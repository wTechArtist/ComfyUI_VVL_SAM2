import torch
from PIL import Image, ImageDraw
import numpy as np
import os
import json
import re
import copy
import supervision as sv
from typing import Tuple, Optional, Any, List

try:
    from florence_sam_processor import process_image
    from utils.sam import model_to_config_map as sam_model_to_config_map
    from utils.sam import load_sam_image_model, run_sam_inference
    from utils.florence import load_florence_model, run_florence_inference, \
        FLORENCE_OPEN_VOCABULARY_DETECTION_TASK, FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
    from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
    from mask_cleaner import remove_small_regions
except ImportError:
    # We're running as a module
    from .florence_sam_processor import process_image
    from .utils.sam import model_to_config_map as sam_model_to_config_map
    from .utils.sam import load_sam_image_model, run_sam_inference
    from .utils.florence import load_florence_model, run_florence_inference, \
        FLORENCE_OPEN_VOCABULARY_DETECTION_TASK, FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
    from .utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
    from .mask_cleaner import remove_small_regions

# GroundingDINO imports (adapted from node.py)
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
import glob

# GroundingDINO specific imports
try:
    from local_groundingdino.datasets import transforms as T
    from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
    from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
    from local_groundingdino.models import build_model as local_groundingdino_build_model
except ImportError:
    print("Warning: GroundingDINO dependencies not found. Please install them.")
    T = None
    local_groundingdino_clean_state_dict = None
    local_groundingdino_SLConfig = None
    local_groundingdino_build_model = None

logger = logging.getLogger('vvl_GroundingDinoSAM2')

# GroundingDINO model configurations
groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

# Format conversion helpers
def tensor2pil(t_image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)

# GroundingDINO utility functions (adapted from node.py)
def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f'using extra model: {destination}')
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination

def load_groundingdino_model(model_name):
    if local_groundingdino_SLConfig is None:
        raise ImportError("GroundingDINO dependencies not available")
        
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir_name
        ),
    )

    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    
    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir_name,
        ),
    )
    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    return dino

def groundingdino_predict(dino_model, image, prompt, threshold):
    def load_dino_image(image_pil):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(dino_model, dino_image, prompt, threshold)
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt

# Color palette and annotators for image annotation
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX
)

def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image

def calculate_iou(box1, box2):
    """计算两个边界框的IoU (Intersection over Union)"""
    # box格式: [x1, y1, x2, y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集面积
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area

def remove_duplicate_boxes(boxes, object_names, iou_threshold=0.5):
    """使用NMS算法去除重复的边界框"""
    if boxes.shape[0] == 0:
        return boxes, object_names
    
    # 计算每个框的面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 按面积排序（保留较大的框）
    order = torch.argsort(areas, descending=True)
    
    keep_indices = []
    remaining = order.tolist()
    
    while remaining:
        # 取出当前最大面积的框
        current_idx = remaining[0]
        keep_indices.append(current_idx)
        remaining.remove(current_idx)
        
        if not remaining:
            break
        
        # 计算当前框与其余框的IoU
        current_box = boxes[current_idx]
        to_remove = []
        
        for other_idx in remaining:
            other_box = boxes[other_idx]
            iou = calculate_iou(current_box, other_box)
            
            if iou > iou_threshold:
                to_remove.append(other_idx)
        
        # 移除重叠的框
        for idx in to_remove:
            remaining.remove(idx)
    
    # 返回去重后的boxes和对应的object_names
    keep_indices = sorted(keep_indices)
    filtered_boxes = boxes[keep_indices]
    filtered_object_names = [object_names[i] for i in keep_indices] if object_names else []
    
    return filtered_boxes, filtered_object_names

def filter_by_area(output_images, output_masks, detections_with_masks, object_names, image_size, 
                  min_area_ratio=0.0001, max_area_ratio=0.9):
    """
    根据mask面积大小过滤分割结果
    
    Args:
        output_images: SAM2分割得到的遮罩图像列表
        output_masks: SAM2分割得到的mask列表  
        detections_with_masks: supervision.Detections对象
        object_names: 对象名称列表
        image_size: 图像尺寸 (width, height)
        min_area_ratio: 最小面积比例（相对于图像总面积）
        max_area_ratio: 最大面积比例（相对于图像总面积）
    
    Returns:
        过滤后的 (output_images, output_masks, detections_with_masks, object_names)
    """
    if not output_masks or detections_with_masks is None:
        return output_images, output_masks, detections_with_masks, object_names
    
    width, height = image_size
    total_area = width * height
    min_area_pixels = total_area * min_area_ratio
    max_area_pixels = total_area * max_area_ratio
    
    keep_indices = []
    filtered_reasons = []
    
    # 计算每个mask的面积并判断是否保留
    for i, mask_tensor in enumerate(output_masks):
        # 将tensor转换为numpy数组并计算面积
        if len(mask_tensor.shape) == 3:
            # 如果是3D tensor (H, W, C)，取第一个通道
            mask_array = mask_tensor[:, :, 0].numpy() > 0.5
        else:
            # 如果是2D tensor (H, W)
            mask_array = mask_tensor.numpy() > 0.5
        
        mask_area = np.sum(mask_array)
        area_ratio = mask_area / total_area
        
        if mask_area < min_area_pixels:
            filtered_reasons.append(f"太小 (面积比例: {area_ratio:.4f} < {min_area_ratio})")
        elif mask_area > max_area_pixels:
            filtered_reasons.append(f"太大 (面积比例: {area_ratio:.4f} > {max_area_ratio})")
        else:
            keep_indices.append(i)
    
    # 如果有被过滤的项目，打印信息
    if len(keep_indices) < len(output_masks):
        filtered_count = len(output_masks) - len(keep_indices)
        print(f"VVL_GroundingDinoSAM2: 基于面积过滤掉 {filtered_count} 个分割结果:")
        for i, reason in enumerate(filtered_reasons):
            if i not in keep_indices:
                obj_name = object_names[i] if i < len(object_names) else f"object_{i+1}"
                print(f"  - {obj_name}: {reason}")
    
    # 过滤结果
    if not keep_indices:
        # 如果所有结果都被过滤掉
        return [], [], None, []
    
    # 过滤output_images和output_masks
    filtered_output_images = [output_images[i] for i in keep_indices] if output_images else []
    filtered_output_masks = [output_masks[i] for i in keep_indices]
    filtered_object_names = [object_names[i] for i in keep_indices] if object_names else []
    
    # 过滤detections_with_masks
    if detections_with_masks is not None:
        # 创建新的detections对象，只包含保留的索引
        if hasattr(detections_with_masks, 'xyxy') and detections_with_masks.xyxy is not None:
            filtered_xyxy = detections_with_masks.xyxy[keep_indices]
        else:
            filtered_xyxy = None
            
        if hasattr(detections_with_masks, 'mask') and detections_with_masks.mask is not None:
            filtered_mask = detections_with_masks.mask[keep_indices]
        else:
            filtered_mask = None
            
        if hasattr(detections_with_masks, 'confidence') and detections_with_masks.confidence is not None:
            filtered_confidence = detections_with_masks.confidence[keep_indices]
        else:
            filtered_confidence = None
            
        if hasattr(detections_with_masks, 'class_id') and detections_with_masks.class_id is not None:
            filtered_class_id = detections_with_masks.class_id[keep_indices]
        else:
            filtered_class_id = None
        
        # 创建新的Detections对象
        filtered_detections = sv.Detections(
            xyxy=filtered_xyxy,
            mask=filtered_mask,
            confidence=filtered_confidence,
            class_id=filtered_class_id
        )
        
        # 复制data字典
        if hasattr(detections_with_masks, 'data') and detections_with_masks.data:
            filtered_detections.data = {}
            for key, value in detections_with_masks.data.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == len(output_masks):
                    if isinstance(value, list):
                        filtered_detections.data[key] = [value[i] for i in keep_indices]
                    else:
                        filtered_detections.data[key] = value[keep_indices]
                else:
                    filtered_detections.data[key] = value
    else:
        filtered_detections = None
    
    return filtered_output_images, filtered_output_masks, filtered_detections, filtered_object_names

def sam2_segment(sam_model, image, boxes):
    """SAM2 segmentation function adapted for SAM2"""
    if boxes.shape[0] == 0:
        return [], [], None
    
    # Convert PIL image to numpy
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    
    # Use SAM2 inference
    # Create a dummy sv.Detections object with the boxes
    detections = sv.Detections(xyxy=boxes.numpy())
    
    # Use the existing SAM2 inference function
    detections_with_masks = run_sam_inference(sam_model, image, detections)
    
    # Convert masks to the expected format
    output_masks = []
    output_images = []
    
    if detections_with_masks.mask is not None:
        for mask in detections_with_masks.mask:
            # Convert mask to PIL Image
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
            output_masks.append(pil2tensor(mask_pil))
            
            # Create masked image
            image_np_copy = copy.deepcopy(image_np)
            if len(image_np_copy.shape) == 3:
                image_np_copy[~mask] = np.array([0, 0, 0])
            else:
                image_np_copy[~mask] = np.array([0, 0, 0, 0])
            
            masked_image_pil = Image.fromarray(image_np_copy)
            output_images.append(pil2tensor(masked_image_pil.convert("RGB")))
    
    return output_images, output_masks, detections_with_masks

# Global variables for GroundingDINO and Florence2 model management
GROUNDING_DINO_MODEL = None
FLORENCE_MODEL = None
FLORENCE_PROCESSOR = None
CURRENT_GROUNDING_DINO_MODEL_NAME = None

def lazy_load_grounding_dino_florence_models(grounding_dino_model_name: str, load_florence2: bool = True):
    global GROUNDING_DINO_MODEL, FLORENCE_MODEL, FLORENCE_PROCESSOR, CURRENT_GROUNDING_DINO_MODEL_NAME
    
    # Load GroundingDINO model
    if GROUNDING_DINO_MODEL is None or CURRENT_GROUNDING_DINO_MODEL_NAME != grounding_dino_model_name:
        GROUNDING_DINO_MODEL = load_groundingdino_model(grounding_dino_model_name)
        CURRENT_GROUNDING_DINO_MODEL_NAME = grounding_dino_model_name
    
    # Load Florence-2 model (for caption generation when needed)
    if load_florence2 and (FLORENCE_MODEL is None or FLORENCE_PROCESSOR is None):
        device = comfy.model_management.get_torch_device()
        try:
            from utils.florence import load_florence_model
        except ImportError:
            from .utils.florence import load_florence_model
        FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=device)

class VVL_GroundingDinoSAM2:
    @classmethod
    def INPUT_TYPES(cls):
        grounding_dino_models = list(groundingdino_model_list.keys())
        
        return {
            "required": {
                "sam2_model": ("VVL_SAM2_MODEL", {"tooltip": "SAM2分割模型，用于对检测到的对象进行精确分割"}),
                "grounding_dino_model": (grounding_dino_models, {
                    "default": grounding_dino_models[0],
                    "tooltip": "GroundingDINO目标检测模型，用于根据文本提示检测图像中的对象。SwinT_OGC模型较小但速度快，SwinB模型较大但精度更高"
                }),
                "image": ("IMAGE", {"tooltip": "输入的图像，支持批量处理多张图像"}),
                "prompt": ("STRING", {
                    "default": "",
                    "tooltip": "目标检测的文本提示词，用逗号分隔多个对象，如'person,car,dog'。留空时将使用Florence-2自动生成描述"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "目标检测的置信度阈值，值越高检测越严格，建议范围0.2-0.5。过低会产生误检，过高可能漏检"
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "IoU阈值用于去除重复检测框，值越高保留的重叠框越多。建议0.3-0.7，避免同一对象被重复分割"
                }),
            },
            "optional": {
                "external_caption": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "tooltip": "外部提供的图像描述文本，用逗号分隔多个对象。当prompt为空时，优先使用此描述进行目标检测"
                }),
                "load_florence2": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否加载Florence-2模型用于自动生成图像描述。当prompt和external_caption都为空时，将自动描述图像内容并进行检测"
                }),
                "min_area_ratio": ("FLOAT", {
                    "default": 0.002, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.0001, 
                    "tooltip": "最小面积比例（相对于图像总面积），用于过滤太小的分割结果。0.002表示占图像0.2%以下的区域将被过滤掉"
                }),
                "max_area_ratio": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "tooltip": "最大面积比例（相对于图像总面积），用于过滤太大的分割结果。0.2表示占图像20%以上的区域将被过滤掉，避免背景误检"
                }),
                "remaining_area_mask": ("MASK", {
                    "tooltip": "可选：用于计算剩余区域的输入mask。当提供时，节点将在输出object_masks中新增一个mask，代表(输入mask ∩ 非已分割区域)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING",)
    RETURN_NAMES = ("annotated_image", "object_masks", "detection_json", "object_names",)
    FUNCTION = "_process_image"
    CATEGORY = "💃rDancer"
    OUTPUT_IS_LIST = (False, True, False, True)

    def _process_image(self, sam2_model: dict, grounding_dino_model: str, image: torch.Tensor, 
                      prompt: str = "", threshold: float = 0.3, iou_threshold: float = 0.5, external_caption: str = "", 
                      load_florence2: bool = True, min_area_ratio: float = 0.0001, max_area_ratio: float = 0.9,
                      remaining_area_mask: Optional[torch.Tensor] = None):
        
        # 从SAM2模型字典中获取模型和设备信息
        sam2_model_instance = sam2_model['model']
        device = sam2_model['device']
        
        # 加载GroundingDINO和Florence2模型
        lazy_load_grounding_dino_florence_models(grounding_dino_model, load_florence2)
        
        prompt_clean = prompt.strip() if prompt else ""
        external_caption_clean = external_caption.strip() if external_caption else ""
        
        annotated_images, object_masks_list, detection_jsons, final_object_names = [], [], [], []
        
        for i, img_tensor in enumerate(image):
            img_pil = tensor2pil(img_tensor).convert("RGB")
            
            # Determine processing mode
            current_detection_phrases = []
            detection_mode_info = ""

            if prompt_clean != "":
                # Mode 1: Direct prompt with GroundingDINO
                current_detection_phrases = [p.strip() for p in prompt_clean.split(',') if p.strip()]
                if not current_detection_phrases and prompt_clean:
                    current_detection_phrases = [prompt_clean.strip()]
                detection_mode_info = f"direct prompt list: {current_detection_phrases}"
                
            elif external_caption_clean != "":
                # Mode 2: Use external caption for grounding
                current_detection_phrases = [c.strip() for c in external_caption_clean.split(',') if c.strip()]
                if not current_detection_phrases and external_caption_clean:
                    current_detection_phrases = [external_caption_clean.strip()]
                detection_mode_info = f"external caption list: {current_detection_phrases}"
                
            else:
                # Mode 3: Generate caption with Florence-2, then use for grounding
                if load_florence2 and FLORENCE_MODEL is not None and FLORENCE_PROCESSOR is not None:
                    print(f"VVL_GroundingDinoSAM2: Image {i} - Generating caption with Florence-2.")
                    _, result_caption = run_florence_inference(
                        model=FLORENCE_MODEL,
                        processor=FLORENCE_PROCESSOR,
                        device=device,
                        image=img_pil,
                        task=FLORENCE_DETAILED_CAPTION_TASK
                    )
                    generated_caption = result_caption[FLORENCE_DETAILED_CAPTION_TASK]
                    current_detection_phrases = [c.strip() for c in generated_caption.split(',') if c.strip()]
                    if not current_detection_phrases and generated_caption:
                        current_detection_phrases = [generated_caption.strip()]
                    detection_mode_info = f"Florence-2 generated caption list: {current_detection_phrases}"
                else:
                    print("VVL_GroundingDinoSAM2: Florence-2 model not available, skipping caption generation.")
                    current_detection_phrases = []
                    detection_mode_info = "No detection phrases available"

            print(f"VVL_GroundingDinoSAM2: Image {i} - Mode: {detection_mode_info}")

            object_names = []
            all_boxes_list = []
            
            if not current_detection_phrases:
                print(f"VVL_GroundingDinoSAM2: Image {i} - No detection phrases to process.")
                boxes = torch.zeros((0,4))
            else:
                for phrase_idx, phrase in enumerate(current_detection_phrases):
                    boxes_single = groundingdino_predict(GROUNDING_DINO_MODEL, img_pil, phrase, threshold)
                    if boxes_single.shape[0] > 0:
                        all_boxes_list.append(boxes_single)
                        object_names.extend([phrase] * boxes_single.shape[0])
                
                if len(all_boxes_list) > 0:
                    boxes = torch.cat(all_boxes_list, dim=0)
                else:
                    boxes = torch.zeros((0,4))
            
            # Fallback logic if no boxes found with initial threshold
            if boxes.shape[0] == 0 and threshold > 0.15 and current_detection_phrases:
                fallback_thresh = max(0.1, threshold * 0.5)
                print(f"VVL_GroundingDinoSAM2: Image {i} - No boxes found with threshold {threshold}. Lowering to {fallback_thresh} and retrying.")
                
                all_boxes_list_fallback = []
                object_names_fallback = []

                for phrase_idx, phrase in enumerate(current_detection_phrases):
                    boxes_single_fallback = groundingdino_predict(GROUNDING_DINO_MODEL, img_pil, phrase, fallback_thresh)
                    if boxes_single_fallback.shape[0] > 0:
                        all_boxes_list_fallback.append(boxes_single_fallback)
                        object_names_fallback.extend([phrase] * boxes_single_fallback.shape[0])
                
                if len(all_boxes_list_fallback) > 0:
                    boxes = torch.cat(all_boxes_list_fallback, dim=0)
                    object_names = object_names_fallback
                else:
                    object_names = [] 
                    boxes = torch.zeros((0,4))

            print(f"VVL_GroundingDinoSAM2: Image {i} - Total boxes found for SAM2 input: {boxes.shape[0]}")
            if boxes.shape[0] > 0:
                print(f"VVL_GroundingDinoSAM2: Image {i} - Corresponding object names: {object_names}")

            # 应用边界框去重逻辑，避免重复分割同一个对象
            if boxes.shape[0] > 0:
                boxes_before_dedup = boxes.shape[0]
                boxes, object_names = remove_duplicate_boxes(boxes, object_names, iou_threshold)
                boxes_after_dedup = boxes.shape[0]
                if boxes_before_dedup != boxes_after_dedup:
                    print(f"VVL_GroundingDinoSAM2: Image {i} - Removed {boxes_before_dedup - boxes_after_dedup} duplicate boxes (IoU threshold: {iou_threshold})")
                    print(f"VVL_GroundingDinoSAM2: Image {i} - Final boxes for SAM2: {boxes_after_dedup}")

            if boxes.shape[0] == 0:
                print("VVL_GroundingDinoSAM2: No objects detected.")
                # Create empty results
                annotated_images.append(pil2tensor(img_pil))
                detection_jsons.append(json.dumps({
                    "image_width": img_pil.width,
                    "image_height": img_pil.height,
                    "objects": []
                }, ensure_ascii=False, indent=2))
                continue
            
            # Use SAM2 for segmentation
            output_images, output_masks, detections_with_masks = sam2_segment(sam2_model_instance, img_pil, boxes)
            
            # 应用面积过滤（如果有分割结果）
            if output_masks and (min_area_ratio > 0 or max_area_ratio < 1.0):
                print(f"VVL_GroundingDinoSAM2: Image {i} - 应用面积过滤（最小比例: {min_area_ratio}, 最大比例: {max_area_ratio}）")
                output_images, output_masks, detections_with_masks, object_names = filter_by_area(
                    output_images, output_masks, detections_with_masks, object_names, 
                    (img_pil.width, img_pil.height), min_area_ratio, max_area_ratio
                )
                
                # 如果所有结果都被面积过滤掉了
                if not output_masks:
                    print(f"VVL_GroundingDinoSAM2: Image {i} - 所有分割结果都被面积过滤掉了")
                    annotated_images.append(pil2tensor(img_pil))
                    detection_jsons.append(json.dumps({
                        "image_width": img_pil.width,
                        "image_height": img_pil.height,
                        "objects": []
                    }, ensure_ascii=False, indent=2))
                    continue
            
            # 处理 remaining_area_mask，生成剩余区域mask
            if remaining_area_mask is not None:
                # 取得当前批次对应的输入mask
                if remaining_area_mask.ndim == 4:
                    cur_input_mask = remaining_area_mask[i]
                else:
                    cur_input_mask = remaining_area_mask

                # 转换为 (H, W) 的布尔数组
                input_mask_np = cur_input_mask.cpu().numpy()
                if input_mask_np.ndim == 3:
                    # 处理形状为 (C, H, W) 或 (H, W, C)
                    if input_mask_np.shape[0] == 1:  # (1, H, W)
                        input_mask_np = input_mask_np[0]
                    else:  # (H, W, 1) 或其他
                        input_mask_np = input_mask_np[:, :, 0]
                input_mask_bool = np.squeeze(input_mask_np) > 0.5

                # 合并已有的所有mask
                combined_existing = np.zeros_like(input_mask_bool, dtype=bool)
                for m_tensor in output_masks:
                    m_np = m_tensor.cpu().numpy()
                    if m_np.ndim == 3:
                        if m_np.shape[0] == 1:
                            m_np = m_np[0]
                        else:
                            m_np = m_np[:, :, 0]
                    combined_existing |= (m_np > 0.5)

                # 计算剩余区域 (输入mask 交 非已分割区域)
                remain_bool = np.logical_and(input_mask_bool, np.logical_not(combined_existing))

                if np.sum(remain_bool) > 0:
                    # 清理零碎区域，只保留最大的连通域
                    remain_uint8 = (remain_bool * 255).astype(np.uint8)
                    remain_cleaned = remove_small_regions(remain_uint8, keep_largest_n=1)
                    remain_bool_cleaned = remain_cleaned > 127
                    
                    # 如果清理后还有区域存在，则继续处理
                    if np.sum(remain_bool_cleaned) > 0:
                        # 生成mask tensor
                        remain_mask_pil = Image.fromarray(remain_cleaned).convert("L")
                        remain_mask_tensor = pil2tensor(remain_mask_pil)
                        output_masks.append(remain_mask_tensor)

                        # 生成对应的masked image
                        img_np_full = np.array(img_pil)
                        img_np_copy = copy.deepcopy(img_np_full)
                        if len(img_np_copy.shape) == 3:
                            img_np_copy[~remain_bool_cleaned] = np.array([0, 0, 0])
                        else:
                            img_np_copy[~remain_bool_cleaned] = np.array([0, 0, 0, 0])
                        remain_image_pil = Image.fromarray(img_np_copy)
                        output_images.append(pil2tensor(remain_image_pil.convert("RGB")))

                        # 计算bbox（基于清理后的mask）
                        ys, xs = np.where(remain_bool_cleaned)
                        x_min, x_max = int(xs.min()), int(xs.max())
                        y_min, y_max = int(ys.min()), int(ys.max())
                        bbox_tensor = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

                        # 更新 detections_with_masks
                        if detections_with_masks is None:
                            detections_with_masks = sv.Detections(xyxy=bbox_tensor.unsqueeze(0), mask=np.asarray([remain_bool_cleaned]))
                        else:
                            # 根据现有 xyxy 的数据类型决定拼接方式，避免 numpy 与 tensor 冲突
                            if hasattr(detections_with_masks, 'xyxy') and detections_with_masks.xyxy is not None:
                                if isinstance(detections_with_masks.xyxy, np.ndarray):
                                    bbox_np = bbox_tensor.cpu().numpy()[None, :]
                                    detections_with_masks.xyxy = np.concatenate([detections_with_masks.xyxy, bbox_np], axis=0)
                                else:
                                    detections_with_masks.xyxy = torch.cat([detections_with_masks.xyxy, bbox_tensor.unsqueeze(0)], dim=0)
                            else:
                                # 初始为空时沿用 bbox 的类型
                                detections_with_masks.xyxy = bbox_tensor.unsqueeze(0)
                            # mask
                            if hasattr(detections_with_masks, 'mask') and detections_with_masks.mask is not None:
                                detections_with_masks.mask = np.concatenate([detections_with_masks.mask, remain_bool_cleaned[None, :, :]], axis=0)
                            else:
                                detections_with_masks.mask = np.asarray([remain_bool_cleaned])
                        
                        # 追加名称
                        object_names.append("remaining_area")

            # 将最终的对象名称添加到列表中
            final_object_names.extend(object_names)
            
            # 使用supervision库的标注器来标注图像
            if len(object_names) > 0 and detections_with_masks is not None:
                # 创建标签列表，确保长度与检测结果匹配
                labels = []
                for j in range(len(detections_with_masks)):
                    if j < len(object_names):
                        labels.append(object_names[j])
                    else:
                        labels.append(f"object_{j+1}")
                
                # 设置detections的data字典来存储标签
                if not hasattr(detections_with_masks, 'data') or detections_with_masks.data is None:
                    detections_with_masks.data = {}
                detections_with_masks.data['class_name'] = labels
                
                # 使用与app.py相同的annotate_image函数
                annotated_img = annotate_image(img_pil, detections_with_masks)
                annotated_images.append(pil2tensor(annotated_img))
            else:
                # 如果没有检测到对象，则返回原始图像
                annotated_images.append(pil2tensor(img_pil))
            
            # Add masks to the list
            if output_masks:
                object_masks_list.extend(output_masks)
            
            # Create detection JSON
            detection_json = {
                "image_width": img_pil.width,
                "image_height": img_pil.height,
                "objects": []
            }
            
            # 使用过滤后的检测结果来生成JSON
            if detections_with_masks is not None and hasattr(detections_with_masks, 'xyxy') and detections_with_masks.xyxy is not None:
                for j, bbox in enumerate(detections_with_masks.xyxy):
                    bbox_2d = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    detection_json["objects"].append({
                        "name": object_names[j] if j < len(object_names) else f"object_{j+1}",
                        "bbox_2d": bbox_2d
                    })
            
            # Format JSON with single-line bbox_2d
            json_str = json.dumps(detection_json, ensure_ascii=False, indent=2)
            json_str = re.sub(r'"bbox_2d":\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]', 
                             r'"bbox_2d": [\1,\2,\3,\4]', json_str)
            detection_jsons.append(json_str)
        
        # Stack results
        annotated_images_stacked = torch.stack(annotated_images) if annotated_images else torch.empty(0)
        final_detection_json = detection_jsons[0] if detection_jsons else "{}"
        
        return (annotated_images_stacked, object_masks_list, final_detection_json, final_object_names)

# Node registration
NODE_CLASS_MAPPINGS = {
    "VVL_GroundingDinoSAM2": VVL_GroundingDinoSAM2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_GroundingDinoSAM2": "VVL GroundingDINO + SAM2"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 