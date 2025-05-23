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
    from app import process_image
    from utils.sam import model_to_config_map as sam_model_to_config_map
    from utils.sam import load_sam_image_model, run_sam_inference
    from utils.florence import load_florence_model, run_florence_inference, \
        FLORENCE_OPEN_VOCABULARY_DETECTION_TASK, FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
    from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
except ImportError:
    # We're running as a module
    from .app import process_image
    from .utils.sam import model_to_config_map as sam_model_to_config_map
    from .utils.sam import load_sam_image_model, run_sam_inference
    from .utils.florence import load_florence_model, run_florence_inference, \
        FLORENCE_OPEN_VOCABULARY_DETECTION_TASK, FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
    from .utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES

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
                "sam2_model": ("VVL_SAM2_MODEL",),
                "grounding_dino_model": (grounding_dino_models, {"default": grounding_dino_models[0]}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": ""}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "external_caption": ("STRING", {"multiline": True, "default": ""}),
                "load_florence2": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING",)
    RETURN_NAMES = ("annotated_image", "object_masks", "masked_image", "detection_json",)
    FUNCTION = "_process_image"
    CATEGORY = "💃rDancer"
    OUTPUT_IS_LIST = (False, True, False, False)

    def _process_image(self, sam2_model: dict, grounding_dino_model: str, image: torch.Tensor, 
                      prompt: str = "", threshold: float = 0.3, external_caption: str = "", 
                      load_florence2: bool = True):
        
        # 从SAM2模型字典中获取模型和设备信息
        sam2_model_instance = sam2_model['model']
        device = sam2_model['device']
        
        # 加载GroundingDINO和Florence2模型
        lazy_load_grounding_dino_florence_models(grounding_dino_model, load_florence2)
        
        prompt_clean = prompt.strip() if prompt else ""
        external_caption_clean = external_caption.strip() if external_caption else ""
        
        annotated_images, object_masks_list, masked_images, detection_jsons = [], [], [], []
        
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

            if boxes.shape[0] == 0:
                print("VVL_GroundingDinoSAM2: No objects detected.")
                # Create empty results
                annotated_images.append(pil2tensor(img_pil))
                masked_images.append(pil2tensor(Image.new("RGB", img_pil.size, (0, 0, 0))))
                detection_jsons.append(json.dumps({
                    "image_width": img_pil.width,
                    "image_height": img_pil.height,
                    "objects": []
                }, ensure_ascii=False, indent=2))
                continue
            
            # Use SAM2 for segmentation
            output_images, output_masks, detections_with_masks = sam2_segment(sam2_model_instance, img_pil, boxes)
            
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
            
            # Create masked image (combine all masks)
            if output_images:
                # Use the first masked image as representative
                masked_images.append(output_images[0])
            else:
                masked_images.append(pil2tensor(Image.new("RGB", img_pil.size, (0, 0, 0))))
            
            # Create detection JSON
            detection_json = {
                "image_width": img_pil.width,
                "image_height": img_pil.height,
                "objects": []
            }
            
            for j, bbox in enumerate(boxes):
                if j < len(object_names):
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
        masked_images_stacked = torch.stack(masked_images) if masked_images else torch.empty(0)
        final_detection_json = detection_jsons[0] if detection_jsons else "{}"
        
        return (annotated_images_stacked, object_masks_list, masked_images_stacked, final_detection_json)

# Node registration
NODE_CLASS_MAPPINGS = {
    "VVL_GroundingDinoSAM2": VVL_GroundingDinoSAM2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_GroundingDinoSAM2": "VVL GroundingDINO + SAM2"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 