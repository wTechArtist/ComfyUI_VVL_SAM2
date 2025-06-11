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
    from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
    from mask_cleaner import remove_small_regions
except ImportError:
    # We're running as a module
    from .florence_sam_processor import process_image
    from .utils.sam import model_to_config_map as sam_model_to_config_map
    from .utils.sam import load_sam_image_model, run_sam_inference
    from .utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
    from .mask_cleaner import remove_small_regions

# GroundingDINO imports
import logging
import comfy.model_management

# Import GroundingDINO loader
try:
    from grounding_dino_loader import groundingdino_model_list
    from local_groundingdino.datasets import transforms as T
except ImportError:
    try:
        from .grounding_dino_loader import groundingdino_model_list
        from local_groundingdino.datasets import transforms as T
    except ImportError:
        print("Warning: GroundingDINO dependencies not found. Please install them.")
        T = None
        groundingdino_model_list = {}

logger = logging.getLogger('vvl_GroundingDinoSAM2')

# Format conversion helpers
def tensor2pil(t_image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)

# GroundingDINO prediction function

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

# 兼容不同版本的supervision
try:
    # 新版本 supervision (>= 0.21.0)
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
except AttributeError:
    # 旧版本 supervision (0.6.0)
    BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE)
    MASK_ANNOTATOR = sv.MaskAnnotator(color=COLOR_PALETTE)
    
    # supervision 0.6.0 没有 LabelAnnotator，创建一个简单的替代品
    class SimpleLabelAnnotator:
        def __init__(self, color=None, **kwargs):
            self.color = color
            
        def annotate(self, scene, detections, labels=None):
            # 如果没有标签，直接返回原图像
            if not labels or len(labels) == 0:
                return scene
            
            import cv2
            output = scene.copy()
            
            for i, (box, label) in enumerate(zip(detections.xyxy, labels)):
                x1, y1, x2, y2 = map(int, box)
                # 在框的左上角绘制标签
                cv2.putText(output, str(label), (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return output
    
    LABEL_ANNOTATOR = SimpleLabelAnnotator(color=COLOR_PALETTE)

def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image

def resolve_duplicate_names(object_names):
    """
    解决对象名称重复问题，为重复的名称自动添加数字后缀
    例如：['person', 'person', 'car', 'person'] -> ['person', 'person_2', 'car', 'person_3']
    
    Args:
        object_names: 原始对象名称列表
    
    Returns:
        resolved_names: 解决重复后的对象名称列表
    """
    if not object_names:
        return object_names
    
    name_counts = {}
    resolved_names = []
    
    for name in object_names:
        # 清理名称，去除可能已存在的数字后缀
        base_name = name
        if '_' in name:
            parts = name.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                base_name = '_'.join(parts[:-1])
        
        if base_name in name_counts:
            name_counts[base_name] += 1
            resolved_name = f"{base_name}_{name_counts[base_name]}"
        else:
            name_counts[base_name] = 1
            resolved_name = base_name
        
        resolved_names.append(resolved_name)
    
    return resolved_names

def verify_data_consistency(object_names, detections_with_masks, output_masks, image_index=0):
    """
    验证对象名称、检测结果和mask之间的数据一致性
    
    Args:
        object_names: 对象名称列表
        detections_with_masks: supervision.Detections对象
        output_masks: mask tensor列表
        image_index: 图像索引（用于日志）
    
    Returns:
        bool: 是否一致
    """
    issues = []
    
    # 检查基本数量
    name_count = len(object_names) if object_names else 0
    
    if detections_with_masks is not None:
        if hasattr(detections_with_masks, 'xyxy') and detections_with_masks.xyxy is not None:
            bbox_count = len(detections_with_masks.xyxy)
        else:
            bbox_count = 0
            
        if hasattr(detections_with_masks, 'mask') and detections_with_masks.mask is not None:
            detection_mask_count = len(detections_with_masks.mask)
        else:
            detection_mask_count = 0
    else:
        bbox_count = 0
        detection_mask_count = 0
    
    output_mask_count = len(output_masks) if output_masks else 0
    
    # 验证数量一致性
    if name_count != bbox_count:
        issues.append(f"对象名称数量({name_count})与bbox数量({bbox_count})不匹配")
    
    if name_count != output_mask_count:
        issues.append(f"对象名称数量({name_count})与输出mask数量({output_mask_count})不匹配")
        
    if bbox_count != detection_mask_count:
        issues.append(f"bbox数量({bbox_count})与检测mask数量({detection_mask_count})不匹配")
    
    # 打印验证结果
    if issues:
        print(f"❌ VVL_GroundingDinoSAM2: Image {image_index} - 数据一致性验证失败:")
        for issue in issues:
            print(f"   - {issue}")
        print(f"   对象名称: {object_names}")
        return False
    else:
        print(f"✅ VVL_GroundingDinoSAM2: Image {image_index} - 数据一致性验证通过")
        print(f"   总数: {name_count}个对象")
        print(f"   对象名称: {object_names}")
        return True

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
    """使用NMS算法去除重复的边界框（仅作为初步过滤）"""
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

def calculate_mask_containment_ratio(mask1, mask2):
    """计算 mask2 被 mask1 包含的比例"""
    # 确保两个mask的形状相同
    if mask1.shape != mask2.shape:
        return 0.0
    
    # 转换为布尔数组
    mask1_bool = mask1 > 0.5
    mask2_bool = mask2 > 0.5
    
    # 计算 mask2 的面积
    mask2_area = np.sum(mask2_bool)
    if mask2_area == 0:
        return 0.0
    
    # 计算 mask2 被 mask1 包含的区域
    intersection = np.sum(mask1_bool & mask2_bool)
    
    # 返回包含比例
    return intersection / mask2_area

def remove_duplicate_masks_by_containment(output_images, output_masks, detections_with_masks, object_names, containment_threshold=0.8):
    """基于实际mask形状去除被包含的重复分割结果"""
    if not output_masks or len(output_masks) <= 1:
        return output_images, output_masks, detections_with_masks, object_names
    
    # 转换所有mask为numpy数组用于计算
    masks_np = []
    for mask_tensor in output_masks:
        if len(mask_tensor.shape) == 3:
            # 如果是3D tensor (H, W, C)，取第一个通道
            mask_np = mask_tensor[:, :, 0].numpy()
        else:
            # 如果是2D tensor (H, W)
            mask_np = mask_tensor.numpy()
        masks_np.append(mask_np)
    
    # 计算每个mask的面积，按面积排序（保留大的）
    mask_areas = [np.sum(mask > 0.5) for mask in masks_np]
    sorted_indices = sorted(range(len(mask_areas)), key=lambda i: mask_areas[i], reverse=True)
    
    keep_indices = []
    
    for i in sorted_indices:
        current_mask = masks_np[i]
        should_keep = True
        
        # 检查当前mask是否被已保留的任何mask包含
        for kept_idx in keep_indices:
            kept_mask = masks_np[kept_idx]
            containment_ratio = calculate_mask_containment_ratio(kept_mask, current_mask)
            
            if containment_ratio > containment_threshold:
                # 当前mask被包含程度超过阈值，不保留
                obj_name = object_names[i] if i < len(object_names) else f"object_{i+1}"
                kept_obj_name = object_names[kept_idx] if kept_idx < len(object_names) else f"object_{kept_idx+1}"
                print(f"VVL_GroundingDinoSAM2: 移除被包含的mask - '{obj_name}'被'{kept_obj_name}'包含{containment_ratio:.2%}")
                should_keep = False
                break
        
        if should_keep:
            keep_indices.append(i)
    
    # 按原始顺序排序保留的索引
    keep_indices = sorted(keep_indices)
    
    if len(keep_indices) == len(output_masks):
        # 没有需要移除的
        return output_images, output_masks, detections_with_masks, object_names
    
    # 过滤结果
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
    
    print(f"VVL_GroundingDinoSAM2: 基于mask包含关系过滤掉 {len(output_masks) - len(keep_indices)} 个重复分割结果")
    
    return filtered_output_images, filtered_output_masks, filtered_detections, filtered_object_names

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

# GroundingDINO model management functions (removed - now using dedicated loader node)

class VVL_GroundingDinoSAM2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam2_model": ("SAM2MODEL", {"tooltip": "SAM2分割模型，用于对检测到的对象进行精确分割"}),
                "grounding_dino_model": ("GROUNDING_DINO_MODEL", {"tooltip": "GroundingDINO目标检测模型，用于根据文本提示检测图像中的对象"}),
                "image": ("IMAGE", {"tooltip": "输入的图像，支持批量处理多张图像"}),
                "external_caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "图像描述文本或待检测对象列表，用逗号分隔，如'person,car,dog'"
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
                "mask_containment_threshold": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "基于实际mask形状的包含阈值，用于移除被其他mask包含的重复分割。设为0时禁用此功能，值越高越严格，0.8表示被包含80%以上才移除"
                }),
            },
            "optional": {
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

    def _process_image(self, sam2_model, grounding_dino_model, image, external_caption, 
                      threshold=0.3, iou_threshold=0.5, mask_containment_threshold=0.8,
                      min_area_ratio=0.002, max_area_ratio=0.2, remaining_area_mask=None):
        
        try:
            # 输入验证和安全检查
            if not isinstance(image, torch.Tensor):
                raise ValueError(f"Expected image to be torch.Tensor, got {type(image)}")
            
            if image.dim() != 4:
                raise ValueError(f"Expected image to have 4 dimensions (batch, height, width, channels), got {image.dim()}")
                
            batch_size = image.shape[0]
            print(f"VVL_GroundingDinoSAM2: Processing {batch_size} images")
            
            # 从SAM2模型字典中获取模型和设备信息
            if not isinstance(sam2_model, dict) or 'model' not in sam2_model:
                raise ValueError("Invalid sam2_model format")
                
            sam2_model_instance = sam2_model['model']
            device = sam2_model.get('device', 'cpu')
            
            # 从传入的grounding_dino_model字典中获取模型
            if not isinstance(grounding_dino_model, dict) or 'model' not in grounding_dino_model:
                raise ValueError("Invalid grounding_dino_model format")
                
            grounding_dino_model_instance = grounding_dino_model['model']
            
            # 清理输入文本
            external_caption_clean = external_caption.strip() if external_caption else ""
            
            # 初始化结果列表
            annotated_images = []
            object_masks_list = []
            detection_jsons = []
            final_object_names = []
            
            # 处理每张图像
            for i in range(batch_size):
                try:
                    img_tensor = image[i]
                    img_pil = tensor2pil(img_tensor).convert("RGB")
                    
                    print(f"VVL_GroundingDinoSAM2: Processing image {i+1}/{batch_size}")
                    
                    # 解析检测短语
                    current_detection_phrases = []
                    if external_caption_clean:
                        current_detection_phrases = [c.strip() for c in external_caption_clean.split(',') if c.strip()]
                        if not current_detection_phrases and external_caption_clean:
                            current_detection_phrases = [external_caption_clean.strip()]
                        print(f"VVL_GroundingDinoSAM2: Image {i+1} - Detection phrases: {current_detection_phrases}")
                    else:
                        print(f"VVL_GroundingDinoSAM2: Image {i+1} - No external_caption provided")
                    
                    # 执行目标检测
                    object_names = []
                    all_boxes_list = []
                    
                    if not current_detection_phrases:
                        boxes = torch.zeros((0, 4))
                    else:
                        for phrase in current_detection_phrases:
                            boxes_single = groundingdino_predict(grounding_dino_model_instance, img_pil, phrase, threshold)
                            if boxes_single.shape[0] > 0:
                                all_boxes_list.append(boxes_single)
                                object_names.extend([phrase] * boxes_single.shape[0])
                        
                        boxes = torch.cat(all_boxes_list, dim=0) if all_boxes_list else torch.zeros((0, 4))
                    
                    # 后备逻辑：如果没有检测到对象，尝试降低阈值
                    if boxes.shape[0] == 0 and threshold > 0.15 and current_detection_phrases:
                        fallback_thresh = max(0.1, threshold * 0.5)
                        print(f"VVL_GroundingDinoSAM2: Image {i+1} - Retrying with lower threshold {fallback_thresh}")
                        
                        for phrase in current_detection_phrases:
                            boxes_single = groundingdino_predict(grounding_dino_model_instance, img_pil, phrase, fallback_thresh)
                            if boxes_single.shape[0] > 0:
                                all_boxes_list.append(boxes_single)
                                object_names.extend([phrase] * boxes_single.shape[0])
                        
                        boxes = torch.cat(all_boxes_list, dim=0) if all_boxes_list else torch.zeros((0, 4))
                    
                    print(f"VVL_GroundingDinoSAM2: Image {i+1} - Found {boxes.shape[0]} boxes")
                    
                    if boxes.shape[0] == 0:
                        # 没有检测到对象，返回原始图像
                        annotated_images.append(pil2tensor(img_pil))
                        detection_jsons.append(json.dumps({
                            "image_width": img_pil.width,
                            "image_height": img_pil.height,
                            "objects": []
                        }, ensure_ascii=False, indent=2))
                        continue
                    
                    # 去重复边界框
                    if boxes.shape[0] > 0:
                        boxes, object_names = remove_duplicate_boxes(boxes, object_names, iou_threshold)
                        print(f"VVL_GroundingDinoSAM2: Image {i+1} - After deduplication: {boxes.shape[0]} boxes")
                    
                    # SAM2分割
                    output_images, output_masks, detections_with_masks = sam2_segment(sam2_model_instance, img_pil, boxes)
                    
                    # 应用各种过滤器
                    if output_masks and len(output_masks) > 1 and mask_containment_threshold > 0:
                        output_images, output_masks, detections_with_masks, object_names = remove_duplicate_masks_by_containment(
                            output_images, output_masks, detections_with_masks, object_names, containment_threshold=mask_containment_threshold
                        )
                    
                    if output_masks and (min_area_ratio > 0 or max_area_ratio < 1.0):
                        output_images, output_masks, detections_with_masks, object_names = filter_by_area(
                            output_images, output_masks, detections_with_masks, object_names, 
                            (img_pil.width, img_pil.height), min_area_ratio, max_area_ratio
                        )
                    
                    # 处理剩余区域mask
                    if remaining_area_mask is not None and output_masks:
                        try:
                            if remaining_area_mask.dim() == 4 and i < remaining_area_mask.shape[0]:
                                cur_input_mask = remaining_area_mask[i]
                            elif remaining_area_mask.dim() == 3:
                                cur_input_mask = remaining_area_mask
                            else:
                                print(f"VVL_GroundingDinoSAM2: Warning - Cannot process remaining_area_mask for image {i+1}")
                                cur_input_mask = None
                            
                            if cur_input_mask is not None:
                                # 处理剩余区域逻辑...
                                input_mask_np = cur_input_mask.cpu().numpy()
                                if input_mask_np.ndim == 3:
                                    if input_mask_np.shape[0] == 1:
                                        input_mask_np = input_mask_np[0]
                                    else:
                                        input_mask_np = input_mask_np[:, :, 0]
                                input_mask_bool = np.squeeze(input_mask_np) > 0.5
                                
                                # 合并已有mask
                                combined_existing = np.zeros_like(input_mask_bool, dtype=bool)
                                for m_tensor in output_masks:
                                    m_np = m_tensor.cpu().numpy()
                                    if m_np.ndim == 3:
                                        if m_np.shape[0] == 1:
                                            m_np = m_np[0]
                                        else:
                                            m_np = m_np[:, :, 0]
                                    combined_existing |= (m_np > 0.5)
                                
                                # 计算剩余区域
                                remain_bool = np.logical_and(input_mask_bool, np.logical_not(combined_existing))
                                
                                if np.sum(remain_bool) > 0:
                                    remain_uint8 = (remain_bool * 255).astype(np.uint8)
                                    remain_cleaned = remove_small_regions(remain_uint8, keep_largest_n=1)
                                    remain_bool_cleaned = remain_cleaned > 127
                                    
                                    if np.sum(remain_bool_cleaned) > 0:
                                        remain_mask_pil = Image.fromarray(remain_cleaned).convert("L")
                                        remain_mask_tensor = pil2tensor(remain_mask_pil)
                                        output_masks.append(remain_mask_tensor)
                                        
                                        img_np_copy = copy.deepcopy(np.array(img_pil))
                                        img_np_copy[~remain_bool_cleaned] = np.array([0, 0, 0])
                                        remain_image_pil = Image.fromarray(img_np_copy)
                                        output_images.append(pil2tensor(remain_image_pil.convert("RGB")))
                                        
                                        # 更新检测结果
                                        ys, xs = np.where(remain_bool_cleaned)
                                        x_min, x_max = int(xs.min()), int(xs.max())
                                        y_min, y_max = int(ys.min()), int(ys.max())
                                        bbox_tensor = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
                                        
                                        if detections_with_masks is None:
                                            detections_with_masks = sv.Detections(xyxy=bbox_tensor.unsqueeze(0), mask=np.asarray([remain_bool_cleaned]))
                                        else:
                                            if hasattr(detections_with_masks, 'xyxy') and detections_with_masks.xyxy is not None:
                                                if isinstance(detections_with_masks.xyxy, np.ndarray):
                                                    bbox_np = bbox_tensor.cpu().numpy()[None, :]
                                                    detections_with_masks.xyxy = np.concatenate([detections_with_masks.xyxy, bbox_np], axis=0)
                                                else:
                                                    detections_with_masks.xyxy = torch.cat([detections_with_masks.xyxy, bbox_tensor.unsqueeze(0)], dim=0)
                                            
                                            if hasattr(detections_with_masks, 'mask') and detections_with_masks.mask is not None:
                                                detections_with_masks.mask = np.concatenate([detections_with_masks.mask, remain_bool_cleaned[None, :, :]], axis=0)
                                        
                                        object_names.append("remaining_area")
                        
                        except Exception as e:
                            print(f"VVL_GroundingDinoSAM2: Error processing remaining_area_mask for image {i+1}: {e}")
                    
                    # 解决重复名称
                    if object_names:
                        object_names = resolve_duplicate_names(object_names)
                    
                    # 验证数据一致性
                    verify_data_consistency(object_names, detections_with_masks, output_masks, image_index=i+1)
                    
                    # 添加到最终结果
                    final_object_names.extend(object_names)
                    
                    # 标注图像
                    if len(object_names) > 0 and detections_with_masks is not None:
                        labels = []
                        detection_count = len(detections_with_masks)
                        
                        for j in range(detection_count):
                            if j < len(object_names):
                                labels.append(object_names[j])
                            else:
                                labels.append(f"object_{j+1}")
                        
                        if not hasattr(detections_with_masks, 'data') or detections_with_masks.data is None:
                            detections_with_masks.data = {}
                        detections_with_masks.data['class_name'] = labels
                        
                        annotated_img = annotate_image(img_pil, detections_with_masks)
                        annotated_images.append(pil2tensor(annotated_img))
                    else:
                        annotated_images.append(pil2tensor(img_pil))
                    
                    # 添加masks到列表
                    if output_masks:
                        object_masks_list.extend(output_masks)
                    
                    # 创建检测JSON
                    detection_json = {
                        "image_width": img_pil.width,
                        "image_height": img_pil.height,
                        "objects": []
                    }
                    
                    if detections_with_masks is not None and hasattr(detections_with_masks, 'xyxy') and detections_with_masks.xyxy is not None:
                        for j, bbox in enumerate(detections_with_masks.xyxy):
                            bbox_2d = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                            obj_name = object_names[j] if j < len(object_names) else f"object_{j+1}"
                            detection_json["objects"].append({
                                "name": obj_name,
                                "bbox_2d": bbox_2d
                            })
                    
                    json_str = json.dumps(detection_json, ensure_ascii=False, indent=2)
                    json_str = re.sub(r'"bbox_2d":\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]', 
                                     r'"bbox_2d": [\1,\2,\3,\4]', json_str)
                    detection_jsons.append(json_str)
                    
                except Exception as e:
                    print(f"VVL_GroundingDinoSAM2: Error processing image {i+1}: {e}")
                    # 添加空结果以保持批次一致性
                    annotated_images.append(pil2tensor(tensor2pil(img_tensor).convert("RGB")))
                    detection_jsons.append(json.dumps({
                        "image_width": 512,
                        "image_height": 512,
                        "objects": []
                    }, ensure_ascii=False, indent=2))
            
            # 堆叠结果
            annotated_images_stacked = torch.stack(annotated_images) if annotated_images else torch.empty(0)
            final_detection_json = detection_jsons[0] if detection_jsons else "{}"
            
            print(f"VVL_GroundingDinoSAM2: Processing completed. Generated {len(object_masks_list)} masks")
            
            return (annotated_images_stacked, object_masks_list, final_detection_json, final_object_names)
            
        except Exception as e:
            print(f"VVL_GroundingDinoSAM2: Critical error: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回安全的默认结果
            batch_size = image.shape[0] if hasattr(image, 'shape') else 1
            empty_images = torch.zeros((batch_size, 512, 512, 3))
            empty_json = json.dumps({"image_width": 512, "image_height": 512, "objects": []})
            
            return (empty_images, [], empty_json, [])

# Node registration
NODE_CLASS_MAPPINGS = {
    "VVL_GroundingDinoSAM2": VVL_GroundingDinoSAM2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_GroundingDinoSAM2": "VVL GroundingDINO + SAM2"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 