import os
from typing import Tuple, Optional, Any, List

import cv2
# import gradio as gr # Gradio phones home, we don't want that
import numpy as np
# import spaces
import supervision as sv
import torch
from PIL import Image
from tqdm import tqdm
import gc
import copy # Added for deepcopy

import comfy.model_management as mm

try:
    from utils.video import generate_unique_name, create_directory, delete_directory

    from utils.florence import load_florence_model, run_florence_inference, \
        FLORENCE_OPEN_VOCABULARY_DETECTION_TASK, FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
    from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
    from utils.sam import load_sam_image_model, run_sam_inference, load_sam_video_model, model_to_config_map # Added model_to_config_map
except ImportError:
    # We're running as a module
    from .utils.video import generate_unique_name, create_directory, delete_directory

    from .utils.florence import load_florence_model, run_florence_inference, \
        FLORENCE_OPEN_VOCABULARY_DETECTION_TASK, FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
    from .utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
    from .utils.sam import load_sam_image_model, run_sam_inference, load_sam_video_model, model_to_config_map # Added model_to_config_map

# MARKDOWN = """
# # Florence2 + SAM2 🔥

# <div>
#     <a href="https://github.com/facebookresearch/segment-anything-2">
#         <img src="https://badges.aleen42.com/src/github.svg" alt="GitHub" style="display:inline-block;">
#     </a>
#     <a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-images-with-sam-2.ipynb">
#         <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" style="display:inline-block;">
#     </a>
#     <a href="https://blog.roboflow.com/what-is-segment-anything-2/">
#         <img src="https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg" alt="Roboflow" style="display:inline-block;">
#     </a>
#     <a href="https://www.youtube.com/watch?v=Dv003fTyO-Y">
#         <img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube" style="display:inline-block;">
#     </a>
# </div>

# This demo integrates Florence2 and SAM2 by creating a two-stage inference pipeline. In 
# the first stage, Florence2 performs tasks such as object detection, open-vocabulary 
# object detection, image captioning, or phrase grounding. In the second stage, SAM2 
# performs object segmentation on the image.
# """

# IMAGE_PROCESSING_EXAMPLES = [
#     [IMAGE_OPEN_VOCABULARY_DETECTION_MODE, "https://media.roboflow.com/notebooks/examples/dog-2.jpeg", 'straw, white napkin, black napkin, hair'],
#     [IMAGE_OPEN_VOCABULARY_DETECTION_MODE, "https://media.roboflow.com/notebooks/examples/dog-3.jpeg", 'tail'],
#     [IMAGE_CAPTION_GROUNDING_MASKS_MODE, "https://media.roboflow.com/notebooks/examples/dog-2.jpeg", None],
#     [IMAGE_CAPTION_GROUNDING_MASKS_MODE, "https://media.roboflow.com/notebooks/examples/dog-3.jpeg", None],
# ]
# VIDEO_PROCESSING_EXAMPLES = [
#     ["videos/clip-07-camera-1.mp4", "player in white outfit, player in black outfit, ball, rim"],
#     ["videos/clip-07-camera-2.mp4", "player in white outfit, player in black outfit, ball, rim"],
#     ["videos/clip-07-camera-3.mp4", "player in white outfit, player in black outfit, ball, rim"]
# ]

# VIDEO_SCALE_FACTOR = 0.5
# VIDEO_TARGET_DIRECTORY = "tmp"
# create_directory(directory_path=VIDEO_TARGET_DIRECTORY)

DEVICE = None #torch.device("cuda")
# DEVICE = torch.device("cpu")

if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

FLORENCE_MODEL, FLORENCE_PROCESSOR = None, None
SAM_IMAGE_MODEL = None
# SAM_VIDEO_MODEL = load_sam_video_model(device=DEVICE)
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX, thickness=2) # Default thickness
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS, # Changed from TOP_CENTER for better visibility
    text_color=sv.Color.from_hex("#000000"),
    text_scale=0.5, # Default scale
    text_thickness=1, # Default thickness
    text_padding=2, # Default padding
    border_radius=3 # Default radius
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    opacity=0.5 # Default opacity
)


# IoU计算和NMS函数
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


def annotate_image_pil(image_pil: Image.Image, detections: sv.Detections) -> Image.Image:
    image_np = np.array(image_pil.convert("RGB"))
    output_image_np = image_np.copy()

    if detections.mask is not None and len(detections.mask) > 0:
        output_image_np = MASK_ANNOTATOR.annotate(scene=output_image_np, detections=detections)
    
    if len(detections.xyxy) > 0:
        output_image_np = BOX_ANNOTATOR.annotate(scene=output_image_np, detections=detections)
        
        labels_to_annotate = []
        # Prefer 'label' if present, then 'class_name', then fallback
        label_source_key = None
        if 'label' in detections.data and detections.data['label'] is not None and len(detections.data['label']) == len(detections):
            label_source_key = 'label'
        elif 'class_name' in detections.data and detections.data['class_name'] is not None and len(detections.data['class_name']) == len(detections):
            label_source_key = 'class_name'
        
        if label_source_key:
            labels_to_annotate = [str(name) for name in detections.data[label_source_key]]
        else:
            labels_to_annotate = [f"obj_{i}" for i in range(len(detections))]

        if labels_to_annotate:
             output_image_np = LABEL_ANNOTATOR.annotate(scene=output_image_np, detections=detections, labels=labels_to_annotate)

    return Image.fromarray(output_image_np)


# def on_mode_dropdown_change(text):
#     return [
#         gr.Textbox(visible=text == IMAGE_OPEN_VOCABULARY_DETECTION_MODE),
#         gr.Textbox(visible=text == IMAGE_CAPTION_GROUNDING_MASKS_MODE),
#     ]

def lazy_load_models(device_target: torch.device, sam_image_model_name: str):
    global SAM_IMAGE_MODEL, loaded_sam_image_model, FLORENCE_MODEL, FLORENCE_PROCESSOR, DEVICE
    
    if DEVICE is None or device_target.type != DEVICE.type or (SAM_IMAGE_MODEL is not None and loaded_sam_image_model != sam_image_model_name):
        offload_models(delete_all=True) # Clear all if device or SAM model type changes
        DEVICE = device_target

    if SAM_IMAGE_MODEL is None or loaded_sam_image_model != sam_image_model_name:
        if SAM_IMAGE_MODEL is not None: # Unload previous SAM if different
            # SAM models might have their own unload logic or just del
            if hasattr(SAM_IMAGE_MODEL, 'unload_model'): SAM_IMAGE_MODEL.unload_model()
            elif hasattr(SAM_IMAGE_MODEL, 'to'): SAM_IMAGE_MODEL.to('cpu')
            del SAM_IMAGE_MODEL
            SAM_IMAGE_MODEL = None
            gc.collect()
            mm.soft_empty_cache()
        SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE, checkpoint=sam_image_model_name)
        loaded_sam_image_model = sam_image_model_name
    elif SAM_IMAGE_MODEL is not None: # Ensure it's on the correct device if already loaded
         SAM_IMAGE_MODEL.model.to(DEVICE)

    if FLORENCE_MODEL is None or FLORENCE_PROCESSOR is None:
        if FLORENCE_MODEL is not None: del FLORENCE_MODEL # Should already be offloaded by offload_models
        if FLORENCE_PROCESSOR is not None: del FLORENCE_PROCESSOR
        FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
    elif FLORENCE_MODEL is not None: # Ensure it's on the correct device
        FLORENCE_MODEL.to(DEVICE)

def offload_models(delete_all=False):
    global SAM_IMAGE_MODEL, FLORENCE_MODEL, FLORENCE_PROCESSOR, loaded_sam_image_model
    offload_device = mm.unet_offload_device()
    do_gc = False

    if SAM_IMAGE_MODEL is not None:
        if delete_all:
            if hasattr(SAM_IMAGE_MODEL, 'unload_model'): SAM_IMAGE_MODEL.unload_model()
            elif hasattr(SAM_IMAGE_MODEL, 'to'): SAM_IMAGE_MODEL.to('cpu')
            del SAM_IMAGE_MODEL
            SAM_IMAGE_MODEL = None
            loaded_sam_image_model = None
            do_gc = True
        else:
            SAM_IMAGE_MODEL.model.to(offload_device)
           
    if FLORENCE_MODEL is not None:
        if delete_all:
            FLORENCE_MODEL.to('cpu')
            del FLORENCE_MODEL
            FLORENCE_MODEL = None
            do_gc = True
        else:
            FLORENCE_MODEL.to(offload_device)
           
    if FLORENCE_PROCESSOR is not None and delete_all:
        del FLORENCE_PROCESSOR
        FLORENCE_PROCESSOR = None
        do_gc = True # Processor is small, but for consistency
    
    if do_gc:
        gc.collect()
    mm.soft_empty_cache()

def process_image_f2s2(device_target: torch.device, sam_image_model_name: str, image_pil: Image.Image, \
                        prompt_str: str, keep_model_loaded: bool, external_caption_str: str = "", \
                        nms_iou_threshold: float = 0.5, # New NMS threshold
                        ) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[list], Optional[Image.Image], Optional[dict]]:
    lazy_load_models(device_target, sam_image_model_name)

    prompt_clean = prompt_str.strip() if prompt_str else ""
    external_caption_clean = external_caption_str.strip() if external_caption_str else ""
    current_mode = None
    text_param_for_florence = None
    caption_for_grounding = None

    if prompt_clean: # Mode 1: Open Vocabulary Detection
        current_mode = IMAGE_OPEN_VOCABULARY_DETECTION_MODE
        text_param_for_florence = prompt_clean
    elif external_caption_clean: # Mode 2: Phrase Grounding with External Caption
        current_mode = IMAGE_CAPTION_GROUNDING_MASKS_MODE
        caption_for_grounding = external_caption_clean
        print(f"F2S2: Using external caption for grounding: {caption_for_grounding}")
    else: # Mode 3: Detailed Caption + Phrase Grounding
        current_mode = IMAGE_CAPTION_GROUNDING_MASKS_MODE
        # Florence will generate the caption internally in _process_image_enhanced_f2s2
        print("F2S2: No prompt or external caption; Florence-2 will generate detailed caption for grounding.")

    # _process_image_enhanced_f2s2 returns: annotated_image_pil, mask_list_np, final_detections_sv, final_object_names_list
    annotated_pil, object_masks_np_list, final_sv_detections_post_sam, final_object_names = _process_image_enhanced_f2s2(\
        current_mode, image_pil, text_param_for_florence, caption_for_grounding, nms_iou_threshold\
    )

    # Convert numpy masks to PIL images (grayscale) for object_masks output
    object_masks_pil_list = []
    if object_masks_np_list is not None and len(object_masks_np_list) > 0:
        for mask_np_single in object_masks_np_list:
            object_masks_pil_list.append(Image.fromarray((mask_np_single * 255).astype(np.uint8)).convert("L"))

    # Create merged mask for the masked_image output
    merged_mask_pil = None
    if object_masks_np_list and len(object_masks_np_list) > 0:
        combined_mask_np = np.any(np.array(object_masks_np_list), axis=0) if len(object_masks_np_list) > 1 else object_masks_np_list[0]
        merged_mask_pil = Image.fromarray((combined_mask_np * 255).astype(np.uint8)).convert("L")
    else: # No masks found, create an empty mask
        merged_mask_pil = Image.new("L", image_pil.size, 0)

    # Create the final masked_image (original image content where mask is white)
    masked_image_pil = Image.new("RGB", image_pil.size, (0, 0, 0))
    masked_image_pil.paste(image_pil, mask=merged_mask_pil)

    # Prepare detection_json from final_sv_detections_post_sam
    # These detections are post-NMS and post-SAM, so their xyxy might be slightly different
    # from pre-SAM boxes if SAM refines them, but names are from pre-SAM (NMS'd) stage.
    detection_json_output = {
        "image_width": image_pil.width,
        "image_height": image_pil.height,
        "objects": []
    }
    if final_sv_detections_post_sam is not None and len(final_sv_detections_post_sam) > 0:
        # Ensure object names are available, matching the length of detections
        names_for_json = final_object_names
        if len(names_for_json) != len(final_sv_detections_post_sam):
             # Fallback if names list doesn't match, though it should
            names_for_json = [f"obj_{k}" for k in range(len(final_sv_detections_post_sam))]

        for k, bbox_coords_np in enumerate(final_sv_detections_post_sam.xyxy):
            detection_json_output["objects"].append({
                "name": str(names_for_json[k]) if k < len(names_for_json) else f"object_{k+1}",
                "bbox_2d": [int(c) for c in bbox_coords_np]
            })

    if not keep_model_loaded:
        offload_models()

    return annotated_pil, merged_mask_pil, object_masks_pil_list, masked_image_pil, detection_json_output


@torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16) # Autocast at Florence/SAM level if needed
def _process_image_enhanced_f2s2(\
    mode: str, \
    image_input_pil: Image.Image, \
    text_input_str: Optional[str] = None, \
    custom_caption_for_grounding_str: Optional[str] = None,\
    nms_iou_thr: float = 0.5 # NMS IoU threshold
) -> Tuple[Optional[Image.Image], Optional[List[np.ndarray]], Optional[sv.Detections], Optional[List[str]]]:
    global SAM_IMAGE_MODEL, FLORENCE_MODEL, FLORENCE_PROCESSOR, DEVICE

    if not image_input_pil:
        return None, None, sv.Detections.empty(), []
       
    detections_pre_sam = sv.Detections.empty()
    object_names_pre_sam = []

    if mode == IMAGE_OPEN_VOCABULARY_DETECTION_MODE:
        if not text_input_str:
            print("F2S2: OVD mode requires a text prompt.")
            return annotate_image_pil(image_input_pil, sv.Detections.empty()), [], sv.Detections.empty(), []
           
        phrases = [p.strip() for p in text_input_str.split(",") if p.strip()]
        if not phrases and text_input_str: phrases = [text_input_str.strip()]

        all_phrase_detections = []
        for phrase in phrases:
            _, florence_result = run_florence_inference(\
                model=FLORENCE_MODEL, processor=FLORENCE_PROCESSOR, device=DEVICE,\
                image=image_input_pil, task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK, text=phrase\
            )
            # sv.Detections.from_lmm populates xyxy, confidence, and potentially data['label']
            phrase_detections = sv.Detections.from_lmm(\
                lmm=sv.LMM.FLORENCE_2, result=florence_result, resolution_wh=image_input_pil.size\
            )
            # Store the original phrase with these detections before merging/NMS
            if len(phrase_detections) > 0:
                if not hasattr(phrase_detections, 'data') or phrase_detections.data is None: phrase_detections.data = {}
                phrase_detections.data['original_phrase'] = [phrase] * len(phrase_detections)
            all_phrase_detections.append(phrase_detections)

        if all_phrase_detections:
            detections_pre_sam = sv.Detections.merge(all_phrase_detections)
            # Extract original phrases for NMS (to keep them associated)
            if 'original_phrase' in detections_pre_sam.data and detections_pre_sam.data['original_phrase'] is not None:
                object_names_pre_sam = list(detections_pre_sam.data['original_phrase'])
            else: # Fallback if merge didn't preserve, or single phrase
                object_names_pre_sam = [phrases[0]] * len(detections_pre_sam) if len(phrases)==1 else []
                # This fallback needs improvement if original_phrase isn't consistently there post-merge.
                # For now, assume simple cases or that 'label' might be populated by from_lmm.
                if not object_names_pre_sam and 'label' in detections_pre_sam.data:
                    object_names_pre_sam = list(detections_pre_sam.data['label'])

    elif mode == IMAGE_CAPTION_GROUNDING_MASKS_MODE:
        caption_to_use = custom_caption_for_grounding_str
        if not caption_to_use:
            print("F2S2: Generating detailed caption with Florence-2 for grounding.")
            _, florence_cap_result = run_florence_inference(\
                model=FLORENCE_MODEL, processor=FLORENCE_PROCESSOR, device=DEVICE,\
                image=image_input_pil, task=FLORENCE_DETAILED_CAPTION_TASK\
            )
            caption_to_use = florence_cap_result[FLORENCE_DETAILED_CAPTION_TASK]
            print(f"F2S2: Generated caption for grounding: {caption_to_use}")
        else:
            print(f"F2S2: Using provided caption for grounding: {caption_to_use}")
        
        _, florence_ground_result = run_florence_inference(\
            model=FLORENCE_MODEL, processor=FLORENCE_PROCESSOR, device=DEVICE,\
            image=image_input_pil, task=FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, text=caption_to_use\
        )
        detections_pre_sam = sv.Detections.from_lmm(\
            lmm=sv.LMM.FLORENCE_2, result=florence_ground_result, resolution_wh=image_input_pil.size\
        )
        # `from_lmm` for grounding task should populate `data['label']` with the grounded phrases
        if 'label' in detections_pre_sam.data and detections_pre_sam.data['label'] is not None:
            object_names_pre_sam = list(detections_pre_sam.data['label'])
        else:
            object_names_pre_sam = [f"grounded_obj_{k}" for k in range(len(detections_pre_sam))]
    else:
        print(f"F2S2: Unsupported mode: {mode}")
        return annotate_image_pil(image_input_pil, sv.Detections.empty()), [], sv.Detections.empty(), []
    
    # --- Apply NMS to detections_pre_sam ---
    detections_after_nms = sv.Detections.empty()
    object_names_after_nms = []

    if len(detections_pre_sam) > 0:
        # Ensure confidence is present for NMS; if not, assign a default (e.g., 1.0)
        if detections_pre_sam.confidence is None:
            print("F2S2: Warning - Detections lack confidence scores for NMS. Assigning default 1.0.")
            detections_pre_sam.confidence = np.ones(len(detections_pre_sam))
        
        # Preserve original names/labels through NMS by temporarily storing them in data
        # The `non_max_suppression` method in supervision might not preserve all custom data fields
        # depending on version or if class_agnostic=False is used with class_id. 
        # Simplest is to re-associate after NMS based on indices if class_agnostic=True.
        if not object_names_pre_sam and 'label' in detections_pre_sam.data: # Backup
             object_names_pre_sam = list(detections_pre_sam.data['label'])
        elif not object_names_pre_sam:
            object_names_pre_sam = [f"obj_{k}" for k in range(len(detections_pre_sam))]

        # Store names before NMS, as NMS might only return indices or a subset of Detections
        temp_detections_for_nms = copy.deepcopy(detections_pre_sam) # NMS might modify in place or return new
        if not hasattr(temp_detections_for_nms, 'data') or temp_detections_for_nms.data is None: temp_detections_for_nms.data = {}
        temp_detections_for_nms.data['_nms_original_names'] = np.array(object_names_pre_sam)

        print(f"F2S2: Detections before NMS: {len(temp_detections_for_nms)}")
        # 使用supervision库的正确NMS方法
        try:
            # 尝试新版本的API
            detections_after_nms = temp_detections_for_nms.with_nms(threshold=nms_iou_thr, class_agnostic=True)
            # 获取保留的原始名称
            if hasattr(detections_after_nms, 'data') and '_nms_original_names' in detections_after_nms.data:
                object_names_after_nms = list(detections_after_nms.data['_nms_original_names'])
            else:
                # 如果NMS后data丢失，重新赋值
                if len(detections_after_nms) <= len(object_names_pre_sam):
                    object_names_after_nms = object_names_pre_sam[:len(detections_after_nms)]
                else:
                    object_names_after_nms = [f"obj_{k}" for k in range(len(detections_after_nms))]
        except AttributeError:
            # 如果with_nms也不存在，使用手动NMS实现
            print("F2S2: Using manual NMS implementation")
            
            # 转换为torch tensor格式
            boxes_tensor = torch.from_numpy(temp_detections_for_nms.xyxy)
            filtered_boxes, filtered_names = remove_duplicate_boxes(boxes_tensor, object_names_pre_sam, nms_iou_thr)
            
            # 转换回supervision格式
            if len(filtered_boxes) > 0:
                detections_after_nms = sv.Detections(
                    xyxy=filtered_boxes.numpy(),
                    confidence=temp_detections_for_nms.confidence[:len(filtered_boxes)] if temp_detections_for_nms.confidence is not None else None
                )
                object_names_after_nms = filtered_names
            else:
                detections_after_nms = sv.Detections.empty()
                object_names_after_nms = []
        print(f"F2S2: Detections after NMS: {len(detections_after_nms)}")
    else:
        print("F2S2: No detections from Florence-2 to process with NMS/SAM.")

    # --- SAM2 Segmentation on NMS'd boxes ---
    final_detections_with_masks_sv = sv.Detections.empty() # This will hold xyxy, masks, and names
    individual_masks_np_list = [] # List of numpy bool masks

    if len(detections_after_nms) > 0:
        # run_sam_inference expects PIL image and sv.Detections(xyxy=...) It returns sv.Detections with .mask populated.
        final_detections_with_masks_sv = run_sam_inference(SAM_IMAGE_MODEL, image_input_pil, detections_after_nms)
        
        if final_detections_with_masks_sv.mask is not None and len(final_detections_with_masks_sv.mask) > 0:
            individual_masks_np_list = [mask_np for mask_np in final_detections_with_masks_sv.mask]
        
        # Ensure original (NMS'd) names are carried to the final detections object for annotation
        if not hasattr(final_detections_with_masks_sv, 'data') or final_detections_with_masks_sv.data is None: 
            final_detections_with_masks_sv.data = {}
        final_detections_with_masks_sv.data['label'] = np.array(object_names_after_nms) # Use 'label' as key for consistency

    # --- Annotation ---
    # Annotate using the detections that have masks and the correct NMS'd names
    annotated_image_output_pil = annotate_image_pil(image_input_pil, final_detections_with_masks_sv)
    
    return annotated_image_output_pil, individual_masks_np_list, final_detections_with_masks_sv, object_names_after_nms


# Backward compatibility wrapper function
def process_image(device_target: torch.device, sam_image_model_name: str, image_pil: Image.Image, 
                 prompt_str: str, keep_model_loaded: bool, external_caption_str: str = "", 
                 nms_iou_threshold: float = 0.5) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[list], Optional[Image.Image], Optional[dict]]:
    """
    Backward compatibility wrapper for process_image_f2s2 with IoU threshold support.
    """
    return process_image_f2s2(
        device_target=device_target,
        sam_image_model_name=sam_image_model_name,
        image_pil=image_pil,
        prompt_str=prompt_str,
        keep_model_loaded=keep_model_loaded,
        external_caption_str=external_caption_str,
        nms_iou_threshold=nms_iou_threshold
    )


# @spaces.GPU(duration=300)
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
# def process_video(
#     video_input, text_input, progress=gr.Progress(track_tqdm=True)
# ) -> Optional[str]:
#     if not video_input:
#         gr.Info("Please upload a video.")
#         return None

#     if not text_input:
#         gr.Info("Please enter a text prompt.")
#         return None

#     frame_generator = sv.get_video_frames_generator(video_input)
#     frame = next(frame_generator)
#     frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     texts = [prompt.strip() for prompt in text_input.split(",")]
#     detections_list = []
#     for text in texts:
#         _, result = run_florence_inference(
#             model=FLORENCE_MODEL,
#             processor=FLORENCE_PROCESSOR,
#             device=DEVICE,
#             image=frame,
#             task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
#             text=text
#         )
#         detections = sv.Detections.from_lmm(
#             lmm=sv.LMM.FLORENCE_2,
#             result=result,
#             resolution_wh=frame.size
#         )
#         detections = run_sam_inference(SAM_IMAGE_MODEL, frame, detections)
#         detections_list.append(detections)

#     detections = sv.Detections.merge(detections_list)
#     detections = run_sam_inference(SAM_IMAGE_MODEL, frame, detections)

#     if len(detections.mask) == 0:
#         gr.Info(
#             "No objects of class {text_input} found in the first frame of the video. "
#             "Trim the video to make the object appear in the first frame or try a "
#             "different text prompt."
#         )
#         return None

#     name = generate_unique_name()
#     frame_directory_path = os.path.join(VIDEO_TARGET_DIRECTORY, name)
#     frames_sink = sv.ImageSink(
#         target_dir_path=frame_directory_path,
#         image_name_pattern="{:05d}.jpeg"
#     )

#     video_info = sv.VideoInfo.from_video_path(video_input)
#     video_info.width = int(video_info.width * VIDEO_SCALE_FACTOR)
#     video_info.height = int(video_info.height * VIDEO_SCALE_FACTOR)

#     frames_generator = sv.get_video_frames_generator(video_input)
#     with frames_sink:
#         for frame in tqdm(
#                 frames_generator,
#                 total=video_info.total_frames,
#                 desc="splitting video into frames"
#         ):
#             frame = sv.scale_image(frame, VIDEO_SCALE_FACTOR)
#             frames_sink.save_image(frame)

#     inference_state = SAM_VIDEO_MODEL.init_state(
#         video_path=frame_directory_path,
#         device=DEVICE
#     )

#     for mask_index, mask in enumerate(detections.mask):
#         _, object_ids, mask_logits = SAM_VIDEO_MODEL.add_new_mask(
#             inference_state=inference_state,
#             frame_idx=0,
#             obj_id=mask_index,
#             mask=mask
#         )

#     video_path = os.path.join(VIDEO_TARGET_DIRECTORY, f"{name}.mp4")
#     frames_generator = sv.get_video_frames_generator(video_input)
#     masks_generator = SAM_VIDEO_MODEL.propagate_in_video(inference_state)
#     with sv.VideoSink(video_path, video_info=video_info) as sink:
#         for frame, (_, tracker_ids, mask_logits) in zip(frames_generator, masks_generator):
#             frame = sv.scale_image(frame, VIDEO_SCALE_FACTOR)
#             masks = (mask_logits > 0.0).cpu().numpy().astype(bool)
#             if len(masks.shape) == 4:
#                 masks = np.squeeze(masks, axis=1)

#             detections = sv.Detections(
#                 xyxy=sv.mask_to_xyxy(masks=masks),
#                 mask=masks,
#                 class_id=np.array(tracker_ids)
#             )
#             annotated_frame = frame.copy()
#             annotated_frame = MASK_ANNOTATOR.annotate(
#                 scene=annotated_frame, detections=detections)
#             annotated_frame = BOX_ANNOTATOR.annotate(
#                 scene=annotated_frame, detections=detections)
#             sink.write_frame(annotated_frame)

#     delete_directory(frame_directory_path)
#     return video_path


# with gr.Blocks() as demo:
#     gr.Markdown(MARKDOWN)
#     with gr.Tab("Image"):
#         image_processing_mode_dropdown_component = gr.Dropdown(
#             choices=IMAGE_INFERENCE_MODES,
#             value=IMAGE_INFERENCE_MODES[0],
#             label="Mode",
#             info="Select a mode to use.",
#             interactive=True
#         )
#         with gr.Row():
#             with gr.Column():
#                 image_processing_image_input_component = gr.Image(
#                     type='pil', label='Upload image')
#                 image_processing_text_input_component = gr.Textbox(
#                     label='Text prompt',
#                     placeholder='Enter comma separated text prompts')
#                 image_processing_submit_button_component = gr.Button(
#                     value='Submit', variant='primary')
#             with gr.Column():
#                 image_processing_image_output_component = gr.Image(
#                     type='pil', label='Image output')
#                 image_processing_text_output_component = gr.Textbox(
#                     label='Caption output', visible=False)

#         with gr.Row():
#             gr.Examples(
#                 fn=process_image,
#                 examples=IMAGE_PROCESSING_EXAMPLES,
#                 inputs=[
#                     image_processing_mode_dropdown_component,
#                     image_processing_image_input_component,
#                     image_processing_text_input_component
#                 ],
#                 outputs=[
#                     image_processing_image_output_component,
#                     image_processing_text_output_component
#                 ],
#                 run_on_click=True
#             )
#     with gr.Tab("Video"):
#         video_processing_mode_dropdown_component = gr.Dropdown(
#             choices=VIDEO_INFERENCE_MODES,
#             value=VIDEO_INFERENCE_MODES[0],
#             label="Mode",
#             info="Select a mode to use.",
#             interactive=True
#         )
#         with gr.Row():
#             with gr.Column():
#                 video_processing_video_input_component = gr.Video(
#                     label='Upload video')
#                 video_processing_text_input_component = gr.Textbox(
#                     label='Text prompt',
#                     placeholder='Enter comma separated text prompts')
#                 video_processing_submit_button_component = gr.Button(
#                     value='Submit', variant='primary')
#             with gr.Column():
#                 video_processing_video_output_component = gr.Video(
#                     label='Video output')
#         with gr.Row():
#             gr.Examples(
#                 fn=process_video,
#                 examples=VIDEO_PROCESSING_EXAMPLES,
#                 inputs=[
#                     video_processing_video_input_component,
#                     video_processing_text_input_component
#                 ],
#                 outputs=video_processing_video_output_component,
#                 run_on_click=True
#             )

#     image_processing_submit_button_component.click(
#         fn=process_image,
#         inputs=[
#             image_processing_mode_dropdown_component,
#             image_processing_image_input_component,
#             image_processing_text_input_component
#         ],
#         outputs=[
#             image_processing_image_output_component,
#             image_processing_text_output_component
#         ]
#     )
#     image_processing_text_input_component.submit(
#         fn=process_image,
#         inputs=[
#             image_processing_mode_dropdown_component,
#             image_processing_image_input_component,
#             image_processing_text_input_component
#         ],
#         outputs=[
#             image_processing_image_output_component,
#             image_processing_text_output_component
#         ]
#     )
#     image_processing_mode_dropdown_component.change(
#         on_mode_dropdown_change,
#         inputs=[image_processing_mode_dropdown_component],
#         outputs=[
#             image_processing_text_input_component,
#             image_processing_text_output_component
#         ]
#     )
#     video_processing_submit_button_component.click(
#         fn=process_video,
#         inputs=[
#             video_processing_video_input_component,
#             video_processing_text_input_component
#         ],
#         outputs=video_processing_video_output_component
#     )
#     video_processing_text_input_component.submit(
#         fn=process_video,
#         inputs=[
#             video_processing_video_input_component,
#             video_processing_text_input_component
#         ],
#         outputs=video_processing_video_output_component
#     )

# demo.launch(debug=False, show_error=True)
