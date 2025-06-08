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
    from grounding_dino_sam2 import (
        tensor2pil, pil2tensor, load_groundingdino_model, groundingdino_predict,
        annotate_image, resolve_duplicate_names, verify_data_consistency,
        calculate_iou, remove_duplicate_boxes, calculate_mask_containment_ratio,
        remove_duplicate_masks_by_containment, filter_by_area, sam2_segment,
        groundingdino_model_list
    )
except ImportError:
    # We're running as a module
    from .florence_sam_processor import process_image
    from .utils.sam import model_to_config_map as sam_model_to_config_map
    from .utils.sam import load_sam_image_model, run_sam_inference
    from .utils.florence import load_florence_model, run_florence_inference, \
        FLORENCE_OPEN_VOCABULARY_DETECTION_TASK, FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK
    from .utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
    from .mask_cleaner import remove_small_regions
    from .grounding_dino_sam2 import (
        tensor2pil, pil2tensor, load_groundingdino_model, groundingdino_predict,
        annotate_image, resolve_duplicate_names, verify_data_consistency,
        calculate_iou, remove_duplicate_boxes, calculate_mask_containment_ratio,
        remove_duplicate_masks_by_containment, filter_by_area, sam2_segment,
        groundingdino_model_list
    )

import logging
import comfy.model_management

logger = logging.getLogger('vvl_GroundingDinoSAM2_Video')

# 导入原始模块的全局变量和函数，而不是重新定义
try:
    import grounding_dino_sam2 as gd_sam2_module
except ImportError:
    from . import grounding_dino_sam2 as gd_sam2_module


class VVL_GroundingDinoSAM2_VideoSequence:
    """
    基于GroundingDINO + SAM2的视频序列处理节点
    输入图片序列，输出带有分割结果的图片序列
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        grounding_dino_models = list(groundingdino_model_list.keys())
        
        return {
            "required": {
                "sam2_model": ("VVL_SAM2_MODEL", {"tooltip": "SAM2分割模型，用于对检测到的对象进行精确分割"}),
                "grounding_dino_model": (grounding_dino_models, {
                    "default": grounding_dino_models[0],
                    "tooltip": "GroundingDINO目标检测模型，用于根据文本提示检测图像中的对象"
                }),
                "image_sequence": ("IMAGE", {"tooltip": "输入的图像序列，用于视频处理。支持批量图像"}),
                "prompt": ("STRING", {
                    "default": "",
                    "tooltip": "目标检测的文本提示词，用逗号分隔多个对象，如'person,car,dog'。留空时将使用Florence-2自动生成描述"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "目标检测的置信度阈值，建议范围0.2-0.5"
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "IoU阈值用于去除重复检测框，建议0.3-0.7"
                }),
                "consistency_threshold": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "帧间一致性阈值，用于保持相同对象在不同帧中的标识一致。值越高要求越严格"
                }),
            },
            "optional": {
                "external_caption": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "tooltip": "外部提供的图像描述文本，用逗号分隔多个对象"
                }),
                "load_florence2": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否加载Florence-2模型用于自动生成图像描述"
                }),
                "min_area_ratio": ("FLOAT", {
                    "default": 0.002, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.0001, 
                    "tooltip": "最小面积比例，用于过滤太小的分割结果"
                }),
                "max_area_ratio": ("FLOAT", {
                    "default": 0.2, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "tooltip": "最大面积比例，用于过滤太大的分割结果"
                }),
                "use_reference_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否使用第一帧作为参考帧来保持对象标识的一致性"
                }),
                "mask_containment_threshold": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "基于实际mask形状的包含阈值，用于移除被其他mask包含的重复分割。设为0时禁用此功能"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING",)
    RETURN_NAMES = ("annotated_sequence", "sequence_masks", "sequence_info", "sequence_object_names",)
    FUNCTION = "_process_video_sequence"
    CATEGORY = "💃rDancer"
    OUTPUT_IS_LIST = (False, True, False, True)

    def calculate_bbox_similarity(self, bbox1, bbox2, image_size):
        """计算两个边界框的相似度"""
        # 归一化坐标
        w, h = image_size
        bbox1_norm = [bbox1[0]/w, bbox1[1]/h, bbox1[2]/w, bbox1[3]/h]
        bbox2_norm = [bbox2[0]/w, bbox2[1]/h, bbox2[2]/w, bbox2[3]/h]
        
        # 计算中心点距离
        center1 = [(bbox1_norm[0] + bbox1_norm[2])/2, (bbox1_norm[1] + bbox1_norm[3])/2]
        center2 = [(bbox2_norm[0] + bbox2_norm[2])/2, (bbox2_norm[1] + bbox2_norm[3])/2]
        center_dist = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        
        # 计算面积相似度
        area1 = (bbox1_norm[2] - bbox1_norm[0]) * (bbox1_norm[3] - bbox1_norm[1])
        area2 = (bbox2_norm[2] - bbox2_norm[0]) * (bbox2_norm[3] - bbox2_norm[1])
        area_similarity = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # 计算IoU
        iou = calculate_iou(bbox1, bbox2)
        
        # 综合相似度 (IoU权重最高)
        similarity = 0.6 * iou + 0.2 * area_similarity + 0.2 * (1 - min(center_dist, 1.0))
        return similarity

    def match_objects_across_frames(self, reference_detections, current_detections, 
                                   reference_names, current_names, image_size, threshold=0.5):
        """在帧间匹配对象，保持标识一致性"""
        if reference_detections is None or current_detections is None:
            return current_names, list(range(len(current_names))) if current_names else []
        
        ref_boxes = reference_detections.xyxy if hasattr(reference_detections, 'xyxy') else []
        cur_boxes = current_detections.xyxy if hasattr(current_detections, 'xyxy') else []
        
        if len(ref_boxes) == 0 or len(cur_boxes) == 0:
            return current_names, list(range(len(current_names))) if current_names else []
        
        # 计算相似度矩阵
        similarity_matrix = np.zeros((len(ref_boxes), len(cur_boxes)))
        for i, ref_box in enumerate(ref_boxes):
            for j, cur_box in enumerate(cur_boxes):
                similarity_matrix[i, j] = self.calculate_bbox_similarity(ref_box, cur_box, image_size)
        
        # 匹配对象（使用贪心算法）
        matched_names = current_names.copy()
        matched_indices = list(range(len(current_names)))
        used_ref_indices = set()
        
        # 按相似度从高到低排序
        pairs = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                if similarity_matrix[i, j] > threshold:
                    pairs.append((i, j, similarity_matrix[i, j]))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 执行匹配
        for ref_idx, cur_idx, similarity in pairs:
            if ref_idx not in used_ref_indices and cur_idx < len(matched_names):
                # 使用参考帧的对象名称
                if ref_idx < len(reference_names):
                    matched_names[cur_idx] = reference_names[ref_idx]
                    matched_indices[cur_idx] = ref_idx
                    used_ref_indices.add(ref_idx)
        
        return matched_names, matched_indices

    def _process_video_sequence(self, sam2_model: dict, grounding_dino_model: str, 
                               image_sequence: torch.Tensor, prompt: str = "", 
                               threshold: float = 0.3, iou_threshold: float = 0.5, 
                               consistency_threshold: float = 0.7, external_caption: str = "", 
                               load_florence2: bool = True, min_area_ratio: float = 0.002, 
                               max_area_ratio: float = 0.2, use_reference_frame: bool = True,
                               mask_containment_threshold: float = 0.8):
        
        # 加载模型
        gd_sam2_module.lazy_load_grounding_dino_florence_models(grounding_dino_model, load_florence2)
        
        # 从SAM2模型字典中获取模型和设备信息
        sam2_model_instance = sam2_model['model']
        device = sam2_model['device']
        
        sequence_length = image_sequence.shape[0]
        print(f"VVL_GroundingDinoSAM2_VideoSequence: 处理视频序列，共 {sequence_length} 帧")
        
        annotated_images = []
        sequence_masks_list = []
        sequence_object_names = []
        
        # 参考帧信息（用于保持对象一致性）
        reference_detections = None
        reference_names = []
        reference_frame_idx = 0
        
        # 序列信息统计
        total_objects_detected = 0
        frames_with_objects = 0
        
        for i, img_tensor in enumerate(image_sequence):
            img_pil = tensor2pil(img_tensor).convert("RGB")
            print(f"VVL_GroundingDinoSAM2_VideoSequence: 处理第 {i+1}/{sequence_length} 帧")
            
            # 使用与单帧处理相同的逻辑确定检测短语
            prompt_clean = prompt.strip() if prompt else ""
            external_caption_clean = external_caption.strip() if external_caption else ""
            
            current_detection_phrases = []
            
            if prompt_clean:
                current_detection_phrases = [p.strip() for p in prompt_clean.split(',') if p.strip()]
                if not current_detection_phrases and prompt_clean:
                    current_detection_phrases = [prompt_clean.strip()]
            elif external_caption_clean:
                current_detection_phrases = [c.strip() for c in external_caption_clean.split(',') if c.strip()]
                if not current_detection_phrases and external_caption_clean:
                    current_detection_phrases = [external_caption_clean.strip()]
            else:
                # 仅在第一帧或参考帧生成描述，后续帧复用
                if i == 0 and load_florence2 and gd_sam2_module.FLORENCE_MODEL is not None and gd_sam2_module.FLORENCE_PROCESSOR is not None:
                    print(f"VVL_GroundingDinoSAM2_VideoSequence: 第 {i+1} 帧 - 使用Florence-2生成描述")
                    _, result_caption = run_florence_inference(
                        model=gd_sam2_module.FLORENCE_MODEL,
                        processor=gd_sam2_module.FLORENCE_PROCESSOR,
                        device=device,
                        image=img_pil,
                        task=FLORENCE_DETAILED_CAPTION_TASK
                    )
                    generated_caption = result_caption[FLORENCE_DETAILED_CAPTION_TASK]
                    current_detection_phrases = [c.strip() for c in generated_caption.split(',') if c.strip()]
                    if not current_detection_phrases and generated_caption:
                        current_detection_phrases = [generated_caption.strip()]
                elif i > 0 and reference_names:
                    # 后续帧使用参考帧的对象名称
                    current_detection_phrases = list(set(reference_names))
            
            # GroundingDINO检测
            object_names = []
            all_boxes_list = []
            
            if current_detection_phrases:
                # 验证模型是否正确加载
                if gd_sam2_module.GROUNDING_DINO_MODEL is None:
                    print(f"VVL_GroundingDinoSAM2_VideoSequence: 错误 - GROUNDING_DINO_MODEL是None!")
                    raise ValueError("GroundingDINO模型未正确加载")
                
                for phrase in current_detection_phrases:
                    print(f"VVL_GroundingDinoSAM2_VideoSequence: 第 {i+1} 帧 - 检测短语: '{phrase}'")
                    boxes_single = groundingdino_predict(gd_sam2_module.GROUNDING_DINO_MODEL, img_pil, phrase, threshold)
                    if boxes_single.shape[0] > 0:
                        all_boxes_list.append(boxes_single)
                        object_names.extend([phrase] * boxes_single.shape[0])
                        print(f"VVL_GroundingDinoSAM2_VideoSequence: 第 {i+1} 帧 - 短语 '{phrase}' 检测到 {boxes_single.shape[0]} 个框")
                    else:
                        print(f"VVL_GroundingDinoSAM2_VideoSequence: 第 {i+1} 帧 - 短语 '{phrase}' 未检测到对象")
                
                if len(all_boxes_list) > 0:
                    boxes = torch.cat(all_boxes_list, dim=0)
                else:
                    boxes = torch.zeros((0,4))
            else:
                boxes = torch.zeros((0,4))
            
            # 去重检测框
            if boxes.shape[0] > 0:
                boxes, object_names = remove_duplicate_boxes(boxes, object_names, iou_threshold)
            
            if boxes.shape[0] == 0:
                print(f"VVL_GroundingDinoSAM2_VideoSequence: 第 {i+1} 帧 - 未检测到对象")
                annotated_images.append(pil2tensor(img_pil))
                continue
            
            # SAM2分割
            output_images, output_masks, detections_with_masks = sam2_segment(sam2_model_instance, img_pil, boxes)
            
            # 基于实际mask形状去除重复分割（在面积过滤之前进行）
            if output_masks and len(output_masks) > 1 and mask_containment_threshold > 0:
                print(f"VVL_GroundingDinoSAM2_VideoSequence: 第 {i+1} 帧 - 应用基于mask的去重逻辑 (阈值: {mask_containment_threshold})")
                output_images, output_masks, detections_with_masks, object_names = remove_duplicate_masks_by_containment(
                    output_images, output_masks, detections_with_masks, object_names, containment_threshold=mask_containment_threshold
                )
            
            # 面积过滤
            if output_masks and (min_area_ratio > 0 or max_area_ratio < 1.0):
                output_images, output_masks, detections_with_masks, object_names = filter_by_area(
                    output_images, output_masks, detections_with_masks, object_names, 
                    (img_pil.width, img_pil.height), min_area_ratio, max_area_ratio
                )
            
            if not output_masks:
                print(f"VVL_GroundingDinoSAM2_VideoSequence: 第 {i+1} 帧 - 所有分割结果被过滤")
                annotated_images.append(pil2tensor(img_pil))
                continue
            
            # 帧间对象匹配（保持一致性）
            if use_reference_frame and reference_detections is not None:
                matched_names, matched_indices = self.match_objects_across_frames(
                    reference_detections, detections_with_masks, reference_names, 
                    object_names, (img_pil.width, img_pil.height), consistency_threshold
                )
                object_names = matched_names
                matched_count = len([idx for idx in matched_indices if idx < len(reference_names)])
                print(f"VVL_GroundingDinoSAM2_VideoSequence: 第 {i+1} 帧 - 对象匹配完成，匹配到 {matched_count} 个已知对象")
            
            # 设置参考帧
            if use_reference_frame and (reference_detections is None or i == reference_frame_idx):
                reference_detections = detections_with_masks
                reference_names = object_names.copy()
                print(f"VVL_GroundingDinoSAM2_VideoSequence: 第 {i+1} 帧设为参考帧，包含 {len(reference_names)} 个对象")
            
            # 解决对象名称重复问题
            object_names = resolve_duplicate_names(object_names)
            
            # 验证数据一致性
            verify_data_consistency(object_names, detections_with_masks, output_masks, image_index=i)
            
            # 统计信息
            total_objects_detected += len(object_names)
            frames_with_objects += 1
            
            # 图像标注
            if len(object_names) > 0 and detections_with_masks is not None:
                labels = object_names[:len(detections_with_masks)]
                
                if not hasattr(detections_with_masks, 'data') or detections_with_masks.data is None:
                    detections_with_masks.data = {}
                detections_with_masks.data['class_name'] = labels
                
                annotated_img = annotate_image(img_pil, detections_with_masks)
                annotated_images.append(pil2tensor(annotated_img))
            else:
                annotated_images.append(pil2tensor(img_pil))
            
            # 添加masks和对象名称
            if output_masks:
                sequence_masks_list.extend(output_masks)
            sequence_object_names.extend(object_names)
            
            print(f"VVL_GroundingDinoSAM2_VideoSequence: 第 {i+1} 帧处理完成，检测到 {len(object_names)} 个对象")
        
        # 生成序列信息
        avg_objects_per_frame = total_objects_detected / frames_with_objects if frames_with_objects > 0 else 0
        unique_object_names = list(set(sequence_object_names))
        
        sequence_info = {
            "total_frames": sequence_length,
            "frames_with_objects": frames_with_objects,
            "total_objects_detected": total_objects_detected,
            "average_objects_per_frame": round(avg_objects_per_frame, 2),
            "unique_object_types": len(unique_object_names),
            "object_types": unique_object_names,
            "processing_settings": {
                "grounding_dino_model": grounding_dino_model,
                "threshold": threshold,
                "iou_threshold": iou_threshold,
                "consistency_threshold": consistency_threshold,
                "use_reference_frame": use_reference_frame,
                "mask_containment_threshold": mask_containment_threshold,
                "min_area_ratio": min_area_ratio,
                "max_area_ratio": max_area_ratio
            }
        }
        
        sequence_info_json = json.dumps(sequence_info, ensure_ascii=False, indent=2)
        
        # 堆叠结果
        annotated_images_stacked = torch.stack(annotated_images) if annotated_images else torch.empty(0)
        
        print(f"VVL_GroundingDinoSAM2_VideoSequence: 序列处理完成")
        print(f"  - 总帧数: {sequence_length}")
        print(f"  - 有对象的帧数: {frames_with_objects}")
        print(f"  - 检测到的对象总数: {total_objects_detected}")
        print(f"  - 平均每帧对象数: {avg_objects_per_frame:.2f}")
        print(f"  - 唯一对象类型: {len(unique_object_names)}")
        print(f"  - 对象类型: {unique_object_names}")
        
        return (annotated_images_stacked, sequence_masks_list, sequence_info_json, sequence_object_names)


# Node registration
NODE_CLASS_MAPPINGS = {
    "VVL_GroundingDinoSAM2_VideoSequence": VVL_GroundingDinoSAM2_VideoSequence
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_GroundingDinoSAM2_VideoSequence": "VVL GroundingDINO + SAM2 视频序列"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 