import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import json
import supervision as sv  # 添加supervision库用于图像标注

try:
    from segment_anything import SamAutomaticMaskGenerator
except ImportError as e:
    raise ImportError("segment_anything 库未安装，请执行 pip install git+https://github.com/facebookresearch/segment-anything.git") from e

from ..utils.panoptic_utils import mask_to_bbox, build_detection_json

# 全局缓存 generator，提高批量速度
_AUTO_GENERATORS = {}

# 定义颜色调色板和mask标注器（参考VVL_GroundingDinoSAM2）
COLORS_HEX = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2', '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB', '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS_HEX)
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
    """标注图像函数（参考VVL_GroundingDinoSAM2）"""
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image

def _get_generator(sam_model, **kwargs):
    key = id(sam_model)
    if key not in _AUTO_GENERATORS:
        _AUTO_GENERATORS[key] = SamAutomaticMaskGenerator(sam_model, **kwargs)
    return _AUTO_GENERATORS[key]

def tensor2pil(t_image: torch.Tensor) -> Image.Image:
    """转换tensor到PIL图像"""
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """转换PIL图像到tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)

class SAM1AutoEverything:
    """SAM1 AutomaticMaskGenerator 一键分割节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam1_model": ("SAM_MODEL", {"tooltip": "由 SAMModelLoader 加载的 SAM1 模型（sam_vit_h / sam_vit_l / sam_vit_b 或 sam_hq_vit_* 系列）"}),
                "image": ("IMAGE", {"tooltip": "输入图像，支持批量处理"}),
            },
            "optional": {
                "points_per_side": ("INT", {"default": 32, "min": 4, "max": 128, "tooltip": "生成网格密度（点数）。值越大 -> mask 更多且更小；越小 -> mask 更少且更大"}),
                "pred_iou_thresh": ("FLOAT", {"default": 0.86, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "预测 IoU 阈值。过滤掉置信度低的 mask，越高越严格"}),
                "stability_score_thresh": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "稳定度阈值。过滤轮廓不稳定的 mask"}),
                "max_mask_count": ("INT", {"default": 50, "min": 1, "max": 4096, "tooltip": "最终按面积排序，仅保留前 N 个最大 mask"}),
                "min_mask_area": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "过滤小于该像素面积的 mask"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("annotated_image", "object_masks", "detection_json", "object_names")
    OUTPUT_IS_LIST = (False, True, False, True)
    FUNCTION = "_generate"
    CATEGORY = "💃rDancer/Panoptic"

    def _generate(self, sam1_model: dict, image: torch.Tensor, points_per_side: int = 32,
                  pred_iou_thresh: float = 0.86, stability_score_thresh: float = 0.92,
                  max_mask_count: int = 256, min_mask_area: int = 0):
        sam_model = sam1_model['model'] if isinstance(sam1_model, dict) else sam1_model
        device = sam_model.device if hasattr(sam_model, 'device') else torch.device('cpu')

        generator = _get_generator(
            sam_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

        annotated_images = []
        masks_out: List[torch.Tensor] = []
        object_names: List[str] = []
        detection_json_str = "{}"

        # 目前按批次逐张处理
        for idx, img_t in enumerate(image):
            img_np = (img_t.cpu().numpy() * 255).astype(np.uint8)
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            img_pil = Image.fromarray(img_np)

            results = generator.generate(img_np)
            # 根据 area 过滤、排序
            results = sorted(results, key=lambda x: x['area'], reverse=True)
            filtered = [r for r in results if r['area'] >= min_mask_area][:max_mask_count]

            bboxes, names, masks_for_detection = [], [], []
            for i, r in enumerate(filtered):
                m = torch.from_numpy(r['segmentation'].astype(np.float32))
                masks_out.append(m)
                names.append(f"mask_{i+1}")
                bboxes.append(mask_to_bbox(r['segmentation']))
                masks_for_detection.append(r['segmentation'])

            object_names = names  # 最终返回最后一张图名字；多图可合并
            detection_json = build_detection_json(img_pil.width, img_pil.height, names, bboxes)
            detection_json_str = json.dumps(detection_json, ensure_ascii=False)

            # 创建标注图像（参考VVL_GroundingDinoSAM2的方法）
            if filtered and masks_for_detection:
                # 创建supervision的Detections对象
                xyxy_boxes = []
                for bbox in bboxes:
                    # bbox格式转换为xyxy
                    x1, y1, x2, y2 = bbox
                    xyxy_boxes.append([x1, y1, x2, y2])
                
                xyxy_boxes = np.array(xyxy_boxes)
                masks_array = np.array(masks_for_detection)
                
                detections = sv.Detections(
                    xyxy=xyxy_boxes,
                    mask=masks_array
                )
                
                # 设置标签
                detections.data = {'class_name': names}
                
                # 标注图像
                annotated_img = annotate_image(img_pil, detections)
                annotated_images.append(pil2tensor(annotated_img))
            else:
                # 如果没有检测到对象，返回原始图像
                annotated_images.append(img_t)

        # 堆叠标注图像
        annotated_images_stacked = torch.stack(annotated_images) if annotated_images else torch.empty(0)

        return (annotated_images_stacked, masks_out, detection_json_str, object_names)

# 节点注册
NODE_CLASS_MAPPINGS = {"SAM1AutoEverything": SAM1AutoEverything}
NODE_DISPLAY_NAME_MAPPINGS = {"SAM1AutoEverything": "VVL SAM1 Auto Everything"} 