import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import json

try:
    from segment_anything import SamAutomaticMaskGenerator
except ImportError as e:
    raise ImportError("segment_anything 库未安装，请执行 pip install git+https://github.com/facebookresearch/segment-anything.git") from e

from ..utils.panoptic_utils import mask_to_bbox, build_detection_json

# 全局缓存 generator，提高批量速度
_AUTO_GENERATORS = {}

def _get_generator(sam_model, **kwargs):
    key = id(sam_model)
    if key not in _AUTO_GENERATORS:
        _AUTO_GENERATORS[key] = SamAutomaticMaskGenerator(sam_model, **kwargs)
    return _AUTO_GENERATORS[key]

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
                "max_mask_count": ("INT", {"default": 256, "min": 1, "max": 4096, "tooltip": "最终按面积排序，仅保留前 N 个最大 mask"}),
                "min_mask_area": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "过滤小于该像素面积的 mask"}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "STRING")
    RETURN_NAMES = ("object_masks", "detection_json", "object_names")
    OUTPUT_IS_LIST = (True, False, True)
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

            bboxes, names = [], []
            for i, r in enumerate(filtered):
                m = torch.from_numpy(r['segmentation'].astype(np.float32))
                masks_out.append(m)
                names.append(f"mask_{i+1}")
                bboxes.append(mask_to_bbox(r['segmentation']))

            object_names = names  # 最终返回最后一张图名字；多图可合并
            detection_json = build_detection_json(img_pil.width, img_pil.height, names, bboxes)
            detection_json_str = json.dumps(detection_json, ensure_ascii=False)

        return (masks_out, detection_json_str, object_names)

# 节点注册
NODE_CLASS_MAPPINGS = {"SAM1AutoEverything": SAM1AutoEverything}
NODE_DISPLAY_NAME_MAPPINGS = {"SAM1AutoEverything": "VVL SAM1 Auto Everything"} 