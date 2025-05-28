import torch
import numpy as np
from PIL import Image
from typing import List
import json

try:
    from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
except ImportError as e:
    raise ImportError("缺少 transformers 库，请执行 pip install transformers timm") from e

from ..utils.panoptic_utils import mask_to_bbox, build_detection_json

# 全局模型缓存
_M2F_MODEL = None
_M2F_PROCESSOR = None
_CURRENT_MODEL_NAME = None

class Mask2FormerPanoptic:
    """Mask2Former Panoptic 分割节点"""

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = [
            "facebook/mask2former-swin-large-coco-panoptic",
            "facebook/mask2former-swin-base-coco-panoptic",
            "facebook/mask2former-swin-tiny-ade-semantic"
        ]
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像，支持批量"}),
            },
            "optional": {
                "model_name": (model_choices, {"default": model_choices[0], "tooltip": "HuggingFace 上的 Mask2Former Panoptic 权重名称"}),
                "confidence_thresh": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Mask2Former 提取实例的置信度阈值，越高越严格"}),
                "min_mask_area": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "小于该像素面积的 mask 将被过滤"}),
                "max_mask_count": ("INT", {"default": 256, "min": 1, "max": 2048, "tooltip": "按面积排序，仅保留前 N 个 mask"}),
                "merge_same_class": ("BOOLEAN", {"default": True, "tooltip": "是否将同类别 Stuff 区域合并为一个 mask"}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "STRING")
    RETURN_NAMES = ("object_masks", "detection_json", "object_names")
    OUTPUT_IS_LIST = (True, False, True)
    FUNCTION = "_segment"
    CATEGORY = "💃rDancer/Panoptic"

    def _load_model(self, model_name: str, device: torch.device):
        global _M2F_MODEL, _M2F_PROCESSOR, _CURRENT_MODEL_NAME
        if _M2F_MODEL is None or _CURRENT_MODEL_NAME != model_name:
            _M2F_PROCESSOR = Mask2FormerImageProcessor.from_pretrained(model_name)
            _M2F_MODEL = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
            _CURRENT_MODEL_NAME = model_name
        return _M2F_MODEL, _M2F_PROCESSOR

    def _segment(self, image: torch.Tensor, model_name: str = "facebook/mask2former-swin-large-coco-panoptic",
                 confidence_thresh: float = 0.5, min_mask_area: int = 0, max_mask_count: int = 256,
                 merge_same_class: bool = True):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, processor = self._load_model(model_name, device)

        masks_out: List[torch.Tensor] = []
        object_names: List[str] = []
        detection_json_str = "{}"

        id2label = model.config.id2label

        for idx, img_t in enumerate(image):
            img_np = (img_t.cpu().numpy() * 255).astype(np.uint8)
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            img_pil = Image.fromarray(img_np)
            inputs = processor(images=img_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            result = processor.post_process_panoptic_segmentation(
                outputs, target_sizes=[img_pil.size[::-1]], threshold=confidence_thresh
            )[0]
            seg_map = result["segmentation"].cpu().numpy().astype(np.int32)
            segments_info = result["segments_info"]

            masks_tmp, names_tmp, bboxes_tmp = [], [], []
            for s in segments_info:
                mask = (seg_map == s["id"])
                area = mask.sum()
                if area < min_mask_area:
                    continue
                label_id = s["label_id"]
                name = id2label.get(label_id, f"class_{label_id}")
                masks_tmp.append(torch.from_numpy(mask.astype(np.float32)))
                names_tmp.append(name)
                bboxes_tmp.append(mask_to_bbox(mask))

            # 如果需要按类别合并 stuff，可实现，这里简单保留实例
            # 限制数量
            if len(masks_tmp) > max_mask_count:
                # 按面积排序截断
                areas = [m.sum().item() for m in masks_tmp]
                keep_idx = np.argsort(areas)[::-1][:max_mask_count]
                masks_tmp = [masks_tmp[i] for i in keep_idx]
                names_tmp = [names_tmp[i] for i in keep_idx]
                bboxes_tmp = [bboxes_tmp[i] for i in keep_idx]

            masks_out.extend(masks_tmp)
            object_names = names_tmp
            detection_json = build_detection_json(img_pil.width, img_pil.height, names_tmp, bboxes_tmp)
            detection_json_str = json.dumps(detection_json, ensure_ascii=False)

        return (masks_out, detection_json_str, object_names)

NODE_CLASS_MAPPINGS = {"Mask2FormerPanoptic": Mask2FormerPanoptic}
NODE_DISPLAY_NAME_MAPPINGS = {"Mask2FormerPanoptic": "Mask2Former Panoptic"} 