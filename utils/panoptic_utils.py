import numpy as np
from typing import List, Dict

def mask_to_bbox(mask: np.ndarray):
    """将二值mask转换为xyxy格式的边界框"""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return [x1, y1, x2, y2]

def build_detection_json(image_w: int, image_h: int, names: List[str], bboxes: List[List[int]]):
    data = {
        "image_width": image_w,
        "image_height": image_h,
        "objects": []
    }
    for n, bb in zip(names, bboxes):
        data["objects"].append({"name": n, "bbox_2d": bb})
    return data 