import json
import math
from typing import Dict, Any, List, Tuple


class VVL_DetectionScaler:
    """
    专门处理detection_json的图像尺寸和边界框等比例缩放节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detection_json": ("STRING", {
                    "multiline": True,
                    "tooltip": "来自VVL检测节点的JSON检测结果，包含图像尺寸和对象边界框信息"
                }),
                "max_edge_size": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "目标最长边大小。图像的长宽两个值中的最长边将被缩放到此尺寸，另一边按比例自动缩放"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("scaled_detection_json", "scale_ratio", "new_width", "new_height")
    FUNCTION = "_scale_detection_json"
    CATEGORY = "💃rDancer"
    
    def _scale_detection_json(self, detection_json: str, max_edge_size: int) -> Tuple[str, float, int, int]:
        """
        等比例缩放detection_json中的图像尺寸和边界框坐标
        
        Args:
            detection_json (str): 输入的JSON字符串
            max_edge_size (int): 目标最长边大小
            
        Returns:
            tuple: (缩放后的JSON字符串, 缩放比例, 新宽度, 新高度)
        """
        try:
            # 解析JSON
            detection_data = json.loads(detection_json)
        except json.JSONDecodeError as e:
            print(f"VVL_DetectionScaler: JSON解析错误: {e}")
            return detection_json, 1.0, 0, 0
        
        # 检查必要字段
        if "image_width" not in detection_data or "image_height" not in detection_data:
            print("VVL_DetectionScaler: 缺少image_width或image_height字段")
            return detection_json, 1.0, 0, 0
        
        # 获取原始尺寸
        original_width = detection_data["image_width"]
        original_height = detection_data["image_height"]
        
        if original_width <= 0 or original_height <= 0:
            print(f"VVL_DetectionScaler: 无效的图像尺寸 {original_width}x{original_height}")
            return detection_json, 1.0, 0, 0
        
        # 计算缩放比例
        scale_ratio = self._calculate_scale_ratio(original_width, original_height, max_edge_size)
        
        # 计算新尺寸
        new_width = int(round(original_width * scale_ratio))
        new_height = int(round(original_height * scale_ratio))
        
        # 直接修改原始数据，保持完全相同的结构
        scaled_data = json.loads(detection_json)  # 重新解析以保持原始结构
        
        # 更新图像尺寸
        scaled_data["image_width"] = new_width
        scaled_data["image_height"] = new_height
        
        # 缩放每个对象的边界框，保持对象的所有其他字段不变
        if "objects" in scaled_data and isinstance(scaled_data["objects"], list):
            for obj in scaled_data["objects"]:
                if isinstance(obj, dict) and "bbox_2d" in obj and isinstance(obj["bbox_2d"], list) and len(obj["bbox_2d"]) == 4:
                    obj["bbox_2d"] = self._scale_bbox(obj["bbox_2d"], scale_ratio)
        
        # 转换回JSON字符串，保持原始格式风格
        # 首先尝试保持原始的缩进格式
        original_is_compact = '\n' not in detection_json.strip()
        
        if original_is_compact:
            # 原始JSON是紧凑格式，保持紧凑
            scaled_json = json.dumps(scaled_data, ensure_ascii=False, separators=(',', ':'))
        else:
            # 原始JSON有缩进，使用相同的缩进风格
            scaled_json = json.dumps(scaled_data, ensure_ascii=False, indent=2)
            # 保持bbox_2d为单行格式（如果原始就是这样）
            import re
            if re.search(r'"bbox_2d":\s*\[\d+,\d+,\d+,\d+\]', detection_json):
                scaled_json = re.sub(r'"bbox_2d":\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]', 
                                    r'"bbox_2d": [\1,\2,\3,\4]', scaled_json)
        
        print(f"VVL_DetectionScaler: 原始尺寸 {original_width}x{original_height} -> "
              f"缩放后尺寸 {new_width}x{new_height} (比例: {scale_ratio:.4f})")
        
        return scaled_json, scale_ratio, new_width, new_height
    
    def _calculate_scale_ratio(self, width: int, height: int, max_edge_size: int) -> float:
        """
        计算缩放比例，使最长边等于目标尺寸
        
        Args:
            width (int): 原始宽度
            height (int): 原始高度
            max_edge_size (int): 目标最长边尺寸
            
        Returns:
            float: 缩放比例
        """
        max_edge = max(width, height)
        return max_edge_size / max_edge
    
    def _scale_bbox(self, bbox: List[int], scale_ratio: float) -> List[int]:
        """
        缩放边界框坐标
        
        Args:
            bbox (List[int]): 原始边界框 [x1, y1, x2, y2]
            scale_ratio (float): 缩放比例
            
        Returns:
            List[int]: 缩放后的边界框
        """
        if len(bbox) != 4:
            print(f"VVL_DetectionScaler: 无效的边界框格式: {bbox}")
            return bbox
        
        x1, y1, x2, y2 = bbox
        
        # 应用缩放比例
        scaled_x1 = int(round(x1 * scale_ratio))
        scaled_y1 = int(round(y1 * scale_ratio))
        scaled_x2 = int(round(x2 * scale_ratio))
        scaled_y2 = int(round(y2 * scale_ratio))
        
        # 确保坐标顺序正确
        min_x = min(scaled_x1, scaled_x2)
        max_x = max(scaled_x1, scaled_x2)
        min_y = min(scaled_y1, scaled_y2)
        max_y = max(scaled_y1, scaled_y2)
        
        return [min_x, min_y, max_x, max_y]


# Node registration
NODE_CLASS_MAPPINGS = {
    "VVL_DetectionScaler": VVL_DetectionScaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_DetectionScaler": "VVL Detection Scaler"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 