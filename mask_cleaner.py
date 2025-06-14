"""
VVL Mask Cleaner Node for ComfyUI
用于清理SAM2等分割结果的mask，填补内部空洞并清除零碎区域
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple, Union


def count_holes(mask: np.ndarray) -> int:
    """计算mask中的空洞数量"""
    # 反转mask，将空洞变成白色区域进行计数
    inverted = cv2.bitwise_not(mask)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(inverted)
    return num_labels - 1  # 减去背景


def count_regions(mask: np.ndarray) -> int:
    """计算mask中的白色区域数量"""
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask)
    return num_labels - 1  # 减去背景


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """
    填补mask内部的空洞
    
    核心思路：
    - 对每个白色连通域，检测其内部真正的黑色空洞
    - 只填补完全被白色区域包围的黑色区域，不改变边缘形状
    """
    # 1. 找到所有白色连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    # 2. 创建处理后的mask副本
    filled_mask = mask.copy()
    
    # 3. 对每个白色连通域进行处理（跳过背景label=0）
    for label_id in range(1, num_labels):
        # 创建当前连通域的mask
        current_region = (labels == label_id).astype(np.uint8) * 255
        
        # 获取当前连通域的边界框
        x, y, w, h = stats[label_id, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
        
        # 提取边界框区域进行处理（提高效率）
        region_roi = current_region[y:y+h, x:x+w]
        
        # 使用漫水填充法检测内部空洞
        # 创建一个稍大的画布，边界填充0
        padded_region = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded_region[1:h+1, 1:w+1] = region_roi
        
        # 从边界开始漫水填充，标记所有与边界连通的黑色区域
        # 这些区域不是内部空洞
        temp_mask = padded_region.copy()
        
        # 从四个边界开始漫水填充
        for start_point in [(0, 0), (0, w+1), (h+1, 0), (h+1, w+1)]:
            if temp_mask[start_point] == 0:
                cv2.floodFill(temp_mask, None, start_point, 128)
        
        # 从边界的所有黑色像素开始填充
        for i in range(h + 2):
            for j in [0, w + 1]:  # 左右边界
                if temp_mask[i, j] == 0:
                    cv2.floodFill(temp_mask, None, (j, i), 128)
        
        for j in range(w + 2):
            for i in [0, h + 1]:  # 上下边界
                if temp_mask[i, j] == 0:
                    cv2.floodFill(temp_mask, None, (j, i), 128)
        
        # 提取真正的内部空洞（值仍为0的黑色区域）
        internal_holes = (temp_mask[1:h+1, 1:w+1] == 0) & (region_roi == 0)
        
        # 填补内部空洞
        region_roi_filled = region_roi.copy()
        region_roi_filled[internal_holes] = 255
        
        # 将填补后的区域更新到结果mask中
        filled_mask[y:y+h, x:x+w] = region_roi_filled
    
    return filled_mask


def remove_small_regions(mask: np.ndarray) -> np.ndarray:
    """
    清除零碎的小遮罩，只保留最大的区域
    
    核心思路：
    - 分析所有白色连通域的面积
    - 按面积排序，只保留最大的区域
    - 删除其他所有区域
    """
    # 1. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    # 2. 按面积排序（排除背景label=0）
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)  # 按面积从大到小排序
    
    # 3. 创建新的清理后的mask
    cleaned_mask = np.zeros_like(mask)
    
    # 4. 只保留最大的区域
    if len(areas) > 0:
        label_id, area = areas[0]
        cleaned_mask[labels == label_id] = 255
    
    return cleaned_mask


def process_mask(mask: np.ndarray, processing_mode: str = "both") -> Tuple[np.ndarray, str]:
    """
    主处理函数 - 按顺序执行mask清理操作
    """
    processed_mask = mask.copy()
    processing_info = []
    
    # 步骤1：填补内部空洞
    if processing_mode in ["both", "fill_only"]:
        original_holes = count_holes(processed_mask)
        processed_mask = fill_internal_holes(processed_mask)
        filled_holes = original_holes - count_holes(processed_mask)
        processing_info.append(f"已填补{filled_holes}个内部空洞")
    
    # 步骤2：清除零碎遮罩
    if processing_mode in ["both", "clean_only"]:
        original_regions = count_regions(processed_mask)
        processed_mask = remove_small_regions(processed_mask)
        remaining_regions = count_regions(processed_mask)
        removed_regions = original_regions - remaining_regions
        processing_info.append(f"已清理{removed_regions}个零碎遮罩，保留{remaining_regions}个主要区域")
    
    return processed_mask, "; ".join(processing_info)


class VVL_MaskCleaner:
    """
    VVL Mask清理节点 - 填补空洞和清除零碎遮罩
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {
                    "tooltip": "输入的mask列表，来自SAM2等分割节点"
                }),
            },
            "optional": {
                # 处理模式
                "processing_mode": (["both", "fill_only", "clean_only"], {
                    "default": "both",
                    "tooltip": "处理模式：both=填洞+清理，fill_only=只填洞，clean_only=只清理"
                }),
            }
        }
    
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("cleaned_masks", "processing_info")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "clean_masks"
    CATEGORY = "💃rDancer"
    
    def clean_masks(self, masks, processing_mode="both"):
        """
        清理mask的主函数
        """
        cleaned_masks = []
        all_processing_info = []
        
        # 确保masks是列表格式
        if not isinstance(masks, list):
            masks = [masks]
        
        for i, mask in enumerate(masks):
            # 处理不同的输入格式
            if isinstance(mask, torch.Tensor):
                # 处理tensor格式
                if mask.dim() == 2:
                    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                elif mask.dim() == 3:
                    # 如果是3维tensor，取第一个通道
                    mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
                else:
                    raise ValueError(f"Unsupported mask tensor dimension: {mask.dim()}")
            elif isinstance(mask, np.ndarray):
                # 处理numpy数组
                if mask.dtype == np.float32 or mask.dtype == np.float64:
                    mask_np = (mask * 255).astype(np.uint8)
                else:
                    mask_np = mask.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported mask type: {type(mask)}")
            
            # 确保mask是二值化的
            _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
            
            # 处理单个mask
            try:
                cleaned_mask_np, info = process_mask(
                    mask_np, processing_mode
                )
                all_processing_info.append(f"Mask {i+1}: {info}")
            except Exception as e:
                # 如果处理失败，返回原始mask
                cleaned_mask_np = mask_np
                all_processing_info.append(f"Mask {i+1}: 处理失败 - {str(e)}")
            
            # 转换回tensor格式
            cleaned_mask_tensor = torch.from_numpy(cleaned_mask_np.astype(np.float32) / 255.0)
            cleaned_masks.append(cleaned_mask_tensor)
        
        # 合并处理信息
        final_info = "\n".join(all_processing_info)
        
        return (cleaned_masks, final_info)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "VVL_MaskCleaner": VVL_MaskCleaner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_MaskCleaner": "VVL Mask Cleaner"
} 