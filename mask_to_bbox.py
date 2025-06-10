import torch
import numpy as np
from typing import List, Tuple, Optional

def get_mask_bbox(mask_tensor: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    """
    从mask tensor中获取边界框坐标
    
    Args:
        mask_tensor: 输入的mask tensor (H, W) 或 (H, W, C)
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max) 或 None（如果mask为空）
    """
    # 处理tensor维度
    mask_np = mask_tensor.cpu().numpy()
    
    if mask_np.ndim == 3:
        # 如果是3D tensor，取第一个通道或最大值
        if mask_np.shape[2] == 1:
            mask_np = mask_np[:, :, 0]
        else:
            mask_np = np.max(mask_np, axis=2)
    elif mask_np.ndim == 1:
        # 如果是1D，尝试重塑
        side_len = int(np.sqrt(len(mask_np)))
        if side_len * side_len == len(mask_np):
            mask_np = mask_np.reshape(side_len, side_len)
        else:
            raise ValueError(f"Cannot reshape 1D mask of length {len(mask_np)} to 2D")
    
    # 二值化mask（阈值0.5）
    mask_bool = mask_np > 0.5
    
    # 查找非零像素的位置
    coords = np.where(mask_bool)
    
    if len(coords[0]) == 0:
        return None  # 空mask
        
    y_coords, x_coords = coords
    x_min, x_max = int(x_coords.min()), int(x_coords.max())
    y_min, y_max = int(y_coords.min()), int(y_coords.max())
    
    return (x_min, y_min, x_max, y_max)

def create_bbox_mask(bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> torch.Tensor:
    """
    根据边界框创建填充的矩形mask
    
    Args:
        bbox: (x_min, y_min, x_max, y_max)
        image_size: (width, height)
        
    Returns:
        torch.Tensor: 填充的矩形mask
    """
    width, height = image_size
    x_min, y_min, x_max, y_max = bbox
    
    # 创建空白mask
    mask = np.zeros((height, width), dtype=np.float32)
    
    # 确保坐标在有效范围内
    x_min = max(0, min(x_min, width - 1))
    x_max = max(0, min(x_max, width - 1))
    y_min = max(0, min(y_min, height - 1))
    y_max = max(0, min(y_max, height - 1))
    
    # 填充矩形区域
    mask[y_min:y_max+1, x_min:x_max+1] = 1.0
    
    return torch.from_numpy(mask)

class VVL_MaskToBBox:
    """
    将mask转换为边界框的节点
    接收mask列表，输出边界框坐标和填充的矩形mask
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "object_masks": ("MASK", {
                    "tooltip": "来自VVL_GroundingDinoSAM2节点的object_masks输出，包含多个分割mask"
                }),
                "expand_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "边界框扩展像素数，用于在原始边界框基础上向外扩展指定像素"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "MASK", "STRING")
    RETURN_NAMES = ("bbox_coordinates", "filled_masks", "center_points")
    FUNCTION = "process_masks"
    CATEGORY = "💃rDancer"
    OUTPUT_IS_LIST = (False, True, False)
    
    def process_masks(self, object_masks, expand_pixels=0):
        """
        处理mask列表，提取边界框并创建填充的矩形mask
        """
        try:
            print(f"VVL_MaskToBBox: 接收到的 object_masks 类型: {type(object_masks)}")
            
            # 处理输入数据
            if object_masks is None:
                print("VVL_MaskToBBox: object_masks 为 None")
                return ("", [], "")
            
            # 由于VVL_GroundingDinoSAM2的OUTPUT_IS_LIST=(False, True, False, True)
            # object_masks应该已经是列表格式传递过来的
            if isinstance(object_masks, list):
                mask_list = object_masks
            elif isinstance(object_masks, torch.Tensor):
                # 如果是单个tensor，转换为列表
                if object_masks.dim() == 4:  # 批量tensor (N, H, W, C)
                    mask_list = [object_masks[i] for i in range(object_masks.shape[0])]
                else:  # 单个tensor
                    mask_list = [object_masks]
            else:
                print(f"VVL_MaskToBBox: 不支持的输入类型: {type(object_masks)}")
                return ("", [], "")
            
            if len(mask_list) == 0:
                print("VVL_MaskToBBox: mask列表为空")
                return ("", [], "")
            
            print(f"VVL_MaskToBBox: 处理 {len(mask_list)} 个mask")
            
            # 使用第一个mask的尺寸确定输出尺寸
            first_mask = mask_list[0]
            if first_mask.dim() >= 2:
                ref_height, ref_width = first_mask.shape[-2], first_mask.shape[-1]
                image_size = (ref_width, ref_height)
                print(f"VVL_MaskToBBox: 使用mask尺寸 {image_size}")
            else:
                image_size = (512, 512)  # 默认尺寸
                print(f"VVL_MaskToBBox: 使用默认尺寸 {image_size}")

            # 处理每个mask
            filled_masks = []
            bbox_coords_list = []
            center_points_list = []
            
            for i, mask_tensor in enumerate(mask_list):
                try:
                    # 获取边界框
                    bbox = get_mask_bbox(mask_tensor)
                    
                    if bbox is None:
                        print(f"VVL_MaskToBBox: Mask {i+1} 为空，跳过")
                        continue
                    
                    x_min, y_min, x_max, y_max = bbox
                    
                    # 扩展边界框
                    if expand_pixels > 0:
                        width, height = image_size
                        x_min = max(0, x_min - expand_pixels)
                        y_min = max(0, y_min - expand_pixels)
                        x_max = min(width - 1, x_max + expand_pixels)
                        y_max = min(height - 1, y_max + expand_pixels)
                        expanded_bbox = (x_min, y_min, x_max, y_max)
                    else:
                        expanded_bbox = bbox
                    
                    bbox_coords_list.append(f"[{expanded_bbox[0]},{expanded_bbox[1]},{expanded_bbox[2]},{expanded_bbox[3]}]")
                    
                    # 计算中心点坐标
                    center_x = (expanded_bbox[0] + expanded_bbox[2]) // 2
                    center_y = (expanded_bbox[1] + expanded_bbox[3]) // 2
                    center_points_list.append(f"[{center_x},{center_y}]")
                    
                    # 创建填充的矩形mask
                    filled_mask = create_bbox_mask(expanded_bbox, image_size)
                    filled_masks.append(filled_mask)
                    
                    print(f"VVL_MaskToBBox: Mask {i+1} -> BBox: {expanded_bbox}")
                    
                except Exception as e:
                    print(f"VVL_MaskToBBox: 处理mask {i+1}时出错: {e}")
                    continue
            
            # 生成输出
            bbox_coords_str = "; ".join(bbox_coords_list)
            center_points_str = "; ".join(center_points_list)
            
            print(f"VVL_MaskToBBox: 成功处理 {len(filled_masks)} 个边界框")
            
            return (bbox_coords_str, filled_masks, center_points_str)
            
        except Exception as e:
            print(f"VVL_MaskToBBox: 处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            return ("", [], "")

# 节点注册
NODE_CLASS_MAPPINGS = {
    "VVL_MaskToBBox": VVL_MaskToBBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_MaskToBBox": "VVL Mask转边界框",
} 