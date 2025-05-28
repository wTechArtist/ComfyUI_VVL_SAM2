# VVL_MaskCleaner 节点需求文档

## 节点概述
创建一个极简易用的mask清理节点，专门解决SAM2分割输出的两个核心问题：
1. 填补mask内部的空洞（如山体中被草树挖掉的部分）
2. 清除周围零碎的小遮罩，只保留主要区域

## 核心功能需求

### 功能1：填补内部空洞
**问题描述：**
- 山体mask中间因为草、树等被误识别而产生黑色空洞
- 完整对象的mask被挖出不应该有的洞
- 需要保持对象的完整性和连续性

**解决方案：**
- 检测每个白色连通域内部的黑色空洞
- 自动填补所有检测到的内部空洞
- 使用形态学闭运算识别并填补内部空洞

**实现逻辑：**
1. 对每个mask进行连通域分析，找到所有白色连通域
2. 对每个白色连通域分别进行空洞填补处理
3. 使用形态学闭运算检测每个连通域内部的黑色空洞
4. 填补所有检测到的内部空洞

### 功能2：清除零碎遮罩
**问题描述：**
- mask周围有很多小的白色噪点
- 分割算法产生的不相关小区域影响效果
- 需要只保留主要的对象区域

**解决方案：**
- 检测所有白色连通域的面积
- 只保留面积最大的N个白色区域
- 删除其他小的零碎遮罩

**实现逻辑：**
1. 分析mask中所有白色连通域
2. 按面积大小排序
3. 只保留最大的N个区域
4. 将其他区域设为黑色

## 节点参数设计

### 输入参数（极简版）
```python
{
    "required": {
        "masks": ("MASK", {
            "tooltip": "输入的mask列表，来自SAM2等分割节点"
        }),
    },
    "optional": {
        # 核心控制参数
        "keep_largest_n": ("INT", {
            "default": 1,
            "min": 1,
            "max": 10,
            "tooltip": "保留最大的N个白色区域，其他区域会被删除"
        }),
        
        # 处理模式
        "processing_mode": (["both", "fill_only", "clean_only"], {
            "default": "both",
            "tooltip": "处理模式：both=填洞+清理，fill_only=只填洞，clean_only=只清理"
        }),
    }
}
```

### 输出参数
```python
RETURN_TYPES = ("MASK", "STRING")
RETURN_NAMES = ("cleaned_masks", "processing_info")
OUTPUT_IS_LIST = (True, False)
```

## 详细算法实现

### 算法1：智能空洞填补
```python
def fill_internal_holes(mask):
    """
    填补mask内部的空洞
    
    核心思路：
    - 对每个白色连通域，检测其内部的黑色空洞
    - 填补所有检测到的内部空洞
    """
    import cv2
    import numpy as np
    
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
        
        # 使用形态学闭运算填补内部空洞
        # 动态调整核大小，确保能够填补大部分内部空洞
        kernel_size = max(5, min(w, h) // 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        filled_roi = cv2.morphologyEx(region_roi, cv2.MORPH_CLOSE, kernel)
        
        # 将填补后的区域更新到结果mask中
        filled_mask[y:y+h, x:x+w] = filled_roi
    
    return filled_mask
```

### 算法2：零碎遮罩清理
```python
def remove_small_regions(mask, keep_largest_n=1):
    """
    清除零碎的小遮罩，只保留最大的N个区域
    
    核心思路：
    - 分析所有白色连通域的面积
    - 按面积排序，只保留最大的N个
    - 删除其他所有区域
    """
    import cv2
    import numpy as np
    
    # 1. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    # 2. 按面积排序（排除背景label=0）
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)  # 按面积从大到小排序
    
    # 3. 创建新的清理后的mask
    cleaned_mask = np.zeros_like(mask)
    
    # 4. 只保留最大的N个区域
    for i in range(min(keep_largest_n, len(areas))):
        label_id, area = areas[i]
        cleaned_mask[labels == label_id] = 255
    
    return cleaned_mask
```

### 主处理函数
```python
def process_mask(mask, keep_largest_n=1, processing_mode="both"):
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
        processed_mask = remove_small_regions(processed_mask, keep_largest_n)
        remaining_regions = count_regions(processed_mask)
        removed_regions = original_regions - remaining_regions
        processing_info.append(f"已清理{removed_regions}个零碎遮罩，保留{remaining_regions}个主要区域")
    
    return processed_mask, "; ".join(processing_info)

def count_holes(mask):
    """计算mask中的空洞数量"""
    # 反转mask，将空洞变成白色区域进行计数
    inverted = cv2.bitwise_not(mask)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(inverted)
    return num_labels - 1  # 减去背景

def count_regions(mask):
    """计算mask中的白色区域数量"""
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask)
    return num_labels - 1  # 减去背景
```

## ComfyUI节点实现

### 完整节点类
```python
import torch
import cv2
import numpy as np

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
                # 核心控制参数
                "keep_largest_n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "tooltip": "保留最大的N个白色区域，其他区域会被删除"
                }),
                
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
    
    def clean_masks(self, masks, keep_largest_n=1, processing_mode="both"):
        """
        清理mask的主函数
        """
        cleaned_masks = []
        all_processing_info = []
        
        for i, mask in enumerate(masks):
            # 转换为numpy数组进行处理
            if isinstance(mask, torch.Tensor):
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            else:
                mask_np = (mask * 255).astype(np.uint8)
            
            # 处理单个mask
            cleaned_mask_np, info = process_mask(
                mask_np, keep_largest_n, processing_mode
            )
            
            # 转换回tensor格式
            cleaned_mask_tensor = torch.from_numpy(cleaned_mask_np.astype(np.float32) / 255.0)
            cleaned_masks.append(cleaned_mask_tensor)
            all_processing_info.append(f"Mask {i+1}: {info}")
        
        # 合并处理信息
        final_info = "\n".join(all_processing_info)
        
        return (cleaned_masks, final_info)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "VVL_MaskCleaner": VVL_MaskCleaner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVL_MaskCleaner": "VVL Mask清理器"
}
```

## 使用场景和参数建议

### 场景1：单对象分割优化
**问题**：山体、建筑物等单个对象有内部空洞和周围噪点
**推荐设置**：
- `keep_largest_n=1`
- `processing_mode="both"`

**效果**：填补内部空洞，只保留最大的主体区域

### 场景2：只需要填补空洞
**问题**：mask质量不错，只是有内部空洞需要填补
**推荐设置**：
- `processing_mode="fill_only"`

**效果**：只填补空洞，不删除任何区域

### 场景3：只需要清理噪点
**问题**：对象完整，但周围有很多小噪点
**推荐设置**：
- `keep_largest_n=1`
- `processing_mode="clean_only"`

**效果**：只保留最大区域，删除所有小噪点

### 场景4：多对象场景
**问题**：需要保留多个主要对象
**推荐设置**：
- `keep_largest_n=3`（或其他数量）
- `processing_mode="both"`

**效果**：保留3个最大的对象，每个对象都填补内部空洞

### 场景5：极简处理
**问题**：只想要最干净的单个主体
**推荐设置**：
- `keep_largest_n=1`
- `processing_mode="both"`

**效果**：最终只得到一个完整、干净的主体区域

## 参数说明

### 核心参数
1. **`keep_largest_n`** (默认: 1)
   - 保留面积最大的N个白色区域
   - 其他所有区域都会被删除
   - 适用范围：1-10个区域

2. **`processing_mode`** (默认: "both")
   - `"both"`: 填补空洞 + 清理零碎区域（推荐）
   - `"fill_only"`: 只填补内部空洞
   - `"clean_only"`: 只清理零碎区域

### 默认参数（适合大多数情况）
- `keep_largest_n`: 1（只保留最大区域）
- `processing_mode`: "both"（完整清理）

### 参数调优建议
- **单对象场景**：`keep_largest_n=1`
- **多对象场景**：`keep_largest_n=2-5`（根据实际对象数量）
- **只填洞不清理**：`processing_mode="fill_only"`
- **只清理不填洞**：`processing_mode="clean_only"`

## 技术实现要点

### 1. 性能优化
- 使用边界框ROI处理，减少计算量
- OpenCV高效连通域算法
- 动态调整形态学核大小
- 批量处理多个mask

### 2. 内存管理
- 逐个处理mask，避免内存峰值
- 及时释放中间结果
- 使用numpy数组进行计算

### 3. 边界情况处理
- 空mask检查和处理
- 全黑/全白mask的特殊处理
- 异常尺寸mask的兼容性
- 参数合理性验证

### 4. 质量保证
- 处理前后统计信息对比
- 参数有效性检查
- 错误恢复和日志记录

## 算法特点

### 优势
1. **极简参数**：只有2个核心参数，易于理解和使用
2. **逻辑清晰**：填洞+清理，功能明确
3. **自动化程度高**：无需复杂调参，默认设置适用大多数场景
4. **处理效果好**：针对SAM2分割结果的常见问题优化

### 适用场景
- SAM2分割结果后处理
- 任何需要清理的二值mask
- 对象分割质量提升
- 批量mask处理

### 局限性
- 不适合需要保留细小细节的场景
- 对于复杂形状可能过度简化
- 填洞操作不可逆

## 开发和测试计划

### 开发优先级
1. **P0（核心功能）**：基础空洞填补和零碎遮罩清理
2. **P1（重要功能）**：ComfyUI集成和批量处理
3. **P2（增强功能）**：性能优化和边界处理
4. **P3（高级功能）**：智能参数推荐

### 测试用例
1. **功能测试**：各种类型mask的处理效果
2. **性能测试**：大批量和高分辨率mask处理
3. **边界测试**：异常输入和极端参数
4. **集成测试**：与其他ComfyUI节点的兼容性

---

## 总结

这个VVL_MaskCleaner节点设计极简实用，专注解决SAM2分割结果的核心问题：
- **只有2个参数**：`keep_largest_n` 和 `processing_mode`
- **功能明确**：填补空洞 + 清理零碎区域
- **使用简单**：默认参数适用大多数场景
- **效果显著**：显著提升mask质量

节点遵循"简单就是美"的设计哲学，让用户能够轻松获得高质量的mask处理结果。 