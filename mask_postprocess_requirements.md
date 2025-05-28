# VVL_MaskPostProcessor 节点需求文档

## 节点概述
创建一个mask后处理节点，用于优化和清理从VVL_GroundingDinoSAM2等分割节点输出的mask，解决分割结果中的常见质量问题。

## 主要功能需求

### 1. 填补内部空洞（Hole Filling）
**问题描述：**
- 分割出的mask内部存在不应该有的空洞
- 例如：山体mask中间因为草、树等被误识别而产生的窟窿
- 完整对象被分割成不连续的区域

**解决方案：**
- **形态学闭运算（Morphological Closing）**：使用结构元素填补小空洞
- **连通域分析**：识别主要区域，填补内部小空洞
- **凸包补全**：对于规则形状，使用凸包算法补全
- **基于面积的过滤**：只填补小于阈值的空洞，保留大的有意义空洞

**参数配置：**
- `hole_fill_method`: 选择填补方法（morphological/connected_components/convex_hull/area_based）
- `max_hole_size`: 最大填补空洞面积阈值（像素数或面积比例）
- `kernel_size`: 形态学运算的核大小
- `iterations`: 形态学运算迭代次数

### 2. 清除零散噪点（Noise Removal）
**问题描述：**
- mask周围存在零散的小噪点
- 分割算法产生的不相关小区域
- 孤立的像素点或小连通域

**解决方案：**
- **连通域过滤**：移除面积小于阈值的连通域
- **形态学开运算**：去除小的突起和噪点
- **中值滤波**：平滑mask边缘，去除孤立点
- **主区域保留**：只保留最大的N个连通域

**参数配置：**
- `noise_removal_method`: 噪点清除方法（connected_components/morphological/median_filter/keep_largest）
- `min_area_threshold`: 最小保留区域面积阈值
- `keep_top_n`: 保留最大的N个连通域
- `filter_kernel_size`: 滤波核大小

### 3. 边缘平滑优化（Edge Smoothing）
**问题描述：**
- mask边缘锯齿状，不够平滑
- 边缘存在细小的凹凸

**解决方案：**
- **高斯模糊 + 二值化**：平滑边缘后重新二值化
- **形态学平滑**：使用开闭运算组合
- **边缘检测优化**：基于梯度的边缘优化
- **轮廓近似**：使用多边形近似简化轮廓

**参数配置：**
- `edge_smooth_method`: 边缘平滑方法
- `blur_radius`: 高斯模糊半径
- `smooth_iterations`: 平滑迭代次数

### 4. 形状完整性修复（Shape Integrity Repair）
**问题描述：**
- 对象的自然形状被破坏
- 应该连续的区域被分割

**解决方案：**
- **距离变换填补**：基于距离变换的形状修复
- **轮廓凸化**：修复凹陷的轮廓
- **形状先验约束**：基于预期形状的修复

**参数配置：**
- `shape_repair_method`: 形状修复方法
- `repair_strength`: 修复强度
- `preserve_details`: 是否保留细节

## 节点输入输出

### 输入参数（Input Types）
```python
"required": {
    "masks": ("MASK", {"tooltip": "输入的mask列表，来自分割节点的输出"}),
    "processing_mode": (["auto", "manual"], {
        "default": "auto",
        "tooltip": "处理模式：auto自动处理，manual手动调参"
    }),
},
"optional": {
    # 空洞填补参数
    "enable_hole_filling": ("BOOLEAN", {
        "default": True,
        "tooltip": "是否启用空洞填补功能"
    }),
    "hole_fill_method": (["morphological", "connected_components", "area_based"], {
        "default": "morphological",
        "tooltip": "空洞填补方法"
    }),
    "max_hole_size": ("INT", {
        "default": 1000,
        "min": 1,
        "max": 50000,
        "tooltip": "最大填补空洞大小（像素数）"
    }),
    "hole_fill_kernel_size": ("INT", {
        "default": 5,
        "min": 3,
        "max": 21,
        "step": 2,
        "tooltip": "空洞填补的形态学核大小"
    }),
    
    # 噪点清除参数
    "enable_noise_removal": ("BOOLEAN", {
        "default": True,
        "tooltip": "是否启用噪点清除功能"
    }),
    "noise_removal_method": (["connected_components", "morphological", "keep_largest"], {
        "default": "connected_components",
        "tooltip": "噪点清除方法"
    }),
    "min_area_threshold": ("INT", {
        "default": 100,
        "min": 1,
        "max": 10000,
        "tooltip": "最小保留区域面积阈值（像素数）"
    }),
    "keep_top_n": ("INT", {
        "default": 1,
        "min": 1,
        "max": 10,
        "tooltip": "保留最大的N个连通域"
    }),
    
    # 边缘平滑参数
    "enable_edge_smoothing": ("BOOLEAN", {
        "default": False,
        "tooltip": "是否启用边缘平滑功能"
    }),
    "edge_smooth_method": (["gaussian", "morphological"], {
        "default": "gaussian",
        "tooltip": "边缘平滑方法"
    }),
    "blur_radius": ("FLOAT", {
        "default": 1.0,
        "min": 0.1,
        "max": 5.0,
        "step": 0.1,
        "tooltip": "高斯模糊半径"
    }),
    
    # 高级参数
    "preview_mode": ("BOOLEAN", {
        "default": False,
        "tooltip": "预览模式：显示处理前后对比"
    }),
    "batch_processing": ("BOOLEAN", {
        "default": True,
        "tooltip": "批量处理：对所有mask应用相同参数"
    }),
}
```

### 输出参数（Output Types）
```python
RETURN_TYPES = ("MASK", "IMAGE", "STRING")
RETURN_NAMES = ("processed_masks", "preview_image", "processing_report")
OUTPUT_IS_LIST = (True, False, False)
```

## 技术实现方案

### 1. 核心算法库
- **OpenCV**：形态学操作、连通域分析、滤波
- **scikit-image**：高级图像处理算法
- **NumPy**：数组操作和计算

### 2. 处理流程
1. **输入验证**：检查mask格式和尺寸
2. **预处理**：统一mask格式，归一化
3. **主处理**：按顺序应用各种处理算法
4. **后处理**：格式转换，质量检查
5. **输出生成**：生成处理后的mask和报告

### 3. 自动模式算法
当选择auto模式时，自动分析mask特征：
- 计算空洞数量和大小分布
- 分析噪点密度和分布
- 评估边缘平滑度
- 自动选择最优参数组合

### 4. 质量评估指标
- **空洞填补率**：填补的空洞数量/总空洞数量
- **噪点清除率**：清除的小区域数量/总小区域数量
- **边缘平滑度**：处理前后边缘复杂度对比
- **形状保真度**：处理后与原始形状的相似度

## 错误处理和边界情况

### 1. 输入异常处理
- 空mask处理：返回空结果
- 格式不匹配：自动转换或报错
- 尺寸不一致：统一处理或分别处理

### 2. 参数边界检查
- 参数范围验证
- 无效参数组合检测
- 默认参数回退机制

### 3. 性能优化
- 大尺寸mask的分块处理
- 批量处理的内存管理
- 算法复杂度控制

## 用户体验优化

### 1. 预览功能
- 处理前后对比显示
- 关键参数效果预览
- 实时参数调整反馈

### 2. 智能推荐
- 根据mask特征推荐参数
- 常用参数组合预设
- 历史参数记录

### 3. 批量处理
- 统一参数应用到所有mask
- 个别mask特殊处理
- 处理进度显示

## 测试用例

### 1. 基础功能测试
- 各种大小和形状的mask
- 不同类型的空洞和噪点
- 边界情况处理

### 2. 性能测试
- 大批量mask处理
- 高分辨率mask处理
- 内存使用监控

### 3. 质量测试
- 处理效果主观评估
- 量化指标验证
- 与其他工具对比

## 未来扩展方向

### 1. AI增强
- 基于深度学习的mask优化
- 语义感知的处理策略
- 自适应参数调整

### 2. 高级功能
- 多尺度处理
- 时序一致性（视频mask）
- 交互式编辑

### 3. 集成优化
- 与其他ComfyUI节点的深度集成
- 工作流优化建议
- 自动化pipeline

---

## 开发优先级

1. **P0（核心功能）**：基础空洞填补和噪点清除
2. **P1（重要功能）**：边缘平滑和自动模式
3. **P2（增强功能）**：预览模式和批量处理
4. **P3（高级功能）**：智能推荐和性能优化 