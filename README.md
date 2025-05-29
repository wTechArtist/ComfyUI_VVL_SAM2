# ComfyUI_VVL_SAM2

VVL SAM2 插件，提供基于 SAM2、Florence2、GroundingDINO 等模型的高级分割功能。

## 功能特性

- **Florence2 + SAM2**：结合 Florence2 视觉语言模型和 SAM2 的高质量分割
- **GroundingDINO + SAM2**：基于文本提示的精确物体检测与分割
- **SAM1 Auto Everything**：SAM1 一键分割所有物体
- **SAM2 模型加载器**：支持多种 SAM2 模型
- **SAM1 模型加载器**：支持多种 SAM1 模型（包括 SAM-HQ）
- **Mask 清理工具**：提供 mask 后处理功能

## 节点说明

### VVL SAM2 Loader
加载 SAM2 模型的专用加载器。

### VVL SAM1 Loader  
加载 SAM1 模型的专用加载器，支持以下模型：
- sam_vit_h (2.56GB) - 最高精度
- sam_vit_l (1.25GB) - 平衡性能
- sam_vit_b (375MB) - 轻量级
- sam_hq_vit_h/l/b - 高质量版本（需要安装 sam_hq）

### VVL Florence2 + SAM2
结合 Florence2 和 SAM2 的智能分割节点。

### VVL GroundingDINO + SAM2
基于文本提示的物体检测与分割。

### VVL SAM1 Auto Everything
使用 SAM1 进行一键全景分割，自动检测图像中的所有物体。

**输入参数：**
- `sam1_model`: 由 VVL SAM1 Loader 加载的模型
- `image`: 输入图像（支持批量）
- `points_per_side`: 生成网格密度
- `pred_iou_thresh`: 预测 IoU 阈值
- `stability_score_thresh`: 稳定度阈值
- `max_mask_count`: 最大 mask 数量
- `min_mask_area`: 最小 mask 面积

**输出：**
- `annotated_image`: 标注后的图像
- `object_masks`: 分割得到的 mask 列表
- `detection_json`: 检测结果 JSON
- `object_names`: 对象名称列表

### VVL Mask Cleaner
提供 mask 清理和后处理功能。

## 安装要求

基础依赖：
```
torch
torchvision
segment-anything
supervision
```

可选依赖（用于 SAM-HQ）：
```
git+https://github.com/SysCV/sam-hq.git
```

## 使用方法

1. 将插件放置在 ComfyUI 的 `custom_nodes` 目录下
2. 重启 ComfyUI
3. 在节点菜单中找到 `💃rDancer` 分类
4. 根据需要选择相应的节点

## 连线说明

**正确的连接方式：**
1. 添加 `VVL SAM1 Loader` 节点
2. 添加 `VVL SAM1 Auto Everything` 节点
3. 将 `VVL SAM1 Loader` 的输出连接到 `VVL SAM1 Auto Everything` 的 `sam1_model` 输入

## 注意事项

- 首次使用时会自动下载模型，请确保网络连接正常
- SAM-HQ 模型需要额外安装 sam-hq 库
- 建议使用 GPU 以获得更好的性能
- 模型会缓存在 ComfyUI 的 models 目录下
- 如果连线识别有问题，请重启 ComfyUI

## 更新日志

- v1.1.1: 修复 VVL SAM1 Loader 连线识别问题
- v1.1.0: 添加独立的 VVL SAM1 Loader，支持完整的 SAM1 模型管理
- v1.0.0: 初始版本，支持 SAM2、Florence2、GroundingDINO 集成 