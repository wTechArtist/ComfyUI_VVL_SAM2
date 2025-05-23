# Florence2SAM2 节点 - 检测信息JSON功能

## 新增功能

这个节点现在已经增强，可以返回检测到的对象的详细信息，包括对象名称和边界框坐标，以JSON格式输出。

## 输出内容

节点现在有4个输出：

1. **annotated_image** (IMAGE) - 带有标注的图像
2. **object_masks** (MASK, 列表) - 每个检测对象的分割掩码
3. **masked_image** (IMAGE) - 应用掩码后的图像  
4. **detection_json** (STRING) - 包含检测信息的JSON字符串

## JSON格式

第四个输出 `detection_json` 返回以下格式的JSON字符串：

```json
{
  "image_width": 1000,
  "image_height": 562,
  "objects": [
    {
      "name": "对象名称1",
      "bbox_2d": [x1, y1, x2, y2]
    },
    {
      "name": "对象名称2", 
      "bbox_2d": [x1, y1, x2, y2]
    }
  ]
}
```

### 字段说明：

- `image_width`: 输入图像的宽度（像素）
- `image_height`: 输入图像的高度（像素）
- `objects`: 检测到的对象列表
  - `name`: 对象的名称/标签
  - `bbox_2d`: 边界框坐标 [x1, y1, x2, y2]，其中(x1,y1)是左上角，(x2,y2)是右下角

## 使用方法

### 开放词汇检测模式
当你提供一个prompt（如"cat, dog, person"）时，节点会：
1. 检测prompt中指定的对象
2. 为每个检测到的对象返回其对应的名称和边界框

### 自动描述模式  
当prompt为空时，节点会：
1. 自动生成图像描述
2. 从描述中提取关键短语进行定位
3. 返回检测到的对象及其边界框信息

## 示例输出

```json
{
  "image_width": 800,
  "image_height": 600,
  "objects": [
    {
      "name": "cat",
      "bbox_2d": [120, 150, 300, 400]
    },
    {
      "name": "dog",
      "bbox_2d": [350, 200, 600, 500]
    }
  ]
}
```

## 注意事项

- 边界框坐标使用图像像素坐标系，(0,0)为图像左上角
- 对象名称来自于用户输入的prompt或Florence2自动生成的描述
- 如果没有检测到任何对象，objects数组将为空
- JSON字符串使用UTF-8编码，支持中文等非ASCII字符 