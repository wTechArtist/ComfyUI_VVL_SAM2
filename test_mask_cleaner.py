"""
测试VVL_MaskCleaner节点的功能
"""

import torch
import cv2
import numpy as np
from vvl_nodes_mask_cleaner import VVL_MaskCleaner, fill_internal_holes, remove_small_regions
import matplotlib.pyplot as plt


def create_test_mask_with_holes():
    """创建一个带有内部空洞的测试mask"""
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # 创建一个大的白色区域（主体）
    cv2.rectangle(mask, (50, 50), (200, 200), 255, -1)
    
    # 在内部创建一些黑色空洞
    cv2.circle(mask, (100, 100), 20, 0, -1)  # 空洞1
    cv2.rectangle(mask, (150, 150), (170, 170), 0, -1)  # 空洞2
    
    # 添加一些零碎的小区域
    cv2.circle(mask, (30, 30), 5, 255, -1)  # 小噪点1
    cv2.circle(mask, (220, 220), 8, 255, -1)  # 小噪点2
    cv2.rectangle(mask, (10, 200), (20, 210), 255, -1)  # 小噪点3
    
    return mask


def create_test_mask_multi_objects():
    """创建一个包含多个对象的测试mask"""
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # 创建三个主要对象
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)  # 对象1（大）
    cv2.circle(mask, (150, 50), 30, 255, -1)  # 对象2（中）
    cv2.ellipse(mask, (180, 180), (40, 25), 45, 0, 360, 255, -1)  # 对象3（中）
    
    # 在对象1中创建空洞
    cv2.circle(mask, (50, 50), 10, 0, -1)
    
    # 添加一些小噪点
    cv2.circle(mask, (200, 100), 3, 255, -1)
    cv2.circle(mask, (100, 150), 2, 255, -1)
    
    return mask


def visualize_results(original, filled, cleaned, title="Mask Cleaning Results"):
    """可视化处理结果"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original Mask")
    axes[0].axis('off')
    
    axes[1].imshow(filled, cmap='gray')
    axes[1].set_title("After Hole Filling")
    axes[1].axis('off')
    
    axes[2].imshow(cleaned, cmap='gray')
    axes[2].set_title("After Cleaning")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"test_results_{title.replace(' ', '_').lower()}.png")
    plt.close()


def test_basic_functions():
    """测试基础函数功能"""
    print("=== 测试基础函数 ===")
    
    # 测试1：单对象带空洞
    print("\n测试1：单对象带空洞")
    mask1 = create_test_mask_with_holes()
    filled1 = fill_internal_holes(mask1)
    cleaned1 = remove_small_regions(filled1, keep_largest_n=1)
    visualize_results(mask1, filled1, cleaned1, "Single Object with Holes")
    
    # 测试2：多对象场景
    print("\n测试2：多对象场景")
    mask2 = create_test_mask_multi_objects()
    filled2 = fill_internal_holes(mask2)
    cleaned2 = remove_small_regions(filled2, keep_largest_n=2)
    visualize_results(mask2, filled2, cleaned2, "Multiple Objects")


def test_node_class():
    """测试节点类功能"""
    print("\n=== 测试节点类 ===")
    
    # 创建节点实例
    node = VVL_MaskCleaner()
    
    # 创建测试数据
    mask_np = create_test_mask_with_holes()
    mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0)
    
    # 测试不同的处理模式
    print("\n测试处理模式：both")
    cleaned_masks, info = node.clean_masks([mask_tensor], keep_largest_n=1, processing_mode="both")
    print(f"处理信息：{info}")
    
    print("\n测试处理模式：fill_only")
    cleaned_masks, info = node.clean_masks([mask_tensor], keep_largest_n=1, processing_mode="fill_only")
    print(f"处理信息：{info}")
    
    print("\n测试处理模式：clean_only")
    cleaned_masks, info = node.clean_masks([mask_tensor], keep_largest_n=1, processing_mode="clean_only")
    print(f"处理信息：{info}")
    
    # 测试批量处理
    print("\n测试批量处理")
    mask_list = [mask_tensor, mask_tensor, mask_tensor]
    cleaned_masks, info = node.clean_masks(mask_list, keep_largest_n=1, processing_mode="both")
    print(f"批量处理信息：\n{info}")
    print(f"处理了 {len(cleaned_masks)} 个mask")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    node = VVL_MaskCleaner()
    
    # 测试空mask
    print("\n测试1：空mask")
    empty_mask = torch.zeros((256, 256))
    cleaned_masks, info = node.clean_masks([empty_mask], keep_largest_n=1, processing_mode="both")
    print(f"空mask处理信息：{info}")
    
    # 测试全白mask
    print("\n测试2：全白mask")
    full_mask = torch.ones((256, 256))
    cleaned_masks, info = node.clean_masks([full_mask], keep_largest_n=1, processing_mode="both")
    print(f"全白mask处理信息：{info}")
    
    # 测试只有小噪点的mask
    print("\n测试3：只有小噪点的mask")
    noise_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(noise_mask, (50, 50), 3, 255, -1)
    cv2.circle(noise_mask, (100, 100), 2, 255, -1)
    cv2.circle(noise_mask, (150, 150), 4, 255, -1)
    noise_tensor = torch.from_numpy(noise_mask.astype(np.float32) / 255.0)
    cleaned_masks, info = node.clean_masks([noise_tensor], keep_largest_n=1, processing_mode="both")
    print(f"噪点mask处理信息：{info}")


if __name__ == "__main__":
    print("开始测试VVL_MaskCleaner节点...")
    
    # 运行所有测试
    test_basic_functions()
    test_node_class()
    test_edge_cases()
    
    print("\n测试完成！")
    print("生成的测试结果图片已保存到当前目录。") 