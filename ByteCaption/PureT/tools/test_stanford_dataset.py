#!/usr/bin/env python3
"""测试 Stanford Dogs 数据集加载"""
import sys
sys.path.insert(0, '/home/Yu_zhen/pureT/ByteCaption')

from datasets import load_from_disk
from PIL import Image

# 加载数据集
print("加载 HuggingFace 数据集...")
train_ds = load_from_disk('/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_hf/train')
val_ds = load_from_disk('/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_hf/validation')

print(f"训练集大小: {len(train_ds)}")
print(f"验证集大小: {len(val_ds)}")

# 显示样本
print("\n=== 训练集样本 ===")
sample = train_ds[0]
print(f"Image ID: {sample['image_id']}")
print(f"Breed: {sample['breed']}")
print(f"Caption: {sample['caption'][:100]}...")
print(f"Image path: {sample['image_path']}")

# 检查图像是否存在
import os
if os.path.exists(sample['image_path']):
    print(f"✓ 图像存在")
    img = Image.open(sample['image_path'])
    print(f"  尺寸: {img.size}")
    print(f"  模式: {img.mode}")
else:
    print(f"✗ 图像不存在: {sample['image_path']}")

print("\n=== 验证集样本 ===")
sample = val_ds[0]
print(f"Image ID: {sample['image_id']}")
print(f"Breed: {sample['breed']}")
print(f"Caption: {sample['caption'][:100]}...")

print("\n数据集准备完成！")
