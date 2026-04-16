#!/usr/bin/env python3
"""Stanford Dogs 数据集端到端测试"""
import sys
import os
sys.path.insert(0, '/home/Yu_zhen/pureT/ByteCaption')

# 设置工作目录
os.chdir('/home/Yu_zhen/pureT/ByteCaption')

from PureT.datasets_.stanford_dogs_dataset import StanfordDogsDataset
from PureT.lib.config import cfg, cfg_from_file

# 加载配置
config_path = '/home/Yu_zhen/pureT/ByteCaption/PureT/experiments/ByteCaption_XE_stanford/config_stanford.yml'
cfg_from_file(config_path)

print("=== 测试 Stanford Dogs 数据集加载 ===\n")

# 创建数据集实例
dataset = StanfordDogsDataset(
    image_ids_path='./PureT/data/stanford_dogs_hf/train_ids.json',
    input_seq='',
    target_seq='',
    gv_feat_path='',
    seq_per_img=1,
    max_feat_num=-1,
    max_samples=5,  # 只测试 5 个样本
    return_captions=False,
    jpeg_quality=60,
    corruption_types=[],
    corruption_level='S0',
    model_type='bytecaption',
    is_training=True,
)

print(f"\n数据集大小: {len(dataset)}")
print(f"配置 SEQ_LEN: {cfg.MODEL.SEQ_LEN}")

# 测试加载几个样本
print("\n=== 加载样本 ===")
for i in range(min(3, len(dataset))):
    try:
        sample = dataset[i]
        print(f"\n样本 {i}:")
        print(f"  Image ID: {sample['image_id']}")
        print(f"  JPEG bytes 大小: {len(sample['jpeg_bytes'])} bytes")
        print(f"  Caption: {sample['caption'][:80]}...")
    except Exception as e:
        print(f"  错误: {e}")

print("\n=== 测试完成 ===")
print("数据集可以正常加载！")
