#!/usr/bin/env python3
"""
将 mini_coco 转换为 HuggingFace 格式。
"""

import os
import json
from pathlib import Path
from PIL import Image
import datasets

def prepare_mini_coco_hf():
    """将 mini_coco 转换为 HuggingFace 格式。"""

    # 读取 ID 文件
    train_ids_path = "PureT/data/coco_karpathy/mini_coco/train_ids.json"
    val_ids_path = "PureT/data/coco_karpathy/mini_coco/validation_ids.json"

    with open(train_ids_path, 'r') as f:
        train_ids = json.load(f)

    with open(val_ids_path, 'r') as f:
        val_ids = json.load(f)

    # 训练数据
    train_dir = Path("PureT/data/coco_karpathy/mini_coco/train")
    train_data = []

    # 示例字幕（用于测试）
    sample_captions = [
        "A photo of a cat",
        "A picture of a dog",
        "An image showing people",
        "A scene with buildings"
    ]

    for i, img_name in enumerate(train_ids):
        img_path = train_dir / img_name
        if img_path.exists():
            img = Image.open(img_path)
            caption = sample_captions[i % len(sample_captions)]
            train_data.append({
                'image': img,
                'caption': caption,
                'id': i,
                'license': 1,
                'coco_url': '',
                'height': img.height,
                'width': img.width,
                'date_captured': '2024-01-01',
                'flickr_url': ''
            })
            print(f"Added train sample {i}: {img_name}")

    # 验证数据
    val_dir = Path("PureT/data/coco_karpathy/mini_coco/validation")
    val_data = []

    for i, img_name in enumerate(val_ids):
        img_path = val_dir / img_name
        if img_path.exists():
            img = Image.open(img_path)
            caption = sample_captions[i % len(sample_captions)]
            val_data.append({
                'image': img,
                'caption': caption,
                'id': len(train_data) + i,
                'license': 1,
                'coco_url': '',
                'height': img.height,
                'width': img.width,
                'date_captured': '2024-01-01',
                'flickr_url': ''
            })
            print(f"Added val sample {i}: {img_name}")

    # 创建 HuggingFace 数据集
    train_dataset = datasets.Dataset.from_list(train_data)
    val_dataset = datasets.Dataset.from_list(val_data)

    # 保存到磁盘
    output_dir = Path("PureT/data/coco_karpathy/mini_coco_hf")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving train dataset...")
    train_dataset.save_to_disk(str(output_dir / "train"))

    print(f"Saving validation dataset...")
    val_dataset.save_to_disk(str(output_dir / "validation"))

    print(f"\nSuccess! HuggingFace dataset saved to {output_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print("\nUpdate config file to use:")
    print(f"DATA_LOADER.TRAIN_ID: '{output_dir}/train_ids.json'")
    print(f"DATA_LOADER.VAL_ID: '{output_dir}/validation_ids.json'")

    # 创建对应的 ID 文件
    train_ids_output = {str(i): train_ids[i] for i in range(len(train_ids))}
    val_ids_output = {str(len(train_data) + i): val_ids[i] for i in range(len(val_ids))}

    with open(output_dir / "train_ids.json", 'w') as f:
        json.dump(train_ids_output, f, indent=2)

    with open(output_dir / "validation_ids.json", 'w') as f:
        json.dump(val_ids_output, f, indent=2)

    print(f"\nCreated ID files:")
    print(f"- {output_dir}/train_ids.json")
    print(f"- {output_dir}/validation_ids.json")

if __name__ == '__main__':
    prepare_mini_coco_hf()
