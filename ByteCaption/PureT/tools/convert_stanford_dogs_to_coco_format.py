#!/usr/bin/env python3
"""
Convert label.log to COCO-style format for ByteCaption training.
"""
import os
import json
import random
from pathlib import Path

# 配置
LABEL_LOG = "/home/Yu_zhen/pureT/ByteCaption/PureT/label.log"
IMAGE_BASE_DIR = "/home/Yu_zhen/pureT/ByteCaption/PureT/stanford_dogs_jpeg"
OUTPUT_DIR = "/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_coco"
TRAIN_SPLIT = 0.8  # 80% 训练, 20% 验证

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "validation"), exist_ok=True)

print(f"正在读取 {LABEL_LOG}...")

records = []
current_record = None

with open(LABEL_LOG, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        # 跳过标题行
        if line.startswith('original_folder'):
            continue
        # 检查是否是新记录的开始
        if line.startswith('n0'):
            # 保存上一条记录
            if current_record:
                records.append(current_record)
            # 解析新记录
            parts = line.split('\t')
            if len(parts) >= 4:
                folder, breed, filename, desc_start = parts[0], parts[1], parts[2], parts[3]
                current_record = {
                    'folder': folder,
                    'breed': breed,
                    'filename': filename,
                    'description': desc_start
                }
        elif current_record and line.strip():
            # 继续累积描述
            current_record['description'] += ' ' + line.strip()

# 保存最后一条记录
if current_record:
    records.append(current_record)

print(f"成功解析 {len(records)} 条记录")

# 验证图片是否存在
print("\n验证图片...")
valid_records = []
missing_count = 0

for record in records:
    # 检查 train 和 test 目录
    img_path_train = os.path.join(IMAGE_BASE_DIR, 'train', record['folder'], record['filename'])
    img_path_test = os.path.join(IMAGE_BASE_DIR, 'test', record['folder'], record['filename'])

    if os.path.exists(img_path_train):
        record['image_path'] = img_path_train
        valid_records.append(record)
    elif os.path.exists(img_path_test):
        record['image_path'] = img_path_test
        valid_records.append(record)
    else:
        missing_count += 1
        if missing_count <= 5:
            print(f"  缺失: {record['filename']}")

print(f"有效记录: {len(valid_records)}, 缺失: {missing_count}")

# 打乱并分割数据集
random.seed(1234)
random.shuffle(valid_records)

split_idx = int(len(valid_records) * TRAIN_SPLIT)
train_records = valid_records[:split_idx]
val_records = valid_records[split_idx:]

print(f"\n数据集划分:")
print(f"  训练集: {len(train_records)} 条")
print(f"  验证集: {len(val_records)} 条")

# 生成唯一的 image_id
def make_image_id(record):
    # 从文件名提取数字部分作为 ID
    name_without_ext = record['filename'].replace('.jpg', '')
    # 尝试提取数字
    parts = name_without_ext.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        return int(parts[-1])
    return hash(record['filename']) % 1000000

# 转换为 COCO 格式
def create_coco_format(records, split_name):
    images = []
    annotations = []

    for idx, record in enumerate(records):
        image_id = make_image_id(record)

        # 添加图片信息
        images.append({
            'image_id': image_id,
            'file_name': record['filename'],
            'folder': record['folder'],  # 添加文件夹信息
            'breed': record['breed']  # 添加品种信息
        })

        # 添加标注信息
        annotations.append({
            'image_id': image_id,
            'caption': record['description'].strip('"')
        })

    coco_data = {
        'images': images,
        'annotations': annotations,
        'dataset': 'Stanford Dogs with Descriptions',
        'split': split_name,
        'total_images': len(images),
        'total_annotations': len(annotations)
    }

    return coco_data

# 生成训练和验证数据
print("\n生成 COCO 格式数据...")
train_coco = create_coco_format(train_records, 'train')
val_coco = create_coco_format(val_records, 'validation')

# 保存文件
train_json = os.path.join(OUTPUT_DIR, 'train', 'captions.json')
val_json = os.path.join(OUTPUT_DIR, 'validation', 'captions.json')
train_ids_json = os.path.join(OUTPUT_DIR, 'train_ids.json')
val_ids_json = os.path.join(OUTPUT_DIR, 'validation_ids.json')

with open(train_json, 'w', encoding='utf-8') as f:
    json.dump(train_coco, f, indent=2, ensure_ascii=False)
print(f"保存: {train_json}")

with open(val_json, 'w', encoding='utf-8') as f:
    json.dump(val_coco, f, indent=2, ensure_ascii=False)
print(f"保存: {val_json}")

# 生成 image_ids 文件
train_ids = [img['image_id'] for img in train_coco['images']]
val_ids = [img['image_id'] for img in val_coco['images']]

with open(train_ids_json, 'w', encoding='utf-8') as f:
    json.dump(train_ids, f, indent=2)
print(f"保存: {train_ids_json}")

with open(val_ids_json, 'w', encoding='utf-8') as f:
    json.dump(val_ids, f, indent=2)
print(f"保存: {val_ids_json}")

# 生成数据集信息
dataset_info = {
    'name': 'Stanford Dogs Caption Dataset',
    'version': '1.0',
    'description': 'Stanford Dogs Dataset with AI-generated captions',
    'total_records': len(valid_records),
    'train_size': len(train_records),
    'val_size': len(val_records),
    'breed_count': len(set(r['breed'] for r in valid_records)),
    'source': 'Generated from label.log'
}

info_path = os.path.join(OUTPUT_DIR, 'dataset_info.json')
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, indent=2)
print(f"保存: {info_path}")

print("\n=== 转换完成 ===")
print(f"输出目录: {OUTPUT_DIR}")
print(f"训练集: {len(train_records)} 条")
print(f"验证集: {len(val_records)} 条")
print(f"品种数: {dataset_info['breed_count']}")

print("\n示例数据:")
print(f"  训练集第1条:")
print(f"    Image ID: {train_coco['images'][0]['image_id']}")
print(f"    File: {train_coco['images'][0]['file_name']}")
print(f"    Caption: {train_coco['annotations'][0]['caption'][:100]}...")
