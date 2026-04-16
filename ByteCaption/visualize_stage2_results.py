#!/usr/bin/env python3
"""
Stage 2 结果可视化脚本
展示预测正确的样本图片
"""
import os
import json
import sys
import glob
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/Yu_zhen/pureT/ByteCaption')

def load_results(result_file):
    """加载评估结果"""
    with open(result_file, 'r') as f:
        return json.load(f)

def load_annotations(ann_file):
    """加载验证集标注"""
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # 创建image_id到标注的映射
    id_to_ann = {}
    for ann in data.get('annotations', []):
        id_to_ann[ann['image_id']] = ann

    return id_to_ann

def extract_breed_name(caption):
    """从caption中提取品种名"""
    import re
    caption = caption.strip()
    # 匹配 "the dog is a [breed]" 格式
    match = re.search(r'the\s+dog\s+is\s+a\s+([^.]+?)(?:\.|$)', caption, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return caption.lower()

def find_image_path(image_id, data_root):
    """查找图片文件路径"""
    # 在多个可能的目录中查找
    search_paths = [
        f"{data_root}/train",
        f"{data_root}/test",
        f"{data_root}/validation",
        f"{data_root}/val",
        "PureT/stanford_dogs_jpeg/train",
        "PureT/stanford_dogs_jpeg/test",
        "PureT/stanford_dogs_jpeg/validation",
    ]

    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue

        # 查找npy文件
        npy_files = glob.glob(f"{search_path}/{image_id}.npy")
        if npy_files:
            return npy_files[0]

        # 查找jpg文件
        jpg_files = glob.glob(f"{search_path}/{image_id}.jpg")
        if jpg_files:
            return jpg_files[0]

        # 查找JPEG文件
        JPEG_files = glob.glob(f"{search_path}/{image_id}.JPEG")
        if JPEG_files:
            return JPEG_files[0]

    return None

def display_predictions(result_file, ann_file, num_samples=10, show_correct_only=True):
    """展示预测结果"""
    print("="*80)
    print("Stage 2 预测结果可视化")
    print("="*80)

    # 加载数据
    results = load_results(result_file)
    id_to_ann = load_annotations(ann_file)

    print(f"\n总预测数量: {len(results)}")

    # 分析预测结果
    correct_predictions = []
    incorrect_predictions = []

    for result in results:
        image_id = result['image_id']
        predicted_caption = result['caption']
        predicted_breed = extract_breed_name(predicted_caption)

        # 获取真实标注
        if image_id not in id_to_ann:
            continue

        gt_caption = id_to_ann[image_id]['caption']
        gt_breed = extract_breed_name(gt_caption)

        is_correct = (predicted_breed == gt_breed)

        sample_info = {
            'image_id': image_id,
            'predicted_breed': predicted_breed,
            'predicted_caption': predicted_caption,
            'gt_breed': gt_breed,
            'gt_caption': gt_caption,
            'correct': is_correct
        }

        if is_correct:
            correct_predictions.append(sample_info)
        else:
            incorrect_predictions.append(sample_info)

    print(f"预测正确: {len(correct_predictions)}")
    print(f"预测错误: {len(incorrect_predictions)}")
    print(f"准确率: {len(correct_predictions) / (len(correct_predictions) + len(incorrect_predictions)) * 100:.2f}%")

    # 展示预测正确的样本
    if show_correct_only and correct_predictions:
        samples_to_show = correct_predictions[:num_samples]
    elif not show_correct_only and incorrect_predictions:
        samples_to_show = incorrect_predictions[:num_samples]
    else:
        samples_to_show = []

    print("\n" + "="*80)
    print(f"展示 {'预测正确' if show_correct_only else '预测错误'} 的样本 (前{len(samples_to_show)}个):")
    print("="*80)

    for i, sample in enumerate(samples_to_show):
        print(f"\n样本 {i+1}:")
        print(f"  图片ID: {sample['image_id']}")
        print(f"  预测品种: {sample['predicted_breed']}")
        print(f"  真实品种: {sample['gt_breed']}")
        print(f"  预测caption: {sample['predicted_caption']}")
        print(f"  真实caption: {sample['gt_caption']}")
        print(f"  结果: {'✓ 正确' if sample['correct'] else '✗ 错误'}")

        # 查找图片路径
        img_path = find_image_path(sample['image_id'], 'PureT/stanford_dogs_120breeds')
        if img_path:
            print(f"  图片路径: {img_path}")
        else:
            print(f"  图片路径: 未找到 (图片ID: {sample['image_id']})")

def main():
    """主函数"""
    # 使用最佳结果的JSON文件
    result_file = '/home/Yu_zhen/pureT/ByteCaption/PureT/experiments/ByteCaption_Stage2_FirstSent/result/result_step_12000.json'
    ann_file = '/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_120breeds/validation/annotations.json'

    print("正在分析结果...")
    display_predictions(result_file, ann_file, num_samples=20, show_correct_only=True)

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == '__main__':
    main()
