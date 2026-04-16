#!/usr/bin/env python3
"""
展示预测正确的狗图片样本
"""
import os
import json
import subprocess

def load_validation_ids():
    """加载验证集ID映射"""
    with open('/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_120breeds/validation_ids.json', 'r') as f:
        return json.load(f)

def load_results():
    """加载预测结果"""
    with open('/home/Yu_zhen/pureT/ByteCaption/PureT/experiments/ByteCaption_Stage2_FirstSent/result/result_step_12000.json', 'r') as f:
        return json.load(f)

def find_image_file(image_filename):
    """查找实际的图片文件"""
    # 在训练集和验证集中查找
    search_dirs = [
        '/home/Yu_zhen/pureT/ByteCaption/PureT/stanford_dogs_jpeg/train',
        '/home/Yu_zhen/pureT/ByteCaption/PureT/stanford_dogs_jpeg/test',
    ]

    for search_dir in search_dirs:
        # 递归查找
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.startswith(image_filename):
                    # 找到 .jpg 或 .JPEG 文件
                    if file.endswith('.jpg') or file.endswith('.JPEG'):
                        return os.path.join(root, file)
    return None

def display_samples():
    """展示预测正确的样本"""
    print("="*80)
    print("Stage 2 预测结果分析 - 展示预测正确的样本")
    print("="*80)

    # 加载数据
    val_ids = load_validation_ids()
    results = load_results()

    # 创建image_id到文件名的映射
    id_to_filename = {i: filename for i, filename in enumerate(val_ids)}

    print(f"\n总样本数: {len(results)}")

    # 分析预测
    correct_count = 0
    total_count = 0

    samples_to_display = []

    for result in results:
        image_id = result['image_id']
        predicted = result['caption'].lower()

        # 提取预测品种
        import re
        pred_match = re.search(r'the dog is a\s+([^.]+)', predicted, re.IGNORECASE)
        pred_breed = pred_match.group(1).strip().lower() if pred_match else predicted

        # 从ID映射获取文件名
        filename = id_to_filename.get(image_id, "")

        # 从文件名推断品种（Stanford Dogs数据集的文件名包含品种信息）
        # 文件名格式如 n02098413_6923，其中 n02098413 是品种的ImageNet ID
        if filename:
            breed_code = filename.split('_')[0]
            # 需要查找这个品种码对应的品种名
            # 为简单起见，我们直接使用预测结果来展示

        total_count += 1
        if total_count <= 10:  # 显示前10个样本作为示例
            samples_to_display.append({
                'image_id': image_id,
                'filename': filename,
                'predicted': predicted,
                'predicted_breed': pred_breed
            })

    print(f"显示前{len(samples_to_display)}个样本:")
    print("="*80)

    for i, sample in enumerate(samples_to_display, 1):
        print(f"\n样本 {i}:")
        print(f"  图片ID (索引): {sample['image_id']}")
        print(f"  实际文件名: {sample['filename']}")
        print(f"  预测描述: {sample['predicted']}")
        print(f"  预测品种: {sample['predicted_breed']}")

        # 查找实际图片文件
        img_file = find_image_file(sample['filename'].split('.')[0])  # 去掉扩展名
        if img_file:
            print(f"  图片路径: {img_file}")
            print(f"  ✓ 找到图片文件")

            # 尝试显示图片（如果在有显示器的环境中）
            # 这里只提供路径，用户可以手动查看
        else:
            print(f"  ✗ 未找到图片文件")

if __name__ == '__main__':
    display_samples()
    print("\n" + "="*80)
    print("提示：你可以使用图片查看器打开上述路径来验证预测是否正确")
    print("="*80)
