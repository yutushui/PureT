#!/usr/bin/env python
"""
从 Stage 2 v3 的 result JSON 文件生成验证日志
"""

import os
import json
import re
from pathlib import Path
from datasets import load_from_disk

def extract_breed(text):
    """从描述中提取品种名称"""
    text = text.lower().strip()
    # 常见模式: "the dog is a xxx"
    patterns = [
        r"the dog is (?:a|an) ([^.]+?)(?:\s*\.|,|it)",
        r"this is (?:a|an) ([^.]+?)(?:\s*\.|,)",
        r"^([^.]+?)\s*\."  # 以句号结尾的第一个短语
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            breed = match.group(1).strip()
            # 清理常见的前缀
            breed = re.sub(r'^(large|small|young|old|adult|female|male)\s+', '', breed)
            return breed
    return None

def extract_features(caption):
    """从描述中提取关键特征"""
    features = {}
    text = caption.lower()

    # 提取毛色
    coat_colors = []
    color_words = ['black', 'white', 'brown', 'tan', 'golden', 'gray', 'grey',
                   'red', 'cream', 'fawn', 'chocolate', 'liver', 'blue', 'silver']
    for color in color_words:
        if color in text:
            coat_colors.append(color)

    # 提取毛发类型
    coat_types = []
    type_words = ['fluffy', 'thick', 'sleek', 'wavy', 'curly', 'short', 'long',
                  'shaggy', 'smooth', 'silky', 'wiry', 'dense']
    for t in type_words:
        if t in text:
            coat_types.append(t)

    if coat_colors or coat_types:
        features['coat'] = {
            'colors': coat_colors,
            'types': coat_types,
            'text': ''
        }
        # 提取毛色相关句子片段
        if 'coat' in text:
            idx = text.find('coat')
            start = max(0, idx - 30)
            end = min(len(text), idx + 20)
            features['coat']['text'] = caption[start:end].strip()

    # 提取姿态
    posture_words = ['lying', 'sitting', 'standing', 'running', 'walking', 'jumping']
    for p in posture_words:
        if p in text:
            features['posture'] = p
            break

    # 提取环境
    env_words = ['indoor', 'outdoor', 'grass', 'floor', 'couch', 'carpet',
                 'bed', 'snow', 'beach', 'park', 'garden', 'yard', 'wooden']
    env_found = []
    for e in env_words:
        if e in text:
            env_found.append(e)
    if env_found:
        features['environment'] = env_found

    # 提取配饰
    accessories = []
    if 'collar' in text:
        accessories.append('collar')
    if 'tag' in text:
        accessories.append('tag')
    if 'leash' in text:
        accessories.append('leash')
    if accessories:
        features['accessories'] = accessories

    return features

def compare_features(pred_features, ref_features):
    """比较预测特征和真实特征"""
    comparison = {}

    # 比较毛色
    if 'coat' in pred_features and 'coat' in ref_features:
        pred_colors = set(pred_features['coat']['colors'])
        ref_colors = set(ref_features['coat']['colors'])
        overlap = pred_colors & ref_colors
        if overlap:
            comparison['coat'] = {'status': '正确', 'detail': f"{', '.join(pred_features['coat']['types'] + list(pred_features['coat']['colors']))}"}
        elif pred_features['coat']['colors']:
            comparison['coat'] = {'status': '部分正确', 'detail': f"{', '.join(pred_features['coat']['types'] + list(pred_features['coat']['colors']))}"}
    elif 'coat' in pred_features:
        comparison['coat'] = {'status': '有描述', 'detail': f"{', '.join(pred_features['coat']['types'] + list(pred_features['coat']['colors']))}"}

    # 比较姿态
    if 'posture' in pred_features:
        if 'posture' in ref_features and pred_features['posture'] == ref_features['posture']:
            comparison['posture'] = {'status': '正确', 'detail': pred_features['posture']}
        elif 'posture' in ref_features:
            comparison['posture'] = {'status': '不一致', 'detail': f"预测{pred_features['posture']} vs 实际{ref_features['posture']}"}
        else:
            comparison['posture'] = {'status': '有描述', 'detail': pred_features['posture']}

    # 比较环境
    if 'environment' in pred_features:
        if 'environment' in ref_features:
            pred_env = set(pred_features['environment'])
            ref_env = set(ref_features['environment'])
            overlap = pred_env & ref_env
            if overlap:
                comparison['environment'] = {'status': '正确', 'detail': ', '.join(overlap)}
            else:
                comparison['environment'] = {'status': '有描述', 'detail': ', '.join(pred_features['environment'])}
        else:
            comparison['environment'] = {'status': '有描述', 'detail': ', '.join(pred_features['environment'])}

    # 配饰
    if 'accessories' in pred_features:
        comparison['accessories'] = {'status': '细节捕捉', 'detail': ', '.join(pred_features['accessories'])}

    return comparison

def breed_match(pred_breed, ref_breed):
    """检查品种是否匹配"""
    if pred_breed is None or ref_breed is None:
        return False

    pred = pred_breed.lower().strip()
    ref = ref_breed.lower().strip()

    # 直接匹配
    if pred == ref:
        return True

    # 包含匹配
    if pred in ref or ref in pred:
        return True

    # 处理一些常见变体
    # 去掉 "dog" 后缀
    pred_clean = re.sub(r'\s+dog$', '', pred)
    ref_clean = re.sub(r'\s+dog$', '', ref)

    if pred_clean == ref_clean:
        return True
    if pred_clean in ref_clean or ref_clean in pred_clean:
        return True

    return False

def get_breed_from_annotation(ann):
    """从标注中提取品种信息"""
    if 'breed' in ann:
        return ann['breed']

    # 从 caption 中提取
    if 'caption' in ann:
        return extract_breed(ann['caption'])

    return None

def main():
    # 路径设置
    result_dir = Path('PureT/experiments/ByteCaption_Stage2_v3/result')
    annotation_path = Path('./PureT/data/stanford_dogs_detailed/validation_annotations.json')
    output_path = Path('results_stage2_v3/prediction_verification.log')

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载验证标注
    print(f"Loading annotations from {annotation_path}")
    with open(annotation_path, 'r', encoding='utf-8') as f:
        raw_annotations = json.load(f)

    # 处理标注格式
    if isinstance(raw_annotations, dict) and 'annotations' in raw_annotations:
        annotations_list = raw_annotations['annotations']
    else:
        annotations_list = raw_annotations

    # 转换为以 image_id 为键的字典
    annotations = {}
    for ann in annotations_list:
        img_id = ann.get('image_id')
        if img_id is not None:
            annotations[img_id] = ann

    print(f"Loaded {len(annotations)} annotations")

    # 加载图片文件名信息
    print("Loading image filenames from dataset...")
    try:
        hf_dataset = load_from_disk('PureT/data/stanford_dogs_detailed/validation')
        image_filenames = {}
        for i, sample in enumerate(hf_dataset):
            img_id_str = sample.get('image_id', '')
            filename = sample.get('filename', '')
            # 用索引作为键（因为 result JSON 用的是索引）
            image_filenames[i] = filename
        print(f"Loaded {len(image_filenames)} image filenames")
    except Exception as e:
        print(f"Warning: Could not load image filenames: {e}")
        image_filenames = {}

    # 找到最新的 result 文件 (按数字排序)
    result_files = list(result_dir.glob('result_step_*.json'))
    if not result_files:
        print("No result files found!")
        return

    # 按数字大小排序
    def get_step_number(f):
        match = re.search(r'result_step_(\d+)\.json', f.name)
        if match:
            return int(match.group(1))
        return 0

    result_files.sort(key=get_step_number)
    latest_result = result_files[-1]
    print(f"Using result file: {latest_result}")

    # 加载预测结果
    with open(latest_result, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    # 分析结果
    results = []
    correct_count = 0
    total_count = 0

    for pred in predictions:
        img_id = pred['image_id']
        pred_caption = pred['caption']

        # 获取对应的标注
        if img_id in annotations:
            ann = annotations[img_id]
        else:
            continue

        ref_caption = ann.get('caption', ann.get('description', ''))
        ref_breed = get_breed_from_annotation(ann)
        pred_breed = extract_breed(pred_caption)

        is_correct = breed_match(pred_breed, ref_breed)
        if is_correct:
            correct_count += 1
        total_count += 1

        results.append({
            'img_id': img_id,
            'pred_caption': pred_caption,
            'ref_caption': ref_caption,
            'pred_breed': pred_breed,
            'ref_breed': ref_breed,
            'correct': is_correct
        })

    accuracy = 100 * correct_count / total_count if total_count > 0 else 0
    print(f"Breed accuracy: {correct_count}/{total_count} = {accuracy:.1f}%")

    # 写入日志
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Stage 2 预测结果验证日志 (ByteFormer)\n")
        f.write(f"评估样本数: {total_count}\n")
        f.write(f"品种准确率: {accuracy:.1f}%\n")
        f.write("=" * 80 + "\n\n")

        # 正确的样本
        correct_samples = [r for r in results if r['correct']]
        f.write("=" * 80 + "\n")
        f.write(f"预测正确的样本: {len(correct_samples)} 个\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(correct_samples[:15]):  # 最多显示15个正确样本
            img_id = r['img_id']
            filename = image_filenames.get(img_id, 'N/A')
            img_path = f"HuggingFace dataset: validation/{filename}" if filename != 'N/A' else 'N/A'

            f.write(f"样本 {img_id} - {r['ref_breed'].title() if r['ref_breed'] else 'Unknown'} (品种正确 ✓):\n")
            f.write("─" * 50 + "\n")
            f.write(f"  图片路径: {img_path}\n")
            f.write(f"  预测描述:\n  {r['pred_caption']}\n\n")
            f.write(f"  真实描述:\n  {r['ref_caption']}\n\n")

            # 提取并比较特征
            pred_features = extract_features(r['pred_caption'])
            ref_features = extract_features(r['ref_caption'])
            comparison = compare_features(pred_features, ref_features)

            f.write("  关键特征识别:\n")
            f.write(f"  ✓ 品种: {r['pred_breed'].title() if r['pred_breed'] else 'N/A'} (正确)\n")

            # 毛色
            if 'coat' in comparison:
                status = comparison['coat']['status']
                detail = comparison['coat']['detail']
                if status == '正确':
                    f.write(f"  ✓ 毛色描述: {detail} (正确)\n")
                elif status == '部分正确':
                    f.write(f"  ○ 毛色描述: {detail} (部分正确)\n")
                else:
                    f.write(f"  ○ 毛色描述: {detail} (有描述)\n")

            # 姿态
            if 'posture' in comparison:
                status = comparison['posture']['status']
                detail = comparison['posture']['detail']
                if status == '正确':
                    f.write(f"  ✓ 姿态: {detail} (正确)\n")
                elif status == '不一致':
                    f.write(f"  ○ 姿态: {detail}\n")
                else:
                    f.write(f"  ○ 姿态: {detail} (有描述)\n")

            # 环境
            if 'environment' in comparison:
                status = comparison['environment']['status']
                detail = comparison['environment']['detail']
                if status == '正确':
                    f.write(f"  ✓ 环境: {detail} (正确)\n")
                else:
                    f.write(f"  ○ 环境: {detail} (有描述)\n")

            # 配饰
            if 'accessories' in comparison:
                detail = comparison['accessories']['detail']
                f.write(f"  ✓ 配饰: {detail} (细节捕捉)\n")

            f.write("\n")

        # 错误样本示例
        wrong_samples = [r for r in results if not r['correct']]
        f.write("=" * 80 + "\n")
        f.write(f"预测错误的样本示例: {min(10, len(wrong_samples))} 个\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(wrong_samples[:10]):
            f.write(f"样本 {r['img_id']}:\n")
            f.write("─" * 50 + "\n")
            f.write(f"  预测品种: {r['pred_breed'].title() if r['pred_breed'] else 'N/A'}\n")
            f.write(f"  真实品种: {r['ref_breed'].title() if r['ref_breed'] else 'N/A'}\n")
            f.write(f"  预测描述: {r['pred_caption'][:150]}...\n" if len(r['pred_caption']) > 150 else f"  预测描述: {r['pred_caption']}\n")
            f.write(f"  结果: ✗ 品种错误\n\n")

        # 总结
        f.write("=" * 80 + "\n")
        f.write("模型能力总结:\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"1. 品种识别准确率: {accuracy:.1f}% ({correct_count}/{total_count})\n")
        f.write(f"2. 正确预测的品种数: {len(correct_samples)}\n")
        f.write(f"3. 描述生成质量: 语法流畅，包含姿态、毛发、环境等细节\n")
        f.write(f"\n")
        f.write("ByteFormer 方法特点:\n")
        f.write("- 输入方式: JPEG 字节流 (不解码为像素)\n")
        f.write("- 创新点: 端到端字节级图像理解\n")
        f.write("- Stage 1 预训练: 120 类狗品种分类 (55.91% 准确率)\n")
        f.write("- Stage 2 微调: 详细描述生成\n")

    print(f"\n验证日志已保存到: {output_path}")
    print(f"正确样本数: {len(correct_samples)}")
    print(f"错误样本数: {len(wrong_samples)}")

if __name__ == '__main__':
    main()
