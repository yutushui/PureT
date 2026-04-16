#!/usr/bin/env python3
"""
创建验证集可视化HTML - 显示实际图片和预测结果
"""
import os
import json
import base64
from pathlib import Path

def load_data():
    """加载所有数据"""
    # 验证集ID
    with open('/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_120breeds/validation_ids.json') as f:
        val_ids = json.load(f)

    # 图片信息
    with open('/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_120breeds/validation_image_info.json') as f:
        img_info = json.load(f)

    # 标注
    with open('/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_120breeds/validation/annotations.json') as f:
        annotations = json.load(f)

    # 预测结果
    result_path = '/home/Yu_zhen/pureT/ByteCaption/PureT/experiments/ByteCaption_Stage2_FirstSent/result/result_step_12000.json'
    if os.path.exists(result_path):
        with open(result_path) as f:
            results = json.load(f)
    else:
        results = None

    return val_ids, img_info, annotations, results

def image_to_base64(img_path):
    """将图片转换为base64编码"""
    try:
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

def create_html(num_samples=20):
    """创建HTML可视化文件"""

    print("正在加载数据...")
    val_ids, img_info, annotations, results = load_data()

    print(f"验证集大小: {len(val_ids)}")
    print(f"预测结果: {'已加载' if results else '未找到'}")

    # 创建HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>验证集可视化 - 真实图片</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .summary {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .sample {
            border: 1px solid #ddd;
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
            display: flex;
            gap: 20px;
        }
        .img-container {
            flex: 0 0 300px;
        }
        .img-container img {
            max-width: 300px;
            max-height: 300px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .info {
            flex: 1;
        }
        .info-item {
            margin: 10px 0;
        }
        .label {
            font-weight: bold;
            color: #555;
            display: inline-block;
            min-width: 120px;
        }
        .ground-truth {
            color: #4CAF50;
            font-weight: bold;
        }
        .predicted {
            color: #2196F3;
            font-weight: bold;
        }
        .correct {
            background-color: #e8f5e9;
            padding: 5px 10px;
            border-radius: 3px;
            color: #2e7d32;
        }
        .incorrect {
            background-color: #ffebee;
            padding: 5px 10px;
            border-radius: 3px;
            color: #c62828;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐕 验证集可视化 - 真实图片</h1>
        <div class="summary">
            <strong>说明：</strong>这里显示了验证集样本的实际图片、真实品种和模型预测结果。
            <br>图片直接嵌入HTML，可以离线查看。
        </div>
"""

    # 统计正确率
    correct_count = 0
    total_count = 0

    for i in range(min(num_samples, len(img_info))):
        info = img_info[i]
        gt_caption = annotations['annotations'][i]['caption']
        gt_breed = gt_caption.replace('The dog is a ', '').replace('.', '')

        # 获取预测结果
        pred_caption = ""
        pred_breed = ""
        is_correct = None

        if results and i < len(results):
            pred_caption = results[i]['caption']
            pred_breed = pred_caption.replace('The dog is a ', '').replace('.', '').lower()
            gt_breed_lower = gt_breed.lower()
            is_correct = pred_breed == gt_breed_lower

            total_count += 1
            if is_correct:
                correct_count += 1

        # 图片路径
        img_path = f"/home/Yu_zhen/pureT/ByteCaption/PureT/stanford_dogs_jpeg/{info['split']}/{info['folder']}/{info['filename']}"

        # 转换为base64
        b64_data = image_to_base64(img_path)
        img_src = f"data:image/jpeg;base64,{b64_data}" if b64_data else ""

        html += f"""
        <div class="sample">
            <div class="img-container">
                {'<img src="' + img_src + '" alt="Dog">' if b64_data else '<p style="color:red">图片加载失败</p>'}
            </div>
            <div class="info">
                <div class="info-item">
                    <span class="label">样本编号:</span> {i}
                </div>
                <div class="info-item">
                    <span class="label">文件名:</span> {info['filename']}
                </div>
                <div class="info-item">
                    <span class="label">文件夹:</span> {info['folder']}
                </div>
                <div class="info-item">
                    <span class="label">真实品种:</span>
                    <span class="ground-truth">{gt_breed}</span>
                </div>
                <div class="info-item">
                    <span class="label">真实标注:</span>
                    <span class="ground-truth">"{gt_caption}"</span>
                </div>
"""

        if pred_caption:
            html += f"""
                <div class="info-item">
                    <span class="label">模型预测:</span>
                    <span class="predicted">"{pred_caption}"</span>
                </div>
                <div class="info-item">
                    <span class="label">结果:</span>
                    <span class="{'correct' if is_correct else 'incorrect'}">{'✓ 正确' if is_correct else '✗ 错误'}</span>
                </div>
"""

        html += """
            </div>
        </div>
"""

    # 添加统计信息
    if total_count > 0:
        accuracy = correct_count / total_count * 100
        html += f"""
        <div class="summary">
            <strong>统计信息：</strong>
            <br>显示样本数: {min(num_samples, len(img_info))}
            <br>预测正确: {correct_count} / {total_count}
            <br>准确率: {accuracy:.2f}%
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    # 保存HTML
    output_path = '/home/Yu_zhen/pureT/ByteCaption/validation_visualization.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ HTML文件已创建: {output_path}")
    print(f"   包含 {min(num_samples, len(img_info))} 个样本")
    print(f"   图片已嵌入，可直接用浏览器打开")

    return output_path

if __name__ == '__main__':
    create_html(num_samples=30)
