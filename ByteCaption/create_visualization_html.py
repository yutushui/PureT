#!/usr/bin/env python3
"""
创建包含正确预测样本的HTML文件
"""
import os
import json
import base64
from pathlib import Path

def load_validation_ids():
    """加载验证集ID映射"""
    with open('/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_120breeds/validation_ids.json', 'r') as f:
        return json.load(f)

def load_results():
    """加载预测结果"""
    with open('/home/Yu_zhen/pureT/ByteCaption/PureT/experiments/ByteCaption_Stage2_FirstSent/result/result_step_12000.json', 'r') as f:
        return json.load(f)

def load_annotations():
    """加载标注"""
    with open('/home/Yu_zhen/pureT/ByteCaption/PureT/data/stanford_dogs_120breeds/validation/annotations.json', 'r') as f:
        data = json.load(f)
    # 创建ID到标注的映射
    return {ann['image_id']: ann['caption'] for ann in data['annotations']}

def find_image_path(image_filename):
    """查找图片文件"""
    search_dirs = [
        '/home/Yu_zhen/pureT/ByteCaption/PureT/stanford_dogs_jpeg/train',
        '/home/Yu_zhen/pureT/ByteCaption/PureT/stanford_dogs_jpeg/test',
    ]

    for search_dir in search_dirs:
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.startswith(image_filename):
                    if file.endswith('.jpg') or file.endswith('.JPEG'):
                        return os.path.join(root, file)
    return None

def create_html():
    """创建HTML可视化文件"""

    print("正在创建HTML文件...")

    val_ids = load_validation_ids()
    results = load_results()
    annotations = load_annotations()

    # 获取前20个样本
    samples = []
    for result in results[:20]:
        image_id = result['image_id']
        filename = val_ids[image_id]
        predicted = result['caption']
        gt_caption = annotations.get(image_id, "")

        img_path = find_image_path(filename.split('.')[0])

        samples.append({
            'image_id': image_id,
            'filename': filename,
            'predicted': predicted,
            'ground_truth': gt_caption,
            'img_path': img_path
        })

    # 创建HTML
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Stage 2 预测结果可视化</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
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
        }
        .sample-header {
            background-color: #f8f9fa;
            padding: 10px;
            margin: -15px -15px 10px -15px;
            border-bottom: 1px solid #ddd;
            border-radius: 8px 8px 0 0;
            font-weight: bold;
        }
        .img-container {
            display: flex;
            gap: 20px;
            align-items: flex-start;
        }
        .img-wrapper {
            flex: 0 0 auto;
        }
        .img-wrapper img {
            max-width: 300px;
            height: auto;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .info {
            flex: 1;
        }
        .info-item {
            margin: 8px 0;
        }
        .label {
            font-weight: bold;
            color: #555;
            display: inline-block;
            min-width: 120px;
        }
        .predicted {
            color: #2196F3;
        }
        .ground-truth {
            color: #4CAF50;
        }
        .note {
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐕 Stage 2 预测结果可视化</h1>

        <div class="summary">
            <strong>说明：</strong>这里显示了前20个验证集样本，包含图片、预测结果和真实标注。
            <br>
            你可以通过对比图片和文字描述来验证模型的预测是否准确。
        </div>

        <div class="note">
            ⚠️ <strong>数据集问题注意：</strong>发现验证集的图片文件都在Lhasa文件夹下，
            但标注却是不同的品种。这是数据集准备过程中的错误。
            <strong>请以实际图片为准来判断模型预测是否正确！</strong>
        </div>
"""

    for i, sample in enumerate(samples, 1):
        img_path = sample['img_path']
        filename = sample['filename']

        # 转换图片路径为相对路径
        rel_path = img_path.replace('/home/Yu_zhen/pureT/ByteCaption/', '../')

        html_content += f"""
        <div class="sample">
            <div class="sample-header">
                样本 {i} (图片ID: {sample['image_id']})
            </div>
            <div class="img-container">
                <div class="img-wrapper">
                    <img src="{rel_path}" alt="Dog image {i}" onerror="this.parentElement.innerHTML='<p>图片加载失败</p>'">
                </div>
                <div class="info">
                    <div class="info-item">
                        <span class="label">文件名:</span>
                        {filename}
                    </div>
                    <div class="info-item">
                        <span class="label">模型预测:</span>
                        <span class="predicted">"{sample['predicted']}"</span>
                    </div>
                    <div class="info-item">
                        <span class="label">真实标注:</span>
                        <span class="ground-truth">"{sample['ground_truth']}"</span>
                    </div>
                    <div class="info-item">
                        <span class="label">图片路径:</span>
                        {img_path}
                    </div>
                </div>
            </div>
        </div>
"""

    html_content += """
    </div>

    <div style="text-align: center; margin-top: 30px; color: #666;">
        <p>提示：如果图片无法加载，请直接在终端中查看图片路径</p>
    </div>
</body>
</html>
"""

    # 保存HTML文件
    output_path = '/home/Yu_zhen/pureT/ByteCaption/stage2_predictions_visualization.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✅ HTML文件已创建: {output_path}")
    print("\n使用方法：")
    print("1. 将HTML文件下载到本地电脑")
    print("2. 用浏览器打开HTML文件")
    print("3. 查看图片和预测结果")
    print(f"\n文件路径: {output_path}")
    print(f"下载命令: scp root123-Super-server:{output_path} .")

if __name__ == '__main__':
    create_html()
