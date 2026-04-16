#!/usr/bin/env python3
"""
评估 ByteCaption 中 ByteFormer backbone 在 Stanford Dogs 上的分类准确率
"""

import os
import sys
import json
import re
import torch
import torch.nn as nn
import numpy as np
import types
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.config import cfg, cfg_from_file
from models.bytecaption_model import PureT_byteformer
from datasets_.coco_dataset import CocoDataset
from torch.utils.data import DataLoader
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
import argparse


def extract_breed_from_caption(caption):
    """从描述中提取品种名称"""
    match = re.search(r'The dog is a ([^.]+)\.', caption, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return None


def load_ground_truth(ann_file, img_ids_file):
    """加载品种标签"""
    with open(img_ids_file, 'r') as f:
        img_ids_data = json.load(f)

    with open(ann_file, 'r') as f:
        data = json.load(f)

    id_to_breed = {}
    for ann in data['annotations']:
        breed = extract_breed_from_caption(ann['caption'])
        if breed:
            id_to_breed[ann['image_id']] = breed

    return id_to_breed


def extract_backbone_features(model, dataloader, device, id_to_breed, max_samples=200, opts=None):
    """提取 backbone 特征"""
    model.eval()
    model.backbone.eval()
    features = []
    labels = []
    image_ids = []

    print(f"提取 backbone 特征 (最多 {max_samples} 个样本)...")

    batch_count = 0
    success_count = 0

    # 直接访问数据集获取 image_id
    dataset = dataloader.dataset

    with torch.no_grad():
        for batch in tqdm(dataloader, total=min(max_samples // dataloader.batch_size + 1, len(dataloader))):
            if len(features) >= max_samples:
                break

            batch_count += 1

            try:
                # 获取数据集索引（batch 中的第一个元素）
                indices_tensor = batch[0]
                if isinstance(indices_tensor, torch.Tensor):
                    dataset_idx = int(indices_tensor[0][0].item())
                else:
                    dataset_idx = int(indices_tensor[0])

                # 从 HuggingFace 数据集获取实际的 image_id
                try:
                    hf_sample = dataset.ds[dataset_idx]
                    actual_image_id = hf_sample.get('image_id', dataset_idx)
                    if isinstance(actual_image_id, torch.Tensor):
                        actual_image_id = int(actual_image_id.item())
                    else:
                        actual_image_id = int(actual_image_id)
                except:
                    actual_image_id = dataset_idx

                # 查找 JPEG bytes
                jpeg_bytes = None
                for item in batch:
                    if isinstance(item, tuple):
                        for subitem in item:
                            if isinstance(subitem, bytes) and len(subitem) > 1000:
                                jpeg_bytes = subitem
                                break
                    elif isinstance(item, bytes) and len(item) > 1000:
                        jpeg_bytes = item
                        break

                if jpeg_bytes is None:
                    continue

                # 获取品种标签
                breed = id_to_breed.get(actual_image_id)
                if breed is None:
                    if batch_count <= 3:
                        print(f"  [Batch {batch_count}] 图像 ID {actual_image_id} (idx={dataset_idx}) 没有品种标签")
                    continue

                # 处理 JPEG bytes - 使用 byteformer collate 函数
                buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                sample_tensor = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
                sample_dict = {"samples": sample_tensor, "targets": torch.tensor(0)}

                collated = byteformer_image_collate_fn([sample_dict], opts)
                input_ids = collated["samples"].unsqueeze(0).to(device)  # [1, L]

                # 通过 backbone
                outputs = model.backbone(input_ids)
                feat = outputs.last_hidden_state  # [1, L, D]
                feat = feat.mean(dim=1).cpu().numpy()[0]  # [D]

                features.append(feat)
                labels.append(breed)
                image_ids.append(actual_image_id)
                success_count += 1

                if success_count <= 5:
                    print(f"  ✓ 成功提取特征 {success_count}: ID={actual_image_id}, breed={breed}, feat_shape={feat.shape}")

            except Exception as e:
                if batch_count <= 3 or success_count == 0:
                    print(f"  [Batch {batch_count}] 处理出错: {e}")
                continue

    print(f"\n成功提取 {len(features)} 个特征")
    return np.array(features), labels, image_ids


def train_and_evaluate_classifier(X, y):
    """训练并评估分类器"""
    # 转换为数值标签
    unique_breeds = sorted(set(y))
    breed_to_idx = {breed: i for i, breed in enumerate(unique_breeds)}
    y_numeric = np.array([breed_to_idx[breed] for breed in y])

    print(f"\n数据统计:")
    print(f"  样本数: {len(y)}")
    print(f"  品种数: {len(unique_breeds)}")
    print(f"  特征维度: {X.shape[1]}")

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练分类器
    print("\n训练 Logistic Regression 分类器...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, verbose=0)
    clf.fit(X_train_scaled, y_train)

    # 评估
    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)

    return train_acc, test_acc, unique_breeds


def main():
    import argparse
    parser = argparse.ArgumentParser(description='评估 backbone 分类能力')
    parser.add_argument('--folder', type=str, default='PureT/experiments/ByteCaption_XE_stanford')
    parser.add_argument('--config', type=str, default='config_coco.yml')
    parser.add_argument('--max_samples', type=int, default=200)
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.join(args.folder, args.config)
    cfg_from_file(config_path)

    # 创建一个能响应 getattr() 的 opts 对象
    class OptsWrapper:
        def __init__(self):
            self.data = {}

        def __getattr__(self, key):
            # 处理点分隔的路径
            if '.' in key:
                parts = key.split('.')
                value = self.data
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part, {})
                    else:
                        # 返回 False 作为默认值
                        return False
                return value if value not in [None, {}] else False
            return self.data.get(key, False)

        def set(self, key, value):
            self.data[key] = value

        # 添加常用的方法
        def strip(self):
            return ""

        def __bool__(self):
            return False

        def __eq__(self, other):
            return False

    opts = OptsWrapper()
    # 设置禁用的增强
    opts.set("image_augmentation.shuffle_bytes.enable", False)
    opts.set("image_augmentation.mask_positions.enable", False)
    opts.set("image_augmentation.byte_stream_corrupter.enable", False)
    opts.set("image_augmentation.random_uniform_noise.enable", False)
    opts.set("image_augmentation.byte_permutation.enable", False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载品种标签
    print("\n加载品种标签...")
    id_to_breed = load_ground_truth(
        cfg.INFERENCE.VAL_ANNFILE,
        cfg.DATA_LOADER.VAL_ID
    )
    print(f"  样本数: {len(id_to_breed)}")
    print(f"  品种数: {len(set(id_to_breed.values()))}")

    # 创建数据集
    print("\n创建数据集...")
    val_dataset = CocoDataset(
        image_ids_path=cfg.DATA_LOADER.VAL_ID,
        input_seq='',
        target_seq='',
        gv_feat_path='',
        seq_per_img=1,
        max_feat_num=-1,
        return_captions=False,
        model_type='bytecaption',
        is_training=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # 加载模型
    print("\n加载模型...")
    model = PureT_byteformer()
    model = model.to(device)

    # 加载检查点
    import glob
    checkpoint_files = glob.glob(os.path.join(args.folder, 'snapshot', '*.pth'))
    if checkpoint_files:
        checkpoint_path = os.path.join(args.folder, 'snapshot', 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = checkpoint_files[0]

        print(f"  加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 只加载 backbone
        backbone_state = {}
        for k, v in checkpoint.items():
            if 'backbone.byteformer' in k:
                new_key = k.replace('backbone.byteformer.', '')
                backbone_state[new_key] = v

        if backbone_state:
            model.backbone.byteformer.load_state_dict(backbone_state, strict=False)
            print(f"  已加载 {len(backbone_state)} 个 backbone 参数")
        else:
            print("  警告: 没有找到 backbone 权重")

    # 提取特征
    features, labels, image_ids = extract_backbone_features(
        model, val_loader, device, id_to_breed, args.max_samples, opts
    )

    if len(features) == 0:
        print("\n错误: 没有提取到任何特征！")
        return

    # 训练分类器
    train_acc, test_acc, breeds = train_and_evaluate_classifier(features, labels)

    # 打印结果
    print("\n" + "=" * 70)
    print("Backbone 分类能力评估结果")
    print("=" * 70)
    print(f"\n训练集准确率: {100*train_acc:.1f}%")
    print(f"测试集准确率: {100*test_acc:.1f}%")
    print(f"随机猜测基线: {100/len(breeds):.1f}%")

    print("\n结论:")
    if test_acc > 70:
        print("  ✓ Backbone 特征质量很好！")
        print("  → 问题在于 decoder，需要微调语言生成部分")
    elif test_acc > 40:
        print("  △ Backbone 特征质量一般")
        print("  → 建议：解冻 backbone，在 Stanford Dogs 上微调")
    else:
        print("  ✗ Backbone 特征质量较差")
        print("  → 建议：检查权重加载，或重新训练 backbone")

    print(f"\n品种列表 ({len(breeds)}):")
    for i, breed in enumerate(breeds[:10]):
        print(f"  {i+1}. {breed}")
    if len(breeds) > 10:
        print(f"  ... 还有 {len(breeds)-10} 个品种")


if __name__ == '__main__':
    main()
