#!/usr/bin/env python
"""
生成 Stage 2 预测验证日志
对验证集进行推理并记录详细结果
"""

import os
import sys
import torch
import json
import argparse
from PIL import Image
from tqdm import tqdm

# 设置路径
sys.path.insert(0, '/home/Yu_zhen/pureT/ByteCaption')
sys.path.insert(0, '/home/Yu_zhen/pureT/ByteCaption/PureT')

from lib.config import cfg, cfg_from_file
import lib.utils as utils
import models
from datasets_.coco_dataset import CocoDataset

def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    cfg_from_file('PureT/experiments/ByteCaption_Stage2_v3/config_coco.yml')

    model = models.create(cfg.MODEL.TYPE)

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 移除 'module.' 前缀 (DataParallel 保存的模型)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model

def extract_breed(text):
    """从描述中提取品种名称"""
    text = text.lower()
    # 常见模式: "the dog is a xxx"
    if "the dog is a" in text:
        parts = text.split("the dog is a")
        if len(parts) > 1:
            breed_part = parts[1].strip().split(".")[0].strip()
            return breed_part
    return None

def breed_match(pred_breed, ref_breed):
    """检查品种是否匹配"""
    if pred_breed is None or ref_breed is None:
        return False
    pred_breed = pred_breed.lower().strip()
    ref_breed = ref_breed.lower().strip()
    # 简单匹配
    return pred_breed in ref_breed or ref_breed in pred_breed

def main():
    # 使用 CPU 或共享 GPU（避免独占）
    device = torch.device('cpu')

    # 加载配置
    cfg_from_file('PureT/experiments/ByteCaption_Stage2_v3/config_coco.yml')

    # 加载词汇表
    vocab_path = cfg.INFERENCE.VOCAB
    vocab = utils.load_vocab(vocab_path)

    # 加载验证集
    val_dataset = CocoDataset(
        image_ids_path='./PureT/data/stanford_dogs_detailed/validation_ids.json',
        input_seq=None,
        target_seq=None,
        gv_feat_path='',
        seq_per_img=1,
        max_feat_num=-1,
        return_captions=True
    )

    # 加载模型
    checkpoint_path = 'PureT/experiments/ByteCaption_Stage2_v3/snapshot/caption_model_20.pth'
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # 加载验证标注
    with open('./PureT/data/stanford_dogs_detailed/validation_annotations.json', 'r') as f:
        annotations = json.load(f)

    # 评估样本数
    num_samples = min(100, len(val_dataset))

    results = []
    correct_breed_count = 0

    print(f"Evaluating {num_samples} samples...")

    for idx in tqdm(range(num_samples)):
        try:
            indices, captions, gv_feat, att_feats = val_dataset[idx]

            # 准备输入
            att_feats = att_feats.unsqueeze(0).to(device) if att_feats.dim() == 3 else att_feats.to(device)
            att_mask = torch.ones(att_feats.size(0), att_feats.size(1)).to(device)

            # 推理
            with torch.no_grad():
                kwargs = {
                    cfg.PARAM.ATT_FEATS: att_feats,
                    cfg.PARAM.ATT_FEATS_MASK: att_mask,
                    'BEAM_SIZE': 1,
                    'GREEDY_DECODE': True
                }
                seq, _ = model.module.decode(**kwargs) if hasattr(model, 'module') else model.decode(**kwargs)

            # 解码
            pred_sent = utils.decode_sequence(vocab, seq)[0]

            # 获取参考描述
            img_id = str(idx)
            ref_caption = captions[0] if captions else ""

            # 提取品种
            pred_breed = extract_breed(pred_sent)
            ref_breed = extract_breed(ref_caption)

            # 检查品种匹配
            is_correct = breed_match(pred_breed, ref_breed)
            if is_correct:
                correct_breed_count += 1

            results.append({
                'idx': idx,
                'pred_sent': pred_sent,
                'ref_sent': ref_caption,
                'pred_breed': pred_breed,
                'ref_breed': ref_breed,
                'correct': is_correct
            })

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    # 写入日志
    log_path = 'results_stage2_v3/prediction_verification_full.log'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Stage 2 预测结果验证日志 (ByteFormer)\n")
        f.write(f"评估样本数: {num_samples}\n")
        f.write(f"品种准确率: {correct_breed_count}/{num_samples} = {100*correct_breed_count/num_samples:.1f}%\n")
        f.write("=" * 80 + "\n\n")

        # 正确的样本
        correct_samples = [r for r in results if r['correct']]
        f.write("=" * 80 + "\n")
        f.write(f"预测正确的样本: {len(correct_samples)} 个\n")
        f.write("=" * 80 + "\n\n")

        for i, r in enumerate(correct_samples[:20]):  # 最多显示20个
            f.write(f"样本 {r['idx']}:\n")
            f.write(f"  预测品种: {r['pred_breed']}\n")
            f.write(f"  真实品种: {r['ref_breed']}\n")
            f.write(f"  预测描述: {r['pred_sent'][:200]}...\n" if len(r['pred_sent']) > 200 else f"  预测描述: {r['pred_sent']}\n")
            f.write(f"  结果: ✓ 正确\n\n")

        # 所有样本详情
        f.write("\n" + "=" * 80 + "\n")
        f.write("所有样本详情:\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"样本 {r['idx']}:\n")
            f.write(f"  预测描述: {r['pred_sent']}\n")
            f.write(f"  真实描述: {r['ref_sent']}\n")
            f.write(f"  品种匹配: {'✓ 正确' if r['correct'] else '✗ 错误'}\n")
            f.write("-" * 40 + "\n\n")

    print(f"结果已保存到 {log_path}")
    print(f"品种准确率: {correct_breed_count}/{num_samples} = {100*correct_breed_count/num_samples:.1f}%")

if __name__ == '__main__':
    main()
