#!/bin/bash
# ByteCaption Stage 2 训练脚本 V3 - 正确加载Stage 1预训练权重
# 关键修复: 添加 --load_weights 参数

cd /home/Yu_zhen/pureT/ByteCaption
source venv/bin/activate
export PYTHONPATH=/home/Yu_zhen/pureT/ByteCaption:/home/Yu_zhen/pureT/ByteCaption/PureT:$PYTHONPATH

# 清理旧checkpoint (从头开始训练)
rm -rf PureT/experiments/ByteCaption_Stage2_v3/snapshot/*

# 启动训练
# --load_weights: 加载Stage 1权重 (关键!)
# --freeze_backbone: 冻结backbone参数
CUDA_VISIBLE_DEVICES=1 python -u PureT/main.py \
  --folder PureT/experiments/ByteCaption_Stage2_v3 \
  --dataset coco \
  --load_weights \
  --freeze_backbone \
  --val_samples 50 \
  --eval_steps 300 \
  > results_stage2_v3/training.log 2>&1

echo "训练完成"
