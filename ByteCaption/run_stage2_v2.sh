#!/bin/bash
# ByteCaption Stage 2 训练脚本 V2
# 改进: 增加验证集、早停、减少epochs

cd /home/Yu_zhen/pureT/ByteCaption
source venv/bin/activate
export PYTHONPATH=/home/Yu_zhen/pureT/ByteCaption:/home/Yu_zhen/pureT/ByteCaption/PureT:$PYTHONPATH

# 清理旧checkpoint
rm -rf PureT/experiments/ByteCaption_Stage2_FirstSent/snapshot/*

# 启动训练
# --val_samples 500: 增加验证集到500样本
# --eval_steps 500: 每500步评估一次（而不是50步）
# --early_stop_patience 10: 10次评估没改善才停止
# --load_weights: 加载Stage 1权重
CUDA_VISIBLE_DEVICES=2 python -u PureT/main.py \
  --folder PureT/experiments/ByteCaption_Stage2_FirstSent \
  --dataset coco \
  --load_weights \
  --val_samples 500 \
  --eval_steps 500 \
  --early_stop_patience 10 \
  > results_stage2_v2/training.log 2>&1

echo "训练完成"
