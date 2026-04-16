#!/bin/bash
# Stage 2 训练脚本 - 详细描述版本
# 使用 Stage 1 的预训练权重，生成详细的图片描述

cd /home/Yu_zhen/pureT/ByteCaption

# 激活虚拟环境
source venv/bin/activate

# 设置GPU
export CUDA_VISIBLE_DEVICES=1

# 训练配置
EXPERIMENT_DIR="PureT/experiments/ByteCaption_Stage2_Detailed"
CONFIG_FILE="${EXPERIMENT_DIR}/config_coco.yml"
LOG_FILE="${EXPERIMENT_DIR}/training.log"

echo "=============================================="
echo "Stage 2 训练 - 详细描述版本"
echo "=============================================="
echo "配置文件: $CONFIG_FILE"
echo "日志文件: $LOG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查预训练权重
BACKBONE_PATH="/home/Yu_zhen/pureT/ByteCaption/results/stanford_dogs_byteformer_stage1/checkpoint_ema_best.pt"
if [ ! -f "$BACKBONE_PATH" ]; then
    echo "错误: 预训练权重不存在: $BACKBONE_PATH"
    exit 1
fi

# 检查数据集
TRAIN_DS="PureT/data/stanford_dogs_detailed/train"
VAL_DS="PureT/data/stanford_dogs_detailed/validation"
if [ ! -d "$TRAIN_DS" ] || [ ! -d "$VAL_DS" ]; then
    echo "错误: 数据集不存在"
    echo "  训练集: $TRAIN_DS"
    echo "  验证集: $VAL_DS"
    exit 1
fi

echo "数据集检查通过"
echo "开始训练..."

# 运行训练
PYTHONPATH=/home/Yu_zhen/pureT/ByteCaption:$PYTHONPATH \
python PureT/main.py \
  --folder $EXPERIMENT_DIR \
  2>&1 | tee $LOG_FILE

echo "=============================================="
echo "训练完成"
echo "=============================================="
