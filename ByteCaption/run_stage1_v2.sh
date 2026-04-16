#!/bin/bash
# Stage 1 训练脚本 V2 - 150 epochs，更强数据增强

cd /home/Yu_zhen/pureT/ByteCaption

# 创建新的结果目录
mkdir -p results_stage1_v2

# 激活虚拟环境
source venv/bin/activate

# 使用GPU 1
export CUDA_VISIBLE_DEVICES=1

# 启动训练
# 配置文件: stanford_dogs_stage1_v2.yaml
# 改进: 150 epochs, batch=32, lr=0.0003, 颜色抖动, 旋转
PYTHONPATH=/home/Yu_zhen/pureT/ByteCaption:$PYTHONPATH \
python corenet/cli/main_train.py \
  --common.config-file stanford_dogs_stage1_v2.yaml \
  > results_stage1_v2/training.log 2>&1

echo "Stage 1训练完成"
