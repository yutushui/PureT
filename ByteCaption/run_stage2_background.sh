#!/bin/bash
cd /home/Yu_zhen/pureT/ByteCaption
source venv/bin/activate
export PYTHONPATH=/home/Yu_zhen/pureT/ByteCaption:/home/Yu_zhen/birds/corenet-main:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

# 训练日志
nohup python PureT/main.py \
    --folder PureT/experiments/ByteCaption_Stage2_FirstSent \
    --dataset coco \
    --load_weights \
    --eval_steps 500 \
    --val_samples 20 \
    --disable_wandb \
    > results_stage2/training.log 2>&1 &

echo $! > results_stage2/training.pid
echo "训练已在后台启动"
echo "日志文件: results_stage2/training.log"
echo "进程ID保存在: results_stage2/training.pid"

# 实时监控训练进度的命令
echo ""
echo "=== 实时监控命令 ==="
echo "查看训练日志: tail -f results_stage2/training.log"
echo "查看GPU使用: nvidia-smi"
echo "查看进程: ps aux | grep main.py"
echo "停止训练: kill \$(cat results_stage2/training.pid)"
