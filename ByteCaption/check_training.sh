#!/bin/bash
# 检查训练状态脚本

LOG_FILE="/home/Yu_zhen/pureT/ByteCaption/results_stage2_v3/training.log"
CHECKPOINT_FILE="/home/Yu_zhen/pureT/ByteCaption/.training_check"

# 检查是否有错误
if grep -q "RuntimeError\|CUDA error\|Traceback\|Assertion.*failed" "$LOG_FILE" 2>/dev/null; then
    echo "[$(date)] ERROR detected in training log!"
    
    # 检查是否已经处理过这个错误
    LAST_ERROR=$(grep -n "RuntimeError\|CUDA error\|Traceback" "$LOG_FILE" | tail -1 | cut -d: -f1)
    if [ -f "$CHECKPOINT_FILE" ]; then
        LAST_CHECKED=$(cat "$CHECKPOINT_FILE")
        if [ "$LAST_ERROR" = "$LAST_CHECKED" ]; then
            echo "[$(date)] Error already handled, skipping..."
            exit 0
        fi
    fi
    
    echo "$LAST_ERROR" > "$CHECKPOINT_FILE"
    
    # 杀掉当前训练进程
    pkill -f "PureT/main.py.*ByteCaption_Stage2_v3" 2>/dev/null
    sleep 3
    
    # 清空日志并重新启动
    echo "" > "$LOG_FILE"
    cd /home/Yu_zhen/pureT/ByteCaption
    source venv/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    PYTHONPATH=/home/Yu_zhen/pureT/ByteCaption:$PYTHONPATH python PureT/main.py --folder PureT/experiments/ByteCaption_Stage2_v3 2>&1 | tee "$LOG_FILE" &
    
    echo "[$(date)] Training restarted!"
else
    echo "[$(date)] Training running normally."
    # 获取最新进度
    LAST_STEP=$(grep "Step [0-9]*:" "$LOG_FILE" | tail -1)
    echo "[$(date)] Latest: $LAST_STEP"
fi
