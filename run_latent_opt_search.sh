#!/bin/bash

# ==========================================================
# Latent Space Optimization - 异步多卡高并发调度脚本
# ==========================================================
# 机器总共有 4 张卡 (0,1,2,3)
# 策略: 拆分为两个槽位 (Slot 1: GPU 0,1 | Slot 2: GPU 2,3)
# 每个槽位中，cuda:0 给主模型，cuda:1 给 vLLM (对应参数 --vllm_gpus 1)

# 1. 定义模型数组
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
)

# 2. 定义辅助函数：根据模型自动匹配它的 4 个专属数据集
get_data_files() {
    local model=$1
    local base_dir="/workspace/yiqiuguo/lsrl/gen_results"
    
    if [[ "$model" == *"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"* ]]; then
        echo "$base_dir/deepseek-r1-distill-qwen-1.5b_math-500_rollout8_run20260405_101242_1854.jsonl"
        # echo "$base_dir/deepseek-r1-distill-qwen-1.5b_omni-math_rollout8_run20260405_101742_2d08.jsonl"
    elif [[ "$model" == *"Qwen/Qwen2.5-1.5B-Instruct"* ]]; then
        echo "$base_dir/qwen2.5-1.5b-instruct_math-500_rollout8_run20260405_095904_59a5.jsonl"
        # echo "$base_dir/qwen2.5-1.5b-instruct_omni-math_rollout8_run20260405_100133_7614.jsonl"
    elif [[ "$model" == *"Qwen/Qwen3-0.6B"* ]]; then
        echo "$base_dir/qwen3-0.6b_math-500_rollout8_run20260405_100209_3eb8.jsonl"
        # echo "$base_dir/qwen3-0.6b_omni-math_rollout8_run20260405_101531_f081.jsonl"
    elif [[ "$model" == *"Qwen/Qwen3-1.7B"* ]]; then
        echo "$base_dir/qwen3-1.7b_math-500_rollout8_run20260405_095839_1499.jsonl"
        # echo "$base_dir/qwen3-1.7b_omni-math_rollout8_run20260405_101117_1952.jsonl"
    fi
}

# 3. 将所有的实验组合排入任务队列
TASKS=()

for MODEL in "${MODELS[@]}"; do
    FILES=($(get_data_files "$MODEL"))
    for DATA_FILE in "${FILES[@]}"; do
        for KL_WEIGHT in 0.0 1.0; do
            for OPTIMIZER in "adam" "frank_wolfe"; do
                
                # 拼接完整的 Python 运行命令 (注意内部的引号转义)
                CMD="python /workspace/yiqiuguo/lsrl/latent_optimizer_v2.py \
                    --model_name \"$MODEL\" \
                    --file_path \"$DATA_FILE\" \
                    --vllm_gpus 1 \
                    --batch_size 1 \
                    --chunk_size 1024 \
                    --optimizer \"$OPTIMIZER\" \
                    --learning_rate 2e-3 \
                    --fw_gamma 0.1 \
                    --steps 50 \
                    --kl_weight $KL_WEIGHT \
                    --eval_every 5 \
                    --eval_k 32 \
                    --mask_strategy \"first_k\" \
                    --mask_max_k 32000 \
                    --grad_direction \"positive\" \
                    --conn_type \"original\" \
                    --reg_type \"lm\" \
                    --early_stop \
                    --early_stop_threshold 5e-4"
                
                TASKS+=("$CMD")
            done
        done
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo "📦 任务队列构建完成，总计共有 $TOTAL_TASKS 个运行组合。"
echo "🚀 开始异步双卡槽调度..."
echo "--------------------------------------------------------"

# 4. 智能异步调度器 (Dual-Slot Dispatcher)
PID_SLOT1=""
PID_SLOT2=""
CURRENT_TASK=0

for TASK_CMD in "${TASKS[@]}"; do
    ((CURRENT_TASK++))
    
    # 无限循环检测是否有空闲槽位
    while true; do
        SLOT1_FREE=true
        SLOT2_FREE=true
        
        # 检查 Slot 1 的进程是否还在存活
        if [ -n "$PID_SLOT1" ] && kill -0 "$PID_SLOT1" 2>/dev/null; then
            SLOT1_FREE=false
        fi
        
        # 检查 Slot 2 的进程是否还在存活
        if [ -n "$PID_SLOT2" ] && kill -0 "$PID_SLOT2" 2>/dev/null; then
            SLOT2_FREE=false
        fi
        
        # 如果至少有一个槽位空闲，跳出等待，去发射新任务
        if [ "$SLOT1_FREE" = true ] || [ "$SLOT2_FREE" = true ]; then
            break
        fi
        
        # 两个都在忙，休息 10 秒再查
        sleep 10
    done

    # 优先将任务发射到 Slot 1
    if [ "$SLOT1_FREE" = true ]; then
        echo "🟢 [Task $CURRENT_TASK/$TOTAL_TASKS] -> 分配至 Slot 1 (GPU: 0, 1)"
        # 局部临时环境变量，不污染全局，保证子进程独立锁定 GPU 0和1
        CUDA_VISIBLE_DEVICES="0,1" eval "$TASK_CMD" &
        PID_SLOT1=$!
        
    # Slot 1 忙碌，则发射到 Slot 2
    elif [ "$SLOT2_FREE" = true ]; then
        echo "🔵 [Task $CURRENT_TASK/$TOTAL_TASKS] -> 分配至 Slot 2 (GPU: 2, 3)"
        # 局部临时环境变量，不污染全局，保证子进程独立锁定 GPU 2和3
        CUDA_VISIBLE_DEVICES="2,3" eval "$TASK_CMD" &
        PID_SLOT2=$!
    fi
    
    # 稍微停顿 3 秒，错开两个任务的极高 IO/显存分配峰值，防止框架抢占锁
    sleep 3
done

echo "⏳ 所有任务均已发射！等待最后几个后台任务执行完毕..."
wait
echo "🎉 恭喜！全部 $TOTAL_TASKS 个实验执行完毕！"