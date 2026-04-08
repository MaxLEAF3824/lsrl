#!/bin/bash

# ==========================================
# 批量大模型离线数学推理评测脚本
# ==========================================

# 1. 定义要评测的模型列表
MODELS=(
    # "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# 2. 定义要评测的数据集列表
DATASETS=(
    # "HuggingFaceH4/MATH-500"
    # "KbsdJames/Omni-MATH"
    # "Hothan/OlympiadBench"
    # "math-ai/amc23"
    "mnoukhov/dapo_math_14k_en_openinstruct"
)

# 3. 固定的采样和硬件参数
NUM_GPUS=4
N=8
TEMP=1.0
MAX_TOKENS=32768
TOP_P=0.95

# ==========================================
# 开始双层循环执行任务
# ==========================================
echo "🚀 启动自动化评测流水线 | 总计 ${#MODELS[@]} 个模型 x ${#DATASETS[@]} 个数据集..."

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "========================================================"
        echo "🔥 当前执行 -> 模型: $MODEL | 数据集: $DATASET"
        echo "========================================================"
        
        # 调用 Python 脚本并传入参数
        python /workspace/yiqiuguo/lsrl/vllm_gen_v2.py \
            --MODEL_ID "$MODEL" \
            --DATASET_ID "$DATASET" \
            --NUM_GPUS "$NUM_GPUS" \
            --SAMPLING_PARAMS_n "$N" \
            --SAMPLING_PARAMS_temperature "$TEMP" \
            --SAMPLING_PARAMS_max_tokens "$MAX_TOKENS" \
            --SAMPLING_PARAMS_top_p "$TOP_P"
            
        # 错误捕获：检查上一条命令的退出状态码
        if [ $? -eq 0 ]; then
            echo "✅ 成功完成: $MODEL 在 $DATASET 上的评测任务。"
        else
            echo "❌ 发生错误: $MODEL 在 $DATASET 上运行失败，将跳过并继续下一个任务..."
            # 如果你希望遇到报错就彻底停掉整个脚本，可以把下面这行取消注释：
            # exit 1
        fi
    done
done

echo ""
echo "🎉 所有评测任务已全部执行完毕！"