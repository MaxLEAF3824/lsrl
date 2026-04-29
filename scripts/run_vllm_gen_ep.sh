#!/bin/bash

# ==========================================
# 0. 接收并解析命令行参数
# ==========================================
# 设置默认值，防止没传参数时报错
RUN_SEED=26418
NODE_TOTAL=1
NODE_RANK=0
export WANDB_API_KEY=469ecac017511ec7e2e95fc2f1bab23668dfc776

# 解析外部传入的参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --RUN_SEED) RUN_SEED="$2"; shift ;;
        --NODE_TOTAL|--NT|-NT) NODE_TOTAL="$2"; shift ;;
        --NODE_RANK|--NR|-NR) NODE_RANK="$2"; shift ;;
        *) echo "❌ 报错: 遇到未知参数 $1"; exit 1 ;;
    esac
    shift
done


# ==========================================
# 批量大模型离线数学推理评测脚本
# ==========================================

# 1. 定义要评测的模型列表
MODELS=(
    # "Qwen/Qwen3-0.6B"
    # "Qwen/Qwen3-1.7B"
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # "/workspace/yiqiuguo/lsrl/checkpoints/radiant-bee-213/step474"
    # "/workspace/yiqiuguo/lsrl/checkpoints/gallant-firebrand-212/step688"
    # "/workspace/yiqiuguo/verl/checkpoints/verl_grpo_math_gb200/qwen3_1.7b_math_gb200/global_step_150/actor_hf_step_150"
    # "/workspace/yiqiuguo/lsrl/checkpoints/light-planet-301/step477"
    # "Qwen/Qwen3-1.7B-Base"
    # "/workspace/yiqiuguo/verl/checkpoints/verl_grpo_math_gb200/qwen3_1.7b_math_gb200/global_step_550/actor_hf_step_550"
    # "/workspace/yiqiuguo/lsrl/checkpoints/latopt-distill-exp-dapo-step0-base/step153"
    "/workspace/yiqiuguo/lsrl/checkpoints/latopt-distill-exp-dapo-qwen3-1.7b-lm_loss_contrastivedebug/step12"
)

# 2. 定义要评测的数据集列表
DATASETS=(
    # "mnoukhov/dapo_math_14k_en_openinstruct"
    "HuggingFaceH4/MATH-500"
    # "KbsdJames/Omni-MATH"
    "math-ai/aime25"
    "HuggingFaceH4/aime_2024"
    "math-ai/amc23"
)

# 3. 固定的采样和硬件参数
NUM_GPUS=4
N=8
TEMP=1.0
MAX_TOKENS=8192
TOP_P=0.95

cd /workspace/yiqiuguo/lsrl
# ==========================================
# 开始双层循环执行任务
# ==========================================
echo "🚀 启动自动化评测流水线 | 总计 ${#MODELS[@]} 个模型 x ${#DATASETS[@]} 个数据集..."
echo "🌍 集群配置 -> 总节点数: $NODE_TOTAL | 当前节点: $NODE_RANK | 随机种子: ${RUN_SEED:-未设置(使用随机UUID)}"

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "========================================================"
        echo "🔥 当前执行 -> 模型: $MODEL | 数据集: $DATASET"
        echo "========================================================"
        
        # 动态拼接额外的可选参数（如果传入了 RUN_SEED 才加进去）
        EXTRA_ARGS=""
        if [ -n "$RUN_SEED" ]; then
            EXTRA_ARGS="--RUN_SEED $RUN_SEED"
        fi
        
        # 调用 Python 脚本并传入所有参数
        python /workspace/yiqiuguo/lsrl/vllm_gen.py \
            --MODEL_ID "$MODEL" \
            --DATASET_ID "$DATASET" \
            --NUM_GPUS "$NUM_GPUS" \
            --SAMPLING_PARAMS_n "$N" \
            --SAMPLING_PARAMS_temperature "$TEMP" \
            --SAMPLING_PARAMS_max_tokens "$MAX_TOKENS" \
            --SAMPLING_PARAMS_top_p "$TOP_P" \
            --NODE_TOTAL "$NODE_TOTAL" \
            --NODE_RANK "$NODE_RANK" \
            $EXTRA_ARGS
            
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

echo "🎉 所有评测任务已全部执行完毕！"