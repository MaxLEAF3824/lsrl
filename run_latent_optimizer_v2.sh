#!/bin/bash

# ==========================================================
# 潜在空间重推理优化 - V2 高吞吐批处理架构
# ==========================================================

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# DATA_FILE="/workspace/yiqiuguo/lsrl/gen_results/deepseek-r1-distill-qwen-1.5b_omni-math_rollout8_run20260405_101742_2d08.jsonl"
# DATA_FILE="/workspace/yiqiuguo/lsrl/qwen3-1.7b_math-500_rollout8_len32768_final.jsonl"
MODEL="Qwen/Qwen3-1.7B"
DATA_FILE="/workspace/yiqiuguo/lsrl/gen_results/qwen3-1.7b_math-500_rollout8_run20260405_095839_1499.jsonl"
# DATA_FILE="/workspace/yiqiuguo/lsrl/gen_results/qwen3-1.7b_omni-math_rollout8_run20260405_101117_1952.jsonl"

# 注意：现在的 batch_size 指的是计算梯度时的并行样本数，而不是数据集总数。
# 整个活跃数据集会在每个 Global Step (Epoch) 被跑通一遍。
BATCH_SIZE=1
CHUNK_SIZE=1024 

echo "🚀 开始启动Latent Optimization训练..."

python /workspace/yiqiuguo/lsrl/latent_optimizer_v2.py \
    --model_name "$MODEL" \
    --file_path "$DATA_FILE" \
    --vllm_gpus 1 2 3 \
    --batch_size $BATCH_SIZE \
    --chunk_size $CHUNK_SIZE \
    --optimizer "adam" \
    --learning_rate 2e-3 \
    --fw_gamma 0.1 \
    --steps 50 \
    --kl_weight 0.0 \
    --eval_every 5 \
    --eval_k 32 \
    --mask_strategy "first_k" \
    --mask_max_k 32000 \
    --grad_direction "positive" \
    --conn_type "original" \
    --reg_type "lm" \
    --early_stop \
    --early_stop_threshold 5e-4
