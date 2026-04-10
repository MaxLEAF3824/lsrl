#!/bin/bash

# ==========================================================
# 潜在空间重推理优化 - V2 高吞吐批处理架构 + 异步蒸馏框架
# ==========================================================

export CUDA_VISIBLE_DEVICES="0,1,2,3"

MODEL="Qwen/Qwen3-1.7B"
DATA_FILE="/workspace/yiqiuguo/lsrl/gen_results/qwen3-1.7b_dapo_math_14k_en_openinstruct_rollout8_run20260409_n098_node0256.jsonl"

BATCH_SIZE=1
CHUNK_SIZE=1024 

echo "🚀 开始启动Latent Optimization & Distillation 训练..."
cd /workspace/yiqiuguo/lsrl/
python /workspace/yiqiuguo/lsrl/latent_optimizer_v5.py \
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
    --eval_every 10 \
    --eval_k 32 \
    --mask_strategy "first_k" \
    --mask_max_k 32000 \
    --grad_direction "positive" \
    --conn_type "original" \
    --reg_type "lm" \
    --early_stop \
    --early_stop_threshold 5e-4 \
    --distill_epochs 3 \
    --distill_lr 2e-5 \
    --distill_ce_loss_weight 0.0 \
    --distill_eval_every 500 \
    --distill_batch_size 1 \
    --distill_grad_accum_steps 4 \
    --distill_sample_filter \
    --distill_eval_datasets HuggingFaceH4/MATH-500 math-ai/aime25 math-ai/amc23 HuggingFaceH4/aime_2024
# openai/gsm8k 