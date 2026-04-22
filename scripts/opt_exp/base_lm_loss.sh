#!/bin/bash

# ==========================================================
# 潜在空间重推理优化 - V5 高吞吐批处理架构 + 异步蒸馏框架
# ==========================================================

export CUDA_VISIBLE_DEVICES="0,1,2,3"

MODEL="Qwen/Qwen3-1.7B"
DATA_FILE="/workspace/yiqiuguo/lsrl/gen_results/qwen3-1.7b_math-500_rollout8_run20260405_095839_1499.jsonl"


echo "🚀 开始启动Latent Optimization & Distillation 训练..."
cd /workspace/yiqiuguo/lsrl/
torchrun --nproc_per_node=4 /workspace/yiqiuguo/lsrl/latent_optimizer_v7.py \
    --model_name "$MODEL" \
    --file_path "$DATA_FILE" \
    --run_name latopt-exp-$(basename "$0" .sh) \
    --vllm_gpus 0 1 2 3 \
    --big_batch_size 512 \
    --batch_size 1 \
    --chunk_size 512 \
    --optimizer "frank_wolfe" \
    --learning_rate 2e-3 \
    --fw_gamma 0.1 \
    --steps 50 \
    --kl_weight 0.5 \
    --eval_every 5 \
    --eval_k 32 \
    --mask_strategy "first_k" \
    --mask_max_k 32000 \
    --grad_direction "positive" \
    --conn_type "original" \
    --reg_type "lm" \
    --skip_distill \
    --early_stop \
    --early_stop_threshold 5e-4 \
    --distill_epochs 3 \
    --distill_lr 2e-5 \
    --distill_ce_loss_weight 0.0 \
    --distill_eval_every 500 \
    --distill_batch_size 1 \
    --distill_grad_accum_steps 4 \
    --distill_eval_datasets HuggingFaceH4/MATH-500
# openai/gsm8k 