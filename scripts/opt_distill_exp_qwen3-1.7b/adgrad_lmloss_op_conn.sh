#!/bin/bash

# ==========================================================
# 潜在空间重推理优化 - V5 高吞吐批处理架构 + 异步蒸馏框架
# ==========================================================

export CUDA_VISIBLE_DEVICES="0,1,2,3"

MODEL="Qwen/Qwen3-1.7B"
DATA_FILE="/workspace/yiqiuguo/lsrl/gen_results/qwen3-1.7b_dapo_math_14k_en_openinstruct_rollout8_run20260408_221824_ff22.jsonl"


echo "🚀 开始启动Latent Optimization & Distillation 训练..."
cd /workspace/yiqiuguo/lsrl/
torchrun --nproc_per_node=4 /workspace/yiqiuguo/lsrl/latent_optimizer_v7.py \
    --model_name "$MODEL" \
    --file_path "$DATA_FILE" \
    --run_name latopt-distill-exp-dapo-qwen3-1.7b-$(basename "$0" .sh) \
    --vllm_gpus 0 1 2 3 \
    --big_batch_size 1024 \
    --batch_size 1 \
    --chunk_size 1024 \
    --optimizer "frank_wolfe" \
    --learning_rate 2e-3 \
    --fw_gamma 0.1 \
    --steps 50 \
    --kl_weight 0.2 \
    --adaptive_grad top_k \
    --eval_every 5 \
    --eval_k 8 \
    --eval_modes pure forced \
    --mask_strategy "first_k" \
    --mask_max_k 32000 \
    --grad_direction "positive" \
    --conn_type "on-policy" \
    --reg_type "lm" \
    --max_samples 4000 \
    --early_stop \
    --early_stop_threshold 5e-4 \
    --distill_epochs 3 \
    --distill_lr 2e-5 \
    --distill_ce_loss_weight 0.0 \
    --distill_eval_every 99999 \
    --distill_batch_size 1 \
    --distill_grad_accum_steps 4 \
    --distill_eval_datasets HuggingFaceH4/MATH-500
# openai/gsm8k 