#!/bin/bash

# ==========================================================
# 潜在空间重推理优化 - V5 高吞吐批处理架构 + 异步蒸馏框架
# ==========================================================

export CUDA_VISIBLE_DEVICES="0,1,2,3"
WANDB_MODE=disabled
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
DATA_FILE="/workspace/yiqiuguo/lsrl/gen_results/qwen2.5-1.5b-instruct_dapo_math_14k_en_openinstruct_rollout8_run20260429_3871.jsonl"

echo "🚀 开始启动Latent Optimization & Distillation 训练..."
cd /workspace/yiqiuguo/lsrl/
torchrun --nproc_per_node=4 /workspace/yiqiuguo/lsrl/latent_optimizer_v7.py \
    --model_name "$MODEL" \
    --file_path "$DATA_FILE" \
    --run_name latopt-distill-exp-dapo-qwen2-lm_loss_contrastive_conn_hard$(basename "$0" .sh) \
    --vllm_gpus 0 1 2 3 \
    --big_batch_size 512 \
    --batch_size 1 \
    --chunk_size 512 \
    --optimizer "frank_wolfe" \
    --learning_rate 2e-3 \
    --fw_gamma 0.1 \
    --steps 20 \
    --kl_weight 1.0 \
    --eval_every 999 \
    --eval_k 8 \
    --eval_modes pure fast \
    --mask_strategy "first_k" \
    --mask_max_k 32000 \
    --grad_direction "contrastive" \
    --conn_type "original" \
    --reg_type "lm" \
    --max_samples 4000 \
    --early_stop \
    --early_stop_threshold 1e-1 \
    --skip_distill_eval \
    --skip_start_eval \
    --distill_type conn_hard \
    --distill_epochs 2 \
    --distill_lr 1e-6 \
    --distill_eval_every 99999 \
    --distill_batch_size 1 \
    --distill_grad_accum_steps 1 \
    --distill_eval_datasets HuggingFaceH4/MATH-500