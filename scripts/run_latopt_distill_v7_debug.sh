#!/bin/bash

# ==========================================================
# 潜在空间重推理优化 - V5 高吞吐批处理架构 + 异步蒸馏框架
# ==========================================================

export CUDA_VISIBLE_DEVICES="0,1,2,3"

MODEL="/workspace/yiqiuguo/verl/checkpoints/verl_grpo_math_gb200/qwen3_1.7b_math_gb200/global_step_150/actor_hf_step_150"
DATA_FILE="/workspace/yiqiuguo/lsrl/gen_results/actor_hf_step_150_dapo_math_14k_en_openinstruct_rollout8_run20260415_19f2.jsonl"


echo "🚀 开始启动Latent Optimization & Distillation 训练..."
cd /workspace/yiqiuguo/lsrl/
torchrun --nproc_per_node=4 /workspace/yiqiuguo/lsrl/latent_optimizer_v7.py \
    --model_name "$MODEL" \
    --file_path "$DATA_FILE" \
    --run_name latopt-exp-dapo-$(basename "$0" .sh) \
    --vllm_gpus 0 1 2 3 \
    --big_batch_size 512 \
    --batch_size 2 \
    --chunk_size 1024 \
    --optimizer "frank_wolfe" \
    --learning_rate 2e-3 \
    --adaptive_grad top_k \
    --adaptive_grad_top_k 256 \
    --fw_gamma 0.1 \
    --steps 20 \
    --kl_weight 0.1 \
    --eval_modes pure \
    --eval_every 999 \
    --eval_k 8 \
    --mask_strategy "first_k" \
    --mask_max_k 0.8 \
    --grad_direction "positive" \
    --conn_type "original" \
    --reg_type "lm" \
    --max_samples 200 \
    --skip_start_eval \
    --early_stop \
    --early_stop_threshold 5e-4 \
    --early_stop_correctness_threshold 0.8 \
    --distill_epochs 3 \
    --distill_lr 2e-5 \
    --distill_ce_loss_weight 0.0 \
    --distill_eval_every 500 \
    --distill_batch_size 1 \
    --distill_grad_accum_steps 4 \
    --distill_eval_datasets HuggingFaceH4/MATH-500
# openai/gsm8k 