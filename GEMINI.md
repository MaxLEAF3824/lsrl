# Project Overview

This project focuses on **Latent Space Optimization and Distillation** for Large Language Models (LLMs), with a specific emphasis on mathematical reasoning tasks. It explores advanced techniques to optimize model outputs directly in the latent space (using custom optimization strategies like Frank-Wolfe) and distill these optimized behaviors back into the models. 

**Key Technologies:**
*   **Deep Learning Framework:** PyTorch (utilizing DistributedDataParallel / FSDP for multi-GPU training)
*   **LLM Inference & Generation:** vLLM, Hugging Face Transformers
*   **Optimization:** Custom Frank-Wolfe Optimizer (`frank_wolfe_optimizer.py`)
*   **Experiment Tracking:** Weights & Biases (wandb)
*   **Data Handling:** Hugging Face `datasets`

# Architecture & Main Components

*   `latent_optimizer_v7.py`: The core script implementing the unified pipeline for latent optimization and model distillation. It handles distributed setup, model loading, and iterative optimization loops.
*   `frank_wolfe_optimizer.py`: Defines the `FrankWolfeOptimizer` class, which manages optimization steps, adaptive gamma computation, and restricts optimization directions to top-k token embeddings.
*   `vllm_gen.py`: A specialized script for high-throughput text generation using vLLM. It features multi-processing, GPU isolation, and micro-batching for robustness during large-scale rollouts.
*   `eval_jsonl.py`: An evaluation utility designed to process JSONL generation results. It computes metrics like standard Pass@k and leverages tokenizer parallelization for speed.
*   `math_utils.py` & `math_wrong_dataset.py`: Domain-specific utilities for parsing mathematical expressions (e.g., extracting `\boxed{}` answers), verifying correctness, and structuring datasets.
*   `scripts/`: A crucial directory containing numerous shell scripts used to configure and launch various training and generation experiments.

# Building and Running

Experiments are primarily launched using the shell scripts found in the `scripts/` directory. These scripts typically set up the environment (like `CUDA_VISIBLE_DEVICES`) and invoke Python scripts via `torchrun` for distributed execution.

**Example Training/Optimization Command (Reference: `scripts/run_latopt_distill_v7_debug.sh`):**

```bash
torchrun --nproc_per_node=4 latent_optimizer_v7.py \
    --model_name <PATH_TO_MODEL> \
    --file_path <PATH_TO_DATA.jsonl> \
    --run_name <EXPERIMENT_NAME> \
    --optimizer "frank_wolfe" \
    --distill_epochs 3 \
    # ... additional parameters
```

**Example Evaluation:**
To evaluate the correctness of generated outputs stored in JSONL format, you can use the evaluation script:
```bash
python eval_jsonl.py # Note: May require editing the script or passing arguments for specific files.
```

# Development Conventions

*   **Bash Script Orchestration:** The project relies heavily on bash scripts to manage the complex arguments required for `torchrun` and to ensure consistent environment configuration across runs.
*   **WandB Logging:** Extensive use of Weights and Biases for tracking both the optimization history and the distillation process metrics.
*   **Resource Management:** Scripts are designed with multi-GPU architectures in mind, including explicit handling of vLLM GPU memory utilization and OMP thread limits to prevent crashes.
*   **Data Formats:** JSONL is the standard format for intermediate generations, optimization histories, and evaluation data.
