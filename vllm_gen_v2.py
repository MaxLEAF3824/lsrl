import json
import os
import time
import uuid
import argparse
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from datasets import load_dataset

prompt_template = """You are an expert mathematician. Please solve the following math problem.

You must strictly adhere to the following output format:
1. First, write out your detailed, step-by-step reasoning process. You MUST enclose your entire reasoning process within `<think>` and `</think>` tags.
2. Provide your final mathematical answer enclosed within `\\boxed{{}}`. 

Problem:
{question}

"""

# ---------------------------------------------------------
# 单 GPU 推理逻辑
# ---------------------------------------------------------
def run_inference(rank, data_chunk, run_id, model_id, sampling_kwargs):
    """
    rank: 显卡编号
    data_chunk: 该卡负责的数据子集
    run_id: 本次运行的唯一标识符
    model_id: 模型路径/ID
    sampling_kwargs: 采样参数字典
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.95
    )

    # 动态初始化采样参数
    sampling_params = SamplingParams(**sampling_kwargs)
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    
    for item in data_chunk:
        # 兼容不同数据集的题目字段名 (MATH 是 problem, OlympiadBench 等可能是 question)
        # question_text = item.get('problem', item.get('question', item.get('problem_v1', None)))
        question_text = item.get('q', None)
        
        # prompt_text = prompt_template.format(question=question_text)
        messages = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text += "<think>"
        formatted_prompts.append(prompt_text)
    
    outputs = llm.generate(formatted_prompts, sampling_params)

    temp_file = f"temp_{run_id}_gpu_{rank}.jsonl"
    with open(temp_file, "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            res_list = [res.text.strip() for res in output.outputs]
            entry = data_chunk[i].copy() 
            entry["responses"] = res_list
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✨ GPU {rank} 处理完毕，样本数: {len(data_chunk)}")

# ---------------------------------------------------------
# 主程序控制
# ---------------------------------------------------------
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="vLLM Offline Inference for Math Datasets")
    
    parser.add_argument("--MODEL_ID", type=str, default="Qwen/Qwen3-1.7B", help="Path or ID of the model")
    parser.add_argument("--DATASET_ID", type=str, default="HuggingFaceH4/MATH-500", help="Dataset ID on HuggingFace")
    parser.add_argument("--NUM_GPUS", type=int, default=4, help="Number of GPUs to use for DP inference")
    
    # 采样参数
    parser.add_argument("--SAMPLING_PARAMS_n", type=int, default=8, help="Number of outputs to generate per prompt (rollout)")
    parser.add_argument("--SAMPLING_PARAMS_temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--SAMPLING_PARAMS_max_tokens", type=int, default=32768, help="Maximum number of generated tokens")
    parser.add_argument("--SAMPLING_PARAMS_top_p", type=float, default=0.95, help="Top-p sampling parameter")
    
    args = parser.parse_args()

    # 组装采样参数字典，方便传递给子进程
    sampling_kwargs = {
        "n": args.SAMPLING_PARAMS_n,
        "temperature": args.SAMPLING_PARAMS_temperature,
        "max_tokens": args.SAMPLING_PARAMS_max_tokens,
        "top_p": args.SAMPLING_PARAMS_top_p
    }

    mp.set_start_method('spawn', force=True)

    # 生成唯一运行 ID (格式：时间戳+随机短ID)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]

    print(f"正在从 HuggingFace 加载数据集: {args.DATASET_ID}...")

    # 根据不同的 DATASET_ID 处理特定的 split 逻辑
    if args.DATASET_ID == "HuggingFaceH4/MATH-500":
        dataset = load_dataset(args.DATASET_ID, split='test')
        q_key = "problem"
    elif args.DATASET_ID == "KbsdJames/Omni-MATH":
        dataset = load_dataset(args.DATASET_ID, split='test')
        q_key = "problem"
    elif args.DATASET_ID == "Hothan/OlympiadBench":
        dataset = load_dataset(args.DATASET_ID, name="OE_TO_maths_en_COMP", split='train')
        q_key = "problem_v1"
    elif args.DATASET_ID == "math-ai/amc23":
        dataset = load_dataset(args.DATASET_ID, split='test')
        q_key = "question"
    elif args.DATASET_ID == "mnoukhov/dapo_math_14k_en_openinstruct":
        dataset = load_dataset(args.DATASET_ID, split='train')
        q_key = "prompt"
    elif args.DATASET_ID == "openai/gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split='test')
        q_key = "question"
    elif args.DATASET_ID == "HuggingFaceH4/aime_2024":
        dataset = load_dataset(args.DATASET_ID, split='train')
        q_key = "problem"
    elif args.DATASET_ID == "math-ai/aime25":
        dataset = load_dataset(args.DATASET_ID, split='test')
        q_key = "problem"
    else:
        dataset = load_dataset(args.DATASET_ID, split='test')
        q_key = "problem"
    
    all_data = [item|{"q": q_key} for item in dataset]
    total_len = len(all_data)
    
    chunk_size = (total_len + args.NUM_GPUS - 1) // args.NUM_GPUS
    chunks = [all_data[i:i + chunk_size] for i in range(0, total_len, chunk_size)]

    print(f"🚀 开始 DP 并行推理 | Run ID: {run_id} | 总题数: {total_len} | 模型: {args.MODEL_ID}")

    processes = []
    for i in range(len(chunks)):
        # 将解析后的参数传递给子进程
        p = mp.Process(target=run_inference, args=(i, chunks[i], run_id, args.MODEL_ID, sampling_kwargs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 合并结果
    model_name = args.MODEL_ID.split("/")[-1].lower()
    dst_name = args.DATASET_ID.split("/")[-1].lower()
    final_output = f"/workspace/yiqiuguo/lsrl/gen_results/{model_name}_{dst_name}_rollout{args.SAMPLING_PARAMS_n}_run{run_id}.jsonl"

    with open(final_output, "w", encoding="utf-8") as outfile:
        for i in range(args.NUM_GPUS):
            temp_file = f"temp_{run_id}_gpu_{i}.jsonl"
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                os.remove(temp_file) 

    print(f"✅ 任务全部完成！最终结果已保存至: {final_output}")