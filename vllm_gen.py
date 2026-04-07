import json
import os
import time
import uuid
from regex import D
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams
from datasets import load_dataset

prompt_template = """You are an expert mathematician. Please solve the following math problem.

You must strictly adhere to the following output format:
1. First, write out your detailed, step-by-step reasoning process. You MUST enclose your entire reasoning process within `<think>` and `</think>` tags.
2. Provide your final mathematical answer enclosed within `\\boxed{}`. 

Problem:
{question}

"""
# ---------------------------------------------------------
# 1. 核心配置
# ---------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-1.7B"  # 确保路径指向最新的 Qwen3
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # 确保路径指向最新的 Qwen3
# MODEL_ID = "Qwen/Qwen3-0.6B"  # 确保路径指向最新的 Qwen3
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

DATASET_ID = "HuggingFaceH4/MATH-500"  # 远程 MATH500 地址
# DATASET_ID = "openai/gsm8k"  # 远程 GSM8K 地址
NUM_GPUS = 4

# 采样参数 (rollout_num=8, temp=1.0, max_new=4096)
SAMPLING_PARAMS = SamplingParams(
    n=8,
    temperature=1.0,
    max_tokens=32768,
    top_p=0.95,
)

# ---------------------------------------------------------
# 2. 单 GPU 推理逻辑 (新增 run_id 参数)
# ---------------------------------------------------------
def run_inference(rank, data_chunk, run_id):
    """
    rank: 显卡编号
    data_chunk: 该卡负责的数据子集
    run_id: 本次运行的唯一标识符
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.95
    )

    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for item in data_chunk:
        prompt_text = prompt_template.format(question=item['problem'])
        messages = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text += "<think>"
        formatted_prompts.append(prompt_text)
    
    outputs = llm.generate(formatted_prompts, SAMPLING_PARAMS)

    # 【修改】：temp_file 现在包含 run_id
    temp_file = f"temp_{run_id}_gpu_{rank}.jsonl"
    with open(temp_file, "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            res_list = [res.text.strip() for res in output.outputs]
            entry = data_chunk[i].copy() 
            entry["responses"] = res_list
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✨ GPU {rank} 处理完毕，样本数: {len(data_chunk)}")

# ---------------------------------------------------------
# 3. 主程序控制
# ---------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    # 【新增】：生成唯一运行 ID (格式：时间戳+随机短ID)
    # 例如：20231027_143005_a1b2
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]

    print(f"正在从 HuggingFace 加载数据集: {DATASET_ID}...")
    from datasets import load_dataset

    # 假设你的 DATASET_ID 变量来自配置或命令行参数
    if DATASET_ID == "HuggingFaceH4/MATH-500":
        dataset = load_dataset(DATASET_ID, split='test')

    elif DATASET_ID == "KbsdJames/Omni-MATH":
        # Omni-MATH 官方仓库通常将所有数据放在 train split 中
        dataset = load_dataset(DATASET_ID, split='train')

    elif DATASET_ID == "Hothan/OlympiadBench":
        # OlympiadBench 包含多个子集，必须通过 name 参数指定纯文本英语竞赛子集
        # 评测通常使用 test split
        dataset = load_dataset(DATASET_ID, name="OE_TO_maths_en_COMP", split='test')

    elif DATASET_ID == "UGMathBench/ugmathbench":
        # UGMathBench 通常包含 test 或默认 split，按规范读取 test
        dataset = load_dataset(DATASET_ID, split='test')

    elif DATASET_ID == "math-ai/amc23":
        # math-ai 组织的单一年份真题（如 AMC 23）数据量较小，通常全量存放在 train split 中
        dataset = load_dataset(DATASET_ID, split='train')

    else:
        # 默认兜底逻辑
        dataset = load_dataset(DATASET_ID, split='test')
    
    all_data = [item for item in dataset]
    total_len = len(all_data)
    
    chunk_size = (total_len + NUM_GPUS - 1) // NUM_GPUS
    chunks = [all_data[i:i + chunk_size] for i in range(0, total_len, chunk_size)]

    print(f"🚀 开始 DP 并行推理 | Run ID: {run_id} | 总题数: {total_len}")

    processes = []
    for i in range(len(chunks)):
        # 【修改】：将 run_id 传给进程
        p = mp.Process(target=run_inference, args=(i, chunks[i], run_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 合并结果
    model_name = MODEL_ID.split("/")[-1].lower()
    dst_name = DATASET_ID.split("/")[-1].lower()
    final_output = f"{model_name}_{dst_name}_rollout{SAMPLING_PARAMS.n}_run{run_id}.jsonl"

    with open(final_output, "w", encoding="utf-8") as outfile:
        for i in range(NUM_GPUS):
            # 【修改】：读取带 run_id 的临时文件
            temp_file = f"temp_{run_id}_gpu_{i}.jsonl"
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                os.remove(temp_file) 

    print(f"✅ 任务全部完成！最终结果已保存至: {final_output}")