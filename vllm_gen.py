import json
import os
import time
import uuid
import argparse
import random
import glob
import torch.multiprocessing as mp
from datasets import load_dataset

prompt_template = """You are an expert mathematician. Please solve the following math problem.

You must strictly adhere to the following output format:
1. First, write out your detailed, step-by-step reasoning process. You MUST enclose your entire reasoning process within `<think>` and `</think>` tags.
2. Provide your final mathematical answer enclosed within `\\boxed{{}}`. 

Problem:
{question}

"""

# 统一输出目录，方便集中管理和扫描
BASE_OUT_DIR = "/workspace/yiqiuguo/lsrl/gen_results"

# ---------------------------------------------------------
# 单 GPU 推理逻辑 (微批次追加写入防暴毙)
# ---------------------------------------------------------
def run_inference(rank, data_chunk, run_id, model_id, sampling_kwargs, node_rank):
    if not data_chunk:
        print(f"✨ GPU {rank} 分配到的数据为空 (可能已全部续传完毕)，直接退出。")
        return

    # 获取动态空闲端口的辅助函数
    import socket
    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    # ================= 核心修复区开始 =================
    # 0. 错开启动时间，防止多进程在同一微秒拿到同一个"空闲"端口
    import time
    time.sleep(rank * 1.0) 

    # 1. 绝对优先：隔离 GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    
    # 2. 动态分配绝对安全的空闲端口
    master_port = str(get_free_port())
    vllm_port = str(get_free_port())
    os.environ["MASTER_PORT"] = master_port
    os.environ["VLLM_PORT"] = vllm_port
    
    # 3. 限制 DataLoader 线程数
    os.environ["OMP_NUM_THREADS"] = "4"

    print(f"🔧 GPU {rank} 分配动态端口: MASTER_PORT={master_port}, VLLM_PORT={vllm_port}")

    # 5. 环境完全干净且隔离后，再进行局部 Import！
    from vllm import LLM, SamplingParams
    # ================= 核心修复区结束 =================
    
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.95
    )

    sampling_params = SamplingParams(**sampling_kwargs)
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    
    for item in data_chunk:
        question_text = item.get('q', None)
        messages = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text += "<think>"
        formatted_prompts.append(prompt_text)
    
    # temp 文件统一存放到结果目录
    temp_file = os.path.join(BASE_OUT_DIR, f"temp_{run_id}_node{node_rank}_gpu_{rank}.jsonl")
    
    # 核心：游击战微批次模式
    MICRO_BATCH_SIZE = 50  # 每 50 题落盘一次
    total_len = len(formatted_prompts)
    
    print(f"🚀 GPU {rank} 启动！分配任务: {total_len} 题，每 {MICRO_BATCH_SIZE} 题追加存档一次。")

    for i in range(0, total_len, MICRO_BATCH_SIZE):
        chunk_prompts = formatted_prompts[i : i + MICRO_BATCH_SIZE]
        chunk_data = data_chunk[i : i + MICRO_BATCH_SIZE]
        
        # 关闭 tqdm 防止控制台被多个进程刷屏变花
        outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=False)

        # 跑完一批，立刻追加模式 ("a") 落盘
        with open(temp_file, "a", encoding="utf-8") as f:
            for j, output in enumerate(outputs):
                res_list = [res.text.strip() for res in output.outputs]
                entry = chunk_data[j].copy() 
                entry["responses"] = res_list
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        current_done = min(i + MICRO_BATCH_SIZE, total_len)
        print(f"💾 [GPU {rank} 存档] 进度: {current_done}/{total_len} ({(current_done/total_len):.1%})")
    
    print(f"✨ GPU {rank} 本次任务彻底处理完毕！")

# ---------------------------------------------------------
# 主程序控制
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Offline Inference with Auto-Resume")
    
    parser.add_argument("--MODEL_ID", type=str, default="Qwen/Qwen3-1.7B", help="Path or ID of the model")
    parser.add_argument("--DATASET_ID", type=str, default="HuggingFaceH4/MATH-500", help="Dataset ID")
    parser.add_argument("--NUM_GPUS", type=int, default=4, help="Number of GPUs to use for DP inference")
    
    parser.add_argument("--SAMPLING_PARAMS_n", type=int, default=8, help="Number of rollouts")
    parser.add_argument("--SAMPLING_PARAMS_temperature", type=float, default=1.0)
    parser.add_argument("--SAMPLING_PARAMS_max_tokens", type=int, default=32768)
    parser.add_argument("--SAMPLING_PARAMS_top_p", type=float, default=0.95)
    
    parser.add_argument("--NODE_TOTAL", type=int, default=1, help="总节点数")
    parser.add_argument("--NODE_RANK", type=int, default=0, help="当前节点编号")
    parser.add_argument("--RUN_SEED", type=int, default=None, help="跨机统一的数字种子")
    
    args = parser.parse_args()
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    sampling_kwargs = {
        "n": args.SAMPLING_PARAMS_n,
        "temperature": args.SAMPLING_PARAMS_temperature,
        "max_tokens": args.SAMPLING_PARAMS_max_tokens,
        "top_p": args.SAMPLING_PARAMS_top_p
    }

    mp.set_start_method('spawn', force=True)

    current_date = time.strftime("%Y%m%d")
    if args.RUN_SEED is not None:
        random.seed(args.RUN_SEED)
        short_hex = "".join(random.choices("0123456789abcdef", k=4))
        run_id = f"{current_date}_{short_hex}"
    else:
        run_id = current_date + "_" + uuid.uuid4().hex[:4]

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
        ddatasets = load_dataset(args.DATASET_ID, "main", split="test")
        q_key = "question"
    elif args.DATASET_ID == "HuggingFaceH4/aime_2024":
        dataset = load_dataset(args.DATASET_ID, split="train")
        q_key = "problem"
    elif args.DATASET_ID == "math-ai/aime25":
        dataset = load_dataset(args.DATASET_ID, split="test")
        q_key= "problem"
    else:
        dataset = load_dataset(args.DATASET_ID, split='test')
        q_key = "problem"
    
    all_data = [(item|{"q": item[q_key]}) for item in dataset]
    global_total_len = len(all_data)
    
    # ---------------------------------------------------------
    # 1. 跨机级数据绝对切片 (雷打不动)
    # ---------------------------------------------------------
    node_chunk_size = (global_total_len + args.NODE_TOTAL - 1) // args.NODE_TOTAL
    start_idx = args.NODE_RANK * node_chunk_size
    end_idx = min(start_idx + node_chunk_size, global_total_len)
    
    node_data = all_data[start_idx:end_idx]
    original_node_len = len(node_data)
    
    print(f"🌍 集群初始分配: 本节点 {args.NODE_RANK} 负责全局索引 {start_idx} ~ {end_idx-1} (共 {original_node_len} 题)")

    # ---------------------------------------------------------
    # 2. 本地自动扫描断点续传 (Auto-Resume)
    # ---------------------------------------------------------
    finished_questions = set()
    
    # 只要是你这个 run_id 的 temp 文件，统统扫一遍
    temp_pattern = os.path.join(BASE_OUT_DIR, f"temp_{run_id}_node{args.NODE_RANK}_gpu_*.jsonl")
    for f_path in glob.glob(temp_pattern):
        with open(f_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'q' in item:
                        finished_questions.add(item['q'])
                except json.JSONDecodeError:
                    pass # 忽略被强杀时写了一半的脏数据

    # 在本节点的数据里，剔除掉已经生成好的
    node_data = [d for d in node_data if d['q'] not in finished_questions]
    skipped_count = original_node_len - len(node_data)
    
    total_len = len(node_data)
    if skipped_count > 0:
        print(f"♻️ [自动续传] 侦测到本地已完成 {skipped_count} 题，跳过生成！剩余需要跑: {total_len} 题。")
    else:
        print("💡 [全新任务] 未检测到本 run_id 的本地存档，全量开跑！")

    if total_len == 0:
        print(f"🎉 节点 {args.NODE_RANK} 的所有任务均已通过续传完成，直接进入大一统合并阶段！")
    else:
        # ---------------------------------------------------------
        # 3. 单机内多卡数据分配
        # ---------------------------------------------------------
        chunk_size = (total_len + args.NUM_GPUS - 1) // args.NUM_GPUS
        chunks = [node_data[i:i + chunk_size] for i in range(0, total_len, chunk_size)]

        print(f"🚀 开始 DP 并行推理 | Run ID: {run_id} | 模型: {args.MODEL_ID}")

        processes = []
        for i in range(len(chunks)):
            p = mp.Process(target=run_inference, args=(i, chunks[i], run_id, args.MODEL_ID, sampling_kwargs, args.NODE_RANK))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    # ---------------------------------------------------------
    # 4. 节点内合并结果与清理 (原子化写入)
    # ---------------------------------------------------------
    model_name = args.MODEL_ID.split("/")[-1].lower()
    dst_name = args.DATASET_ID.split("/")[-1].lower()
    
    node_suffix = f"_node{args.NODE_RANK}" if args.NODE_TOTAL > 1 else ""
    
    global_final_output = os.path.join(BASE_OUT_DIR, f"{model_name}_{dst_name}_rollout{args.SAMPLING_PARAMS_n}_run{run_id}.jsonl")
    node_final_output = os.path.join(BASE_OUT_DIR, f"{model_name}_{dst_name}_rollout{args.SAMPLING_PARAMS_n}_run{run_id}{node_suffix}.jsonl") if args.NODE_TOTAL > 1 else global_final_output

    temp_node_output = node_final_output + ".tmp"

    # 将本节点下所有的临时碎片合并且去重 (以防万一续传时有小重叠)
    merged_q_set = set()
    with open(temp_node_output, "w", encoding="utf-8") as outfile:
        # 首先，如果本节点以前跑出过完整的正式文件，把它读进来
        if os.path.exists(node_final_output):
             with open(node_final_output, "r", encoding="utf-8") as infile:
                 for line in infile:
                     try:
                         item = json.loads(line)
                         if 'q' in item and item['q'] not in merged_q_set:
                             merged_q_set.add(item['q'])
                             outfile.write(line)
                     except: pass
                     
        # 其次，读入刚刚跑完的 temp 文件
        for i in range(args.NUM_GPUS):
            temp_file = os.path.join(BASE_OUT_DIR, f"temp_{run_id}_node{args.NODE_RANK}_gpu_{i}.jsonl")
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8") as infile:
                    for line in infile:
                         try:
                             item = json.loads(line)
                             # 去重保护
                             if 'q' in item and item['q'] not in merged_q_set:
                                 merged_q_set.add(item['q'])
                                 outfile.write(line)
                         except: pass
                # 碎片合并完成，销毁临时文件
                os.remove(temp_file)

    os.replace(temp_node_output, node_final_output)
    print(f"✅ 节点 {args.NODE_RANK} 自身任务全部完成并安全落盘: {node_final_output}")

    # ---------------------------------------------------------
    # 5. 跨机集群等待与大合并 (仅由 Rank 0 执行)
    # ---------------------------------------------------------
    if args.NODE_TOTAL > 1 and args.NODE_RANK == 0:
        print("\n👑 [Master 节点 0] 已完工，进入轮询等待模式，准备合并其他节点数据...")
        
        all_expected_files = [os.path.join(BASE_OUT_DIR, f"{model_name}_{dst_name}_rollout{args.SAMPLING_PARAMS_n}_run{run_id}_node{i}.jsonl") for i in range(args.NODE_TOTAL)]
        
        while True:
            missing_nodes = [str(i) for i, f in enumerate(all_expected_files) if not os.path.exists(f)]
            
            if not missing_nodes:
                print("\n🎉 所有节点文件均已集齐！开始最终的全局合并...")
                break
                
            print(f"⏳ 等待中... 尚未完工的节点: [{', '.join(missing_nodes)}]，5秒后重新检查...")
            time.sleep(5)
            
        with open(global_final_output, "w", encoding="utf-8") as global_out:
            for f in all_expected_files:
                with open(f, "r", encoding="utf-8") as node_in:
                    global_out.write(node_in.read())
                    
        print(f"🚀🚀🚀 全局合并大功告成！完美结果文件: {global_final_output}")