import os
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["FLASHINFER_LOG_LEVEL"] = "WARNING"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
import json
import torch
import torch.distributed as dist
import torch.nn.functional as F
import gc
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from vllm import LLM, SamplingParams

# 尝试导入生产环境的判题工具，如果缺失则使用降级逻辑
try:
    from math_utils import is_correct_v3
except ImportError:
    def is_correct_v3(pred, answer):
        return str(answer).strip().lower() in str(pred).strip().lower()

def main():
    # ==========================================
    # 1. 初始化 Distributed 环境
    # ==========================================
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"🚀 启动梯度对抗搜索评估 (Grad-Perturb) | 总进程数: {world_size}")
        print(f"{'=' * 60}")

    # ==========================================
    # 2. Mock 数据准备
    # ==========================================
    file_path = "/workspace/yiqiuguo/lsrl/gen_results/qwen2.5-1.5b-instruct_math-500_rollout32_run20260429_3871.jsonl"
    
    if rank == 0 and not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(json.dumps({"problem": "If x=2, what is x+3?", "responses": ["Let me think. I think x+3 is 4."], "answer": "5"}) + "\n")
            f.write(json.dumps({"problem": "What is the capital of France?", "responses": ["The capital of France is London, as everyone knows."], "answer": "Paris"}) + "\n")
    
    dist.barrier()

    # ==========================================
    # 3. 读取并切分数据 (Data Parallelism)
    # ==========================================
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    k = 8          # 取 TOP-K 个梯度最大的位置
    top_n = 50      # 在每个位置取 TOP-N 个模型预测的候选词进行梯度余弦比对
    max_new_tokens = 8192

    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    chunk_size = math.ceil(len(all_lines) / world_size)
    my_lines = all_lines[rank * chunk_size : (rank + 1) * chunk_size]
    
    if rank == 0:
        print(f"📦 数据总数: {len(all_lines)}, 每个进程处理约 {chunk_size} 条")

    # ==========================================
    # STAGE 1: HF 寻找对抗替换分支 (Grad-Perturbation)
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    ).to(device)
    
    hf_model.requires_grad_(False)
    hf_model.eval()
    
    # 提取模型的词表 Embedding 矩阵，用于后续计算余弦相似度
    word_embeddings = hf_model.get_input_embeddings().weight.detach()

    all_tasks = []
    metadata = [] 

    for idx, line in enumerate(tqdm(my_lines, desc=f"🔍 Rank {rank} 计算梯度替换点", position=rank)):
        item = json.loads(line)
        question = item.get('problem', item.get('question', ''))
        response = item['responses'][0]
        answer = item.get('answer', '')
        
        global_idx = rank * chunk_size + idx
        metadata.append({"global_idx": global_idx, "answer": answer})

        chat_prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(chat_prompt)
        response_ids = tokenizer.encode(response)
        
        P = len(prompt_ids)
        R = len(response_ids)
        if R < 4:
            continue
            
        input_ids = torch.tensor([prompt_ids + response_ids]).to(device)
        
        # 挂载 Embedding 梯度
        embeds = hf_model.get_input_embeddings()(input_ids).detach()
        embeds.requires_grad_(True)
        
        outputs = hf_model(inputs_embeds=embeds)
        logits = outputs.logits[0]
        
        half_R = R // 2
        
        # Cross Entropy Loss
        logits_second_half = logits[P + half_R - 1 : P + R - 1]
        targets_second_half = input_ids[0, P + half_R : P + R]
        
        loss = F.cross_entropy(logits_second_half, targets_second_half)
        loss.backward()
        
        # 计算前 50% 的梯度模长分数
        grads_first_half = embeds.grad[0, P : P + half_R]
        embeds_first_half = embeds[0, P : P + half_R]
        saliency_scores = (grads_first_half * embeds_first_half).sum(dim=-1).abs()
        
        # 取 TOP-K 个梯度最大的位置 (如果 half_R 不足 k，则取 half_R)
        actual_k = min(k, half_R)
        topk_rel_indices = torch.topk(saliency_scores, actual_k).indices.tolist()

        # 遍历这 K 个关键位置，进行单词替换
        for rel_idx in topk_rel_indices:
            pos = P + rel_idx
            
            # 当前位置的梯度 [hidden_size]
            grad_vec = embeds.grad[0, pos]
            
            # 预测当前位置词的 Logits 是在 pos - 1 步输出的
            logits_at_pos = logits[pos - 1]
            
            # 获取该位置预测概率 TOP-N 的候选词
            _, top_n_indices = torch.topk(logits_at_pos, top_n)
            
            # 提取这 N 个候选词的 Embedding [TOP_N, hidden_size]
            candidate_embeds = word_embeddings[top_n_indices]
            
            # 计算梯度与候选词 Embedding 的余弦相似度
            # grad_vec.unsqueeze(0) 会广播计算
            sims = F.cosine_similarity(grad_vec.unsqueeze(0), candidate_embeds, dim=-1)
            
            # 找出相似度最高的那个候选词
            best_candidate_idx = torch.argmax(sims).item()
            best_candidate_token = top_n_indices[best_candidate_idx].item()
            
            # 🌟 构建新 Prefix: 取完整的前 50%，仅替换这一个关键 Token
            new_prefix = input_ids[0, : P + half_R].tolist()
            new_prefix[pos] = best_candidate_token
            
            all_tasks.append({
                "global_idx": global_idx,
                "algo": "Grad_Perturb",
                "prefix_ids": new_prefix
            })

    # ==========================================
    # STAGE 2: 彻底清理 HF 占用
    # ==========================================
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    # ==========================================
    # STAGE 3: 初始化 vLLM Engine
    # ==========================================
    if rank == 0:
        print("\n" + "=" * 60)
        print("⚡ [STAGE 3] 初始化 vLLM Engine 进行分支延展...")

    with torch.cuda.device(device):
        vllm_engine = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
            data_parallel_size=world_size,
            distributed_executor_backend="external_launcher",
            max_num_seqs=1024,
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
            enforce_eager=True,
        )
    
    # 因为我们替换了词，可能会触发模型的不同思维链，所以可以给一点 temperature 配合
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_new_tokens)
    
    inputs = [{"prompt_token_ids": task["prefix_ids"]} for task in all_tasks]
    
    vllm_outputs = vllm_engine.generate(prompts=inputs, sampling_params=sampling_params, use_tqdm=(rank==0))

    # ==========================================
    # STAGE 4: 本地评估与全局 Reduce
    # ==========================================
    results_map = {meta["global_idx"]: {"Grad_Perturb": [], "answer": meta["answer"]} for meta in metadata}
    
    for task, output in zip(all_tasks, vllm_outputs):
        gen_text = output.outputs[0].text
        results_map[task["global_idx"]][task["algo"]].append(gen_text)

    local_perturb_passed = 0
    local_valid_count = 0

    for g_idx, data in results_map.items():
        if len(data["Grad_Perturb"]) == 0: 
            continue
        local_valid_count += 1
        answer = data["answer"]
        
        if any(is_correct_v3(text, answer) for text in data["Grad_Perturb"]):
            local_perturb_passed += 1

    stats_tensor = torch.tensor([local_valid_count, local_perturb_passed], dtype=torch.long, device=device)
    dist.reduce(stats_tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        total_valid = stats_tensor[0].item()
        total_perturb = stats_tensor[1].item()
        
        print("\n" + "=" * 60)
        print(f"📈 全局 DP 评估摘要 (基于 {world_size} 卡聚类)")
        print("-" * 60)
        print(f"🔹 Total Valid Questions: {total_valid}")
        if total_valid > 0:
            print(f"🔹 Grad-Perturb (Top {k} positions) Pass@{k}: {(total_perturb / total_valid) * 100:.2f}%")
        print("=" * 60)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()