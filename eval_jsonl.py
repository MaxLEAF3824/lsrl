import os
import math
import argparse
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# 如果你有 math_verify 相关依赖请保持导入
# import math_verify
# from math_verify import parse, verify
from math_utils import is_correct_v3

# 开启 Tokenizer 底层多线程并行加速 (这一行非常关键，配合批量编码才能起效)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def extract_model_and_dataset(file_path):
    """
    从文件路径中提取模型名和数据集名。
    解决带有下划线的数据集名（如 dapo_math_14k_en_openinstruct）被错误截断的问题。
    """
    basename = os.path.basename(file_path)
    
    # 1. 截断后缀：以 "_rollout" 作为分界线，丢弃后面的采样和时间戳参数
    if "_rollout" in basename:
        prefix = basename.split("_rollout")[0]
    else:
        prefix = basename.split(".jsonl")[0] # 兜底逻辑
        
    # 2. 从左向右只切分 1 次
    if "_" in prefix:
        model_name_local, dataset_name = prefix.split("_", 1)
    else:
        model_name_local = prefix
        dataset_name = "unknown_dataset"
    return model_name_local, dataset_name

model_hf_mapping = {
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B"
}

def calc_pass_at_k(n: int, c: int, k: int) -> float:
    """计算标准 Pass@k 的期望值"""
    if k > n:
        return None 
    if n - c < k:
        return 1.0  
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

def eval_jsonl_fast(file_path, possible_ks=[1, 4, 8, 16, 32], return_length=True, verbose=False):
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return
        
    # 解析名称 
    model_name_local, dataset_name = extract_model_and_dataset(file_path)
    if verbose:
        print("=" * 60)
        print(f"🚀 开始评估 | 模型: \033[1;36m[{model_name_local}]\033[0m | 数据集: \033[1;33m[{dataset_name}]\033[0m")
        print("=" * 60)
        print("⏳ 正在加载 Tokenizer (Qwen/Qwen3-1.7B)...")
    
    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    
    all_items = []
    all_preds_flat = []
    
    # ==========================================
    # 1. 第一阶段：读取数据并收集所有预测字符串 (避免在此处进行耗时计算)
    # ==========================================
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            all_items.append(item)
            preds = item.get('responses', [])
            for p in preds:
                all_preds_flat.append(str(p))

    total_responses_count = len(all_preds_flat)
    total_response_tokens = 0

    # ==========================================
    # 2. 第二阶段：循环外一次性批量 Tokenize (充分利用底层多线程加速)
    # ==========================================
    if total_responses_count > 0 and return_length:
        if verbose:
            print(f"⏳ 正在批量统计 Token 数量 (共 {total_responses_count} 条回复)...")
        
        # 批量编码提速：关闭不需要的 attention_mask 和特殊 token 以节省内存
        encoded_outputs = tokenizer(
            all_preds_flat, 
            add_special_tokens=False, 
            return_attention_mask=False,
            return_token_type_ids=False
        )
        # 快速累加长度
        total_response_tokens = sum(len(ids) for ids in encoded_outputs["input_ids"])
        avg_tokens = (total_response_tokens / total_responses_count) if total_responses_count > 0 else 0.0
    else:
        avg_tokens = None
    # ==========================================
    # 3. 第三阶段：计算正确率和 Pass@k
    # ==========================================
    max_preds = 0 
    total_questions = 0
    
    pass_k_sums = {k: 0.0 for k in possible_ks}
    valid_k_counts = {k: 0 for k in possible_ks}
    incorrect_problems = []
    correct_examples = []
    incorrect_examples = []
    
    if verbose:
        print("⏳ 正在计算正确率与 Pass@k...")
        
    for item in tqdm(all_items, disable=not verbose, desc="Evaluating"):
        gold_answer = item.get('answer', item.get('ground_truth', None))
        preds = item.get('responses', [])
        n = len(preds)
        
        if n == 0:
            continue
            
        max_preds = max(max_preds, n)
        total_questions += 1
        
        c = 0
        for p in preds:
            p_str = str(p)
            is_corr = is_correct_v3(p, gold_answer)
            if is_corr:
                c += 1
                if verbose and len(correct_examples) < 1:
                    correct_examples.append({
                        "problem": str(item.get('problem', '')),
                        "gold": str(gold_answer),
                        "pred": p_str
                    })
            else:
                if verbose and len(incorrect_examples) < 1:
                    incorrect_examples.append({
                        "problem": str(item.get('problem', '')),
                        "gold": str(gold_answer),
                        "pred": p_str
                    })
        
        if c == 0:
            incorrect_problems.append({
                "problem": str(item.get('problem', ''))[:100] + "...",
                "gold": gold_answer
            })
        
        # 实时累加 Pass@k 的值
        for k in possible_ks:
            if n >= k:
                pass_k_sums[k] += calc_pass_at_k(n, c, k)
                valid_k_counts[k] += 1

    # ==========================================
    # 4. 汇总及打印结果
    # ==========================================
    target_ks = [k for k in possible_ks if k <= max_preds]
    
    summary_results = {
        "Model": model_name_local,
        "Dataset": dataset_name,
        "Total Questions": total_questions,
        "Max Responses": max_preds
    }


    summary_results["Avg Tokens/Resp"] = avg_tokens
    
    for k in target_ks:
        if valid_k_counts[k] > 0:
            mean_val = pass_k_sums[k] / valid_k_counts[k]
        else:
            mean_val = 0.0
        summary_results[f"Pass@{k} (%)"] = mean_val * 100

    if verbose:
        print("\n" + "=" * 60)
        print("📈 评估结果摘要 (Evaluation Summary)")
        print("-" * 60)
        
        print("\n" + "=" * 60)
        print("✅ 示例：正确的回复 (Top 1 Correct Examples)")
        print("-" * 60)
        for i, ex in enumerate(correct_examples, 1):
            prob_trunc = ex['problem'][:2000].replace('\n', ' ') + ("..." if len(ex['problem']) > 2000 else "")
            pred_trunc = ex['pred'][:2000].replace('\n', ' ') + ("..." if len(ex['pred']) > 2000 else "")
            print(f"\033[1;32m[正确样例 {i}]\033[0m")
            print(f"  📝 问题: {prob_trunc}")
            print(f"  🎯 标准答案: {ex['gold']}")
            print(f"  🤖 模型回复: {pred_trunc}\n")

        print("=" * 60)
        print("❌ 示例：错误的回复 (Top 1 Incorrect Examples)")
        print("-" * 60)
        for i, ex in enumerate(incorrect_examples, 1):
            prob_trunc = ex['problem'][:2000].replace('\n', ' ') + ("..." if len(ex['problem']) > 2000 else "")
            pred_trunc = ex['pred'][:2000].replace('\n', ' ') + ("..." if len(ex['pred']) > 2000 else "")
            print(f"\033[1;31m[错误样例 {i}]\033[0m")
            print(f"  📝 问题: {prob_trunc}")
            print(f"  🎯 标准答案: {ex['gold']}")
            print(f"  🤖 模型回复: {pred_trunc}\n")
            
        print("=" * 60 + "\n")
        
        for key, value in summary_results.items():
            if isinstance(value, float):
                if "Pass@" in key:
                    print(f" 🔹 {key:<20}: \033[1;32m{value:>6.2f}%\033[0m")
                else:
                    print(f" 🔹 {key:<20}: \033[1;32m{value:>6.2f}\033[0m")
            else:
                print(f" 🔹 {key:<20}: {value:>7}")
        
    return summary_results

if __name__ == "__main__":
    # === 命令行参数解析 ===
    parser = argparse.ArgumentParser(description="LLM 数学/逻辑推理评估脚本 (计算 Pass@k)")
    
    # 必填位置参数
    parser.add_argument("file", type=str, help="需要评估的 JSONL 文件路径")
    
    # 可选参数
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16, 32], 
                        help="指定要计算的 Pass@k 列表。例如: --ks 1 4 8 (默认: 1 4 8 16 32)")
    
    args = parser.parse_args()
    
    # 执行评估
    res = eval_jsonl_fast(args.file, possible_ks=args.ks, verbose=True)