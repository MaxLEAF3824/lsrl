import math_verify
from math_verify import parse, verify
import os
import orjson as json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import datetime
import math
from tqdm import tqdm
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py


from math_utils import is_correct_v3

# 开启 Tokenizer 底层多线程并行加速
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
        
    # 2. 从左向右只切分 1 次 (因为你的模型名称里都是连字符，没有下划线)
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

import os
import json
import math
import argparse
import pandas as pd
from tqdm import tqdm

# ==========================================
# 假设这些是你原本代码中定义好的依赖和函数
# 请确保它们在你的实际脚本中正常导入
# from your_utils import extract_model_and_dataset, is_correct_v3, model_hf_mapping, loaded_tokenizers
# from transformers import AutoTokenizer
# ==========================================

def calc_pass_at_k(n: int, c: int, k: int) -> float:
    """计算标准 Pass@k 的期望值"""
    if k > n:
        return None 
    if n - c < k:
        return 1.0  
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

def eval_jsonl_fast(file_path, possible_ks=[1, 4, 8, 16, 32], verbose=False):
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return
        
    # 解析名称 (请确保该函数已定义)
    model_name_local, dataset_name = extract_model_and_dataset(file_path)
    if verbose:
        print("=" * 60)
        print(f"🚀 开始评估 | 模型: \033[1;36m[{model_name_local}]\033[0m | 数据集: \033[1;33m[{dataset_name}]\033[0m")
        print("=" * 60)
    
    max_preds = 0 
    total_questions = 0
    
    # 🌟 优化核心：使用字典直接累加结果，彻底抛弃 Pandas
    pass_k_sums = {k: 0.0 for k in possible_ks}
    valid_k_counts = {k: 0 for k in possible_ks}
    incorrect_problems = []
    
    # 新增：用于在 verbose 模式下展示的具体回复样例
    correct_examples = []
    incorrect_examples = []
    
    # 1. 边读边算 (Streaming) - 只需遍历一次文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            gold_answer = item.get('answer', item.get('ground_truth', None))
            preds = item.get('responses', [])
            n = len(preds)
            
            if n == 0:
                continue
                
            max_preds = max(max_preds, n)
            total_questions += 1
            
            # 展开生成器表达式，以便在评估时捕获具体的 correct/incorrect 样例
            c = 0
            for p in preds:
                is_corr = is_correct_v3(p, gold_answer)
                if is_corr:
                    c += 1
                    # 收集正确的样例 (最多 3 个)
                    if verbose and len(correct_examples) < 3:
                        correct_examples.append({
                            "problem": str(item.get('problem', '')),
                            "gold": str(gold_answer),
                            "pred": str(p)
                        })
                else:
                    # 收集错误的样例 (最多 3 个)
                    if verbose and len(incorrect_examples) < 3:
                        incorrect_examples.append({
                            "problem": str(item.get('problem', '')),
                            "gold": str(gold_answer),
                            "pred": str(p)
                        })
            
            if c == 0:
                incorrect_problems.append({
                    "problem": str(item.get('problem', ''))[:100] + "...",
                    "gold": gold_answer
                })
            
            # 实时累加 Pass@k 的值
            for k in possible_ks:
                if n >= k:  # 仅当样本的预测数量满足 k 时才计算
                    pass_k_sums[k] += calc_pass_at_k(n, c, k)
                    valid_k_counts[k] += 1

    # 动态过滤掉超过实际生成数量的 K 值
    target_ks = [k for k in possible_ks if k <= max_preds]
    
    # 2. 汇总结果 (直接用累加值求平均)
    summary_results = {
        "Model": model_name_local,
        "Dataset": dataset_name,
        "Total Questions": total_questions,
        "Max Responses": max_preds
    }
    
    for k in target_ks:
        # 避免除以 0 的情况
        if valid_k_counts[k] > 0:
            mean_val = pass_k_sums[k] / valid_k_counts[k]
        else:
            mean_val = 0.0
        summary_results[f"Pass@{k} (%)"] = mean_val * 100

    # 3. 美观打印
    if verbose:
        print("\n" + "=" * 60)
        print("📈 评估结果摘要 (Evaluation Summary)")
        print("-" * 60)
        
        # 打印收集到的正确回复样例
        print("\n" + "=" * 60)
        print("✅ 示例：正确的回复 (Top 3 Correct Examples)")
        print("-" * 60)
        for i, ex in enumerate(correct_examples, 1):
            prob_trunc = ex['problem'][:20000].replace('\n', ' ') + ("..." if len(ex['problem']) > 20000 else "")
            pred_trunc = ex['pred'][:20000].replace('\n', ' ') + ("..." if len(ex['pred']) > 20000 else "")
            print(f"\033[1;32m[正确样例 {i}]\033[0m")
            print(f"  📝 问题: {prob_trunc}")
            print(f"  🎯 标准答案: {ex['gold']}")
            print(f"  🤖 模型回复: {pred_trunc}\n")

        # 打印收集到的错误回复样例
        print("=" * 60)
        print("❌ 示例：错误的回复 (Top 3 Incorrect Examples)")
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
                print(f" 🔹 {key:<20}: \033[1;32m{value:>6.2f}%\033[0m")
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