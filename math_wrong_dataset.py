import os
# 必须在导入或使用 transformers 之前设置，防止 HF Tokenizer 的 Rust 后端与多进程发生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from multiprocessing import Pool
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from math_verify import parse as math_verify_parse
from math_verify import verify as math_verify_verify
from math_utils import is_correct_v3, last_boxed_only_string, remove_boxed, is_equiv

# =========================================================================
# 1. 第一阶段：验证和过滤错误答案 (多进程 Worker)
# =========================================================================
def _process_single_item(item):
    """提取需要处理的题目和 '过程完整但结果错误' 的回复"""
    gold_answer = item.get("answer", item.get("ground_truth", None))
    responses = item.get("responses", [])
    
    if not any(is_correct_v3(p, gold_answer) for p in responses):
        complete_but_wrong_responses = []
        for res in responses:
            boxed_str = last_boxed_only_string(res)
            if boxed_str:
                parsed_res = math_verify_parse(remove_boxed(boxed_str))
                parsed_gold = math_verify_parse(gold_answer.strip())
                if not math_verify_verify(parsed_res, parsed_gold):
                    complete_but_wrong_responses.append(res)
        
        if complete_but_wrong_responses:
            return {
                "problem": item.get("q", item.get("problem", item.get("question", None))),
                "gold_answer": gold_answer,
                "complete_wrong_responses": complete_but_wrong_responses,
            }
    return None

# =========================================================================
# 2. 第二阶段：分词和截断组装 (多进程 Worker)
# =========================================================================
_worker_tok = None

def _init_tokenize_worker(tok):
    global _worker_tok
    _worker_tok = tok

def _tokenize_and_format_sample(args):
    """处理单个样本的 Tokenize 和格式化逻辑"""
    sample_idx, sample, thinking_ratio, old_thinking_pattern = args
    tok = _worker_tok
    results = []
    
    # prompt content 仅包含问题本身
    prompt_list = [{"role": "user", "content": sample["problem"]}]
    # 按照 DAPO 格式，这是一个 object array
    prompt_arr = np.array(prompt_list, dtype=object)

    for response_idx, wrong_response in enumerate(sample["complete_wrong_responses"]):
        uid = f"sample{sample_idx}_resp{response_idx}"
        try:
            # 计算 answer_length 用于统计
            tokens = tok.encode(wrong_response, add_special_tokens=False)
            answer_length = len(tokens)

            results.append({
                "uid": uid,
                "data_source": "math_wrong",
                "prompt": prompt_arr,
                "ability": "MATH",
                "reward_model": {
                    "ground_truth": sample["gold_answer"].strip(),
                },
                "answer_text": wrong_response,
                "question": sample["problem"],
                "answer_length": answer_length,
            })
        except Exception:
            pass
            
    return results

# =========================================================================
# 3. Dataset 类及构建函数 (整合多进程)
# =========================================================================
class MathWrongDataset(Dataset):
    def __init__(self, raw_samples, tok, thinking_ratio, max_samples, old_thinking_pattern=False, num_workers=16):
        self.flat_data = []
        
        # 准备分发给各个子进程的参数
        worker_args = [
            (idx, sample, thinking_ratio, old_thinking_pattern) 
            for idx, sample in enumerate(raw_samples)
        ]
        
        # 使用多进程加速 Tokenizer 操作
        chunksize = max(1, len(worker_args) // (num_workers * 4))
        with Pool(processes=num_workers, initializer=_init_tokenize_worker, initargs=(tok,)) as pool:
            for batch_results in tqdm(
                pool.imap_unordered(_tokenize_and_format_sample, worker_args, chunksize=chunksize), 
                total=len(worker_args), 
                desc=f"Tokenizing Dataset ({num_workers} Workers)"
            ):
                if batch_results:
                    self.flat_data.extend(batch_results)
                    # 达到数量上限可提前退出迭代（注：imap_unordered 可能会在后台多处理几个，我们稍后做精确截断）
                    if len(self.flat_data) >= max_samples:
                        break

        # 精确截断到 max_samples
        self.flat_data = self.flat_data[:max_samples]
        
        if len(self.flat_data) > 0:
            avg_len = np.mean([d['answer_length'] for d in self.flat_data])
            print(f"✅ 构建完成! data_size: {len(self.flat_data)} | avg len: {avg_len:.1f}")
        else:
            print("⚠️ 警告: 构建完成，但未提取到任何有效数据！")

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        return self.flat_data[idx]


def build_math_wrong_dataset(file_path: str, tok: AutoTokenizer, thinking_ratio, max_samples, num_workers: int = 16) -> MathWrongDataset:
    results = [json.loads(line) for line in open(file_path, "r", encoding="utf-8")]
    new_wrong_data = []
    
    # 阶段 1：多进程过滤出完整但错误的回答
    chunksize = max(1, len(results) // (num_workers * 4))
    with Pool(processes=num_workers) as pool:
        for processed_item in tqdm(
            pool.imap_unordered(_process_single_item, results, chunksize=chunksize), 
            total=len(results), 
            desc=f"Filtering Responses ({num_workers} Workers)"
        ):
            if processed_item is not None:
                new_wrong_data.append(processed_item)

    # 阶段 2：多进程进行 Tokenizer 截断与转换（Dataset 内部实现）
    return MathWrongDataset(
        raw_samples=new_wrong_data, 
        tok=tok, 
        thinking_ratio=thinking_ratio, 
        max_samples=max_samples, 
        old_thinking_pattern=False, 
        num_workers=num_workers
    )


if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Build math wrong dataset from jsonl and save as parquet.")
    parser.add_argument("file_path", type=str, help="Path to the input jsonl file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path for tokenizer.")
    parser.add_argument("--thinking_ratio", type=float, default=0.8, help="Ratio to split thinking and response.")
    parser.add_argument("--max_samples", type=int, default=9999999, help="Maximum number of samples to process.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker processes.")
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    print(f"Building dataset from {args.file_path}...")
    dataset = build_math_wrong_dataset(
        args.file_path, 
        tokenizer, 
        args.thinking_ratio, 
        args.max_samples, 
        args.num_workers
    )
    
    if dataset.flat_data:
        df = pd.DataFrame(dataset.flat_data)
        
        # Generate output path: original_filename_wrong.parquet
        base, _ = os.path.splitext(args.file_path)
        output_path = f"{base}_wrong.parquet"
        
        print(f"Saving {len(df)} samples to {output_path}...")
        df.to_parquet(output_path)
        print("Done!")
    else:
        print("No valid data found to save.")
