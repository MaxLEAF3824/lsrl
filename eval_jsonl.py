import math_verify
from math_verify import parse, verify
import os
import json
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


def compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(e)

    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if not isinstance(s, str) or not s:
        return ""
    
    # 去除首尾可能存在的空白字符
    s = s.strip()
    
    # 情况 1: 标准的 \boxed{...}
    if s.startswith("\\boxed{"):
        s = s[len("\\boxed{"):]
        # 如果末尾有闭合括号，则去掉它
        if s.endswith("}"):
            s = s[:-1]
        return s
    
    # 情况 2: 带空格的 \boxed ...
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    
    # 情况 3: 连在一起的非标准格式，如 \boxed123
    if s.startswith("\\boxed"):
        return s[len("\\boxed"):]
        
    return s

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{" in string:
        splits = string.split("\\text{")
        assert len(splits) == 2
        return splits[1][:-1]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def extract_answer(text):
    if not isinstance(text, str):
        return ""
    # 匹配 \boxed{内容},取最后一个匹配项（通常是最终结论）
    patterns = [
        r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', # 处理嵌套大括号
        r'The answer is[:\s]*([^\.]+)',             # 备选：非 LaTeX 格式
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return ""

def is_correct_v3(model_output, ground_truth):
    """
    model_output: 模型生成的完整文本 (包含思考过程)
    ground_truth: 数据集里的标准答案 (通常是已经提取出来的答案字符串)
    """
    try:
        # 1. 找到模型输出中最后一个 \boxed{} 的内容
        extracted_box = last_boxed_only_string(model_output)
        
        if extracted_box is None:
            # 如果没找到 boxed，这道题通常判定为错
            return False
            
        # 2. 去掉 \boxed{} 外壳
        # 注意：这里加个 try 因为 remove_boxed 里面有 assert，失败会报错
        try:
            model_answer = remove_boxed(extracted_box)
        except:
            # 处理类似 \boxed 后面没带括号的极端情况
            model_answer = extracted_box.replace("\\boxed", "").strip("{} ")

        # 3. 使用官方的等价性判断函数
        # is_equiv 内部会调用那套复杂的 strip_string 逻辑
        return is_equiv(model_answer, ground_truth)
        
    except Exception as e:
        # 打印错误方便调试，实际跑大规模数据时可以关掉
        # print(f"Eval Error: {e}")
        return False

def is_correct_v4(model_output, ground_truth):
    """
    model_output: 模型生成的完整文本 (包含思考过程)
    ground_truth: 数据集里的标准答案 (通常是已经提取出来的答案字符串)
    """
    try:
        # 1. 找到模型输出中最后一个 \boxed{} 的内容
        extracted_box = last_boxed_only_string(model_output)
        
        if extracted_box is None:
            # 如果没找到 boxed，这道题通常判定为错
            return False
            
        # 2. 去掉 \boxed{} 外壳
        # 注意：这里加个 try 因为 remove_boxed 里面有 assert，失败会报错
        try:
            model_answer = remove_boxed(extracted_box)
        except:
            # 处理类似 \boxed 后面没带括号的极端情况
            model_answer = extracted_box.replace("\\boxed", "").strip("{} ")

        def is_equiv_math_verify(str1, str2, verbose=False):
            if str1 is None and str2 is None:
                print("WARNING: Both None")
                return True
            if str1 is None or str2 is None:
                return False

            try:
                ss1 = strip_string(str1)
                ss2 = strip_string(str2)
                if verbose:
                    print(ss1, ss2)
                return verify(parse(ss1), parse(ss2))
            except Exception:
                return verify(parse(str1), parse(str2))
        
        # 3. 使用魔改的等价性判断函数
        # is_equiv 内部会调用那套复杂的 strip_string 逻辑
        # return is_equiv_math_verify(model_answer, ground_truth)
        return is_equiv(model_answer, ground_truth)
        
    except Exception as e:
        # 打印错误方便调试，实际跑大规模数据时可以关掉
        # print(f"Eval Error: {e}")
        return False
    
def apply_chat_template(tok, prompt):
    # 构造对话结构
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用 Chat Template
    # tokenize=False 返回字符串，add_generation_prompt=True 添加助手引导符
    prompt_text = tok.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. 核心修改：在助手回复的开头强制注入 <think> 标记
    # 这样模型会从 <think> 之后开始补全，从而触发推理逻辑
    prompt_text += "<think>"
    return prompt_text

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

def eval_jsonl(file_path, possible_ks=[1, 4, 8, 16, 32], verbose=False):
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在: {file_path}")
        return
        
    # 解析名称 (请确保该函数已定义)
    model_name_local, dataset_name = extract_model_and_dataset(file_path)
    if verbose:
        print("=" * 60)
        print(f"🚀 开始评估 | 模型: \033[1;36m[{model_name_local}]\033[0m | 数据集: \033[1;33m[{dataset_name}]\033[0m")
        print("=" * 60)
    
    # Tokenizer 逻辑 (请确保依赖已导入)
    # hf_model_path = model_hf_mapping.get(model_name_local, model_name_local)
    # if hf_model_path not in loaded_tokenizers:
    #     loaded_tokenizers[hf_model_path] = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True, use_fast=True)
    # tok = loaded_tokenizers[hf_model_path]
    
    # 1. 读取数据
    results = []
    max_preds = 0 
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            results.append(item)
            preds = item.get('responses', [])
            max_preds = max(max_preds, len(preds))
            
    # 动态过滤掉超过实际生成数量的 K 值
    target_ks = [k for k in possible_ks if k <= max_preds]
    
    eval_data = []
    incorrect_problems = []
    
    # 2. 遍历数据并计算 Correctness 与 Pass@k
    for item in results:
        gold_answer = item.get('answer', item.get('ground_truth', None))
        preds = item.get('responses', [])
        n = len(preds)
        
        # 判断正确性 (请确保 is_correct_v3 已定义)
        correctness = [is_correct_v3(p, gold_answer) for p in preds]
        c = sum(correctness)
        
        problem_metrics = {
            "problem": item.get('problem', ''),
            "gold": gold_answer,
            "n": n,
            "c": c,
            "any_correct": c > 0
        }
        
        for k in target_ks:
            problem_metrics[f"pass@{k}"] = calc_pass_at_k(n, c, k)
            
        eval_data.append(problem_metrics)
        
        if c == 0:
            incorrect_problems.append({
                "problem": item.get('problem', '')[:100] + "...",
                "gold": gold_answer
            })
            
    eval_df = pd.DataFrame(eval_data)
    total_questions = len(eval_df)
    
    # 3. 汇总与美观打印
    summary_results = {
        "Model": model_name_local,
        "Dataset": dataset_name,
        "Total Questions": total_questions,
        "Max Responses": max_preds
    }
    
    for k in target_ks:
        col_name = f"pass@{k}"
        summary_results[f"Pass@{k} (%)"] = float(eval_df[col_name].dropna().mean() * 100)

    if verbose:
        print("\n" + "=" * 60)
        print("=" * 60)
        print("📈 评估结果摘要 (Evaluation Summary)")
        print("-" * 60)
        
        for key, value in summary_results.items():
            if isinstance(value, float):
                print(f" 🔹 {key:<20}: \033[1;32m{value:>6.2f}%\033[0m")
            else:
                print(f" 🔹 {key:<20}: {value:>7}")
        print("=" * 60 + "\n")
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
    res = eval_jsonl(args.file, possible_ks=args.ks, verbose=True)