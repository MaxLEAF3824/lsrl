import math_verify
from math_verify import parse, verify
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
    string = string.replace("\%", "")  # noqa: W605

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