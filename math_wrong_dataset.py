from math_verify import parse as math_verify_parse
from math_verify import verify as math_verify_verify
from torch.utils.data import Dataset, DataLoader
import json
from math_utils import is_correct_v3, last_boxed_only_string, remove_boxed, is_equiv
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import numpy as np

class MathWrongDataset(Dataset):
    def __init__(self, raw_samples, tok, thinking_ratio, max_samples, old_thinking_pattern=False):
        self.flat_data = []
        for sample_idx, sample in tqdm(enumerate(raw_samples), leave=True, total=len(raw_samples), desc="Building MathWrongDataset..."):
            messages = [{"role": "user", "content": sample["problem"]}]
            question_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>"
            gt_text = sample["gold_answer"] + "}\n"

            for response_idx, wrong_response in enumerate(sample["complete_wrong_responses"]):
                uid = f"sample{sample_idx}_resp{response_idx}"
                try:
                    if old_thinking_pattern:
                        parts = wrong_response.split("</think>")
                        if len(parts) == 2:
                            thinking_text = parts[0]
                            # 既然 len(parts) == 2，直接取 parts[1] 即可
                            after_thinking_text = parts[1] 
                            assert "\\boxed{" in after_thinking_text, f"[{uid}] 缺少 boxed 结果"
                            pred_box_text = last_boxed_only_string(after_thinking_text)
                            connector_text = after_thinking_text.split(pred_box_text)[0] + "\\boxed{"
                            pred_text = remove_boxed(pred_box_text) + "}\n"
                    else:
                        tokens = tok.encode(wrong_response, add_special_tokens=False)
                        answer_length = len(tokens)
                        split_idx = int(len(tokens) * thinking_ratio)
                        thinking_text = tok.decode(tokens[:split_idx])
                        # 将剩余的 20% 作为 after_thinking_text，防止后续处理报错
                        after_thinking_text = tok.decode(tokens[split_idx:])
                        # 同样需要确保后 20% 包含结果，如果不包含会触发 except 直接跳过该样本
                        assert "\\boxed{" in after_thinking_text, f"[{uid}] 截断后的后20%文本中缺少 boxed 结果"
                        pred_box_text = last_boxed_only_string(after_thinking_text)
                        connector_text = after_thinking_text.split(pred_box_text)[0] + "\\boxed{"
                        pred_text = remove_boxed(pred_box_text) + "}\n"

                    self.flat_data.append(
                        {
                            "uid": uid,
                            "question_text": question_text,
                            "answer_text": wrong_response,
                            "thinking_text": thinking_text,
                            "connector_text": connector_text,
                            "pred_text": pred_text,
                            "gt_text": gt_text,
                            "problem": sample["problem"],
                            "answer_length": answer_length,
                        }
                    )
                except Exception:
                    pass
            if len(self.flat_data) > max_samples:
                break
        self.flat_data = self.flat_data[:max_samples]
        # self.flat_data = sorted(self.flat_data, key=lambda x: -len(x["thinking_text"]))
        print(f"data_size:{len(self.flat_data)} avg len: {np.mean([d['answer_length'] for d in self.flat_data])}")

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        return self.flat_data[idx]


def build_math_wrong_dataset(file_path: str, tok: AutoTokenizer, thinking_ratio, max_samples) -> MathWrongDataset:
    results = [json.loads(line) for line in open(file_path, "r", encoding="utf-8")]
    new_wrong_data = []
    for item in tqdm(results, desc="Processing Data", leave=False):
        gold_answer = item.get("answer", item.get("ground_truth", None))
        responses = item.get("responses", [])
        if not any(is_correct_v3(p, gold_answer) for p in responses):
            complete_but_wrong_responses = [
                res
                for res in responses
                if last_boxed_only_string(res)
                and not math_verify_verify(
                    math_verify_parse(remove_boxed(last_boxed_only_string(res))),
                    math_verify_parse(gold_answer.strip()),
                )
            ]
            if complete_but_wrong_responses:
                new_wrong_data.append(
                    {
                        "problem": item.get("q", item.get("problem", item.get("question", None))),
                        "gold_answer": gold_answer,
                        "complete_wrong_responses": complete_but_wrong_responses,
                    }
                )
    return MathWrongDataset(new_wrong_data, tok, thinking_ratio, max_samples)
