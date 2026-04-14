from math_verify import parse as math_verify_parse
from math_verify import verify as math_verify_verify
from torch.utils.data import Dataset, DataLoader
import json
from math_utils import is_correct_v3, last_boxed_only_string, remove_boxed, is_equiv
from tqdm.auto import tqdm
from transformers import AutoTokenizer


class MathWrongDataset(Dataset):
    def __init__(self, raw_samples, tok):
        self.flat_data = []
        for sample_idx, sample in enumerate(raw_samples):
            messages = [{"role": "user", "content": sample["problem"]}]
            question_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>"
            gt_text = sample["gold_answer"].strip() + "}"

            for response_idx, wrong_response in enumerate(sample["complete_wrong_responses"]):
                uid = f"sample{sample_idx}_resp{response_idx}"
                try:
                    parts = wrong_response.split("</think>")
                    thinking_text = parts[0]
                    after_thinking_text = parts[1] if len(parts) > 1 else ""
                    assert "\\boxed{" in after_thinking_text, f"[{uid}] 缺少 boxed 结果"

                    pred_box_text = last_boxed_only_string(after_thinking_text)
                    connector_text = after_thinking_text.split(pred_box_text)[0] + "\\boxed{"
                    pred_text = remove_boxed(pred_box_text) + "}"

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
                        }
                    )
                    self.flat_data = sorted(self.flat_data, key=lambda x: -len(x["thinking_text"]))
                except Exception:
                    pass

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        return self.flat_data[idx]


def build_math_wrong_dataset(file_path: str, tok: AutoTokenizer) -> MathWrongDataset:
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
    return MathWrongDataset(new_wrong_data, tok)
