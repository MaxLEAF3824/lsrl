import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import math
import argparse
import tokenize
import traceback
import gzip
import torch
import torch.nn.functional as F
import pandas as pd
import time
import asyncio
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from rouge_score import rouge_scorer
from math_verify import parse as math_verify_parse
from math_verify import verify as math_verify_verify
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
import random

# =========================================================================
# [0] 依赖引入 (确保 math_utils 和 vllm_workers 可用)
# =========================================================================
from math_utils import is_correct_v3, last_boxed_only_string, remove_boxed, is_equiv
from vllm_workers import VLLMDPWorkerPool


# =========================================================================
# [🌟 架构升级] 辅助函数：跨设备搬运优化器状态
# =========================================================================
def move_optimizer_state(optimizer, device):
    for param, state in optimizer.state.items():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


# =========================================================================
# [1] 数据集管理
# =========================================================================
class MathWrongDataset(Dataset):
    def __init__(self, raw_samples, tok):
        self.flat_data = []
        index = 0
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
                    # debug使用，观察长句子显存占用
                    # self.flat_data = sorted(
                    #     self.flat_data, key=lambda x: -len(x["answer_text"])
                    # )
                    index += 1
                except Exception:
                    pass

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        return self.flat_data[idx]


def build_math_wrong_dataset(file_path: str, tok: AutoTokenizer) -> MathWrongDataset:
    print(f"📄 读取文件: {file_path}")
    results = [json.loads(line) for line in open(file_path, "r", encoding="utf-8")]

    new_wrong_data = []
    for item in tqdm(results, desc="Processing Data"):
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


# =========================================================================
# [2] 优化器定义
# =========================================================================
class FrankWolfeOptimizer:
    def __init__(self, vocab_embeddings):
        self.W_emb = vocab_embeddings

    def step(self, latent_tensor, gamma):
        if latent_tensor.grad is None:
            return
        grad = latent_tensor.grad.to(self.W_emb.device).detach()
        with torch.no_grad():
            scores = torch.matmul(grad, self.W_emb.T)
            best_vocab_indices = torch.argmin(scores, dim=-1)
            best_embeds = self.W_emb[best_vocab_indices].to(latent_tensor.device)
            latent_tensor.copy_((1 - gamma) * latent_tensor + gamma * best_embeds)
            latent_tensor.grad.zero_()


# =========================================================================
# [3] 主流程
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--file_path",
        type=str,
        default="/workspace/yiqiuguo/lsrl/qwen3-1.7b_math-500_rollout8_len32768_final.jsonl",
    )
    parser.add_argument("--vllm_gpus", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--big_batch_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_k", type=int, default=32)
    parser.add_argument(
        "--mask_strategy",
        type=str,
        default="top_k_entropy",
        choices=["top_k_entropy", "first_k"],
    )
    parser.add_argument("--mask_max_k", type=int, default=32768)
    parser.add_argument(
        "--grad_direction",
        type=str,
        default="positive",
        choices=["positive", "negative"],
    )
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "frank_wolfe"])
    parser.add_argument("--fw_gamma", type=float, default=0.1)
    parser.add_argument("--conn_type", type=str, default="original", choices=["fast", "original"])
    parser.add_argument("--reg_type", type=str, default="lm", choices=["kl", "lm"])
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_threshold", type=float, default=1e-3)
    parser.add_argument("--skip_distill", action="store_true", help="是否跳过蒸馏阶段")
    parser.add_argument("--distill_epochs", type=int, default=3)
    parser.add_argument("--distill_lr", type=float, default=2e-5)
    parser.add_argument("--distill_ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--distill_eval_every", type=int, default=500, help="每多少步执行一次蒸馏评测")
    parser.add_argument(
        "--distill_eval_datasets",
        type=str,
        nargs="+",
        default=["HuggingFaceH4/MATH-500"],
    )
    parser.add_argument("--distill_batch_size", type=int, default=1)
    parser.add_argument("--distill_grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=9999999, help="限制处理的最大样本数量")
    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(project="L-GRPO-Math500", config=vars(args))
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    model.requires_grad_(False)
    model.gradient_checkpointing_disable()
    model.eval()

    vllm_engine = VLLMDPWorkerPool(model_name=args.model_name, gpu_ids=args.vllm_gpus)

    def get_embeds(text):
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        return model.get_input_embeddings()(ids).detach(), ids

    embeds_end_think, ids_end_think = get_embeds("</think>")
    embeds_fast_conn, ids_fast_conn = get_embeds("The final answer is \n&&\n\\boxed{")
    all_embeddings = model.get_input_embeddings().weight.detach()
    eval_device = torch.device("cuda:0") # 假设你评测在 cuda:0
    all_embeddings_norm_eval_gpu = F.normalize(all_embeddings.to(eval_device), dim=-1).to(torch.bfloat16)
    
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    wrong_dataset = build_math_wrong_dataset(args.file_path, tokenizer)
    all_raw_data = wrong_dataset.flat_data
    if args.max_samples is not None:
        all_raw_data = all_raw_data[: args.max_samples]

    # [🔥 核心改动] 拆分为外层 Big Batch
    raw_data_chunks = [all_raw_data[i : i + args.big_batch_size] for i in range(0, len(all_raw_data), args.big_batch_size)]

    global_history = {
        d["uid"]: {
            "uid": d["uid"],
            "problem": d["question_text"],
            "gt_text": d["gt_text"],
            "steps": [],
        }
        for d in all_raw_data
    }

    # 💥 全新设计：轻量级全局蒸馏数据集，仅保存 original_text, target_logits (top-100), new_correct_ids
    global_distill_dataset = []

    embeds_end_think_cpu = embeds_end_think.cpu()
    embeds_fast_conn_cpu = embeds_fast_conn.cpu()

    run_name = wandb.run.name if wandb.run is not None else f"opt_v2_{args.optimizer}_{args.reg_type}"
    history_filename = f"./optimization_histories/optimization_history_{run_name}.jsonl"
    os.makedirs(os.path.dirname(history_filename), exist_ok=True)

    def save_current_history():
        tmp_filename = history_filename + ".tmp"
        with open(tmp_filename, "w", encoding="utf-8") as f:
            f.write(json.dumps({"config": vars(args)}, ensure_ascii=False) + "\n")
            for u in global_history.keys():
                f.write(json.dumps(global_history[u], ensure_ascii=False) + "\n")
        os.replace(tmp_filename, history_filename)

    save_current_history()

    executor = ThreadPoolExecutor(max_workers=1)

    global_total_pure_acc, global_total_forced_acc, global_total_fast_acc = (0.0, 0.0, 0.0)
    global_total_rouge_l, global_total_change_ratio = 0.0, 0.0

    # ==================================================
    # [🚀 异步后台逻辑定义]
    # ==================================================
    def run_eval_async(eval_step, eval_uids, latents_snapshot, chunk_static_data, return_valid_answers=False):
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            eval_device_id = 0
            eval_device = torch.device(f"cuda:{eval_device_id}")

            tqdm.write(f"\n🚀 [后台评测] Step {eval_step} 启动! 正在 {eval_device} 计算潜空间漂移...")
            total_pure_acc, total_forced_acc, total_fast_acc = 0.0, 0.0, 0.0
            total_rouge_l, total_change_ratio = 0.0, 0.0

            eval_pure_inputs, eval_forced_inputs, eval_fast_inputs = [], [], []
            step_metrics = {uid: {} for uid in eval_uids}

            # [🔥 核心优化 1]：创建一个独立的 CUDA 流，让后台评测与主循环的 backward 彻底分离并行
            eval_stream = torch.cuda.Stream(device=eval_device)

            with torch.no_grad(), torch.cuda.stream(eval_stream):
                # 确保大词表 Normalize 在独立流和高效率数据类型下进行

                for uid in eval_uids:
                    ct_cpu = latents_snapshot[uid]
                    sd = chunk_static_data[uid]

                    target_flat = ct_cpu.squeeze(0)
                    L = target_flat.shape[0]

                    # [🔥 核心优化 2]：在 GPU 上预分配结果数组，避免循环内的 .cpu() 阻塞
                    nearest_token_ids_gpu = torch.empty(L, dtype=torch.long, device=eval_device)

                    # [🔥 核心优化 3]：回归正常的 chunk size，让矩阵大小保持在几十 MB 级别，最大化利用 L2 Cache
                    drift_chunk_size = 2048

                    for c_start in range(0, L, drift_chunk_size):
                        c_end = min(c_start + drift_chunk_size, L)
                        chunk_eval_gpu = target_flat[c_start:c_end].to(eval_device, non_blocking=True)
                        chunk_norm = F.normalize(chunk_eval_gpu, dim=-1).to(torch.bfloat16)

                        sim_chunk = torch.matmul(chunk_norm, all_embeddings_norm_eval_gpu.T)
                        nearest_token_ids_gpu[c_start:c_end] = torch.argmax(sim_chunk, dim=-1)

                        # 及时清理临时的大概率分布矩阵
                        del chunk_eval_gpu, chunk_norm, sim_chunk

                    # 整个序列算完后，一次性搬回 CPU（只需阻塞一次）
                    nearest_token_ids = nearest_token_ids_gpu.cpu()
                    del nearest_token_ids_gpu

                    orig_ids = sd["ids_think"].squeeze(0)
                    changed_mask = nearest_token_ids != orig_ids
                    change_ratio = changed_mask.float().mean().item()

                    decoded_nearest_text = tokenizer.decode(nearest_token_ids).strip()
                    original_thinking_clean = sd["thinking_text"]
                    rouge_l_f1 = scorer.score(original_thinking_clean[:10000], decoded_nearest_text[:10000])["rougeL"].fmeasure

                    step_metrics[uid]["change_ratio"] = change_ratio
                    step_metrics[uid]["rouge_L"] = rouge_l_f1

                    total_change_ratio += change_ratio
                    total_rouge_l += rouge_l_f1

                    embeds_q_cpu = sd["embeds_q_cpu"]
                    embeds_conn_cpu = sd["embeds_conn_cpu"]

                    eval_pure_inputs.append(torch.cat([embeds_q_cpu, ct_cpu, embeds_end_think_cpu], dim=1).squeeze(0))
                    eval_forced_inputs.append(
                        torch.cat(
                            [embeds_q_cpu, ct_cpu, embeds_end_think_cpu, embeds_conn_cpu],
                            dim=1,
                        ).squeeze(0)
                    )
                    eval_fast_inputs.append(
                        torch.cat(
                            [
                                embeds_q_cpu,
                                ct_cpu,
                                embeds_end_think_cpu,
                                embeds_fast_conn_cpu,
                            ],
                            dim=1,
                        ).squeeze(0)
                    )


            # [🔥 核心优化 4]：强制同步刚才的流，确保数据计算真实落盘再往下走
            eval_stream.synchronize()

            with torch.cuda.device(eval_device):
                torch.cuda.empty_cache()

            tqdm.write(f"⚡ [后台评测] 漂移计算完毕，准备发送数据至 vLLM Worker...")
            modes = [
                ("pure", eval_pure_inputs, 2048),
                ("forced", eval_forced_inputs, 128),
                ("fast", eval_fast_inputs, 128),
            ]

            valid_answers = {}  # 新增：用于存储正确答案的字典

            for mode, inputs_list, max_toks in modes:
                tqdm.write(f"   --> vLLM 正在并行生成 {mode} 模式...")
                batch_outputs = vllm_engine.generate(
                    inputs_list,
                    {
                        "max_tokens": max_toks,
                        "temperature": 0.7,
                        "n": args.eval_k,
                        "skip_special_tokens": False,
                    },
                )
                if not batch_outputs:
                    continue

                for idx, uid in enumerate(eval_uids):
                    gt_text = global_history[uid]["gt_text"]
                    gt_token_len = len(tokenizer.encode(gt_text, add_special_tokens=False))

                    correct_count = 0
                    for output_ids in batch_outputs[idx]:
                        if mode != "pure":
                            output_ids = output_ids[:gt_token_len]
                        gen_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                        is_corr = False
                        if mode == "pure":
                            if is_correct_v3(gen_text, gt_text.replace("}", "")):
                                is_corr = True
                                if return_valid_answers and uid not in valid_answers:
                                    valid_answers[uid] = output_ids
                        else:
                            ans = gen_text.replace("$", "").replace("}", "").strip()
                            if is_equiv(ans, gt_text.replace("}", "")):
                                is_corr = True
                        if is_corr:
                            correct_count += 1

                    acc = correct_count / args.eval_k
                    step_metrics[uid][f"{mode}_acc"] = acc
                    if mode == "pure" and len(batch_outputs[idx]) > 0:
                        sample_gen = tokenizer.decode(batch_outputs[idx][0], skip_special_tokens=True)
                        step_metrics[uid]["sample_gen_text"] = sample_gen

                    if mode == "pure":
                        total_pure_acc += acc
                    elif mode == "forced":
                        total_forced_acc += acc
                    elif mode == "fast":
                        total_fast_acc += acc

            tqdm.write(
                f"\n✅ [后台评测完成 | Step {eval_step}] Pure: {total_pure_acc / len(eval_uids):.2%} | Forced: {total_forced_acc / len(eval_uids):.2%} | Fast: {total_fast_acc / len(eval_uids):.2%}"
            )

            res = {
                "step": eval_step,
                "uids": eval_uids,
                "step_metrics": step_metrics,
                "avg_metrics": {
                    "eval/avg_pure_acc": total_pure_acc / len(eval_uids),
                    "eval/avg_forced_acc": total_forced_acc / len(eval_uids),
                    "eval/avg_fast_acc": total_fast_acc / len(eval_uids),
                    "eval/avg_rouge_L": total_rouge_l / len(eval_uids),
                    "eval/avg_change_ratio": total_change_ratio / len(eval_uids),
                },
            }
            if return_valid_answers:
                res["valid_answers"] = valid_answers
            return res
        except Exception as e:
            tqdm.write(f"\n❌ [后台评测严重崩溃] {str(e)}\n{traceback.format_exc()}")
            return None

    def process_eval_results(res):
        if res is None:
            return
        eval_step, eval_uids, eval_metrics = (
            res["step"],
            res["uids"],
            res["step_metrics"],
        )

        wandb.log({"global_step": eval_step, **res["avg_metrics"]})

        for uid in eval_uids:
            for step_record in global_history[uid]["steps"]:
                if step_record["step"] == eval_step:
                    step_record["metrics"].update(eval_metrics[uid])
                    break
        save_current_history()

    # =========================================================================
    # [🔥 外层大循环] Big Batch 迭代控制内存占用
    # =========================================================================
    big_batch_pbar = tqdm(
        enumerate(raw_data_chunks),
        total=len(raw_data_chunks),
        desc="🌍 Big Batch 总进度",
        position=0,
        leave=True,
    )
    for chunk_idx, current_raw_data in big_batch_pbar:
        static_data_cpu = {}
        global_latents = torch.nn.ParameterDict()
        active_uids = []
        eval_future = None

        # --- Phase 1: Pre-computing ---
        tqdm.write(f"⚙️ Preparing Static Data & Latents for Chunk {chunk_idx + 1}...")
        prep_batch_size = args.batch_size
        for i in tqdm(
            range(0, len(current_raw_data), prep_batch_size),
            desc=f"Pre-computing Chunk {chunk_idx + 1}",
            position=1,
            leave=False,
        ):
            batch_samples = current_raw_data[i : i + prep_batch_size]
            batch_embeds_full, batch_info = [], []

            for d in batch_samples:
                uid = d["uid"]
                active_uids.append(uid)

                embeds_q, ids_q = get_embeds(d["question_text"])
                embeds_conn, ids_conn = get_embeds(d["connector_text"]) if args.conn_type == "original" else (embeds_fast_conn, ids_fast_conn)
                embeds_gt, ids_gt = get_embeds(d["gt_text"])
                embeds_pred, ids_pred = get_embeds(d["pred_text"])
                embeds_think, ids_think = get_embeds(d["thinking_text"])

                curr_think = torch.nn.Parameter(embeds_think.detach().cpu().clone())
                global_latents[uid] = curr_think

                think_start_idx = ids_q.shape[1] - 1
                think_end_idx = think_start_idx + ids_think.shape[1]
                full_emb = torch.cat([embeds_q, embeds_think, embeds_end_think, embeds_conn], dim=1)
                batch_embeds_full.append(full_emb)

                batch_info.append(
                    {
                        "uid": uid,
                        "ids_q": ids_q,
                        "ids_conn": ids_conn,
                        "ids_gt": ids_gt,
                        "ids_pred": ids_pred,
                        "ids_think": ids_think,
                        "thinking_text": d["thinking_text"],
                        "start": think_start_idx,
                        "end": think_end_idx,
                        "len": full_emb.shape[1],
                        "embeds_q": embeds_q,
                        "embeds_conn": embeds_conn,
                    }
                )

            max_len = max(info["len"] for info in batch_info)
            padded_embeds, attn_masks = [], []

            for j, emb in enumerate(batch_embeds_full):
                diff = max_len - emb.shape[1]
                if diff > 0:
                    pad = torch.zeros((1, diff, emb.shape[2]), device=device, dtype=model.dtype)
                    padded_embeds.append(torch.cat([emb, pad], dim=1))
                    mask = torch.cat([torch.ones(emb.shape[1]), torch.zeros(diff)]).to(device)
                else:
                    padded_embeds.append(emb)
                    mask = torch.ones(max_len).to(device)
                attn_masks.append(mask.unsqueeze(0))

            with torch.no_grad():
                outputs = model(
                    inputs_embeds=torch.cat(padded_embeds, dim=0),
                    attention_mask=torch.cat(attn_masks, dim=0),
                )
                all_logits = outputs.logits

            for j, info in enumerate(batch_info):
                uid = info["uid"]
                think_logits = all_logits[j, info["start"] : info["end"], :].unsqueeze(0)
                probs = F.softmax(think_logits, dim=-1)

                if args.mask_strategy == "top_k_entropy":
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    mask = torch.zeros_like(entropy, dtype=torch.float32)
                    mask.scatter_(
                        1,
                        torch.topk(entropy, k=min(args.mask_max_k, entropy.shape[1]), dim=1).indices,
                        1.0,
                    )
                else:
                    mask = torch.zeros(
                        (1, info["ids_think"].shape[1]),
                        dtype=torch.float32,
                        device=device,
                    )
                    mask[:, : args.mask_max_k] = 1.0

                topk_probs, topk_indices = torch.topk(probs, k=100, dim=-1)

                static_data_cpu[uid] = {
                    "ids_q": info["ids_q"].cpu(),
                    "ids_conn": info["ids_conn"].cpu(),
                    "ids_gt": info["ids_gt"].cpu(),
                    "ids_pred": info["ids_pred"].cpu(),
                    "ids_think": info["ids_think"].cpu(),
                    "grad_mask": mask.unsqueeze(-1).cpu(),
                    "topk_probs": topk_probs.cpu(),
                    "topk_indices": topk_indices.cpu(),
                    "thinking_text": info["thinking_text"],
                    "embeds_q_cpu": info["embeds_q"].cpu(),
                    "embeds_conn_cpu": info["embeds_conn"].cpu(),
                }
            del all_logits, outputs, padded_embeds, attn_masks, batch_embeds_full
        torch.cuda.empty_cache()

        # --- init optimizer ----
        global_opts = {}
        if args.optimizer == "adam":
            for uid, param in global_latents.items():
                global_opts[uid] = torch.optim.Adam([param], lr=args.learning_rate)
        elif args.optimizer == "frank_wolfe":
            optimizer = FrankWolfeOptimizer(all_embeddings)

        # --- Phase 2: 优化循环 ---
        step_pbar = tqdm(
            range(args.steps),
            desc=f"🏃 Chunk {chunk_idx + 1} Steps",
            position=1,
            leave=False,
        )
        for step in step_pbar:
            global_step_id = chunk_idx * args.steps + step

            if eval_future is not None and eval_future.done():
                process_eval_results(eval_future.result())
                eval_future = None

            if not active_uids:
                print(f"🎉 Chunk {chunk_idx + 1} 的所有样本均已触发 Early Stop！")
                break

            progress = step / max(1, args.steps - 1)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_lr = (args.learning_rate * 0.1) + (args.learning_rate * 0.9) * cosine_factor
            current_fw_gamma = (args.fw_gamma * 0.1) + (args.fw_gamma * 0.9) * cosine_factor

            step_metrics = {uid: {} for uid in active_uids}
            (
                epoch_gt_loss_sum,
                epoch_kl_loss_sum,
                epoch_lm_loss_sum,
                epoch_total_loss_sum,
            ) = 0.0, 0.0, 0.0, 0.0

            mini_batches = [active_uids[i : i + args.batch_size] for i in range(0, len(active_uids), args.batch_size)]

            # [修改点 3]：修改第三层 Mini Batch 进度条 (position=2, leave=False 跑完一个 step 后自动清理)
            inner_pbar = tqdm(
                mini_batches,
                desc=f"🔥 Mini Batches (Step {step})",
                position=2,
                leave=False,
            )

            for batch_uids in inner_pbar:
                curr_think_list, curr_mask_list = [], []
                embeds_q_list, embeds_conn_list, embeds_gt_list, embeds_pred_list = (
                    [],
                    [],
                    [],
                    [],
                )
                (
                    ids_q_list,
                    ids_think_list,
                    ids_conn_list,
                    ids_gt_list,
                    ids_pred_list,
                ) = [], [], [], [], []

                for uid in batch_uids:
                    sd = static_data_cpu[uid]
                    curr_mask_list.append(sd["grad_mask"].to(device))

                    p = global_latents[uid]
                    p.data = p.data.to(device)
                    if p.grad is not None:
                        p.grad.data = p.grad.data.to(device)
                    curr_think_list.append(p)

                    if args.optimizer == "adam":
                        opt = global_opts[uid]
                        move_optimizer_state(opt, device)
                        for pg in opt.param_groups:
                            pg["lr"] = current_lr
                    elif args.optimizer == "frank_wolfe":
                        if p.grad is not None:
                            p.grad.zero_()

                    with torch.no_grad():
                        emb_q = model.get_input_embeddings()(sd["ids_q"].to(device))
                        emb_conn = model.get_input_embeddings()(sd["ids_conn"].to(device))
                        emb_gt = model.get_input_embeddings()(sd["ids_gt"].to(device))
                        emb_pred = model.get_input_embeddings()(sd["ids_pred"].to(device))

                    embeds_q_list.append(emb_q)
                    embeds_conn_list.append(emb_conn)
                    embeds_gt_list.append(emb_gt)
                    embeds_pred_list.append(emb_pred)

                    ids_q_list.append(sd["ids_q"].to(device))
                    ids_think_list.append(sd["ids_think"].to(device))
                    ids_conn_list.append(sd["ids_conn"].to(device))
                    ids_gt_list.append(sd["ids_gt"].to(device))
                    ids_pred_list.append(sd["ids_pred"].to(device))

                full_embeds_list = []
                for i in range(len(batch_uids)):
                    target_emb = embeds_gt_list[i] if args.grad_direction == "positive" else embeds_pred_list[i]
                    full_embeds_list.append(
                        torch.cat(
                            [
                                embeds_q_list[i],
                                curr_think_list[i].to(model.dtype),
                                embeds_end_think,
                                embeds_conn_list[i],
                                target_emb,
                            ],
                            dim=1,
                        )
                    )

                max_len = max(emb.shape[1] for emb in full_embeds_list)
                attention_mask_list = []
                for i, emb in enumerate(full_embeds_list):
                    if emb.shape[1] < max_len:
                        pad_emb = torch.zeros(
                            (emb.shape[0], max_len - emb.shape[1], emb.shape[2]),
                            device=device,
                            dtype=model.dtype,
                        )
                        full_embeds_list[i] = torch.cat([emb, pad_emb], dim=1)
                        attn = torch.zeros((emb.shape[0], max_len), device=device, dtype=torch.long)
                        attn[:, : emb.shape[1]] = 1
                        attention_mask_list.append(attn)
                    else:
                        attention_mask_list.append(torch.ones((emb.shape[0], max_len), device=device, dtype=torch.long))

                full_embeds_batch = torch.cat(full_embeds_list, dim=0)
                full_attention_mask = torch.cat(attention_mask_list, dim=0)

                # [1] 前向计算得到全局 last_hidden，并切断计算图
                last_hidden = model.model(inputs_embeds=full_embeds_batch, attention_mask=full_attention_mask).last_hidden_state

                # 设置为叶子节点，用于在 Chunk 循环中临时收集梯度
                last_hidden_detached = last_hidden.detach().requires_grad_(True)

                batch_size_cur = len(batch_uids)
                epoch_gt_loss_sum_temp = 0.0
                epoch_kl_loss_sum_temp = 0.0
                epoch_lm_loss_sum_temp = 0.0
                epoch_total_loss_sum_temp = 0.0

                for i, uid in enumerate(batch_uids):
                    sd = static_data_cpu[uid]
                    gt_pos = ids_q_list[i].shape[1] + ids_think_list[i].shape[1] + ids_end_think.shape[1] + ids_conn_list[i].shape[1] - 1
                    target_ids = ids_gt_list[i] if args.grad_direction == "positive" else ids_pred_list[i]

                    # === 1. 计算 GT Loss 并立即反向传播 ===
                    target_logits = model.lm_head(last_hidden_detached[[i], gt_pos : gt_pos + target_ids.shape[1], :])
                    gt_loss = F.cross_entropy(
                        target_logits.view(-1, target_logits.size(-1)),
                        target_ids.view(-1),
                    )
                    if args.grad_direction == "negative":
                        gt_loss = -gt_loss

                    # 除以 batch_size_cur 对应你原逻辑中的 torch.stack().mean()
                    if step > 0:
                        (gt_loss / batch_size_cur).backward()

                    # === 2. 准备 Chunking ===
                    think_start = ids_q_list[i].shape[1] - 1
                    think_end = think_start + ids_think_list[i].shape[1]
                    think_len = think_end - think_start

                    kl_sum_val, logprobs_sum_val = 0.0, 0.0
                    orig_probs_topk = sd["topk_probs"].to(device)
                    orig_indices_topk = sd["topk_indices"].to(device)

                    # target_ids_think 只是普通的 LongTensor，不参与梯度计算，放外面完全安全
                    target_ids_think = ids_think_list[i]

                    # === 3. Chunk 循环 ===
                    for c_start in range(0, think_len, args.chunk_size):
                        c_end = min(c_start + args.chunk_size, think_len)

                        # [🔥 修复点 1] 直接从叶子节点 last_hidden_detached 动态切片，确保图不被共享
                        h_chunk = last_hidden_detached[[i], think_start + c_start : think_start + c_end, :]

                        logits_chunk = model.lm_head(h_chunk)
                        lse_chunk = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)

                        orig_p_chunk = orig_probs_topk[:, c_start:c_end, :]
                        orig_idx_chunk = orig_indices_topk[:, c_start:c_end, :]
                        curr_log_probs_topk_chunk = torch.gather(logits_chunk, -1, orig_idx_chunk) - lse_chunk
                        kl_chunk = (orig_p_chunk * (torch.log(orig_p_chunk + 1e-10) - curr_log_probs_topk_chunk)).sum(dim=-1)

                        # [🔥 修复点 2] 放弃循环外的 torch.cat，直接从叶子节点 curr_think_list[i] 动态切片取所需片段
                        embeds_chunk = curr_think_list[i][:, c_start : c_end, :]

                        target_ids_chunk = target_ids_think[:, c_start:c_end]
                        orig_embeds_chunk = model.get_input_embeddings()(target_ids_chunk).detach()

                        orig_norm = torch.norm(orig_embeds_chunk, p=2, dim=-1, keepdim=True)
                        curr_norm = torch.norm(embeds_chunk, p=2, dim=-1, keepdim=True)
                        scaled_embeds_chunk = embeds_chunk * (orig_norm / (curr_norm + 1e-8))

                        score_0_chunk = (h_chunk * scaled_embeds_chunk).sum(dim=-1)
                        mask_chunk = torch.zeros_like(logits_chunk, dtype=torch.bool)
                        mask_chunk.scatter_(2, target_ids_chunk.unsqueeze(-1), True)

                        new_logits_chunk = torch.where(mask_chunk, score_0_chunk.unsqueeze(-1), logits_chunk)

                        logprobs_chunk_val = (score_0_chunk - torch.logsumexp(new_logits_chunk, dim=-1)).sum()

                        kl_sum_val += kl_chunk.sum().item()
                        logprobs_sum_val += logprobs_chunk_val.item()

                        # 【单 Chunk 即时反向传播】
                        if step > 0:
                            s_kl_chunk = kl_chunk.sum() / think_len
                            s_lm_chunk = -logprobs_chunk_val / think_len
                            s_reg_chunk = s_kl_chunk if args.reg_type == "kl" else s_lm_chunk
                            chunk_loss = (args.kl_weight * s_reg_chunk) / batch_size_cur

                            chunk_loss.backward()

                        # 强制销毁当前 Chunk 的大显存占用变量
                        del h_chunk, logits_chunk, lse_chunk, curr_log_probs_topk_chunk
                        del new_logits_chunk, mask_chunk, score_0_chunk, kl_chunk, logprobs_chunk_val
                        del embeds_chunk, scaled_embeds_chunk

                    # === 4. 汇总单个样本的 Loss 指标（仅供记录，不计入新图）===
                    s_gt = gt_loss.item()
                    s_kl = kl_sum_val / think_len
                    s_lm = -(logprobs_sum_val / think_len)
                    s_reg = s_kl if args.reg_type == "kl" else s_lm
                    s_total = s_gt + args.kl_weight * s_reg

                    step_metrics[uid] = {
                        "total_loss": s_total,
                        "gt_loss": s_gt,
                        "kl_loss": s_kl,
                        "lm_loss": s_lm,
                    }

                    epoch_gt_loss_sum_temp += s_gt
                    epoch_kl_loss_sum_temp += s_kl
                    epoch_lm_loss_sum_temp += s_lm
                    epoch_total_loss_sum_temp += s_total

                # === 5. 更新全局累计变量和进度条 ===
                epoch_gt_loss_sum += epoch_gt_loss_sum_temp
                epoch_kl_loss_sum += epoch_kl_loss_sum_temp
                epoch_lm_loss_sum += epoch_lm_loss_sum_temp
                epoch_total_loss_sum += epoch_total_loss_sum_temp

                inner_pbar.set_postfix(
                    {
                        "Total": f"{epoch_total_loss_sum_temp / batch_size_cur:.3f}",
                        "GT": f"{epoch_gt_loss_sum_temp / batch_size_cur:.3f}",
                    }
                )

                # === 6. 最终将全局梯度传递回模型 ===
                if step > 0:
                    last_hidden.backward(last_hidden_detached.grad)

                for i, uid in enumerate(batch_uids):
                    p = curr_think_list[i]
                    mask = curr_mask_list[i]
                    if p.grad is not None:
                        p.grad.data.mul_(mask.to(device).to(p.dtype))
                    if args.optimizer == "adam":
                        opt = global_opts[uid]
                        if step > 0:
                            opt.step()
                        opt.zero_grad(set_to_none=True)
                        move_optimizer_state(opt, torch.device("cpu"))
                    elif args.optimizer == "frank_wolfe":
                        if step > 0:
                            optimizer.step(p, gamma=current_fw_gamma)

                    p.data = p.data.cpu().pin_memory()

                del full_embeds_batch, full_attention_mask, last_hidden

                time.sleep(0.005)

            wandb.log(
                {
                    "global_step": global_step_id,
                    "train/epoch_total_loss": epoch_total_loss_sum / len(active_uids),
                    "train/epoch_gt_loss": epoch_gt_loss_sum / len(active_uids),
                    "train/lr": current_lr,
                    "active_samples": len(active_uids),
                }
            )

            next_active_uids = []
            for uid in active_uids:
                global_history[uid]["steps"].append({"step": global_step_id, "metrics": step_metrics[uid]})
                gt_loss = step_metrics[uid]["gt_loss"]
                if args.early_stop and gt_loss < args.early_stop_threshold:
                    pass
                else:
                    next_active_uids.append(uid)
            active_uids = next_active_uids

            if step % args.eval_every == 0 and step != args.steps - 1:
                if eval_future is not None:
                    process_eval_results(eval_future.result())
                    eval_future = None

                eval_uids = list(active_uids)
                latents_snapshot = {uid: global_latents[uid].detach().clone() for uid in eval_uids}
                eval_future = executor.submit(
                    run_eval_async,
                    global_step_id,
                    eval_uids,
                    latents_snapshot,
                    static_data_cpu,
                )

            if step % args.eval_every == 0 or step == args.steps - 1:
                save_current_history()

        if eval_future is not None:
            print("\n⏳ 训练步骤完成，等待后台收尾最后一轮 Eval...")
            process_eval_results(eval_future.result())

        # =========================================================================
        # [Phase 3: Final Eval & 数据提取清理]
        # =========================================================================
        print(f"\n🎉 开始 Chunk {chunk_idx + 1} 的 Final Evaluation ...")
        chunk_all_uids = [d["uid"] for d in current_raw_data]

        final_latents_cpu = {uid: global_latents[uid].detach().cpu() for uid in chunk_all_uids}

        final_eval_result = run_eval_async(
            eval_step=global_step_id, 
            eval_uids=chunk_all_uids, 
            latents_snapshot=final_latents_cpu, 
            chunk_static_data=static_data_cpu,
            return_valid_answers=True
        )
        process_eval_results(final_eval_result)
        
        # 累加全局统计指标供最后打印
        avg_metrics = final_eval_result["avg_metrics"]
        num_chunk = len(chunk_all_uids)
        global_total_pure_acc += avg_metrics["eval/avg_pure_acc"] * num_chunk
        global_total_forced_acc += avg_metrics["eval/avg_forced_acc"] * num_chunk
        global_total_fast_acc += avg_metrics["eval/avg_fast_acc"] * num_chunk
        global_total_rouge_l += avg_metrics["eval/avg_rouge_L"] * num_chunk
        global_total_change_ratio += avg_metrics["eval/avg_change_ratio"] * num_chunk

        # 拿出我们需要的金标答案
        valid_answers = final_eval_result.get("valid_answers", {})

        print(f"🎯 正在为 Chunk {chunk_idx + 1} 提取 Target Logits... (筛选出有效样本: {len(valid_answers)})")
        model.eval()
        with torch.no_grad():
            for uid, best_output_ids in valid_answers.items():
                sd = static_data_cpu[uid]
                ct = final_latents_cpu[uid].to(device).to(model.dtype)
                emb_q = sd["embeds_q_cpu"].to(device)

                # Forward 计算原 teacher 模型的 target logits
                full_emb = torch.cat([emb_q, ct], dim=1)
                logits = model(inputs_embeds=full_emb).logits

                think_start = emb_q.shape[1] - 1
                think_end = full_emb.shape[1] - 1
                think_logits = logits[0, think_start:think_end, :]

                probs = F.softmax(think_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=100, dim=-1)

                # 转换 successful generation 回 tensor (作为蒸馏的 CE 标签)
                ids_new_ans = torch.tensor([best_output_ids], dtype=torch.long)

                # 直接保存最精简的信息进入全局蒸馏池
                global_distill_dataset.append(
                    {
                        "uid": uid,
                        "ids_q": sd["ids_q"].cpu(),
                        "ids_orig_think": sd["ids_think"].cpu(),
                        "target_probs": topk_probs.cpu(),
                        "target_indices": topk_indices.cpu(),
                        "ids_new_ans": ids_new_ans.cpu(),
                        "gt_text": global_history[uid]["gt_text"],
                        "problem": global_history[uid]["problem"],
                    }
                )

        # 🚀 内存回收
        del static_data_cpu, global_latents, final_latents_cpu
        if args.optimizer == "adam":
            del global_opts
        torch.cuda.empty_cache()
        save_current_history()

    # ==================================================
    # Phase 4: Big Batch 循环彻底结束，全局统计打印
    # ==================================================
    total_samples = len(all_raw_data)
    print("\n" + "=" * 60)
    print(
        f"📊 [Global Final Results] Pure Acc: {global_total_pure_acc / total_samples:.2%} | Forced Acc: {global_total_forced_acc / total_samples:.2%} | Fast Acc: {global_total_fast_acc / total_samples:.2%}"
    )
    print(f"📉 [Global Final Latent Drift] ROUGE-L: {global_total_rouge_l / total_samples:.4f} | Change Ratio: {global_total_change_ratio / total_samples:.2%}")

    wandb.log(
        {
            "global_step": len(raw_data_chunks) * args.steps,
            "final/avg_pure_acc": global_total_pure_acc / total_samples,
            "final/avg_forced_acc": global_total_forced_acc / total_samples,
            "final/avg_fast_acc": global_total_fast_acc / total_samples,
            "final/avg_rouge_L": global_total_rouge_l / total_samples,
            "final/avg_change_ratio": global_total_change_ratio / total_samples,
        }
    )

    # =========================================================================
    # Phase 5: 模型蒸馏阶段 (Distillation)
    # =========================================================================
    print("\n" + "=" * 60)
    if args.skip_distill:
        print("⏭️  根据参数 --skip_distill，跳过模型蒸馏阶段。")
        wandb.finish()
        vllm_engine.close()
        return
    print("🔥 Phase 5: 启动模型蒸馏阶段...")
    print("=" * 60)

    wandb.define_metric("model_train_step")
    wandb.define_metric("distill/train/*", step_metric="model_train_step")
    wandb.define_metric("distill/eval/*", step_metric="model_train_step")

    distill_dataset = global_distill_dataset
    print(f"📦 蒸馏数据集就绪，有效过滤后样本数: {len(distill_dataset)} / {total_samples}")

    if len(distill_dataset) == 0:
        print("⚠️ 没有符合条件的样本进行蒸馏，流程结束。")
        wandb.finish()
        vllm_engine.close()
        return

    # 准备异步评测环境
    def pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

    def load_eval_datasets(dataset_names):
        datasets_dict = {}
        for ds_id in dataset_names:
            try:
                if "MATH-500" in ds_id:
                    ds = load_dataset(ds_id, split="test")
                    q_key, a_key = "problem", "answer"
                elif "amc23" in ds_id:
                    ds = load_dataset(ds_id, split="test")
                    q_key, a_key = "question", "answer"
                elif "gsm8k" in ds_id:
                    ds = load_dataset(ds_id, "main", split="test")
                    q_key, a_key = "question", "answer"
                elif "aime_2024" in ds_id:
                    ds = load_dataset(ds_id, split="train")
                    q_key, a_key = "problem", "answer"
                elif "aime25" in ds_id:
                    ds = load_dataset(ds_id, split="test")
                    q_key, a_key = "problem", "answer"
                else:
                    print(f"⚠️ 未知的评测集: {ds_id}，跳过")
                    continue
                datasets_dict[ds_id] = {"data": ds, "q_key": q_key, "a_key": a_key}
                print(f"✅ 成功加载评测集: {ds_id} (样本数: {len(ds)})")
            except Exception as e:
                print(f"❌ 加载评测集 {ds_id} 失败: {e}")
        return datasets_dict

    eval_datasets = load_eval_datasets(args.distill_eval_datasets)
    train_eval_data = []
    for item in distill_dataset:
        train_eval_data.append(
            {
                "problem_formatted": item["problem"],
                "answer_gt": item["gt_text"].replace("}", ""),
            }
        )

    eval_datasets["train_dataset"] = {
        "data": train_eval_data,
        "q_key": "problem_formatted",
        "a_key": "answer_gt",
        "pre_formatted": True,
    }
    print(f"✅ 成功将训练集加入评测: train_dataset (样本数: {len(train_eval_data)})")

    distill_eval_future = None
    distill_step = 0

    def run_distill_eval_async(step, current_state_dict_path):
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            tqdm.write(f"\n🚀 [后台蒸馏评测] 正在同步权重至 vLM...")
            vllm_engine.update_weights(current_state_dict_path)

            metrics_to_log = {"model_train_step": step}
            total_pass1, total_pass8, total_pass16, total_pass32 = [], [], [], []

            for ds_id, ds_info in tqdm(
                eval_datasets.items(),
                desc="🌐 蒸馏评测总体进度",
                leave=False,
                position=1,
                dynamic_ncols=True,
            ):
                ds, q_key, a_key = ds_info["data"], ds_info["q_key"], ds_info["a_key"]
                is_pre_formatted = ds_info.get("pre_formatted", False)

                current_n = 8
                if "MATH-500" in ds_id or "train_dataset" in ds_id:
                    current_n = 8
                elif any(x in ds_id for x in ["aime_2024", "aime25", "amc23"]):
                    current_n = 32

                sp_dict = {
                    "max_tokens": 32768,
                    "temperature": 0.7,
                    "n": current_n,
                    "skip_special_tokens": False,
                }

                prompts = []
                for item in ds:
                    if is_pre_formatted:
                        prompts.append(item[q_key])
                    else:
                        msg = [{"role": "user", "content": item[q_key]}]
                        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

                tqdm.write(f"   --> vLLM 正在生成 {ds_id} (并发 {current_n})...")
                batch_outputs = vllm_engine.generate(prompts, sp_dict, input_type="texts")

                ds_pass1, ds_pass8, ds_pass16, ds_pass32 = 0.0, 0.0, 0.0, 0.0

                for idx, item in enumerate(
                    tqdm(
                        ds,
                        desc=f"📊 正在校验 {ds_id} 答案",
                        leave=False,
                        position=2,
                        dynamic_ncols=True,
                    )
                ):
                    gt_ans = str(item[a_key]).strip()
                    correct_count = 0

                    for output_ids in batch_outputs[idx]:
                        gen_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                        ans_cand = last_boxed_only_string(gen_text)
                        if ans_cand and is_equiv(remove_boxed(ans_cand), gt_ans):
                            correct_count += 1

                    ds_pass1 += pass_at_k(current_n, correct_count, 1)
                    if current_n >= 8:
                        ds_pass8 += pass_at_k(current_n, correct_count, 8)
                    if current_n >= 16:
                        ds_pass16 += pass_at_k(current_n, correct_count, 16)
                    if current_n >= 32:
                        ds_pass32 += pass_at_k(current_n, correct_count, 32)

                num_samples = len(ds)
                metrics_to_log[f"distill/eval/{ds_id}/pass@1"] = ds_pass1 / num_samples
                total_pass1.append(ds_pass1 / num_samples)

                if current_n >= 8:
                    metrics_to_log[f"distill/eval/{ds_id}/pass@8"] = ds_pass8 / num_samples
                    total_pass8.append(ds_pass8 / num_samples)
                if current_n >= 16:
                    metrics_to_log[f"distill/eval/{ds_id}/pass@16"] = ds_pass16 / num_samples
                    total_pass16.append(ds_pass16 / num_samples)
                if current_n >= 32:
                    metrics_to_log[f"distill/eval/{ds_id}/pass@32"] = ds_pass32 / num_samples
                    total_pass32.append(ds_pass32 / num_samples)

            if total_pass1:
                metrics_to_log["distill/eval/avg_pass@1"] = sum(total_pass1) / len(total_pass1)
            if total_pass8:
                metrics_to_log["distill/eval/avg_pass@8"] = sum(total_pass8) / len(total_pass8)
            if total_pass16:
                metrics_to_log["distill/eval/avg_pass@16"] = sum(total_pass16) / len(total_pass16)
            if total_pass32:
                metrics_to_log["distill/eval/avg_pass@32"] = sum(total_pass32) / len(total_pass32)

            tqdm.write(f"✅ [后台蒸馏评测完成] 均值 Pass@1: {metrics_to_log.get('distill/eval/avg_pass@1', 0):.2%}")
            return metrics_to_log

        except Exception as e:
            tqdm.write(f"\n❌ [蒸馏评测严重崩溃] {str(e)}\n{traceback.format_exc()}")
            return None

    # 5.4 开启模型训练状态
    model.requires_grad_(True)
    model.train()
    model.gradient_checkpointing_enable()
    distill_optimizer = torch.optim.AdamW(model.parameters(), lr=args.distill_lr)

    distill_history_file = f"./optimization_histories/distill_history_{run_name}.jsonl"
    os.makedirs(os.path.dirname(distill_history_file), exist_ok=True)

    save_dir = f"./checkpoints/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    shm_model_path = "/dev/shm/distill_model_weights.pt"

    print(f"🏃 开启蒸馏训练，共 {args.distill_epochs} Epochs...")

    for epoch in range(args.distill_epochs):

        random.shuffle(distill_dataset)

        batch_size = args.distill_batch_size
        accum_steps = args.distill_grad_accum_steps
        mini_batches = [distill_dataset[i : i + batch_size] for i in range(0, len(distill_dataset), batch_size)]
        pbar = tqdm(mini_batches, desc=f"Distill Epoch {epoch + 1}/{args.distill_epochs}")

        distill_optimizer.zero_grad()

        for b_idx, batch in enumerate(pbar):
            batch_loss, batch_kl, batch_ce = 0.0, 0.0, 0.0

            for item in batch:
                uid = item["uid"]
                ids_q = item["ids_q"].to(device)
                ids_think = item["ids_orig_think"].to(device)  # 蒸馏使用的是 ORIGINAL Think IDs!
                ids_new_ans = item["ids_new_ans"].to(device)  # 新提取的正确生成的回答

                target_probs = item["target_probs"].to(device)  # [seq_len, 100]
                target_indices = item["target_indices"].to(device)

                # 构建完整的输入序列: Q + Original Think + EndThink + New Answer
                full_ids = torch.cat([ids_q, ids_think, ids_end_think.to(device), ids_new_ans], dim=1)
                outputs = model(full_ids)
                logits = outputs.logits

                # === KL Loss (Think Portion) ===
                think_start = ids_q.shape[1] - 1
                think_end = think_start + target_probs.shape[0]
                pred_think_logits = logits[0, think_start:think_end, :]

                pred_lse = torch.logsumexp(pred_think_logits, dim=-1, keepdim=True)
                pred_log_probs_topk = torch.gather(pred_think_logits, -1, target_indices) - pred_lse

                kl_elementwise = target_probs * (torch.log(target_probs + 1e-10) - pred_log_probs_topk)
                kl_loss = kl_elementwise.sum(dim=-1).mean()

                # === CE Loss (Answer Portion) ===
                ans_start = full_ids.shape[1] - ids_new_ans.shape[1] - 1
                ans_end = full_ids.shape[1] - 1
                pred_ans_logits = logits[0, ans_start:ans_end, :]

                ce_loss = F.cross_entropy(pred_ans_logits, ids_new_ans[0])
                total_loss = kl_loss + args.distill_ce_loss_weight * ce_loss

                (total_loss / (batch_size * accum_steps)).backward()

                batch_loss += total_loss.item()
                batch_kl += kl_loss.item()
                batch_ce += ce_loss.item()

                with open(distill_history_file, "a", encoding="utf-8") as f:
                    record = {
                        "model_train_step": distill_step,
                        "epoch": epoch,
                        "uid": uid,
                        "total_loss": total_loss.item(),
                        "kl_loss": kl_loss.item(),
                        "ce_loss": ce_loss.item(),
                        "gen_length": full_ids.shape[1],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == len(mini_batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                distill_optimizer.step()
                distill_optimizer.zero_grad()

                avg_kl, avg_ce, avg_loss = (
                    batch_kl / batch_size,
                    batch_ce / batch_size,
                    batch_loss / batch_size,
                )
                pbar.set_postfix(
                    {
                        "Loss": f"{avg_loss:.3f}",
                        "KL": f"{avg_kl:.3f}",
                        "CE": f"{avg_ce:.3f}",
                    }
                )

                wandb.log(
                    {
                        "model_train_step": distill_step,
                        "distill/train/total_loss": avg_loss,
                        "distill/train/kl_loss": avg_kl,
                        "distill/train/ce_loss": avg_ce,
                        "distill/train/lr": distill_optimizer.param_groups[0]["lr"],
                    }
                )

                if distill_step % args.distill_eval_every == 0:
                    if distill_eval_future is not None:
                        tqdm.write("⏳ 等待上一轮蒸馏评测结束...")
                        res = distill_eval_future.result()
                        if res:
                            wandb.log(res)

                    torch.save(model.state_dict(), shm_model_path)
                    distill_eval_future = executor.submit(run_distill_eval_async, distill_step, shm_model_path)
                    if distill_step > 0:
                        model.save_pretrained(f"{save_dir}/step{distill_step}/")

                distill_step += 1

    if distill_eval_future is not None:
        print("\n⏳ 蒸馏训练已全部完成，等待后台收尾上一轮的 Eval...")
        res = distill_eval_future.result()
        if res:
            wandb.log(res)

    # 2. 强制执行 Final Eval (针对最终的模型权重)
    print("\n🏁 启动 Final Eval (最终权重评测)...")
    torch.save(model.state_dict(), shm_model_path)
    
    # 这里可以直接同步运行最后一次评测，以确保流程完全走完
    final_res = run_distill_eval_async(distill_step, shm_model_path)
    if final_res:
        # 可以为 final_eval 加上特定的标记，方便在 wandb 中区分
        final_res_logged = {f"final_{k}": v for k, v in final_res.items() if k != "model_train_step"}
        final_res_logged["model_train_step"] = distill_step
        wandb.log(final_res_logged)
        print("✅ Final Eval 数据已记录至 wandb。")
    
    if distill_step > 0:
        model.save_pretrained(f"{save_dir}/step{distill_step}/")
    
    print(f"🎉 蒸馏全流程结束！保存模型文件到{save_dir}")

    wandb.finish()
    vllm_engine.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        print("\n" + "=" * 60)
        print("🧹 开始执行清理和保存操作...")
        
        # 定义需要检查和清理的共享内存文件路径
        shm_files = [
            "/dev/shm/distill_model_weights.pt",
        ]
        
        for shm_path in shm_files:
            if os.path.exists(shm_path):
                try:
                    # 2. 从共享内存中删除文件，释放 RAM
                    os.remove(shm_path)
                    print(f"🗑️ 已清理共享内存文件释放空间: {shm_path}")
                except Exception as e:
                    print(f"⚠️ 处理 {shm_path} 时发生错误: {e}")
                    
        print("=" * 60)