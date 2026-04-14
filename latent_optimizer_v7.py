import os
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import math
import argparse
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from rouge_score import rouge_scorer

from datasets import load_dataset
import random
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.utils.data import Dataset, DataLoader
from math_utils import is_correct_v3, last_boxed_only_string, remove_boxed, is_equiv
from math_wrong_dataset import build_math_wrong_dataset, MathWrongDataset
from vllm import LLM, SamplingParams
import os

# =========================================================================
# [🌟] 辅助函数：跨设备搬运优化器状态
# =========================================================================
def move_optimizer_state(optimizer, device):
    for param, state in optimizer.state.items():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


# =========================================================================
# [2] 优化器定义
# =========================================================================
class FrankWolfeOptimizer:
    def __init__(self, vocab_embeddings, adaptive_mode="off", top_k=0, top_p=0.0):
        self.W_emb = vocab_embeddings
        self.adaptive_mode = adaptive_mode
        self.top_k = top_k
        self.top_p = top_p

    def step(self, latent_tensor, gamma):
        if latent_tensor.grad is None:
            return None, None
            
        grad = latent_tensor.grad.to(self.W_emb.device).detach()
        seq_len = grad.shape[1]
        
        with torch.no_grad():
            grad_norms = torch.norm(grad, p=2, dim=-1) # [batch, seq_len]
            eff_gamma = torch.full_like(grad_norms, gamma)
            
            if self.adaptive_mode == "top_k":
                k = min(self.top_k, seq_len)
                if k > 0:
                    topk_vals, _ = torch.topk(grad_norms, k, dim=-1)
                    threshold = topk_vals[:, -1:]
                    mask = (grad_norms >= threshold).float()
                    eff_gamma = eff_gamma * mask
            elif self.adaptive_mode == "top_p":
                sorted_norms, sorted_idx = torch.sort(grad_norms, dim=-1, descending=True)
                cum_probs = torch.cumsum(sorted_norms, dim=-1) / (torch.sum(sorted_norms, dim=-1, keepdim=True) + 1e-8)
                mask_sorted = cum_probs <= self.top_p
                mask_sorted = torch.cat([torch.ones_like(mask_sorted[:, :1]), mask_sorted[:, :-1]], dim=-1)
                mask = torch.zeros_like(grad_norms).scatter_(-1, sorted_idx, mask_sorted.float())
                eff_gamma = eff_gamma * mask
            elif self.adaptive_mode == "soft":
                max_norm = torch.max(grad_norms, dim=-1, keepdim=True)[0] + 1e-8
                eff_gamma = gamma * (grad_norms / max_norm)

            scores = torch.matmul(grad, self.W_emb.T)
            best_vocab_indices = torch.argmin(scores, dim=-1)
            best_embeds = self.W_emb[best_vocab_indices].to(latent_tensor.device)
            
            eff_gamma_expanded = eff_gamma.unsqueeze(-1).to(latent_tensor.device)
            latent_tensor.copy_((1 - eff_gamma_expanded) * latent_tensor + eff_gamma_expanded * best_embeds)
            latent_tensor.grad.zero_()
            
            return best_vocab_indices, eff_gamma


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--file_path", type=str, default="/workspace/yiqiuguo/lsrl/qwen3-1.7b_math-500_rollout8_len32768_final.jsonl")
    parser.add_argument("--vllm_gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--big_batch_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_k", type=int, default=32)
    parser.add_argument("--eval_modes", type=int, nargs="+", default=["pure", "forced", "fast"])
    parser.add_argument("--mask_strategy", type=str, default="top_k_entropy", choices=["top_k_entropy", "first_k"])
    parser.add_argument("--mask_max_k", type=int, default=32768)
    parser.add_argument("--grad_direction", type=str, default="positive", choices=["positive", "negative"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "frank_wolfe"])
    parser.add_argument("--fw_gamma", type=float, default=0.1)
    # 修改原有的 conn_type 和 reg_type
    parser.add_argument("--conn_type", type=str, default="original", choices=["fast", "original", "on-policy"])
    parser.add_argument("--reg_type", type=str, default="lm", choices=["kl", "lm"])
    
    # 新增的参数
    parser.add_argument("--adaptive_grad", type=str, default="off", choices=["off", "top_k", "top_p", "soft"])
    parser.add_argument("--adaptive_grad_top_k", type=int, default=128)
    parser.add_argument("--adaptive_grad_top_p", type=float, default=0.8)
    parser.add_argument("--gamma_decay", type=str, default="cosine", choices=["constant", "cosine", "fw"])
    parser.add_argument("--early_stop_correctness_threshold", type=float, default=1.01) # 默认大于1，即不开启
    parser.add_argument("--n_conn_grad", type=int, default=1)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_threshold", type=float, default=1e-3)
    parser.add_argument("--skip_distill", action="store_true")
    parser.add_argument("--distill_epochs", type=int, default=3)
    parser.add_argument("--distill_lr", type=float, default=2e-5)
    parser.add_argument("--distill_ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--distill_eval_every", type=int, default=500)
    parser.add_argument("--distill_eval_datasets", type=str, nargs="+", default=["HuggingFaceH4/MATH-500"])
    parser.add_argument("--distill_batch_size", type=int, default=1)
    parser.add_argument("--distill_grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=9999999)
    return parser.parse_args()


# =========================================================================
# [4] 主流程
# =========================================================================
def main():
    # 初始化分布式环境
    args = parse_args()

    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        wandb.init(project="L-GRPO-Math500", config=vars(args))
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
    model.requires_grad_(False)
    model.eval()

    with torch.cuda.device(device):
        vllm_engine = LLM(
            model=args.model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
            data_parallel_size=world_size,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=0.2,
            dtype="bfloat16",
            enable_prompt_embeds=True,
            max_model_len=40960,
            enforce_eager=True,  # 🔥 强制 Eager 模式，关闭 CUDA Graph（重点）
        )

    def get_embeds(text):
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        return model.get_input_embeddings()(ids).detach(), ids

    embeds_end_think, ids_end_think = get_embeds("</think>")
    embeds_fast_conn, ids_fast_conn = get_embeds("The final answer is \n&&\n\\boxed{")
    all_embeddings = model.get_input_embeddings().weight.detach()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    if rank == 0:
        print(f"📄 读取文件: {args.file_path}")
    
    wrong_dataset = build_math_wrong_dataset(args.file_path, tokenizer)
    
    all_raw_data = wrong_dataset.flat_data
    if args.max_samples is not None:
        all_raw_data = all_raw_data[: args.max_samples]

    my_raw_data = [all_raw_data[i] for i in range(len(all_raw_data)) if i % world_size == rank]

    raw_data_chunks = [my_raw_data[i : i + args.big_batch_size] for i in range(0, len(my_raw_data), args.big_batch_size)]

    global_history = {}
    if rank == 0:
        global_history = {d["uid"]: {"uid": d["uid"], "problem": d["question_text"], "gt_text": d["gt_text"], "steps": []} for d in all_raw_data}

    # 每个 Rank 维护自己的局部数据 local_history
    local_history = {d["uid"]: {"uid": d["uid"], "problem": d["question_text"], "gt_text": d["gt_text"], "steps": []} for d in my_raw_data}
    local_distill_dataset = []
    global_early_stopped_uids = set()

    embeds_end_think_cpu = embeds_end_think.cpu()
    embeds_fast_conn_cpu = embeds_fast_conn.cpu()

    run_name = wandb.run.name if (wandb.run is not None and rank == 0) else f"opt_v2_{args.optimizer}_{args.reg_type}"
    # [修改]：统一写到一个文件里，去除 _rank{rank} 后缀
    history_filename = f"./optimization_histories/optimization_history_{run_name}.jsonl"
    if rank == 0:
        os.makedirs(os.path.dirname(history_filename), exist_ok=True)

    # [新增逻辑]：同步收集各节点的 local_history 并由 Rank 0 写入
    def sync_and_save_history():
        # 1. 创建接收容器 (仅 rank 0 需要有效容器，其他 rank 传 None 即可)
        gathered_histories = [None] * world_size if rank == 0 else None
        
        # 2. 收集所有节点的 local_history
        dist.gather_object(local_history, gathered_histories, dst=0)
        
        # 3. Rank 0 汇总并保存
        if rank == 0:
            for r_history in gathered_histories:
                for uid, history_data in r_history.items():
                    # 替换更新 global_history 中对应 uid 的数据点
                    global_history[uid] = history_data
            
            # 使用临时文件写入后替换（保证读取时不会读到残缺的 jsonl）
            tmp_filename = history_filename + ".tmp"
            with open(tmp_filename, "w", encoding="utf-8") as f:
                # 第一行：Config
                f.write(json.dumps({"config": vars(args)}, ensure_ascii=False) + "\n")
                # 后续行：每个 UID 一行
                for uid in global_history.keys():
                    f.write(json.dumps(global_history[uid], ensure_ascii=False) + "\n")
            os.replace(tmp_filename, history_filename)

    # 初始触发一次同步和保存
    sync_and_save_history()

    pos_big = rank * 3
    pos_step = rank * 3 + 1
    pos_mini = rank * 3 + 2

    # ==================================================
    # [🚀 同步评估逻辑]
    # ==================================================
    def run_eval_sync(eval_step, eval_uids, latents_snapshot, chunk_static_data, return_valid_answers=False):
        # 1. 无论是否有数据，优先初始化好所有的基础变量，防止后续报错
        step_metrics = {uid: {} for uid in eval_uids}
        local_pure_acc, local_forced_acc, local_fast_acc = 0.0, 0.0, 0.0
        local_total_change_ratio = 0.0
        valid_answers = {}
        
        # 2. 只有当前卡上分配到了数据，才执行繁重的计算和 vLLM 推理
        if len(eval_uids) > 0:
            eval_pure_inputs, eval_forced_inputs, eval_fast_inputs = [], [], []
            
            tqdm.write(f"\n[R{rank}] 🚀 [后台评测] Step {eval_step} 启动! 正在 {device} 计算潜空间漂移...")
            with torch.no_grad():
                for uid in eval_uids:
                    ct_cpu = latents_snapshot[uid]
                    sd = chunk_static_data[uid]
                    target_flat = ct_cpu.squeeze(0).to(device)

                    L = target_flat.shape[0]
                    nearest_token_ids_gpu = torch.empty(L, dtype=torch.long, device=device)
                    for c_start in range(0, L, args.chunk_size):
                        c_end = min(c_start + args.chunk_size, L)
                        chunk_eval_gpu = target_flat[c_start:c_end]
                        chunk_norm = F.normalize(chunk_eval_gpu, dim=-1).to(torch.bfloat16)
                        sim_chunk = torch.matmul(chunk_norm, all_embeddings.T)
                        nearest_token_ids_gpu[c_start:c_end] = torch.argmax(sim_chunk, dim=-1)

                    nearest_token_ids = nearest_token_ids_gpu.cpu()
                    orig_ids = sd["ids_think"].squeeze(0)
                    changed_mask = nearest_token_ids != orig_ids
                    change_ratio = changed_mask.float().mean().item()

                    decoded_nearest_text = tokenizer.decode(nearest_token_ids).strip()

                    step_metrics[uid]["change_ratio"] = change_ratio
                    step_metrics[uid]["curr_thinking_text"] = decoded_nearest_text
                    local_total_change_ratio += change_ratio

                    embeds_q_cpu = sd["embeds_q_cpu"]
                    embeds_conn_cpu = sd["embeds_conn_cpu"]
                    eval_pure_inputs.append(torch.cat([embeds_q_cpu, ct_cpu, embeds_end_think_cpu], dim=1).squeeze(0))
                    eval_forced_inputs.append(torch.cat([embeds_q_cpu, ct_cpu, embeds_end_think_cpu, embeds_conn_cpu], dim=1).squeeze(0))
                    eval_fast_inputs.append(torch.cat([embeds_q_cpu, ct_cpu, embeds_end_think_cpu, embeds_fast_conn_cpu], dim=1).squeeze(0))

            tqdm.write(f"[R{rank}]⚡ [后台评测] 漂移计算完毕，准备发送数据至vLLM")

            modes = ["pure", "forced", "fast"]
            max_toks = {"pure": 2048, "forced": 128, "fast": 128}
            my_results = {}

            for mode in modes:
                if mode == "pure":
                    inputs_tensors = eval_pure_inputs
                elif mode == "forced":
                    inputs_tensors = eval_forced_inputs
                else:
                    inputs_tensors = eval_fast_inputs

                if len(inputs_tensors) > 0:
                    with torch.cuda.device(device):
                        vllm_inputs = [{"prompt_embeds": emb.to(device)} for emb in inputs_tensors]
                        sp = SamplingParams(max_tokens=max_toks[mode], temperature=0.7, n=args.eval_k, skip_special_tokens=False)
                        outputs = vllm_engine.generate(vllm_inputs, sampling_params=sp, use_tqdm=True)

                    my_results[mode] = []
                    for out in outputs:
                        my_results[mode].append([list(comp.token_ids) for comp in out.outputs])
                else:
                    my_results[mode] = []

            for mode in args.eval_modes:
                tqdm.write(f"[R{rank}] 🚀 Step{eval_step} {mode}模型评测开始")
                batch_outputs = my_results[mode]
                for idx, uid in enumerate(eval_uids):
                    gt_text = local_history[uid]["gt_text"]
                    gt_token_len = len(tokenizer.encode(gt_text, add_special_tokens=False))
                    correct_count = 0

                    if len(batch_outputs) > 0:
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
                                    # 收集错误的回答用于 on-policy connector
                                    if "wrong_answers" not in step_metrics[uid]:
                                        step_metrics[uid]["wrong_answers"] = []
                                    step_metrics[uid]["wrong_answers"].append(gen_text)
                            else:
                                ans = gen_text.replace("$", "").replace("}", "").strip()
                                if is_equiv(ans, gt_text.replace("}", "")):
                                    is_corr = True

                            if is_corr:
                                correct_count += 1

                    acc = correct_count / args.eval_k if args.eval_k > 0 else 0
                    step_metrics[uid][f"{mode}_acc"] = acc

                    if mode == "pure" and len(batch_outputs[idx]) > 0:
                        sample_gen = tokenizer.decode(batch_outputs[idx][0], skip_special_tokens=True)
                        step_metrics[uid]["sample_gen_text"] = sample_gen
                    
                    if mode == "pure":
                        local_pure_acc += acc
                    elif mode == "forced":
                        local_forced_acc += acc
                    elif mode == "fast":
                        local_fast_acc += acc
        else:
            # 如果没有数据，打印一条 log 方便调试
            tqdm.write(f"[R{rank}] ⚠️ Step {eval_step}: 暂无分配到数据，直接进入同步等待...")


        # 3. ===== 极度关键区：所有卡必须到达这里 =====
        # 将统计结果放入 tensor。统一使用 float32，避免类型不匹配引发 NCCL 报错
        local_counts = torch.tensor([
            local_pure_acc, 
            local_forced_acc, 
            local_fast_acc, 
            local_total_change_ratio, 
            len(eval_uids)
        ], dtype=torch.float32, device=device)

        # 如果有卡没有数据，它会提供 [0, 0, 0, 0, 0]，完美融入 SUM 运算而不破坏结果
        dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
        
        global_uids_count = local_counts[4].item()

        avg_metrics = {}
        if global_uids_count > 0:
            avg_metrics = {
                "eval/avg_pure_acc": local_counts[0].item() / global_uids_count,
                "eval/avg_forced_acc": local_counts[1].item() / global_uids_count,
                "eval/avg_fast_acc": local_counts[2].item() / global_uids_count,
                "eval/avg_change_ratio": local_counts[3].item() / global_uids_count, 
            }
            
        res = {"step": eval_step, "uids": eval_uids, "step_metrics": step_metrics, "avg_metrics": avg_metrics}
        
        if return_valid_answers:
            res["valid_answers"] = valid_answers
            
        return res

    def process_eval_results(res):
        if res is None:
            return
        eval_step, eval_uids, eval_metrics = res["step"], res["uids"], res["step_metrics"]
        if rank == 0:
            metrics_display = "  ".join([f"{k.replace('eval/avg_', '')}: {v:.4f}" for k, v in res["avg_metrics"].items()])
            tqdm.write(f"--- Eval Step {eval_step} ---")
            tqdm.write(f"Samples: {int(len(eval_uids))} | {metrics_display}")
            wandb.log({"global_step": eval_step, **res["avg_metrics"]})
        for uid in eval_uids:
            for step_record in local_history[uid]["steps"]:
                if step_record["step"] == eval_step:
                    step_record["metrics"].update(eval_metrics[uid])
                    break

    # =========================================================================
    # [🔥 外层大循环] Big Batch 迭代控制内存占用
    # =========================================================================
    big_batch_pbar = tqdm(enumerate(raw_data_chunks), total=len(raw_data_chunks), desc=f"[R{rank}] 🌍 Big Batch", position=pos_big, leave=True)
    for chunk_idx, current_raw_data in big_batch_pbar:
        static_data_cpu = {}
        global_latents = torch.nn.ParameterDict()
        active_uids = []

        # --- Phase 1: Pre-computing ---
        for i in tqdm(range(0, len(current_raw_data), args.batch_size), desc=f"[R{rank}] Pre-computing", position=pos_step, leave=False):
            batch_samples = current_raw_data[i : i + args.batch_size]
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
                outputs = model(inputs_embeds=torch.cat(padded_embeds, dim=0), attention_mask=torch.cat(attn_masks, dim=0))
                all_logits = outputs.logits

            for j, info in enumerate(batch_info):
                uid = info["uid"]
                probs = F.softmax(all_logits[j, info["start"] : info["end"], :].unsqueeze(0), dim=-1)

                if args.mask_strategy == "top_k_entropy":
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    mask = torch.zeros_like(entropy, dtype=torch.float32)
                    mask.scatter_(1, torch.topk(entropy, k=min(args.mask_max_k, entropy.shape[1]), dim=1).indices, 1.0)
                else:
                    mask = torch.zeros((1, info["ids_think"].shape[1]), dtype=torch.float32, device=device)
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
        torch.cuda.empty_cache()

        global_opts = {}
        if args.optimizer == "adam":
            for uid, param in global_latents.items():
                global_opts[uid] = torch.optim.Adam([param], lr=args.learning_rate)
        elif args.optimizer == "frank_wolfe":
            optimizer = FrankWolfeOptimizer(all_embeddings)

        # =========================================================================
        # [🔥 内层Step循环] Big Batch内优化步数循环
        # =========================================================================
        step_pbar = tqdm(range(args.steps), desc=f"[R{rank}] 🏃 Steps", position=pos_step, leave=False)
        for step in step_pbar:
            global_step_id = chunk_idx * args.steps + step

            local_active_count = torch.tensor([len(active_uids)], device=device, dtype=torch.int32)
            dist.all_reduce(local_active_count, op=dist.ReduceOp.SUM)
            if local_active_count.item() == 0:
                break

            current_lr = 0.0
            epoch_gt_loss_sum, epoch_kl_loss_sum, epoch_lm_loss_sum, epoch_total_loss_sum = 0.0, 0.0, 0.0, 0.0
            step_metrics = {uid: {} for uid in active_uids}

            if len(active_uids) > 0:
                progress = step / max(1, args.steps - 1)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                current_lr = (args.learning_rate * 0.1) + (args.learning_rate * 0.9) * cosine_factor
                progress = step / max(1, args.steps - 1)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                current_lr = (args.learning_rate * 0.1) + (args.learning_rate * 0.9) * cosine_factor
                
                if args.gamma_decay == "constant":
                    current_fw_gamma = args.fw_gamma
                elif args.gamma_decay == "cosine":
                    current_fw_gamma = (args.fw_gamma * 0.1) + (args.fw_gamma * 0.9) * cosine_factor
                elif args.gamma_decay == "fw":
                    current_fw_gamma = 2.0 / (step + 2)

                # 初始化 FW 优化器（移到这里，传入自适应参数）
                if args.optimizer == "frank_wolfe":
                    optimizer = FrankWolfeOptimizer(
                        all_embeddings, 
                        adaptive_mode=args.adaptive_grad, 
                        top_k=args.adaptive_grad_top_k, 
                        top_p=args.adaptive_grad_top_p
                    )

                mini_batches = [active_uids[i : i + args.batch_size] for i in range(0, len(active_uids), args.batch_size)]
                inner_pbar = tqdm(mini_batches, desc=f"[R{rank}] 🔥 Minibatch", position=pos_mini, leave=False)

                for batch_uids in inner_pbar:
                    curr_think_list, curr_mask_list = [], []
                    embeds_q_list, embeds_conn_list, embeds_gt_list, embeds_pred_list = [], [], [], []
                    ids_q_list, ids_think_list, ids_conn_list, ids_gt_list, ids_pred_list = [], [], [], [], []

                    # 确定当前 batch 最大的 connector 数量
                    max_conns = 1
                    if args.conn_type == "on-policy":
                        max_conns = max([len(static_data_cpu[uid].get("active_conns", [(None, None)])) for uid in batch_uids])

                    batch_size_cur = len(batch_uids)
                    
                    # 循环处理每一个 Connector (时间换空间，避免 OOM)
                    for c_idx in range(max_conns):
                        full_embeds_list = []
                        valid_indices = []
                        
                        for i, uid in enumerate(batch_uids):
                            sd = static_data_cpu[uid]
                            conns = sd.get("active_conns", [(sd["embeds_conn_cpu"], sd["ids_conn_cpu"])])
                            if c_idx < len(conns):
                                emb_conn, _ = conns[c_idx]
                                target_emb = embeds_gt_list[i] if args.grad_direction == "positive" else embeds_pred_list[i]
                                full_emb = torch.cat([embeds_q_list[i], curr_think_list[i].to(model.dtype), embeds_end_think, emb_conn.to(device), target_emb], dim=1)
                                full_embeds_list.append(full_emb)
                                valid_indices.append(i)
                        
                        if not full_embeds_list: continue

                        # Padding 和 Forward
                        max_len = max(emb.shape[1] for emb in full_embeds_list)
                        attention_mask_list = []
                        for i, emb in enumerate(full_embeds_list):
                            if emb.shape[1] < max_len:
                                pad_emb = torch.zeros((emb.shape[0], max_len - emb.shape[1], emb.shape[2]), device=device, dtype=model.dtype)
                                full_embeds_list[i] = torch.cat([emb, pad_emb], dim=1)
                                attn = torch.zeros((emb.shape[0], max_len), device=device, dtype=torch.long)
                                attn[:, : emb.shape[1]] = 1
                                attention_mask_list.append(attn)
                            else:
                                attention_mask_list.append(torch.ones((emb.shape[0], max_len), device=device, dtype=torch.long))

                        full_embeds_batch = torch.cat(full_embeds_list, dim=0)
                        full_attention_mask = torch.cat(attention_mask_list, dim=0)

                        last_hidden = model.model(inputs_embeds=full_embeds_batch, attention_mask=full_attention_mask).last_hidden_state
                        last_hidden_detached = last_hidden.detach().requires_grad_(True)

                        for local_idx, i in enumerate(valid_indices):
                            uid = batch_uids[i]
                            sd = static_data_cpu[uid]
                            conns = sd.get("active_conns", [(sd["embeds_conn_cpu"], sd["ids_conn_cpu"])])
                            _, id_conn = conns[c_idx]
                            id_conn = id_conn.to(device)
                            num_conns = len(conns)

                            # 1. 计算 GT Loss
                            gt_pos = ids_q_list[i].shape[1] + ids_think_list[i].shape[1] + ids_end_think.shape[1] + id_conn.shape[1] - 1
                            target_ids = ids_gt_list[i] if args.grad_direction == "positive" else ids_pred_list[i]
                            target_logits = model.lm_head(last_hidden_detached[[local_idx], gt_pos : gt_pos + target_ids.shape[1], :])
                            gt_loss = F.cross_entropy(target_logits.view(-1, target_logits.size(-1)), target_ids.view(-1))
                            if args.grad_direction == "negative": gt_loss = -gt_loss

                            # 2. 计算 REG Loss (KL 或 新版 LM Loss)
                            think_start = ids_q_list[i].shape[1] - 1
                            think_len = ids_think_list[i].shape[1]
                            reg_loss_val = 0.0
                            lm_loss_hard_val = 0.0

                            if args.reg_type == "kl":
                                # (保留你原有的 KL Loss Chunk 计算逻辑，累加到 reg_loss_val，注意除以 num_conns)
                                pass # 这里为了简洁省略，直接用你原来的 KL 代码即可
                            
                            elif args.reg_type == "lm":
                                # --- 新版 LM Loss 逻辑 ---
                                curr_think_norm = F.normalize(curr_think_list[i], p=2, dim=-1)
                                sim = torch.matmul(curr_think_norm, all_embeddings.T.to(device))
                                curr_think_hard_ids = torch.argmax(sim, dim=-1) # [1, think_len]
                                
                                # A. 可导的 LM Loss (Chunked 节省显存)
                                lm_loss_sum = 0.0
                                for c_start in range(0, think_len - 1, args.chunk_size):
                                    c_end = min(c_start + args.chunk_size, think_len - 1)
                                    h_chunk = last_hidden_detached[[local_idx], think_start + c_start : think_start + c_end, :]
                                    logits_chunk = model.lm_head(h_chunk)
                                    labels_chunk = curr_think_hard_ids[:, c_start + 1 : c_end + 1]
                                    loss_chunk = F.cross_entropy(logits_chunk.reshape(-1, logits_chunk.size(-1)), labels_chunk.reshape(-1), reduction='sum')
                                    lm_loss_sum += loss_chunk
                                reg_loss_val = lm_loss_sum / (think_len - 1)

                                # B. 不可导的 LM Loss Hard (仅用于观察，只在 c_idx==0 时算一次)
                                if c_idx == 0 and step % 5 == 0: # 没必要每步都算，省点时间
                                    with torch.no_grad():
                                        hard_embeds = model.get_input_embeddings()(curr_think_hard_ids)
                                        hard_full_emb = torch.cat([embeds_q_list[i], hard_embeds], dim=1)
                                        hard_logits = model(inputs_embeds=hard_full_emb).logits
                                        hard_think_logits = hard_logits[:, ids_q_list[i].shape[1]-1 : ids_q_list[i].shape[1]-1 + think_len - 1, :]
                                        lm_loss_hard_val = F.cross_entropy(hard_think_logits.reshape(-1, hard_think_logits.size(-1)), curr_think_hard_ids[:, 1:].reshape(-1)).item()
                                        step_metrics[uid]["lm_loss_hard"] = lm_loss_hard_val

                            # 3. 组合 Loss 并 Backward
                            s_gt = gt_loss / num_conns
                            s_reg = reg_loss_val / num_conns
                            s_total = s_gt + args.kl_weight * s_reg

                            if step > 0:
                                (s_total / batch_size_cur).backward()

                            # 记录 Metrics (累加多个 Conn 的结果)
                            if "total_loss" not in step_metrics[uid]:
                                step_metrics[uid].update({"total_loss": 0, "gt_loss": 0, "reg_loss": 0})
                            step_metrics[uid]["total_loss"] += s_total.item()
                            step_metrics[uid]["gt_loss"] += s_gt.item()
                            step_metrics[uid]["reg_loss"] += s_reg.item() if isinstance(s_reg, torch.Tensor) else s_reg

                            epoch_gt_loss_sum += s_gt.item()
                            epoch_total_loss_sum += s_total.item()

                        if step > 0:
                            last_hidden.backward(last_hidden_detached.grad)

                    for i, uid in enumerate(batch_uids):
                        p, mask = curr_think_list[i], curr_mask_list[i]
                        fw_ids, fw_weights = None, None
                        
                        if p.grad is not None:
                            p.grad.data.mul_(mask.to(device).to(p.dtype))
                            
                        if args.optimizer == "adam":
                            opt = global_opts[uid]
                            if step > 0: opt.step()
                            opt.zero_grad(set_to_none=True)
                            move_optimizer_state(opt, torch.device("cpu"))
                        elif args.optimizer == "frank_wolfe" and step > 0:
                            fw_ids, fw_weights = optimizer.step(p, gamma=current_fw_gamma)
                            
                        p.data = p.data.cpu().pin_memory()
                        
                        # 记录 FW 的动作到 History
                        if fw_ids is not None:
                            step_metrics[uid]["fw_action_ids"] = fw_ids.cpu().tolist()
                            step_metrics[uid]["fw_action_weights"] = fw_weights.cpu().tolist()

                # 更新 active_uids 时，剔除 Early Stop 的样本
                next_active_uids = []
                for uid in active_uids:
                    local_history[uid]["steps"].append({"step": global_step_id, "metrics": step_metrics[uid]})
                    
                    is_gt_converged = args.early_stop and step_metrics[uid]["gt_loss"] < args.early_stop_threshold
                    is_pure_acc_converged = uid in global_early_stopped_uids
                    
                    if not (is_gt_converged or is_pure_acc_converged):
                        next_active_uids.append(uid)
                        
                local_active_len = len(active_uids)
                active_uids = next_active_uids
            else:
                local_active_len = 0

            local_loss_tensor = torch.tensor([epoch_total_loss_sum, epoch_gt_loss_sum, local_active_len], device=device, dtype=torch.float32)
            dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)

            if rank == 0 and local_loss_tensor[2].item() > 0:
                wandb.log(
                    {
                        "global_step": global_step_id,
                        "train/epoch_total_loss": local_loss_tensor[0].item() / local_loss_tensor[2].item(),
                        "train/epoch_gt_loss": local_loss_tensor[1].item() / local_loss_tensor[2].item(),
                        "train/lr": current_lr,
                        "active_samples": local_loss_tensor[2].item(),
                    }
                )

            # 找到这段代码：
            if step % args.eval_every == 0 and step != args.steps - 1 and step > 0:
                eval_uids = list(active_uids) if len(active_uids) > 0 else []
                latents_snapshot = {uid: global_latents[uid].detach().clone() for uid in eval_uids}
                res = run_eval_sync(global_step_id, eval_uids, latents_snapshot, static_data_cpu)
                process_eval_results(res)
                sync_and_save_history()
                
                # ========== 新增：Early Stop (Pure Acc) 与 On-Policy Connector ==========
                if res is not None:
                    for uid in eval_uids:
                        metrics = res["step_metrics"][uid]
                        # 1. 检查 Pure Acc Early Stop
                        if metrics.get("pure_acc", 0.0) >= args.early_stop_correctness_threshold:
                            if uid not in global_early_stopped_uids:
                                global_early_stopped_uids.add(uid)
                                tqdm.write(f"[R{rank}] 🛑 UID {uid} Pure Acc 达标，提前终止优化！")
                        
                        # 2. 构造 On-Policy Connector
                        if args.conn_type == "on-policy" and uid not in global_early_stopped_uids:
                            wrongs = metrics.get("wrong_answers", [])
                            if len(wrongs) > 0:
                                new_conns = []
                                for w_text in wrongs:
                                    if len(new_conns) >= args.n_conn_grad: break
                                    try:
                                        parts = w_text.split("</think>")
                                        after_thinking_text = parts[1] if len(parts) > 1 else ""
                                        if "\\boxed{" in after_thinking_text:
                                            pred_box_text = last_boxed_only_string(after_thinking_text)
                                            connector_text = after_thinking_text.split(pred_box_text)[0] + "\\boxed{"
                                            emb_c, id_c = get_embeds(connector_text)
                                            new_conns.append((emb_c.cpu(), id_c.cpu()))
                                    except Exception:
                                        pass
                                if len(new_conns) > 0:
                                    static_data_cpu[uid]["active_conns"] = new_conns
                # =====================================================================

        chunk_all_uids = [d["uid"] for d in current_raw_data]
        final_latents_cpu = {uid: global_latents[uid].detach().cpu() for uid in chunk_all_uids}

        final_eval_result = run_eval_sync(global_step_id, chunk_all_uids, final_latents_cpu, static_data_cpu, return_valid_answers=True)
        process_eval_results(final_eval_result)
        sync_and_save_history()

        valid_answers = final_eval_result.get("valid_answers", {})

        model.eval()
        with torch.no_grad():
            for uid, best_output_ids in valid_answers.items():
                sd = static_data_cpu[uid]
                ct = final_latents_cpu[uid].to(device).to(model.dtype)
                emb_q = sd["embeds_q_cpu"].to(device)

                full_emb = torch.cat([emb_q, ct], dim=1)
                logits = model(inputs_embeds=full_emb).logits

                think_start = emb_q.shape[1] - 1
                think_end = full_emb.shape[1] - 1
                think_logits = logits[0, think_start:think_end, :]

                probs = F.softmax(think_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=100, dim=-1)

                local_distill_dataset.append(
                    {
                        "uid": uid,
                        "ids_q": sd["ids_q"].cpu(),
                        "ids_orig_think": sd["ids_think"].cpu(),
                        "target_probs": topk_probs.cpu(),
                        "target_indices": topk_indices.cpu(),
                        "ids_new_ans": torch.tensor([best_output_ids], dtype=torch.long).cpu(),
                        "gt_text": local_history[uid]["gt_text"],
                        "problem": local_history[uid]["problem"],
                    }
                )

        del static_data_cpu, global_latents, final_latents_cpu
        torch.cuda.empty_cache()
        

    dist.barrier()
    if args.skip_distill:
        if rank == 0:
            wandb.finish()
        return

    all_distill = [None] * world_size
    dist.all_gather_object(all_distill, local_distill_dataset)
    if rank == 0:
        global_distill_dataset = [item for sublist in all_distill for item in sublist]
    else:
        global_distill_dataset = [None]

    dist.broadcast_object_list([global_distill_dataset], src=0)
    global_distill_dataset = global_distill_dataset[0]

    if rank == 0:
        print(f"\n🔥 启动 FSDP 模型蒸馏阶段... 过滤后全局总样本数: {len(global_distill_dataset)}")

    if len(global_distill_dataset) == 0:
        if rank == 0:
            wandb.finish()
        return

    if rank == 0:

        def pass_at_k(n, c, k):
            if n - c < k:
                return 1.0
            return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

        eval_datasets = {}
        for ds_id in args.distill_eval_datasets:
            try:
                if "MATH-500" in ds_id:
                    ds, q_key, a_key = load_dataset(ds_id, split="test"), "problem", "answer"
                elif "amc23" in ds_id:
                    ds, q_key, a_key = load_dataset(ds_id, split="test"), "question", "answer"
                elif "gsm8k" in ds_id:
                    ds, q_key, a_key = load_dataset(ds_id, "main", split="test"), "question", "answer"
                elif "aime_2024" in ds_id:
                    ds, q_key, a_key = load_dataset(ds_id, split="train"), "problem", "answer"
                elif "aime25" in ds_id:
                    ds, q_key, a_key = load_dataset(ds_id, split="test"), "problem", "answer"
                else:
                    continue
                eval_datasets[ds_id] = {"data": ds, "q_key": q_key, "a_key": a_key}
            except Exception:
                pass

        train_eval_data = [{"problem_formatted": item["problem"], "answer_gt": item["gt_text"].replace("}", "")} for item in global_distill_dataset]
        eval_datasets["train_dataset"] = {"data": train_eval_data, "q_key": "problem_formatted", "a_key": "answer_gt", "pre_formatted": True}

    def run_distill_eval_sync(step, current_state_dict_path):
        if rank == 0:
            # 🚀 对应传入 local_rank 隔离
            print(f"🔄 [Rank 0] 正在更新本地 vLLM 权重...")
            state_dict = torch.load(current_state_dict_path, map_location="cpu")

            executor = vllm_engine.llm_engine.model_executor
            if hasattr(executor, "driver_worker"):
                vllm_model = executor.driver_worker.model_runner.model
            else:
                vllm_model = executor.model_runner.model

            vllm_model.load_weights(state_dict.items())
            del state_dict
            torch.cuda.empty_cache()

            metrics_to_log = {"model_train_step": step}
            total_pass1, total_pass8 = [], []

            for ds_id, ds_info in eval_datasets.items():
                ds, q_key, a_key = ds_info["data"], ds_info["q_key"], ds_info["a_key"]
                is_pre_formatted = ds_info.get("pre_formatted", False)
                current_n = 32 if any(x in ds_id for x in ["aime_2024", "aime25", "amc23"]) else 8

                prompts = []
                for item in ds:
                    if is_pre_formatted:
                        prompts.append(item[q_key])
                    else:
                        prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": item[q_key]}], tokenize=False, add_generation_prompt=True))

                sp = SamplingParams(max_tokens=32768, temperature=0.7, n=current_n, skip_special_tokens=False)
                outputs = vllm_engine.generate(prompts, sampling_params=sp, use_tqdm=True)

                ds_pass1, ds_pass8 = 0.0, 0.0
                for idx, item in enumerate(ds):
                    gt_ans = str(item[a_key]).strip()
                    correct_count = 0

                    out_ids_batch = [list(comp.token_ids) for comp in outputs[idx].outputs]
                    for output_ids in out_ids_batch:
                        ans_cand = last_boxed_only_string(tokenizer.decode(output_ids, skip_special_tokens=True))
                        if ans_cand and is_equiv(remove_boxed(ans_cand), gt_ans):
                            correct_count += 1

                    ds_pass1 += pass_at_k(current_n, correct_count, 1)
                    if current_n >= 8:
                        ds_pass8 += pass_at_k(current_n, correct_count, 8)

                num_samples = len(ds)
                metrics_to_log[f"distill/eval/{ds_id}/pass@1"] = ds_pass1 / num_samples
                total_pass1.append(ds_pass1 / num_samples)
                if current_n >= 8:
                    metrics_to_log[f"distill/eval/{ds_id}/pass@8"] = ds_pass8 / num_samples
                    total_pass8.append(ds_pass8 / num_samples)

            if total_pass1:
                metrics_to_log["distill/eval/avg_pass@1"] = sum(total_pass1) / len(total_pass1)
            if total_pass8:
                metrics_to_log["distill/eval/avg_pass@8"] = sum(total_pass8) / len(total_pass8)

            wandb.log(metrics_to_log)
            tqdm.write(f"✅ [Rank 0] 蒸馏评测完成，Avg Pass@1: {metrics_to_log.get('distill/eval/avg_pass@1', 0):.2%}")

        dist.barrier()

    torch.cuda.empty_cache()
    model.requires_grad_(True)
    model.train()
    model.gradient_checkpointing_enable()

    fsdp_model = FSDP(model, device_id=local_rank)
    distill_optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=args.distill_lr)

    sampler = DistributedSampler(global_distill_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    def distill_collate_fn(batch):
        return batch

    dataloader = DataLoader(global_distill_dataset, batch_size=args.distill_batch_size, sampler=sampler, collate_fn=distill_collate_fn)

    shm_model_path = "/dev/shm/distill_model_weights.pt"
    distill_step = 0

    for epoch in range(args.distill_epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"[R{rank}] Distill Epoch {epoch + 1}/{args.distill_epochs}", leave=False)

        distill_optimizer.zero_grad()
        for b_idx, batch in enumerate(pbar):
            batch_loss, batch_kl, batch_ce = 0.0, 0.0, 0.0

            for item in batch:
                ids_q = item["ids_q"].to(device)
                ids_think = item["ids_orig_think"].to(device)
                ids_new_ans = item["ids_new_ans"].to(device)
                target_probs = item["target_probs"].to(device)
                target_indices = item["target_indices"].to(device)

                full_ids = torch.cat([ids_q, ids_think, ids_end_think.to(device), ids_new_ans], dim=1)
                outputs = fsdp_model(full_ids)
                logits = outputs.logits

                think_start = ids_q.shape[1] - 1
                think_end = think_start + target_probs.shape[0]
                pred_think_logits = logits[0, think_start:think_end, :]

                pred_lse = torch.logsumexp(pred_think_logits, dim=-1, keepdim=True)
                pred_log_probs_topk = torch.gather(pred_think_logits, -1, target_indices) - pred_lse
                kl_loss = (target_probs * (torch.log(target_probs + 1e-10) - pred_log_probs_topk)).sum(dim=-1).mean()

                ans_start = full_ids.shape[1] - ids_new_ans.shape[1] - 1
                ans_end = full_ids.shape[1] - 1
                ce_loss = F.cross_entropy(logits[0, ans_start:ans_end, :], ids_new_ans[0])

                total_loss = kl_loss + args.distill_ce_loss_weight * ce_loss
                (total_loss / args.distill_grad_accum_steps).backward()

                batch_loss += total_loss.item()
                batch_kl += kl_loss.item()
                batch_ce += ce_loss.item()

            if (b_idx + 1) % args.distill_grad_accum_steps == 0 or (b_idx + 1) == len(dataloader):
                fsdp_model.clip_grad_norm_(1.0)
                distill_optimizer.step()
                distill_optimizer.zero_grad()

                local_metrics = torch.tensor([batch_loss, batch_kl, batch_ce], device=device)
                dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)

                if rank == 0:
                    wandb.log(
                        {
                            "model_train_step": distill_step,
                            "distill/train/total_loss": local_metrics[0].item() / world_size,
                            "distill/train/kl_loss": local_metrics[1].item() / world_size,
                            "distill/train/ce_loss": local_metrics[2].item() / world_size,
                            "distill/train/lr": distill_optimizer.param_groups[0]["lr"],
                        }
                    )

                if distill_step % args.distill_eval_every == 0:
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
                        cpu_state_dict = fsdp_model.state_dict()
                    if rank == 0:
                        torch.save(cpu_state_dict, shm_model_path)
                    run_distill_eval_sync(distill_step, shm_model_path)

                distill_step += 1

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = fsdp_model.state_dict()

    if rank == 0:
        torch.save(cpu_state_dict, shm_model_path)
    run_distill_eval_sync(distill_step, shm_model_path)

    if rank == 0:
        save_dir = f"./checkpoints/{run_name}/step{distill_step}/"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, state_dict=cpu_state_dict)
        tokenizer.save_pretrained(save_dir)
        print(f"🎉 蒸馏全流程结束！模型权重已保存到 {save_dir}")
        wandb.finish()
        if os.path.exists(shm_model_path):
            os.remove(shm_model_path)
        vllm_engine.close()


if __name__ == "__main__":
    main()
