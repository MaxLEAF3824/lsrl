import os
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["FLASHINFER_LOG_LEVEL"] = "WARNING"
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
import datetime
from datasets import load_dataset
import random
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.utils.data import Dataset, DataLoader
from math_utils import is_correct_v3, last_boxed_only_string, remove_boxed, is_equiv
from math_wrong_dataset import build_math_wrong_dataset, MathWrongDataset
from vllm import LLM, SamplingParams
from frank_wolfe_optimizer import FrankWolfeOptimizer

# =========================================================================
# [🌟] 辅助函数：跨设备搬运优化器状态 & Token 混合比例更新
# =========================================================================
def move_optimizer_state(optimizer, device):
    for param, state in optimizer.state.items():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

# [🌟 修改点1] 新增函数：用于维护每个位置的 Soft Token IDs 混合比例
def update_token_mix(current_mix, fw_ids, fw_weights, mask):
    """
    更新 token 混合字典。
    current_mix: list of dict, 形如 [{'1024': 1.0}, {'200': 0.8, '300': 0.2}, ...]
    fw_ids: Tensor (seq_len,)
    fw_weights: Tensor (seq_len,)
    mask: Tensor (seq_len,)
    """
    for i in range(len(current_mix)):
        m = mask[i].item()
        if m > 0:
            w = fw_weights[i].item() * m  # 实际应用的更新步长
            new_id = str(int(fw_ids[i].item()))
            
            # 原有比例衰减 (1 - w)
            for k in current_mix[i].keys():
                current_mix[i][k] *= (1.0 - w)
                
            # 加上新的 fw_id 比例
            current_mix[i][new_id] = current_mix[i].get(new_id, 0.0) + w
    return current_mix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--file_path", type=str, default="/workspace/yiqiuguo/lsrl/qwen3-1.7b_math-500_rollout8_len32768_final.jsonl")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--vllm_gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.1)
    parser.add_argument("--big_batch_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--thinking_ratio", type=float, default=0.8)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_k", type=int, default=32)
    parser.add_argument("--eval_modes", type=str, nargs="+", default=["pure", "forced", "fast"])
    parser.add_argument("--mask_strategy", type=str, default="top_k_entropy", choices=["top_k_entropy", "first_k"])
    parser.add_argument("--mask_max_k", type=float, default=32768)
    parser.add_argument("--grad_direction", type=str, default="positive", choices=["positive", "negative", "contrastive"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "frank_wolfe"])
    parser.add_argument("--fw_gamma", type=float, default=0.1)
    parser.add_argument("--fw_restrict_k", type=int, default=0, help="限制 FW 优化器在 Top-K 个候选词内寻优。")
    parser.add_argument("--conn_type", type=str, default="original", choices=["fast", "original", "on-policy"])
    parser.add_argument("--reg_type", type=str, default="lm", choices=["kl", "lm"])
    parser.add_argument("--adaptive_grad", type=str, default="off", choices=["off", "top_k", "top_p", "soft"])
    parser.add_argument("--adaptive_grad_top_k", type=int, default=128)
    parser.add_argument("--adaptive_grad_top_p", type=float, default=0.8)
    parser.add_argument("--gamma_decay", type=str, default="cosine", choices=["constant", "cosine", "fw"])
    parser.add_argument("--early_stop_correctness_threshold", type=float, default=1.01)
    parser.add_argument("--n_conn_grad", type=int, default=1)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_threshold", type=float, default=1e-3)
    parser.add_argument("--skip_distill", action="store_true")
    parser.add_argument("--skip_start_eval", action="store_true")
    parser.add_argument("--distill_epochs", type=int, default=1)
    parser.add_argument("--distill_lr", type=float, default=2e-5)
    parser.add_argument("--distill_ce_loss_weight", type=float, default=1.0)

    # [🌟 修改点3.1] 移除旧版参数，新增蒸馏类型和截断参数
    parser.add_argument("--distill_type", type=str, default="original_soft_kl", 
                        choices=["original_soft_kl", "new_soft_kl", "opsd", "conn_hard", "all_hard"])
    parser.add_argument("--max_distill_length", type=int, default=1024, help="OPSD模式下截取计算KL的最大长度")
    
    parser.add_argument("--distill_eval_every", type=int, default=99999)
    parser.add_argument("--distill_eval_datasets", type=str, nargs="+", default=["HuggingFaceH4/MATH-500"])
    parser.add_argument("--distill_eval_max_tokens", type=int, default=8192)
    parser.add_argument("--distill_batch_size", type=int, default=1)
    parser.add_argument("--distill_grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=9999999)
    
    parser.add_argument("--distill_only", action="store_true", help="跳过优化阶段，直接从本地加载数据集进行蒸馏")
    parser.add_argument("--distill_dataset_path", type=str, default="./default_distill_dataset.pt", help="蒸馏数据集读取路径")
    parser.add_argument("--skip_distill_eval", action="store_true", help="跳过蒸馏阶段的所有评测")
    
    return parser.parse_args()


# =========================================================================
# [4] 主流程
# =========================================================================
def main():
    args = parse_args()

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        wandb.init(project="L-GRPO-Math500", name=args.run_name, config=vars(args))
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
    model.requires_grad_(False)
    model.eval()

    need_vllm = not (args.distill_only and args.skip_distill_eval)
    vllm_engine = None
    if need_vllm:
        if rank == 0:
            print("🚀 初始化 vLLM Engine...")
        with torch.cuda.device(device):
            vllm_engine = LLM(
                model=args.model_name,
                trust_remote_code=True,
                tensor_parallel_size=1,
                data_parallel_size=world_size,
                distributed_executor_backend="external_launcher",
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                dtype="bfloat16",
                enable_prompt_embeds=True,
                max_model_len=32768,
                enforce_eager=True,  
            )

    def get_embeds(text):
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        return model.get_input_embeddings()(ids).detach(), ids

    embeds_fast_conn, ids_fast_conn = get_embeds("\nTherefore, the final answer is \n$$\n\\boxed{")
    all_embeddings = model.get_input_embeddings().weight.detach()

    if rank == 0: print(f"📄 读取文件: {args.file_path}")
    wrong_dataset = build_math_wrong_dataset(args.file_path, tokenizer, args.thinking_ratio, args.max_samples)
    
    all_raw_data = wrong_dataset.flat_data
    if args.max_samples is not None:
        all_raw_data = all_raw_data[: args.max_samples]

    my_raw_data = [all_raw_data[i] for i in range(len(all_raw_data)) if i % world_size == rank]
    raw_data_chunks = [my_raw_data[i : i + args.big_batch_size] for i in range(0, len(my_raw_data), args.big_batch_size)]

    global_history = {}
    if rank == 0:
        global_history = {d["uid"]: {"uid": d["uid"], "problem": d["question_text"], "gt_text": d["gt_text"], "steps": []} for d in all_raw_data}

    local_history = {d["uid"]: {"uid": d["uid"], "problem": d["question_text"], "gt_text": d["gt_text"], "steps": []} for d in my_raw_data}
    local_distill_dataset = []
    
    # [🌟 修改点1] 初始化软 token 记录的字典
    global_token_mixes = {}
    
    embeds_fast_conn_cpu = embeds_fast_conn.cpu()

    run_name = wandb.run.name if (wandb.run is not None and rank == 0) else f"opt_v2_{args.optimizer}_{args.reg_type}"
    history_filename = f"./optimization_histories/optimization_history_{run_name}.jsonl"
    if rank == 0:
        os.makedirs(os.path.dirname(history_filename), exist_ok=True)

    def sync_and_save_history():
        gathered_histories = [None] * world_size if rank == 0 else None
        dist.gather_object(local_history, gathered_histories, dst=0)
        
        if rank == 0:
            for r_history in gathered_histories:
                for uid, history_data in r_history.items():
                    global_history[uid] = history_data
            
            tmp_filename = history_filename + ".tmp"
            with open(tmp_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps({"config": vars(args)}, ensure_ascii=False) + "\n")
                for uid in global_history.keys():
                    f.write(json.dumps(global_history[uid], ensure_ascii=False) + "\n")
            os.replace(tmp_filename, history_filename)

    if not args.distill_only:
        sync_and_save_history()

    pos_big = rank * 3
    pos_step = rank * 3 + 1
    pos_mini = rank * 3 + 2

    # ==================================================
    # [🚀 同步评估逻辑] （此处维持不变，计算 ACC 和纯模型前向）
    # ==================================================
    def run_eval_sync(eval_step, eval_uids, latents_snapshot, chunk_static_data, return_valid_answers=False):
        if vllm_engine is None: return {}
        step_metrics = {uid: {} for uid in eval_uids}
        local_pure_acc, local_forced_acc, local_fast_acc, local_total_change_ratio = 0.0, 0.0, 0.0, 0.0
        valid_answers, eval_pure_inputs, eval_forced_inputs, eval_fast_inputs = {}, [], [], []
        
        if len(eval_uids) > 0:
            tqdm.write(f"\n[R{rank}] 🚀 [评测] Step {eval_step} 启动! 正在 {device} 计算潜空间漂移...")
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

                    step_metrics[uid]["change_ratio"] = change_ratio
                    step_metrics[uid]["curr_thinking_ids"] = nearest_token_ids.tolist()
                    local_total_change_ratio += change_ratio

                    embeds_q_cpu = sd["embeds_q_cpu"]
                    embeds_conn_cpu = sd["embeds_conn_cpu"]
                    eval_pure_inputs.append(torch.cat([embeds_q_cpu, ct_cpu], dim=1).squeeze(0))
                    eval_forced_inputs.append(torch.cat([embeds_q_cpu, ct_cpu, embeds_conn_cpu], dim=1).squeeze(0))
                    eval_fast_inputs.append(torch.cat([embeds_q_cpu, ct_cpu, embeds_fast_conn_cpu], dim=1).squeeze(0))

            tqdm.write(f"[R{rank}]⚡ [评测] 漂移计算完毕，准备发送数据至vLLM")
        else:
            tqdm.write(f"[R{rank}] ⚠️ Step {eval_step}: 暂无分配到数据，参与空轮转以保持 NCCL 同步...")

        modes = ["pure", "forced", "fast"]
        max_toks = {"pure": 8192, "forced": 32, "fast": 32}
        my_results = {}

        for mode in args.eval_modes:
            if mode == "pure": inputs_tensors = eval_pure_inputs
            elif mode == "forced": inputs_tensors = eval_forced_inputs
            else: inputs_tensors = eval_fast_inputs

            with torch.cuda.device(device):
                vllm_inputs = [{"prompt_embeds": emb.contiguous().cpu()} for emb in inputs_tensors]
                sp = SamplingParams(max_tokens=max_toks[mode], temperature=1.0, n=args.eval_k, skip_special_tokens=False)
                outputs = vllm_engine.generate(vllm_inputs, sampling_params=sp, use_tqdm=(rank == 0))

            my_results[mode] = []
            for out in outputs:
                my_results[mode].append([list(comp.token_ids) for comp in out.outputs])

        if len(eval_uids) > 0:
            for mode in args.eval_modes:
                tqdm.write(f"[R{rank}] 🚀 Step{eval_step} {mode}模型评测开始")
                batch_outputs = my_results[mode]
                for idx, uid in enumerate(eval_uids):
                    gt_text = local_history[uid]["gt_text"]
                    gt_token_len = len(tokenizer.encode(gt_text, add_special_tokens=False))
                    correct_count = 0

                    if len(batch_outputs) > 0:
                        for output_ids in batch_outputs[idx]:
                            if mode != "pure": output_ids = output_ids[:gt_token_len]
                            gen_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                            is_corr = False

                            if mode == "pure":
                                if is_correct_v3(gen_text, gt_text.replace("}", "")):
                                    is_corr = True
                                    if return_valid_answers and uid not in valid_answers:
                                        valid_answers[uid] = output_ids
                                else:
                                    if "wrong_answers" not in step_metrics[uid]: step_metrics[uid]["wrong_answers"] = []
                                    step_metrics[uid]["wrong_answers"].append(gen_text)
                            else:
                                ans = gen_text.replace("$", "").replace("}", "").strip()
                                if is_equiv(ans, gt_text.replace("}", "")): is_corr = True

                            if is_corr: correct_count += 1

                    acc = correct_count / args.eval_k if args.eval_k > 0 else 0
                    step_metrics[uid][f"{mode}_acc"] = acc

                    if mode == "pure" and len(batch_outputs[idx]) > 0:
                        step_metrics[uid]["sample_gen_text"] = tokenizer.decode(batch_outputs[idx][0], skip_special_tokens=True)
                    
                    if mode == "pure": local_pure_acc += acc
                    elif mode == "forced": local_forced_acc += acc
                    elif mode == "fast": local_fast_acc += acc

        local_counts = torch.tensor([local_pure_acc, local_forced_acc, local_fast_acc, local_total_change_ratio, len(eval_uids)], dtype=torch.float32, device=device)
        dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
        global_uids_count = local_counts[4].item()

        avg_metrics = {}
        if global_uids_count > 0:
            avg_metrics = {"eval/avg_pure_acc": local_counts[0].item() / global_uids_count, "eval/avg_forced_acc": local_counts[1].item() / global_uids_count, "eval/avg_fast_acc": local_counts[2].item() / global_uids_count, "eval/avg_change_ratio": local_counts[3].item() / global_uids_count}
            
        res = {"step": eval_step, "uids": eval_uids, "step_metrics": step_metrics, "avg_metrics": avg_metrics}
        if return_valid_answers: res["valid_answers"] = valid_answers
        return res    
    
    def process_eval_results(res):
        if not res: return
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

    global_early_stopped_uids = set()

    # =========================================================================
    # [🔥 分支1: 执行完整的大循环优化并生成蒸馏集]
    # =========================================================================
    if not args.distill_only:
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

                    # [🌟 修改点1] 注册最初的 Token 权重，1.0 是确定的 Original IDs
                    global_token_mixes[uid] = [{str(token_id): 1.0} for token_id in ids_think[0].tolist()]

                    think_start_idx = ids_q.shape[1] - 1
                    think_end_idx = think_start_idx + ids_think.shape[1]
                    full_emb = torch.cat([embeds_q, embeds_think, embeds_conn], dim=1)
                    batch_embeds_full.append(full_emb)
                    batch_info.append({
                        "uid": uid, "ids_q": ids_q, "ids_conn": ids_conn,
                        "ids_gt": ids_gt, "ids_pred": ids_pred, "ids_think": ids_think,
                        "thinking_text": d["thinking_text"], "start": think_start_idx,
                        "end": think_end_idx, "len": full_emb.shape[1],
                        "embeds_q": embeds_q, "embeds_conn": embeds_conn,
                    })

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

                    if 0 < args.mask_max_k < 1: real_mask_max_k = int(args.mask_max_k * probs.shape[1])
                    else: real_mask_max_k = int(args.mask_max_k)
                    
                    if args.mask_strategy == "top_k_entropy":
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                        mask = torch.zeros_like(entropy, dtype=torch.float32)
                        mask.scatter_(1, torch.topk(entropy, k=min(real_mask_max_k, entropy.shape[1]), dim=1).indices, 1.0)
                    else:
                        mask = torch.zeros((1, info["ids_think"].shape[1]), dtype=torch.float32, device=device)
                        mask[:, : real_mask_max_k] = 1.0

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

            # =========================================================================
            # [🔥 内层Step循环] Big Batch内优化步数循环
            # =========================================================================
            step_pbar = tqdm(range(args.steps), desc=f"[R{rank}] 🏃 Steps", position=pos_step, leave=False)
            for step in step_pbar:
                global_step_id = chunk_idx * args.steps + step

                local_active_count = torch.tensor([len(active_uids)], device=device, dtype=torch.int32)
                dist.all_reduce(local_active_count, op=dist.ReduceOp.SUM)
                if local_active_count.item() == 0: break

                current_lr = 0.0
                epoch_gt_loss_sum, epoch_kl_loss_sum, epoch_lm_loss_sum, epoch_total_loss_sum = 0.0, 0.0, 0.0, 0.0
                step_metrics = {uid: {} for uid in active_uids}

                if len(active_uids) > 0:
                    progress = step / max(1, args.steps - 1)
                    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    current_lr = (args.learning_rate * 0.1) + (args.learning_rate * 0.9) * cosine_factor
                    
                    if args.gamma_decay == "constant": current_fw_gamma = args.fw_gamma
                    elif args.gamma_decay == "cosine": current_fw_gamma = (args.fw_gamma * 0.1) + (args.fw_gamma * 0.9) * cosine_factor
                    elif args.gamma_decay == "fw": current_fw_gamma = 2.0 / (step + 2) * args.fw_gamma

                    if args.optimizer == "frank_wolfe":
                        optimizer = FrankWolfeOptimizer(all_embeddings, adaptive_mode=args.adaptive_grad, top_k=args.adaptive_grad_top_k, top_p=args.adaptive_grad_top_p)

                    mini_batches = [active_uids[i : i + args.batch_size] for i in range(0, len(active_uids), args.batch_size)]
                    inner_pbar = tqdm(mini_batches, desc=f"[R{rank}] 🔥 Minibatch", position=pos_mini, leave=False)

                    for batch_uids in inner_pbar:
                        curr_think_list, curr_mask_list = [], []
                        embeds_q_list, embeds_conn_list, embeds_gt_list, embeds_pred_list = [], [], [], []
                        ids_q_list, ids_think_list, ids_conn_list, ids_gt_list, ids_pred_list = [], [], [], [], []

                        for uid in batch_uids:
                            sd = static_data_cpu[uid]
                            curr_mask_list.append(sd["grad_mask"].to(device))
                            p = global_latents[uid]
                            p.data = p.data.to(device)
                            if p.grad is not None: p.grad.data = p.grad.data.to(device)
                            curr_think_list.append(p)

                            if args.optimizer == "adam":
                                opt = global_opts[uid]
                                move_optimizer_state(opt, device)
                                for pg in opt.param_groups: pg["lr"] = current_lr
                            elif args.optimizer == "frank_wolfe" and p.grad is not None:
                                p.grad.zero_()

                            with torch.no_grad():
                                embeds_q_list.append(model.get_input_embeddings()(sd["ids_q"].to(device)))
                                embeds_conn_list.append(model.get_input_embeddings()(sd["ids_conn"].to(device)))
                                embeds_gt_list.append(model.get_input_embeddings()(sd["ids_gt"].to(device)))
                                embeds_pred_list.append(model.get_input_embeddings()(sd["ids_pred"].to(device)))

                            ids_q_list.append(sd["ids_q"].to(device))
                            ids_think_list.append(sd["ids_think"].to(device))
                            ids_conn_list.append(sd["ids_conn"].to(device))
                            ids_gt_list.append(sd["ids_gt"].to(device))
                            ids_pred_list.append(sd["ids_pred"].to(device))

                        max_conns = 1
                        if args.conn_type == "on-policy":
                            max_conns = max([len(static_data_cpu[uid].get("active_conns", [(None, None)])) for uid in batch_uids])
                        batch_size_cur = len(batch_uids)
                        
                        for c_idx in range(max_conns):
                            # [省略前向传播、对比方向及 Loss 计算逻辑：维持原有，无需改动]
                            full_embeds_list_pos, full_embeds_list_neg = [], []
                            valid_indices_pos, valid_indices_neg = [], []
                            
                            for i, uid in enumerate(batch_uids):
                                sd = static_data_cpu[uid]
                                conns = sd.get("active_conns", [(sd["embeds_conn_cpu"], sd["ids_conn"])])
                                if args.grad_direction == "contrastive":
                                    emb_conn_pos = embeds_fast_conn_cpu.to(device)
                                    target_emb_pos = embeds_gt_list[i]
                                    full_emb_pos = torch.cat([embeds_q_list[i], curr_think_list[i].to(model.dtype), emb_conn_pos, target_emb_pos], dim=1)
                                    full_embeds_list_pos.append(full_emb_pos)
                                    valid_indices_pos.append(i)
                                    if c_idx < len(conns):
                                        emb_conn_neg, _ = conns[c_idx]
                                        full_emb_neg = torch.cat([embeds_q_list[i], curr_think_list[i].to(model.dtype), emb_conn_neg.to(device), embeds_pred_list[i]], dim=1)
                                        full_embeds_list_neg.append(full_emb_neg)
                                        valid_indices_neg.append(i)
                                else:
                                    if c_idx < len(conns):
                                        emb_conn, _ = conns[c_idx]
                                        target_emb = embeds_gt_list[i] if args.grad_direction == "positive" else embeds_pred_list[i]
                                        full_emb = torch.cat([embeds_q_list[i], curr_think_list[i].to(model.dtype), emb_conn.to(device), target_emb], dim=1)
                                        full_embeds_list_pos.append(full_emb)
                                        valid_indices_pos.append(i)
                            
                            if not full_embeds_list_pos: continue

                            def pad_and_get_mask(emb_list):
                                max_len = max(emb.shape[1] for emb in emb_list)
                                padded_list, mask_list = [], []
                                for emb in emb_list:
                                    if emb.shape[1] < max_len:
                                        pad = torch.zeros((emb.shape[0], max_len - emb.shape[1], emb.shape[2]), device=device, dtype=model.dtype)
                                        padded_list.append(torch.cat([emb, pad], dim=1))
                                        m = torch.zeros((emb.shape[0], max_len), device=device, dtype=torch.long)
                                        m[:, :emb.shape[1]] = 1
                                        mask_list.append(m)
                                    else:
                                        padded_list.append(emb)
                                        mask_list.append(torch.ones((emb.shape[0], max_len), device=device, dtype=torch.long))
                                return torch.cat(padded_list, dim=0), torch.cat(mask_list, dim=0)

                            full_embeds_batch_pos, full_attention_mask_pos = pad_and_get_mask(full_embeds_list_pos)
                            
                            if step == 0:
                                with torch.no_grad():
                                    last_hidden_pos = model.model(inputs_embeds=full_embeds_batch_pos, attention_mask=full_attention_mask_pos).last_hidden_state
                                last_hidden_detached_pos = last_hidden_pos.detach()
                            else:
                                last_hidden_pos = model.model(inputs_embeds=full_embeds_batch_pos, attention_mask=full_attention_mask_pos).last_hidden_state
                                last_hidden_detached_pos = last_hidden_pos.detach().requires_grad_(True)

                            last_hidden_neg, last_hidden_detached_neg = None, None
                            if args.grad_direction == "contrastive" and full_embeds_list_neg:
                                full_embeds_batch_neg, full_attention_mask_neg = pad_and_get_mask(full_embeds_list_neg)
                                if step == 0:
                                    with torch.no_grad():
                                        last_hidden_neg = model.model(inputs_embeds=full_embeds_batch_neg, attention_mask=full_attention_mask_neg).last_hidden_state
                                    last_hidden_detached_neg = last_hidden_neg.detach()
                                else:
                                    last_hidden_neg = model.model(inputs_embeds=full_embeds_batch_neg, attention_mask=full_attention_mask_neg).last_hidden_state
                                    last_hidden_detached_neg = last_hidden_neg.detach().requires_grad_(True)

                            for i, uid in enumerate(batch_uids):
                                if i not in valid_indices_pos: continue
                                local_idx_pos = valid_indices_pos.index(i)
                                sd = static_data_cpu[uid]
                                conns = sd.get("active_conns", [(sd["embeds_conn_cpu"], sd["ids_conn"])])
                                num_conns = len(conns)

                                if args.grad_direction == "contrastive":
                                    id_conn_pos = ids_fast_conn.to(device)
                                    target_ids_pos = ids_gt_list[i]
                                else:
                                    _, id_conn_pos = conns[c_idx]
                                    id_conn_pos = id_conn_pos.to(device)
                                    target_ids_pos = ids_gt_list[i] if args.grad_direction == "positive" else ids_pred_list[i]

                                gt_pos = ids_q_list[i].shape[1] + ids_think_list[i].shape[1] + id_conn_pos.shape[1] - 1
                                target_logits_pos = model.lm_head(last_hidden_detached_pos[[local_idx_pos], gt_pos : gt_pos + target_ids_pos.shape[1], :])
                                
                                gt_loss_raw = F.cross_entropy(target_logits_pos.view(-1, target_logits_pos.size(-1)), target_ids_pos.view(-1))
                                gt_scale_factor = 1.0 / (num_conns * batch_size_cur)
                                if args.grad_direction == "negative": 
                                    gt_scale_factor = -gt_scale_factor

                                if step > 0:
                                    (gt_loss_raw * gt_scale_factor).backward()
                                    
                                s_gt_val = gt_loss_raw.item() / num_conns

                                think_start = ids_q_list[i].shape[1] - 1
                                think_len = ids_think_list[i].shape[1]
                                reg_loss_val = 0.0

                                if args.reg_type == "kl":
                                    kl_sum_val = 0.0
                                    kl_scale_factor = args.kl_weight / (think_len * num_conns * batch_size_cur)
                                    orig_probs_topk, orig_indices_topk = sd["topk_probs"].to(device), sd["topk_indices"].to(device)
                                    for c_start in range(0, think_len, args.chunk_size):
                                        c_end = min(c_start + args.chunk_size, think_len)
                                        h_chunk = last_hidden_detached_pos[[local_idx_pos], think_start + c_start : think_start + c_end, :]
                                        logits_chunk = model.lm_head(h_chunk)
                                        lse_chunk = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)
                                        orig_p_chunk = orig_probs_topk[:, c_start:c_end, :]
                                        orig_idx_chunk = orig_indices_topk[:, c_start:c_end, :]
                                        curr_log_probs_topk_chunk = torch.gather(logits_chunk, -1, orig_idx_chunk) - lse_chunk
                                        kl_chunk_sum = (orig_p_chunk * (torch.log(orig_p_chunk + 1e-10) - curr_log_probs_topk_chunk)).sum()
                                        if step > 0: (kl_chunk_sum * kl_scale_factor).backward()
                                        kl_sum_val += kl_chunk_sum.item()
                                    reg_loss_val = kl_sum_val / think_len

                                elif args.reg_type == "lm":
                                    with torch.no_grad():
                                        curr_think_hard_ids = torch.empty((1, think_len), dtype=torch.long, device=device)
                                        curr_think_norm = F.normalize(curr_think_list[i], p=2, dim=-1)
                                        for c_start in range(0, think_len, args.chunk_size):
                                            c_end = min(c_start + args.chunk_size, think_len)
                                            chunk_norm = curr_think_norm[:, c_start:c_end, :]
                                            sim_chunk = torch.matmul(chunk_norm, all_embeddings.T.to(device))
                                            curr_think_hard_ids[:, c_start:c_end] = torch.argmax(sim_chunk, dim=-1)
                                    
                                    lm_loss_sum_val = 0.0
                                    lm_scale_factor = args.kl_weight / (max(1, think_len - 1) * num_conns * batch_size_cur)
                                    
                                    for c_start in range(0, think_len - 1, args.chunk_size):
                                        c_end = min(c_start + args.chunk_size, think_len - 1)
                                        h_chunk = last_hidden_detached_pos[[local_idx_pos], think_start + c_start : think_start + c_end, :]
                                        logits_chunk = model.lm_head(h_chunk)
                                        labels_chunk = curr_think_hard_ids[:, c_start : c_end]
                                        
                                        loss_chunk_sum = F.cross_entropy(
                                            logits_chunk.reshape(-1, logits_chunk.size(-1)), 
                                            labels_chunk.reshape(-1), 
                                            reduction='sum'
                                        )
                                        
                                        if step > 0:
                                            (loss_chunk_sum * lm_scale_factor).backward()
                                            
                                        lm_loss_sum_val += loss_chunk_sum.item()
                                        
                                    reg_loss_val = lm_loss_sum_val / max(1, think_len - 1)

                                neg_loss_val = 0.0
                                if args.grad_direction == "contrastive" and i in valid_indices_neg:
                                    local_idx_neg = valid_indices_neg.index(i)
                                    _, id_conn_neg = conns[c_idx]
                                    id_conn_neg = id_conn_neg.to(device)
                                    id_pred_neg = ids_pred_list[i]
                                    neg_target_ids = torch.cat([id_conn_neg, id_pred_neg], dim=1)
                                    neg_start_idx = ids_q_list[i].shape[1] + ids_think_list[i].shape[1] - 1
                                    neg_total_len = neg_target_ids.shape[1]
                                    neg_scale_factor = -1.0 / (neg_total_len * num_conns * batch_size_cur)
                                    neg_loss_sum_val = 0.0
                                    for c_start in range(0, neg_total_len, args.chunk_size):
                                        c_end = min(c_start + args.chunk_size, neg_total_len)
                                        h_chunk_neg = last_hidden_detached_neg[[local_idx_neg], neg_start_idx + c_start : neg_start_idx + c_end, :]
                                        logits_chunk_neg = model.lm_head(h_chunk_neg)
                                        labels_chunk_neg = neg_target_ids[:, c_start : c_end]
                                        loss_chunk_neg_sum = F.cross_entropy(logits_chunk_neg.reshape(-1, logits_chunk_neg.size(-1)), labels_chunk_neg.reshape(-1), reduction='sum')
                                        if step > 0: (loss_chunk_neg_sum * neg_scale_factor).backward()
                                        neg_loss_sum_val += loss_chunk_neg_sum.item()
                                    neg_loss_val = - (neg_loss_sum_val / neg_total_len)

                                s_total_val = s_gt_val + (neg_loss_val / num_conns) + args.kl_weight * (reg_loss_val / num_conns)
                                if "total_loss" not in step_metrics[uid]: step_metrics[uid].update({"total_loss": 0, "gt_loss": 0, "neg_loss": 0, "reg_loss": 0})
                                step_metrics[uid]["total_loss"] += s_total_val
                                step_metrics[uid]["gt_loss"] += s_gt_val
                                step_metrics[uid]["neg_loss"] += (neg_loss_val / num_conns)
                                step_metrics[uid]["reg_loss"] += (reg_loss_val / num_conns)
                                epoch_gt_loss_sum += s_gt_val
                                epoch_total_loss_sum += s_total_val

                            if step > 0:
                                last_hidden_pos.backward(last_hidden_detached_pos.grad)
                                if args.grad_direction == "contrastive" and last_hidden_neg is not None and last_hidden_detached_neg.grad is not None:
                                    last_hidden_neg.backward(last_hidden_detached_neg.grad)
                            
                            del last_hidden_pos, last_hidden_detached_pos, full_embeds_batch_pos, full_attention_mask_pos
                            if args.grad_direction == "contrastive":
                                del last_hidden_neg, last_hidden_detached_neg
                                if 'full_embeds_batch_neg' in locals():
                                    del full_embeds_batch_neg, full_attention_mask_neg

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
                                if args.fw_restrict_k > 0:
                                    restrict_k = min(args.fw_restrict_k, sd["topk_indices"].shape[-1])
                                    current_topk_indices = sd["topk_indices"][:, :, :restrict_k].to(device)
                                else:
                                    current_topk_indices = None
                                
                                fw_ids, fw_weights = optimizer.step(p, gamma=current_fw_gamma, valid_token_indices=current_topk_indices)
                                p.grad = None
                                
                                # [🌟 修改点1] 实时记录与更新 Token 混合分布
                                global_token_mixes[uid] = update_token_mix(
                                    current_mix=global_token_mixes[uid],
                                    fw_ids=fw_ids.squeeze(0).cpu(),
                                    fw_weights=fw_weights.squeeze(0).cpu(),
                                    mask=mask.squeeze(-1).squeeze(0).cpu()
                                )

                            p.data = p.data.cpu().pin_memory()
                            
                            # 保存 FW action 到日志
                            if fw_ids is not None:
                                step_metrics[uid]["fw_action_ids"] = fw_ids.cpu().tolist()
                                step_metrics[uid]["fw_action_weights"] = fw_weights.cpu().tolist()
                                step_metrics[uid]["curr_token_mix"] = global_token_mixes[uid]

                    next_active_uids = []
                    for uid in active_uids:
                        local_history[uid]["steps"].append({"step": global_step_id, "metrics": step_metrics[uid]})
                        if args.grad_direction in ["positive","contrastive"]: is_gt_converged = args.early_stop and step_metrics[uid]["gt_loss"] < args.early_stop_threshold
                        else: is_gt_converged = False
                        is_pure_acc_converged = uid in global_early_stopped_uids
                        if not (is_gt_converged or is_pure_acc_converged): next_active_uids.append(uid)
                            
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

                start_eval = (step > 0) if args.skip_start_eval else True
                if step % args.eval_every == 0 and step != args.steps - 1 and start_eval:
                    eval_uids = list(active_uids) if len(active_uids) > 0 else []
                    latents_snapshot = {uid: global_latents[uid].detach().clone() for uid in eval_uids}
                    res = run_eval_sync(global_step_id, eval_uids, latents_snapshot, static_data_cpu)
                    process_eval_results(res)
                    sync_and_save_history()
                    
                    if res is not None and res:
                        for uid in eval_uids:
                            metrics = res["step_metrics"][uid]
                            if metrics.get("pure_acc", 0.0) >= args.early_stop_correctness_threshold:
                                if uid not in global_early_stopped_uids:
                                    global_early_stopped_uids.add(uid)
                                    tqdm.write(f"[R{rank}] 🛑 UID {uid} Pure Acc 达标，提前终止优化！")
                            
                            if args.conn_type == "on-policy" and uid not in global_early_stopped_uids:
                                wrongs = metrics.get("wrong_answers", [])
                                if len(wrongs) > 0:
                                    new_conns = []
                                    for w_text in wrongs:
                                        if len(new_conns) >= args.n_conn_grad: break
                                        try:
                                            after_thinking_text = w_text
                                            if "\\boxed{" in after_thinking_text:
                                                pred_box_text = last_boxed_only_string(after_thinking_text)
                                                connector_text = after_thinking_text.split(pred_box_text)[0] + "\\boxed{"
                                                emb_c, id_c = get_embeds(connector_text)
                                                new_conns.append((emb_c.cpu(), id_c.cpu()))
                                        except Exception:
                                            pass
                                    if len(new_conns) > 0:
                                        static_data_cpu[uid]["active_conns"] = new_conns

            chunk_all_uids = [d["uid"] for d in current_raw_data]
            final_latents_cpu = {uid: global_latents[uid].detach().cpu() for uid in chunk_all_uids}

            final_eval_result = run_eval_sync(global_step_id, chunk_all_uids, final_latents_cpu, static_data_cpu, return_valid_answers=True)
            process_eval_results(final_eval_result)
            sync_and_save_history()
            valid_answers = final_eval_result.get("valid_answers", {})

            # [🌟 修改点2] 精简并改造保存给蒸馏数据集的数据
            for uid, best_output_ids in valid_answers.items():
                sd = static_data_cpu[uid]
                
                # 为 OPSD 预处理 Token
                p1_text = f"Problem: {local_history[uid]['problem']}\nHere is a reference thinking process:\n"
                p2_text = "\nAfter understanding the reference thinking process, please try to solve this problem using your own approach below:\nAnswer:\n"
                opsd_p1_ids = tokenizer.encode(p1_text, return_tensors="pt", add_special_tokens=False).cpu()
                opsd_p2_ids = tokenizer.encode(p2_text, return_tensors="pt", add_special_tokens=False).cpu()

                local_distill_dataset.append(
                    {
                        "uid": uid,
                        "gt_text": local_history[uid]["gt_text"],
                        "problem": local_history[uid]["problem"],
                        "distill_data": {
                            "ids_q": sd["ids_q"].cpu(),
                            "ids_orig_think": sd["ids_think"].cpu(),
                            "ids_orig_conn": sd["ids_conn"].cpu(),
                            "ids_orig_pred": sd["ids_pred"].cpu(),     # 用于 OPSD
                            "ids_new_ans": torch.tensor([best_output_ids], dtype=torch.long).cpu(),
                            "think_token_mixes": global_token_mixes[uid],  # [🌟] 保存混合轨迹
                            "opsd_p1_ids": opsd_p1_ids,                    # [🌟] OPSD用 Prompt 1
                            "opsd_p2_ids": opsd_p2_ids                     # [🌟] OPSD用 Prompt 2
                        }
                    }
                )

            del static_data_cpu, global_latents, final_latents_cpu
            torch.cuda.empty_cache()
            
        dist.barrier(device_ids=[local_rank])
        if args.skip_distill:
            if rank == 0: wandb.finish()
            return

        all_distill = [None] * world_size
        dist.all_gather_object(all_distill, local_distill_dataset)
        if rank == 0:
            global_distill_dataset = [item for sublist in all_distill for item in sublist]
            distill_dataset_save_path = f"./distill_datasets/{run_name}/distill_dataset.pt"
            os.makedirs(os.path.dirname(distill_dataset_save_path), exist_ok=True)
            print(f"💾 [Rank 0] 正在持久化保存蒸馏数据集至: {distill_dataset_save_path}")
            torch.save(global_distill_dataset, distill_dataset_save_path)
        else:
            global_distill_dataset = [None]
        dataset_container = [global_distill_dataset]
        dist.broadcast_object_list(dataset_container, src=0)
        global_distill_dataset = dataset_container[0]

    # =========================================================================
    # [🔥 分支2: 直接加载本地预存数据集进行蒸馏 (离线蒸馏)]
    # =========================================================================
    else:
        if rank == 0:
            print(f"📥 [Rank 0] 检测到 distill_only 模式，正在从本地加载蒸馏数据集: {args.distill_dataset_path}")
            global_distill_dataset = torch.load(args.distill_dataset_path, map_location="cpu")
        else:
            global_distill_dataset = [None]
        dataset_container = [global_distill_dataset]
        dist.broadcast_object_list(dataset_container, src=0)
        global_distill_dataset = dataset_container[0]

    if rank == 0: print(f"\n🔥 启动 FSDP 模型蒸馏阶段... 全局总样本数: {len(global_distill_dataset)}")
    if len(global_distill_dataset) == 0:
        if rank == 0: wandb.finish()
        return

    # =========================================================================
    # [2] 准备评测数据集 (跳过评测时仍按原逻辑解析但不会执行)
    # =========================================================================
    def pass_at_k(n, c, k):
        if n - c < k: return 1.0
        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

    eval_datasets = {}
    for ds_id in args.distill_eval_datasets:
        try:
            if "MATH-500" in ds_id: ds, q_key, a_key = load_dataset(ds_id, split="test"), "problem", "answer"
            elif "amc23" in ds_id: ds, q_key, a_key = load_dataset(ds_id, split="test"), "question", "answer"
            elif "gsm8k" in ds_id: ds, q_key, a_key = load_dataset(ds_id, "main", split="test"), "question", "answer"
            elif "aime_2024" in ds_id: ds, q_key, a_key = load_dataset(ds_id, split="train"), "problem", "answer"
            elif "aime25" in ds_id: ds, q_key, a_key = load_dataset(ds_id, split="test"), "problem", "answer"
            else: continue
            eval_datasets[ds_id] = {"data": ds, "q_key": q_key, "a_key": a_key}
        except Exception: pass

    # =========================================================================
    # [3] 定义同步评测函数
    # =========================================================================
    def run_distill_eval_sync(step, current_state_dict_path):
        if vllm_engine is None: return
        if rank == 0: print(f"🔄 [Rank 0] 加载权重至 vLLM: {current_state_dict_path}")
        state_dict = torch.load(current_state_dict_path, map_location="cpu")
        executor = vllm_engine.llm_engine.model_executor
        if hasattr(executor, "driver_worker"): vllm_model = executor.driver_worker.model_runner.model
        else: vllm_model = executor.model_runner.model
        vllm_model.load_weights(state_dict.items())
        del state_dict
        torch.cuda.empty_cache()

        metrics_to_log = {"model_train_step": step}
        total_pass1, total_pass8 = [], []

        for ds_id, ds_info in eval_datasets.items():
            ds, q_key, a_key = ds_info["data"], ds_info["q_key"], ds_info["a_key"]
            is_pre_formatted = ds_info.get("pre_formatted", False)
            current_n = 8 if any(x in ds_id for x in ["aime_2024", "aime25", "amc23"]) else 1

            prompts = []
            for item in ds:
                if is_pre_formatted: prompts.append(item[q_key])
                else: prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": item[q_key]}], tokenize=False, add_generation_prompt=True)+"<think>")

            sp = SamplingParams(max_tokens=8192, temperature=1.0, n=current_n, top_p=0.95, skip_special_tokens=False)
            outputs = vllm_engine.generate(prompts, sampling_params=sp, use_tqdm=(rank == 0))
            if rank == 0:
                ds_pass1, ds_pass8 = 0.0, 0.0
                saved_data_list = []

                for idx, item in enumerate(ds):
                    gt_ans = str(item[a_key]).strip()
                    correct_count = 0
                    
                    responses = []
                    out_ids_batch = [list(comp.token_ids) for comp in outputs[idx].outputs]
                    for output_ids in out_ids_batch:
                        gen_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                        responses.append(gen_text)
                        ans_cand = last_boxed_only_string(gen_text)
                        if ans_cand and is_equiv(remove_boxed(ans_cand), gt_ans):
                            correct_count += 1

                    ds_pass1 += pass_at_k(current_n, correct_count, 1)
                    if current_n >= 8: ds_pass8 += pass_at_k(current_n, correct_count, 8)
                        
                    row_data = dict(item)
                    row_data["responses"] = responses
                    saved_data_list.append(row_data)

                num_samples = len(ds)
                metrics_to_log[f"distill/eval/{ds_id}/pass@1"] = ds_pass1 / num_samples
                total_pass1.append(ds_pass1 / num_samples)
                if current_n >= 8:
                    metrics_to_log[f"distill/eval/{ds_id}/pass@8"] = ds_pass8 / num_samples
                    total_pass8.append(ds_pass8 / num_samples)
                    
                safe_model_name = args.model_name.replace("/", "_")
                safe_ds_id = ds_id.replace("/", "_")
                current_run_name = run_name if 'run_name' in globals() else args.run_name
                
                save_dir = f"./gen_results/{current_run_name}"
                os.makedirs(save_dir, exist_ok=True)
                out_filename = os.path.join(save_dir, f"{safe_model_name}_{safe_ds_id}_rollout{current_n}_{current_run_name}_distill_step_{step}.jsonl")
                
                with open(out_filename, "w", encoding="utf-8") as f:
                    for row in saved_data_list:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                
                print(f"💾 [Rank 0] 数据集 {ds_id} 的生成结果已保存至: {out_filename}")

        if rank == 0:
            if total_pass1: metrics_to_log["distill/eval/avg_pass@1"] = sum(total_pass1) / len(total_pass1)
            if total_pass8: metrics_to_log["distill/eval/avg_pass@8"] = sum(total_pass8) / len(total_pass8)
            wandb.log(metrics_to_log)
            print(f"✅ [Rank 0] 蒸馏评测完成，Avg Pass@1: {metrics_to_log.get('distill/eval/avg_pass@1', 0):.2%}")
        
        dist.barrier(device_ids=[local_rank])

    # =========================================================================
    # [4] FSDP ZeRO-2 初始化 & 工具函数
    # =========================================================================
    import functools
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp.api import ShardingStrategy
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

    # [🌟 修改点3] 获取冻结的完整 embedding，供实时构建 Soft Embeds
    with torch.no_grad():
        frozen_embed_layer = model.get_input_embeddings().weight.detach().clone().to(device)

    def mix_to_soft_embeds(token_mixes):
        """将 token_mixes 转换回 Soft Embeddings (1, seq_len, hidden_dim)"""
        seq_len = len(token_mixes)
        dim = frozen_embed_layer.shape[1]
        soft_embs = torch.zeros((1, seq_len, dim), device=device, dtype=model.dtype)
        for i, mix in enumerate(token_mixes):
            for token_id_str, weight in mix.items():
                tid = int(token_id_str)
                soft_embs[0, i] += weight * frozen_embed_layer[tid]
        return soft_embs

    def mix_to_hard_ids(token_mixes):
        """将 token_mixes 转换回 Hard IDs (argmax weight)"""
        hard_ids = []
        for mix in token_mixes:
            best_id = max(mix, key=mix.get)
            hard_ids.append(int(best_id))
        return torch.tensor([hard_ids], dtype=torch.long, device=device)

    model.config.use_cache = False
    model.requires_grad_(True)
    
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
        model.get_input_embeddings().requires_grad_(False)
    if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        model.get_output_embeddings().requires_grad_(False)
    elif hasattr(model, "lm_head") and model.lm_head is not None:
        model.lm_head.requires_grad_(False)
        
    model.train()
    model.gradient_checkpointing_enable()

    transformer_layer_cls = model.model.layers[0].__class__
    my_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={transformer_layer_cls})

    fsdp_model = FSDP(
        model, 
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        auto_wrap_policy=my_auto_wrap_policy,
        use_orig_params=True,
        sync_module_states=True,
    )
    
    trainable_params = [p for p in fsdp_model.parameters() if p.requires_grad]
    distill_optimizer = torch.optim.AdamW(trainable_params, lr=args.distill_lr)

    sampler = DistributedSampler(global_distill_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    def distill_collate_fn(batch): return batch
    dataloader = DataLoader(global_distill_dataset, batch_size=args.distill_batch_size, sampler=sampler, collate_fn=distill_collate_fn)

    shm_model_path = "/dev/shm/distill_model_weights.pt"
    distill_step = 0
    save_options = StateDictOptions(full_state_dict=True, cpu_offload=True)

    if not args.skip_distill_eval and not args.skip_start_eval:
        cpu_state_dict = get_model_state_dict(fsdp_model, options=save_options)
        if rank == 0: torch.save(cpu_state_dict, shm_model_path)
        dist.barrier(device_ids=[local_rank]) 
        run_distill_eval_sync(distill_step, shm_model_path)
    
    # =========================================================================
    # [5] 蒸馏训练大循环 - 全新多模态支持
    # =========================================================================
    for epoch in range(args.distill_epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"[R{rank}] Distill Epoch {epoch + 1}/{args.distill_epochs}", leave=False)

        distill_optimizer.zero_grad()
        for b_idx, batch in enumerate(pbar):
            batch_loss = 0.0

            for item in batch:
                d_data = item["distill_data"]
                ids_q = d_data["ids_q"].to(device)
                ids_orig_think = d_data["ids_orig_think"].to(device)
                ids_new_ans = d_data["ids_new_ans"].to(device)
                think_mixes = d_data["think_token_mixes"]

                # [🌟 修改点3] 五种蒸馏模式前向计算分支
                if args.distill_type == "original_soft_kl":
                    soft_think_embeds = mix_to_soft_embeds(think_mixes)
                    q_embeds = frozen_embed_layer[ids_q.squeeze(0)].unsqueeze(0)
                    
                    with torch.no_grad():
                        teacher_embeds = torch.cat([q_embeds, soft_think_embeds], dim=1)
                        logits_teacher = fsdp_model(inputs_embeds=teacher_embeds, use_cache=False).logits
                        
                    student_ids = torch.cat([ids_q, ids_orig_think], dim=1)
                    logits_student = fsdp_model(input_ids=student_ids, use_cache=False).logits

                    # 计算 KL
                    t_start = ids_q.shape[1] - 1
                    t_end = student_ids.shape[1] - 1
                    pred_log_probs = F.log_softmax(logits_student[0, t_start:t_end, :], dim=-1)
                    target_probs = F.softmax(logits_teacher[0, t_start:t_end, :], dim=-1)
                    total_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

                elif args.distill_type == "new_soft_kl":
                    soft_think_embeds = mix_to_soft_embeds(think_mixes)
                    q_embeds = frozen_embed_layer[ids_q.squeeze(0)].unsqueeze(0)
                    
                    with torch.no_grad():
                        teacher_embeds = torch.cat([q_embeds, soft_think_embeds], dim=1)
                        logits_teacher = fsdp_model(inputs_embeds=teacher_embeds, use_cache=False).logits
                        
                    hard_think_ids = mix_to_hard_ids(think_mixes)
                    student_ids = torch.cat([ids_q, hard_think_ids], dim=1)
                    logits_student = fsdp_model(input_ids=student_ids, use_cache=False).logits

                    t_start = ids_q.shape[1] - 1
                    t_end = student_ids.shape[1] - 1
                    pred_log_probs = F.log_softmax(logits_student[0, t_start:t_end, :], dim=-1)
                    target_probs = F.softmax(logits_teacher[0, t_start:t_end, :], dim=-1)
                    total_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

                elif args.distill_type == "opsd":
                    ids_orig_conn = d_data["ids_orig_conn"].to(device)
                    ids_orig_pred = d_data["ids_orig_pred"].to(device)
                    p1_ids = d_data["opsd_p1_ids"].to(device)
                    p2_ids = d_data["opsd_p2_ids"].to(device)
                    
                    soft_think_embeds = mix_to_soft_embeds(think_mixes)
                    p1_embeds = frozen_embed_layer[p1_ids.squeeze(0)].unsqueeze(0)
                    p2_embeds = frozen_embed_layer[p2_ids.squeeze(0)].unsqueeze(0)
                    orig_think_embeds = frozen_embed_layer[ids_orig_think.squeeze(0)].unsqueeze(0)
                    conn_embeds = frozen_embed_layer[ids_orig_conn.squeeze(0)].unsqueeze(0)
                    pred_embeds = frozen_embed_layer[ids_orig_pred.squeeze(0)].unsqueeze(0)
                    
                    # Teacher: P1 + Soft + P2 + Orig_Think + Conn + Pred
                    with torch.no_grad():
                        teacher_embeds = torch.cat([p1_embeds, soft_think_embeds, p2_embeds, orig_think_embeds, conn_embeds, pred_embeds], dim=1)
                        logits_teacher = fsdp_model(inputs_embeds=teacher_embeds, use_cache=False).logits
                        
                    # Student: Q + Orig_Think + Conn + Pred
                    student_ids = torch.cat([ids_q, ids_orig_think, ids_orig_conn, ids_orig_pred], dim=1)
                    logits_student = fsdp_model(input_ids=student_ids, use_cache=False).logits

                    # 截取对齐后的尾部序列（Orig_Think + Conn + Pred）
                    tail_len = ids_orig_think.shape[1] + ids_orig_conn.shape[1] + ids_orig_pred.shape[1]
                    distill_len = min(tail_len, args.max_distill_length)
                    
                    # Student 结尾偏移
                    s_end = student_ids.shape[1] - 1
                    s_start = s_end - distill_len
                    # Teacher 结尾偏移
                    t_end = teacher_embeds.shape[1] - 1
                    t_start = t_end - distill_len

                    pred_log_probs = F.log_softmax(logits_student[0, s_start:s_end, :], dim=-1)
                    target_probs = F.softmax(logits_teacher[0, t_start:t_end, :], dim=-1)
                    total_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

                elif args.distill_type == "conn_hard":
                    soft_think_embeds = mix_to_soft_embeds(think_mixes)
                    q_embeds = frozen_embed_layer[ids_q.squeeze(0)].unsqueeze(0)
                    ans_embeds = frozen_embed_layer[ids_new_ans.squeeze(0)].unsqueeze(0)
                    
                    student_embeds = torch.cat([q_embeds, soft_think_embeds, ans_embeds], dim=1)
                    logits_student = fsdp_model(inputs_embeds=student_embeds, use_cache=False).logits
                    
                    # 仅在 new_answer 处计算 CE
                    ans_start = student_embeds.shape[1] - ids_new_ans.shape[1] - 1
                    ans_end = student_embeds.shape[1] - 1
                    total_loss = F.cross_entropy(logits_student[0, ans_start:ans_end, :], ids_new_ans[0])

                elif args.distill_type == "all_hard":
                    hard_think_ids = mix_to_hard_ids(think_mixes)
                    student_ids = torch.cat([ids_q, hard_think_ids, ids_new_ans], dim=1)
                    logits_student = fsdp_model(input_ids=student_ids, use_cache=False).logits
                    
                    # 在 Hard Think + New Answer 处计算 CE
                    target_ids = torch.cat([hard_think_ids, ids_new_ans], dim=1)
                    t_start = ids_q.shape[1] - 1
                    t_end = student_ids.shape[1] - 1
                    total_loss = F.cross_entropy(logits_student[0, t_start:t_end, :], target_ids[0])

                (total_loss / (len(batch) * args.distill_grad_accum_steps)).backward()
                batch_loss += total_loss.item()

            if (b_idx + 1) % args.distill_grad_accum_steps == 0 or (b_idx + 1) == len(dataloader):
                fsdp_model.clip_grad_norm_(1.0)
                distill_optimizer.step()
                distill_optimizer.zero_grad()
                distill_step += 1

                local_metrics = torch.tensor([batch_loss], device=device)
                dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)

                if rank == 0:
                    wandb.log({
                        "model_train_step": distill_step,
                        "distill/train/total_loss": local_metrics[0].item() / world_size,
                        "distill/train/lr": distill_optimizer.param_groups[0]["lr"],
                    })
                
                if not args.skip_distill_eval and distill_step % args.distill_eval_every == 0:
                    cpu_state_dict = get_model_state_dict(fsdp_model, options=save_options)
                    if rank == 0: torch.save(cpu_state_dict, shm_model_path)
                    dist.barrier(device_ids=[local_rank]) 
                    run_distill_eval_sync(distill_step, shm_model_path)

    # =========================================================================
    # [6] 终局保存与清理
    # =========================================================================
    cpu_state_dict = get_model_state_dict(fsdp_model, options=save_options)
    if rank == 0: torch.save(cpu_state_dict, shm_model_path)
    
    dist.barrier(device_ids=[local_rank])
    if not args.skip_distill_eval: run_distill_eval_sync(distill_step, shm_model_path)

    if rank == 0:
        save_dir = f"./checkpoints/{run_name}/step{distill_step}/"
        os.makedirs(save_dir, exist_ok=True)
        print(f"💾 [Rank 0] 正在将模型序列化到硬盘...")
        model.save_pretrained(save_dir, state_dict=cpu_state_dict)
        tokenizer.save_pretrained(save_dir)
        del cpu_state_dict
        print(f"🎉 蒸馏全流程结束！模型权重已保存到 {save_dir}")
        wandb.finish()
        if os.path.exists(shm_model_path): os.remove(shm_model_path)

    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()

if __name__ == "__main__":
    main()