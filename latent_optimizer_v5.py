import os
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
            messages = [{"role": "user", "content": sample['problem']}]
            question_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            gt_text = sample['gold_answer'].strip() + "}"
            
            for response_idx, wrong_response in enumerate(sample['complete_wrong_responses']):
                uid = f"sample{sample_idx}_resp{response_idx}"
                try:
                    parts = wrong_response.split("</think>")
                    thinking_text = parts[0]
                    after_thinking_text = parts[1] if len(parts) > 1 else ""
                    assert "\\boxed{" in after_thinking_text, f"[{uid}] 缺少 boxed 结果"
                    
                    pred_box_text = last_boxed_only_string(after_thinking_text) 
                    connector_text = after_thinking_text.split(pred_box_text)[0] + "\\boxed{"
                    pred_text = remove_boxed(pred_box_text) + "}"
                    
                    self.flat_data.append({
                        "uid": uid, "question_text": question_text,
                        "answer_text": wrong_response, "thinking_text": thinking_text,
                        "connector_text": connector_text, "pred_text": pred_text,
                        "gt_text": gt_text
                    })
                    index += 1
                except Exception:
                    pass

    def __len__(self): return len(self.flat_data)
    def __getitem__(self, idx): return self.flat_data[idx]

def build_math_wrong_dataset(file_path: str, tok: AutoTokenizer) -> MathWrongDataset:
    print(f"📄 读取文件: {file_path}")
    results = [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]
    
    new_wrong_data = []
    for item in tqdm(results, desc="Processing Data"):
        gold_answer = item.get('answer', item.get('ground_truth', None))
        responses = item.get('responses', [])
        if not any(is_correct_v3(p, gold_answer) for p in responses):
            complete_but_wrong_responses = [
                res for res in responses 
                if last_boxed_only_string(res) and not math_verify_verify(math_verify_parse(remove_boxed(last_boxed_only_string(res))), math_verify_parse(gold_answer.strip()))
            ]
            if complete_but_wrong_responses:
                new_wrong_data.append({
                    "problem": item.get('q', None), "gold_answer": gold_answer,
                    "complete_wrong_responses": complete_but_wrong_responses
                })
    return MathWrongDataset(new_wrong_data, tok)

# =========================================================================
# [2] 优化器定义
# =========================================================================
class FrankWolfeOptimizer:
    def __init__(self, vocab_embeddings):
        self.W_emb = vocab_embeddings 

    def step(self, latent_tensor, gamma):
        if latent_tensor.grad is None: return
        grad = latent_tensor.grad.to(self.W_emb.device).detach()
        with torch.no_grad():
            scores = torch.matmul(grad, self.W_emb.T)
            best_vocab_indices = torch.argmin(scores, dim=-1)
            best_embeds = self.W_emb[best_vocab_indices].to(latent_tensor.device)
            latent_tensor.copy_((1 - gamma) * latent_tensor + gamma * best_embeds)
            latent_tensor.grad.zero_()

def quantize_to_int8(tensor):
    scale = tensor.abs().max() / 127.0
    int8_tensor = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return int8_tensor, scale

# =========================================================================
# [3] 主流程
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--vllm_gpus", type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_k", type=int, default=32)
    parser.add_argument("--mask_strategy", type=str, default="top_k_entropy", choices=["top_k_entropy", "first_k"])
    parser.add_argument("--mask_max_k", type=int, default=32768)
    parser.add_argument("--grad_direction", type=str, default="positive", choices=["positive", "negative"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "frank_wolfe"])
    parser.add_argument("--fw_gamma", type=float, default=0.1)
    parser.add_argument("--conn_type", type=str, default="fast", choices=["fast", "original"])
    parser.add_argument("--reg_type", type=str, default="kl", choices=["kl", "lm"])
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_threshold", type=float, default=1e-3)
    parser.add_argument("--distill_epochs", type=int, default=3)
    parser.add_argument("--distill_lr", type=float, default=2e-5)
    parser.add_argument("--distill_ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--distill_eval_every", type=int, default=20, help="每多少步执行一次蒸馏评测")
    parser.add_argument("--distill_sample_filter", action="store_true", help="是否剔除没有优化的样本")
    parser.add_argument("--distill_eval_datasets", type=str, nargs='+', default=["HuggingFaceH4/MATH-500"])
    parser.add_argument("--distill_batch_size", type=int, default=4)
    parser.add_argument("--distill_grad_accum_steps", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    wandb.init(project="L-GRPO-Math500", config=vars(args))
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, 
        dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    model.requires_grad_(False)
    model.eval()  # 👇 [🔥 新增：确保前两阶段绝对闭锁随机性] 👇
    vllm_engine = VLLMDPWorkerPool(model_name=args.model_name, gpu_ids=args.vllm_gpus)
    
    def get_embeds(text):
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        return model.get_input_embeddings()(ids).detach(), ids

    embeds_end_think, ids_end_think = get_embeds("</think>")
    embeds_fast_conn, ids_fast_conn = get_embeds("The final answer is \n&&\n\\boxed{")
    all_embeddings = model.get_input_embeddings().weight.detach()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    wrong_dataset = build_math_wrong_dataset(args.file_path, tokenizer)
    raw_data = wrong_dataset.flat_data
    raw_data = raw_data[:10000] # 限制最大数据量

    # --- Phase 1 ---
    print("⚙️ Preparing Static Data & Latents (Batch Forwarding)...")
    static_data_cpu = {}
    global_latents = torch.nn.ParameterDict()
    active_uids = []
    
    global_history = {
        d['uid']: {"uid": d['uid'], "problem": d['question_text'], "gt_text": d['gt_text'], "steps": []} 
        for d in raw_data
    }

    embeds_end_think_cpu = embeds_end_think.cpu()
    embeds_fast_conn_cpu = embeds_fast_conn.cpu()
    all_embeddings_cpu = all_embeddings.cpu()

    prep_batch_size = args.batch_size
    for i in tqdm(range(0, len(raw_data), prep_batch_size), desc="Pre-computing"):
        batch_samples = raw_data[i : i + prep_batch_size]
        batch_embeds_full = []
        batch_info = []
        
        for d in batch_samples:
            uid = d['uid']
            active_uids.append(uid)
            
            embeds_q, ids_q = get_embeds(d['question_text'])
            embeds_conn, ids_conn = get_embeds(d['connector_text']) if args.conn_type == "original" else (embeds_fast_conn, ids_fast_conn)
            embeds_gt, ids_gt = get_embeds(d['gt_text'])
            embeds_pred, ids_pred = get_embeds(d['pred_text'])
            embeds_think, ids_think = get_embeds(d['thinking_text'])
            
            curr_think = torch.nn.Parameter(embeds_think.detach().cpu().clone())
            global_latents[uid] = curr_think
            
            think_start_idx = ids_q.shape[1] - 1
            think_end_idx = think_start_idx + ids_think.shape[1]
            full_emb = torch.cat([embeds_q, embeds_think, embeds_end_think, embeds_conn], dim=1)
            batch_embeds_full.append(full_emb)
            
            batch_info.append({
                "uid": uid, "ids_q": ids_q, "ids_conn": ids_conn, "ids_gt": ids_gt,
                "ids_pred": ids_pred, "ids_think": ids_think, "thinking_text": d['thinking_text'],
                "start": think_start_idx, "end": think_end_idx, "len": full_emb.shape[1],
                "embeds_q": embeds_q, "embeds_conn": embeds_conn 
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
            think_logits = all_logits[j, info["start"]:info["end"], :].unsqueeze(0)
            probs = F.softmax(think_logits, dim=-1)
            
            if args.mask_strategy == "top_k_entropy":
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                mask = torch.zeros_like(entropy, dtype=torch.float32)
                mask.scatter_(1, torch.topk(entropy, k=min(args.mask_max_k, entropy.shape[1]), dim=1).indices, 1.0)
            else:
                mask = torch.zeros((1, info["ids_think"].shape[1]), dtype=torch.float32, device=device)
                mask[:, :args.mask_max_k] = 1.0
            
            topk_probs, topk_indices = torch.topk(probs, k=100, dim=-1)
            static_data_cpu[uid] = {
                "ids_q": info["ids_q"].cpu(), "ids_conn": info["ids_conn"].cpu(), 
                "ids_gt": info["ids_gt"].cpu(), "ids_pred": info["ids_pred"].cpu(), 
                "ids_think": info["ids_think"].cpu(), "grad_mask": mask.unsqueeze(-1).cpu(),
                "topk_probs": topk_probs.cpu(), "topk_indices": topk_indices.cpu(),
                "thinking_text": info['thinking_text'],
                "embeds_q_cpu": info["embeds_q"].cpu(),
                "embeds_conn_cpu": info["embeds_conn"].cpu()
            }
        
        del all_logits, outputs, padded_embeds, attn_masks, batch_embeds_full
    torch.cuda.empty_cache()
    
    optimal_latents = {uid: {"acc": -1.0, "latent": None} for uid in global_latents.keys()}
    
    global_opts = {}
    if args.optimizer == "adam":
        for uid, param in global_latents.items():
            global_opts[uid] = torch.optim.Adam([param], lr=args.learning_rate)
    elif args.optimizer == "frank_wolfe":
        optimizer = FrankWolfeOptimizer(all_embeddings)

    run_name = wandb.run.name if wandb.run is not None else f"opt_v2_{args.optimizer}_{args.reg_type}"
    history_filename = f"./optimization_histories/optimization_history_{run_name}.jsonl"
    os.makedirs(os.path.dirname(history_filename), exist_ok=True)
    
    # [新增] 专门用于覆盖更新历史文件的闭包函数
    def save_current_history():
        tmp_filename = history_filename + ".tmp"
        with open(tmp_filename, "w", encoding="utf-8") as f:
            # 永远保持第一行是 config
            f.write(json.dumps({"config": vars(args)}, ensure_ascii=False) + "\n")
            # 遍历写入所有 sample 的最新状态
            for u in global_history.keys():
                f.write(json.dumps(global_history[u], ensure_ascii=False) + "\n")
        # 原子替换，防止读取时文件不完整
        os.replace(tmp_filename, history_filename)

    # 初始写入一次
    save_current_history()

    # ==================================================
    # [🚀 异步后台逻辑定义]
    # ==================================================
    executor = ThreadPoolExecutor(max_workers=1)
    eval_future = None

    def run_eval_async(eval_step, eval_uids, latents_snapshot):
        try:
            # 强行注入独立的 Event Loop，避免 vLLM 的异步机制在子线程中崩溃
            asyncio.set_event_loop(asyncio.new_event_loop())

            # [🌟 算力物理隔离] 获取第一个 vLLM 分配的 GPU（如 cuda:1）
            # 这样潜空间漂移计算就完全不会干扰 GPU 0 上的训练前向和反向传播
            eval_device_id = args.vllm_gpus[0] if args.vllm_gpus else 0
            eval_device = torch.device(f"cuda:{eval_device_id}")

            tqdm.write(f"\n🚀 [后台评测] Step {eval_step} 线程启动! 正在 {eval_device} 使用 Chunked 加速计算潜空间漂移...")
            total_pure_acc, total_forced_acc, total_fast_acc = 0.0, 0.0, 0.0
            total_rouge_l, total_change_ratio = 0.0, 0.0
            
            eval_pure_inputs, eval_forced_inputs, eval_fast_inputs = [], [], []
            step_metrics = {uid: {} for uid in eval_uids}
            
            # [🔥 GPU 物理隔离] 将全量词表移至 Eval 专属 GPU 并执行归一化
            with torch.no_grad():
                all_embeddings_norm_eval_gpu = F.normalize(all_embeddings_cpu.to(eval_device), dim=-1)
            
            for uid in eval_uids:
                ct_cpu = latents_snapshot[uid]
                sd = static_data_cpu[uid]
                
                target_flat = ct_cpu.squeeze(0)
                L = target_flat.shape[0]
                nearest_token_ids = torch.empty(L, dtype=torch.long)
                
                # [🔥 GPU Chunked 计算] 限制每次计算量，防止挤爆 vLLM 的显存
                drift_chunk_size = 2048*512  # 大约只产生 150MB 左右的峰值显存占用
                with torch.no_grad():
                    for c_start in range(0, L, drift_chunk_size):
                        c_end = min(c_start + drift_chunk_size, L)
                        
                        # 1. 搬运 Chunk 到 Eval GPU
                        chunk_eval_gpu = target_flat[c_start:c_end].to(eval_device, non_blocking=True)
                        chunk_norm = F.normalize(chunk_eval_gpu, dim=-1)
                        
                        # 2. 在 Eval GPU 上执行矩阵乘法 [chunk, D] @ [D, Vocab]
                        sim_chunk = torch.matmul(chunk_norm, all_embeddings_norm_eval_gpu.T)
                        
                        # 3. 得到结果立马送回 CPU 保存
                        nearest_token_ids[c_start:c_end] = torch.argmax(sim_chunk, dim=-1).cpu()
                        
                        # 4. 及时清理垃圾释放 Eval GPU 显存
                        del chunk_eval_gpu, chunk_norm, sim_chunk
                
                orig_ids = sd["ids_think"].squeeze(0)
                changed_mask = (nearest_token_ids != orig_ids)
                change_ratio = changed_mask.float().mean().item()
                
                decoded_nearest_text = tokenizer.decode(nearest_token_ids).strip()
                original_thinking_clean = sd['thinking_text']
                rouge_l_f1 = scorer.score(original_thinking_clean[:10000], decoded_nearest_text[:10000])['rougeL'].fmeasure
                
                step_metrics[uid]['change_ratio'] = change_ratio
                step_metrics[uid]['rouge_L'] = rouge_l_f1
                
                total_change_ratio += change_ratio
                total_rouge_l += rouge_l_f1

                embeds_q_cpu = sd["embeds_q_cpu"]
                embeds_conn_cpu = sd["embeds_conn_cpu"]
                
                eval_pure_inputs.append(torch.cat([embeds_q_cpu, ct_cpu, embeds_end_think_cpu], dim=1).squeeze(0))
                eval_forced_inputs.append(torch.cat([embeds_q_cpu, ct_cpu, embeds_end_think_cpu, embeds_conn_cpu], dim=1).squeeze(0))
                eval_fast_inputs.append(torch.cat([embeds_q_cpu, ct_cpu, embeds_end_think_cpu, embeds_fast_conn_cpu], dim=1).squeeze(0))

            # 清理 Eval GPU 上的全局归一化词表，彻底打扫战场
            del all_embeddings_norm_eval_gpu
            # 注意这里要清理特定的 Eval GPU
            with torch.cuda.device(eval_device):
                torch.cuda.empty_cache()

            tqdm.write(f"⚡ [后台评测] 漂移计算完毕，准备发送数据至 vLLM Worker...")
            modes = [('pure', eval_pure_inputs, 2048), ('forced', eval_forced_inputs, 128), ('fast', eval_fast_inputs, 128)]
            
            # 这里移除所有的 TQDM，防止破坏主线程的进度条
            for mode, inputs_list, max_toks in modes:
                tqdm.write(f"   --> vLLM 正在并行生成 {mode} 模式...")
                batch_outputs = vllm_engine.generate(inputs_list, {"max_tokens": max_toks, "temperature": 0.7, "n": args.eval_k, "skip_special_tokens": False})
                if not batch_outputs: continue
                
                for idx, uid in enumerate(eval_uids):
                    gt_text = global_history[uid]['gt_text']
                    gt_token_len = len(tokenizer.encode(gt_text, add_special_tokens=False))
                    
                    correct_count = 0
                    for output_ids in batch_outputs[idx]:
                        if mode != 'pure': output_ids = output_ids[:gt_token_len]
                        gen_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                        is_corr = False
                        if mode == 'pure':
                            if is_correct_v3(gen_text, gt_text.replace("}", "")): is_corr = True
                        else:
                            ans = gen_text.replace("$", "").replace("}", "").strip()
                            if is_equiv(ans, gt_text.replace("}", "")): is_corr = True
                        if is_corr: correct_count += 1
                            
                    acc = correct_count / args.eval_k
                    step_metrics[uid][f'{mode}_acc'] = acc
                    if mode == 'pure' and len(batch_outputs[idx]) > 0:
                        sample_gen = tokenizer.decode(batch_outputs[idx][0], skip_special_tokens=True)
                        step_metrics[uid]['sample_gen_text'] = sample_gen
                    
                    if mode == 'pure': total_pure_acc += acc
                    elif mode == 'forced': total_forced_acc += acc
                    elif mode == 'fast': total_fast_acc += acc

            tqdm.write(f"\n✅ [后台评测完成 | Step {eval_step}] Pure: {total_pure_acc/len(eval_uids):.2%} | Forced: {total_forced_acc/len(eval_uids):.2%} | Fast: {total_fast_acc/len(eval_uids):.2%}")
            
            return {
                "step": eval_step, "uids": eval_uids, "step_metrics": step_metrics,
                "latents_snapshot": latents_snapshot,
                "avg_metrics": {
                    "eval/avg_pure_acc": total_pure_acc / len(eval_uids),
                    "eval/avg_forced_acc": total_forced_acc / len(eval_uids),
                    "eval/avg_fast_acc": total_fast_acc / len(eval_uids),
                    "eval/avg_rouge_L": total_rouge_l / len(eval_uids),
                    "eval/avg_change_ratio": total_change_ratio / len(eval_uids)
                }
            }
        except Exception as e:
            # 强制拦截所有异常并打印，拒绝吞没！
            tqdm.write(f"\n❌ [后台评测严重崩溃] {str(e)}")
            tqdm.write(traceback.format_exc())
            return None

    def process_eval_results(res):
        if res is None: return
        eval_step, eval_uids, eval_metrics = res["step"], res["uids"], res["step_metrics"]
        latents_snapshot = res["latents_snapshot"]
        
        wandb.log({"global_step": eval_step, **res["avg_metrics"]})
        
        for uid in eval_uids:
            for step_record in global_history[uid]["steps"]:
                if step_record["step"] == eval_step:
                    step_record["metrics"].update(eval_metrics[uid])
                    break
            if uid in optimal_latents:
                current_pure_acc = eval_metrics[uid].get('pure_acc', 0.0)
                if current_pure_acc > optimal_latents[uid]["acc"]:
                    optimal_latents[uid]["acc"] = current_pure_acc
                    optimal_latents[uid]["latent"] = latents_snapshot[uid]
        
        save_current_history()

    # ==================================================
    # Phase 2: 外层训练循环
    # ==================================================
    for step in range(args.steps):
        # 每次进入新的循环，检查后台任务是否顺利完成了
        if eval_future is not None and eval_future.done():
            process_eval_results(eval_future.result())
            eval_future = None

        if not active_uids:
            print("🎉 所有样本均已触发 Early Stop，优化阶段结束，进入 Final Eval！")
            break
            
        print(f"\n==============================================")
        print(f"🔄 Global Step {step+1}/{args.steps} | Active Samples: {len(active_uids)}")
        print(f"==============================================")
        
        progress = step / max(1, args.steps - 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        current_lr = (args.learning_rate * 0.1) + (args.learning_rate * 0.9) * cosine_factor
        current_fw_gamma = (args.fw_gamma * 0.1) + (args.fw_gamma * 0.9) * cosine_factor
        
        step_metrics = {uid: {} for uid in active_uids}
        epoch_gt_loss_sum, epoch_kl_loss_sum, epoch_lm_loss_sum, epoch_total_loss_sum = 0.0, 0.0, 0.0, 0.0
        
        import random
        random.shuffle(active_uids) 
        mini_batches = [active_uids[i:i + args.batch_size] for i in range(0, len(active_uids), args.batch_size)]
        pbar = tqdm(mini_batches, desc=f"Step {step+1} Optimizing")
        
        for batch_uids in pbar:
            curr_think_list, curr_mask_list = [], []
            embeds_q_list, embeds_conn_list, embeds_gt_list, embeds_pred_list = [], [], [], []
            ids_q_list, ids_think_list, ids_conn_list, ids_gt_list, ids_pred_list = [], [], [], [], []
            
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
                        pg['lr'] = current_lr
                elif args.optimizer == "frank_wolfe":
                    if p.grad is not None: p.grad.zero_()
                
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
                full_embeds_list.append(torch.cat([embeds_q_list[i], curr_think_list[i].to(model.dtype), embeds_end_think, embeds_conn_list[i], target_emb], dim=1))

            max_len = max(emb.shape[1] for emb in full_embeds_list)
            attention_mask_list = []
            for i, emb in enumerate(full_embeds_list):
                if emb.shape[1] < max_len:
                    pad_emb = torch.zeros((emb.shape[0], max_len - emb.shape[1], emb.shape[2]), device=device, dtype=model.dtype)
                    full_embeds_list[i] = torch.cat([emb, pad_emb], dim=1)
                    attn = torch.zeros((emb.shape[0], max_len), device=device, dtype=torch.long)
                    attn[:, :emb.shape[1]] = 1
                    attention_mask_list.append(attn)
                else:
                    attention_mask_list.append(torch.ones((emb.shape[0], max_len), device=device, dtype=torch.long))
                    
            full_embeds_batch = torch.cat(full_embeds_list, dim=0)
            full_attention_mask = torch.cat(attention_mask_list, dim=0)

            last_hidden = model.model(inputs_embeds=full_embeds_batch, attention_mask=full_attention_mask).last_hidden_state

            all_gt_loss, all_kl_div_list, batch_logprobs = [], [], []
            for i, uid in enumerate(batch_uids):
                sd = static_data_cpu[uid]
                gt_pos = ids_q_list[i].shape[1] + ids_think_list[i].shape[1] + ids_end_think.shape[1] + ids_conn_list[i].shape[1] - 1
                target_ids = ids_gt_list[i] if args.grad_direction == "positive" else ids_pred_list[i]
                
                target_logits = model.lm_head(last_hidden[[i], gt_pos : gt_pos + target_ids.shape[1], :])
                gt_loss = F.cross_entropy(target_logits.view(-1, target_logits.size(-1)), target_ids.view(-1))
                if args.grad_direction == "negative": gt_loss = -gt_loss
                all_gt_loss.append(gt_loss)
                
                think_start = ids_q_list[i].shape[1] - 1
                think_end = think_start + ids_think_list[i].shape[1]
                curr_think_hidden = last_hidden[[i], think_start:think_end, :] 
                think_len = curr_think_hidden.shape[1]
                
                kl_sum, logprobs_sum = 0.0, 0.0
                orig_probs_topk = sd["topk_probs"].to(device)
                orig_indices_topk = sd["topk_indices"].to(device)
                
                curr_think_embeds = torch.cat([curr_think_list[i][:, 1:ids_think_list[i].shape[1], :], embeds_end_think], dim=1)
                target_ids_think = torch.cat([ids_think_list[i][:, 1:], ids_end_think], dim=1) 
                
                for c_start in range(0, think_len, args.chunk_size):
                    c_end = min(c_start + args.chunk_size, think_len)
                    h_chunk = curr_think_hidden[:, c_start:c_end, :]
                    logits_chunk = model.lm_head(h_chunk) 
                    lse_chunk = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)
                    
                    orig_p_chunk = orig_probs_topk[:, c_start:c_end, :]
                    orig_idx_chunk = orig_indices_topk[:, c_start:c_end, :]
                    curr_log_probs_topk_chunk = torch.gather(logits_chunk, -1, orig_idx_chunk) - lse_chunk
                    kl_chunk = (orig_p_chunk * (torch.log(orig_p_chunk + 1e-10) - curr_log_probs_topk_chunk)).sum(dim=-1)
                    kl_sum += kl_chunk.sum() 
                    
                    embeds_chunk = curr_think_embeds[:, c_start:c_end, :]
                    target_ids_chunk = target_ids_think[:, c_start:c_end] 
                    orig_embeds_chunk = model.get_input_embeddings()(target_ids_chunk).detach()
                    
                    orig_norm = torch.norm(orig_embeds_chunk, p=2, dim=-1, keepdim=True)
                    curr_norm = torch.norm(embeds_chunk, p=2, dim=-1, keepdim=True)
                    scaled_embeds_chunk = embeds_chunk * (orig_norm / (curr_norm + 1e-8))
                    
                    score_0_chunk = (h_chunk * scaled_embeds_chunk).sum(dim=-1) 
                    mask_chunk = torch.zeros_like(logits_chunk, dtype=torch.bool)
                    mask_chunk.scatter_(2, target_ids_chunk.unsqueeze(-1), True)
                    
                    new_logits_chunk = torch.where(mask_chunk, score_0_chunk.unsqueeze(-1), logits_chunk)
                    logprobs_sum += (score_0_chunk - torch.logsumexp(new_logits_chunk, dim=-1)).sum()
                    
                all_kl_div_list.append(kl_sum / think_len)
                batch_logprobs.append(logprobs_sum / think_len)
                
                s_gt = gt_loss.item()
                s_kl = (kl_sum / think_len).item()
                s_lm = -(logprobs_sum / think_len).item()
                s_reg = s_kl if args.reg_type == "kl" else s_lm
                s_total = s_gt + args.kl_weight * s_reg
                
                step_metrics[uid] = {"total_loss": s_total, "gt_loss": s_gt, "kl_loss": s_kl, "lm_loss": s_lm}

            batch_gt_loss = torch.stack(all_gt_loss).mean()
            batch_kl_loss = torch.stack(all_kl_div_list).mean()
            batch_lm_prior_loss = -torch.stack(batch_logprobs).mean()
            
            reg_loss = batch_kl_loss if args.reg_type == "kl" else batch_lm_prior_loss
            total_loss = batch_gt_loss + args.kl_weight * reg_loss
            
            batch_sample_count = len(batch_uids)
            epoch_gt_loss_sum += batch_gt_loss.item() * batch_sample_count
            epoch_kl_loss_sum += batch_kl_loss.item() * batch_sample_count
            epoch_lm_loss_sum += batch_lm_prior_loss.item() * batch_sample_count
            epoch_total_loss_sum += total_loss.item() * batch_sample_count
            
            pbar.set_postfix({"Total": f"{total_loss.item():.3f}", "GT": f"{batch_gt_loss.item():.3f}"})
            
            total_loss.backward()
            
            for i, uid in enumerate(batch_uids):
                p = curr_think_list[i]
                mask = curr_mask_list[i]
                if p.grad is not None:
                    p.grad.data.mul_(mask.to(device).to(p.dtype))
                if args.optimizer == "adam":
                    opt = global_opts[uid]
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    move_optimizer_state(opt, torch.device("cpu"))
                elif args.optimizer == "frank_wolfe":
                    optimizer.step(p, gamma=current_fw_gamma)

                p.data = p.data.cpu().pin_memory()

            del full_embeds_batch, full_attention_mask, last_hidden
            # torch.cuda.empty_cache()
            
            # [🔥 防卡死黑科技] 每跑完一个 Batch，主动出让一次 GIL，给后台线程留口饭吃！
            time.sleep(0.005)

        wandb.log({
            "global_step": step,
            "train/epoch_total_loss": epoch_total_loss_sum / len(active_uids),
            "train/epoch_gt_loss": epoch_gt_loss_sum / len(active_uids),
            "train/lr": current_lr,
            "active_samples": len(active_uids)
        })

        # ==================================================
        # 记录本地训练指标及执行 Early Stop
        # ==================================================
        next_active_uids = []
        for uid in active_uids:
            global_history[uid]["steps"].append({"step": step, "metrics": step_metrics[uid]})
            gt_loss = step_metrics[uid]["gt_loss"]
            if args.early_stop and gt_loss < args.early_stop_threshold:
                tqdm.write(f"  🔪 [修剪] 样本 {uid} 达标 (Loss: {gt_loss:.4f})，移出队列。")
            else:
                next_active_uids.append(uid)
        active_uids = next_active_uids

        # ==================================================
        # [🚀 触发后台 Eval]
        # ==================================================
        if step % args.eval_every == 0 and step != args.steps - 1:
            if eval_future is not None:
                tqdm.write("⏳ 已经到达下一个 Eval 节点，强制主进程等待后台收尾上一轮评测...")
                process_eval_results(eval_future.result())
                eval_future = None
                
            eval_uids = list(active_uids)
            latents_snapshot = {uid: global_latents[uid].detach().clone() for uid in eval_uids}
            eval_future = executor.submit(run_eval_async, step, eval_uids, latents_snapshot)
        if step % args.eval_every == 0 or step == args.steps - 1:
            save_current_history()

    # ==================================================
    # Phase 3: 收尾动作
    # ==================================================
    if eval_future is not None:
        print("\n⏳ 训练步骤已全部完成，等待后台收尾最后一轮 Eval...")
        process_eval_results(eval_future.result())

    print("\n" + "="*60)
    print("🎉 开始全局 Final Evaluation (此步骤不再异步) ...")
    print("="*60)
    
    all_uids = list(global_history.keys())
    total_pure_acc, total_forced_acc, total_fast_acc = 0.0, 0.0, 0.0
    total_rouge_l, total_change_ratio = 0.0, 0.0

    eval_pure_inputs, eval_forced_inputs, eval_fast_inputs = [], [], []
    
    for uid in all_uids:
        ct = global_latents[uid].to(device).to(all_embeddings.dtype).detach()
        sd = static_data_cpu[uid]
        
        target_flat = ct.squeeze(0)
        sim = torch.matmul(F.normalize(target_flat, dim=-1), F.normalize(all_embeddings, dim=-1).T)
        nearest_token_ids = torch.argmax(sim, dim=-1)
        
        orig_ids = sd["ids_think"].to(device).squeeze(0)
        changed_mask = (nearest_token_ids != orig_ids)
        change_ratio = changed_mask.float().mean().item()
        
        decoded_nearest_text = tokenizer.decode(nearest_token_ids).strip()
        original_thinking_clean = sd['thinking_text']
        rouge_l_f1 = scorer.score(original_thinking_clean[:10000], decoded_nearest_text[:10000])['rougeL'].fmeasure
        
        global_history[uid]["steps"][-1]["metrics"]["change_ratio"] = change_ratio
        global_history[uid]["steps"][-1]["metrics"]["rouge_L"] = rouge_l_f1
        
        total_change_ratio += change_ratio
        total_rouge_l += rouge_l_f1

        with torch.no_grad():
            embeds_q = model.get_input_embeddings()(sd["ids_q"].to(device))
            embeds_conn = model.get_input_embeddings()(sd["ids_conn"].to(device))
        
        eval_pure_inputs.append(torch.cat([embeds_q, ct, embeds_end_think], dim=1).squeeze(0))
        eval_forced_inputs.append(torch.cat([embeds_q, ct, embeds_end_think, embeds_conn], dim=1).squeeze(0))
        eval_fast_inputs.append(torch.cat([embeds_q, ct, embeds_end_think, embeds_fast_conn], dim=1).squeeze(0))

    modes = [('pure', eval_pure_inputs, 2048), ('forced', eval_forced_inputs, 128), ('fast', eval_fast_inputs, 128)]
    
    for mode, inputs_list, max_toks in tqdm(modes, desc="Final Eval Modes"):
        batch_outputs = vllm_engine.generate(inputs_list, {"max_tokens": max_toks, "temperature": 0.7, "n": args.eval_k, "skip_special_tokens": False})
        if not batch_outputs: continue
        
        for idx, uid in enumerate(tqdm(all_uids, desc=f"Processing {mode} Results", leave=False)):
            gt_text = global_history[uid]['gt_text']
            gt_token_len = len(tokenizer.encode(gt_text, add_special_tokens=False))
            
            correct_count = 0
            for output_ids in batch_outputs[idx]:
                if mode != 'pure': output_ids = output_ids[:gt_token_len]
                gen_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                is_corr = False
                
                if mode == 'pure':
                    if is_correct_v3(gen_text, gt_text.replace("}", "")): is_corr = True
                else:
                    ans = gen_text.replace("$", "").replace("}", "").strip()
                    if is_equiv(ans, gt_text.replace("}", "")): is_corr = True
                
                if is_corr: correct_count += 1
                    
            acc = correct_count / args.eval_k
            global_history[uid]["steps"][-1]["metrics"][f'{mode}_acc'] = acc
            
            if mode == 'pure' and len(batch_outputs[idx]) > 0:
                sample_gen = tokenizer.decode(batch_outputs[idx][0], skip_special_tokens=True)
                global_history[uid]["steps"][-1]["metrics"]['sample_gen_text'] = sample_gen
            
            if mode == 'pure': total_pure_acc += acc
            elif mode == 'forced': total_forced_acc += acc
            elif mode == 'fast': total_fast_acc += acc

    print(f"📊 [Final Eval Results] Pure Acc: {total_pure_acc/len(all_uids):.2%} | Forced Acc: {total_forced_acc/len(all_uids):.2%} | Fast Acc: {total_fast_acc/len(all_uids):.2%}")
    print(f"📉 [Final Latent Drift] ROUGE-L: {total_rouge_l/len(all_uids):.4f} | Change Ratio: {total_change_ratio/len(all_uids):.2%}")
    
    wandb.log({
        "global_step": args.steps, 
        "final/avg_pure_acc": total_pure_acc / len(all_uids),
        "final/avg_forced_acc": total_forced_acc / len(all_uids),
        "final/avg_fast_acc": total_fast_acc / len(all_uids),
        "final/avg_rouge_L": total_rouge_l / len(all_uids),
        "final/avg_change_ratio": total_change_ratio / len(all_uids)
    })
    
    # ==================================================
    # Phase 4: 全局落盘 (INT8 量化及 GZIP 压缩)
    # ==================================================
    print(f"📁 正在写入全部 {len(all_uids)} 个样本的完整历史记录...")
    save_current_history()
    
    try:
        print(f"💾 正在打包并压缩保存 Optimized Embeddings (仅保存 Optimal & INT8 量化)...")
        saved_dataset = []
        for d in raw_data:
            uid = d['uid']
            
            opt_raw = optimal_latents[uid]["latent"]
            if opt_raw is None:
                # 已经是安全的 CPU Tensor 
                opt_raw = global_latents[uid].detach().clone()
                
            int8_latent, scale = quantize_to_int8(opt_raw)
            
            item_pack = {
                "metadata": d,  
                "last_optimal_metrics": {
                    "optimal_pure_acc": max(optimal_latents[uid]["acc"], 0.0)
                },
                "tensors": {
                    "optimal_embeds_int8": int8_latent,
                    "optimal_scale": scale
                }
            }
            saved_dataset.append(item_pack)
            
        tensor_filename = f"./optimization_histories/optimized_embeds_{run_name}.pt.gz"
        os.makedirs(os.path.dirname(tensor_filename), exist_ok=True)
        with gzip.open(tensor_filename, 'wb') as f:
            torch.save(saved_dataset, f)
            
        print(f"✅ Embeddings 已成功以 INT8 格式压缩并保存至 {tensor_filename}")
    except Exception as e:
        print(f"⚠️ 保存 Embeddings 时发生错误: {e}")
    
    
# =========================================================================
    # Phase 5: 模型蒸馏阶段 (Distillation)
    # =========================================================================
    print("\n" + "="*60)
    print("🔥 Phase 5: 启动模型蒸馏阶段...")
    print("="*60)
    
    wandb.define_metric("model_train_step")
    wandb.define_metric("distill/train/*", step_metric="model_train_step")
    wandb.define_metric("distill/eval/*", step_metric="model_train_step")
    
    # 5.1 数据集筛选与组装
    distill_dataset = []
    for uid in all_uids:
        acc = optimal_latents[uid]["acc"]
        if args.distill_sample_filter and acc <= 0:
            continue # 剔除没有任何提升的样本
            
        sd = static_data_cpu[uid]
        # 获取最新的潜变量（作为优化的 target_think）
        opt_think_embeds = optimal_latents[uid]["latent"]
        if opt_think_embeds is None:
            opt_think_embeds = global_latents[uid].detach().clone()
        
        # 组装数据：Question + Think(Target) + EndThink + Connector + Answer(GT)
        distill_dataset.append({
            "uid": uid,
            "ids_q": sd["ids_q"],
            "embeds_q": sd["embeds_q_cpu"],
            "target_think_embeds": opt_think_embeds,
            "ids_think": sd["ids_think"],
            "ids_gt": sd["ids_gt"], # GT 作为 Answer
            "gt_text": global_history[uid]['gt_text']
        })
        
    print(f"📦 蒸馏数据集就绪，有效样本数: {len(distill_dataset)} / {len(all_uids)}")
    
    if len(distill_dataset) == 0:
        print("⚠️ 没有符合条件的样本进行蒸馏，流程结束。")
        return

    # 5.2 预计算 Target Soft Logits
    print("⚙️ 正在预计算 Target Soft Logits (Top-100) 防止内存爆满...")
    target_soft_logits_dict = {}
    model.eval()
    with torch.no_grad():
        for item in tqdm(distill_dataset, desc="Precomputing Target Logits"):
            uid = item["uid"]
            emb_q = item["embeds_q"].to(device)
            emb_think = item["target_think_embeds"].to(device).to(model.dtype)
            
            full_emb = torch.cat([emb_q, emb_think], dim=1)
            outputs = model(inputs_embeds=full_emb)
            
            start_idx = emb_q.shape[1] - 1
            end_idx = full_emb.shape[1] - 1
            think_logits = outputs.logits[:, start_idx:end_idx, :] 
            
            # 👇【修改点：实时计算 softmax 并仅保留 Top-100 的 prob 和 index】👇
            probs = F.softmax(think_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=100, dim=-1)
            
            # 保存到 CPU 内存，内存占用从单样本 ~1.2GB 直降到约 ~1.5MB
            target_soft_logits_dict[uid] = {
                "probs": topk_probs.cpu(),
                "indices": topk_indices.cpu()
            }
            
    torch.cuda.empty_cache()

    # 5.3 准备异步评测环境
    def pass_at_k(n, c, k):
        """计算 pass@k"""
        if n - c < k: return 1.0
        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

    def load_eval_datasets(dataset_names):
        datasets_dict = {}
        for ds_id in dataset_names:
            try:
                if "MATH-500" in ds_id:
                    ds = load_dataset(ds_id, split='test')
                    q_key, a_key = "problem", "answer"
                elif "amc23" in ds_id:
                    ds = load_dataset(ds_id, split='test')
                    q_key, a_key = "question", "answer"
                elif "gsm8k" in ds_id:
                    ds = load_dataset(ds_id, "main", split='test')
                    q_key, a_key = "question", "answer"
                elif "aime_2024" in ds_id:
                    ds = load_dataset(ds_id, split='train')
                    q_key, a_key = "problem", "answer"
                elif "aime25" in ds_id:
                    ds = load_dataset(ds_id, split='test')
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
    # =========================================================================
    # [新增] 将训练集（蒸馏数据集）加入评测字典中，监控训练拟合度
    # =========================================================================
    train_eval_data = []
    for item in distill_dataset:
        uid = item["uid"]
        train_eval_data.append({
            "problem_formatted": global_history[uid]["problem"], # 已经过 chat_template 格式化的文本
            "answer_gt": item["gt_text"].replace("}", "")        # 去除构建时追加的 '}' 符号以便正确比对
        })
    
    eval_datasets["train_dataset"] = {
        "data": train_eval_data,
        "q_key": "problem_formatted",
        "a_key": "answer_gt",
        "pre_formatted": True  # 标记该数据集已包含 template，跳过 apply_chat_template
    }
    print(f"✅ 成功将训练集加入评测: train_dataset (样本数: {len(train_eval_data)})")
    # =========================================================================
    distill_eval_future = None
    distill_step = 0
    
    def run_distill_eval_async(step, current_state_dict_path):
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            tqdm.write(f"\n🚀 [后台蒸馏评测] 正在同步权重至 vLM...")
            
            # 通知 vLLM 从 /dev/shm 热加载最新权重
            vllm_engine.update_weights(current_state_dict_path)
            
            # [修改] 采样数 n 降为 16
            metrics_to_log = {"model_train_step": step}
            
            # 重新引入 16 和 32 的聚合列表，以支持动态计算
            total_pass1, total_pass8, total_pass16, total_pass32 = [], [], [], []
            
            for ds_id, ds_info in tqdm(eval_datasets.items(), desc="🌐 蒸馏评测总体进度", leave=False, position=1, dynamic_ncols=True):
                ds, q_key, a_key = ds_info["data"], ds_info["q_key"], ds_info["a_key"]
                is_pre_formatted = ds_info.get("pre_formatted", False)
                
                # [🔥 新增逻辑]：根据不同的数据集名称动态设置 n
                current_n = 8  # 默认兜底
                if "MATH-500" in ds_id or "train_dataset" in ds_id:
                    current_n = 8
                elif any(x in ds_id for x in ["aime_2024", "aime25", "amc23"]):
                    current_n = 32
                
                # 为当前数据集构建专用的生成参数
                sp_dict = {"max_tokens": 32768, "temperature": 0.7, "n": current_n, "skip_special_tokens": False}
                
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
                
                for idx, item in enumerate(tqdm(ds, desc=f"📊 正在校验 {ds_id} 答案", leave=False, position=2, dynamic_ncols=True)):
                    gt_ans = str(item[a_key]).strip()
                    correct_count = 0
                    
                    for output_ids in batch_outputs[idx]:
                        gen_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                        ans_cand = last_boxed_only_string(gen_text)
                        if ans_cand and is_equiv(remove_boxed(ans_cand), gt_ans):
                            correct_count += 1
                            
                    # [🔥 新增逻辑]：动态计算当前 n 允许的 pass@k，传入的总数是 current_n
                    ds_pass1 += pass_at_k(current_n, correct_count, 1)
                    if current_n >= 8: ds_pass8 += pass_at_k(current_n, correct_count, 8)
                    if current_n >= 16: ds_pass16 += pass_at_k(current_n, correct_count, 16)
                    if current_n >= 32: ds_pass32 += pass_at_k(current_n, correct_count, 32)
                
                num_samples = len(ds)
                
                # [🔥 新增逻辑]：只记录有效的 pass@k 指标
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

            # 只有当对应列表里有数据时才计算平均值
            if total_pass1: metrics_to_log["distill/eval/avg_pass@1"] = sum(total_pass1)/len(total_pass1)
            if total_pass8: metrics_to_log["distill/eval/avg_pass@8"] = sum(total_pass8)/len(total_pass8)
            if total_pass16: metrics_to_log["distill/eval/avg_pass@16"] = sum(total_pass16)/len(total_pass16)
            if total_pass32: metrics_to_log["distill/eval/avg_pass@32"] = sum(total_pass32)/len(total_pass32)
                
            tqdm.write(f"✅ [后台蒸馏评测完成] 均值 Pass@1: {metrics_to_log.get('distill/eval/avg_pass@1', 0):.2%}")
            return metrics_to_log
            
        except Exception as e:
            tqdm.write(f"\n❌ [蒸馏评测严重崩溃] {str(e)}\n{traceback.format_exc()}")
            return None
    
    # 5.4 开启模型训练状态
    model.requires_grad_(True)
    model.train()  # 🌟 [必须新增]：从预计算的 eval 模式切回 train 模式
    distill_optimizer = torch.optim.AdamW(model.parameters(), lr=args.distill_lr)
    
    distill_history_file = f"./optimization_histories/distill_history_{run_name}.jsonl"
    os.makedirs(os.path.dirname(distill_history_file), exist_ok=True)
    
    shm_model_path = "/dev/shm/distill_model_weights.pt"
    
    print(f"🏃 开启蒸馏训练，共 {args.distill_epochs} Epochs...")
    
    for epoch in range(args.distill_epochs):
        import random
        random.shuffle(distill_dataset)
        
        batch_size = args.distill_batch_size
        accum_steps = args.distill_grad_accum_steps
        
        mini_batches = [distill_dataset[i:i + batch_size] for i in range(0, len(distill_dataset), batch_size)]
        pbar = tqdm(mini_batches, desc=f"Distill Epoch {epoch+1}/{args.distill_epochs}")
        
        distill_optimizer.zero_grad()
        
        for b_idx, batch in enumerate(pbar):
            batch_loss = 0.0
            batch_kl = 0.0
            batch_ce = 0.0
            
            # 为了处理不同长度序列，我们依旧拆解 forward 或者利用 attention_mask
            # 简化实现，这里沿用逐样本处理（可根据显存调整）
            for item in batch:
                uid = item["uid"]
                ids_q = item["ids_q"].to(device)
                ids_think = item["ids_think"].to(device)
                ids_gt = item["ids_gt"].to(device)
                # 👇【修改点：取出保存的 top100 数据】👇
                target_info = target_soft_logits_dict[uid]
                target_probs = target_info["probs"].to(device)[0]      # [seq_len, 100]
                target_indices = target_info["indices"].to(device)[0]  # [seq_len, 100]
                
                # 构建完整的输入序列: Q + Think + EndThink + Conn + GT
                full_ids = torch.cat([ids_q, ids_think, ids_end_think.to(device), ids_fast_conn.to(device), ids_gt], dim=1)
                
                outputs = model(full_ids)
                logits = outputs.logits
                
                # == 计算 KL Loss ==
                # 定位 think 部分预测的位置
                think_start = ids_q.shape[1] - 1
                think_end = think_start + target_probs.shape[0]
                pred_think_logits = logits[0, think_start:think_end, :]
                
                # Soft KL Div
                pred_lse = torch.logsumexp(pred_think_logits, dim=-1, keepdim=True)
                pred_log_probs_topk = torch.gather(pred_think_logits, -1, target_indices) - pred_lse
                
                kl_elementwise = target_probs * (torch.log(target_probs + 1e-10) - pred_log_probs_topk)
                kl_loss = kl_elementwise.sum(dim=-1).mean() # 对 vocab 聚合求和，对 seq 聚合求平均
                
                # == 计算 CE Loss ==
                # 定位 Answer 预测的位置
                ans_start = full_ids.shape[1] - ids_gt.shape[1] - 1
                ans_end = full_ids.shape[1] - 1
                pred_ans_logits = logits[0, ans_start:ans_end, :]
                
                ce_loss = F.cross_entropy(pred_ans_logits, ids_gt[0])
                
                total_loss = kl_loss + args.distill_ce_loss_weight * ce_loss
                
                # 损失按累积步数缩放
                (total_loss / (batch_size * accum_steps)).backward()
                
                batch_loss += total_loss.item()
                batch_kl += kl_loss.item()
                batch_ce += ce_loss.item()
                
                # 记录 JSONL (每条样本详情)
                with open(distill_history_file, "a", encoding="utf-8") as f:
                    record = {
                        "model_train_step": distill_step,
                        "epoch": epoch,
                        "uid": uid,
                        "total_loss": total_loss.item(),
                        "kl_loss": kl_loss.item(),
                        "ce_loss": ce_loss.item(),
                        "gen_length": full_ids.shape[1]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 梯度累加更新
            if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == len(mini_batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                distill_optimizer.step()
                distill_optimizer.zero_grad()
                
                avg_kl = batch_kl / batch_size
                avg_ce = batch_ce / batch_size
                avg_loss = batch_loss / batch_size
                
                pbar.set_postfix({"Loss": f"{avg_loss:.3f}", "KL": f"{avg_kl:.3f}", "CE": f"{avg_ce:.3f}"})
                
                wandb.log({
                    "model_train_step": distill_step,
                    "distill/train/total_loss": avg_loss,
                    "distill/train/kl_loss": avg_kl,
                    "distill/train/ce_loss": avg_ce,
                    "distill/train/lr": distill_optimizer.param_groups[0]['lr']
                })
                
                # ======== 触发异步评测 ========
                if distill_step % args.distill_eval_every == 0:
                    if distill_eval_future is not None:
                        tqdm.write("⏳ 等待上一轮蒸馏评测结束...")
                        res = distill_eval_future.result()
                        if res: wandb.log(res)
                        
                    # 落盘当前模型至共享内存
                    torch.save(model.state_dict(), shm_model_path)
                    distill_eval_future = executor.submit(run_distill_eval_async, distill_step, shm_model_path)
                
                distill_step += 1

    # 收集最后一轮评测
    if distill_eval_future is not None:
        print("\n⏳ 蒸馏训练已全部完成，等待后台收尾最后一轮 Eval...")
        res = distill_eval_future.result()
        if res: wandb.log(res)

    print("🎉 蒸馏全流程结束！")
    wandb.finish()
    vllm_engine.close()
    
if __name__ == "__main__":
    main()
