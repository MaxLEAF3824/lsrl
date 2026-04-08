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
        gold_answer = item.get('answer', '')
        responses = item.get('responses', [])
        if not any(is_correct_v3(p, gold_answer) for p in responses):
            complete_but_wrong_responses = [
                res for res in responses 
                if last_boxed_only_string(res) and not math_verify_verify(math_verify_parse(remove_boxed(last_boxed_only_string(res))), math_verify_parse(gold_answer.strip()))
            ]
            if complete_but_wrong_responses:
                new_wrong_data.append({
                    "problem": item.get('problem', ''), "gold_answer": gold_answer,
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
    parser.add_argument("--distill_eval_every", type=int, default=50)
    parser.add_argument("--distill_sample_filter", action="store_true", help="是否只保留优化后 pure_acc 有提升的样本")
    parser.add_argument("--distill_datasets", type=str, nargs='+', default=["HuggingFaceH4/MATH-500", "openai/gsm8k", "math-ai/aime25", "math-ai/amc23","HuggingFaceH4/aime_2024"])
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
                drift_chunk_size = 2048  # 大约只产生 150MB 左右的峰值显存占用
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
            torch.cuda.empty_cache()
            
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
    # [🌟 Phase 5: 蒸馏阶段 (Latent-to-Text Model Distillation)]
    # =========================================================================
    print("\n" + "="*60)
    print("🚀 开启模型蒸馏阶段 (Distillation from Optimized Latents)")
    print("="*60)

    # 1. 过滤样本 (distill_sample_filter)
    distill_uids = []
    for uid in all_uids:
        if args.distill_sample_filter:
            # 如果 optimal acc 依然是 0，说明这个样本在隐状态优化期间根本没学出来，剔除。
            if optimal_latents[uid]["acc"] <= 0.0:
                continue
        distill_uids.append(uid)
    print(f"📊 蒸馏数据集筛选完成: 过滤前 {len(all_uids)} 个，过滤后保留 {len(distill_uids)} 个有效样本。")

    if len(distill_uids) == 0:
        print("⚠️ 没有符合条件的蒸馏样本，提前结束运行。")
        return

    # 2. 预计算教师模型 Soft Labels (KL Target)
    # 根据你的要求，这里在训练前一次性预计算好目标逻辑分布，避免训练时反复跑无梯度前向
    print("🧠 正在预计算教师模型 (Teacher Model) 的 Soft Labels...")
    teacher_soft_targets = {}
    model.eval() # 确保模型处于 eval 模式
    
    for uid in tqdm(distill_uids, desc="预计算 Soft Targets"):
        sd = static_data_cpu[uid]
        # 获取最优的隐状态
        opt_latent = optimal_latents[uid]["latent"].to(device) if optimal_latents[uid]["latent"] is not None else global_latents[uid].to(device)
        
        # 拼接 teacher 的输入: [Question] + [Optimized_Think_Latents] + [</think>]
        with torch.no_grad():
            embeds_q = model.get_input_embeddings()(sd["ids_q"].to(device))
            teacher_input_embeds = torch.cat([embeds_q, opt_latent.to(model.dtype), embeds_end_think], dim=1)
            teacher_outputs = model(inputs_embeds=teacher_input_embeds)
            
            # 提取 Think 部分的 Logits 作为 KL 散度的目标
            think_start_idx = sd["ids_q"].shape[1] - 1
            think_end_idx = think_start_idx + opt_latent.shape[1]
            teacher_think_logits = teacher_outputs.logits[0, think_start_idx:think_end_idx, :].detach().cpu()
            teacher_soft_targets[uid] = teacher_think_logits
            
        del teacher_outputs, teacher_input_embeds, embeds_q, opt_latent
        torch.cuda.empty_cache()

    # 3. 释放优化器占用的显存，准备切换模型状态为可训练
    del global_latents, static_data_cpu, global_opts
    torch.cuda.empty_cache()
    
    # 将 LLM 设为可训练状态
    model.requires_grad_(True)
    distill_optimizer = torch.optim.Adam(model.parameters(), lr=args.distill_lr)
    
    wandb.define_metric("distill_step")
    wandb.define_metric("distill/*", step_metric="distill_step")
    
    # 辅助函数：计算 pass@k
    def calc_pass_at_k(n, c, k):
        if c == 0: return 0.0
        if n - c < k: return 1.0
        # 1 - C(n-c, k) / C(n, k)
        return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

    # 4. 异步评测逻辑定义
    from datasets import load_dataset
    distill_eval_future = None
    distill_history_filename = f"./optimization_histories/distill_history_{run_name}.jsonl"
    distill_eval_lock = asyncio.Lock()

    def run_distill_eval_async(step, current_state_dict_path):
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            tqdm.write(f"\n🚀 [后台蒸馏评测] Step {step} 启动！正在更新 vLLM Worker 权重...")
            
            # 让 vLLM 从 /dev/shm 读取并热更新权重
            vllm_engine.update_weights(current_state_dict_path)
            tqdm.write(f"✅ [后台蒸馏评测] 权重同步完毕，开始跑测试集...")
            
            eval_metrics = {}
            step_record = {"step": step, "datasets": {}}
            
            for ds_id in args.distill_datasets:
                tqdm.write(f"  👉 正在评测数据集: {ds_id}")
                if ds_id == "HuggingFaceH4/MATH-500":
                    ds = load_dataset(ds_id, split='test')
                    q_key, a_key = "problem", "answer"
                elif ds_id == "math-ai/amc23":
                    ds = load_dataset(ds_id, split='test')
                    q_key, a_key = "question", "answer"
                elif ds_id == "openai/gsm8k":
                    ds = load_dataset(ds_id, "main", split='test')
                    q_key, a_key = "question", "answer"
                elif ds_id == "HuggingFaceH4/aime_2024":
                    ds = load_dataset(ds_id, split='train')
                    q_key, a_key = "problem", "answer"
                elif ds_id == "math-ai/aime25":
                    ds = load_dataset(ds_id, split='test')
                    q_key, a_key = "problem", "answer"
                else:
                    tqdm.write(f"⚠️ 未知数据集 {ds_id}，跳过。")
                    continue

                prompts = []
                gt_answers = []
                for item in ds:
                    messages = [{"role": "user", "content": item[q_key]}]
                    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt_text)
                    gt_answers.append(item[a_key])
                
                # 每个问题生成 32 个回复用于计算 pass@k
                eval_k_samples = 32
                sp_dict = {"max_tokens": 4096, "temperature": 0.7, "n": eval_k_samples, "skip_special_tokens": True}
                
                # 传入字符串 list 给改写过的 vllm_worker
                batch_outputs = vllm_engine.generate(prompts, sp_dict)
                
                if not batch_outputs: continue
                
                ds_pass1, ds_pass8, ds_pass16, ds_pass32 = 0.0, 0.0, 0.0, 0.0
                ds_gen_lengths = []
                dataset_sample_records = []
                
                for idx, out_ids_list in enumerate(batch_outputs):
                    gt_ans = str(gt_answers[idx])
                    correct_count = 0
                    sample_texts = []
                    
                    for out_ids in out_ids_list:
                        ds_gen_lengths.append(len(out_ids))
                        gen_text = tokenizer.decode(out_ids)
                        sample_texts.append(gen_text)
                        # 这里复用你的 math_verify 或者 is_correct_v3 逻辑
                        try:
                            # 简单容错提取
                            pred_ans = gen_text.split("</think>")[-1].strip()
                            if is_correct_v3(pred_ans, gt_ans):
                                correct_count += 1
                        except Exception:
                            pass
                    
                    n = eval_k_samples
                    c = correct_count
                    ds_pass1 += calc_pass_at_k(n, c, 1)
                    ds_pass8 += calc_pass_at_k(n, c, min(8, n))
                    ds_pass16 += calc_pass_at_k(n, c, min(16, n))
                    ds_pass32 += calc_pass_at_k(n, c, min(32, n))
                    
                    dataset_sample_records.append({
                        "question": prompts[idx], "gt": gt_ans,
                        "correct_count": c, "samples": sample_texts[:2] # 只存2个样例避免文件爆炸
                    })
                
                num_items = len(prompts)
                p1, p8, p16, p32 = ds_pass1/num_items, ds_pass8/num_items, ds_pass16/num_items, ds_pass32/num_items
                avg_len = sum(ds_gen_lengths)/len(ds_gen_lengths) if ds_gen_lengths else 0
                
                eval_metrics.update({
                    f"distill_{ds_id}/pass@1": p1, f"distill_{ds_id}/pass@8": p8,
                    f"distill_{ds_id}/pass@16": p16, f"distill_{ds_id}/pass@32": p32,
                    f"distill_{ds_id}/avg_gen_length": avg_len
                })
                
                step_record["datasets"][ds_id] = {
                    "metrics": {"pass@1": p1, "pass@8": p8, "pass@16": p16, "pass@32": p32, "avg_len": avg_len},
                    "details": dataset_sample_records
                }
                tqdm.write(f"    📊 {ds_id} Pass@1: {p1:.2%} | Pass@32: {p32:.2%}")

            # 计算多数据集宏平均
            macro_p1 = sum(eval_metrics[f"distill_{d}/pass@1"] for d in args.distill_datasets) / len(args.distill_datasets)
            eval_metrics["distill_avg/macro_pass@1"] = macro_p1
            
            # 落盘 History
            with open(distill_history_filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_record, ensure_ascii=False) + "\n")
                
            return {"step": step, "metrics": eval_metrics}
            
        except Exception as e:
            tqdm.write(f"\n❌ [后台蒸馏评测崩溃] {str(e)}")
            tqdm.write(traceback.format_exc())
            return None

    def process_distill_eval(res):
        if res is None: return
        wandb.log({"distill_step": res["step"], **res["metrics"]})

    # 5. 蒸馏主训练循环
    global_distill_step = 0
    model.train()
    
    for epoch in range(args.distill_epochs):
        random.shuffle(distill_uids)
        
        # 每次 epoch 开始，或 eval_every 触发时，检查后台评测
        distill_batches = [distill_uids[i:i + args.batch_size] for i in range(0, len(distill_uids), args.batch_size)]
        pbar = tqdm(distill_batches, desc=f"Distill Epoch {epoch+1}/{args.distill_epochs}")
        
        for batch_uids in pbar:
            # 阻塞等待上一次评测完成（如果存在且未完成，也可以选择不阻塞而是跳过本次发起，这里遵循你"必须等待"的设计）
            if global_distill_step % args.distill_eval_every == 0 and global_distill_step > 0:
                if distill_eval_future is not None:
                    pbar.write("⏳ 等待上一轮后台评测收尾...")
                    process_distill_eval(distill_eval_future.result())
                    distill_eval_future = None
                
                # 将模型当前权重保存到 /dev/shm 内存盘中，极速 I/O
                shm_path = f"/dev/shm/vllm_weights_{run_name}.pt"
                # 仅保存可训练的模型参数，避免 OOM
                torch.save(model.state_dict(), shm_path)
                
                # 提交异步评测任务
                distill_eval_future = executor.submit(run_distill_eval_async, global_distill_step, shm_path)

            distill_optimizer.zero_grad()
            batch_total_loss, batch_kl_loss, batch_ce_loss = 0.0, 0.0, 0.0
            
            for uid in batch_uids:
                # 获取数据
                raw_d = next(d for d in raw_data if d["uid"] == uid)
                original_think_text = raw_d["thinking_text"]
                optimized_answer_text = raw_d["gt_text"] # 优化的结果实际上就是 GT Answer 的形式
                
                # 拼接完整的原始文本 (Teacher 使用 Latent，Student 使用原始 Text)
                text_input = raw_d["question_text"] + original_think_text + embeds_end_think_cpu + optimized_answer_text
                
                # 对输入进行 Tokenize (务必确保不要加双重 bos 符号)
                input_ids = tokenizer.encode(text_input, return_tensors="pt", add_special_tokens=False).to(device)
                
                # 定位 Think 区域与 Answer 区域
                ids_q_len = tokenizer.encode(raw_d["question_text"], add_special_tokens=False, return_tensors="pt").shape[1]
                ids_think_len = tokenizer.encode(original_think_text, add_special_tokens=False, return_tensors="pt").shape[1]
                
                think_start = ids_q_len - 1
                think_end = think_start + ids_think_len
                answer_start = think_end + 1 # 加上 </think> 的长度
                
                # 学生模型前向传播
                student_outputs = model(input_ids=input_ids)
                student_logits = student_outputs.logits[0] # Shape: [SeqLen, Vocab]
                
                student_think_logits = student_logits[think_start:think_end, :]
                student_answer_logits = student_logits[answer_start:-1, :] # 切掉最后一个因为 shift
                
                # KL Loss (Think 阶段)
                target_soft = teacher_soft_targets[uid].to(device)
                
                # 由于 original_think_text 和 opt_latent 长度可能因为 Tokenize 有细微差异
                # 这里做截断对齐（取最小长度）
                min_think_len = min(student_think_logits.shape[0], target_soft.shape[0])
                stu_think_align = student_think_logits[:min_think_len]
                tgt_soft_align = target_soft[:min_think_len]
                
                tgt_probs = F.softmax(tgt_soft_align, dim=-1)
                stu_log_probs = F.log_softmax(stu_think_align, dim=-1)
                kl_loss = F.kl_div(stu_log_probs, tgt_probs, reduction='batchmean')
                
                # CE Loss (Answer 阶段)
                target_answer_ids = input_ids[0, answer_start+1:] # 错位 Shift 计算 Next Token
                min_ans_len = min(student_answer_logits.shape[0], target_answer_ids.shape[0])
                ce_loss = F.cross_entropy(
                    student_answer_logits[:min_ans_len], 
                    target_answer_ids[:min_ans_len]
                )
                
                total_loss = kl_loss + ce_loss * args.distill_ce_loss_weight
                total_loss = total_loss / len(batch_uids) # 梯度累加归一化
                total_loss.backward()
                
                batch_kl_loss += kl_loss.item()
                batch_ce_loss += ce_loss.item()
                batch_total_loss += total_loss.item() * len(batch_uids) # 还原真实 total
            
            distill_optimizer.step()
            
            wandb.log({
                "distill_step": global_distill_step,
                "distill/total_loss": batch_total_loss / len(batch_uids),
                "distill/kl_loss": batch_kl_loss / len(batch_uids),
                "distill/ce_loss": batch_ce_loss / len(batch_uids),
                "distill/lr": args.distill_lr
            })
            
            pbar.set_postfix({"Loss": f"{batch_total_loss / len(batch_uids):.3f}", "CE": f"{batch_ce_loss / len(batch_uids):.3f}"})
            global_distill_step += 1

    # 6. 收尾最终评测
    if distill_eval_future is not None:
        print("\n⏳ 蒸馏结束，等待后台收尾最后一轮 Eval...")
        process_distill_eval(distill_eval_future.result())
        
    print("\n🎉 蒸馏流水线彻底结束！")
    
    # 最后将蒸馏好的模型永久落盘
    final_save_path = f"./models/distilled_{run_name}"
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"💾 蒸馏最终权重已保存至: {final_save_path}")

    wandb.finish()
    vllm_engine.close()
    
if __name__ == "__main__":
    main()
