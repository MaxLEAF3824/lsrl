import os
import json
import math
import argparse
import tokenize
import traceback
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from rouge_score import rouge_scorer
from math_verify import parse as math_verify_parse
from math_verify import verify as math_verify_verify

# =========================================================================
# [0] 依赖引入 (确保 math_utils 和 vllm_workers 可用)
# =========================================================================
from math_utils import is_correct_v3, last_boxed_only_string, remove_boxed, is_equiv
from vllm_workers import VLLMDPWorkerPool

# =========================================================================
# [1] 数据集与内存优化管理
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
        grad = latent_tensor.grad.detach()
        with torch.no_grad():
            scores = torch.matmul(grad, self.W_emb.T)
            best_vocab_indices = torch.argmin(scores, dim=-1)
            latent_tensor.copy_((1 - gamma) * latent_tensor + gamma * self.W_emb[best_vocab_indices])
            latent_tensor.grad.zero_()

# =========================================================================
# [3] 主流程 (Epoch-based 架构)
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--vllm_gpus", type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--steps", type=int, default=40, help="实际上是 Epoch 数量")
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_k", type=int, default=32)
    parser.add_argument("--mask_strategy", type=str, default="top_k_entropy", choices=["top_k_entropy", "first_k"])
    parser.add_argument("--mask_max_k", type=int, default=32768)
    parser.add_argument("--grad_direction", type=str, default="positive", choices=["positive", "negative"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "frank_wolfe"])
    parser.add_argument("--conn_type", type=str, default="fast", choices=["fast", "original"])
    parser.add_argument("--reg_type", type=str, default="kl", choices=["kl", "lm_prior"])
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_threshold", type=float, default=1e-3)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"🚀 Starting Latent Optimization with Config: {vars(args)}")
    
    wandb.init(project="L-GRPO-Math500", config=vars(args))
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading Main Model with Flash Attention 2 to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, 
        dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) 
    
    vllm_engine = VLLMDPWorkerPool(model_name=args.model_name, gpu_ids=args.vllm_gpus)
    
    def get_embeds(text):
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        return model.get_input_embeddings()(ids).detach(), ids

    embeds_end_think, ids_end_think = get_embeds("</think>")
    embeds_fast_conn, ids_fast_conn = get_embeds("The final answer is \n&&\n\\boxed{")
    all_embeddings = model.get_input_embeddings().weight.detach()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    wrong_dataset = build_math_wrong_dataset(args.file_path, tokenizer)
    
    # 临时测试：截取前 部分 个样本跑 Demo
    raw_data = wrong_dataset.flat_data[74:82]
    # ==================================================
    # Phase 1: 构建 CPU 静态缓存 & GPU 活跃潜变量注册表
    # ==================================================
    print("⚙️ Preparing Static Data (CPU) & Latents (GPU)...")
    static_data_cpu = {}
    global_latents = torch.nn.ParameterDict()
    active_uids = []
    
    global_history = {
        d['uid']: {"uid": d['uid'], "problem": d['question_text'], "gt_text": d['gt_text'], "steps": []} 
        for d in raw_data
    }

    for d in tqdm(raw_data, desc="Pre-computing"):
        uid = d['uid']
        active_uids.append(uid)
        
        embeds_q, ids_q = get_embeds(d['question_text'])
        embeds_conn, ids_conn = get_embeds(d['connector_text']) if args.conn_type == "original" else (embeds_fast_conn, ids_fast_conn)
        embeds_gt, ids_gt = get_embeds(d['gt_text'])
        embeds_pred, ids_pred = get_embeds(d['pred_text'])
        embeds_think, ids_think = get_embeds(d['thinking_text'])
        
        curr_think = torch.nn.Parameter(embeds_think.detach().clone())
        global_latents[uid] = curr_think
        
        with torch.no_grad():
            orig_full = torch.cat([embeds_q, embeds_think, embeds_end_think, embeds_conn], dim=1)
            orig_out = model(inputs_embeds=orig_full)
            think_start_idx = ids_q.shape[1] - 1
            think_end_idx = think_start_idx + ids_think.shape[1]
            orig_probs = F.softmax(orig_out.logits[:, think_start_idx:think_end_idx, :], dim=-1)
            
            if args.mask_strategy == "top_k_entropy":
                entropy = -torch.sum(orig_probs * torch.log(orig_probs + 1e-10), dim=-1)
                mask = torch.zeros_like(entropy, dtype=torch.float32)
                mask.scatter_(1, torch.topk(entropy, k=min(args.mask_max_k, entropy.shape[1]), dim=1).indices, 1.0)
            else:
                mask = torch.zeros_like(ids_think, dtype=torch.float32)
                mask[:, :args.mask_max_k] = 1.0
                
            topk_probs, topk_indices = torch.topk(orig_probs, k=100, dim=-1)
            
            static_data_cpu[uid] = {
                "ids_q": ids_q.cpu(), "embeds_q": embeds_q.cpu(),
                "ids_conn": ids_conn.cpu(), "embeds_conn": embeds_conn.cpu(),
                "ids_gt": ids_gt.cpu(), "embeds_gt": embeds_gt.cpu(),
                "ids_pred": ids_pred.cpu(), "embeds_pred": embeds_pred.cpu(),
                "ids_think": ids_think.cpu(),
                "grad_mask": mask.unsqueeze(-1).cpu(),
                "topk_probs": topk_probs.cpu(), "topk_indices": topk_indices.cpu(),
                "thinking_text": d['thinking_text']
            }
            del orig_full, orig_out, orig_probs
            
    torch.cuda.empty_cache()
    
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(global_latents.values(), lr=args.learning_rate)
    elif args.optimizer == "frank_wolfe":
        optimizer = FrankWolfeOptimizer(all_embeddings)

    # 从 wandb 获取当前 run 的名字 (即使 wandb 自动追加了随机后缀也能准确获取)
    run_name = wandb.run.name if wandb.run is not None else f"opt_v2_{args.optimizer}_{args.reg_type}"
    history_filename = f"optimization_history_{run_name}.jsonl"
    print(f"📁 History 将保存至: {history_filename}")
    
    # 打开文件
    history_file = open(history_filename, "a", encoding="utf-8")
    
    # 👇 新增：在文件第 0 行写入当前的 config 字典
    history_file.write(json.dumps({"config": vars(args)}, ensure_ascii=False) + "\n")
    history_file.flush()
    # 👆 新增结束

    # ==================================================
    # Phase 2: 外层循环 (Global Steps / Epochs)
    # ==================================================
    for step in range(args.steps):
        if not active_uids:
            print("🎉 所有样本均已触发 Early Stop，训练提前结束！")
            break
            
        print(f"\n==============================================")
        print(f"🔄 Global Step {step+1}/{args.steps} | Active Samples: {len(active_uids)}")
        print(f"\n==============================================")
        
        progress = step / max(1, args.steps - 1)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        current_lr = (args.learning_rate * 0.1) + (args.learning_rate * 0.9) * cosine_factor
        
        if args.optimizer == "adam":
            for pg in optimizer.param_groups: pg['lr'] = current_lr
        
        step_metrics = {uid: {} for uid in active_uids}
        
        # 记录 Epoch 全局累计 Loss
        epoch_gt_loss_sum = 0.0
        epoch_kl_loss_sum = 0.0
        epoch_lm_loss_sum = 0.0
        epoch_total_loss_sum = 0.0
        
        # --------------------------------------------------
        # 2.1 内层循环: 动态 Batch 迭代优化
        # --------------------------------------------------
        import random
        random.shuffle(active_uids) 
        
        mini_batches = [active_uids[i:i + args.batch_size] for i in range(0, len(active_uids), args.batch_size)]
        pbar = tqdm(mini_batches, desc=f"Step {step+1} Optimizing")
        
        for batch_uids in pbar:
            if args.optimizer == "adam": optimizer.zero_grad(set_to_none=True)
            elif args.optimizer == "frank_wolfe":
                for uid in batch_uids:
                    if global_latents[uid].grad is not None: global_latents[uid].grad.zero_()

            curr_think_list, curr_mask_list = [], []
            embeds_q_list, embeds_conn_list, embeds_gt_list, embeds_pred_list = [], [], [], []
            ids_q_list, ids_think_list, ids_conn_list, ids_gt_list, ids_pred_list = [], [], [], [], []
            
            for uid in batch_uids:
                curr_think_list.append(global_latents[uid])
                sd = static_data_cpu[uid]
                
                curr_mask_list.append(sd["grad_mask"].to(device))
                embeds_q_list.append(sd["embeds_q"].to(device))
                embeds_conn_list.append(sd["embeds_conn"].to(device))
                embeds_gt_list.append(sd["embeds_gt"].to(device))
                embeds_pred_list.append(sd["embeds_pred"].to(device))
                
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
                    mask = torch.zeros_like(logits_chunk, dtype=torch.bool)
                    mask.scatter_(2, target_ids_chunk.unsqueeze(-1), True)
                    
                    new_logits_chunk = torch.where(mask, score_0_chunk.unsqueeze(-1), logits_chunk)
                    logprobs_sum += (score_0_chunk - torch.logsumexp(new_logits_chunk, dim=-1)).sum()
                    
                    del h_chunk, logits_chunk, lse_chunk, orig_p_chunk, orig_idx_chunk
                    del curr_log_probs_topk_chunk, kl_chunk, embeds_chunk, target_ids_chunk
                    del orig_embeds_chunk, scaled_embeds_chunk, score_0_chunk, mask, new_logits_chunk
                    
                all_kl_div_list.append(kl_sum / think_len)
                batch_logprobs.append(logprobs_sum / think_len)
                
                all_kl_div_list.append(kl_sum / think_len)
                batch_logprobs.append(logprobs_sum / think_len)
                
                # 👇 新增与修改：计算单样本的 total_loss 并写入 step_metrics
                s_gt = gt_loss.item()
                s_kl = (kl_sum / think_len).item()
                s_lm = -(logprobs_sum / think_len).item()
                
                # 根据你的 config 计算加权 total_loss
                s_reg = s_kl if args.reg_type == "kl" else s_lm
                s_total = s_gt + args.kl_weight * s_reg
                
                step_metrics[uid] = {
                    "total_loss": s_total,
                    "gt_loss": s_gt,
                    "kl_loss": s_kl,
                    "lm_loss": s_lm
                }

            # Batch 级 Loss 聚合
            batch_gt_loss = torch.stack(all_gt_loss).mean()
            batch_kl_loss = torch.stack(all_kl_div_list).mean()
            batch_lm_prior_loss = -torch.stack(batch_logprobs).mean()
            
            reg_loss = batch_kl_loss if args.reg_type == "kl" else batch_lm_prior_loss
            total_loss = batch_gt_loss + args.kl_weight * reg_loss
            
            # 更新 Epoch 总 Loss (累加计算均值)
            batch_sample_count = len(batch_uids)
            epoch_gt_loss_sum += batch_gt_loss.item() * batch_sample_count
            epoch_kl_loss_sum += batch_kl_loss.item() * batch_sample_count
            epoch_lm_loss_sum += batch_lm_prior_loss.item() * batch_sample_count
            epoch_total_loss_sum += total_loss.item() * batch_sample_count
            
            # 丰富进度条内容
            pbar.set_postfix({
                "Total": f"{total_loss.item():.3f}", 
                "GT": f"{batch_gt_loss.item():.3f}",
                "KL": f"{batch_kl_loss.item():.3f}",
                "LM": f"{batch_lm_prior_loss.item():.3f}"
            })
            
            total_loss.backward()
            for mask, latent in zip(curr_mask_list, curr_think_list):
                latent.grad.mul_(mask.to(latent.dtype))
                
            if args.optimizer == "adam": optimizer.step()
            elif args.optimizer == "frank_wolfe":
                for latent, mask in zip(curr_think_list, curr_mask_list):
                    optimizer.step(latent, gamma=0.01 + 0.09 * cosine_factor)

            del full_embeds_batch, full_attention_mask, last_hidden
            torch.cuda.empty_cache()

        # --------------------------------------------------
        # 2.2 全局评测 (ROUGE-L, Acc, Wandb Logging)
        # --------------------------------------------------
        # 1. 记录 Epoch 级别各种平均 Loss 到 WandB
        wandb.log({
            "global_step": step,
            "train/epoch_total_loss": epoch_total_loss_sum / len(active_uids),
            "train/epoch_gt_loss": epoch_gt_loss_sum / len(active_uids),
            "train/epoch_kl_loss": epoch_kl_loss_sum / len(active_uids),
            "train/epoch_lm_loss": epoch_lm_loss_sum / len(active_uids),
            "train/lr": current_lr,
            "active_samples": len(active_uids)
        })

        if step % args.eval_every == 0 or step == args.steps - 1:
            total_pure_acc, total_forced_acc, total_fast_acc = 0.0, 0.0, 0.0
            total_rouge_l, total_change_ratio = 0.0, 0.0
            
            print(f"🚀 [Global Eval] 1/2 计算潜空间 Token 漂移度 (ROUGE-L & Change Ratio)...")
            for uid in active_uids:
                ct = global_latents[uid].to(all_embeddings.dtype).detach()
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
                
                step_metrics[uid]['change_ratio'] = change_ratio
                step_metrics[uid]['rouge_L'] = rouge_l_f1
                
                total_change_ratio += change_ratio
                total_rouge_l += rouge_l_f1

            print(f"🚀 [Global Eval] 2/2 发送 {len(active_uids)} 个潜变量至 vLLM Worker Pool...")
            eval_pure_inputs, eval_forced_inputs, eval_fast_inputs = [], [], []
            for uid in active_uids:
                ct = global_latents[uid].to(all_embeddings.dtype).detach()
                sd = static_data_cpu[uid]
                embeds_q = sd["embeds_q"].to(device)
                embeds_conn = sd["embeds_conn"].to(device)
                
                eval_pure_inputs.append(torch.cat([embeds_q, ct, embeds_end_think], dim=1).squeeze(0))
                eval_forced_inputs.append(torch.cat([embeds_q, ct, embeds_end_think, embeds_conn], dim=1).squeeze(0))
                eval_fast_inputs.append(torch.cat([embeds_q, ct, embeds_end_think, embeds_fast_conn], dim=1).squeeze(0))

            modes = [('pure', eval_pure_inputs, 2048), ('forced', eval_forced_inputs, 128), ('fast', eval_fast_inputs, 128)]
            
            for mode, inputs_list, max_toks in modes:
                batch_outputs = vllm_engine.generate(inputs_list, {"max_tokens": max_toks, "temperature": 0.7, "n": args.eval_k, "skip_special_tokens": False})
                if not batch_outputs: continue
                
                for idx, uid in enumerate(active_uids):
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
                    
                    # 👇 新增：在 pure 模式下，保存第一个生成的文本，方便事后分析
                    if mode == 'pure' and len(batch_outputs[idx]) > 0:
                        # 获取第一个生成的序列并解码
                        sample_gen = tokenizer.decode(batch_outputs[idx][0], skip_special_tokens=True)
                        step_metrics[uid]['sample_gen_text'] = sample_gen
                    # 👆 新增结束
                    
                    if mode == 'pure': total_pure_acc += acc
                    elif mode == 'forced': total_forced_acc += acc
                    elif mode == 'fast': total_fast_acc += acc

            # 2. 完整记录评测平均值到终端与 WandB
            print(f"📊 [Eval Results] Pure Acc: {total_pure_acc/len(active_uids):.2%} | Forced Acc: {total_forced_acc/len(active_uids):.2%} | Fast Acc: {total_fast_acc/len(active_uids):.2%}")
            print(f"📉 [Latent Drift] ROUGE-L: {total_rouge_l/len(active_uids):.4f} | Change Ratio: {total_change_ratio/len(active_uids):.2%}")
            
            wandb.log({
                "global_step": step,
                "eval/avg_pure_acc": total_pure_acc / len(active_uids),
                "eval/avg_forced_acc": total_forced_acc / len(active_uids),
                "eval/avg_fast_acc": total_fast_acc / len(active_uids),
                "eval/avg_rouge_L": total_rouge_l / len(active_uids),
                "eval/avg_change_ratio": total_change_ratio / len(active_uids)
            })

        # --------------------------------------------------
        # 2.3 Early Stop 动态修剪
        # --------------------------------------------------
        next_active_uids = []
        for uid in active_uids:
            global_history[uid]["steps"].append({"step": step, "metrics": step_metrics[uid]})
            
            gt_loss = step_metrics[uid]["gt_loss"]
            if args.early_stop and gt_loss < args.early_stop_threshold:
                print(f"  🔪 [修剪] 样本 {uid} 达标 (Loss: {gt_loss:.4f})，已写出并移出训练队列。")
                history_file.write(json.dumps(global_history[uid], ensure_ascii=False) + "\n")
                history_file.flush()
                del static_data_cpu[uid] 
            else:
                next_active_uids.append(uid)
                
        active_uids = next_active_uids

    if active_uids:
        print(f"\n📁 写入剩余 {len(active_uids)} 个未早停的样本历史...")
        for uid in active_uids:
            history_file.write(json.dumps(global_history[uid], ensure_ascii=False) + "\n")
        history_file.flush()
    
    history_file.close()
    wandb.finish()
    vllm_engine.close()
    print(f"✅ 训练执行完毕！全局记录已存储于 {history_filename}")

if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    # wrong_dataset = build_math_wrong_dataset("/workspace/yiqiuguo/lsrl/qwen3-1.7b_math-500_rollout8_len32768_final.jsonl", tokenizer)
    # print('wrong_dataset: ', len(wrong_dataset))
    # for i,d in enumerate(wrong_dataset.flat_data):
    #     print(i,d['uid'])
    main()