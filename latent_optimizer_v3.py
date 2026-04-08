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
# [🌟 架构升级] 辅助函数：跨设备搬运优化器状态
# =========================================================================
def move_optimizer_state(optimizer, device):
    """将原生优化器的内部张量状态（如动量 exp_avg 等）搬运到指定设备"""
    for param, state in optimizer.state.items():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

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
        self.W_emb = vocab_embeddings  # 在 GPU 上

    def step(self, latent_tensor, gamma):
        if latent_tensor.grad is None: return
        # 梯度传输到 GPU
        grad = latent_tensor.grad.to(self.W_emb.device).detach()
        with torch.no_grad():
            scores = torch.matmul(grad, self.W_emb.T)
            best_vocab_indices = torch.argmin(scores, dim=-1)
            best_embeds = self.W_emb[best_vocab_indices].to(latent_tensor.device)
            latent_tensor.copy_((1 - gamma) * latent_tensor + gamma * best_embeds)
            latent_tensor.grad.zero_()

def quantize_to_int8(tensor):
    """对隐状态进行对称 INT8 量化"""
    scale = tensor.abs().max() / 127.0
    int8_tensor = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return int8_tensor, scale

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
    parser.add_argument("--fw_gamma", type=float, default=0.1, help="Frank-Wolfe 优化器的初始 Gamma 步长")
    parser.add_argument("--conn_type", type=str, default="fast", choices=["fast", "original"])
    parser.add_argument("--reg_type", type=str, default="kl", choices=["kl", "lm"])
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

    # ==================================================
    # Phase 1: 构建 CPU 静态缓存 & CPU 潜变量注册表 (Batch 优化版)
    # ==================================================
    print("⚙️ Preparing Static Data & Latents (Batch Forwarding)...")
    static_data_cpu = {}
    global_latents = torch.nn.ParameterDict()
    active_uids = []
    
    global_history = {
        d['uid']: {"uid": d['uid'], "problem": d['question_text'], "gt_text": d['gt_text'], "steps": []} 
        for d in raw_data
    }

    # 设置预处理的 Batch Size
    prep_batch_size = args.batch_size
    for i in tqdm(range(0, len(raw_data), prep_batch_size), desc="Pre-computing (Batched)"):
        batch_samples = raw_data[i : i + prep_batch_size]
        
        batch_embeds_full = []
        batch_info = []
        
        for d in batch_samples:
            uid = d['uid']
            active_uids.append(uid)
            
            # 获取各部分 Embeddings
            embeds_q, ids_q = get_embeds(d['question_text'])
            embeds_conn, ids_conn = get_embeds(d['connector_text']) if args.conn_type == "original" else (embeds_fast_conn, ids_fast_conn)
            embeds_gt, ids_gt = get_embeds(d['gt_text'])
            embeds_pred, ids_pred = get_embeds(d['pred_text'])
            embeds_think, ids_think = get_embeds(d['thinking_text'])
            
            # 初始化 Latent (在 CPU)
            curr_think = torch.nn.Parameter(embeds_think.detach().cpu().clone())
            global_latents[uid] = curr_think
            
            # 记录位置信息
            think_start_idx = ids_q.shape[1] - 1
            think_end_idx = think_start_idx + ids_think.shape[1]
            
            full_emb = torch.cat([embeds_q, embeds_think, embeds_end_think, embeds_conn], dim=1)
            batch_embeds_full.append(full_emb)
            
            batch_info.append({
                "uid": uid, "ids_q": ids_q, "ids_conn": ids_conn, "ids_gt": ids_gt,
                "ids_pred": ids_pred, "ids_think": ids_think, "thinking_text": d['thinking_text'],
                "start": think_start_idx, "end": think_end_idx, "len": full_emb.shape[1]
            })

        # --- Batch Forward 开始 ---
        max_len = max(info["len"] for info in batch_info)
        padded_embeds = []
        attn_masks = []
        
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
            outputs = model(inputs_embeds=torch.cat(padded_embeds, dim=0), 
                           attention_mask=torch.cat(attn_masks, dim=0))
            all_logits = outputs.logits # [B, L, Vocab]

        # --- 处理与存储结果 ---
        for j, info in enumerate(batch_info):
            uid = info["uid"]
            # 提取该样本对应的 thinking 部分的 logits
            think_logits = all_logits[j, info["start"]:info["end"], :].unsqueeze(0)
            probs = F.softmax(think_logits, dim=-1)
            
            # 计算 Entropy Mask
            if args.mask_strategy == "top_k_entropy":
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                mask = torch.zeros_like(entropy, dtype=torch.float32)
                mask.scatter_(1, torch.topk(entropy, k=min(args.mask_max_k, entropy.shape[1]), dim=1).indices, 1.0)
            else:
                mask = torch.zeros((1, info["ids_think"].shape[1]), dtype=torch.float32, device=device)
                mask[:, :args.mask_max_k] = 1.0
            
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
                "thinking_text": info['thinking_text']
            }
        
        # 显存清理
        del all_logits, outputs, padded_embeds, attn_masks, batch_embeds_full
    torch.cuda.empty_cache()
    
    optimal_latents = {uid: {"acc": -1.0, "latent": None} for uid in global_latents.keys()}
    
    # [🌟 架构升级] 为每个数据单独实例化优化器对象，状态持久化在 CPU 内存中
    global_opts = {}
    if args.optimizer == "adam":
        for uid, param in global_latents.items():
            global_opts[uid] = torch.optim.Adam([param], lr=args.learning_rate)
    elif args.optimizer == "frank_wolfe":
        optimizer = FrankWolfeOptimizer(all_embeddings)

    run_name = wandb.run.name if wandb.run is not None else f"opt_v2_{args.optimizer}_{args.reg_type}"
    history_filename = f"./optimization_histories/optimization_history_{run_name}.jsonl"
    os.makedirs(os.path.dirname(history_filename), exist_ok=True)
    print(f"📁 History 将保存至: {history_filename}")
    history_file = open(history_filename, "a", encoding="utf-8")
    history_file.write(json.dumps({"config": vars(args)}, ensure_ascii=False) + "\n")
    history_file.flush()

    # ==================================================
    # Phase 2: 外层循环 (Global Steps / Epochs)
    # ==================================================
    for step in range(args.steps):
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
        
        # --------------------------------------------------
        # 2.1 内层循环: 动态 Batch 迭代优化
        # --------------------------------------------------
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
                
                # [🌟 架构升级] 按需加载：将参数原地变更为 GPU 张量
                p = global_latents[uid]
                p.data = p.data.to(device)
                if p.grad is not None:
                    p.grad.data = p.grad.data.to(device)
                curr_think_list.append(p)
                
                # [🌟 架构升级] 针对本 Batch 的数据，将 Adam 状态拉上 GPU 并更新 LR
                if args.optimizer == "adam":
                    opt = global_opts[uid]
                    move_optimizer_state(opt, device)
                    for pg in opt.param_groups:
                        pg['lr'] = current_lr
                elif args.optimizer == "frank_wolfe":
                    if p.grad is not None: p.grad.zero_()
                
                # 动态获取 Embeddings
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
                # 注意此处 curr_think_list[i] 在 GPU 上
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
                    
                    del h_chunk, logits_chunk, lse_chunk, orig_p_chunk, orig_idx_chunk
                    del curr_log_probs_topk_chunk, kl_chunk, embeds_chunk, target_ids_chunk
                    del orig_embeds_chunk, scaled_embeds_chunk, score_0_chunk, mask_chunk, new_logits_chunk
                    
                all_kl_div_list.append(kl_sum / think_len)
                batch_logprobs.append(logprobs_sum / think_len)
                
                s_gt = gt_loss.item()
                s_kl = (kl_sum / think_len).item()
                s_lm = -(logprobs_sum / think_len).item()
                s_reg = s_kl if args.reg_type == "kl" else s_lm
                s_total = s_gt + args.kl_weight * s_reg
                
                step_metrics[uid] = {
                    "total_loss": s_total, "gt_loss": s_gt, "kl_loss": s_kl, "lm_loss": s_lm
                }

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
            
            pbar.set_postfix({
                "Total": f"{total_loss.item():.3f}", "GT": f"{batch_gt_loss.item():.3f}",
                "KL": f"{batch_kl_loss.item():.3f}", "LM": f"{batch_lm_prior_loss.item():.3f}"
            })
            
            total_loss.backward()
            
            # [🌟 架构升级] 在 GPU 上应用 Mask，执行 Optimizer Step，随后释放回 CPU
            for i, uid in enumerate(batch_uids):
                p = curr_think_list[i]  # p 本质上就是 global_latents[uid]
                mask = curr_mask_list[i]
                
                if p.grad is not None:
                    # 在 GPU 上施加 Mask
                    p.grad.data.mul_(mask.to(device).to(p.dtype))
                
                if args.optimizer == "adam":
                    opt = global_opts[uid]
                    # 在 GPU 执行高性能 Step 计算
                    opt.step()
                    # 清理该样本在 GPU 的梯度释放显存
                    opt.zero_grad(set_to_none=True)
                    # 将更新后的动量状态搬回 CPU 进行持久化
                    move_optimizer_state(opt, torch.device("cpu"))
                elif args.optimizer == "frank_wolfe":
                    # FW 原地修改参数，它接受当前在 GPU 的张量 p
                    optimizer.step(p, gamma=current_fw_gamma)

                # 将 Latent 数据本身搬回 CPU，且确保持续存在于 Pinned Memory 中以加速后续轮次
                p.data = p.data.cpu().pin_memory()

            del full_embeds_batch, full_attention_mask, last_hidden
            torch.cuda.empty_cache()

        # --------------------------------------------------
        # 2.2 中间层评测 (Eval) & Logging
        # --------------------------------------------------
        wandb.log({
            "global_step": step,
            "train/epoch_total_loss": epoch_total_loss_sum / len(active_uids),
            "train/epoch_gt_loss": epoch_gt_loss_sum / len(active_uids),
            "train/epoch_kl_loss": epoch_kl_loss_sum / len(active_uids),
            "train/epoch_lm_loss": epoch_lm_loss_sum / len(active_uids),
            "train/lr": current_lr,
            "active_samples": len(active_uids)
        })

        if step % args.eval_every == 0 and step != args.steps - 1:
            total_pure_acc, total_forced_acc, total_fast_acc = 0.0, 0.0, 0.0
            total_rouge_l, total_change_ratio = 0.0, 0.0
            
            print(f"🚀 [Intermediate Eval] 计算潜空间漂移 & 发送 {len(active_uids)} 个潜变量至 vLLM ...")
            eval_pure_inputs, eval_forced_inputs, eval_fast_inputs = [], [], []
            
            for uid in active_uids:
                # 此时 global_latents[uid] 的 .data 均安稳地停留在 CPU，不会 OOM
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
                
                step_metrics[uid]['change_ratio'] = change_ratio
                step_metrics[uid]['rouge_L'] = rouge_l_f1
                
                total_change_ratio += change_ratio
                total_rouge_l += rouge_l_f1

                with torch.no_grad():
                    embeds_q = model.get_input_embeddings()(sd["ids_q"].to(device))
                    embeds_conn = model.get_input_embeddings()(sd["ids_conn"].to(device))
                
                eval_pure_inputs.append(torch.cat([embeds_q, ct, embeds_end_think], dim=1).squeeze(0))
                eval_forced_inputs.append(torch.cat([embeds_q, ct, embeds_end_think, embeds_conn], dim=1).squeeze(0))
                eval_fast_inputs.append(torch.cat([embeds_q, ct, embeds_end_think, embeds_fast_conn], dim=1).squeeze(0))

            modes = [('pure', eval_pure_inputs, 2048), ('forced', eval_forced_inputs, 128), ('fast', eval_fast_inputs, 128)]
            
            for mode, inputs_list, max_toks in tqdm(modes, desc=f"Step {step} Eval Modes"):
                batch_outputs = vllm_engine.generate(inputs_list, {"max_tokens": max_toks, "temperature": 0.7, "n": args.eval_k, "skip_special_tokens": False})
                if not batch_outputs: continue
                
                for idx, uid in enumerate(tqdm(active_uids, desc=f"Processing {mode} Results", leave=False)):
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
                
                if mode == 'pure':
                    for uid in active_uids:
                        current_pure_acc = step_metrics[uid]['pure_acc']
                        if current_pure_acc > optimal_latents[uid]["acc"]:
                            optimal_latents[uid]["acc"] = current_pure_acc
                            # 此刻 global_latents 本身就在 CPU，安全赋值
                            optimal_latents[uid]["latent"] = global_latents[uid].detach().clone()

            print(f"📊 [Inter-Eval] Pure: {total_pure_acc/len(active_uids):.2%} | Forced: {total_forced_acc/len(active_uids):.2%} | Fast: {total_fast_acc/len(active_uids):.2%}")
            
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
                print(f"  🔪 [修剪] 样本 {uid} 达标 (Loss: {gt_loss:.4f})，移出训练队列等待最终 Eval。")
                if "topk_probs" in static_data_cpu[uid]:
                    del static_data_cpu[uid]["topk_probs"]
                    del static_data_cpu[uid]["topk_indices"]
                    del static_data_cpu[uid]["grad_mask"]
            else:
                next_active_uids.append(uid)
                
        active_uids = next_active_uids

    # ==================================================
    # Phase 3: 终局全量评测 (Final Post-hoc Eval)
    # ==================================================
    print("\n" + "="*60)
    print("🎉 优化循环结束，开始执行全局 Final Evaluation ...")
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
    for uid in all_uids:
        history_file.write(json.dumps(global_history[uid], ensure_ascii=False) + "\n")
    history_file.flush()
    history_file.close()
    
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
    
    wandb.finish()
    vllm_engine.close()
    print(f"🎉 训练彻底结束！")

if __name__ == "__main__":
    main()