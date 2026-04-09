import os
import json
import math
import argparse
import traceback
import gzip
import torch
import torch.nn.functional as F
import asyncio
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

from math_utils import is_equiv, last_boxed_only_string, remove_boxed
from vllm_workers import VLLMDPWorkerPool

def dequantize_from_int8(tensor_int8, scale, target_dtype):
    """将 INT8 反量化为模型的 dtype (通常为 bfloat16)"""
    return (tensor_int8.to(torch.float32) * scale).to(target_dtype)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--merged_embeds_path", type=str, required=True, help="第一阶段合并后的 optimized_embeds_full.pt.gz 路径")
    parser.add_argument("--vllm_gpus", type=int, nargs='+', default=[1, 2, 3])
    
    parser.add_argument("--distill_epochs", type=int, default=3)
    parser.add_argument("--distill_lr", type=float, default=2e-5)
    parser.add_argument("--distill_ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--distill_eval_every", type=int, default=20)
    parser.add_argument("--distill_sample_filter", action="store_true", help="是否剔除没有优化的样本")
    parser.add_argument("--distill_eval_datasets", type=str, nargs='+', default=["HuggingFaceH4/MATH-500"])
    parser.add_argument("--distill_batch_size", type=int, default=4)
    parser.add_argument("--distill_grad_accum_steps", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    
    run_name = f"distill_v2_from_{os.path.basename(args.merged_embeds_path)}"
    wandb.init(project="L-GRPO-Math500-Distill", name=run_name, config=vars(args))
    wandb.define_metric("model_train_step")
    wandb.define_metric("distill/train/*", step_metric="model_train_step")
    wandb.define_metric("distill/eval/*", step_metric="model_train_step")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    
    print("🚀 正在加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, 
        dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    
    vllm_engine = VLLMDPWorkerPool(model_name=args.model_name, gpu_ids=args.vllm_gpus)
    executor = ThreadPoolExecutor(max_workers=1)
    
    def get_embeds(text):
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        return model.get_input_embeddings()(ids).detach(), ids

    embeds_end_think, ids_end_think = get_embeds("</think>")
    embeds_fast_conn, ids_fast_conn = get_embeds("The final answer is \n&&\n\\boxed{")

    print(f"📦 加载潜变量数据: {args.merged_embeds_path}")
    with gzip.open(args.merged_embeds_path, 'rb') as f:
        saved_dataset = torch.load(f)

    # =========================================================================
    # Phase 5.1 数据集筛选与组装 (包含重构 Tokenizer IDs)
    # =========================================================================
    distill_dataset = []
    print("⚙️ 正在重新组装蒸馏输入特征并反量化...")
    for item_pack in tqdm(saved_dataset, desc="Rebuilding Dataset"):
        acc = item_pack["last_optimal_metrics"].get("optimal_pure_acc", 0)
        if args.distill_sample_filter and acc <= 0:
            continue
            
        d = item_pack["metadata"]
        uid = d["uid"]
        
        # 重新生成 Token IDs 和 Embeds (避免之前存储占用过多磁盘)
        ids_q = tokenizer.encode(d['question_text'], return_tensors="pt", add_special_tokens=False)
        embeds_q_cpu = model.get_input_embeddings()(ids_q.to(device)).cpu().detach()
        ids_think = tokenizer.encode(d['thinking_text'], return_tensors="pt", add_special_tokens=False)
        ids_gt = tokenizer.encode(d['gt_text'], return_tensors="pt", add_special_tokens=False)
        
        # 反量化 Optimized Think Embeds
        opt_think_embeds = dequantize_from_int8(
            item_pack["tensors"]["optimal_embeds_int8"],
            item_pack["tensors"]["optimal_scale"],
            model.dtype
        )
        
        distill_dataset.append({
            "uid": uid,
            "ids_q": ids_q,
            "embeds_q": embeds_q_cpu,
            "target_think_embeds": opt_think_embeds,
            "ids_think": ids_think,
            "ids_gt": ids_gt, 
            "gt_text": d['gt_text']
        })
        
    print(f"📦 蒸馏数据集就绪，有效样本数: {len(distill_dataset)} / {len(saved_dataset)}")
    if len(distill_dataset) == 0:
        print("⚠️ 没有符合条件的样本进行蒸馏，流程结束。")
        return

    # =========================================================================
    # Phase 5.2 预计算 Target Soft Logits
    # =========================================================================
    print("⚙️ 正在预计算 Target Soft Logits...")
    target_soft_logits_dict = {}
    model.eval()
    with torch.no_grad():
        for item in tqdm(distill_dataset, desc="Precomputing Target Logits"):
            uid = item["uid"]
            emb_q = item["embeds_q"].to(device)
            emb_think = item["target_think_embeds"].to(device)
            
            full_emb = torch.cat([emb_q, emb_think], dim=1)
            outputs = model(inputs_embeds=full_emb)
            
            start_idx = emb_q.shape[1] - 1
            end_idx = full_emb.shape[1] - 1
            think_logits = outputs.logits[:, start_idx:end_idx, :] 
            
            target_soft_logits_dict[uid] = think_logits.cpu()
            
    torch.cuda.empty_cache()

    # =========================================================================
    # Phase 5.3 准备异步评测环境
    # =========================================================================
    def pass_at_k(n, c, k):
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
    distill_eval_future = None
    distill_step = 0
    
    def run_distill_eval_async(step, current_state_dict_path):
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            tqdm.write(f"\n🚀 [后台蒸馏评测] 正在同步权重至 vLM...")
            vllm_engine.update_weights(current_state_dict_path)
            
            sp_dict = {"max_tokens": 32768, "temperature": 0.7, "n": 32, "skip_special_tokens": False}
            metrics_to_log = {"model_train_step": step}
            
            total_pass1, total_pass8, total_pass16, total_pass32 = [], [], [], []
            
            for ds_id, ds_info in tqdm(eval_datasets.items(), desc="🌐 蒸馏评测总体进度", leave=False):
                ds, q_key, a_key = ds_info["data"], ds_info["q_key"], ds_info["a_key"]
                
                prompts = []
                for item in ds:
                    msg = [{"role": "user", "content": item[q_key]}]
                    prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
                
                tqdm.write(f"   --> vLLM 正在生成 {ds_id} (并发 32)...")
                batch_outputs = vllm_engine.generate(prompts, sp_dict, input_type="texts")
                
                ds_pass1, ds_pass8, ds_pass16, ds_pass32 = 0.0, 0.0, 0.0, 0.0
                
                for idx, item in enumerate(tqdm(ds, desc=f"📊 正在校验 {ds_id} 答案", leave=False)):
                    gt_ans = str(item[a_key]).strip()
                    correct_count = 0
                    
                    for output_ids in batch_outputs[idx]:
                        gen_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                        ans_cand = last_boxed_only_string(gen_text)
                        if ans_cand and is_equiv(remove_boxed(ans_cand), gt_ans):
                            correct_count += 1
                            
                    ds_pass1 += pass_at_k(32, correct_count, 1)
                    ds_pass8 += pass_at_k(32, correct_count, 8)
                    ds_pass16 += pass_at_k(32, correct_count, 16)
                    ds_pass32 += pass_at_k(32, correct_count, 32)
                
                num_samples = len(ds)
                metrics_to_log[f"distill/eval/{ds_id}/pass@1"] = ds_pass1 / num_samples
                metrics_to_log[f"distill/eval/{ds_id}/pass@8"] = ds_pass8 / num_samples
                metrics_to_log[f"distill/eval/{ds_id}/pass@16"] = ds_pass16 / num_samples
                metrics_to_log[f"distill/eval/{ds_id}/pass@32"] = ds_pass32 / num_samples
                
                total_pass1.append(ds_pass1 / num_samples)
                total_pass8.append(ds_pass8 / num_samples)
                total_pass16.append(ds_pass16 / num_samples)
                total_pass32.append(ds_pass32 / num_samples)

            if total_pass1:
                metrics_to_log["distill/eval/avg_pass@1"] = sum(total_pass1)/len(total_pass1)
                metrics_to_log["distill/eval/avg_pass@8"] = sum(total_pass8)/len(total_pass8)
                metrics_to_log["distill/eval/avg_pass@16"] = sum(total_pass16)/len(total_pass16)
                metrics_to_log["distill/eval/avg_pass@32"] = sum(total_pass32)/len(total_pass32)
                
            tqdm.write(f"✅ [后台蒸馏评测完成] 均值 Pass@1: {metrics_to_log.get('distill/eval/avg_pass@1', 0):.2%}")
            return metrics_to_log
            
        except Exception as e:
            tqdm.write(f"\n❌ [蒸馏评测严重崩溃] {str(e)}\n{traceback.format_exc()}")
            return None
    
    # =========================================================================
    # Phase 5.4 开启模型训练状态
    # =========================================================================
    model.requires_grad_(True)
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
            
            for item in batch:
                uid = item["uid"]
                ids_q = item["ids_q"].to(device)
                ids_think = item["ids_think"].to(device)
                ids_gt = item["ids_gt"].to(device)
                target_logits = target_soft_logits_dict[uid].to(device)
                
                full_ids = torch.cat([ids_q, ids_think, ids_end_think.to(device), ids_fast_conn.to(device), ids_gt], dim=1)
                
                outputs = model(full_ids)
                logits = outputs.logits
                
                think_start = ids_q.shape[1] - 1
                think_end = think_start + target_logits.shape[1]
                pred_think_logits = logits[0, think_start:think_end, :]
                
                log_probs = F.log_softmax(pred_think_logits, dim=-1)
                target_probs = F.softmax(target_logits[0], dim=-1)
                kl_loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
                
                ans_start = full_ids.shape[1] - ids_gt.shape[1] - 1
                ans_end = full_ids.shape[1] - 1
                pred_ans_logits = logits[0, ans_start:ans_end, :]
                
                ce_loss = F.cross_entropy(pred_ans_logits, ids_gt[0])
                
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
                        "gen_length": full_ids.shape[1]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
                
                if distill_step % args.distill_eval_every == 0:
                    if distill_eval_future is not None:
                        tqdm.write("⏳ 等待上一轮蒸馏评测结束...")
                        res = distill_eval_future.result()
                        if res: wandb.log(res)
                        
                    torch.save(model.state_dict(), shm_model_path)
                    distill_eval_future = executor.submit(run_distill_eval_async, distill_step, shm_model_path)
                
                distill_step += 1

    if distill_eval_future is not None:
        print("\n⏳ 蒸馏训练已全部完成，等待后台收尾最后一轮 Eval...")
        res = distill_eval_future.result()
        if res: wandb.log(res)

    print("🎉 蒸馏全流程结束！")
    wandb.finish()
    vllm_engine.close()
    
if __name__ == "__main__":
    main()