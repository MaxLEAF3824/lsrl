import torch
import gzip
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 引入你的评测函数和 vLLM 引擎
from math_utils import is_correct_v3
from vllm_workers import VLLMDPWorkerPool

# 1. 基础配置与加载
MODEL_NAME = "Qwen/Qwen3-1.7B"
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

print("⏳ Loading Model & Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEVICE)
model.requires_grad_(False)

print("⏳ Loading VLLM Engine...")
vllm_engine = VLLMDPWorkerPool(model_name=MODEL_NAME, gpu_ids=[3])

# 2. 读取压缩的优化历史
file_path = "/workspace/yiqiuguo/lsrl/optimization_histories/optimized_embeds_dulcet-aardvark-117.pt.gz"
print(f"📁 Unzipping and loading {file_path}...")
with gzip.open(file_path, 'rb') as f:
    saved_data = torch.load(f)

print(f"✅ Loaded {len(saved_data)} samples. Starting Restoration and Evaluation...")

# 获取辅助 Embeddings
ids_end_think = tokenizer.encode("</think>", return_tensors="pt", add_special_tokens=False).to(DEVICE)
embeds_end_think = model.get_input_embeddings()(ids_end_think).detach()

eval_inputs = []
gt_texts = []
saved_accs = []

# 3. 批量还原 Optimal Latents
for item in tqdm(saved_data, desc="Restoring INT8 Tensors"):
    metadata = item["metadata"]
    gt_text = metadata["gt_text"]
    
    # 获取原始的 thinking embeddings (Base)
    ids_think = tokenizer.encode(metadata["thinking_text"], return_tensors="pt", add_special_tokens=False).to(DEVICE)
    base_embeds = model.get_input_embeddings()(ids_think).detach()
    
    # 取出 Optimal Delta
    opt_dict = item["tensors"]["optimal_delta"]
    if opt_dict is None:
        # 如果 optimal 是 None，说明没有更新或退回到 last
        opt_dict = item["tensors"]["last_delta"]
        
    q_tensor = opt_dict["q_tensor"].to(DEVICE)
    scale = opt_dict["scale"]
    
    # 💥 核心：反量化恢复出连续的浮点隐状态！
    restored_opt_latent = base_embeds + (q_tensor.to(model.dtype) * scale)
    
    # 组装纯净模式 (Pure Mode) 的输入: [Question] + [Restored Thinking] + [</think>]
    ids_q = tokenizer.encode(metadata["question_text"], return_tensors="pt", add_special_tokens=False).to(DEVICE)
    embeds_q = model.get_input_embeddings()(ids_q).detach()
    
    full_embeds = torch.cat([embeds_q, restored_opt_latent, embeds_end_think], dim=1).squeeze(0)
    
    eval_inputs.append(full_embeds)
    gt_texts.append(gt_text)
    saved_accs.append(item["last_optimal_metrics"]["optimal_pure_acc"])

# 4. 送入 vLLM 重新生成
EVAL_K = 32
print(f"🚀 Sending {len(eval_inputs)} restored latents to vLLM (n={EVAL_K})...")
batch_outputs = vllm_engine.generate(
    eval_inputs, 
    {"max_tokens": 2048, "temperature": 0.7, "n": EVAL_K, "skip_special_tokens": False}
)

# 5. 评测并对比误差
print("\n" + "="*50)
print("📊 INT