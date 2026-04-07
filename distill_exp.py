import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoConfig

# ==========================================
# 1. 数据集准备: Mock SoftInputsDatasets
# ==========================================
class SoftInputsDatasets(Dataset):
    def __init__(self, num_samples=100, seq_len=32, vocab_size=151936, hidden_size=2048):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Hard 输入 (离散词表ID)
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        
        # Soft 输入 (纯连续特征向量 - 用于 Teacher 前向)
        inputs_embeds_soft = torch.randn((self.seq_len, self.hidden_size))
        
        # 混合输入 (部分是正常词的 embedding，部分是特殊微调的 embedding)
        mixed_inputs_embeds = torch.randn((self.seq_len, self.hidden_size))
        
        # Target 标记：0=正常词，1=纯Embedding
        target_types = torch.randint(0, 2, (self.seq_len,))
        
        # 标签构建：正常词给ID，纯Embedding位置设为-100(告诉交叉熵忽略它)
        target_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        target_ids[target_types == 1] = -100 
        
        # 纯 Embedding 的目标向量 (仅在 target_types == 1 时有意义)
        target_embeds = torch.randn((self.seq_len, self.hidden_size))
        
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "inputs_embeds_soft": inputs_embeds_soft,
            "mixed_inputs_embeds": mixed_inputs_embeds,
            "target_types": target_types,
            "target_ids": target_ids,
            "target_embeds": target_embeds,
            "attention_mask": attention_mask
        }

# ==========================================
# 2. 核心损失函数定义 (纯函数)
# ==========================================

def compute_kl_loss(model, batch, device, temperature=1.0, alpha=1.0):
    """Mode 1: 混合 KL + CE对齐 (Hard预测常规词，Soft去拟合Teacher的Logits)"""
    input_ids = batch["input_ids"].to(device)
    inputs_embeds_soft = batch["inputs_embeds_soft"].to(device)
    target_types = batch["target_types"].to(device)
    target_ids = batch["target_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Teacher Forward (Soft)
    with torch.no_grad():
        outputs_soft = model(inputs_embeds=inputs_embeds_soft, attention_mask=attention_mask)
        logits_soft = outputs_soft.logits.detach()

    # Student Forward (Hard)
    outputs_hard = model(input_ids=input_ids, attention_mask=attention_mask)
    logits_hard = outputs_hard.logits

    # 错位对齐
    shift_logits_hard = logits_hard[:, :-1, :].contiguous()
    shift_logits_soft = logits_soft[:, :-1, :].contiguous()
    shift_target_types = target_types[:, 1:].contiguous()
    shift_target_ids = target_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    hard_mask = (shift_target_types == 0) & (shift_mask == 1)
    soft_mask = (shift_target_types == 1) & (shift_mask == 1)

    loss_ce, loss_kl = 0.0, 0.0

    if hard_mask.any():
        loss_ce = F.cross_entropy(shift_logits_hard[hard_mask], shift_target_ids[hard_mask])

    if soft_mask.any():
        log_prob_hard = F.log_softmax(shift_logits_hard[soft_mask] / temperature, dim=-1)
        prob_soft = F.softmax(shift_logits_soft[soft_mask] / temperature, dim=-1)
        loss_kl = F.kl_div(log_prob_hard, prob_soft, reduction='batchmean') * (temperature ** 2)

    return loss_ce + alpha * loss_kl


def compute_hidden_mse_loss(model, batch, device, alpha=1.0):
    """Mode 5: 混合 Hidden MSE + CE对齐 (隐空间状态直接对齐)"""
    input_ids = batch["input_ids"].to(device)
    inputs_embeds_soft = batch["inputs_embeds_soft"].to(device)
    target_types = batch["target_types"].to(device)
    target_ids = batch["target_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs_soft = model(inputs_embeds=inputs_embeds_soft, attention_mask=attention_mask, output_hidden_states=True)
        hidden_soft = outputs_soft.hidden_states[-1].detach()

    outputs_hard = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    logits_hard = outputs_hard.logits
    hidden_hard = outputs_hard.hidden_states[-1]

    shift_logits_hard = logits_hard[:, :-1, :].contiguous()
    shift_hidden_hard = hidden_hard[:, :-1, :].contiguous()
    shift_hidden_soft = hidden_soft[:, :-1, :].contiguous()
    
    shift_target_types = target_types[:, 1:].contiguous()
    shift_target_ids = target_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    hard_mask = (shift_target_types == 0) & (shift_mask == 1)
    soft_mask = (shift_target_types == 1) & (shift_mask == 1)

    loss_ce, loss_mse = 0.0, 0.0

    if hard_mask.any():
        loss_ce = F.cross_entropy(shift_logits_hard[hard_mask], shift_target_ids[hard_mask])

    if soft_mask.any():
        loss_mse = F.mse_loss(shift_hidden_hard[soft_mask], shift_hidden_soft[soft_mask])

    return loss_ce + alpha * loss_mse


def compute_masked_ce_loss(model, batch, device):
    """Mode 3: Masked CE Loss (让模型学会作为Context去理解连续向量)"""
    mixed_inputs = batch["mixed_inputs_embeds"].to(device)
    target_ids = batch["target_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    vocab_size = model.config.vocab_size

    outputs = model(inputs_embeds=mixed_inputs, attention_mask=attention_mask)
    
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()

    # 自动忽略 label 为 -100 的纯 Embedding 位置
    loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)
    return loss


def compute_soft_logit_loss(model, batch, device):
    """Mode 4: 软 Logit 点积 (将目标连续向量作为扩展词表)"""
    mixed_inputs = batch["mixed_inputs_embeds"].to(device)
    target_types = batch["target_types"].to(device)
    target_ids = batch["target_ids"].to(device)
    target_embeds = batch["target_embeds"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    vocab_size = model.config.vocab_size

    outputs = model(inputs_embeds=mixed_inputs, attention_mask=attention_mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    
    # 构建扩展词表
    base_logits = model.lm_head(hidden_states)
    soft_logit_scores = torch.sum(hidden_states * target_embeds, dim=-1, keepdim=True)
    
    # 缓解点积数值过大导致的 softmax 饱和
    scale_factor = hidden_states.shape[-1] ** 0.5 
    soft_logit_scores = soft_logit_scores / scale_factor

    extended_logits = torch.cat([base_logits, soft_logit_scores], dim=-1)

    shift_logits = extended_logits[:, :-1, :].contiguous()
    shift_target_types = target_types[:, 1:].contiguous()
    shift_target_ids = target_ids[:, 1:].contiguous()
    
    # 修改软目标位置的 label 为 V (即新加的这一列)
    final_labels = shift_target_ids.clone()
    final_labels[shift_target_types == 1] = vocab_size

    loss = F.cross_entropy(shift_logits.view(-1, vocab_size + 1), final_labels.view(-1), ignore_index=-100)
    return loss

# ==========================================
# 3. 训练主循环
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 模型初始化 (用小模型跑通测试) ---
    print("Initializing dummy Qwen config...")
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    config.num_hidden_layers = 2 # 减少层数方便快速无OOM测试
    model = AutoModelForCausalLM.from_config(config).to(device)
    model.train()

    # --- 2. 数据与优化器 ---
    dataset = SoftInputsDatasets(num_samples=64, seq_len=32, vocab_size=config.vocab_size, hidden_size=config.hidden_size)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- 3. 配置参数 ---
    num_epochs = 2
    # 可选列表: "kl_mixed", "hidden_mse_mixed", "masked_ce", "soft_logit"
    MODE = "hidden_mse_mixed" 
    print(f"Starting training loop with Mode: [{MODE}]...")
    
    # --- 4. 纯循环迭代 ---
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 根据模式分发计算 Loss
            if MODE == "kl_mixed":
                loss = compute_kl_loss(model, batch, device, alpha=1.0)
            elif MODE == "hidden_mse_mixed":
                loss = compute_hidden_mse_loss(model, batch, device, alpha=1.0)
            elif MODE == "masked_ce":
                loss = compute_masked_ce_loss(model, batch, device)
            elif MODE == "soft_logit":
                loss = compute_soft_logit_loss(model, batch, device)
            else:
                raise ValueError(f"Unknown mode: {MODE}")

            loss.backward()
            
            # 梯度裁剪 (防止软标签或点积对齐导致初期梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if step % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Step [{step}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.4f} ---")

if __name__ == "__main__":
    main()