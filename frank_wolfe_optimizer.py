import torch


class FrankWolfeOptimizer:
    def __init__(self, vocab_embeddings, adaptive_mode="off", top_k=0, top_p=0.0):
        self.W_emb = vocab_embeddings
        self.adaptive_mode = adaptive_mode
        self.top_k = top_k
        self.top_p = top_p

    def step(self, latent_tensor, gamma, valid_token_indices=None):
        if latent_tensor.grad is None:
            return None, None
            
        grad = latent_tensor.grad.to(self.W_emb.device).detach()
        seq_len = grad.shape[1]
        
        with torch.no_grad():
            grad_norms = torch.norm(grad, p=2, dim=-1) # [batch, seq_len]
            eff_gamma = torch.full_like(grad_norms, gamma)
            
            # --- 保持原有的 Adaptive Gamma 逻辑不变 ---
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

            # =======================================================
            # 新增功能核心：限制优化方向到 Top-K Token 候选集
            # =======================================================
            if valid_token_indices is not None:
                # valid_token_indices 的形状应为: [batch, seq_len, K]
                # 获取候选词的 embeddings: [batch, seq_len, K, hidden_dim]
                candidate_embeds = self.W_emb[valid_token_indices].to(latent_tensor.device)
                
                # 计算受限候选词的点积得分
                # grad shape: [batch, seq_len, hidden_dim] -> unsqueeze -> [batch, seq_len, 1, hidden_dim]
                # 对应位相乘后求和，得到每个候选词的分数: [batch, seq_len, K]
                scores = torch.sum(grad.unsqueeze(-2) * candidate_embeds, dim=-1)
                
                # 在 K 个候选项中找到内积最小（最优）的索引
                best_k_idx = torch.argmin(scores, dim=-1) # [batch, seq_len]
                
                # 将局部的 K 索引映射回全局的真实 Vocab Indices
                best_vocab_indices = torch.gather(
                    valid_token_indices.to(latent_tensor.device), 
                    dim=2, 
                    index=best_k_idx.unsqueeze(-1)
                ).squeeze(-1) # [batch, seq_len]
                
                best_embeds = self.W_emb[best_vocab_indices].to(latent_tensor.device)
            else:
                # 原有的全词表打分逻辑
                scores = torch.matmul(grad, self.W_emb.T)
                best_vocab_indices = torch.argmin(scores, dim=-1)
                best_embeds = self.W_emb[best_vocab_indices].to(latent_tensor.device)
            
            # 执行 FW 步进
            eff_gamma_expanded = eff_gamma.unsqueeze(-1).to(latent_tensor.device)
            latent_tensor.copy_((1 - eff_gamma_expanded) * latent_tensor + eff_gamma_expanded * best_embeds)
            latent_tensor.grad.zero_()
            
            return best_vocab_indices, eff_gamma