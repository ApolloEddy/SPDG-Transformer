import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FrozenStructuralPrior(nn.Module):
    def __init__(self, seq_len: int, sparsity: float = 0.1, pattern: str = 'local'):
        super().__init__()
        self.seq_len = seq_len
        self.sparsity = sparsity
        self.pattern = pattern
        
        self.register_buffer('prior_mask', self._generate_prior_mask())
    
    def _generate_prior_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.seq_len, self.seq_len)
        
        if self.pattern == 'local':
            window_size = int(self.seq_len * self.sparsity)
            for i in range(self.seq_len):
                start = max(0, i - window_size // 2)
                end = min(self.seq_len, i + window_size // 2 + 1)
                mask[i, start:end] = 1.0
        
        elif self.pattern == 'global':
            num_global = int(self.seq_len * self.sparsity)
            for i in range(self.seq_len):
                indices = torch.randperm(self.seq_len)[:num_global]
                mask[i, indices] = 1.0
        
        elif self.pattern == 'block':
            block_size = int(self.seq_len * (1 - self.sparsity) ** 0.5)
            num_blocks = self.seq_len // block_size
            for bi in range(num_blocks):
                for bj in range(num_blocks):
                    if abs(bi - bj) <= 1:
                        start_i, end_i = bi * block_size, min((bi + 1) * block_size, self.seq_len)
                        start_j, end_j = bj * block_size, min((bj + 1) * block_size, self.seq_len)
                        mask[start_i:end_i, start_j:end_j] = 1.0
        
        return mask
    
    def forward(self) -> torch.Tensor:
        return self.prior_mask


class ResidualSurprise(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor, delta_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        norm_delta_x = torch.norm(delta_x, dim=-1, keepdim=True)
        
        u = norm_delta_x / (norm_x + self.eps)
        
        aggregated_u = u.mean(dim=1)
        
        return u, aggregated_u


class DynamicGating(nn.Module):
    def __init__(self, lambda_max: float = 10.0, alpha: float = 5.0):
        super().__init__()
        self.lambda_max = lambda_max
        self.alpha = alpha
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        lambda_coef = self.lambda_max * torch.exp(-self.alpha * u)
        return lambda_coef


class ClippingAttentionEngine(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.residual_surprise = ResidualSurprise()
        self.dynamic_gating = DynamicGating()
    
    def forward(
        self,
        x: torch.Tensor,
        prior_mask: torch.Tensor,
        delta_x: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        A = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if delta_x is not None:
            u, aggregated_u = self.residual_surprise(x, delta_x)
            lambda_coef = self.dynamic_gating(aggregated_u)
            lambda_coef = lambda_coef.view(batch_size, 1, 1, 1)
            
            B = prior_mask.unsqueeze(0).unsqueeze(0) * lambda_coef
            
            L = A + B
            
            mask = (lambda_coef > 1.0).squeeze()
            if mask.any():
                threshold = B[mask].mean(dim=-1, keepdim=True)
                L[mask] = torch.where(L[mask] < threshold, float('-inf'), L[mask])
        else:
            L = A
            u = None
        
        attention_weights = F.softmax(L, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention_weights, u
        return output, None, u


class SPDGMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        sparsity: float = 0.1,
        pattern: str = 'local',
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        
        self.structural_prior = FrozenStructuralPrior(seq_len, sparsity, pattern)
        self.attention_engine = ClippingAttentionEngine(d_model, n_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        prior_mask = self.structural_prior()
        
        delta_x = None
        if self.training:
            residual = x
            x_norm = self.norm1(x)
            attn_out, attn_weights, u = self.attention_engine(x_norm, prior_mask, delta_x, return_attention)
            x = residual + attn_out
            delta_x = attn_out
        else:
            residual = x
            x_norm = self.norm1(x)
            prev_x = x.clone()
            attn_out, attn_weights, u = self.attention_engine(x_norm, prior_mask, None, return_attention)
            x = residual + attn_out
            delta_x = attn_out
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        if return_attention:
            return x, attn_weights, u
        return x, None, u
