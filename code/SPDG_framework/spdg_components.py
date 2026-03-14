import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_ops import (
    AttentionOutput,
    efficient_spdg_attention,
)

class FrozenStructuralPrior(nn.Module):
    """
    Optimized Prior Mask generator supporting random patterns for ablation study.
    """
    def __init__(self, seq_len: int, sparsity: float = 0.1, pattern: str = 'local'):
        super().__init__()
        self.seq_len = seq_len
        self.sparsity = sparsity
        self.pattern = pattern
        prior_mask = self._generate_prior_mask()
        self.register_buffer('prior_mask', prior_mask)

    def _generate_prior_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)
        if self.pattern == 'local':
            window_size = max(1, int(self.seq_len * self.sparsity))
            rows = torch.arange(self.seq_len).unsqueeze(1)
            cols = torch.arange(self.seq_len).unsqueeze(0)
            mask = torch.abs(rows - cols) <= (window_size // 2)
        elif self.pattern == 'random':
            # Ablation Case: Random Sparsity
            mask = torch.rand(self.seq_len, self.seq_len) < self.sparsity
            # Keep diagonal to avoid total disconnection
            mask.fill_diagonal_(True)
        elif self.pattern == 'block':
            block_size = max(1, int(self.seq_len * (1 - self.sparsity) ** 0.5))
            num_blocks = (self.seq_len + block_size - 1) // block_size
            for bi in range(num_blocks):
                for bj in range(num_blocks):
                    if abs(bi - bj) <= 1:
                        mask[bi*block_size:(bi+1)*block_size, bj*block_size:(bj+1)*block_size] = True
        return mask[:self.seq_len, :self.seq_len]
    
    def forward(self) -> torch.Tensor:
        return self.prior_mask


class ResidualSurprise(nn.Module):
    """
    Normalized surprise calculation supporting uncalibrated ablation.
    """
    def __init__(self, eps: float = 1e-8, calibrate: bool = True):
        super().__init__()
        self.eps = eps
        self.calibrate = calibrate
        # Learnable scaling
        self.u_scale = nn.Parameter(torch.ones(1) * 0.1) if calibrate else None
    
    def forward(self, x: torch.Tensor, delta_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_delta_x = torch.norm(delta_x, p=2, dim=-1, keepdim=True)
        u_raw = norm_delta_x / (norm_x + self.eps)
        
        # Ablation Case: Calibration vs Raw
        u_scaled = u_raw * self.u_scale if self.calibrate else u_raw
        aggregated_u = u_scaled.mean(dim=1)
        return u_scaled, aggregated_u


class DynamicGating(nn.Module):
    """
    Gating module supporting fixed lambda ablation.
    """
    def __init__(self, lambda_max: float = 10.0, alpha: float = 5.0, mode: str = 'dynamic'):
        super().__init__()
        self.lambda_max = lambda_max
        self.alpha = alpha
        self.mode = mode # 'dynamic', 'fixed_sparse', 'fixed_full'
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self.mode == 'fixed_full':
            return torch.zeros_like(u)
        elif self.mode == 'fixed_sparse':
            return torch.full_like(u, self.lambda_max)
        # Default: lambda = lambda_max * exp(-alpha * u)
        return self.lambda_max * torch.exp(-self.alpha * u)


class SPDGMultiHeadAttention(nn.Module):
    """
    Standard-aligned Multi-Head Attention parameterized for ablation studies.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        sparsity: float = 0.1,
        pattern: str = 'local',
        dropout: float = 0.1,
        lambda_max: float = 10.0,
        alpha: float = 5.0,
        gating_mode: str = 'dynamic'
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.seq_len = seq_len
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.structural_prior = FrozenStructuralPrior(seq_len, sparsity, pattern)
        self.dynamic_gating = DynamicGating(lambda_max, alpha, mode=gating_mode)
    
    def forward(
        self,
        x: torch.Tensor,
        u_prev: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if u_prev is None:
            u_prev = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        
        lambda_coef = self.dynamic_gating(u_prev).view(batch_size, 1, 1, 1)
        prior_mask = self.structural_prior()[:seq_len, :seq_len]
        
        attn_out = efficient_spdg_attention(
            q, k, v,
            prior_mask=prior_mask,
            lambda_coef=lambda_coef,
            attention_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            training=self.training
        )
        out = attn_out.output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        return out, lambda_coef.view(batch_size, 1)


class SPDGTransformerLayer(nn.Module):
    """
    Transformer Layer with ablation configuration.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        sparsity: float = 0.1,
        pattern: str = 'local',
        gating_mode: str = 'dynamic',
        calibrate: bool = True
    ):
        super().__init__()
        self.self_attn = SPDGMultiHeadAttention(
            d_model, n_heads, seq_len, sparsity, pattern, dropout, gating_mode=gating_mode
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_surprise = ResidualSurprise(calibrate=calibrate)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        u_prev: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = src
        nx = self.norm1(x)
        attn_out, _ = self.self_attn(nx, u_prev=u_prev, attention_mask=src_mask)
        x = x + self.dropout1(attn_out)
        nx = self.norm2(x)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(nx))))
        x = x + self.dropout2(ffn_out)
        _, aggregated_u = self.residual_surprise(src, x - src)
        return x, aggregated_u
