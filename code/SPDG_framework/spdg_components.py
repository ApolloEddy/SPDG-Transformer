import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from attention_ops import (
    AttentionOutput,
    scaled_dot_product_attention,
    sparse_scaled_dot_product_attention,
)


class FrozenStructuralPrior(nn.Module):
    def __init__(self, seq_len: int, sparsity: float = 0.1, pattern: str = 'local'):
        super().__init__()
        self.seq_len = seq_len
        self.sparsity = sparsity
        self.pattern = pattern

        prior_mask, indices = self._generate_prior_mask()
        self.register_buffer('prior_mask', prior_mask)
        self.register_buffer('prior_indices', indices)
        self.register_buffer('prior_index_mask', self._generate_index_mask())

    def _generate_prior_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)

        if self.pattern == 'local':
            window_size = max(1, int(self.seq_len * self.sparsity))
            for i in range(self.seq_len):
                start = max(0, i - window_size // 2)
                end = min(self.seq_len, i + window_size // 2 + 1)
                mask[i, start:end] = True
        
        elif self.pattern == 'global':
            num_global = max(1, int(self.seq_len * self.sparsity))
            for i in range(self.seq_len):
                indices = torch.randperm(self.seq_len)[:num_global]
                mask[i, indices] = True
        
        elif self.pattern == 'block':
            block_size = int(self.seq_len * (1 - self.sparsity) ** 0.5)
            block_size = max(1, block_size)
            num_blocks = math.ceil(self.seq_len / block_size)
            for bi in range(num_blocks):
                for bj in range(num_blocks):
                    if abs(bi - bj) <= 1:
                        start_i, end_i = bi * block_size, min((bi + 1) * block_size, self.seq_len)
                        start_j, end_j = bj * block_size, min((bj + 1) * block_size, self.seq_len)
                        mask[start_i:end_i, start_j:end_j] = True

        neighbor_indices, _ = self._indices_from_mask(mask)

        return mask, neighbor_indices

    def _indices_from_mask(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = mask.nonzero(as_tuple=False)
        grouped = [[] for _ in range(mask.size(0))]
        for row, col in indices.tolist():
            grouped[row].append(col)
        max_neighbors = max(len(cols) for cols in grouped)
        neighbor_indices = torch.full((mask.size(0), max_neighbors), 0, dtype=torch.long)
        neighbor_mask = torch.zeros((mask.size(0), max_neighbors), dtype=torch.bool)
        for row, cols in enumerate(grouped):
            if cols:
                neighbor_indices[row, : len(cols)] = torch.tensor(cols, dtype=torch.long)
                neighbor_mask[row, : len(cols)] = True
            else:
                neighbor_indices[row, 0] = row
                neighbor_mask[row, 0] = True

        return neighbor_indices, neighbor_mask

    def _generate_index_mask(self) -> torch.Tensor:
        _, neighbor_mask = self._indices_from_mask(self.prior_mask)
        return neighbor_mask
    
    def forward(self) -> torch.Tensor:
        return self.prior_mask

    def get_indices(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.prior_indices[:seq_len].clone()
        mask = self.prior_index_mask[:seq_len].clone()

        if seq_len < self.seq_len:
            valid = indices < seq_len
            mask = mask & valid
            indices = indices.clamp(max=seq_len - 1)

        return indices, mask


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

        self.dynamic_gating = DynamicGating()
    
    def forward(
        self,
        x: torch.Tensor,
        prior_mask: torch.Tensor,
        prior_indices: torch.Tensor,
        prior_index_mask: torch.Tensor,
        u_prev: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if u_prev is None:
            u_prev = torch.zeros(batch_size, 1, device=x.device)

        lambda_coef = self.dynamic_gating(u_prev).view(batch_size, 1, 1, 1)

        sparse_threshold = 1.0
        use_sparse = (lambda_coef.view(batch_size) >= sparse_threshold)

        outputs = []
        attention_infos = [] if return_attention else None

        for idx in range(batch_size):
            if use_sparse[idx]:
                result: AttentionOutput = sparse_scaled_dot_product_attention(
                    Q[idx : idx + 1],
                    K[idx : idx + 1],
                    V[idx : idx + 1],
                    prior_indices[:seq_len],
                    index_mask=prior_index_mask[:seq_len],
                    attention_mask=attention_mask[idx : idx + 1] if attention_mask is not None else None,
                    dropout_p=self.dropout.p,
                    training=self.training,
                    return_attention=return_attention,
                )
            else:
                prior_bias = torch.where(
                    prior_mask[:seq_len, :seq_len],
                    torch.tensor(0.0, device=x.device, dtype=x.dtype),
                    torch.tensor(-1.0, device=x.device, dtype=x.dtype),
                )
                logit_bias = lambda_coef[idx] * prior_bias
                logit_bias = logit_bias.unsqueeze(0).unsqueeze(0)
                result = scaled_dot_product_attention(
                    Q[idx : idx + 1],
                    K[idx : idx + 1],
                    V[idx : idx + 1],
                    attention_mask=attention_mask[idx : idx + 1] if attention_mask is not None else None,
                    logit_bias=logit_bias,
                    dropout_p=self.dropout.p,
                    training=self.training,
                    return_attention=return_attention,
                )
                if return_attention:
                    result.attention_info["prior_bias"] = lambda_coef[idx].item()

            outputs.append(result.output)
            if return_attention:
                attention_infos.append(result.attention_info)

        attention_output = torch.cat(outputs, dim=0)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attention_output = self.out_proj(attention_output)

        if return_attention:
            return attention_output, attention_infos
        return attention_output, None


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
        self.residual_surprise = ResidualSurprise()
        
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
        attention_mask: Optional[torch.Tensor] = None,
        u_prev: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list], Optional[torch.Tensor]]:
        prior_mask = self.structural_prior()

        residual_input = x
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attention_engine(
            x_norm,
            prior_mask,
            *self.structural_prior.get_indices(x.shape[1]),
            u_prev,
            attention_mask,
            return_attention,
        )
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        delta_x = x - residual_input
        u, aggregated_u = self.residual_surprise(residual_input, delta_x)

        if return_attention:
            return x, attn_weights, aggregated_u
        return x, None, aggregated_u
