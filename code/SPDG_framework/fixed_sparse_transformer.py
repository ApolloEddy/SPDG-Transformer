from typing import Optional, Tuple

import torch
import torch.nn as nn

from attention_ops import sparse_scaled_dot_product_attention
from spdg_components import FrozenStructuralPrior


class FixedSparseMultiHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        sparsity: float = 0.1,
        pattern: str = 'local',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.seq_len = seq_len
        self.sparsity = sparsity
        self.pattern = pattern

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.structural_prior = FrozenStructuralPrior(seq_len, sparsity, pattern)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        indices, index_mask = self.structural_prior.get_indices(seq_len)
        result = sparse_scaled_dot_product_attention(
            Q,
            K,
            V,
            indices,
            index_mask=index_mask,
            attention_mask=attention_mask,
            dropout_p=self.dropout.p,
            training=self.training,
            return_attention=return_attention,
        )

        output = result.output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        if return_attention:
            return output, result.attention_info
        return output, None


class FixedSparseTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, sparsity: float = 0.1, pattern: str = 'local', dropout: float = 0.1):
        super().__init__()
        self.attention = FixedSparseMultiHead(d_model, n_heads, seq_len, sparsity, pattern, dropout)
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
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        x = self.norm1(x)
        attn_out, attn_weights = self.attention(x, attention_mask, return_attention)
        x = residual + attn_out
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        if return_attention:
            return x, attn_weights
        return x, None


class FixedSparseTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        seq_len: int = 512,
        sparsity: float = 0.1,
        pattern: str = 'local',
        dropout: float = 0.1,
        n_classes: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.sparsity = sparsity
        self.pattern = pattern
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        
        self.layers = nn.ModuleList([
            FixedSparseTransformerLayer(d_model, n_heads, seq_len, sparsity, pattern, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if n_classes is not None:
            self.classifier = nn.Linear(d_model, n_classes)
        else:
            self.classifier = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_u: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        attention_weights_list = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, attention_mask, return_attention)
            if return_attention and attn_weights is not None:
                attention_weights_list.append(attn_weights)
        
        x = self.norm(x)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(x.size()).float()
            x = x * mask
        
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        
        if return_attention:
            return logits, attention_weights_list, None
        else:
            return logits, None, None
    
    def get_sparsity_stats(self) -> dict:
        stats = {
            'model_type': 'Fixed-Sparse-Transformer',
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'seq_len': self.seq_len,
            'sparsity': self.sparsity,
            'pattern': self.pattern
        }
        return stats
