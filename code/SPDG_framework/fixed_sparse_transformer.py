import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FixedSparseMultiHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, sparsity: float = 0.1, pattern: str = 'local', dropout: float = 0.1):
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
        
        self.register_buffer('sparse_mask', self._generate_sparse_mask())
    
    def _generate_sparse_mask(self) -> torch.Tensor:
        mask = torch.ones(self.seq_len, self.seq_len)
        
        if self.pattern == 'local':
            window_size = int(self.seq_len * self.sparsity)
            for i in range(self.seq_len):
                mask[i, :max(0, i - window_size // 2)] = 0.0
                mask[i, min(self.seq_len, i + window_size // 2 + 1):] = 0.0
        
        elif self.pattern == 'global':
            num_global = int(self.seq_len * self.sparsity)
            for i in range(self.seq_len):
                indices = torch.randperm(self.seq_len)[:num_global]
                sparse_mask = torch.ones(self.seq_len)
                sparse_mask[indices] = 0.0
                mask[i] = sparse_mask
        
        elif self.pattern == 'block':
            block_size = int(self.seq_len * (1 - self.sparsity) ** 0.5)
            num_blocks = self.seq_len // block_size
            for bi in range(num_blocks):
                for bj in range(num_blocks):
                    if abs(bi - bj) > 1:
                        start_i, end_i = bi * block_size, min((bi + 1) * block_size, self.seq_len)
                        start_j, end_j = bj * block_size, min((bj + 1) * block_size, self.seq_len)
                        mask[start_i:end_i, start_j:end_j] = 0.0
        
        return mask
    
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
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if seq_len <= self.seq_len:
            sparse_mask = self.sparse_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores.masked_fill(sparse_mask == 0.0, float('-inf'))
        else:
            attention_scores = attention_scores.masked_fill(torch.ones_like(attention_scores, dtype=torch.bool) == False, float('-inf'))
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention_weights
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
