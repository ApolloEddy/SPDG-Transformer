import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Any
from spdg_components import SPDGTransformerLayer

class FixedSparseTransformer(nn.Module):
    """
    Fixed Sparse Transformer aligned with SPDG implementation.
    The gating coefficient is fixed to the maximum (lambda_max) to enforce the prior mask.
    """
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
        n_classes: Optional[int] = None,
        dim_feedforward: int = 2048,
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
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            SPDGTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                seq_len=seq_len,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                sparsity=sparsity,
                pattern=pattern
            ) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        if n_classes is not None:
            self.classifier = nn.Linear(d_model, n_classes)
        else:
            self.classifier = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_embedding, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_u: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Any]], Optional[List[torch.Tensor]]]:
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        for layer in self.layers:
            # Force surprise to be 0 so lambda becomes lambda_max (Fixed Sparse)
            # lambda = lambda_max * exp(-alpha * 0) = lambda_max
            u_force_sparse = torch.zeros((batch_size, 1), device=x.device, dtype=x.dtype)
            x, _ = layer(x, u_prev=u_force_sparse, src_mask=attention_mask)
        
        x = self.norm(x)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        else:
            pooled = x.mean(dim=1)
            
        logits = self.classifier(pooled)
        return logits, None, None

    def get_sparsity_stats(self) -> dict:
        return {
            'model_type': 'Fixed-Sparse-Control',
            'd_model': self.d_model,
            'sparsity': self.sparsity,
            'pattern': self.pattern
        }
