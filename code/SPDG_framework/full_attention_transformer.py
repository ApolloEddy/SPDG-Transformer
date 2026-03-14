import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Any
from spdg_components import SPDGTransformerLayer

class FullAttentionTransformer(nn.Module):
    """
    Standard Transformer aligned with SPDG implementation to ensure fair experimental comparison.
    The only difference is that the gating coefficient is fixed to 0.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        seq_len: int = 512,
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
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # Use the SAME layer implementation as SPDG, but with sparsity logic disabled in forward
        self.layers = nn.ModuleList([
            SPDGTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                seq_len=seq_len,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                sparsity=0.0, # NO SPARSITY pattern
                pattern='local'
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
        
        # In Full Attention, we pass u_prev as a huge negative number or 
        # modify the layer to ignore gating. 
        # A cleaner way: pass a lambda_override of 0.
        
        for layer in self.layers:
            # Force surprise to be very high so lambda becomes 0 (Full Attention)
            # lambda = lambda_max * exp(-alpha * u)
            # If u is very large (e.g. 100), lambda -> 0
            u_force_full = torch.full((batch_size, 1), 100.0, device=x.device, dtype=x.dtype)
            x, _ = layer(x, u_prev=u_force_full, src_mask=attention_mask)
        
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
            'model_type': 'Full-Attention-Control',
            'd_model': self.d_model,
            'status': 'standard-aligned-shared-ops'
        }
