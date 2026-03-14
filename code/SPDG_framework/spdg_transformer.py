import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Any
from spdg_components import SPDGTransformerLayer

class SPDGTransformer(nn.Module):
    """
    Performance-optimized SPDG Transformer.
    Re-engineered for standard PyTorch execution patterns to ensure maximum speed.
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
        
        # Standard Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Fixed: Ensure pos_embedding size matches the intended seq_len
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # High-performance SPDG Layers
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
        # Xavier initialization for standard alignment
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Special handling for pos_embedding
        nn.init.normal_(self.pos_embedding, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_u: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Any]], Optional[List[torch.Tensor]]]:
        batch_size, seq_len = input_ids.shape
        
        # 1. Embedding + Positional
        # Correctly slicing pos_embedding to match input seq_len
        x = self.token_embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        u_list = []
        u_prev = None # Initial surprise is None (zeros in gating)
        
        # 2. Sequential Layer Processing
        for layer in self.layers:
            # Standard sequential flow: Surprise from layer N feeds into layer N+1
            x, u = layer(x, u_prev=u_prev, src_mask=attention_mask)
            if return_u:
                u_list.append(u)
            u_prev = u # Feedback loop
        
        # 3. Final Norm and Pooling
        x = self.norm(x)
        
        # Use pooling for classification (Standard BERT-style)
        if attention_mask is not None:
            # Masked average pooling
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        else:
            pooled = x.mean(dim=1)
            
        logits = self.classifier(pooled)
        
        # Standard return format
        return logits, None, (u_list if return_u else None)
    
    def get_sparsity_stats(self) -> dict:
        return {
            'model_type': 'SPDG-Transformer-Optimized',
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'seq_len': self.seq_len,
            'sparsity': self.sparsity,
            'pattern': self.pattern,
            'status': 'standard-aligned'
        }
