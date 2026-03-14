import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class AttentionOutput:
    output: torch.Tensor
    attention_info: Optional[Dict[str, Any]] = None

def efficient_spdg_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prior_mask: torch.Tensor,
    lambda_coef: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    penalty_value: float = -10.0,
) -> AttentionOutput:
    """
    Standard-aligned efficient SPDG attention (Vectorized).
    Implements SPDG logic via masking to leverage GPU parallelism.
    
    Args:
        q, k, v: (B, H, L, D) - Head dimension D
        prior_mask: (L, L) - Boolean mask (True where prior exists)
        lambda_coef: (B, 1, 1, 1) - Dynamic gating scale
        attention_mask: (B, L) or (B, 1, 1, L) or (B, 1, L, L)
    """
    B, H, L, D = q.shape
    
    # 1. Scaled Dot-Product: (B, H, L, D) @ (B, H, D, L) -> (B, H, L, L)
    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    
    # 2. Dynamic SPDG Penalty
    inv_prior = (~prior_mask).to(q.dtype).unsqueeze(0).unsqueeze(0) # (1, 1, L, L)
    spdg_penalty = lambda_coef * inv_prior * penalty_value
    logits = logits + spdg_penalty
    
    # 3. Standard Masking (Padding/Causal)
    if attention_mask is not None:
        # Standardize attention_mask to (B, 1, 1, L) or (B, 1, L, L)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1) # (B, 1, L, L)
            
        logits = logits.masked_fill(attention_mask == 0, float("-inf"))
        
    # 4. Softmax & Dropout
    weights = F.softmax(logits, dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=training)
    
    # 5. Weighted Sum
    output = torch.matmul(weights, v)
    
    return AttentionOutput(output=output)
