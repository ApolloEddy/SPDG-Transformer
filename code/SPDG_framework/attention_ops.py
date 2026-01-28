import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AttentionOutput:
    output: torch.Tensor
    attention_info: Optional[dict]


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    logit_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    return_attention: bool = False,
) -> AttentionOutput:
    head_dim = q.size(-1)
    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    if logit_bias is not None:
        logits = logits + logit_bias

    if attention_mask is not None:
        key_mask = attention_mask[:, None, None, :]
        logits = logits.masked_fill(key_mask == 0, float("-inf"))

    weights = F.softmax(logits, dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=training)
    output = torch.matmul(weights, v)

    if return_attention:
        return AttentionOutput(
            output=output,
            attention_info={"mode": "full", "weights": weights},
        )
    return AttentionOutput(output=output, attention_info=None)


def sparse_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    index_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    logit_bias: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    return_attention: bool = False,
) -> AttentionOutput:
    head_dim = q.size(-1)
    k_selected = k[:, :, indices]
    v_selected = v[:, :, indices]

    # Memory efficient implementation using einsum
    # q: (B, H, L, D)
    # k_selected: (B, H, L, W, D)
    # logits: (B, H, L, W)
    logits = torch.einsum('bhld,bhlwd->bhlw', q, k_selected) / math.sqrt(head_dim)

    if logit_bias is not None:
        logit_bias_selected = logit_bias[:, :, indices]
        logits = logits + logit_bias_selected

    if index_mask is not None:
        mask = index_mask.unsqueeze(0).unsqueeze(0)
        logits = logits.masked_fill(mask == 0, float("-inf"))

    if attention_mask is not None:
        key_mask = attention_mask[:, None, :]
        key_mask_selected = key_mask[:, :, indices]
        logits = logits.masked_fill(key_mask_selected == 0, float("-inf"))

    weights = F.softmax(logits, dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=training)
    output = (weights.unsqueeze(-1) * v_selected).sum(dim=3)

    if return_attention:
        return AttentionOutput(
            output=output,
            attention_info={
                "mode": "sparse",
                "weights": weights,
                "indices": indices,
                "index_mask": index_mask,
                "seq_len": q.size(-2),
            },
        )
    return AttentionOutput(output=output, attention_info=None)
