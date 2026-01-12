import unittest

import torch

from attention_ops import sparse_scaled_dot_product_attention
from spdg_components import FrozenStructuralPrior, SPDGMultiHeadAttention


class SparseAttentionTests(unittest.TestCase):
    def test_sparse_attention_shapes(self) -> None:
        batch_size = 2
        seq_len = 8
        n_heads = 2
        head_dim = 4

        q = torch.randn(batch_size, n_heads, seq_len, head_dim)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim)

        prior = FrozenStructuralPrior(seq_len=seq_len, sparsity=0.5, pattern="local")
        indices, index_mask = prior.get_indices(seq_len)

        result = sparse_scaled_dot_product_attention(
            q, k, v, indices, index_mask=index_mask, return_attention=True
        )
        self.assertEqual(result.output.shape, (batch_size, n_heads, seq_len, head_dim))
        self.assertEqual(result.attention_info["mode"], "sparse")

    def test_spdg_gating_prefers_sparse_on_low_surprise(self) -> None:
        d_model = 16
        seq_len = 8
        n_heads = 2

        attention = SPDGMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            seq_len=seq_len,
            sparsity=0.5,
            pattern="local",
            dropout=0.0,
        )

        x = torch.randn(1, seq_len, d_model)
        u_prev = torch.zeros(1, 1)

        _, attention_info, _ = attention(x, attention_mask=None, u_prev=u_prev, return_attention=True)
        first = attention_info[0]
        self.assertEqual(first["mode"], "sparse")


if __name__ == "__main__":
    unittest.main()
