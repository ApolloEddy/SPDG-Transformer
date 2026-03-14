# SPDG-Transformer 高性能重构规格说明 (Spec)

## 1. 核心目标
将原有的基于 Python 逻辑拆分（Dynamic Branching）的 SPDG 实现，重构为基于 **向量化掩码（Vectorized Masking）** 的标准 PyTorch 实现。
- **性能目标**：推理速度提升 5x - 10x（针对长序列和大 Batch）。
- **兼容性**：完全兼容 `torch.compile`，支持 FP16/BF16。
- **一致性**：架构上对齐 `nn.TransformerEncoderLayer`。

## 2. 架构设计

### 2.1 核心算子：`efficient_spdg_attention`
- **逻辑变化**：
  - 弃用 `k[:, :, indices]` 这种非连续内存切片。
  - 采用 `Logits = (Q @ K^T) / sqrt(d) + Lambda * (1 - Prior_Mask) * Penalty`。
  - 通过 `Penalty` (如 -10.0) 在 Softmax 前压制非先验区域的权重，实现“动态稀疏”效果。
- **数据流**：保持 (B, H, L, D) 的连续张量流，消除 CPU 同步。

### 2.2 动态门控：`DynamicGating` & `ResidualSurprise`
- **向量化**：`Surprise` 的计算必须是全向量化的（使用 `torch.norm`）。
- **广播机制**：`lambda_coef` 需能自动广播至 (B, H, L, L) 以配合注意力计算。

### 2.3 模块封装：`SPDGTransformerLayer`
- **Pre-Norm 结构**：提升训练稳定性。
- **状态流转**：`u_prev` (Surprise) 需在层间或步骤间显式传递，形成闭环。

## 3. 实现计划 (Steps)

1. **算子层 (Ops)**: 验证 `efficient_spdg_attention` 的等价性。
2. **组件层 (Components)**: 重构 `FrozenStructuralPrior`，优化先验掩码生成速度（使用 `torch.arange` 向量化）。
3. **架构层 (Transformer)**: 整合 `SPDGTransformer`，确保接口标准。
4. **验证层 (Verification)**: 
   - 逻辑验证：运行 `test_attention.py`。
   - 性能验证：对比重构前后的 `run_flops_analysis.py` 结果。

## 4. 关键验证点 (Definition of Done)
- [ ] 所有测试用例通过 (Passed)。
- [ ] 移除所有 `if use_sparse:` 类型的 Python 动态分支。
- [ ] 显存占用保持平稳，无内存泄漏（由于频繁切片导致的内存空洞）。
