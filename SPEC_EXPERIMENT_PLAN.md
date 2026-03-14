# SPDG-Transformer 实验验证规格说明 (Spec)

## 1. 实验目标
验证重构并校准后的 SPDG-Transformer 在以下维度的表现：
- **逻辑有效性**：观察 $\lambda$ 是否随 $u$ (Surprise) 动态切换。
- **计算效率**：对比 Full-Attention、Fixed-Sparse 与 SPDG 的推理延迟与 FLOPs。
- **收敛性能**：在标准序列任务上的准确率与 Loss 下降曲线。

## 2. 实验矩阵 (Experiment Matrix)
所有实验基于：`d_model=256`, `n_heads=8`, `n_layers=4`, `seq_len=512`。

| 实验组 ID | 模型类型 | 门控状态 | 先验稀疏度 (Sparsity) | 核心变量 |
| :--- | :--- | :--- | :--- | :--- |
| **EXP_FULL** | Full-Attention | 强制 $\lambda=0$ | 0.0 | 基准 (Baseline) |
| **EXP_FIXED** | Fixed-Sparse | 强制 $\lambda=10$ | 0.1 | 纯先验对比 |
| **EXP_SPDG** | SPDG (Proposed) | **动态切换** | 0.1 | 本文核心逻辑 |

## 3. 评估指标 (Metrics)
1. **Performance**: Inference latency (ms), Memory Peak (MB).
2. **Quality**: Cross-Entropy Loss, Accuracy (%).
3. **Internal Dynamics**: Gating distribution ($\lambda$ mean/std).

## 4. 自动化流程
1. 调用 `bench/run_all.py` 进行端到端性能测试。
2. 调用 `main.py` 进行 10 Epochs 的训练任务。
3. 收集 `results/` 下的 JSON 数据进行多维分析。
