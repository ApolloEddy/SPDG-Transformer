# SPDG-Transformer: 静态先验与动态门控注意力框架

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

SPDG-Transformer (Static-Prior Dynamic-Gating Transformer) 是一个创新的注意力机制框架，旨在解决全注意力机制（Full Attention）计算效率低和稀疏注意力（Sparse Attention）准确率下降的“两难困境”。

通过结合**冻结结构先验 (Frozen Structural Prior)** 和 **动态门控机制 (Dynamic Gating)**，SPDG 实现了计算资源的自适应分配：对简单的模式化输入“省力”，对复杂的语义输入“发力”。

---

## 🚀 核心特性

- **⚡ 理论效率提升**: 理论复杂度从 $O(N^2)$ 降低至 $O(N)$（在长序列下优势显著）。
- **🧠 智能自适应**: 独创的“残差惊喜 (Residual Surprise)”指标，自动感知输入难度。
- **💪 向量化工程优化 (New)**: 移除了低效的 Python 循环，使用 `torch.index_select` 实现向量化门控，大幅优化推理效率。
- **📊 完整的实验框架**: 内置微基准测试、FLOPs 分析、可视化工具和自动化报告生成。

---

## 🛠️ 环境安装

### 1. 基础环境

请确保安装 Python 3.9+。

### 2. 安装依赖

直接从 `requirements.txt` 安装所有依赖包：

```bash
pip install -r requirements.txt
```

*(注意：如果您的环境没有 CUDA，请自行安装 pytorch-cpu 版本)*

---

## 🏃‍♂️ 快速开始 (Quick Start)

### 1. 快速验证 (Fast Verify)

运行此指令以确认环境配置正确且代码无 Bug。它会使用极少量数据运行 1 个 Epoch。

```bash
python code/SPDG_framework/main.py --fast --device cuda
```

### 2. 生成学术证据 (Academic Validation)

这是本项目最核心的价值验证，运行以下脚本可复现论文中的关键图表。

**证据 A：理论计算量碾压 (FLOPs Scaling)**
生成 Full vs SPDG 的 FLOPs 对比图，证明 $O(N)$ 优势。

```bash
python code/SPDG_framework/bench/run_flops_analysis.py
```

*产出*: `results/SPDG_experiment/figures/flops_scaling_log.png`

**证据 B：智能门控可视化 (Gating Intelligence)**
生成“难度 vs 惊喜”的 S 曲线，证明模型确实“懂”输入。
*(需先运行过 --fast 获取 checkpoint)*

```bash
python code/SPDG_framework/bench/visualize_gating.py
```

*产出*: `results/SPDG_experiment/figures/gating_intelligence.png`

### 3. 运行完整实验 (Full Experiment)

执行完整的训练、评估流程。包含 Synthetic 和 GLUE (SST-2) 数据集。
*(注意：GLUE 数据集初次运行需要联网下载)*

```bash
python code/SPDG_framework/main.py --device cuda
```

---

## 📂 项目结构

```
SPDG-Transformer/
├── code/SPDG_framework/
│   ├── spdg_components.py      # [核心] SPDG 算法实现 (含向量化优化)
│   ├── spdg_transformer.py     # 模型组装
│   ├── train.py                # 训练器
│   ├── main.py                 # 统一入口
│   └── bench/                  # 验证脚本 (FLOPs, Memory Check等)
├── results/                    # 实验结果输出目录
├── requirements.txt            # 项目依赖
└── README.md                   # 说明文档
```

## ⚠️ 硬件说明

- **显存要求**: 完整训练建议 8GB+ VRAM。
- **序列长度**:
  - 本地单卡训练建议 `seq_len <= 2048`。
  - 在 8GB 显存下，`seq_len=4096` 可能导致 OOM（Full Attention 机制本身的限制）。

---

## 引用

如果您使用了本项目的代码或思路，请引用：

```bibtex
@article{spdg2026,
  title={SPDG-Transformer: Static-Prior Dynamic-Gating Attention},
  author={SPDG Research Team},
  year={2026}
}
```
