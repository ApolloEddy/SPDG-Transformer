# SPDG-Transformer 实验框架

## 项目概述

SPDG-Transformer（Static-Prior Dynamic-Gating Transformer）是一个创新的注意力机制框架，通过冻结结构先验和动态门控机制实现计算资源的自适应分配。本项目提供了完整的实验框架，包括模型实现、训练脚本、指标计算、可视化和报告生成。

## 核心特性

### 1. SPDG-Transformer
- **冻结结构先验 (Frozen Structural Prior B)**: 基于全局统计的稀疏注意力模式
- **残差惊喜 (Residual Surprise u)**: 量化模型不确定性的轻量级指标
- **动态门控 (Dynamic Gating λ)**: 根据不确定性自适应调整计算量
- **裁剪注意力引擎 (Clipping Attention Engine)**: 基于门控系数的注意力裁剪

### 2. 基线模型
- **Full-Attention Transformer**: 标准全注意力模型（性能上限）
- **Fixed-Sparse Transformer**: 固定稀疏注意力模型（无刹车基线）

## 项目结构

```
SPDG-Transformer/
├── code/
│   └── SPDG_framework/
│       ├── spdg_components.py          # SPDG核心组件
│       ├── spdg_transformer.py         # SPDG模型
│       ├── full_attention_transformer.py  # 全注意力模型
│       ├── fixed_sparse_transformer.py   # 固定稀疏模型
│       ├── datasets.py                 # 数据集加载
│       ├── train.py                    # 训练脚本
│       ├── metrics.py                  # 指标计算
│       ├── visualization.py            # 可视化
│       ├── report_generator.py         # 报告生成
│       ├── validate_models.py          # 模型验证
│       ├── main.py                    # 主流程
│       └── __init__.py
├── data/
│   └── SPDG_experiment/
│       ├── difficulty_mapping.csv      # 难度映射表
│       └── [数据集文件]
├── results/
│   └── SPDG_experiment/
│       ├── metrics/
│       │   ├── training_metrics.csv
│       │   ├── inference_metrics.csv
│       │   ├── difficulty_experiment_metrics.csv
│       │   └── summary.json
│       ├── figures/
│       │   ├── difficulty_vs_computation.png
│       │   ├── accuracy_comparison.png
│       │   ├── inference_time_comparison.png
│       │   ├── training_curves.png
│       │   ├── flops_comparison.png
│       │   ├── efficiency_accuracy_tradeoff.png
│       │   └── summary_figure.png
│       ├── checkpoints/
│       └── SPDG_experiment_report.md
├── env/
│   └── spdg_env.yml
├── article.txt
└── README.md
```

## 环境配置

### 1. 创建Conda环境

```bash
conda env create -f env/spdg_env.yml
conda activate spdg_env
```

### 2. 依赖库

- Python 3.10
- PyTorch 2.1.0
- xFormers 0.0.22.post7
- transformers 4.35.2
- datasets 2.15.0
- matplotlib 3.8.2
- seaborn 0.13.0
- pandas 2.1.3
- numpy 1.24.3

## 使用方法

### 1. 模型验证

验证所有模型是否正常工作：

```bash
cd code/SPDG_framework
python validate_models.py
```

### 2. 运行完整实验流程

运行所有实验（训练、评估、可视化、报告生成）：

```bash
cd code/SPDG_framework
python main.py
```

### 3. 分步执行

只运行特定步骤：

```bash
# 只运行模型验证
python main.py --validate-only

# 跳过训练
python main.py --skip-training

# 跳过评估
python main.py --skip-evaluation

# 跳过可视化
python main.py --skip-visualization

# 跳过报告生成
python main.py --skip-report

# 指定设备
python main.py --device cuda
```

### 4. 单独运行训练

```bash
python train.py
```

### 5. 生成报告

```bash
python report_generator.py
```

## 实验设计

### 实验1: 动态效率

**目标**: 验证计算量随输入难度自适应调整

**指标**:
- X轴: 样本难度（聚合u值）
- Y轴: QK对计算比例（归一化FLOPs）

**预期结果**: S形曲线，低难度样本使用稀疏注意力，高难度样本使用全注意力

### 实验2: 鲁棒性测试

**目标**: 在反统计输入上保持准确率

**数据集**: 逻辑反转数据集、花园路径句子

**对比模型**:
1. SPDG-Transformer
2. Full-Attention Transformer（性能上限）
3. Fixed-Sparse Transformer（无刹车基线）

**预期结果**: SPDG匹配全注意力准确率，显著优于固定稀疏模型

## 模型配置

### SPDG-Transformer

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 512 | 模型维度 |
| n_heads | 8 | 注意力头数 |
| n_layers | 6 | 层数 |
| seq_len | 512 | 序列长度 |
| sparsity | 0.1 | 稀疏度 |
| pattern | 'local' | 稀疏模式 |
| lambda_max | 10.0 | 最大门控系数 |
| alpha | 5.0 | 门控衰减系数 |

### 基线模型

使用相同的超参数配置，确保公平对比。

## 输出说明

### 1. 训练指标 (training_metrics.csv)

- epoch: 训练轮数
- train_loss: 训练损失
- train_acc: 训练准确率
- val_loss: 验证损失
- val_acc: 验证准确率
- learning_rate: 学习率
- epoch_time: 每轮训练时间

### 2. 推理指标 (inference_metrics.csv)

- model_name: 模型名称
- dataset_name: 数据集名称
- accuracy: 准确率
- mean_inference_time: 平均推理时间
- throughput: 吞吐量（样本/秒）
- total_flops: 总FLOPs
- flops_per_token: 每个token的FLOPs
- mean_sparsity: 平均稀疏度
- mean_computation_ratio: 平均计算比例

### 3. 难度实验指标 (difficulty_experiment_metrics.csv)

- model_name: 模型名称
- difficulty_level: 难度等级
- aggregated_u: 聚合u值
- computation_ratio: 计算比例
- accuracy: 准确率
- inference_time: 推理时间

### 4. 可视化图表

所有图表保存为PNG格式，分辨率300dpi：
- difficulty_vs_computation.png: 难度-计算比例曲线
- accuracy_comparison.png: 准确率对比
- inference_time_comparison.png: 推理时间对比
- training_curves.png: 训练曲线
- flops_comparison.png: FLOPs对比
- efficiency_accuracy_tradeoff.png: 效率-准确率权衡
- summary_figure.png: 实验总结图

### 5. 实验报告

Markdown格式的完整实验报告，包含：
- 实验概述
- 模型配置详情
- 数据集配置
- 训练配置
- 实验步骤回顾
- 结果总结
- 详细指标对比
- 结果分析
- 结论与建议
- 附录

## 核心算法

### 残差惊喜计算

```
u = ||Δx|| / (||x|| + ε)
```

其中：
- x: 当前层输入
- Δx: 残差更新
- ε: 数值稳定性常数

### 动态门控系数

```
λ(u) = λ_max · exp(-αu)
```

其中：
- λ_max: 最大门控系数（默认10.0）
- α: 衰减系数（默认5.0）
- u: 残差惊喜

### 注意力调制

```
L = A + λ(u) · B
```

其中：
- A: 标准注意力logits
- B: 结构先验
- λ(u): 动态门控系数

## 性能优化

1. **计算开销**: 计算u和λ的复杂度为O(Nd)，相对于O(N²)的注意力可忽略
2. **内存优化**: 使用梯度检查点和混合精度训练
3. **推理加速**: 支持xFormers优化的注意力计算

## 扩展应用

SPDG框架可扩展到其他模态：

- **视频**: 时间连续性 vs 场景切换
- **生物学**: 蛋白质/DNA序列的保守基序 vs 罕见突变
- **时间序列**: 稳定趋势 vs 异常检测
- **音频**: 静音/平稳语音 vs 情感爆发

## 引用

如果使用本代码，请引用：

```bibtex
@article{spdg2024,
  title={SPDG-Transformer: Static-Prior Dynamic-Gating Attention},
  author={SPDG Research Team},
  year={2024}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请联系项目维护者。

## 更新日志

### v1.0.0 (2024-01-05)
- 初始版本发布
- 实现SPDG-Transformer核心组件
- 实现基线模型
- 完整的实验框架
- 模型验证通过
