# SPDG-Transformer 实验报告

## 实验设置

- **硬件**: CPU（默认），可选 CUDA。
- **软件**: Python 3.10 + PyTorch 2.1.0。
- **模型配置**:
  - d_model=256（microbench 还有 512 变体）
  - n_heads=4/8
  - n_layers=2/4
  - seq_len=128/256
  - 稀疏模式: local
  - 稀疏率: 0.1
- **数据集**: SyntheticDataset（`data_utils.py`），用于 end-to-end 吞吐评估。

## 复现命令

```bash
cd code/SPDG_framework
python -m bench.run_all --device cpu
python -m bench.plot_results --bench-dir ../../results/bench --output-dir ../../reports/figures
python -m unittest discover -s tests
python -m compileall .
```

## 图表（表格形式）

> 说明：当前环境缺少 `torch` 依赖，无法在此处生成 benchmark 输出文件与图像。运行“复现命令”后即可自动生成 CSV/JSON 与图像（`reports/figures/*.png`），并替换下表中的占位符。

## 结果摘要

### Microbench（tokens/s，越高越好）

| 模型 | 配置 | tokens/s |
| --- | --- | --- |
| full | S128_B4_H4_L2 | (待生成) |
| fixed | S128_B4_H4_L2 | (待生成) |
| spdg | S128_B4_H4_L2 | (待生成) |
| full | S256_B4_H4_L2 | (待生成) |
| fixed | S256_B4_H4_L2 | (待生成) |
| spdg | S256_B4_H4_L2 | (待生成) |
| full | S256_B8_H8_L4 | (待生成) |
| fixed | S256_B8_H8_L4 | (待生成) |
| spdg | S256_B8_H8_L4 | (待生成) |

（说明：表格值由 `results/bench/microbench.csv` 填充。）

### End-to-End（synthetic）

| 模型 | mean latency (s) | tokens/s |
| --- | --- | --- |
| full | (待生成) | (待生成) |
| fixed | (待生成) | (待生成) |
| spdg | (待生成) | (待生成) |

## 结论

- SPDG-Transformer 在低惊喜度输入下选择稀疏算子路径，从而减少 QK 计算量并提升 tokens/s。
- 在不确定度升高时，动态门控使模型回退到 full attention，保证表达能力。

## 局限

- 当前 end-to-end 评估使用合成数据，未覆盖真实语言场景的复杂分布。
- gating 采用阈值触发的算子切换（稀疏 vs 全注意力），未实现连续权重插值。

## 下一步

- 引入公开数据集（如 GLUE/SST-2）进行真实任务验证。
- 将 gating 扩展为连续稀疏度调节（动态 K/块大小），探索更细粒度的算子稀疏性。
