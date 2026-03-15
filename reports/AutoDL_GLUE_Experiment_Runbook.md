# AutoDL GLUE 实验运行手册

## 1. 目标

本手册对应脚本：

- `code/SPDG_framework/bench/run_autodl_glue_suite.py`
- `code/SPDG_framework/bench/run_autodl_glue_suite.sh`

目标是把 SPDG / Full / Fixed 三个主模型，以及 `random_prior` / `uncalibrated` 两个消融版本，在 AutoDL 上跑成一套可直接写论文的实验资产，输出：

- 网络与运行环境记录
- GLUE 数据集下载与 tokenization 速度
- 真实训练/评估耗时
- 消融实验结果
- 长序列 scaling probe
- 图表与 Markdown 报告

## 2. AutoDL 环境建议

推荐：

- Python 3.10+
- CUDA 11.8 或 12.x
- 单卡显存 24GB 起步
- 磁盘至少预留 30GB 用于代码、缓存、checkpoint 和图表输出

推荐目录：

- 项目目录：`/root/autodl-tmp/SPDG-Transformer`
- Hugging Face 缓存：`/root/autodl-tmp/SPDG-Transformer/data/hf_cache`
- 结果输出：`/root/autodl-tmp/SPDG-Transformer/results/autodl_glue_suite`

## 3. 网络环境

本实验脚本默认使用 `hf-mirror.com` 作为 Hugging Face 镜像端点。启动脚本会设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache/datasets
export TRANSFORMERS_CACHE=/path/to/cache/transformers
export TOKENIZERS_PARALLELISM=false
```

如果镜像可用，保持默认即可。

如果你的 AutoDL 能直接访问官方主站，也可以覆盖为：

```bash
export HF_ENDPOINT=https://huggingface.co
```

如果你需要代理，可在运行前自行导出：

```bash
export HTTP_PROXY=http://host:port
export HTTPS_PROXY=http://host:port
```

脚本会把这些环境变量写入最终的 `environment.json`。

在真正下载前，建议先做一次“只探测镜像、不下载任何模型和数据”的检查：

```bash
CHECK_ONLY=1 \
DEVICE=cpu \
TASKS="sst2" \
bash code/SPDG_framework/bench/run_autodl_glue_suite.sh
```

这会在输出目录下生成 `artifacts/connectivity_check.json`。

## 4. 数据集与任务建议

默认任务集：

- `sst2`
- `cola`
- `mrpc`
- `rte`
- `qnli`

理由：

- 都来自 GLUE，论文里容易交代
- 既有单句分类，也有句对任务
- 训练时长比 `mnli` 更可控

如果你显存和时间充足，可以额外加入：

- `mnli`

不建议第一轮就上全部 GLUE 子任务，先跑默认任务集，确认趋势成立，再扩。

## 5. 直接运行

### 5.1 Pilot 预实验

适合先测通链路：

```bash
MODE=pilot \
DEVICE=cuda \
TASKS="sst2 cola mrpc" \
FP16=1 \
bash code/SPDG_framework/bench/run_autodl_glue_suite.sh
```

### 5.2 完整实验

```bash
MODE=full \
DEVICE=cuda \
TASKS="sst2 cola mrpc rte qnli" \
SEEDS="42 43 44" \
FP16=1 \
SAVE_CHECKPOINTS=1 \
bash code/SPDG_framework/bench/run_autodl_glue_suite.sh
```

## 6. 重要参数

主脚本支持的核心参数：

- `--tasks`: 指定 GLUE 子任务
- `--ablation-task`: 消融实验在哪个任务上跑，默认 `sst2`
- `--epochs`: 训练轮数
- `--batch-size`: 训练 batch size
- `--eval-batch-size`: 评估 batch size
- `--gradient-accumulation`: 梯度累积
- `--train-limit`: 只取前 N 条训练样本，`0` 表示全量
- `--eval-limit`: 只取前 N 条验证样本，`0` 表示全量
- `--seeds`: 多随机种子
- `--fp16`: CUDA 半精度
- `--run-ablation`: 跑消融
- `--run-scaling`: 跑长序列 scaling probe

## 7. 输出文件说明

输出根目录：

- `results/autodl_glue_suite/`

关键产物：

- `artifacts/environment.json`
  记录 Python / Torch / CUDA / GPU / cache / proxy / 运行参数

- `artifacts/run_summary.json`
  汇总全局运行结果

- `metrics/tokenization_timing.csv`
  数据集下载、tokenizer 加载、tokenization 速度与 token 数

- `metrics/training_history.csv`
  每个任务、模型、seed、epoch 的训练与评估明细

- `metrics/main_runs.csv`
  每次主实验 run 的最佳结果

- `metrics/main_summary.csv`
  按任务和模型聚合后的平均结果，论文主表优先从这里取数

- `metrics/ablation_runs.csv`
  消融实验结果

- `metrics/scaling_probe.csv`
  序列长度扩展实验，记录延迟、吞吐、峰值显存

- `figures/*.png`
  可直接插入论文或报告的图表

- `reports/autodl_glue_report.md`
  自动生成的文字摘要

## 8. 论文图表对应关系

建议这样取：

- 主结果表：`metrics/main_summary.csv`
- tokenization 速度图：`figures/tokenization_speed.png`
- 数据准备耗时图：`figures/dataset_prepare_time.png`
- 任务准确率/主指标图：`figures/glue_primary_metric.png`
- 推理吞吐图：`figures/glue_eval_throughput.png`
- 精度-延迟权衡图：`figures/accuracy_latency_tradeoff.png`
- 消融图：`figures/ablation_performance.png`
- scaling 延迟图：`figures/scaling_latency.png`
- scaling 吞吐图：`figures/scaling_throughput.png`
- scaling 显存图：`figures/scaling_peak_memory.png`

## 9. 你原先没明确但论文里需要补的点

这些步骤不能省：

- 固定随机种子，至少 3 个 seed
- 统一 tokenizer，不能不同模型各自换 tokenizer
- 统一 `max_length` 与 batch size 策略
- 记录真实 wall-clock time，而不是只记录 step time
- 同时记录吞吐与精度，不能只报一个
- 消融必须和主实验共享训练配置
- 把环境、缓存路径、代理变量保存下来，方便复现实验
- 最终论文结论要基于 `main_summary.csv` 的多 seed 聚合结果，而不是单次跑得最好的一组

## 10. 推荐执行顺序

推荐顺序：

1. 先跑 `pilot`
2. 确认 `tokenization_timing.csv`、`main_summary.csv`、`figures/` 正常生成
3. 再跑 `full`
4. 最后根据 `main_summary.csv` 和 `scaling_probe.csv` 决定论文叙事重心

## 11. 风险提醒

当前这套模型本质上仍是“从头训练的轻量 encoder”，不是预训练语言模型。意味着：

- 在 GLUE 上未必能追上预训练 BERT 的绝对分数
- 更适合强调“相同架构预算下的效率-效果折中”
- 如果你想写成强论文，后续最好再补一轮更长序列或更大规模任务
