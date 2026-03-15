# 变更日志

## 2026-03-15

### Fixed
- 修复 `C:\Users\Eddy\.codex\skills\spec-coding\SKILL.md` 因 UTF-8 BOM 导致的技能导入失败问题；文件现已改为无 BOM 的 UTF-8 编码，frontmatter 可被正常识别。

### Notes
- 本次仓库内仅新增该日志文件，用于记录对外部技能文件的修复动作。

### Changed
- 将用户级 `pip` 配置 `C:\Users\Eddy\pip\pip.ini` 切换为官方源 `https://pypi.org/simple`，移除原清华镜像配置，便于后续安装 `PyYAML` 等依赖时直接使用官方仓库。
- 已验证 `python -m pip config list -v` 当前读取结果为 `global.index-url='https://pypi.org/simple'`。

### Added
- 下载并保留 `PyYAML-6.0.3-cp38-cp38-win_amd64.whl` 到 `downloads/`，作为当前 Python 3.8 x64 环境可复用的本地安装包。

### Fixed
- 绕过 `pip` 在当前网络环境下继承异常代理的问题，改为直连 `PyPI` 下载 wheel 并完成本地安装；`python -c "import yaml"` 现可正常导入。
- 重新执行 `C:\Users\Eddy\.codex\skills\.system\skill-creator\scripts\quick_validate.py` 后，`spec-coding` 技能校验结果已恢复为 `Skill is valid!`。

### Added
- 新增 `code/SPDG_framework/bench/run_paper_benchmark.py`，用于在不依赖现有损坏主入口的前提下，直接下载官方 `SST-2` 与 `IMDb` 数据源，完成真实文本分类预实验、三模型对比和长序列 scaling probe。
- 新增 `results/paper_benchmark_pilot/` 预实验结果输出，包含 `classification_results.json`、`scaling_results.json` 与 `summary.json`。

### Fixed
- 修复论文预实验脚本在限制样本数时直接截断数据导致类别分布失真的问题；受限子集现在按标签轮转采样，避免 `IMDb` 等数据集出现单类子集造成的伪高精度。

### Notes
- 当前 CPU 预实验仅说明“真实数据链路已跑通，SPDG 在小样本 `SST-2` 上有轻微准确率优势，但在 CPU scaling 上尚未表现出稳定速度优势”，不能据此直接宣称论文级先进性成立。

### Added
- 新增 `code/SPDG_framework/bench/run_autodl_glue_suite.py`，面向 AutoDL 服务器提供完整的 GLUE 论文实验入口，覆盖环境记录、官方数据集下载、tokenization 速度统计、真实训练/评估耗时、消融实验、scaling probe、图表生成与 Markdown 报告输出。
- 新增 `code/SPDG_framework/bench/run_autodl_glue_suite.sh`，固化 Linux/AutoDL 运行所需的 Hugging Face 缓存目录、依赖安装与一键启动参数。
- 新增 `reports/AutoDL_GLUE_Experiment_Runbook.md`，说明网络环境、任务选择、输出文件含义、论文取数路径与风险点。

### Fixed
- 修复 AutoDL GLUE 主脚本中梯度累积尾批次可能未执行优化器更新的问题，避免最后一个不足 `gradient_accumulation` 的 batch 被静默丢弃。
- 修复训练阶段峰值显存统计被评估阶段覆盖的问题，训练与评估峰值显存现分别记录。

### Notes
- 当前本地仅完成 Python 级静态验证：`run_autodl_glue_suite.py` 已通过 AST 解析与 `--help` 检查；由于当前 Windows 环境没有 `bash`，`run_autodl_glue_suite.sh` 未做本地语法执行验证。
