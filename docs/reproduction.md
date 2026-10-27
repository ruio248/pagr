# 复现指南

以下命令从仓库根目录执行。所有实验只用 CPU；无需远端训练服务、GPU、API key、TRQAM 环境或原作者绝对路径。

## 环境

建议 Python 3.13.11，建立独立 `.venv`：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebook]'
```

Linux 若只需要 CPU wheel，可在安装项目之前执行：

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[notebook]'
```

`pyproject.toml` 固定直接数值依赖；完整参考环境见 `configs/environment.json`。不保证任意 Python 3.11–3.13/操作系统都有 bitwise 相同结果。项目声明的 Python 范围不等于所有组合均已实测。CI 使用 Ubuntu / Python 3.13.11；本地参考重放环境见 `provenance/publication_validation.json`（验证完成后生成）。

另在全新、不继承 system-site-packages 的 macOS arm64 虚拟环境中安装非 editable wheel：`pip check`、5 项测试、218 文件校验和完整一条 baseline replay 均通过。其全部已安装包版本和结果在 [`provenance/clean_install_validation.json`](../provenance/clean_install_validation.json)。这不替代尚未成功启动的 Ubuntu hosted CI；notebook 的执行检查使用前述参考环境。

## 第一层：无需训练，查看与验证

```bash
python -m pagr verify
python -m unittest discover -s tests -v
python -m pagr summary
```

`verify` 对照 SHA256 manifest 校验 218 个 shipped 源代码/资产文件；检查完整性，不判断算法有效。42 个 reference JSON、42 个最终 actor、原始数据/critic 和替代 critic 已包含，无需下载大数据。

## 第二层：用相同冻结输入重放训练

单 run：

```bash
python -m pagr replay --task arm --seed 18001 --method am_adam --output runs/arm_18001_am
python -m pagr replay --task arm --seed 18001 --method silu --output runs/arm_18001_silu
```

全部 42 run：

```bash
python -m pagr replay-suite --output runs/all_42
```

也可分别 `--task navigation` / `--task arm`，每个 21 run。只支持固定任务、seed 和七个方法；另行探索应保存新协议，不覆盖本参考配置。

每个 run 保存 `result.json`、`comparison.json`、`actor.pt`、`evaluation.npz`，部分方法另有 checkpoint / guard blocks。comparison 对比所有 final 指标，报告 exact、max error、1e-4 tolerance；这不是只复制已存在 JSON。完整 suite 会写 `suite_comparison.json`。

算法运行时间以每个 run 的 `seconds` 为准，受机器影响；初次包安装可能远慢于 toy 训练。重放生成的 rollout 数组可达数百 MB，默认在被 Git 忽略的 `runs/`，不会随仓库上传。

## 第三层：重新生成原始离线数据和拟合 critic

```bash
python -m pagr prepare --task arm --seed 18001 --output runs/generated_inputs
python -m pagr replay --task arm --seed 18001 --method silu --inputs runs/generated_inputs --output runs/regenerated_arm_silu
```

`prepare` 在相同已知动力学下从头采集该 seed 的离线 endpoints 与 6144 次终点观测，拟合 5 个 ReLU、5 个 SiLU、RBF 和 residual critic。它不是把 oracle 作为 actor 标签。若只需要原始 ReLU 可加 `--original-only`，此时只能 replay 前四种方法。

为全部 task/seed 重复 prepare 后，才可对该 `--inputs` 根目录运行完整 suite。生成 tensor archive 的字节 hash 可能因序列化细节不同；应另外比较数组和 state_dict 数值，不能只凭 archive hash 判不同数据。

固定输入 replay 与从零 prepare 是不同层次的复现。重新拟合在不同硬件产生浮点差异时，模型选择也可能变化，不应悄悄调整超参数以逼近 reference。

## Notebook

### 两份逐行展开的研究 notebook

1. [`01_motivation_analytic.ipynb`](../notebooks/01_motivation_analytic.ipynb)：解析的“值准、梯度不准”例子；一次实际 AM–SGD 更新的精确期望损害；三维线性 Gaussian 下的逐轨迹/条件均值误差；有限样本 AM 回归、状态反馈、同 KL 控制性能；梯度修复与值偏移对照；精确全局优化的反例。A 是 SGD，B 是冻结轨迹回归，不冒充多轮 Adam 实验。
2. [`02_flow_am_terminal_gradient.ipynb`](../notebooks/02_flow_am_terminal_gradient.ipynb)：四模式行为分布、真实 FM 预训练、有限标量奖励模型、随机生成、lean AM、learned/oracle/2x 对照、实际 Adam 和 conditional averaging 诊断。三个新固定 seeds，不从历史模型或结果 replay。完整设置见 [`configs/flow_am.json`](../configs/flow_am.json)。

旧 Notebook 2 为[历史导航 SOC 实验](../notebooks/archive/02_navigation_am_and_candidates.ipynb)，源码与已执行输出按 SHA-256 原样归档，原位置保留供旧审计兼容。它不是预训练 Flow 实验。

```bash
python scripts/execute_notebook.py --notebook notebooks/01_motivation_analytic.ipynb --output runs/notebooks/01_motivation.executed.ipynb
python scripts/execute_notebook.py --notebook notebooks/02_flow_am_terminal_gradient.ipynb --output runs/notebooks/02_flow_am.executed.ipynb
```

每条命令使用当前解释器对应的临时 kernel；不会偷偷切换到系统 Python。默认每个 cell 允许 1800 秒，可用 `--timeout` 调整。输出文件已存在时必须明确加 `--replace` 或换文件名。每次 Run All 另外创建唯一的 `runs/01-motivation-*` / `runs/02-flow-am-formal-*` 目录，保存全部表格、图、协议与模型，不覆盖上一次。Flow 的失败会写入该目录 `events.jsonl` 并抛错，不会静默跳过 seed。

同名 `.py` 是可直接执行的 percent-cell 源文件，代码和 notebook 一致；脚本模式建议使用非交互绘图后端：

```bash
MPLBACKEND=Agg python notebooks/01_motivation_analytic.py
MPLBACKEND=Agg python notebooks/02_flow_am_terminal_gradient.py
```

若只编辑新 Flow `.py`，使用 `python scripts/build_research_notebooks.py --name 02_flow_am_terminal_gradient --replace`（**只清除新 notebook 的执行输出**），再执行。不要无意重建 Notebook 1 或归档文件。轻量测试包含 source/notebook 一致性、旧导航审计以及新的 Flow AM 离散误差传播、VJP、有限差分、oracle guard 和 RNG coupling。

独立开发 smoke（seed 91001，不使用正式 seeds，不能据此宣称预训练质量或方法效果）：

```bash
PAGR_FLOW_PROFILE=smoke python scripts/execute_notebook.py --notebook notebooks/02_flow_am_terminal_gradient.ipynb --output runs/notebooks/02_flow_am.smoke.ipynb
```

正式环境变量默认 `formal`。每个 Flow run 内含 `config.json`、环境/source hash、数据、base/reward/actor checkpoints、optimizer history、CSV/JSON、诊断 NPZ、十张 PNG、执行事件与自动结论。`runs/` 默认 gitignored，以免普通重跑覆盖或误提交产物；已审计的[正式运行目录](../runs/02-flow-am-formal-drdkjy4g)作为例外纳入仓库，包含 checkpoint。开发 smoke 与被替代的初次运行仅保留本地及开发日志，不作为正式结果上传。

对新Flow运行做产物与checkpoint重放审计（目录改成实际输出；output必须是新的路径）：

```bash
python scripts/validate_flow_notebook.py --run runs/02-flow-am-formal-drdkjy4g --output runs/02-flow-am-validation.json
```

该脚本核对配置、执行source hash、Notebook已执行输出、冻结参数、所有训练步是否有限、归档字节，并从保存checkpoint重新生成全部38组最终评估。当前本地参考环境下逐样本数组精确相等；其他平台可能存在浮点差异，不应隐瞒为算法变化。

**仅针对历史导航 notebook**，完整执行后可用下面的命令检查从零训练的结果；该旧 validator 不适用于新 Flow 数据；将示例目录替换为旧 notebook 最后打印的目录：

```bash
python scripts/validate_research_notebooks.py --analytic-run runs/01-motivation-EXAMPLE --toy-run runs/02-navigation-EXAMPLE
```

该审计以本地参考环境下的精确一致性为标准，包括 12 组数组、30 个 critic state_dict、15 个最终 actor、所有最终/细网格指标；在其他平台不保证 bitwise 一致。`provenance/notebook_validation.json` 记录本次执行结果，不替代尚未通过的 Ubuntu hosted CI。

**方法边界**：解析例子允许人为构造误差，用于机制隔离；自然导航不做梯度腐化。Robust guard 是经验有限代价差 safeguard，不是新的 HJBI 方程，仍保留历史 6 项严格 support-certificate 警告。报告所有方法/seed 的负结果。三 seeds 的区间是探索性描述，不是独立确认。

### 保留的历史总览 notebook

[`notebooks/reproduce.ipynb`](../notebooks/reproduce.ipynb) 包含全部结果、两条真实训练重放、路径图、固定诊断及审计警告。两条示范选同一个 seed 18001，并非挑选收益最大的 seed 18003；整表始终显示全部 42 run。

```bash
python scripts/execute_notebook.py
```

默认输出 `runs/notebooks/reproduce.executed.ipynb`，不覆盖已提交 notebook。可传 `--output` 指定新文件。每次执行用不同临时 run 目录，避免覆盖先前实验。Notebook 中静态历史结果与本次训练结果分别标注。

## 代码边界

公共入口为 `pagr.runtime` 与 `python -m pagr`。`src/pagr/_legacy/` 保留实验代码目录结构以减少重构改变；不要直接运行它们原有的历史 main / freeze / audit 命令（它们包含旧路径假设及历史实验入口）。包装层把读数据路径转向 bundled assets，并将训练输出放到调用者指定目录。

公开导出只对 source 的 `torch.load(..., weights_only=False)` 改为 `weights_only=True`，不改变训练数学；JSON 内机器路径已脱敏；TeX 图路径改为相对路径。每个变化和原/发布 hash 都记在 `provenance/export_manifest.json`。不执行不可信 pickle，不要求用户上传个人凭据。

已有 PDF 可直接阅读，无需 LaTeX。若重编译历史 TeX，需要 XeLaTeX、xeCJK/fontspec 等包及原稿指定的 `Songti SC` 中文字体；这不是 Python 依赖，也不随仓库分发。没有该字体的系统需在自己的副本中改用已安装的中文字体；不保证由此生成的排版与历史 PDF 字节一致。

运行时 JSON 可能记录**执行机器自己的本地路径**，因此 runs 默认不纳入 Git。分享新结果前也应做同样脱敏，不要直接上传整个 runs 或虚拟环境。

## 常见误读

- 终点真实函数只用于离线观测生成/评估/诊断；代码里存在解析 g 不代表 actor 训练用了它。
- 模型 guard 的额外模拟不能算作零计算开销，虽然真实终点查询为 0。
- `summary` 读历史 artifact；`replay` 才真正重新训练。
- 本地同环境精确复现不是新 seed 实验，也不是验证 novelty。
- `verify` / CI 通过不清除历史审计中 6 项 solver gap 警告。
