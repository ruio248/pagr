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

### 二维避障设计 smoke（正式三 seed 尚未启动）

[已执行的二维 Notebook](../notebooks/02_qam_point_obstacle.ipynb)当前展示的是唯一目录
[`02-qam-point-obstacle-smoke-e6eff27267`](../runs/02-qam-point-obstacle-smoke-e6eff27267/report.md)
的单 seed、N=10 **设计检查**，而非正式高精度参考结果。当前数值表在预选点的独立 MC、网格和策略积分核验中未完全收敛；`--profile formal` 因而会主动报错，不会静默用粗表跑三 seed。既有解析、16D 和 MC 诊断完全保留。

要重跑同等 smoke 预算（会创建另一个唯一目录，不覆盖已保存结果）：

```bash
.venv/bin/python -c 'import importlib.util,sys; from dataclasses import replace; from pathlib import Path; p=Path("notebooks/02_qam_point_obstacle.py"); s=importlib.util.spec_from_file_location("point",p); m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m); c,_=m.load_config();c=replace(c,actor_updates=10,curve_updates=(0,5,10),curve_episodes=256,final_episodes=512);print(m.run_experiment(c,(59000,),"smoke"))'
.venv/bin/python scripts/validate_point_obstacle_run.py runs/02-qam-point-obstacle-smoke-e6eff27267
```

上述校验命令的目录是已保存的参考 smoke；重跑时应替换为新打印的目录，并按源码调用 `render_showcase`、`render_curves` 和 `mechanism_probe` 生成展示产物。本轮数值失败细节和计算成本见该 run 的报告；不要将其单 seed 差距当作已验证的梯度因果效应。

#### 冻结当前策略的 reference 精度审计

[`审计点`](../configs/reference_audit_points.json)先于结果在独立提交中固定，覆盖 17 个几何位置、时刻和动作尺度。以下命令只读取现有 smoke 的 Ordinary `N=0` Actor/Critic，在新目录中计算逐轴加密的数值表、每点两份独立的 8192-rollout 环境 MC、以及动作有限差分；**不会训练 Actor，也不会解锁正式三 seed**：

```bash
.venv/bin/python scripts/audit_point_reference.py --output runs/point-reference-audit-my-replay
```

这个审计相当耗时，完成的数值表带 Actor、Base、环境源码、审计点和配置 hash 缓存；中断后用相同命令安全续跑，这些科学输入改变会报错而不是混用数据。审计脚本自身的续跑/展示修复单独记录在 `runner-revision-history.json`。输出中的 `reference-audit/precision_summary.json` 是 fail-closed 的精度判定；即使单个 `N=0` 策略通过，也不能自动代表未来每个更新后策略都通过。未经验证的 reference 不得作为“真值”解释 Critic 梯度误差。

已保存的审计目录是 [`02-qam-point-obstacle-reference-audit-n0-v2`](../runs/02-qam-point-obstacle-reference-audit-n0-v2/report.md)，其主 gate 为 **FAIL**；在复制仓库中不要用重放命令覆盖此目录。重放结束后，可以用下列只读校验检查完整性：

```bash
.venv/bin/python scripts/validate_point_reference_audit.py runs/02-qam-point-obstacle-reference-audit-n0-v2
```

81²/64-policy 与 121²/32-policy 的单轴加密是主 gate 失败后的诊断，不会逆向改变原 `precision_summary.json`。完整误差来源、计算成本和九项结论见已保存的 [`report.md`](../runs/02-qam-point-obstacle-reference-audit-n0-v2/report.md)。

### 两份主研究 notebook 与联合训练诊断扩展

1. [`01_motivation_analytic.ipynb`](../notebooks/01_motivation_analytic.ipynb)：解析的“值准、梯度不准”例子；一次实际 AM–SGD 更新的精确期望损害；三维线性 Gaussian 下的逐轨迹/条件均值误差；有限样本 AM 回归、状态反馈、同 KL 控制性能；梯度修复与值偏移对照；精确全局优化的反例。A 是 SGD，B 是冻结轨迹回归，不冒充多轮 Adam 实验。
2. [`02_qam_td_terminal_error.ipynb`](../notebooks/02_qam_td_terminal_error.ipynb)：当前主 Notebook 2，在同一个16维八步稀疏任务上从真实环境转移做 TD，拟合条件 `Q(k,y,a)`，训练条件 Flow，并分别执行固定参考后续策略的 Learned/Oracle AM 提取（A）及当前 Actor 后继动作的最小联合 TD+AM（B）。[`配置`](../configs/qam_td_terminal_error.json)预先冻结3个正式 seed。报告和新目录中的 checkpoint、数据、数值核验分开保存；源码直接在[同名 `.py`](../notebooks/02_qam_td_terminal_error.py)展开。
3. [`02b_joint_mc_reference_diagnostic.ipynb`](../notebooks/02b_joint_mc_reference_diagnostic.ipynb)：从 Notebook 2 的 B 分支保存的联合 Actor/Critic/Adam checkpoint 做只读的当前策略 Monte Carlo 梯度诊断；不是训练新算法，也不复用固定参考策略解析 Q 冒充当前策略真值。完整参数在[诊断配置](../configs/qam_joint_mc_diagnostic.json)，结果在[独立报告](../runs/02-qam-joint-mc-diagnostic-d026b21f0a/report.md)。

旧的 [`02_flow_am_dimension_coupling.ipynb`](../notebooks/02_flow_am_dimension_coupling.ipynb) 原样保留并[归档说明](../notebooks/archive/reward_regression_previous_main/README.md)：它是三任务的单步带噪奖励回归，不是多步 TD-Q。原来的只读重分析输出及历史核验仍保留，下述旧命令仅用于重放该历史实验。

此前二维四模式 Flow Notebook 2 及其负结果保留在[历史归档](../notebooks/archive/flow_am_four_mode/README.md)和原位置，仍可运行；它不是新任务的 `independent_d2` 成绩。更早的导航 SOC Notebook 2 为[另一个历史归档](../notebooks/archive/02_navigation_am_and_candidates.ipynb)，也不是预训练 Flow 实验。

当前主 Notebook 2 的正式执行命令（每次创建唯一新 run；正式训练较长）是：

```bash
PAGR_QAM_PROFILE=formal python scripts/execute_notebook.py --notebook notebooks/02_qam_td_terminal_error.ipynb --output runs/notebooks/02_qam_td_formal.executed.ipynb --timeout 72000
```

`PAGR_QAM_PROFILE=smoke` 是默认短链路检查，参数更小，不可当作正式结果。正式设置为3个新 seed，N=0/50/…/300，critic 预训练10000更新、5成员、固定最终 checkpoint，Base 2500更新，A/B各300 Actor 更新；A和B的目标/解释不同。完整数值 Oracle 验证在每次 run 开始前执行。若要重新构建空白 notebook，`python scripts/build_research_notebooks.py --name 02_qam_td_terminal_error --replace` 会清除该 notebook 现有执行输出，请先保存或使用不同位置。

Notebook 2b 使用已随仓库保存的正式 run，不重做训练；完整诊断仍需约44万条固定动作后的模拟轨迹及配对闭环评价，每次创建新的 `runs/02-qam-joint-mc-diagnostic-*` 目录：

```bash
python scripts/execute_notebook.py --notebook notebooks/02b_joint_mc_reference_diagnostic.ipynb --output runs/notebooks/02b_joint_mc.executed.ipynb
python scripts/validate_joint_mc_diagnostic.py runs/02-qam-joint-mc-diagnostic-d026b21f0a
```

审计命令的目录要替换成此次实际打印的新目录；第二行显示的是已提交参考结果。若重新从 `.py` 构建此 notebook，执行输出会被清除；先保留当前已执行版本。两份 MC 分组用于估计参考梯度的噪声，不把它称为精确 Oracle。

```bash
python scripts/execute_notebook.py --notebook notebooks/01_motivation_analytic.ipynb --output runs/notebooks/01_motivation.executed.ipynb
PAGR_DIM_SOURCE_RUN=runs/02-flow-am-dim-formal-8q5obfng python scripts/execute_notebook.py --notebook notebooks/02_flow_am_dimension_coupling.ipynb --output runs/notebooks/02_flow_am_fixed_updates.executed.ipynb
```

每条命令使用当前解释器对应的临时 kernel；不会偷偷切换到系统 Python。默认每个 cell 允许 1800 秒，可用 `--timeout` 调整。输出文件已存在时必须明确加 `--replace` 或换文件名。设置 `PAGR_DIM_SOURCE_RUN` 时，Notebook 2 核对原训练配置、审计和模型/数据/checkpoint来源hash，创建唯一 `runs/02-flow-am-update-analysis-formal-*`，**不训练、不重新评价未保存模型，也不修改原run**。不设置该环境变量则会新建 `runs/02-flow-am-dim-formal-*` 并完整训练三任务；这不是本轮要求的重分析命令。

同名 `.py` 是可直接执行的 percent-cell 源文件，代码和 notebook 一致；脚本模式建议使用非交互绘图后端：

```bash
MPLBACKEND=Agg python notebooks/01_motivation_analytic.py
PAGR_DIM_SOURCE_RUN=runs/02-flow-am-dim-formal-8q5obfng MPLBACKEND=Agg python notebooks/02_flow_am_dimension_coupling.py
```

若只编辑历史三任务 Flow `.py`，使用 `python scripts/build_research_notebooks.py --name 02_flow_am_dimension_coupling --replace`（**会清除该历史 notebook 的执行输出**），再执行。不要无意重建 Notebook 1 或其他归档文件。历史轻量测试见 `tests/test_flow_am_dimension_coupling.py` 和 `tests/test_fixed_update_analysis.py`；后者核对N/K/B、配对差和KL恒等式、512/4096两种评价来源、只读复用与原始文件hash。

历史三任务 Notebook 2 当时先用独立 smoke seed 99001 跑通，再以固定的 38001–38003 seed 运行。仅重做该历史轻量检查时可执行：

```bash
PAGR_DIM_PROFILE=smoke python scripts/execute_notebook.py --notebook notebooks/02_flow_am_dimension_coupling.ipynb --output runs/notebooks/02_flow_am_dim.smoke.ipynb
```

该历史正式训练配置默认 `PAGR_DIM_PROFILE=formal`。已完成的原run保存每 seed 的行为/标量数据、校准噪声、9个Base、45个critic、18个actor的checkpoint、per-path最终评价、诊断CSV、环境和source/config hash。使用更新后的审计脚本核对原run全部27组 Frozen/Learned/Oracle 最终 checkpoint 回放；其原始执行源码由[初版已执行notebook归档](../notebooks/archive/dimension_coupling_initial/README.md)提供，不会错误地拿后改版本冒充旧源码（将输出改成新路径，勿覆盖历史审计）：

```bash
python scripts/validate_flow_am_dimension_coupling.py --run runs/02-flow-am-dim-formal-8q5obfng --output runs/02-flow-am-update-analysis-formal-EXAMPLE/source-replay-validation.json
```

以下 smoke、验证与 K 网格描述只属于**归档的二维四模式 Flow notebook**，不在本轮三个任务上自动重跑：

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
