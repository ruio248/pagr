# 结果、motivation 与审计边界

## 二维避障长训练扩展：300 次更新之后，差距仍在

[Notebook 2c](../notebooks/02c_qam_point_long_training.ipynb) 和
[完整统计报告](../runs/02-qam-point-obstacle-long-mc-v2-20261029/report.md)
比较三个 seed（59001–59003）的 Ordinary QAM / 当前策略 MC-reference QAM，
各进行 300 次 Actor 更新及 300 次 TD 更新，固定第 300 次 checkpoint，不挑最佳中间模型。
两条分支在每个 seed 内共享初始化、离线数据、训练随机流和配对测试噪声。

最终另用 4096 个独立测试回合，得到：

| Seed | Ordinary 成功率 | MC-reference 成功率 | 配对差（百分点） |
|---|---:|---:|---:|
| 59001 | 52.29% | 65.06% | +12.77 |
| 59002 | 49.71% | 66.41% | +16.70 |
| 59003 | 51.37% | 65.92% | +14.55 |
| 三 seed 均值 | 51.12% | 65.80% | +14.67 |

成功率的 seed SD 分别为 1.31 和 0.68 个百分点；环境折扣回报均值分别为 -0.03543 与 0.21031。
结果支持在本任务和这套固定超参数下存在可持续的梯度参考收益空间，**不是仅在很短训练时暂时领先**。
但它不支持“只要训练更久，差距就会持续扩大”：另一组固定 1024 回合的曲线中，
平均差距在 N=10/25/50/100/150/200/300 分别为
13.57/9.08/12.04/12.79/13.54/14.65/15.07 个百分点，有波动，早期优势已经很明显。
1024 回合曲线与 4096 回合最终评价是不同随机流，不能静默拼接。

三个 seed 的 reference 在首次检查点 N=10 均超过 60% 成功率，Ordinary 到 N=300 均未达到；
两者都未达到预设的 70% 或 80% 门槛。这里的 crossing 只是固定检查点上的首次观测，不是精确到达时刻。

数值表版 v1 在首 seed、reference N=50 的近目标端点未通过独立梯度核验，已停止并保留原现场。
进一步加密未消除偏差，因此 v2 改用每个训练端点 8192 条 IID 当前策略模拟轨迹的 score 梯度，
所有 Actor 从 N=0 重启，没有放宽原梯度阈值。
新的 21 次抽样梯度审计全部通过；[完整产物核验](../runs/02-qam-point-obstacle-long-mc-v2-20261029/validation.json)
核对了六条分支的更新数、Adam、随机状态、归档源码/协议，并实际重放六个最终 checkpoint 的 4096 回合评价。
这不等于证明每次更新、每一个输入点的梯度都精确。标量 Q 一致性仍只是本轮诊断，不作梯度 admission 否决。

必须保留的边界：

- Reference 共额外使用 471,859,200 条从抽样中间状态开始的模拟 continuation；记录的训练更新计算时间约 1334.72 秒，Ordinary 约 4.88 秒，不含共同预训练、测试、审计和保存开销。这不是同计算预算的算法优势。
- seed 59001 参与了数值开发，三个 seed 都属于探索性证据。这里没有提出或验证新的 PAGR 抑制算法。
- 没有由此证明“Q 数值已准确，只有梯度不准”，也没有单独识别 adjoint 放大的因果作用。
- 成功率与环境回报不是包含 KL/控制能量的完整 AM 提取目标；本轮没有证明该完整控制目标改善。

## 二维避障：终端一致的路径级鲁棒 Hamiltonian 候选

在二维点质量避障任务上实现并测试了一个独立候选：[实现](../scripts/terminal_consistent_robust_qam.py)、[配对实验 runner](../scripts/run_terminal_consistent_robust_qam.py)及[冻结协议](../configs/terminal_consistent_robust_qam.json)。每条采样 Flow 路径使用一个共享的终端梯度扰动 $q$，将它沿离散 Lean Adjoint 传播，并对冻结路径上的凸二次内层最大化求解；这是一种**路径级近似**，不同样本的 $q$ 可不同，因此不是全局一致的 `delta(s,a1)` 梯度场，也不声称是严格 HJBI 控制律。

开发 seed `59000` 用 300 次更新和 2048 个 held-out episode 按成功率、碰撞、回报的冻结规则选择扰动半径。规则选择的总体最优是 `epsilon_scale=0`；所有正半径都没有提高开发成功率，较大的半径还恶化了结果。为检查非零扰动是否仍可能在新 seed 上有用，随后将 `0.05` 作为**探索性敏感性检查**，而非开发阶段选出的胜者；三个新 seed `59101–59103` 各进行 300 次 Actor 更新，每个 seed 的两分支共享初始化、TD/Actor 更新预算和评价噪声，最终各用4096个 episode 评测：

| 方法 | 成功率（均值±seed SD） | 碰撞率（均值±seed SD） | 折扣回报（均值±seed SD） |
|---|---:|---:|---:|
| Lean AM | 53.34 ± 3.20% | 46.66 ± 3.20% | 0.0028 ± 0.0536 |
| 路径级鲁棒候选（scale 0.05） | 53.22 ± 3.10% | 46.78 ± 3.10% | 0.0008 ± 0.0519 |

鲁棒候选相对 Lean AM 的配对成功率差为 **−0.12 个百分点**（seed 差为 −0.17、−0.20、0.00 个百分点）：0 胜、1 平、2 负。每 seed 的50条配对轨迹中，未观察到成功/失败标签互换。更新诊断显示平均扰动范数达到约束边界，但对名义总漂移的平均比例仅约 `0.9995`，且 costate target shift 约 `0.0067`；因此这次非零半径产生的实际控制信号变化很小，与几乎重合的学习/评测表现相符。更大的半径在开发集上表现更差，不支持单纯增大半径。

**结论：本次候选没有超过 Lean AM，也没有验证鲁棒抑制有效。** 这不否定终端梯度误差影响控制学习的 motivation；它说明当前“梯度误差球 + 冻结路径 + 二次 Hamiltonian 惩罚”构造，在这个 toy、这套半径尺度和有限样本下未形成有效改善。三个探索性 seed、一个任务不能支持普遍结论；成功率也不是完整含能量/KL的 AM 目标。完整本地训练数据、曲线、checkpoint 与逐路径评价在 `runs/02-qam-point-obstacle-terminal-consistent-robust-dev-20261029-182005-558a/formal/`，notebook/代码单测只验证实现与数值求解，不是方法有效性的证明。

### Oracle-radius 与梯度方向诊断（中止的探索实验）

[完整记录](oracle_radius_gradient_diagnostic.md)与[固定配置](../configs/oracle_radius_gradient_diagnostic.json)。新候选将每个样本的 MC-reference/learned 梯度误差范数作为鲁棒半径，但不提供误差方向；这属于特权诊断，不是可部署方法。seed 59103 在 N=240 中止，故只有 59101、59102 两个 seed 完成 300 次更新及4096回合评价：成功率配对差分别为 −6.18、−5.93 个百分点。两次均明显低于 Lean AM；不报告三 seed 正式均值，也不继续该候选训练。

对已有 MC 梯度 audit 的只读分解中，156 个可靠非零点的 mean relative error 从 learned 的0.883降到 magnitude-only的0.859、direction-only的0.688。direction-only 使用了特权参考方向，未进行 actor 训练，不能视为方法收益。结果提示误差方向可能比单纯误差半径更关键，但没有提供一个无需 Oracle 的可靠方向估计器，也没有证明该机制会改善控制性能。完整证据边界和部分 checkpoint 说明见上述报告。

## 当前主 Notebook 2：多步稀疏任务的 TD-Q 梯度实验

完整执行输出在[新 formal run](../runs/02-qam-td-formal-31c55defef/report.md)，[Notebook 2](../notebooks/02_qam_td_terminal_error.ipynb) 已实际执行，6个代码单元没有错误。三个预设 seed `49001–49003` 共享同一个16维、8步任务设计、目标和半径；训练各512 episode（4096转移），成功数分别为113/107/108，只有最后环境步有非零奖励。解析 `Q^{pi_ref}` 的独立 Monte Carlo、中心差分、Bellman、终止与数值边界核验通过。[验证记录](../runs/02-qam-td-formal-31c55defef/validation.json)确认历史单步 notebook 字节未动，所有预设固定 `N` checkpoint、数据和机制 NPZ 均存在。不能把这项新任务说成官方 QAM 使用的 benchmark。

实验 A 的 Critic **只用真实转移和 TD backup** 训练10000步、5成员，后继动作来自固定 `pi_ref`，最终 checkpoint 固定且没有用 Oracle 选模型。验证集的 `Q_hat−Q_ref` RMSE 分别为0.1686/0.1673/0.1663，模型的 `Q` 非恒零，但误差相对约0.18的真实Q均值并不小；因此本轮**未实现“值几乎准确而仅梯度明显失真”的自然前提**。共同路径的 N=300 终点动作梯度相对 RMS 误差均值约1.005。`k=0` 的即时奖励恒为0，而 `Q_ref` 随动作变化且梯度非零，证明这不是即时奖励梯度回归。

固定 Actor 更新次数 `N=300`，使用与曲线不同、独立的512条高样本最终闭环 episode 和512个固定提取状态（各方法同 seed 共享评价噪声）：

| seed | A Learned J_extract ↓ | A Oracle J_extract ↓ | A Learned 成功率 ↑ | A Oracle 成功率 ↑ | B 联合成功率 ↑ |
|---|---:|---:|---:|---:|---:|
| 49001 | -1.93193 | -2.01373 | 0.56836 | 0.60742 | 0.67969 |
| 49002 | -2.08695 | -2.14977 | 0.57813 | 0.65625 | 0.68945 |
| 49003 | -2.12377 | -2.19145 | 0.56641 | 0.65430 | 0.64844 |

实验 A 的配对 `J_extract` 差（Learned − Oracle，正值表示 Oracle 更好）为0.08180/0.06282/0.06768，均值0.07077，3/3同号；闭环成功率差（Oracle − Learned）为0.03906/0.07813/0.08789，均值0.06836，3/3同号。这支持在**本任务、固定 pi_ref critic 目标、此有限训练预算**下存在可恢复的估计梯度相关损失；三个探索 seed 不证明普遍规律，也不能把 `J_extract` 当环境回报。原始固定N=0/50/…/300曲线和每 seed 表在[报告](../runs/02-qam-td-formal-31c55defef/report.md)及[图](../runs/02-qam-td-formal-31c55defef/fixed-N-and-propagation.png)。

同一个 Actor/相同路径的终点误差、`ν_j δp_{j+1}` 标签误差、Matching 参数梯度差、克隆相同 Adam history 后的虚拟步差均被保存；最大伴随传播残差为约`1.9e-6`、参数投影恒等式残差约`3.3e-8`。N=300 的伴随误差在终端约1.07 RMS，逆传到起点约0.0042，**没有出现必然放大**，但仍改变 Matching/Adam 更新。[每维及方向统计](../runs/02-qam-td-formal-31c55defef/gradient-dimension-direction-diagnostics.csv)和[每维图](../runs/02-qam-td-formal-31c55defef/terminal-gradient-per-dimension.png)另存，未改变训练结果。

实验 B 复用 A 的 Critic 预训练初始化，之后每次 TD 后继动作由当前 Actor 的同一 SDE sampler 给出，并交替执行一次 Critic/一次 AM Actor 更新。最终闭环成功率均值0.67253；它验证联合 TD→Q动作梯度→Lean AM→Actor 的调用链及本设置中的性能提升，**但没有精确当前策略 Oracle，不能把 A 的 `Q^{pi_ref}` 当作 B 的梯度误差真值**，也不能与 A 的 Oracle 差混表。两分支均不是完整官方工程复现。额外解析 Oracle 标签计算的64样本 CPU microbenchmark（单批约0.126–0.129 ms，Learned critic 自动微分约0.342–0.347 ms）[单独记录](../runs/02-qam-td-formal-31c55defef/terminal-label-compute-cost.json)；不含训练/rollout，不代表两方法 wall-clock 效率。没有提出或测试新的抑制算法。

### 联合 TD＋AM 的当前策略 Monte Carlo 梯度诊断（Notebook 2b）

[Notebook 2b](../notebooks/02b_joint_mc_reference_diagnostic.ipynb)只读取 B 的三个 seed、`N=0/150/300` 保存的 Actor/Critic/Adam 状态；每个快照固定 `k=0/4/7` 各一个独立评价状态，不重新训练。对生成动作端点固定首动作，之后用**该快照当前 Actor**闭环模拟；利用已知高斯环境转移的 score 恒等式估计 `∇_a Q^{π_actor}`。每个点两份独立8192条轨迹，最后一步环境噪声在 `k<7` 时解析积分以降低方差。它是有限样本的参考估计，不是精确 Oracle，也不在正常 TD＋AM 训练中使用。[完整27点和9个快照结果](../runs/02-qam-joint-mc-diagnostic-d026b21f0a/report.md)、[图](../runs/02-qam-joint-mc-diagnostic-d026b21f0a/joint-mc-diagnostic.png)与[审计](../runs/02-qam-joint-mc-diagnostic-d026b21f0a/validation.json)。

在选定27点上，critic–MC 动作梯度差的 L2 范数均值约0.348，两份独立MC梯度差均值约0.104；27/27点前者较大。相同路径的终点标签差经 Lean adjoint 传到 matching（九快照标签差 RMS 均值约0.233），虚拟 Adam 步差范数均值约0.0285，而MC两半估计造成的对应步差均值约0.0141。这提供了**当前策略下误差可传播到更新**的诊断证据，但MC差异不是正式同时置信界，也不能保证每一点的真实梯度偏差已被精确测出。

这一轮并**不支持**“Q值已比较准确、主要是梯度错”的更强命题：27点的Q值平均绝对偏差约0.238，而MC Q均值约0.406。三状态微批量的一步虚拟更新，在2048条共享噪声闭环评价中，MC参考梯度分支相对learned分支有8/9个正的点估计，但只有2/9个未经多重比较校正的逐快照95%区间排除零；按三个训练seed先对三个N取平均，平均成功率差约+0.00233、探索性seed层95% t区间约`[-0.00203, +0.00669]`。这不是完整联合再训练，也没有确认最终性能提升或新算法胜出。

以下旧章节为历史导航/Arm与单步 Flow 结果，数据来源包括 `src/pagr/assets/reference/summary.json`、`mechanism.json`、`guard_diagnosis.json`、`delivery_audit.json`。所有表格保留全部方法，不筛掉失败 seed；它们与上述多步 TD-Q 实验分开解释。

## 能说什么，不能说什么

当前结果支持把“值是否接近”和“导数对训练是否有用”分开测量。**尚未完整证明**“值误差很小、梯度误差较大，经传播导致最终性能下降”这条因果链，更未证明已有候选普遍优于原始 lean AM + Adam。

Arm 上 SiLU 的平均代价改善 2.598%，仅 2/3 seed 胜出；Navigation 则恶化 3.635%，0/3 胜出。不能把这个任务选择后的结果作为稳定算法贡献。AM 2× rollout 在 Arm 上改善 1.497%、3/3 胜出，但来自双倍 actor rollout 预算。

Nominal / robust finite guards 两个任务均 0/3 胜出；robust 相对原 AM 平均恶化 Navigation 0.896%、Arm 2.153%。RBF 在 Navigation 上平均改善 0.778%、仅 1/3 胜出，在 Arm 上明显恶化。Residual 没有达到开发门槛。

## 梯度机制诊断

在原始 AM 的共同 checkpoint（500/1000），使用每 checkpoint 1024 条新诊断路径，避免把两个已不同的 policy distribution 直接当成纯 estimator 对比。诊断不反向用于训练。

Arm，原 ReLU → SiLU，记录的聚合指标：

| 指标 | 原 ReLU | SiLU |
|---|---:|---:|
| 终点值相对 RMS error | 0.214854 | 0.214489 |
| 终点梯度相对误差 | 0.328274 | 0.235331 |
| Lean matching 参数梯度相对误差 | 0.745226 | 0.573662 |
| Lean matching 参数梯度 cosine | 0.641817 | 0.818089 |
| 共享 Adam history 下的 step 相对误差 | 0.306537 | 0.245857 |
| 共享 Adam history 下的 step cosine | 0.948382 | 0.936584 |

相近的值误差与不同的梯度/传播误差确实可以并存。不过这里值相对误差约 21%，不能描述为“几乎完美”。Adam step cosine 还变差了，说明终点梯度改善不等于所有更新指标改善。这是有限诊断证据，不是全链条因果证明。

Guard 的 held-out 诊断中，17 个修改 block 有 11 个帮助、6 个有害；存在方向正确的局部决策并不代表全程最终代价改善。

## 数值审计：明确保留失败标记

- 原始离线 42 个输入文件的 hash 保持一致。
- 历史审计中 6 个原生 AM 最终指标精确复现。
- 1200 项 convex support 检查出现 **6 项**超过 2e-5 严格阈值的 gap 警告，最大约 0.001082434。
- 对最终实际选中的 300 个 block，审计记录了 empirical candidate-set nonincrease；这不是 true-J 证书。
- 原 `delivery_audit.json` 的 `passed` 仍为 **false**，`passed_required_integrity_checks` 为 true。公开包不把它改为全绿。

本仓库 CI 的通过表示基础实现/完整性测试通过，**不表示历史所有 numerical certificate 通过，也不表示研究假设成立**。软件测试和研究有效性是两层不同证据。

## 历史比较的限制

早期 full-adjoint / optimizer 配置存在不同对照，不能用于声称当前全 lean AM/Adam 被击败。历史 all-lean terminal-consistent minimax 在 Arm 上相对原 AM/Adam 约恶化 1.38%，3 seed 均未胜出；这些记录保留在 `assets/reference/history`。

当前 follow-up 是观察先前负结果后继续进行的开发阶段实验，复用 seed/数据，并不是一次从未看过数据的预注册确认试验。`protocol.json` 记录各阶段规则，但不将整个探索历史倒写成单次预注册成功实验。未来需要新 seed、明确主比较、公平预算、误差覆盖验证和机制干预，才能提高结论强度。

## 预训练 Flow AM 平台

上一轮平台正式运行：`runs/02-flow-am-formal-drdkjy4g`。配置 SHA-256：`34c154ef5e49f980bcd0cc03d8286b783006b4ea3ad4902b23dda3592b1dd49b`。该历史运行由当时实际执行的 Notebook 2 生成；其后又追加了下文历史诊断。参数/样本预算见 [设置](settings.md#预训练-flow-am固定协议)，端点和 KL 定义见 [方法](methods.md#预训练-flow-am平台)，数据/checkpoint/每条评估结果的 SHA-256 和独立重放见 [审计](../provenance/flow_am_validation.json)。

本轮完成3个真实 FM 模型、15个 scalar reward 模型、9次主 actor 微调和2次首 seed 的K64重训。四个行为模式均被 base ODE 覆盖，最近中心模式占比约23%–28%；base held-out FM loss从6.58/6.78/6.79降至4.16/4.31/4.27。FM loss包含随机桥的不可约条件方差，不能以其不为零判定没学到 Flow；同时 base 不是精确分布模型。

### 主结果：固定最终 checkpoint，SDE K32

| 方法 | 真实奖励，mean ± seed SD ↑ | 真实 J，mean ± seed SD ↓ | energy / path KL MC均值 |
|---|---:|---:|---:|
| Frozen Flow | 0.783288 ± 0.030332 | -1.174933 ± 0.045498 | 0 |
| AM-learned | 1.174199 ± 0.018269 | -1.466112 ± 0.055979 | 0.295187 |
| AM-oracle | 1.186407 ± 0.019661 | -1.465560 ± 0.058587 | 0.314051 |
| AM-learned-2x | 1.199065 ± 0.018959 | -1.474315 ± 0.056979 | 0.324283 |

Oracle减learned的逐seed真实奖励差为 `+0.012126, +0.015801, +0.008696`；均值 **+0.012208**，95%配对t区间 `[0.003380, 0.021036]`。三个seed上都提高奖励，但这不是含能量控制目标的同义词。

真实J差为 `+0.003203, -0.005095, +0.003547`；均值 **+0.000552**，95%区间 `[-0.011603, 0.012707]`，oracle仅1/3 seed的J更低。增益被更大的控制能量大体抵消。因此：**该K32设置中存在奖励层面的梯度替换收益；尚未确认真实regularized objective的稳定收益空间。** 2x表现更好但花了双倍训练轨迹，不能作为同预算新算法贡献。

### 机制证据，不等同于性能保证

共同 AM-learned step300，三seed平均：值RMSE/真实奖励标准差 **14.28%**，终端梯度相对RMS误差 **30.43%**，matching参数梯度相对误差 **32.38%**。参数梯度cosine约 **0.9372**，相同Adam历史下实际步cosine约 **0.9935**。说明原始梯度误差、参数投影与实际优化步不能互相替代。

此处数值误差并非可忽略，不能把它称为“值已近乎准确”的完整自然实例证明。离散base Jacobian乘积和actor投影恒等式在真实Flow上通过数值验证，但并不是最优控制性能因果链的普适证明。

step300，固定早中间时刻 `t=0.2875` 的条件续采样：逐路径标签误差RMS平均约 **2.158**，条件均值范数平均约 **0.149**。每个状态256条未来样本，其均值分量MC SE平均约0.092/0.082，条件均值范数还有有限样本偏差；不能据此宣布真实条件偏差为零。时刻/状态逐项均值、方差和SE均保存。Ensemble disagreement仍未校准为真实误差。

虚拟一步共享同一份Adam历史，在另外2048条公共评估噪声上测量真实奖励和J变化；正式actor不被这些诊断更新。它不保证每一步oracle都优于learned，也不等同于真实J的无偏policy gradient。

### 网格与 sampler 敏感性

全体最终actor在K64上仅重评一次；另外首 seed 28001重新训练了learned/oracle K64，参数预算不变：

| seed28001 / 方法 | K32训练、K32评估 reward / J | K32训练、K64评估 reward / J | K64重训、K64评估 reward / J |
|---|---:|---:|---:|
| learned | 1.154013 / -1.401517 | 1.165193 / -1.419058 | 1.196715 / -1.406138 |
| oracle | 1.166139 / -1.398314 | 1.175985 / -1.413551 | 1.192400 / -1.417050 |

重训后首seed的oracle reward反而略低，J更低；不能把K32的奖励排名写成对离散化稳定。此检查只有一seed，固定epsilon不变，不证明理想time-zero SDE收敛。

同一K32训练actor用128步RK2 ODE部署时，oracle相对learned的奖励平均 **+0.010561**，3/3 seed，配对区间 `[0.001795, 0.019327]`。SDE和ODE绝对回报、模式占比与分布均不同，完整表/图已输出。SDE前epsilon采用base warmup，ODE从0起就用fine velocity，这也是不能默认两者相等的原因之一。

### 实现/开发审计与限制

- 当时版本的notebook六个代码单元格实际执行，无错误输出；57/57运行时核验通过。
- 本地15项unittest通过；新产物validator101/101通过，38组保存checkpoint重新评估的逐样本reward/cost/energy或ODE动作完全一致。
- 首轮完整实验之后发现跨seed加法子流碰撞，保留 `runs/02-flow-am-formal-1_blv1k9` 和其完整执行notebook，明确不作为最终独立seed统计。修正为互斥namespace后，在同一组seed编号、固定场景/预算下重跑；没有为了胜出改奖励、优化器或方法。详见 [开发日志](../provenance/flow_am_development.json)。
- 所有旧assets、参考结果、报告、审计和Notebook1不变；旧Notebook2复制归档且字节一致。
- 截至该历史四模式 bandit 阶段，没有新候选或新 HJBI 贡献。后续二维避障路径级鲁棒候选及其负结果见上文；目前仍没有传播感知抑制算法已被验证有效的结论。三 seed 和固定预算都只是初步研究证据。

## Oracle AM 收益空间诊断（追加开发实验）

最新完整运行：runs/02-flow-am-formal-8u0sh2xk，报告见该目录的 oracle-diagnostic/report.md，固定配置见 configs/oracle_am_diagnostic.json，逐项审计见 oracle-diagnostic/validation-v2.json。这是同一组三个已查看的开发 seed 28001–28003，不是独立确认实验。旧三臂和首 seed K64 在新目录重训，数据、模型、0/150/300 actor/Adam 状态与逐路径评估对旧运行精确重放；Oracle-B128 三 seed 与 K64 后两 seed 为新增训练。没有新增 uncertainty 算法、重选超参或修改真实目标。

K32、B64/B128 × learned/oracle、固定300次更新的均值（括号内为训练 seed SD）：

| 方法 | 真实奖励 ↑ | KL/energy ↓ | 真实 J ↓ |
|---|---:|---:|---:|
| Frozen Flow | 0.783288 | 0 | −1.174933 |
| learned B64 | 1.174199 (0.018269) | 0.295187 (0.030292) | −1.466112 (0.055979) |
| oracle B64 | 1.186407 (0.019661) | 0.314051 (0.032642) | −1.465560 (0.058587) |
| learned B128 | 1.199065 (0.018959) | 0.324283 (0.030429) | −1.474315 (0.056979) |
| oracle B128 | 1.218299 (0.034497) | 0.357274 (0.029051) | −1.470175 (0.055714) |

以正值表示 Oracle 改善 J 的配对增益，A64=−0.000552（1/3 seed为正；描述性95%配对t区间 [−0.012707, 0.011603]），A128=−0.004140（0/3为正；[−0.008650, 0.000371]）。两个 batch 的 Oracle 均提高平均奖励（+0.012208、+0.019234），但也提高 KL/energy（+0.018864、+0.032991），因此未观察到稳定的真实 J 可恢复收益。这不是“Oracle 与 learned 完全等效”的证明。终点相对 Frozen 的平均位移也随 Oracle 增大，逐 seed 见完整报告。

固定300次更新时 learned 从B64到B128的 J 改善均值 S_learned=+0.008203，3/3 seed；Oracle 的 S_oracle=+0.004615，2/3 seed。交互 I=A64−A128=+0.003588，描述性区间 [−0.008039, 0.015215]，不支持稳定的 batch×标签交互结论。B128/300消耗38400条训练轨迹，B64/300只消耗19200条，故这并非纯采样方差效应。在同为19200条的 B64/300 对 B128/150 中，learned 的 S=−0.029415；后者 Adam 步数只有一半，因此同轨迹数也不是同优化预算。

共同 checkpoint 上，4个预定状态 × 2个batch × 3 seed × 32次独立噪声，共768对 learned/oracle 匹配梯度及虚拟 Adam 更新，每对用1024条新的共同评价路径。以 learned step300 为例，B64 的三seed平均 trCov(qL)/trCov(qO)/trCov(d)/trCov(qO,d) 约为 7.134/10.357/1.093/−2.158；B128 对应 3.958/5.983/0.647/−1.336。交叉项为负且不可忽略，不能把两方差直接相加。B128 下这些有限批次的方差迹较低，但平均标签差范数并未消失（B64约0.978，B128约0.970）；32次均值不是真正已知的 bias。

同一 learned step300 的原始梯度 cosine 平均从 B64 0.924 到 B128 0.937；共享 Adam history 的实际一步 cosine 约 0.989/0.990，相对参数步差仍约 0.143/0.136。这支持“在测量的共享历史上，Adam 改变了标签差异对当前一步的作用”，不意味着它消除了误差或独立训练历史相同。Step0 的实际一步 cosine 只有约0.85，说明阶段依赖。裁剪对梯度方向只做缩放，768对中 learned 有47次、oracle有114次超过 norm 5；原始与裁剪后 cosine 相同并非没发生裁剪。虚拟一步的 D_J 与 D_R、逐路径差标准误、批间 SD 已分别保存；在 learned step300，D_J 的平均值为 B64约−0.000220、B128约−0.000288，但这些局部步不足以推出300步的长期排序。

K32 actor 在 K32/K64 上重评，K64的 learned/oracle B64 也已在全部三个 seed 重训；至少两个 seed 的 Oracle–learned J 符号随训练网格改变，因此尚未得到网格稳定的结论。K64没有完整2×2，不能推断 batch×网格交互。

当前数据最支持的是“Oracle 奖励增益与更大控制代价相抵消”，同时显示有限 batch 波动和 Adam 历史会改变局部梯度的作用；三者对最终 J 的独立因果份额仍无法区分。保存的四臂曲线、所有 checkpoint、逐路径数组、probe 原始参数梯度、配对效应与训练预算均在新运行目录。7个代码单元格已执行无错误；原平台重放验证101项、57项运行时核验和新增收益空间验证492项通过，19项单元测试通过。这些是实现/数值完整性证据，不是研究假设的证明。
# 历史单步 Notebook 2：维度 × 耦合探索实验

历史这轮在三个新任务上分别预训练 Base Flow、拟合五成员有限标量奖励模型，再在同一 Base 上各跑 Learned/Oracle AM；seed 38001–38003，固定300次actor更新、最终4096条共同评价噪声。完整每 seed 的真实 `J=PathKL−1.5 E[R]`、奖励、PathKL、数据、checkpoint和机制 CSV 在[正式运行报告](../runs/02-flow-am-dim-formal-8q5obfng/report.md)。[旧 notebook](../notebooks/02_flow_am_dimension_coupling.ipynb)已实际执行；[201项产物检查与27组逐路径精确重放](../runs/02-flow-am-dim-formal-8q5obfng/validation.json)通过。当时未开发新的误差抑制算法。

| 任务 | Frozen J/d | Learned J/d | Oracle J/d | `G=(J_L−J_O)/d` | 终端梯度相对RMS误差 | `q_L−q_O` 相对差 | 共享Adam一步相对差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| independent_d2 | 1.80694±0.05941 | 0.81932±0.01480 | 0.81707±0.01673 | 0.00224±0.00194 | 0.1132 | 0.5758 | 0.3246 |
| independent_d16 | 1.88185±0.05087 | 0.89733±0.03056 | 0.84998±0.01546 | 0.04735±0.03203 | 0.4047 | 0.9731 | 0.6551 |
| coupled_d16 | 1.95956±0.02122 | 1.00072±0.01534 | 0.97265±0.01194 | 0.02807±0.00578 | 0.3063 | 0.9078 | 0.5804 |

`J/d` 和 `G` 的 ± 是三个训练 seed 的 SD，不是每条评估路径的独立训练不确定性；机制列是固定 Learned 最终 checkpoint 上每 seed 8组B64共同路径的均值。所有 9 个任务–seed 配对的 G 都为正，且 Oracle 的真实 J 均优于 Frozen；这次高维平台至少显示了有限预算下的 Oracle 收益空间。值得注意的是，`independent_d16` 的 Oracle 平均奖励并不高于 Learned（−0.3330 对 −0.3295），但路径 KL/d 较低（0.3505 对 0.4030），因此真实 J 更好。若只看奖励将得出错误结论。

对用户关心的五个问题，实际结果是：

1. **增加任务相关维度时**，终端梯度相对RMS误差从 0.1132 增至 0.4047，G 从 0.00224 增至 0.04735。三 seed 都同号，但这只是固定数据预算下的探索性维度效应，不是维度必然导致误差的定律。
2. **加入非线性行为耦合时**，所测 Base Jacobian 非对角能量占比从 0.1819 增至 0.3746；传播算子最大奇异值从 0.0465 变为 0.0727，仍远小于1。局部结构确有变化，测得的是收缩而非强制放大。
3. **耦合没有带来更大收益空间**：与独立 d16 相比，终端梯度误差下降到 0.3063，G 下降到 0.02807。故“非线性依赖必然放大梯度危害”不受本轮数据支持。
4. **支持的部分**是高维独立任务相对二维出现更明显的自然估计梯度误差、局部AM更新差和同目标 Oracle 收益；**未支持的部分**是耦合进一步放大的预测。三个 seed、固定网络/预算下不能推断普遍效果，也没有证明某个新算法优于AM。
5. **仍无法区分**终端梯度本身、critic 值拟合难度、actor路径状态分布、行为先验、Base拟合质量和有限优化的各自因果贡献。最终诊断的值RMSE/真实奖励SD分别约0.130、0.884、0.398；B的标量估计也明显更难，故不能说“值几乎完全正确但梯度独立变坏”。

Base 的 held-out FM loss/d 分别约 1.556、1.576、1.644；Base SDE对行为的固定随机投影距离约0.0825、0.0772、0.0766，邻接相关误差RMS约0.0384、0.0365、0.0321。这些指标、每维均值/方差和耦合分布的逆变换检查保存在各 seed 的 `base-quality.json`；它们显示可运行的预训练与有限维分布检查，但随机投影距离或二维散点不证明完整16维分布拟合。真实训练没有用真值指导 critic/learned actor，Oracle 仅是终点标签替换的诊断，也不是可部署上界。旧二维四模式 Flow 的负结果按[字节级归档](../notebooks/archive/flow_am_four_mode/README.md)保留，不能并入本轮 `independent_d2` 统计。

### 同 Actor 更新次数的只读重分析

这节**没有新训练，也没有新增checkpoint评价**。它读取原正式 run 的 `learning-curves.csv`：`N=0,50,100,150,200,250,300`，每个任务/seed两臂在同一512条评价噪声下测得；训练时只保存了0/150/300的模型与优化器checkpoint。最终4096条路径评价来自另一独立噪声流，分表呈现。固定 `K=64`、`B=64`；`N` 是Actor的实际 `optimizer.step()` 次数，不是生成步数或运行时间。

正的 `ΔR=(R_Oracle−R_Learned)/d` 表示Oracle任务奖励更高，正的 `ΔJ=(J_Learned−J_Oracle)/d` 表示Oracle原随机控制目标更低。下表为512路径曲线的三训练seed配对均值，括号中是优势同向seed数；完整七点、逐seed数值、SD与KL分解见[固定N报告](../runs/02-flow-am-update-analysis-formal-q6xpxevn/report.md)和[机器可读表](../runs/02-flow-am-update-analysis-formal-q6xpxevn/paired-updates-by-seed.csv)。

| 任务 | N=50：ΔR / ΔJ | N=150：ΔR / ΔJ | N=300：ΔR / ΔJ |
|---|---:|---:|---:|
| independent_d2 | −0.00060 (1/3) / +0.00117 (1/3) | −0.00092 (1/3) / +0.00281 (3/3) | −0.00212 (1/3) / +0.00168 (3/3) |
| independent_d16 | −0.00118 (1/3) / +0.00717 (3/3) | −0.00587 (1/3) / +0.03924 (3/3) | −0.00348 (1/3) / +0.04711 (3/3) |
| coupled_d16 | +0.00634 (3/3) / +0.01038 (3/3) | +0.01395 (2/3) / +0.02789 (3/3) | +0.01441 (2/3) / +0.02842 (3/3) |

所以，对于“同样更新N次”的问题，答案必须分指标：二维独立任务的Oracle奖励无稳定优势，`J` 的小差距到中后段才在三seed同向；十六维独立任务从N=50开始 `J` 在三seed同向更好，差距到后段增大，但平均奖励反而略低，主要伴随Oracle更低的Path KL；十六维耦合任务从N=50起两指标平均均占优，但后段奖励只在2/3 seed同向。独立的4096路径最终评价也保留这个区分：三任务的Oracle `J` 都是3/3同向，奖励同向数依次为1/3、1/3、2/3。`ΔJ=1.5ΔR−ΔKL`逐seed逐N核对成立。未做同KL校准，最终 `J` 降低百分比仅是性能百分比，不是训练步数或运行时间加速比。

这组结果支持“终端标签真值可能在有限N内改变控制表现”，但不表明Oracle一般更快、更高奖励或渐近更优；也不能单独区分终点梯度误差、critic拟合、Base质量与actor状态分布。三seed、300步为探索性范围，Oracle真实梯度只作为诊断对照。新的[分析来源清单](../runs/02-flow-am-update-analysis-formal-q6xpxevn/source-manifest.json)和[原始已执行notebook归档](../notebooks/archive/dimension_coupling_initial/README.md)明确区分重分析与历史训练。
