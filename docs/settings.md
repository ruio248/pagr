# 实验设置：以实现和冻结协议为准

前半部分描述仓库中可直接重放的历史 42-run follow-up：2 个任务 × 3 个 seed × 7 个方法。历史协议保留在 `src/pagr/assets/reference/history/`，不能混用其 full-adjoint 或非 Adam 设置来解释本表。新的 Flow AM 固定协议见文末，不能与旧动力学实验混为一组结果。

机器可读任务配置为 [`configs/tasks.json`](../configs/tasks.json)，方法配置为 [`configs/methods.json`](../configs/methods.json)。历史 task config 中存在继承的 `updates`、`batch` 等字段；**最新 actor 训练使用方法 `Options` 的 1000 / 64，而不是这些历史默认值**。CLI 不读取 `configs/tasks.json` 作为可调覆盖文件；它是冻结 `assets/original/protocol.json` 的可读镜像。

## 1. 共用训练设置

| 设置 | 实际值 |
|---|---|
| 训练 seed | 18001、18002、18003；全部为复用的开发 seed |
| Actor | 两个宽度 64 的 Tanh 隐藏层；末层权重和 bias 初始化为 0 |
| Adjoint | 全部 lean；不计算 actor 的状态 Jacobian，不与 full adjoint 混比 |
| 更新次数 | 1000；固定取最终 checkpoint |
| Actor optimizer | 原生 PyTorch Adam，lr=0.002，betas=(0.9, 0.999)，eps=1e-8，weight_decay=0 |
| 梯度裁剪 | 参数梯度 L2 norm 上限 5 |
| 每次 rollout batch | 64；仅 `am_2x` 为 128 |
| Actor 初始化 RNG | seed + 300000 |
| 训练 Brownian RNG | seed + 400000 |
| 训练时真实终点值/梯度 | 不用于参数更新；训练轨迹使用冻结 critic 的梯度 |
| 在线真实终点观测查询 | 0；不是 offline-to-online 新采样实验 |
| 硬件与 dtype | CPU，torch 单线程；网络/动力学 float32；核回归/凸优化的部分运算 float64 |

模型 rollout、真实终点观测、评估 oracle 三种预算分别记录，不能混称“同样的数据量/算力”。已知动力学和已知 running cost 在训练中可用；不学习动力学模型。

## 2. 原始终点 critic

| 设置 | 值 |
|---|---|
| 数据 | 同一动力学下随机 behavior controller 的终点 |
| 训练/验证终点数 | 1024 / 512，两个任务相同 |
| 每个终点的观测 | 4 次执行噪声下的物理终点代价取均值 |
| 离线标量观测总数 | (1024+512)×4 = 6144 / task / seed |
| 数据生成 RNG | seed + 200000 |
| Ensemble | 5 个；第 0 个不用 bootstrap，其余对同一训练集 bootstrap |
| 网络 | 两层宽 128 ReLU 隐藏层；输出乘 20 |
| 输入缩放 | Navigation 除以 [3,3,2,2]；Arm 除以 2 |
| 训练 loss | mean(((prediction−observation)/20)²) |
| 优化器 | Adam lr=0.002，默认 betas/eps；mini-batch=128 |
| 每个 critic 更新 | Navigation 2200；Arm 2500 |
| 选择规则 | 第 1 步及每 50 步检查**带噪标量验证 MSE**，保留最低者 |
| Critic 初始化 RNG | seed + 100000 + member×1009 |
| Bootstrap/batch RNG | seed + 900000 + member×1013 |

不使用真实终点梯度标签来训练 critic，也不按控制 return 选择 critic checkpoint。验证集不是独立的最终效果确认集，尤其不能消除重复开发造成的选择偏差。

## 3. 七个方法及预算

| CLI 方法 | 改动 | Actor rollouts | 额外 model rollouts | 总训练 model rollouts |
|---|---|---:|---:|---:|
| `am_adam` | 五个 ReLU critic 的平均终点梯度 + 原生 lean AM/Adam | 64000 | 0 | 64000 |
| `am_2x` | 同上，仅 batch 128 | 128000 | 0 | 128000 |
| `nominal_guard` | 每 20 步对参数位移做 nominal scalar-value 检查 | 64000 | 64000 | 128000 |
| `robust_guard` | 同上，检查对共享 critic mixture 的最坏差值 | 64000 | 64000 | 128000 |
| `smooth` | 换成单个 Gaussian RBF scalar critic | 64000 | 0 | 64000 |
| `residual` | 原 ensemble mean + RBF scalar residual | 64000 | 0 | 64000 |
| `silu` | 两个 ReLU 换 SiLU，其余 critic 与 actor 训练规则不变 | 64000 | 0 | 64000 |

Guard 每个 block 比较 α∈{0,0.25,0.5,1} 的 4 个策略，每个策略 320 条轨迹；50×4×320=64000。全部使用相同 Brownian 随机数（common random numbers），不是只算非零候选。Critic value 查询为 64000×5=320000。这些是已知动力学下的模拟和估计值查询，不是新物理终点标签。所有评估成本另计；各方法 wall-clock、导数运算和 critic 拟合开销并不相等。

Guard RNG：seed + 800000 + block×10007，block 从 1 开始。Adam moments 对所有 proposal mini-batch 前进，拒绝后也**不回滚 moments**，只插值 actor 参数。α=0 总包含在候选中，同分时优先选 0。

Robust mixture：5 维 simplex，uniform prior，KL radius ρ=0.3；验证平方误差不超过 uniform mixture 的 1.1 倍。一次比较的候选策略和原策略必须用**同一个** mixture。SLSQP 最大 300 次迭代，ftol=1e-11，权重下界 1e-12；可行性检查阈值 1e-7。求解器成功不等于严格最优性证明，已知警告见结果页。

RBF：用训练集各坐标均值/样本标准差标准化（std 下限 0.1），Gaussian length scale ∈{0.5,1,2,4}，ridge∈{0.001,0.01,0.1,1,10}，共 20 组；仅按原带噪验证 MSE 选择。直接回归目标是 y；residual 目标是 y−原 ensemble mean。模型具有显式平滑先验，不等于校准的梯度不确定性估计。

SiLU：相同训练/验证数据、参数数目、初始参数、bootstrap/mini-batch stream、优化器、更新预算及验证选择规则；只替换两处激活函数。结果属于 smooth-activation ablation，不能归因于新的 HJBI 控制律。

## 4. Navigation：二维避障与 docking

状态 x=(p₁,p₂,v₁,v₂)∈R⁴，控制/噪声维度 2，Euler–Maruyama。Horizon=2，K=40，dt=0.05。起点 (-2,0,0,0)，目标 (2,0,0,0)。线性参考跟踪 drift，stiffness=0.8，damping=1；控制与噪声只作用于速度，σ=0.5。所有方法共享 reference arch=0.9，不是某个候选额外得到的路线先验。

两个圆障碍 (cx,cy,r)=(-0.65,0.2,0.4)、(0.65,-0.2,0.4)。Running penalty 是每个障碍的 Gaussian bump：幅度 55，宽度 0.85r；不是硬不可穿越约束。评估 collision 对每条离散路径线段与圆的交点检查。

物理终点代价：位置平方误差权重 1，速度平方误差权重 0.25，quartic 权重 0；另有幅度 35 的 docking capture penalty，位置/速度尺度 0.4/0.5。终点执行噪声位置 std=0.1，速度 std=0.15。真实终点函数 g 是该随机物理代价的解析期望，评估固定这个 g。

Success：整条轨迹无碰撞、终点位置误差 <0.4、速度误差 <0.5。优化目标为积分 running cost + 0.5∫||u||²dt + g；success/collision 是报告指标，不作为另一种训练 objective。

## 5. Arm：六关节随机位姿生成

状态/控制/噪声均为 6 维。时间是 pose generation time，**不是完整刚体机械臂动力学或部署实验**。Horizon=1，K=32，dt=1/32；drift=−(q−q_start)，σ=0.7。起始关节角 (0,0.2,−0.1,0.1,−0.1,0)。

六段长度 (0.8,0.65,0.55,0.4,0.35,0.25)，末端目标 (1.4,1.3)，方向目标 π/2。物理终点代价：末端位置平方误差×12 + 方向项 2(1−cos(角差)) + posture 平方误差×0.1。执行噪声为独立关节高斯 std=0.15；真实 g 采用其解析期望。

无障碍，running state cost=0；仍有 0.5∫||u||²dt。Success：末端距离 <0.35 且 wrapped orientation error <0.5 rad。

## 6. 评估与统计

- 每 100 updates（含 step 0）记录 1024-path 学习曲线，RNG=seed+500000；这些固定噪声下的重复评估不是独立实验。
- 主结果：固定第 1000 次更新的 actor，4096 paths，RNG=seed+600000。不按真实评估值选 best checkpoint。
- 次要细网格评估：步数翻倍（80 / 64）、2048 paths，RNG=seed+700000；不可替代原预设主指标。
- `cost_mc_se` 是单个 actor 的 Monte Carlo 标准误，不是跨训练 seed 的标准误。
- `mean` 是 3 seed 算术平均；`reduction_pct`=(baseline_mean−candidate_mean)/baseline_mean×100%；`wins` 是严格低于配对 baseline 的 seed 数。
- 保留的 paired t 区间仅是开发数据上的描述性统计；3 个复用 seed 和多候选搜索不能提供确认性显著性或普适性证据。
- 原 finite-guard 门槛：任一任务相对两个 Adam baseline 都至少平均改善 1% 且 3/3 seed 胜出；后续 estimator 实验的门槛至少要求相对原 AM 改善 1% 且 3/3。没有新候选通过。

## 7. 保存与环境

公开包保存全部原始 42 个离线输入文件（每 seed 为 data.npz、fit.json、5 个 critic），替代 critic、42 个最终 actor、逐 run JSON、协议、诊断与报告。原始逐步大 rollout 数组未上传；`replay` 会重新生成它们。

原始环境的 Python 和核心科学计算包版本见 [`configs/environment.json`](../configs/environment.json)。这固定了核心版本，不是所有操作系统/所有传递依赖的完整锁文件。没有保证跨 CPU 架构 bitwise 相同；每次 replay 都生成数值比较报告。

## 预训练 Flow AM 固定协议

机器可读参数为 [`configs/flow_am.json`](../configs/flow_am.json)，源码中的 frozen dataclass 与其一致，测试会核对。以下与前面的历史导航/Arm 完全分开。

| 项目 | 新 Flow 实验 |
|---|---|
| 正式 / 开发 seeds | 28001–28003 / 独立 smoke 91001 |
| RNG 分配 | root_seed × 10000000 + role_offset；同 seed 方法共享相应子流，跨 seed 命名空间互斥 |
| 行为数据 | 4096 train、2048 held-out，四等权 Gaussian；中心 `(±1.5,±1.5)`，每轴 std=0.4 |
| 真实奖励 | 宽度0.85的四 bump，高度0.2/0.8/1.4/2；减 `0.025*||a||²` |
| 观测 | 512 train + 256 validation 个 scalar observations；独立 Gaussian noise，std=0.2（variance=0.04） |
| FM 网络 / 训练 | SiLU MLP，64×64 hidden；输入 x、t、sin(pi*t)、cos(pi*t)；2500 updates，batch256，lr0.001 |
| FM 选择 | 固定 held-out bridge loss，每100步评估，最早达到最小值的 checkpoint；不看真实 reward |
| Reward ensemble | 五个64×64 SiLU scalar MLP；bootstrap同一批观测，1800 updates，batch128，lr0.001；最早最小 noisy validation MSE |
| 微调预算 | 300 updates；AM-learned / oracle batch64（19200条），learned-2x batch128（38400条）；base frozen |
| Actor Adam | lr0.0003，betas0.9/0.999，eps1e-8，global norm clip5；beta固定1.5 |
| 端点 | 0→epsilon0.05由 frozen base RK2 warmup8步；epsilon→1 EM K32；无确定性最后一步 |
| 曲线 / 最终 | 每50步768 paths；固定最终 actor，4096 paths，另报 Monte Carlo SE |
| 机制诊断 | AM-learned step0/150/300，各256条共同路径，learned与oracle标签对照 |
| 虚拟 Adam 步 | 两份相同 optimizer history；2048条独立于诊断/训练的新公共评估噪声；不污染正式 actor |
| Conditional | K的1/4、1/2、3/4处，诊断路径中前3个状态，各256个未来样本；保存均值各分量 SE 和方差迹 |
| K/2K | 全部最终 actor 在K64上重新评估；只在预定首 seed 28001额外重训 learned和oracle K64 |
| ODE | 同一最终 fine velocity，从0→1，128步RK2；单独报告 reward、模式占比和 empirical sliced W2；不报告 action KL |

初次实现使用 `root_seed+offset`，首次完整运行之后审计发现跨 seed 子流重合。所有旧结果保留，但最终统计改用修复后命名空间的重跑；没有按方法胜负调整场景或优化预算。完整开发/重启记录见 [`flow_am_development.json`](../provenance/flow_am_development.json)。不能把这次实现开发过程称为完全事前注册的确认性实验。

三个正式 seed 只提供初步配对诊断。2x多花训练样本；机制诊断/虚拟步带来额外评估计算，不计入正式 actor 更新预算。单个 actor 的 `*_mc_se` 与跨 seed SD/配对t区间分开。表格中的 ODE cost/energy 空值、终端时刻不存在的 matching label 空值是“不适用”，不是被掩盖的训练 NaN。

### Oracle 收益空间追加诊断

追加实验的机器可读固定参数在 configs/oracle_am_diagnostic.json，继承上表的行为数据、真实奖励、预训练 Flow、奖励 ensemble、beta、端点和 Adam。它使用同一组三个已查看开发 seed，不是新确认实验。

| 项目 | 追加诊断 |
|---|---|
| K32 主对照 | learned/oracle × B64/B128，全部300次 Adam 更新；Frozen 同噪声参照 |
| 主 checkpoint | 0、150、300；最终4096条共同且独立于训练的 SDE 评估路径 |
| 同轨迹预算 | B64/300 与 B128/150，均19200条；更新次数不相同 |
| 重复梯度 checkpoint | learned-B64 的0/150/300，以及 oracle-B64 的300 |
| 重复数 / batch | 每 seed、checkpoint、B64/B128 各32次独立训练噪声；B64是B128的样本前缀 |
| 配对虚拟一步 | 同 actor/Adam history，learned/oracle 梯度分别更新；每对1024条新共同评价路径 |
| RNG role | 1000000起，checkpoint各隔100000；repetition各隔4；训练增量和评价增量各占连续两个 role |
| K64敏感性 | learned/oracle B64，全部三 seed 重训；不构成 K64 的完整2×2 |
| 额外成本 | 训练轨迹、生成时间步、Adam步、真值梯度点数、checkpoint运行时间和独立probe耗时分别保存 |

正式新增产物在 runs/02-flow-am-formal-8u0sh2xk/oracle-diagnostic/report.md。每组高维梯度与逐路径虚拟一步差值均以压缩数组保存，用于复算方差交叉项和配对 Monte Carlo 标准误。旧运行与失败 smoke 输出保持原样，不参与最终统计。
