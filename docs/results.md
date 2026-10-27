# 结果、motivation 与审计边界

前半部分为历史导航/Arm结果，数据来源：`src/pagr/assets/reference/summary.json`、`mechanism.json`、`guard_diagnosis.json`、`delivery_audit.json`。所有表格保留全部方法，不筛掉失败 seed。新的预训练 Flow AM 结果见文末，与历史实验分开解释。

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

最终本地运行：`runs/02-flow-am-formal-drdkjy4g`。配置 SHA-256：`34c154ef5e49f980bcd0cc03d8286b783006b4ea3ad4902b23dda3592b1dd49b`。全部结果由实际执行的 [新 Notebook 2](../notebooks/02_flow_am_terminal_gradient.ipynb) 生成。参数/样本预算见 [设置](settings.md#预训练-flow-am固定协议)，端点和 KL 定义见 [方法](methods.md#预训练-flow-am平台)，数据/checkpoint/每条评估结果的 SHA-256 和独立重放见 [新审计](../provenance/flow_am_validation.json)。

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

- 新notebook六个代码单元格实际执行，无错误输出；57/57运行时核验通过。
- 本地15项unittest通过；新产物validator101/101通过，38组保存checkpoint重新评估的逐样本reward/cost/energy或ODE动作完全一致。
- 首轮完整实验之后发现跨seed加法子流碰撞，保留 `runs/02-flow-am-formal-1_blv1k9` 和其完整执行notebook，明确不作为最终独立seed统计。修正为互斥namespace后，在同一组seed编号、固定场景/预算下重跑；没有为了胜出改奖励、优化器或方法。详见 [开发日志](../provenance/flow_am_development.json)。
- 所有旧assets、参考结果、报告、审计和Notebook1不变；旧Notebook2复制归档且字节一致。
- 没有新候选，没有新HJBI贡献，也没有传播感知抑制算法已被验证的结论。三seed、一个四模式二维bandit与固定训练预算，只是初步研究平台。
