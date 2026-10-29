# Oracle-radius 与梯度方向诊断：中止的探索性结果

**状态：停止；候选未胜过 Lean AM；三 seed 训练未完成。** 本报告记录 2026-10-29 的开发诊断。训练在 seed 59103 的第 240 次更新后中止；该 seed 没有最终 checkpoint 评价。以下不构成完成的三 seed 正式对比，也不支持把当前候选包装成有效算法。

## 问题与方法

此前的路径级鲁棒候选在开发阶段基本选中零半径。这里进一步使用当前策略的高样本 Monte Carlo 梯度估计，给每个 actor batch 样本提供一个特权半径

\[
\epsilon_i=\|g_{\mathrm{MC},i}-g_{\mathrm{critic},i}\|,
\]

并把这个半径而非误差向量交给路径级鲁棒求解器。每个参考梯度使用 8192 条 simulator continuation。对比从同一 Lean AM 初始化开始的两臂，共享离线数据、初始化、TD/Actor 更新次数、优化器与评价噪声。三个预先列出的 seed 为 59101–59103，每臂计划 300 次 Actor 更新。

这是一个**特权诊断**：实际训练时借用了 MC 梯度误差的真实范数，不能部署为只访问 learned critic 的算法；有限样本误差范数是点估计，不是置信上界，也不保证真实误差落在该半径球内。它检验的是“即使知道误差大小、但不知道方向，当前对称误差球是否有帮助”，而不是一个可用的 uncertainty estimator。

另做了一项只读的梯度幅值/方向分解，复用已有的 held-out 当前策略 MC 梯度审计，不更新模型。筛选规则预先固定为 \(\|g_{MC}\|>\max(0.03,3\|SE_{MC}\|)\)。在保留的点上，`magnitude_only` 保持 learned 梯度方向、替换成 MC 梯度范数；`direction_only` 保持 learned 梯度范数、替换成 MC 梯度方向。后者仍使用特权参考方向，因此只是误差归因诊断，不是可部署方法或策略性能实验。

## Oracle-radius 训练：已完成的两个 seed

两个 seed 都完成 300 次更新，并各用 4096 个 episode 做独立最终评价。成功率差定义为 Oracle-radius 减 Lean AM：

| Seed | Lean AM | Oracle-radius 候选 | 配对差 | Lean 失败→候选成功 | Lean 成功→候选失败 |
|---|---:|---:|---:|---:|---:|
| 59101 | 54.91% | 48.73% | −6.18 pp | 35 | 288 |
| 59102 | 55.47% | 49.54% | −5.93 pp | 99 | 342 |
| 两个已完成 seed 的描述性均值 | 55.19% | 49.13% | −6.05 pp | — | — |

两次完整配对都明显恶化；成功轨迹被变成失败的数量远多于反向修复数量。这个两 seed 均值仅描述已完成的两次探索，不是预设三 seed 结果，也没有 seed 59103 的最终闭环成功率。不能把它写成显著性结论或一般规律。

第三个 seed 59103 中止时已完成更新 `N=240`，每 10 次更新保存的训练状态和训练日志均保留；尚无 N=300 checkpoint、4096 回合最终评价或 paired outcome，因此不计入上表和任何汇总均值。训练进程已中断，后续没有继续训练。

## 梯度方向 / 幅值只读审计

在已有的三个审计 seed、四个 checkpoint 上共有 192 个 Critic/MC 梯度配对点；按上面的信噪门槛保留 156 点：

| 统计量 | 结果 |
|---|---:|
| 梯度方向 cosine 均值 / 中位数 | 0.434 / 0.725 |
| cosine < 0 的比例 | 23.1% |
| cosine < 0.5 的比例 | 34.0% |
| learned/reference 梯度范数比中位数 | 0.191 |
| Learned 梯度相对误差均值 | 0.883 |
| 仅替换幅值后的相对误差均值 | 0.859 |
| 仅替换方向后的相对误差均值 | 0.688 |

误差按 \(\|g_{MC}\|\) 归一化。该分解说明在这组被筛选的梯度标签上，参考方向替换比只修正幅值更能降低相对梯度误差；但梯度方向本身仍有明显误差，而且参考方向是 Oracle 信息。它**没有训练 direction-only actor**，所以不能据此推出 direction-only AM 的闭环表现，也不能证明某个方向估计器能从普通训练数据中恢复该方向。

## 结论与停止决定

1. 这个对称“误差半径 + 路径级内层鲁棒化”候选没有收益信号：两个已完成的配对 seed 均比 Lean AM 低约 6 个百分点；第三 seed 不完整。按当前实现与这批证据，应停止该候选，不继续为它加训练预算或调半径。
2. 梯度误差不只是尺度误差。只纠正误差大小仍留下很高的梯度相对误差；使用特权真实方向的离线分解看起来更有解释力，但仍未形成可部署的方向可靠性方法。
3. 本结果**既没有否定“终点梯度误差会影响 AM 学习”的 motivation，也没有证明该 motivation**。它只显示这个特定鲁棒化构造没有把 Oracle-radius 信息转化成更好的闭环控制。
4. 可选的一步虚拟 Adam 更新诊断虽在代码路径中预留，但本轮没有运行；没有用它补造策略结果。也没有启动新的训练。

## 可复现材料

- 固定协议：[配置](../configs/oracle_radius_gradient_diagnostic.json)
- 实验 runner：[脚本](../scripts/run_oracle_radius_gradient_diagnostic.py)
- 鲁棒目标及逐样本半径实现：[实现](../scripts/terminal_consistent_robust_qam.py)
- 单测：[tests/test_terminal_consistent_robust_qam.py](../tests/test_terminal_consistent_robust_qam.py)
- 本机未纳入版本控制的完整、部分运行产物位于 `runs/02-qam-point-obstacle-oracle-radius-gradient-diagnostic-20261029-direction-v3/`；其中含完成 seed 的 final/paired CSV、第三 seed N=240 的 latest checkpoint/训练日志，以及只读梯度分解表与图。此目录受 `.gitignore` 忽略，避免把大型 checkpoint 和本地绝对路径写入公开仓库；推送的报告包含关键汇总数值。
