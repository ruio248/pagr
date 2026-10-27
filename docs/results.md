# 结果、motivation 与审计边界

数据来源：`src/pagr/assets/reference/summary.json`、`mechanism.json`、`guard_diagnosis.json`、`delivery_audit.json`。所有表格保留全部方法，不筛掉失败 seed。

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
