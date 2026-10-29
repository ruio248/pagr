# PointObstacle Q-gradient magnitude × direction mechanism experiment

## 五个问题：先给结论

1. **Direction-only 是否提高闭环成功率？** N=100: +0.15 pp (有极小表观增加（6/4096 个 episode），不足以视为实质改善); final N=100: +0.15 pp.
2. **Magnitude-only 是否提高闭环成功率？** N=100: +9.96 pp (是); final N=100: +9.96 pp.
3. **Direction-only 恢复 Full-reference gap 的比例？** N=100: 1.33%; final N=100: 1.33%.
4. **经过 Q-gradient → terminal adjoint → Lean Adjoint → parameter gradient → Adam，direction-only 是否更接近 full-reference update？** N=100 的 parameter-gradient 距离更接近: 否；虚拟 Adam-step 距离更接近: 否。
5. **当前证据是否支持 direction error 是主要可恢复失效来源？** **not supported**。仅单 seed；按 N=100 的预注册闭环差距与虚拟 Adam-step 结果判定，不能推广到其他 seed/任务。

## 设计与协议

单 seed `59001`，源 run `runs/02-qam-point-obstacle-long-mc-v2-20261029`。四臂共享相同离线转移、Base、初始 Actor/Critic/target、Adam 初始化、TD/Actor 随机流、批次索引、Flow 噪声和评价噪声。Critic 在每臂继续普通 TD 更新；reference gradient 只用于 Actor terminal label，绝不进入 TD target、Critic loss、checkpoint 选择、奖励、Base 或评价。

`magnitude_only` 保留 g_L 方向并替换为 ||g_R||；`direction_only` 保留 ||g_L|| 并替换为 g_R 方向；`full_reference` 使用 g_R；Learned 使用 g_L。MC reference 每 endpoint 为8192条 IID continuation，由 long-MC-v2 的同一个 `independent_mc` score estimator 估计，后续动作来自该 branch 当前 Actor。

第一阶段为 `100` 次 Actor 和 TD 更新；检查点为 [0, 10, 25, 50, 75, 100]，曲线每臂每点 1024 回合，N=100 final 每臂 4096 回合。
口径说明：源 Long-MC-v2 中约 52.29% / 65.06% 是 N=300 的 final 对照；本机制实验按预先固定的准入规则停在 N=100，因此这里的四臂 final 数值是 N=100，不应当作同一训练步数直接对比。N=100 的 1024 回合 learned 曲线与源曲线精确重放；4096 回合 final 是本实验固定共同噪声 bank 上的独立评价。

## N=100 admission 表

| Method | Success % | Collision % | Return | Δ vs Learned | Gap recovery |
|---|---:|---:|---:|---:|---:|
| learned | 51.34 | 48.66 | -0.03227 | +0.00 pp | 0.00% |
| magnitude_only | 61.30 | 38.70 | 0.13469 | +9.96 pp | 90.47% |
| direction_only | 51.49 | 48.51 | -0.02909 | +0.15 pp | 1.33% |
| full_reference | 62.35 | 37.65 | 0.15166 | +11.01 pp | 100.00% |

Admission: direction recovery `1.33%`; direction gain `+0.15 pp`; predeclared continuation decision: **stop at N=100**.

## Gradient-label decomposition

下表为 N=100 前最后一批训练 label 的逐样本诊断汇总（每 branch 64 个点）。范数比为该批次 ||gL|| 均值除以 ||gR|| 均值；eL/eM/eD 的均值仍按逐样本相对误差直接平均，可能被 ||gR|| 接近零的样本放大，因此同时报告中位数与 p90，且保留原始 per-sample 项。

| Branch | ||gL|| | ||gR|| | cosine mean | mean-norm ratio | mean eL/eM/eD | median eL/eM/eD | p90 eL/eM/eD | cos<0 | cos<0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned | 0.08278 | 0.2069 | 0.457 | 0.400 | 1.18e+05/0.84/1.18e+05 | 0.971/0.842/0.858 | 14.9/1.82/14 | 0.172 | 0.375 |
| magnitude_only | 0.08448 | 0.1883 | 0.289 | 0.449 | 1.17e+05/0.987/1.17e+05 | 1.39/0.908/0.893 | 11.8/1.89/10.5 | 0.297 | 0.469 |
| direction_only | 0.08463 | 0.2089 | 0.439 | 0.405 | 1.2e+05/0.853/1.2e+05 | 0.965/0.849/0.841 | 15.6/1.81/14 | 0.219 | 0.422 |
| full_reference | 0.08768 | 0.1882 | 0.408 | 0.466 | 1.3e+05/0.883/1.3e+05 | 1.5/0.823/0.892 | 18/1.81/16.2 | 0.203 | 0.422 |

原始 per-sample 向量范数、rowwise norm ratio、cosine、reference MC standard errors 与三种相对误差保存在 `gradient-decomposition-per-sample.csv`；相对误差使用 protocol 中固定的 epsilon。旧格式结果中的逐样本误差列曾被批次均值覆盖，本报告按保存的 ||gL||、||gR|| 与 cosine 用欧氏距离恒等式恢复逐样本值，不重跑训练或 MC。zero-vector fraction 单独记录在 `zero-step-gradient-sanity.json` 和逐样本 CSV 中。

## Lean Adjoint / Actor gradient / Adam 虚拟更新

所有虚拟比较都在同一个 Learned branch Actor、Critic、Base、paths 和 optimizer history 上完成，不执行 optimizer.step。对每个终端标签构造 adjoint、matching parameter gradient，并用克隆的同一 Adam state 算虚拟一步。

| N | Method | cos(q,qR) | rel ||q−qR|| | cos(Adam step,ref) | rel Adam-step distance |
|---:|---|---:|---:|---:|---:|
| 0 | direction_only | 0.9762 | 0.7411 | 0.8505 | 0.5468 |
| 0 | full_reference | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 0 | learned | 0.9291 | 0.7324 | 0.7994 | 0.6333 |
| 0 | magnitude_only | 0.9698 | 0.2440 | 0.8422 | 0.5618 |
| 50 | direction_only | 0.2039 | 0.9792 | 0.5239 | 0.8589 |
| 50 | full_reference | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 50 | learned | 0.4417 | 0.9297 | 0.5943 | 0.8393 |
| 50 | magnitude_only | 0.9539 | 0.3002 | 0.9466 | 0.3230 |
| 100 | direction_only | 0.0621 | 1.0010 | 0.3408 | 0.9405 |
| 100 | full_reference | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| 100 | learned | 0.3103 | 0.9684 | 0.4323 | 0.9099 |
| 100 | magnitude_only | 0.9298 | 0.3684 | 0.9276 | 0.3805 |

## 50 条配对轨迹

### N=100

同一组初始状态、环境噪声、exogenous z/Flow 随机量。状态转移按任务定义以碰撞/成功/超时终止。

| Method | Success | Collision | Return |
|---|---:|---:|---:|
| learned | 58.0% | 42.0% | 0.0806 |
| magnitude_only | 64.0% | 36.0% | 0.1834 |
| direction_only | 60.0% | 40.0% | 0.1138 |
| full_reference | 68.0% | 32.0% | 0.2503 |

Paired transition counts: `[{"comparison": "learned_fail_to_direction_only_success", "count": 1}, {"comparison": "learned_success_to_direction_only_fail", "count": 0}, {"comparison": "learned_fail_to_magnitude_only_success", "count": 4}, {"comparison": "learned_success_to_magnitude_only_fail", "count": 1}, {"comparison": "learned_fail_to_full_reference_success", "count": 6}, {"comparison": "learned_success_to_full_reference_fail", "count": 1}, {"comparison": "learned_collision_to_direction_success", "count": 1}, {"comparison": "learned_collision_to_direction_only_success", "count": 1}, {"comparison": "learned_collision_to_magnitude_only_success", "count": 4}, {"comparison": "learned_collision_to_full_reference_success", "count": 6}]`.

## 预先固定的解释边界

- `direction_only ≈ full_reference` 且 `magnitude_only ≈ learned` 才支持本 seed/task 的方向主导解释。
- `direction_only > learned` 但远低于 full reference 表示方向有帮助但不足。
- `direction_only ≈ learned < full_reference` 表示离线方向误差改善没有转化为闭环因果收益。
- 若 magnitude-only 优于 direction-only，记录幅值侧更受支持；不事后改写。
- 本实验只有一个探索 seed；没有新算法、无调参、无 best-checkpoint 选择。结论不能推广。

## 文件与状态

- `mechanism_results.csv`: 主结果表。
- `curves.csv`: 共同1024 episode噪声下的各固定 checkpoint 曲线。
- `learned-replay-validation.csv`: 对源 run 已有检查点的 Learned 臂逐指标精确重放；源未记录 N=75 时明确标记为 unavailable。
- `gradient-decomposition-per-sample.csv`: 每10次更新的原始梯度诊断。
- `actor_update_decomposition.csv`: N=0/50/100及（若准入）N=150/300的虚拟梯度与Adam分析。
- `test50-N*/`: 四臂图片、overlay、配对 outcome CSV 和轨迹 NPZ。
- Stage status: `stage1-complete`。
