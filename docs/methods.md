# 公式与代码对应

本页区分研究 motivation、现有算法和未被证明的部分。PAGR 是研究方向名，不能将全部 ablation 统称为一个已成功的方法。

## 固定真实目标

设控制与噪声共用矩阵 S，

$$dX_t=(b(X_t,t)+S u_\theta(X_t,t))dt+S\,dW_t,$$

$$J(\theta)=\mathbb E\left[g(X_T)+\int_0^T\left(\ell(X_t)+\frac12\|u_\theta(X_t,t)\|^2\right)dt\right].$$

这里真实 g 和 J 不随方法改变。训练只获得带噪终点代价的离线观测并拟合 ĝ。把估计梯度误差建模为 δ=∇ĝ−∇g，不意味着真实任务的终点代价被对手改变。

## 历史导航 / Arm 代码的 lean matching

对固定采样轨迹，令 Fₖ 是不含 actor 状态 Jacobian 的 Euler drift Jacobian。当前两个任务的 F 不依赖状态。列向量记号下：

$$a_K=\nabla\hat g(X_K),\qquad a_k=F_k^\top a_{k+1}+\Delta t\,\nabla\ell(X_k),\qquad y_k=-S^\top a_{k+1}.$$

$$L_\mathrm{AM}(\theta;X,y)=\frac{\Delta t}{2}\,\mathbb E_B\sum_{k=0}^{K-1}\|u_\theta(X_k,t_k)-\operatorname{stopgrad}(y_k)\|^2.$$

轨迹 X 和标签 y 对此次反传冻结；Adam 接收 q=∇θL_AM（裁剪 norm 5 后更新）。它不是沿策略依赖轨迹完整求导的 ∇θJ。实现见 `lean_am.matching_gradients` / `lean_am.update`，公共入口经 `pagr.runtime.backend` 加载。

## 终点误差怎样传播

在**同一条冻结轨迹、相同 dynamics 和 running cost**下，终点误差 δ 引起

$$\Delta a_{k+1}=P_{k+1}\delta,\qquad P_{k+1}=F_{k+1}^\top\cdots F_{K-1}^\top,\qquad P_K=I,$$

$$\Delta y_k=-S^\top P_{k+1}\delta,$$

$$\Delta q=\Delta t\,\mathbb E_B\sum_k D_\theta u_\theta(X_k,t_k)^\top S^\top P_{k+1}\delta.$$

这是冻结路径 matching gradient 的误差表达式。只有通过此算子投影到 actor 参数更新方向的部分会在这一步产生作用；原始终点梯度误差的范数或 ensemble variance 不是充分的影响度量。

但下一步以后轨迹分布、模型和 Adam moments 都可能改变，不能直接由此式推出最终真实 J 的排序。梯度裁剪和 Adam 也是非线性/历史依赖变换。即使使用真实 ∇g，lean matching 的 q 也不应不加条件地称为真实策略梯度。

## 目前实际尝试的 finite-value guard

先用相同 lean AM + Adam 做 20 次更新，得到参数 proposal θ⁺；候选为 θ_α=θ+α(θ⁺−θ)，α∈{0,1/4,1/2,1}。

第 m 个 scalar critic 定义模型代价 Ĵ_m。对候选和 incumbent 采用相同 Brownian 样本，估计 d_m(α)=Ĵ_m(θ_α)−Ĵ_m(θ)。可行权重集为

$$\mathcal W=\{w\ge0:\mathbf1^\top w=1,\;\mathrm{KL}(w\|\pi)\le0.3,\;w^\top Cw\le1.1\pi^\top C\pi\},\quad\pi_m=1/5.$$

C 是原 noisy validation residual 的 Gram matrix。选择

$$\alpha_*\in\arg\min_{\alpha\in\{0,1/4,1/2,1\}}\max_{w\in\mathcal W}w^\top d(\alpha).$$

同一个 w 评估候选与 incumbent，避免比较两个不同终点函数。Nominal ablation 固定 w=π。α=0 的 empirical difference 是 0，因此精确求解时选定候选的这个有限样本、有限集合 worst-case difference 不大于 0。

**这个结论仅涉及选定的 empirical model costs。** 它不保证真实 g 落在 ensemble hull 中，不保证梯度误差覆盖，不保证真实 J 下降，也不是由一个新的 HJBI PDE 严格推出的控制律。数值求解还有独立的误差警告。实现为 `value_guard.guard` / `worst_difference`。

该方法的实测负结果也符合这一限制：scalar validation 约束可以排除一些坏 critic，但不能自动识别对控制有害的梯度偏差。所有成员共享偏差时，最坏 mixture 仍可能错误。不能据此把额外守护模拟解释为已经实现了“传播感知”的可靠性估计。

## Estimator ablations

`anchored_am.py` 用 Gaussian RBF 拟合 g 或 NN residual；`smooth_network.py` 替换 ReLU 为 SiLU。二者都改变终点标签来源，复用相同 lean AM/Adam。它们是诊断干预，不是新的最优控制方程，也不是已校准的 uncertainty model。

后续若主张 PAGR 为算法，应另外定义可计算的传播后风险、校准方法、与真实控制目标的关系和独立 seed 实验；当前发布不补写尚未完成的结论。

## 预训练 Flow AM 平台

主入口：[02_flow_am_terminal_gradient.ipynb](../notebooks/02_flow_am_terminal_gradient.ipynb)，完整核心实现直接展开在同名 `.py` 与单元格中。固定一个条件、二维动作、单步终止，`Q_true=R`；不是导航动力学，也没有 TD、moving base 或在线 critic 更新。

理想线性桥为 `X_t=(1-t)Z+tA`（noise→action），`nu²=2(1-t)/t`，base drift `2v_base-X/t`，control `2(v_theta-v_base)/nu`，终端代价 `-beta R`。Actor 完整复制真实 FM 训练选出的 base；冻结权重不阻止 base 的输入 VJP。

实际网格：先用所有方法共享的冻结 base 做 8 步 RK2，从 0 到 epsilon=0.05，再在 `[epsilon,1]` 做 K=32 个 EM Gaussian 转移。所有漂移和 diffusion 使用左端 `t_k`，最后一个左端严格小于 1，故每一步的 covariance `h nu_k² I` 都正定。终端梯度在实际 noisy `X_K` 上计算，没有最后一步去噪或 action clipping。

对冻结路径，`F_k=I+h(2 D_x v_base(X_k,t_k)-I/t_k)`，`p_K=-beta grad R_hat(X_K)`，`p_k=F_k^T p_(k+1)`。局部 matching loss 配对 `(X_k,p_(k+1))`：

```text
L = h/(2B) sum_(i,k) || 2(v_theta-v_base)/nu_k + nu_k p_(k+1) ||²
delta q = 2h/B sum_(i,k) D_theta v_theta(X_k,t_k)^T delta p_(k+1)
```

Paths、terminal labels、base Jacobians、adjoints 全部 detach；仅当前 actor 的局部输出参与参数反传，不重复加 energy，不通过 actor 状态 Jacobian 做 BPTT。改写 velocity target 时必须保留 `2h/nu_k²` 权重。Lean matching 梯度不是一般意义下真实 J 的精确梯度。

有限-chain KL 来自实际 Gaussian 转移均值差：`||m_theta-m_base||²/(2h nu²)=h||u||²/2`。共同初始分布（包含冻结 warmup）在 chain rule 中抵消；因此对本实验定义的全部随机链，这是精确恒等式，期望的样本平均仍有 Monte Carlo 误差。它不等于 ODE action KL，也不包括理想时间零模型中被省略的受控过程。新 J 是该有限链上的 `E[energy-beta R]`。

### 参考与端点差异

已读取 [Adjoint Matching v5 §4/5.2、Algorithm 1、G.1](https://arxiv.org/html/2409.08861v5)、[QAM v3 §3–4](https://arxiv.org/html/2601.14234v3)；并实际通过 GitHub API 读取 [官方 agents/qam.py](https://github.com/ColinQiyangLi/qam/blob/2726d767c9a0a7a46d49693f0391f73dc2cf58ac/agents/qam.py)，commit `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`，`adj_matching`、`actor_loss`、`compute_flow_actions`。

官方代码使用 `sigma_h²=2(1-t+h)/(t+h)`、漂移 `2v-X/(t+h)`，最后一步为 base ODE；其 backward 时间参数也与 forward 不完全相同。本平台采用明确的 next-index 离散伴随，**不是官方代码逐行复现**，不把论文 same-time 或官方其他索引约定预判为 bug。B 的 HTML Eq.5 所示加号与 Eq.7 不一致；本实现使用 Eq.7，并由线性桥 score `(t v-X)/(1-t)` 独立核对漂移符号。

固定 epsilon 的截断起点、学习到而非精确的 FM、有限步数与离散 lean 近似均阻止我们宣称严格 memoryless 或精确 exponential tilt。K/2K 检查只考察该固定-epsilon 数值平台；不能把它解释为 epsilon→0 收敛。ODE 额外从 0 到 1 积分 fine velocity，和 SDE 的 base warmup 并不相同，报告其分布差异而不默认相等。

### Oracle AM 收益空间诊断

Notebook 2 追加章节把终点梯度来源（learned/真实奖励 oracle）和 rollout batch（64/128）显式拆成 2×2；优化器、端点、伴随配对、matching 权重、训练次数及真实目标均不变。训练分支不再靠方法名字符串推断梯度来源。Oracle 只替换终点梯度标签，仍不是完整真实策略梯度。Frozen Flow 是同一最终噪声上的参照。

同更新比较取四臂的 update 300；同轨迹预算比较取 B64/update 300 和 B128/update 150，均为 19200 条，但后者只有一半 Adam 步。每臂保留 update 0/150/300 checkpoint 和共同的4096条 SDE 最终评价路径。真实目标仍是完整有限链 KL 减去 β 倍真实奖励，奖励、energy、J 均按路径保存。

对每个已有训练 seed，令 L/O 分别表示 learned/oracle。J 越小越好时，定义 A_B=J_LB−J_OB、S_L=J_L64−J_L128、S_O=J_O64−J_O128、I=A64−A128=S_L−S_O。奖励使用相反的“越大越好”方向：A_B=R_OB−R_LB、S_L=R_L128−R_L64、S_O=R_O128−R_O64，仍有 I=S_L−S_O。区间仅是三个已查看开发 seed 的描述性配对 t 区间。

在 learned actor 的 0/150/300 和 oracle actor 的 300 checkpoint，固定同一个 actor 与 Adam history，每 seed、每 batch 做32次新训练噪声重复；单次内两种标签使用完全相同的 rollout 状态，B64 是 B128 的前缀。记冻结路径 matching 梯度 qL、qO 与 d=qL−qO。保存高维原始梯度用于复核，只计算协方差迹：

$$\operatorname{trCov}(q_L)=\operatorname{trCov}(q_O)+\operatorname{trCov}(d)+2\operatorname{trCov}(q_O,d).$$

mean(d) 是32次重复的均值估计，不是真正已知的总体 bias；qO 也不是 ∇θJ。对每一对 q 使用独立复制、完全相同的 Adam 一阶/二阶矩和计数器，按正式训练的 norm clip 与 Adam 更新，然后用新的1024条共同评价噪声直接比较 oracle 虚拟一步减 learned 虚拟一步的 J、reward。Monte Carlo SE 从每条路径的配对差计算；批间 SD、MC SE、训练 seed 差异分开报告。正式 actor、optimizer 和全局训练 RNG 不被虚拟步改写。终点投影 qL=cqO+r⊥ 在 qO 接近零时标记未定义，不删掉重复。

K32 actor 分别在 K32/K64 有限链上评价；K64 的 learned/oracle B64 在全部三个 seed 重新训练并在共同 K64 噪声上评价。这只是网格敏感性，不构成 K64 的完整梯度来源×batch 交互实验。所有新结果写入独立 run，原正式 run 和此前负结果不覆盖。
