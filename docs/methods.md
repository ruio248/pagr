# 公式与代码对应

本页区分研究 motivation、现有算法和未被证明的部分。PAGR 是研究方向名，不能将全部 ablation 统称为一个已成功的方法。

## 固定真实目标

设控制与噪声共用矩阵 S，

$$dX_t=(b(X_t,t)+S u_\theta(X_t,t))dt+S\,dW_t,$$

$$J(\theta)=\mathbb E\left[g(X_T)+\int_0^T\left(\ell(X_t)+\frac12\|u_\theta(X_t,t)\|^2\right)dt\right].$$

这里真实 g 和 J 不随方法改变。训练只获得带噪终点代价的离线观测并拟合 ĝ。把估计梯度误差建模为 δ=∇ĝ−∇g，不意味着真实任务的终点代价被对手改变。

## 当前代码的 lean matching

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
