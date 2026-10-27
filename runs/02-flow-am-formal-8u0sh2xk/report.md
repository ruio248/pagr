# Flow AM execution report (formal)

Seeds: [28001, 28002, 28003]. 9 main fine-tunes + 2 pre-specified first-seed 2K retrains; 3 frozen bases. Fixed final step 300. Runtime 47.1s.

Primary sampler: shared frozen-base RK2 warmup to epsilon, then all-Gaussian EM chain. This is a truncated-start, finite-step approximation, not exact time-zero memoryless AM. The complete implemented-chain KL equals expected conditional Gaussian KL; numerical estimates have MC error. The separate ODE integrates fine velocity from 0 to 1 and has no reported KL.

Oracle minus learned paired differences (positive reward is good; negative cost is good):

- SDE reward: mean 0.012208, seed SD 0.003554; 95% t interval [0.003379884381169918, 0.021035814475634212], favorable seeds 3/3.

- SDE cost: mean 0.000552, seed SD 0.004893; 95% t interval [-0.011602779025240432, 0.012706827519742246], favorable seeds 1/3.

- ODE reward: mean 0.010561, seed SD 0.003529; 95% t interval [0.0017953484410397728, 0.01932688927612064], favorable seeds 3/3.

三 seed 的真实离散 J 差异区间包含零，尚未确认稳定收益空间；这既不是普适证明，也不否定理论动机。

Reward, regularized J and ODE/SDE outcomes must be interpreted separately. Ensemble disagreement is not a calibrated error estimate; all mechanism diagnostics use common learned-actor checkpoints. Virtual Adam steps clone the current optimizer state and use fresh common evaluation noise without changing formal training.

No new candidate, covariance regularizer, damping, guard, adaptive beta or oracle-dependent checkpoint selection was introduced. Learned branches made zero new true reward/gradient queries during updates. The first-seed K/2K retrain is not a three-seed convergence study. Diagnostics query truth only outside the learned-training guard.

Implementation checks: 57/57. Failed executions, if any, remain in their separate events.jsonl; no trajectory clipping or nan_to_num.
