# PAGR — Propagation-Aware Gradient Robustification

终点梯度可靠性、信用分配与控制性能的研究记录。

**状态：探索性研究，不是已经验证有效的新算法。** PAGR 是暂定的、与具体框架无关的研究方向名称；当前使用 lean Adjoint Matching（AM）作为可复现试验平台。仓库保留负结果、开发阶段的选择过程和数值审计警告。

> Under a fixed true control objective, terminal-value estimation and terminal-gradient reliability are different questions. This repository studies how terminal-gradient errors propagate through adjoints into actor learning signals. It does **not** establish a generally superior robust-control algorithm or a new HJBI equation.

## 研究问题

在固定真实控制目标下，即使终点代价的数值估计较准确，其梯度误差仍可能通过 adjoint 传播，污染中间时刻的学习信号，降低有限训练预算下的控制性能。我们希望刻画并抑制其中真正有害的部分，同时保留有效的控制信号。

区分三件事：终点数值误差、传播后的标签/actor 更新误差、最终真实控制代价。梯度 disagreement 不是已校准的梯度误差；局部 matching loss 的下降也不是一般意义的真实控制代价下降。

当前 toy 数据来自同一动力学下的自然离线轨迹和带执行噪声的终点观测，**没有为得到正结果而人为叠加高频梯度污染**。二维避障是四维位置–速度状态；另一任务是六关节平面臂的随机位姿生成，不是实物机器人实验。

## 当前结果

固定 3 个开发 seed（18001–18003），每种方法 1000 次 actor 更新；表中是真实评估代价，越低越好。每个最终 actor 用 4096 条评估轨迹，统计单位仍是 **3 个训练 seed**，不是 12288 次独立训练。

| 方法 | Navigation 平均代价 | Arm 平均代价 | 解释 |
|---|---:|---:|---|
| 原始 lean AM + Adam | 19.467734 | 10.973526 | 主基线 |
| AM + Adam，2×训练轨迹 | 19.413852 | 10.809225 | 额外预算对照，不是新算法 |
| Nominal value guard | 19.612436 | 11.095938 | 两任务均 0/3 seed 胜出 |
| Robust value guard | 19.642161 | 11.209811 | 两任务均 0/3 seed 胜出 |
| RBF critic | 19.316217 | 13.045337 | 对任务敏感 |
| Residual critic repair | 19.364881 | 11.017859 | 没有稳定收益 |
| SiLU critic | 20.175479 | 10.688394 | Arm 平均改善 2.60%，但仅 2/3 胜出；Navigation 恶化 |

**没有新的候选方法通过所记录的开发门槛。** 所有 seed 被复用于探索，不能把局部正结果称为独立确认，更不能由此宣称 motivation 已被完整因果验证。详见 [结果与证据边界](docs/results.md)。

![Historical learning curves and arm seed detail; complete method results are in the table above](reports/figures/followup_results.png)

## 安装与快速运行

参考环境为 Python 3.13.11、PyTorch 2.9.1、NumPy 2.3.5、SciPy 1.16.3，CPU、单线程。项目不需要 CUDA、远端服务器、API key 或下载私有数据。

```bash
git clone https://github.com/ruio248/pagr.git
cd pagr
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebook]'
python -m pagr verify
python -m unittest discover -s tests -v
python -m pagr summary
python -m pagr replay --task arm --seed 18001 --method am_adam --output runs/arm_18001_am
```

不做 notebook 时安装 `python -m pip install -e .` 即可。系统需提供 Python 3.13；不必使用 Conda。Windows 用 `.venv\Scripts\Activate.ps1` 激活。Linux CPU-only PyTorch 安装方式、完整 42-run 重放、从零生成离线数据、执行 notebook、跨平台误差说明见 [复现指南](docs/reproduction.md)。

每次训练的输出必须使用新的空目录；默认不覆盖已有结果，也不向包内的参考数据写入。命令行直接重跑冻结设置，不提供暗中更换超参数或选最优测试 checkpoint 的选项。

## 导航

| 内容 | 入口 |
|---|---|
| 完整实验参数、随机种子、预算和指标 | [docs/settings.md](docs/settings.md) |
| 公式、lean adjoint、有限位移 guard 与实现对应 | [docs/methods.md](docs/methods.md) |
| 可运行 notebook：全结果、真实重放、轨迹、机制与警告 | [notebooks/reproduce.ipynb](notebooks/reproduce.ipynb) |
| 历史实验报告及可编译 TeX 源文件 | [reports/lean_am_followup.pdf](reports/lean_am_followup.pdf)、[TeX](reports/lean_am_followup.tex) |
| 全部 42 个逐 seed 结果和最终 actor | [src/pagr/assets/reference/followup](src/pagr/assets/reference/followup) |
| 离线数据、原始/替代 critic | [src/pagr/assets](src/pagr/assets) |
| 当前实现的轻量测试 | [tests/test_research.py](tests/test_research.py) |
| 来源、公开包转换和 SHA256 | [provenance/export_manifest.json](provenance/export_manifest.json) |
| GitHub 设置、隐私边界、许可证状态 | [docs/repository.md](docs/repository.md) |

## 公开范围与许可

这不是整个工作区的备份。只收录当前梯度可靠性研究的必要代码、数据、checkpoint、结果、报告和复现入口；不含 TRQAM、机器人部署项目、SSH/代理配置、聊天附件、个人路径、密钥、虚拟环境和多 GB 的重复 rollout 缓存。

**按作者要求暂不添加许可证，后续决定。** Public 表示可公开查看，不等于已选择 MIT/Apache 等开源许可。本项目没有发布到 PyPI；`pagr-research` 是此仓库的本地安装包名。关于 GitHub 无许可证仓库的说明见 [GitHub licensing documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。
