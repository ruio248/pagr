# GitHub 与公开包设置

## 仓库定位

- 仓库：`ruio248/pagr`，个人账号，**Public**，默认分支 `main`。
- 暂定全名：Propagation-Aware Gradient Robustification。AM 是当前测试平台，不限定未来只能使用 AM。
- 当前包版本：`0.1.0a1`，研究原型；未发布 PyPI package、GitHub Release 或网站。
- **不添加许可证**，作者后续决定；没有 MIT/Apache 等授权声明。公开可见和授予开源许可证是不同操作。
- Git commit 使用 GitHub noreply 邮箱，不公开个人邮件地址。

## 协作与合并

Issues 开启；Wiki、Projects、Discussions 关闭。允许 squash merge，关闭 merge commit / rebase merge；合并后删除 PR head branch。暂未配置分支保护/ruleset 或强制 PR 审查，作者可直接更新 `main`；这也意味着 CI 并非强制合并门槛，未来多人协作可另行开启。

仓库不带组织成员邀请、外部 webhook、GitHub Pages 部署或任何部署密钥。没有导入其他工作区目录的 Git 历史；这是经过筛选的新初始发布。

## CI 与安全

`.github/workflows/ci.yml` 在 main push、pull request、手动触发时执行：

1. Ubuntu 24.04 / Python 3.13.11 / PyTorch CPU。
2. 从非 editable wheel 安装项目，验证 bundled 资产和实现测试。
3. 完整重放一条 Arm seed 18001 原生 AM/Adam（1000 updates），验证所有最终指标有限并报告与参考误差。

不会自动运行整个 42-run suite，不运行远端 GPU 任务，不上传原始数据到第三方。不设置 Actions secret，也不使用 `pull_request_target`。GITHUB_TOKEN 默认只读，不能批准 PR；workflow 的 `contents` 也是 read。官方 checkout/setup-python action 固定 commit SHA，checkout 不保留凭据。

CI 是软件 smoke test，不验证 novelty、因果结论或跨平台数值完全相同。历史研究审计 `passed=false` 和 6 项 solver warnings 被明确测试保留。

发布时 GitHub secret scanning 和 push protection 开启；Dependabot 自动安全更新关闭（不自动改依赖或发版本升级 PR）。Actions 仓库级 allowlist 保持平台默认 `all`，SHA pinning 未在仓库级强制，但当前 workflow 的两项 action 均已手动固定 SHA；后续修改 workflow 时仍需遵循这一要求。

机器可读的初始设置回读见 `provenance/github_settings.json`；它是发布时快照而非永远不变的保证。设置变动后请更新此页和快照。

## 包含与排除

包含：curated 实验 source、公共 CLI 包装层、全部最新 42-run 结果与最终 actor、自然离线数据、原始及替代 critic、冻结配置、历史比较摘要、可执行 notebook、历史 PDF/TeX 和图、来源 hash、可复现检查。

排除：TRQAM、机器人部署工程、个人聊天及粘贴附件、远端地址与代理/SSH 凭据、虚拟环境、用户配置、无关工作区内容、多 GB rollout 缓存。`runs/`、`.venv/`、`tmp/`、`dist/`、密钥样式文件均在 `.gitignore`。

发布前检查文本、notebook 输出、PDF 文本/元数据、tensor archive 的序列化元数据及 PNG 元数据，筛查常见 token/密钥/个人绝对路径。这是有界静态检查，不宣称能检测所有类型的秘密。

## 来源与最小变更

`provenance/export_manifest.json` 逐文件记录相对原工作区路径、原始 SHA256、发布 SHA256 和转换：

- 原训练源文件在原位置不修改；公开副本仅将 tensor 读取改为 `weights_only=True`。
- JSON 中作者机器的绝对路径脱敏，数值结果不改。
- TeX 图路径改为相对路径；既有 PDF 字节保持不变。
- 公共运行包装层单独放在 `src/pagr/runtime.py`，避免依赖原工作区目录。

PDF 为 2026-10-25 的历史记录，其中提到的是当时 notebook 的 robust-guard/SiLU 示范。公开版 notebook 是新整理的可移植入口，真实重跑 baseline/SiLU，并展示全部方法；它不伪装为原 notebook 的字节副本。历史报告中“原源文件不变”指原工作区，公开副本的安全加载修改见 manifest。

未添加许可证不是一个开源许可证选择。GitHub 相关说明见 [Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。
