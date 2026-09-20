# Seed / Taiji 长期工作记忆

> 只留跨轮次仍有效的契约、纪律与判据；归档细节见 §5。整理：2026-09-20。

## 1 架构契约（机器强制，不得放宽凑绿）

- `taiji/` 自足：禁 import seed / seed_platform / neuroplex / transformers（含传递性）；AST 级强制于
  `tests/taiji_native/test_architecture_contract.py`、`test_naming_boundary_contract.py`。
- `instruments/` → taiji 单向（仅 `content_digest`）；`taiji/` 对 instruments 零引用。语义 encoder 的
  embedder 必须显式注入（fail closed）。`neuroplex/` 不得 import seed/taiji。
- 红线：`taiji-document-embedder-v1` 的 payload 格式与 digest 锚不得变更。

## 2 环境硬约束

- Python 用 `C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe`（managed 3.13 无 torch/ruff）。
- **bash 无 ls/cat/grep/head/tail/mkdir/rm**：文件用 Read/Write/Edit/Glob/Grep，目录/过滤用 `python -c`
  或 `| python.exe -c "…"`。**管道里缺失命令会 SIGPIPE 杀掉上游 Python**（白跑过 9 分钟训练）。
- **反引号在任何 shell 引号里都会被命令替换**（已踩 4 次）⇒ 含反引号/代码片段一律走 Write/Edit。
  严禁 heredoc；`python -c` 内避免长中文。`reg.exe` 被安全策略拦截。
- 跑 `tests/` 必须 `CODEBUDDY_SAFE_DELETE_ENABLED=0`（批删守卫劫持 `Path.unlink`/`os.remove` ⇒ 大批
  gate 假失败；`dangerouslyDisableSandbox` 绕不开）。全量 pytest ~15 min 会 SIGTERM ⇒ 后台或分批。
- 读仓库文件的 gate 必须显式传 `SeedRuntime.load(..., workspace_root=PROJECT_ROOT)`（不继承 override）。

## 3 工作流与文档纪律

流程：预注册 → 实现 → 门禁 → 报告 → 提交；负结果**如实落账，绝不改绿**。文档分四类：
冻结判据/预注册**只追加**（新结论写新预注册）；冻结证据**只追加**（被覆盖则恢复归档版 + 另存新版）；
导航/状态文档（03 的"唯一下一步"）**必须改**；原文写错**改 + 注明原值**。

- 归属证明用引用图检查；回归测试**双向钉住**；**守卫必须红/绿各跑一次证明能响**（否则是装饰）。
- 测试改完读回全文（Edit 可能匹配错缩进而静默失效）；同一文件多处替换**必须串行**。临时探针用毕即删。
- 链接前缀：`plans/reference/*` → `../../`；`plans/active/roadmap/*` → `../../../`；改完跑链接校验。
- 正结果先问"相邻设置能否复现"；交互效应声明前必须做规模扫描。提交信息写 `.git/COMMIT_MSG_*` 再 `-F`。
- ⚠️ 本仓可能有**并行会话**：审计结论标取证时刻、提交前重跑；别清理"以为是垃圾"的未跟踪内容。

## 4 可复用判据

- **边际退化**：不同输入生成同一字节串且等于 train 众数 ⇒ 学到边际分布，前向/机制层解释全不成立。
- **序列级 ≠ 位置级**：序列级依赖建立 ≠ 位置级条件化成立（TF 首错常集中 `pos_0`）。
- **预算分层**：否决只在「对照臂在该预算下已能产出非零 exact」时有效；"改动生效但两臂逐位相同"
  ⇒ 是**预算不足以表达**，不是假设被否决。
- **抖动 vs 系统性**：1/N seeds ⇒ 抖动可放行；N/N 且机制读数退化 ⇒ 系统性不得放行。
- **`counterfactual._run_scripted` 不能判合同合法性**：直调 `execute_tool`，绕过 `policy_for` 的拦截。
- **写进代码的预期值必须有可否决通路**。
- **仓库卫生**（全文见 `docs/REPO_HYGIENE_RULES.md`，守卫 `tests/test_repo_secret_guard.py`）：
  判泄漏比**在位值 vs 历史 blob 指纹**（不是数提交、不是看 ignore）；多路径合并查询不能逐路径归因；
  目录规则不覆盖子文件（按命名约定通配 + 反向钉住）。历史重写用 `git clone --mirror` 镜像隔离、
  **永不**在主仓库原地做；`filter-repo` 不重写自定义 ref 与远端 `refs/pull/*`；改完重建 commit-graph，
  旧 sha 全部失效。

## 5 当前状态与归档索引

- **产品默认基座 `checkpoints/seed_beta.pt`**（16M tick，trainer=`train_seed_corpus`，来源登记 v2）；
  DEBT-I9 未结项。H 阈值已在新底重标并冻结（绑 设备/链路/checkpoint）。**M5 限定退出已获批准**
  （2026-09-20），**R2 语言能力是其显式排除项**。
- **R2 在办：受控重训语言读出合同草案 v0**（`plans/reference/M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md`，
  臂 A/B + 反事实 C）。**步骤 0 已执行**：CPU 下 A 0.0023630 / B 0.0031833 / C 0.0020120 s·tick⁻¹，
  峰值 276.152 MB；提出 **N ≤ 16M/臂 ≈ 33.6 h**；**两次批准（§5 数值线、§4 预算）未申请** ⇒ 不授权训练。
  前置：先证明能落盘 checkpoint 并在新进程恢复；写靶一律隔离，`checkpoints/*.pt` 只读。
- R2 内容绑定 v1–v6 全判停；D1–D8 机制轴结项（`plans/reference/M5_R2_*`）。
- 回退备份 `E:/Seed-backup-{git,secrets}-20260919`；未确认前别跑 `git gc`/`prune`。
