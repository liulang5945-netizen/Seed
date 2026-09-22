# Seed / Taiji 长期工作记忆

> 只留跨轮次仍有效的契约/纪律/判据；细节见 §5 与 `docs/*_RULES.md`。整理 2026-09-22。

## 1 架构契约（机器强制）

- `taiji/` 自足：禁 import seed / seed_platform / neuroplex / transformers（含传递性）；AST 级强制于
  `tests/taiji_native/test_{architecture,naming_boundary}_contract.py`。
- `instruments/` → taiji 单向（仅 `content_digest`）；`taiji/` 对 instruments 零引用。语义 encoder 的
  embedder 必须显式注入（fail closed）。`neuroplex/` 不得 import seed/taiji。
- 红线：`taiji-document-embedder-v1` 的 payload 格式与 digest 锚不得变更。

## 2 环境硬约束

- Python 用 `C:/Users/23747/AppData/Local/Programs/Python/Python312/python.exe`（managed 3.13 无 torch/ruff）。
- **bash 无 ls/cat/grep/head/tail/mkdir/rm**：文件用 Read/Write/Edit/Glob/Grep，目录/过滤用 `python -c`
  或 `| python.exe -c "…"`。**管道里缺失命令会 SIGPIPE 杀掉上游 Python**。
- **反引号在任何 shell 引号里都会被命令替换**（已踩 4 次）⇒ 含反引号/代码片段一律走 Write/Edit。
  严禁 heredoc；`python -c` 内避免长中文。`reg.exe` 被拦截。
- **跑 `tests/` 与任何大批删除的构建步骤必须 `CODEBUDDY_SAFE_DELETE_ENABLED=0`**：批删守卫劫持
  `Path.unlink`/`os.remove` ⇒ 中止或长时间无进展，**极易误诊为磁盘 I/O 卡死**，沙箱开关绕不开。
  全量 pytest ~15 min 会 SIGTERM。读仓库文件的 gate 必须显式传
  `SeedRuntime.load(..., workspace_root=PROJECT_ROOT)`。

## 3 工作流与文档纪律

预注册 → 实现 → 门禁 → 报告 → 提交；负结果**如实落账，绝不改绿**。文档四类：冻结判据/预注册与
冻结证据**只追加**（新结论写新预注册；被覆盖则恢复归档版+另存）；导航/状态文档（03 的"唯一下一步"）
**必须改**；原文写错**改+注明原值**。

- 归属证明用引用图检查；回归测试**双向钉住**；**守卫必须红/绿各跑一次证明能响**（否则是装饰）。
- 测试改完读回全文（Edit 可能匹配错缩进而静默失效）；同一文件多处替换**必须串行**。探针用毕即删。
- 链接前缀 `plans/reference/*`→`../../`、`plans/active/roadmap/*`→`../../../`；改完跑链接校验。
- 正结果先问"相邻设置能否复现"；交互效应声明前做规模扫描。提交信息写 `.git/COMMIT_MSG_*` 再 `-F`。
- ⚠️ **本仓常有并行会话**：结论标取证时刻、提交前重跑 `git log`；别清理未跟踪内容。

## 4 可复用判据

- **边际退化**：不同输入生成同一字节串且等于 train 众数 ⇒ 学到边际分布，前向/机制层解释全不成立。
- **预算分层**：否决只在「对照臂在该预算下已能产出非零 exact」时有效；"改动生效但两臂逐位相同"
  ⇒ 是**预算不足以表达**，不是假设被否决。
- **抖动 vs 系统性**：1/N seeds ⇒ 抖动可放行；N/N 且机制读数退化 ⇒ 系统性不得放行。
- **长跑/批处理**：幂等重跑要用**多臂**测试证明；时刻进文件名换安全字符（`:` 在 Windows 非法）；
  driver 分臂记状态；**报速率报中位窗口不报累计**；空窗口指标写 `null` 不写 `0.0`。
- **`counterfactual._run_scripted` 不能判合同合法性**：绕过 `policy_for` 的拦截。
- **写进代码的预期值必须有可否决通路**（否则是装饰）。
- **仓库卫生**（全文 `docs/REPO_HYGIENE_RULES.md`）：判泄漏比**在位值 vs 历史 blob 指纹**（≠数提交、
  ≠看 ignore）；目录规则不覆盖子文件（命名约定通配+反向钉住）。历史重写用 `git clone --mirror` 镜像
  隔离、**永不**原地做；`filter-repo` 不重写自定义 ref 与远端 `refs/pull/*`。
- **桌面/前端工具链与打包**（全文 `docs/DESKTOP_AUTOMATION_PITFALLS.md`）：Vite/Vitest **不**重写
  `./x.js`→`x.ts`；`ELECTRON_RUN_AS_NODE=1` 让 `electron.exe --version` 打印 Node 版本（像"二进制没装"）；
  NSIS `/D=` 路径在 bash 里**必须加引号**。

## 5 当前状态与归档索引

- **产品默认基座 `checkpoints/seed_beta.pt`**（16M tick，来源登记 v2）；DEBT-I9 未结项。H 阈值绑
  `(设备,链路,checkpoint)`。**M5 限定退出已获批准**（2026-09-20），**R2 语言能力是其显式排除项**。
- **R2 执行中：受控重训语言读出**（合同 `plans/reference/M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md`）：
  A/B 已跑满（写入面**实测** `predictive_readout` / `motor`），C 续跑中；判决器（M1/M2/K2）就位。
  `checkpoints/*.pt` 只读、写靶隔离。
- **产品侧工程支线**（不入研究主线）：前端 TS 地基 + Electron 壳 + 打包链路，见
  `plans/reference/FRONTEND_TS_VS_HARNESS_ADOPTION_DECISION_BRIEF_20260921.md`。
- 回退备份 `E:/Seed-backup-{git,secrets}-20260919`；未确认前别跑 `git gc`/`prune`。
  R2 v1–v6 判停、D1–D8 结项见同一目录。
