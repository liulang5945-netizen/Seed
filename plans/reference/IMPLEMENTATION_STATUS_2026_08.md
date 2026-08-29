# Seed / Taiji 实现事实参考

> 这是当前代码事实的短摘要，不承担执行顺序。执行顺序只看 [当前路线入口](../active/SEED_DEVELOPMENT_ROADMAP_2026_08.md)，架构合同看 `plans/active/` 根目录的四份文档。

## 所有权

- `taiji/` 不导入 `seed`、`neuroplex` 或 `transformers`；native core 保持独立。
- `seed/` 负责产品、运行时、Workbench 和外部 provider 边界；`neuroplex/` 是冻结的 Legacy Transformer 对照。
- Taiji 产生结构化 `ActionIntent`、`ContentPlan`、`WorldAffordance` 和状态/证据 lineage；语言 provider 只负责把已形成的内容实现为可读表达。
- 工作台工具必须来自当前 content-addressed capability snapshot；前端、prompt 或 provider 不得维护第二份工具表。

## 当前可声明的闭环

- `InputFrame → Observation/PerceptEvent → WorldState/WorldEvent` 的输入边界、来源和 checkpoint lineage 已存在。
- Executive 从认知状态和真实 affordance grounding 选择候选；候选经 Workbench admission、policy、executor 和真实 after-state/Outcome 回写。
- IDE 语言识别综合扩展名、shebang、内容、manifest、邻近文件、LSP/toolchain 证据；高置信时可逆地自动切换，歧义时 `ask_user`。
- 文件 patch/create/rename/delete/undo 和 terminal/diagnostics/test/build 已有结构化预览、审批、原子执行、冲突检测、输出预算和 checkpoint 续跑边界。
- MCP-shaped 本地 registry、有限多步 loop、WorldEvent freshness、successor graph、recovery handoff、branch portfolio 和只读客户端 snapshot 已有合同。
- provider artifact 具备 registry、内容寻址、loader、startup、回滚和客户端观测边界；provider 不能绕过 Taiji 选择工具或伪造认知状态。

## 明确不能宣称

- 当前不是开放域通用智能，不是完整的人脑仿真，也不是“规模扩大就自动进化”。结构成长必须由真实任务误差/容量压力触发，并经过 holdout、lesion、资源预算和 rollback。
- 当前没有 CUDA 实测结论；CPU profile 只能作为参考基线。没有 CUDA-capable 主机前不提交 fused/sparse kernel，也不把 CPU 结果宣传成 CUDA 加速。
- 只读 Workbench canary 不等于写入自治、开放域自然语言工具选择、外部 MCP 生命周期或长程规划完成。
- 小型模拟 Gate 只属于 S0 机制证据，必须逐级升级到 replay/sandbox 和真实 packaged client/workbench。

## 验证基线

最近一次已记录的完整回归为 Python `560 passed, 6 skipped`、前端 `42 files / 237 passed`（2026-08-29 训练 ETA/进度分母修复，见 [02_GATES_AND_CI.md §14.18](../active/roadmap/02_GATES_AND_CI.md)；上一轮为三修提交 `cd39632` 的 `556 passed`，本轮新增 4 例训练进度契约）；同轮实测 native boundary `6 entrypoints PASS`、API contract `45 literals PASS`、Ruff `All checks passed`、核心 mypy `0 errors / 44 files`、ESLint `0 errors`、`npm run build` 通过。本轮未复测覆盖率，上一次记录值为 `45.13%`（对应 `550/233` 那轮），不得当作当前值引用。另有一项本轮实测的运行时基线：**CPU 原生训练吞吐 ≈147 字节/s**（`scripts/archive/diagnostics/diag_eta_rate.py`，窗口比值 1.00~1.05 稳定），此前代码注释里的「≈311 ticks/s」是无出处的过期数字，已作废。`black --check` 本机因缓存目录不可写挂死（见 [02_GATES_AND_CI.md §14.6](../active/roadmap/02_GATES_AND_CI.md)），由 CI 腿覆盖。后续改动不得只更新数字，必须把对应报告、Gate 和失败边界一起保留。

随后两轮实测基线（同日，checkpoint 往返等价性 Gate 与 recovery portfolio 审计 Gate）：

- **checkpoint 往返 Gate（`58976d6`，见 [02_GATES_AND_CI.md §14.19](../active/roadmap/02_GATES_AND_CI.md)）**：Python `563 passed, 6 skipped`（新增 5 例：等价性往返 3 例），核心 mypy `0 errors / 44 files`，前端 `42 files / 237 passed`，API contract `45 literals PASS`。
- **recovery portfolio 审计 Gate（工作树，见 [02_GATES_AND_CI.md §14.20](../active/roadmap/02_GATES_AND_CI.md)）**：Python `568 passed, 6 skipped`（新增审计门禁 5 例），核心 mypy `0 errors / 44 files`，前端 `43 files / 242 passed`（+`RecoveryPortfolioAuditPanel` 5 例），API contract `46 literals PASS`（新增 `/api/workbench/taiji/recovery-branch/context`），native boundary `6 entrypoints PASS`，ESLint `0 errors`，`npm run build` 通过。OpenAPI 基线已按 `--snapshot-update` 重生成。
- **recovery portfolio S2 packaged-client 现场证据（2026-08-29）**：最终 `dist/Seed/Seed.exe` 在 Legacy-off、native runtime、8138 自定义端口和真实 `LOCALAPPDATA` 下启动成功；受限数据根自动降级到 `dist/Seed/user_data`，不需手工 Qt 环境。Workspace、右侧检查器和恢复组合审计面板真实可见，所有关键 API 请求 8138/200，Playwright 无页面错误、无 Legacy/Transformer/HF/GGUF 标记；`runtime/status` 为 `seed:seed_corpus.pt` / `is_taiji=true` / `is_seed=true`，native capability snapshot revision `4`。本次为结构化 `portfolio_empty` 空态，非空 branch/tombstone 仍以 S0/S1 replay 为证据。完整记录见 [packaged_client_s2_20260829.json](../../reports/packaged_client_s2_20260829.json)。

## 关联资料

- [Taiji 核心目标](../active/TAIJI_CORE_REQUIREMENTS.md)
- [Taiji Native Architecture v1](../active/TAIJI_NATIVE_ARCHITECTURE_V1.md)
- [架构方向决策](../active/ARCHITECTURE_DIRECTION_2026_08.md)
- [Seed 产品与运行时架构](../active/SEED_ARCHITECTURE.md)
- [持续门禁与 CI](../active/roadmap/02_GATES_AND_CI.md)
