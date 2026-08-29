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
- **W7-G0 合同冻结（2026-08-29）**：五份 R1–R5 manifest 已进入 `plans/manifests/`，由 `tests/test_w7_gate_manifests.py` 的 3 个结构测试看守；它们冻结后续实现的真实输入/输出、trace、资源预算、checkpoint、red proof、holdout/lesion、失败隔离、rollback 和越界边界。R1/R2/R3/R5 为 `contract_frozen`，R4 明确为 `hardware-blocked`；这不是任何后续能力已完成的声明。冻结后的执行入口曾是 W7-R1 provider watchdog，当前入口以实时路线文件为准。
- **本轮回归边界（2026-08-29）**：S2 相关定向后端测试 `26 passed`，frontend `apiClient` 定向测试 `9 passed`，W7-G0 manifest 测试 `3 passed`，release/package 与最终 packaged-client canary 通过。全量 Python 回归在本机受 Temp/工作区目录 ACL 拒绝阻断（首轮 `481 passed, 6 skipped, 91 errors`，隔离 basetemp 仍出现同类权限错误并在 pytest 收尾失败），因此不把本机全量结果写成 CI 绿灯；当前能力声明仍以此前已记录的 CI/定向 Gate 为准。
- **W7-R1 S0/S1/S2 provider watchdog（2026-08-29）**：健康记录改为以 `artifact_id + artifact_digest` 内容寻址，防止同 ID 内容替换继承旧失败计数；S0 的健康/失败阈值/单次回退/cooldown/native fallback 通过，S1 又用真实 `TSKV8Adapter.native_checkpoint()` 验证 artifact、digest、registry、健康计数和阈值后的下一次探针 lineage 可恢复。S2 在明确 Legacy-off 的 `dist/Seed/Seed.exe` 默认 8000 端口 canary 中验证 backend/runtime/UI 启动、provider 健康字段只读投影、网络端口绑定以及无 Legacy/Transformer/HF 标记；8 个 API 请求均 8000/200，无页面错误或请求失败。S2 仅运行 native-readable 内置器官，外部 provider artifact 轮换未被虚报为已验证。报告见 [S0](../../reports/taiji_w7_r1_provider_watchdog_20260829.json)、[S1](../../reports/taiji_w7_r1_provider_watchdog_s1_20260829.json) 与 [S2](../../reports/taiji_w7_r1_provider_watchdog_s2_20260829.json)，定向 provider 回归 `21 passed`。R1 已完成，随后进入 W7-R2-S0。
- **W7-R2 S0/S1/S2 interaction-group（2026-08-29）**：新增通用 trace-grounded pair evaluator，输入不透明 owner/context、Outcome、恢复效果、资源和 checkpoint revision；按 context 内四格 counterfactual 估计 contribution、正/负 interaction、recovery effect、uncertainty、资源成本和 lesion，训练 digest 不吸收 holdout Outcome。S1 使用真实 `TSKV8Adapter` 生成 `Event/Outcome` 并经 `taiji-native-v1` checkpoint 精确回放；S2 再以真实 `SeedRuntime + WorkbenchEnvironment` 执行 workspace list/search/read，观察 capability snapshot、native executive selection、world evidence 和 recovery retry。16 个 S2 replay record 全部一致，2 个 admitted group、4 个拒绝候选，role label 输入数为 0；跨 revision、holdout 污染、资源压力、source digest 篡改和缺 Workbench 证据均 fail-closed。报告见 [S1](../../reports/taiji_w7_r2_interaction_groups_s1_20260829.json) 与 [S2](../../reports/taiji_w7_r2_interaction_groups_s2_20260829.json)；R2 完成，下一步为 W7-R3-S0 visual/desktop evidence。
- **W7-R3 S0/S1 visual/desktop（2026-08-29）**：恢复生命页五维雷达作为主视觉，Taiji 原生摘要改为紧凑辅助卡片；删除侧边栏重复生命状态脉冲块；“状态依据”从非生命页移除，仅在生命页底部默认折叠；修复托盘“生命状态”先恢复窗口再切换 `#/life`。本轮又修复了 packaged client 的真实空白窗口：PyInstaller 前端改为逐文件打包并校验源码/包内 211 个文件的集合与字节，启用 `--clean`，冻结 Qt 显式绑定 `QtWebEngineProcess.exe`；当前 CPU-only/受限主机默认使用 `--disable-gpu --single-process`，不使用 `--no-sandbox`。重新打包的 Legacy-off/native `dist/Seed/Seed.exe` 在 8148 端口出现 `loadFinished(ok=True)`，实际请求当前 hash 版 JS/CSS、runtime bootstrap/status、聊天和训练接口，health canary 为 ok；前端 `43 files / 245 passed`、Vite build、ESLint 0 errors、桌面契约与项目身份 `13 passed`。证据见 [R3-S1](../../reports/taiji_w7_r3_visual_desktop_s1_20260829.json)；Chrome 未安装，真实 Windows 任务栏/托盘、截图和高 DPI/窄窗口属于尚未完成的 R3-S2。

## 关联资料

- [Taiji 核心目标](../active/TAIJI_CORE_REQUIREMENTS.md)
- [Taiji Native Architecture v1](../active/TAIJI_NATIVE_ARCHITECTURE_V1.md)
- [架构方向决策](../active/ARCHITECTURE_DIRECTION_2026_08.md)
- [Seed 产品与运行时架构](../active/SEED_ARCHITECTURE.md)
- [持续门禁与 CI](../active/roadmap/02_GATES_AND_CI.md)
