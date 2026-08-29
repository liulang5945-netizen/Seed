# Seed / Taiji 当前执行状态

> 本文件是当前执行状态的唯一实时摘要。详细历史记录按日期保存在 `plans/archive/history/`，不从历史文档恢复下一步。

## 2026-08-29 状态快照

### 已闭合的执行层

- W0–W3 的 native Workbench 版本合同、真实 workspace 只读/受控写入、语言证据与 IDE 自主切换、终端、审批、MCP registry、有限循环、checkpoint continuation 和真实跨文件任务闭环已具备证据。
- W4 的 HF/GGUF/Transformer/Legacy 产品语义清理已完成；合法的 provider 只负责语言 realization，不选择工具、不拥有 Taiji cognition。
- W5 的客户端真实性接入已完成；前端以 native capability、runtime/provider/homeostasis/training/knowledge evidence 为状态来源，旧 Legacy 调用不能回流。其中 homeostasis 一路直到 2026-08-29（提交 `cd39632`）才真正接通：此前 `taiji/adapter.py` 没有 homeostatic 读访问器，`api/models_runtime.py` 的 `LifeNeedsPayload` 又给四个需求字段各设了 `default: 50.0`，于是空 `needs: {}` 在传输层被补成四个编造值、原生 `stress` 被静默丢弃——**客户端显示「已接入原生」而数据全是假的，且门禁全绿**。现已改为 `dict[str, float] = {}`（缺测就不出现，不再编造），并由 `tests/seed/test_native_life_status.py` 的反编造断言看守。
- W6 的 typed native facade 和产品页拆分已完成至 Settings 共享面板收口；组件不越权持有 native API 副作用，前端回归保持可见。
- current Gate（recovery portfolio 客户端审计回放）的 S0/S1 已在代码层闭合，S2 packaged-client 现场取证也已完成：只读绑定键 `GET /taiji/recovery-branch/context`、结构化错误码、`RecoveryPortfolioAuditPanel`（右栏属性检查器，事件投影驱动，stale-keep-last / 切 loop 清空 / 只读）在最终 Legacy-off 客户端真实 Workspace 路径可见。客户端实际观察到 native checkpoint `seed:seed_corpus.pt` 和 capability snapshot revision `4`；所有关键 API 请求 8138/200，无页面错误或 Legacy 标记。详见 [02_GATES_AND_CI.md §14.20](02_GATES_AND_CI.md) 与 [S2 证据](../../../reports/packaged_client_s2_20260829.json)。
- W7-R1 provider watchdog 的 S0/S1/S2 已闭合：健康状态按 `artifact_id + artifact_digest` 隔离，native adapter checkpoint 可恢复健康 lineage；Legacy-off 的默认打包客户端能以只读方式投影 provider 状态、artifact digest 字段和结构化健康计数。S2 当前运行的是 `native-readable`（artifact digest 为空表示原生内置器官，不冒充外部内容寻址 artifact），外部 provider 轮换未在客户端 canary 中宣称完成。详见 [02_GATES_AND_CI.md §14.24](02_GATES_AND_CI.md) 与 [S2 证据](../../../reports/taiji_w7_r1_provider_watchdog_s2_20260829.json)。
- W7-R2 interaction-group 的 S0/S1/S2 已通过：S0 以不透明 `owner_id` 和真实 task context 的四格 trace 作为输入，S1 把真实 `TSKV8Adapter` 的 `Event/Outcome` 投影并逐 episode 精确回放，S2 再通过真实 `SeedRuntime + WorkbenchEnvironment` 执行 workspace list/search/read、native executive selection、world evidence 和 recovery retry。三层均保持 holdout 只读、lesion/失败事件可追溯、trace digest 与 checkpoint revision/owner-policy lineage 可恢复；跨上下文混淆、holdout 反向污染、资源超限、checkpoint digest 篡改和无 Workbench 证据均 fail-closed。没有预设神经元角色，也没有写回 executive 或 provider。详见 [02_GATES_AND_CI.md §14.25–14.27](02_GATES_AND_CI.md)、[S0 证据](../../../reports/taiji_w7_r2_interaction_groups_20260829.json)、[S1 证据](../../../reports/taiji_w7_r2_interaction_groups_s1_20260829.json) 与 [S2 证据](../../../reports/taiji_w7_r2_interaction_groups_s2_20260829.json)。
- P6 provider artifact、provider startup、客户端观测和训练/回滚合同已接通；P7 executive、grounding、world evidence、bounded successor graph 和 recovery portfolio 已形成可恢复只读闭环。
- W7-R3-S1 已完成重新验证：前端打包由只比较 `index.html` 收紧为 211 个文件的集合与字节一致性，PyInstaller 使用 `--clean`，冻结 Qt 显式绑定 `QtWebEngineProcess.exe`。当前受限 CPU-only 主机的 QWebEngine 多进程路径会卡在根 HTML，因此默认桌面 shell 使用 `--disable-gpu --single-process`，未使用 `--no-sandbox`。新包在 8148 端口真实启动并记录 `loadFinished(ok=True)`，实际加载当前 hash 版 JS/CSS、runtime bootstrap/status、聊天和训练接口；这证明客户端不再出现 health 绿但界面空白的启动假绿，不等于 R3-S2 的真实 Windows 任务栏/托盘/DPI 现场证据已完成。

### 必须保持的边界

- 当前默认自主执行只覆盖 Taiji-owned、freshness-valid、受能力快照约束的只读 Workbench 路径；写入自治、开放域自然语言工具选择和外部 MCP 生命周期仍未宣称完成。
- recovery portfolio 审计 Gate 的 S2 packaged-client 现场取证已完成；本次启动展示的是无持久化 portfolio 的结构化空态，非空 branch/tombstone 排序仍由 S0/S1 replay 证据覆盖，不把空态 canary 宣称为非空恢复演示。最终客户端同时修复了 Qt 无 GPU 启动降级、真实后端端口透传、受限数据目录的非阻塞降级和 `/api/health.taiji_available` 状态不一致。
- interaction-group、视觉/桌面体验、CUDA、开放域学习和结构自进化没有取消，只能按 W7 顺序推进；CUDA 在当前 CPU-only 主机上保持 `hardware-blocked`。
- W7-R3 视觉层的两处渲染缺陷已修复并收敛（不改变任何运行时语义，不新增前端认知状态）：训练页 `.tk-card h3` 内联 `<svg class="ic">` 此前在组件与全局样式中都无尺寸规则，按 SVG 默认 300×150 渲染并在 flex 标题内把「检查点列表」挤成换行，现由 `TrainingOverviewPanel.vue` / `TrainingView.vue` 统一的 `.tk-card h3 .ic` 规则约束，同时删除 `TrainingView.vue` 里同目的的内联 `style` 硬补丁；生命状态页 `NeedsPentagram.vue` 的数据面此前用 6% 透明度的 `--primary-subtle` 填充加灰色 `--ink-muted` 描边，与同页 `--chart-*` 渐变面板不一致，现改为 `--chart-1 → --chart-3` 径向渐变加 `--chart-2` 描边与描边式顶点，仍全部走主题 token 以兼容五套 `data-theme`。顺带修掉两个既存失效项：数据面由 `<polygon>` 改 `<path>` 使 `transition: d` 真正生效，`critical` 半径由 CSS 几何覆写改为模板绑定以兼容旧 WebView；`polygon.pentagram-guide`×5 / `circle.pentagram-dot`×5 / `text.pentagram-label`×5 / `.ckpt-item button` 等被测试锁定的选择器保持不变，前端 43 文件 245 例回归、ESLint 与 build 全绿。
- W7-R5-S0 已通过：生产 Taiji Workbench 选择路径在真实只读执行后，将执行前保存的 source affordance 与感知/世界上下文绑定到真实 `Outcome`，再调用 `record_executive_outcome()`；intent、当前 executive decision 和 source affordance 任一不一致都会 fail-closed。`learn=False` 保持评测冻结，`learn=True` 时 `online_updates` 增长；checkpoint 往返保留 `fit_updates`/`online_updates` 并恢复同一最后选择。证据见 [R5-S0](../../../reports/taiji_w7_r5_s0_learning_channel_20260829.json)。
- 训练前必须先验证 checkpoint 能保存、恢复并继续产生等价的 lineage、预算、结构和 provider artifact 状态；任何只在内存中成立的训练结果不算 Gate 证据。该往返等价性准入已由 [04_EXECUTION_PLAN.md §3](04_EXECUTION_PLAN.md) 的 `test_checkpoint_roundtrip_contract.py`（3 例）满足。
- **W7-R3-S2 页面级取证已完成，Windows shell 取证仍 `tool-blocked`**：用户已安装 Chrome，最终包在生命页与 900px/760px 窄布局下完成真实页面证据；响应式修复解决了顶栏裁切和主体挤压。Windows Computer Use 能枚举唯一 Seed 窗口，但无法激活窗口，返回 `failed to activate captured window`，截图落到桌面背景，因此任务栏、托盘、通知和高 DPI 不作通过声明。证据见 [R3-S2](../../../reports/taiji_w7_r3_visual_desktop_s2_20260829.json)。
- 内化（把 skill/mcp 这类外挂知识学进权重后再删除）与效应器身体成长属于 W7-R5 范围。已冻结的 [taiji_w7_r5_open_domain_growth_v1.json](../../manifests/taiji_w7_r5_open_domain_growth_v1.json) 只覆盖开放域结构成长（`status: contract_frozen`，`implementation.status: not_started`），**不覆盖内化转换器与可注册效应器**；后两者需要一份新的、尚未创建的 R5 manifest。在该新 manifest 冻结前不得宣称任何内化或自注册能力已具备。

## 当前唯一下一步

W7-R5-S0 已完成。事实依据：`api/seed_runtime.py` 在选择并准入 Taiji-owned 只读 Workbench candidate 后，执行真实 Workbench 环境，保存执行前的 source affordance 与 affordance context；构造 `WorkbenchTaijiEvidence`、提交 `record_world_event` 后，在 `learn=True` 时调用 `record_executive_outcome`。调用前严格校验当前 decision、intent_id 和 source_affordance_id，任一不一致即 fail-closed；`learn=False` 不产生学习副作用，供评测和回放冻结使用。定向回归与 checkpoint 往返证据见 [R5-S0](../../../reports/taiji_w7_r5_s0_learning_channel_20260829.json)。

明确非本切片：不写 `taiji/internalization.py` 转换器、不做可删性判据、不动 `seed_platform/workbench.py` L447 的 read-only 投影限制、不引入效应器注册表。

## 后续唯一顺序

1. W7-R1 / W7-R2 的 S0/S1/S2（已完成）。
2. **W7-R3-S2（当前唯一下一步）**：在能够激活并捕获 Seed 窗口的 Windows Computer Use 会话中，沿同一新包补齐真实窗口、任务栏、托盘通知和高 DPI 证据；页面级窄布局证据已完成。
3. W7-R5-S0 前置切片（已完成）：`record_executive_outcome` 已接通并通过 checkpoint 往返。
4. 冻结 R5 内化与效应器成长 manifest，再按 [04_EXECUTION_PLAN.md §9](04_EXECUTION_PLAN.md) 的 R5-A（知识内化转换器 + 可删性判据）与 R5-B（效应器注册表 L0→L4）推进。
5. W7-R4 CUDA 在获得真实 CUDA 主机前保持 `hardware-blocked`，不用 CPU 结果替代。

每个方向仍是先做可证伪 Gate，再接入真实运行时，最后才更新产品展示。

## 更新规则

- 当前快照只保留已验证的能力和明确的未完成边界；历史数字、实验过程和一次性失败原因移入 archive。
- 新能力必须在这里留下“owner、真实输入、结构化输出、checkpoint 归属、失败模式和 Gate”；否则只能算实验记录。
- 若实现与本文件、架构文档或 CI 事实冲突，先暂停下一步，修正唯一事实源并提交，再恢复执行。
- W7-G0 的五份 manifest 与结构门禁已提交；R4 当前硬件状态仍为 `hardware-blocked`，不可用 CPU 结果替代 CUDA 证据。
