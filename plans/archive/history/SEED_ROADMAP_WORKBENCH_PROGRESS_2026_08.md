# Seed / Taiji 路线执行记录：Workbench Closure 进展

> 本文由原总路线图按职责拆分而来。原始行号：1870–2238；本文件属于当前计划资料或历史证据，具体身份以本文件所在目录为准。
> 这是已完成 slice 与历史下一步的连续记录；最后一条状态已提炼到当前执行入口。


- 不删除 P1–P7 核心架构讨论、requirements、native architecture 和本路线；它们仍是后续开发的实时依据。
- 测试过程日志、一次性调试探针、已被后续 Gate 覆盖的执行记录可继续进入 archive；核心决策和未关闭缺口不归档。
- W0 已闭合（2026-08-29）；在 W1 之前不增加写文件自治、终端自治、MCP 或新的研究 Gate。interaction-group、provider watchdog、
  CUDA、视觉美化、Legacy Agent 和新格式支持仍完整保留在 W7/后续边界，不是删除；小型模拟 Gate 继续作为各阶段的 S0 验证工具。
- W0 首批实现已落地（2026-08-29）：Seed 已拥有版本化 workbench 合同、内容寻址 capability snapshot、只读
  `WorkbenchEnvironment`、core router、runtime capability projection、`planned → policy → executing → outcome` 审计链，且
  `ActionIntent → ToolCall` 的结构化工具路径已与 motor-symbol `settle_action` 解耦，工作台摘要感知值遵守 Taiji byte sensor 值域。
- W0 前端投影已接线（2026-08-29）：`WorkspaceView` 按 native capability 懒加载目录、`MonacoEditor` 通过 native read 打开文件，
  页面显示 snapshot/最近 outcome，`editor.open` outcome 可由统一 audit projection 驱动 IDE 打开；旧 Legacy 写入、终端、重命名仍未被
  冒充为 native。
- W0 checkpoint continuation 已通过：checkpoint round-trip 恢复 capability snapshot、workbench audit 和 tick，审计阶段保持
  `planned → policy → executing → outcome`；失效 snapshot、越界路径、错误 sensor 值域和断开环境均 fail-closed。
- W0 packaged-client/real-workbench S2 canary 已通过：`SEED_ENABLE_LEGACY=0` 下真实 `dist/Seed/Seed.exe` 成功启动后端，
  `/api/workbench/capabilities` 与 `/api/workbench/files?path=.` 均返回 200，native workspace bytes 与 capability snapshot 可读；
  打包期间发现的 Qt6/ICU DLL 冲突已在 `desktop/seed.spec` 过滤，并纳入 release 检查。该证据确认打包客户端启动和 native route
  可用，不等同于 W5 的 GUI 视觉、DPI、托盘或人工点击验收。

**W1 语言识别、选择与 IDE 自主切换退出 Gate 已通过（2026-08-29）。** `ProgrammingLanguageRegistry` 以内容寻址规则统一
扩展名、shebang、内容、manifest、邻近文件、可选 LSP 与 toolchain 证据；`programming_language_id/editor_language_id`、
confidence、provenance、registry revision、explanation 与 user override 已进入 Workbench state。`editor.set_language` 已成为
Taiji-native 可逆 action：高置信且与证据一致时允许 Taiji 自动切换，低置信、语言冲突或 `.h` 等歧义场景返回 `ask_user`；
runner/LSP 上下文和可用工具链快照与同一语言选择绑定，显式用户覆盖可撤销并按文件 digest 失效，checkpoint 不会把旧覆盖
错误应用到新内容，Monaco 已提供“自动检测”入口。holdout 覆盖 `.h`、无扩展 shebang、多语言 monorepo、Vue/TS、notebook、
markdown code block、错误扩展名和 filename-only lesion；API/OpenAPI、runtime/checkpoint 与 Monaco 动态 projection 已接通，
后端 Workbench 合同 `8 passed`、Monaco 回归 `10 passed`，前端 lint `0 errors`、构建通过。该 Gate 是语言/IDE 合同闭环，
不代表 W2 runner 已可执行。

**已完成（2026-08-29）：W2 首批受控执行合同与 executor。** native capability snapshot 升至 revision 3，新增
`workspace.apply_patch/create/rename/delete/undo` 与 `terminal.run`。文件修改已收敛为 UTF-8 结构化 text-replace 和统一
transaction：before/after SHA-256、原子写入、冲突 fail-closed、唯一单次 undo token；create/rename/delete 共享同一撤销模型。
终端执行已收敛为 argv-only、明确 `shell=False`、workspace 内 cwd、bounded timeout/output、env allowlist 与 expected artifacts，
非零退出和超时都会产生失败 outcome；runtime 会保留 executor 返回的真实 transaction payload，旧只读能力保持兼容投影。
文件事务、终端边界、审批策略与失败结果回归共 `11 passed`，ruff、Black、py_compile 和 diff check 通过。该 slice 只完成
contract/executor，不把直接 executor 调用等同于产品自治：写入和终端仍由 policy 默认返回 `ask_user`，未完成 IDE 预览/审批、
真实 diagnostics/test/build outcome 回写以及 checkpoint 续跑 Gate。

**已完成（2026-08-29）：W2 第二 slice 的审批、预览与真实 outcome 闭环。** `/api/workbench/preview` 对精确 action request
做不落盘验证并生成短期一次性 approval token；`/api/workbench/execute` 只有携带同一请求绑定的 token 才能执行高风险写入/终端，
重放、过期、参数或 snapshot 漂移均 fail-closed，审计请求只记录 approval presence。`terminal.run` 已增加 command/diagnostics/test/build
execution kind、结构化 diagnostics、expected artifacts、after-state，并按 timeout、exit code、诊断错误和缺失产物综合计算 success；
runtime 和通用前端 projection 已接入 preview/execute client。后端 Workbench 合同 `12 passed`、前端完整回归 `187 passed`、构建通过，
ruff/Black/py_compile/diff check 通过。该 slice 完成的是审批/结果合同，不等于 checkpoint 后 undo/approval 状态和真实临时项目续跑已通过。

**已完成（2026-08-29）：W2 退出 Gate 的 checkpoint 续跑与真实临时项目闭环。** transaction state 随 SeedRuntime checkpoint
保存并恢复 undo lineage，approval token 明确为 session-scoped、重启后失效；恢复后重新预览/审批可完成撤销。临时多文件项目已完成
语言识别→patch 预览/执行→test 产物→diagnostics 失败回写链路，预览无副作用，冲突/输出洪泛/cwd 漂移/超时/进程中断均有
fail-closed 证据；旧 `runtime_service` 边界测试同步到 native capability 事实。Seed/native 回归 `320 passed, 1 skipped`，W2
退出 Gate 通过，具备进入 W3 的证据。

**已完成（2026-08-29）：W3 第一纵切片的 native MCP registry 与 canary Gate。** 新增 Seed-owned `McpToolRegistry`，以内容寻址
registry snapshot、版本化 input schema、source/risk/timeout/output budget 和 registry revision 统一 MCP-shaped 工具合同；native
Workbench 新增 `mcp.list/invoke`，仅接入无安装、无网络副作用的本地 `workspace-summary` canary。参数 schema、registry revision、
未知/禁用工具、动态风险审批和输出超限均经过 Workbench policy/executor，失败时 fail-closed 并保留 outcome；API 的 capabilities 与
`/api/workbench/mcp`、前端 `mcpRegistry` projection 已接通。该 slice 没有接回 Legacy `mcp_manager`，不等于外部 MCP 生命周期管理、
真实远端服务连接或多步有限自治循环已完成。Workbench 定向回归 `18 passed`，前端 `187 passed`、构建通过、ESLint `0 errors/17 warnings`。

**已完成（2026-08-29）：W3 第二 slice 的 MCP identity、checkpoint 与 loop preflight Gate。** MCP registry 内容身份已随 runtime
checkpoint 保存/恢复；单次 Workbench request、Outcome 和返回 ToolCall 共享 capability snapshot/registry snapshot binding，审批
摘要也纳入 registry identity。`/api/workbench/loop/preflight` 与前端 `preflightLoop` 已接通，loop 只做不执行 admission，强制最多
8 步、总预算不超过 32 units、拒绝重复调用、首错终止和 `after_each_step` checkpoint 边界。Workbench 定向回归 `19 passed`，
Seed/native 全量回归 `320 passed, 1 skipped`，前端 `187 passed`、构建通过、ESLint `0 errors/17 warnings`。该 slice 不等于真正的
多步执行、逐步 checkpoint 提交或外部 MCP 生命周期管理。

**已完成（2026-08-29）：W3 第三 slice 的受预检有限多步执行 Gate。** 新增 `/api/workbench/loop/execute` 与前端 `executeLoop`，
只接受 preflight identity 未漂移的 native Workbench request；每个已尝试步骤都真实执行、写入 ToolCall/Outcome audit 并保存 checkpoint，
遇到失败立即停止并保留已完成前缀，恢复后重放已提交 request 会 fail-closed。真实成功两步、失败停机和 checkpoint 恢复重放定向回归
`21 passed`，Seed/native 全量回归 `320 passed, 1 skipped`，前端 `187 passed`、构建通过、ESLint `0 errors/17 warnings`。该 slice 尚未
扩展到跨文件 patch/test/diagnostics 任务，也未接入外部 MCP 生命周期。

**已完成（2026-08-29）：W3 退出 Gate 的真实跨文件代码任务 loop。** 在现有 preflight/execute/checkpoint 约束内，真实临时项目已完成
语言识别→跨文件 patch→test/build 产物→diagnostics 失败→checkpoint 恢复→创建修复标记→diagnostics 重试；失败后只从未提交步骤继续，
已提交 request 的旧审批令牌即使失效也会先被 checkpoint 提交历史拒绝，避免误报为普通审批失败或重复副作用。去掉 Taiji planner 或
WorkbenchEnvironment 任一侧均 fail-closed，仍不接外部安装、网络服务或开放式自治。Workbench 定向回归 `22 passed`，Seed/native 全量
回归 `320 passed, 1 skipped`，前端 `187 passed`、生产构建通过、ESLint `0 errors/17 warnings`。

**已完成（2026-08-29）：W4 第一 slice 的正式产品语义残留清理。** 前端已移除 GGUF 导出、旧模型发布、Agent 参数配置、Cortex
认知主体切换和旧 Agent 日志筛选；能力页只展示 Seed-owned native capability registry，不再提供 Legacy MCP 安装/市场管理入口；聊天请求
不再携带 `engine/agent_max_iterations/agent_temperature`，生命状态页不再调用历史 `/api/life/*` 动作接口，训练 composable 的旧生命动作、
发布和 GGUF 死路径已删除。合法的 native MCP projection、语言 provider artifact 和离线 benchmark 未被误删，仍保留在各自边界内。
前端新增原生能力与无历史生命接口回归，完整 Vitest 为 `22 files / 188 passed`，ESLint 为 `0 errors / 15 warnings`，生产构建通过。

**已完成（2026-08-29）：W4 第二 slice 的后端 artifact/settings/OpenAPI 边界迁移。** 新增平台统一 artifact 词表
`taiji_checkpoint`、`language_provider_artifact`、`legacy_benchmark_artifact` 及 `/api/artifacts` 原生清单；新增
`/api/runtime/activate` 和 `/api/settings/runtime`，桌面 Seed 激活不再调用全局 model switch。设置 schema 升至 v2：旧
`model_type/model_name/gguf_path/LoRA/量化` 字段不会被猜测激活，明确的 `self/seed + 安全 checkpoint` 仅迁移为 Taiji runtime，
其余进入带来源和原因的 quarantine。旧 model/HF/GGUF/publish/Cortex switch 接口保留短期 410 迁移桩并从默认 OpenAPI 隐藏；
`ChatRequest` 删除 `engine/agent_*`，Legacy 依赖和路由改为显式 `SEED_ENABLE_LEGACY=1` 才启用。完整 Python CI 为
`526 passed, 6 skipped`，ruff、编译和 diff check 通过。

**已完成（2026-08-29）：W4 退出 Gate。** provider artifact 内容寻址、首轮 chat canary、运行时 watchdog/回退，Taiji 零 Transformer
导入、Legacy-off 启动与 native API 边界回归共 `34 passed`；合法 Qwen provider 仍只位于语言器官集成边界，NeuroPlex 仍仅为显式离线
benchmark/兼容 profile，不进入默认客户端和 Taiji cognition。

**已完成（2026-08-29）：W5 第一 slice 的客户端真实性审计与旧调用清理。** 本轮按
`source/owner/freshness/availability` 逐页核对 Chat、Life、Agent/能力、Training、Settings、KB：

1. Workspace 的目录切换、系统选目录和快捷路径已分别收口到 native `/api/workbench/workspace`、`/api/system/select_folder`、
   `/api/system/quick_paths`；原生 Workbench 统一承担文件创建、UTF-8 digest-checked patch、重命名、删除和 argv-only terminal，所有副作用
   都经 preview→approval→execute，目录创建和资源管理器唤起这类未登记能力从界面移除。Monaco 保存现在携带原始 digest、编码和完整性，
   冲突/非 UTF-8/截断快照 fail-closed。
2. KB 页不再在默认客户端路径调用 `/api/rag/*` 或挂载 Legacy `FileUploadQueue`；它只读取统一 runtime snapshot 中是否真实出现
   `knowledge.*` capability，没有能力时明确显示待接入边界。这样“资料库管理”不会被误报为 Taiji 已具备 provenance Observation 检索。
3. Chat/Life/Settings 的旧 ByteSensor/ByteMotor、固定 257 单元和 tokenizer/embedding 断言文案已移除；Life 只显示 runtime、语言器官、
   Workbench 和 needs 是否上报的真实字段。Seed active 时 runtime status 正确报告 `is_taiji=true`，不再依赖 Legacy `app_state` 标志。
4. 删除未被引用且仍携带旧 engine/workspace 写入路径的 `useWorkspaceBridge.js`，移除未使用的模型市场 locale key；frontend 现存 RAG 字符串
   仅限通用组件测试夹具，不属于产品调用路径。

本轮证据：前端 Vitest `22 files / 185 passed`，ESLint `0 errors / 13 warnings`，生产构建通过；后端原生 Workbench/系统路由与平台边界回归
`72 passed, 1 skipped`，ruff、compileall、diff check 通过。尚未宣称 knowledge capability 已实现，也尚未进行 packaged route-level smoke。

**已完成（2026-08-29）：W5 第二 slice 的真实状态接入。** `/api/runtime/status.tools` 现在携带 Workbench capability snapshot 的
`snapshot_id/revision/source/owner/observed_at`，并纳入稳定的 `ToolsPayload` 合同；前端 `runtimeStore.statusEvidence` 统一从 runtime
status、provider artifact 状态、Workbench snapshot、life/training section 生成 `source/owner/freshness/availability` 投影。
Chat、Life、Agent/能力、Training、Settings、KB 六个入口均展示对应证据；homeostasis/self-state 在 `needs` 未上报时明确为“未上报”，
不再用默认值或前端表达推断冒充运行时状态。前端 Vitest `23 files / 187 passed`、ESLint `0 errors`、生产构建通过；后端定向回归
`65 passed, 1 skipped`，OpenAPI、ruff、compileall、diff check 通过。

**已完成（2026-08-29）：W5 第三 slice 的 packaged route-level smoke 与 frontend/source capability contract。** 前端源码门禁固定检查
六个产品入口必须挂载状态证据投影，并禁止 Legacy RAG/model/HF/GGUF/engine 旧路径重新进入产品源码；明确保留的
`FileUploadQueue` 仅作为未挂载的训练上传兼容组件。CI 已在 frontend lint 后执行该门禁。生产 `vite preview` 逐路由 smoke 使用精确
主容器选择器，覆盖 7 路由、错误页排除、6 个状态证据页面、导航、聊天/训练交互和移动端，共 `35 项 / 0 失败`；此前误测旧的
5173 开发进程已查明并改为显式 4173 生产预览地址，避免验证对象漂移。后端 Legacy-off 启动合同增加 runtime/workbench/system 原生路由，
定向回归 `67 passed, 1 skipped`，frontend `23 files / 187 passed`、ESLint、生产构建、native boundary、ruff、compileall、diff check 全部通过。

**已完成（2026-08-29）：W6 第一 slice 的 OpenAPI→frontend endpoint contract。** 新增版本化 OpenAPI 快照约束，快照测试现在同时
比较 operation 的 method、query parameters 和 request body；前端 `check-api-contract` 读取该快照，校验产品源码中的 83 个 API
字面量及直接调用的 method、查询参数和 JSON 顶层字段，并已接入 CI。未使用且仍携带 Legacy `/api/taiji/upload` 的
`useChatUpload.js` 已删除；`select_folder` 的标题参数已正式纳入后端接口与快照。W6 第一片证据为后端 `67 passed, 1 skipped`、
前端 Vitest `23 files / 187 passed`、native boundary/API contract、ESLint、生产构建、Ruff、compileall、diff check 全部通过。

**已完成（2026-08-29）：W6 第二 slice 的 typed/native API facade 第一批迁移。** 新增 `nativeApi` facade，集中维护 runtime、
Workbench 和 system 三组 native 端点、查询参数构造、JSON 序列化和错误解包；Workbench projection、runtime health/status、Workspace
快速路径、PathSelector 已切换到命名操作，不再在这些入口自行拼接 URL 或重复序列化请求体。facade 路径本身已纳入 API contract
门禁，并新增 4 项 facade 单测；一次发现并修复 Workspace 快速路径把 payload 误当 Response 的边界回归。证据为前端 Vitest
`24 files / 191 passed`、API contract/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第三 slice 的 typed/native API facade 第二批迁移。** settings、auth、chat、training 和 App 健康/版本入口
已切换到 `nativeApi` 命名操作；普通 JSON 的 URL、method、请求体序列化和错误解包不再散落在页面/Store。流式聊天、附件上传、训练原生
训练和检查点恢复保留为显式 raw-response 方法，避免把 SSE/FormData 错当成 JSON；`nativeApiPaths` 覆盖对应 OpenAPI 路径。证据为前端
Vitest `24 files / 191 passed`、API contract/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第四 slice 的特殊传输边界收口。** 训练产品页改用 `nativeDatasetUpload` 语义开关，native dataset 上传通过
命名 facade 操作，通用上传组件的旧 endpoint prop 仅保留在隔离兼容路径；新增 chat/训练 SSE 与 FormData 的 raw-response 契约测试，覆盖
取消信号、非 2xx、空 body 和 multipart body 不被 JSON 化。证据为前端 Vitest `24 files / 194 passed`、API contract/native boundary、
ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第五 slice 的 WorkspaceView 第一阶段结构拆分与最小观测点。** 路径选择对话框已拆为
`WorkspacePathDialog`，父视图只负责 native 工作台状态、路径切换和 preview→approval→execute 协调；原生 API facade 新增请求数、成功/失败、
最后状态和延迟 snapshot，为后续 trace/SLO 提供统一观测入口。父视图移除已迁移对话框的专属样式，新增组件与 facade 观测回归；前端
Vitest `25 files / 196 passed`、API/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第六 slice 的 WorkspaceView 文件树展示拆分。** 文件树渲染、展开/折叠图标、工具栏和树节点事件转发已抽为
`WorkspaceFileTree`；父视图继续唯一拥有目录加载、文件读写、快捷打开、编辑器联动和 native mutation 状态。移动端隐藏规则和原有树样式
随组件迁移，新增组件级事件回归；前端 Vitest `26 files / 197 passed`、API/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第七 slice 的 WorkspaceView 编辑器/终端协调拆分。** 编辑器区域、终端显示、终端尺寸事件和 Monaco
保存事件转发已抽为 `WorkspaceEditorPane`；父视图通过显式 expose 引用继续读取当前文件、保存状态和编辑器动作，仍保留 native approval
handler、文件状态与 mutation 流程。新增组件级回归；前端 Vitest `27 files / 198 passed`、API/native boundary、ESLint、生产构建全部通过。

**已完成（2026-08-29）：W6 第八 slice 的 SettingsView 运行环境区拆分。** 运行环境状态与未认证终端开关已抽为
`SettingsRuntimePanel`；父视图继续唯一持有 `nativeApi` 设置持久化、GET/POST 竞态防护、失败回滚和运行时健康探测。新增组件级开关事件回归，
前端 Vitest `28 files / 199 passed`、API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**已完成（2026-08-29）：W6 第九 slice 的 nativeApi trace/SLO 观测面板。** 新增 `RuntimeApiMetricsPanel`，由 facade snapshot 只展示请求路径、
请求/成功/失败计数、平均延迟和最后状态码，不采集或展示请求正文；`RuntimeEvidenceStrip` 仅在 Settings 显式开启该面板，并在组件卸载时清理刷新计时器。
前端 Vitest `29 files / 201 passed`、API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**已完成（2026-08-29）：W6 第十片的 TrainingView 训练概览拆分。** Loss 画布、真实进度、吞吐/ETA 指标和检查点列表已抽为
`TrainingOverviewPanel`；训练启动、暂停、停止、SSE 状态和检查点恢复仍由父视图与 `useTraining` 负责，组件仅通过 props/events 连接。
父视图同步移除已迁移的概览样式与绘图 watcher。前端 Vitest `30 files / 203 passed`、API contract/native boundary、ESLint、生产构建全部通过
（ESLint `0 errors / 13 warnings`）；中途发现并修复了组件 props 与模块状态同名造成的 `vue/no-dupe-keys` CI 错误。

**已完成（2026-08-29）：W6 第十一片的 TrainingView 数据集面板拆分。** 上传入口、文件表格、选择/删除/预览事件和样本预览已抽为
`TrainingDatasetPanel`；父视图继续唯一持有数据集加载、删除、预览和 native 上传协调，组件通过显式 props/events 连接。旧数据集专属样式已从父视图移除。
前端 Vitest `31 files / 205 passed`、API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**已完成（2026-08-29）：W6 第十二片的 TrainingView 日志与训练控制展示区拆分。** 日志展示/清空事件已抽为
`TrainingLogPanel`，暂停/恢复/停止控制已抽为 `TrainingControlBar`；父视图继续唯一持有 SSE、训练状态和 native 操作，
组件仅通过显式 props/events 连接。新增组件级回归，前端 Vitest `33 files / 209 passed`、API contract/native boundary、
ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**已完成（2026-08-29）：W6 第十三片的 ChatView 消息与输入区展示拆分。** 消息欢迎区、示例对话、消息气泡、原始输出和消息动作已抽为
`ChatMessageList`，输入框、快捷模板、附件选择和停止入口已抽为 `ChatComposer`；父视图继续唯一持有流式 reader、会话状态、附件上传和
native API 副作用，组件仅通过显式 props/events 连接。前端 Vitest `35 files / 213 passed`、API contract/native boundary、ESLint、
生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**已完成（2026-08-29）：W6 第十四片的 LifeStatusView 原生状态证据展示拆分。** Taiji 原生运行态卡片、语言器官/工作台/needs
状态和推进链路已抽为 `LifeNativeStatusPanel`；父视图继续唯一持有 runtime snapshot、轮询、生命活动边界和报告导出，组件通过显式
props 连接。前端 Vitest `36 files / 215 passed`、API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**已完成（2026-08-29）：W6 第十五片的 LifeStatusView needs 与表达展示拆分。** needs 五维图、生命表达、需求明细和生命事件流已抽为
`LifeNeedsDashboard`；父视图继续唯一持有 runtime snapshot、轮询、生命活动边界、需求值归一化和报告导出，组件只接收显式 props。
前端 Vitest `37 files / 217 passed`、API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**已完成（2026-08-29）：W6 第十六片的 SettingsView 通用设置展示拆分。** 主题、语言、时区、界面密度表单已抽为
`SettingsGeneralPanel`；组件不读写 store/API，父视图继续唯一持有设置加载、保存竞态、native API 副作用和失败回滚。前端 Vitest `38 files / 219 passed`、
API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**当前唯一下一步：开始 W6 第十七片的 SettingsView Taiji 参数展示拆分。** 先隔离局部激活阈值、响应超时、自动巩固和睡眠模式表单，
父视图继续唯一持有设置加载/保存竞态、native API 副作用和失败回滚；每片保持 `nativeApi` 单一入口、补齐组件回归并在提交前跑完整 CI，
不得先做视觉包装或 CUDA kernel。

**已完成（2026-08-29）：W6 第十七片的 SettingsView Taiji 参数展示拆分。** 局部激活阈值、响应超时、自动巩固和睡眠模式表单已抽为
`SettingsTaijiPanel`；组件不读写 store/API，父视图继续唯一持有参数校验、设置加载、保存竞态、native API 副作用和失败回滚。前端 Vitest `39 files / 221 passed`、
API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**当前唯一下一步：开始 W6 第十八片的 SettingsView 数据与隐私展示拆分。** 先隔离对话保留、自动清理和数据导出表单，父视图继续唯一持有设置加载/保存竞态、
导出聚合、重置确认、native API 副作用和失败回滚；每片保持 `nativeApi` 单一入口、补齐组件回归并在提交前跑完整 CI，
不得先做视觉包装或 CUDA kernel。

**已完成（2026-08-29）：W6 第十八片的 SettingsView 数据与隐私展示拆分。** 对话保留、自动清理、数据导出和危险操作展示已抽为
`SettingsPrivacyPanel`；组件不读写 store/API，父视图继续唯一持有设置保存竞态、导出聚合、重置确认、native API 副作用和失败回滚。前端 Vitest `40 files / 223 passed`、
API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**当前唯一下一步：开始 W6 第十九片的 SettingsView 关于与许可展示拆分。** 先隔离版本元信息和开源许可入口，父视图继续唯一持有版本加载、许可弹窗状态和 native API 副作用；
每片保持 `nativeApi` 单一入口、补齐组件回归并在提交前跑完整 CI，不得先做视觉包装或 CUDA kernel。

**已完成（2026-08-29）：W6 第十九片的 SettingsView 关于与许可展示拆分。** 版本元信息和开源许可入口已抽为
`SettingsAboutPanel`；组件不持有许可弹窗状态，父视图继续唯一持有版本加载、弹窗状态和 native API 副作用。前端 Vitest `41 files / 225 passed`、
API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**当前唯一下一步：开始 W6 第二十片的 SettingsView 共享表单样式与设置契约收口。** 在不改变父层状态拥有权的前提下，提取重复的设置展示契约和可复用样式边界，
补齐跨面板结构回归与组件注册检查；每片提交前跑完整 CI，不得先做视觉包装或 CUDA kernel。

**已完成（2026-08-29）：W6 第二十片的 SettingsView 共享面板结构、样式与契约收口。** 五个设置分区统一经由
`SettingsPanelSection` 渲染标题/面板骨架，共享控件样式集中到 `assets/styles/settings-panels.css`，新增跨面板结构回归；父层状态拥有权和 native API 边界不变。
前端 Vitest `42 files / 230 passed`、API contract/native boundary、ESLint、生产构建全部通过（ESLint `0 errors / 13 warnings`）。

**路线校准（2026-08-29 live audit）：** 旧的“W7 第一片还要接入真实 Workbench 工具”表述已过时。现有主线提交与回归已经证明
W0–W3 的 native Workbench 版本合同、只读/受控写入、语言证据与 IDE 切换、终端、审批、MCP registry、有限循环、checkpoint continuation
和真实跨文件任务闭环；W4 的 HF/GGUF/Transformer/Legacy 产品语义清理、W5 的客户端真实性接入、P6 provider watchdog 与 P3
interaction-group attribution 也已有对应证据。继续重复建设 Workbench 基础执行器属于路径偏移。

**已完成（2026-08-29）：聊天→Workbench 原生任务事件桥接。** 新增显式 `/api/chat/workbench/stream` transport，要求调用方提交已经由
Taiji 形成的结构化 `ActionIntent`，通过同一个 `SeedRuntime` 执行并把 `planned → policy → executing → outcome` 审计事件实时投影到聊天
SSE；IDE 仍从 `/api/workbench/events` 读取同一审计记录。前端 native facade、ChatStore、消息轨迹展示和 OpenAPI/产品 smoke 已接通。
该片只完成“执行与观测桥”，不把语言 provider 或自然语言启发式解析冒充 Taiji 的工具选择器。

**CI 收口（2026-08-29）：** 新增桥接后的全量 Python 回归 `530 passed, 6 skipped`，前端 `42 files / 232 passed`，native/API
contract、Ruff 主门禁与 B/SIM blocking、Black `26.5.1`、核心 mypy 和生产构建均通过；同时清理了 CI 发现的 8 个存量 Black 格式文件与
Workbench loop 的显式 `zip(strict=True)` 门禁问题。

**当前唯一下一步：建立 Taiji-owned 的只读任务准入层。** 在不由语言 provider、前端或自然语言硬编码选择工具的前提下，把 Taiji 当前
`CognitiveState/GoalPlanner/WorldAffordance` 产生的候选绑定到 Workbench capability snapshot，形成“候选→ActionIntent→聊天/IDE 共享执行→真实
after-state→Outcome”的第一条自主只读任务 Gate；无 Taiji 产生的 intent、能力快照漂移或候选与 capability 不一致时必须 fail-closed。
该 Gate 通过前不进入写入自治、开放域自然语言工具选择、CUDA kernel 或新的视觉包装。

**已完成（2026-08-29）：Taiji-owned 只读任务准入 Gate。** Workbench 新增 `TaijiTaskAdmission`，只接受带
`WorldAffordance` lineage 的 `ExecutiveCandidate`，由 `TSKV8Adapter` 的 `ExecutiveController` 在当前认知状态中选择候选，再绑定
当前 content-addressed capability snapshot；snapshot 漂移、candidate tick 过期、未接地候选、未知/禁用 capability、未声明参数和非
`read_only` 风险均在执行前 fail-closed。SeedRuntime 新增不执行的 `/api/workbench/taiji/admit` 与原子选择-准入-执行的
`/api/workbench/taiji/execute`，两者均不读取 prompt、不由语言 provider 选工具，并通过 JSON-safe decision projection、native API facade
和 OpenAPI baseline 暴露；真实只读文件 after-state/Outcome、路由、Workbench 回归与前端 facade 回归均通过。该 Gate 关闭的是
“Taiji 候选到 Workbench 的责任边界”，不宣称当前默认 Seed 已经自动生成 workspace affordance，也不开放写入/终端自治。

**当前唯一下一步：建立 Workbench capability snapshot 到 Taiji WorldAffordance 的受控投影 Gate。** 让当前快照中的可用只读能力以
带 capability snapshot/revision lineage 的世界 affordance 进入 Taiji，而不是在 adapter 或 API 中维护第二份工具表；只生成已连接且
参数契约可验证的候选，能力快照变化时让旧 affordance/candidate 失效，并用 `list/read/stat/search` 的真实 workspace evidence 验证
“能力发现→Taiji 候选→准入→after-state”闭环。该 Gate 通过前不进入写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

**已完成（2026-08-29）：Workbench capability snapshot→Taiji WorldAffordance 投影 Gate。** `CapabilitySnapshot` 新增受控投影：
只有显式提供的结构化 `parameter_bindings` 才会生成 `WorldAffordance`，其 `affordance_id` 由 capability/参数内容寻址，且保留
`workbench-snapshot`、capability revision 和 capability id lineage；未知/禁用/非 `read_only` capability、未声明参数和非 mapping
绑定均 fail-closed。`TSKV8Adapter.set_world_affordances()` 只接收通用世界 affordance，不导入 Workbench 或复制工具表；grounding 会保留
外部世界 lineage。SeedRuntime 与 `/api/workbench/taiji/project`、frontend native facade 已接通，API 使用 JSON-safe projection，
并可继续进入现有 Taiji ExecutiveController→Workbench admission→after-state/Outcome 链。该 Gate 证明能力合同能进入 Taiji 世界模型，
不宣称 `list/read/stat/search` 结果已经自动变成 WorldEvent，也不开放写入/终端自治。

**已完成（2026-08-29）：真实 workspace evidence→Taiji WorldEvent/WorldState freshness Gate。** WorkBench 的真实
`workspace.list/read/stat/search` after-state 结果现在由带 request/intent/call/capability snapshot/tick lineage 的
`WorkbenchTaijiEvidence` 封装，写入 Taiji `WorldEvent`，并保留 after-state digest 与受限结果内容；`TSKV8Adapter.record_world_event()`
只接受当前 world tick 的 typed event，事件和 affordance 失效状态进入 native checkpoint。每次真实 workspace 证据到达后，当前投影的
WorkBench affordance 会被精确清除，旧 candidate 不会继续沿用；下一轮必须重新投影并重新准入。真实临时项目覆盖 evidence、世界事件、
失效和 checkpoint continuation，WorkBench 定向回归 `30 passed`。该 Gate 仍不开放写入/终端自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

**已完成（2026-08-29）：capability/world freshness 的重新投影闭环。** `WorkbenchTaijiEvidence` 可以从已持久化的最新
`WorldEvent` 恢复并校验 event id、snapshot、参数、结果和 after-state digest；同一当前 tick 且同一 capability snapshot 下，才允许由
最新真实 evidence 自动生成结构化 projection binding。重新投影后的 affordance identity 同时绑定 evidence/after-state lineage，旧
candidate 因当前 world affordance id 不存在而 fail-closed；新增 `/api/workbench/taiji/reproject` 和 native facade，明确拒绝旧 evidence、
snapshot 漂移及失败 evidence。真实回归覆盖“旧 candidate 被拒绝→新 evidence→新 affordance→新 candidate→read-only admission→Outcome”、
`list/read/stat/search` 事件顺序和 runtime checkpoint continuation；全量 Python 回归 `540 passed, 6 skipped`，前端 `42 files / 233 passed`，
API/native boundary、Ruff、B/SIM、Black、核心 mypy、OpenAPI 与生产构建均通过。该闭环仍不开放写入/终端自治、开放域自然语言工具选择、
CUDA kernel 或视觉包装。

**已完成（2026-08-29）：不依赖手工 candidate 的真实只读自主 canary。** `SeedRuntime` 默认挂载 Taiji 原生
`LearnedAffordanceFeatures` 与 `ExecutiveController`，从当前 evidence reprojected affordance 直接合成候选；真实临时 workspace 已验证
在不注入 `ExecutiveCandidate`、不读取 prompt、不调用语言 provider、也不经过前端的情况下，连续完成 `workspace.list → workspace.read` 的
选择、准入、执行、WorldEvent 回写、当前 affordance 精确失效和最新 evidence 重新投影。checkpoint 保存/恢复后不会重复执行已消费事件，
也不会复用旧 affordance identity；WorkBench 定向回归 `33 passed`，全量 Python 回归 `541 passed, 6 skipped`，覆盖率 `44.53%`，
Ruff 主门禁/B-SIM、Black、核心 mypy 均通过。该 canary 证明 Taiji-owned 只读闭环已脱离手工 candidate，仍不开放写入自治、开放域
自然语言工具选择、CUDA kernel 或视觉包装。

**当前唯一下一步：建立多证据 successor graph 的有限多步 checkpoint continuation Gate。** 将 `list/search` 产生的多个
`read/stat` successor 纳入 Taiji-owned 的统一候选排序与 bounded loop，逐步提交每个已消费前缀的 checkpoint；验证中途失败只保留已完成
前缀，最新 WorldEvent 到达后旧 sibling candidate 全部失效，恢复后只从最新 evidence 重新投影，不重复消费或绕过累计预算。该 Gate
通过前不进入写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

**已完成（2026-08-29）：多证据 successor graph 的有限多步 checkpoint continuation Gate。** 新增
`seed-taiji-successor-graph-v1` runtime/API contract：Taiji 从当前多个 `WorldAffordance` 统一合成候选，逐步执行真实只读
`workspace.list/read/stat/search`，每个成功 evidence 到达后清空旧 frontier 并只从最新 WorldEvent 重新投影 successor；旧 sibling
candidate 因 frontier identity 变化全部失效。累计预算、已消费 request/affordance/event identity 和 frontier 进入 checkpoint，恢复时
重新投影并校验 frontier，禁止重复消费、旧 identity 复用、snapshot/budget/frontier 漂移；每次已尝试 step 都在副作用后提交 checkpoint。
定向 Workbench 回归 `35 passed`，OpenAPI 严格快照 `2 passed`，全量 Python 回归 `543 passed, 6 skipped`，覆盖率 `44.60%`，
Ruff 主门禁/B-SIM、Black、核心 mypy 均通过。该 Gate 仍只覆盖 bounded read-only graph，不开放写入自治、开放域自然语言工具选择、
CUDA kernel 或视觉包装。

**已完成（2026-08-29）：successor graph 的失败前缀与可审计 recovery Gate。** 真实 `workspace` 读取失败、snapshot 漂移和
checkpoint 写入失败均进入明确的 recovery 状态；checkpoint 持久化 `completed_prefix`、失败码/原因、latest evidence、剩余 frontier、
已消费 request/affordance/event identity。每个副作用前先保存 in-flight reservation，post-checkpoint 失败时恢复不会重试未知 step；
恢复入口对旧 loop 返回零步 `recovery_needed`，只允许外部取得 freshness-valid successor 后以新的受控 graph 继续，且不会自动扩大能力或进入
写入自治。WorkBench 定向回归 `38 passed`，全量 Python 回归 `546 passed, 6 skipped`，覆盖率 `44.70%`，Ruff 主门禁/B-SIM、Black、
核心 mypy 均通过。该 Gate 仍不开放开放域自然语言工具选择、CUDA kernel 或视觉包装。

**已完成（2026-08-29）：显式 recovery handoff 与 fresh-evidence continuation Gate。** `recovery_needed` 已从诊断结果提升为
可审计的受控恢复协议：WorkBench 必须先产生新的当前 tick/当前 capability snapshot 成功 evidence；handoff 校验 parent loop、failure
event、snapshot、schema 与 frontier lineage，生成新的 recovery loop identity，只把未消费且 freshness-valid 的 successor 交给 Taiji
Executive。旧 loop identity 会永久 retired，失败前缀、parent failure、source evidence 和 after-state provenance 均保留，budget 与
completed prefix 不重置；runtime/API 定向回归 `6 passed`，全量 Python 回归 `548 passed, 6 skipped`，覆盖率 `44.77%`，Ruff/B-SIM、
Black、核心 mypy 和 OpenAPI 均通过。该 Gate 仍不开放写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

**已完成（2026-08-29）：recovery source evidence 的 failure-context compatibility Gate。** recovery failure 现在持久化
失败 capability、完整结构化参数和路径上下文；handoff 除 current tick、snapshot、schema、frontier freshness 外，要求新 evidence 与
parent failure 的 capability/parameters 精确兼容。无关 capability、跨路径 evidence 和缺少失败上下文均 fail-closed，不能借 recovery
handoff 绕过失败原因；runtime/API 定向回归 `5 passed`，全量 Python 回归 `548 passed, 6 skipped`，覆盖率 `44.77%`，Ruff/B-SIM、
Black、核心 mypy 与 OpenAPI 均通过。该 Gate 仍不开放写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

**已完成（2026-08-29）：recovery branch portfolio 与多候选 fail-closed selection Gate。** 同一 parent failure 现在可以保存多个
通过 current tick、capability snapshot、failure capability/parameters 和 after-state digest 校验的 recovery evidence 分支；每条 branch
拥有内容寻址 identity、独立 loop identity、预算/完成前缀、frontier、consumed request/affordance/event provenance 和可恢复状态。Taiji
只能从 active 且未复活的 branch 中选择，选择时会永久 retire parent/当前 loop、重新校验 source evidence 与 frontier，并把 branch 以继承的
budget/prefix 继续执行；失败、checkpoint failure、旧 loop 和已选择 branch 不得绕过 portfolio 再次执行，checkpoint 会保留完整 portfolio。
新增 `/api/workbench/taiji/recovery-branch/register` 与 `/api/workbench/taiji/recovery-branch/select`，定向 recovery/successor 回归 `7 passed`，
OpenAPI 严格快照 `2 passed`，全量 Python 回归 `549 passed, 6 skipped`，覆盖率 `44.92%`，Ruff、Black、核心 mypy 均通过。该 Gate
仍只覆盖 bounded read-only recovery，不开放写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

**已完成（2026-08-29）：recovery branch portfolio 的长期 liveness 与容量淘汰 Gate。** portfolio 现在有明确的 native 最大 branch
槽位和 tick TTL；`active/selected/completed/failed/expired` 生命周期被统一校验，过期分支进入 `expired`，容量不足时只按
`last_touched_tick + branch_id` 的确定性顺序淘汰终态分支。淘汰记录保留 branch/loop/source evidence/after-state digest 墓碑并加入
retired loop 集合，旧 branch 即使从工作集移除也不能重新注册或选择；若没有可安全淘汰的终态分支，注册直接 fail-closed。维护操作只
更新 portfolio，不选择、不执行分支，并与完整 checkpoint 原子关联；新增 `/api/workbench/taiji/recovery-branch/maintain`。定向
portfolio 回归 `2 passed`，OpenAPI 严格快照 `2 passed`，全量 Python 回归 `550 passed, 6 skipped`，覆盖率 `45.00%`，Ruff、Black、核心
mypy 均通过。该 Gate 仍只覆盖 bounded read-only recovery，不开放写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

**已完成（2026-08-29）：recovery portfolio 的跨 checkpoint 一致性与并发互斥 Gate。** portfolio 增加单调 `revision`，维护、注册、选择和
successor continuation 可携带 expected revision；checkpoint 恢复后旧 revision 重放会 fail-closed。同一 SeedRuntime 的 successor、handoff、
branch register/select/maintain 统一进入可重入互斥锁，避免两个执行者同时选择或更新同一 branch；successor 执行完成后会把 branch 的 loop、
预算、完成前缀、frontier、event lineage 和 liveness 一并提升到新 revision 再落盘。新增 revision stale 回归，定向 portfolio/recovery `4 passed`，
全量 Python 回归 `550 passed, 6 skipped`，覆盖率 `45.10%`，Ruff、Black、核心 mypy 和 OpenAPI 严格快照均通过。该 Gate 仍只覆盖 bounded
read-only recovery，不开放写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

**已完成（2026-08-29）：recovery portfolio 的只读状态投影与审计可观测 Gate。** 新增不执行的
`GET /api/workbench/taiji/recovery-branch/portfolio`：以当前 snapshot/WorldState 读取 portfolio revision、tick、容量、TTL、
active/selected/completed/failed/expired/evicted 计数、branch 的最小身份/血缘/预算/frontier 摘要和 liveness due 列表；不返回
parameters、完整 evidence 或任何可直接执行的 candidate payload，evicted/tombstone 只作为审计信息。投影在 checkpoint restore 后保持
revision、branch identity 和顺序一致，过期状态可被观察但不会在只读查询中偷偷改变持久化状态；维护仍由显式 maintain Gate 完成。
WorkBench portfolio 定向回归 `2 passed`，OpenAPI 严格快照 `2 passed`，全量 Python 回归 `550 passed, 6 skipped`，覆盖率 `45.13%`，Ruff、
Black、核心 mypy 均通过。该 Gate 仍只覆盖 bounded read-only recovery，不开放写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

**已完成（2026-08-29）：recovery portfolio snapshot 的客户端消费与只读审计回放 Gate。** frontend native facade 增加
`taijiRecoveryPortfolio` 明确查询路径与可选 expected revision，`useWorkbenchProjection` 提供只读 `recoveryPortfolio` 状态和
`refreshRecoveryPortfolio(parentLoopId, expectedRevision)`，不新增执行/写入方法；facade 回归验证 query/body 边界，后端回归验证
checkpoint restore、revision stale、expired/evicted tombstone 和 branch 顺序，前端不会把 snapshot 摘要拼装为 candidate。前端 Vitest
`42 files / 233 passed`、native-boundary、API contract、ESLint `0 errors / 13 warnings`、生产构建均通过。该 Gate 仍只覆盖 bounded
read-only recovery，不开放写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。

**当前唯一下一步：建立 recovery portfolio 的客户端审计回放视图 Gate。** 在已有 native projection 消费层上增加只读审计模型/视图，按
revision 展示 branch 生命周期、容量压力、source evidence/after-state lineage 和 eviction tombstone；视图不得触发 maintain/select/execute，
也不得显示可直接复用的 parameters。通过前不进入写入自治、开放域自然语言工具选择、CUDA kernel 或视觉包装。
