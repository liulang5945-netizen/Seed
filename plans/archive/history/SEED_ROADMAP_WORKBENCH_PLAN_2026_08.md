# Seed / Taiji 路线执行记录：Workbench Closure 规划

> 本文由原总路线图按职责拆分而来。原始行号：1490–1869；本文件属于当前计划资料或历史证据，具体身份以本文件所在目录为准。
> 这是全盘审计后形成的 W0–W7 规划原文；当前状态以 active/roadmap/03_CURRENT_EXECUTION.md 为准。

### 16.1 全盘审计后的路线校准：从研究 Gate 转向产品执行闭环（2026-08-28）

#### 16.1.1 审计结论

本轮按用户要求暂停功能开发，只核对 `main@6e2204b` 的真实代码、计划、API、前端和客户端链路。结论不是 Taiji 缺少一个 IDE
按钮，而是项目存在一条系统级断层：**Taiji 内已经构造出世界、计划、`ActionIntent`、`ToolCall` 和 `Outcome` 等认知/效应器合同，
Seed 产品却没有一个 Taiji-native 的执行平面把这些合同接到 IDE、文件、终端、LSP、诊断和 MCP。**

这解释了为什么研究 Gate 数量持续增加，客户端仍像若干互不相连的面板：当前 Taiji 能在模拟环境中证明工具闭环，语言 provider 能形成
可读文本，IDE 也能被人手操作，但三者没有共享同一 capability、权限、执行、结果和状态合同。继续沿 interaction-group attribution、provider
watchdog 或更多小型数值 Gate 纵深推进，会扩大内部证明数量，却不会关闭产品最关键的因果闭环，属于路径偏移。

因此主线立即重排为 **Workbench Closure W0–W7**。P1–P7 的既有成果保留为认知基础，不回滚；抽象 recovery attribution、provider
watchdog、CUDA/fused kernel 和新视觉打磨全部冻结，直到真实工作台纵切片通过。

#### 16.1.2 当前进度的分层事实

| 层 | 已完成事实 | 尚未完成/不能宣称 |
|---|---|---|
| Taiji cognition | P1–P7 已覆盖版本化状态、感知/世界/工作空间、记忆、规划、结构生长、生成、工具合同与大量 checkpoint/lesion Gate | 未证明开放域智能；大量 Gate 仍是小型数值/模拟环境 |
| 语言器官 | `native-readable`、外部 Qwen provider、训练/安全准入、内容寻址、registry 和原子轮换已落地 | provider 只负责表达，不拥有 IDE/工具权限；watchdog 尚未做且不再是当前瓶颈 |
| 产品运行时 | `SeedRuntime.chat()` 可走 Taiji 输入边界并返回可读文本；桌面壳、标题栏、托盘、构建和 CI 已收束 | 原生聊天没有 tool event、执行循环或 IDE after-state；不能自主完成代码任务 |
| 工作台 | Monaco、文件树、人工保存、Python run 和交互式终端 UI 已存在 | API 被归为 Legacy 可选路由；没有 Taiji-native capability registry、事务、审批、outcome 或自主语言选择 |
| 工具/MCP | NeuroPlex 路线有 ReAct、工具表、MCP 和插件历史实现；Taiji 有通用 `TaijiToolEnvironment` 协议 | 原生模式上报空工具列表；现有 MCP/Agent 路由没有接入 Taiji，部分前后端路径/参数还不一致 |
| 训练/发布 | Taiji native checkpoint、训练恢复、provider artifact 已存在 | 产品页仍把 GGUF、LoRA 合并和 Legacy 模型发布当作 Taiji 正式能力 |
| 前端产品口径 | provider 回退、运行时和错误中心已有一定可观测性 | 多处界面仍展示 TSK-v8 旧叙事、Cortex 热切换、Legacy life/Agent 配置和未实现能力 |
| 工程门禁 | 主分支与远端同步，最近 CI 已全绿；前端 185 tests，跨平台/容器/启动门禁已建立 | API/前端 capability 契约没有生成或一致性门禁；“界面有入口但后端不存在/原生模式不注册”仍可全绿 |

规模事实也支持“先收执行平面”的判断：当前仓库约 202 个 API route decorators，17 个 Seed/API/desktop 文件仍直接导入 NeuroPlex；
`taiji/adapter.py` 已达约 9300 行，Taiji native 有 85 个测试文件、68 个 eval 脚本和 253 个跟踪报告。项目不缺继续增加局部 Gate 的能力，
缺的是把这些能力变成一个真实、可观测、可撤销的产品纵切片。

#### 16.1.3 已确认的问题清单与根因

| 编号 | 代码证据 | 实际问题 | 根因分类 |
|---|---|---|---|
| G1 | `api/routes_chat.py::_seed_event_generator()` 只调用 `seed_runtime.chat()` 并返回 `final` 文本 | Seed 原生聊天不会生成/执行工作台工具事件 | 产品执行链缺失 |
| G2 | `taiji/adapter.py::generate_tool_call/execute_tool_call` 只被 Taiji 测试消费，`api/`/`seed/` 无调用者 | P6 工具合同停留在模拟环境，没有产品适配器 | research→product 断层 |
| G3 | `api/app.py::_register_routers()` 通过 `_load_optional_router()` 挂载 `routes_agent_workspace` | 关闭 Legacy 时，内置 IDE 的文件 API 一起消失 | 所有权分类错误 |
| G4 | `frontend/src/composables/useWorkspaceBridge.js` 全仓无调用者 | 文件打开、命令、错误回流只是注释承诺 | 死桥/假接线 |
| G5 | `MonacoEditor.vue` 用硬编码列表、扩展名表和组件内 `ref` 切换语言 | Taiji、后端和 checkpoint 不知道当前编程语言，也无法自主选择 | 状态只在 UI |
| G6 | 原生模式的 `runtime_service._tools_section()` 返回空列表，`runtimeStore.modelLifecycle` 却宣称可工具调用/自主探索 | 产品状态与真实 capability 相互矛盾 | 双重真相源 |
| G7 | `routes_agent_workspace.py` 的 run/create/analyze 仍导入 `neuroplex.agent_ext` | 即使界面可用，也不是 Taiji-native 工作台执行器 | Legacy 反向占位 |
| G8 | `AgentConfigView.vue` 使用 `/api/mcp/start/{id}`、`install/{id}` 等路径，后端要求 `/api/mcp/start` + JSON body；搜索参数也不一致 | MCP 面板存在可稳定复现的前后端合同漂移 | 无契约 Gate |
| G9 | TrainingView/useTraining/locales 展示 GGUF 导出和“合并 LoRA 权重”，后端只返回“Seed 不支持” | 旧 HF/GGUF 模型格式仍被呈现为正式产品操作 | 迁移残留 |
| G10 | `routes_settings.py`、`routes_models.py`、`seed_platform.config` 仍保存 GGUF/HF/model_type API 与字段 | 前端残留背后还有设置、OpenAPI、测试快照和兼容数据残留 | 只隐藏 UI 不够 |
| G11 | Settings 仍允许 Seed↔Cortex 热切换，Agent/ReAct/MCP/RAG 只在 Legacy router 下出现 | “Legacy 仅离线对照”没有落实到产品边界 | 架构决策未产品化 |
| G12 | Chat/Life/Settings 仍出现“不经过学习式 embedding”“ByteSensor→ByteMotor 即 Taiji”等旧文案 | 产品继续传播已被 2026-08-25 架构纠正否定的方向 | 文案/心智模型漂移 |
| G13 | `ChatRequest`、Agent 设置仍暴露 `engine/temperature/max_iterations`，Seed 原生分支实际忽略这些字段 | 用户配置看似有效，实际不进入原生运行时 | 幽灵配置 |
| G14 | `taiji/adapter.py`、主要 Vue view 和路线图持续膨胀 | 新能力容易继续堆进巨型文件并产生隐藏耦合 | 模块边界债 |

#### 16.1.4 术语和所有权重新钉定

后续接口禁止继续使用含义模糊的 `language` 或 `model_type`：

| 概念 | 规范名 | 决策权 | 状态/证据 |
|---|---|---|---|
| 人类自然语言表达器 | `natural_language_backend` | Taiji 生成合同 + Seed provider loader | provider artifact、Gate、health |
| IDE 编程语言 | `programming_language_id` | Taiji 可提出/选择；Seed capability 执行 | 文件内容、扩展名、manifest、LSP、confidence、provenance |
| 文件语法高亮 | `editor_language_id` | Workbench projection | 可与 programming language 相同，但不是认知主体 |
| 运行器/工具链 | `runner_id` / `toolchain_id` | Seed capability registry + policy | 可用性、版本、平台、资源、权限 |
| Taiji 保存格式 | `taiji_checkpoint_format` | Taiji/Seed checkpoint contract | `seed-native-v1` 兼容信封与 native payload |
| 外部嘴巴资产 | `language_provider_artifact` | Seed 集成边界 | 可使用 HF/Transformers/LoRA，但仅是末端器官 |
| 导入/导出适配格式 | `artifact_adapter_format` | Seed 发布工具 | 不得成为认知架构或全局 runtime 开关 |

HF 本身不是禁词：Hugging Face 数据集、缓存、Qwen/Transformers provider 和 adapter 可以继续存在于数据/语言器官集成边界。
必须清除的是把 HF/GGUF/LoRA 当作 Taiji 核心 checkpoint、全局模型类型或正式产品认知切换的 UI/API。合法的外部 provider
能力移动到“语言器官资产”语境，不再出现在“Taiji 模型格式”语境。

#### 16.1.5 目标架构

```text
User / environment observation
        |
        v
Taiji perception -> world/self/memory/goal -> plan -> ActionIntent
                                                    |
                                                    v
                                           structured ToolCall
                                                    |
                                                    v
Seed Workbench Capability Plane
  registry -> snapshot/freshness -> policy/approval -> transaction/executor -> audit
       |             |                    |                    |
       |             |                    |                    +-> file/terminal/LSP/MCP result
       |             |                    +-> deny / ask / allow / budget
       |             +-> current files, languages, tools, permissions, versions
       +-> typed schemas, risk, reversibility, resource cost
                                                    |
                                                    v
typed WorkbenchOutcome + after-state + diagnostics + provenance
                                                    |
                                                    v
Taiji Outcome / Observation / episodic+procedural memory / online credit / replan

Frontend IDE: observes the same snapshot, transaction and audit stream; it never becomes the hidden executor.
Language provider: realizes ExpressionPlan only; it never receives workbench authority.
```

工作台 capability 至少分为：

- `workspace.list/read/stat/search`：只读、可默认自动执行；
- `editor.open/reveal/set_language/diagnostics`：可撤销 UI/分析状态；
- `workspace.apply_patch/create/rename/delete`：文件事务，必须有 before digest、patch、after digest 和撤销记录；
- `terminal.run/test/build/debug`：命令 schema、cwd、timeout、环境变量白名单、资源预算和完整结果；
- `toolchain.detect/select`：识别项目语言、LSP、解释器/编译器，不静默安装依赖；
- `mcp.list/invoke`：通过统一 schema 注册，不能继续直接复用 NeuroPlex registry；
- `dependency.install/network/destructive`：高风险能力，除非用户预先建立窄 allowlist，否则必须显式审批。

#### 16.1.6 不可破坏的不变量

1. Taiji 决定做什么；Seed 决定能力是否存在、是否获准以及如何安全执行；前端只观察和承载人机控制。
2. 不允许语言 provider、ReAct、RAG、工作流或 UI 先决定动作，再让 Taiji 只做文案/打分。
3. 每个 action 必须绑定 `intent_id/call_id/capability_revision/world_tick`，每个 outcome 必须绑定真实执行和 after-state。
4. capability snapshot、审批、预算、执行和 outcome 必须可 checkpoint/重启续接；过期 snapshot fail-closed。
5. 文件修改使用事务/patch，不把任意自然语言直接写盘；执行前后可 diff、可撤销、可审计。
6. IDE 编程语言允许 Taiji 自主切换，但自动执行只限高置信、可逆的 `editor.set_language`；若会改变 runner、安装依赖、
   执行命令或覆盖未保存状态，必须进入对应风险 Gate。
7. UI 不得展示后端/原生模式未注册的能力；API 不得保留永远返回“不支持”的正式操作来制造假能力。
8. Legacy 只保留离线 benchmark/兼容启动，不再作为正式客户端的隐藏能力供应商。
9. 每个阶段先证明门禁能变红，再验收绿；CI 必须同时跑 native、legacy-off、frontend contract、Windows 和 packaged smoke。
10. CUDA 继续暂缓；工作台闭环不依赖本机硬件升级，不能以 CUDA 为阻塞理由。

#### 16.1.7 唯一顺序：Workbench Closure W0–W7

以下是严格顺序，不是可并行菜单。前一阶段退出 Gate 未通过，不进入后一阶段。

##### W0：Workbench Capability Contract + 只读真实纵切片

目标是先打通最小但真实的 `Taiji → Seed → IDE/workspace → Taiji` 回路，而不是先做万能 Agent。

工作项：

1. 在 Seed 产品边界定义版本化 `CapabilityDescriptor`、`CapabilitySnapshot`、`WorkbenchActionRequest`、
   `ExecutionPolicyDecision`、`WorkbenchTransaction` 和 `WorkbenchOutcome`；Taiji 继续只使用自身 `ToolCall/Outcome`。
2. 新建 Taiji-native `WorkbenchEnvironment(TaijiToolEnvironment)`，首批只注册 `workspace.list/read/stat/search`、
   `editor.open` 和 `editor.diagnostics.read`；不得导入 NeuroPlex。
3. 把工作区基础 API 从 Legacy optional router 中拆出为 core router；Legacy create-project/analyze/install 单独隔离或返回明确
   `legacy_only`，不再让 IDE 是否存在取决于 `SEED_ENABLE_LEGACY`。
4. SeedRuntime 新增 action event stream：`planned → policy → executing → outcome`；前端聊天与 IDE 订阅同一事件，
   `editor.open` 由状态投影驱动，不再通过无人消费的 window event bridge。
5. `/api/runtime/status` 只从 capability registry 上报工具；删除“空工具列表但宣称可自主探索”的推断文案。
6. 建立第一个真实 canary：在临时工作区放入未见文件，Taiji 形成读取意图，Seed 执行读取，文件 digest/内容摘要作为
   `WorkbenchOutcome` 回写，checkpoint 后可继续，关闭 environment 时 fail-closed。

退出 Gate：

- legacy-off 启动仍能打开 IDE、列目录和读取文件；
- 一条真实 read-only action 从 `ActionIntent` 到 UI 可见 outcome 全链路保留同一 lineage；
- 断开 WorkbenchEnvironment、篡改 capability revision、路径越界和过期 snapshot 均确定性失败；
- 任何前端显示的 capability 均存在于 OpenAPI/runtime snapshot，契约测试可通过故意删端点变红；
- 未实现 write/terminal/MCP 时 UI 明确显示“未授权/未接入”，不得伪装可用。

##### W1：编程语言识别、选择与 IDE 自主切换

1. 用 `ProgrammingLanguageEvidence` 统一扩展名、shebang、文件内容、项目 manifest、邻近文件、LSP 与 toolchain 可用性；
   现有 `extToLang` 只降为一个低权重证据源。
2. `programming_language_id`、`editor_language_id`、confidence、provenance、capability revision 和用户 override 进入
   Workbench state；Monaco 不再维护第二份隐藏真相。
3. 注册可逆 `editor.set_language` action。高置信且不改变运行器/文件内容时可由 Taiji 自动执行；低置信、语言冲突或会改变
   toolchain 时产生 `ask_user` policy outcome。
4. 语言列表由 backend capability 动态提供；未知语言保持 `plaintext`，不得因不在硬编码数组而丢失状态。
5. 用 `.h`、无扩展 shebang、多语言 monorepo、Vue/TS、notebook/markdown code block 和错误扩展名建立 holdout；
   filename-only lesion 必须显著退化，证明不是扩展名查表。

退出 Gate：Taiji 能解释“为何选择该语言”、自主切换后 Monaco/LSP/runner snapshot 一致，用户 override 可保持并撤销，
checkpoint/重启不会把旧语言状态错误应用到新文件。

##### W2：受控写入、终端与测试执行

1. 文件修改只接受结构化 patch/transaction，包含 before digest、目标路径、预期 after digest、冲突处理和 undo token；
   create/rename/delete 使用同一事务模型。
2. 终端从交互式 WebSocket UI 中抽出非交互 `terminal.run` executor，参数包含 argv、cwd、timeout、env allowlist、
   output limit 和 expected artifacts；不把 shell 字符串直接拼接执行。
3. capability 风险分级采用渐进自治：只读默认自动；可逆编辑按用户 autonomy policy；写入需预览/撤销；安装、网络、删除和
   破坏性命令默认显式审批。
4. 真实 diagnostics/test/build 结果回写 Taiji；成功不以 exit code 单独判断，还要记录 diagnostics、产物和 after-state。
5. 故意覆盖未保存文件、cwd 漂移、超时、输出洪泛、部分 patch 冲突和进程中断，验证原子失败与恢复。

退出 Gate：Taiji 可在临时项目中读文件、选择语言、生成 patch、运行测试、观察失败、重规划并修复；全过程可审计、可撤销、
checkpoint 续跑不重复执行已提交事务。

##### W3：原生工具/MCP registry 与自主循环

1. 将 workspace/terminal/LSP 与 MCP 都适配到同一 Seed capability registry；复用协议思想，不复用 NeuroPlex 认知/工具选择器。
2. MCP 管理 API 与前端按 OpenAPI 生成/校验，修复 path/body/query 漂移；安装/启动服务与调用工具分开授权。
3. Taiji `SelfState` 保存可用工具、权限、成功率、延迟、资源成本和最近失败，不把 UI localStorage 当自我模型。
4. 以真实 outcome 更新 affordance、procedural memory、world model 和 replan；语言 provider 只解释结果，不做隐藏 tool selection。
5. 建立有限 horizon autonomous task loop，并有 step/time/resource budget、取消、暂停、人工接管和恢复。

退出 Gate：在全新临时项目完成一个跨文件、诊断、测试的代码任务；去掉 Taiji planner 或 WorkbenchEnvironment 任一侧均失败，
证明不是 Legacy ReAct/外部 decoder 偷做；多次 checkpoint 不重复工具副作用。

##### W4：HF/GGUF/Transformer/Legacy 产品残留迁移

1. 前端删除 GGUF 导出按钮、LoRA 合并发布文案、无效的 `engine/temperature/max_iterations` 和正式产品 Cortex 热切换；
   外部 Qwen/LoRA 只在“语言器官资产”页面/高级配置中出现。
2. 后端将 artifact 分类收敛为 `taiji_checkpoint`、`language_provider_artifact`、`legacy_benchmark_artifact`；删除全局
   `model_type=gguf/self/cortex` 语义。
3. 对已保存 `gguf_path/model_type/model_name` 做一次显式设置迁移：能识别则转到 legacy/provider 配置，不能识别则隔离并提示，
   不静默猜测；旧端点先返回版本化 deprecation/410，再在一个兼容窗口后删除。
4. OpenAPI snapshot、Pydantic model、settings schema、frontend locales/composables/tests 一次清完；`download_hf` 等永远“不支持”
   的正式路由不能继续留在产品 API。
5. NeuroPlex 保留离线 benchmark CLI、固定数据/报告和 opt-in compatibility profile；默认客户端、主导航和 runtime status 不再暴露。

退出 Gate：frontend/source/OpenAPI/core settings 中不再存在 GGUF 或认知主体热切换；`taiji/` 仍零 Transformer import；
Qwen provider canary 仍通过，证明清理的是错误产品语义而不是合法语言器官。

##### W5：客户端全部内容与 Taiji 实际能力对齐

1. Chat 首屏、Life、Agent、Training、Settings、KB 的每一项状态标注真实 source、owner、freshness 和可用性；删除
   “ByteSensor→ByteMotor 即完整 Taiji”“不经过学习式 embedding”等已失效文案。
2. Life 面板改读 Taiji homeostasis/self-state；尚无原生数据的卡片隐藏或标为 roadmap，不再用 Legacy scheduler 代填。
3. Agent 配置改为 autonomy policy、capability scope、预算和审批偏好；不再展示 Seed 原生不消费的 ReAct 温度/迭代配置。
4. KB/RAG 只有在检索结果能作为带 provenance 的 Observation 进入 Taiji 时才称“知识能力”；否则仅称资料库管理。
5. 建立 route-level packaged smoke 和 capability screenshot/state contract，防止“页面可见但功能未接”再次全绿。

退出 Gate：默认客户端只展示 Taiji-native 实际能力；断开任一后端 capability 时 UI 自动降级且不保留假按钮/假状态；
文案、health、runtime snapshot 和可执行行为一致。

##### W6：模块化、契约生成与发布可靠性

1. 按 perception/world/memory/planning/execution/language/checkpoint facade 拆分 9300 行 adapter；保持公开 facade 和 checkpoint
   兼容，不做无验证的大重写。
2. 大型 Vue view 拆为 view model + typed API client + 可复用 panels；所有 mutation 经统一 client，不散落 URL 字符串。
3. 从 OpenAPI/capability schema 生成或校验前端 client，CI 检查每个前端调用有端点、method/body/query 对齐，native/legacy-off
   注册表与界面 capability 一致。
4. 增加真实任务 trace、action latency、policy deny、rollback、checkpoint resume 和 outcome learning 指标；研究 report 与产品 SLO 分离。
5. 发版门禁覆盖源码、dist、打包客户端、legacy-off、首次工作区、升级设置迁移和进程回收。

退出 Gate：模块拆分前后 checkpoint digest/行为等价；故意制造一个 URL、schema、能力状态或 checkpoint 漂移时 CI 必红；
打包客户端完成 W2/W3 canary。

##### W7：后续可靠性、研究、性能与产品体验工作包

W7 不是把 provider watchdog、interaction-group、小型模拟 Gate、CUDA 或视觉体验永久搁置，而是把它们从“现在就继续加功能”
改为**有前置条件、有升级路径、有真实验收的后续工作包**。其中小型模拟 Gate 是 W0–W7 全程使用的验证层，不需要等到 W7
才恢复；其余工作包只有在所依赖的产品合同稳定后才进入实施，避免继续用内部模拟替代尚未闭合的工作台能力。
W7 的实施阶段仍在 W0–W6 全部通过后开始；下表的进入条件是各工作包除总顺序外还必须满足的证据条件，不是允许提前插队。

排程定位如下：

| 工作包 | 是否保留 | 进入条件 | 在总路线中的位置 |
|---|---|---|---|
| 小型模拟 Gate | 保留且立即作为验证手段使用 | 对应阶段已有可故意打红的因果假设 | W0–W7 横切，不单独宣称产品完成 |
| provider runtime watchdog | 完整保留 | W0 的事件、审计、checkpoint lineage 与 W3 registry 稳定 | W7-R1 |
| interaction-group / recovery attribution | 完整保留 | W2/W3 已产生真实多步失败、恢复和工具 outcome trace | W7-R2 |
| 视觉与桌面体验收口 | 完整保留 | W5 已完成能力、状态、文案与路由真实性对齐 | W7-R3 |
| CUDA 与性能优化 | 完整保留，当前仅硬件验证受阻 | W6 固定 CPU 基线、checkpoint 合同，并取得可用 CUDA 主机 | W7-R4 |
| 开放域学习与结构自进化 | 完整保留 | 上述真实任务 trace、资源指标和 causal lesion 可共同支撑增长决策 | W7-R5 |

###### W7-G0：三层 Gate 梯度——小型模拟不取消，但必须向真实环境毕业

小型模拟 Gate 的正确定位是低成本证明机制是否存在，而不是能力终点。此梯度从 W0 开始适用于每一个工作包：

1. **S0 确定性小型模拟**：最小数值世界、固定 seed、边界输入和单一因果变量；必须先通过 lesion、错误输入或断开关键组件证明
   Gate 会红，再验证实现后变绿。S0 可以阻止错误实现进入下一层，但不得单独形成“已具备通用能力”的产品声明。
2. **S1 replay / sandbox Gate**：使用脱敏的真实 action/outcome trace、临时仓库、失败重放和 checkpoint 中断续接；验证机制能处理
   非理想顺序、工具错误、状态漂移和资源限制，而不是只适配手写 toy schema。
3. **S2 packaged-client / real-workbench canary**：在打包客户端、legacy-off 和真实工作台 capability 下完成用户任务；以真实文件、
   diagnostics、命令结果、UI 状态和 audit lineage 作为最终证据。

每个新 Gate 必须在 manifest 中声明 `claim`、`owner`、`S0/S1/S2 level`、`red proof`、`graduation target`、输入摘要、
checkpoint revision 和替代了哪些旧报告。S1/S2 已覆盖的 S0 执行日志进入 archive，只保留可复现脚本、合同和最终报告，避免
模拟报告无限堆积。任何能力若只有 S0 证据，路线图必须显式写“模拟机制成立，产品能力未验收”。

退出 Gate：每项长期能力都能从当前 claim 追溯到对应 S0/S1/S2 证据；故意移除关键组件会在最低适用层变红；不存在用 S0
通过结果替代 S2 产品完成声明的情况。

###### W7-R1：provider runtime watchdog、稳定回退与恢复

目标不是让外部 Qwen/provider 变成 Taiji 的认知主体，而是保证作为“语言器官”的 provider 在运行时退化时可检测、可隔离、
可回退，且失败不会污染 Taiji 的认知状态、checkpoint 或下一次请求。

工作项：

1. 为每次语言 realization 建立版本化健康记录，至少区分 `accepted`、语义校验失败、validated fallback、timeout、加载异常、
   artifact 漂移和 canary 失败；健康状态按 artifact digest 隔离，禁止跨版本继承计数。
2. 使用连续失败阈值、滚动接受率、冷却期和恢复迟滞共同决定 `healthy/degraded/quarantined/probing`，避免单次抖动触发回退，
   也避免 provider 在 active/previous 间频繁振荡。
3. 自动回退只允许落到 registry 中 allowlisted、内容寻址仍有效且 canary 通过的 previous version；previous 漂移、过期或不存在时，
   必须 fail closed 到 `native-readable`，不得选择任意本地模型。
4. watchdog 状态、计数、冷却期限、active/previous revision 和最后失败原因进入 checkpoint；重启后继续原状态，但不得重放已经完成
   的语言请求或泄漏 prompt/history。
5. runtime status、聊天 final event 和异常中心显示 active/fallback/quarantine/probe、artifact revision 和可操作原因；前端只观察，
   不自行决定轮换或清空错误。
6. canary 覆盖“active 连续退化 → previous 原子回退 → 冷却 → 隔离 probe → 恢复 active”，以及 previous 漂移、进程中断、
   并发请求和 checkpoint continuation；任何失败不能形成半提交 registry。

退出 Gate：provider 退化可在请求级 trace 中重现；回退目标经过内容寻址和 canary 重新确认；重启前后 watchdog 决策一致；移除
health source、篡改 previous digest 或关闭语义 validator 时 Gate 确定性变红；Taiji 在 provider 全部不可用时仍通过
`native-readable` 给出可读、来源清楚的降级输出。

###### W7-R2：interaction-group 与 recovery attribution 的真实任务化

interaction-group 不再为了增加“神经元群”概念而扩展，而是用于解释和改善真实多策略、多工具、多记忆源共同参与时的成功、失败与恢复。

工作项：

1. 从 W2/W3 的真实 task trace 定义 interaction observation：参与的 workspace route、memory source、planner branch、tool call、
   recovery action、资源成本和最终 outcome；不得按名称或手工角色表直接指定贡献。
2. 在预算内实现可计算的边际归因：先使用 leave-one-group-out、成对交互和局部反事实；只有证据显示高阶交互必要时才提高阶数，
   不默认做指数级全子集搜索。
3. group 的形成、合并、拆分、休眠和剪枝由持续贡献、互补性、冲突率、恢复价值和资源预算驱动；结构变化写入 provenance，
   可单独回滚，不得破坏无关 group 的 digest 与 checkpoint 状态。
4. 把 recovery attribution 回写到 workspace routing、procedural/episodic/semantic memory 和 planning policy，但保留各 owner 的更新边界，
   不建立一个重新包办全部学习的中心控制器。
5. 使用未见工具组合、跨文件故障、错误诊断、部分 patch 冲突和多步恢复建立 holdout；与 single-strategy、no-group、random-group
   和 no-attribution 做同预算对照。

退出 Gate：interaction-group 在至少一类真实工作台任务上稳定优于最强单策略和随机分组；lesion 能定位退化来源；错误归因可局部回滚；
计算与内存开销随 group 数量保持有界；不存在只有 group 数量增加、任务成功率和恢复效率不改善的“规模即进化”声明。

###### W7-R3：视觉、桌面外壳与交互体验收口

视觉工作不取消，但必须建立在 W5 的真实 capability 和状态模型上；否则只会把错误的 Legacy/HF/伪 Agent 内容包装得更漂亮。

工作项：

1. 统一客户端信息架构：侧边栏在目标窗口高度内完整显示核心导航，低频项进入显式二级入口；工作台、Taiji 状态、训练、语言器官、
   设置和异常中心的层级与真实 owner 对齐，不再使用滚动隐藏关键入口。
2. 建立设计 token 和可复用组件，统一字体、间距、圆角、阴影、色彩、分隔、focus ring、loading/empty/disabled/error/fallback/
   approval/executing/rollback 状态，清除各页面独立硬编码样式。
3. 收口 Windows 桌面品牌资产：窗口/任务栏、系统托盘、托盘通知和打包产物使用同一 Taiji logo 来源与多尺寸资源；应用内允许
   低成本流转动画，任务栏和系统通知使用平台兼容的静态帧；圆润外壳在缩放、最大化和系统阴影下不裁切内容。
4. UI 只展示 registry 中真实存在的 capability，并完整呈现 action lineage、审批、执行、回退和 provider 降级；视觉状态不能掩盖
   capability 不可用、Legacy-only 或实验性边界。
5. 覆盖键盘导航、焦点顺序、对比度、reduced-motion、100/125/150/200% DPI、多显示器、浅深色和小窗口；禁止为追求动效
   牺牲可访问性、启动时间或托盘稳定性。
6. 建立 route screenshot/state contract 与 packaged-client smoke，重点检查侧边栏溢出、窗口圆角、任务栏/托盘/通知图标、真实状态源、
   首屏任务完成路径和异常降级。

退出 Gate：默认打包客户端在各 DPI 下无关键导航滚动、裁切和空白壳；桌面所有品牌入口一致；关闭 capability/provider 时 UI 能准确降级；
视觉回归、可访问性和打包 smoke 均可通过故意破坏 token、图标或状态绑定而变红。

###### W7-R4：CUDA、跨设备一致性与测量驱动的性能优化

CUDA 不是取消，而是**当前缺少可验证硬件**。在获得 CUDA 主机前允许整理 device abstraction、benchmark schema 和 CPU 基线，
但不得提交“已适配 CUDA”或“已加速”的能力结论；CUDA 主机不可用也不阻塞 W0–W7-R3。

工作项：

1. 在 W6 固定代表性 CPU workload、数据 manifest、seed、checkpoint revision、精度指标、峰值内存、吞吐与延迟，先 profile 出真实热点；
   没有测量证据的模块不进入 CUDA 优化。
2. 建立显式 device/dtype/capability 合同，禁止模块内部私自选择设备；CPU-only、CUDA unavailable、显存不足和设备切换均有确定性回退。
3. 验证 CPU → CUDA → CPU checkpoint continuation，覆盖 optimizer/local-learning 状态、随机状态、稀疏结构、provider artifact 引用和
   长序列中断；旧 CPU checkpoint 必须继续可读。
4. 先做算子迁移和批处理/向量化，再按 profile 证据评估 mixed precision、稀疏布局与 fused kernel；自定义 kernel 必须保留参考实现、
   数值对照和硬件 capability fallback。
5. 跨设备验证 deterministic/tolerance 边界、NaN/Inf、OOM 恢复、吞吐、p50/p95 latency、峰值显存和能耗代理；性能提升不能以
   破坏 Gate、checkpoint 或学习质量为代价。

退出 Gate：在真实 CUDA 主机上通过 CPU/CUDA 数值与 checkpoint 一致性；目标 workload 达到预先登记的加速和显存阈值；移除 CUDA、
降低 capability 或触发 OOM 时自动回落且结果可审计；只有实测热点才允许保留 fused/sparse kernel。

###### W7-R5：开放域学习与结构自进化

自进化不等于持续增加神经元数量。它必须由真实任务上的长期误差、容量拥塞、恢复失败和新分布证据触发，并同时允许生长、重组、
巩固、休眠、剪枝和回退。

工作项：

1. 聚合 W2/W3/W7-R2 的真实失败簇，区分表示容量不足、记忆干扰、路由冲突、世界模型误差、工具缺失和语言 realization 失败，
   防止所有问题都被误判为“需要扩大神经元规模”。
2. 为 perception/workspace/memory/world/planning 分别定义可观测的 capacity pressure 与 growth proposal；结构增长由局部 owner 提议，
   由全局资源治理器按收益、预算和可回滚性批准，不使用固定任务名触发器。
3. 新增单元/连接/group 先在隔离 shadow 状态学习，通过 holdout、lesion 和资源收益 Gate 后原子并入；未获益或产生漂移时恢复旧 topology，
   checkpoint 同时保存结构 revision 和参数状态。
4. 开放域 world/semantic/skill 学习从真实 provenance Observation 与 outcome 中形成，语言 provider 只负责表达；新知识必须能追溯、
   冲突检测、遗忘控制并被任务成功率验证。
5. 长期评测同时记录能力收益、遗忘、恢复时间、参数/连接规模、内存、延迟和能耗代理；禁止只报告规模扩大或训练 loss 下降。

退出 Gate：至少一类未见真实任务触发结构增长后，在同预算 holdout 上优于冻结结构，并且无关能力遗忘受控；growth lesion、错误增长和
rollback Gate 均成立；当容量压力消失时系统不会继续无界扩张。只有满足这些条件，才能把“自然生长式迭代”作为 Taiji 已实现能力。

W7 的实际实施顺序固定为 **G0 贯穿全程，R1 → R2 → R3 → R4 → R5**。若到达 R4 时仍没有 CUDA 主机，R4 标记为
`hardware-blocked`，可先整理不声称完成的基线和测试资产，但不得绕过真实 CUDA Gate 把它记为完成；R5 的结构增长仍必须等待
真实资源数据，不能因为 CUDA 暂缺而改用更多 toy Gate 代替。

#### 16.1.8 立即冻结和归档边界
