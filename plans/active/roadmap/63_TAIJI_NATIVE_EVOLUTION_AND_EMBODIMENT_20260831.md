# Taiji 原生认知训练、经验进化与 Seed 客户端热插拔总路线

> 状态：`E0 complete / E1 complete / E2-A complete / E2-B complete / E3-0 complete / E3-1 complete / E3-2 complete / E3-3 complete / E3-4 complete / E4 complete / E5 active`。本文件自 2026-08-31 起取代原 P7-1a provider artifact 升级作为当前主线，并于 2026-09-01 按“Skill/MCP 可成为知识语料、MCP 执行侧由客户端继承”完成修订。Qwen/provider 质量问题保留为语言器官支线，不再阻塞 Taiji 本体训练；唯一执行事实源 `03_CURRENT_EXECUTION.md` 当前指向 E5：Seed-owned 客户端插件 runtime。

## 1. 路线决策

当前最优路线不是继续扩张 provider 外围，也不是从零训练一个模仿 Transformer 的通用语言模型，而是直接把项目已经拥有的 Taiji 原生神经元、突触、记忆、interaction-group、结构成长、Workbench Outcome 和 capability lifecycle 连接成可训练、可验证、可持续生长的系统。

路线由三条相互独立但可协作的增长链组成：

1. **脑进化**：Taiji 自有突触权重、局部神经元状态、记忆、路由、interaction-group 和结构拓扑依据真实经验更新。
2. **客户端身体进化**：Seed 客户端通过可装载、可卸载、可隔离、可回滚的插件扩展界面、IDE/Workbench、Skill、MCP、可视化和工具能力；客户端就是 Taiji 与用户及环境交互的身体壳层。
3. **语言进化**：provider/codec 只改善输入理解候选与输出表达，不拥有 Goal、记忆、规划、工具选择、结构成长或执行权。

三条链共享同一份可追溯经验和语料来源，但不得共享所有权。Skill/MCP 自身的说明、schema、示例、约束和领域资料可以经过治理后成为 Taiji 知识语料；它们的真实调用与结果成为经验语料。MCP 的连接器、执行接口、权限、资源和 UI 不写入模型，而是形成客户端 capability/plugin 候选，由 Seed 客户端进化继承。客户端插件热插拔改进的是 Seed 客户端，不进入 Taiji 神经网络内部。

## 2. 现有基础与真实缺口

### 2.1 已经存在的原生基础

- `taiji/neuron_region.py` 已实现可训练稀疏突触、持续膜电位/活动/trace/threshold、局部误差学习、神经元增长、lesion 和 checkpoint。
- `taiji/neuron_network.py` 已实现异质区域、跨区域稀疏连接、可学习路由、区域/连接/神经元增长与剪枝、split/merge、资源约束和 checkpoint。
- `taiji/procedural_memory.py` 已能从真实 episodic action trace 内化单步和多步程序性技能；它属于 Taiji 内部记忆，不等于外部 Skill 包。
- `taiji/interaction_groups.py`、`interaction_group_online.py` 和 `interaction_structural_bridge.py` 已有 train/holdout 分区、在线 Outcome credit、失败拒绝、checkpoint、rollback 和结构候选桥接。
- `seed_platform/workbench.py` 已产生绑定 request/intent/call/capability/snapshot 的 `WorkbenchOutcome`，并有真实 IDE、文件、terminal 和 MCP-shaped 执行结果。
- `seed_platform/capability_registry.py` 已有内容寻址 bundle、candidate、shadow、active、replace、retire、rollback、tombstone、资源 reservation、disposer 和 checkpoint。
- `seed_platform/mcp_registry.py` 已有 Seed-owned、内容寻址的 MCP-shaped 工具描述和调用校验。

### 2.2 当前缺口

- Workbench、interaction-group、结构 evidence、MCP 和 capability lifecycle 各自有记录，但没有统一、append-only、可供训练消费的经验合同，也没有把 Skill/MCP artifact 转换为可信知识语料的合同。
- 当前没有 Seed-owned 外部 Skill registry；已有的 procedural skill 是 Taiji 内部学习结果，不能冒充可热插拔 Skill 包。
- `api/routes_agent_mcp.py` 的市场、安装、启动和插件接口仍调用 Legacy `neuroplex.agent_ext.mcp_manager`；它不能直接成为 Taiji 的可信身体或训练数据源。
- 当前 capability registry 记录逻辑 executor/disposer 身份，但还没有连接 Vue 客户端扩展槽、QWebEngine 宿主、后端 capability host、依赖服务、作用域装载、健康状态、状态迁移和原子热替换的完整客户端插件合同。
- 现有在线学习 Gate 多为受限 evaluator/canary；尚缺从统一真实经验到原生突触/记忆/路由更新，再到未见任务收益的标准训练闭环。
- P7-1 已证明 Qwen2.5-0.5B-Instruct 的语义质量不足，但这只阻塞该 provider 成为生产语言入口，不阻塞 Taiji 从结构化 Workbench、Skill 和 MCP 经验学习。

## 3. 从 DeepSeek Harness 采纳什么

参考：

- 官方总览：https://www.deepseek.com/harness/en/
- 插件与生命周期：https://deepseek-harness.github.io/deepseek-harness/en/develop/framework/
- 组合与 HMR：https://deepseek-harness.github.io/deepseek-harness/en/develop/cordis-tutorial/06-composition-and-hmr
- Skill 子系统：https://deepseek-harness.github.io/deepseek-harness/en/reference/subsystems/skills

可采纳的系统原则：

- 客户端 capability 是独立插件，不通过修改 PyQt/QWebEngine 壳或 Taiji 特权核心来接入；
- 插件有稳定 identity、依赖、作用域、装载/卸载和 cleanup 生命周期；
- 依赖消失时自动卸载，依赖恢复后可重新装载；
- 注册行为必须作为 effect 被完整撤销，不能在热更新后泄漏旧注册；
- session/执行过程使用 append-only event stream，支持恢复、分叉、搜索和重放；
- Skill 可以来自多个 provider/scope，近作用域覆盖远作用域，但内容与版本必须可追溯。

明确不照搬的部分：

- 不把 Cordis/JavaScript runtime 直接搬入 Taiji；Seed 只采纳生命周期语义。
- 不把开发态 HMR 等同于生产态自进化。生产态使用候选、shadow、原子切换、checkpoint 和 rollback。
- 不允许模型生成任意源码后自动安装、导入或执行。
- 不把插件热插拔说成模型权重更新；插件扩展 Seed 客户端，Taiji learner 改变认知。
- 不强制所有内部神经器官都插件化；Taiji 核心状态必须保持原生 checkpoint 的一致性和性能。

## 4. 目标架构

```text
用户 / 环境 / Seed 客户端 / IDE / Skill / MCP / Client Plugin
                 |
        +--------+------------------+
        |                           |
        v                           v
 Artifact Corpus Adapter       Runtime Experience Adapter
        |                           |
        +------ provenance / redaction / partition ------+
                 |
                 v
 EvolutionCorpus + EvolutionExperience append-only ledger
         |           |              |
         |           |              +--> 审计 / replay / dataset builder
         |           +-----------------> 客户端缺口与 capability candidate
         +-----------------------------> Taiji native learning
                                                  |
                              +-------------------+-------------------+
                              |                   |                   |
                         突触/路由更新        记忆/Skill 内化       结构候选
                              |                   |                   |
                              +--------- trial checkpoint -----------+
                                                  |
                                  shadow / holdout / retention / lesion
                                                  |
                                      admit / rollback / quarantine
```

### 4.1 认知平面

Owner 仅为 `taiji/`：

- 感知、Goal、WorldState、规划、执行反馈和不确定性；
- 局部突触学习、跨区域 credit 和可塑性调制；
- working/episodic/semantic/procedural memory；
- interaction-group 形成、选择与在线更新；
- 结构 pressure、candidate、shadow、admission、lesion 和 rollback。

### 4.2 经验平面

Owner 为新的 Seed/Taiji 边界合同：

- Skill/MCP artifact 的说明、schema、示例、约束和领域资料由 Seed 解析为内容寻址的 `EvolutionCorpusArtifact`；
- 原始外部执行结果由 Seed 采集、脱敏、内容寻址为 `EvolutionExperience`；
- Taiji 只消费已准入、分区明确、来源完整的经验视图；
- corpus/experience ledger 是 append-only 事实源，dataset 是从 ledger 派生的可丢弃视图；
- 训练、holdout、retention 和安全对抗分区在记录产生时绑定，之后不得重标。

### 4.3 Seed 客户端身体平面

Owner 为 Seed-owned client extension host 与 capability runtime。当前客户端由 PyQt6 原生壳、QWebEngineView、Vue SPA 和 FastAPI backend 组成，因此热插拔必须分层：

- **不可热替换根壳**：`desktop/main.py` 的窗口、托盘、任务栏、进程管理、QWebChannel 安全桥和升级恢复。它保持最小、签名、重启更新，插件不能覆盖。
- **Vue 客户端扩展层**：可热插拔页面、侧栏入口、IDE panel、状态面板、命令、可视化和设置页，通过稳定 slot/route/command API 接入。
- **后端 capability host**：可热插拔 Workbench adapter、数据 provider、Skill、MCP connector 和受控工具；所有执行仍走 policy/approval/Outcome。
- **外部 Skill**：程序性先验、操作流程和领域约束，可在客户端被发现、装载和调用。
- **MCP**：外部传感器、执行器和服务连接，在客户端呈现但由后端隔离执行。
- **client plugin**：把 UI extension、Skill、MCP、capability/service 中的一项或多项组成可装载客户端器官。

插件可以让客户端增加新的工作台页面、编辑器辅助、状态视图、领域工作流、MCP 工具和可视化，但不能替换标题栏/托盘安全逻辑、绕过 API/Workbench 或直接访问 Taiji 内存。

### 4.4 治理平面

Owner 为 registry、policy、checkpoint 和 gate：

- provenance、内容 digest、签名/来源、权限、依赖和资源预算；
- staged/shadow/active/quarantine/rollback 生命周期；
- train/holdout 防泄漏、污染检测、凭据脱敏和 prompt-injection 标记；
- 每次认知或身体变更的 parent/child checkpoint 与精确 rollback。

## 5. 统一进化语料与经验合同

E1 同时定义 `EvolutionCorpusArtifact` 与 `EvolutionExperience`。前者描述“Skill/MCP 本身能教给 Taiji 什么”，后者描述“实际使用后发生了什么”。二者都不是大而全的自由 JSON，而是版本化、内容寻址、append-only 的事实记录。

### 5.1 必需字段

| 范围 | 字段 |
|---|---|
| 身份 | `experience_id`、`format_version`、`previous_event_digest`、`event_digest` |
| 来源 | `source_kind`、`source_id`、`source_version`、`source_digest`、`scope_id` |
| 血缘 | `episode_id`、`request_id`、`intent_id`、`call_id`、`parent_checkpoint_digest` |
| 认知上下文 | `percept_digest`、`goal_digest`、`world_state_digest`、`plan_digest`、`uncertainty` |
| 行动 | `capability_id`、`capability_snapshot_id`、`arguments_digest`、`approval_id` |
| 结果 | `status`、`success`、`result_digest`、`error_code`、`reward_components`、`user_correction_digest` |
| 资源 | `latency_ms`、`cpu_ms`、`memory_bytes`、`output_bytes`、`side_effect_count` |
| 数据治理 | `partition`、`taint_flags`、`redaction_revision`、`retention_policy` |
| 客户端绑定 | `client_snapshot_id`、`skill_digest`、`mcp_server_digest`、`mcp_schema_digest`、`plugin_digest`，未使用时为空 |

### 5.2 语料来源类型

- `skill_artifact`：Skill 标题、适用条件、步骤、示例、反例、约束、参考资料和版本血缘。
- `mcp_artifact`：server/tool 描述、JSON schema、示例、错误语义、资源/权限合同和领域文档。
- `client_plugin_artifact`：客户端页面/命令/能力说明和用户可见 affordance；不包含可执行源码。
- `verified_domain_material`：Skill/MCP 明确引用且允许训练使用的领域资料。

语料适配器把 artifact 拆成 `knowledge`、`procedure`、`affordance`、`constraint`、`example`、`counterexample` 六类单元，保留原始 artifact digest、chunk digest、许可/用途、scope、语言、置信度和依赖。相互冲突的来源并存，不在采集阶段静默合并成唯一真相。

### 5.3 经验来源类型

- `workbench`：真实 IDE/文件/terminal/本地 MCP-shaped Outcome。
- `skill`：外部 Skill 的发现、选择、调用步骤、完成/失败、用户修正和版本。
- `mcp`：server/tool schema、调用、结果、超时、断连、审批与重试。
- `client_plugin`：客户端 UI/能力装载、依赖解析、shadow、健康、卸载、回滚和资源变化。
- `user_correction`：用户对目标、计划、执行结果或解释的明确纠正。
- `provider`：语言候选的成功/失败，只作为表达或语义证据，不获得执行所有权。

### 5.4 EvolutionCorpusArtifact 必需字段

| 范围 | 字段 |
|---|---|
| 身份 | `corpus_id`、`format_version`、`artifact_digest`、`chunk_digest` |
| 来源 | `source_kind`、`source_id`、`source_version`、`publisher`、`scope_id` |
| 内容 | `unit_kind`、`content_digest`、`relation_digests`、`language`、`confidence` |
| 能力语义 | `capability_semantics`、`input_schema_digest`、`output_schema_digest`、`constraint_digests` |
| 治理 | `license/use_policy`、`taint_flags`、`redaction_revision`、`partition`、`retention_policy` |
| 血缘 | `supersedes_digest`、`dependency_digests`、`admission_revision` |

模型训练读取的是通过 corpus admission 的派生单元，不直接读取插件目录或 MCP server 的任意文件。

### 5.5 不进入训练的内容

- 未脱敏凭据、token、环境变量、私有路径明文和超出 retention policy 的内容；
- 未验证来源、用途不允许训练或带未处理 prompt injection 的 Skill/MCP 内容；
- holdout/retention 标签及其可逆推出信息；
- 插件源码、MCP server 可执行文件、shell command、安装脚本和动态 import 路径；这些属于客户端器官 artifact，不属于知识语料；
- evaluator 预期答案、最终工具绑定或人工分数伪装成模型观测；
- 只有“成功”结论、没有输入/行动/结果/血缘的日志。

## 6. Skill 作为知识语料、经验来源与客户端部件

### 6.1 两种 Skill 必须区分

1. **外部 Skill 包**：可装载说明、流程和资源，属于身体层，必须版本化、内容寻址、作用域隔离。
2. **Taiji procedural memory**：从多次真实经验内化出的慢速程序性记忆，属于认知层，保存于 native checkpoint。

外部 Skill 可以成为模型知识和程序性语料，但不以原始文件直接覆盖权重。正确流程是：

```text
发现 Skill -> 校验来源/用途 -> EvolutionCorpusArtifact -> corpus admission
          \-> scoped mount -> 真实调用 -> Outcome/用户修正 -> EvolutionExperience
两类证据汇合 -> knowledge/procedural memory 候选
-> holdout/lesion/retention -> 内化或拒绝
```

### 6.2 Skill 经验记录

- Skill artifact digest、provider、版本、scope 和依赖；
- 被选择的上下文以及未被选择的候选；
- 实际读取的 Skill section/步骤 digest，避免记录整包未使用内容；
- 每一步 ActionIntent、Workbench/MCP 调用与 Outcome；
- 用户中断、修正、跳步、重试和最终验收；
- 资源消耗、完成率、泛化任务族和 lesion 后变化。

Skill artifact 本身还要形成知识语料：适用条件进入 concept/affordance 学习，步骤进入 procedural 候选，约束/反例进入拒绝与校准学习，参考资料进入 semantic knowledge 候选。真实 Outcome 决定这些语料应被增强、降权、冲突标记还是拒绝，不能仅因 Skill 自述“正确”就获得高权重。

### 6.3 Skill 内化 Gate

- 至少有多个独立成功 episode，且不能只是同一模板重复；
- 相对“不使用 Skill”基线有显著成功率或样本效率提升；
- 移除外部 Skill 后，Taiji procedural memory 仍能在未见变体完成任务；
- 错误 Skill、过期 Skill 和冲突 Skill 必须降低置信或触发澄清，不能污染旧能力；
- 内化失败不删除 Skill，只保留 rejected candidate 与原因。

## 7. MCP 作为经验来源与外部器官

MCP 同时提供两类可内化内容，但不是认知主体：

1. **认知部分**：tool/server 描述、schema、示例、约束、错误语义、领域资料和成功/失败轨迹，转换为 `EvolutionCorpusArtifact + EvolutionExperience`，可训练 Taiji 的知识、affordance、规划和程序记忆。
2. **客户端器官部分**：连接器、协议握手、工具调用、权限、资源、凭据边界、UI 和 disposer，转换为 `ClientCapabilityInheritanceCandidate`，由 Seed 客户端插件生命周期继承。

这里的“硬件部分”在当前软件项目中指身体侧可执行器官；若 MCP 对接真实设备，物理设备仍在客户端之外，Seed 继承的是经过治理的驱动/连接/能力接口。外部 MCP server 必须先转换成 Seed-owned capability candidate，才能被 Taiji 看见和使用。

```text
MCP discover
-> server identity + docs/schema/examples
-> EvolutionCorpusArtifact -> Taiji knowledge candidate
-> ClientCapabilityInheritanceCandidate
-> permission/resource/policy review
-> staged connection
-> shadow schema/health probe
-> active scoped mount
-> Workbench policy/approval execute
-> Outcome + EvolutionExperience
```

每次 MCP 经验必须绑定 server digest、tool schema digest、arguments digest、返回 digest、超时/错误、审批、环境前后状态和 capability snapshot。MCP 文本结果默认带 `untrusted_external_content`，只能作为观测，不能作为系统指令或自动安装请求。

### 7.1 MCP 内化的双产物

| 产物 | Owner | 内容 | 准入结果 |
|---|---|---|---|
| `CognitiveInternalizationArtifact` | Taiji | 知识、关系、程序、affordance、约束、失败模式 | semantic/procedural/world/route update candidate |
| `ClientCapabilityInheritanceCandidate` | Seed 客户端 | connector、schema、executor/disposer、permission、UI、resource、health | client plugin/capability candidate |

两条产物共享 MCP artifact digest 和验证报告，但独立准入、独立 checkpoint、独立 rollback。认知部分通过不代表客户端执行器安全；客户端 capability 通过也不代表模型已学会何时使用。

### 7.2 客户端继承等级

- L0 `referenced`：只认识 MCP 文档/schema，不连接 server。
- L1 `mounted`：客户端插件挂载外部 MCP connector，仍依赖外部 server。
- L2 `adapted`：Seed-owned adapter 固化 schema/policy/UI，外部 server 仍承担执行。
- L3 `native-capability`：在许可、安全和独立 oracle 允许时，由 Seed-owned executor 实现等价能力，不再依赖 MCP protocol；必须与原 MCP 做差分/回归/资源/rollback Gate。

不能通过复制未知 MCP 源码或模型自动生成 executor 直接进入 L3。L2→L3 是客户端工程迁移，不是模型训练动作。

当前 `api/routes_agent_mcp.py -> neuroplex.agent_ext.mcp_manager` 的 `/api/mcp/*` 仍是 Legacy 兼容能力，不进入 E1 verified corpus/experience ledger；原 `/api/plugins/marketplace`、marketplace refresh 和 workspace upload 已由 E5-2 统一退役为 Seed-owned 410 tombstone。真实第三方 MCP/plugin 仍需等 E5-3 runtime canary 与后续 E6 Gate。

## 8. Seed 客户端插件合同与热插拔生命周期

### 8.1 ClientPluginManifest

每个客户端插件至少声明：

- `plugin_id`、`version`、`artifact_digest`、`publisher`、`signature_status`；
- `provides`、`requires`、依赖版本范围和 optional dependency；
- `ui_extensions`：route、sidebar、panel、command、settings、visualization slot 及其静态 artifact digest；
- `backend_extensions`：所含 Skill/MCP/capability/service 的 digest；
- `scope`：global、workspace、task 或 session；
- `effect`、`risk`、permissions、network/filesystem/process policy；
- CPU、memory、latency、output、side-effect 和并发预算；
- UI mount/unmount identity、backend executor/disposer identity、health probe、state schema 和 migration identity；
- shadow/holdout/rollback Gate 与 quarantine 原因。

manifest 只描述可执行身份，不包含任意源码、shell、module path 或自动安装指令。第一方受信任 UI 扩展使用内容寻址的静态 ESM bundle；第三方 UI 默认运行在 sandboxed iframe/WebView 中，只能通过版本化 host bridge 请求能力，不能直接访问主页面 store、QWebChannel 或文件系统。

### 8.2 生命周期

```text
discovered
  -> verified
  -> resolved
  -> installed
  -> staged
  -> shadow
  -> ready
  -> active
  -> degraded -> draining -> detached
                       \-> rolled_back

任意验证/安全失败 -> quarantined
```

- `discovered`：只记录 artifact，不进入客户端菜单、路由或能力表。
- `verified`：digest、签名、兼容版本和 manifest schema 通过。
- `resolved`：前后端依赖、权限和资源可满足。
- `installed`：artifact 写入版本目录，但未装载。
- `staged`：UI 和 backend 分别装载到隔离 scope，不向用户或 Taiji 发布。
- `shadow`：隐藏 UI 完成 handshake/render/cleanup，后端完成健康、schema、资源和无副作用/可逆性测试。
- `ready`：UI snapshot 与 capability snapshot 均已准备，可以原子提交。
- `active`：客户端 slot/route/command 与 backend capability 同时可见；调用仍受 Workbench policy。
- `degraded`：健康或依赖异常，只停止新调用，不丢审计。
- `draining`：等待在途调用结束并执行 disposer。
- `detached`：注册、子插件、资源 reservation 和 service effect 已撤销。
- `rolled_back`：同时恢复上一 client extension snapshot、capability snapshot 和插件状态。
- `quarantined`：来源、schema、行为或资源异常，不自动复活。

### 8.3 开发热更新与生产热替换

- 开发态可在受控客户端进程使用 HMR 语义：卸载旧 UI route/slot/command 和 backend effect、装载新版本、验证无监听器、路由、store、timer、WebSocket 或 executor 注册泄漏。
- 生产态不执行源码 HMR；使用旧版本 active、新版本 shadow 的 blue/green 切换，并对 `client_snapshot + capability_snapshot` 做两阶段原子提交。
- 任一依赖消失时，消费者自动进入 degraded/draining；依赖恢复后重新走 shadow，不能直接恢复 active。
- 插件状态迁移失败时恢复旧 UI、旧 backend、旧 snapshot；不能只恢复 registry 而遗留新页面、菜单、事件监听或缓存。
- PyQt 原生根壳、任务栏/托盘、QWebChannel 和 backend worker 的二进制升级需要安全重启，不伪装成无重启热插拔。

## 9. 脑—客户端协同进化的归因规则

每个失败 episode 先分类，再只允许一种主要干预进入 trial 分支：

| 失败来源 | 首选干预 |
|---|---|
| 已有能力但选择错误 | interaction-group、route 或 planner credit |
| 同类输入预测持续错误 | 局部突触/世界模型更新 |
| 重复多步流程且执行正确 | procedural memory 内化 |
| 工作空间或区域容量不足 | 神经元/区域/连接结构候选 |
| 当前客户端 affordance 根本不存在 | client capability/plugin candidate |
| 需要外部信息或执行器 | MCP capability candidate |
| 只是不够可读或语言歧义 | provider/codec 候选，不改执行认知 |

不能在同一 trial 同时新增客户端插件、扩大神经元并更新路由后再把全部收益归给“自进化”。每个变更必须有 no-change、weight-only、memory-only、route-only、structure-only 或 client-plugin-only 对照；通过后才允许下一层组合。

## 10. 分阶段实施路线

### E0：路线与所有权收敛

状态：**已完成（2026-09-01）**。

交付：统一术语、现状缺口、DeepSeek Harness 采纳边界、数据合同、生命周期、阶段 Gate 和唯一下一步。

退出 Gate：只有 `03_CURRENT_EXECUTION.md` 决定当前动作；阶段总计划、implementation status 和 manifest 均一致指向 E1，P7-1a 降为语言支线。

### E1：统一进化语料/经验合同与 checkpoint 前置 Gate

目标：不训练、不装插件，先建立可信经验事实源。

实现范围：

- 新建独立 `EvolutionCorpusArtifact`、`EvolutionExperience`、统一 append-only ledger 和 canonical digest；
- 先接 `WorkbenchOutcome` 适配器，再用确定性 Skill/MCP/client-plugin lifecycle fixture 验证合同，不接 Legacy manager；
- append-only、去重、previous digest 链、partition 锁定、taint/redaction、checkpoint roundtrip；
- 损坏、截断、重排、重复、跨 partition 改写和 parent checkpoint drift 均在 learner mutation 前拒绝；
- 保存进程 A 的 ledger + Taiji parent checkpoint，进程 B 恢复后追加一条经验，digest 和序号连续。

Gate：

- checkpoint 可写、关闭进程可恢复、继续写入、损坏 fail-closed；
- 同一 Outcome 幂等，内容不同但 ID 相同拒绝；
- holdout/retention 不能被训练 dataset builder 读取；
- 未脱敏 secret fixture 不能入账；
- corpus/experience ledger 不改变突触、记忆、拓扑或 capability snapshot。

退出物：E1 report、manifest 更新、定向测试和 checkpoint fixture。**已完成（2026-09-01）。** report 的 8 项 Gate 全部通过，定向测试 5/5 通过；E1 未触碰 Taiji 权重、拓扑、客户端 UI 或 Legacy MCP。

### E2：真实 Skill/MCP 语料与经验适配器

目标：把 Skill/MCP artifact 本身转换为知识语料，并把真实 Workbench、Skill/MCP 调用和客户端插件 lifecycle 投影为经验合同。

**当前执行阶段。** E1 ledger 合同已冻结为输入边界；E2-A/B 已完成来源适配器和 Seed-owned registry/lifecycle 的确定性状态流；E3-0 已完成训练 checkpoint 预检；E3-1 已完成 route/interaction credit；E3-2 已完成动态 action kind 的 procedural memory intake；E3-3 已完成真实 `WorldTransition` 进入现有局部世界模型；E3-4 已完成多 seed fixed-capacity 对照和持续失败触发前置 Gate：当前 fixed-capacity holdout/retention 通过，结构候选不得生成；E4 已完成受治理 Skill/MCP artifact 知识向 Taiji-owned semantic/procedural/affordance artifact 的内化，并证明外部描述关闭后仍可读取内部状态。E5-0/E5-1/E5-2 已完成 Seed-owned client extension host 的合同、API/Vue 接线和 Legacy surface 退役；当前进入 E5-3：在真实 slot 中验证本地只读 extension 的挂载、卸载、失败回收与 snapshot 回滚；不把 digest-only 结果转成伪状态，也不把 Legacy MCP manager 接入 verified ledger。

历史阶段证据摘要：E2-A 已通过 [source adapter report](../../../reports/taiji_w7_e2_source_adapters_20260901.json)，E2-B 已通过 [source registry report](../../../reports/taiji_w7_e2b_source_registry_20260901.json)，E3-0/E3-1/E3-2/E3-3/E3-4/E4 的对应 Gate 也已通过；当前阶段事实以本文件后面的 E5-0/E5-1/E5-2/E5-3 条目和 `03_CURRENT_EXECUTION.md` 为准。

顺序：

1. Workbench Outcome + user correction；
2. Seed-owned external Skill registry：artifact corpus + scoped invocation；
3. Seed-owned MCP registry：docs/schema/example corpus + connection/tool lifecycle；
4. client extension/capability plugin artifact 与 lifecycle。

Gate：每个来源至少包含成功、失败、拒绝、超时/取消、重启恢复和版本变化；相同语义的来源字段一致，来源特有字段只进入 client binding；Legacy MCP 记录不能进入 verified dataset。

### E3：Taiji 本体的第一条直接训练闭环

目标：第一次让已准入的 Skill/MCP 知识语料和统一真实经验直接更新 Taiji-owned 参数/状态，而不是训练 Qwen/provider。

训练对象按最小充分原则依次开放：

1. interaction-group/route credit；
2. procedural memory；
3. world prediction 与局部突触；
4. 只有固定容量对照失败时才提出结构增长。

训练前必须通过 E1 checkpoint Gate。每次训练生成 parent、trial、admitted 三类 checkpoint，保存 optimizer/local learner state、随机状态、dataset digest、partition manifest、经验 cursor 和资源 ledger。

首个任务族使用真实 Workbench 的 workspace/editor/MCP-shaped/terminal 受控组合，但 train 与 holdout 按任务族和环境状态隔离，不复用同一模板的表面改写。

Gate：相对 frozen/no-learning、replay-only 和随机更新对照，未见任务成功率、恢复能力或样本效率有可复现提升；旧任务 retention 达标；lesion 移除对应收益；rollback 精确恢复 parent。

### E4：Skill/MCP 知识与程序内化

目标：证明 Skill/MCP 不只是 prompt 或工具目录，而能由 artifact 语料与真实 Outcome 共同形成 Taiji semantic/procedural/affordance knowledge。

- 选择一个多步 IDE Skill 和一个 MCP tool family，保留完整 artifact、仅调用轨迹、无 artifact/Skill/MCP 和错误/过期 artifact 对照；
- 记录调用步骤、用户修正和 Outcome；
- 训练只读取 train partition；
- 在未见项目结构中关闭外部 Skill/MCP 描述注入，验证内化后的 Taiji 是否仍知道领域关系、何时选择能力和如何组织步骤；
- 对 procedural memory lesion 后收益应消失，旧程序记忆应保持。

Gate：知识问答不是唯一指标；必须同时通过未见任务规划/能力选择、内部化收益、错误 artifact 抗污染、checkpoint continuation、跨 scope 不泄漏和 lesion 因果性。

状态：**已完成（2026-09-01）**。E4 使用 Skill/MCP artifact 与对应成功 Outcome 的 train/holdout/retention 分区，分别更新 Taiji-owned semantic、procedural 和 affordance 器官；provider、外部描述、客户端 executor 均未进入训练输入或模型输出。Gate 结果见 [E4 artifact internalization report](../../../reports/taiji_w7_e4_artifact_internalization_20260901.json) 与 [E4 tests](../../../tests/taiji_native/test_artifact_internalization.py)：10/10 检查通过，semantic holdout loss 为 `0.0000399`，procedural holdout/retention 为 `1.0`，lesion holdout 为 `0.25`，affordance holdout MSE 约为 `1.15e-12`；检疫、身份隔离、外部描述关闭和 checkpoint continuation 均 fail-closed。

### E5：Seed-owned 客户端插件 runtime

目标：在现有 PyQt + QWebEngine + Vue + FastAPI 客户端上建立统一 extension host，并在 capability registry 上增加 service dependency、scope、health、state migration 和递归 disposer，不重写 registry。

- 建立 `ClientPluginManifest`、client extension snapshot 和 lifecycle ledger；
- Vue 提供稳定的 route/sidebar/panel/command/settings/visualization slots；QWebEngine/PyQt 根壳保持保护域；
- UI 与 backend 使用两阶段 prepare/commit，任一侧失败都不发布半个插件；
- 先接纯本地、只读、可独立 oracle 的插件；
- 完成 dependency loss/recovery、开发 HMR effect cleanup、生产 blue/green、in-flight draining、状态迁移失败 rollback；
- 将 `routes_agent_mcp.py` 从 Legacy manager 迁移到 Seed-owned runtime；不能安全迁移的市场/安装接口进入 tombstone。

**E5-0 已完成（2026-09-01）**：`seed_platform/client_extension_host.py` 已建立声明式 `ClientPluginManifest`、内容寻址 `ClientExtensionSnapshot`、可配置 slot/protected-shell policy、两阶段 prepare/commit、blue/green 状态迁移、依赖健康/quarantine、in-flight draining、递归 disposer、rollback、checkpoint/tamper 校验和生命周期审计。E5-0 [Gate report](../../../reports/taiji_w7_e5_client_extension_host_20260901.json) 的 12/12 检查通过；它只证明 client-body contract 和可回滚状态机，不证明 Vue/API/Workbench 已接线，不接 Legacy manager，也不执行插件源码。

**E5-1 已完成（2026-09-01）**：`api/routes_client_extensions.py` 已把 host 接入原生 API，`frontend/src/composables/nativeApi.js` 与 `useClientExtensions.js` 只转发内容寻址 snapshot、两阶段 prepare/commit、依赖和 rollback；`App.vue` 注入 client-body state 并在启动时读取 snapshot。E5-1 [Gate report](../../../reports/taiji_w7_e5_1_client_snapshot_integration_20260901.json) 的 8/8 检查通过，API 与 Workbench capability snapshot 绑定、stale snapshot 拒绝、声明式 slot projection 和 Taiji/Legacy 边界均保持。

**E5-2 已完成（2026-09-01）**：`api/routes_plugins.py` 已成为显式 410 tombstone，所有旧 `/api/plugins`、enable/disable/delete/install、marketplace、marketplace refresh 和 upload 入口均 fail-closed 指向 `/api/client-extensions`；`routes_agent_mcp.py` 与 `routes_agent_workspace.py` 的重复旧路由已清除，前端无旧 `/api/plugins` 引用。E5-2 [Gate report](../../../reports/taiji_w7_e5_2_legacy_plugin_surface_20260901.json) 的 9/9 检查通过；未删除不属于本阶段的 `/api/mcp/*` Legacy 兼容能力，也未接入真实第三方 MCP/plugin。当前进入 E5-3：Seed-owned 本地只读 extension 的真实 slot runtime canary。

Gate：旧 client/capability snapshot 可恢复，旧路由/组件/监听器/executor 注册无泄漏，在途调用不丢失，资源 reservation 归还，异常插件 quarantine，Taiji cognition checkpoint 不被插件覆盖。

### E6：MCP 客户端器官继承

目标：把已在 E2/E4 形成认知语料的 MCP，独立转换为客户端器官候选；认知和执行能力在此重新汇合，但不混淆准入。

- discover 同时产生 corpus candidate 与 `ClientCapabilityInheritanceCandidate`，不自动连接/安装；
- schema、权限、网络、凭据、资源和 side effect 进入 policy；
- shadow 只做健康/schema/只读探测；
- active 后通过 Workbench approval 执行；
- active 后的新成功/失败经验回到 E1 ledger，持续校正 E3/E4 learner；
- 至少完成 L1 mounted，并评审一个满足许可/独立 oracle 的 L2 adapted；L3 native-capability 只有差分 Gate 证明等价且资源收益为正才实施。

Gate：schema drift、server substitution、prompt injection、断连、超时、部分返回、凭据泄漏和 rollback 全部 fail-closed。

### E7：脑—客户端协同选择器

目标：Taiji 能根据经验判断“应该学习已有能力，还是向 Seed 客户端申请新能力”，但不能直接安装插件。

输出只能是下列候选之一：`weight_update`、`memory_consolidation`、`route_update`、`structure_candidate`、`client_capability_candidate`、`clarify_or_stop`。

Gate：

- 已有能力可解决时不申请插件；
- 缺少 affordance 时不靠增加突触伪造执行器；
- 语言失败不触发结构增长；
- 资源不足时降级/停止而不是无限增长；
- client-plugin-only 与 brain-only 对照能够解释收益归属。

### E8：长期持续进化与数据飞轮

目标：跨重启、跨版本、跨项目累计经验，同时防止灾难性遗忘和数据污染。

- bounded replay、优先级采样、失败与纠正样本平衡；
- 周期性 sleep/play consolidation；
- drift、retention、calibration、resource efficiency 和插件健康监控；
- 经验压缩保留 digest/provenance，不改写原始 ledger；
- 回退单个认知更新或单个客户端插件，不回滚无关进化。

Gate：多周期净能力收益、旧能力保持、checkpoint 大小/延迟预算、污染隔离和分支合并策略通过。

### E9：规模与 CUDA

状态：当前主机 `hardware-blocked`，不阻塞 E1–E8 的 CPU/native 正确性。

获得 CUDA 主机后复跑相同 workload、数值一致性、checkpoint 跨设备恢复和 profiler；只有真实热点证明需要时才实现 fused/sparse kernel。规模增长由收益/资源曲线驱动，不以参数数量本身作为进化指标。

## 11. 统一 Gate 与核心指标

### 11.1 能力指标

- 未见任务成功率、部分完成率和恢复成功率；
- 达到目标所需真实 episode 数和用户纠正次数；
- 跨项目/任务族 transfer；
- world prediction error、规划 calibration 和错误停止质量；
- Skill 内化后关闭外部 Skill 的保持率；
- MCP/客户端插件相对无插件基线的因果收益。

### 11.2 安全与保持指标

- old-task retention regression；
- holdout 泄漏、taint 命中、secret redaction 和 prompt-injection 拒绝；
- lesion 后目标收益下降且无关能力保持；
- rollback 后 checkpoint、topology、client snapshot、registry、plugin state 和 resource ledger 一致；
- provider、Skill、MCP、frontend 均不能生成最终执行权。

### 11.3 资源指标

- 参数/突触/神经元/连接增量；
- checkpoint 大小与恢复时间；
- CPU/GPU、memory、latency、I/O 和输出字节；
- 每单位资源带来的 holdout gain；
- 客户端插件的 UI mount/render、空闲、backend 调用和卸载资源；
- 结构增长相对 weight/memory/route-only 对照的净收益。

## 12. 禁止事项

- 不把外部 Skill 文本、MCP 返回或插件说明直接拼接为训练真值。
- 不允许模型生成源码、shell、依赖或 manifest 后自动安装/执行。
- 不允许 provider、Skill、MCP 或 frontend 拥有 Taiji Goal、ActionIntent、policy 或结构 admission。
- 不允许 holdout、retention、evaluator 预期答案进入训练。
- 不允许同一 trial 同时改变 Taiji 认知和客户端插件后声称单一因果收益。
- 不通过硬编码操作词表、神经元角色、任务 ID 或 prompt 分支提高 Gate。
- 不因本机无 CUDA 而停止原生正确性、checkpoint 和 CPU 小规模训练。
- 不在 checkpoint 写入/恢复/损坏回退未通过前启动长训练。
- 不把插件数量、参数数量或神经元数量本身当成智能增长。

## 13. 文件所有权建议

E1–E7 优先按现有 owner 增量接线，避免继续膨胀 `taiji/adapter.py` 和 `api/seed_runtime.py`：

- `taiji/evolution_experience.py`：Taiji 可消费的 corpus/experience 合同和训练视图；
- `seed_platform/evolution_ledger.py`：artifact corpus 与 runtime experience ledger、脱敏、内容寻址和 checkpoint；
- `seed_platform/skill_registry.py`：外部 Skill artifact/scope/lifecycle；
- `seed_platform/client_plugins.py`：客户端插件 manifest、backend dependency、health 和 lifecycle；
- `frontend/src/extensions/`：受保护的客户端 extension host、slots、sandbox bridge 和 snapshot 投影；
- `api/routes_client_extensions.py`：只读状态、安装审批、activate/rollback 的 API projection；
- `desktop/main.py`：保持最小保护根壳，只提供签名资产、QWebEngine/QWebChannel 边界和安全重启，不加载任意插件源码；
- `seed_platform/mcp_registry.py`：扩展 MCP server/tool identity，不承担认知；
- `taiji/evolution_credit.py`：脑更新类型选择和归因；
- `api/seed_runtime.py`：只保留 facade/装配，不成为 ledger 或 learner owner；
- `api/routes_agent_mcp.py`：迁移到 Seed-owned client plugin runtime 后仅保留 API projection。

具体文件名可在实现时微调，但 owner 和依赖方向不可改变：`taiji/` 不导入 `seed_platform`，Seed 通过版本化 DTO 把已治理经验投影给 Taiji。

## 14. CI、提交与决策节点

- 每个 E 阶段先运行新合同的定向单元测试与 evaluator，再运行受影响回归；全量 CI 按当前用户决定在阶段收口时统一修复，未运行不标记为通过。
- E1、E3、E5 是强 checkpoint Gate；任一损坏恢复或 lineage 错误立即停止功能推进。
- E3 完成后是第一个需要讨论的决策点：若直接原生训练没有超过 frozen/route/memory 对照，应先修学习目标和 credit，不扩大参数规模。
- E5 完成后是第二个决策点：只有隔离、状态迁移和 rollback 可靠，才允许接真实第三方 MCP/plugin。
- 本轮提交 E4 与 E5-0 实现、定向测试、evaluator、report 和计划同步；不推送、不运行全量 CI。全量 CI 仍按用户决定在阶段收口时统一修复。

## 15. E1 首个执行切片

`03_CURRENT_EXECUTION.md` 当前指向 **E5-3：Seed-owned 本地只读 extension 的真实 slot runtime canary**；E1 已完成并作为全部后续训练与客户端继承的事实源，E5-0/E5-1/E5-2 已分别冻结 host contract、完成 API/Vue 接线并退役 Legacy plugin surface。

它优先于真实第三方 MCP 和插件 HMR，因为必须先在受控本地 slot 中证明 Seed-owned extension 的挂载、卸载、失败回收和 snapshot 回滚；E5-3 不修改 Taiji cognition checkpoint，也不把插件执行权写入模型。
