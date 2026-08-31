# Seed / Taiji 实现事实参考

> 事实快照：2026-09-01。本文件只描述当前代码事实、能力边界和证据入口，不决定执行顺序。当前动作见 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md)，E1–E9 阶段见 [63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md](../active/roadmap/63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md)。

> 最新状态覆盖：P7-1 真实 Qwen 语义质量 Gate 已完成测量但未通过；Qwen2.5-0.5B-Instruct 仅保留为实验/回退 provider。主线已重定向为 E1：先建立 Skill/MCP 知识语料、Workbench/Skill/MCP 经验和客户端插件 lifecycle 共用的内容寻址 ledger 与 checkpoint Gate，再直接训练 Taiji 本体。provider 升级降为语言器官支线。

## 1. 身份、所有权与依赖

- `taiji/` 是 Taiji-owned substrate 与认知纵切片。正式 Taiji 路径不导入 `seed`、`seed_platform`、`neuroplex` 或 Transformers。
- `seed/`、`api/`、`seed_platform/`、`frontend/` 和桌面壳负责产品/runtime、Workbench、policy、provider 装配、UI 和外部副作用，不得隐藏实现 goal、memory、plan 或 tool choice。
- `neuroplex/` 是冻结的 Legacy Transformer 对照，只允许离线 benchmark 与显式兼容；不能成为 Taiji runtime 的认知主体。
- Qwen/Transformers 位于 `seed/language_provider.py` 的语言 provider 边界。provider 可以表达 ContentPlan，不能拥有 intent、tool、memory 或结构成长决策。
- capability、programming language 和 MCP 均由 backend registry/snapshot 提供；前端不得维护第二份能力真相。

## 2. 当前代码形成的能力

### Taiji 原生认知构件

- 输入边界、perception、world state/event、workspace、executive、planning、generation 和 homeostasis 已有 Taiji-owned 合同与可执行纵切片。
- working、episodic、semantic、procedural memory 和 world learning 已有原型；状态进入 native checkpoint。
- `neuron_network.py`、`neuron_region.py`、`sparse.py`、`local_learning.py` 与 cross-region 路径提供向量化神经元群、稀疏连接、局部学习和持续状态；没有把一个 Transformer checkpoint 当作一个神经元。
- interaction-group 从真实 trace、贡献、冲突、恢复和 lesion 形成，不按 `zh/en/code/math` 或固定“规划神经元”角色硬编码知识。
- structural growth 已覆盖 candidate、pressure、arbitration、validation、measurement、admission、rollback、retention、lineage、checkpoint continuation 和 artifact consumption policy。

### Workbench 与 IDE

- Workbench 已注册文件/目录读取、搜索、stat、编辑器打开、语言识别与切换、结构化 patch/create/rename/delete/undo、terminal、diagnostics、test/build 和 MCP-shaped 能力。
- programming-language registry 综合扩展名、shebang、内容、manifest、邻近文件、toolchain 和 LSP 证据；`editor.set_language` 是可逆 UI 动作，高置信可准入，低置信或冲突返回 `ask_user`，user override 优先。
- side-effecting 动作经过 preview、policy、approval、before/after digest、原子执行和 Outcome；有限 loop、successor graph、recovery handoff/portfolio 与 checkpoint continuation 已存在。
- capability registry 已替代旧工具名 `elif` 分派，支持内容寻址 bundle、snapshot、candidate、shadow、resource reservation、replacement、rollback 和 tombstone。

### 语言、provider 与客户端

- `native-readable` 是无外部 provider 时的默认可读表层；`structured-stub` 只保留为显式调试 codec。
- language provider artifact 具备 manifest/digest、registry、activation、rotation、health observation、watchdog、previous/native rollback 和 checkpoint 状态合同。
- 前端从 native facade 读取 runtime、provider、homeostasis、training、knowledge 和 Workbench 状态。
- 前端 live 页面没有 HF/GGUF/Transformer 模型格式切换；隐藏兼容 API 与配置层仍保留部分 tombstone/迁移字段。

### R5A / R5B / R5C

- R5A：grounded Outcome DTO、train-only replay、内容寻址、原生局部 learner、holdout/retention/lesion、checkpoint lineage 和 candidate-only 删除评审已落地。
- R5B：capability bundle/candidate、digest-only shadow、approval、resource reservation、disposer audit、registry dispatch、replacement/rollback 和 checkpoint 已落地；没有纯计算 executor 因缺真实候选而被强行实施。
- R5C-S0–S52：长期 Workbench evidence、pressure、candidate batch、独立 measurement owner、measured artifact、原子 admission/rollback、lineage retention、磁盘 restart、外部 content-addressed store、只读 audit/reconciliation、measurement sidecar、partial recovery、strict verified bridge 和显式 artifact consumption policy 已落地。
- S52 已将 `require_verified_measurements: bool` 收敛为内容寻址、可 checkpoint、可审计的 policy；新运行时默认 `verified-only`，历史 replay 需显式 `legacy-compatible`。

### E0 原生进化与客户端身体边界

- 现有代码尚无 `EvolutionCorpusArtifact`/`EvolutionExperience` 统一 ledger；Workbench Outcome、interaction trace、结构 evidence 和 capability lifecycle 仍是分散事实源。
- 外部 Skill registry 尚未实现；`taiji/procedural_memory.py` 中的 skill 是模型从 episode 内化出的程序记忆，不等于外部 Skill artifact。
- Skill/MCP artifact 的说明、schema、示例、约束和领域资料将成为受治理的模型知识语料；真实调用、失败和用户修正将成为经验语料。
- MCP 内化必须生成两个独立候选：Taiji-owned 认知内化 artifact，以及 Seed 客户端的 connector/executor/permission/resource/UI capability 继承候选；二者共享 provenance，但独立准入、checkpoint 和 rollback。
- 客户端当前是 PyQt6 + QWebEngineView + Vue SPA + FastAPI backend，尚无 extension host。后续热插拔只覆盖 Vue route/sidebar/panel/command/settings/visualization 和后端 Workbench/Skill/MCP capability；`desktop/main.py` 根壳、托盘、任务栏、QWebChannel 和进程管理只能安全重启更新。
- `api/routes_agent_mcp.py` 仍指向未实现的 Legacy `neuroplex.agent_ext.mcp_manager`；这些接口数据不能作为 verified 训练语料或客户端插件证据。
- 详细阶段 E1–E9 和 DeepSeek Harness 采纳边界见 [63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md](../active/roadmap/63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md)。

## 3. 当前不能声明的能力

- 普通 `/api/chat/stream` 的旧路径仍不能在没有预制结构化 intent 时自主完成 IDE 任务；新的 P2-8 入口已不要求调用方提供 `ActionIntent`，P2-9/P2-10/P2-11/P2-12 已收回只读多步、IDE 语言链和自然语言写入的最终绑定，P2-13 已将两阶段协议接入原生 API、OpenAPI 和前端 transport，但真实 provider 语义证据进入聊天 UI 的完整用户旅程仍未验收。
- P2-1 red Gate 已以真实 `/api/chat/workbench/stream` 请求确认旧路径缺口：无 intent 的自然语言请求返回 422 且无 Workbench 副作用；P2-2 已建立 Taiji-owned `TaskInterpretation`/Goal evidence；P2-3 已把显式 resolved evidence 接入非执行 planner；P2-8 已完成一条单步自然语言→Taiji 感知→语义 evidence→live affordance→ActionIntent→受控只读 Workbench→checkpoint/recovery 闭环，但尚未形成无外部参数绑定的自主语义 grounding。
- P2-4 已证明 Taiji 可消费 Workbench 语言 evidence 并形成非执行 `editor.set_language` intent，user override 优先、歧义询问；这仍不等于从普通自然语言自主完成语言选择和 IDE 执行闭环。
- P2-5 已证明一条真实 resolved-task canary 可读取文件、执行证据驱动的语言切换、经 preview/approval 写入可逆 patch，并在 runtime checkpoint 重启后 undo/recovery 恢复原文件；连续失败、预算和续接仍待 P2-6。
- P2-6 已证明 bounded Workbench loop 在真实失败步骤处停止并逐步 checkpoint，超预算在执行前拒绝，失败 checkpoint 可重启后用 fresh request 续接，旧的 read/language 能力仍可用；P2-7 已补齐受控语义多步分解，但仍不是自主语义 provider 闭环。
- P6-1a 已由 [semantic provider interface canary](../../reports/taiji_w7_p6_1a_semantic_provider_interface_20260831.json) 补齐独立 provider 请求/准入边界：请求以输入与上下文摘要内容寻址，不携带 capability/tool/parameter/intent；provider 只能返回 `SemanticEvidenceProposal`，Taiji 决定 resolved 状态并派生 decomposition，解释阶段无 ActionIntent、tool call 或 Workbench 副作用；未挂载 provider 时返回明确的 `semantic_provider_not_attached`。P6-1b 已用测试注入 provider 完成 provider evidence 驱动的聊天 plan/execute 只读 Workbench 旅程，不将测试 provider 冒充生产模型；真实 packaged provider 与浏览器现场进入 P6-1c。
- P3-1 已证明独立 SemanticEvidenceProposal 只携带内容寻址的目标/约束/语义步骤，Taiji 依据实时 input digest、tick、置信度和歧义度派生 Goal/TaskDecomposition；错配、低置信和 capability/tool/intent 字段 fail closed，provider proposal 可 checkpoint，且没有 Workbench 副作用。
- P3-2 已证明两个确定性 provider artifact 在 registry active/previous 轮换后，provider provenance 仍可审计，但同一输入的语义决策、步骤 grounding、工具 kind/参数和 no-side-effect 边界保持一致；该结果不代替真实 packaged-client provider rotation/watchdog/restart rebinding。
- P3-3 已证明现有 provider watchdog、rotation、fallback、checkpoint/restart rebinding 的确定性 packaged-client 集成 seam；P6-1c 又在本机 Hugging Face 缓存中找到并加载 Qwen2.5-0.5B-Instruct，真实 semantic evidence、浏览器字段和 provider 失败回退 Gate 已通过，但真实 Qwen packaged-client 轮换/watchdog/重启现场仍未验收。
- P4-1 已证明真实 Workbench interaction-group 执行产生 native world evidence、native executive selection、recovery trace 和 exact checkpoint replay，并观察到 holdout/lesion 效应；这仍不等于开放域稳定的 `1+1>2`。
- provider watchdog 的 native/合同证据和 P3-3 确定性 seam 不等于真实 Qwen packaged-client 已完成轮换/重绑现场验收；P6-1c 的本地 Qwen semantic evidence、浏览器字段和失败回退 Gate 已完成，但不等于安装包交付或开放域质量验收。
- R5C 主要证明结构成长机制安全、原子、可恢复；尚未证明开放域未见任务上持续优于只调整权重/路由/记忆的对照。
- interaction-group 已形成、可恢复并通过真实 Workbench 闭环 Gate；P4-2 又固定了误差驱动状态转移、跨区域/内容 credit、资源/预算拒绝、神经元/结构 rollback 和 checkpoint continuation；P4-3 已证明已准入互补组在真实 Workbench train/holdout 上相对最强单体、稠密平均和随机单体期望均有 `0.75` reward margin；P4-4 已证明只用 train-only evidence 的 native selector 在三组 seed 中选择同一互补组；P4-5 已证明四个 context family 的留一族迁移保持同一选择和 `0.75` holdout margin，但这些族仍复用同一对底层 capability；P4-6 已补充异质 capability 成员画像、正则化关系 transfer、两个未见组合目标、负组合/资源/未知成员 fail-closed 和 checkpoint/replay 保持；P4-7 已补充三轮 future Workbench 对照，关系 transfer 平均任务分数 `1.0`，单体权重/单路由/历史记忆对照 `0.2`；P4-8 已补充三轮真实在线 Outcome 写回/准入/失败拒绝/重启/rollback；P4-9 已补充在线成功 Outcome 与独立 holdout/retention 结构证据的 sealed pressure、候选仲裁、shadow validation、结构准入和拓扑 rollback；P4-10a 已证明首次结构扩容在两个未见三动作 Workbench context 上超过固定容量对照，P4-10b 已证明两个独立在线周期将容量从 2→3→4 并在未见四动作任务上保持收益；P4-11 已证明 editor+MCP 跨域结构收益与旧 workspace 能力保留；P4-12 已证明 terminal 三域治理、审批/资源边界、失败停止、checkpoint 恢复和 rollback；P2-8 已完成第一条 Taiji-owned 单步自然语言 Workbench 闭环，P2-9 已用声明式 semantic contract 收回最终参数绑定，P2-10 已完成多步 grounding/recovery，P2-11 已完成由当前文件证据驱动的 IDE language chain，P2-12 已完成 Taiji-owned digest-checked 自然语言受控写入，P2-13 已完成 API/OpenAPI/前端两阶段 transport，P5-1/P5-2/P5-3 已完成自然语言 Workbench 协议、grounding engine 和执行边界模块化，P6-1a/P6-1b/P6-1c 已完成独立 provider 接口、测试注入旅程、真实 Qwen semantic artifact、浏览器字段和失败回退 Gate；下一步只进入真实 Qwen packaged-client lifecycle 现场 Gate。
- HF/GGUF/Transformer 已退出前端主语义，不等于所有配置、隐藏 tombstone 和 Legacy 文件已物理删除。
- P7-1 的真实 Qwen 语义质量 Gate 已实测但未通过：当前 0.5B artifact 只能作为实验/回退 provider；更强 provider artifact 保留为语言器官支线，不再是当前主线。不得在 Taiji 层增加硬编码解析分支。
- E1 尚未实现，因此当前还不能声明 Skill/MCP 已作为训练语料、MCP 能力已被客户端继承、Taiji 已从这些来源更新权重，或客户端支持插件热插拔。
- 页面层通过不等于 Windows 任务栏、托盘、通知、圆角、高 DPI 和安装包现场通过。
- CPU Gate 不能替代 CUDA profile、数值一致性或跨设备 checkpoint 证据。
- 定向 evaluator/report 不能替代当前提交后的全量 CI；CI 按用户决定暂缓。P2-8 的 report 只证明其预注册确定性切片，不证明真实 provider 质量或开放域自主 grounding。

## 4. 主要实现风险

| 风险 | 当前事实 | 后续处理 |
|---|---|---|
| 路线偏移 | S18–S52 对 artifact 生命周期持续微分片，用户可见闭环未同步闭合 | S52 后停止默认存储扩张，转入自然语言→Workbench 纵切片 |
| 大文件耦合 | `taiji/adapter.py` 约 558 KB，`api/seed_runtime.py` 约 164 KB，`taiji/planning.py` 约 130 KB，`seed_platform/workbench.py` 约 126 KB，`taiji/contracts.py` 约 121 KB | 端到端闭环稳定后按 owner 提取模块，保留 facade 与 checkpoint migration |
| 计划膨胀 | `plans/active/roadmap/` 已积累 61 个 S52 前文件 | 当前状态已压缩；后续校验链接后批量归档完成分片 |
| 永久兼容层 | `gguf_path`、`download_hf` 与隐藏 tombstone 仍存在 | 建立迁移期限和删除 Gate，不让兼容字段永久进入产品模型 |
| Git 分叉 | attached `codex/interaction-group-incremental` 落后 137 提交且有 5 个未提交文件 | 只读审计后再吸收或删除，不直接合并到 main |
| 证据过度声明 | 大量 canary `gate.passed=true`，但 CI/CUDA/Windows/开放域收益未同步完成 | 能力声明按证据层级分开；未运行保持未验证 |

## 5. 当前证据入口

| 范围 | 权威入口 | 已证明边界 |
|---|---|---|
| 核心需求 | [TAIJI_CORE_REQUIREMENTS.md](../active/TAIJI_CORE_REQUIREMENTS.md) | CR-1–CR-10 与不可回退项目目的 |
| 架构身份 | [ARCHITECTURE_DIRECTION_2026_08.md](../active/ARCHITECTURE_DIRECTION_2026_08.md) | Taiji/Seed/TSK-v8/Legacy 与成熟技术采纳规则 |
| Workbench 合同 | `seed_platform/workbench.py`、`programming_languages.py`、`capability_registry.py` | registry、language、policy、execution、Outcome 与 rollback |
| provider | [R1-S0](../../reports/taiji_w7_r1_provider_watchdog_20260829.json)、[S1](../../reports/taiji_w7_r1_provider_watchdog_s1_20260829.json)、[S2](../../reports/taiji_w7_r1_provider_watchdog_s2_20260829.json) | native/合同级 watchdog 与可观测性；真实外部 packaged 轮换未宣称 |
| interaction-group | [R2-S0](../../reports/taiji_w7_r2_interaction_groups_20260829.json)、[S1](../../reports/taiji_w7_r2_interaction_groups_s1_20260829.json)、[S2](../../reports/taiji_w7_r2_interaction_groups_s2_20260829.json) | native replay、真实只读 Workbench 与恢复；结构写回未宣称 |
| desktop | [R3-S1](../../reports/taiji_w7_r3_visual_desktop_s1_20260829.json)、[R3-S2](../../reports/taiji_w7_r3_visual_desktop_s2_20260829.json) | 页面与窄布局；Windows shell 仍 tool-blocked |
| R5A | [S0](../../reports/taiji_w7_r5a_s0_internalization_20260830.json)、[S1](../../reports/taiji_w7_r5a_s1_internalization_20260830.json) | internalization DTO/learner、holdout、lesion、checkpoint 与候选边界 |
| R5B | [registry](../../reports/taiji_w7_r5b_s1_capability_registry_20260830.json)、[candidate](../../reports/taiji_w7_r5b_l1_capability_candidate_20260830.json)、[shadow](../../reports/taiji_w7_r5b_l2_capability_shadow_20260830.json)、[resource](../../reports/taiji_w7_r5b_l3_capability_resource_20260830.json) | candidate→shadow→resource/rollback；不代表自动生成任意 executor |
| R5C 总合同 | [taiji_w7_r5_open_domain_growth_v1.json](../manifests/taiji_w7_r5_open_domain_growth_v1.json) | S0–S52 scope、evidence 和 next stage |
| R5C S41–S48 | [store](../../reports/taiji_w7_r5c_s41_structural_artifact_store_20260831.json)、[bridge](../../reports/taiji_w7_r5c_s42_runtime_artifact_store_bridge_20260831.json)、[batch](../../reports/taiji_w7_r5c_s44_runtime_artifact_store_batch_20260831.json)、[audit](../../reports/taiji_w7_r5c_s47_runtime_artifact_store_audit_projection_20260831.json)、[reconcile](../../reports/taiji_w7_r5c_s48_artifact_store_runtime_reconciliation_20260831.json) | 外部 store、runtime bridge、parent 顺序、只读 audit/reconciliation |
| R5C S49–S52 | [sidecar](../../reports/taiji_w7_r5c_s49_structural_artifact_measurement_sidecar_20260831.json)、[recovery](../../reports/taiji_w7_r5c_s50_structural_artifact_measurement_bundle_recovery_20260831.json)、[verified bridge](../../reports/taiji_w7_r5c_s51_runtime_verified_measurement_bridge_20260831.json)、[policy](../../reports/taiji_w7_r5c_s52_artifact_consumption_policy_20260831.json) | verified/legacy 区分、partial recovery、strict bridge、explicit policy/default boundary |
| P2-1 | [natural-language red Gate](../../reports/taiji_w7_p2_natural_language_workbench_red_gate_20260831.json) | 无预制 `ActionIntent` 时接口明确拒绝自然语言 Workbench 请求；未声明自主规划 |
| P2-2 | [TaskInterpretation/Goal evidence Gate](../../reports/taiji_w7_p2_task_interpretation_goal_evidence_20260831.json) | 自然语言进入内容寻址、带 provenance/约束/不确定性的 Goal evidence；无 ActionIntent/tool/Workbench 副作用 |
| P2-3 | [Task planner integration Gate](../../reports/taiji_w7_p2_task_planner_integration_20260831.json) | resolved Goal evidence 结合 affordance、资源预算和置信度进入非执行 planner；candidate evidence 在 planner 前澄清 |
| P2-4 | [Language evidence planner Gate](../../reports/taiji_w7_p2_language_evidence_planner_20260831.json) | Workbench 语言 evidence 进入 Taiji 非执行 `editor.set_language` intent；user override 优先，歧义在 ActionIntent 前澄清 |
| P2-5 | [Reversible IDE canary](../../reports/taiji_w7_p2_reversible_ide_canary_20260831.json) | 真实文件读取、语言切换、preview/approval、patch、Outcome、checkpoint、重启后 undo/recovery 与原文件恢复通过 |
| P2-6 | [IDE restart/recovery canary](../../reports/taiji_w7_p2_ide_restart_recovery_20260831.json) | bounded loop 失败停止、逐步 checkpoint、超预算前置拒绝、失败重启后 fresh request 续接和旧能力保持通过 |
| P2-7 | [Task decomposition canary](../../reports/taiji_w7_p2_task_decomposition_20260831.json) | 语义步骤绑定 Goal、禁止执行字段、Taiji 非执行 grounding 和 native checkpoint roundtrip 通过；不声明自然语言已自主理解 |
| P2-8 | [Natural-language Workbench canary](../../reports/taiji_w7_p2_8_natural_language_workbench_20260831.json) | 三个独立 seed 通过自然语言→Taiji 感知→provider semantic evidence→live affordance→Taiji ActionIntent→真实只读 Workbench→checkpoint/save-load；低置信停止、provider 执行字段拒绝；仍使用显式后端 `parameter_bindings`，不声明无外部绑定的开放域 grounding |
| P2-9 | [Semantic grounding canary](../../reports/taiji_w7_p2_9_semantic_grounding_20260831.json) | 三个独立 seed 在不传外部 `parameter_bindings` 时通过声明式 capability semantic contract 生成唯一 live binding，由 Taiji 形成 ActionIntent 并完成真实只读 Workbench；未知语义、provider 参数注入、checkpoint/save-load 均 fail-closed/通过；不声明多步自主 grounding |
| P2-10 | [Multi-step grounding/recovery canary](../../reports/taiji_w7_p2_10_multistep_grounding_recovery_20260831.json) | 三个独立 seed 在不传外部 `parameter_bindings` 时分别 grounding 并执行 `workspace.read → workspace.stat`；两步 checkpoint roundtrip 通过，故意失败在第 0 步停止并保存 checkpoint，fresh request 恢复成功；不声明真实 IDE 语言链或开放域自主性 |
| P2-11 | [IDE language chain canary](../../reports/taiji_w7_p2_11_ide_language_chain_20260831.json) | 三个独立 seed 在不传外部 `parameter_bindings` 时完成 `workspace.read → workspace.programming_language.resolve → editor.set_language`；provider 不提交最终语言 ID，Taiji 从当前文件/Workbench 证据派生绑定，结果进入 Outcome/checkpoint/recovery；user override 与歧义在 ActionIntent 前停止；不声明自然语言受控写入或开放域自主性 |
| P2-12 | [Natural-language write canary](../../reports/taiji_w7_p2_12_natural_language_write_20260831.json) | Taiji 从实时 UTF-8 文件证据派生唯一文本替换和前后 digest，显式 preview/approval 后完成 `workspace.apply_patch`；checkpoint/undo/recovery、缺失审批 fail-closed 和外部 digest 冲突写入前停止通过；不声明产品 API/前端已接线或开放域自主性 |
| P2-13 | [Natural-language Workbench API canary](../../reports/taiji_w7_p2_13_natural_language_workbench_api_20260831.json) | `plan/approve/execute` 通过原生 HTTP/OpenAPI/前端 transport 暴露；最终 patch/digest/binding 仍由 Taiji 所有，重复审批幂等，缺失 token、过期/未知 plan fail-closed，完成写入产生 checkpoint；不声明真实 provider 语义证据生成或完整聊天 UI 用户旅程 |
| P5-1 | [Natural-language Workbench modularization canary](../../reports/taiji_w7_p5_1_natural_language_workbench_modularization_20260831.json) | `api/natural_language_workbench.py` 独立拥有 plan/approve/execute 协议，`SeedRuntime` 保留兼容 facade；无 runtime/provider 循环依赖，P2-12/P2-13 行为 Gate 保持通过；不声明完整聊天 UI 用户旅程 |
| P5-2 | [Natural-language Workbench grounding modularization canary](../../reports/taiji_w7_p5_2_natural_language_workbench_grounding_modularization_20260831.json) | `api/workbench_grounding.py` 独立拥有 live language/patch grounding，`SeedRuntime` 只保留薄 facade；digest/语言证据与 P2-9/P2-10/P2-11/P2-12/P2-13/P5-1 回归 Gate 保持通过；不声明真实 provider 或开放域自主性 |
| P5-3 | [Natural-language Workbench execution modularization canary](../../reports/taiji_w7_p5_3_natural_language_workbench_execution_modularization_20260831.json) | `api/workbench_execution.py` 独立拥有 request binding、preflight、approval-plan preparation、execution/outcome 和 side-effect projection，`SeedRuntime` 只保留 facade；P2-13/P5-1/P5-2 回归 Gate 保持通过；不声明真实 provider 或开放域自主性 |
| P6-1a | [Semantic provider interface canary](../../reports/taiji_w7_p6_1a_semantic_provider_interface_20260831.json) | `SemanticEvidenceProvider` 和内容寻址 `SemanticProviderRequest` 建立独立 provider seam；provider 只能提交经过 Taiji 校验的 semantic evidence，解释阶段不产生 ActionIntent、tool call 或 Workbench 副作用；未挂载状态显式可观测；不声明真实 provider 质量或完整聊天旅程 |
| P6-1b | [Backend journey test](../../tests/taiji_native/test_p6_1b_chat_workbench_journey.py) / [Frontend journey test](../../frontend/src/__tests__/ChatView.test.js) | 测试注入 provider 的 semantic evidence 沿 `/interpret → natural-language/plan → natural-language/execute` 进入只读 Workbench；前端只转发 Taiji 返回的 evidence 与 snapshot，不生成 binding、patch、digest 或 intent；不声明真实 packaged provider 或浏览器现场 |
| P6-1c | [Qwen semantic provider report](../../reports/taiji_w7_p6_1c_qwen_semantic_provider_20260831.json) / [browser field report](../../reports/taiji_w7_p6_1c_qwen_browser_field_20260831.json) / [fallback report](../../reports/taiji_w7_p6_1c_provider_failure_fallback_20260831.json) / [adapter tests](../../tests/seed/test_semantic_provider_qwen.py) | 本机 allowlisted Qwen2.5-0.5B-Instruct 真实加载并输出 semantic evidence；适配器拒绝未知/执行字段、兼容单数 `constraint` 后由 Taiji 完成 `resolved` interpretation/decomposition；浏览器聊天卡片、Taiji plan/execute 和前端无执行字段注入均通过，provider 失败会返回 degraded/Goal-only 且不产生 Workbench 副作用；确定性 packaged lifecycle seam 见 P3-3，后续仅需真实 Qwen packaged-client 现场 |
| P6-1d | [Packaged Qwen lifecycle report](../../reports/taiji_w7_p6_1d_packaged_qwen_20260831.json) / [packaged evaluator](../../scripts/training/eval_taiji_p6_1d_packaged_qwen.py) / [PyInstaller spec](../../desktop/seed.spec) | 新构建的 `SeedBackend.exe` 在两个独立进程周期中显式挂载同一 allowlisted Qwen artifact，均完成 activation、真实 semantic admission 和 `resolved` interpretation；停止后重启重新绑定同一 provider，evidence digest 保持一致；错误 model digest 返回 500 并 fail-closed；不声明多版本真实 rotation/watchdog、安装器 UI、模型质量或 CUDA |
| P7-1 | [Qwen semantic quality report](../../reports/taiji_w7_p7_1_qwen_semantic_quality_20260831.json) / [quality evaluator](../../scripts/training/eval_taiji_p7_1_semantic_quality.py) | 真实 Qwen2.5-0.5B-Instruct 的 8 个固定中文案例 Gate 未通过：provider success、只读约束保持和执行字段隔离均为 `1.0`，但清晰案例通过率仅 `0.2857`，模糊请求未达到高歧义；已定位 stat/search、query 丢失、语言协议不匹配、path 污染和过度自信等失败。该 artifact 保留为实验/回退 provider，不能进入生产语义默认入口 |
| E0 | [Native evolution and client embodiment route](../active/roadmap/63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md) | 已定义 Skill/MCP artifact 知识语料、真实 Outcome 经验、MCP 认知内化与客户端 capability 双产物、DeepSeek Harness 生命周期采纳边界和 E1–E9；当前唯一动作是 E1 合同/checkpoint Gate，不代表功能已实现 |
| P7-1a（支线） | P7-1 quality report 与 provider seam | 更强 semantic provider artifact 升级保留为语言器官支线；不阻塞 E1–E3，也不能污染 Taiji cognition、memory、topology 或执行边界 |
| P3-1 | [Semantic provider boundary canary](../../reports/taiji_w7_p3_1_semantic_provider_boundary_20260831.json) | provider 只能提交内容寻址语义 evidence；Taiji 派生 Goal/分解、保留不确定性、拒绝错配/执行字段并 checkpoint；无 Workbench 副作用 |
| P3-2 | [Provider rotation invariance canary](../../reports/taiji_w7_p3_2_provider_rotation_invariance_20260831.json) | 确定性 artifact active/previous 轮换、provider provenance 审计、同一任务语义/grounding 不变性和无执行副作用通过；不声明真实 packaged provider 已通过 |
| P3-3 | [Packaged provider lifecycle canary](../../reports/taiji_w7_p3_3_packaged_provider_lifecycle_20260831.json) | 确定性 rotation/watchdog/fallback/checkpoint/restart rebinding 集成 seam 通过；真实 Qwen semantic artifact 已在 P6-1c 后端和浏览器字段 Gate 使用，但真实 Qwen packaged-client lifecycle 现场仍未验收 |
| P4-1 | [Interaction-group Workbench canary](../../reports/taiji_w7_p4_1_interaction_group_workbench_20260831.json) | 真实 Workbench 世界证据、native executive selection、恢复轨迹、精确 checkpoint replay 与 holdout/lesion 观测通过；不声明开放域 `1+1>2` |
| P4-2 | [Small simulation canary](../../reports/taiji_w7_p4_2_small_simulation_20260831.json) | 误差驱动状态转移、跨区域/内容 credit、资源/预算 fail-closed、神经元/结构 rollback 与 checkpoint continuation 通过；不声明真实长期收益 |
| P4-3 | [Workbench longitudinal gain canary](../../reports/taiji_w7_p4_3_workbench_longitudinal_gain_20260831.json) | 已准入互补组在真实 Workbench train/holdout 上相对最强单体、稠密平均和随机单体期望均有 `0.75` reward margin；冲突组为负对照，旧 capability/资源/lesion/recovery/checkpoint 保持；不声明在线自主选组 |
| P4-4 | [Interaction-group learning canary](../../reports/taiji_w7_p4_4_interaction_group_learning_20260831.json) | train-only 原生 selector 在三组 seed 选择同一互补组，holdout outcome 翻转不改变选择，预算/冲突 fail-closed，holdout 组合收益与旧 capability/lesion/recovery/checkpoint 保持；不声明多任务族迁移 |
| P4-5 | [Interaction-group multifamily canary](../../reports/taiji_w7_p4_5_interaction_group_multifamily_20260831.json) | 四个真实 Workbench context family 中留出任一互补族时，selector 只消费其余 train evidence，三组 seed 选择同一互补组，holdout margin 为 `0.75`，冲突/预算/泄漏/旧 capability/lesion/recovery/checkpoint 保持；底层 capability 对未变化 |
| P4-6 | [Interaction-group transfer canary](../../reports/taiji_w7_p4_6_interaction_group_transfer_20260831.json) | 5 个不同 Workbench capability 训练族、2 个 train 未见目标组合、1 个负组合对照；train-only 单体画像/正则化关系模型在 3 个 seed/顺序下选择正目标，holdout 相对最强单体至少 `0.5`，资源/未知成员 fail-closed，旧 Workbench/lesion/recovery/replay/checkpoint 保持；这是有界归纳 transfer，不是开放域收益 |
| P4-7 | [Open-domain interaction gain canary](../../reports/taiji_w7_p4_7_open_domain_interaction_gain_20260831.json) | 3 个不同 future Workbench 组合、3 个 seed 下关系 transfer 平均分 `1.0`，单体权重/单路由/历史记忆均为 `0.2`；评分来自 native action success/status，future 不提前更新 learner，transfer lesion、候选回滚、checkpoint、资源/未知成员拒绝和旧任务保持；仍是受限 future 对照 |
| P4-8 | [Online interaction writeback canary](../../reports/taiji_w7_p4_8_online_interaction_writeback_20260831.json) | 3 个 seed 各执行三轮真实 Workbench 在线反馈；成功 terminal Outcome 写回后关系模型变化，失败反馈不改 learner，重启保留审计，rollback 恢复父状态并保留 rolled-back 记录，holdout 写回拒绝，native world/replay、lesion 和旧 capability 保持；这是受控在线学习边界，不是开放域自进化 |
| P4-9 | [Online interaction structural bridge canary](../../reports/taiji_w7_p4_9_online_interaction_structural_bridge_20260831.json) | 3 个 seed 的真实 Workbench 在线成功/失败轮中，只有已准入成功 Outcome 与独立 holdout/retention 证据形成 sealed structural pressure；候选经 checkpoint 恢复、structural arbitration、shadow validation、admission 后增加神经元，rollback 恢复拓扑和预算，失败反馈被排除；证明在线证据可安全抵达结构候选，不证明结构后的未见任务净收益或开放域自进化 |
| P4-10a | [Structural workspace net-gain canary](../../reports/taiji_w7_p4_10_structural_workspace_net_gain_20260831.json) | 3 个 seed 的首次在线结构扩容将 live workspace 容量由 2 增至 3；两个未见三动作 Workbench context 的结构组平均得分 `1.0`，固定容量的 interaction-weight/router/memory 对照均为 `0.0`；真实 success/status、失败反馈排除、checkpoint、lesion、资源扣减和 topology rollback 通过；仅证明一次有界结构净收益，连续增长仍待 P4-10b |
| P4-10b | [Continuous structural growth canary](../../reports/taiji_w7_p4_10b_continuous_structural_growth_20260831.json) | 3 个 seed 在首轮 2→3 后接收两个新的未见在线 context，第二轮扩容至 4；两个未见四动作 Workbench context 结构组平均得分 `1.0`，固定容量 interaction-weight/router/memory 对照均为 `0.0`，首轮任务 retention、失败排除、第二轮 lesion、双 checkpoint、资源边界和顺序 rollback 通过；仍是两周期有界证据 |
| P4-11 | [Cross-domain structural gain canary](../../reports/taiji_w7_p4_11_cross_domain_structural_gain_20260831.json) | 3 个 seed 的真实 `editor.open` / `mcp.list` 记录进入训练；在线证据驱动的结构准入将 live workspace 容量由 2 增至 3，两个未见三动作 editor+MCP 任务平均得分 `1.0`，固定容量对照均为 `0.0`，旧 workspace 任务保持，lesion、checkpoint、资源和 topology rollback 通过；这是有界跨域容量证据，不是能力发现或开放域自进化 |
| P4-12 | [Terminal three-domain governance canary](../../reports/taiji_w7_p4_12_terminal_three_domain_governance_20260831.json) | 3 个 seed 的真实 `terminal.run` 正/负 Outcome 进入训练；结构准入后两个未见三动作 editor+MCP+terminal 任务完成，固定容量对照为 `0.0`，terminal 显式 approval 且 shell/argv/timeout/output/artifact 受限，失败停止、checkpoint fresh recovery、旧能力保持、lesion 和 topology/budget rollback 通过；这是有界三域治理与结构证据，不是开放域自进化 |

所有报告只证明报告内预注册的 Gate。报告文件存在或 `gate.passed=true` 不自动扩大成产品、CI、CUDA、Windows shell、开放域收益或通用智能结论。

## 6. Git 与验证快照

- 2026-08-31 本地核对：`main` 与本地 `origin/main` 同指向 `40d018d`；本轮没有 fetch 或 push，因此不是新的远端在线确认。
- 当前 main 包含本轮 S52 policy、P2-2–P4-12 implementation/test/canary/report 和计划更新，并新增 P2-8 自然语言 Workbench 单步闭环证据，尚未提交；多个 pytest/output 临时目录对 Git 报告访问拒绝，未跟踪现场不能据此宣称全部清洁。
- attached `codex/interaction-group-incremental` worktree 位于 `.codex/worktrees/interaction-group-incremental`，落后 `origin/main` 137 个提交，修改 5 个文件。
- 最近一次已记录的远端全绿基线早于当前 R5C-S18–S52 大批改动；当前代码的全量 CI 状态保持“暂缓/未验证”。

## 7. 维护规则

1. 本文件只写当前事实，不再追加逐轮“下一步”。
2. 新能力必须同时记录 owner、输入/输出、checkpoint、Gate、失败模式、rollback 和不能声明的边界。
3. 完成分片的详细数字留在 report/manifest/route slice；失效或重复摘要进入 archive。
4. 当前执行顺序永远只由 `03_CURRENT_EXECUTION.md` 决定。
