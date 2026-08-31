# Seed / Taiji 当前执行状态

> 快照日期：2026-09-01。本文件是“现在做什么”的唯一事实源；当前原生进化路线见 [63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md](63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md)，阶段背景见 [62_POST_S51_PROJECT_CONVERGENCE_20260831.md](62_POST_S51_PROJECT_CONVERGENCE_20260831.md)，完成分片的细节由各路线文件、manifest、report 和 Git 历史追溯。

## 1. 当前结论

- Taiji 原生架构方向有效：Taiji 拥有认知状态与决策，Seed 只承载产品/runtime，Legacy NeuroPlex 只保留离线对照。
- W0–W6、W7-R1/R2 已形成基线；R3 页面层已有证据，Windows shell 仍为 `tool-blocked`；R4 CUDA 为 `hardware-blocked`。
- R5A 知识内化、R5B 效应器候选和 R5C 结构成长已建立内容寻址、验证、原子准入、rollback、checkpoint 和 lineage 基础。
- R5C-S51 已通过 verified measurement bridge 的定向 native/CPU Gate；R5C-S52 已补齐统一、可 checkpoint、可审计的 artifact consumption policy。
- 当前最大能力缺口不是 artifact 生命周期，而是自然语言任务虽已进入 Taiji Goal evidence、语言 evidence、受限语义分解、独立 provider evidence contract、P2-8 单步闭环、P2-9 声明式 semantic grounding、P2-10 多步 grounding/recovery、P2-11 IDE 语言链、P2-12 自然语言受控写入链、P2-13 产品 API/前端传输边界、P5-1 协议编排模块化、P5-2 grounding engine 模块化和 P5-3 执行边界模块化，P6-1a/P6-1b/P6-1c/P6-1d 又建立了独立 semantic provider 请求/准入接口、测试注入旅程、真实 Qwen 后端/浏览器字段旅程、失败回退和冻结 packaged backend 重启重绑现场，仍未证明真实 provider 质量、多版本真实 rotation/watchdog 现场和开放域长期能力收益。E1 已补齐统一进化语料/经验合同与 checkpoint ledger；E2-A/B 已完成脱离执行的 Skill/MCP/client-plugin 语料投影和 Seed-owned registry/lifecycle 接线；E3-0 已通过训练前 checkpoint 预检，E3-1 已通过首条原生 route/interaction credit 学习 Gate，E3-2 已通过 procedural memory intake Gate，E3-3 已通过真实 WorldTransition 的 world prediction/local update Gate，E3-4 已通过固定容量对照与结构增长触发前置 Gate，E4 已通过 Skill/MCP 知识与程序内化 Gate。P2-8 的显式 `parameter_bindings` 只保留为兼容 seam，不能扩大为开放域自主 IDE 结论。
- P2-4/P2-5/P2-6/P2-7/P2-8/P2-9/P2-10/P2-11/P2-12/P2-13/P3-1/P3-2/P3-3/P4-1/P4-2/P4-3/P4-4/P4-5/P4-6/P4-7/P4-8/P4-9/P4-10a/P4-10b/P4-11/P4-12/P5-1/P5-2/P5-3/P6-1a/P6-1b/P6-1c/P6-1d 已把 Workbench 的语言证据、Taiji 派生语言绑定、`editor.set_language`、preview/approval、文件 patch、Outcome、checkpoint、undo、失败停止、预算边界、重启续接、无工具语义分解、provider 权限边界、同任务决策不变性、真实交互组工作台闭环、小型模拟中的状态转移/credit/rollback/continuation、互补组的真实 Workbench 因果收益、train-only interaction-group 学习选择、同一 capability 对在多任务 context 上的留一族迁移、异质 capability 成员/未见组合的 train-only 关系 transfer、三轮 future Workbench 对照收益、三轮真实在线 Outcome 写回/准入/回滚、在线证据到结构候选的受控桥接、首次结构扩容的未见三动作净收益、第二个独立周期的连续结构扩容、editor+MCP 跨域结构收益与旧 workspace 能力保留，以及 terminal 三域的 approval、资源边界、失败停止、checkpoint 恢复和 rollback 接到 Taiji 受控链路；P6-1c/P6-1d 的真实 Qwen semantic artifact、浏览器字段、失败回退、冻结 packaged backend 重启重绑和错误 digest fail-closed Gate 已通过，下一步转入语义质量基线，不扩张 artifact 基础设施。
- provider artifact 的确定性轮换/watchdog/restart rebinding 集成 seam、provider 异常 Goal-only 回退 Gate 已通过；本机 Hugging Face 缓存中的 Qwen2.5-0.5B-Instruct 已通过内容寻址 artifact 校验、真实语义输出和浏览器字段 Gate，但真实 Qwen 安装包的轮换/watchdog/重启现场仍未验收。
- P6-1a 已通过 [semantic provider interface report](../../../reports/taiji_w7_p6_1a_semantic_provider_interface_20260831.json)：独立 provider 只能接收内容寻址请求并返回 `SemanticEvidenceProposal`；Taiji 在解释边界决定 resolved 状态、派生 decomposition，且不产生 ActionIntent、tool call 或 Workbench 副作用。没有 provider 时，产品 API 明确返回 `semantic_provider_not_attached`，保持 Goal-only 候选。
- P6-1b 已通过 [backend journey test](../../../tests/taiji_native/test_p6_1b_chat_workbench_journey.py) 与 [frontend journey test](../../../frontend/src/__tests__/ChatView.test.js)：测试注入 provider 的 evidence 已沿 `/interpret → natural-language/plan → natural-language/execute` 进入只读 Workbench，前端只转发 Taiji 返回的 semantic evidence 和当前 snapshot，不生成 binding、patch、digest 或 intent。
- P6-1c 后端真实 artifact Gate 已通过 [Qwen semantic provider report](../../../reports/taiji_w7_p6_1c_qwen_semantic_provider_20260831.json)：本机 Qwen2.5-0.5B-Instruct 通过显式模型 digest allowlist 和真实本地加载；模型输出仅进入 semantic evidence，Taiji 派生 `resolved` interpretation/decomposition，ActionIntent、tool call 和 Workbench side effect 均为空。适配器只对模型实际出现的单数 `constraint` 做显式受限兼容归一化，未知执行字段仍 fail-closed。
- P6-1c 浏览器字段 Gate 已通过 [browser field report](../../../reports/taiji_w7_p6_1c_qwen_browser_field_20260831.json)：客户端通过环境变量显式挂载同一 Qwen artifact，聊天 UI 显示真实目标证据和语义步骤，Taiji 计划并执行只读 Workbench；interpret/plan/execute 请求均无 `parameter_bindings`、patch、digest 或 intent 注入，浏览器无 pageerror。provider 异常回退由 [fallback report](../../../reports/taiji_w7_p6_1c_provider_failure_fallback_20260831.json) 证明 degraded/Goal-only/无副作用；确定性 packaged lifecycle 由 [lifecycle report](../../../reports/taiji_w7_p3_3_packaged_provider_lifecycle_20260831.json) 证明。二者仍不等于真实 Qwen 安装包的轮换/watchdog/重启现场验收。
- P6-1d 冻结 packaged backend Gate 已通过 [packaged Qwen report](../../../reports/taiji_w7_p6_1d_packaged_qwen_20260831.json)：新构建的 `SeedBackend.exe` 在两个独立进程周期中均通过显式 model digest 挂载 Qwen、activation、真实 semantic admission 和 `resolved` interpretation；停止后重启重新绑定同一 artifact，两个周期 evidence digest 一致，解释阶段无 ActionIntent/tool call/side effect；错误 digest 的 packaged activation 返回 500 并 fail-closed。该 Gate 不等于多版本真实 rotation/watchdog、安装器 UI 或模型质量。
- 前端 live UI 已无 HF/GGUF/Transformer 格式切换；配置和隐藏兼容 API 仍有迁移残留，后续按退役清单收口。

### P7-1 质量结论（2026-08-31）

- 真实 Qwen2.5-0.5B-Instruct 的固定 8 案例质量 Gate 未通过：provider success、只读约束保持和执行字段隔离均为 `1.0`，但清晰案例通过率为 `0.2857`，模糊请求高歧义判定为 `false`。
- 已观察到的失败是语义协议质量问题：`stat→search`、搜索词丢失、语言识别输出非协议 `check`、语言设置污染 path、模糊请求编造路径并过度自信。不得通过继续堆 prompt 分支或硬编码词表来掩盖。
- 当前 artifact 只保留在实验/回退路径；在更强 provider artifact 通过同一 Gate 前，不得把真实 Qwen 接入宣传为可读语言器官或生产级自主 IDE 入口。
- 路线已从“先升级 provider”校正为“先训练 Taiji 本体并建立持续进化数据闭环”：Skill/MCP 本身的说明、schema、示例、约束和领域资料成为受治理的知识语料，Skill/MCP/Workbench 的真实调用 Outcome 成为经验语料；MCP 的 connector、executor、permission、resource 和 UI 则形成客户端 capability/plugin 候选，由 Seed 客户端进化继承。DeepSeek Harness 的插件生命周期原则只用于 Seed 客户端的页面、IDE/Workbench、Skill、MCP、可视化和工具热插拔，不进入 Taiji 神经网络内部。完整路线见 [63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md](63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md)。
- P7-1 provider artifact 升级降为语言器官支线；当前 Qwen 仍是实验/回退 artifact，其质量失败不阻塞结构化经验驱动的 Taiji 原生训练。

## 2. 仓库与证据边界

- 2026-08-31 核对时，当前 checkout 为 `main`，`main` 与 `origin/main` 同指向 `92f7ac7`；本轮未执行 fetch/push，因此只声明核对时的本地 remote-tracking 状态。
- 当前 main 已包含 S52、P2-2/P2-3/P2-4/P2-5/P2-6/P2-7/P2-8/P2-9/P2-10/P2-11/P3-1/P3-2/P3-3/P4-1/P4-2/P4-3/P4-4/P4-5/P4-6/P4-7/P4-8/P4-9/P4-10a/P4-10b/P4-11/P4-12/P5-1/P5-2/P5-3/P6-1a/P6-1b/P6-1c/P6-1d/E5-0/E5-1/E5-2/E5-3/E6-0 的提交、测试、canary/report 和计划更新；工作树中的 pytest/output 临时目录访问拒绝仅属于 Windows 测试环境噪声，不把它扩大成“全量 CI 已通过”。
- `codex/interaction-group-incremental` 仍附着在独立 worktree，比 `origin/main` 落后 137 个提交，并有 5 个未提交文件。它不进入当前开发，不强删、不自动合并。
- S18–S52 的大量 evaluator/report 证明机制 Gate，不等于正常 CI 全量通过，也不等于开放域智能或自进化收益已经成立。
- 前端 UI 支线（不属于 P2 主线）：生命需求雷达图已从 `3664322` 的全中性配色改为「总和分档整图换色」。档位由 `sum(5 needs)` 决定，阈值 `200 / 350` 由既有单项阈值 `40 / 70` 乘维度数推导，代码只维护 `WATCH_LEVEL=40`、`ALERT_LEVEL=70` 一处常量。色相在 `themes.css` 五套主题各新增 `--needs-tier-calm/watch/alert`（浅色 `#2c82d6 / #ff8a15 / #e5372c`，dark 提亮为 `#5aa9f0 / #ffa23d / #f55b50`），不复用 `--warning`/`--danger`，因两者五套主题同值且未为暗色提亮。单项 `> 70` 仍只控制轴线加粗定位，取色一律走 `--tier`，因此整图任何时刻只有一个非中性色相。验证：`vitest` 43 文件 251 例通过、`eslint` 无告警、`npm run build` 成功。

## 3. 当前能力声明

| 可以声明 | 仍不能声明 |
|---|---|
| Taiji-owned 持续状态、局部学习、异质区域/神经元群和多类记忆原型 | 完整人脑等价、AGI 或无限自进化 |
| Workbench 文件/搜索/语言识别、受控编辑/undo、terminal/MCP、Outcome 和有限 successor loop；P2-8/P2-9/P2-10/P2-11/P2-12 可在确定性语义证据下由 Taiji 完成单步、多步闭环、声明式能力绑定/恢复、真实 IDE 语言切换和 digest-checked 受控写入；P2-13 已将 plan/approve/execute 暴露到原生 API 与前端传输门面；P5-1/P5-2/P5-3 已将协议编排、grounding engine、执行边界移出 runtime facade | 用户只说一句自然语言就能在无外部参数绑定下自主完成开放域 IDE 任务并正确切换语言 |
| 结构 candidate 的证据聚合、验证、准入、回滚、checkpoint 和有界 lineage；editor+MCP+terminal 三域结构收益与治理 Gate 已通过 | 结构扩大已在更广开放域持续带来净能力收益 |
| native-readable 默认语言表层与 provider artifact/watchdog 合同 | Qwen/provider 是 Taiji 大脑，或 packaged-client provider 已完成生产验收 |
| 自然语言可形成内容寻址、带 provenance/约束/不确定性的 Taiji Goal evidence | Taiji 已从自然语言自主解析出正确工具或 ActionIntent |
| resolved Goal evidence 可经当前 affordance/资源/置信度进入非执行 planner；语言证据可形成并在真实 canary 中执行 `editor.set_language` 与可逆 patch；P2-12 还可在 Taiji 生成 patch、preview、显式 approval 后执行 `workspace.apply_patch`；P2-13 已提供前端只保存 plan/approval/outcome 的两阶段 transport；P5-1/P5-2/P5-3 已保持兼容 facade 并拆出协议、grounding、执行边界；失败 loop 可 checkpoint/重启续接 | 普通 candidate 自然语言已具备可执行语义，或真实 provider 语义证据已接入完整聊天用户旅程并形成开放域 IDE 闭环 |
| provider 可提交受输入 digest/tick 约束、内容寻址且无执行字段的语义 evidence；Taiji 派生 Goal/分解并保留不确定性；P2-8/P2-9/P2-10/P2-11/P2-12 可由 Taiji 从当前感知/affordance 产生 intent、声明式参数绑定、实时语言绑定、digest-checked patch 和有界恢复；P2-13 的 API/前端 transport 仍只传 Taiji 计划、审批和结果；确定性 artifact 轮换后同一任务决策保持不变 | provider 已被证明拥有 Taiji 的认知、工具选择、ActionIntent、policy 或执行权；真实 packaged provider 质量、真实聊天 UI 用户旅程和生产化轮换仍未验收 |
| 前端主语义已退出 HF/GGUF/Transformer 格式切换 | 所有 Legacy 配置/tombstone 已物理删除 |

## 4. 当前唯一下一步

S52 已完成并由 [CPU canary](../../../reports/taiji_w7_r5c_s52_artifact_consumption_policy_20260831.json) 证明：新运行时默认 `verified-only`，历史 replay 必须显式 `legacy-compatible`，策略和 artifact audit 可 checkpoint，失败在 native mutation 前保持原子。P2-1 red Gate 由 [natural-language Workbench report](../../../reports/taiji_w7_p2_natural_language_workbench_red_gate_20260831.json) 证明无 intent 时执行接口仍拒绝普通自然语言请求。

P2-2 已完成并由 [TaskInterpretation/Goal evidence report](../../../reports/taiji_w7_p2_task_interpretation_goal_evidence_20260831.json) 证明：普通自然语言可进入 Taiji-owned、内容寻址、可 checkpoint 的 Goal evidence；当前状态仍是 `candidate`，confidence 为 0、ambiguity 为 1，且没有 ActionIntent、tool 或 Workbench 副作用。P2-3 已由 [planner integration report](../../../reports/taiji_w7_p2_task_planner_integration_20260831.json) 证明：resolved evidence 可结合 affordance、资源预算和置信度进入非执行 planner，未解析 evidence 在 planner 前澄清。P2-4 已由 [language evidence planner report](../../../reports/taiji_w7_p2_language_evidence_planner_20260831.json) 证明：高置信语言证据可形成非执行 `editor.set_language` intent，user override 优先，歧义语言在 ActionIntent 前询问。P2-5 已由 [reversible IDE canary report](../../../reports/taiji_w7_p2_reversible_ide_canary_20260831.json) 证明：真实文件读取、语言切换、preview/approval、patch、Outcome、checkpoint、重启后 undo/recovery 和原文件恢复均通过。P2-6 已由 [IDE restart/recovery report](../../../reports/taiji_w7_p2_ide_restart_recovery_20260831.json) 证明：失败步骤停止、逐步 checkpoint、超预算前置拒绝、失败 checkpoint 重启后新 request 续接和旧能力保持均通过。P2-7 已由 [task decomposition report](../../../reports/taiji_w7_p2_task_decomposition_20260831.json) 证明：有界语义步骤绑定当前 Goal、native checkpoint 可恢复、Taiji 可对每一步做非执行 grounding，但不能把 provider/语义字段直接变成 tool 或 ActionIntent。

P3-1 已由 [semantic provider boundary report](../../../reports/taiji_w7_p3_1_semantic_provider_boundary_20260831.json) 证明：provider proposal 内容寻址且受实时 input digest/tick 校验，Taiji 决定 resolved/candidate/ambiguous 并派生 Goal/分解，低置信、错配和执行字段在 mutation 前拒绝，provider evidence 可 checkpoint，整个边界无 Workbench 副作用。

P3-2 已由 [provider rotation invariance report](../../../reports/taiji_w7_p3_2_provider_rotation_invariance_20260831.json) 证明：artifact registry 的 active/previous 轮换、provider provenance 审计、同一输入的语义决策与 Workbench grounding 不变性通过，且无 Workbench 副作用；该 Gate 只使用确定性 metadata/proposal，不等于真实 packaged provider 已通过。P3-3 已由 [packaged provider lifecycle report](../../../reports/taiji_w7_p3_3_packaged_provider_lifecycle_20260831.json) 证明：现有 watchdog、rotation、fallback、checkpoint/restart rebinding 的确定性集成 seam 通过；P6-1d 又由真实 [packaged Qwen report](../../../reports/taiji_w7_p6_1d_packaged_qwen_20260831.json) 证明冻结 backend 的显式挂载、双周期重启重绑和错误 digest fail-closed。多版本真实 Qwen rotation/watchdog 仍未验收。

P4-1 已由 [interaction-group Workbench report](../../../reports/taiji_w7_p4_1_interaction_group_workbench_20260831.json) 证明：真实 Workbench capability execution 产生 native world evidence、executive selection、recovery trace 和 exact checkpoint replay，并保留 interaction-group 的 holdout/lesion 证据；该 Gate 不单独宣称开放域 `1+1>2`。

P4-2 已由 [small simulation report](../../../reports/taiji_w7_p4_2_small_simulation_20260831.json) 证明：误差驱动状态转移、跨区域/内容 credit 改变选择、资源/预算 fail-closed、神经元与结构 rollback，以及 checkpoint continuation 全部通过；这是确定性 CPU 机制 Gate，不等于真实 provider 质量或开放域收益。P4-3 已由 [Workbench longitudinal gain report](../../../reports/taiji_w7_p4_3_workbench_longitudinal_gain_20260831.json) 证明：真实 Workbench 互补任务的已准入组合在 train/holdout 上均以 `0.75` reward margin 超过最强单体、稠密平均和随机单体期望，冲突组保持负对照，旧 Workbench capability、资源、lesion、recovery 和 checkpoint replay 保持；该 Gate 尚未证明在线学习器自主选择组合。

P4-4 已由 [interaction-group learning report](../../../reports/taiji_w7_p4_4_interaction_group_learning_20260831.json) 证明：只用 train-only 候选时三组 seed 选择同一互补组，holdout outcome 翻转不改变选择，预算不足 fail-closed，holdout 组合仍以 `0.75` reward margin 超过单体/稠密/随机对照，冲突组不被选，且旧 capability、lesion、recovery 和 checkpoint replay 保持。P4-5 已由 [multifamily transfer report](../../../reports/taiji_w7_p4_5_interaction_group_multifamily_20260831.json) 证明：四个真实 Workbench context family 中留出任一互补族时，selector 仅消费其余 train evidence，三组 seed 均选中同一互补组，holdout margin 保持 `0.75`，冲突/预算/泄漏/旧 capability/lesion/recovery/checkpoint 均通过；但四族仍复用同一对底层 capability。P4-6 已由 [heterogeneous transfer report](../../../reports/taiji_w7_p4_6_interaction_group_transfer_20260831.json) 证明：5 个不同 capability 训练族、2 个从未在 train group 中出现的目标组合和 1 个负组合对照，通过 train-only 单体画像与正则化关系模型，在 3 个 seed/顺序排列下选择未见目标，holdout 相对最强单体至少 `0.5`，资源超限和无单体证据成员 fail-closed，checkpoint、旧 Workbench、lesion 与 replay 保持；该 Gate 仍是有界归纳 transfer，不等于开放域自进化。P4-7 已由 [open-domain interaction gain report](../../../reports/taiji_w7_p4_7_open_domain_interaction_gain_20260831.json) 证明：3 个不同 future Workbench 组合、3 个 seed 下，关系 transfer 平均任务分数为 `1.0`，只用单体权重、单路由和历史记忆均为 `0.2`；future 只使用真实 action success/status 投影评分，未提前写入 learner，transfer lesion 显著降低收益，候选回滚、checkpoint、资源/未知成员拒绝和旧任务保持均通过；该结果仍是受限 future 对照，不是开放域 AGI 证据。P4-8 已由 [online interaction writeback report](../../../reports/taiji_w7_p4_8_online_interaction_writeback_20260831.json) 证明真实在线 Outcome 写回、失败拒绝、重启和 rollback；P4-9 已由 [online interaction structural bridge report](../../../reports/taiji_w7_p4_9_online_interaction_structural_bridge_20260831.json) 证明在线成功证据经过独立 holdout/retention、结构仲裁、shadow validation、admission 和 rollback 后才改变拓扑。

P4-11 已完成 editor+MCP 跨能力域结构收益与旧 workspace 能力保留 Gate：三组 seed 的真实 Workbench `editor.open` / `mcp.list` 训练记录经过在线证据、结构准入后，容量 2→3；两个未见三动作跨域任务结构组均为 `1.0`，固定容量对照均为 `0.0`，旧 workspace 任务保持成功，lesion 去掉收益，checkpoint、资源与 rollback 通过。P4-12 已完成 terminal 三域治理 Gate：真实 `terminal.run` 训练包含正/负 Outcome，三组 seed 的结构组完成两个未见三动作 editor+MCP+terminal 组合，固定容量对照为 `0.0`；terminal 必须显式 approval，且 shell、argv、timeout、output、artifact 均受限，失败 loop 停止、checkpoint 恢复后的 fresh request 成功，旧 editor+MCP+workspace 能力、lesion、topology/budget rollback 全部保持。该证据仍是有界三域结构与治理证据，不是开放域自进化。

P2-11 已由 [IDE language chain report](../../../reports/taiji_w7_p2_11_ide_language_chain_20260831.json) 证明：三个独立 seed 在无外部 `parameter_bindings` 下完成 `workspace.read → workspace.programming_language.resolve → editor.set_language`；provider 未提交最终语言 ID，Taiji 从当前文件/Workbench 证据派生绑定，切换结果进入 Outcome/checkpoint/recovery，用户 override 和歧义均在新 ActionIntent 前停止。回归：[P2-11 test](../../../tests/taiji_native/test_ide_language_chain.py)。

**E1 已完成**：`EvolutionCorpusArtifact`、`EvolutionExperience` 和 Seed-owned append-only ledger 已落地；Workbench Outcome 只投影来源/结果 digest，不把原始结果写入经验；准入、train/holdout 分区、敏感字段脱敏、幂等/冲突拒绝、hash chain、checkpoint 关闭进程恢复/续接和篡改 fail-closed 均由 [E1 report](../../../reports/taiji_w7_e1_evolution_ledger_20260901.json) 与 5 个定向测试证明。E1 没有训练权重、安装插件、接 Legacy MCP manager 或修改客户端 UI。

**E2-A/B 已完成**：Skill/MCP/client-plugin 的说明、schema、示例、约束和版本血缘已转换为 `EvolutionCorpusArtifact`；Seed-owned registry 已记录发现、版本冲突、staged/shadow/active、失败隔离和 checkpoint 重绑；生命周期事件已幂等写入 E1 ledger。适配器只保留内容 digest，敏感值脱敏，执行源码/安装入口/Legacy manager 均不进入 verified ledger；[E2-A report](../../../reports/taiji_w7_e2_source_adapters_20260901.json) 与 [E2-B report](../../../reports/taiji_w7_e2b_source_registry_20260901.json) 通过，定向测试 12/12 通过。

**当前唯一动作：E6-4 真实第三方 MCP 连接/客户端激活前的授权决策**。E3-0～E6-3 已按本文件事实源完成：E6-0 冻结 MCP client capability candidate 合同，E6-1 完成 Seed-owned API/registry shadow lifecycle，E6-2 完成显式 activation proposal，E6-3 仅调用 ClientExtensionHost.prepare 完成本地合成器官 dry-run；全部阶段均保持 checkpoint、snapshot 绑定、rollback 和 fail-closed 边界。下一步只讨论真实连接所需的权限、凭据、网络范围、用户授权、撤销和审计条件，不自动连接第三方 MCP、不激活客户端器官、不把客户端执行器写入 Taiji。

客户端热插拔的对象明确为 Seed 客户端：Vue 页面/路由/侧栏/IDE panel/命令/设置/可视化与后端 Workbench/Skill/MCP capability 通过 `client snapshot + capability snapshot` 两阶段原子切换；`desktop/main.py` 的 PyQt/QWebEngine 根壳、托盘、任务栏、QWebChannel 和进程管理属于保护域，只能安全重启更新。一次 MCP 内化产生两个独立候选：Taiji-owned `CognitiveInternalizationArtifact` 和 Seed-owned `ClientCapabilityInheritanceCandidate`；前者学习知识/程序/affordance，后者让客户端继承连接与执行能力，二者独立准入和回滚。

## 5. 当前阻塞与暂缓项

- **CI：按用户决定暂缓。** 未运行/未修复不能标记为通过，恢复后统一收口累积问题。
- **CUDA：`hardware-blocked`。** 当前主机无可用 CUDA，不用 CPU 结果替代 GPU 结论。
- **Windows shell：`tool-blocked`。** 真实任务栏、托盘、通知、DPI 与窗口现场证据待工具可用后补齐。
- **Git 收束：暂缓。** attached worktree 含未提交变化，必须先审计再决定吸收或删除。
- **提交/推送：本轮已提交，暂不推送。** E6-0 为 `90616a7`，E6-1 为 `f8abd0a`，E6-2 为 `68ab08b`，E6-3 已完成；不执行远端同步。
- **语义 provider 质量：Gate 未通过但不阻塞 E1–E3。** 当前 Qwen2.5-0.5B-Instruct 保留在实验/回退路径，不进入生产默认语义入口；结构化 Workbench/Skill/MCP 经验可先驱动 Taiji 原生学习。
- **客户端插件：E5-0/E5-1/E5-2/E5-3/E6-0/E6-1/E6-2/E6-3 已完成，E6-4 待授权决策。** `seed_platform/client_extension_host.py`、`api/routes_client_extensions.py`、`frontend/src/composables/useClientExtensions.js` 和 `frontend/src/components/ClientExtensionSlot.vue` 已接入声明式 client snapshot 两阶段发布与真实 slot projection；旧 `/api/plugins`、marketplace/upload 已统一 410 tombstone，重复入口和前端旧引用已清除；E6-0 冻结 MCP client capability candidate 合同，E6-1 接入 Seed-owned API/registry shadow lifecycle，E6-2 生成显式 activation proposal，E6-3 完成本地合成器官的 prepare-only dry-run。当前仍不能声称真实第三方热插拔，下一步只处理连接/激活授权条件。

## 6. 事实源

- 项目目的：[TAIJI_CORE_REQUIREMENTS.md](../TAIJI_CORE_REQUIREMENTS.md)
- 架构身份与成熟技术采纳：[ARCHITECTURE_DIRECTION_2026_08.md](../ARCHITECTURE_DIRECTION_2026_08.md)
- 当前阶段总计划：[62_POST_S51_PROJECT_CONVERGENCE_20260831.md](62_POST_S51_PROJECT_CONVERGENCE_20260831.md)
- 原生认知、Skill/MCP 数据与客户端热插拔总路线：[63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md](63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md)
- S52 细化合同：[61_R5C_S52_ARTIFACT_CONSUMPTION_POLICY_20260831.md](61_R5C_S52_ARTIFACT_CONSUMPTION_POLICY_20260831.md)
- 代码事实索引：[IMPLEMENTATION_STATUS_2026_08.md](../../reference/IMPLEMENTATION_STATUS_2026_08.md)
- R5C 执行 manifest：[taiji_w7_r5_open_domain_growth_v1.json](../../manifests/taiji_w7_r5_open_domain_growth_v1.json)
