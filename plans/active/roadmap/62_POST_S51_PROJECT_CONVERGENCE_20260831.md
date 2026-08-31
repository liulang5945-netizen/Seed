# Seed / Taiji 后 S51 项目收敛与开发计划

> 状态：2026-08-31 当前阶段总计划。执行入口仍以 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md) 的唯一下一步为准；本文件定义该动作之后的顺序、边界和退出 Gate。

## 1. 本轮全盘结论

项目没有停留在“原始神经元实验”阶段。当前主线已经具备 Taiji-owned 持续状态、异质区域/神经元群、工作空间、世界/记忆/规划、受控 Workbench、语言器官、provider artifact、interaction-group、结构提案、验证、准入、回滚、checkpoint 和长期 lineage 等大量原生构件。

真正的问题是实现重心发生了偏移：R5C-S18–S51 对 artifact、retention、replay、恢复和审计进行了很深的工程加固，但用户可见的完整认知纵切片仍未闭合。P2-8 已取消执行接口对调用方预制 `ActionIntent` 的要求并完成一条确定性单步闭环；但它仍由调用方提供后端 `parameter_bindings`，因此“Taiji 从自然语言和当前环境独立完成语义到能力绑定、执行、验证并从 Outcome 学习”还不能声明完成。

后续不再以增加更多 artifact-store 微分片作为默认推进方式。S52 是该基础设施线的收口节点；之后主线必须转向真实任务闭环和能力收益。

## 2. 当前事实与缺口

| 范围 | 已落地事实 | 关键缺口 |
|---|---|---|
| Taiji 身份 | Taiji 是唯一认知主体；Seed 是产品/runtime；Legacy NeuroPlex 只做离线对照 | 仍需用端到端行为证明所有权，而不只靠包依赖和合同声明 |
| 神经元与结构成长 | 向量化区域、稀疏连接、局部学习、interaction-group、growth/prune/split/merge、验证/回滚/checkpoint 已有实现 | 尚未证明开放域未见任务上，结构变化持续优于“只调已有权重/路由”的对照 |
| Workbench | 文件读取、搜索、语言识别、`editor.set_language`、patch/undo、terminal/MCP、policy/approval、Outcome 和有限 successor loop 已存在 | 自然语言输入不能直接产生 Taiji-owned goal/plan/ActionIntent；IDE 自主操作尚未形成真实用户旅程 |
| 语言器官 | `native-readable` 为默认表层；`structured-stub` 为调试边界；Qwen/Transformers 只位于 provider 适配层；artifact 轮换和健康 watchdog 有合同测试 | 缺 packaged-client 下真实 provider 轮换、失败回退、重启重绑和质量 Gate；provider 不能被误用为认知规划器 |
| 自进化证据 | R5A/R5B/R5C 已覆盖知识候选、效应器候选、结构候选及有界生命周期 | 当前主要证明“机制安全可恢复”，没有充分证明“能力真实增长、旧能力保持、资源收益为正” |
| 产品边界 | 前端实时页面已不提供 HF/GGUF/Transformer 格式切换；相关 API 多数退化为隐藏 tombstone | 配置层仍保留 `gguf_path`、`download_hf` 等迁移字段；需要明确保留期限并最终删除，而不是永久兼容 |
| 桌面体验 | 页面层和生命状态已有一轮收敛 | Windows 任务栏、托盘、通知 logo、高 DPI 和真实窗口仍缺现场证据；不得用页面截图替代 |
| 工程结构 | 测试、evaluator 和报告覆盖面很大 | `taiji/adapter.py`、`api/seed_runtime.py`、`seed_platform/workbench.py`、`taiji/planning.py`、`taiji/contracts.py` 已成为高耦合大文件；plans 也积累了大量完成分片 |
| Git/工作树 | 2026-08-31 核对时 `main` 与 `origin/main` 同指向 `92f7ac7` | 附着的 `codex/interaction-group-incremental` 比 `origin/main` 落后 137 个提交且有 5 个未提交文件，必须隔离审计，不能强删或直接混入主线 |

## 3. 统一执行原则

1. 每一阶段必须闭合一个真实用户旅程，不能只因为某个内部 ledger 多了字段就继续切下一片。
2. provider 可以做语言编码/表达和候选语义证据，但不能拥有 goal、tool choice、memory、policy 或最终 ActionIntent。
3. 新结构只有在现有权重、路由、记忆和策略已不足，并通过 holdout、lesion、资源、旧能力保持与 rollback 后才能准入。
4. 训练或在线学习前，先通过 checkpoint 保存、进程关闭、恢复、继续；失败时禁止开始训练。
5. 新成长路径使用 verified artifact；历史证据兼容必须显式声明原因，不允许静默升级或伪造 measurement facts。
6. CUDA 与 Windows shell 证据继续保留，但硬件/工具不可用时不阻塞 CPU/native 主线，也不得宣称通过。
7. CI 依用户当前决定暂缓；暂缓不等于通过。恢复 CI 时集中收口全部累积问题，在全绿前不继续发布功能声明。
8. 已完成过程进入 archive；核心需求、当前架构、当前缺口和未完成 Gate 留在 active/reference。

## 4. 分阶段开发顺序

### P0：事实源与工作区收敛

目标：恢复“一个当前状态、一个当前下一步、一个阶段总计划”。

- `03_CURRENT_EXECUTION.md` 只保留当前事实和唯一动作；S0–S52 的逐轮说明由分片文档、manifest、report 和 Git 历史承担。
- `04_EXECUTION_PLAN.md` 只作为阶段/已完成证据索引，不再产生新的并列下一步。
- 后续把已完成的 R5C 分片从 active 批量归档，并保留一个内容索引；归档前先校验所有链接和报告存在。
- 对附着 worktree 的 5 个未提交文件做差异审计，区分“main 已吸收”“仍有独立价值”“应放弃”；未得出可追溯结论前不删除 worktree/branch。
- 对 Git 无法读取的 pytest 临时目录做目标路径核对，清理动作必须在单独授权范围内进行，不能误删 `output/` 证据。

退出 Gate：active 中只有一处“当前唯一下一步”；当前状态、manifest 和实现代码不互相矛盾；没有为了整理文档丢失未提交代码或证据。

### P1：R5C-S52 artifact consumption policy 收口

状态：**已完成（2026-08-31）**；证据：[S52 CPU canary](../../../reports/taiji_w7_r5c_s52_artifact_consumption_policy_20260831.json)。

目标：以显式策略对象替代 `require_verified_measurements: bool`，并把 S52 定义为 artifact 基础设施线的最后一个默认分片。

- owner：Taiji artifact policy 与 checkpoint；SeedRuntime 只解析、投影和执行已绑定策略。
- 策略：新成长强制 `verified-only`；历史回放只能显式选择 `legacy-compatible`，并记录 reason、artifact status、policy revision/digest。
- checkpoint：policy、迁移/回滚记录和 audit 可保存、恢复、重放；策略变化不能改写旧事实。
- 原子性：多 candidate 先全量解析策略与 sidecar，任何一项失败都不消费 sibling、不扣预算、不改 topology。
- 回滚：恢复上一 policy revision，不删除 artifact，不伪造 sidecar，不改变已完成 lineage。

退出 Gate：verified/legacy/tampered/missing/multi-candidate/checkpoint/rollback 全部通过定向 CPU canary。此后若没有端到端任务暴露具体缺陷，不再创建 S53 式存储分片。

### P2：自然语言到 Workbench 的 Taiji-owned 纵向闭环

状态：**P2-13 已完成（2026-08-31）**；证据：[P2-1 natural-language Workbench red Gate](../../../reports/taiji_w7_p2_natural_language_workbench_red_gate_20260831.json)、[P2-2 TaskInterpretation/Goal evidence Gate](../../../reports/taiji_w7_p2_task_interpretation_goal_evidence_20260831.json)、[P2-3 planner integration Gate](../../../reports/taiji_w7_p2_task_planner_integration_20260831.json)、[P2-4 language evidence planner Gate](../../../reports/taiji_w7_p2_language_evidence_planner_20260831.json)、[P2-5 reversible IDE canary](../../../reports/taiji_w7_p2_reversible_ide_canary_20260831.json)、[P2-6 IDE restart/recovery canary](../../../reports/taiji_w7_p2_ide_restart_recovery_20260831.json)、[P2-7 task decomposition canary](../../../reports/taiji_w7_p2_task_decomposition_20260831.json)、[P2-8 natural-language Workbench canary](../../../reports/taiji_w7_p2_8_natural_language_workbench_20260831.json)、[P2-9 semantic grounding canary](../../../reports/taiji_w7_p2_9_semantic_grounding_20260831.json)、[P2-10 multi-step grounding/recovery canary](../../../reports/taiji_w7_p2_10_multistep_grounding_recovery_20260831.json)、[P2-11 IDE language chain canary](../../../reports/taiji_w7_p2_11_ide_language_chain_20260831.json)、[P2-12 natural-language write canary](../../../reports/taiji_w7_p2_12_natural_language_write_20260831.json)、[P2-13 API/前端 transport canary](../../../reports/taiji_w7_p2_13_natural_language_workbench_api_20260831.json)。P2-8 已闭合第一条 Taiji-owned 单步自然语言→感知→语义 evidence→live grounding→ActionIntent→Workbench execution→checkpoint/recovery 链路；P2-9 将 capability 的语义要求和语义槽→参数映射做成内容寻址的声明式合同，P2-10 又在不传外部 `parameter_bindings` 的情况下完成多步只读任务、失败停止、失败 checkpoint 和 fresh request 恢复，P2-11 再把真实 IDE 语言切换纳入同一条无外部最终绑定的三步链，P2-12 收回自然语言编辑中的 digest/patch 参数生成，P2-13 将 plan/approve/execute 以原生 API、OpenAPI 和前端 transport 暴露，并验证幂等审批、缺失 token、过期/未知 plan 的 fail-closed。该阶段仍未证明真实 provider 质量或完整聊天 UI 用户旅程。

目标：解决当前最大的产品与架构漏洞，让 Taiji 在受控边界内真正操作自带 IDE，并逐步收回从语义到具体能力参数的认知 ownership。

执行链固定为：

```text
用户输入
  -> Observation / Percept
  -> Taiji Goal 与任务约束
  -> 当前 WorldState / Workbench affordances
  -> Taiji Plan / ActionIntent
  -> preview / policy / approval
  -> IDE 或工具执行
  -> EnvironmentOutcome
  -> World / Memory / Learning
  -> ContentPlan / LanguageOrgan
  -> 可读输出
```

子阶段：

1. 已完成 red Gate：普通 `/api/chat/workbench/stream` 在没有预制 intent 时返回 422，并明确暴露当前缺口，未执行任何 Workbench 副作用。
2. 已完成 Taiji-owned `TaskInterpretation`/goal evidence：自然语言以 input/evidence digest、provenance、约束和不确定性进入 Goal；当前只形成 `candidate`，不执行、不选工具。provider 若参与，只能产生带 provenance 的语义候选，不能直接返回 tool 名或最终 intent。
3. 已完成 Goal evidence、当前 affordance、资源/风险/置信度到 Taiji planner 的非执行接线：仅显式 `resolved` evidence 可形成 ActionIntent，普通 `candidate` 必须澄清；禁止 `if prompt contains ... -> tool` 语义硬编码。
4. 已完成语言证据接线：现有 `workspace.programming_language.resolve` 的高置信、无冲突结果可进入 Taiji 非执行 `editor.set_language` intent；低置信/冲突询问，用户 override 始终优先，规划阶段不执行副作用。
5. 已完成真实可逆编程任务 canary：读取文件 → 识别/切换编辑器语言 → 形成 patch → preview → 审批 → 写入 → 重新读取验证 → Outcome → checkpoint → undo/recovery。
6. 已完成失败恢复与重启续接 canary：bounded loop 在真实失败处停止并逐步 checkpoint，超预算在执行前拒绝，失败 checkpoint 可用 fresh request 续接，旧 Workbench 能力保持。
7. 已完成受控的自然语言多步任务分解 canary：Taiji-owned semantic evidence 产生有界、可验证的 Workbench step candidates，逐步执行前保留 clarification、preview/policy、预算和 checkpoint 边界，provider 不能直接注入工具名或 intent。
8. 已完成独立的语义 evidence provider contract：provider 只提交内容寻址的目标/约束/语义步骤候选，Taiji 校验输入 digest、tick、置信度和禁止字段后，才决定是否形成 TaskInterpretation/TaskDecomposition；provider 不拥有 Goal 定案、工具选择、ActionIntent、policy 或执行权。
9. 已完成 provider artifact 轮换与同一任务决策不变性 canary：在同一输入/WorldState/affordance 下轮换两个确定性 artifact，provider provenance 可审计但语义、grounding 和工具决策保持一致，且无 Workbench 执行副作用。
10. 已完成 packaged-client provider lifecycle 的确定性集成 seam：现有 guarded chat canary 与 health policy 验证了 provider 发布后劣化回退、回退状态 checkpoint、重启重绑和隔离版本不复活；本机没有可验收的真实 provider 模型目录，因此该部分保持 `asset-unverified`，不宣称真实模型质量通过。
11. 已完成 P4-1 interaction-group Workbench Gate：真实 Workbench capability execution 产生 native world evidence、native executive selection、recovery trace 和 exact checkpoint replay，并观察到 holdout/lesion 效应；该证据不单独扩大为开放域 `1+1>2`。
12. 已完成 P4-2 小型模拟状态转移、credit、rollback 与 checkpoint continuation Gate：误差驱动状态转移、跨区域/内容 credit 改变选择、资源/预算 fail-closed、神经元与结构 rollback，以及 checkpoint continuation 全部通过；该结果是确定性 CPU 机制证据，不等于真实 provider 质量或开放域收益。
13. 已完成 P4-3 真实 Workbench 纵向收益与旧能力保持 Gate：互补任务的已准入组合在 train/holdout 上均以 `0.75` reward margin 超过最强单体、稠密平均和随机单体期望，冲突组保持负对照，旧 Workbench capability、资源、lesion、recovery 和 checkpoint replay 保持；该 Gate 尚未证明在线学习器自主选择组合。
14. 已完成 P4-4 interaction-group 学习选择、跨 seed 稳定性与旧能力保持 Gate：只用 train-only 候选时三组 seed 选择同一互补组，holdout outcome 翻转不改变选择，预算不足 fail-closed，holdout 组合仍以 `0.75` reward margin 超过单体/稠密/随机对照，冲突组不被选，且旧 capability、lesion、recovery 和 checkpoint replay 保持。
15. 已完成 P4-5 多任务族与留一族 holdout 迁移 Gate：四个真实 Workbench context family 中留出任一互补族时，selector 仅消费其余 train evidence，三组 seed 均选中同一互补组，holdout margin 保持 `0.75`，冲突/预算/泄漏/旧 capability/lesion/recovery/checkpoint 均通过；但四族仍复用同一对底层 capability。
16. 已完成 P4-6 异质成员与未见组合 transfer Gate：5 个不同 capability 训练族、2 个未见组合目标和 1 个负组合对照，使用 train-only 单体证据和可 checkpoint 的正则化关系模型，3 个 seed/顺序均选中对应正向目标，holdout 相对最强单体至少 `0.5`，未知成员/资源超限 fail-closed，旧 Workbench、lesion、recovery、replay 和 checkpoint 保持；该结果限定为有界归纳 transfer。
17. 已完成 P4-7 开放域长期收益与权重/路由/记忆对照 Gate：3 个不同 future Workbench 组合、3 个 seed 下，关系 transfer 完成所有双动作任务，平均分 `1.0`，只用单体权重、单路由和历史记忆为 `0.2`；真实 action success/status 作为评分源，future 未提前写入 learner，transfer lesion、候选回滚、checkpoint、资源/未知成员拒绝和旧任务保持均通过；仍不外推为开放域 AGI。
18. 已完成 P4-8 真实在线 Outcome 写回、学习准入、重启与 rollback Gate：3 个 seed 各执行三轮真实 Workbench 反馈；成功 terminal Outcome 写回后关系模型变化，失败反馈不改变 learner，重启保留 audit，显式 rollback 恢复父状态并留下 rolled-back 记录，holdout 分区禁止进入在线反馈，原生 world/replay、lesion 和旧 Workbench 保持；该结果只证明受控在线学习边界，不等于开放域持续自进化。
19. 已完成 P4-9 在线 interaction evidence 到结构候选的受控桥接 Gate：3 个 seed 的真实 Workbench 在线成功/失败轮中，只有已准入成功 Outcome 被桥接；独立 holdout/retention 证据与在线 evidence 共同形成 sealed pressure，候选经 checkpoint 恢复、structural arbitration、shadow validation、结构 admission 后才增加神经元，rollback 恢复拓扑和预算；失败反馈未进入 pressure，且未绕过治理链路。
20. 已完成 P4-10a 在线结构候选的首次未见任务净收益 Gate：3 个 seed 中，真实在线成功 Outcome 与独立 holdout/retention 证据经过结构桥接、arbitration、shadow validation 和 admission 后，live workspace 容量由 2 增至 3；两个未见三动作 Workbench context 的结构组平均得分为 `1.0`，只调 interaction 权重、路由或记忆且容量固定为 2 的对照均为 `0.0`，并通过失败在线反馈排除、checkpoint continuation、lesion 容量影响、资源扣减和 topology rollback；这是一次有界净收益证据，不是连续自进化结论。
21. 已完成 P4-10b 第二个独立周期的连续结构增长 Gate：3 个 seed 在第一轮容量 2→3 后接收两个新的、训练中未见的在线 context，形成新的 sealed pressure 并完成第二轮容量 3→4；第二轮两个未见四动作 Workbench context 的结构组平均得分为 `1.0`，固定容量 interaction-weight/router/memory 对照均为 `0.0`，第一轮三动作任务在第二轮后保持成功，lesion 移除第二轮容量后收益消失，两个 admission 均 checkpointable，顺序 rollback 恢复拓扑、容量和预算；这是两周期有界增长证据，不是无限自进化结论。
22. 已完成 P4-11 editor+MCP 跨能力域结构收益与旧 workspace 能力保留 Gate：三组 seed 的真实 Workbench `editor.open` / `mcp.list` 训练记录经过在线证据、结构准入后，live workspace 容量由 2 增至 3；两个未见三动作跨域任务结构组平均得分为 `1.0`，固定容量 interaction-weight/router/memory 对照均为 `0.0`，旧 workspace 任务保持成功，lesion 移除跨域收益，native status、checkpoint、资源和 rollback 证据通过；该结果仍是 editor+MCP 的有界跨域容量证据。
23. 已完成 P4-12 terminal 三域治理与恢复 Gate：真实 `terminal.run` 正/负 Outcome 进入训练；三组 seed 完成两个未见三动作 editor+MCP+terminal 组合，固定容量对照为 `0.0`；terminal 显式 approval、无 shell、argv/timeout/output/artifact 资源边界、失败停止、checkpoint 恢复后的 fresh request、旧能力保持、lesion 与 topology/budget rollback 全部通过；该结果仍是有界三域结构与治理证据。
24. 已完成 P2-8 Taiji-owned 自然语言 Workbench 单步闭环 Gate：自然语言输入先进入 Taiji 感知，provider 只能提交无执行字段的语义 evidence；Taiji 将证据重新绑定当前 tick，结合 live affordance 形成 ActionIntent，经过既有 preflight/policy/approval 后执行真实只读 Workbench，并验证低置信停止、执行字段 fail-closed 和 checkpoint/save-load。证据：[P2-8 report](../../../reports/taiji_w7_p2_8_natural_language_workbench_20260831.json)，回归：[P2-8 test](../../../tests/taiji_native/test_natural_language_workbench.py)。边界是确定性 provider evidence 与显式后端参数绑定，不宣称开放域语义理解。
25. 已完成 P2-9 Taiji-owned semantic grounding Gate：能力描述加入内容寻址的声明式 `semantic_requirements` 与 `semantic_parameters` 合同，Taiji 在不接收外部 `parameter_bindings` 时根据语义步骤生成唯一 live binding，并继续走 ActionIntent、Workbench preflight/执行、Outcome、checkpoint；零候选、多候选和 provider 参数/执行字段均 fail-closed。证据：[P2-9 report](../../../reports/taiji_w7_p2_9_semantic_grounding_20260831.json)，回归：[P2-9 test](../../../tests/taiji_native/test_semantic_grounding.py)。
26. 已完成 P2-10 Taiji-owned multi-step grounding/recovery Gate：三种子均在无外部 `parameter_bindings` 下独立 grounding 两个只读语义步骤 `workspace.read → workspace.stat`，按顺序执行并 checkpoint；故意失败在第 0 步停止，失败 checkpoint 可被 fresh request 恢复并完成新任务。证据：[P2-10 report](../../../reports/taiji_w7_p2_10_multistep_grounding_recovery_20260831.json)，回归：[P2-10 test](../../../tests/taiji_native/test_multistep_grounding_recovery.py)。
27. 已完成 P2-11 Taiji-owned IDE language chain Gate：三个独立 seed 在无外部 `parameter_bindings` 下完成 `workspace.read → workspace.programming_language.resolve → editor.set_language`；provider 只提交操作/路径/override 语义，最终 `programming_language_id` 由 Taiji 根据当前文件/Workbench 证据派生，切换结果进入 Outcome/checkpoint/recovery，用户 override 和歧义均在 ActionIntent 前停止。证据：[P2-11 report](../../../reports/taiji_w7_p2_11_ide_language_chain_20260831.json)，回归：[P2-11 test](../../../tests/taiji_native/test_ide_language_chain.py)。
28. **已完成 P2-12**：Taiji 从 P2-11 当前文件证据派生 digest/structured patch，经过 preview/policy/approval 后执行 `workspace.apply_patch`，并验证 Outcome、checkpoint、undo/recovery，以及冲突/歧义/审批缺失的写入前停止。证据：[P2-12 report](../../../reports/taiji_w7_p2_12_natural_language_write_20260831.json)，回归：[P2-12 test](../../../tests/taiji_native/test_natural_language_write.py)。
29. **已完成 P2-13**：将 P2-12 的 Taiji-owned `plan → preview/approval → execute` 协议接入原生 API、OpenAPI baseline 和前端 `nativeApi`/composable；重复审批幂等，缺失 token、snapshot/tick 漂移和未知 plan fail-closed。证据：[P2-13 report](../../../reports/taiji_w7_p2_13_natural_language_workbench_api_20260831.json)，回归：[P2-13 test](../../../tests/taiji_native/test_natural_language_workbench_api.py) 与前端 native API/composable tests。
30. **已完成 P5-1**：将 P2-12/P2-13 已验证的 plan/approve/execute 编排从 `api/seed_runtime.py` 拆到 [`api/natural_language_workbench.py`](../../../api/natural_language_workbench.py)，保留 runtime facade、checkpoint/approval/rollback 合同和前端 transport。证据：[P5-1 report](../../../reports/taiji_w7_p5_1_natural_language_workbench_modularization_20260831.json)，回归：[P5-1 test](../../../tests/taiji_native/test_natural_language_workbench_modularization.py)。
31. **已完成 P5-2**：将“语义步骤 → 实时 Workbench 证据 → 声明式 binding / digest-checked patch”的 grounding engine 拆到 [`api/workbench_grounding.py`](../../../api/workbench_grounding.py)，`SeedRuntime` 只保留兼容 facade；语言 ID 仍由 live evidence 派生，文本 patch 仍由当前 digest 计算，P2-9/P2-10/P2-11/P2-12/P2-13 与 P5-1 回归保持通过。证据：[P5-2 report](../../../reports/taiji_w7_p5_2_natural_language_workbench_grounding_modularization_20260831.json)，回归：[P5-2 test](../../../tests/taiji_native/test_natural_language_workbench_grounding_modularization.py)。
32. **已完成 P5-3**：将 planning outcome → Workbench request binding → preflight → execution/outcome 边界拆到 [`api/workbench_execution.py`](../../../api/workbench_execution.py)，`SeedRuntime` 只保留调用 facade；当前 capability/MCP snapshot 绑定、approval-plan 准备、执行前置检查和 side-effect 投影均保留，P2-13/P5-1/P5-2 与自然语言 Workbench 回归保持通过。证据：[P5-3 report](../../../reports/taiji_w7_p5_3_natural_language_workbench_execution_modularization_20260831.json)，回归：[P5-3 test](../../../tests/taiji_native/test_natural_language_workbench_execution_modularization.py)。
33. **已完成 P6-1a**：建立独立 `SemanticEvidenceProvider` / `SemanticProviderRequest` 接口；请求以输入与上下文摘要内容寻址，不携带 capability/tool/parameter/intent，provider 只能提交 `SemanticEvidenceProposal`，Taiji 负责 admission、Goal 状态和语义分解，解释阶段无 ActionIntent、tool call 或 Workbench 副作用。证据：[P6-1a report](../../../reports/taiji_w7_p6_1a_semantic_provider_interface_20260831.json)，回归：[P6-1a test](../../../tests/taiji_native/test_semantic_provider_interface.py)。
34. **已完成 P6-1b**：用测试注入 provider 走通聊天端 `/interpret → natural-language/plan → natural-language/execute` 只读 Workbench 旅程；后端验证真实 TestClient transport，前端验证 semantic evidence 只进入 Taiji plan，客户端不生成 binding、patch、digest 或 intent。证据：[P6-1b backend journey](../../../tests/taiji_native/test_p6_1b_chat_workbench_journey.py)，[P6-1b frontend journey](../../../frontend/src/__tests__/ChatView.test.js)。该 Gate 不等于真实 provider 模型验收。
35. **当前唯一动作**：执行 P6-1c 真实 packaged semantic provider artifact / 浏览器现场 Gate；预检已确认当前仓库没有可加载的真实 checkpoint、tokenizer、safetensors 或 adapter 目录，因此只在真实 allowlisted artifact 提供后，验证 watchdog/rotation/restart rebinding、首轮 semantic evidence canary、浏览器聊天卡片和 Workbench plan/approval/execute 现场，确认 provider 失败时回到 Goal-only/clarification 边界。禁止把测试 provider、历史 manifest 或诊断 corpus 冒充产品模型，禁止前端重新生成 patch/digest/intent。

退出 Gate：用户只提供自然语言和工作区，Taiji 不依赖外部预制 `ActionIntent` 或最终 `parameter_bindings` 完成 P2-1 至 P2-13 的受控协议链；所有动作可审计、可恢复、可解释，provider 移除后决策链仍属于 Taiji。P2-12 已证明写入参数由 Taiji 基于当前内容派生，P2-13 已证明产品 API/前端 transport 不重新夺回这些 ownership；P6-1b 已证明测试注入 provider 的完整 transport 旅程，P6-1c 仍需真实 packaged provider artifact 和浏览器现场验收。

### P3：语言器官与 provider 生产化

目标：让语言层稳定充当“嘴巴/语言接口”，同时保持 Taiji 认知所有权。

- `native-readable` 继续作为无外部 provider 时的真实默认；`structured-stub` 仅可显式调试。
- 用相同 ContentPlan 对 native-readable 与外部 provider 评估可读性、约束保持、事实遗漏、幻觉和延迟，禁止比较两个不同认知结果。
- 在 packaged client 中验证 provider artifact 内容寻址、版本轮换、watchdog、previous→native 降级、cooldown、重启重绑与失败通知；当前确定性集成 seam 已通过，真实模型资产与质量 Gate 仍未验收。
- provider 输出必须经过约束检查；不得写 Taiji memory、改变 intent、调用工具或绕过 policy。

退出 Gate：真实客户端完成至少一次成功轮换、一次失败回退和一次 checkpoint 重启；语言后端变化不改变同一任务的工具决策与事实约束。

### P4：interaction-group、学习与自进化收益 Gate

目标：从“机制可以运行”升级为“能力确实增长”。provider watchdog、interaction-group 和小型模拟 Gate 都在本阶段继续，不搁置。

- interaction-group：在预注册任务上比较最强单体、稠密平均、随机分组与学习分组，只有组合完成单体不能完成的任务才声明 `1+1>2`；P4-6 已增加异质成员画像和未见组合关系 transfer，P4-7 已增加三轮 future 对照，P4-8 已增加真实 Outcome 写回/拒绝/重启/rollback，但不把有界归纳或受控在线更新外推成开放域智能。
- 长期收益：P4-7 已在真实 Workbench future holdout 上和只调权重/路由/记忆对照比较；后续必须扩展任务域与连续反馈，而不是继续堆叠一次性任务族或合成 reward。
- 在线成长：P4-8 已验证 Outcome 能在显式 Gate 后写回 relation learner 并可重启/rollback；P4-9 已验证在线 interaction evidence 只有在独立 holdout/retention、arbitration、shadow validation、admission 和 rollback 之后才能抵达结构变化；P4-10a 已验证首次结构扩容可在两个未见三动作 Workbench context 上超过固定容量的权重/路由/记忆对照；P4-10b 已验证第二个独立在线周期将容量从 2→3→4 并在未见四动作任务上保持收益；P4-11 已验证 editor+MCP 跨域结构收益与旧 workspace 保留；P4-12 已验证 terminal 三域治理、资源/审批、失败停止与 checkpoint 恢复。自然语言主线当前转向 P2-13 产品 API/前端接线，避免后端 Gate 与真实用户旅程脱节。
- 小型模拟 Gate：只用于快速验证状态转移、credit 和 rollback，不替代真实 Workbench longitudinal Gate。
- R5A：证明知识内化后可降低外部 artifact 依赖，同时旧任务不退化；未过独立删除评审前不物理删除来源。
- R5B：只有真实新工具能力通过 shadow、approval、resource 和 disposer Gate 后才进入 registry；不从任意源码直接注册 executor。
- R5C：先比较权重/路由/记忆调整；只有持续容量不足时才申请结构增长。结构变化必须在未见任务上产生净收益，并通过 lesion、资源和 rollback。
- 睡眠/玩耍/稳态：接入同一真实任务纵向证据，验证不同模式的因果作用，避免成为外围 scheduler 展示。

退出 Gate：至少一个长期任务族同时证明新能力收益、旧能力保持、资源边界、因果 lesion 和 checkpoint continuation；“自进化”声明以该证据为起点，不以新增参数数量为依据。

### P5：工程模块化与 Legacy 残留退役

目标：降低大文件和永久兼容层对架构上限的限制，但不做无证据的大重写。

- 按 owner 拆分 `taiji/adapter.py`：checkpoint/state、provider、recovery、structural lifecycle、workbench projection 分离；adapter 只保留组合与兼容 facade。
- 拆分 `api/seed_runtime.py`：runtime lifecycle、Workbench orchestration、structural artifact bridge、persistence/status 分离。P5-1 已先把自然语言 Workbench plan/approval/execute 协议移至 `api/natural_language_workbench.py`，P5-2 已将语义 grounding engine 移至 `api/workbench_grounding.py`，P5-3 已将 request/preflight/execution boundary 移至 `api/workbench_execution.py`，均保留 runtime facade。
- 拆分 `seed_platform/workbench.py`：capability registry、policy/preflight、language/editor、workspace executor、terminal/MCP executor 分离。
- `taiji/contracts.py` 按 observation/world/goal/action/memory/language 划分，但保持版本化导出和 checkpoint 迁移。
- 为 HF/GGUF/Transformer 残留建立退役清单：前端 live 路径保持禁止；隐藏 API tombstone 只保留一个明确迁移窗口；`gguf_path`、`download_hf` 等配置在迁移测试通过后删除。
- Legacy NeuroPlex 保持离线 benchmark，不允许重新成为 Seed 启动默认或 Taiji runtime 依赖。

退出 Gate：模块依赖方向可静态检查；旧 checkpoint 可显式迁移；前端、OpenAPI 和运行配置不再暴露模型格式切换；行为回归不因拆分改变。

### P6：客户端真实性与桌面体验

目标：让客户端展示真实 Taiji，而不是重复入口、固定文案或与实现不一致的状态。

- 生命状态只保留一个主入口和一个详情视图；全局页面不再顶置重复状态条。
- 多维状态采用真实 runtime projection，明确“已测事实、估计值、不可用”，不造假百分比。
- 侧边栏无需滚动即可访问主功能；IDE、知识、训练、生命和设置名称与当前 Taiji 能力对齐。
- Windows 应用图标、任务栏、托盘、通知弹窗统一使用同一 Taiji logo；窗口圆角、DPI、键盘导航和 reduced-motion 一并现场验证。
- HF/GGUF/Transformer 格式切换不得重新出现在产品 UI。

退出 Gate：packaged `Seed.exe` 在真实 Windows 桌面完成窗口、任务栏、托盘、通知、DPI 和关键用户旅程取证；前端字节与安装包内容一致。

### P7：集中 CI、仓库和发布收口

目标：按用户决定，在功能主线完成一轮后统一处理累积 CI，而不是把“未运行”写成“通过”。

- 先跑与变更相关的定向 Gate，再跑 Python lint/type/test、前端 lint/test/build、OpenAPI、Legacy-off、checkpoint、Windows/package 全矩阵。
- 对所有失败按根因分组，一次只修一个根因，直到无 skipped/允许失败的关键 Gate。
- 审计 `main`、attached worktree、local refs、`origin/main`；保存有价值的未提交变化后再收束到只保留 main。
- 生成发布 manifest，绑定 commit、前端字节、Seed.exe、checkpoint format、报告 digest 和能力声明。

退出 Gate：CI 全绿、工作树和 refs 收敛、发布包与 main 同源、远端同步状态通过实时核验。

### P8：CUDA 独立恢复线

当前保持 `hardware-blocked`，不删除、不伪装完成。真实 CUDA 主机到位后按固定顺序执行：同一 workload CPU profiler → CUDA profiler → CPU→CUDA→CPU checkpoint → 数值/结构/预算一致性 → 热点证据评审。只有 profiler 证明现有算子是瓶颈，才实现 fused/sparse kernel。

## 5. 当前唯一下一步

> 状态修订：P4-11 与 P4-12 已分别完成 editor+MCP 跨域结构收益、terminal 三域治理与恢复 Gate，P2-8/P2-9/P2-10/P2-11/P2-12/P2-13/P5-1/P5-2/P5-3 已完成 Taiji-owned 自然语言单步、多步闭环、声明式 semantic grounding/recovery、真实 IDE 语言链、digest-checked 受控写入、API/前端 transport、协议编排、grounding engine 和执行边界模块化；P6-1a/P6-1b 已完成独立 provider 接口和测试注入聊天旅程，下方历史总结中的旧当前动作已由本节末的执行入口覆盖，现行唯一动作是 P6-1c 真实 packaged semantic provider artifact / 浏览器现场 Gate。

S52 已实现并通过 **artifact consumption policy**，P2-1 至 P2-9 已分别固定“无 intent 不执行”“自然语言先成为 Taiji Goal evidence”“resolved evidence 才能进入非执行 planner”“语言 evidence 才能形成非执行语言 intent”“真实可逆 IDE 操作可经 approval、checkpoint 和 recovery 完成”“失败可停止、预算可拒绝、重启可续接”“语义分解不能携带执行字段”“Taiji 可从已验证语义 evidence、当前感知和 live affordance 形成并执行一条单步 ActionIntent”“能力声明式 semantic contract 可在无外部 parameter_bindings 时产生唯一 live binding”的边界；P2-10/P2-11/P2-12/P2-13/P5-1/P5-2/P5-3 又分别闭合多步 grounding/recovery、实时 IDE 语言绑定、digest-checked 自然语言受控写入、API/前端 transport、协议编排、grounding engine 和执行边界模块化。P3-1 固定 provider 只能提交内容寻址语义 evidence，P3-2 证明确定性 artifact 轮换不改变同任务决策，P3-3 的确定性 provider lifecycle seam 通过但真实模型资产未验收，P4-1 已证明真实 Workbench interaction-group 闭环与恢复证据，P4-2 已固定小型模拟的状态转移、credit、rollback 和 checkpoint continuation，P4-3 已证明已准入互补组的真实 Workbench 因果收益，P4-4 已证明 train-only interaction-group 学习选择与跨 seed 稳定性，P4-5 已证明同一 capability 对在多 context 上的留一族迁移，P4-6 已证明异质成员/未见组合的有界 train-only 关系 transfer，P4-7 已证明三轮 future Workbench 对照收益，P4-8 已证明三轮真实在线 Outcome 写回/准入/拒绝/重启/rollback，P4-9 已证明在线 interaction evidence 到结构候选的受控桥接，P4-10a 已证明一次结构扩容的未见任务净收益，P4-10b 已证明两个独立周期的连续结构增长，P4-11 已证明 editor+MCP 跨域结构收益与旧 workspace 保留，P4-12 已证明 terminal 三域治理与恢复。P6-1a/P6-1b 已分别完成独立 provider 接口和测试注入聊天旅程；当前后继切换为 **P6-1c 真实 packaged semantic provider artifact / 浏览器现场 Gate**，不继续追加 artifact-store 微分片。

当前明确不做：CI 全量修复、提交、推送、CUDA、Windows shell 现场美化、attached worktree 删除或 output/pytest 证据目录清理；P3-3/P6-1c 的真实模型资产仍未验收，不宣称真实 packaged provider 质量、浏览器现场或开放域持续自进化已完成。

> 执行入口覆盖：上文历史序列中的 P4-9、P4-10a、P4-10b、P4-11、P4-12、P2-8、P2-9、P2-10、P2-11、P2-12、P2-13、P5-1、P5-2、P5-3、P6-1a 与 P6-1b 当前动作已完成；当前唯一执行动作以 `03_CURRENT_EXECUTION.md` 为准，即 P6-1c 真实 packaged semantic provider artifact / 浏览器现场 Gate。
