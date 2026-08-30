# Seed / Taiji 后续详细开发计划

> 计划基线：2026-08-30。本文件给出后续阶段、依赖和验收，不创建第二个即时入口；当前只执行 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md) 指定的一项。已完成的 W0–W7 历史蓝图见 [SEED_W7_EXECUTION_PLAN_SNAPSHOT_20260829.md](../../archive/history/SEED_W7_EXECUTION_PLAN_SNAPSHOT_20260829.md)。

## 1. 收敛后的架构主线

Taiji 不从“原始神经元模拟”重新发明全部技术，也不让 Transformer/provider 接管 cognition。后续开发沿四条有所有权的链路闭合：

```text
真实任务 Outcome ─→ R5A 知识内化 ─→ Taiji-owned 选择/记忆参数
能力包与执行器 ───→ R5B 效应器注册 ─→ Workbench 身体能力
长期误差/容量压力 ─→ R5C 结构提案 ─→ shadow/holdout/lesion/rollback
已形成的 ContentPlan ─→ 语言 provider ─→ 可读输出（不参与前三条决策）
```

这四条链共享 lineage、资源账本和 checkpoint，但不能共享 owner。特别是：

- **描述性知识可以在验证内化后删除；真实执行器不能因“模型学会选择”而删除。**
- **能力注册不等于能力自治；只有 Taiji 的 structured affordance 选择、policy 准入和真实 Outcome 回写闭合后才算身体成长。**
- **结构增加不等于进化；只有旧能力保持、资源可接受且可回滚的增益才进入稳定模型。**

## 2. 阶段总览

| 顺位 | 工作包 | 目标 | 退出产物 |
|---|---|---|---|
| C0 | 计划与事实源收敛 | 去除多重下一步、拆出历史、固定阻塞边界 | 本轮计划提交 |
| C1 | W7-R5-G1 合同分离 | 分别冻结知识内化与效应器成长合同 | 两份 manifest + contract red/green |
| C2 | W7-R5A 知识内化 | 让真实 Outcome 形成可恢复、可 lesion 的 Taiji-owned 学习 | S0/S1/S2 + 可删性账本 |
| C3 | W7-R5B 效应器成长 | 从硬编码分派演进为内容寻址、可撤销注册生命周期 | L0–L3；L4 独立评审 |
| C4 | W7-R5C 自进化闭环 | 让长期证据触发局部结构变化，而非全量重训 | shadow→admit/rollback 纵向 Gate |
| C5 | 语言与 provider 成熟 | 提高可读表达并验证外部 artifact 轮换 | packaged canary + 质量/安全回归 |
| C6 | 工作台自治扩展 | 在已有只读闭环上逐级开放受控写入和长程任务 | 审批/撤销/恢复/长期评测 |
| C7 | 阻塞线补证 | 恢复 R3 Windows shell 与 R4 CUDA | 各自 S2/硬件报告 |
| C8 | 发布收口 | 全门禁、文档、安装包和远端发布一致 | release candidate |

R3/R4 未通过时不得声明相应能力，但它们不再作为 R5 的伪串行依赖。C7 一旦具备工具或硬件即可插入执行；插入后仍须单独提交，不与 R5 改动混合。

## 3. C1：W7-R5-G1 合同分离（已完成基线）

### 目标

创建两份而不是一份混合 manifest：

- `taiji_w7_r5_internalization_v1.json`：知识/规则内化、replay、可删性和遗忘边界；
- `taiji_w7_r5_effector_registry_v1.json`：能力包、执行器注册、snapshot、隔离、卸载和回滚。

选择分离的原因是失败后果不同：错误删除描述知识可以回滚 artifact；错误删除执行器等同截肢。把两者塞进一个 Gate 会允许某一侧通过掩盖另一侧未验证。

### 必须冻结的合同

- owner 与禁止依赖；真实 input/output DTO；内容 digest 与 revision；
- checkpoint 必存字段、旧版本兼容和 continuation；
- S0/S1/S2、red proof、holdout、lesion、资源、失败隔离、rollback；
- “可以物理删除什么、永远不能自动删除什么”；
- 与既有 `taiji_w7_r5_open_domain_growth_v1.json` 的依赖关系，三者不得互相冒充完成。

### Gate

已扩展 `tests/test_w7_gate_manifests.py`：缺失/混合 owner、缺 checkpoint、认知越权和错误删除边界会红，合法合同通过；R5B manifest 已推进到 `s1_dispatch_integrated`，R5A 的实现状态由其 S0/S1/S2 分阶段记录。效应器注册表仍由 R5B 独立拥有，避免 R5A 与 R5B 的 owner 边界漂移。

## 4. C2：W7-R5A 知识内化（下一阶段）

### S0：纯 DTO 转换与确定性 replay

- 在 `taiji/internalization.py` 定义不依赖 `seed_platform` 的 Outcome/evidence DTO、内容 digest 和训练样本转换器。
- 真实 affordance 必须来自 grounding；失败记录不得凭 `capability_id` 造出 affordance。
- `reward_terms` 缺测即缺失，越界直接 fail-closed；样本 ID 由 evidence + affordance 内容寻址。
- 有界 replay buffer 按内容键去重，训练/holdout 分区不可写穿。

退出：相同 checkpoint、manifest、evidence digest 和 seed 产生相同样本/账本；污染 holdout、伪造 grounding、重复 evidence 或越界奖励均红。

### S1：native checkpoint 与离线巩固（已完成 synthetic canary）

- Seed runtime 只负责把真实 Workbench Outcome 投影为 DTO；Taiji owner 批量巩固，不逐条重置优化器状态。
- checkpoint 保存 replay digest、训练计数、外部 artifact 绑定、`external → shadow → internalized/tombstone` 生命周期。
- 恢复后继续一步，选择结果、计数、lineage 与预算一致。

当前实现：`taiji/internalization_learner.py` 使用无优化器重置的归一化局部更新；父 checkpoint 在 trial mutation 前保存，holdout/retention 只读，feature/grounding lesion 可观测，恢复后 online counter 可继续。S1 synthetic native canary 已通过；S2-B 已在真实只读 Workbench 任务上补齐纵向 selection、train-only preference、外挂移除、feature/grounding lesion、旧任务保持、checkpoint recovery 与 candidate-only deletion boundary；S2-C 已用独立 seed 11/29 与 task slice 通过稳定性、资源和独立删除评审。

### S2：真实 Workbench 纵向证据

- S2-A 已完成：`SeedRuntime` 只把当前、签名验证过的只读 `workbench.evidence` 与其同 snapshot 的重投影 grounded successor affordance 投影成 `GroundedOutcomeEvidence`。运行时不可写 replay、训练 learner 或推进 lifecycle；陈旧 snapshot、旧 affordance 和失败 evidence 全部 fail-closed。
- S2-B 已完成：使用未参与训练的新任务组合执行真实只读 Workbench；比较外部规则存在与移除后的选择质量，并完成外部/内化/grounding/retention/checkpoint 五类 Gate。通过后只推进 Taiji replay 生命周期并生成可恢复的 deletion candidate，不删除外部 artifact。
- S2-C 已完成：`InternalizationStabilityTrial`/`InternalizationStabilityGate` 在独立 seed 与任务切片上汇总收益、保持、lesion、指标离散度、bounded resource counters 和 checkpoint digest；`IndependentDeletionReview` 独立检查 artifact 内容寻址、候选理由、lifecycle/manifest/checkpoint 绑定和物理删除边界。真实 Workbench 用例以两个实际 seed/task slice 在主机系统临时工作区执行通过；未通过稳定性或评审时，artifact 保持 active，不能进入 R5B/R5C。
- 只有 S2-C 稳定性与独立删除评审也通过，才允许讨论外部描述的物理删除提交；真实执行器、MCP 通道和 capability bundle 永远不在此删除范围内。R5B-L0 已先建立 capability bundle 的独立生命周期合同，物理删除仍不在范围内。

## 5. C3：W7-R5B 效应器成长

### L0：注册表重构，能力集合不变

- 已新建 `seed_platform/capability_registry.py`，完成 bundle 内容寻址、snapshot/revision、生命周期记录、disposer 约束、stale fail-closed 和 checkpoint roundtrip 合同；`WorkbenchEnvironment.execute_tool()` 已改为 registry resolve → executor identity → 原生执行表，旧 `elif tool_name` 分派已移除。全量 enabled capability 覆盖、request/approval snapshot binding、原子 replacement/rollback 和 checkpoint continuation 已有定向/直接集成证据，`scripts/training/eval_capability_registry.py` 的 R5B-S1 evaluator 也已报告 `gate.passed=true`。
- 注册返回 disposer；卸载、替换和失败回滚不直接修改全局散列表。
- `CapabilitySnapshot` 由已装配 bundle 内容生成 digest + revision；装配变化后旧 evidence 自动 stale。
- 15 个既有能力、错误码、policy 和 Legacy-off 行为逐项等价。

当前退出：registry 自身与 Workbench request/approval 的未知/禁用/陈旧能力全部 fail-closed，注册不自动激活，side-effecting bundle 缺 disposer 不能注册；核心 replacement/rollback Gate 已通过，最终退出还要在正常 CI 环境跑完整 Workbench 文件回归。

### L1：候选能力包

- 包含 schema、effect/risk/reversibility、权限、资源、版本、内容 digest、执行入口和卸载器。
- 校验、预编译和保存候选与激活分离；候选默认 `proposed`，不因落盘自动可执行。
- 文本描述只供审计，不供 provider/LLM 选择工具。

当前实现：`CapabilityCandidate` 将 bundle 与 rationale、evidence digests、有限数值 resource budget、evaluation gates 和 metadata 组成独立内容寻址 artifact；`CapabilityRegistry.propose()` 不注册 bundle，`validate_candidate()` 只进入 `validated`，`reject_candidate()` 保留拒绝审计；candidate 与 lifecycle 均进入 checkpoint，嵌套 executable-source 字段 fail-closed。L1 evaluator 已通过，下一步进入 L2 shadow 差异 Gate。

### L2：shadow 与审批

- 在相同输入上做影子执行或无副作用模拟；记录结果、after-state、资源和风险差异。
- 需要真实副作用的能力必须经过产品 policy/用户审批；Taiji 不可绕过。

当前实现：`CapabilityShadowObservation` 以 digest 绑定输入、baseline/candidate 输出、after-state 和资源指标；`evaluate_shadow()` 只解析 registry 中的 shadow bundle，不调用 executor，read-only 等价、policy deny、stale snapshot、side-effect detection 和 side-effect approval 均有独立结果。L2 evaluator 已通过，下一步进入 L3 原子激活和资源回滚。

### L3：可撤销激活

- 原子更新 snapshot、registry、资源账本和 checkpoint；失败恢复上一装配。
- 真实 Outcome 回写 R5A/R5C，但注册表本身不学习认知内容。

当前实现：registry 激活与 replacement 先计算 active bundle 的完整 resource reservation，超限会在任何状态提交前 fail-closed；`resource_ledger` 与 prior reservation 进入 checkpoint，rollback 恢复父 active set 与资源使用，并对 disposer 释放留下审计事件而不执行未知 source。L3 evaluator 已通过，L4 仅允许进入独立架构评审。

### L4：纯计算执行体替代

只有无外部副作用、可用独立 oracle 完全验证的纯计算能力可进入 L4。它需要单独架构评审，不随 L0–L3 自动推进。

2026-08-30 评审结论见 [05_R5B_L4_PURE_COMPUTATION_REVIEW_20260830.md](05_R5B_L4_PURE_COMPUTATION_REVIEW_20260830.md)：当前 workspace、IDE、terminal、MCP 和编辑能力都不能证明同时满足显式值输入、确定性无副作用、独立 oracle 三项条件，因此暂不实施任何 L4 executor。后续只有新候选满足该准入合同，才重新打开评审。

## 6. C4：W7-R5C 结构成长与自进化

复用已有 structural growth、topology ledger、neuron/region growth 与 rollback 基础，不另起“原始神经元”架构。R5C-S0 先把 R5A/R5B 的长期真实 evidence 做成内容寻址、去重、可 checkpoint 的观察窗口；触发输入来自窗口聚合后的持续错误簇、恢复不足、容量饱和、遗忘和资源压力，而不是单个 tick、规模目标或人工标签。

R5C-S0 已实现于 `taiji/structural_evidence.py` 与 `TSKV8Adapter`：standalone/cross-region 的真实 runtime observation 进入统一 ledger；窗口有明确容量、单调 tick、重复证据索引、内容 digest、封存摘要与 checkpoint roundtrip。它只保存事实，不调用 growth controller，不提交 topology。

### R5C-S1：窗口摘要 pressure 架构评审

S1 只评审并实现从多个已封存窗口形成 growth pressure 的最小事实投影。必须同时满足：跨窗口/跨任务片证据来源可追溯；holdout 与 retention 仍保持只读隔离；资源 pressure 来自 ledger/运行时计量而不是目标规模；单个窗口、单个 demo 或人工标签不能直接产生 proposal；窗口摘要 checkpoint 恢复后 digest、lineage 和去重状态一致。S1 通过后，才允许将聚合 pressure 交给既有 `AdaptiveStructuralGrowthController` 生成候选，仍不自动 admit。

S1 已实现：`StructuralRuntimeObservation` 使用显式 task slice/partition，`StructuralEvidenceLedger` 按上下文隔离窗口，`project_structural_growth_pressure()` 生成非突变 projection。S1 canary 已验证跨两个 train task slices、独立 holdout、single-slice rejection、projection roundtrip 和 ledger immutability。

### R5C-S2：candidate-only growth-controller bridge

S2 只允许 projection 经过 digest 去重和 controller 的可配置阈值映射，生成待验证的 structural candidate；必须保存 parent checkpoint，不能消耗长期 budget 或提交 topology。projection 重放、旧 checkpoint、holdout/retention 不足、controller 状态写入失败均 fail-closed；candidate 必须继续走现有 shadow/holdout/lesion/rollback 账本。

S2 已实现：`TSKV8Adapter.propose_structural_candidate_from_pressure()` 已完成 holdout-gated 单向桥接、projection digest 去重、parent checkpoint 绑定和外部 evidence tick 的 checkpoint continuation；`eval_taiji_structural_bridge.py` 的 Gate 已通过。candidate materialization 仍只进入 pending proposal，未改变 neuron/region topology，也未消耗 structural budget。

### R5C-S3：candidate validation Gate（已完成）

S3 只验证 candidate，不执行真实 admission。每个 candidate 必须先保存 trial checkpoint，再在 shadow 中记录结构变化的 after-state/resource digest，随后由独立 holdout、retention 和 lesion 检查收益是否真实、旧能力是否保持、候选是否具有因果贡献。任何失败都必须原子拒绝、归还 reservation、保留 parent active 并写入 tombstone；恢复后不能复活 rejected candidate。

S3 的最小交付是一个候选验证器和独立 Gate，复用现有 topology ledger、resource ledger 和 rollback，不新增第二套结构生命周期。通过前不得把 candidate 变成 admitted topology，也不得宣称“自进化已完成”。

S3A 已完成：`StructuralCandidateValidation` 将 holdout shadow 的 checkpoint、拓扑和预算不变性写入可恢复记录；S3B 已完成：`StructuralValidationGateDecision` 将 holdout/retention/lesion/resource/budget 阈值集中为无副作用 policy；S3C 已完成：adapter 将真实 validation record、pending proposal 和 retention/lesion/resource 指标绑定到 policy，accepted 仍 pending、failed 原子 rejected；S3D 已完成一个受限 atomic admission canary，验证通过的 candidate 单次 commit、预算精确扣减、幂等和 restore。下一步是跨 seed/task slice 稳定性与 rollback。

### R5C-S3D：atomic admission transaction（已完成）

S3D 才允许对通过 S3C 的 decision 尝试一次 topology admission。流程必须在同一可恢复生命周期内保存 parent/trial checkpoint、reservation、commit 后结构摘要和 rollback/tombstone；任一 checkpoint、资源、拓扑 roundtrip、retention 或 lesion 条件失败，都恢复 parent、归还 reservation 并留下拒绝原因。不能把 `commit_*` 直接暴露为成长触发器，也不能只靠内存状态宣称成功。

S3D 已完成一个受限 neuron admission canary：policy-approved candidate 通过既有 commit transaction 单次增长，预算精确扣减，重复调用幂等，admission result 可 checkpoint restore。下一步转向跨 seed/task slice 稳定性与 rollback，不能把单个 canary 外推为长期自进化能力。

### R5C-S4：cross-seed/task-slice stability and rollback（已完成）

S4 要求同一 candidate contract 在独立 seed、独立 task slice 和新 holdout 上重复，比较 admission 前后收益、旧任务 retention、lesion causal effect、资源和恢复时间；至少一个成功 admission 与一个失败/rollback 样本必须同时存在。只有稳定性和 rollback 都通过，才可讨论更大结构预算或多步生长。

固定生命周期：

1. 保存 parent checkpoint；
2. owner 提出局部连接、神经元、区域、记忆容量或 pruning/merge 候选；
3. 在预算内 shadow learn；
4. 独立 holdout 与 lesion；
5. 比较收益、遗忘、恢复时间、参数/连接、内存、延迟和能耗近似；
6. 原子 admit，或恢复 parent 并写 tombstone；
7. 跨 seed/任务片稳定后才扩大长期容量。

### R5C-S5：multi-step bounded growth and checkpoint continuation（已完成）

S5 证明结构成长不是一次性演示：同一 parent checkpoint 的 structural budget 为 2 时，必须能恢复并连续完成两次受 policy 约束的局部 admission；每次 admission 都要生成新的 child checkpoint、保留前一条 lineage，并精确扣减资源预算。预算归零后，即使候选的 holdout、retention 和 lesion 指标通过，也必须因 resource state/budget 不足 fail-closed，不能扩大预算、不能提交 topology、不能在重启后复活被拒绝候选。

S5 的唯一 canary 为 `scripts/training/eval_taiji_structural_continuation.py`，报告 `gate.passed=true`，覆盖：两步 topology `u0→u1→u2→u3`、两个 admission lineage、checkpoint continuation、预算 2→1→0、第三候选 rejection、rejection restore 和初始模型 bootstrap 边界。该 Gate 仍不代表无限自进化、自动预算扩容、全量重训或真实开放域质量收益。

### R5C-S6：Workbench evidence 驱动的长期增长调度（已完成）

S6 把真实 Workbench 作为外部身体/环境接入结构成长证据，但不把“工具成功”直接等同于神经元增长。每次证据必须绑定真实 `WorkbenchOutcome` 的 request/intent/call/capability snapshot 和内容摘要；结构观测使用独立、可 checkpoint 的单调运行时钟，避免把动作起始 tick（首次合法值为 0）误当作结构时钟。

调度器只消费新的 sealed evidence window，并按 `network_id/region_id/task_slice_id/partition` 隔离 train、holdout、retention；满足最小跨任务训练窗口与独立 holdout 后，才调用已有 pressure projection 和 candidate-only bridge。调度结果、已消费窗口 digest、projection digest、candidate id 和 revision 全部进入 checkpoint；重复调度必须返回 `no_new_sealed_window`，不能重复创建候选。

S6 canary 为 `scripts/training/eval_taiji_workbench_structural_scheduler.py`，报告 `gate.passed=true`：三次真实 Workbench 成功读取产生三份结构证据，跨两个 train task slice 与一个 holdout 窗口形成 candidate，restore 后 scheduler/candidate 保持，拓扑仍为 candidate-only。

### R5C-S7：调度候选的验证闭环与 Workbench 长期 continuation（已完成）

S7 将 S6 生成的 candidate 接到既有 shadow→holdout→retention→lesion→policy→admission 生命周期。Workbench 只能提供可追溯的真实执行结果和显式评估指标；不能通过单次成功、单个窗口或人工标注绕过候选验证。

S7 已交付候选级 orchestrator：从 checkpoint 恢复 scheduler candidate，保存 parent/trial，要求独立验证输入和指标，调用现有 `validate_structural_candidate_shadow()` 与 `evaluate_structural_candidate_gate()`，只有 policy 通过才允许 `admit_structural_candidate()`；失败必须保留 parent、归还预算并记录可恢复拒绝。`eval_taiji_workbench_growth_continuation.py` 已验证真实 Workbench evidence、shadow、policy、atomic admission、预算精确扣减、checkpoint restore 和重复 continuation 幂等。

### R5C-S8：多候选调度、冲突仲裁与长期 continuation（已完成）

S8 解决 S7 仍未覆盖的真实调度问题：同一批 sealed windows 可能产生多个候选，候选可能争用同一 structural budget、作用于同一区域或具有互相冲突的 topology proposal。调度器必须先建立可恢复的 candidate batch，再按内容寻址的优先级、资源成本、证据新鲜度、跨任务收益和区域冲突进行确定性仲裁；不能依赖列表顺序、随机数或“先到先得”隐藏决策。

S8 的边界：仲裁只决定哪些 candidate 进入验证队列，不直接 admission；同一 batch 中未获选候选必须记录 deferred/rejected 原因，不能丢失或自动复活。通过验证的候选仍逐个走 shadow→policy→admission，预算 reservation 必须原子隔离，任一失败不能污染其他候选。checkpoint 必须恢复 batch、排序依据、reservation、候选状态和审计 digest；重复调度必须幂等。

S8 canary 需要至少三类候选：不同区域无冲突、同区域资源冲突、同一 projection 重放；覆盖确定性排序、预算隔离、deferred/rejected 恢复、单候选 admission 后的剩余批次 continuation，以及重复运行输出 digest 一致。`scripts/training/eval_taiji_structural_arbitration.py` 已报告 `gate.passed=true`。S8 只建立 reservation，不并行提交 topology，也不自动增加长期预算。

### R5C-S9：多轮 continuation、跨区域容量压力与可逆回滚（已完成）

S9 把 S8 的单个 batch 放进可恢复的长期闭环：每轮只处理显式提供验证输入的 selected candidate，未处理候选保持 reservation；容量压力必须以只读快照反映区域占用、候选队列、reservation 与 structural budget，不能用“规模目标”替代真实压力。admitted candidate 必须绑定 parent/child checkpoint，回滚必须恢复 parent topology、重开对应预算并留下内容寻址 audit；回滚后新的 Workbench evidence 可以让旧 deferred candidate 重新参与仲裁，但不能静默复活旧状态或覆盖历史。

S9 已实现于 `taiji/structural_continuation.py` 与 `TSKV8Adapter`，并由 `scripts/training/eval_taiji_structural_continuation_recovery.py` 报告 `gate.passed=true`：首轮和恢复后的第二轮 admission、跨区域 capacity pressure、rollback、checkpoint restore、新 evidence 再仲裁和重复 rollback 幂等均通过。该 Gate 仍不代表并行 admission、无限预算或开放域质量收益。

### R5C-S10：真实 Workbench 多区域证据驱动的候选批次调度（已完成）

S10 的目标是清除 S8/S9 canary 中“候选由测试 harness 组装”的最后一处边界：从两个以上真实 Workbench 区域与 task slice 的成功 Outcome 生成各自的 sealed evidence window 和 pressure projection，再由运行时一次性建立 `StructuralCandidateBatch`，进入 S8 的确定性仲裁与 S9 的 continuation/rollback。候选身份必须绑定真实 request/intent/call/capability snapshot、窗口 digest、region/task-slice 和 parent checkpoint，不能由前端或 provider 直接指定结构操作。

S10 必须证明：多区域真实 evidence 可产生多个互不相同且可追溯的 candidate；同区域冲突、跨区域预算不足、重复 Outcome、checkpoint restore 和新一轮 evidence 都保持确定性；Workbench 只提供事实，不绕过 shadow、holdout、retention、lesion、resource policy 或 rollback。S10 不扩展 CUDA、CI、开放域语言质量或无限结构预算。

S10 已实现：`schedule_structural_candidate_batch_from_workbench_evidence()` 对多个真实 Workbench 区域分别运行 sealed-window scheduler，再把新 candidate 一次性交给 `StructuralCandidateBatch`；`StructuralWorkbenchBatchScheduleResult` 保存请求 digest、源窗口、区域、candidate、batch 和 scheduler revision。`scripts/training/eval_taiji_workbench_multi_region_batch.py` 已报告 `gate.passed=true`：6 次真实 `workspace.read` 产生两个区域的 2 train+1 holdout 窗口，两个 candidate 在同一 batch 中被确定性 reserved，checkpoint restore 与重复调度保持相同结果，topology/budget 未被仲裁改变。

### R5C-S11：真实多区域 batch 的 shadow→policy→admission→rollback 全生命周期（已完成）

S11 把 S10 生成的真实多区域 batch 接入完整 continuation，而不是停在 candidate-only。每个 selected candidate 必须使用与其真实 evidence/region 对齐的 holdout 输入、retention、lesion 和 resource 指标，按既有 S7/S8 contract 逐个 shadow→policy→admission；一个区域候选失败时，其他区域候选不能被污染，未处理 reservation 必须可跨 checkpoint 保持。

S11 必须证明：真实多区域 batch 中两个 candidate 可以在独立验证后跨 checkpoint 逐个 admission；同区域冲突或单候选 policy rejection 只影响对应 candidate；admission 后容量压力可观察，任一 admitted candidate 可回滚到自己的 parent checkpoint 并重开预算；重复 continuation/rollback 幂等，Workbench evidence、provider 和前端都不能绕过结构 Gate。S11 不扩展 CUDA、CI、无限预算或开放域语言质量。

S11 已实现：复用 `continue_structural_candidate_batch()` 将 S10 的真实多区域候选逐个走 shadow→policy→atomic admission，并在独立 checkpoint 分支中验证单候选失败隔离；`rollback_structural_candidate_batch()` 恢复对应 parent topology、重开预算并保留 rollback audit。`scripts/training/eval_taiji_workbench_multi_region_lifecycle.py` 已报告 `gate.passed=true`：第一候选先 admission，第二候选失败不污染第一候选；另一恢复路径完成第二候选 admission，随后回滚且 checkpoint restore 后重复回滚幂等。

### R5C-S12：真实 Workbench replay 驱动的 validation artifact（已完成）

S12 解决当前闭环仍保留的最后一个证据边界：Workbench Outcome 已真实接入，但 holdout、retention、lesion、resource 等验证指标仍由 canary 显式传入。`WorkbenchStructuralValidationArtifact` 现已把真实 Workbench replay、候选前后 checkpoint、独立 holdout 输入/输出、retention 对照、lesion 对照与资源计量绑定到同一内容寻址 artifact，再由既有 policy 读取，不让“工具成功”直接变成准入分数。

S12 已证明：同一真实 Outcome replay 在 checkpoint restore 后保持相同 validation artifact digest；缺失、篡改或 holdout replay 不匹配的数据 fail-closed；artifact 只提供事实，policy 仍决定 admission；单候选失败不污染其他候选，且 artifact 与 checkpoint lineage 可审计。`api/seed_runtime.py` 已提供同一公共入口，`scripts/training/eval_taiji_workbench_validation_artifact.py` 报告 `gate.passed=true`。S12 不扩展 CUDA、CI、无限预算或开放域语言质量。

### R5C-S13：多区域 batch 的 replay validation artifact continuation（已完成）

S13 已将 S12 从单候选闭环推进到 S10/S11 的真实多区域 batch：每个 selected candidate 由其所属 region、task slice 和真实 Workbench Outcome 生成独立 validation artifact，再由 batch continuation 消费；调用方不再直接注入 holdout、retention、lesion、resource 指标集合。artifact 之间不能跨 region、跨 candidate 或跨 parent checkpoint 混用，单个 artifact 失败只能释放对应 reservation，不能污染其他候选。

S13 已证明：多区域 batch 的 artifact 集合按 candidate/artifact digest 建立并 checkpoint；第一候选 admission 后第二候选可绑定新的 parent checkpoint；任一 artifact 缺失、篡改或错配都会 fail-closed；合法 artifact 仍逐候选经过既有 shadow→policy→atomic admission，失败分支维持 S11 的隔离，恢复后 batch/artifact digest 与重复消费幂等。`api/seed_runtime.py` 已提供公共入口，`scripts/training/eval_taiji_workbench_validation_artifact_batch.py` 报告 `gate.passed=true`。S13 不扩展 CUDA、CI、无限预算或开放域语言质量。

### R5C-S14：独立 replay measurement owner（已完成）

S14 解决 S12/S13 仍暴露的最后一处硬编码边界：canary 不再把 metrics 直接传入 batch continuation，artifact 构建阶段也不再由 evaluator 手工填写 holdout gain、retention regression、lesion effect 和 resource state。`StructuralValidationMeasurements` 作为 Taiji-owned、可复用的 replay measurement owner，由 baseline/candidate/lesion 的实际观测和原始容量 pressure 计算 metrics，再把计算结果和输入 digest 一起写入 artifact。

S14 已证明：同一 parent checkpoint、候选 trial、目标/旧任务 replay 和资源观测产生确定性 metrics；任何缺失 baseline、错配 candidate、越界测量或跨任务混用都 fail-closed；metric producer 不改变 topology、不拥有 admission 权限，policy 仍只消费其输出；checkpoint restore 后测量和 artifact digest 一致。`scripts/training/eval_taiji_workbench_validation_measurements.py` 报告 `gate.passed=true`。S14 不扩展 CUDA、CI、无限预算或开放域语言质量。

### R5C-S15：实测多区域 artifact batch（已完成）

S15 已把 S14 的 measurement owner 接入 S13 多区域 artifact batch。每个 candidate 都从自己的 baseline/candidate/lesion replay 与容量快照获得 measured metrics，形成独立 artifact；batch continuation 只消费这些 artifact，不恢复任何批次级手工分数。第一候选 admission 后，第二候选从新 parent checkpoint 重新测量并继续，确保顺序性与 lineage 正确。

S15 已证明：两个真实 Workbench region candidate 的 policy 分别消费自己的 measured holdout/retention/lesion/resource 指标，artifact resource digest 对应原始容量测量，增量 batch admission 完成，artifact batch checkpoint restore 与重复消费幂等。`scripts/training/eval_taiji_workbench_measured_artifact_batch.py` 报告 `gate.passed=true`。S15 不扩展 CUDA、CI、无限预算或开放域语言质量。

### R5C-S16：多轮 measured evidence continuation 与 rollback（已完成）

S16 将当前已验证的单轮 measured artifact batch 放入长期循环：新的真实 Workbench evidence 必须形成新的 sealed windows、measurement artifacts 和 candidate batch；后续轮次只能在上一轮 checkpoint/rollback lineage 上继续，不能复用陈旧 artifact、重复扣预算或把一次性收益当作长期稳定。

S16 已证明：两轮真实 evidence→measured artifact→batch admission 可以跨 checkpoint 延续；第二轮使用新 sealed windows、新 candidate 和新 artifact digest，旧 artifact 在 parent 变化后 fail-closed；任一候选 rollback 恢复到该轮第一候选后的 parent topology、budget、artifact batch 和 reservation，其他已验证区域不受污染；多轮后的 retention/lesion/resource 指标仍由 measurement owner 计算。期间发现并修复 `StructuralGrowthScheduleState` 的全局 cooldown 饿死问题，改为按 `network_id:region_id` 保存 stream cursor，同时兼容旧 checkpoint。`scripts/training/eval_taiji_workbench_measured_multi_round.py` 报告 `gate.passed=true`。S16 不扩展 CUDA、CI、无限预算或开放域语言质量。

### R5C-S17：多轮 artifact/measurement integrity 与 provenance closure（已完成）

S17 收敛 S16 暴露的内容寻址边界：validation artifact 已验证 payload digest，但 measurement payload 的 `measurement_digest` 在反序列化时仍只做非空检查；此外，多轮 ledger 需要明确区分“当前轮新 evidence”与“历史 lineage evidence”，避免后续实现把可追溯历史误当成新的触发事实。

S17 已证明：篡改任一 measurement metric、raw probe digest、resource digest 或 measurement format 都 fail-closed；artifact 显式绑定 measurement digest，且 artifact digest、candidate evidence、parent/trial checkpoint 和 source window lineage 可交叉验证；旧格式 artifact（无 measurement digest）仍可读取；checkpoint restore 后 integrity/provenance ledger 与 Gate 结果一致。`scripts/training/eval_taiji_workbench_integrity.py` 报告 `gate.passed=true`。S17 不扩展 CUDA、CI、无限预算或开放域语言质量。

### R5C-S18：多轮 ledger compactness 与跨轮 evidence 消费审计（下一步）

S18 处理多轮运行进入长期阶段后的审计边界：S16 已证明新 stream cursor 和新窗口能持续推进，S17 已证明 artifact/measurement payload 完整，但当前 pressure projection 仍会读取历史 sealed summaries，且多个 lineage ledger 依赖统一有界截断。下一步要明确“历史可追溯”与“本轮可消费”的区别，并验证在多轮压缩后仍不会重复触发、丢失 parent lineage 或让旧 artifact 重新获得准入资格。

S18 必须证明：每个 evidence stream 的已消费窗口、保留窗口和 provenance summary 在容量上界内可恢复；窗口被 compact 后仍保留内容 digest、来源 task slice、partition、parent/child lineage 和消费状态；重复调度不会创建候选，跨轮旧 artifact 仍 fail-closed；compact 前后 candidate/batch/artifact/admission/rollback 的可审计摘要一致。S18 不扩展 CUDA、CI、无限预算或开放域语言质量。

禁止全量从零训练作为默认迭代方式；允许基础模型版本升级时进行受控迁移，但必须保留父 lineage、旧能力回归和可逆转换。

## 7. C5–C6：语言成熟与工作台自治

### 语言/provider

- `native-readable` 保持默认产品表层；`structured-stub` 只作为显式无损调试 codec。
- 完成真实外部 provider artifact 的 packaged-client 轮换、失败回退、重启重绑和安全 canary；R1 的 native-only S2 不冒充该能力。
- 质量评估绑定相同 ContentPlan，比较可读性、约束保持、事实遗漏和 fallback；provider 不改 intent、tool 或 memory。

### 工作台自治

- 继续沿已有 IDE 语言识别/高置信自动切换、policy、预览、审批、undo 和 Outcome 链扩展。
- 默认自治先从可逆、小影响写入开始，再到跨文件任务；每层设置预算、人工接管点和 checkpoint continuation。
- HF/GGUF/Transformer 继续只存在于 provider/离线对照边界，前端不得恢复模型格式切换残留。

## 8. C7–C8：阻塞线与发布

### R3 Windows shell

工具可用后只补真实窗口、任务栏、托盘通知、高 DPI、键盘导航和 reduced-motion 证据；已通过页面证据不重做。失败不修改 Taiji checkpoint。

### R4 CUDA

真实 CUDA 主机到位后，先运行同一 CPU workload 的 profiler，再验证 CPU→CUDA→CPU checkpoint、结构/lineage/预算一致和数值容差；只有热点证据支持时才评审 fused/sparse kernel。

### 发布收口

- 后端、前端、桌面、Legacy-off、checkpoint、manifest、OpenAPI、安全和安装包 Gate 全部执行且无 skipped；
- `dist/Seed/Seed.exe`、前端字节、报告 digest、文档状态和 Git commit 对齐；
- 当前计划只留未完成阶段，已完成执行日志归档；
- 发布前实时检查本地 `main`、其他 worktree refs、`origin/main` 和远端同步，不把“已提交”写成“已推送”。

## 9. 每个 slice 的固定交付格式

1. 先写 red contract 与失败证据；
2. 做最小 owner 内实现，不顺手扩展相邻能力；
3. 运行与改动范围匹配的阻塞 Gate；
4. 生成内容寻址报告，记录未验证边界；
5. 更新 manifest、实现事实和唯一下一步；
6. 单一主题提交；若 CI 红，下一提交只修 CI，不能继续堆功能。
