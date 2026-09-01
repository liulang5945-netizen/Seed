# Seed / Taiji W7-R5 实施与证据索引

> 计划基线：2026-08-30；2026-08-31 起仅作为 W7-R5 已完成分片与原始 Gate 的证据索引，不再决定后续阶段。当前只执行 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md) 指定的一项，S52 后的交付顺序阶段见 [01_SCOPE_AND_PHASES.md](01_SCOPE_AND_PHASES.md) 第 6–7 节。已完成的 W0–W7 历史蓝图见 [SEED_W7_EXECUTION_PLAN_SNAPSHOT_20260829.md](../../archive/history/SEED_W7_EXECUTION_PLAN_SNAPSHOT_20260829.md)。

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

### R5 编号的接管关系（2026-09-01 收敛）

上述 R5A/R5B/R5C 三链在 2026-08-31 的路线修正后已被本文件第 9–18 节的 E 系列阶段接管。本文件自此只作为 R5 已完成分片的证据索引，**其中任何“下一阶段”“后续”表述都不再产生动作**：

| 原 R5 链 | 接管阶段 | 接管后的形态 |
|---|---|---|
| R5A 知识内化 | E2/E4 | 语料与经验合同 + `CognitiveInternalizationArtifact`（owner: Taiji） |
| R5B 效应器成长 | E5/E6 | Seed 客户端插件生命周期 + `ClientCapabilityInheritanceCandidate`（owner: Seed 客户端） |
| R5C 结构成长 | E8 | 长期数据飞轮下的局部结构候选与单项回滚 |
| R3 Windows shell | 仍为 `tool-blocked` | 不被 E 系列接管，工具可用后按 C7 单独补证 |
| R4 CUDA | E9（`hardware-blocked`） | 不阻塞 E1–E8 的 CPU/原生正确性 |

接管不等于作废：R5A/R5B 已通过的 Gate 与报告仍是 E 系列的前置证据，但**新的准入判定只走 E 系列的 Gate**，避免同一能力在两套编号里各自声明完成。

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

2026-09-01 收敛后，C1–C8 之后的当前主线为 E1–E9 原生进化与客户端体化线，路线决策、目标架构、语料/经验合同、Skill/MCP/插件合同、归因规则、未闭合阶段和 owner 见本文件第 9–18 节；E 系列不重编号 C 系列，两者按 owner 与 Gate 直接衔接。

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

已扩展 `tests/test_w7_gate_manifests.py`：缺失/混合 owner、缺 checkpoint、认知越权和错误删除边界会红，合法合同通过；R5B manifest 当前实现状态为 `l4_reviewed_no_go`，R5A 为 `s2_stability_gate_implemented`，两者的分阶段记录由各自 manifest 的 `implementation` 持有。效应器注册表仍由 R5B 独立拥有，避免 R5A 与 R5B 的 owner 边界漂移。

## 4. C2：W7-R5A 知识内化（已交付基线，由 E2/E4 接管）

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

当前实现：`CapabilityCandidate` 将 bundle 与 rationale、evidence digests、有限数值 resource budget、evaluation gates 和 metadata 组成独立内容寻址 artifact；`CapabilityRegistry.propose()` 不注册 bundle，`validate_candidate()` 只进入 `validated`，`reject_candidate()` 保留拒绝审计；candidate 与 lifecycle 均进入 checkpoint，嵌套 executable-source 字段 fail-closed。L1 evaluator 已通过，L2 shadow 差异 Gate 与 L3 资源/rollback Gate 也已完成。

### L2：shadow 与审批

- 在相同输入上做影子执行或无副作用模拟；记录结果、after-state、资源和风险差异。
- 需要真实副作用的能力必须经过产品 policy/用户审批；Taiji 不可绕过。

当前实现：`CapabilityShadowObservation` 以 digest 绑定输入、baseline/candidate 输出、after-state 和资源指标；`evaluate_shadow()` 只解析 registry 中的 shadow bundle，不调用 executor，read-only 等价、policy deny、stale snapshot、side-effect detection 和 side-effect approval 均有独立结果。L2 evaluator 与 L3 原子激活、资源 reservation、rollback Gate 均已通过。

### L3：可撤销激活

- 原子更新 snapshot、registry、资源账本和 checkpoint；失败恢复上一装配。
- 真实 Outcome 回写 R5A/R5C，但注册表本身不学习认知内容。

当前实现：registry 激活与 replacement 先计算 active bundle 的完整 resource reservation，超限会在任何状态提交前 fail-closed；`resource_ledger` 与 prior reservation 进入 checkpoint，rollback 恢复父 active set 与资源使用，并对 disposer 释放留下审计事件而不执行未知 source。L3 evaluator 已通过，L4 仅允许进入独立架构评审。

### L4：纯计算执行体替代

只有无外部副作用、可用独立 oracle 完全验证的纯计算能力可进入 L4。它需要单独架构评审，不随 L0–L3 自动推进。

2026-08-30 评审结论见 [05_R5B_L4_PURE_COMPUTATION_REVIEW_20260830.md](../../archive/history/roadmap_shards/05_R5B_L4_PURE_COMPUTATION_REVIEW_20260830.md)：当前 workspace、IDE、terminal、MCP 和编辑能力都不能证明同时满足显式值输入、确定性无副作用、独立 oracle 三项条件，因此暂不实施任何 L4 executor。后续只有新候选满足该准入合同，才重新打开评审。

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

### R5C-S18：多轮 ledger compactness 与跨轮 evidence 消费审计（已完成）

S18 处理多轮运行进入长期阶段后的审计边界：S16 已证明新 stream cursor 和新窗口能持续推进，S17 已证明 artifact/measurement payload 完整，但当前 pressure projection 仍会读取历史 sealed summaries，且多个 lineage ledger 依赖统一有界截断。下一步要明确“历史可追溯”与“本轮可消费”的区别，并验证在多轮压缩后仍不会重复触发、丢失 parent lineage 或让旧 artifact 重新获得准入资格。

S18 已证明：`StructuralEvidenceLedger.audit_consumption()` 区分 evaluated、consumed、unconsumed、retained、compacted 与 orphaned window digest，并按 network/region stream 给出消费状态；`compact_consumed_windows()` 只移动已消费且不是各 stream 最新保留窗口的 sealed summary，将 active evidence index 转成有界 provenance record，保留 window/evidence digest、来源 network/region、task slice、partition、tick 范围和消费 scheduler revision。adapter 提供只读 audit 与显式 compaction 入口，active projection 不读取 compacted history，重复 compact/duplicate replay 幂等，篡改 provenance、ledger checkpoint restore 和新一轮 evidence continuation 均有 Gate。`scripts/training/eval_taiji_structural_evidence_compaction.py` 报告 `gate.passed=true`。S18 不扩展 CUDA、CI、无限预算或开放域语言质量。

### R5C-S19：压缩后 provenance-aware pressure projection 与跨轮候选边界（已完成）

S19 已处理 S18 留下的消费边界：新增 `StructuralEvidencePressureSnapshot`，在不恢复历史窗口为 active evidence 的前提下保存 train/holdout/retention pressure 聚合所需的最小统计；`project_structural_growth_pressure()` 读取 active summaries + 明确传入的历史 snapshot，压缩前后输出相同 projection digest。scheduler 仍只以未评估的 active sealed window 触发，因此重复调度不会创建 candidate，candidate、snapshot、audit 与 ledger checkpoint 恢复保持一致；tampered snapshot fail-closed。`scripts/training/eval_taiji_structural_provenance_projection.py` 报告 `gate.passed=true`。

S19 已证明：同一 active evidence 集合在 compact 前后产生等价 pressure/candidate identity；compacted digest 只能通过 snapshot 作为历史聚合事实，不能单独触发新 pressure；新窗口仍可继续进入 active ledger；旧 candidate/batch/artifact 的 parent/evidence digest 校验边界未被放宽；checkpoint restore 保持 candidate 与 snapshot。candidate/batch/artifact/admission/rollback 在多轮压缩前后的全链路摘要一致性由 S20 显式审计。S19 不扩展 CUDA、CI、无限预算或开放域语言质量。

### R5C-S20：跨轮 candidate/artifact/admission/rollback lineage 审计（已完成）

S20 将 S18/S19 的 evidence 历史层与既有 candidate batch、validation artifact、admission、rollback 账本做一次跨 checkpoint 对齐。重点是证明压缩只改变 evidence 的存储形态，不改变 candidate 的 source window/evidence digest、parent/child checkpoint、reservation、预算回退和旧 artifact 的 stale/错配拒绝。

S20 必须证明：compact 前后同一 candidate/batch/artifact/admission/rollback 链的 digest 与 lineage 摘要一致；压缩后的历史不会让被拒绝、已回滚或旧 parent artifact 重新进入准入；新 evidence 形成的新 batch 使用新 source digest；checkpoint continuation 与 rollback 仍是原子且幂等。S20 不扩展 CUDA、CI、无限预算或开放域语言质量。

S20 已实现并通过：`scripts/training/eval_taiji_structural_lineage_compaction.py` 复用真实 Workbench measured artifact batch，验证 compaction 只改变 evidence 存储与 checkpoint digest，不改写 candidate/batch source/evidence lineage；压缩后旧 parent artifact fail-closed 且不改变 topology/budget；原未压缩 parent 分支仍能准入并 rollback；admission/rollback checkpoint digest、reservation 和预算恢复保持绑定；compacted checkpoint 的 candidate/audit 与 measurement provenance 可恢复。报告 `gate.passed=true`。S20 不扩展 CUDA、CI、无限预算或开放域语言质量。

### R5C-S21：长序列压缩保留、分支恢复与资源账本压力 Gate（已完成）

S21 的目标是把 S20 的单次跨轮 lineage 审计推进到长序列边界：在多个 evidence round、多个 stream、混合 unconsumed/consumed/compacted 状态、候选 admission/rollback 和 reservation 释放同时存在时，确认 compaction 的保留上限、pressure snapshot 数量和结构资源账本都保持有界且可恢复。该 slice 仍不扩大预算，也不把“能持续运行”误写成无限自进化。

S21 必须证明：连续至少三轮真实 Workbench evidence 后，按 stream 的 `keep_latest_per_stream` 与 compacted provenance 上限稳定；每轮 checkpoint restore 后 active/unconsumed window、snapshot、candidate、artifact batch、reservation、admission/rollback audit 的摘要一致；压缩中途失败不产生半提交状态；资源压力达到上限时只隔离当前候选/批次，不污染其他 stream；释放 reservation 后新 evidence 可以在相同上限内继续推进；旧 artifact 与旧 batch 仍 fail-closed。Gate 只允许通过现有 bounded CPU/native canary，不涉及 CUDA、CI、前端或物理删除。

S21 已实现并通过：`scripts/training/eval_taiji_structural_long_sequence_stress.py` 在有限 compacted-window cap=16 下执行三轮、每轮六次真实 Workbench evidence；第一轮压缩后为 4 compacted/2 active，第二轮为 10 compacted/2 active，第三轮达到 16 compacted/2 active。未消费窗口在调度前保持 active，降低 cap 的 OverflowError 保持 ledger digest、active/compacted 集合不变；第二、三轮 rollback 恢复预算并隔离同批另一候选，最终 candidate、pressure snapshot、admission/rollback、audit 与 cap 均 checkpoint roundtrip。报告 `gate.passed=true`。S21 不扩展 CUDA、CI、无限预算或开放域语言质量。

### R5C-S22：受保护 candidate/batch lineage 保留（已完成）

S22 的目标是补齐 S21 暴露的另一类长期边界：evidence ledger 已有明确 compact cap，但 candidate/batch 记录若继续按字典排序盲删，可能误删仍有 reservation、仍待延续或仍可 rollback 的 lineage。第一阶段先把“活动 lineage 不得淘汰”从隐含约定变成 adapter-owned retention contract。

S22 已实现并通过：`_record_structural_candidate_batch()` 只淘汰终结 batch；活动 reservation、deferred candidate 和尚未 rollback 的 admitted candidate 始终保留。pending candidate queue 只淘汰未被活动 batch 引用的项；validation、gate、artifact、admission、rollback 与 artifact-batch 记录在仍有 live continuation/rollback 依赖时保留。lineage limit=1 的真实 Workbench retention canary 与 R5C 定向回归通过，`eval_taiji_structural_lineage_retention.py` 报告 `gate.passed=true`。当受保护记录自身超过目标上限时暂不强删，避免以“达标”为名破坏 lineage。

### R5C-S23：跨 candidate/artifact/admission/rollback 账本的协同终结保留 Gate（已完成）

S23 负责完成 S22 暂未做的协同淘汰：候选、artifact、artifact batch、validation、gate decision、admission、rollback、capacity/schedule audit 必须作为一个有引用关系的 lineage 图进行终结判断，而不是每个列表单独按最后 N 条截断。

S23 必须证明：只有没有 active reservation、pending/deferred candidate、未完成 artifact batch、可用 rollback、pending topology proposal 或 checkpoint 引用的终结子图才可一起淘汰；淘汰必须原子，不能留下孤立 artifact/admission/rollback；被淘汰的旧链再次 replay、rollback 或 admission 必须稳定 fail-closed 且不会复活；受保护子图超出容量时返回可观测 retention pressure，不静默丢数据；checkpoint restore 与新 evidence 继续调度保持确定性。S23 只做 native/CPU lineage graph 与可逆审计，不扩大预算、不执行物理删除、不处理 CI/CUDA/前端。

S23 已实现并通过：新增 `StructuralLineageRetentionResult` 与 `TSKV8Adapter.compact_structural_lineage_history()`，以 candidate batch 为根将 candidate、artifact/artifact-batch、validation、gate、admission、rollback、proposal 和相关 schedule audit 作为同一 lineage 子图协同淘汰；只有无活动 reservation、pending/deferred continuation、pending topology proposal 或 rollback 依赖的终结子图可被移除，异常时原子恢复。真实 Workbench canary `scripts/training/eval_taiji_structural_lineage_compaction.py` 报告 `gate.passed=true`，覆盖活动 lineage 保留、关联记录整组移除、content-addressed result、checkpoint restore、旧 replay/rollback fail-closed、压缩后新 evidence 调度确定性和 protected pressure；新增定向用例 `4 passed`，既有 R5C 定向回归 `31 passed`。S23 不扩大预算、不执行物理删除、不处理 CI/CUDA/前端。

### R5C-S24：运行时维护边界接入协同 lineage 压缩与自动触发审计 Gate（已完成）

S24 负责把 S23 的显式压缩能力接入结构运行时维护边界，但不把 retention 变成隐式后台删除。维护入口必须在 evidence/scheduler/continuation 的确定性边界调用协同压缩，输入使用显式 `max_batches` 或配置快照，输出持久化 retention result digest、被保护 batch、pressure 和删除计数；无可淘汰子图时保持纯观察，不改变 topology、budget、active evidence 或 provider/Workbench 行为。

S24 必须证明：自动触发只发生在明确维护周期且幂等；维护前后活动 reservation、pending/deferred candidate、未完成 artifact batch、rollbackable admission 和 pending topology proposal 不变；终结 lineage 的各关联 ledger 同步变化；失败不产生半压缩 checkpoint；restore 后再次进入维护周期结果确定；retention pressure 可向上层观测但不能绕过人工/策略边界执行结构成长。Gate 只覆盖 native/CPU 运行时维护与真实 Workbench evidence，不扩大 structural budget、不做物理删除、不处理 CI/CUDA/前端。

S24 已实现并通过：`run_structural_maintenance_cycle()` 增加显式 `lineage_retention_max_batches`；默认维护不触发压缩，只有维护 owner 明确传入正上限时才执行 S23 协同 lineage 保留。结果 digest、保护 batch、pressure、删除计数进入 structural checkpoint，source/target digest 排除 audit 自身以确保重复维护幂等；非法上限原子失败，维护前后 topology/budget 保持不变。真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_maintenance.py` 报告 `gate.passed=true`，S18–S24 定向回归为 `24 passed`。S24 不引入隐式后台线程、不扩大预算、不执行物理删除、不处理 CI/CUDA/前端。

### R5C-S25：SeedRuntime 显式 lineage maintenance audit 可观测契约 Gate（已完成）

S25 负责把 S24 的 adapter-owned maintenance 边界接到 SeedRuntime 的显式调用面，保证产品/runtime 层能够读取一份完整、内容寻址、可 checkpoint 恢复的 maintenance audit，而不需要直接依赖 Taiji 内部 ledger。该入口只编排已存在的 candidate maintenance 与可选 lineage retention，不拥有结构成长决策权，不自动启动后台清理，也不把 retention pressure 当作准入信号。

S25 必须证明：SeedRuntime 的显式入口能返回 candidate maintenance results 与 retention audit 的稳定 payload；缺省参数不触发 retention，正上限才触发一次 S24 维护；checkpoint restore 后 audit/result digest、保护 lineage 和 pressure 保持一致；非法上限、旧/缺失 runtime state 与 payload 篡改 fail-closed，失败不改变 topology、budget 或 ledger；入口不会把 provider/frontend/Workbench 副作用混入 Taiji structural maintenance。Gate 只覆盖 native/CPU runtime contract 与真实 Workbench evidence，不扩大预算、不做物理删除、不处理 CI/CUDA/前端。

S25 已实现并通过：新增 `StructuralMaintenanceAudit`，将 SeedRuntime 每次显式维护调用的 candidate results、当次 retention result 和 structural runtime tick 组成 content-addressed payload；SeedRuntime 只负责稳定投影，Taiji 继续拥有 candidate lifecycle、retention policy、topology 与 budget。默认调用不会触发或重放 checkpoint 中旧 audit，显式正上限才投影本次 retention；Seed checkpoint restore 保留 Taiji audit state，非法上限和篡改 payload fail-closed。真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_runtime.py` 报告 `gate.passed=true`，新增定向用例 `3 passed`，S18–S25 相关回归 `27 passed`。S25 不启动后台清理、不改变 provider/frontend/Workbench 副作用边界、不处理 CI/CUDA/前端。

### R5C-S26：runtime 只读 structural maintenance 状态投影 Gate（已完成）

S26 负责把 S25 的调用级 audit 接入 SeedRuntime 的只读状态查询，使上层能够看到当前 structural runtime tick、最近一次 retention audit、保护 lineage 与 pressure，而不必读取 Taiji 内部 ledger。状态查询必须是纯 projection：不得触发 maintenance、改变 checkpoint、重算 candidate、执行 provider/frontend/Workbench 副作用，且在没有 audit 时返回明确空态。

S26 必须证明：状态查询在维护前后不会产生副作用；显式维护后能完整投影最近 retention audit 的 format/digest/status/pressure/保护与删除摘要；Seed checkpoint restore 后投影一致；没有历史 audit 时不会伪造结果；状态返回值篡改或缺失字段不能被当作新的结构决策输入。Gate 只覆盖 native/CPU runtime status contract，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S26 已实现并通过：`SeedRuntime.structural_maintenance_status()` 以及 `status()["structural_maintenance"]` 提供只读 projection，包含 format、structural runtime tick、最近 retention audit 和 pressure；没有 audit 时返回显式空态。真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_status.py` 报告 `gate.passed=true`，新增定向用例 `3 passed`；查询前后 checkpoint digest、topology、budget 不变，Seed checkpoint restore 后状态一致。S26 不把 status 作为结构决策输入，不启动后台维护，不处理 CI/CUDA/前端。

### R5C-S27：版本化、内容寻址的 lineage retention policy Gate（已完成）

S27 负责把 S24–S26 目前传递的裸 `max_batches` 收敛成 Taiji-owned、可 checkpoint、可审计的 retention policy。policy 必须携带 format/revision/上限和保护规则的明确身份，维护调用使用 policy snapshot 而不是隐式读取全局常量；旧的显式整数入口只保留为受控兼容层，不能成为新的内部事实源。policy 只决定历史保留边界，不得决定 candidate 准入、topology growth、provider 行为或 frontend 状态。

S27 必须证明：相同 policy payload 在不同运行实例产生相同 policy digest；缺失/未知 revision、非正上限、越界保护规则和篡改 digest fail-closed；policy checkpoint restore 后 retention result、status projection 和 lineage 保护集合一致；切换 policy 只影响后续显式 maintenance，不改写已有 audit 或当前 topology/budget；旧整数兼容入口与 policy 入口不能产生两套语义。Gate 只覆盖 native/CPU retention policy，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S27 已实现并通过：新增 `StructuralLineageRetentionPolicy`，以 revision、max_batches、固定安全 protection rules 和 policy digest 作为 retention 的唯一显式身份；`max_batches` 仅作为兼容输入在边界转换为同一 policy，policy 与 retention result 一起 checkpoint，SeedRuntime audit/status 同步投影 policy。未知 revision、非法保护集合、policy/result 不一致、双重输入和 digest 篡改均 fail-closed；切换 policy 只影响后续显式 maintenance。真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_policy.py` 报告 `gate.passed=true`，新增定向用例 `4 passed`，S18–S27 相关回归 `34 passed`。S27 不开放后台维护、不扩大预算、不处理 CI/CUDA/前端。

### R5C-S28：retention policy 可迁移生命周期与回滚 Gate（已完成）

S28 负责验证 policy revision 演进不会破坏既有 lineage：新 policy 必须通过显式 migration 产生，保留旧 policy/result 的 provenance，并能在 migration 失败时恢复原 policy、retention audit、status projection 与结构状态。迁移只处理 retention policy schema/边界，不修改 candidate、topology、budget、provider 或 Workbench 副作用。

S28 必须证明：旧 v1 policy 可被明确识别并迁移到兼容版本；迁移前后 safe protection invariants 不减弱，policy/result/status digest 绑定可追溯；未知版本、非法迁移和目标 policy 不兼容时 fail-closed 且 checkpoint 原子不变；迁移后的显式 maintenance 与未迁移路径语义可比较，回滚恢复旧 policy 和旧 audit；没有迁移请求时不发生隐式升级。Gate 只覆盖 native/CPU policy lifecycle，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S28 已实现并通过：支持显式相邻 v1→v2 policy migration，v2 保持相同 max_batches、mode 与安全 protection rules；migration 记录 source/target/status/digest，随 checkpoint 恢复，并可显式 rollback 到旧 policy 而不改写旧 retention result、topology 或 budget。无请求不隐式迁移，非法安全语义、篡改和不一致 checkpoint fail-closed。真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_policy_migration.py` 报告 `gate.passed=true`，新增定向用例 `3 passed`，S18–S28 相关回归 `37 passed`。S28 不开放后台维护、不扩大预算、不处理 CI/CUDA/前端。

### R5C-S29：SeedRuntime 磁盘 checkpoint 下的 policy 迁移继续与回滚 Gate（已完成）

S29 负责验证 S28 不只在内存对象中成立：通过 `SeedRuntime.save()` / `SeedRuntime.load()` 的真实磁盘 checkpoint 继续运行，确认 policy、migration、retention result、status projection 和受保护 lineage 的绑定不丢失；加载后必须能显式继续 maintenance 或回滚 migration。保存失败、文件载荷篡改和 checkpoint 版本错配必须 fail-closed，不能留下半写状态或改变当前结构。

S29 必须证明：迁移后的 runtime checkpoint 可真实落盘并加载；加载前后 policy/migration/result/status digest 与保护集合一致；加载后 rollback 恢复旧 policy 和旧 audit，继续维护不会复活已删除 lineage 或改变拓扑预算；磁盘 checkpoint 篡改/缺失字段/不兼容版本拒绝且原文件与运行中状态不变；没有显式迁移/继续请求时不产生隐式动作。Gate 只覆盖 native/CPU 磁盘恢复，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S29 已实现并通过：真实 `SeedRuntime.save()` / `SeedRuntime.load()` 将 retention audit、v1/v2 policy、migration、status、result 与已删除 terminal lineage 写入磁盘并恢复；恢复后显式 rollback 可再次保存/加载，topology、structural budget 和旧 result 不变。tampered migration 与缺失配置字段 fail-closed，原 checkpoint 字节与运行中状态保持不变。定向用例 `1 passed`，真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_disk_checkpoint.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S30：重启恢复后的显式 structural maintenance continuation Gate（已完成）

S30 负责验证 checkpoint 恢复不是终点：加载后的 runtime 必须能够在同一 policy/migration 语义下接收一轮新的真实 Workbench evidence，显式执行一次 structural maintenance continuation，并区分新动作与历史 audit。恢复过程不得重放旧 maintenance、重复消费旧 evidence、复活已压缩 lineage 或绕过 protection rules。

S30 必须证明：恢复前后的 evidence cursor、policy、migration、retention result 和 lineage provenance 一致；恢复后新 evidence 只进入新的 task slice/stream，显式 maintenance 只消费新窗口并生成新的可寻址 audit；旧 window/candidate 不重复调度，已删除 lineage 不复活，protected pressure 仍可观测；新一轮 checkpoint 再恢复后 continuation cursor、audit 与 topology/budget 一致。非法 stale cursor、重复旧 evidence、混用旧 policy 和新 policy 的请求必须 fail-closed 且不改变当前状态。Gate 只覆盖 native/CPU restart continuation，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S30 已实现并通过：加载后的 SeedRuntime 接收 6 条新的真实 Workbench evidence，新的 task slices 推进 structural runtime tick 与 scheduler revision，并只创建新的 candidate batch；显式 maintenance 产生新 retention audit，已删除 terminal lineage 不复活，第二次 checkpoint restore 保留 continuation state，默认 maintenance 不重放旧 audit。真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_restart_continuation.py` 报告 `gate.passed=true`，新增定向用例 `1 passed`。本 slice 不处理 CI/CUDA/前端。

### R5C-S31：重启后候选准入、回滚与 checkpoint continuation Gate（已完成）

S31 负责把 S30 的“新 evidence 能继续流动”推进到一次受限候选生命周期：在重启恢复后的 runtime 上，对新 batch 执行 candidate-only replay validation、五类 Gate、单次 atomic admission，再 checkpoint 恢复并验证 rollback。该阶段只允许已有结构成长合同消费新 evidence，不把 runtime restart 当成重新训练，也不让迁移 policy 绕过 candidate validation 或资源预算。

S31 必须证明：新 batch 的每个 candidate 都绑定新的 evidence window、parent checkpoint、holdout/retention/lesion/resource measured facts；准入前后 checkpoint、reservation、topology 和 structural budget 可审计；恢复后只能继续未完成 candidate，重复 replay、旧 parent artifact、跨 batch candidate 或 stale policy fail-closed；成功 admission 后 rollback 可恢复父结构与预算，并在第二次 checkpoint restore 后保持 rollback lineage 和 rejected/accepted 状态一致。Gate 只覆盖 native/CPU restart candidate lifecycle，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S31 已实现并通过：重启后的新 batch 先经过 candidate-only holdout replay、validation gate 与 atomic admission，第一 candidate 后 checkpoint 恢复再完成第二 candidate；随后 rollback 恢复父结构/预算并再次 checkpoint，policy migration 与 rollback lineage 保持，跨批次 candidate continuation 现在 fail-closed 且无状态变化。真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_restart_admission.py` 报告 `gate.passed=true`，新增定向用例 `1 passed`。本 slice 不处理 CI/CUDA/前端。

### R5C-S32：重启后 replay-bound validation artifact 与 measured continuation Gate（已完成）

S32 负责把 S31 使用的 candidate-only replay 事实进一步收紧为可寻址的 Workbench validation artifact：artifact 必须绑定新 evidence、candidate、region、parent/trial checkpoint 和独立 measured holdout/retention/lesion/resource 事实，且能够跨一次重启继续消费。该阶段禁止用恢复后的临时输入重新“重算一套看似相同”的指标冒充原始测量，也禁止旧 artifact 借重启复活。

S32 必须证明：新 artifact 的输入与 measurement digest 在保存/加载后完全一致；正确 candidate/parent/replay 才能进入 validation、policy 和 atomic admission；篡改 metric、raw replay、artifact binding、旧 parent 或跨 batch candidate 均 fail-closed 且不改变 reservation、topology、budget 或 cursor；成功 admission 的 artifact provenance 可在第二次 checkpoint 后恢复，重复消费返回幂等结果而不产生第二次结构变更。rollback 的 artifact 联合语义由下一 slice 单独验收。Gate 只覆盖 native/CPU replay-bound measured continuation，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S32 已实现并通过：独立 replay measurement owner 生成的 Workbench validation artifact 以 JSON 内容寻址 payload 保存；容量 pressure snapshot 先完成再保存 parent checkpoint，重启后只消费保存的 artifact/replay。篡改 measurement digest 在候选粒度 fail-closed 且预算/拓扑不变，两个 measured artifact 跨 native checkpoint 完成 atomic admission，最终 checkpoint 后重复消费返回 `already_applied` 且 artifact batch complete。定向用例 `1 passed`，R5C-S18–S32 相关回归 `31 passed`，真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_restart_artifact.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S33：artifact provenance 与 rollback/恢复闭环 Gate（已完成）

S33 负责把 S32 的 artifact ledger 与既有 rollback、retention maintenance 和 checkpoint continuation 组合起来，验证“已准入但随后回滚”的候选不会被 artifact 重放复活，同时保留可审计的 artifact provenance 和可恢复的资源账本。

S33 必须证明：artifact 准入后 rollback 会明确改变 candidate/artifact batch 的生命周期状态，但不会删除或伪造历史 artifact；旧 artifact、旧 replay、重复 rollback 和跨 batch artifact 都 fail-closed 或幂等，且不重新扣预算、不恢复 topology；显式 retention 只淘汰无 live lineage 的完整 artifact 子图，活动/rollbackable lineage 继续受保护；保存、加载、维护和再次 rollback 后，artifact digest、admission/rollback lineage、policy/result 与 topology/budget 保持一致。Gate 只覆盖 native/CPU artifact rollback/recovery，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S33 已实现并通过：已回滚候选再次收到旧 artifact 时明确返回 `rolled_back`，不再误报 `already_applied`；artifact provenance 在 rollback checkpoint 后保持可审计，双候选均回滚后显式 retention 原子淘汰完整 artifact lineage，压缩后旧 batch 回放 fail-closed 且无状态变化，再次 checkpoint 不复活 rollback lineage。定向测试 `2 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_artifact_rollback.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S34：多批次 artifact lineage 隔离与 retention pressure Gate（已完成）

S34 负责把 S33 的单 batch rollback/compaction 语义提升到多 batch 并存场景：一个活动 batch 与多个终结 batch 同时存在时，retention 必须只淘汰无 live lineage 的 artifact 子图，不污染活动 batch 的 replay、预算、拓扑或 checkpoint continuation。

S34 必须证明：活动 reservation/pending/deferred/rollbackable candidate 的 artifact lineage 在小 retention limit 下持续受保护；多个终结 batch 可按完整子图逐批淘汰，artifact、artifact-batch、validation、gate、admission、rollback 和 schedule audit 不发生跨 batch 混删；终结 batch 的旧 artifact 回放 fail-closed，活动 batch 仍可 measured admission/rollback；checkpoint 前后 retention pressure、保护集合、artifact digest、预算和拓扑一致。Gate 只覆盖 native/CPU multi-batch artifact retention，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S34 已实现并通过：在两个 batch 并存时，小 retention limit 只淘汰终结 batch 的完整 artifact 子图，活动 batch、reservation、pending lineage、预算、拓扑和 checkpoint continuation 均保持；终结 batch 的旧 artifact 回放无状态变化。定向测试 `1 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_multi_batch_artifact.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S35：artifact batch 输入隔离与部分失败原子性 Gate（已完成）

S35 负责补齐 replay-bound artifact batch 的输入边界：未知 candidate/replay mapping key 不能被静默忽略；一个 candidate 的 artifact 或 replay 失败时，只能在候选粒度 fail-closed，同时允许同 batch 的其他合法 measured candidate 继续完成 validation、policy 与 atomic admission。

S35 必须证明：unknown artifact/replay key fail-closed 且完全原子；错误 candidate binding、缺字段 replay 和 malformed artifact 不污染另一 candidate 的 artifact、admission、reservation 或 topology；合法 candidate 仍可准入，失败 candidate 的资源/状态变化可审计并可 checkpoint；重复提交只返回既有终态，不产生第二次结构变更。Gate 只覆盖 native/CPU artifact batch 输入隔离，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S35 已实现并通过：artifact batch 对未知 candidate/replay mapping key 显式拒绝且原子不变；malformed artifact 只使自身 candidate fail-closed，合法 sibling 在 checkpoint 后仍可 measured admission，重复提交保持 topology/budget 不变。定向测试 `4 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_structural_lineage_artifact_batch_isolation.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S36：SeedRuntime artifact batch 投影与稳定结构绑定 Gate（已完成）

S36 负责把 native artifact batch contract 提升到 SeedRuntime 工作台边界：运行时只能做线程安全和稳定 payload 投影，不得重新计算 measured facts、吞掉异常或绕过 Taiji 的 candidate/artifact/validation/admission/rollback/retention 所有权。

S36 必须证明：SeedRuntime wrapper 与 native adapter 返回语义一致；外部保存的 artifact/replay 在 runtime checkpoint 重启后可完成 measured admission 并幂等重复；unknown key 继续 fail-closed 且原子；容量测量造成的 structural parent state 变化必须在保存后绑定，不能放宽 digest 校验掩盖漂移。tamper/stale parent/cross-batch failure 与并发提交由下一 slice 单独验收。Gate 只覆盖 native/CPU runtime artifact batch contract，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S36 已实现并通过：SeedRuntime 在“测量完成后保存 → runtime checkpoint 重启 → 消费外部 artifact”路径上父 digest 稳定匹配，wrapper 不重算指标；未知 key 原子拒绝，artifact provenance 可恢复，重复消费幂等。定向测试 `1 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_runtime_structural_artifact_batch.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S37：SeedRuntime artifact 失败隔离与并发提交边界 Gate（已完成）

S37 负责补齐 runtime artifact 的异常和并发边界：tamper、stale parent、错误 binding、缺字段 replay、跨 batch mapping key 必须显式失败；多个线程同时提交同一合法 artifact 时只能产生一次真实 admission，不能重复扣预算或写入不一致 lineage。

S37 必须证明：错误输入按候选级/调用级原子边界 fail-closed，合法 sibling 不被污染；同一 artifact 并发提交最多一次 admission，另一调用返回既有终态；runtime/native 的 artifact、candidate、batch、budget、topology projection 一致；checkpoint restore 后成功 provenance 与失败记录不漂移。Gate 只覆盖 native/CPU runtime artifact failure isolation 与 concurrency，不扩大预算、不开放后台自动维护、不处理 CI/CUDA/前端。

S37 已实现并通过：runtime 侧 stale parent、tamper、跨 batch 输入全部 fail-closed；并发提交同一 artifact 只产生一次真实 admission，另一次幂等返回，预算只扣一次，重启后 provenance 保留。定向测试 `2 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_runtime_structural_artifact_failure_concurrency.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S38：多轮 SeedRuntime artifact 生命周期与 retention pressure Gate（已完成）

S38 将 S36–S37 的 runtime artifact contract 推进到至少三轮真实 Workbench evidence：每轮必须由新的 task slices 产生独立 batch，经过 measured artifact、重启、成功/失败/rollback 或重复提交，并在小 retention 上限下验证活动 lineage 受到保护、终结子图可被完整淘汰。

S38 必须证明：三轮 batch/artifact lineage 不跨轮串线；malformed/stale/rollback/repeat 按 contract fail-closed 或幂等；每轮 save/load 后 structural budget、topology、cursor、policy、artifact digest 与 runtime projection 一致；retention pressure 只淘汰无 live lineage 的完整终结子图。Gate 只覆盖 native/CPU SeedRuntime 多轮 artifact 生命周期与有界 retention，不声明无限扩张、自动增预算、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S38 已实现并通过：三轮新的真实 Workbench evidence 创建独立 batch；第一轮活动 reservation 经 save/load 保持，第二轮 measured artifact 顺序准入后双 rollback，第三轮 malformed artifact 只使单候选 fail-closed 并保留 sibling reservation；小上限 retention 仅淘汰第二轮终结 artifact 子图，旧 batch replay 无状态变化，最终 checkpoint 保留 policy/cursor/budget/topology。定向测试 `1 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_runtime_structural_artifact_multi_round.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S39：retention 后活动 lineage 继续执行 Gate（已完成）

S39 验证 retention 后的活动 batch 仍然是可继续的 live lineage：SeedRuntime 从 retention 后的磁盘 checkpoint 恢复，使用当前状态重新生成 measured artifact，完成两个 candidate 的 admission，再执行 rollback 与第二次 checkpoint；不能复用旧 parent digest，也不能复活已淘汰 batch/artifact。

S39 必须证明：受保护 reservation 可在 retention 后继续消费新的 measured artifact；admission、预算扣减、topology、artifact provenance 和 rollback 可恢复且幂等；已删除终结 batch 的旧 artifact 继续 fail-closed；post-retention checkpoint 的 cursor、policy、audit、budget 和 topology projection 一致。Gate 只覆盖 native/CPU SeedRuntime retention 后 continuation，不声明无限扩张、自动增预算、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S39 已实现并通过：retention 后从磁盘恢复活动 batch，使用当前 checkpoint 重新生成 measured artifact，完成两个 candidate 的 admission、rollback 和幂等重复；已删除终结 batch 的旧 artifact 仍 fail-closed，二次 checkpoint 保留活动 batch、policy、cursor、budget 与 topology。定向测试 `1 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_runtime_structural_artifact_post_retention.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S40：重复 retention pressure 循环与有界增长 Gate（已完成）

S40 将 retention 验证扩展到至少五轮新的 Workbench task slices：保留第一轮活动 reservation，后续四轮依次完成 measured artifact admission、rollback 和 `max_batches=1` terminal-only retention，检查多次压力下活动 lineage、预算、topology、cursor、policy 和 checkpoint projection 不漂移。

S40 必须证明：每次 retention 只淘汰当轮终结 batch，活动 batch 始终受保护；每轮 admission/rollback 精确且重启不重放；最终已删除 batch/artifact 无法回放且无状态变化；lineage record 数量与资源计数保持在 policy/lineage 上限内。Gate 只覆盖 native/CPU SeedRuntime 重复 retention pressure，不声明无限扩张、自动增预算、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S40 已实现并通过：在第一轮活动 reservation 持续存在时，后续四轮新的 Workbench task slices 均完成 measured artifact admission、双 rollback 与 `max_batches=1` terminal-only retention；每轮 save/load 通过，终结 batch/artifact 只被当轮淘汰，最终只保留活动 batch，记录与容量快照受 lineage limit 约束，删除 batch 的 artifact replay fail-closed。定向测试 `1 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_runtime_structural_artifact_repeated_retention.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S41：外部 measured artifact store 与跨进程交接 Gate（已完成）

S41 将 measured artifact 从调用栈里的 Python 对象提升为不可变 content-addressed JSON：artifact 以自身 digest 命名，能够写入磁盘、从另一运行实例读取并再次进入现有 SeedRuntime batch contract；重复/并发写入幂等，字节碰撞、篡改和非法 digest fail-closed。

S41 必须证明：外部 roundtrip 不改变 artifact/measurement/evidence facts；store 不绕过 candidate、batch、parent、replay、admission、rollback 或 retention 所有权；任何 store 或交接失败不改变 runtime topology、budget、candidate 或 batch。Gate 只覆盖 native/CPU artifact persistence，不声明无限存储、自动垃圾回收、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S41 已实现并通过：measured artifact 由自身 digest 命名为不可变 canonical JSON，重复/初次并发写入幂等，Windows replace 竞态不产生半文件，篡改读取和字节碰撞 fail-closed；外部 store 读取的 payload 经 SeedRuntime checkpoint 后仍按既有 batch contract 完成 admission 与重复幂等。定向测试 `1 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_structural_artifact_store.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S42：SeedRuntime artifact store bridge 与原子交接 Gate（已完成）

S42 新增显式 SeedRuntime store bridge：调用方只提供 candidate→artifact digest 引用和 replay，runtime 在任何状态改变前完成 batch-bound key 与 store payload 校验，再一次性复用 native artifact/admission contract。unknown key、非法/缺失/篡改 artifact 或 replay 错配必须 fail-closed 且不留下部分状态。

S42 必须证明：外部 digest 引用在 runtime checkpoint 重启后可 admission 并幂等重复；store 解析失败不会改变 topology、budget、candidate、batch 或 provenance；replay 错配只遵守 native 的候选级 fail-closed，不污染 sibling、topology 或全局预算；bridge 不执行删除、不接管 retention，不改变既有 parent/replay/admission/rollback 所有权。Gate 只覆盖 native/CPU SeedRuntime bridge，不声明无限存储、自动垃圾回收、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S42 已实现并通过：SeedRuntime 新增显式 artifact store bridge，未知 candidate key 在解析前拒绝，缺失 digest 在 runtime 变更前失败；合法外部 digest 经 checkpoint 重启后完成 admission，重复提交返回 `already_applied` 且预算只扣一次。定向测试 `1 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_runtime_artifact_store_bridge.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S43：多 candidate artifact 引用预解析与原子性 Gate（已完成）

S43 验证 bridge 在多 candidate 引用场景下先完成全部外部 artifact 解析，再进入 native batch contract；任一 artifact 缺失、篡改或非法时，第一个合法 artifact 也不能提前触发 admission。失败后在同一 checkpoint 上提交合法 sibling，必须仍能正常 admission 与幂等重复。

S43 必须证明：多 candidate store resolution 是 all-or-nothing，失败不改变 topology、budget、candidate/batch 状态或 provenance；合法 sibling 后续只发生一次预算扣减，bridge 不改变 native 候选级 replay、parent、retention 或 rollback 所有权。Gate 只覆盖 native/CPU SeedRuntime 多 artifact preflight，不声明无限扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S43 已实现并通过：多 candidate artifact digest 引用先全量预解析，第二项缺失时第一项不提前 admission，batch/candidate 状态与 checkpoint digest 原子不变；同一 checkpoint 随后合法提交 admission，重复提交幂等。定向测试 `1 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_runtime_artifact_store_preflight.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S44：多 artifact 外部 batch 交接与 parent 顺序 Gate（已完成）

S44 验证两个 candidate 的 measured artifact 可以分别从外部 store 交接：第一个 artifact 在重启后 admission，第二个 artifact 必须在第一个 admission 后重新测量并绑定 child parent checkpoint；最终同一 batch complete，批量重复提交只返回既有终态。

S44 必须证明：每个 candidate 只 admission 一次且预算精确扣减；第二 artifact 不复用旧 parent；checkpoint restore 后两个 artifact provenance、parent/trial digest 和 batch state 一致；bridge 不改变 native replay、retention、rollback 或 store 所有权。Gate 只覆盖 native/CPU SeedRuntime 多 artifact external batch continuation，不声明并行无序 admission、无限扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S44 已实现并通过：两个外部 measured artifact 按 parent 顺序跨三次 checkpoint 完成同一 batch，预算精确扣减、provenance 保留，完整 mapping 重复均为 `already_applied`。定向测试 `1 passed`，Ruff 通过，真实 Workbench CPU canary `scripts/training/eval_taiji_runtime_artifact_store_batch.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S45：runtime retention 与外部 artifact store 生命周期分离 Gate（已完成）

S45 验证 runtime retention 只压缩 batch/candidate/artifact 的 lineage 引用，不直接删除外部 immutable store 文件；store 中保留的旧 artifact 仍不能绕过已删除 batch 重新进入 runtime。自动垃圾回收不在本 slice 内实现。

S45 必须证明：终结 batch 被 retention 移除后，外部 artifact 文件字节、digest 和 measurement facts 保持；从 store 读取旧 artifact 回放已删除 batch 必须 fail-closed 且 checkpoint digest 不变；活动 batch、store artifact、retention audit 的所有权在 checkpoint restore 后一致。Gate 只覆盖 native/CPU 生命周期隔离，不声明自动垃圾回收、无限存储、无限扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S45 已实现并通过：runtime retention 仅淘汰终结 lineage，不删除外部 immutable artifact；旧 store artifact 不能复活已删除 batch，活动 batch/store/audit 所有权跨 checkpoint 保持。`eval_taiji_runtime_retention_store_separation.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S46：外部 structural artifact store 只读 inventory / audit Gate（已完成）

S46 在 S45 的生命周期分离之上增加只读 inventory / audit：按稳定 digest 顺序列出外部 artifact，并重新验证文件名 digest、artifact digest、canonical bytes 与 measurement digest。审计只观察外部文件，不改变 runtime、lineage、budget、retention audit 或 checkpoint，也不执行自动垃圾回收。

S46 必须证明：健康 inventory 与逐项 load 事实一致且跨 checkpoint 稳定；runtime retention 后 orphan 仍可审计但不能重新进入已删除 batch；篡改 payload、measurement facts、非法文件名和额外异常文件 fail-closed，且不发生自动删除或修复。Gate 只覆盖 native/CPU store audit，不声明自动删除、无限存储、无限扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S46 已实现并通过：`StructuralValidationArtifactStore.inventory()` / `audit()` 按 digest 稳定返回 artifact 与 measurement 事实摘要，重新验证文件名 digest、artifact digest、canonical bytes 和 measurement digest；runtime retention 后的 orphan 保持可审计但不能回放已删除 batch，篡改和非法文件 fail-closed 且不自动修复。定向测试 `1 passed`，Ruff 通过，`eval_taiji_runtime_retention_store_audit.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S47：SeedRuntime 外部 artifact store audit 只读投影 Gate（已完成）

S47 将 S46 的外部 store inventory 接入一个显式的 SeedRuntime 只读观察入口。入口只接受已存在的 `StructuralValidationArtifactStore`，返回稳定的 store audit 事实与当前 runtime lineage 可见性对照，不把外部 artifact 注册回 runtime，不触发 retention、replay、budget、candidate/batch 或 checkpoint 变化。

S47 必须证明：同一 checkpoint 上重复查询返回相同 audit digest；store 中的 runtime orphan 能被标识为“外部存在、runtime 不可消费”，活动/已知 lineage 的对应关系不被错误推断；查询前后 runtime checkpoint、拓扑、budget、retention audit 和 store 文件字节不变，缺失/篡改 store 仍 fail-closed。Gate 只覆盖 native/CPU 只读观察投影，不声明自动注册、自动删除、无限扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S47 已实现并通过：SeedRuntime 提供 `project_structural_artifact_store_audit()`，以稳定 `audit_digest` 投影 store inventory 与 runtime lineage visibility；checkpoint restore 后结果一致，retention 后 orphan 只被标识为外部存在，篡改查询 fail-closed，未注册外部 artifact。定向测试 `1 passed`，Ruff 通过，`eval_taiji_runtime_artifact_store_audit_projection.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S48：artifact store 与 runtime lineage 只读对账 Gate（已完成）

S48 将 S47 projection 升级为显式 v2，并补充反向对账：runtime 已记录的 artifact digest、artifact batch 已引用的 digest、外部 store 已存在的 digest，以及两类 missing-store 集合都按稳定顺序输出。对账不回写、不修复、不删除，也不把缺失外部文件误判为 runtime orphan。

S48 必须证明：健康 store 上 missing 集合为空；runtime 已记录但未进入 store 的 artifact 只产生对应 missing digest；external orphan 与 runtime-missing 不互相误报；重复查询和 checkpoint restore 稳定，缺失/篡改文件仍 fail-closed 且 runtime/store 全部只读。Gate 只覆盖 native/CPU projection reconciliation，不声明自动修复、自动注册、自动删除、无限扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S48 已实现并通过：projection 升级为显式 v2，补充 runtime artifact 与 batch 引用集合及 missing-store digest；runtime 已记录但未进 store 的 artifact 与 retention 后 external orphan 分别被识别，checkpoint restore、篡改 fail-closed 和全链路只读通过。定向回归 `2 passed`，Ruff 通过，`eval_taiji_runtime_artifact_store_runtime_reconciliation.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S49：measured artifact 的 measurement-fact sidecar Gate（已完成）

S49 为新 measured artifact 增加独立的 canonical measurement sidecar：artifact 仍由自身 digest 命名，measurement facts 由 measurement digest 命名；store 可以使用 `StructuralValidationMeasurements.from_payload()` 独立重算 measurement digest。早期只写 artifact 的 legacy 文件保留，但必须明确标记为 `legacy_unverified`，不得伪造已不存在的 facts。

S49 必须证明：新 bundle 的 artifact/measurement roundtrip 与 `verified` inventory 通过；measurement facts、sidecar 文件名或 artifact/measurement 绑定篡改均 fail-closed 且不删除原文件；legacy artifact 仍可走既有 runtime contract 但不会被误报 verified；sidecar 审计、重复写入和 checkpoint restore 不改变 runtime/store 状态。Gate 只覆盖 native/CPU measurement-fact persistence，不声明自动修复、自动删除、无限扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S49 已实现并通过：`put_measured_artifact()` 以 measurement digest 写入 canonical sidecar，`load_measurements()` 独立重算 measurement digest，inventory 区分 `verified` 与 `legacy_unverified`；绑定冲突、sidecar 篡改 fail-closed 且不删除，runtime 仍经既有 artifact bridge 消费。定向回归 `4 passed`，Ruff 通过，`eval_taiji_structural_artifact_measurement_sidecar.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S50：measurement bundle 部分写入后的显式恢复 Gate（已完成）

S50 验证 artifact 与 measurement sidecar 双文件 bundle 的不完整写入边界：sidecar-only 或 artifact-only 不得被当作 verified，也不自动删除现场；同一完整 bundle 重试必须可恢复、幂等，冲突仍 fail-closed。该 slice 只验证恢复协议，不引入自动清理或 runtime 注册。

S50 必须证明：sidecar-only audit fail-closed 后补交 artifact 可恢复为 verified；artifact-only 明确为 legacy_unverified 后补交匹配 sidecar 可升级为 verified；重复写入不改变 immutable bytes，冲突不覆盖；恢复和审计不触发 runtime、checkpoint、budget、topology 或 lineage 变化。Gate 只覆盖 native/CPU bundle recovery，不声明自动修复、自动删除、无限扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S50 已实现并通过：sidecar-only 与 artifact-only partial state 均 fail-closed 且不删除，补交完整 bundle 后幂等恢复为 `verified`，冲突重试不覆盖原字节，runtime checkpoint 保持不变。定向回归 `5 passed`，Ruff 通过，`eval_taiji_structural_artifact_measurement_bundle_recovery.py` 报告 `gate.passed=true`。本 slice 不处理 CI/CUDA/前端。

### R5C-S51：verified measurement artifact bridge Gate（已完成）

S51 为 SeedRuntime artifact-store bridge 增加显式 `require_verified_measurements` 严格模式：调用方选择严格模式时，artifact 与 measurement sidecar 必须在 runtime 变更前全部通过 store 独立校验；历史兼容调用仍可消费 legacy artifact-only 文件。该默认边界已由 S52 的显式 policy 取代，多 candidate 解析仍遵循全量预解析和原子 batch contract。

S51 必须证明：verified bundle 严格模式可消费，legacy 兼容调用可消费但严格模式 fail-closed；多 candidate 任一 sidecar 缺失或篡改时不发生部分 admission、预算扣减或 checkpoint 变化；严格模式重复消费保持既有幂等，runtime contract 不新增自动注册或自动升级。Gate 只覆盖 native/CPU verified bridge，不声明默认强制迁移、自动删除、无限扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

S51 已实现并通过：`load_verified_artifact()` 独立验证 artifact 与 measurement sidecar，SeedRuntime bridge 增加显式 `require_verified_measurements`；verified bundle strict consumption、legacy 兼容调用/strict 拒绝、multi-candidate strict preflight 原子性均通过。其历史报告不代表当前默认策略；当前默认由 S52 policy canary 定义。本 slice 不处理 CI/CUDA/前端。

### R5C-S52：artifact consumption policy 与 verified 默认边界 Gate（已完成）

S52 将 S51 的布尔开关收敛为显式、内容寻址、可 checkpoint 的 artifact consumption policy：新成长路径推荐 verified-only，历史证据只能由显式 legacy-compatible policy 回放；policy、原因和 artifact status 必须进入只读 audit，不能绕过原有 parent、replay、candidate、batch 或 resource contract。

S52 已证明：verified-only 接受 verified、拒绝 legacy/missing/tampered；legacy-compatible 只允许明确旧证据且保留原因；policy save/load/rollback 稳定；多 candidate policy resolution all-or-nothing。CPU canary 报告为 [taiji_w7_r5c_s52_artifact_consumption_policy_20260831.json](../../../reports/taiji_w7_r5c_s52_artifact_consumption_policy_20260831.json)，不声明自动迁移、自动删除、无限扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。S52 后唯一后继切换为 P2 的自然语言→Workbench red Gate。

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

## 9. E 系列主线路线决策

当前最优路线不是继续扩张 provider 外围，也不是从零训练一个模仿 Transformer 的通用语言模型，而是直接把项目已经拥有的 Taiji 原生神经元、突触、记忆、interaction-group、结构成长、Workbench Outcome 和 capability lifecycle 连接成可训练、可验证、可持续生长的系统。

路线由三条相互独立但可协作的增长链组成：

1. **脑进化**：Taiji 自有突触权重、局部神经元状态、记忆、路由、interaction-group 和结构拓扑依据真实经验更新。
2. **客户端身体进化**：Seed 客户端通过可装载、可卸载、可隔离、可回滚的插件扩展界面、IDE/Workbench、Skill、MCP、可视化和工具能力；客户端就是 Taiji 与用户及环境交互的身体壳层。
3. **语言进化**：provider/codec 只改善输入理解候选与输出表达，不拥有 Goal、记忆、规划、工具选择、结构成长或执行权。

三条链共享同一份可追溯经验和语料来源，但不得共享所有权。Skill/MCP 自身的说明、schema、示例、约束和领域资料可以经过治理后成为 Taiji 知识语料；它们的真实调用与结果成为经验语料。MCP 的连接器、执行接口、权限、资源和 UI 不写入模型，而是形成客户端 capability/plugin 候选，由 Seed 客户端进化继承。客户端插件热插拔改进的是 Seed 客户端，不进入 Taiji 神经网络内部。

## 10. 从 DeepSeek Harness 采纳与不采纳的部分

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

## 11. E 系列目标架构

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

## 12. 统一进化语料与经验合同

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

## 13. Skill 语料、经验与内化 Gate

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

## 14. MCP 双产物内化与客户端继承等级

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

当前 `api/routes_agent_mcp.py -> neuroplex.agent_ext.mcp_manager` 的 `/api/mcp/*` 仍是 Legacy 兼容能力，不进入 E1 verified corpus/experience ledger；原 `/api/plugins/marketplace`、marketplace refresh 和 workspace upload 已由 E5-2 统一退役为 Seed-owned 410 tombstone。真实第三方 MCP/plugin 仍需等 E6-1 API/registry shadow 接线及后续 E6 Gate。

## 15. Seed 客户端插件合同与热插拔生命周期

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

## 16. 脑—客户端协同进化的归因规则

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

## 17. E1–E9 阶段与未闭合 Gate

E0–E6 的逐条完成记录与当时状态引自被归档的原总路线快照 [63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md](../../archive/history/63_TAIJI_NATIVE_EVOLUTION_AND_EMBODIMENT_20260831.md)，其状态行为：

> 状态：`E0 complete / E1 complete / E2-A complete / E2-B complete / E3-0 complete / E3-1 complete / E3-2 complete / E3-3 complete / E3-4 complete / E4 complete / E5 complete / E6-0 complete / E6-1 complete / E6-2 complete / E6-3 complete / E6-4 complete / E6-5a complete / E6-5b deferred-by-decision`。本文件自 2026-08-31 起取代原 P7-1a provider artifact 升级作为当前主线，并于 2026-09-01 按“Skill/MCP 可成为知识语料、MCP 执行侧由客户端继承”完成修订。Qwen/provider 质量问题保留为语言器官支线，不再阻塞 Taiji 本体训练。E6-5b 已按用户决定搁置：它索取的是外部治理输入（具体第三方 MCP、连接方式、网络范围、凭据引用、审批人、撤销责任）而非工程推导，且 `connection_attempted` 恒为 False 使未完成态即最安全态；主线已回归 Taiji 本体，唯一执行事实源 `03_CURRENT_EXECUTION.md` 不再指向 E6-5b。

以下三个阶段尚未退出，是 E 系列当前仍生效的阶段合同。

### E7：脑—客户端协同选择器

状态：五条 Gate 与六类输出互斥全部 `complete`（owner `taiji/evolution_credit.py`，回归 `tests/taiji_native/test_evolution_credit.py` 12 passed，`tests/taiji_native` 全量 428 passed / 1 skipped）。结构增长准入不在本模块重算，而是消费 `CapacityGrowthTrigger` 的许可位；「缺少 affordance」以「能力未注册」为可验证代理，不引入第二套 affordance 开关。第五条消融归属 Gate 由 `attribute_brain_client_ablation()` 闭合：同一经验集在「能力已注册」与「能力未注册」两种 registry 下各跑一臂，每条 episode 只能落入 `brain_only`、`client_plugin_only`、`unattributed` 之一，`clarify_or_stop`（含资源耗尽与无能力标识）一律归 `unattributed` 而不计入任何一侧收益，归属结果内容寻址、跨臂搬账即 `digest mismatch`。

目标：Taiji 能根据经验判断“应该学习已有能力，还是向 Seed 客户端申请新能力”，但不能直接安装插件。

输出只能是下列候选之一：`weight_update`、`memory_consolidation`、`route_update`、`structure_candidate`、`client_capability_candidate`、`clarify_or_stop`。

Gate：

- 已有能力可解决时不申请插件；
- 缺少 affordance 时不靠增加突触伪造执行器；
- 语言失败不触发结构增长；
- 资源不足时降级/停止而不是无限增长；
- client-plugin-only 与 brain-only 对照能够解释收益归属。

### E8：长期持续进化与数据飞轮

状态：bounded replay 采样合同 `complete`，其余 Gate 项 `pending`（owner `taiji/evolution_training.py`，回归 `tests/taiji_native/test_evolution_training.py` 7 passed，`tests/taiji_native` 全量 432 passed / 1 skipped）。`internalization.py` 的 `BoundedReplayBuffer` 是硬上限（满则 `BufferError`），不承担优先级淘汰；`select_bounded_replay()` 补的是容量收紧时的淘汰次序：`EVOLUTION_REPLAY_TIERS = ("correction", "failure", "success")` 按层填充，`correction` 与 `failure` 为 `EVOLUTION_REPLAY_RETAINED_TIERS`，容量不足以容纳纠正与失败证据时拒绝采样并报 `capacity cannot drop retained evidence`，因此「旧能力保持」从事后指标变成采样期 fail-closed。层级判据取 `EvolutionExperience` 既有的 `user_correction_digest` 与 `success`，不新增经验字段；`BoundedReplaySelection` 对 `experience_id` 排序后内容寻址，与输入顺序无关、跨重启可复现，删改选中集合即 `selection digest mismatch`。未新建模块，`config.py` 的 11 个 `replay_*` 参数面保持单一。多周期净能力收益、checkpoint 大小/延迟预算、污染隔离仍缺判据；分支合并策略需外部治理输入。

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

## 18. E 系列文件所有权

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

## 19. 每个 slice 的固定交付格式

1. 先写 red contract 与失败证据；
2. 做最小 owner 内实现，不顺手扩展相邻能力；
3. 运行与改动范围匹配的阻塞 Gate；
4. 生成内容寻址报告，记录未验证边界；
5. 更新 manifest、实现事实和唯一下一步；
6. 单一主题提交；若 CI 红，下一提交只修 CI，不能继续堆功能。
