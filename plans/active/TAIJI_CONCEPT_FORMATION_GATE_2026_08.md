# Taiji 多信号 Concept Formation Gate

更新时间：2026-08-26

## 目标

Concept 必须是跨经历形成的可追踪不变量，而不是把单个 cue、episode 或人工标签改名为
“概念”。当前 Gate 要求概念形成同时受到三类证据约束：

1. **感知 latent**：不同经历中的局部表征具有足够相似度；
2. **世界证据**：对象集合和关系结构一致。对象 ID 可以变化，但关系形状仍需保持；
3. **Outcome 证据**：行动结果的成功性和有界 reward 表征一致。

三类信号的组合权重由 `TaijiConfig.concept_signal_weights` 配置，默认值为
`(latent=0.45, world=0.35, outcome=0.20)`；聚类阈值和历史血缘长度同样属于配置，而不在
运行时写死。

## 当前运行时事实

- 观测产生 `Event/Assembly`，结算把事件、程序集、对象和关系引用写入
  `EpisodicMemoryRecord`；
- 语义巩固只接受有事件血缘、且来自至少两个 episode 的经历；
- 相同关系形状但不同对象 ID 的跨 schema 经历可以形成同一临时 Concept；
- 第三个未见对象实例会扩展已有 Concept 的支持集并递增 `update_count`；
- 删除事件/程序集、对象/关系或 Outcome 证据中的任一类信号都会阻止该概念形成；
- `Concept` 保存 support event/assembly/object/relation IDs、latent prototype、outcome
  均值与一致性，可经 native checkpoint 恢复；训练前 checkpoint 检查仍是前置条件。
- `ConceptFormationOrgan` 已从 `TSKV8Adapter` 提取，独立拥有多信号匹配、概念 identity、支持集
  更新和 checkpoint；adapter 现在只负责把 episodic evidence 接入该器官。
- 器官现在具有可配置容量、塑性率、强度剪枝和显式 `lesion`；容量曲线在 1/2/4 个槽位下
  分别保留 1/2/4 个概念，checkpoint 与 lesion 对照均通过。
- `ConceptMatch` 已成为器官的查询边界；运行时把匹配 concept IDs 写入 `MemoryState`，并把
  匹配度 × 概念置信度 × outcome 质量映射为 `PlanningCandidate.concept_affinity`，由 planner
  的可配置 `concept_weight` 消费；lesion 后该 prior 为零并改变窄规划对照。
- 下游迁移 Gate 已通过：schema 数量 1/2/4/8 的未见对象与关系查询均保持 100% 规划迁移；
  无 concept prior 的对照不迁移，容量 1/2 显示可测的概念干扰；event/assembly、world、
  outcome 三类证据 lesion 均 fail-closed；器官 checkpoint 和 native runtime checkpoint
  均恢复相同概念与查询能力。
- 多步 sequence Gate 已通过：Concept 从 episode 的时间顺序形成 `action_sequences`，
  rollout 以可配置 `concept_sequence_weight` 消费；正确顺序击败高即时收益的反转序列，
  1/2/4/8 schema scale 均通过，前缀匹配有效、反转匹配为零；concept lesion 后选择回到
  反转对照，adapter native checkpoint 恢复选择结果，真实失败仍触发并恢复 replan 状态。
- 状态条件 suffix Gate 已通过：`ConceptSequenceTrace` 从真实 `WorldTransition` 保存每一步
  的 before/after latent、prediction error 和未来折扣后的 step credit；部分执行后能从
  after-state 重新检索剩余 suffix，完全错位动作与错误状态 fail-closed；运行时保留环境
  after-state，不被后续感知覆盖，organ/native checkpoint 恢复 trace、计划和 suffix affinity。
- 变量 horizon / 分支塑性 Gate 已通过：同一 Concept 内保留共享前缀后的两条不同 trace，
  `suffix_sequence_affinity` 能在 horizon=1/2/3 下区分正确分支和反转动作；即时收益略高的
  对照分支仍被正确 suffix prior 淘汰。对分支特有的 `confirm` 真实转移进行 outcome/error
  增量更新时只命中对应 trace，visits 与 step credit 发生 EMA 更新，另一条分支保持不变；
  trace lesion 保留 Concept identity 但使 sequence prior 为零，organ checkpoint 恢复更新后的
  visits/credit。该 Gate 报告为 `reports/taiji_concept_branch_20260826.json`。
- trace capacity / selective branch Gate 已通过：`trace_capacity=1/2/4` 分别保留 1/2/2 条
  分支，容量为 1 时按 trace strength 保留更强的 good 分支，容量为 2/4 时 alternative
  分支可加入但不会压过正确 suffix；增量巩固会在同一 Concept 中加入新 trace，按稳定
  `trace_id` 删除单一分支后其他分支不变，checkpoint 恢复剩余 trace identity。该 Gate 报告为
  `reports/taiji_concept_trace_capacity_20260826.json`。
- online branch birth Gate 已通过：`ConceptFormationOrgan` 接受不命中已有 trace 的连续真实
  `WorldTransition` 链，形成带稳定 `trace_id` 的新分支；重复链不产生副本，新分支立即可按
  after-state 检索，负 outcome/高 prediction error 会降低其 credit 而不抹掉 identity，
  `TSKV8Adapter.grow_online_concept_branch` 发布后 native checkpoint 仍能恢复该分支；
  `settle_action` 的 episode buffer 可在 terminal 时自动触发 branch birth，中途 checkpoint
  恢复 1 步 buffer 后继续完成同一分支。该 Gate 报告为
  `reports/taiji_concept_online_birth_20260826.json`。
- branch attribution Gate 已通过：多个同时激活的 Concept 不再共享写入同一条在线链；器官
   按 match confidence、已学习的 before/after-state 证据和 prediction-error fit 选择唯一
   owner。低置信度、近似平分的跨 concept 干扰和 owner trace lesion 均 fail-closed；真实
   `settle_action` episode buffer 只保留 owner，并可经 native checkpoint 继续完成 branch
   birth。配置中的权重、最低分数和最小胜出间隔均由 `TaijiConfig` 管理。该 Gate 报告为
   `reports/taiji_concept_branch_attribution_20260826.json`。
- structural growth budget/rollback Gate 已通过：`StructuralGrowthRequest` 将 owner 的新分支
  变更记录为版本化 proposal；预算不足时 fail-closed，候选必须经过 trial checkpoint roundtrip、
  trace lesion 与 replayability 验证，接受后扣减 `DevelopmentState.structural_budget`，native
  checkpoint 可恢复 request，rollback 可恢复父结构并返还预算。该 Gate 报告为
  `reports/taiji_structural_growth_20260826.json`。
- synapse topology proposal Gate 已通过：`StructuralTopologyProposal` 使用稳定的 substrate 与
  单元坐标描述一次 rewire，不依赖 action/intent；`TaijiFabric` 只允许对现有合法固定 fan-in
  bank 提案和应用，局部学习后的 donor response 在 holdout probe 上提升，fabric checkpoint
  可恢复拓扑，functional lesion 会移除新增接触的贡献，父 payload 可恢复原拓扑。该 Gate 报告为
  `reports/taiji_topology_proposal_20260826.json`。
- runtime topology ledger Gate 已通过：`TSKV8Adapter` 接管 topology proposal 的 parent
  checkpoint、资源成本与 `DevelopmentState.structural_budget`；proposal 经 fabric checkpoint
  roundtrip 后才能接受，native checkpoint 能继续恢复 ledger，rollback 只允许按最新接受顺序
  执行，预算耗尽时 fail-closed。该 Gate 报告为
  `reports/taiji_topology_runtime_ledger_20260826.json`。
- neuron growth Gate 已通过：`AdaptiveNeuronRegion` 使用稳定 `unit_id`、显式活动/阈值/膜电位/
  trace 状态、稀疏输入与递归突触；新增单元只追加状态和新突触行，旧单元的身份、拓扑和权重
  保持不变。新单元可通过局部 error × eligibility 学习 holdout，functional lesion 会使其活动
  失效，器官 checkpoint 与 `TSKV8Adapter` native checkpoint 均能恢复；runtime ledger 负责预算、
  接受、零预算拒绝和最新顺序 rollback。该 Gate 报告为
  `reports/taiji_neuron_growth_20260826.json`。
- cross-region wiring Gate 已通过：`AdaptiveNeuronNetwork` 以显式 source/target region 和
  stable `connection_id` 建立稀疏跨区突触；上游活动能驱动下游，连接 lesion 后下游活动归零；
  上游 neuron growth 会迁移连接输入维度并保持旧支持/权重，网络 checkpoint、native checkpoint、
  预算接受/拒绝和逆序 rollback 均已覆盖。该 Gate 报告为
  `reports/taiji_cross_region_20260826.json`。

## 边界与已知限制

这只是数据驱动的临时形成 Gate，不等同于开放域语义、符号知识或通用理解。形成、匹配、
更新和证据索引现在由 Taiji 自有 `ConceptFormationOrgan`/注册表负责；`TSKV8Adapter` 只
保留兼容 API 与接线，不再承载概念形成规则。器官已有独立 checkpoint、容量治理、塑性更新、
剪枝和 lesion，并已通过语义检索→规划的窄消费路径；这仍不等于开放域概念已被充分验证。

## 下一步唯一入口

学习型跨区域协作 Gate 已完成：`CrossRegionCooperationLearner` 为显式连接维护可 checkpoint
的 prediction-error、holdout-transfer、resource-state EMA 与探索状态，`AdaptiveNeuronNetwork`
按 learner 和资源预算选择路径；学习路径在 holdout 证据上优于固定全连接/随机基线，并通过
connection/region lesion 与 checkpoint continuation；在线 credit loop 已接入真实 network tick，
由 expected target activity 自动计算 prediction error 和 holdout transfer。下一步进入 substrate
驱动的自动结构成长 Gate 已完成：持续误差、资源可用性和 holdout 增益只能生成 neuron proposal，
必须通过 DevelopmentState budget、checkpoint trial、functional lesion 与 reverse rollback 才能
出生。区域级 proposal Gate 也已完成：持续区域瓶颈可以生成带非语义 child region identity、
显式 dynamics 和 topology role 的 region proposal；`AdaptiveNeuronNetwork` 保持已有区域和
执行顺序，adapter ledger 负责预算、checkpoint trial、functional region lesion、显式跨区域
连接和逆向 rollback，且零预算 fail-closed。该 Gate 报告为
`reports/taiji_region_growth_20260826.json`。当前下一步是独立验证新生区域及其连接在未见输入上的
holdout 目标改善，再允许其进入长期发展记录；该 post-growth validation Gate 已通过：两次
未见输入的相对 holdout gain 为 `0.8735`，checkpoint continuation 后仍通过，未通过验证的
区域会阻断跨区连接。当前下一步是建立资源感知的 retention/pruning Gate，继续禁止按固定
action/intent 表决定增长。
