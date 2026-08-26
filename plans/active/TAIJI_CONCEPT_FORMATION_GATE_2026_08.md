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

## 边界与已知限制

这只是数据驱动的临时形成 Gate，不等同于开放域语义、符号知识或通用理解。形成、匹配、
更新和证据索引现在由 Taiji 自有 `ConceptFormationOrgan`/注册表负责；`TSKV8Adapter` 只
保留兼容 API 与接线，不再承载概念形成规则。器官已有独立 checkpoint、容量治理、塑性更新、
剪枝和 lesion，并已通过语义检索→规划的窄消费路径；这仍不等于开放域概念已被充分验证。

## 下一步唯一入口

在同一 Concept 的多分支 trace 上加入容量/干扰曲线与分支增删：验证分支数量增长时正确
分支仍可由 after-state 与 outcome/error 选择，淘汰或 lesion 单一 trace 不得破坏其他分支，
并在多分支 runtime continuation 与 checkpoint 后保持一致；继续禁止固定 action/intent 表。
