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

## 边界与已知限制

这只是数据驱动的临时形成 Gate，不等同于开放域语义、符号知识或通用理解。当前实现暂时
位于 `TSKV8Adapter`，用于证明真实 runtime lineage；不得继续在 adapter 中无界增加认知
逻辑。下一步应把形成、匹配、更新和证据索引提取到 Taiji 自有语义器官/注册表，使其拥有
独立 checkpoint、容量策略、可塑性和 lesion 接口。

## 下一步唯一入口

提取 `ConceptFormationOrgan`（名称可在实现时确定）并让 adapter 只负责调用边界：该器官
接收版本化经历证据，产生稳定的 concept identity、支持集更新和可回滚 checkpoint；随后用
跨 schema/未见任务迁移、容量/干扰曲线和三类信号 lesion 重新验收。不得把扩大参数量或
增加固定标签表作为替代。
