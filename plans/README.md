# Seed / Taiji 计划与架构入口

> 最新修订：2026-09-10。当前唯一执行计划是 [Seed / Taiji 唯一执行计划](active/roadmap/03_CURRENT_EXECUTION.md) 顶部的「2026-09-10 修订执行序列」。旧正文及归档中的“下一步”均失效。

M4.R0～R12 已完成并归档，但 2026-09-09 的代码/报告复审发现：该系列主要测量 F1 byte-prediction 的固定容量 continuation，R2 的固定等权读出槽、R7/R10 不一致的 owner 图、未进入普通主路径的 adaptive network，以及 exact-zero Gate 都不足以代表 CR-4/A8 的继承式结构成长。历史报告保留，过度外推已撤销。

最新实际状态：v2 R2 的 fast/slow 与真实 replay 有小规模 S/G 证据；R4 formal 在 S/G 上优于小容量对照，但未稳定超过 fixed-large；R5 条件路由候选被否决。R6 九 cell 接线运行通过，但全部无训练，且旧评分把“不接纳反馈”计为任务成功率 0，不能当作学习提升。评分语义修正已完成：任务成功、反馈准入和参数更新现在分离，wiring-canary 不能进入 learning-formal Gate；旧报告仍保留为历史证据。整体保持 `can_promote=false`，暂停直接沿旧 admission 启动学习 formal。完整依据见 [M4 v1/v2 实际结果复审](reference/M4_V1_V2_RESULT_REVIEW_2026_09_10.md)。

**当前唯一下一步：为 K runtime 建立真实 continuation-learning contract，并完成 B3-K 单步 pilot。** B3 的 S/G slice 已在同一 inherited model 17 parent 上通过：fast+replay 使用 424 个真实 replay event，S/G 均改善，fresh restore/read-only/rollback 通过；但当前 R6 K 仍是 wiring-only（`training_update_steps=0`），所以完整 B3 尚未完成，也没有晋级。训练前 checkpoint 保存恢复预检已通过。随后依次进行独立 holdout 正式比较及 v1 兼容对照，再按证据恢复结构成长。Skill/MCP、Workbench、插件、provider、CUDA 和视觉均已排入计划，不再并行抢占模型学习主线。

## 权威文档

| 文档 | 唯一职责 |
|---|---|
| [TAIJI_CORE_REQUIREMENTS.md](active/TAIJI_CORE_REQUIREMENTS.md) | 长期使命与 CR-1～CR-10，不随实验归档 |
| [TAIJI_NATIVE_ARCHITECTURE_V1.md](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | 目标架构、对象、学习体系与设计约束 |
| [TAIJI_CONTINUAL_DEVELOPMENT_V2.md](active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | M4 v2 的快/慢学习、主路径结构成长、课程与准入 Gate |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 原生身份、成熟技术与 Legacy 边界 |
| [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) | Seed、Workbench、provider 与副作用所有权 |
| [03_CURRENT_EXECUTION.md](active/roadmap/03_CURRENT_EXECUTION.md) | **唯一执行顺序、详细任务、验收、日程和下一步** |
| [TAIJI_RESEARCH_REVIEW_2026_09_06.md](reference/TAIJI_RESEARCH_REVIEW_2026_09_06.md) | 本次审视、最小复现、证据失效范围与技术参考，不另设执行路线 |
| [M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md](reference/M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md) | R0～R12 原始数值、有效技术资产与复审后的解释边界 |
| [M4_V1_V2_RESULT_REVIEW_2026_09_10.md](reference/M4_V1_V2_RESULT_REVIEW_2026_09_10.md) | v1/v2 实际对比、R6 评分混淆与本次修订依据，不另设执行路线 |
| [IMPLEMENTATION_STATUS_2026_08.md](reference/IMPLEMENTATION_STATUS_2026_08.md) | 当前实现事实与能力声明边界 |

## 目录与维护

- `active/` 保留根需求、当前架构和唯一计划。兼容入口 `roadmap/01、02、04` 不决定顺序。
- `reference/` 保存当前事实、证据解释和 owner 边界。
- `archive/` 保存已完成/被替代的计划、实验讨论和调试过程；仍有效的设计结论先提炼到 active。
- `manifests/` 保存版本化数据、评估与实验合同；旧报告不因新判定而被原地改绿。
- 每轮更新唯一计划并提交精确范围；不以新增编号文档、checkpoint 文件大小或测试数量代替能力收益。
- 历史入口：[本次研究重审归档](archive/history/research_review_20260906/README.md)、[2026-09-01 收敛归档](archive/history/roadmap_convergence_20260901/README.md)、[总归档索引](archive/README.md)。
