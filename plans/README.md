# Seed / Taiji 计划与架构入口

> 最新修订：2026-09-06。当前唯一执行计划是 [模型优先统一开发计划](active/roadmap/03_CURRENT_EXECUTION.md)。所有归档中的“下一步”均失效。

本次已完成当前研究内容的系统重审与详细排程。Taiji 已有真实小规模训练和窄能力证据；active 评分错路由、旧 B5 holdout 重复和能力/保持/增量混用需先纠正。代码尚未修复，后续由用户按计划推进。

当前唯一下一步是统一计划的 **M2.R0：修复 active/protected 评分与版本化能力判定**。后续按“真实续训 → 表征与时间学习 → 语义表达 → 联合保持 → 真实任务 → 连续成长”推进。Skill/MCP、客户端热插拔、provider、视觉和 CUDA 都保留具体排期与解冻条件。

## 权威文档

| 文档 | 唯一职责 |
|---|---|
| [TAIJI_CORE_REQUIREMENTS.md](active/TAIJI_CORE_REQUIREMENTS.md) | 长期使命与 CR-1～CR-10，不随实验归档 |
| [TAIJI_NATIVE_ARCHITECTURE_V1.md](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | 目标架构、对象、学习体系与设计约束 |
| [ARCHITECTURE_DIRECTION_2026_08.md](active/ARCHITECTURE_DIRECTION_2026_08.md) | 原生身份、成熟技术与 Legacy 边界 |
| [SEED_ARCHITECTURE.md](active/SEED_ARCHITECTURE.md) | Seed、Workbench、provider 与副作用所有权 |
| [03_CURRENT_EXECUTION.md](active/roadmap/03_CURRENT_EXECUTION.md) | **唯一执行顺序、详细任务、验收、日程和下一步** |
| [TAIJI_RESEARCH_REVIEW_2026_09_06.md](reference/TAIJI_RESEARCH_REVIEW_2026_09_06.md) | 本次审视、最小复现、证据失效范围与技术参考，不另设执行路线 |
| [IMPLEMENTATION_STATUS_2026_08.md](reference/IMPLEMENTATION_STATUS_2026_08.md) | 当前实现事实与能力声明边界 |

## 目录与维护

- `active/` 保留根需求、当前架构和唯一计划。兼容入口 `roadmap/01、02、04` 不决定顺序。
- `reference/` 保存当前事实、证据解释和 owner 边界。
- `archive/` 保存已完成/被替代的计划、实验讨论和调试过程；仍有效的设计结论先提炼到 active。
- `manifests/` 保存版本化数据、评估与实验合同；旧报告不因新判定而被原地改绿。
- 每轮更新唯一计划并提交精确范围；不以新增编号文档、checkpoint 文件大小或测试数量代替能力收益。
- 历史入口：[本次研究重审归档](archive/history/research_review_20260906/README.md)、[2026-09-01 收敛归档](archive/history/roadmap_convergence_20260901/README.md)、[总归档索引](archive/README.md)。
