# Seed / Taiji 计划与架构入口

> 2026-09-10 依据 main 4629bc8c 的源码、checkpoint 与报告复审。唯一执行顺序：[当前计划](active/roadmap/03_CURRENT_EXECUTION.md)。

目前已有五类 K worker 和 fast/slow＋replay 实现。C-stage v2 在 D/R/A 小型评估上四门通过，FS 比 C 的平均 MSE 改善约 0.001868，弱类改善约 0.002495。该组合使用额外 replay；三组 v4 K worker 权重实测相同，全五类泛化与独立拆分收益仍待验证，can_promote=false。

**唯一下一步：同 replay、同预算的 FS / 直接 continuation 配对诊断。** 随后依次完成有效信号/五类数据合同、独立效果验证、S/G/K 联合状态整合与结构成长。SGK v1 暂停，待纠正保持判据方向、资源计量和课程独立性后发布 v2。

- [本轮结果复审](reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)：实际收益、预算混淆、父 worker 重复、seed 循环、恢复缺口与修订依据。
- [机器审计](../reports/taiji_m4v2_plan_result_review_20260910.json)：源码摘要、权重对比、资源计数和 retention 反例。
- [历史执行流水](archive/history/20260910_result_review/EXECUTION_HISTORY.md)：保留此前全部记录，旧“下一步”不再授权执行。
- [历史计划入口](archive/history/20260910_result_review/PLAN_INDEX_HISTORY.md)：此前入口及研究进展记录。

## 权威文档

| 文档 | 职责 |
|---|---|
| [当前执行计划](active/roadmap/03_CURRENT_EXECUTION.md) | 唯一下一步、依赖顺序和验收条件 |
| [核心需求](active/TAIJI_CORE_REQUIREMENTS.md) | 长期目标 |
| [原生架构](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | 核心对象和学习体系 |
| [继承式成长 v2](active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | 持续学习与结构成长研究依据 |
| [架构方向](active/ARCHITECTURE_DIRECTION_2026_08.md) | 原生身份、成熟技术与 legacy 边界 |
| [Seed 架构](active/SEED_ARCHITECTURE.md) | 客户端、Workbench、provider 所有权 |
| [结果复审](reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md) | 本轮实测结论与限制 |

## 维护规则

active 保留核心需求、架构和唯一执行计划；reference 保留证据解释；archive 保存完成/失效的执行与调试流水；manifests 保存版本化合同。被替代的预注册须标明状态，历史 JSON/权重保留，不通过改写旧成绩制造通过。

每轮按实际结果更新当前计划并提交。知识语料、MCP/IDE、客户端热插拔、watchdog、CUDA 和视觉发布已排入当前计划 P5。
