# 计划归档索引

这里保存已完成、被新路线替代或仅用于历史追溯的计划。归档文档保留当时的实验上下文和决策证据，但不提供当前执行顺序。

## 归档分类

| 目录 | 内容边界 |
|---|---|
| `architecture_design/` | 被 Native v1 替代的早期完整架构草案 |
| `audits/` | 历史审计、缺口盘点和方向评估；大审计已拆成 origins/findings/directions |
| `authored/` | 已冻结的早期设计原则、启动标准和 Legacy 机制说明 |
| `history/` | 阶段收束、对话/实验/项目事件和路线执行日志 |
| `implementation/` | 已完成的修复、旧实现和兼容层实施记录 |
| `reference/` | 只用于历史或训练背景的参考资料 |

## 最近一次路线拆分

| 文档 | 归档原因 |
|---|---|
| [SEED_ROADMAP_RELEASE_LOG_2026_08.md](history/SEED_ROADMAP_RELEASE_LOG_2026_08.md) | 原总路线 P8 产品与发布记录已完成，移出当前执行文件 |
| [SEED_ROADMAP_EXECUTION_FOUNDATION_2026_08.md](history/SEED_ROADMAP_EXECUTION_FOUNDATION_2026_08.md) | 路线校准前的历史 Gate 与执行记录 |
| [SEED_ROADMAP_WORKBENCH_PLAN_2026_08.md](history/SEED_ROADMAP_WORKBENCH_PLAN_2026_08.md) | 全盘审计形成的 W0–W7 规划原文，当前摘要已提炼到 active |
| [SEED_ROADMAP_WORKBENCH_PROGRESS_2026_08.md](history/SEED_ROADMAP_WORKBENCH_PROGRESS_2026_08.md) | Workbench Closure 已完成 slice 和历史下一步连续记录 |
| [PLANS_INDEX_STATUS_SNAPSHOT_20260829.md](history/PLANS_INDEX_STATUS_SNAPSHOT_20260829.md) | 旧入口同时承载导航、状态和日志，现已拆分 |
| [ARCHITECTURE_COMPROMISE_ORIGINS.md](audits/ARCHITECTURE_COMPROMISE_ORIGINS.md) | 原架构妥协审计的早期机制与历史状态 |
| [ARCHITECTURE_COMPROMISE_FINDINGS.md](audits/ARCHITECTURE_COMPROMISE_FINDINGS.md) | 原架构妥协审计的系统性缺口与组件 findings |
| [ARCHITECTURE_COMPROMISE_DIRECTIONS.md](audits/ARCHITECTURE_COMPROMISE_DIRECTIONS.md) | 原架构妥协审计的改进方向与候选路线 |
| [SEED_GATE_CI_HISTORY_2026_08.md](history/SEED_GATE_CI_HISTORY_2026_08.md) | 2026-08-29 以前累计的 Gate、CI 事故、修复和逐轮证据；活动文件只保留仍生效规则 |
| [SEED_W7_EXECUTION_PLAN_SNAPSHOT_20260829.md](history/SEED_W7_EXECUTION_PLAN_SNAPSHOT_20260829.md) | R1–R5 原详细蓝图与已完成过程；后续阶段已重写为 C0–C8 活动计划 |
| [SEED_NATIVE_PHASE_HISTORY_20260829.md](history/SEED_NATIVE_PHASE_HISTORY_20260829.md) | P0–P7 原滚动实现记录；活动版只保留长期范围、阶段状态和退出门槛 |

其他归档文档的用途见各自文件头部说明。原路线、旧 NeuroPlex/Transformer 设计和测试调试记录都不应被重新解释为当前 Taiji 事实。

## 使用规则

- 项目使命和长期核心讨论必须保留在 `plans/active/TAIJI_CORE_REQUIREMENTS.md` 与 `plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md`。
- 当前执行顺序只看 `plans/active/SEED_DEVELOPMENT_ROADMAP_2026_08.md`，当前唯一下一步只看 `plans/active/roadmap/03_CURRENT_EXECUTION.md`。
- 当前身份、所有权和产品边界只看 `plans/active/ARCHITECTURE_DIRECTION_2026_08.md` 与 `plans/active/SEED_ARCHITECTURE.md`。
- 归档文档中的“下一步”均是历史记录，不得直接恢复执行。
