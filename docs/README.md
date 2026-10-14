# Seed 项目地图

本页是目录与阅读导航。当前工作顺序只由 [03 当前执行](../plans/active/roadmap/03_CURRENT_EXECUTION.md) 决定；本页不复制动态实验状态。

## 按目的找内容

| 目的 | 入口 |
| --- | --- |
| 了解整个项目及成熟度 | [全项目评估](../plans/reference/PROJECT_WIDE_ASSESSMENT_20261014.md)（2026-10-14 快照） |
| 理解长期目标 | [核心需求](../plans/active/TAIJI_CORE_REQUIREMENTS.md) |
| 理解认知架构 | [原生架构](../plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md)、[代码 Wiki](../TAIJI_CODE_WIKI.md) |
| 查看当前主线与下一步 | [当前执行](../plans/active/roadmap/03_CURRENT_EXECUTION.md)、[N 主线规划](../plans/active/roadmap/09_NEXT_MAINLINE_PLAN.md) |
| 查看本轮开发计划修订 | [完整修订方案](../plans/reference/PLAN-N-01_development_revision_20261014.md)、[N1 实施附件](../plans/reference/PLAN-N1-06_carrier_consumer_alignment_20261014.md) |
| 查实验规格、裁决和评估 | [参考件索引](../plans/reference/README.md)、[研究完成度](../plans/reference/N_MAINLINE_COMPLETION_AUDIT_20261011.md)（含后继追加） |
| 查产品收官范围及遗留 | [阶段收束](../plans/reference/M0_M7_PROJECT_CONSOLIDATION_20261006.md)、[M6 批准书](../plans/reference/M6_SCOPE_EXCLUSION_APPROVAL_20261006.md) |
| 查连续工程记录 | [08 台账](../plans/active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md) |
| 查失效路线与旧阶段 | [归档索引](../plans/archive/README.md) |
| 查收纳结果和本地产物 | [本轮整理记录](../plans/reference/PROJECT_ORGANIZATION_20261014.md) |
| 恢复已冷归档的实验 | [本地产物与恢复说明](../outputs/README.md) |

## 目录归属

| 目录 | 放什么 | 维护方式 |
| --- | --- | --- |
| `taiji/` | 原生认知内核 | 活动研究代码 |
| `seed/`、`seed_platform/`、`api/` | 产品运行时、平台路径和接口 | 与内核分层维护 |
| `taiji-harness/` | web/桌面产品及打包，独立 pnpm workspace | 遵循内部 AGENTS.md |
| `neuroplex/` | 历史兼容实现及回归涉及的代码 | 保留兼容，不因旧名称批量删除 |
| `instruments/`、`scripts/`、`tests/` | 仪器、可复用任务和测试 | 可复用脚本留在 scripts，历史脚本已有 archive/legacy 分区 |
| `plans/active/` | 根需求、架构、活动调度 | 当前执行以 03 为准 |
| `plans/reference/` | 预注册、批准书、判读、专题评估 | 文件日期是证据时点，不自动等于失效 |
| `plans/archive/` | 历史计划与被替代设计 | 保留追溯，旧下一步不生效 |
| `reports/`、`artifacts/`、`design/` | 原始报告、证据资产、设计记录 | 不覆盖或删除冻结证据 |
| `output/`、`outputs/` | 已跟踪证据和本地运行产物混合 | 逐项登记；不可整体忽略、移动或删除 |
| `checkpoints/`、`data/`、`taiji_data/` | 权重、训练及运行数据 | 清理前确认恢复与引用关系 |
| `security/`、`.local/`、`.workbuddy/` | 凭据、会话与项目记忆 | 保留，不作为普通缓存清理 |

目录规范的权威清单见 [FOLDER_STRUCTURE_RULES](FOLDER_STRUCTURE_RULES.md)，凭据及 Git 卫生见 [REPO_HYGIENE_RULES](REPO_HYGIENE_RULES.md)。
