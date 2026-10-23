# 历史设计与证据恢复索引

当前开发只读[03队首](../active/roadmap/03_CURRENT_EXECUTION.md)、[10详细计划](../active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)与[09设计继承](../active/roadmap/09_NEXT_MAINLINE_PLAN.md)。本目录保留独有设计/证据与必要兼容路径，不再要求每次文档整理复制整份过程流水。

## 留在工作树的历史

| 分类 | 用途 |
|---|---|
| `architecture_design/` | 早期架构及仍被根需求引用的原始依据 |
| `audits/` | 历史妥协/缺口盘点，结论按后继证据解释 |
| `authored/` | 原设计原则与启动标准 |
| `implementation/` | 旧实现/兼容层说明 |
| `reference/` | 独有历史参考资料 |
| `history/` | 必要阶段原件、既有索引与兼容文件；日期不代表仍执行 |
| `history/roadmap_shards/` | 旧W7等分片引用，当前不按旧切片开工 |

## 本次删除与恢复

2026-10-23已将6份整篇重复过渡快照移出工作树；原始计划和当时归档版本均已在Git提交中保存。08删除重复操作流水、保留同步手册/登记标题/必要逐字处方；reference索引删除滚动状态正文，改为主题导航。准确文件映射、SHA、行尾及恢复命令见[历史整理记录](../reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md)。本目录不再把2026-09-10写为“最新归档”。

既有结果层见[研究复核](../reference/TAIJI_RESEARCH_REVIEW_2026_09_06.md)、[M5蒸馏墓碑](../reference/M5_DISTILLATION_TOMBSTONE.md)、[项目收束](../reference/M0_M7_PROJECT_CONSOLIDATION_20261006.md)。它们的冻结判读、批准范围和失败保留；已使用的语料、权重、来源记录及会话数据不按“旧”批量删除。

## 留存规则

- 活动索引维护当前职责，不保留并列历史队首；详细包由10、当前动作由03维护。
- 唯一设计依据、冻结合同/来源SHA、原始评价结果、未完成工作及受用方读取的内容保留原位。
- 重复过程文本在结论蒸馏、引用迁移、Git精确恢复验证后可删除；恢复原文时不把旧额度/权限/判据当新授权。
- 可再生缓存与测试残留按实际白名单清理，保留含源码快照的临时根；不通过重写Git或prune来“清历史”。
