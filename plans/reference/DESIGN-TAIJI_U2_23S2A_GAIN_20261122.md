# 23-S.2a 预注册：增益补门试点（3 臂 × 1 seed；≤600 CPU；零 check）

> [预注册] — 2026-11-22（D-20261122-277）。上游 [23-S 全门映射](MAPPING-TAIJI_U2_23S_GATES_20261122.md)：唯一缺口为「增益≥0.05 缺同协议基线」。**试点，不签 23-S。**

## 1. 设计（同 23-S.1b 协议；seed 20261120）
- R（路由臂）：沿用 23-S.1b 该 seed 读数（Q = min(CN unseen, P-B, P-D) = 0.8359）；
- S（共享臂）：全 72 单元、无任务路由、无保护，同步骤训练；
- F（冻结臂）：CN 阶段同 R；CN 后冻结全部单位参数，P-B/P-D 只训读出头；
- Q := min(CN unseen, P-B, P-D)；增益 := Q_R − max(Q_S, Q_F)。

## 2. 判据（试点）
- 试点增益 ≥0.05 ⇒ `23s2a_gain_positive`（进入 23-S.2b 3 seed 正式确认，≤1,800 CPU）；
- 否则 ⇒ `23s2a_gain_absent`；超帽 ⇒ `budget_stop`；工程失败 ⇒ `engineering_failure`。

## 3. 披露
1 seed 试点，只作机制归因；不签 23-S；零 check/final；固定分区实验层机制。
