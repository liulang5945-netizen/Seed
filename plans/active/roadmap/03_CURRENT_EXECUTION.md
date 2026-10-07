# 当前执行（N 主线，2026-10-06 起）

> **本文是 N 主线的活动文档**。M 主线（M0–M8）已于 2026-10-06 收束：其执行日志（含全部停靠块与历史读数）封存于 [archive/history/m_series_execution_20261006/03_CURRENT_EXECUTION_M.md](../../archive/history/m_series_execution_20261006/03_CURRENT_EXECUTION_M.md)——**历史读数与"下一步"指示不得引用为现行**；引用历史请带批次出处。

## 现行状态（2026-10-06 收束时点）

- **M6 provider 与产品收口**：收官（[批准书 20261006](../../reference/M6_SCOPE_EXCLUSION_APPROVAL_20261006.md)，02:145 声明入档；台账 ㊵-455）。
- **M7 CI 与发布**：限定开线，**CI 已绿**（`m7-ci` workflow：build＋doc-sync 43 叶＋hygiene 18 叶，run 37453182065；台账 ㊵-456/457）。发布面 owner 裁暂缓；剩余＝manifest 对齐、测试面接 CI（前置 ㊵-145 未裁）、ubuntu-26 复验（2026-10-19 后）。
- **M8 CUDA 与规模化**：摘除主线（重启＝真设备＋能力就绪＋owner 单独批准）。
- **A 支线／R2 线**：已收官（SPEC-A-26 结案归档；R2 v1 标定判停结案）。

## 活动规划与导航

- **下一轮主线（N 系列）**：[09_NEXT_MAINLINE_PLAN.md](09_NEXT_MAINLINE_PLAN.md)——架构与模型（2026-10-07 依 **VISION 全文**重订：主线叙事"修复期→加速期→补全期"、每 N 项六要素、§1.1 VISION 全文接线总表（§4/§5/§8/§9/§10/§15/§16/§17/§18 逐项锚点）、owner 待批决策点按时间序汇总、预注册三件模板（§16.1/§18.8/§13））。北极星＝经验→能力转化速率，S7 四组基线为第 0 点。排序不变：N1 P1 接口修复（S5 便宜证伪最先）→ N2 巩固通电 → N3 容量决定 → N4 读取形态补全 → N5 自进化唤醒（N3 裁乙即启动，一体两裁）→ N6 工程债随实施项清。
- **项目收束**：[M0_M7_PROJECT_CONSOLIDATION_20261006](../../reference/M0_M7_PROJECT_CONSOLIDATION_20261006.md)——逐阶段终态、跨阶段遗留登记（10 项带重启条件）、继承资产。
- **连续台账**：[08_UPSTREAM_SYNC_PLAYBOOK.md](08_UPSTREAM_SYNC_PLAYBOOK.md)（㊵-455＝M6 收官条；㊵-456 起＝M7/N 系列时代；更早条目为 M 时代历史）。

## N 系列执行日志（自下而上追加，每项开工前预注册判据）

（暂无——N1/S5 预注册待 owner 点头后开始。）
