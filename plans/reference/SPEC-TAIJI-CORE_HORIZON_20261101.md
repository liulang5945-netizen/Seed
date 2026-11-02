# 09.7b：可持续片段长度定位（不授予正式训练）

> 文档状态：[合同] — 09.7 以 `retention_breach` 结案后，定位"保持仍在门内的最长片段"并报告该处的净收益。**训练仍未准入**。

日期：2026-11-01。候选取 09.7 同一点（parts=5、units=3、lr=1e-5），两臂同 09.7。

## 1. 09.7 留下的缺口

09.7 的 32 步片段里：净收益随步数单调增长（8→32 步：HOLDOUT 1.739e-4→7.187e-4，SECOND_HOLDOUT 1.903e-4→8.147e-4），但**两臂都突破保持门**（只重放 0.084752、出生 0.085631）——是总训练量破坏保持，不是出生造成的。缺口是我只测了末点的保持漂移，**没有逐检查点记账**，所以无法说明它从哪一步起越线。

## 2. 判据（先声明、后测量）

同两臂、同 32 步，但**每 8 步同时记保持漂移与两条评估生涯的净收益**。

1. **可持续点**：设 `steps*` 为最后一个满足"两臂漂移都 ≤0.030"的检查点。若 `steps*` 存在且该点两条生涯的净收益都 ≥1e-4 ⇒ `sustainable_horizon_found`，并冻结 `steps*`。
2. 若 `steps*` 存在但该点净收益 <1e-4 ⇒ `sustainable_but_no_net_benefit`。
3. 若 8 步处就已超门 ⇒ `no_sustainable_horizon`（与 09.6.8/09.6.9 的 8 步读数矛盾，须查面）。
4. **8 步一致性**：8 步处的漂移与净收益必须复现 09.6.8/09.6.9（净收益 ±20%、漂移 ≤0.030），否则 `arm_face_mismatch`。

## 3. 预算、出口与交付

- 预算：≤1800 CPU 秒 / 3600 墙钟秒；超限 `budget_stop`。
- 出口：`sustainable_horizon_found`／`sustainable_but_no_net_benefit`／`no_sustainable_horizon`／`arm_face_mismatch`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`scripts/training/verify_taiji_fragment_horizon.py`、`reports/taiji_core_fragment_horizon_20261101.json`、MAINLINE/03 同步。
- **不声称**：任务能力、长程偿付、跨参考图外推；`steps*` 只是"保持仍在门内的最长声明检查点"，不是"结构生长的可持续上限"。
