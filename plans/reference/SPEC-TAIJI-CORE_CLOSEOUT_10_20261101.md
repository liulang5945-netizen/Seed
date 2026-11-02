# 10 阶段收束依据与负结果卡（不授予训练，收口决定权在 owner）

> 文档状态：[合同] — 10.1–10.7 的四个预定出口已全部判负（`model_below_random`／`contrast_below_floor`／`action_gap_below_floor`／`training_below_chance_plus`）。本合同只把已经发生的读数收拢成一张可复核的卡；**不改口径、不重跑任何探针、不训练、不发布权重**。是否以本卡收口 10、以及 11 是否继续保持不准入，按 §3 的出口由 owner 裁定。

日期：2026-11-01。`taiji/living_core.py` 字节保持不变。

## 1. 输入（六件，缺一即 `engineering_failure`）

1. [10.4 选择读数](../../reports/taiji_core_subject_selection_10_4_20261101.json)
2. [10.5 目标条件化接口](../../reports/taiji_core_goal_conditioned_10_5_20261101.json)
3. [10.6 动作特定构造](../../reports/taiji_core_action_verdict_10_6_20261101.json)
4. [10.7 第一次有界训练](../../reports/taiji_core_goal_training_10_7_20261101.json)
5. [11 准入报告](../../reports/taiji_core_l3_admission_11_20261101.json)
6. [09 收束卡](../../reports/taiji_core_09_closeout_20261101.json)（只提供同一候选的身份摘要）

本步不产生新读数，只做一致性复核。

## 2. 判据（每条都能翻）

1. **同一候选**：四份探针报告的 `parent_model_digest` 两两相等，且等于 09 收束卡的 `identity.parent_model_digest` ⇒ 否则 `identity_mismatch`。
2. **出口在册**：10.4＝`model_below_random`、10.5＝`contrast_below_floor`、10.6＝`action_gap_below_floor`、10.7＝`training_below_chance_plus`，逐条精确比对 ⇒ 任一不符即 `exit_not_recorded`。
3. **无能力声称**：10.4 `capability_passed is False`；10.5／10.6／10.7 `capability_claimed is False`；四份 `training_admitted is False` ⇒ 否则 `capability_claim_forbidden`。
4. **常量偏好机制可复现**：10.7 的未训练与两条训练臂，命中序列各自必须恰为 `FTFTFT` 或 `TFTFTF` 之一、命中数都是 3/6；且两条训练臂的序列必须**不同**（证明训练换了一个常量，而不是变成条件化）⇒ 否则 `mechanism_not_reproduced`。
5. **保持门读数在册**：lr=1e-4 的 `retention_drift > 0.030`、lr=1e-5 的 `retention_drift ≤ 0.030` ⇒ 否则 `mechanism_not_reproduced`。
6. **损失降幅在册**：lr=1e-4 的 `last_loss / first_loss ≤ 0.70` ⇒ 否则 `mechanism_not_reproduced`。
7. **权重不出库**：10.7 `weights_published is False` 且 `default_model_replaced is False` ⇒ 否则 `weights_published_forbidden`。
8. **准入未回退**：11 报告 `status == step11_not_admitted`、`training_admitted is False`、`l3_scope_started is False`、`P2_one_capability_by_the_model.passed is False` ⇒ 否则 `admission_regressed`。
9. **预算在册**：每份探针报告 `cpu_seconds ≤ budget.cpu_seconds`；本卡自身 ≤120 CPU 秒／240 墙钟秒 ⇒ 否则 `budget_stop`。

判据 4 只约束"命中序列是交替的"这一可机检事实；它**不**证明主体"理解"了目标，也不排除别的解释——机制句只描述读数形状。

## 3. 出口与卡字段

- 出口：`closeout_10_evidence_complete`／`identity_mismatch`／`exit_not_recorded`／`capability_claim_forbidden`／`mechanism_not_reproduced`／`weights_published_forbidden`／`admission_regressed`／`budget_stop`／`engineering_failure`。
- 卡必须带：`stage_exits`（四项出口原文）、`p2_satisfied: false`、`step11_status`、`stage_close_decision: "owner_pending"`、`mechanism`（一句机器可复核的描述）、`limits`。
- **通过只表示**：四个预定出口都判负、读数可复核、允许声称的句子没有越界。**不表示**主体能选择、能规划或能绑定，不表示任务能力，不表示 10 的收口已获 owner 批准，也不授权任何训练。

## 4. 交付

本合同、`scripts/training/verify_taiji_closeout_10.py`、`tests/taiji_native/test_closeout_10.py`、`reports/taiji_core_10_closeout_v2_20261101.json`、MAINLINE／03／索引同步。

## 5. 执行更正（2026-11-01，同日）

首版卡写出后，验证器被 `black` 格式化（仓库口径 `black --check .`），脚本字节变化 ⇒ 首版卡 `source_hashes` 里的自绑定过期。按「不删旧件、升版重跑」处理：`reports/taiji_core_10_closeout_20261101.json`（v1）保留，[v2](../../reports/taiji_core_10_closeout_v2_20261101.json) 为现行绑定。两次的九项读数与状态逐项相同，只有墙钟与那一枚哈希不同。
