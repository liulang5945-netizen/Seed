# 10.3：预测、有界 rollout 与选择·契约与基线（不授予训练）

> 文档状态：[合同] — owner 已于 2026-11-01 批准立项。本合同冻结预测-选择接口、账本分账、四条对照臂与判分口径；交付的是**参考规划器基线**，模型侧预测器不在本步。

日期：2026-11-01。10.1 的环境与 10.2 的绑定契约不改；`taiji/living_core.py` 字节保持不变。

## 1. 接口与账本（契约）

`plan(binding, state, depth)` → `ActionIntent`：

- `action`：选中的动作（`(object, place)`）；
- `predicted`：对该动作**预测**的下一状态；
- `imagined`：有界 rollout 中**想象**到的后续状态序列（深度 ≤ `depth`）；
- `expected_score`：rollout 下的预测终态分；
- `explanation`：一句中文说明（**仅供阅读**）；
- `ledger`：逐步记录 `{step, action, predicted, imagined, actual, predicted_matches_actual}`。

**三类 Outcome 分开记**：`predicted`（模型对下一步的预测）、`imagined`（rollout 内部状态）、`actual`（真实执行后的状态）不得写在同一个字段里，也不得相互覆盖。

## 2. 四条对照臂

1. `planner`：参考规划器（内部转移模型与世界一致，深度 2）。
2. `greedy`：简单策略——每步选"让当前对象离目标位置最近"的动作，不做 rollout、不预测。
3. `no_prediction`：声明种子下随机合法动作，不做任何预测。
4. `perturbed_model`：参考规划器，但内部转移模型**故意错位**（把"移到 A"预测成"移到 B"），用来证明 `predicted_matches_actual` 这一栏能变假。
5. `wrong_goal`：参考规划器，但绑定被**改错**（目标位置换成另一个位置）。

## 3. 判据

1. **分账成立**：每步 `predicted`／`imagined`／`actual` 三个字段都在场且互不覆盖；`imagined` 长度**不超过**声明深度，且当 rollout 未因目标达成提前停止时应等于深度（达成后继续想象只会给已解决的局面白记成本）。否则 `ledger_not_separated`。
2. **预测可比且能变假**：`planner` 的 `predicted_matches_actual` 全为真；`perturbed_model` 至少一步为假。两者同时成立才算这一栏有判别力，否则 `prediction_column_blind`。
3. **规划优于简单策略**：在全部夹具上 `planner` 终态分 ≥ `greedy`，且至少一条严格更高。否则 `planner_no_advantage`。
4. **目标条件真的被用上**：至少一条夹具上 `wrong_goal` 的终态分低于 `planner`。否则 `goal_condition_unused`。
5. **不凭解释判分**：评分函数只接收终态与步数；`explanation` 不进入任何判分路径（数据流检查＋AST 扫描 `planner` 模块里评分调用不接受文本）。否则 `explanation_scored`。
6. **随机臂不占优**：`no_prediction` 不得在全部夹具上追平 `planner`（否则夹具太窄）。否则 `fixture_too_narrow`。

## 4. 预算、出口与交付

- 预算：≤300 CPU 秒 / 600 墙钟秒；超限 `budget_stop`。
- 出口：`passed_10_3_prediction_contract_only`／`ledger_not_separated`／`prediction_column_blind`／`planner_no_advantage`／`goal_condition_unused`／`explanation_scored`／`fixture_too_narrow`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`taiji/goal_planner.py`、`tests/taiji_native/test_goal_planner.py`、`scripts/training/verify_taiji_goal_planner_10_3.py`、`reports/taiji_core_goal_planner_10_3_20261101.json`、MAINLINE/03 同步。
- **通过只表示**预测-选择契约、分账与四条对照成立，且参考规划器在该夹具集上按声明工作；**不表示**模型能预测后果或作出选择，也不表示 11/12/13 准入。**训练仍未批准。**
