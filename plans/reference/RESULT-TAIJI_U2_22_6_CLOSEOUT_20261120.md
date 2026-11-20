# 结果卡：22.6 自驱与递归发展（结案 `autonomous_choice_negative / recursive_negative`）

> 文档状态：[结果卡/结案] — 2026-11-20（D-20261119-148）。上游：[22.6 开卡备忘](BRIEF-TAIJI_U2_22_6_OPENING_20261120.md)、[22.6.0 C 任务合同](../manifests/taiji_u2_22_6_0_c_task_20261120.json)、[22.5.1 正式结案](RESULT-TAIJI_U2_22_5_1_FORMAL_CHECK2_20261120.md)。**本卡只汇总已在「条件预研」中落盘的有界探针，不新增训练、不读 check/final。**

## 1. 子卡与读数

| 子卡 | 判词 | 关键读数 | 价 | 回执 |
|---|---|---|---|---|
| 22.6.0 C 任务 fixture | **`c_task_ready`** | C1 拒收（`answer_without_observe` 唯一）、C2 `wait14_observe_answer`、C3 `wait0_observe_answer`；8 seeds 重放一致；期限/行动成本/等待/无新提示齐备 | 9.64 CPU（帽 40） | [run](../../reports/taiji_u2_22_6_0_c_fixture_run_20261120.json) |
| 22.6.1 收益/风险/成本预测器 | **`predictor_gain_pass`** | 1,536 行；held-out Huber **0.23398 vs 基线 0.27287**，改善 **0.1425 ≥ 0.10**；oracle 上界 0.8191 | 20.69 CPU（帽 40） | [run](../../reports/taiji_u2_22_6_1_predictor_run_20261120.json) |
| 22.6.2 选择策略（离线/贪心） | **`policy_gain_absent`** | learnable **0.6637**、fixed observe-t0 0.6467、**fixed wait-deadline 0.8300**、reactive 0.6267、wait 控制 −0.15 | 见回执 | [run](../../reports/taiji_u2_22_6_2_policy_run_20261120.json) |
| 22.6.2b 上下文成本敏感策略 | **`policy_gain_absent`** | 与 22.6.2 同向：未超过最佳固定策略；bootstrap β 重跑仍未过 | 见回执 | [run](../../reports/taiji_u2_22_6_2b_policy_20261120.json) |
| 22.6.3 两次相依改进（可行性） | **`recursive_feasibility_absent`** | `dependent_refine 0.2` vs `fixed_parent 0.2`，差值 **0.0** | 21.09 CPU | [run](../../reports/taiji_u2_22_6_3_recursive_feasibility_20261120.json) |
| 22.6.3 多 seed | **`recursive_multiseed_absent`** | mean margin 0.06、**min margin −0.10**（要求 min≥0.05、mean≥0.10）；5 seed 中 2 个 ≤0 | 87.16 CPU | [run](../../reports/taiji_u2_22_6_3_recursive_multiseed_20261120.json) |
| 22.6.4b 校准 | **`calibration_already_adequate`** | 现有校准已达标，无额外收益 | 见回执 | [run](../../reports/taiji_u2_22_6_4b_calibration_20261120.json) |

## 2. 主终点核对（读结果后不改口径）

- 主终点要求：冻结效用（任务收益−归一成本−旧损伤）相对最强固定/反应策略 **≥0.05**、world 95% 下界 **>0**、**3 seed 同向**、A 损伤 ≤0.02。
- 实测：learnable 0.6637 vs 最强固定 `fixed_wait_deadline` 0.8300 ⇒ **Δ=−0.1663**，且两种策略方法（离线贪心、上下文成本敏感）同向负；主要原因是 C 任务中「等待到期限」本身极强，学习策略未找到超越它的信息—成本组合。
- 结论：**主终点未达**，且 22.6.2/2b 两次独立方法一致 ⇒ 不属统计噪声。

## 3. 递归子卡

- 要求：至少 2 次**相依**改进决策（后次读真已改变的主体），并有固定父反事实。
- 实测：可行性卡 `dependent − fixed = 0.0`；多 seed 卡 `mean 0.06 / min −0.10`，未过预检门 ⇒ **两次相依改进未建立**。

## 4. 判词与收口

- **22.6 结案**：
  - `autonomous_choice`：**negative（范围内）**——C 任务上两次独立策略方法均未超过最强固定策略；与之前 C 池三次负（D-60/123/124）方向一致，本卡是官方 C 任务的独立确认；
  - `recursive_development`：**negative（范围内）**——两次相依改进未建立；与 D-102（重复输入）、D-108（顺序两步）一致。
- 依据：铁律 7（三探针止损）已触发；不追加预算、不新增局部探针、不改门值。
- 范围限定：本卡只覆盖 22.6.0 冻结的 C 任务与 22.6.3 的相依改进构造；不推广为「自主选择/递归发展不可能」。
- **队列**：22.6 结案，主线队首推进至 **22.7**（成熟生长，已有 `growth_benefit_not_established_in_package_22` 预研，开卡时须先处理该负结果）。
- 不签：22.7+、C/D/J、final、模型成型；不改变 22.5.1 的 `joint_gate_check2_pass / b_parent_earned`。

## 5. 预算账

本卡不新增花费；引用预研实测：22.6.0 `9.64`＋22.6.1 `20.69`＋22.6.3 两张 `21.09+87.16`＋其余回执内读数，均在 22.6 计划帽 9,600 CPU 的预研范围内，无超帽。
