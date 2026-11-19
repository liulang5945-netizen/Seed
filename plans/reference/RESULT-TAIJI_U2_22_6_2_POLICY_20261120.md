# 结果卡：22.6.2 策略对照（`policy_gain_absent`，负面，已执行）

> 文档状态：[设计卡＋结果卡] — 2026-11-20。合同：[`taiji_u2_22_6_2_policy_20261120.json`](../manifests/taiji_u2_22_6_2_policy_20261120.json)（`status=frozen`、`training_authority=owner:standing-authorization:2026-11-20#recommendation-default`、`result.closed=true`）。回执：[run](../../reports/taiji_u2_22_6_2_policy_run_20261120.json)／[smoke](../../reports/taiji_u2_22_6_2_policy_smoke_20261120.json)。**零 check/final。**

## 1. 预注册协议

- 池：32 个已准入 world（`pool_admitted`，D-20261119-24），每 world 3 session × 16 决策；成本冻结（观测 `0.02`、等待 `0.01`/tick、求证 `0.10`）。
- 臂（同一确定性 tick 流，3 个平局种子）：
  - `learnable`：重训 22.6.1 预测器配方（MLP 10→32→4、固定 seed），每决策点按预测效用**贪心 argmax** 行动；
  - `fixed_observe_t0`：t0 观测后立即答；
  - `fixed_wait_deadline`：等到 deadline、观测、作答（本池最强固定，预期 `0.83`）；
  - `reactive_stability`：每 tick 观测，最近两次观测一致即答；
  - `wait_control`：等到 deadline 但不观测、以最近词作答。
- 主判据：`learnable − 最强固定 ≥0.05`、world 配对单侧 95% 下界 `>0`、3 seed 同向；出口顺序 `engineering_failure→not_admitted→budget_stop→policy_gain_absent→policy_gain_pass`。

## 2. 读数（3 seed 完全一致；实耗 `93.48 ≤ 帽 130 CPU`）

| 臂 | 均值效用 |
|---|---|
| **`fixed_wait_deadline`（最强固定）** | **0.8300** |
| `learnable` | 0.6637 |
| `fixed_observe_t0` | 0.6467 |
| `reactive_stability` | 0.6267 |
| `wait_control` | −0.1500 |

`learnable − 最强固定 = −0.1663`，配对单侧 95% 下界 `−0.1998`，3 seed 全部为负 ⇒ **判 `policy_gain_absent`**（预注册负出口）。

## 3. 判读（为什么负、负得是否有信息）

1. **贪心用预测器 ≠ 更好策略**：预测器标签是「在冻结延续规则下」的实现效用，且训练分布来自「每 tick 观测」的默认行为；贪心 argmax 在闭环状态里会过早 `ANSWER`/`VERIFY`，锁定了过期信息——这正是「会估计收益 ≠ 会做决策」的证据。
2. **最强固定是纯等待策略**（0.83）：要超过它，必须**只在低风险时提前作答**；预测器没学到这种风险条件，反而系统性提前答，故低于两个简单固定臂。
3. **不是机制失败而是选择机制失败**：实体/材料/记录通道未参与本卡；负结果指向「需要闭环反馈训练或风险校准规则」，而不是「主体不能选择」。
4. **边界**：贪心闭环与标签的延续语义不完全一致（已在协议中写明）；本卡不签任何自主/选择能力，也不读 check/final。

## 4. 定价修正（披露）

设计稿估 `30 CPU／帽 90`，实测 `93.48`（预测器重训＋32 world 重放＋5 臂模拟）⇒ **原估低估 3 倍**；合同包价改为 `97.5`、帽改为 `130`（实际 93.48）。另：提交前移除一个未使用局部变量（无语义变化），合同 `amendment` 记录旧 pin 与新 pin，已执行结果不变。

## 5. 唯一后继（不复活旧窗）

- 本卡按 `policy_gain_absent` 收束，**不在同一范围调参重开**。
- 后继候选（下一张新卡，需预注册与单独定价）：① **闭环反馈策略学习**（在 C session 内用真实结果更新风险/成本阈值，而非只用离线标签）；② **风险校准选择规则**（用预测器方差/保守下界代替 argmax）；③ 更硬池子（提高提前答风险）——但先做 22.6.3 的「两次相依改进决策」设计，用本负结果作对照。
