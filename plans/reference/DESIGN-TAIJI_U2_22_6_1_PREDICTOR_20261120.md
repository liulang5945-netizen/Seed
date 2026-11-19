# 设计卡（草稿）：22.6.1 gain/risk/cost 预测器

> 文档状态：[设计卡→合同已执行] — 2026-11-20。22.6.1 合同已冻结并执行：**`predictor_gain_pass`**（held-out Huber 改善 14.25%，20.69≤帽 40 CPU，D-20261119-26）。依附：22.6.0 C 任务合同（`c_task_ready`）＋32 world 开发池（`pool_admitted`，最强固定 `0.83`）。**本卡只做零训练的数据集生成＋预测器协议＋定价；预测器训练须在合同冻结后按常设授权记录并执行。** 数据集回执：[`taiji_u2_22_6_1_labels_20261120.json`](../../reports/taiji_u2_22_6_1_labels_20261120.json)（1,536 行、`2.38 CPU`、`0` 优化器步、`0 check/final`），数据：`output/s2_22_6_1/labels.jsonl`（脚本确定性重生成；回执含 sha256）。

## 1. 任务与答案域

- 每个决策点（32 world × 3 session × 16 tick＝1,536 点）预测四个候选动作的**实现效用**：
  - `ANSWER_NOW`：用最近观测词作答并锁定；
  - `OBSERVE_THEN_DEADLINE`：当前 tick 观测→等到 deadline→deadline 再观测→作答；
  - `VERIFY_THEN_NOW`：花 1 次 VERIFY 取当前真值词→立即作答；
  - `REST_THEN_DEADLINE`：不再观测，deadline 用最近词作答。
- 标签＝**已实现的效用**（`1[answer==deadline truth] − 累计操作成本`），不是模型偏好；成本冻结：观测 `0.02`、等待 `0.01`/tick、求证 `0.10`。
- 特征（10 维，全部决策时可见、无真值泄漏）：`tick`、`decisions_left`、`staleness`（距上次观测 tick 数）、最近观测词 4 维 one-hot、`observations`、`verify_left`、`cost_so_far`。

## 2. 数据划分与判据（读前冻结）

- 划分：32 world 中 **24 train / 8 held-out**（按 world 划分，session 不跨拆分）；标签统计：`ANSWER_NOW 0.5681`、`OBSERVE_THEN_DEADLINE 0.755`、`VERIFY_THEN_NOW 0.5423`、`REST_THEN_DEADLINE 0.4931`。
- 预测器：小 MLP（10→32→4，Huber 损失），训练仅用 train split；held-out 只读一次出终值。
- **主判据（22.6.1）**：held-out 归一 Huber 误差比「fit 常数均值基线」低 **≥10%**；基线＝每动作在 train 上的均值常数预测。
- **分账（预注册）**：点预测误差（归一 Huber）与选择收益（argmax 策略在 held-out 上的效用 − 最强固定 `0.83`）分开报；选择收益在 22.6.2 主判据使用，不在本卡签字。
- 出口顺序：`engineering_failure` → `not_admitted`（池/数据摘要不符）→ `budget_stop` → `predictor_insufficient`（Huber 未达标）→ `predictor_gain_pass`。

## 3. 定价

| 项 | CPU |
|---|---|
| 标签数据集（已实测） | `2.38` |
| 预测器训练（1,536 行 × 200 epochs × 小 MLP） | `<1` |
| held-out 评测＋复算＋工程余量 | `≈5` |
| **包价估** | **`≈10`** |
| **硬帽（×3）** | **`30 CPU／10 min wall`** |

零 check/final；训练 step 数量级 `10^5` 参数更新以内，按常设授权记录授权。

## 4. 边界与后继

- 本卡不签任何选择/自主能力，不读 check/final；预测器只是 22.6.2 策略对照的组件之一。
- 后继：① 冻结 22.6.1 合同（判据摘要＋数据/池 sha256 pin＋报价）→ smoke（数据完整性＋基线可复算）→ 训练执行（常设授权记录）→ 出判词；② 通过则进 22.6.2 策略对照（可学习策略/固定日程/反应 ν/等待控制，以 `0.83` 为最强固定基线）；③ 未过则按 `predictor_insufficient` 收束，禁同范围第 3 次修订（22.5.1 纪律）。
