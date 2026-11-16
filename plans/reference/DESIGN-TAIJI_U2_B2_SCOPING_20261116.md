# 22.5.B.2 设计卡：把纠正可塑性定域到记录在场

> 卡号：`22.5.B.2`｜日期：2026-11-16｜状态：**已结案（负，informative）**——见[结果卡](RESULT-TAIJI_U2_B2_SCOPING_20261116.md)（D-20261116-11，`retention_broken_under_scoping`）与[B 线收束卡](CLOSEOUT-TAIJI_U2_B_LINE_20261116.md)；本卡保留冻结时的设计原文。
> 契约 [`taiji_u2_b2_scoping_20261116.json`](../manifests/taiji_u2_b2_scoping_20261116.json)（`criteria_sha256=75daaf6f…`）；[smoke 回执](../../reports/taiji_u2_b2_scoping_smoke_20261116.json)、[定价回执](../../reports/taiji_u2_b2_scoping_price_20261116.json)。
> 前置：[B.1 结果卡](RESULT-TAIJI_U2_B1_SEPARATION_20261116.md)（D-20261116-09）——参数分离兑现但学成记录盲全局平移；探针证明 presence 信号存在却被忽略，容量/剂量排除。owner 已裁决**立项 B.2 功能定域**（D-20261116-10）。

## 1 这张卡问什么

**给纠正 delta 加上记录在场门控之后，纠正与保持能不能同时成立——且保持是在记录在场条件下挣到的，不是构造出来的？**

相对 B.1 的**唯一结构变量**：`correction="v2"`（身份 `taiji-u2-subject-v7`），`delta = adapter([record, context]) × valid.any()`——`valid` 是 copy 分支自己的 presence 信号。无新参数（仍 4 张量/408 标量，23,150/198/42 状态字段），起点仍是 A.5 反馈前 bundle（**不是** B.1 反馈后），可训集/数据/协议/阈值逐字同 B.1 lean。

## 2 保持门为什么这次既可失败、又不构造性通过

- **无记录 A 回合**：门闭 ⇒ delta 精确 0 ⇒ A 输出逐位不变。这一项**降级为结构监视项**（monitor-only），明令不得引为"保持被挣到"——这是 A.5 陷阱的自觉规避。
- **有记录 distractor A 回合**（新保持臂）：32 个 fit 问题各配一条**关于不同实体**的合法释放记录，正确答案仍是冻结真值。门开 ⇒ 训好的 adapter 在线 ⇒ **能**打坏它们。损伤 = 反馈前 distractor 精确 − 反馈后 distractor 精确（基线逐 seed 实测；出生 adapter 零 ⇒ 基线 == v4 带记录）。门 ≤0.02。
- **distractor 从不参与训练**（训练只 B 相关记录）⇒ 对无关记录的中性必须泛化，不是背出来的。

## 3 门（读数前冻结，阈值＝22.5.1 父门自身口径）

| 判据 | 阈值／定义 |
|---|---|
| 主项 | 无记录 A fit 32 ≥ **0.80**（表达项；损伤为结构监视） |
| 纠正率 | 24 首过错例轮末 ≥ **0.80**，3 seed 全部 |
| **保持（distractor）** | distractor 损伤 ≤ **0.02**，逐 seed，**可失败** |
| 反馈增益 | learned − control ≥ **0.05**；72 对 world 配对 95% 下界 **>0** |
| 稳定性 | 首过对→二遍错 ≤ max(1, 0.10×首过对数) |
| 反事实臂 | 同实体第三位置记录须被跟随 ≥ **0.80**（凡纠正过门的 seed） |
| 敲除臂 | cut 下 B ≤ 首过+0.05 且 A 摘要 == 基线（带记录也精确还原 v4） |
| 门结构 | 闭精确零／开等于未门控 adapter／返回 context 逐位核验 |
| 路径纯度 | 176 非可训基张量逐位 == start |
| 控制臂 | 3 例曝光零更新，参数逐位不变、二遍 == 首过 |
| 恢复 | 独立进程 32 A＋24 B＋**32 distractor** 逐字节 |

协议＝B.1 lean（k=8、不回滚、无诊断重问、轮末重问进门）。评测贪婪精确字节＋EOS＋UTF-8。

## 4 出口（优先序）

1. `retention_broken_under_scoping`（任一 seed distractor 损伤 >0.02）；
2. `knockout_failed`（含门结构核验失败）；
3. `engineering_failure`（纯度/恢复/控制）；
4. `counterfactual_failed`；
5. **`scoping_earns_joint_gate`**（全门 3 seed）⇒ 联合父门在定域路径下可达（机制可达性）；
6. `correction_absent_under_scoping`／`correction_partial_under_scoping`；
7. `budget_stop`、`not_identifiable_by_design`、`pricing_over_budget`。

**预注册的 informative 负结果**：presence 定域 ≠ relevance 定域。adapter 只在 B 相关记录上训过，distractor 对它是分布外——若 distractor 保持失败，读作"presence 门不够，需 relevance 条件化或检索"，不是工程失败。

## 5 能签与不能签

**能签**：定域这一结构干预下纠正与保持能否同时成立（保持在记录在场条件下挣到），及门结构/纯度/敲除归因。**不能签**：父/能力/check②/22.6；无记录 A 的 0 损伤不得引为保持证据；正结果后继是正式父卡，负结果后继是 B.3 裁决（relevance／检索），不自动加剂量。

## 6 预算（实测单位）

| 单位 | CPU |
|---|---:|
| 记录条件更新（含 adapter＋门） | **0.40625** |
| B 回合 | 0.21875 |
| A 回合 | 0.046875 |

`scoping_3seed`：576 更新＋96 distractor 回合＋全臂，估计 **428.719**，帽 **493.027 CPU／986.054 wall**（设计帽 600/1200 内，15% 余量）。wiring 另账：smoke **177.13**（3 seed×继承/门闭/门开/distractor 出生中性/敲除/梯度/门结构/合法释放，0 优化器步）、定价 **6.95**（8 步于丢弃模型）。

## 7 唯一的请求（是非题）

**是否授权按 `scoping_3seed` 实价执行 `22.5.B.2`（帽 493.027 CPU／986.054 wall，576 更新＋96 distractor 回合，3 seed）？**

目前 **0 正式优化器调用、0 训练步、0 checkpoint、0 能力声称、0 A 父**。