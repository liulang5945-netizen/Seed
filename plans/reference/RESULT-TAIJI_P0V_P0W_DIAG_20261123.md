# 结果卡：P0-V 修订任务（4 维 flag 真实 tick）＋P0-W held-out 抑制复测

> [结果卡·探索诊断] — 2026-11-23（D-20261123-359/360）。脚本 [diag_p0v](../../scripts/training/diag_p0v_20261123.py)／[diag_p0w](../../scripts/training/diag_p0w_20261123.py)；回执 [p0v](../../reports/taiji_p0v_diag_20261123.json)／[p0w](../../reports/taiji_p0w_diag_20261123.json)。**1 seed、111.08/107.05 CPU、零 check；探索诊断。**

## 1. P0-V：4 维 one-hot flag 放在真实 tick（`p0v_not_pressured`）

| 项 | 读数 |
|---|---|
| 正常中文（训练后/保持） | 1.000 / 1.000 |
| CN 探针 train/**test** | 0.885 / **0.625** |
| 出生探针 test | 0.430 |
| 修订 acc（400 次） | 0.574 |
| 迁移 / 清零适配器 | 0.506 / 0.432 |

判定 `p0v_not_pressured`。

## 2. P0-W：held-out 抑制复测（`p0w_suppression_holds`）

CN 基座 1.000；4 类 one-hot flag 注入三处放置、探针 75/25 划分：

| 通道组 | all_ticks test | tick0 test | last_tick test |
|---|---|---|---|
| 0:4（对照） | **1.000** | 0.477 | 0.828 |
| **32:36** | **0.242** | **0.281** | **0.219** |
| **40:44** | **0.203** | **0.203** | **0.227** |

⇒ **32:36/40:44 在 held-out 上确实被抑制（≤0.31）**；0:4 在 all_ticks/last_tick 可传输。

## 3. 关键判读：P0-V 的「不压力」是**标签结构伪影**

- P0-V 中 flag 4 类、仅 flag=0 为正常回答 ⇒ **3/4 样本答案 = 配对另一色**；一个**读不到 flag** 的探针只要「总答另一色」即可 ≈0.75 上限。
- 因此 CN 探针 test 0.625 **不代表 flag 被解码**（flag-blind 基线就是 0.75），是**类别不平衡**造成的假压力信号。
- P0-W 用平衡 4 类 flag（直接预测 flag 本身）证明：**抑制成立**。

## 4. 修正后的修订任务设计（下一轮）

1. **平衡 flag**：4 维 one-hot，其中 {0,1}=正常答案、{2,3}=翻转 → flag-blind 基线 = 0.5；
2. 压力判据：held-out CN 探针 ≤0.55（平衡基线 0.5，阈值仍成立）；
3. 任务降到 **4 类**（配对颜色对固定子集）以降低学习难度；
4. 保留字符读取（write_in）＋单元级适配器读 flag。

## 5. 披露

单 seed、合成面；不签 24-S/25-S/29/final/C/D/J；零 check/final。
