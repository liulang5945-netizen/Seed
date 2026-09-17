# M5 P3b-v2 pilot 结项：按预注册反假设**归因架构层**

日期：2026-09-17。状态：**已结项**。
依据：[P3b-v2 预注册](M5_P3B_V2_GOAL_ALIGNED_PREREGISTRATION_FROZEN_20260917.md) §3 判据 / §4 停止线。

## §1 判定

**J6（dev `exact_response` > 0）未达成：`0/20`** ✗

按 §4 停止线第 2 条「**J6 未达成（dev exact 仍为 0）⇒ 目标对齐未发生，按反假设结项**」
⇒ **缺口归因架构层**。

## §2 实测数据

**dev（20 episodes，`--defer-final`，final 未读）**

| 指标 | baseline（零步） | child（2 epochs × 12） | 变化 |
|---|---|---|---|
| `teacher_forced_accuracy` | 0.0000 | **0.1969** | **+0.1969** ✓ |
| `teacher_forced_mean_surprise` | 5.5984 | **4.5277** | **−1.0707** ✓ |
| **`exact_response`** | **0/20** | **0/20** | **0** ✗ |

**其它已记录信号**：`teacher_forced_mean_surprise_delta = −1.0707`；
`baseline_prompt_sensitivity_rate = 1.0`；`child_prompt_sensitivity_rate = 1.0`；
`child_output_changed_after_update_rate = 1.0`；`checkpoint_read_only = true`；
`native_mode_only = true`；`external_provider = false`。

**train（40 episodes）**：`teacher_forced_accuracy = 0.2957`、`surprise = 3.7158`；
逐 episode 轨迹（`reports/p3b_v2_pilot_20260917/progress.jsonl`）：
`response_accuracy` 0.0222 → 0.4222 单调上升，`surprise` 5.5274 → 2.2529 单调下降。

**输出样本**（`dev-fact-01`，参考回答「根据已知信息，是白色。」）：
```
丸渋渋渋游䯯色色色色艂||||end>
<|||||
```
⇒ 仍是不可读字节，但**已出现 `end>` 结构片段**。

## §3 结论：三方叠加仍不足

P3b-v2 相对既往工作的差异化是**三者叠加**：
① H3.7B 的 prior/credit 一致性修复；② 语料 **3.3×** 放大（24 → 80 行）；
③ byte-aligned 16-byte 窗口 target。

在 pilot 预算（2 epochs × 12 episodes）下：

- **该组合确实产生了"可测量的泛化"** —— dev teacher-forced accuracy `0 → 0.1969`、
  surprise `−1.07`，且 `child_output_changed_after_update_rate = 1.0`（更新确实改变了输出）；
- **但没有产生"可自行生成的正确答案"** —— `exact_response` 仍 `0/20`，输出仍不可读。

⇒ 按预注册**归因架构层**：在「逐字节自监督 + 局部信用」的架构下，
**"teacher-forced 分布对齐"与"能自行生成正确 token 序列"之间存在未被跨越的鸿沟**。
这与 H3.5 复审的既有结论一致 ——「`prefix state → 单步 byte 局部误差` **缺少贯穿回答的内容计划**」。

## §4 界限（本结项**不能**主张什么）

- 只覆盖 **pilot 预算**（2 epochs × 12 episodes、80 行语料）⇒ **不能**断言"更大预算也无效"；
  那需要**新的预注册与新授权**。
- **不**把 dev accuracy `0.1969` 表述为"语言能力" —— 它是 **teacher-forced 逐字节准确率**，
  不是生成质量；**不**据此晋级 L2、**不**宣称能力收益。
- **final 未读**（`--defer-final`）。
- 训练**起点是从零构建**（非 H3.7B child），理由见预注册 §7.3。

## §5 下一步（需决策）

pilot 已按规则把缺口归因**架构层** ⇒ 下一步不是"再加机制"或"再加预算"，而是架构级决策：

- **(a) 接受架构层结论**，把 R2 主线转向其他方向（例如产品侧迁移评估）；
- **(b) 设计"内容计划"层的架构改动**（H3.5 复审已指出该缺口的具体形态）并另立预注册；
- **(c) 先补 DEBT-I7（测试隔离）**，恢复"全量测试"作为可信验收手段 ——
  目前全量会混入 5 个假失败，有掩盖真失败的风险。

**建议顺序：先 (c) 再 (a)/(b)** —— 因为无论走哪条路，**都需要一个可信的验收手段**。
