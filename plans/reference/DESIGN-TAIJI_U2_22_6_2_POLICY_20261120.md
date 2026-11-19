# 设计卡：22.6.2 策略对照（已执行并收束为负面）

> 文档状态：[设计卡→已执行] — 2026-11-20。合同：[`taiji_u2_22_6_2_policy_20261120.json`](../manifests/taiji_u2_22_6_2_policy_20261120.json)。**执行结果：`policy_gain_absent`**（学习型贪心策略 0.6637 对最强固定 0.83，gain −0.1663；见 [结果卡](RESULT-TAIJI_U2_22_6_2_POLICY_20261120.md)）。

## 1. 目的与预注册

在同一 32 world 准入池（D-20261119-24）上比较：`learnable`（22.6.1 预测器 greedy argmax）、`fixed_observe_t0`、`fixed_wait_deadline`（最强固定）、`reactive_stability`、`wait_control`；判据＝`learnable − 最强固定 ≥0.05`、world 配对单侧 95% 下界 `>0`、3 seed 同向。成本冻结观测 0.02／等待 0.01／求证 0.10；`0 check/final`。

## 2. 定价与出口

- 实测 `93.48 ≤ 帽 130 CPU`（原估 30／帽 90 低估，已在结果卡披露修正）。
- 出口顺序：`engineering_failure → not_admitted → budget_stop → policy_gain_absent → policy_gain_pass`。

## 3. 结果摘要

`fixed_wait_deadline 0.83` > `learnable 0.6637` > `fixed_observe_t0 0.6467` > `reactive_stability 0.6267` > `wait_control −0.15`；gain −0.1663（LB −0.1998，3 seed 全负）⇒ **`policy_gain_absent`**。判读与后继见[结果卡 §3–§5](RESULT-TAIJI_U2_22_6_2_POLICY_20261120.md)：贪心用离线收益预测不等于决策改进；后继应做闭环反馈策略学习或风险校准规则，不在同范围调参重开。
