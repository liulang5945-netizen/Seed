# 10.5：目标条件化选择接口与目标敏感性探针（不授予训练）

> 文档状态：[合同] — owner 已于 2026-11-01 裁定「先换目标条件化接口」。这是对 10.4 `model_below_random` 的直接回应：**先证明接口会随目标变化**，再谈强弱。**不训练。**

日期：2026-11-01。候选与 10.4 相同（09.8 主体），环境与夹具同 10.1b。

## 1. 接口（替换 10.4 的绝对似然）

对动作句 `s` 与两个目标文本 `g`（匹配）与 `h`（错配）：

```
contrast(s; g, h) = NLL(s | h) − NLL(s | g)
```

匹配目标让该动作句更可能时，`contrast > 0`。选择规则：取 `contrast` 最大的动作。

与 10.4 的关键差别：10.4 比的是**绝对似然**，被训练生涯的字面先验主导（19/19 全选同一对象）；这里比的是**同一句话在两个目标下的差**，字面先验在相减中互相抵消。

## 2. 目标敏感性探针（主判据）

取共享同一初态的一对目标：A＝「把蓝色杯子放到窗台」、B＝「把红色印章放到窗台」（容量规则下该初态只有一个空位，两个目标各有**一个合法且正确的动作**，且互不相同）。

判据经过两次更正（都在测量之前）：

- 初稿用「换目标后选择翻转」——**只有 2 个候选时 `argmax` 在任何非零差下都会翻转**，几乎恒真；
- 二稿改为「正确动作的 contrast 更高」——在对比度对目标反号的构造下，两个目标下的方向条件**互相等价**，仍是恒真。

因此本步**只判幅度**，方向只披露：

1. **对比度可分**：`|contrast(杯子→窗台) − contrast(印章→窗台)| ≥ 1e-3`。低于此 ⇒ `contrast_below_floor`。
2. **零对照**：把匹配与错配设成**同一文本**时，两动作的 contrast 必须**恰为 0**、gap 恰为 0；否则说明差值不只来自目标差异 ⇒ `nonzero_null`。
3. **方向披露**：哪个动作的 contrast 更高只入表、不作门——只有 2 个候选时它由符号唯一决定，判它没有信息量。
4. **同候选**与**确定性**为前置：父摘要等于 09.8 卡片、重复调用逐位相同。

## 3. 预算与出口

- 预算：≤600 CPU 秒 / 1200 墙钟秒；超限 `budget_stop`。
- 出口：`goal_conditioned_selection`（幅度与零对照都过）／`contrast_below_floor`／`nonzero_null`／`candidate_identity_mismatch`／`nondeterministic_selection`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`taiji/goal_conditioned_scorer.py`、`tests/taiji_native/test_goal_conditioned_scorer.py`、`scripts/training/verify_taiji_goal_conditioned_10_5.py`、`reports/taiji_core_goal_conditioned_10_5_20261101.json`、MAINLINE/03 同步。
- **通过只表示**这个接口在该对上会随目标改变选择、且对比度高于声明底座；**不表示**选择正确、不表示任务能力、不表示 P2 满足——P2 仍需在 10.4 的同一判据上赢过对照。
