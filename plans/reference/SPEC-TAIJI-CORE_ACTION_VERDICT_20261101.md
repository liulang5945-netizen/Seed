# 10.6：动作特定构造——是非题打分（不授予训练）

> 文档状态：[合同] — owner 已于 2026-11-01 裁定「试动作特定构造」。10.4 证明主体不读目标、10.5 证明读到的目标只是共模；本步把动作放进前缀，用是非题让分数在**结构上**依赖动作。**不训练。**

日期：2026-11-01。候选与 10.5 相同（09.8 主体），世界同 10.1b。

## 1. 构造（与前两步的差别）

- 10.4：`prefix = 目标`，`target = 动作句`，取绝对似然 ⇒ 字面先验主导。
- 10.5：`contrast = NLL(动作句|错配目标) − NLL(动作句|匹配目标)` ⇒ 目标读到了，但两句被推得一样多（共模）。
- **10.6**：`prefix = 局面 + 目标 + 动作句`，`target ∈ {是, 否}`；分数
  `verdict(action) = NLL(否 | prefix) − NLL(是 | prefix)`，
  越大表示模型越倾向"该动作能达成目标"。**动作在 prefix 里**，所以每个动作的前缀都不同。

## 2. 判据

1. **答案可分**：至少一个前缀上 `|NLL(是) − NLL(否)| > 1e-6`。若答案 token 在模型眼里完全等价，整个构造无信号 ⇒ `answer_tokens_indistinguishable`。
2. **动作可分**：同一局面、同一目标下，正确动作与另一动作的 `verdict` 差 ≥ `1e-3` ⇒ `action_specific_signal`；否则 `action_gap_below_floor`。
3. **方向**：正确动作的 verdict 更高 ⇒ 记为 `correct_action_preferred`（**只披露、不作门**：两个候选时方向由符号决定）。
4. **同候选**（父摘要等于 09.8 卡片）与**确定性**（重复调用逐位相同）为前置。
5. **零对照**：同一前缀配同一答案对两次，结果必须相同（差恰为 0）。

## 3. 预算与出口

- 预算：≤600 CPU 秒 / 1200 墙钟秒；超限 `budget_stop`（终点处判）。
- 出口：`action_specific_signal`／`action_gap_below_floor`／`answer_tokens_indistinguishable`／`candidate_identity_mismatch`／`nondeterministic_selection`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`taiji/action_verdict_scorer.py`、`tests/taiji_native/test_action_verdict_scorer.py`、`scripts/training/verify_taiji_action_verdict_10_6.py`、`reports/taiji_core_action_verdict_10_6_20261101.json`、MAINLINE/03 同步。
- **通过只表示**这个构造在该局面上能区分动作；**不表示选择正确、不表示任务能力、不表示 P2 满足**——P2 仍须在 10.4 的判据上赢过对照。
