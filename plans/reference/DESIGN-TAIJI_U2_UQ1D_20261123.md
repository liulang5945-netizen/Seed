# UQ-1d 预注册：多 seed 汇聚校准＋Clopper–Pearson 风险控制（有限样本；1 seed、≤300 CPU、零 check）

> [预注册] — 2026-11-23（D-20261123-313）。上游 [UQ-1c](RESULT-TAIJI_U2_UQ1C_20261123.md)（D-312 `uq1c_risk_unmet`）：Wilson 上界把 P24 测试风险从 0.170 收窄到 **0.139** 仍 >0.12；诊断＝**校准（seed 212，256）与测试（seed 712，256）在该覆盖下难度不同**，且置换对照实现缺陷（conf/correct 同置换）。本卡：① 校准/测试各 **1024**（4 个生成种子汇聚）以稳定分布；② **Clopper–Pearson 精确单侧 95% 上界**；③ 置换对照只打乱正确性（破坏与置信的关联）；④ 容忍 0.03（测试集抽样 ~1.5σ）。判据清单：[`taiji_u2_uq1d_criteria_20261123.json`](../manifests/taiji_u2_uq1d_criteria_20261123.json)（`criteria_sha256` 读前冻结）。**按常设授权执行，待 owner 回溯。**

## 1. 协议

- 主体/训练同 UQ-1（P24 500、P27 600、分块 48 维头、隔离、wd=0）；
- 校准 1024 = 4 个种子（P24 offsets 212–215，P27 offsets 81–84）；测试 1024（P24 offsets 712–715，P27 offsets 981–984）；
- 风险控制：按置信度降序前缀 k，e_k=错误数，CP 上界 UCB(k)=Beta⁻¹₀.₉₅(e_k+1, k−e_k)；取最大 k 使 UCB(k) ≤ α_t 且 k/n ≥0.30；τ=第 k 置信度；
- 任务条件 α：P24 0.10、P27 0.04；**容忍 0.03**；最小覆盖 0.40；
- 对照：置换校准（仅打乱正确性序列）、始终作答。

## 2. 判据（读前冻结）

| 门 | 判据 |
|---|---|
| G0 基线胜任 | P24 测试 acc ≥0.70 且 P27 ≥0.70（1024 测试） |
| G1 风险受控 | 修正策略测试风险 ≤ α_t+0.03（≤0.13 / ≤0.07） |
| G2 覆盖可用 | 覆盖率 ≥0.40（两任务） |
| G3 反馈因果 | 置换校准策略在两任务上均不满足「风险 ≤α_t+0.03 且覆盖 ≥0.40」 |
| G4 校准必要性 | 始终作答至少在一个任务上违反 α_t+0.03 |
| G5 身份/保持 | capsule 前向差 ≤1e-6；P24 在 P27 后 drop ≤0.001 |
| G6 预算/纪律 | 单 seed 训练、≤300 CPU、`criteria_sha256` 读前校验、零 check/final |

**出口**：全过⇒`uq1d_verified`；G1 或 G2 不过⇒`uq1d_risk_unmet`；G3 不过⇒`uq1d_feedback_weak`；G4 不过⇒`uq1d_calibration_optional`；G0 不过⇒`uq1d_baseline_insufficient`；超帽⇒`budget_stop`；工程异常⇒`engineering_failure`。**不调阈、不改门。**

## 3. 披露

- CP 界为精确单侧 95% 界（交换性假设：校准/测试同分布、独立）；多 seed 汇聚提高稳定性但不消除分布差异；
- 合成面；α 为实验口径；「求助」＝弃答动作；单训练 seed（校准/测试多 seed 数据）；
- 不回溯 UQ-1/UQ-1b/UQ-1c 判词；不签 24-S/29/final/C/D/J；零 check/final。
