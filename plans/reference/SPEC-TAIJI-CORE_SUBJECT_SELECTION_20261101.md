# 10.4：主体自身参与选择的第一个能力读数（不授予训练）

> 文档状态：[合同] — owner 已于 2026-11-01 批准。这是 11 的 P2 直路：让**模型本身**在 10.1b 环境上选动作，并如实报告它是赢是输。**不训练、不授予训练。**

日期：2026-11-01。`taiji/living_core.py` 字节保持不变。

## 1. 候选与选择机制（不使用动作头、不使用外部策略）

- **候选**：09.8 卡片的同一主体（R021 参考图 ＋ 共识出生 parts=5/units=3 ＋ 8 步 lr=1e-5），重建方式与其验证器逐位相同；报告记录父模型摘要以核对同一身份。
- **模型如何选**：对每个合法动作渲染一句中文（如「把蓝色杯子放到窗台」），把它与当前 Goal 文本拼成一条生涯（`prompt = Goal`、`target = 动作句`），用**主体自身的条件 NLL** 给该动作打分，取 NLL 最低者。选择只依赖主体参数与 Goal 文本，不读终态、不读评分、不使用贪心启发式。
- **对照**：`greedy`（10.3 的启发式）与 `random`（声明种子）各跑一次。

## 2. 判据

1. **同候选**：报告里的父模型摘要必须等于 09.8 卡片 `identity.parent_model_digest`；否则 `candidate_identity_mismatch`。
2. **确定性**：同一夹具重复调用选出的动作序列逐位相同；否则 `nondeterministic_selection`。
3. **参与而不是空转**：每次选择打分的动作数必须等于该状态的**合法动作数**（容量规则下为 2：每个对象只有一个空位可去），不得退化；否则 `selector_degenerate`。（初稿写死为 4，是忽略了 10.1b 的容量规则，已在执行前更正。）
4. **胜负如实报**：比较三者终态总分。主体 ≥ 随机 + 0.05 ⇒ `model_advantage_over_random`；主体 = 贪心 ⇒ `model_matches_greedy`；否则 `model_below_random`。**三种都是合法出口，不得只报对自己有利的那条。**
5. 报告 `model_involved: true`，并写清 `capability_passed` 是真是假。

## 3. 预算与出口

- 预算：≤600 CPU 秒 / 1200 墙钟秒；超限 `budget_stop`。
- 出口：`model_advantage_over_random`／`model_matches_greedy`／`model_below_random`／`candidate_identity_mismatch`／`nondeterministic_selection`／`selector_degenerate`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`taiji/subject_action_scorer.py`、`tests/taiji_native/test_subject_action_scorer.py`、`scripts/training/verify_taiji_subject_selection_10_4.py`、`reports/taiji_core_subject_selection_10_4_20261101.json`、MAINLINE/03 同步。
- **通过（即 `model_advantage_over_random` 或 `model_matches_greedy`）只表示**主体在这个 6 夹具的合成中文环境上的选择优于随机；**不表示**自然中文连续任务能力、不表示 L3 能力、不表示 11 准入自动成立（P1/P3 仍未解）。若判 `model_below_random`，那也是一份**有价值的负读数**，必须原样入账。
