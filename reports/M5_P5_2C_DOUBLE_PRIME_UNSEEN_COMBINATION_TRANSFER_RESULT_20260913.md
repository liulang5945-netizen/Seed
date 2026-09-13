# P5.2c″ 未见组合迁移 — 结果报告

- **格式**: `taiji-p5-2c-double-prime-unseen-combination-transfer-report-v1`
- **预注册（冻结）**: `plans/reference/M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md`
- **runner**: `scripts/training/eval_taiji_p5_2c_double_prime_unseen_combination_transfer_gate.py`
- **原始报告**: `reports/taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json`
- **回归测试**: `tests/taiji_native/test_p5_2c_double_prime_unseen_combination_gate.py`（14 用例）
- **路线**: C（设计修复：两个未见组合，均经实测验证能力面不相交）
- **墙钟**: 22.293 s / 上限 900 s

---

## 1. 结论

**`outcome = transfer_signal_constant`**（三态之一，非 `failed`、非 `transfer_no_gain`、非 `unseen_combination_transfer_supported`）

| 项 | 值 |
|---|---|
| `status` | `completed` |
| `outcome` | `transfer_signal_constant` |
| `experiment_passed` | `false` |
| `growth_admitted` | `false` |
| `can_promote` | `false` |
| 九门 | **8 过 / 1 未过**（仅 `transfer_and_budget`） |

`transfer_signal_constant` 的判据（预注册 §6 预先固定）：机械门全过、增益门未过，**且两对未见组合均不产生正的场景增益**（`any_held_out_pair_positive = false`）。这与 `transfer_no_gain` 的区别是实质性的：

- `transfer_no_gain`（P5.2c′）：**唯一一个**未见组合无增益 ⇒ 样本量 1，无法区分「该组合无增益」与「迁移整体无信号」。
- `transfer_signal_constant`（本轮）：**两个**未见组合**都**无增益 ⇒ 信号在两个独立结构位置上表现一致，更像「当前表征下的常量」而非「单点噪声」。

---

## 2. 本轮只修测量仪器 —— 边界声明（不得省读）

本轮是**路线 C**，即**实验设计修复**。它**不**触碰 profile 表征、**不**触碰 pair 关系项形式（分别为路线 A 与路线 B）。

因此：

- 本轮的**负结果只能归因于「该表征下不可行」**，**不能**归因于「迁移不可行」。
- 若本轮是正结果，也**不得**声称「表征问题已解决」。

**表征缺陷仍在的直接证据**（本轮实测，非引用）：

```json
"member_evidence": {
  "profiles": 4,
  "contributions": {
    "member-a": 0.5, "member-b": 0.5, "member-c": 0.5, "member-d": 0.5
  },
  "contribution_uniform": true
}
```

四个成员的 contribution **完全相等**，learner 无法区分成员。这正是路线 A 的靶点。在 contribution 均等的条件下，任何「组合优于单体」的证据都极难出现 —— 因为 learner 没有理由偏好任何成员，预测的 `predicted_interaction` 也只能落在同一常量附近（本轮为 `-0.09375`）。

---

## 3. 路线 C 的三项目标达成情况

| # | 目标 | 结果 | 证据 |
|---|---|---|---|
| 1 | 未见组合数 ≥ 2 | **达成** | `unseen_sample_count = 2`（P5.2c′ 为 1） |
| 2 | 至少 1 对具互补潜力 | **达成（2 对）** | `held_out_pair_kinds = ["DISJOINT", "DISJOINT"]` |
| 3 | 解决 / 量化 block-3 不可达 | **达成（如实量化，未隐藏）** | `block3_uniformly_unreachable = true`，`discriminating_fraction = 0.75` |

### 3.1 目标 2 —— 能力面实测与互补性分类

能力面在**本轮由真实执行测得**，不引用设计文档旧值（报告内已带 `note` 字段声明这一点）：

```
member-a = {0}      member-b = {1}      member-c = {2}      member-d = {0}
```

block ↔ 模板为精确对应：block-0=`lang_confirm`、block-1=`patch_persist`、block-2=`create_persist`、block-3=`header_override`。

6 对完整分类：

| pair | union | overlap | kind | 是否持有 |
|---|---|---|---|---|
| `a+b` | [0,1] | — | `DISJOINT` | 否（观测） |
| **`a+c`** | [0,2] | — | **`DISJOINT`** | **是（P₁\*）** |
| `a+d` | [0] | [0] | **`EQUAL`** | 否（观测） |
| **`b+c`** | [1,2] | — | **`DISJOINT`** | 否（观测） |
| **`b+d`** | [0,1] | — | **`DISJOINT`** | **是（P₂\*）** |
| `c+d` | [0,2] | — | `DISJOINT` | 否（观测） |

**6 对中 5 对 DISJOINT，仅 `a+d` 为 EQUAL。** 关键设计决策：

- 持有 **`a+c`** 与 **`b+d`**，二者均实测 DISJOINT；
- **刻意把冗余对 `a+d` 留在观测面内**，使 learner 能观察到「某些组合不产生超额收益」，而不是只看到互补对 —— 否则 learner 会学到「任意组合都该有增益」的错映射。

这正是 P5.2c′ 的设计缺陷所在：它按字典序索引 `(0,3)` 持有，恰好落在**唯一**的冗余对 `a+d` 上，用一个结构性冗余的组合去检验「组合迁移能力」，结论自然失真。

### 3.2 目标 3 —— block-3 如实量化

```json
"block3_audit": {
  "block": 3,
  "contexts": ["p52a-validation-103", "p52a-validation-107", "p52a-validation-111"],
  "episodes": 66,
  "successes": 0,
  "success_rate": 0.0,
  "stop_reason_classes": { "all_members_exhausted": 6, "contract_intercepted": 60 },
  "uniformly_unreachable": true,
  "discriminating_contexts": 9,
  "total_contexts": 12,
  "discriminating_fraction": 0.75
}
```

`header_override` 模板对**所有** cell 都不可达（66 次尝试 0 成功，其中 60 次被 `contract_intercepted` 拦截 —— 即被上层契约主动拒绝，而非能力不足）。**有效区分面只有 9/12 个 context（0.75）**。

本轮**没有**修复这个问题（修它需要改契约层，属另一议题），但**把它量化并显式披露**，使后续所有 P5.x 结论都能带上这个折算因子。

---

## 4. 证据构造（唯一结构性改动）

| 项 | P5.2c′ | **P5.2c″（本轮）** |
|---|---|---|
| train episodes | 176 | **144** |
| holdout episodes | 88 | **88（不变）** |
| 移除 episode 数 | 16 | **32** |
| 持有 pair 数 | 1 | **2** |
| 观测 pair 数 | 5 | **4** |
| 候选 pair 数 | 6 | **6（不变）** |

移除口径：**两对持有组合的联合 `(T,T)` cell 从全部 8 个 train context 中整体移除**。计算结果 `8 × 2 × 2 = 32`，与报告 `removed_episode_count = 32` 一致。

**为什么必须整体移除**：`_estimate_pair` 只要有**任一** train context 四 cell 齐备即返回估计。部分移除无效 —— 这是一个机械事实，已由回归测试 `test_removal_covers_whole_train_partition_for_both_pairs` 显式固化为断言。

**「未见」而非「无支持」**：两对持有组合的每个成员在 train 中仍各保留 **16 条** singleton 证据。

```json
"held_out_pairs": {
  "member-a+member-c": {
    "in_observed_records": false, "joint_count_in_train": 0, "joint_count_in_holdout": 8,
    "singleton_counts_in_train": { "member-a": 16, "member-c": 16 }
  },
  "member-b+member-d": {
    "in_observed_records": false, "joint_count_in_train": 0, "joint_count_in_holdout": 8,
    "singleton_counts_in_train": { "member-b": 16, "member-d": 16 }
  }
}
```

入场审计 `passed = true`，`conditions = []`（零违规）。

---

## 5. 为什么拒绝「扩大成员集」

roadmap 早先计划包含「增加成员数」以制造更多未见组合。本轮**明确拒绝**该路线，理由为实证：

1. `maximum_pairwise_candidates = 32` ⇒ `C(n,2) ≤ 32` ⇒ **n ≤ 8**，理论天花板存在但远未触及。
2. **P5.2a 只有 4 个 train 模板族**（`lang_confirm` / `patch_undo` / `create_undo` / `header_override`），每族 10 个 task，且**每族对应一个成员**。第 5/6 成员**无新族可专精**，只会复制既有能力面 ⇒ **纯增冗余**。
3. 互补性**已经存在**：6 对中 5 对 DISJOINT。缺的不是成员，而是**实验设计**。

runner 内以常量 `member_set_widening_rejected_because` 显式记录该决策，回归测试 `test_does_not_widen_member_set` 断言成员数恒为 4。

---

## 6. 九门逐项

| # | 门 | 结果 | 关键量 |
|---|---|---|---|
| 1 | `static_checks` | ✅ | ruff 全仓通过（`All checks passed!`）+ py_compile |
| 2 | `evidence_admissibility` | ✅ | 入场审计通过，`conditions = []` |
| 3 | `unseen_combination_identity` | ✅ | **两对**均 `in_observed_records=false`、`joint_count_in_train=0`、`joint_count_in_holdout=8` |
| 4 | `unseen_context_holdout` | ✅ | 4 个未见 context 严格隔离 |
| 5 | `label_opaqueness` | ✅ | 持有由不透明索引 `(0,2)`/`(1,3)` 决定 |
| 6 | `real_execution` | ✅ | 真实调用，`interventions_happened=true` |
| 7 | `rejection_recovery` | ✅ | 9 项全 true（详见 §7） |
| 8 | `prediction_binding_and_calibration` | ✅ | 2 样本，符号一致率 `1.0`，绝对误差中位数 `0.25` |
| 9 | **`transfer_and_budget`** | ❌ | **`object_gain = -0.5`** vs `required = 1.65` |

门 9 的判据（预注册 §3.6 已扩展）要求**四项同时成立**：

```
margin_cleared              = false   ← 未达
collaboration_holds         = false   ← 未达
any_held_out_positive       = false   ← 未达（本轮新增要求）
replica_consistent          = true
total_wall <= 900s          = true    (22.3s)
```

---

## 7. 对照与预测面

### 7.1 对照摘要

```json
"control_summary": {
  "strongest_control": "strongest_singleton",
  "strongest_control_gain": 1.5,
  "object_gain": -0.5,
  "margin": 0.15,
  "required": 1.65,
  "margin_cleared": false,
  "collaboration_holds": false,
  "any_held_out_pair_positive": false,
  "margin_bearing_controls": [
    "lesion_learner", "no_learning", "random_combination",
    "strongest_singleton", "train_only_simple_regression"
  ]
}
```

**最强的对手仍然是单体**：逐 context 取最强 singleton，平均增益 `1.5`；而组合的增益是 `-0.5`。这意味着**组合不但没有协同，比单体还更差** —— learner 选出的组合在两个 context 上输给单体，结论不是「打平」，而是「负向」。

### 7.2 两对持有组合的实测

| pair | kind | 场景增益 | 平均已实现交互 | 优于最强单体的 context 数 | 是否被选为对象 |
|---|---|---|---|---|---|
| `member-a+member-c` | `DISJOINT` | **-0.5** | -0.5 | **0 / 4** | 是（P₁\*） |
| `member-b+member-d` | `DISJOINT` | **0.0** | 0.0 | **0 / 4** | 否 |

两对**都没有**在任何 context 上超过最强单体。`b+d` 恰好打平（0.0），`a+c` 反而是负的。

### 7.3 预测绑定

learner 在打分**之前**已完成绑定（`bound_before_scoring = true`）：

```json
"selected_member_ids": ["member-a", "member-c"],
"selected_is_a_held_out_pair": true,
"predicted_interaction": -0.09375,
"uncertainty": 0.5609,
"support": 4
```

**预测方向是对的但幅度严重不足**：预测 `-0.09375`，实测 `-0.5`。即 learner「知道」这对组合不会有好结果，但**严重低估了负面程度**（绝对误差 `0.40625`）。这与 §2 的贡献均等缺陷一致：learner 只能给所有组合一个接近常量的预测。

### 7.4 校准（门 8）

```json
"calibration": {
  "unseen_sample_count": 2,
  "comparable_pair_count": 2,
  "median_absolute_error": 0.25,
  "sign_match_rate": 1.0,
  "small_sample_caveat": "the unseen surface has cardinality 2 by construction;
                          these pooled statistics rest on those two pairs and
                          are not an independent sample",
  "thresholds": { "minimum_sign_match_rate": 0.5,
                  "maximum_median_absolute_error": 0.35,
                  "fixed_at_freeze_time": true }
}
```

| 指标 | 阈值 | P5.2c′ | **P5.2c″** | 判定 |
|---|---|---|---|---|
| 符号一致率 | ≥ 0.5 | 1.0 | **1.0** | 过 |
| 绝对误差中位数 | ≤ 0.35 | 0.3417 | **0.25** | 过（改善） |
| 未见样本数 | — | 1 | **2** | 改善 |

**门 8 虽过，但样本量仍只有 2**，报告内已带 `small_sample_caveat` 明确声明这不是独立样本。不得把门 8 的通过读作「校准已被验证」。

---

## 8. 拒绝 / 恢复探针

```json
"rejections": {
  "holdout_record_rejected": true,
  "terminal_record_rejected": true,
  "tampered_checkpoint_rejected": true,
  "stale_lineage_rejected": true,
  "unknown_member_fails_closed": true,
  "observed_pairs_not_reselectable": true,
  "held_out_pair_predictable_P1": true,
  "held_out_pair_predictable_P2": true,
  "all_held_out_pairs_predictable": true
}
```

**双向断言**（本轮强化）：既要求**观测对不可被重选**，也要求**两个持有对都可被预测**。只查一个方向会接受「探针失效但恰好通过」的中间状态。

恢复探针（新进程重载）：

```json
"recovery": {
  "fresh_process": true, "returncode": 0,
  "restored_selection": { "group_id": "transfer-group:a19bf62a1bb7e730ca3f99f6",
                          "member_ids": ["member-a","member-c"],
                          "predicted_interaction": -0.09375 },
  "in_process_selection": { /* 完全一致 */ },
  "expected_pair": ["member-a","member-c"],
  "selection_reproduced": true
}
```

自一致性 `reverse_order_selection_matches = true`；复本一致 `replica_consistent = true`。

---

## 9. 已知代价（预注册 §7 如实记录）

1. **`P₁*` 是结构性指定的**，因此对照 C4（正确组合）与「正确答案」重合，已从 margin 计算中**排除**，避免自我认证。`fixed_combination_gain_excluded_from_margin = -0.5` 单列披露。
2. **候选面仍为 6 对、观测面退化为 4 对**，learner 的观测多样性下降。
3. **block-3 仍不可达**，区分面 `0.75`，所有增益统计都应按此折算理解。
4. **恢复效应未测量**，记为 0，**未编造数据**（`recovery_effect_disclosure`）。

---

## 10. 对「先 C 后 A」决策的验证

用户决策为「先 C 后 AB」。本轮结论对这次排期给出了明确支持：

| 维度 | P5.2c′ | P5.2c″ | 说明 |
|---|---|---|---|
| 未见组合数 | 1 | **2** | 仪器修复有效 |
| 持有对性质 | `EQUAL`（唯一冗余对） | **`DISJOINT` × 2** | 设计缺陷已消除 |
| 未见样本校准 | 单样本 | **2 样本** | 结论稳健性上升 |
| 结论 | `transfer_no_gain` | **`transfer_signal_constant`** | **结论更精确** |
| block-3 | 未量化 | **已量化（0.75）** | 后续结论可带折算因子 |
| 表征缺陷 | 已发现 | **仍在（`contribution_uniform`）** | 归路线 A |

**关键点**：修好仪器后，结论从「1 个样本的 no_gain」变成「2 个样本的 constant」。如果先做 A 再做 C，我们将**无法判断** A 的收益是来自表征修复还是来自仪器修复 —— 两个变量混在同一轮，归因不可分离。先 C 后 A 保住了这个可分离性。

**同时必须诚实指出**：`transfer_signal_constant` **比 `transfer_no_gain` 更负面，不是更正面**。两个互补组合都拿不到增益，说明障碍不在「组合选得不好」，而在更底层。这为路线 A 提供了更强的动机，但也意味着**路线 A 的预期不应被抬高**。

---

## 11. 唯一下一步

进入 **路线 A：profile 表征修复**。

- 靶点：`contribution_uniform = true` —— 四个成员 contribution 全为 `0.5`，learner 无法区分成员。
- 硬约束：**必须先新预注册**（不得沿用本预注册的任何判据）；`growth_admitted=false`、`can_promote=false` 贯穿。
- 不得反向改判据、放宽判据凑通过、覆写本报告、重跑挑结果。
- 路线 B（pair 关系项形式）在路线 A 出结果后再定。

**归因纪律**：本轮结论**只**支持「该表征下两个互补组合均无正增益」；**不**支持「迁移不可行」。

---

## 12. 复现

```bash
# 全仓静态检查
python -m ruff check .

# 回归测试（35 用例：14 + 9 + 12）
python -m pytest tests/taiji_native/test_p5_2c_double_prime_unseen_combination_gate.py \
                 tests/taiji_native/test_p5_2c_prime_unseen_combination_gate.py \
                 tests/taiji_native/test_intervention_reality_gate.py -q

# 重跑 gate（约 22 s）
python scripts/training/eval_taiji_p5_2c_double_prime_unseen_combination_transfer_gate.py
```

已核验：在同一工作区连续两次执行，`outcome`、`gates_failed`、两对增益、校准值**完全一致**（`elapsed_seconds` 22.293 s → 22.707 s 属正常抖动）。
