# M5 P5.2c′ 未见组合迁移 Gate — 结果报告

> 执行日期：2026-09-13。报告提交前代码基线：`0d133676`（runner 与测试为本轮新增，见文末提交）。
> 预注册：[M5_P5_2C_PRIME_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md](../plans/reference/M5_P5_2C_PRIME_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md)（冻结，未修改）
> 机器报告：[taiji_p5_2c_prime_unseen_combination_transfer_20260913.json](taiji_p5_2c_prime_unseen_combination_transfer_20260913.json)
> 前身：[P5.2c 原预注册](../plans/reference/M5_P5_2C_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md)（`blocked_at_entry_audit`，**未修改、未覆写**）与[入场审计](M5_P5_2C_ENTRY_AUDIT_P5_2B_DEFECT_20260913.md)

## 1. 结论（三态）

**`transfer_no_gain`**

- 九门中 **8 门通过**，仅门 9 `transfer_and_budget` 未过。
- 机械门（1–7）全部通过 ⇒ 实验**有效执行**，结论是科学结论而非接线事故。
- 门 8 `prediction_binding_and_calibration` **通过**（符号一致率 `1.0` ≥ `0.5`；绝对误差中位数 `0.3417` ≤ `0.35`）。
- 门 9 未过原因：`object_gain = 0.0`，`required = 1.65`（最强对照 `strongest_singleton = 1.5` + `MARGIN 0.15`），**margin 未过**，且 `collaboration_holds = false`（4 个 holdout context 中 0 个优于最强单体）。

按预注册 §6，`transfer_no_gain` = 「门 1–7 全过、门 8 或门 9 未过——cohort 内无迁移收益或无预测校准」。本例为**无迁移收益**，且校准成立。

**`growth_admitted=false`、`can_promote=false` 贯穿，未改变。**

## 2. 入场审计（预注册 §3.2 与 §4 的硬条件）— 全部成立

| 断言 | 期望 | 实测 | 结论 |
|---|---|---|---|
| `P*` 不在 `observed_records` | 缺席 | **缺席** | ✅ |
| `P*` 的 `(T,T)` 在 train 出现次数 | `0` | **0** | ✅ |
| `P*` 的 `(T,T)` 在 holdout 出现次数 | `>0` | **8** | ✅ |
| 其余 5 对均在 `observed_records` | 全在 | **5/5 全在** | ✅ |
| `P*` 两成员在 train 的 singleton 证据 | 均 >0 | `member-a: 16`、`member-d: 16` | ✅ |
| 移除台账条目数 | `8 × 2 = 16` | **16** | ✅ |
| episode 数 | train 160 / holdout 88 | **160 / 88** | ✅ |
| 非 baseline 零步 cell | `0` | **0**（`interventions_happened=true`） | ✅ |

移除动作**只作用于证据构造层**：`P*` 的两个成员在 train 中各自保留 16 条 singleton 观测与全部 baseline 观测，因此该组合是**未见（unseen）**而非**无支持（unsupported）**。learner 拥有完整单体证据，缺的只是这一对的**联合**效应——这正是「未见组合」的定义。

## 3. 设计有效性核查

| 项 | 结果 |
|---|---|
| 单次判定为 `P*` | ✅ `selected_is_designated_unseen_pair = true` |
| 选择发生在评分之前 | ✅ `parent_checkpoint_digest` 在执行前冻结 |
| `P*` 的 5 对同侪在 `unseen_only=True` 下不可选 | ✅ `observed_pairs_not_reselectable = true` |
| `P*` 在 `unseen_only=True` 下可选 | ✅ `designated_pair_predictable = true`（两方向都钉住） |
| 全新进程恢复后选择逐位一致 | ✅ `selection_reproduced = true` |
| 反转观测顺序后选择与预测逐位一致 | ✅ `reverse_order_selection_matches = true` |
| replica 逐位一致 | ✅ `replica_consistent = true` |
| 拒绝项 7/7 成立 | ✅ 含 holdout 泄漏、篡改 digest、stale lineage、未知成员 |
| 安全违规 | **0** |
| wall | **21.5s** / 900s |

## 4. 主判据与对照（预注册 §3.5 / §3.6）

MARGIN = `0.15`（沿用 P5.2a 冻结常数，未改）。

| 对照 | mean_gain_vs_strongest_single |
|---|---|
| **object（`P*` = member-a+member-d）** | **0.000000** |
| C1 `no_learning` | −0.500000 |
| C2 `strongest_singleton`（逐 context 最强单体） | **+1.500000** |
| C3 `random_combination` | −0.500000 |
| C4 `fixed_combination` | +0.000000 |
| C5 `train_only_simple_regression` | +0.000000 |
| C6 `lesion_learner` | +0.000000 |

- **`required = 1.5 + 0.15 = 1.65`，object = 0.0 ⇒ margin 未过。**
- **C2 是决定性的**：逐 context 的最强单体收益（+1.5）远高于 pair（0.0）。即「单独用最强的那个成员」在 holdout 上比「两个成员一起用」更好——pair 相对最强单体**没有任何增益**。
- C4 / C5 / C6 与 object 同为 0.0，说明三者选中的都是等价最优（`P*` 或与之等价的 pair），**不构成对 object 的超越**。
- 按预注册 §3.5 已披露的代价，C4 因与正确答案重合被**排除在带 margin 的对照之外**（`margin_bearing_controls` 已列出：C1/C2/C3/C5/C6 + singleton）。

### 逐 context 分解（object）

| context | baseline | member-a | member-d | pair | realized_interaction | gain_vs_strongest_single | 最强单体 |
|---|---|---|---|---|---|---|---|
| `p52a-validation-108` | −1.0 | 1.0 | 1.0 | 1.0 | **−2.0** | 0.0 | member-a |
| `p52a-validation-109` | −1.0 | −1.0 | −1.0 | −1.0 | 0.0 | 0.0 | member-a |
| `p52a-validation-110` | −1.0 | −1.0 | −1.0 | −1.0 | 0.0 | 0.0 | member-a |
| `p52a-validation-111` | −1.0 | −1.0 | −1.0 | −1.0 | 0.0 | 0.0 | member-a |

- **4 个 context 中 0 个优于最强单体**（`collaboration_holds = false`）。
- `108` 的 `realized_interaction = −2.0` 是**负交互**：pair 与两个单体各自都成功（1.0 vs baseline −1.0），但按 factorial 公式 `pair − first − second + baseline = 1 − 1 − 1 + (−1) = −2`。这两个成员在 block-0 上做的是**同一件事**（互相冗余），因此联合使用没有额外收益——它们的贡献被重复计算了。
- `109/110/111` 三处 pair 与最强单体**完全相同**（`−1.0`），即 pair 未带来差异。

## 5. 预测校准（门 8）— 通过，但须标注样本量

| pair | 在 train 中被观测 | 预测 interaction | 实测 interaction | 绝对误差 | 符号一致 |
|---|---|---|---|---|---|
| `member-a+member-b` | 是 | — (不可预测) | −0.5 | — | — |
| `member-a+member-c` | 是 | — | −0.5 | — | — |
| **`member-a+member-d`（`P*`）** | **否** | **−0.15833** | **−0.5** | **0.34167** | **是** |
| `member-b+member-c` | 是 | — | 0.0 | — | — |
| `member-b+member-d` | 是 | — | 0.0 | — | — |
| `member-c+member-d` | 是 | — | −0.5 | — | — |

- pooled `sign_match_rate = 1.0`、`median_absolute_error = 0.3417` ⇒ **门 8 通过**。
- **必须标注的限制**：未见组合面基数 = 1，故上述统计**只建立在 1 个样本上**（`comparable_pair_count = 1`），不是独立样本量，不做显著性主张。报告已在 `calibration.small_sample_caveat` 显式记录。
- 值得注意的是 learner 的预测方向是**正确的**：它预测 `P*` 有**负**交互（−0.158），实测亦为负（−0.5）。即 learner 在一定程度上**预见到这对组合不会产生协同**——但由于 margin 判据要求的是「优于最强单体」，方向正确并不足以通过。

## 6. 根因分析：为什么没有迁移收益

报告提供了三组**互相独立**的证据，指向同一个结构性解释。

### 6.1 四个成员 profile 的 contribution 完全相同

```
member-a: 0.5   member-b: 0.5   member-c: 0.5   member-d: 0.5
```

- 4 个成员在 profile 维度上**完全没有区分度**。
- 后果一：C5（`train_only_simple_regression`，按 profile 贡献之和对 pair 排序）在**所有 pair 上给出相同分数**，退化为按字典序取第一个 → 与 `no_learning` 同类，不具备选择能力。
- 后果二：learner 拟合的交互项**没有可用的成员级特征**去外推到未见组合。它能利用的只有「成员身份」这一离散标识，而 `P*` 的身份在训练中从未与联合结果共现过。于是它对 `P*` 的预测只能回落到其他 5 对的平均交互水平附近（−0.158 是 5 个已观测 pair 交互值的某种收缩估计）。

### 6.2 成功矩阵显示成员按 block 高度专门化，且组合只是单体的并集

| cell | block-0 | block-1 | block-2 | block-3 | 合计 |
|---|---|---|---|---|---|
| `none`（baseline） | 0/6 | 0/6 | 0/6 | 0/6 | **0/24** |
| `member-a` | **6/6** | 0/6 | 0/6 | 0/6 | 6/24 |
| `member-b` | 0/6 | **6/6** | 0/6 | 0/6 | 6/24 |
| `member-c` | 0/6 | 0/6 | **6/6** | 0/6 | 6/24 |
| `member-d` | **6/6** | 0/6 | 0/6 | 0/6 | 6/24 |
| `member-a+member-b` | 6/6 | 0/6 | 0/6 | 0/6 | 6/24 |
| `member-a+member-c` | 6/6 | 0/6 | 0/6 | 0/6 | 6/24 |
| **`member-a+member-d`（`P*`）** | **6/6** | 0/6 | 0/6 | 0/6 | **6/24** |
| `member-b+member-c` | 0/6 | 6/6 | 6/6 | 0/6 | 12/24 |
| `member-b+member-d` | 6/6 | 6/6 | 0/6 | 0/6 | 12/24 |
| `member-c+member-d` | 0/6 | 0/6 | 6/6 | 0/6 | 6/24 |

关键读数：

1. **baseline 全 0**（`0/24`），确认任务确实需要成员介入——没有伪成功。
2. **`member-a` 与 `member-d` 的成功面完全相同**（都是 block-0 的 `6/6`，其余全 0）。`P*` 的联合面（`6/6`）**没有超出任何一个单体**。这正是 §4 里 `gain_vs_strongest_single = 0.0` 与 `108` 的负交互（−2.0）的来源：**`member-a` 与 `member-d` 在功能上冗余**。
3. **互补性由能力面的「不重叠」决定，而非由成员身份决定**。矩阵中「联合 > 任一单体」的 pair 有**两个**：
   - `member-b+member-c` = `[0,6,6,0]` → `12/24`（b 覆盖 block-1，c 覆盖 block-2，互不重叠）
   - `member-b+member-d` = `[6,6,0,0]` → `12/24`（d 覆盖 block-0，b 覆盖 block-1，互不重叠）

   二者是**同一个结构事实的两个实例**：`member-d` 的能力面与 `member-a` **完全相同**（均只覆盖 block-0），因此「d 与 b 互补」等价于「a 与 b 互补」。反过来说，`member-a+member-d` 之所以冗余，正是因为它的两个成员**共享同一能力面** `{block-0}`。

4. **block-3 全部为 0/6**，包括 baseline 与所有组合（此前 P5.2b 已记录其原因为 `contract_intercepted:language_evidence_ambiguous`，属合同层合理拦截）。即 block-3 是**不可达任务面**，对任何组合都无差异。

### 6.2b 对 §6.1 的更正（归因的关键一环）

上文曾把互补 pair 描述为「唯一」的 `b+c`，这是**不准确的缩写**，此处更正：互补 pair 有**两个**（`b+c`、`b+d`），它们的共同点是**成员能力面不重叠**。

这处更正改变了归因的落点。真正的问题不是「互补 pair 恰好已被观测」，而是：

> **能力面信息在 learner 的 profile 里完全丢失。**

4 个成员的 `contribution` 全部等于 `0.5`，即在 learner 看来 **4 个成员是不可区分的**。它无从得知：

- `member-a` 与 `member-d` 的能力面**相同**（故二者组合必然冗余）；
- `member-b`、`member-c` 与它们的能力面**不重叠**（故与任一个组合都互补）。

learner 唯一能利用的是「成员身份」这一离散标识，而 `P*` 的身份在训练中从未与联合结果共现。于是它对 `P*` 的预测只能回落到已观测 5 对的平均交互附近——这与实测 `−0.158` vs `−0.5` 的「方向正确但幅度收缩」完全吻合。

**结论**：本 gate 的 `transfer_no_gain` 不能读作「该 cohort 无联合增益潜力」。矩阵明确显示存在真实互补（`12/24` vs 单体 `6/24`）。失败发生在**表征层**——profile 没有携带做出该判断所需的变量。

### 6.3 家族覆盖列联表

```
positive_gain_rows = 24 / 72      positive_by_block  = {0: 12, 1: 6, 2: 6}
nonpositive        = 48 / 72      nonpositive_by_blk = {0: 6, 1: 12, 2: 12, 3: 18}
```

- 正增益只出现在 block-0/1/2；block-3 的 18 行全为非正（任务不可达）。
- block-0 的分布**不均衡**（12 正 / 6 非正）：因为 `member-a` 与 `member-d` 都覆盖 block-0，凡是含 a 或 d 的 pair 都得正增益——这是**冗余**而非**互补**。

### 6.4 综合判断

| 层次 | 结论 |
|---|---|
| **直接原因** | `P*` 在 holdout 上的 `realized_interaction = −0.5`（4 context 平均），相对最强单体增益 `0.0`，远未达到 `1.65` 的门槛 |
| **机制原因** | `P*` 的两个成员**共享同一能力面**（`{block-0}`），联合不产生超额收益 |
| **表征原因（关键）** | 4 个成员 profile 的 `contribution` **全等 `0.5`**，learner 无法区分成员，因而无法判断哪两个成员的能力面**不重叠**。而「不重叠」正是本 cohort 中联合增益的**唯一来源**（见 §6.2 第 3 条：`b+c` 与 `b+d` 均达 `12/24`） |
| **设计层含义** | 移除动作按**结构**（字典序索引 `(0,3)`）选取未见 pair，未按能力面互补性选取，故落到一对能力面相同的成员上。但这属于**抽样位置**问题；即使换到互补 pair，learner 也缺乏识别它的特征——§6.2b 已证 |
| **不应得出的结论** | **不可**读作「该 cohort 无联合增益潜力」。矩阵显示互补性真实存在（`12/24` vs 单体 `6/24`）。失败发生在**表征层与设计层，不在能力层** |

## 7. 这次实验证明了什么、没证明什么

**证明了**：

- 「从 train 全分区移除指定 pair 的联合 cell」是构造非空未见组合面的**有效且干净**的机制（预注册 §2 的实证在本轮再次复现）。
- learner 在只缺联合证据、单体证据完整时，能给出**方向正确**的预测（预测 −0.158 vs 实测 −0.5），且通过门 8 的校准阈值（在 1 个样本上）。
- P5.2b 的三层修复（判据 / 任务 / 绑定+重置）在本 gate 中持续有效：`zero_step = 0`、baseline 全 0、合同路径全部真实执行。

**没有证明**（不可外推）：

- **不能**声称 learner 具备「在多个未见组合中选择」的能力——候选面基数为 1（预注册 §7 已预先披露此代价）。
- **不能**声称「未见组合迁移」在一般意义上不可行——本轮只检验了 **1 个** 未见组合，且它恰为冗余 pair。
- **不能**把 `transfer_no_gain` 解释为「组合学习无效」——矩阵显示 `member-b+member-c` 有真实互补性（12/24 > 任一单体 6/24），只是它已被观测。

## 8. 停止点判定（预注册 §7）

命中条目：**`transfer_no_gain` → 回模型/特征归因（profile 维度、pair 关系项形式、不确定性口径），需新预注册再试；本 gate 不因归因结论被追认为通过。**

具体归因方向（须在新预注册中冻结后才能执行）：

1. **profile 维度无区分度**是首要嫌疑：4 个成员 contribution 全等 0.5，learner 没有成员级特征可外推。应检验「profile 是否需要携带能力面信息（如该成员覆盖哪些 block/cell 类型）」。
2. **pair 关系项的形式**：当前 `_pair_features` 只支持 pair 的成员身份组合，无法表达「互补 vs 冗余」这一区分——而 §6 表明这才是真正决定联合增益的结构。
3. **未见组合的选取原则**：本轮按结构索引选取，结果落到冗余 pair 上。若新预注册要保留「选择能力」的检验力，应同时满足：(a) 未见组合数 ≥2；(b) 未见 pair 中包含至少 1 对具有互补潜力的组合。这需要更大的成员集与整体移除的配合（成本更高，预注册 §7 已注明）。
4. **block-3 不可达**：`contract_intercepted:language_evidence_ambiguous` 使 1/4 的 context 对所有组合都是 0——有效区分面只有 3/4。若要在更高功效下重测，应先解决 block-3 的任务可达性（这可能触及 P5.2a 任务定义，需独立预注册）。

## 9. 纪律遵守声明

- 未修改、未覆写 P5.2c 原预注册与其 `blocked_at_entry_audit` 报告。
- 未放宽任何门或阈值；门 8 阈值 `0.5`/`0.35` 与 MARGIN `0.15` 均为预注册冻结值，本轮未调整。
- 未重跑挑结果；本报告为首次执行结果。
- 未把 `transfer_no_gain` 改绿为通过，也未创建 P5.2c′ 之外的第二份报告路径。
- `growth_admitted=false`、`can_promote=false` 贯穿。
- 临时探针脚本（`scripts/probe_p52cp_surface.py`）用毕即删。
- 语义映射只存在于 runner 配置节，未进入 learner/evaluator 输入；报告不含语义→成员对应表。

## 10. 产物

| 文件 | 说明 |
|---|---|
| `scripts/training/eval_taiji_p5_2c_prime_unseen_combination_transfer_gate.py` | 新 runner（门 9 未过，如实落 `transfer_no_gain`） |
| `tests/taiji_native/test_p5_2c_prime_unseen_combination_gate.py` | 回归守卫 **9 passed**：钉住「部分移除无效 / 全分区移除有效」「移除仅限联合 cell，保留边缘证据」「entry audit 三向 fail-closed」「阈值字面量冻结」「C4 与答案重合且被排除出 margin」 |
| `reports/taiji_p5_2c_prime_unseen_combination_transfer_20260913.json` | 机器报告 |
| 本文件 | 结果分析 |
