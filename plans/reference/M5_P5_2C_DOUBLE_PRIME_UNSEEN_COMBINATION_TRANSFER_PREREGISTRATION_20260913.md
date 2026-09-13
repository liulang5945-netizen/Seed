# M5 P5.2c″ — 设计修复版未见组合迁移 Gate 预注册（冻结版，路线 C）

> 冻结日期：2026-09-13。状态：**冻结，待执行**。
> 决策依据：[P5.2c′ 下一步决策](../active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md) —— 用户决策「**先 C 后 A B**」。
> 取代关系：本文**不修改、不追溯改写** P5.2c 原预注册（`blocked_at_entry_audit`）、其入场审计、
> P5.2b 报告，以及 P5.2c′ 预注册与结果（`transfer_no_gain`）。以上全部保持冻结原貌、继续有效。
> 本文为**新预注册**：P5.2c′ 的停止点判定要求 `transfer_no_gain` 后回归因并另立新预注册。

## §0 本文与路线 C 的关系（先读，避免误读定位）

**本文只修「测量仪器」，不修表征。** 路线 C 的产出是**一个可信的实验装置**，不是关于迁移能力的结论。
若本 gate 得到正结果，**不得**据此声称表征问题已解决；若得到负结果，**不得**据此声称迁移不可能。
表征修复属路线 A，须在本 gate 之后另行预注册。

路线 C 的三个既定修复目标（来自决策文档 §2 路线 C）：

| # | P5.2c′ 的设计缺陷 | 本文的修复 |
|---|---|---|
| 1 | 未见组合数 = 1，门 8 校准只建在单样本上 | **持有 2 个 pair**（各 1 个 disjoint） |
| 2 | 未见 pair 按字典序索引选取，落到**能力面相同**的冗余 pair（`a+d`） | **显式选取 2 个能力面不重叠的 pair**，并以实证断言其不重叠 |
| 3 | block-3 对全部组合不可达（`contract_intercepted`），有效区分面仅 3/4 | **在引入前先量化**，并在门中区分「不可达」与「无增益」 |

## §1 研究问题

**当某两个认知成员的联合效应从未在训练证据中被估计过时，learner 能否在未见 context 上，仅凭成员 profile 与已观测的其他组合，预测出该组合的真实联合增益，并显著优于全部冻结对？**

相对 P5.2c′ 的加强：**同一 gate 内持有 ≥2 个未见组合，其中至少 1 对的能力面互不重叠**，使「预测组合价值」不再由单一样本决定。

## §2 机制实证（本轮量化，权威来源 = 本文所附探针证据）

沿用 P5.2c′ 已实证的机制与根因（`_estimate_pair` 只需任一 train context 齐备四 cell 即估计；
「增加成员数」无效——4/5/6 成员实测 unseen 均为 0）。本文**不重复论证该部分**，只补充路线 C 的新实证。

### 2.1 成员能力面（真实执行实测，非推断）

以 12 个 validation context、`block = task_index % 4` 分组（block 与模板**精确对应**：
block-0=`lang_confirm`、block-1=`patch_persist`、block-2=`create_persist`、block-3=`header_override`），
4 个成员的 singleton 成功面实测为：

| 成员 | block-0 | block-1 | block-2 | block-3 | 能力面 |
|---|---|---|---|---|---|
| `member-a` | **3/3** | 0/3 | 0/3 | 0/3 | `{0}` |
| `member-b` | 0/3 | **3/3** | 0/3 | 0/3 | `{1}` |
| `member-c` | 0/3 | 0/3 | **3/3** | 0/3 | `{2}` |
| `member-d` | **3/3** | 0/3 | 0/3 | 0/3 | `{0}` |

**`member-d` 的能力面与 `member-a` 完全相同（均为 `{0}`）。** 这解释了 P5.2c′ 的全部现象：
`a+d` 必然冗余，而 `a+b`/`a+c`/`b+c`/`b+d`/`c+d` 五对能力面均不重叠。

### 2.2 6 个 pair 的互补性分类

| pair | 能力面并/交 | 类型 |
|---|---|---|
| `a+b` | `{0,1}` / `∅` | **DISJOINT（互补）** |
| `a+c` | `{0,2}` / `∅` | **DISJOINT（互补）** |
| `a+d` | `{0}` / `{0}` | **EQUAL（冗余）** |
| `b+c` | `{1,2}` / `∅` | **DISJOINT（互补）** |
| `b+d` | `{0,1}` / `∅` | **DISJOINT（互补）** |
| `c+d` | `{0,2}` / `∅` | **DISJOINT（互补）** |

→ **6 对中 5 对互补，仅 `a+d` 冗余**。P5.2c′ 恰好持有未见的 `a+d`，是**抽样位置**问题。

### 2.3 为什么**不**扩大成员集（关键设计判断，与 roadmap 早先计划相反）

P5.2a 只有 **4 个 train 模板族**（`lang_confirm`/`patch_undo`/`create_undo`/`header_override`），
每个成员专精一族。因此：

- 新增第 5/6 个成员**没有新的族可专精**，只能复制现有能力面 → **增加冗余，不增加互补**；
- 而本 cohort 的互补性**已由现有 4 成员提供**（5/6 对互补）。缺的不是成员，是**设计**。
- 附带约束：`InteractionGroupEvaluatorConfig.maximum_pairwise_candidates = 32`，故 `C(n,2) ≤ 32` ⇒ `n ≤ 8`。

**结论：本文沿用现有 4 个成员，不新增成员、不新增场景、不新增语料、不新增 provider。**

### 2.4 持有 2 对的设计空间与选定

持有 2 对后 `observed = 4`（预算充足，且 4 个成员仍在观测对中出现）。选定：

| 持有 pair | 能力面 | 类型 |
|---|---|---|
| `member-a + member-c` | `{0}` ∪ `{2}` | **DISJOINT** |
| `member-b + member-d` | `{1}` ∪ `{0}` | **DISJOINT** |

**opaque 表述**（按 id 字典序索引，不泄露语义映射）：

- `P₁*` = 索引 `(0, 2)` → `member-a + member-c`
- `P₂*` = 索引 `(1, 3)` → `member-b + member-d`

选择理由（**不是**为了「选一对容易赢的」）：

1. 两对**能力面均不重叠**，故「联合增益」在结构上**可能**存在——这正是研究问题要检验的；
2. 保留在观测面的 4 对中包含**冗余对 `a+d`**，使 learner 有机会（在路线 A 尚未实施的本 gate 中，
   仅靠已观测的**联合**证据）观察到「某些组合无超额收益」，而非只能看到互补组合；
3. 未按 P5.2c′ 的字典序 `(0,3)` 选取，避免重复其抽样偏差。

## §3 设计

### 3.1 成员与角色（零新增语义，零新增数据源）

沿用 P5.2b / P5.2c′ 的 4 个 family-specialist readout（`member-a/b/c/d`，opaque id）。
**不新增成员**（§2.3 已证无效且有害）。语义映射仍只存于 runner 配置节，不进 learner/evaluator 输入。

### 3.2 矩阵与分区（对 P5.2c′ 的结构性改动，显式披露）

沿用前身：`12 context × 11 cells × 2 重复 = 264 episodes` 的真实合同执行，同一 P5.2a validation
context 集合、同一 cell 集合、同一构造机制、同一执行路径。

**改动**：在 **train 分区（`contexts[0:8]`）** 中，**整体移除** `P₁*` 与 `P₂*` 的联合 cell：

| 分区 | context | `P₁*`/`P₂*` 的 `(T,T)` | 四对的 `(T,T)` | 两对成员的 `(F,F)/(T,F)/(F,T)` |
|---|---|---|---|---|
| train | `p52a-validation-100..107` | **移除** | 保留 | 保留 |
| holdout | `p52a-validation-108..111` | **保留（用于评分）** | 保留 | 保留 |

- train episode 数：`8 × 11 × 2 = 176` → 移除 `8 × 2 × 2 = 32` → **144**。
- holdout episode 数：`4 × 11 × 2 = 88`（**不变**）。
- **必须逐条断言**：
  - `P₁*` 与 `P₂*` 均**不出现**在 `train_only_candidates` 的 `observed_records` 中；
  - 两对的 `(T,T)` 在 train 分区出现次数均 **== 0**；
  - 两对的 `(T,T)` 在 holdout 出现次数均 **> 0**；
  - 其余 **4 对均在** `observed_records`；
  - 4 个成员在 train 中各自仍有 **16** 条 singleton 观测（即两对是**未见**而非**无支持**）。

### 3.3 证据来源（一步交叉验证，禁止 train 内部自证）

- **learner 可见证据（唯一）**：`contexts[0:8]` 的 **144** 个 episodes → `build_member_evidence` +
  `train_only_candidates`（预期返回 **4** 个候选）。
- **被评对象**：`P₁*` 与 `P₂*`（须满足 `unseen_only=True` 下的可见性，逐条断言）。
- **未见 context（holdout）**：`contexts[8:12]` → 独立评分，**不进任何 fit/observe/candidate 输入**。
- **禁止**：holdout outcome 回流；用 train context 的 `P₁*`/`P₂*` 实测值充当证据（§3.2 已在结构上排除）；
  在 holdout 上重跑挑结果。

### 3.4 候选—执行—评估流程

1. learner 在 train 分区 fit 完成（4 个观测 pair + 4 个 profile）。
2. `select(candidate_sets_ordered, resource_budget=10.0, unseen_only=True)` 选出候选；
   **必须选出 `P₁*` 或 `P₂*` 之一**（若 `select` 返回 `None` 或返回已观测 pair，按 §6 判 `failed`）。
3. **执行前绑定（不可事后补记）**：冻结 `parent_checkpoint` payload+digest、`member_ids`、
   `predicted_interaction`、`uncertainty`、`resource_cost`、选择时刻。
4. 在 holdout context 上真实执行被选 pair 及其该 context 的 `none` 基线与每个 singleton。
5. 执行后**独立**计算：`realized_interaction = pair − first − second + baseline`（同 `_estimate_pair` 公式，
   但在 holdout 上独立计算），以及 `realized_pair_gain_vs_strongest_single = pair − max(first, second)`。
6. **两对均须评分**（不只评分被选中的那对），以便门 8 校准建立在 **2** 个未见样本上。

### 3.5 对照（全部冻结，同一执行路径、同一评分口径）

| # | 对照 | 定义 |
|---|---|---|
| C1 | no_learning | 不学习：字典序取第一个 profile 完整候选 pair |
| C2 | strongest_singleton | **同一 context 内**实测收益最高的单体（无 pair） |
| C3 | random_combination | 4 个**未观测** pair 中均匀随机选 1（固定 seed，与 learner 无关） |
| C4 | fixed_combination | 固定选 `P₁*`（opaque：索引 `(0,2)`） |
| C5 | train_only_simple_regression | 仅按成员 profile 贡献之和线性排序（不拟合交互项） |
| C6 | lesion_learner | learner 系数清零后的选择（结构对照） |

> **C4 的特别说明**：因 `P₁*` 被结构指定，C4 与「learner 选对 `P₁*`」在上限上重合。这**削弱了 C4 的对照力**，
> 属本设计的已知代价，故 **§5 门 9 不以「> C4」单独作为通过条件**，而要求同时满足预测校准门（门 8），
> 即必须证明**预测**与实现一致，而不只是「选中了正确答案」。此项代价在 §7 显式登记。

### 3.6 主判据（MARGIN 沿用，不改常数）

对**被选 pair**，要求 `mean_holdout(realized_pair_gain_vs_strongest_single)` **>**
`max(对照 C1/C2/C3/C5/C6 同口径值)` + `MARGIN`，`MARGIN = 0.15`
（沿用 P5.2a 冻结的 `FROZEN_MARGIN` 与 P5.2b/P5.2c′ 配对/对照口径传统；本 gate 只改度量对象，不改常数）。

**并须同时满足**门 8 的预测校准要求（见 §5），否则判 `transfer_no_gain`。

**另须满足**（路线 C 的新增要求）：**两对中至少 1 对**的
`mean_holdout(realized_pair_gain_vs_strongest_single) > 0`，否则该 gate 的全部对象都无正增益，
判定为 `transfer_signal_constant`（见 §6）。

### 3.7 必报分账（不作门，禁止省略）

- 预测校准：**pooled（2 个未见样本）** 与 per-pair 的 `predicted_interaction` vs `realized_interaction`
  的绝对误差、符号一致率、相关系数（样本量小须标注，2 个样本不构成独立样本量）。
- 任务成功：per-context、per-template-block 的完整 11 cell 成功率矩阵照实输出。
- **`P₁*`/`P₂*` 专项**：两对在 train 中四 cell 的出现次数（须各为 `0/1/1/1`）、在 holdout 中的真实四 cell 值。
- **互补性实证**：两对成员的能力面并/交（须实测，不得引用本文 §2.1 的旧值充当本轮结果）。
- 移除动作披露：train 分区被移除的确切 episode 标识清单（**32** 条）。
- 每类保持：若 learner 选择改变了某些 context 的成功，逐 context 列「变好/变坏/不变」。
- 最坏组：最差 context block 与最差 cell 实测成功率单列。
- 失败率与预算：`contract_intercepted:*` / `bind_failure:*` / `step_cap` 分类计数、动作总数、wall。
- **家族覆盖归因**：pair 成功与「该 block 是否被成员家族覆盖」的列联表。
- **block-3 专项**：单列 block-3 对**所有** cell（含 baseline）的成功数，
  明确区分「合同层不可达」与「无增益」；报告须给出 block-3 的 `contract_intercepted` 原因计数。

### 3.8 恢复与拒绝

- `checkpoint()` → **全新进程** `from_checkpoint` → 相同候选与预算下 `select` 结果逐位一致。
- 必须**拒绝**（逐项断言 raise 或返回 `None`）：含 holdout 字段的记录 → `observe_records` raise；
  篡改 `checkpoint_digest` / stale `source_trace_digest` / stale `checkpoint_revision` → `from_checkpoint`
  raise；未知成员（无 singleton profile）→ `candidate()` 返回 `None`；**已观测的 4 对**在
  `unseen_only=True` 下 → 不被选中；**`P₁*` 与 `P₂*` 均须可被预测**（两方向都断言）。

### 3.9 预算与确定性

- `TOTAL_SECONDS_CAP = 900.0`（沿 P5.2b/P5.2c′）。
- replica：相同 seed 与相同构造在独立临时 workspace 重跑全矩阵，
  `(episode_id, success, stop_reason, resource_cost)` 排序后逐位一致；构造层随机量按先例显式排除并披露。
- 多 seed 只作**重复测量**（选择稳定性），**不得**当独立样本量，不做显著性主张。

## §4 干预真实性（承接 P5.2b 修复，作为入场硬条件）

复用 P5.2b runner 的 `_intervention_reality()` 判据，并在入场处硬断言：

- 任何非 baseline cell 零步 → `interventions_happened=false` → **本 gate 立即 `failed`**，不进入评分。
- `(F,F)` baseline 零步豁免（「未调用任何成员」即其 treatment）。
- ≥3 层修复（判据 / 任务 `requires_explicit_language_override` / `_bind` 透传 `user_override` +
  episode 重置用显式空 payload）必须保持有效；本 gate 运行时再次核验 `zero_step_episodes_total == 0`。

## §5 九门

| # | 门 | 判据 |
|---|---|---|
| 1 | static_checks | scoped py_compile/ruff/black + 全仓 ruff 通过 + pytest 无新增失败（既有债务见 [技术债登记册](../active/roadmap/05_TECH_DEBT_REGISTER.md)）+ mypy 基线持平 |
| 2 | evidence_admissibility | learner 只消费 train-only 证据；`train_only_candidates` 记录 `status="candidate"` 且 `holdout_interaction is None and holdout_recovery_effect is None`（逐条断言） |
| 3 | **unseen_combination_identity** | **`P₁*` 与 `P₂*` 均不在 learner `observed_records` 中**；两对的 `(T,T)` 在 train 出现 **0** 次；两对的 `(T,T)` 在 holdout 出现 **>0** 次；其余 **4 对均在**；`unseen_only=True`；候选成员全部有 profile；两对成员在 train 各有 **16** 条 singleton 观测 |
| 4 | unseen_context_holdout | 评分 context 与 learner 可见 context 不相交（`[8:12]` vs `[0:8]`）；holdout episode 未进任何 fit/observe/candidate 调用（构造断言） |
| 5 | label_opaqueness | 成员 id opaque；映射只在 runner 配置节；learner/evaluator 输入不含映射与模板名；报告不输出语义→成员对应表 |
| 6 | real_execution | 被选 pair 与全部对照的 outcome 均来自真实合同执行；安全违规 0；provenance 三来源完整；`recovery_effect` 记 0 并披露；`intervention_reality.interventions_happened == true` |
| 7 | rejection_recovery | §3.8 各类拒绝逐项成立 + 全新进程恢复后 `select` 逐位一致 |
| 8 | **prediction_binding_and_calibration** | 执行前留存 `parent_checkpoint` digest、成员集合、预测 interaction、uncertainty、resource_cost（时间早于执行，不可事后补记）；执行后真实收益独立计算并并列输出；**且** pooled（**2** 个未见样本）预测—实现符号一致率 ≥ `0.5`、绝对误差中位数 ≤ `0.35`（阈值在冻结时固定，不由实测反推） |
| 9 | transfer_and_budget | §3.6 主判据成立（含 MARGIN 0.15、> C1/C2/C3/C5/C6）**且**门 8 成立**且**§3.6 的「至少 1 对正增益」成立；replica 一致；wall ≤ 900s；`P₁*`/`P₂*` 在 train 的四 cell 计数各 == `0/1/1/1` |

**防止判据漂移**：门 2/3/4/7 是机械门，失败即 `failed`，不得解释为「科学结论」。
门 8/9 是实质能力门，失败走三态。**门 8 的阈值 `0.5` / `0.35` 在本文冻结时即固定**，重跑不得调整。

## §6 三态

- `unseen_combination_transfer_supported`：九门全过，且 §3.6 主判据与门 8 同时成立。
- `transfer_no_gain`：门 1–7 全过、门 8 或门 9 未过——cohort 内无迁移收益或无预测校准。
- `transfer_signal_constant`：矩阵成立但（a）全部 pair 的 factorial cells outcome 恒定，
  或（b）**两对未见组合的正增益均为 0**（路线 C 新增判据）。
- `failed`：任一机械/合同/干预真实性门失败。
- `blocked_at_entry_audit`：入场前置（§4 或 §3.2 断言）不成立；若出现，说明移除动作未生效，须回接线而非改判据。

## §7 停止点与已知代价

- 机械失败 → 回合同（修接线，不改判据）。
- `unseen_combination_identity` 失败 → §3.2 移除动作未生效，回接线。
- `transfer_signal_constant` → 回 P5.2b 重新设计可干预成员/任务。
- `transfer_no_gain` → **进入路线 A（表征修复）**：按决策文档 §2 路线 A，检验 profile 是否需携带能力面信息，
  需**新预注册**。本 gate 不因归因结论被追认为通过。
- 收益、保持、成本均通过 → 才允许进入 P5.2d 在线回写。
- **已知代价（显式登记）**：
  1. **C4 与 `P₁*` 重合**，故门 9 不以「> C4」单独通过，须同时满足门 8。
  2. **`P₁*`/`P₂*` 由结构指定**（各为 1 对，共 2 对），故本 gate 通过**只主张**
     「预测—实现一致性 + 优于无学习/随机/单体的对照」，
     **不主张** learner 具备「在**多个**未见组合中选出最优」的能力——后者需更大未见面。
  3. **表征缺陷未修**：4 个 profile 的 `contribution` 仍可能全等（P5.2c′ 实测全为 `0.5`），
     故本 gate 的**负**结果不能归因于「迁移不可行」，只能归因于「该表征下不可行」。
  4. **block-3 仍不可达**（若本轮未修）：有效区分面为 3/4，功效受限；报告须显式量化其影响。
- 任何失败都保留原报告与本文，不覆写、不改绿。

## §8 纪律

门值以报告为准逐项对账；禁止反向改判据、剪枝凑数、重跑挑结果；报告不覆写不改绿；
`growth_admitted=false`、`can_promote=false` 贯穿；临时脚本用毕即删；预注册冻结前不训练、不接新数据源、
不读取 sealed；多 seed 只作重复测量；语义映射不进入 learner/evaluator 输入。

**本预注册不修改前身任何冻结产物**：P5.2c 原预注册、P5.2c 入场审计、P5.2b 报告、
P5.2c′ 预注册与 P5.2c′ 报告。

## §9 产物顺序

1. `scripts/training/eval_taiji_p5_2c_double_prime_unseen_combination_transfer_gate.py`（新 runner）
2. `reports/taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json`（新报告；**不覆盖**任何前身报告）
3. 本文（冻结）
4. `plans/active/roadmap/03_CURRENT_EXECUTION.md` 状态同步

## §10 附：本文所依据的探针结论（用毕即删，此处留证）

| 探针 | 结论 |
|---|---|
| 成员数 vs unseen pair 数（P5.2c′ §2） | n=4/5/6 的 unseen pair **均为 0** ⇒ 增加成员数无效 |
| 移除范围 | 部分移除**无效**（剩一个齐备 context 即被估计）；全分区移除**有效** |
| 能力面实测 | `a`=`{0}`、`b`=`{1}`、`c`=`{2}`、`d`=`{0}` ⇒ `a+d` 冗余，其余 5 对互补 |
| 模板族数量 | train 仅 **4** 族 ⇒ 新增成员无新族可专精，只增冗余 |
| 双持有机械验证 | `train=144 / holdout=88 / removed=32`，`observed=4`，两对 joint 均 `train=0 / holdout=8`，4 成员各保留 **16** 条 singleton |
