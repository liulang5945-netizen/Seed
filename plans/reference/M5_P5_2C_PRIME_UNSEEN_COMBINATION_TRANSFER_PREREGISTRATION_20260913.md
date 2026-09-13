# M5 P5.2c′ — 未见组合迁移 Gate 预注册（冻结版，承接 P5.2c 入场审计）

> 冻结日期：2026-09-13。状态：**冻结，待执行**。
> 取代关系：本文**不修改、不追溯改写** [P5.2c 原预注册](M5_P5_2C_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md)；
> 原预注册保持冻结原貌，其 `blocked_at_entry_audit` 结论与 [入场审计](../../reports/M5_P5_2C_ENTRY_AUDIT_P5_2B_DEFECT_20260913.md) 继续有效。
> 本文为**新预注册**，因为原门 3 的缺陷是结构性的（未见组合面为空），必须以新设计重新冻结判据后重跑。
> 修复记录见 [03_CURRENT_EXECUTION.md §4.1](../active/roadmap/03_CURRENT_EXECUTION.md)。

## §1 研究问题

与前身一致：**当某两个认知成员的联合效应从未在训练证据中被估计过时，learner 能否在未见 context 上，
仅凭成员 profile 与已观测的其他组合，预测出该组合的真实联合增益，并显著优于全部冻结对？**

**本轮新增的、使问题可回答的前置**：必须先**构造一个非空的未见组合面**，否则该问题在当前证据结构下
不可问（前身即因此阻塞）。

## §2 阻断缺陷的实证根因（本轮量化，权威来源=本预注册所附探针证据）

前身门 3 不可满足的根因**不是**「成员数不够」，而是 `InteractionGroupEvaluator.train_only_candidates`
的枚举口径：

```python
# taiji/interaction_groups.py:848-851
member_ids = tuple(sorted({member for episode in corpus.train for member in episode.member_ids}))
pairs = tuple(itertools.combinations(member_ids, 2))          # 枚举全部 C(n,2)
...
estimate = self._estimate_pair(corpus.train, members)          # 需要四 cell 齐备
if estimate is None or ...: continue                           # 不齐备者被【静默跳过】
```

配合 `_estimate_pair` 的可用性条件（同文件 928-936 行）：

```python
usable_contexts = [cells for cells in contexts.values()
    if all(cells[key] for key in ((False,False),(True,False),(False,True),(True,True)))]
if not usable_contexts: return None
```

**推论（已实证，见 §3）**：只要有**任意一个** train context 齐备某 pair 的四 cell，该 pair 就被估计并进入
`observed_records`。因此「增加成员数」**完全无效**——成员越多，被观测的 pair 只会更多。

| 成员数 n | `C(n,2)` 候选 pair | 被观测 pair | 未见 pair |
|---|---|---|---|
| 4 | 6 | 6 | **0** |
| 5 | 10 | 10 | **0** |
| 6 | 15 | 15 | **0** |

→ **roadmap 早先记的「成员数 ≥5 使 pair 面 > 已观测数」是错的**，已由本轮探针证伪，不采用。

**使未见组合面非空的唯一可行机制（已实证）**：把某个 pair 的联合 cell `(T,T)` **从 train 全部分区中整体移除**
（该 pair 的 `(F,F)/(T,F)/(F,T)` 仍保留，即两个成员各自仍被单独观测），使其在任何 train context 下都
**凑不齐四 cell** → `_estimate_pair` 返回 `None` → 该 pair 不进入 `observed_records` → 成为真正未见组合。

| 移除范围 | 被观测 pair | 未见 pair |
|---|---|---|
| 不移除 | 6 | 0 |
| 仅从 ctx-0/ctx-1 移除 | 6 | **0**（只保留一个齐备 context 即可被估计） |
| **从全部 train context 移除** | 5 | **1（目标 pair）** |

## §3 设计

### 3.1 成员与角色（零新增语义，零新增数据源）

- 沿用 P5.2b 的 4 个 family-specialist readout（`member-a/b/c/d`，opaque id）。
- **不新增成员**（§2 已证无效）。**不新增场景**、不新增语料、不新增 provider。
- 语义映射（成员 → 训练家族）仍只存于 runner 配置节，不进 learner/evaluator 输入。

### 3.2 矩阵与分区（对 P5.2b 的唯一结构性改动，且该改动被显式披露）

沿用前身：`12 context × 11 cells × 2 重复 = 264 episodes` 的真实合同执行，同一 P5.2a validation
context 集合、同一 cell 集合、同一构造机制、同一执行路径。

**唯一改动**：指定一个 held-out pair（记为 `P*`，opaque 表述：按 id 字典序索引 `(0,3)`，即 P5.2b 中
曾以伪成功 admitted 的那一对）。在 **train 分区（`contexts[0:8]`）** 中，**整体移除** `P*` 的联合 cell：

| 分区 | context | `P*` 的 `(T,T)` | `P*` 的 `(F,F)/(T,F)/(F,T)` | 其他 5 个 pair 的四 cell |
|---|---|---|---|---|
| train | `p52a-validation-100..107` | **移除** | 保留 | 保留 |
| holdout | `p52a-validation-108..111` | **保留（用于评分）** | 保留 | 保留 |

- train episode 数：`8 × 11 × 2 = 176` → 移除 `8 × 2 = 16` → **160**。
- holdout episode 数：`4 × 11 × 2 = 88`（**不变**，`P*` 在此处被真实执行以取得 realized 收益）。
- **必须逐条断言**：`P*` 不出现在 `train_only_candidates` 的 `observed_records` 中；且 `P*` 的
  `(T,T)` 在 train 分区出现次数 == 0。

**为什么这不是「制造看不见的答案」而是合法设计**：`P*` 的两个成员在 train 中各自被完整观测（singleton
与基线均在），learner 拥有全部**单体**证据；它缺的只是这一对**联合**是否成立。这正是「未见组合」的定义。
移除动作发生在**证据构造层**，只影响 learner 的输入，不影响 holdout 上的真实执行与评分。

### 3.3 证据来源（一步交叉验证，禁止 train 内部自证）

- **learner 可见证据（唯一）**：`contexts[0:8]` 的 160 个 episodes → `build_member_evidence` +
  `train_only_candidates`（预期返回 5 个候选）。
- **被评对象**：`P*`（必须满足 `unseen_only=True` 下的可见性，逐条断言）。
- **未见 context（holdout）**：`contexts[8:12]` → 独立评分，**不进任何 fit/observe/candidate 输入**。
- **禁止**：holdout outcome 回流；用 train context 的 `P*` 实测值充当证据（§3.2 已在结构上排除）；
  在 holdout 上重跑挑结果。

### 3.4 候选—执行—评估流程

1. learner 在 train 分区 fit 完成（5 个观测 pair + 4 个 profile）。
2. `select(candidate_sets_ordered, resource_budget=10.0, unseen_only=True)` 选出候选；**必须选出 `P*`**
   （若 `select` 返回 `None` 或返回非 `P*`，按 §6 判 `failed`/`no_gain`，按实际取值）。
3. **执行前绑定（不可事后补记）**：冻结 `parent_checkpoint` payload+digest、`member_ids`、
   `predicted_interaction`、`uncertainty`、`resource_cost`、选择时刻。
4. 在 holdout context 上真实执行 `P*` 及其该 context 的 `none` 基线与每个 singleton（最强单体对照所需）。
5. 执行后**独立**计算：`realized_interaction = pair − first − second + baseline`（同 `_estimate_pair` 公式，
   但在 holdout 上独立计算），以及 `realized_pair_gain_vs_strongest_single = pair − max(first, second)`。

### 3.5 对照（全部冻结，同一执行路径、同一评分口径）

| # | 对照 | 定义 |
|---|---|---|
| C1 | no_learning | 不学习：字典序取第一个 profile 完整候选 pair |
| C2 | strongest_singleton | **同一 context 内**实测收益最高的单体（无 pair） |
| C3 | random_combination | 6 个 pair 中均匀随机选 1（固定 seed 表，与 learner 无关） |
| C4 | fixed_combination | 固定选 `P*`（opaque：索引 `(0,3)`） |
| C5 | train_only_simple_regression | 仅按成员 profile 贡献之和线性排序（不拟合交互项） |
| C6 | lesion_learner | learner 系数清零后的选择（结构对照） |

> **C4 的特别说明**：因 `P*` 现已被结构指定，C4 与「learner 选对」在上限上重合。这**削弱了 C4 的对照力**，
> 属本设计的已知代价，故 **§5 门 9 不以「> C4」单独作为通过条件**，而要求同时满足预测校准门
> （§4 门 8）——即必须证明**预测**与实现一致，而不只是「选中了正确答案」。此项代价在 §7 停止点显式登记。

### 3.6 主判据（MARGIN 沿用，不改常数）

要求 `mean_holdout(realized_pair_gain_vs_strongest_single)` **>**
`max(对照 C1/C2/C3/C5/C6 同口径值)` + `MARGIN`，`MARGIN = 0.15`
（沿用 P5.2a 冻结的 `FROZEN_MARGIN` 与 P5.2b 配对/对照口径传统；本 gate 只改度量对象，不改常数）。

**并须同时满足**门 8 的预测校准要求（见 §5），否则判 `transfer_no_gain`。

### 3.7 必报分账（不作门，禁止省略）

- 预测校准：pooled 与 per-context 的 `predicted_interaction` vs `realized_interaction` 的绝对误差、
  符号一致率、相关系数（样本量小须标注）。
- 任务成功：per-context、per-template-block 的完整 11 cell 成功率矩阵照实输出。
- **`P*` 专项**：`P*` 在 train 中四 cell 的出现次数（须为 `0/1/1/1`）、在 holdout 中的真实四 cell 值。
- 移除动作披露：train 分区被移除的确切 episode 标识清单（16 条）。
- 每类保持：若 learner 选择改变了某些 context 的成功，逐 context 列「变好/变坏/不变」。
- 最坏组：最差 context block 与最差 cell 实测成功率单列。
- 失败率与预算：`contract_intercepted:*` / `bind_failure:*` / `step_cap` 分类计数、动作总数、wall。
- **家族覆盖归因**：pair 成功与「该 block 是否被成员家族覆盖」的列联表。

### 3.8 恢复与拒绝

- `checkpoint()` → **全新进程** `from_checkpoint` → 相同候选与预算下 `select` 结果逐位一致。
- 必须**拒绝**（逐项断言 raise 或返回 `None`）：含 holdout 字段的记录 → `observe_records` raise；
  篡改 `checkpoint_digest` / stale `source_trace_digest` / stale `checkpoint_revision` → `from_checkpoint`
  raise；未知成员（无 singleton profile）→ `candidate()` 返回 `None`；**已观测 pair 在 `unseen_only=True`
  下 → 不被选中**（本 gate 用它验证 `P*` 之外的 5 对确实不可选）。

### 3.9 预算与确定性

- `TOTAL_SECONDS_CAP = 900.0`（沿 P5.2b）。
- replica：相同 seed 与相同构造在独立临时 workspace 重跑全矩阵，
  `(episode_id, success, stop_reason, resource_cost)` 排序后逐位一致；构造层随机量按先例显式排除并披露。
- 多 seed 只作**重复测量**（选择稳定性），**不得**当独立样本量，不做显著性主张。

## §4 干预真实性（承接 P5.2b 修复，作为入场硬条件）

前身被 P5.2b 的空事件 episode 污染（57 个非 baseline 零步 cell）。**本 gate 复用 P5.2b runner 的
`_intervention_reality()` 判据**，并在入场处硬断言：

- 任何非 baseline cell 零步 → `interventions_happened=false` → **本 gate 立即 `failed`**，不进入评分。
- `(F,F)` baseline 零步豁免（「未调用任何成员」即其 treatment）。
- ≥3 层修复（判据 / 任务 `requires_explicit_language_override` / `_bind` 透传 `user_override` +
  episode 重置用显式空 payload）必须保持有效；本 gate 运行时再次核验 `zero_step_episodes_total == 0`。

## §5 九门

| # | 门 | 判据 |
|---|---|---|
| 1 | static_checks | scoped py_compile/ruff/black + 全仓 ruff 通过 + pytest 无新增失败（既有债务见 [技术债登记册](../active/roadmap/05_TECH_DEBT_REGISTER.md)）+ mypy 基线持平 |
| 2 | evidence_admissibility | learner 只消费 train-only 证据；`train_only_candidates` 记录 `status="candidate"` 且 `holdout_interaction is None and holdout_recovery_effect is None`（逐条断言） |
| 3 | **unseen_combination_identity** | **`P*` 不在 learner `observed_records` 中**；`P*` 的 `(T,T)` 在 train 出现 **0** 次；其余 5 对**均在**；`unseen_only=True`；候选成员全部有 profile |
| 4 | unseen_context_holdout | 评分 context 与 learner 可见 context 不相交（`[8:12]` vs `[0:8]`）；holdout episode 未进任何 fit/observe/candidate 调用（构造断言） |
| 5 | label_opaqueness | 成员 id opaque；映射只在 runner 配置节；learner/evaluator 输入不含映射与模板名；报告不输出语义→成员对应表 |
| 6 | real_execution | `P*` 与全部对照的 outcome 均来自真实合同执行；安全违规 0；provenance 三来源完整；`recovery_effect` 记 0 并披露；`intervention_reality.interventions_happened == true` |
| 7 | rejection_recovery | §3.8 各类拒绝逐项成立 + 全新进程恢复后 `select` 逐位一致 |
| 8 | **prediction_binding_and_calibration** | 执行前留存 `parent_checkpoint` digest、成员集合、预测 interaction、uncertainty、resource_cost（时间早于执行，不可事后补记）；执行后真实收益独立计算并并列输出；**且** pooled 预测—实现符号一致率 ≥ `0.5`、绝对误差中位数 ≤ `0.35`（阈值在冻结时固定，不由实测反推） |
| 9 | transfer_and_budget | §3.6 主判据成立（含 MARGIN 0.15、> C1/C2/C3/C5/C6）**且**门 8 成立；replica 一致；wall ≤ 900s；`P*` 在 train 的四 cell 计数 == `0/1/1/1` |

**防止判据漂移**：门 2/3/4/7 是机械门，失败即 `failed`，不得解释为「科学结论」。
门 8/9 是实质能力门，失败走三态。**门 8 的阈值 `0.5` / `0.35` 在本文冻结时即固定**，重跑不得调整。

## §6 三态

- `unseen_combination_transfer_supported`：九门全过，且 §3.6 主判据与门 8 同时成立。
- `transfer_no_gain`：门 1–7 全过、门 8 或门 9 未过——cohort 内无迁移收益或无预测校准。
- `transfer_signal_constant`：矩阵成立但全部 pair 的 factorial cells outcome 恒定（无差异可测）。
- `failed`：任一机械/合同/干预真实性门失败。
- `blocked_at_entry_audit`：入场前置（§4）不成立——**本状态在本文档下应不再出现**；
  若再次出现，说明 §3.2 的移除动作未生效，须回接线而非改判据。

## §7 停止点与已知代价

- 机械失败 → 回合同（修接线，不改判据）。
- `unseen_combination_identity` 失败 → §3.2 移除动作未生效，回接线。
- `transfer_signal_constant` → 回 P5.2b 重新设计可干预成员/任务。
- `transfer_no_gain` → 回模型/特征归因（profile 维度、pair 关系项形式、不确定性口径），需**新预注册**再试；
  本 gate 不因归因结论被追认为通过。
- 收益、保持、成本均通过 → 才允许进入 P5.2d 在线回写。
- **已知代价（显式登记）**：由于 `P*` 被结构指定为唯一未见组合，候选面退化为 1，
  「选择能力」的检验力弱于原设计；且 C4 与正确答案重合（§3.5）。故本 gate 的通过**只主张
  「预测—实现一致性 + 优于无学习/随机/单体的对照」，不主张「learner 具备在多组合中选择的能力」**。
  后者需另行设计（≥2 个未见组合，要求更高成员数配合整体移除，成本更高）。
- 任何失败都保留原报告与本文，不覆写、不改绿。

## §8 纪律

门值以报告为准逐项对账；禁止反向改判据、剪枝凑数、重跑挑结果；报告不覆写不改绿；
`growth_admitted=false`、`can_promote=false` 贯穿；临时脚本用毕即删；预注册冻结前不训练、不接新数据源、
不读取 sealed；多 seed 只作重复测量；语义映射不进入 learner/evaluator 输入。
**本预注册不修改前身 P5.2c 预注册与入场审计报告的任何内容。**

## §9 产物顺序

1. `scripts/training/eval_taiji_p5_2c_prime_unseen_combination_transfer_gate.py`（新 runner）
2. `reports/taiji_p5_2c_prime_unseen_combination_transfer_20260913.json`（新报告；**不覆盖**前身报告）
3. 本文（冻结）
4. `plans/active/roadmap/03_CURRENT_EXECUTION.md` 状态同步
