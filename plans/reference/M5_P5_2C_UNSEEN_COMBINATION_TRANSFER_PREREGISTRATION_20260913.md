# M5 P5.2c — 未见组合迁移 Gate 预注册（冻结版）

日期：2026-09-13。状态：**已冻结，执行被阻塞**（本文提交即冻结；冻结后不接预注册之外数据源、不读取 sealed、不启动训练）。
上游：[推进方案](../active/roadmap/03_CURRENT_EXECUTION.md) §7 与「当前唯一下一步」、P5.2b（[预注册](M5_P5_2B_GROUP_CAUSAL_CORPORA_PREREGISTRATION_20260913.md) / [报告](../../reports/taiji_p5_2b_group_causal_corpora_20260913.json)：`group_causal_corpora_supported`，提交 `0abf463f`）、P5.2a 归因（`predictive_execution_insufficient`，提交 `eff6e1d1`；只读 recon `6de56c10`；零训练探针 `b63eb483`）。

> **入场审计结论（2026-09-13，见 [审计报告](../../reports/M5_P5_2C_ENTRY_AUDIT_P5_2B_DEFECT_20260913.md)）**：本预注册按 §5 门 2/3 接线后**未通过入场条件**——`select(..., unseen_only=True)` 返回 `None`：4 个成员下 `C(4,2)=6` 个 pair 被 `train_only_candidates` **全数观测**，未见组合集合为空（**门 3 结构性不可满足**）。进一步审计定位到上游缺陷：P5.2a `lang_confirm` 模板（context `100/104/108`）的目标状态 == 初始状态，使这些 context 下 9 个干预 cell **零步执行即判成功**（空事件 episode），由此生成的 `member-a+member-d` admitted group（`interaction=0.2222`）**是伪成功而非联合增益**。**P5.2c 停止执行**，出口回到 P5.2b 修复（见审计 §5 三项）。本预注册保留其冻结原貌，不追溯改写；其「4 成员 / 6 pair」设计前提已披露为不成立，任何后续执行必须**新预注册**。

## §1 研究问题

只看在**干预前**就存在的 train trace 证据（成员 singleton profile + 已测 pair 记录），迁移 learner 能否为**从未被执行过的成员组合**预测出有符号的联合增益，并被随后的真实 Workbench 执行独立证实——且该收益在**未参与拟合的 context** 上成立？

两个必须分开的主张，判据不同：

- **协作主张**：被选中的组合的真实收益 > 同一 context 下的最强单体（§4.3 最强单体对照）。
- **泛化主张**：预测—实现一致性成立在**未参与拟合的组合**（learner 从未 `observe` 过的 pair）与**未参与拟合的 context**（learner 从未见过其任何 episode 的 context）上。

本阶段 learner 只产出候选预测，**不承担权限准入**：是否执行由 Workbench 合同（policy/approval）决定，不被 learner 覆盖。

## §2 复用的可干预成员与证据路径（全部既有资产，零新增语义）

- **成员（与 P5.2b 同一真实实例）**：4 个 family-specialist procedural readout，各只在 P5.2a train 的一个模板家族上训练（hidden 64 / epochs 250 / seed `17 + index`），opaque 命名 `member-a/b/c/d`；语义映射表只存于 runner 配置节，不进入任何 learner/evaluator 输入（§5 门 5）。
- **复用路径（唯一允许路径）**：`build_member_evidence → observe_members / observe_records → candidate / select → Workbench 实际执行`。
  - profile 来源：`build_member_evidence(corpus.train, ...)` —— 同 context 同时存在 inactive baseline 与 singleton 才纳入（未知成员 fail-closed）。
  - pair 记录来源：`InteractionGroupEvaluator.train_only_candidates(corpus)` —— 仅由 train trace 估计，返回 `status="candidate"`、**不含任何 holdout 派生字段**、未 admitted。`observe_records` 对含 `holdout_interaction`/`holdout_recovery_effect` 的记录与 terminal 状态记录一律 raise（本 gate 主动断言其拒绝行为，见 §5 门 7）。
  - learner：`InteractionGroupTransferLearner`（`ridge=0.1`）；`_pair_features = (1.0, (c1+c2)/2, c1*c2)`，`uncertainty = residual_rmse + 1/sqrt(max(1, min profile observations))`，`resource_cost = 两成员 profile 成本之和`。
  - 选择：`select(candidate_member_sets, resource_budget=..., unseen_only=True)` —— `unseen_only=True` 使**已 observe 过的 pair 直接失格**，这正是「未见组合」通道；未知成员 `candidate()` 返回 `None`。
- **构造机制（沿用 P5.2b 冻结修订）**：**dual-predict-select** —— 每 tick 真实调用每个在场成员（各自 readout 预测 + 参数绑定尝试，逐成员留痕），选择规则 = 字典序第一个绑定成功者执行、其余记 dissent。逐成员留痕是必需的：episode 的 `member_ids` 由事件 owner 派生，纯 fallback 链下未调用成员无事件，pair cell 结构性无法成形。
- **执行合同（与 P5.2/P5.2a/P5.2b 同一路径）**：`ActionIntent → WorkbenchActionRequest.from_action_intent → policy_for → issue_approval → consume_approval → execute_tool`；参数绑定三来源 `goal_state` / `world_state` / `transaction_token`，逐动作记录 provenance；绑定失败 = 安全停止，不伪造参数。

## §3 已知约束（带入本 gate，不作为可改写判据的借口）

1. **P5.2a 动作头边界**：GRU + 线性 readout 学到的是「具体 goal 文本 → 模板」的**记忆映射**，不具备语义级模板泛化结构（零训练探针在显式 cue 下成功率 0/12）。因此 member-a/d 一类组合的增益**来自互补家族覆盖，而非序列泛化**；本 gate 不主张、也不检验「学到了新的动作序列能力」。
2. **P5.2b 矩阵的周期性归因（本轮量化，权威数据 = P5.2b 报告 `matrix.per_cell_success`）**：`p52a-validation-{100..111}` 按 `index % 4` 严格分块，成功分布与模板块一一对应——块 0（100/104/108）全部含 `member-a` 的 cell 成功（`none` 3/6，其余含 a 的 cell 6/6）；块 1（101/105/109）仅 `member-b`、`member-b-member-c`、`member-b-member-d` 成功（各 6/6）；块 2（102/106/110）仅 `member-c`、`member-b-member-c`、`member-c-member-d` 成功（各 6/6，`member-b` 0/6）；块 3（103/107/111）**全 cell 0/6**。→ 该强周期性说明：pair 增益由**该 context 所属模板家族是否被某成员覆盖**决定，不由序列长度/位置泛化决定，且**句法上 §1 的「协作」判据会退化为「选中了覆盖该家族的那个成员组合」**。故 §4.5 把 context 家族覆盖列为必报分账，§4.3 的最强单体对照在**同一 context 内**取，避免用跨家族平均掩盖这一退化。
3. **门 9 层面的既有事实**：P5.2a 已证明「动作头在分布外退化」；本 gate 的收益若出现，只允许归因到成员选择与覆盖互补，**不允许**表述为序列能力提升。
4. **不测项如实披露**：`recovery_effect` 本 gate 不度量（沿 P5.2b 先例记 0，不伪造）；不度量跨家族动词泛化；不度量真实语料、provider、CUDA 优势、开放域通用能力。

## §4 设计

### 4.1 证据来源与分区（一步交叉验证，禁止 train 内部自证）

- **同一矩阵，一次执行**：沿用 P5.2b 的 `12 context × 11 cells × 2 重复 = 264 episodes` 真实合同执行（同一 P5.2a validation context 集合、同一 cell 集合、同一构造机制）。不新增场景、不新增成员、不新增数据源。
- **learner 可见证据（唯一）**：`contexts[0:8]` 的 episodes → `build_member_evidence` + `train_only_candidates`。8 个 context 的 `index % 4` 分布为 2/2/2/2，四个模板块各 2 个，覆盖对称。
- **未见组合**：learner `observe` 的记录来自 train-only 估计出的 6 个候选 pair；**被评对象 pair 必须在 learner 的 `observed_records` 中不存在**（`unseen_only=True` 的实际效果，逐条断言）。
- **未见 context（holdout）**：`contexts[8:12]`（= `p52a-validation-{108,109,110,111}`，恰为 4 个模板块各 1 个）→ 独立评分，**不进任何 fit/observe/candidate 输入**。holdout context 的 episode 只用于「执行后的真实收益」计算。
- **禁止**：把 holdout 的 outcome 回流进 learner；用 train context 的 pair 实测值充当「未见组合」的收益证据（同 context 自证）；在 holdout 上重跑挑结果。

### 4.2 候选—执行—评估流程

1. learner 在 train 分区 fit 完成 → `select` 在给定候选集合与预算下选出 1 个候选（含 `group_id` / `member_ids` / `source_trace_digest` / `checkpoint_revision` / 预测 interaction / uncertainty / resource_cost）。
2. **执行前绑定（不可事后补记）**：冻结并留存 `parent_checkpoint` payload + digest、候选 `member_ids`、`predicted_interaction`、`uncertainty`、`resource_cost`、选择时刻。
3. 在**未见 context** 上以该成员集合真实执行（同一合同路径、同一 `STEP_CAP`），并同时执行该 context 的 `none` 基线与**每个成员 singleton**（最强单体对照所需）。
4. **执行后独立计算真实收益**：按该 context 的 factorial 四 cell 实测 outcome 计算 `realized_interaction = pair − first − second + baseline`（与 `InteractionGroupEvaluator._estimate_pair` 同式，但在 holdout context 上独立计算），以及 `realized_pair_gain_vs_strongest_single = pair − max(first, second)`。
5. 预测—实现登记：per-context 与 pooled 的 `predicted_interaction`、`realized_interaction`、`abs_error`、符号一致。

### 4.3 对照（全部冻结，同一执行路径、同一评分口径）

| # | 对照 | 定义 | 作用 |
|---|---|---|---|
| C1 | no_learning | 不学习：按字典序取第一个「profile 完整」的候选 pair | 迁移是否带来任何价值 |
| C2 | strongest_singleton | **同一 context 内**实测收益最高的单体（无 pair） | 「协作」判据的分母 |
| C3 | random_combination | 从 6 个 pair 中均匀随机选 1（固定 seed 表，与 learner 无关） | 排除「随便选一对也行」 |
| C4 | fixed_combination | 固定选 P5.2b 中唯一 admitted 的等价组合 `member-a ∪ member-d`（opaque 表述：按 id 字典序索引 `(0,3)`） | 排除「固定答案就够」 |
| C5 | train_only_simple_regression | train-only 上仅用「候选成员 profile 贡献之和」线性排序的简化基线（不用 pair 记录拟合交互项） | 复杂关系项是否有额外价值 |
| C6 | lesion_learner | 将 learner 系数清零后的选择（结构对照） | 选择是否来自学到的关系项 |

**「迁移收益」的判据（主判据载体）**：设被评对象 = learner 选出的组合。要求

`mean_holdout(realized_pair_gain_vs_strongest_single)` **>** `max(对照 C1/C3/C4/C5/C6 在同一批未见 context 上同口径的 `realized_pair_gain_vs_strongest_single`)` + `MARGIN`。

`MARGIN = 0.15` —— **沿用** P5.2a 冻结的 `FROZEN_MARGIN` 常数、P5.2b 的「配对/对照口径」传统（对象是任务成功率差值），本 gate 不改常数、只改度量对象（转为 holdout 上的 pair 增益差）。该沿用在预注册中显式披露，不由实测结果反推。

### 4.4 九门判据（见 §5）与「学习到」的可证伪性

若 learner 在所有未见 context 上选出的组合与 C1/C4 完全相同、或预测 interaction 全部同号且与实现零相关，则本 gate 必须落 `transfer_no_gain`（除非 C2 判据反而成立——那说明增益来自固定组合，属 C4 已覆盖，判 `transfer_no_gain`）。禁止用「learner 也选了同一个组合」记作迁移成功。

### 4.5 必报分账（不作门，但禁止省略）

- 预测校准：pooled 与 per-context 的 `predicted_interaction` vs `realized_interaction` 的绝对值误差、符号一致率、Spearman 或 Pearson 相关系数（样本量小须标注）。
- 任务成功：per-context、per-template-block 的 `none` / 各 singleton / 各 pair 成功率矩阵（完整 11 cell × 4 block 表照实输出）。
- 每类保持：不适用新类时不写编造值；若 learner 选择改变了某些 context 的成功，逐 context 列出「变好/变坏/不变」。
- 最坏组：最差 context block 与最差 cell 的实测成功率必须单列。
- 失败率与预算：`contract_intercepted:*` / `bind_failure:*` / `step_cap` 分类计数、动作总数、wall。
- **家族覆盖归因**：pair 成功与「该 block 是否被成员家族覆盖」的列联表（§3 约束 2 的直接落地）。

### 4.6 恢复与拒绝（§11.2 入场条件）

- learner `checkpoint()` → **全新进程** `from_checkpoint` → 在相同候选集合与预算下的 `select` 结果（含 `group_id`、`member_ids`、预测值）逐位一致。
- 必须**拒绝**（逐项断言 raise 或返回 `None`）：
  - 含 holdout 字段的记录 → `observe_records` raise；
  - 篡改 `checkpoint_digest` / stale `source_trace_digest` / stale `checkpoint_revision` → `from_checkpoint` raise；
  - 未知成员（无 singleton profile）→ `candidate()` 返回 `None`；
  - 已 observe 的 pair 在 `unseen_only=True` 下 → 失格（`select` 不选它）。

### 4.7 预算与确定性

- `TOTAL_SECONDS_CAP = 900.0`（沿 P5.2b）。
- replica：以相同 seed 与相同构造在独立临时 workspace 重跑全矩阵，`(episode_id, success, stop_reason, resource_cost)` 排序后逐位一致；构造层随机量（sensation 派生量）按 P5.2a/P5.2b 先例显式排除并披露。
- **多 seed 声明**：若对 learner 做多 seed 重复，只作**重复测量**（选择稳定性），**不得**当作独立模型/任务样本量，不做显著性主张。

## §5 九门

| # | 门 | 判据 |
|---|---|---|
| 1 | static_checks | scoped py_compile/ruff/black + 全仓 ruff 通过 + pytest 无新增失败（基线既有债务见 P5.2a §3）+ mypy 基线持平 |
| 2 | evidence_admissibility | learner 只消费 train-only 证据：`build_member_evidence` 输入 = train episodes；`train_only_candidates` 返回记录 `status="candidate"` 且 `holdout_interaction is None and holdout_recovery_effect is None`（逐条断言） |
| 3 | unseen_combination_identity | 被评对象 pair 不在 learner `observed_records` 的成员集合中；`unseen_only=True`；候选成员全部有 profile（否则本门失败，不降级）。**实测：本门不可满足——6/6 pair 全被观测，未见组合面为空；本门因失败而阻塞整个 gate，不降级、不放宽** |
| 4 | unseen_context_holdout | 用于评分的 context 与 learner 可见 context 不相交（`contexts[8:12]` vs `contexts[0:8]`），且 holdout episode 未进入任何 fit/observe/candidate 调用（构造断言） |
| 5 | label_opaqueness | 成员 id opaque；映射表只在 runner 配置节；learner/evaluator 输入不含映射与模板名；报告输出不出现语义 → 成员 的对应表 |
| 6 | real_execution | 被评组合与全部对照的 outcome 均来自真实合同执行；安全违规 0；provenance 三来源完整；`recovery_effect` 记 0 并披露 |
| 7 | rejection_recovery | §4.6 四类拒绝逐项成立 + `checkpoint()` 全新进程恢复后 `select` 逐位一致 |
| 8 | prediction_binding | 每条被评候选在执行前留存 `parent_checkpoint` digest、成员集合、预测 interaction、uncertainty、resource_cost（时间早于执行，不可事后补记）；执行后真实收益独立计算，二者并列输出 |
| 9 | transfer_and_budget | §4.3 迁移收益判据成立（含 MARGIN 0.15、> 全部对照）；「协作」判据（> 同 context 最强单体）成立；replica 一致；wall ≤ 900s |

**注意（防止判据漂移）**：门 2/3/4/7 是**机械门**，失败即 `failed`，不得解释为「科学结论」。门 9 是实质能力门，失败走三态中的 `transfer_no_gain`。

## §6 三态

- `unseen_combination_transfer_supported`：九门全过，且 §4.3 迁移收益判据与「协作 > 同 context 最强单体」同时成立。
- `transfer_no_gain`：门 1–8 全过、门 9 未过——即 cohort 内无迁移收益（含预测完全无校准的情形）。
- `transfer_signal_constant`：矩阵成立但全部 pair 的 factorial cells outcome 恒定（无差异可测），预测—实现一致性无定义。
- `failed`：任一机械/合同门（1–8）失败。

## §7 停止点（沿 roadmap §7 出口）

- 机械失败 → 回合同（修接线，不改判据）。
- 无因果信号（`transfer_signal_constant`）→ 回 P5.2b 重新设计可干预成员/任务。
- 有信号但无迁移收益（`transfer_no_gain`）→ 回模型/特征归因（成员 profile 维度、pair 关系项形式、不确定性口径），需**新预注册**再试；本 gate 不因归因结论被追认为通过。
- 收益、保持、成本均通过 → 才允许进入 P5.2d 在线回写。
- 任何失败都保留原报告与本文，不覆写、不改绿。

## §8 纪律

门值以报告为准逐项对账；禁止反向改判据、剪枝凑数、重跑挑结果；报告不覆写不改绿；`growth_admitted=false`、`can_promote=false` 贯穿；临时脚本用毕即删；预注册冻结前不训练、不接新数据源、不读取 sealed；多 seed 只作重复测量；语义映射不进入 learner/evaluator 输入。

## §9 产物顺序

1. `scripts/training/eval_taiji_p5_2c_unseen_combination_transfer_gate.py`（静态检查先行）。
2. 执行落盘 `reports/taiji_p5_2c_unseen_combination_transfer_20260913.json` + 九门对账（含 §4.5 全部必报分账）。
3. roadmap 状态表与唯一下一步更新 + 独立提交。
