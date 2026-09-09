# M5.K1 预注册：技能组合课程 canary（A8 主证据第一步）

> 起草日期：2026-09-09。K 课程定义见 [架构 v2 §6.1](../active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md)：结构化语义 → 世界转移 → Workbench 只读意图 → 隔离执行，每阶段未见组合 + 真实 outcome。本文是 K1 canary 的预注册；确认后实现。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 可证伪假设

> 在同一 runtime/checkpoint 上串联四阶段（语义 → 转移 → 只读意图 → 隔离执行），对**未见任务组合**（训练中未出现过的 目标×文件×语言 组合），完整链产出的意图经真实隔离执行的成功率显著高于冻结语义对照，且语义 lesion 使成功率崩塌——即组合能力来自学习的语义表征，不是 planner 常量或执行运气。

## 2. 串联路径（全部复用既有资产）

| 阶段 | 资产 | 本 canary 的用法 |
|---|---|---|
| 1 语义 | `StructuredSemanticLearner`（semantic_training.py） | 在**已见**组合（文件×语言×目标）上训练 observation→facts/Goal/ContentPlan |
| 2 转移 | `StructuredSemanticTransitionLearner`（semantic_transition.py） | 在多步状态序列上训练（K1 先最小化：单步验证接口贯通） |
| 3 意图 | `NativeReadOnlyIntentPlanner.propose`（read_only_intent.py） | 消费 1+2 输出，对未见组合产出 `ActionIntent`（M3.R3 native 臂路径） |
| 4 执行 | `execute_workbench_intent` → `project_workbench_outcome_for_internalization` → ledger（M5.S6B 链） | 真实隔离执行，真 outcome 按 S6B 政策准入（成功分级/失败负） |

新增胶水（唯一新写）：**参数化任务组合生成器**——生成 (文件, 语言, 目标) 三元组网格，train/holdout 按组合维分割（holdout 组合的每个元素都已在训练出现，但**三元组本身未见**）——这是"组合泛化"的严格定义，避免 S3 的"holdout 太易"教训。

## 3. 臂与判据

| 臂 | 语义 learner | 预期 |
|---|---|---|
| A full-chain | 训练 | 未见组合执行成功率高 |
| B frozen-semantic | 冻结（零训练） | 成功率显著低 |
| C semantic-lesion | 训练后 lesion | 成功率崩塌（因果证明）|

Gate（全部预注册）：

1. **主指标**：A 臂未见组合真实执行成功率 ≥ 0.8，且 A−B ≥ 0.3、A−C ≥ 0.3；
2. 每阶段的既有 Gate 保持（语义 fact F1、unknown fail-closed、planner stale-snapshot、执行 approval/digest/undo）；
3. 真实 outcome 全部经 S6B 政策准入（成功分级、失败负 reward），reward_variance > 0；
4. checkpoint/fresh-restore、core mypy、相关回归不退化。

## 4. 停止线

- canary 先行（单 seed、小网格）；A 臂成功率 < 0.5 → 先归因（语义阶段？planner？执行？），不调阈值硬凑；
- 不引入 provider/联网/真实客户端写入；执行全部在进程私有临时工作区；
- `can_promote=false` 固定——K1 证明的是组合链闭合，不是 A8 完成。

## 5. 明确不做

- 不在本 canary 训练字节预测/fabric（K 的量尺是任务成功率，不是 BPB）；
- 不做多 seed formal（canary 通过后另行预注册）；
- 不把 provider 文本当语义特征。

## 6. 执行记录（2026-09-09，canary 已运行，Gate 未通过）

实现：`scripts/training/eval_taiji_m5_k1_skill_composition.py`；报告：`reports/taiji_m5_k1_skill_composition_20260909.json`（`status=failed`，`can_promote=false`）。

### 6.1 相对原设计的实现修正（均为 harness/语料设计，非阈值调整）

1. **holdout 全覆盖**：`_arm_outcomes` 初版以 holdout[0] 作锚点导致 `typescript_05` 不产生行；改为以最后一个训练世界作初始锚，全部 6 个 holdout 观察逐行预测→提议→真实执行。
2. **拆分防泄漏重构**：转移语料的 before-fact keys 只编码块身份（missing/python/typescript/rust），不含文件标识；训练序列简单取反会在 ts/rs 块内与 train 发生 `input_digest` 碰撞，同一 tuple 传 dev/test 双重违规。dev/test 改为「(before 块, event 文件) 有序对与 train/dev 完全不相交」的未见文件序列。
3. **合成内容去污染**：初版内容 `record {lang} {index}` 撞上 C# 的 `record` content pattern（强度 0.55 > 扩展名 0.14），ts/rs 被判为 csharp，工具链翻转不可观测。改为各语言惯用内容（`interface`/`fn`/`def`，与 M3.R1 fixture 同形）+ 根目录四份 manifest（extension+content 0.69 < resolved 阈值 0.72，需 manifest 证据补足）+ 按索引变长 pad（percept 特征只含 one-hot+标量，byte_length 是同语言文件唯一的区分来源）。
4. **Stage 1/Stage 2 分离**：臂由**语义 learner**（§3 表）定义，故 planner 消费 Stage-1（`StructuredSemanticLearner`，percept→facts/Goal/ContentPlan，无 before 依赖）的事件接地世界；Stage-2 转移按 §2「K1 先最小化：单步验证接口贯通」降为接口贯通检查（`_stage2_interface`，报告型指标，不门控 planner）。否则 A−B≡A−C≡0，臂对比失效。Stage-1 语料中 missing 文件 percept 特征全同（失败读取无内容），只允许 missing_00 入语料；训练序列改为 `(m00, py00-03, m01, ts00-03, rs00-03)`，制造两个不同语言的 recover→file 转移以打破转移头的块级捷径。

### 6.2 结果

| 量 | 值 |
|---|---|
| A unseen（ts05-07 真实执行成功率） | **0.0**（Gate 需 ≥0.8） |
| A 已见组合 | python inspect 成功且 S6B 准入 ✓；rust clarify 真实执行成功 ✓（resolve 成功的投影受「stale evidence」限制，见 6.4） |
| B frozen / C lesion | 全部 `conflict` fail-closed，0.0 ✓（对照行为正确） |
| Stage-2 接口贯通 | 6/6 步 world 良构、契约完好 ✓ |
| Stage-1 拟合 | fact 0.00196 / goal 3.4e-06 / content 1.8e-06（收敛） |

链路闭合验证：语义→意图→执行→S6B 准入全链在**已见**组合上真实跑通；planner 对目标/世界不一致 fail-closed；B/C lesion 臂崩塌符合预期。

### 6.3 归因（A=0.0 < 0.5 停止线触发，按 §4 归因而非调阈值）

失败精确定位在 **Stage-1 fact head 的属性组合**，下游全部正确：

- 实测 holdout ts05（typescript × 工具链可用）：`language::typescript=0.830` ✓（语言属性从 event one-hot 正确组合）；**`toolchain::available=0.135` ✗**（其因果特征——percept 中的 toolchain scalar——值为 1.0）；`toolchain::missing=0.865`（错）。世界因此声明 toolchain missing → goal 读出 clarify-toolchain（goal/content 线性读出对该世界本身一致）→ planner 检测世界与观察不一致，拒绝（fail-closed 正确）。
- 机制：训练 registry 中 (python ↔ available)、(ts/rs ↔ missing) **完全相关**。fact head 是线性欠定问题——存在完美组合的正确解（权重只放 toolchain scalar），但 delta-rule 动力学在完全相关数据下把状态属性绑到了身份特征上。goal head 的合取泛化同理继承边际相关。
- **训练数据无法修复**：训练 registry 里 typescript 工具链必然不可用（这正是组合翻转的设计），(× ts-event@available) 交叉在训练中不可达；harness 层面无解。

### 6.4 已知次要限制（不阻塞）

- resolve 类成功 outcome 的 S6B 投影被「latest WorkBench evidence is stale」拒绝（投影链以 read evidence 为中心）；unseen Gate 只要求 read intent，不受影响。
- missing_03（recover-target）无 planner 路由，正确失败。

### 6.5 判定与唯一后续假设

**K1 canary 判定：未通过（honest fail）**。组合链条的工程闭合已验证，失败源于线性语义 learner 在完全相关训练边际下无法组合 (身份 × 状态) 属性——这是「课程结构属性」问题的语义层版本（同 M4 结论族：分布相关性支配学习结果）。

**唯一建议下一步（K1.1，需新预注册）**：类型化 fact–feature 绑定——每个语义 fact 声明其因果特征来源（从 `WorkbenchObservationSchema.feature_names` 与 fact key 的谓词名规范派生掩码），fact head 按 fact×feature 掩码读取；goal/content 读出掩码到**状态类 facts**（read/language_state/toolchain/target），身份 facts（language）只进世界与 content.semantic_slots（planner 一致性检查已消费后者）。在该设计下，未见三元组的状态投影与已见 inspect 状态向量**逐维相同**（组合发生在 fact 层），goal 读出在分布内 → A 臂可组合且 B/C 仍应崩塌。确认前不改 `taiji/semantic_training.py`、不重跑 K1。

## 7. K1.1 执行记录（2026-09-09，用户确认后实现，Gate 通过）

用户确认 K1.1 方案后实现并重跑 K1。改动两处：

1. **`taiji/semantic_training.py`（`SEMANTIC_TRAINING_VERSION 1→2`）**：`StructuredSemanticLearner` 新增可选 `fact_feature_masks`（fact key → 允许的特征索引；提供时必须**精确覆盖**全部 fact keys，fail-closed）与 `readout_excluded_facts`（goal/content 读出输入中被排除的 facts）。fact head 在初始化、每个训练 epoch 的 delta 更新后、checkpoint 恢复后都重新执行掩码（掩码区权重恒零，delta 更新对允许区的贡献不受影响）；goal/content 读出在 fit 与 predict 两侧一致地对输入乘掩码（被排除列零初始化且零更新，不引入额外自由度）。未提供掩码时行为与 v1 完全一致（向后兼容）；掩码与排除项进入 checkpoint 往返。M2.R3 回归 5 passed、M3.R1 回归 2 passed、目标文件 mypy 0。
2. **K1 脚本（report version 2）**：新增 `_typed_fact_feature_masks` 从 schema 规范派生绑定——`language::*` 读 `language:{value}` one-hot、`language_state::*` 读 `selection:{value}`、`read/target/toolchain/diagnostics` 各读同名 scalar 指示器；`readout_excluded_facts` = 全部 `language::*` 身份 facts。绑定描述写入报告 `typed_binding` 字段。

### 7.1 结果（同预注册判据，未放宽；单 seed canary）

| 量 | 值 | Gate |
|---|---|---|
| A unseen（ts05-07 真实执行成功率） | **1.0**（3/3 全部 `resolved → workspace.read → success=1.0 → S6B 准入`） | ≥0.8 ✓ |
| A−B / A−C | **1.0 / 1.0**（B/C 全部 `conflict` fail-closed） | ≥0.3 ✓ |
| rows | 每臂 unseen 计数 3 | ✓ |
| Stage-1 拟合 | fact 0.0365（掩码约束下的收敛值）/ goal 1.2e-05 / content 2.8e-06 | — |
| Stage-2 接口 | 6/6 良构 | ✓ |

A 臂 6 行完整行为：ts05/06/07（**未见三元组**）inspect 成功+准入、python_05（已见组合）成功+准入、rust_05（已见组合）clarify→resolve 真实执行成功（resolve 成功的 S6B 投影仍受「stale evidence」限制，为 §6.4 已知次要限制）、missing_03 无路由正确失败。报告：`reports/taiji_m5_k1_skill_composition_20260909.json`（`status=passed`，`can_promote=false` 固定）。

### 7.2 结论与边界

- **K1 canary 通过**：四阶段链（类型化语义 → 转移接口 → 只读意图 → 隔离执行）在未见任务组合上以真实执行成功闭合，组合能力依赖学习的语义表征（B frozen / C lesion 崩塌为 0）。这证明的是「组合链闭合 + 语义组合因果」，不是 A8 完成，也不是多 seed 稳健性。
- 类型化绑定的边界如实声明：fact→feature 掩码由**课程 harness 依 schema 规范派生**，是先验的表征约束而非学出的绑定；learner 学到的是掩码内的校准（阈值/读出）。该约束的普适化（无 schema 先验时如何学出绑定）是后续课程问题，不在 K1 结论内。
- 下一形式化步骤需另行预注册：多 seed/order formal（K 量尺=任务成功率），以及 K2（更深的组合维度）设计。formal 前不改判据、不引入新变量。
