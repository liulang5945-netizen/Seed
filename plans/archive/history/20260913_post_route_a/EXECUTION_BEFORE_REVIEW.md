> 历史快照：来源 f9825943；不授予执行许可。当前顺序见 [活动推进方案](../../../active/roadmap/03_CURRENT_EXECUTION.md)。

# Seed / Taiji 当前推进方案

> 更新：2026-09-13。审查基线：`102b81e1`；P5.2b 缺陷修复记录见 §4.1。本文件是唯一执行顺序来源。
> 当前唯一下一步：**路线 A（profile 表征修复）**，见文末；路线 C 已执行完毕（`transfer_signal_constant`）。
> 本轮交付为结果复审和方案重组；以下新增阶段是建议计划，尚未冻结实验判据或获得产品行为切换许可。
> [本轮证据复审](../../../reference/M5_POST_P5_2_REVIEW_20260913.md)记录事实、推断与限制；[整理前执行快照](../../../archive/history/20260913_plan_reorganization/EXECUTION_BEFORE_REVIEW.md)保存原始阶段流水。

## 1. 项目所处位置与本轮目标

长期目标仍是 Taiji 拥有持续状态、记忆、行动选择、异质群体协作和继承式成长；Seed 提供产品、权限和环境执行。既有实验载体用于验证机制，不能成为能力上限。依据为[核心需求](../../../active/roadmap/../TAIJI_CORE_REQUIREMENTS.md)和[原生架构](../../../active/roadmap/../TAIJI_NATIVE_ARCHITECTURE_V1.md)。

当前已经跨过三个阶段：K 轴限定范围晋级；知识来源的受控内容收益验证；Workbench 小型模拟的合同执行与程序动作预测验证。接下来的主要缺口是把预测、执行、真实结果、群体学习和已准入状态串成可恢复的闭环。

本轮整理目标是：把分散的成绩归入明确证据范围，保留失败教训，清除活动计划中的过时指令，并为下一阶段写出依赖、交付物、验收和停止条件。

## 2. 已完成成果及其生效范围

| 工作线 | 已完成事实 | 当前可以使用的资产 | 尚未覆盖 |
|---|---|---|---|
| K 轴晋级 | P4.14 联合课程 4/4，rollout review、runtime 附着和独立批准完成；scorecard v8 的 promotion gate 为 true | K/G 联合状态、恢复/回滚合同、显式附着接口 | 默认行为采用、附着持久化、通用任务能力 |
| G 侧持续学习 | P4.11 两 seed、P4.12 9/9、P4.13 两阶段 9/9 通过 | parent-relative 表征与联合可行域投影 | 开放任务及长程无界累积 |
| P5.1–P5.1e | 内容迁移、对比辨别、语义 encoder 注入、构造同预算内容收益通过；P5.1b 语义改写试验失败 | governed corpus、锚定 encoder、程序学习和对照框架，以及辨别力不足的失败证据 | 各阶段指标不能跨不同实验直接排序 |
| P5.1f | 真实语料试验按原判据失败，门 2/4/5/7/9 未过 | 预算、准入与饱和指标的失败证据 | 不可将其改写为已通过或仅删除失败项 |
| P5.1g | 真实语料配额对照九门通过；trial a-gate 0.651376，超频率基线 0.284376 | calls 配额采样、trial 测量路径、语料内容收益证据 | 两臂仍未准入；不能声称真实语料知识已进入产品有效状态 |
| P5.2 | 合同 Gate 九门通过；180 动作真实路径、30 场景文件撤销恢复；readout holdout/a-gate 1.0 | scripted 场景、真实 Workbench 执行器、readout、trace 投影 | 执行循环尚未消费 readout 预测；群体评估 groups=0 |
| 工程与 CI | 有前端、桌面、provider、插件、语言识别、事务和恢复工程资产 | 复用现有接口与相关回归测试 | 历史局部检查不构成当前 HEAD 全仓 CI 通过 |

晋级以[宣布](../../../reference/M5_K_PROMOTION_DECLARATION_20260912.md)和[scorecard v8 报告](../../../../reports/taiji_m5_k_axis_scorecard_v8_20260912.json)为准：限定五类合成载体，附着 opt-in、进程内状态，默认 chat/workbench 路径保持原样，S 为 control-only evidence，结构成长未触发。

P5.1g/P5.2 报告中的 `can_promote=false` 是各自实验的权限边界，不撤销此前 K 轴晋级；K 轴的 `can_promote=true` 也不向新阶段传递许可。`growth_admitted=false` 仍适用于本轮已有成果。

## 3. 本轮复审修正的三个关键判断

### 3.1 真实语料收益与产品准入分开验收

P5.1g 的 sourced/placebo 工具词表不相交，placebo a-gate 为结构性零；超 per-tick 频率基线的收益提供了额外证据，但仍属于所用词表、语料和测量路径。两臂 consolidate 的 `admitted=false`、`rolled_back=true` 仍须保留。

预注册披露已用校准探针观察 a-gate 并选择 hidden 64/250 epochs，后续正式泛化验收必须使用新的独立测试集。当前报告的“准入不可达”仅适用于已测配置与预算，不能作为所有模型结构上的不可能结论。

### 3.2 P5.2 执行正确与模型控制执行尚有接线间隔

现有 runner 的 `_execute_scene` 遍历 `scene.steps`，使用脚本的动作类型和参数；`_train_readout` 与 `_accuracy` 独立测程序预测。报告证明了合同路径、动作预测以及轨迹格式可用，但没有证明模型的预测决定了那 180 个执行动作。

后续需记录“预测 → intent → request → outcome”的直接关系，并在预测错误时保留失败，不回填脚本答案。模型负责什么、参数绑定器负责什么必须逐字段披露。

### 3.3 Trace 可消费与群体协作证据尚有数据间隔

P5.2 报告的 `interaction_trace.evaluation_summary.groups=0`；每条 episode 都是固定 generator/environment 双 owner。现有 `build_member_evidence` 需要同 context 的 inactive baseline 与 singleton；迁移 learner 还需要 train-only 的群体记录及未知组合。

因此现有 trace 是接口资产，不能直接当作已具备学习信号的群体语料。环境 owner 是审计/执行主体，不能仅通过改名计作第二个认知成员。

## 4. 方向选择与依赖总览

| 方向 | 对当前缺口的作用 | 代价与前置 | 本轮安排 |
|---|---|---|---|
| 深化 P5.2：预测执行、协作归因、在线学习 | 直接推进认知主体、真实行动反馈与异质协作，复用最新资产 | 需要补执行接线、干预对照、未知组合与恢复链 | **主线推荐** |
| 先做插件热插拔/provider watchdog | 改善集成、可用性、故障隔离 | 无法回答模型是否实际选择动作、是否学会协作 | 接口成熟后开展 |
| 立即扩参或结构成长 | 潜在提升容量与专门化 | 当前缺口主要是因果接线/对照，尚无干净扩容必要性证据 | 在实测容量压力成立后重开 |

选择按长期能力上限和依赖判断。主线选“深化 P5.2”；P5.1 的真实语料准入作为部署前硬依赖保留，不能因 trial 读数好而跳过。

| 顺序 | 阶段/工作包 | 交付 | 进入下一阶段条件 |
|---|---|---|---|
| 0 | 证据与工程基线 | 当前复审、命令级 CI 差异清单 | 确认事实边界；相关阻塞定位 |
| 1 | P5.2a 预测驱动执行 | 冻结合同、执行接线、隔离课程报告 | 执行动作来源可追踪，真实任务收益与保持成立。**已执行 `predictive_execution_insufficient`**（[预注册](../../../reference/M5_P5_2A_PREDICTIVE_EXECUTION_PREREGISTRATION_20260913.md) / [报告](../../../../reports/taiji_p5_2a_predictive_execution_20260913.json)：八门过（安全违规 0、恢复/篡改拒绝过），final model `0.5` < 冻结阈值 `0.65` 且与 frequency `0.5` 持平；失败模式 = 训练分布的 patch→undo 关联在无 undo 新组合上触发 bind 失败 ×3 + 过早 create 被合同拦截 ×3，goal_reached 仅 6/12） |
| 2 | P5.2b 群体因果语料 | baseline/singleton/group 干预矩阵 | 同 context 配对成立、非空群体记录、无标签泄漏。**已执行 `group_causal_corpora_supported`；入场审计曾推翻该结论，缺陷已修复并按原判据重跑通过（见下）**（[预注册](../../../reference/M5_P5_2B_GROUP_CAUSAL_CORPORA_PREREGISTRATION_20260913.md) / [报告](../../../../reports/taiji_p5_2b_group_causal_corpora_20260913.json)：九门全过；4 个 family-specialist readout 作真实可干预成员，12 context × 11 cells × 2 重复 = 264 episodes 真实合同执行；**1 对 admitted group（member-a+member-d）interaction `0.2222` 且 holdout 同值复现**，5 对因 `low_confidence` 如实拒绝；成员 profile 4/4；组合机制修订（fallback → dual-predict-select）已在预注册披露——纯 fallback 下未调用成员无事件使 pair cell 结构性无法成形；wall 22.8s。**复核修正：九门门 3/门 6 判据只检查「是否存在」而不检查「干预是否真的发生」，故漏检 block-0 context 的零步伪成功；该 admitted group 无效**） |
| 3 | P5.2c 未见组合迁移 | transfer learner 候选与真实执行对照 | 学习组合优于冻结对照且旧能力保持。**原 gate 门 3 结构性不可满足（阻塞）；根因已定位并实证，已另立新预注册 P5.2c′ 并执行完毕**：预注册已冻结（[P5.2c 原预注册](../../../reference/M5_P5_2C_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md)），但 4 成员下 6 个 pair 全被 `train_only_candidates` 估计，未见组合面为空。审计根因：P5.2a `lang_confirm` 模板（context 100/104/108）目标 == 初始状态，9 个干预 cell 零步执行即判成功（空事件 episode），**P5.2b 的 `member-a+member-d` admitted group（interaction `0.2222`）为伪成功**。该零步缺陷已按 §4.1 三层修复并重跑通过；**未见组合面问题经实证根因是 `_estimate_pair` 只需任一 train context 齐备四 cell 即估计，故「增加成员数」无效（4/5/6 成员实测 unseen 均为 0），唯一可行机制是把指定 pair 的联合 cell 从 train 全分区整体移除**。详见[审计报告](../../../../reports/M5_P5_2C_ENTRY_AUDIT_P5_2B_DEFECT_20260913.md)与[P5.2c′ 新预注册](../../../reference/M5_P5_2C_PRIME_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md) |
| 3′ | **P5.2c′ 未见组合迁移（已执行）** | 非空未见组合面上的迁移判定 | **已执行 `transfer_no_gain`**（[结果报告](../../../../reports/M5_P5_2C_PRIME_UNSEEN_COMBINATION_TRANSFER_RESULT_20260913.md) / [机器报告](../../../../reports/taiji_p5_2c_prime_unseen_combination_transfer_20260913.json)）：入场审计 7/7 成立（`P*` joint 在 train **0** 次、holdout **8** 次，train 160 / holdout 88，移除台账 16 条）；九门 **8 过**，仅门 9 `transfer_and_budget` 未过——`object_gain=0.0` vs `required=1.65`（最强对照 `strongest_singleton=1.5` + MARGIN 0.15），4 context 中 0 个优于最强单体；**门 8 校准通过**（符号一致率 `1.0`，绝对误差中位数 `0.3417`，但样本量 1 须标注）。根因：`P*`（`member-a+member-d`）两成员**功能冗余**（成功面均为 block-0 的 `6/6`，`108` 上 realized interaction `−2.0`），联合无超额收益；而该 cohort 中唯一有真实互补的 `member-b+member-c`（12/24 > 任一单体 6/24）**已被 train 观测**，不在未见面上。4 成员 profile contribution 全等 `0.5`，learner 无成员级区分特征。机械门全过 ⇒ 结论有效，非接线事故 |
| 3″ | **P5.2c″ 未见组合迁移（设计修复，已执行）** | 两个已验证互补的未见组合面上的迁移判定 | **已执行 `transfer_signal_constant`**（[结果报告](../../../../reports/M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_RESULT_20260913.md) / [机器报告](../../../../reports/taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json)）：**路线 C 三项目标全部达成**——(1) 未见组合数 **2**（P5.2c′ 为 1）；(2) 两对均**实测** `DISJOINT`（`a+c`={0}∪{2}、`b+d`={1}∪{0}），且**刻意保留冗余对 `a+d` 在观测面内**使 learner 能观察到「部分组合无超额收益」；(3) block-3 如实量化（66 次尝试 0 成功，60 次 `contract_intercepted`，区分面 `0.75`）。入场审计 `passed=true` / `conditions=[]`；train **144** / holdout **88**，移除 **32** 条（`8×2×2`），4 成员各保留 **16** 条 singleton（属「未见」非「无支持」）。九门 **8 过**，仅门 9 `transfer_and_budget` 未过：`object_gain=-0.5` vs `required=1.65`（最强对照 `strongest_singleton=1.5`），两对场景增益 `a+c=-0.5`、`b+d=0.0`，`any_held_out_pair_positive=false`，**两对均 0/4 context 优于最强单体**。门 8 校准通过且**优于前身**（2 样本、符号一致率 `1.0`、绝对误差中位数 `0.25`，P5.2c′ 为 1 样本/`0.3417`）。**结论比 P5.2c′ 更精确也更负面**：两个互补组合都拿不到增益 ⇒ 障碍不在「组合选得不好」而在更底层；`contribution_uniform=true`（4 成员 contribution 全 `0.5`）⇒ 表征缺陷仍在，归路线 A。**归因纪律：本负结果只支持「该表征下不可行」，不支持「迁移不可行」** |
| 4 | P5.2d 在线结果回写 | 多轮 online → child → 恢复/回滚链 | 学习增益、保持、预算、幂等和中断恢复通过 |
| 5 | P5.1h 真实语料准入 | 独立数据与 retention/准入实验 | 如需消费真实语料 child，必须先通过产品准入 |
| 6 | runtime 行为采用评审 | shadow → opt-in canary → 持久化提案 | 实际使用的全部 artifact 准入通过，独立批准 |
| 7 | P5.3 插件/provider；硬件与视觉 | 稳定接口上的产品完善 | 上游合同冻结，发布验收与对应依赖通过 |

P5.2a–d、P5.1h 是本轮用于拆分交付的建议编号，不是已执行或已冻结的实验。相邻阶段逐门进入，不同时启动多个训练分支。若第 1 阶段需要真实语料 child，第 5 阶段应提前成为该 child 的前置；合成 P5.2 研究可先沿现有隔离资产推进。

**既有技术债**（历史测试失败、架构边界违反、Git 遗留）不进入本表的执行顺序，单独登记在
[技术债登记册](../../../active/roadmap/05_TECH_DEBT_REGISTER.md)。该册只登记与量化，**主线收尾后才进入处置阶段**；
如需判断「某次全量测试失败是否由本次改动引入」，先按该册 §1 的引用图方法论证归属，不要靠重跑手感。

### 4.1 P5.2b 零步缺陷的三层根因与修复记录

审计发现 P5.2b 的 admitted pair 是空事件 episode 造成的伪成功。修复过程中暴露出**同一缺陷的三个独立层次**，逐层修复才恢复真实证据。记录于此，避免后续同类误判。

| 层 | 缺陷 | 检出方式 | 修复 |
|---|---|---|---|
| L1 判据 | `cell_completeness` 只数 episode 条数，`real_execution` 只问「某处是否有过执行」，无法发现「整个 cell 零执行」 | P5.2c 入场审计 | 新增 `_intervention_reality()`：任何非 baseline cell 零步即 `interventions_happened=false`，并让门 3/门 6 消费该证据；`(F,F)` baseline 零步是 treatment 本身，豁免 |
| L2 任务 | `lang_confirm` 目标态 == 初始态，且 python 可由扩展名解析，tick 0 即满足 | 重跑后 offending cell 仍全零 | 加 `requires_explicit_language_override` 字段，`_goal_reached` 要求已记录的 `user_override`；生成期加 `_assert_nontrivial_goals` 守卫 |
| L3 绑定与重置 | ① `_bind` 的 `editor.set_language` 丢弃 `user_override`，任务变成**永不可达**；② `restore_language_state(None)` 是 no-op（`if not payload: return`），上一 episode 的 override 泄漏到下一 episode | 单 episode 探针：`pre=False, post=False` 且同环境内第二个 episode 即零步 | ① `_bind` 在 `requires_explicit_language_override` 时透传 `user_override=True`；② 重置改为显式空 payload |

**关键教训**：仅让任务「不能太容易被满足」是不够的——必须同时验证**改造后仍可被满足**。L2 的首次修复把「伪成功」换成了「必然失败」，而改动后的矩阵在门禁上与「已修复」无法区分（零步消失的原因不同）。因此新增回归测试 `tests/taiji_native/test_intervention_reality_gate.py` 同时钉住两个方向：`pre=False`（非平凡）与 `post=True`（可满足）。

修复后按**原判据**重跑（未放宽任何门）：`zero_step_episodes_total=0`、offender `0`、九门全过、`interventions_happened=true`。各模板成功面呈现真实差异：`lang_confirm`（100/104/108）由 member-a/member-d 及 a+b 达成，`patch_persist`（101/105/109）仅 member-b，`create_persist`（102/106/110）由 member-c 及 c+d，`header_override`（103/107/111）全 0——后者失败原因为 `contract_intercepted:language_evidence_ambiguous`，属合同层合理拦截，非缺陷。所有 context 的 `(F,F)` baseline 成功数为 0，确认任务确实需要成员介入。

## 5. P5.2a：让模型预测实际控制动作

**研究问题**：保持 Workbench 权限合同不变时，程序 readout 的预测能否驱动多步任务，并获得可归因的真实结果收益？

**输入与复用**：

- 复用 `ProceduralSequenceLearner`、锚定 encoder、`WorkbenchEnvironment`、`WorkbenchActionRequest.from_action_intent`。
- P5.2 的 40/12/8 场景留作历史/开发基线；新增最终测试按项目、模板、语言证据冲突和动作组合分组，先登记重复率与隔离规则。
- 冻结父 checkpoint、动作词表、可见 observation、参数 schema、审批策略与资源预算。当前 readout 输出动作类型，不把它描述为已有任意参数生成能力。

**实施步骤**：

1. 先完成零训练接线审计：列出预测 API、recurrent state、停止/拒绝出口、参数来源以及禁止读取的标签字段。
2. 写预注册，冻结候选、对照、评分与数据拆分。场景参考动作只用于监督/评分，在线执行路径不得读取 `scene.steps` 作答案。
3. 预测 capability 后用当前世界状态绑定合法参数；记录哪些参数仍是受控模板提供。缺证据、schema 不符或越权时安全停止。
4. 实际 outcome 更新下一步可见状态；处理文件变化、语言歧义、撤销 token 失效和失败后重观察。
5. 完成保存/独立恢复预检后，按冻结课程执行，产出命令、摘要、trajectory 与失败分类。

**对照与验收**：

- 脚本 oracle 只作合同/任务可达性上界；冻结模型、lesion、train-only 的频率基线作能力对照。
- 每个执行动作可追溯到本轮预测与该时刻 snapshot；运行时不能因预测不匹配而替换为标准答案。
- 主指标为任务最终状态成功率与安全违规数；动作准确率、步数、延迟、恢复率为分账指标。
- 在新模板上保留实质差异任务，避免仅靠目标模板识别和 tick 查表解题。
- 旧场景、低证据拒绝和 undo 保持；checkpoint 恢复前后在相同观察下的选择与结果一致。
- 新阈值从 validation 校准与任务意义确定，测试前冻结；P5.2 的 0.15 动作准确率门不自动迁移为任务成功率门。

**停止点**：无法隔离脚本答案；参数必须由外部 oracle 决定才能完成任务；预测收益只来自模板泄漏；安全边界或恢复失败。需要改变动作所有权或任务定义时先提交失败证据，再讨论。

## 6. P5.2b：建立可识别的群体因果对照

**研究问题**：哪些实际认知成员在相同任务条件下存在超出单体的联合贡献？

1. 盘点可干预成员的真实状态、输入、输出与关闭方式；区分认知成员、宿主执行器、审计 owner。暂不规定多个成员已存在。
2. 对每个 context 固定初始世界、父状态、任务、预算，建立 inactive baseline、每个 singleton、候选组合、必要的成员/连接 lesion。
3. `context_id` 绑定配对条件；同组各臂必须相同。不同 context 允许任务差异；成员名称不得携带语义角色标签供 learner 偷看。
4. 全部 outcome/resource/recovery 指标来自实际执行；缺失对照记“不可估计”，不赋零或伪造记录。
5. train-only 构建 `InteractionGroupMemberEvidence` 与 `InteractionGroupRecord`；holdout 评分独立，不能进入 `observe_records`。
6. 留出“成员已见、组合未见”的验证/测试集合；未知成员没有 singleton 支持时应拒绝，单列为边界。

**验收**：配对完整、profile/群体记录非空、revision/digest 一致、组间差异可测、资源成本真实；测试集不进入拟合。只有一种固定双 owner trace 的输入不能通过。

**停止点**：实际认知成员不足、对照无法执行、所有结果恒定或不能留出新组合。此时先设计可干预任务/成员，不创建虚构群体也不靠扩大重复场景获得“样本量”。

## 7. P5.2c：从 trace 学习未见组合的候选价值

**复用路径**：`build_member_evidence → observe_members/observe_records → candidate/select → Workbench 实际执行`。迁移 learner 预测候选，不承担权限准入。

- 冻结无学习、最强单体、随机组合、固定组合，以及匹配资源的组合策略。必要时加入 train-only 简单回归基线，辨别复杂机制是否有额外价值。
- 每条候选在执行前绑定 parent digest、成员集合、预测收益、不确定性和成本；执行后独立计算真实收益。
- 主张“协作”需在预注册任务上超过最强单体；主张“泛化”需在未参与拟合的组合/context 上成立。
- 报告预测校准、任务成功、每类保持、最坏组、失败率和预算；多个 seed 是重复测量，不能自动当作独立模型/任务。
- checkpoint 保存后用全新进程恢复，重现候选和选择；污染 lineage、holdout 记录及未知成员必须拒绝。

**出口**：机械失败回合同；无因果信号回 P5.2b；有信号但无迁移收益回模型/特征归因；收益、保持、成本均通过才进入在线回写。任何失败都保留原报告与预注册。

## 8. P5.2d：真实 outcome 驱动连续学习

**目标**：把实际执行结果变成可归因的学习更新，并在后续新任务与旧任务上检验作用。

1. 复用 `InteractionGroupOutcomeFeedback` 与 online learner，核验实际 `Outcome`、来源 split、终态、置信度和父状态。
2. 在线新经历更新候选 child，测试/holdout 仅评分；候选选择时间必须早于实际 outcome。
3. 设计多轮 online 与固定学习预算对照，逐轮评估新能力、旧能力、预算和真实执行失败。
4. 明确失败经历的审计保存与可训练准入区别；现有 online 模块对失败/低置信度等反馈拒绝训练，改变该规则需独立预注册。
5. 在 observation、selection、execution、feedback、checkpoint 边界中断恢复；验证重复 outcome 不重复学习、重复 token 不重复执行、stale parent 拒绝。
6. 同时验 learner rollback 和环境事务状态；恢复模型不能被误报为已撤销外部副作用。

**验收**：更新后的收益优于冻结对照、旧类保持，状态/事件可恢复，成本可接受；未达准入的 child 留作实验候选。此阶段不自动开放结构成长。

## 9. P5.1h 与产品采用：保留未解决的准入任务

P5.1g 的真实语料 retention 和产品 admission 仍是部署缺口。下一次相关实验先区分：训练预算不足、表征容量不足、数据分布差异、目标/词表冲突以及保持协议问题；当前结果不足以选定其中一个为唯一根因。

- 使用独立开发/测试划分，新增词表可比较的内容对照，避免收益只由 sourced 覆盖目标词表解释。
- 保留 calls 配额，同时审计总 example/update、有效参数、训练 wall、推理成本、checkpoint 字节；calls 相等不代表全部计算成本相等。
- 可选更高容量/成熟序列机制或保持目标改进，但单轮要冻结主要变化，先做小规模 validation 归因。
- 同时报告 trial 与 admitted 两套结果；通过 trial 性能门但 admission 失败时，不授予产品采用许可。
- 不放宽历史 admission 线掩盖失败；如旧门与应用目标不符，单独写测量修订论证、新版本和批准记录。

产品采用遵循：依赖验收 → shadow 旁路测量 → opt-in canary → 持久化/恢复合同 → 默认行为评审。每步记录采用 artifact、数据范围、用户可见变化、回滚方案。此前 K 轴 opt-in 附着不覆盖这些新许可。

## 10. 后续工程与长期成长

**P5.3 插件/provider**：复用已有 registry、客户端扩展、授权目标绑定和 watchdog。先做接口兼容、生命周期、动态加载/卸载、故障隔离、降级/回滚，性能与权限分账；外部 provider 输出不得算作原生学习收益。具体工作项由前序稳定接口反推。

**结构成长**：当同父代固定容量最强方案在真实多任务干扰/长序列上出现可复现瓶颈，再与同最终有效容量 fixed-large 做公平比较。新结构零影响出生、继承父状态、具有可测贡献，经过 lesion、保持、预算与恢复门。P4.7 关闭的是当时任务/对照下的容量假设，不取消长期 CR-4。

**硬件与产品视觉**：硬件可用后做 CPU 数值一致性、保存恢复与吞吐基准，资源采购/设备切换单独决策。视觉、安装、托盘/任务栏与发布在运行接口稳定后集中验收。阻塞实际研究的客户端故障优先修复。

## 11. CI、训练前检查与成果管理

### 11.1 CI 实况与处理顺序

2026-09-13 查询到最近远端运行 [34579613959](https://github.com/liulang5945-netizen/Seed/actions/runs/34579613959)，对应 `c7bbd389`，并非当前 `102b81e1`：两条 Linux test job 在 Ruff 失败；Windows cancelled；frontend、Docker、两种启动 smoke 成功。

P5.1g/P5.2 预注册引用的 `4 failed / 1199 passed / 6 skipped` 与 `mypy 61` 是历史局部基线；当前 CI 配置的 core mypy 上限为 0。原 Gate 中“static_four_checks=true”不能替代远端 CI 成功，也不能自动抬高 CI 阈值。

后续实现前按实际命令和版本重新建立失败清单：测试名称、错误位置、父提交表现、是否与工作包相关、修复归属。优先修复新增回归和阻塞主线的存量错误；相关必要检查不过则不启动对应正式训练。剩余债务可保留为显式未完成项，发布/默认采用前须满足正式 CI。

本轮只跑文档链接、现有项目身份/单执行入口测试及 diff 检查；不把未执行的全仓测试写成通过。旧测试若硬编码过期阶段标题，应更新为现行标题并保留“只有一个执行入口”和链接可解析约束。

### 11.2 每次训练的入场条件

- 冻结父/child 路径，验证目标目录可写、剩余磁盘及原子保存可用；不得覆盖 parent。
- 保存零步完整状态，再由独立进程恢复；核对参数、优化/学习状态、RNG、数据游标、预算和 lineage。
- 验证中断恢复、错误 parent/digest/schema 拒绝与 rollback；没有全套状态时先补合同。
- 固定 train/validation/retention/final-test 的来源、摘要、隔离规则；validation 调参必须留记录，final-test 未提前用于选择。
- 阈值、样本规模与预算在正式结果前冻结。先执行 scoped lint/format/type/test，再启动 runner。
- 记录代码提交、环境、命令、开始/结束状态与报告；终止/预算超限必须留下可审计失败出口。

### 11.3 交付与清理规则

每阶段交付一份预注册、一份必要机器报告和必要 checkpoint/manifest；更新本文件的状态表与唯一下一步，然后提交。具体 debug 流水进 archive，不在首页累计长篇“用户确认/当前下一步”。

本轮已把旧计划和入口快照归档并重定位链接。旧失败 JSON、权重与被引用实现保留；只有核实绝对路径、引用及恢复方式后才清理临时产物。不得批量删除未知目录、语料或 checkpoint。

## 当前唯一下一步：路线 A —— profile 表征修复（须新预注册）

**已决策**：用户选择「**先 C 后 AB**」。路线 **C（实验设计修复）已执行完毕并出结论** `transfer_signal_constant`（[结果报告](../../../../reports/M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_RESULT_20260913.md)）。

### 路线 C 执行结果（已收尾）

| 量 | P5.2c′ | **P5.2c″** |
|---|---|---|
| 未见组合数 | 1 | **2** |
| 持有对性质 | `EQUAL`（唯一冗余对 `a+d`） | **`DISJOINT` × 2**（`a+c`、`b+d`） |
| train / holdout | 176 / 88 | **144 / 88** |
| 移除 episode | 16 | **32** |
| 观测对 | 5 | **4**（含刻意保留的冗余对 `a+d`） |
| 门 8 样本量 | 1 | **2** |
| 门 8 绝对误差中位数 | `0.3417` | **`0.25`** |
| `object_gain` | `0.0` | **`-0.5`** |
| 最强对照 | `strongest_singleton 1.5` | `strongest_singleton 1.5` |
| 优于最强单体的 context | 0 / 4 | **0 / 4（两对均如此）** |
| block-3 | 未量化 | **已量化（区分面 0.75）** |
| 九门 | 8 过 | **8 过** |
| 结论 | `transfer_no_gain` | **`transfer_signal_constant`** |

**「先 C 后 A」已被验证是正确排期**：若先做 A 再做 C，则「表征修复的收益」与「仪器修复的收益」混在同一轮，**归因不可分离**。先 C 保住了这个可分离性——路线 A 的效果可以直接对照 `transfer_signal_constant` 这个干净基线。

**同时必须诚实指出**：`transfer_signal_constant` **比 `transfer_no_gain` 更负面，不是更正面**。两个互补对都拿不到增益，说明障碍比「组合选取不当」更深。这加强了路线 A 的必要性，但也意味着**路线 A 的预期不应被抬高**。

### 唯一下一步：路线 A

**靶点**：`member_evidence.contribution_uniform = true` —— 4 个成员 contribution 全为 `0.5`，learner 无法区分成员。这是唯一被两轮实验（P5.2c′ 与 P5.2c″）**同时**指向且**尚未触碰**的缺陷。

**硬约束**：

- **必须先新预注册**，不得沿用 P5.2c′/P5.2c″ 的任何判据、阈值或对照。
- `growth_admitted=false`、`can_promote=false` 贯穿。
- 不得反向改判据、放宽判据凑通过、覆写既有报告、重跑挑结果。
- **保留 P5.2c″ 作为干净基线**：路线 A 必须与本轮的 `transfer_signal_constant`（同为 2 个 `DISJOINT` 持有对、同 144/88 划分）对照，否则无法归因。

**路线 B（pair 关系项形式）** 在路线 A 出结果后再定 —— 因为 `contribution_uniform` 会同时污染 profile 项与 pair 关系项，A 未修则 B 的效果不可解释。

**候选归因材料**：[P5.2c′ 下一步决策提示](../../../active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md)（三条路线 A/B/C 的完整论证；其中路线 C 已由本轮完成）。

---

**已决策**：P5.2b 干预真实性缺陷（L1/L2/L3 三层）已按 §4.1 修复并按原判据重跑通过；P5.2c′ 与 P5.2c″ 均已按冻结预注册执行完毕。

**纪律**：命中预注册停止点 ⇒ 回模型/特征归因，**本 gate 不因归因结论被追认为通过**；不修改/不覆写 P5.2c 原预注册、入场审计报告、P5.2b 报告、P5.2c′ 报告与本轮 P5.2c″ 报告；不重跑挑结果；`growth_admitted=false`、`can_promote=false` 贯穿。

---

### 附：P5.2c′ 根因分析（保留为路线 A 的输入证据）

**P5.2c′ 结果**：`transfer_no_gain`（[结果报告](../../../../reports/M5_P5_2C_PRIME_UNSEEN_COMBINATION_TRANSFER_RESULT_20260913.md)）。九门 **8 过**，仅门 9 未过；**入场审计 7/7 成立、门 8 校准通过、机械门全过** ⇒ 结论有效。`object_gain = 0.0`，`required = 1.65`（`strongest_singleton 1.5` + MARGIN `0.15`），优于最强单体的 context **0 / 4**，wall `21.5s` / 900s。

**根因（三层，互相独立地指向同一结论）**：

1. **`P*` 两成员功能冗余**：`member-a` 与 `member-d` 成功面完全相同（均为 block-0 的 `6/6`，其余全 0），`P*` 联合面 `6/6` 未超出任一单体；context-108 上 `realized_interaction = −2.0`。
2. **互补 pair 全在观测面上**：`member-b`→block-1、`member-c`→block-2 互不覆盖，`b+c` 得 `12/24`（> 任一单体 `6/24`）——但已被 train 观测。移除动作按**字典序索引** `(0,3)` 选取，未按互补性选取，故落到冗余 pair。
   - **归因的关键更正**：互补 pair 有**两个**（`b+c`=`[0,6,6,0]`、`b+d`=`[6,6,0,0]`，均 `12/24` > 单体 `6/24`），不是「唯一一个」。二者是同一结构事实的两个实例：**`member-d` 的能力面与 `member-a` 相同**（均只覆盖 block-0），故 `a+d` 冗余、`b+c`/`b+d` 互补。因此 `transfer_no_gain` **不可**读作「该 cohort 无联合增益潜力」——失败在**表征层与设计层，不在能力层**。
3. **profile 无区分度**：4 个成员 contribution 全为 `0.5`，learner 没有成员级特征可外推到未见组合，预测只能回落到已观测 5 对的平均交互水平附近（预测 `−0.158` vs 实测 `−0.5`，方向正确但幅度收缩）。

**这第 3 条是路线 C 修不掉、且两轮实验都复现的缺陷**（P5.2c″ 实测 `contribution_uniform=true`）⇒ 归路线 A。

**路线 D. block-3 不可达（未解决，已量化）**：`contract_intercepted:language_evidence_ambiguous` 使 1/4 的 context 对所有组合均为 0（P5.2c″ 实测 66 次尝试 0 成功、区分面 `0.75`）；提升功效需先解决 block-3 任务可达性（可能触及 P5.2a 任务定义）。

---

### 附：P5.2b 修复前的问题陈述（保留为历史归档）

以下为修复前的原始问题描述，**其中的「成员数 ≥5」方案已被本轮探针证伪**（见 P5.2c′ 预注册 §2），保留以便对照。

**已决策**：用户选择选项 (2) P5.2b。**P5.2b 曾报 `group_causal_corpora_supported`（提交 `0abf463f`），该结论已被入场审计推翻。**

**P5.2c 预注册已按建议冻结**：[预注册](../../../reference/M5_P5_2C_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md)（九门、三态、对照、纪律、必报分账齐全）。**但执行在接线阶段即被入场审计阻塞**（[审计报告](../../../../reports/M5_P5_2C_ENTRY_AUDIT_P5_2B_DEFECT_20260913.md)）：

1. **门 3 结构性不可满足**：4 个成员下 `C(4,2)=6` 个 pair 被 `train_only_candidates` **全数观测**，`select(..., unseen_only=True)` 返回 `None`——「未见组合」集合为空。`_pair_features` 仅支持 pair，三元组 raise，故无扩容余地。
2. **根因（P0，污染 P5.2b 结论）**：P5.2a `_validation_tasks()` 的 `lang_confirm` 模板（`index % 4 == 0`，context `100/104/108`）**目标状态 == 初始状态**——`goal_files = initial` 且 `.py` 的语言 selection 自动成立。实测这些 context 各 22 episodes 中 **20 个零步执行**：tick 循环首行 `_goal_reached` 即真，`steps=[]` → `_project` 产出**空事件 episode**，仍记 `success=True`。context 100/104/108 的 11 个干预 cell 只有 `none`（1 步失败）与 `member-a`（2 步真实执行）有动作，其余 9 个 cell 全部零步判成功。
3. **被推翻的具体结论**：`member-a+member-d` admitted group 的 `interaction = 0.2222` 完全由空事件 cell 生成——`member-d` 单体与 pair cell 都在 block-0 **未执行却记成功**，`(T,T)−(T,F)−(F,T)+(F,F)` 的差值来自「干预生效前任务已满足」，**不是超出单体的联合增益**，而是伪成功。其 `contribution = −0.6667` 同样不可用。
4. **九门判据缺陷（需独立修正）**：`real_execution` 只断言「存在已执行动作 + provenance 合规 + 安全违规 0」，`cell_completeness` 只断言 episode 计数——**两者都只检查存在性，不检查干预是否真的发生**，故对「整 cell 零执行」零检出能力。

**建议交付（按优先级）**：

- **P0-A 判据加固**：`real_execution` 增加**非空事件断言**（每个非 `(F,F)` cell 必须 ≥1 事件）；`cell_completeness` 增加**零步 episode 计数 = 0（除 `(F,F)` 外）**；两门均须对 `(F,F)` baseline 的失败语义做显式例外声明。
- **P0-B 任务修正**：重定义或排除 `lang_confirm` 模板，使 validation context 全部为**非平凡持久目标**（P5.2a 已对 undo 类做过同类排除，但漏了「语言 selection 自动成立」这条路径）。
- **P1 未见组合面**：扩充成员数（≥5，使 pair 面 > 已观测数），或在 train/holdout 划分上**留出整对 pair 不参与估计**；任一方案都需**新预注册**。
  **（2026-09-13 更正）**「扩充成员数 ≥5」**已实证无效**：`_estimate_pair` 只需任一 train context 齐备四 cell 即估计，成员数 4/5/6 实测 unseen pair 均为 0。
  唯一可行机制是**把指定 pair 的联合 cell 从 train 全分区整体移除**，据此另立 P5.2c′ 新预注册。
- 修完后**重新预注册并重跑 P5.2b**，再谈 P5.2c；P5.2c 旧预注册保留冻结原貌，不追溯改写。

**纪律**：不修改、不覆写 P5.2b 报告与预注册（保留为失败证据）；不重跑 P5.2b 挑结果；`growth_admitted=false`、`can_promote=false` 贯穿；临时探测脚本用毕即删。

P5.2a 已按冻结预注册执行并如实落 `predictive_execution_insufficient`（提交 `eff6e1d1`）：八门过、门 9 迁移失败。只读归因 recon（临时脚本已清理）已把失败定位到机制层，**逐场景证据推翻了初步假设**：

- **失败不是 patch→undo 强关联**（训练分布中 apply_patch 后仅 50% 跟 undo、50% 结束），而是 **cue→模板类型识别漂移**：模型在训练 4 模板上完美（train 1.0），但对六种新组合的目标文本落在模板决策边界外——F1（patch 类）被预测为 read→resolve→undo（T0 语言模式与 T1 undo 尾部的混合泄漏）×3、F3（create+语言复合）忽略了 create 直接走语言序列被合同拦截 ×2、F5/F6（已有文件任务）被错误泛化为 list→create 被 preview 冲突拦截 ×2。7 个失败 100% 是动作选择错误，参数绑定与合同路径零失误（provenance/安全审计全过）。
- 频率基线 0.5 的成功是幸存者路径：位置表（read→resolve→set_language→apply_patch）恰好是 F1/F2 参考序列的「无害超集」，apply_patch 的参数绑定从目标状态派生，命中即达成。
- **无停止 token 不是本次失败原因**（0/7 失败是 goal 达成后的多余动作）——停止机制候选（STOP token/完成分类头）针对性低；goal 状态注入 cue 会改变训练合同且不触及类型识别漂移。

**三选项决策进展：零训练探针已执行，选项 1 被否定。** 探针（临时脚本已清理，零训练）：把 12 个 final 场景的 goal 文本替换为携带训练模板类型签名词汇的显式措辞，现有 readout（确定性重训）在显式 cue 下成功率 **0/12**（current 措辞 4/12）——显式词汇使 cue 嵌入偏离训练 goal 分布，预测全面退化为 list/undo 泄漏。**结论：瓶颈不是 cue 措辞，而是 GRU+线性 readout 学到的是「具体 goal 文本 → 模板」的记忆映射，不具备语义级模板泛化结构**——与 P5.1f（retention 天花板）、P5.2a final（位置先验退化）构成同向证据链：固定容量序列机制在新分布上受限。选项 1（P5.2a-v2 cue 结构化）无证据支持，关闭。
