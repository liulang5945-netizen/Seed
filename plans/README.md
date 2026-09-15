# Seed / Taiji 计划与架构入口

> **2026-09-15 用户纠正后的安排**：未来架构只保留一份 [完整 VISION](reference/VISION_FUTURE_TECHNOLOGY.md)，按计算结构、语言生成、训练适应、工程与验证有序展开，不切换主线。[整模型能力评价与用户验收](active/roadmap/07_MINI_MODEL_DELIVERY.md) 取代立即做预览的安排：阶段结项必须检查真实模型输出，基本对话/问答/上下文和代表能力达标后，才冻结最小版本交用户自由提问验收。

> 更新：2026-09-14；总计划审查基线 `cd8e4acc`。执行顺序仅由[当前推进方案](active/roadmap/03_CURRENT_EXECUTION.md)决定。

## 先看项目全局

[总阶段与能力晋级地图](active/roadmap/01_SCOPE_AND_PHASES.md)回答项目处于哪一阶段、各阶段还缺什么；[晋级与发布门禁](active/roadmap/02_GATES_AND_CI.md)回答什么时候可以晋级；[当前执行](active/roadmap/03_CURRENT_EXECUTION.md)只决定下一工作包。

- **已认可**：M0～M3 有限定基础证据；2026-09-12 K 轴已限定晋级。
- **当前主线**：M5 知识与身体中的协作/选择轴，B0 已收束，进入 WP-1 决策；HANDOFF-M4 为局部修法，不是大阶段 M4。
- **下一能力评审**：B1/B2 正式验收并满足保持/恢复等共同门后，评审协作/选择轴，不直接宣布 M5 完成。
- **M5 整体退出**：知识准入、协作/选择、在线/身体必达项闭合后独立评审；随后分别做 M6 产品与 M7 发布验收。
- **仍在总图中**：M4.V2 结构成长、生命调节/记忆/自主学习等长期目标；M8 真设备加速不阻塞 CPU 交付，未被本轮解冻。

本轮修订保留下方 B0 证据索引，不重跑实验或改历史报告。计划按“实验完成 / 能力轴晋级 / 大阶段退出 / 产品发布”分账，不再按局部实验次数判断项目进度。

## 当前阶段

**K 轴已限定晋级；路线 C、A 均已完成。B0 评分与可达性审查已完成，结论是目标当前不可达，故暂停正式训练，进入任务与估计目标的修改审阅。**

边界：K/G 为 opt-in、进程内状态；P5.1g 为未准入 trial；P5.2a 已接真实预测执行但泛化门未过；修复后的 P5.2b 当前 groups=0、rejected=6。A 的预测两档不代表正确排序，跨任务覆盖不代表同任务协作。结构成长未触发。

[B0 设计包](reference/M5_B0_MEASUREMENT_AND_REACHABILITY_AUDIT_20260913.md)给出统一测量字典、旧值复算、可达性上界与复合任务候选；[机制与预检续篇](reference/M5_B0_MECHANISM_AND_TASK_PRECHECK_20260913.md)给出根因与可落地规格；[路线 B 预注册草案](reference/M5_B0_ROUTE_B_PREREGISTRATION_DRAFT_20260913.md)（**未冻结**）给出 H1/H2/H3 分层判据。

本轮关键事实：六个 pair 在四个未见 context 上 **0 次**超过全体单体 oracle，`oracle_all_cell = oracle_singleton = 1.5`，三种候选参照**全部不可达**（缺口 2.15 / 0.15 / 1.15）。根因（续篇）：组合机制是**优先级回退链**（每 tick 只执行第一个绑定成功的成员），**仲裁类机制上界恒 ≤ 0**；24 个 pair×context 单元**零交错轨迹**；现行任务可支撑的参照上限仅 **0.35**，低于最佳可部署单体 0.5 ⇒ 换参照救不了，**任务必须改**。

探针结果（[交接可行性探针](reference/M5_B0_HANDOFF_PROBE_RESULT_20260913.md)）：先复现冻结矩阵 **11/11**；**三层已完全分离** —— 任务层找到达标形态 `create_and_override`（四单体全失败、`k=4/4`、**三种参照全部 feasible**、可支撑参照上限 1.85），但同一候选上**六个 pair 仍全部失败、`interleaved=0`** ⇒ **机制层是唯一阻塞**；根因是交接触发条件是"绑定失败"而非"无进展"，第一个成员独占 episode。据此提出最小修法 **M1（待决策 D5，未实施）**。

反事实测量（[机制修法反事实测量](reference/M5_B0_M1_COUNTERFACTUAL_RESULT_20260913.md)）：先验证候选面**可满足且顺序强制**（反序时 `set_language` 绑定成功但执行失败），再测**六个修法变体**——**五个失败、一个成功**。**`m4_failure_handoff`（+21 行、2 处替换）**：冻结面 **0 回归、2 改善**（`a+b`/`a+c` 各 0.25→0.50），候选面 **`interleaved=4/4`**、**同参照增益 `+2.000` > 1.65**（最严参照）⇒ **H2 协作主张首次可达**。

伪影审计（[M4 伪影审计](reference/M5_B0_M4_ARTIFACT_AUDIT_20260913.md)）：**四项全过** —— ① 特异性：正增益**只**出现在 `create_and_override`（冻结面 −0.5、`dual_requirement` −2.0、`create_then_patch` +0.000）；② 干预真实性：5 次面测量 `interventions_happened` 全 true；③ 机制 lesion：增益随交接消失且四单体全败；④ **3 个种子偏移下候选面增益恒 +2.000，且无种子在冻结面制造增益**。冻结面 2 个改善**逐步归因为真实交接**（基线里 `member-b`/`member-c` 一次机会都没有，M4 才放行）。**M4 仍未实施**；新增停止原因 `all_members_blocked` 需门禁语义审查。

加固轮（[M4 加固轮结果](reference/M5_B0_M4_HARDENING_RESULT_20260913.md)）闭合审计 §6 中**三项无需决策的残余风险**：规模扩到 **12 context / 6 结构变体**（六个变体**各自** `+2.000`，冻结规则 **0/6 为正、零交错**）、种子扩到 **5 个偏移**（恒 `+2.000`）、`all_members_blocked` 消费面**清单化为 11 个文件**，并证明 `contract_intercepted` 在两条规则下**计数相同**（非 M4 引入，增益对比未被污染）。〔**2026-09-15 收窄**：该"相同"只对**未被让位触及**的 `preview_ValueError` 成立（逐格 `36=36`）；`language_assessment_unavailable` 恰恰**会被让位修掉**（`create__observation` 冻结侧 72 次 → M4 侧 0），故它不是恒等项而是机制证据——原判据按子键写，见[N2 冻结版](reference/M5_B0_N2_STOP_REASON_PREREGISTRATION_FROZEN_20260915.md) I5〕。`.h`/`cpp` 的"预期被合同拦截"被**与规则无关的脚本证据推翻并改判**，同时加上"预期与观测不符即拒出报告"的 fail-closed 校验。**界限说明（本轮最重要）**：六个变体只换表面（扩展名/语言/内容形状），**共享同一组合结构** `create + override`，结果指纹**只有 1 种** ⇒ 目前只证**表面稳健**，**结构稳健**须由 T1/T2/T3 重跑（登记 **N1**）；`all_members_blocked` 的门禁语义预注册为落地前置（登记 **N2**）。两次全量重跑 JSON **字节相同**。

N1 结构空间探针（[N1 结构空间结果](reference/M5_B0_STRUCTURE_SPACE_RESULT_20260913.md)）回答"**M4 的收益是否只在 `create + override` 这一种结构下成立**"。先实测发现**计划里的 T1/T3 在冻结 binder 下根本建不出来**（`workspace.create` 绑定 `goal_files[main_path]` ⇒ 创建即达标、`apply_patch` 随后拒绝；`_bind` 把每个动作都解析到 `main_path` ⇒ 第二目标路径不可达），T2 也**没有成员间证据通道**（`predict_episode` 只有一个参数，元组内是同一张量的重复）。故改为**枚举目标谓词可表达的全部结构**：3 内容路线 × 4 语言路线去平凡格 = **11 格 / 22 context / 3 种子**，有效性**在合同之下**判定。**结果：正增益覆盖 `create` 行全部三条语言路由**（各 `+2.000`、`interleaved 0→2`、`member-a+member-c` 两个成员都真实执行），其余 8 格 `0.000`，`patch` 行两规则同 `−2.000`（没有把不可达变成可达），**零回归、预测与观测 11/11 一致**；且 M4 修复的是**两种不同的冻结失败形态**（在不存在的文件上评估语言 ⇒ 合同拦截；首成员独占 ⇒ 步数耗尽）。**界限（本轮同等重要）**：三格**共享同一结果指纹** ⇒ 它们是一个结构因素经由三条路线，不是三次独立确认；可主张的宽度是 **1 个结构因素（存在性前提）× 3 条语言路线 × 2 种失败形态**，而**内容侧的第二种联合必需结构在冻结 binder 下不存在**（登记 **N1a**：要跨内容结构须先决定改 binder）。本轮亦自查出第一版有效性门**绕过 `policy_for`** 的缺陷并修好 —— 该修正**改变了结果**（`create__mismatch` `0.000`→`+2.000`，3 个假矛盾归零），预测函数两版之间未改。两次重跑 JSON **字节相同**，30 项合同测试通过。

**当前状态（2026-09-15）：WP-1 出口、WP-2 语义冻结、WP-3 落地与五条出口验收全部交付。** 取值定案（全取上限档）：D1 = 主判据 `all_singleton_oracle`（`required 1.65`；口径更正见[草案 §10.2](reference/M5_B0_ROUTE_B_PREREGISTRATION_DRAFT_20260913.md)：实测 `+2.000` 本来就是对该参照算的，且 `+2.000 = ceiling_gain 2.0` ⇒ **增益维度零余量**）｜D2 = create 行全部 3 条语言路由｜D3 = 逐格公布 `required`｜D4 = 旧 144/88 降级、新面独立冻结｜**D5 = 落地**｜N2 = 落地前预注册并把 `all_members_blocked` 定为一类终局结果｜**N1a = 开**（WP-6）｜N1a-2 = T2 证据通道单列。
- 序 0 **WP-1.5 扩面**：四条出口全过（6 context / 7 种子、7/7 偏移为正、两跑字节相同）⇒ 冻结前必过门已通过；**闭合 L4/N1b/N1c，未闭合 L1**（指纹仍 3/11）。
- 序 1 **WP-1 出口**：[路线 B 冻结版预注册（binder v1）](reference/M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md) 已落盘。
- 序 2 **WP-2**：[N2 冻结版](reference/M5_B0_N2_STOP_REASON_PREREGISTRATION_FROZEN_20260915.md) + [双向测试](../tests/taiji_native/test_b0_n2_stop_reason_semantics_contract.py)；消费面 **17 文件 / 10 处判断点（4 live gate + 6 测试断言）**（WP-2 落盘时 15/8，WP-3 的规则版本封条测试成为第 16 个消费者 ⇒ J9；出口②复现守卫再加第 17 个 ⇒ J10），扫描器 fail-closed 且 `drift.clean=true`。**计数恒以[重扫报告](../reports/taiji_b0_n2_stop_reason_disposition_20260915.json) 为准，不得由正文填**。
- 序 3 **WP-3 落地**：两处替换已写进 gate（与反事实**逐字节相同**）。**出口①已证**（`variant_is_identity=true`、`added_lines=0`、未重绑定冻结属性）；**出口②③已证**，但**第一次重跑被判无效**——落地使探针基线臂变成被测规则本身（两臂同源 ⇒ `delta ≡ 0`、"无回归"为同义反复），已改为反向还原 revision-0 基线 + 两臂同一函数即失败 + `arm_provenance`，有效重跑与封存报告**逐字段相同**（详见[落地结果 §3](reference/M5_B0_M4_LANDING_RESULT_20260915.md)）；**出口④已证**（落地后全量 **1428 passed / 0 failed / 6 skipped**，对基线 1408/0/6 失败集合无新增；消失的 5 个测试名全是有意重命名，逐个列在落地结果 §4）；**出口⑤已证**（八份 revision-0 产物哈希逐份重算不变 + 六支仪器默认输出改指新名的覆写防护）。同批补做 **gate 端到端实测**：默认路径直跑 24.8 s、九门无失败、报告 `rule_revision=1`，且封存报告哈希不变（`groups=2`/`rejected=4` 只是 gate 表面量，不是能力主张）。范围限定：该组合规则**只存在于 gate 脚本**，`taiji/` 无对应实现 ⇒ **不证明产品机制已支持协作**。

**执行顺序（工作包制）**：WP-1 决策 → WP-1.5 扩面 → 冻结版预注册 → WP-2 语义 → **WP-3 落地（五条出口：① 反事实 ≡ 实现（已证）② 冻结面 0 回归 / 2 改善（已证：反事实探针在已发布源码上 `regressions=0`/`improvements=2`，结构空间 11 格逐字段复现封存报告）③ 达标面 `interleaved>0` 且逐格 `> required`（已证：`interleaved=6`、`+2.000 > 1.65`、7/7 偏移）④ 全量失败集合不新增，**基线 2026-09-15 复采 = 1408 / 0 失败 / 6 跳过**（旧 851/27、949/27 均作废，那 27 项系本地沙箱批量删除守卫伪影）⑤ 历史报告只加"规则版本"指针不改数字 + 哈希封条与默认输出覆写防护）→ WP-4 B1 入场（五件探针**各在其所属 `rule_revision` 下**全绿：预检/交接/加固为 revision-0 归档结论、**落地后不重跑**；**反事实与结构空间为 revision-1 实测复现** + 训练前六门）→ WP-5 下游链**；**WP-6 binder 议题**挂在 WP-3 之后。

**授权状态**：用户 2026-09-15 解除"先不开始实际推进"限制并指示持续推进 ⇒ 计划制定、只读测量、WP-2/WP-3 落地与测试均在执行范围内；**B1 训练已授权并完成**（`representation_discriminative`），**WP-5a B2 选择门已执行 = 负结果**（机械门全过、G4/G5 不过 ⇒ `failed` / `no_task_benefit`）。**仍未授权/仍未做**：binder 修改（WP-6）、产品采用与默认行为切换、push（远端由用户处理，`gh` 未认证 ⇒ 禁止任何"CI 已绿"表述）。**下一执行节点 = CAP-0 整模型能力基线**（07 §5.1：主线结项点先做 CAP-0）；其清点段只读、无需决策，基线段须先冻结评价集（决策节点）。不直接加特征列重跑旧 Gate。详见[推进计划修订](reference/M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md)。

## 最新证据入口

| 证据 | 结论与使用范围 |
|---|---|
| [K 轴晋级宣布](reference/M5_K_PROMOTION_DECLARATION_20260912.md) / [scorecard v8](../reports/taiji_m5_k_axis_scorecard_v8_20260912.json) | K 轴限定晋级与 13 项 promotion gate；不等于默认产品采用或结构成长 |
| [P4.14 联合课程](reference/M5_K_P4_14_JOINT_COURSE_PREREGISTRATION_20260911.md) | K continuation → post-K 重构输入 → G 求解器，4/4 cell |
| [P5.1 内容迁移](reference/M5_P5_1_SOURCED_KNOWLEDGE_TRANSFER_PREREGISTRATION_20260912.md) | 词表重叠下的受控内容收益 |
| [P5.1b 语义改写](reference/M5_P5_1B_SEMANTIC_PARAPHRASE_TRANSFER_PREREGISTRATION_20260912.md) | 失败证据；小语料下辨别力不足 |
| [P5.1c 对比训练](reference/M5_P5_1C_CONTRASTIVE_DISCRIMINATION_PREREGISTRATION_20260912.md) | 内容家族辨别改善 |
| [P5.1d encoder 注入](reference/M5_P5_1D_SEMANTIC_ENCODER_INJECTION_PREREGISTRATION_20260912.md) | 锚定语义 encoder 接入与回归 |
| [P5.1e 构造同预算](reference/M5_P5_1E_SAME_BUDGET_CONTENT_BENEFIT_PREREGISTRATION_20260912.md) | 构造载体内容收益 |
| [P5.1f 真实语料失败](reference/M5_P5_1F_REAL_CORPUS_SAME_BUDGET_PREREGISTRATION_20260912.md) | 配额口径、准入、指标饱和与超预算问题 |
| [P5.1g 配额对照](reference/M5_P5_1G_REAL_CORPUS_QUOTA_BUDGET_PREREGISTRATION_20260912.md) / [报告](../reports/taiji_p5_1g_real_corpus_quota_budget_20260912.json) | trial 内容收益成立；两臂 admission 仍失败 |
| [P5.2 Workbench 合同](reference/M5_P5_2_WORKBENCH_SIMULATION_CONTRACT_PREREGISTRATION_20260912.md) / [报告](../reports/taiji_p5_2_workbench_simulation_contract_20260912.json) | 真实合同执行与动作预测分别验证；尚未闭合预测执行与群体迁移 |
| [P5.2a 预测驱动执行](reference/M5_P5_2A_PREDICTIVE_EXECUTION_PREREGISTRATION_20260913.md) / [报告](../reports/taiji_p5_2a_predictive_execution_20260913.json) | `predictive_execution_insufficient`；八门过、门 9 迁移失败；失败全为动作选择错误 |
| [P5.2b 群体因果语料](reference/M5_P5_2B_GROUP_CAUSAL_CORPORA_PREREGISTRATION_20260913.md) / [报告](../reports/taiji_p5_2b_group_causal_corpora_20260913.json) | 修复后语料合同九门通过；当前 groups=0、rejected=6，旧 admitted pair 无效 |
| [P5.2c′ 未见组合迁移结果](../reports/M5_P5_2C_PRIME_UNSEEN_COMBINATION_TRANSFER_RESULT_20260913.md) | transfer_no_gain；未见面只有一对且冗余，非一般选择能力结论 |
| [P5.2c″ 未见组合迁移结果](../reports/M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_RESULT_20260913.md) | 路线 C 完成，两对留出、旧表示常量；block-3 只量化未修可达性 |
| [P5.2c‴ 表征修复结果](../reports/M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_RESULT_20260913.md) / [本轮复审](reference/M5_POST_ROUTE_A_REVIEW_20260913.md) | 路线 A 完成，秩和区分度提升但收益未过；评分参照和任务上界需审查 |
| [B0 设计包](reference/M5_B0_MEASUREMENT_AND_REACHABILITY_AUDIT_20260913.md) / [审计报告](../reports/taiji_b0_measurement_reachability_audit_20260913.json) | 统一测量字典 v1 草案、32/32 字段复算一致、三种参照全部不可达、五个手算用例；结论=先改任务 |
| [B0 机制与预检续篇](reference/M5_B0_MECHANISM_AND_TASK_PRECHECK_20260913.md) / [预检报告](../reports/taiji_b0_task_reachability_precheck_20260913.json) | 组合机制是**优先级回退链**；仲裁上界恒 ≤0；24 单元**零交错**；逃生通道闭式要求；T1/T2/T3 具体规格 |
| [B0 交接可行性探针结果](reference/M5_B0_HANDOFF_PROBE_RESULT_20260913.md) / [探针报告](../reports/taiji_b0_handoff_feasibility_probe_20260913.json) | 先复现冻结矩阵 **11/11**；**三层完全分离**：任务层已找到达标形态（`k=4/4`、三种参照全 feasible），**机制层是唯一阻塞**；提出最小修法 **M1（待决策 D5）** |
| [B0 机制修法反事实测量](reference/M5_B0_M1_COUNTERFACTUAL_RESULT_20260913.md) / [测量报告](../reports/taiji_b0_m1_counterfactual_20260913.json) | 六个变体**五个失败一个成功**；**`m4_failure_handoff` 不回归（0 回归 / 2 改善）且 `interleaved=4/4`、同参照增益 +2.000 > 1.65** ⇒ **H2 协作主张首次可达**（**当时**未实施，待 D5；D5=落地已于 2026-09-15 完成 ⇒ 本文与报告属 `rule_revision=0` 证据） |
| [B0 M4 伪影审计](reference/M5_B0_M4_ARTIFACT_AUDIT_20260913.md) / [审计报告](../reports/taiji_b0_m4_artifact_audit_20260913.json) | **四项全过**：特异性（正增益只在该面）、干预真实性（无惰性 cell）、机制 lesion（增益随交接消失且四单体全败）、**3 种子下增益恒 +2.000**；冻结面 2 个改善**逐步归因为真实交接** |
| [B0 M4 加固轮结果](reference/M5_B0_M4_HARDENING_RESULT_20260913.md) / [加固报告](../reports/taiji_b0_m4_hardening_20260913.json) | 规模 **12 context / 6 变体**、种子 **5 个**、`all_members_blocked` 消费面 **11 文件清单**；六变体各自 `+2.000` 而冻结规则 0/6；`.h` 合同预期被证据推翻并加 fail-closed 校验。**只证表面稳健，不证结构稳健**（N1）；两次重跑字节相同 |
| [B0 N1 结构空间结果](reference/M5_B0_STRUCTURE_SPACE_RESULT_20260913.md) / [探针报告](../reports/taiji_b0_structure_space_probe_20260913.json) | 枚举目标谓词可表达的**全部 11 格结构**（22 context / 3 种子，有效性在合同之下判定）：**收益不限于** `create + override`（`create` 行三条语言路由各 `+2.000`，修复两种冻结失败形态），其余格 `0.000`/`−2.000`、零回归、预测 11/11 一致。**界限**：三格共享同一结果指纹 ⇒ 独立结构因素仍为 **1（存在性前提）**；T1/T3 在冻结 binder 下不可表达（N1a）。重跑字节相同，30 测试通过 |
| [B0 N2 停止原因处置审查（已被冻结版取代）](reference/M5_B0_N2_STOP_REASON_DISPOSITION_20260914.md) / [处置报告](../reports/taiji_b0_n2_stop_reason_disposition_20260914.json) | **审查面来源，非权威计数**：本文写「实时消费面 14 个文件 / 只有 5 个是真判断点」，**已由 [N2 冻结版](reference/M5_B0_N2_STOP_REASON_PREREGISTRATION_FROZEN_20260915.md) §0 更正为现值 17 文件 / 10 处判断点（4 live + 6 测试断言）**；「`contract_intercepted` 两规则计数相同」亦收窄（见该表下一行）。保留价值：规范定义草案、6 条不变式底稿、脚本排除自身避免自污染的设计 |
| [B0 N1 扩面重跑（WP-1.5，结果 §10）](reference/M5_B0_STRUCTURE_SPACE_RESULT_20260913.md) / [扩面报告](../reports/taiji_b0_structure_space_probe_wide_20260915.json) | **每格 6 context（66 总）/ 7 种子偏移**，同一支探针经 `--contexts-per-cell` / `--seed-offsets` 驱动（**默认值未改** ⇒ 无旗标仍逐字节复现 20260913 报告；**限定**：该字节复现只对 revision-0 源码成立，M4 落地后无旗标重跑改按 gate 的 `RULE_REVISION` 选臂 ⇒ 测量字段仍逐字段相同、但 `rule_delta`/`arm_provenance` 键会变）。三格各 `+2.000`、交错 **`0 → 6`**、**7/7 偏移为正**、`unexplained`/`regress` 皆空、11 格全 `measurable`、两跑字节相同、单轮 **12 分钟**。**闭合 L4（样本规模）**；逃生通道 `required` 由 **2/2 → 5/6**（k 维度首次有余量），9 个「格 × 参照」全 feasible。**未闭合 L1**：指纹仍 3/11 ⇒ 独立结构因素仍为 1，只能靠 WP-6；`ceiling_gain` 与 context 数无关（恒 2.0）⇒ **增益维度仍零余量** |
| [**B0 N2 冻结版预注册**](reference/M5_B0_N2_STOP_REASON_PREREGISTRATION_FROZEN_20260915.md) / [重扫报告](../reports/taiji_b0_n2_stop_reason_disposition_20260915.json) / [双向测试](../tests/taiji_native/test_b0_n2_stop_reason_semantics_contract.py) | **FROZEN 2026-09-15**：`all_members_blocked` = **合法让位后的终局结果（非成功）**，8 条不变式 I1–I8 逐条绑测试；**更正底稿数字**：消费面现值为 **17 文件 / 10 处判断点（4 live gate + 6 测试断言）/ 8 文件仅记录**（WP-2 落盘时 15/8——扫描器把本测试自身登记为 J8，未处置即退出码 1；WP-3 的封条与出口②守卫再加 J9/J10 ⇒ 17/10）。**自查纠偏**：原写「`contract_intercepted:*` 两规则计数相同」被自己的双向测试否证——只有未被让位触及的 `preview_ValueError` 恒 `36=36`，`language_assessment_unavailable` 在 `create__observation` 冻结侧 72 次、M4 侧归零，**那正是让位修复的对象** |
| [**B0 路线 B 冻结版预注册（binder v1）**](reference/M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md) | **FROZEN 2026-09-15**：H2 主判据 = `gain_vs_all_singleton_oracle`（参照 `1.5`、`required 1.65`、`margin 0.15` 来源已写明）；任务规格 = `create` 行 × 3 条语言路由（含模板、成员族、`STEP_CAP`、索引规则）；**逐格阈值推导表**（`required k` 5/4/2 vs `available 6`、`ceiling_gain 2.0`、`max_clearable 1.85`，由 `reference_requirements()` 计算而非手填）；数据隔离（旧 144/88 降级）；G1–G6 分层门；**§6 反例面**（`patch` 行必须保持 `−2.000`）；§7 允许/禁止表述；§14 未闭合项诚实清单。**只适用 binder v1** |
| [**B0 WP-3 落地结果（rule_revision=1）**](reference/M5_B0_M4_LANDING_RESULT_20260915.md) / [落地后探针报告](../reports/taiji_b0_structure_space_probe_m4landed_20260915.json) / [无效首跑物证](../reports/taiji_b0_structure_space_probe_m4landed_degenerate_20260915.json) | HANDOFF-M4 已写进 gate（两处替换与反事实**逐字节相同**）。**出口①已过**：`variant_is_identity=true`、`added_lines=0`、锚点消失、冻结属性未重绑定、四个 revision-0 变体照旧拒绝构造。**出口②③的第一次重跑被判无效并如实登记**：落地后探针的 baseline 臂取的就是已发布规则，两臂同源 ⇒ `delta ≡ 0`、`regresses: none` 均是同义反复；已改为**按 gate 自报 `RULE_REVISION` 选臂**（`build_reverted()` 反向还原 revision-0 基线、两臂同一函数即 `SystemExit`、报告新增 `arm_provenance`），有效重跑**已**与封存扩面报告**逐字段相同**（`rows`/`validity`/`verdict`/`outcome_distinctness`/`seed_sweep` 五块整体相等，四条 landed 守卫测试钉住）⇒ **出口②③已证**。**出口④已证**（1428 / 0 失败 / 6 跳过，对 1408 基线无新增失败）；**出口⑤已证**（八份哈希重算不变 + 覆写防护，并由 gate 默认直跑 24.8 s 实测证明封存文件未被覆写）。**不证明**：产品机制已支持协作（该规则只存在于 gate 脚本，`taiji/` 无实现）；**不闭合** L1（仍 1 个结构因素） |
| [**WP-6 步骤 1：binder 放宽可行性论证（只读）**](reference/M5_B0_WP6_BINDER_FEASIBILITY_20260915.md) | **未改任何代码/冻结面**。把"改 binder"拆成两件事：T3 只缺 binder（目标谓词已支持多文件），T1 还缺非平凡性门与 tick 顶端早退；给出最小改动面实测计数（`_bind` 7 个非测试调用点、`Task.main_path` 19 个构造点）、**会被打破的 6 组冻结断言清单**、以及一处必须先处置的口径不对称（参照臂绕过 `_bind` ⇒ 只放宽被测路径会偏向参照）。可表达性判据须从"源码文本形状"改成"行为"。v2 预注册待 P3b 结论后开 |
| [WP-5a B2 选择预注册（冻结版）](reference/M5_WP5A_B2_SELECTION_PREREGISTRATION_FROZEN_20260915.md) | B2 特有协议：`select=argmax predicted_interaction`、候选面全 factorial 396 episodes 一次取得 treatment+lesion+对照、可部署对照 = best fixed singleton、wall cap 60 s；估计目标与 G1–G6 继承路线 B 冻结版 |
| [WP-5a B2 选择结果（负结果）](reference/M5_WP5A_B2_SELECTION_RESULT_20260915.md) / [报告](../reports/taiji_b0_b2_selection_20260915.json) | **机械门 G1/G2/G3/G6 全过、G4/G5 双不过** ⇒ `failed`（`no_task_benefit`）。选中 `member-b+member-c`（0.141802）而**唯一增益的是 `member-a+member-c`（+2.000，预测 0.083829）**；所选 pair 逐格增益 `0.0`、自身交错 `0`、lesion 不成立；四单体全 `−1.0`（对照并列垫底）。**结论：表示"能区分"未转化为"选得对"**，H1/H2 均不成立；**协作轴不晋级**。稳定字段两跑哈希相同；**冻结三态表缺 (G4✗,G5✗) 标签**已登记为 `preregistration_gap` |

## 文档职责

| 文档 | 职责 |
|---|---|
| [当前推进方案](active/roadmap/03_CURRENT_EXECUTION.md) | 唯一下一步、依赖、交付、验收、讨论节点 |
| [B0 结果整理与推进计划修订](reference/M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md) | B0 十一轮**结果账本**（F1–F12 已结项事实 / L1–L6 界限）+ **工作包计划**（WP-1 决策 → WP-1.5 扩面 → WP-2 停止原因语义 → WP-3 落地五条出口 → WP-4 B1 入场 → WP-5 下游链 → WP-6 binder 议题，含机时与并行债务轨）；§5 附 **2026-09-15 上限优先修订对照表**（每项决策的上限档建议 vs 低成本档差别） |
| [技术债登记册](active/roadmap/05_TECH_DEBT_REGISTER.md) | A1/A2 已结项，历史 SystemExit 待定位；新评分/CI 约束见最新状态补充 |
| [P5.2c′ 决策历史](active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md) | 用户既定顺序先 C 后 AB；C/A/B0 已完成，D1–D5 决策点待定 |
| [B0 设计包](reference/M5_B0_MEASUREMENT_AND_REACHABILITY_AUDIT_20260913.md) | 统一公式、旧值复算、可达性上界、手算用例、复合任务候选、训练前硬门、CI 阻塞清单 |
| [P5.2c″ 预注册](reference/M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md) | 路线 C 的冻结判据：两个已验证互补的未见组合、block-3 量化、**声明只修测量仪器不修表征** |
| [P5.2c″ 结果报告](../reports/M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_RESULT_20260913.md) | `transfer_signal_constant`；九门 8 过；三项目标全达成；含「先 C 后 A」排期验证 |
| [本轮结果复审](reference/M5_POST_P5_2_REVIEW_20260913.md) | 证据核验、边界修正、方案选择依据 |
| [核心需求](active/TAIJI_CORE_REQUIREMENTS.md) | 认知所有权、协作、行动、记忆、持续学习与成长的长期目标 |
| [原生架构](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) / [继承式成长](active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | 状态、学习与成长机制约束 |
| [Seed 架构](active/SEED_ARCHITECTURE.md) / [架构方向](active/ARCHITECTURE_DIRECTION_2026_08.md) | 产品/认知职责和 legacy 边界 |
| [P4 结果复审](reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md) | 历史容量、表征、学习规则等结论来源 |
| [整理前执行快照](archive/history/20260913_plan_reorganization/EXECUTION_BEFORE_REVIEW.md) / [整理前索引](archive/history/20260913_plan_reorganization/PLAN_INDEX_BEFORE_REVIEW.md) | 完整历史流水；旧“下一步”不授予执行许可 |

## 维护与 CI

active 只保留当前决策与核心约束；reference 保留预注册/结果解释；archive 保存历史；manifests 保存冻结数据合同。失败报告不覆盖，临时产物逐项核验后清理。

当前 f9825943 的 CI 34753643532 上次查询时两条 Linux 已在 Ruff 失败；本轮未查询远端（`gh` 未认证）。按命令级基线：本地 `ruff check .` 原有 1 项 `I001`（即上述失败原因），B0 已修复，现为 **All checks passed**。

**全量套件已两次复采**（`tests/taiji_native/`）：`f9825943`+B0 为 **851 用例 / 27 失败 / 0 错误 / 1 跳过（938s）**；**当前 HEAD `fba517cf` 为 949 用例 / 27 失败 / 0 错误 / 1 跳过（801s）**，且**失败集合与上一基线逐位相同（0 新增、0 消失）** ⇒ B0 十一轮新增的 **+98 个测试全部通过**，类别 A 架构边界违反保持归零。这**预验证了 WP-3 出口判据④**。注意本仓库既有失败形态是**顺序/状态污染**，故必须在全量顺序上下文下比对**集合**而非计数。细节见[技术债登记册](active/roadmap/05_TECH_DEBT_REGISTER.md) §2。

**推送受阻（如实记录）**：本会话 `git push` 报 `could not read Username for 'https://github.com'`（无可用凭据；`ls-remote` 可通，属认证问题而非网络）。`origin/main` 仍停在 `f9825943`，本地 `main` 领先若干未推送提交（含 `4a94e9c6`、`104de608`、`b89ca217` 及本轮文档提交）⇒ **CI 不会被本次提交触发**，需用户提供凭据或在本机推送。
