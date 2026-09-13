# Seed / Taiji 计划与架构入口

> 更新：2026-09-13（B0 完成 + M4 加固轮 + N1 结构空间轮）；结果审查基线 `f9825943`。执行顺序仅由[当前推进方案](active/roadmap/03_CURRENT_EXECUTION.md)决定。

## 当前阶段

**K 轴已限定晋级；路线 C、A 均已完成。B0 评分与可达性审查已完成，结论是目标当前不可达，故暂停正式训练，进入任务与估计目标的修改审阅。**

边界：K/G 为 opt-in、进程内状态；P5.1g 为未准入 trial；P5.2a 已接真实预测执行但泛化门未过；修复后的 P5.2b 当前 groups=0、rejected=6。A 的预测两档不代表正确排序，跨任务覆盖不代表同任务协作。结构成长未触发。

[B0 设计包](reference/M5_B0_MEASUREMENT_AND_REACHABILITY_AUDIT_20260913.md)给出统一测量字典、旧值复算、可达性上界与复合任务候选；[机制与预检续篇](reference/M5_B0_MECHANISM_AND_TASK_PRECHECK_20260913.md)给出根因与可落地规格；[路线 B 预注册草案](reference/M5_B0_ROUTE_B_PREREGISTRATION_DRAFT_20260913.md)（**未冻结**）给出 H1/H2/H3 分层判据。

本轮关键事实：六个 pair 在四个未见 context 上 **0 次**超过全体单体 oracle，`oracle_all_cell = oracle_singleton = 1.5`，三种候选参照**全部不可达**（缺口 2.15 / 0.15 / 1.15）。根因（续篇）：组合机制是**优先级回退链**（每 tick 只执行第一个绑定成功的成员），**仲裁类机制上界恒 ≤ 0**；24 个 pair×context 单元**零交错轨迹**；现行任务可支撑的参照上限仅 **0.35**，低于最佳可部署单体 0.5 ⇒ 换参照救不了，**任务必须改**。

探针结果（[交接可行性探针](reference/M5_B0_HANDOFF_PROBE_RESULT_20260913.md)）：先复现冻结矩阵 **11/11**；**三层已完全分离** —— 任务层找到达标形态 `create_and_override`（四单体全失败、`k=4/4`、**三种参照全部 feasible**、可支撑参照上限 1.85），但同一候选上**六个 pair 仍全部失败、`interleaved=0`** ⇒ **机制层是唯一阻塞**；根因是交接触发条件是"绑定失败"而非"无进展"，第一个成员独占 episode。据此提出最小修法 **M1（待决策 D5，未实施）**。

反事实测量（[机制修法反事实测量](reference/M5_B0_M1_COUNTERFACTUAL_RESULT_20260913.md)）：先验证候选面**可满足且顺序强制**（反序时 `set_language` 绑定成功但执行失败），再测**六个修法变体**——**五个失败、一个成功**。**`m4_failure_handoff`（+21 行、2 处替换）**：冻结面 **0 回归、2 改善**（`a+b`/`a+c` 各 0.25→0.50），候选面 **`interleaved=4/4`**、**同参照增益 `+2.000` > 1.65**（最严参照）⇒ **H2 协作主张首次可达**。

伪影审计（[M4 伪影审计](reference/M5_B0_M4_ARTIFACT_AUDIT_20260913.md)）：**四项全过** —— ① 特异性：正增益**只**出现在 `create_and_override`（冻结面 −0.5、`dual_requirement` −2.0、`create_then_patch` +0.000）；② 干预真实性：5 次面测量 `interventions_happened` 全 true；③ 机制 lesion：增益随交接消失且四单体全败；④ **3 个种子偏移下候选面增益恒 +2.000，且无种子在冻结面制造增益**。冻结面 2 个改善**逐步归因为真实交接**（基线里 `member-b`/`member-c` 一次机会都没有，M4 才放行）。**M4 仍未实施**；新增停止原因 `all_members_blocked` 需门禁语义审查。

加固轮（[M4 加固轮结果](reference/M5_B0_M4_HARDENING_RESULT_20260913.md)）闭合审计 §6 中**三项无需决策的残余风险**：规模扩到 **12 context / 6 结构变体**（六个变体**各自** `+2.000`，冻结规则 **0/6 为正、零交错**）、种子扩到 **5 个偏移**（恒 `+2.000`）、`all_members_blocked` 消费面**清单化为 11 个文件**，并证明 `contract_intercepted` 在两条规则下**计数相同**（非 M4 引入，增益对比未被污染）。`.h`/`cpp` 的"预期被合同拦截"被**与规则无关的脚本证据推翻并改判**，同时加上"预期与观测不符即拒出报告"的 fail-closed 校验。**界限说明（本轮最重要）**：六个变体只换表面（扩展名/语言/内容形状），**共享同一组合结构** `create + override`，结果指纹**只有 1 种** ⇒ 目前只证**表面稳健**，**结构稳健**须由 T1/T2/T3 重跑（登记 **N1**）；`all_members_blocked` 的门禁语义预注册为落地前置（登记 **N2**）。两次全量重跑 JSON **字节相同**。

N1 结构空间探针（[N1 结构空间结果](reference/M5_B0_STRUCTURE_SPACE_RESULT_20260913.md)）回答"**M4 的收益是否只在 `create + override` 这一种结构下成立**"。先实测发现**计划里的 T1/T3 在冻结 binder 下根本建不出来**（`workspace.create` 绑定 `goal_files[main_path]` ⇒ 创建即达标、`apply_patch` 随后拒绝；`_bind` 把每个动作都解析到 `main_path` ⇒ 第二目标路径不可达），T2 也**没有成员间证据通道**（`predict_episode` 只有一个参数，元组内是同一张量的重复）。故改为**枚举目标谓词可表达的全部结构**：3 内容路线 × 4 语言路线去平凡格 = **11 格 / 22 context / 3 种子**，有效性**在合同之下**判定。**结果：正增益覆盖 `create` 行全部三条语言路由**（各 `+2.000`、`interleaved 0→2`、`member-a+member-c` 两个成员都真实执行），其余 8 格 `0.000`，`patch` 行两规则同 `−2.000`（没有把不可达变成可达），**零回归、预测与观测 11/11 一致**；且 M4 修复的是**两种不同的冻结失败形态**（在不存在的文件上评估语言 ⇒ 合同拦截；首成员独占 ⇒ 步数耗尽）。**界限（本轮同等重要）**：三格**共享同一结果指纹** ⇒ 它们是一个结构因素经由三条路线，不是三次独立确认；可主张的宽度是 **1 个结构因素（存在性前提）× 3 条语言路线 × 2 种失败形态**，而**内容侧的第二种联合必需结构在冻结 binder 下不存在**（登记 **N1a**：要跨内容结构须先决定改 binder）。本轮亦自查出第一版有效性门**绕过 `policy_for`** 的缺陷并修好 —— 该修正**改变了结果**（`create__mismatch` `0.000`→`+2.000`，3 个假矛盾归零），预测函数两版之间未改。两次重跑 JSON **字节相同**，30 项合同测试通过。

**唯一推荐下一步：审阅 D1–D5 + N2 + N1a（WP-1 决策窗口），据此冻结路线 B 预注册。** 其中 **D5 已有通过四项伪影审计、在 12 context / 6 变体 / 5 种子下复现、并在全部 11 个可表达结构上定位了增益范围的实测可行选项 `m4_failure_handoff`** —— 不落地，H2 协作主张在当前机制下不可达；落地前仍须先处理 **N2**（`all_members_blocked` 门禁语义预注册，N1 使其更必要：三格各产生 24 次该停止原因），并按 N1 的界限读证据（**N1a** binder、**N1b** 每格 2 context、**N1c** 3 种子）。
执行顺序已按工作包修订：**WP-1 决策 → WP-2 停止原因语义 → WP-3 落地（五条可机检出口：反事实 ≡ 实现 / 冻结面仍 0 回归 2 改善 / 达标面 `interleaved>0` 且增益 `>1.65` / 全量失败集合不新增（基线 851/27/0/1）/ 历史报告只加"规则版本"指针不改数字）→ WP-4 B1 入场（五件探针 + 训练前六门）→ WP-5 下游链**。当前**无需决策即可推进**的只有两件、且都是只读：WP-1.5 扩面重跑（每格 4–6 context × 5 种子，≈12–15 分钟）与 WP-2 的"11 文件停止原因消费表"骨架。在 D1–D5 明确前不启动 B1 训练；不直接加特征列重跑旧 Gate。详见[推进计划修订](reference/M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md)。

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
| [B0 机制修法反事实测量](reference/M5_B0_M1_COUNTERFACTUAL_RESULT_20260913.md) / [测量报告](../reports/taiji_b0_m1_counterfactual_20260913.json) | 六个变体**五个失败一个成功**；**`m4_failure_handoff` 不回归（0 回归 / 2 改善）且 `interleaved=4/4`、同参照增益 +2.000 > 1.65** ⇒ **H2 协作主张首次可达**（未实施，待 D5） |
| [B0 M4 伪影审计](reference/M5_B0_M4_ARTIFACT_AUDIT_20260913.md) / [审计报告](../reports/taiji_b0_m4_artifact_audit_20260913.json) | **四项全过**：特异性（正增益只在该面）、干预真实性（无惰性 cell）、机制 lesion（增益随交接消失且四单体全败）、**3 种子下增益恒 +2.000**；冻结面 2 个改善**逐步归因为真实交接** |
| [B0 M4 加固轮结果](reference/M5_B0_M4_HARDENING_RESULT_20260913.md) / [加固报告](../reports/taiji_b0_m4_hardening_20260913.json) | 规模 **12 context / 6 变体**、种子 **5 个**、`all_members_blocked` 消费面 **11 文件清单**；六变体各自 `+2.000` 而冻结规则 0/6；`.h` 合同预期被证据推翻并加 fail-closed 校验。**只证表面稳健，不证结构稳健**（N1）；两次重跑字节相同 |
| [B0 N1 结构空间结果](reference/M5_B0_STRUCTURE_SPACE_RESULT_20260913.md) / [探针报告](../reports/taiji_b0_structure_space_probe_20260913.json) | 枚举目标谓词可表达的**全部 11 格结构**（22 context / 3 种子，有效性在合同之下判定）：**收益不限于** `create + override`（`create` 行三条语言路由各 `+2.000`，修复两种冻结失败形态），其余格 `0.000`/`−2.000`、零回归、预测 11/11 一致。**界限**：三格共享同一结果指纹 ⇒ 独立结构因素仍为 **1（存在性前提）**；T1/T3 在冻结 binder 下不可表达（N1a）。重跑字节相同，30 测试通过 |
| [B0 复合任务候选与路线 B 预注册草案](reference/M5_B0_ROUTE_B_PREREGISTRATION_DRAFT_20260913.md) | **未冻结**；H1 路由 / H2 协作 / H3 排序分列判据；候选已由探针收敛 |

## 文档职责

| 文档 | 职责 |
|---|---|
| [当前推进方案](active/roadmap/03_CURRENT_EXECUTION.md) | 唯一下一步、依赖、交付、验收、讨论节点 |
| [B0 结果整理与推进计划修订](reference/M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md) | B0 十一轮**结果账本**（F1–F12 已结项事实 / L1–L6 界限）+ **工作包计划**（WP-1 决策 → WP-1.5 扩面 → WP-2 停止原因语义 → WP-3 落地五条出口 → WP-4 B1 入场 → WP-5 下游链，含机时与并行债务轨） |
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

当前 f9825943 的 CI 34753643532 上次查询时两条 Linux 已在 Ruff 失败；本轮未查询远端（`gh` 未认证）。按命令级基线：本地 `ruff check .` 原有 1 项 `I001`（即上述失败原因），B0 已修复，现为 **All checks passed**。**全量套件本轮已跑**：851 用例 / 27 失败 / 0 错误 / 1 跳过（938s），**类别 A 架构边界违反已归零（2→0，A1/A2 结项可复现）**，27 项全为 `SystemExit: 1` 且**是旧 28 项的严格子集、无新增失败**。细节见[技术债登记册](active/roadmap/05_TECH_DEBT_REGISTER.md) §2 与[机制续篇](reference/M5_B0_MECHANISM_AND_TASK_PRECHECK_20260913.md) §7。

**推送受阻（如实记录）**：本会话 `git push` 报 `could not read Username for 'https://github.com'`（无可用凭据；`ls-remote` 可通，属认证问题而非网络）。`origin/main` 仍停在 `f9825943`，本地 `main` 领先若干未推送提交（含 `4a94e9c6`、`104de608`、`b89ca217` 及本轮文档提交）⇒ **CI 不会被本次提交触发**，需用户提供凭据或在本机推送。
