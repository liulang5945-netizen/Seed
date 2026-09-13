# M5.K A8 晋级评审合同：veto 逐项裁决与晋级路径冻结

> 冻结日期：2026-09-12。前置：[scorecard v6](M5_K_AXIS_SCORECARD_V6_CONTRACT_20260911.md)（A8 晋级评审两个入场条件齐备：`k_worker_joint_course_completed=true`、`default_runtime_rollout_review_completed=true`）。本合同冻结 A8 评审的**裁决对象、逐项裁决标准、结果映射与 fail-closed 边界**；评审结论须**独立批准**（见 §6），批准前不训练、不接默认 runtime、不解冻任何 owner。A8 准则原文（[原生架构 §11](../active/TAIJI_NATIVE_ARCHITECTURE_V1.md)）：「新任务后保留旧能力，并根据缺口完成可回滚的分化/增长/剪枝」。

## 1. 目的与评审对象

A8 评审裁决两件事：

1. **证据裁决（3 项 veto）**：v3 时代冻结的三项 veto 是否已被后续证据链满足——满足则由 scorecard v7 翻转（机械翻转，证据已入账，不需要新实验）；
2. **授权裁决（1 项）**：是否授权「默认 runtime owner/学习机制附着」进入预注册（这是唯二无法由已有证据翻转的 veto——它们要求产品代码里真实发生附着）。

裁决对象 = v6 `promotion_gates` 中仍为 `false` 的五项：

| veto | 冻结出处 | 字面要求 |
|---|---|---|
| `parent_retention_baseline_present` | v1 scorecard | K1/K2 formal 的 `metric_contract.parent_retention` 存在 |
| `same_parent_continual_s_g_k_evidence` | v1 scorecard | 同一 parent 上的连续 S/G/K 课程证据 |
| `resource_rollback_old_capability_gate` | v1 scorecard / v3 §6 | 晋级课程含资源等价、rollback、旧能力非劣门（SGK v1 的原始形态） |
| `default_runtime_owner_attached` | v1 scorecard | owner 附着于默认 runtime |
| `learning_mechanism_attached_default_runtime` | v3 scorecard | 学习机制（现 = 求解器机制 + K continuation）接入默认 runtime |

## 2. 证据基线（只读，digest 已入账 scorecard v6）

- K1/K2/K3 standalone formal（v1/v2 收束）；C 阶段学习机制（v3，FS 候选）；
- 表示因子化（P4.9）+ 求解器更新机制（P4.11）+ 单任务验证（P4.12，9/9 零方差）+ 两阶段累积课程（P4.13，9/9，向后保持零失败，154 约束全格精确零违反）+ K worker 联合课程（P4.14，4/4，post-K 景观上 G 求解器维持平衡）；
- 默认 runtime rollout review（4/4 cell，消费等价逐字段复现，`fit_called=false`，产品代码与默认 checkpoint 零改动）；
- 保持证据链：P2.4 retention canary（50 条均衡 rehearsal 复现）→ P2.6 交错保持 → P4.4 sibling 身份校准（退化被诚实复现）→ P4.13 向后保持 → review 消费等价覆盖同一保持指标。

## 3. 逐项裁决（标准 + 选项 + 推荐）

### 3.1 `parent_retention_baseline_present` —— 建议翻转（证据裁决）

- **现状**：v1 冻结该 veto 时，保持证据为零。此后 P2.4–review 链在同一 parent 上建立了**多层级保持基线**：50 条均衡 rehearsal canary、sibling 身份校准、retention-newtask 非劣门、向后保持零失败、消费等价复现。
- **选项 A（推荐）**：以 P2.4→review 证据链裁定满足——链条强于 v1 字面要求（v1 只要求 formal 合同里存在 parent_retention 字段）。翻转属机械转录，不需新实验。
- **选项 B**：按 v1 字面重跑 K1/K2 formal 并补 parent_retention 合同——成本一轮 formal，信息量低（同一测量已由 P2.4+ 链在更强对照下完成）。

### 3.2 `same_parent_continual_s_g_k_evidence` —— 建议翻转（含明确诚实边界）

- **现状**：P4.14 已是**同一 parent 上的连续 S/G/K 课程**——K 相（P2.6 机械，学习）→ post-K 重 materialization → G 相（P4.11 合同，学习），S 全程为 runtime evidence（P3.1 合同冻结的架构角色）。
- **选项 A（推荐）**：裁定满足，边界如实入账——**S 不是 learned**：按架构合同 S 是证据/状态器官（K 只读 S evidence，G 持有选择），P3.1–P3.6 已闭合其状态接线/恢复/所有权边界。翻转口径为「同 parent 连续 S/G/K 课程（S 为架构性 control-only evidence）」。
- **选项 B**：预注册 learned-S 实验——成本一条新研究线；且 P3.3 的教训（G-only 零步与训练行为完全相同，直到引入行为 utility 信号）提示 S 未必存在可分离的学习目标；在 A8 主线上属可选深化而非晋级前置。
- **选项 C**：维持 veto——把「S learned」设为晋级硬条件（最保守，A8 主线停摆直至新线完成）。

### 3.3 `resource_rollback_old_capability_gate` —— 建议翻转（后继链映射）

- **现状**：该 veto 的原始形态 = v3 §6 的 SGK v1 晋级课程（资源等价 + rollback + 旧能力非劣，学习机制 = FS 候选）。SGK v1 已暂停；其三项要求在**后继链**中全部以更强形式出现：绝对资源预算（P4.12/13/14 + review，cell 2–16s ≪ 600s cap）、rollback 门（P3.0 合同 + P4.13 9/9 + review 4/4 行为等价）、旧能力非劣（P4.13 向后保持零失败 + review 保持指标逐字段复现）。
- **选项 A（推荐）**：裁定 SGK v1 被后继链取代并满足——翻转时在 scorecard v7 记录「取代映射」（SGK v1 每项要求 → 对应 P4.x/review 证据），SGK v1 文档标记 `superseded`，不删除、不改写。
- **选项 B**：复活 SGK v1 原样执行——成本高且其学习机制（FS 交错 SGD）已被 P4.10/P4.11 证明是该任务上的失败机制（交错 SGD 不可达可行解，投影求解器可达），按已否决路线执行与「不把失败改写成通过」纪律冲突。

### 3.4 `default_runtime_owner_attached` + `learning_mechanism_attached_default_runtime` —— 建议授权进入预注册（不即时翻转）

- **现状**：rollout review 已冻结并验证消费合同（`taiji-default-runtime-rollout-attachment-v1`），但产品代码零改动——这是 review 的设计边界，不是缺口。
- **授权内容**：预注册「默认 runtime 附着」工程步——runtime 侧 adapter 按消费合同实现 fail-closed load（digest/lineage/嵌入不变量/参数计数）、独立恢复、行为等价复验、失败回滚（恢复原 checkpoint）；不改判定阈值、不新增训练、附着前后 `seed_corpus.pt` 与全部研究 artifact digest 不变。
- **翻转时机**：两项 veto 在附着步**执行并通过其自身验收门之后**由 scorecard v7 翻转——不是被本评审翻转。此顺序保证「默认 runtime 已采用该机制」永远有机器证据背书。
- **选项 B**：不授权——A8 主线停在「研究闭合、未接入」状态（合法收束，晋级无从谈起）。

## 4. 结果映射（冻结；批准后禁止调改）

| 分支 | 条件 | 后续 |
|---|---|---|
| `promotion_review_recommended` | §3.1–3.3 三项证据裁决批准 ∧ §3.4 附着授权批准 | scorecard v7 翻转三项证据 veto（机械转录）；唯一下一步 = 附着工程预注册 |
| `promotion_review_partial_<veto>` | 任一裁决被否/改选 B/C | 相应工作项进入计划；其余已批准项照常处理 |
| `promotion_review_deferred` | 整体缓议 | A8 主线停在当前收束态；记录缓议理由 |

## 5. fail-closed 边界（本评审无论如何不发生的事）

- 本评审不宣布 promotion：`can_promote` 只能由 scorecard v7 在五项 veto 全部翻转后机械置位，且置位后仍须独立批准；
- 不训练、不读取 sealed、不修改任何已冻结报告/判据；
- 不在附着预注册批准前改动 `api/`/`seed_platform/`/默认 checkpoint；
- 不把机制证据写成完整认知能力或结构成长：A8 的「分化/增长/剪枝」子句在本轴上**未被触发**（P4.7 关闭容量假设、P4.9–P4.11 在固定容量内消解张力）——这是如实记录，不是跳过；结构成长机械（原生线 budget/rollback 门）保持可用待未来缺口。

## 6. 评审流程（独立批准）

1. 本合同冻结（已发生）；2. 评审人（项目所有者）对 §3 各项给出批准/否决/改选；3. 按 §4 落盘结论并更新唯一计划；4. scorecard v7 仅在批准范围执行机械转录。评审人未明确批准前，一切保持 v6 现状。

## 7. 评审结论（2026-09-12，独立批准已落盘）

评审人对 §3 四项裁决**全部批准**，结果映射落盘为 **`promotion_review_recommended`**：

- §3.1 `parent_retention_baseline_present`：批准选项 A（证据裁决翻转）；
- §3.2 `same_parent_continual_s_g_k_evidence`：批准选项 A（P4.14 即同 parent 连续 S/G/K 课程；S 为架构性 control-only evidence 的边界如实入账）；learned-S 探索线（选项 B）记录为**非阻塞的未来上限提升项**，不作为晋级前置；
- §3.3 `resource_rollback_old_capability_gate`：批准选项 A（后继链取代映射；SGK v1 标记 `superseded`，文档不改写）；
- §3.4 附着授权：批准——`default_runtime_owner_attached` 与 `learning_mechanism_attached_default_runtime` 两项 veto 保持 `false`，待附着步执行并通过其自身验收门后由**后继 scorecard**（v7 之后的下一个版本）翻转。

**评审人常设决策准则（记录在案，适用后续所有决策点）**：遇到需要决策的分叉时，优先选择**上限更高**的选项（更强的集成、更完整的能力面），前提是不违反本仓库的 fail-closed 纪律与已冻结判据。

**勘误（机械编号修正，非判据变更）**：本合同 §3.4 与 §5 原文将「后继 scorecard」误写为「scorecard v7」；按 §4 的冻结映射，v7 = 三项证据 veto 的机械转录，附着两项 veto 由 v7 之后的下一个 scorecard 版本翻转。判据本身无任何变更。
