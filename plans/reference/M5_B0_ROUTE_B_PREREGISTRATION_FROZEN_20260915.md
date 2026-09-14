# M5 路线 B **冻结版**预注册（binder v1）

> 日期：2026-09-15。基线提交 `f23c9953`。前驱：[路线 B 预注册草案（未冻结）](M5_B0_ROUTE_B_PREREGISTRATION_DRAFT_20260913.md)。
> **状态：FROZEN。** 本文的估计目标、参照、任务规格、阈值推导、数据隔离、门禁分层与停止条件**不得在实现中顺手修改**；
> 任何后续新结论一律写**新的**预注册文件，本文原文不动（含负结果，不改绿）。
> 取值来源：用户 2026-09-15 确认「全部按照上限档来」（D1–D5 / N2 / N1a / N1a-2 逐项见草案 §10.2 与[推进计划修订 §2](M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md)「WP-1 取值已确认」表）。
> 适用范围：**仅 binder v1**（`_bind` 把所有动作解析到 `task.main_path`）。WP-6 若改 binder，必须另立版本标注的预注册，本文数值只作 v1 基线。

## §1 研究问题（三项分列，不得互代）

| 编号 | 主张 | 判别式 | 本文状态 |
|---|---|---|---|
| **H1 路由** | 组合覆盖比任何固定单体更广的任务面 | 同一参照下 `U(pair) > U(best deployable control)` | 待 B1 |
| **H2 协作** | 组合在**同一任务**上超过全体单体 oracle | `mean(P − max_i S_i) > 0` **且**经成员/连接 lesion 归因 | **本文主判据**（§2） |
| **H3 排序** | 对未见组合**正确排序**，而非仅区分类别 | 独立排序验证 + 校准样本量达标 | 待 B1 之后 |

H1 通过不授予 H2 主张；H2 通过不授予 H3 主张；三者各自独立验收。

## §2 冻结的估计目标与参照（D1）

**编码**：`SUCCESS_OUTCOME = +1.0`、`FAILURE_OUTCOME = −1.0` ⇒ 单 context 可清增益 `success − B = 2.0`
（[预检脚本](../../scripts/training/audit_taiji_b0_task_reachability_precheck.py) §常量与 `required_combination_only_contexts`）。

| 字段 | 公式 | 角色（冻结） |
|---|---|---|
| `gain_vs_all_singleton_oracle` | `mean(P − max_i S_i)` | **H2 主判据。参照 = `all_singleton_oracle`，`reference_gain = 1.5`** |
| `gain_vs_deployable_control` | `U(policy) − U(frozen deployable control)` | H1 主判据；oracle 标 `deployable=False`，**不作 H1 对照** |
| `best_fixed_singleton` / `best_observed_fixed_pair` | `0.5` / `1.0` | **降为诊断列**，只用于说明增益形状，不作为达标依据 |
| `causal_interaction` | `P − S_i − S_j + B` | 单独报告，不作效用 |
| `cost_account` | calls / steps / wall / recovery | 分账报告；折 utility 的权重须另立预注册 |

**阈值推导（D3）**：逃生通道为
`(success − B)·k / n > reference_gain + margin`，`margin = 0.15`（来源：路线 A 报告 `control_summary.margin`，非本轮反推）。
`required_combination_only_contexts` 取满足该式的最小 `k`；`k > n` 时返回 `n+1`，含义是**参照本身使主张不可达 ⇒ 改参照而不是调 margin**。
旧 1.65 **保留为历史记录**，不作为新阈值来源；新判据**逐格公布**，不使用单一全局阈值。

**n = 6（每格 6 context，WP-1.5 上限档）**，逐格推导与实测（三个联合必需格一致；由
`reference_requirements()` 计算，**不得手填**）：

| 参照 | `reference_gain` | `required`（=gain+margin） | `required k` | `available k` | `feasible` | `ceiling_gain` | `max_clearable_reference` |
|---|---|---|---|---|---|---|---|
| `all_singleton_oracle`（**主判据**） | 1.5 | 1.65 | **5** | 6 | true | 2.0 | 1.85 |
| `best_observed_fixed_pair` | 1.0 | 1.15 | 4 | 6 | true | 2.0 | 1.85 |
| `best_fixed_singleton` | 0.5 | 0.65 | 2 | 6 | true | 2.0 | 1.85 |

**两条必须随本文一起引用的余量性质**（[扩面报告](../../reports/taiji_b0_structure_space_probe_wide_20260915.json) / [结果 §10.3](M5_B0_STRUCTURE_SPACE_RESULT_20260913.md)）：

1. **k 维度有余量**：主判据需 5 个组合必需 context，可用 6 个（草案阶段是 2/2，一个都不能少）。
2. **增益维度零余量**：`ceiling_gain = (success−B)·available/n`，当 `available = n` 时恒为 `2.0`；实测增益恰好 `+2.000`，即**把可清 context 全清**，达标**不是勉强过线**，但也**没有任何缓冲**——任一格退化即跌破 `1.65`。
   ⇒ **风险条款 R-ZS**：任何后续报告若只报均值增益，必须同时报 `required`/`available`/`ceiling_gain`，否则视为不完整报告。

## §3 任务规格（D2，冻结为 create 行 × 3 条语言路由）

| 项 | 冻结值 |
|---|---|
| 候选面 | `create__observation`、`create__override`、`create__mismatch`（**全部 3 条**语言路由） |
| 内容路线 | `create`（初始无文件 ⇒ 单体永远达不到语言子句；存在性前提） |
| 语言路由 | `observation`＝可推断语言且无覆盖；`override`＝可推断语言 + `user_override=True`；`mismatch`＝**不可**推断语言（`javascript`）+ `user_override=True` |
| 可推断语言 | `python`；不可推断 `javascript` |
| 文件形状 | 扩展名 `.py`；`BASE_TEMPLATE = "def run_{i}():\n    return {i}\n"`、`PATCHED_TEMPLATE = "…return {i} + 1\n"`（内容携带**真实** Python 证据，不用 `x = 2` 这类无证据串） |
| context 索引 | `INDEX_BASE = 600`，`i = 600 + ordinal*10 + step`（`step < 10` 内不撞；n=6 安全） |
| 成员族 | `member-a=lang_confirm`、`member-b=patch_undo`、`member-c=create_undo`、`member-d=header_override`；`CELL_MEMBER_SETS` 的**顺序**属冻结面（风险 1：反序 pair 结果不同） |
| 执行预算 | `STEP_CAP = 8`、`REPEATS = 2` |
| 有效性门 | 每格逐 context 在**合同路径**（`_bind` → `policy_for` → 必要时 `issue_approval`/`consume_approval` → `execute_tool`）下脚本化验证：正向达标、反向不达标（顺序强制）、tick0 不已达标、零 `contract_intercepted`；任一不满足 ⇒ 该格 `not measurable` 并**如实剔除**，不得放宽 |

**已知不可表达（冻结为界限，不得声称）**：`patch` 内容路线下联合必需不成立（两规则同 `−2.000`，见 §6）；
T1 串联交接 / T3 共享状态约束在 binder v1 下**不可表达**（L2，有实测见证）；T2 纠错交接**缺成员间证据通道**（L3）。

## §4 数据隔离（D4）

- 旧 **144/88 载体降级为开发回归**，不作为本路线的测试面；新测试面**独立冻结**，规模随上限档（每格 ≥ 6 context）。
- 未见联合 cell 在 **train 全分区**移除（部分移除无效，已有回归测试钉住）。
- 按任务结构 / 项目文本 / 成员组合分组去重；**同一结构事实的多个实例不算独立样本**（本文的 3 条语言路由即属此类，见 §7 限定）。
- 条件化输入只取预测时可见 observation；评分真值不得进入输入。
- 校准样本量 / seed / 阈值 / wall cap 在 development 阶段冻结，不依最终成绩反推。

## §5 门禁（分层，不得合并为一门）

| 层 | 门 | 内容 |
|---|---|---|
| G1 合同门 | 干预真实性 | 非 baseline cell 零步 ⇒ `interventions_happened=false`（P5.2b 三层修复承接） |
| G2 数据门 | 隔离与去重 | §4 全部条款 |
| G3 排序门 | 独立排序验证 | tie-break / 重命名 / 编号置换下方向稳定；校准样本量达下限 |
| G4 任务门 | H1 | 统一参照下优于可部署对照 |
| G5 协作门 | H2 | 超过 `all_singleton_oracle` 参照（`required` 按 §2 逐格）**且** lesion 归因成立 **且** `interleaved > 0` |
| G6 预算门 | 成本与恢复 | 等预算、wall cap、中断恢复、拒绝路径 |

**训练前强制**：五件探针全绿——[上界预检](../../scripts/training/audit_taiji_b0_task_reachability_precheck.py) /
[交接可行性](../../scripts/training/probe_taiji_b0_handoff_feasibility.py) /
[反事实测量](../../scripts/training/probe_taiji_b0_m1_counterfactual.py) /
[M4 加固扫描](../../scripts/training/audit_taiji_b0_m4_hardening.py) /
[结构空间](../../scripts/training/probe_taiji_b0_structure_space.py)——并过 §9.2 六门。

## §6 反例面（必须**不**变的格，双向钉住）

冻结版不仅钉正结果，也钉**不该动**的结果：`none__*` 三格与 `create__none`、`patch__none` 必须保持
`0.000`（两条规则下），`patch__observation/override/mismatch` 必须保持两规则同 `−2.000`。
若新版本使其中任何格转好，**先怀疑仪器或 binder 泄漏**，不得直接报告为能力扩展。

## §7 结论边界与允许表述（H2 的上限，写死）

- 承重结构因素**只有 1 个**：**存在性前提**（初始无文件 ⇒ 单体永达不到语言子句）。三条语言路由共享同一结果指纹
  （`3 distinct fingerprints / 11 cells`，扩面后不变）⇒ 这是**一个因素经由三条路线**，**不是三次独立确认**。
- **允许**：「M4 的收益不限于 `create + override` 一种结构：`create` 行三条语言路由各 `+2.000`，且修复**两种**不同的冻结失败形态（合同拦截型 / 步数耗尽型）」。
- **禁止**：「结构稳健性已被多次独立确认」「独立结构因素 > 1」「跨内容结构稳健」。要突破只能走 **WP-6（改 binder）**，且须满足其出口 ③「新格结果指纹与 create 行三格不同」。
- 均值增益超过 oracle **不等于**一次真实协作事件：`collaboration_achieved` 仍须 G5 的 lesion 归因 + 独立验收（L6）。

## §8 机制修法（D5 = 落地）

`m4_failure_handoff`：**+21 行 / 2 处替换**施加于 `_member_episode`——(a) 失败即让位（自身最近一步未执行 ⇒ 计入 `blocked`，`chosen` 取首个未阻塞成员；全阻塞 ⇒ 记录终局 `all_members_blocked` 并 `finish`）；(b) cue 长度改为**每成员自身**步数。
落地前置与出口见[推进计划修订](M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md) WP-2 / WP-3；
其中 **WP-3 五条出口全过才算落地**：① 反事实探针在新 runner 上退化为恒等；② 冻结面仍 0 回归 / 2 改善；③ 达标面 `interleaved>0` 且同参照增益 `> required`；④ 全量失败集合对基线无新增；⑤ 历史报告只加「规则版本」指针、不改任何数字。

## §9 停止原因语义（N2 摘要，细则见 WP-2 冻结版）

`all_members_blocked` = **合法让位后的终局结果（非成功）**：不等于 `goal_reached`、不等于 `execution_failed`、
不等于 `safety_stops`、不可单独推出「成员无能力」。消费面 **15 个文件 / 8 处判断点（4 live gate + 4 测试断言）**，
逐条处置与不变式 I1–I8 见 **[N2 冻结版预注册](M5_B0_N2_STOP_REASON_PREREGISTRATION_FROZEN_20260915.md)**
（其底稿为[N2 处置审查](M5_B0_N2_STOP_REASON_DISPOSITION_20260914.md)；该审查文中「14 文件 / 5 判断点」为**已被更正**的旧计数）。

## §10 三态结论（扩展草案 §6）

| 三态 | 触发 | 允许表述 |
|---|---|---|
| `task_unreachable` | §2 逐格 `feasible=false` | 不判能力不足；改参照或改任务 |
| `representation_ineffective` | 特征仍不可区分候选 | 只支持"该表示下不可行" |
| `routing_only` | G4 过、G5 不过 | 只支持覆盖更广，**不支持协作** |
| `collaboration_supported` | G4、G5 均过且 lesion 归因成立、`interleaved>0` | 支持协作主张（仍受 §7 界限） |

## §11 必报分账（不作门，禁止省略）

总体收益、逐格差异、最坏类、安全违规、保持、成本（calls/steps/wall/recovery）、校准样本量与符号一致率、
lesion 前后对照、拒绝与恢复路径、三种参照全部分项、`required`/`available`/`ceiling_gain`、结果指纹数、停止原因分布。

## §12 停止点

1. §2 主判据逐格 `feasible=false` ⇒ 不进入 B1。
2. 公平 oracle 无法超过 `required` ⇒ 停止并回到参照讨论，**不得调 margin**。
3. 改变任务定义 / 估计目标 / 冻结阈值 / binder ⇒ **先审阅确认**；本文冻结后不改。
4. 收益保持不过 ⇒ 不得进入 P5.2d。
5. 有效性门在任何一格失败 ⇒ 该格如实剔除并公布原因，禁止用更宽的门换样本。

## §13 纪律

负结果如实落账、绝不改绿；失败报告独立保存不覆盖；报告对自身机制的描述必须与代码同步
（P5.2b 曾有自述错误 ⇒ 由 `frozen_attribute_intact` + 双向合同测试抑制）；回归测试双向钉住；
临时探针用毕即删；`growth_admitted=false`、`can_promote=false` 贯穿，研究通过不自动授权产品采用；
远端未查询时禁止任何「CI 已绿」表述；不 `gc`/`prune`，备份不删。

## §14 本文的**未闭合**项（诚实清单，不得因冻结而消失）

| # | 未闭合 | 归属 |
|---|---|---|
| 1 | 独立结构因素仍为 1（L1）；跨内容结构不可表达（L2）；T2 无证据通道（L3） | WP-6 / N1a-2 |
| 2 | `all_members_blocked` 冻结版语义预注册未写 | WP-2 |
| 3 | M4 未落地；反事实可达 ≠ 能力落地 | WP-3 |
| 4 | 优先级定义（成员序）未冻结 | WP-3 一并 |
| 5 | 成员族若变化须重跑全部五件探针 | 风险 4 |
| 6 | checkpoint 预检六门未取证 | WP-4 前置 |
| 7 | 全量套件的 pre-WP-3 基线尚未在本轮复采 | WP-3 出口④所需 |
