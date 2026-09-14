# M5 B0 / WP-3：HANDOFF-M4 落地结果（`rule_revision = 1`）

> 日期：2026-09-15。落地基线：`97aff6ed`（WP-1 冻结版 + WP-2 语义冻结之后）。
> 预注册依据：[路线 B 冻结版](M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md) §8（**本文不改判据，只报实测**）、
> [N2 冻结版](M5_B0_N2_STOP_REASON_PREREGISTRATION_FROZEN_20260915.md) I1–I8。
> 计划出口与仪器语义：[推进计划修订](M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md) WP-3。
> **状态：验收完成——五条出口全过（① §2｜②③ §3.3 + §3.4｜④ §4｜⑤ §5）。**
> 本文不抹去过程：§3.1 记录了第一次重跑为何**无效**以及当时被误记的"已过"。
> 远端未查询（`gh` 未认证）⇒ 全文不含任何"CI 已绿"表述；所有数字来自本地命令级复采。

## §1 落地了什么

对 [`scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py`](../../scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py)
的 `_member_episode` 施加**与反事实逐字节相同**的 2 处替换（+21 行）：

| 处 | 规则 | 效果 |
|---|---|---|
| (a) 选择 | `chosen = bindable[0]` → 计算 `last_success_index` 与 `blocked`（自身最近一次尝试未执行的成员），取**首个未阻塞**候选；全阻塞 ⇒ 记 `"executed": False` 的终局步并 `finish("all_members_blocked")` | 失败即让位，不再重试到 `STEP_CAP` |
| (b) cue | `cues = tuple([cue] * (len(steps) + 1))` → 按 `own_steps`（**该成员自己**已执行步数）+1 | 后加入的成员不再被问到自己训练范围之外 |

配套（同一提交内）：`RULE_REVISION = 1` / `COMPOSITION_RULE = "m4_failure_handoff"`；
报告 `design` 段发布 `rule_revision` 并把 revision-0 描述作为**带标签的历史字段**保留；
`DEFAULT_REPORT` 改为 revision-1 新文件名，旧 `…_20260913.json` 仅作 `REVISION_0_REPORT` 引用。

## §2 出口①：反事实 ≡ 实现（**已过**）

| 断言 | 实测 |
|---|---|
| gate 函数体逐字节包含 `M4_SELECTION` / `M2_CUE` | `true` / `true` |
| 两个 revision-0 锚点已消失 | `chosen = bindable[0]` `false`、旧 cue 行 `false` |
| `m4_failure_handoff` 判为恒等 | `variant_is_identity=true`、`already_applied=2`、`added_lines=0`、前后行数相等 |
| 冻结属性未被重绑定 | `frozen_attribute_intact=true`，且 `episode_fn is not frozen._member_episode` |
| 未落地的 revision-0 变体仍**拒绝构造** | `m1a_no_progress` / `m1b_tick_rotation` / `m2a_progress_plus_handoff` / `m3_repertoire_aware` = `refused` |

**副作用与两台仪器的差别（2026-09-15 更正，先前一概写"不重跑"是错的）**：`m2_per_member_progress` 也变成恒等——
M4 的第 (b) 处本就是 m2 的全部改动。
- **加固扫描**（为六变体逐一构造反事实）：其中四个现在**拒绝构造** ⇒ 它是 revision-0 仪器，**不重跑**，
  其归档结论原样成立。
- **反事实探针**：它的 `regression_check` 不比"当前源码"，而是拿给定臂去比**已封存的冻结矩阵数字**
  （`_frozen_rates()` 读 `FROZEN_ROUTE_C_REPORT`，与被测臂无关）⇒ **落地后依旧是有意义的两臂陈述**，
  可以重跑并直接给出口②所需的"冻结面 0 回归 / 2 改善"。重跑产物写新文件名（默认输出已改，见 §3.2）。

## §3 出口②③：落地后的实测复现 revision-0 的预测

### §3.1 第一次落地后重跑**无效**（本小节是负结果，按原样记录）

落地后立刻以满档重跑结构空间探针，得到的表是：三格 `frozen +2.000 / m4 +2.000 / delta +0.000`、
`interleaved 6 → 6`、`m4 regresses cells: none`。本文先前据此写"出口②已过"，**该结论作废**：

- 探针的两臂是 `baseline = frozen._member_episode` 与 `audited = 反事实副本`。M4 落地后
  前者**已经就是** M4，而后者因锚点已落地被判恒等（`variant_is_identity=true`，本文 §2 的出口①证据）
  ⇒ **两个 arm 是同一份源码**，`delta` 恒 `0.000`、`regresses: none` 都是同义反复，不是测量结果。
- 冻结版 §8 把出口②写成"**冻结面仍 0 回归 / 2 改善**"（两臂陈述）。以恒等对比去充当它，
  等于用"规则和自己相同"证明"规则没有让任何东西变差"——**这类退化不会被任何断言抓住，只能靠仪器语义发现**。
- 该次产物**未删除**，改名为
  [`taiji_b0_structure_space_probe_m4landed_degenerate_20260915.json`](../../reports/taiji_b0_structure_space_probe_m4landed_degenerate_20260915.json)
  作为这一陷阱的物证（`rule_delta.variant_is_identity=true` 就在文件里）。

### §3.2 仪器修法（同一批改动内）

| 修法 | 位置 | 作用 |
|---|---|---|
| `build_reverted()`：把**已落地**的 variant 沿同一 `(anchor, replacement)` 对**反向**还原成 revision-0 那一臂；部分落地（一对还原、一对未还原）= 既非 revision 0 也非 1 ⇒ `SystemExit` | [`probe_taiji_b0_m1_counterfactual.py`](../../scripts/training/probe_taiji_b0_m1_counterfactual.py) | 落地后仍有真正的"旧规则"可测；`co_varnames` 层面可验（`last_success_index`/`blocked`/`own_steps` 消失） |
| 按 gate 自报的 `RULE_REVISION` **选臂**：`>=1` 时 audited 臂取源码本身、baseline 臂用还原；`0` 时维持原前向 patch 路径 | [`probe_taiji_b0_structure_space.py`](../../scripts/training/probe_taiji_b0_structure_space.py) | 两列语义（`*_frozen`=revision 0，`*_audited`=revision 1）在落地前后保持一致 |
| **两臂同一函数即 `SystemExit`**；报告新增 `shipped_rule_revision` 与 `arm_provenance`（两臂来源、`arms_are_distinct`、正反两个 delta）；控制台首行打印臂来源 | 同上 | 退化不再可能被静默读成结果 |
| `does_not_change[0]` 改为按 revision 取值（不再声称"M4 is NOT implemented"） | 同上 | 报告自述与源码一致 |
| **六支** B0 仪器的默认输出不再指向封存文件（`REVISION_0_OUTPUT` + 新默认名），并由 `test_no_instrument_defaults_its_output_onto_sealed_evidence` 全仓扫描把关 | `probe_taiji_b0_{m1_counterfactual,structure_space,handoff_feasibility}.py`、`audit_taiji_b0_{m4_artifact,m4_hardening,n2_stop_reason_disposition}.py` | 一次忘记 `--output` 就能毁掉本轮基线；此前只在 gate 上修过 |

### §3.3 有效重跑（满档：6 context / 7 种子，与封存扩面报告同规模）

`probe_taiji_b0_structure_space.py --contexts-per-cell 6 --seed-offsets 0 101 202 303 404 505 606`
⇒ [`taiji_b0_structure_space_probe_m4landed_20260915.json`](../../reports/taiji_b0_structure_space_probe_m4landed_20260915.json)。

| 格 | 联合必需 | revision 0（还原臂） | revision 1（已发布） | delta | 交错 |
|---|---|---|---|---|---|
| `create__observation` | 是 | `+0.000` | **`+2.000`** | **`+2.000`** | **`0 → 6`** |
| `create__override` | 是 | `+0.000` | **`+2.000`** | **`+2.000`** | **`0 → 6`** |
| `create__mismatch` | 是 | `+0.000` | **`+2.000`** | **`+2.000`** | **`0 → 6`** |
| `patch__observation/override/mismatch` | 否 | `−2.000` | `−2.000` | `+0.000` | `0 → 0` |
| 其余 5 格 | 否 | `+0.000` | `+0.000` | `+0.000` | `0 → 0` |

判据实测：`arms_are_distinct = true`（`baseline = 还原到 rule_revision 0`、`audited = 已发布源码`）、
`m4 regresses cells = none`、`unexplained changes = none`、
`positive every seed = ['create__mismatch','create__observation','create__override']`（**7/7 偏移**，逐格 `+2.000 > required 1.65`）、
`interleaved = 6`、`grid separation = 3 distinct fingerprints over 11 cells`（**L1 仍未闭合**）、
T1/T3 仍 `inexpressible_under_frozen_binder`，
与封存扩面报告 `b09b3e62…` 的**逐格测量字段：全部相同**（`rows` / `validity` / `verdict` / `outcome_distinctness` / `seed_sweep` 五个顶层块整体相等，由
[`test_b0_structure_space_contract.py`](../../tests/taiji_native/test_b0_structure_space_contract.py) 末尾四条 landed 守卫钉住）。

⇒ **出口②（网格形式）**：已发布规则**逐字段复现**反事实所预测的 11 格结果，含 8 格"必须不动"的反例面；
此处"无回归"是**两臂相减**的结果，不是同义反复。
⇒ **出口③**：`interleaved > 0`（6）且逐格 `+2.000 > required 1.65`，7/7 种子偏移为正。

### §3.4 出口②的另一半：反事实探针在已发布源码上的有效重跑（**已过**）

`probe_taiji_b0_m1_counterfactual.py --output reports/taiji_b0_m1_counterfactual_m4landed_20260915.json`
⇒ [`taiji_b0_m1_counterfactual_m4landed_20260915.json`](../../reports/taiji_b0_m1_counterfactual_m4landed_20260915.json)（新文件，`shipped_rule_revision=1`）。

这台仪器的两臂陈述**没有**因落地而失效：它的 `regression_check` 拿被测臂去比**已封存的冻结矩阵数字**
（`_frozen_rates()` 读旧报告），因此"0 回归 / 2 改善"仍是真陈述。实测：

| 项 | 实测 |
|---|---|
| 未落地的四个变体 | 如实记 `refused`（锚点 `chosen = bindable[0]` 出现 0 次），**不再让整台仪器在第一个变体上退出** |
| `m4_failure_handoff` delta | `added_lines=0`、`identity=True`、两处 `already_applied`（对照封存报告 `added_lines=21`、无恒等键） |
| **冻结面回归（出口②原文判据）** | `no_regression=True`、`regressions=0`、`improvements=2` |
| 候选面（出口③） | `interleaved=4`、`best_pair=member-a+member-c`、`gain=+2.000`、`positive=True`；`reversed_order` 同 `+2.000`/`interleaved=4` |
| 停止原因（已发布规则） | 候选面 `all_members_blocked=48`、`goal_reached=8`、`all_members_exhausted=8`、`contract_intercepted:preview_ValueError=24` |
| 与封存 revision-0 报告的关系 | `regression` 与 `effect` 的测量字段**逐字段相同**（`rows`/`regressions`/`improvements`/`stop_reasons`/`by_order`），已由 `test_b0_m1_counterfactual_contract.py` 三条守卫钉住 |

⇒ **出口②（"冻结面仍 0 回归 / 2 改善"）在已发布源码上直接复现**；§3.1 那次无效重跑不改变这一点，
因为它坏在结构空间探针的 baseline 臂，而非本仪器的冻结面参照。

## §4 出口④：全量套件（**已过**）

| 侧 | 结果 | 用时 |
|---|---|---|
| 改前基线（同一工作树，仅差 WP-3 改动，`%TEMP%/pre_wp3_baseline.xml`） | **1408 passed / 0 failed / 6 skipped** | 1063 s |
| 落地后复采（`%TEMP%/post_wp3_full3.xml`） | **1428 passed / 0 failed / 6 skipped**（退出码 0） | 1065 s |

判据是**失败集合不新增**（不是"必须全绿"，也不得为凑绿放宽任何断言）。两侧失败集合都为空 ⇒ 成立。
用例名逐份比对（两份 junit XML）：新增 20 个 test id，**消失 5 个**，且这 5 个**全部是本轮有意重命名**，
每个都在同模块以双版本形态替换回来（覆盖未丢失）：

| 消失的名字 | 替代者 |
|---|---|
| `test_b0_handoff_probe_contract::test_frozen_gate_still_uses_priority_fallback` | `test_gate_reports_which_composition_rule_it_ships` |
| `test_b0_m1_counterfactual_contract::test_every_anchor_is_unique_in_the_frozen_source` | `test_every_anchor_is_unique_or_explicitly_shipped` |
| `test_b0_m1_counterfactual_contract::test_frozen_gate_still_uses_priority_fallback` | `test_gate_declares_which_rule_revision_it_ships` |
| `test_b0_n2_stop_reason_semantics_contract::test_the_frozen_rule_cannot_produce_the_new_reason` | `test_the_new_reason_only_ever_applies_to_a_handoff_rule`（仍双向：源码按 revision 取值 **且** 归档 revision-0 各格 `stop_reasons_frozen` 不得出现该原因） |
| `test_b0_structure_space_contract::test_probe_is_read_only_and_m4_still_not_implemented` | `test_probe_is_read_only_and_the_archived_report_states_its_scope` |

⇒ 净新增 15 条测试 + 5 条重命名；b0 子集 144 → 163；`ruff check .` 全绿。

## §5 出口⑤：历史证据未被触碰（**已过**）

- 封条：7 份 revision-0 报告 + 1 份 revision-0 处置报告的 sha256 已记入[推进计划修订](M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md) WP-3 表，
  并由 [`test_b0_rule_revision_seal_contract.py`](../../tests/taiji_native/test_b0_rule_revision_seal_contract.py)（J9）逐份重算比对；
  该测试同时要求 gate 的默认输出名**不得**是任何被封存的文件名。**本轮实测**：该模块随 b0 测试组全绿
  ⇒ 八份 revision-0 产物在落地后逐份哈希不变。
- **覆写防护（同批新增，出口⑤的必要条件）**：其余六支仪器的 `DEFAULT_OUTPUT` 原本仍指向封存文件名，
  而写入是原地 replace ⇒ 忘记 `--output` 一次就毁掉基线。已全部改为 `REVISION_0_OUTPUT`（只读）+ 新默认名，
  并由 `test_no_instrument_defaults_its_output_onto_sealed_evidence` 全仓扫描 `DEFAULT_(OUTPUT|REPORT)` 赋值把关。
- 六份解读性文档（交接探针 / 反事实 / 伪影审计 / 加固轮 / 结构空间 / 机制续篇）均已加 `rule_revision = 0` 标注，
  其中加固轮额外说明"M4 仍未实施"是**写作当时的事实**；结构空间 §11 先前那条"落地后重跑 `delta` 归 `0` 不是回归"
  已被本文 §3.1 推翻，故**就地更正为**"两臂同源是仪器缺陷，须由还原臂恢复对照"（§1–§10 的历史数字仍未回改）。
- **未改动任何历史 JSON 的数字**；新结果一律新文件。

## §6 这次落地**没有**证明什么（必须与正面结果同层阅读）

1. **不证明产品机制已支持协作**：该组合规则**只存在于 gate 脚本**，`taiji/` 无对应实现。
   01 号能力账本要求的协作轴「机制正式版本」**仍未闭合**，须另立 runtime 设计与预注册。
2. **不闭合 L1**：指纹仍 3/11 ⇒ 独立结构因素仍为 1；要抬过 1 只能走 WP-6。
3. **不构成能力晋级**：`growth_admitted=false`、`can_promote=false` 不变；反事实/恒等都不等于 `collaboration_achieved`。
4. **零余量仍在**：`ceiling_gain` 与 context 数无关恒 `2.0`，实测增益恰为 `+2.000` ⇒ 任一格退化即跌破 `required`。
   新增守卫确认参照未被移动（三格单体成功率 `0.0 → 0.0`，逐格断言已入测试）。

## §7 落地引发的计划修正（同批完成）

- **WP-4 入场条件原本会变成死锁**："五件探针全绿"含加固扫描，而它现在按设计拒绝运行。
  已在 03 / README / 推进计划三处改为"**各在其所属 `rule_revision` 下**全绿；revision-0 仪器不重跑"。
- **出口④基线更正**：旧"27 项 `SystemExit` 基线"作废——该级联系本地沙箱批量删除守卫伪影，
  本轮未复现且**非本轮修复成果**；详见[技术债登记册](../active/roadmap/05_TECH_DEBT_REGISTER.md) 2026-09-15 节。
- **消费面清单四次长大**：14 → 15 → 16 → **17**（每次都因为"新写的守卫自己读了 `stop_reason`"，被 fail-closed 扫描器当场抓出；
  最后一次正是本文 §3.4 的出口②复现守卫 ⇒ J10）。N2 冻结版 §0 已给出恒定的正确表述与"计数只能来自重扫"的规矩。

## §8 落地后真跑一次 gate 的端到端实测（**补掉的集成缺口，非五条出口之一**）

五条出口与全套件都不曾**运行过 gate 的 `main()`**（测试只 import 它、读它的源码与常量），
"改完 runner 之后它还能不能整体跑通"因此是没被覆盖的。本轮补做一次，**不带任何旗标**（即用默认路径）：

| 项 | 实测 |
|---|---|
| 退出码 / 用时 | `0` / **24.8 s** |
| 九门 | `gates_failed: []`，`status: completed`，`outcome: group_causal_corpora_supported` |
| 表面量 | `groups=2`、`rejected=4`、`matrix_constant=false`、`interventions_happened=true`、`zero_step_intervention_episodes=0` |
| 晋级位 | `growth_admitted=false`、`can_promote=false`（**未因落地而改变**） |
| 报告自述 | [`taiji_p5_2b_group_causal_corpora_m4_20260915.json`](../../reports/taiji_p5_2b_group_causal_corpora_m4_20260915.json) 带 `rule_revision: 1` + `composition_rule`，revision-0 描述作为 `composition_rule_revision_0` 历史字段保留 |

**这次运行同时是覆写防护的实测证据**：默认路径直写之下，封存的
`taiji_p5_2b_group_causal_corpora_20260913.json` 重算 sha256 **仍等于封条值**（`531cf6ef…`），
新数字落在 revision-1 文件名里 ⇒ §5 的防护不是纸面承诺。

**边界**：`groups=2 / rejected=4` 是**已发布规则下 gate 自身的表面量**，
不构成任何能力主张（WP-4 的训练前六门与 B1 入场仍未启动）。
