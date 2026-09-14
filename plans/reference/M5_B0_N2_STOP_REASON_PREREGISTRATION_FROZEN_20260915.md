# M5 B0 / N2 停止原因语义 **冻结版**预注册

> 日期：2026-09-15。基线提交 `f23c9953`。底稿：[N2 停止原因处置审查](M5_B0_N2_STOP_REASON_DISPOSITION_20260914.md) §2/§4/§5（只读审查面）。
> **状态：FROZEN。** 本文是 WP-3 落地 `m4_failure_handoff` 的**语义前置**：新停止原因的含义在改 runner 之前定死，落地后不得回改。
> 取值来源：用户 2026-09-15 确认 **D5 = 落地** 且 **N2 = 落地前预注册**（见[冻结版路线 B](M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md) §8/§9）。
> 证据：[审查报告 2026-09-14](../../reports/taiji_b0_n2_stop_reason_disposition_20260914.json)（审查面来源，`consumer_count = 14`）
> 与[重扫报告 2026-09-15](../../reports/taiji_b0_n2_stop_reason_disposition_20260915.json)（**冻结后的清单守卫**：`consumer_count = 15`、`judgement_sites` 长度 8、`drift.clean = true`、`status = inventory_guard_for_frozen_preregistration`）。

## §0 先更正底稿的一个数字

底稿 §1/§4 写「**5 个判断点**」，实为**判断点站点数与文件数被混为一谈**。以
[2026-09-15 重扫报告](../../reports/taiji_b0_n2_stop_reason_disposition_20260915.json) 为准（`consumer_count = 15`、
`judgement_sites` 长度 8、`drift.clean = true`）：

| 类别 | 数量 | 条目 |
|---|---|---|
| live gate 判据 | **4** | J1 副本一致性（P5.2a `deterministic_surface`）、J2 前缀判据（P5.2a 安全停止计数）、J3 副本一致性（P5.2b `surface`）、J5 子串判据（加固轮 `was_contract_intercepted`） |
| 测试断言 | **4** | J4（`test_b0_m4_artifact_audit_contract`，钉旧规则历史证据）、J6（`test_b0_m4_hardening_contract`）、J7（`test_b0_structure_space_contract`）、J8（`test_b0_n2_stop_reason_semantics_contract`，本包新增的双向守卫） |
| 仅记录 / 聚合 | **8 个文件** | 其余 consumer |

⇒ **正确表述恒为「15 个消费文件 / 8 处判断点（4 live gate + 4 测试断言，分布在 7 个文件），8 个文件仅记录聚合」**。
底稿的「5 / 9」既数错了站点也数错了文件；本文不回改底稿正文，只在此处更正，后续所有文档统一引用本节。
**注**：J8 就是本文 §4 的测试文件——它自己也是新原因的消费者，因此必须先被处置、清单才算干净；
扫描器在存在未处置消费者时**以退出码 1 失败**（"the review surface changed; re-disposition before landing"）。

## §1 两类终局结果的规范定义

| 原因 | 触发条件 | 语义 | 成功？ | 安全违规？ | 可推出单体无能力？ |
|---|---|---|---|---|---|
| `all_members_exhausted` | 当前 tick **没有任何成员能完成 bind** | 组合无可执行动作，安全终止 | **否**（除非终态另行达标） | 否 | **不能** |
| `all_members_blocked` | 成员**可以** bind，但其自身最近一次执行未成功，且所有候选成员均被暂时 block | **失败信号已被消费、让位轮次耗尽**的合法中间停止 | **否**（不得转写为成功） | 否（不带 `contract_intercepted` 前缀） | **不能**（只说明本 episode 未完成） |

二者是**不同的终局**，不得互相折叠：前者是"没有可绑定的动作"，后者是"绑定了但执行失败后无人可让位"。

## §2 强制不变式（逐条可机检）

| # | 不变式 | 机检方式 |
|---|---|---|
| I1 | `stop_reason == "all_members_blocked"` ⇒ `success` 为假，且 `goal_reached` 为假 | 双向测试 T-a |
| I2 | 该原因**不得**计入 `safety_stops`；只有 `contract_intercepted:*` 前缀算合同拦截 | 双向测试 T-b |
| I3 | 该原因**不得**单独计入"成员无能力"或拒绝增长 | 双向测试 T-c |
| I4 | 该原因**只**出现在被审规则（M4）的 run 中；冻结规则的任何 run 不得出现 | 双向测试 T-d（正：三格各 24 次；反：冻结侧恒 0） |
| I5 | **未被让位触及的**合同拦截在两规则下计数必须相同（`contract_intercepted:preview_ValueError`，实测 `36 = 36` 逐格）；**被修复的**拦截必须归零且只出现在该出现的地方（`language_assessment_unavailable` 只在 `create__observation` 的**冻结侧** `72`，M4 侧 `0`） | 双向测试 T-e。**注**：本条原写为"`contract_intercepted:*` 全部相同"，被自己的双向测试否证后收窄—— blanket 版本与归档证据矛盾 |
| I6 | 副本一致性比较必须把 `stop_reason` 纳入 surface；同一 `rule_revision` 下 matrix 与 replica 逐项一致 | WP-3 门（落地后 T-f） |
| I7 | 记录/聚合可保留完整 reason，但**不得**把新 reason 静默折叠为 `all_members_exhausted` | 双向测试 T-g |
| I8 | 不同 `rule_revision` 的报告禁止数字覆盖；旧报告只加"规则版本"指针 | WP-3 出口⑤（T-h） |

## §3 消费面逐条处置（15 文件；4 live gate + 4 测试断言 + 7 仅记录）

处置原则：**live gate 判据（J1/J2/J3/J5）零改动即为安全**，因为新原因不带 `contract_intercepted` 前缀、且副本两侧同步；
**测试断言（J4/J6/J7/J8）须显式双版本化**，不得把旧断言静默改绿。
[扫描脚本](../../scripts/training/audit_taiji_b0_n2_stop_reason_disposition.py) 在存在未处置消费者时**退出码 1**（fail-closed），故本表与代码中的 `EXPECTED_CONSUMERS` 必须同步。

| 文件 | 类 | 处置（冻结） | 断言 |
|---|---|---|---|
| `audit_taiji_b0_m4_artifact.py` | 记录 | 保留 reason，不作阈值 | T-g |
| `audit_taiji_b0_m4_hardening.py` | J5 | 子串判据不变；新原因不匹配前缀 | T-b, T-e |
| `audit_taiji_b0_measurement_reachability.py` | 记录 | 读历史字段，不扫 live reason | — |
| `eval_taiji_p5_2a_predictive_execution_gate.py` | J1, J2 | 副本两侧同步；安全停止只认前缀 | T-b, T-f |
| `eval_taiji_p5_2b_group_causal_corpora_gate.py` | J3 | 同上（同规则内比较） | T-f |
| `eval_taiji_p5_2c_double_prime_…gate.py` | 记录 | `reason.split(":")[0]` 仅报告分类 | T-g |
| `eval_taiji_p5_2c_prime_…gate.py` | 记录 | 同上 | T-g |
| `eval_taiji_p5_2c_triple_prime_…gate.py` | 记录 | 同上 | T-g |
| `eval_taiji_p5_2c_…gate.py` | 记录 | 同上 | T-g |
| `probe_taiji_b0_m1_counterfactual.py` | 记录 | 只报告反事实 stop reason | T-d |
| `probe_taiji_b0_structure_space.py` | 记录 | 聚合两规则 reason 计数 | T-d, T-e |
| `test_b0_m4_artifact_audit_contract.py` | **J4** | **WP-3 须迁移为 `rule_revision` 双版本断言**（反事实落地后退化为恒等，旧断言是历史证据） | T-h |
| `test_b0_m4_hardening_contract.py` | **J6** | 新原因的出现与含义须继续显式断言 | T-d, T-h |
| `test_b0_structure_space_contract.py` | **J7** | 正增益格**必须**出现该原因；冻结侧与零格**必须**不出现 | T-d（双向） |
| `test_b0_n2_stop_reason_semantics_contract.py` | **J8**（本包新增） | 语义守卫自身：断言 I1–I5、I7；它是 WP-3 的**入场守卫**而非 runtime gate，故同样要随 `rule_revision` 迁移 | T-a, T-b, T-c, T-d, T-e, T-g |

## §4 双向测试清单

测试文件：`tests/taiji_native/test_b0_n2_stop_reason_semantics_contract.py`。
每条都必须**正反两向**——正向钉住允许的解释空间，反向钉住禁止的解释空间。

| 编号 | 正向断言 | 反向断言（必须为假） |
|---|---|---|
| T-a | 含 `all_members_blocked` 的 step 其 `goal_reached` 为假 | 不得被读成成功终态 |
| T-b | 该原因不以 `contract_intercepted` 开头 ⇒ 不入 `safety_stops` | 存在某处把它当安全违规 |
| T-c | 三格出现 24 次该原因**同时**该格 `goal_reached=4` 且增益 `+2.000` | 该原因 ⇒ 成员无能力 |
| T-d | M4 侧计数 `> 0`；冻结侧计数 `== 0` | 冻结规则也会产生它（若是，则 M4 与冻结不可区分） |
| T-e | **未被让位触及的**拦截逐格相同（`preview_ValueError` `36 = 36`）；**被修复的** `language_assessment_unavailable` 只在 `create__observation` 冻结侧 `72`、M4 侧 `0`，其余格两侧皆 `0` | 未修复的拦截在两规则间发生变化（那才说明增益对比被污染）；或被修复的拦截在别的格也归零（说明修复与结构无关） |
| T-f | （WP-3 后）同一 `rule_revision` 下 matrix 与 replica 的 reason/success/cost 逐项一致 | 跨 revision 比较被当作确定性证明 |
| T-g | 报告分类不折叠两种 exhausted/blocked | 二者被合并统计 |
| T-h | （WP-3 后）旧报告仅加版本指针、数字未变 | 新规则数字覆盖旧报告 |

**当前落地范围**：T-a…T-e、T-g 依现有归档报告即可判定 ⇒ 本轮写入并通过；
**T-f、T-h 依赖 runner 具备 `rule_revision`** ⇒  reserved，随 WP-3 一并实现（本文先冻结其判据，避免落地后补写变成"事后合理化的门"）。

## §5 `rule_revision` 规则

1. 每次停止原因集合或让位规则变化 ⇒ `rule_revision` 递增，并写入报告与测试断言参数。
2. 跨 revision **禁止**：数字覆盖、把旧报告的格当新报告的格、用旧阈值判新规则达标。
3. 跨 revision **允许**：显式版本化比较（同一 face 在两规则下的 reason 分布对照），即 §3 中 J4/J6 的迁移目标形态。

## §6 出口判据（WP-2）

1. §3 的 14 行处置全部落地（4 个 live 判据零改动并说明理由；3 个测试断言标记为 WP-3 迁移项）。
2. §2 的 I1–I5、I7 由 T-a…T-e、T-g 双向钉住且**通过**。
3. 计数新鲜度：`drift.clean = true` 且 consumer 清单仍为 14；若新增消费者 ⇒ 先补处置再落地 M4。
4. 本文冻结；后续语义变化一律新预注册。

## §7 停止点

1. 若发现任一 live gate 把 `all_members_blocked` 计为成功、安全违规或单体无能力 ⇒ **停止 WP-3**，先修门禁语义。
2. 若 `drift.added` 非空 ⇒ 停止，先给新消费者做处置。
3. 若 M4 落地后 `contract_intercepted` 计数在两规则间不再相同 ⇒ 停止并归因（说明增益对比被污染，见 I5）。
4. 负结果如实落账，不改绿；`growth_admitted=false`、`can_promote=false` 贯穿。
