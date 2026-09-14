# M5 B0 / N2：`all_members_blocked` 停止原因语义审查结果

> 日期：2026-09-14；状态：**只读审查结果，N2 预注册尚未冻结**。
> 依据：[M4 加固轮](M5_B0_M4_HARDENING_RESULT_20260913.md) §5 / §8、
> [M4 伪影审计](M5_B0_M4_ARTIFACT_AUDIT_20260913.md) §5、
> [当前推进方案](../active/roadmap/03_CURRENT_EXECUTION.md) WP-2。
> 证据：[实时处置脚本](../../scripts/training/audit_taiji_b0_n2_stop_reason_disposition.py) /
> [处置报告](../../reports/taiji_b0_n2_stop_reason_disposition_20260914.json) /
> [契约测试](../../tests/taiji_native/test_b0_n2_stop_reason_disposition_contract.py)。
> **M4 仍未实施；本文不改 gate、runner、规则、冻结报告或阈值。**

## §1 结论摘要

1. **当前 live tree 有 14 个 `stop_reason` 消费文件**，而 M4 加固报告记录的 11 个清单已过时：新增的是 N1 结构空间探针和两个加固/结构契约测试。N2 脚本显式排除自身，避免扫描结果自污染。
2. **5 个判断点全部存在且完成初步处置**；其余 9 个文件为记录/聚合，不以停止原因作 gate 阈值。
3. 现有判断点只有两类：
   - **副本一致性**（P5.2a / P5.2b）：比较同一规则下两次运行的完整 surface；`all_members_blocked` 会在两侧同步出现，不改变一致性语义。
   - **`contract_intercepted` 前缀判断**（P5.2a 与加固扫描）：`all_members_blocked` 不带该前缀，不会被误计为安全拦截。
4. 因此，N2 的主要工作不是修改既有 gate，而是**冻结新停止原因的语义**：
   `all_members_blocked = 合法让位后的中间停止，不等于 goal_reached，不等于单体无能力，不等于安全违规`。
5. **N2 仍是 M4 落地前置**：本文完成的是审查面和初步处置，不是冻结版预注册；WP-2 仍需 D5=落地后写入新预注册并加双向测试。

## §2 两个停止原因的规范定义（草案）

| 原因 | 触发条件 | 语义 | 是否成功 | 是否安全违规 | 是否单体无能力 |
|---|---|---|---|---|---|
| `all_members_exhausted` | 当前 tick 没有任何成员能完成 bind | 组合没有可执行动作，安全终止 | **否**（除非终态已另行达标） | 否 | 不能单独推出 |
| `all_members_blocked` | 成员可以 bind，但最近一次执行失败；所有候选成员均被暂时 block | 失败信号已被消费、让位轮次耗尽；安全中间停止 | **否**（不得直接转成成功） | 否 | **不能推出** |

### §2.1 强制不变式

1. `stop_reason == "all_members_blocked"` **不得**被映射为 `success=True` 或 `goal_reached`。
2. `all_members_blocked` **不得**计入 `safety_stops`；只有 `contract_intercepted:*` 前缀才是合同拦截。
3. `all_members_blocked` **不得**单独计入"成员无能力"或拒绝增长；它表示组合层已发生合法让位，但当前 episode 仍未完成。
4. 副本一致性必须把 `stop_reason` 作为 surface 的一部分比较；M4 修改后 matrix 与 replica 必须使用同一规则。
5. 记录/聚合可以保留完整 reason；不得把新 reason 静默折叠到 `all_members_exhausted`。
6. M4 落地后，历史旧报告仍只对旧规则有效；新规则必须另存报告并写明 `rule_revision`。

## §3 Live 消费面：14 个文件

| 文件 | 初步分类 | 处置 |
|---|---|---|
| `scripts/training/audit_taiji_b0_m4_artifact.py` | 记录/聚合 | 保留 reason；不作阈值 |
| `scripts/training/audit_taiji_b0_m4_hardening.py` | 记录 + 合同前缀判断 | `was_contract_intercepted()` 只看 `contract_intercepted`，新 reason 不匹配 |
| `scripts/training/audit_taiji_b0_measurement_reachability.py` | 读取历史报告字段 | 不扫描 live reason，不作阈值 |
| `scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py` | 副本一致性 + 合同前缀判断 | J1/J2：安全；新 reason 不进入 `safety_stops` |
| `scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py` | 副本一致性 | J3：同一规则两次 surface 同步出现，新 reason 不改变一致性判定 |
| `scripts/training/eval_taiji_p5_2c_double_prime_unseen_combination_transfer_gate.py` | 记录/聚合 | `reason.split(":")[0]` 只用于报告分类 |
| `scripts/training/eval_taiji_p5_2c_prime_unseen_combination_transfer_gate.py` | 记录/聚合 | 同上 |
| `scripts/training/eval_taiji_p5_2c_triple_prime_representation_repair_gate.py` | 记录/聚合 | 同上 |
| `scripts/training/eval_taiji_p5_2c_unseen_combination_transfer_gate.py` | 记录/聚合 | 同上 |
| `scripts/training/probe_taiji_b0_m1_counterfactual.py` | 记录/探针 | 只报告反事实 stop reason |
| `scripts/training/probe_taiji_b0_structure_space.py` | 记录/探针 | 只聚合 frozen/audited reason counters |
| `tests/taiji_native/test_b0_m4_artifact_audit_contract.py` | 历史断言 | J4：固定旧规则证据；M4 落地后需迁移为 rule-revision 参数化断言 |
| `tests/taiji_native/test_b0_m4_hardening_contract.py` | 报告契约断言 | 新 reason 的出现与含义须继续显式断言 |
| `tests/taiji_native/test_b0_structure_space_contract.py` | 结构空间报告断言 | 正增益格必须出现 `all_members_blocked`，冻结格不得出现 |

### §3.1 清单新鲜度规则

M4 加固报告曾记录 11 个文件；当前 live scan 为 14 个。差异是：

- `scripts/training/probe_taiji_b0_structure_space.py`
- `tests/taiji_native/test_b0_m4_hardening_contract.py`
- `tests/taiji_native/test_b0_structure_space_contract.py`

N2 脚本自身不计入 live consumer 清单；否则审查脚本会因为读取自身的 `stop_reason` 字符串而自污染。契约测试会钉住这一排除规则。

## §4 5 个判断点（初步处置）

| ID | 位置 | 类型 | 结论 |
|---|---|---|---|
| J1 | P5.2a `deterministic_surface()` | 副本一致性 | 安全：同一规则下两侧同步，不把新 reason 当失败或成功 |
| J2 | P5.2a `startswith("contract_intercepted")` | 合同前缀 | 安全：新 reason 不带前缀，不计为安全拦截 |
| J3 | P5.2b `surface(...)` | 副本一致性 | 安全：M4 落地后两侧均用新规则；旧报告另存 |
| J4 | M4/N1 契约测试中的历史断言 | 测试判断 | 需在 WP-3 改为显式 `rule_revision` 双版本断言，不能把旧断言静默改绿 |
| J5 | 加固轮 `was_contract_intercepted()` | 合同前缀 | 安全：新 reason 不匹配；仍需保留双向测试 |

**当前结论**：没有发现既有 gate 把 `all_members_blocked` 当作成功、合同违规或单体无能力。真正需要新冻结的是**它作为合法让位后的中间停止**的语义，以及 M4 落地后的双版本报告/测试关系。

## §5 双向测试要求（WP-2 出口草案）

落地前至少增加以下断言；本文尚不写入 runner：

1. `all_members_blocked` → `success=False`，且不计入 `safety_stops`。
2. `contract_intercepted:preview_ValueError` → 仍计入合同拦截；不能被新 reason 的出现改变。
3. frozen rule 的 `all_members_exhausted` 与 M4 rule 的 `all_members_blocked` 必须各自保留，不能互相折叠。
4. matrix/replica 在同一 `rule_revision` 下 reason、success、resource_cost 逐项一致。
5. 不同 `rule_revision` 的报告禁止直接做数字覆盖；只能通过显式版本化比较。
6. 正增益格出现 `all_members_blocked` 是允许且可解释的；没有惰性 cell 仍是硬门。

## §6 出口与下一步

- **本轮已完成**：14 文件 live 清单、5 判断点、初步处置、清单漂移守卫、N2 报告与契约测试。
- **本轮未完成**：冻结 N2 预注册、修改既有 gate、实施 M4；这些依赖 D5=落地。
- **唯一下一步**：~~WP-1 书面确认 D1–D5、N2 语义与 N1a~~ —— **已于 2026-09-15 确认（全取上限档）**：D5 = 落地 `m4_failure_handoff`；N2 = **落地前**逐条预注册 14 个消费文件（5 个真判断点）的处置，并把 `all_members_blocked` 定为**一类终局结果**（自带归因桶，不得折算为 `execution_failed` / `goal_reached` / `safety_stops` / "成员无能力"）；N1a = 开，升为 WP-6。⇒ 本文 §2 的规范定义草案与 §5 的 6 条不变式**即为冻结版 N2 预注册的底稿**。
- **执行已授权（2026-09-15 解除限制）**：前置门 WP-1.5 满档扩面**四条出口全过**（每格 6 context / 7 种子、7/7 偏移为正、两跑字节相同）⇒ 本文正转为**冻结版 N2 预注册**（WP-2），随后进入 WP-3 落地。
