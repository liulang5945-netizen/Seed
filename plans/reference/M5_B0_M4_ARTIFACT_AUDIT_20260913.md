# M5 B0：M4 伪影审计（D5 决策前的最后一道证据）

> **规则版本**：本文全部数值只对 **`rule_revision = 0`**（`chosen = bindable[0]` + episode 级 cue 长度）有效。
> 审计对象是"反事实规则 vs 当时的冻结规则"；M4 落地后两者不再是不同规则，本文不回改数字、不重跑覆盖。

> 2026-09-13；基线 `f9825943`。
> 本文审计 [M4 反事实结果](M5_B0_M1_COUNTERFACTUAL_RESULT_20260913.md) 是否成立。
> 这条研究线**曾被零步伪影坑过一次**（`reports/M5_P5_2C_ENTRY_AUDIT_P5_2B_DEFECT_20260913.md`），
> 而 M4 同时做到"产生正增益"和"改善冻结面 2 个 cell"，因此必须逐项排查后才能用于 D5。
> **M4 仍未实施**：不改任何 gate/runner/规则/冻结产物。
> 证据：[审计脚本](../../scripts/training/audit_taiji_b0_m4_artifact.py) /
> [审计报告](../../reports/taiji_b0_m4_artifact_audit_20260913.json)。

## §1 结论：四项审计全部通过，且改善有逐步可解释的轨迹

| # | 审计项 | 结果 |
|---|---|---|
| 1 | **特异性**：正增益只应出现在为交接设计的任务面上 | **通过**。唯一正增益面 = `create_and_override`（+2.000）；冻结面 −0.5、`dual_requirement` −2.0、`create_then_patch` +0.000 |
| 2 | **干预真实性**：无惰性非 baseline cell | **通过**。5 次面测量的 `interventions_happened` 全为 `true` |
| 3 | **机制 lesion**：增益必须随交接消失 | **通过**。同一任务面在冻结规则下增益 ≤ 0；四个单体全部失败 ⇒ 增益既需交接、也需两个成员 |
| 4 | **种子稳健性**：不是单一种子侥幸 | **通过**。3 个种子偏移下候选面增益**恒为 +2.000**（区间 `[+2.000, +2.000]`），且**无一种子在冻结面制造增益** |

**另外**：`frozen._member_episode` 身份在审计前后不变（反事实只在命名空间副本中运行）。

## §2 特异性：正增益只在设计好的任务面上出现

| 面 | 最佳 pair | 同参照增益 | `interleaved` | 干预真实性 |
|---|---|---|---|---|
| `create_and_override`（冻结顺序） | `member-a+member-c` | **+2.000** | **4** | true |
| `create_and_override`（反序） | `member-c+member-d` | **+2.000** | **4** | true |
| 冻结验证面 | `member-a+member-b` | −0.500 | 0 | true |
| `dual_requirement` | `member-a+member-b` | −2.000 | 0 | true |
| `create_then_patch` | `member-a+member-c` | +0.000 | 0 | true |

- 两个**单成员可解**的面上，M4 都没有制造正增益（`dual_requirement` 是 −2.0，`create_then_patch` 恰好 0.0）。
- **冻结面上也没有正增益** ⇒ M4 不会在"没有组合专属可解 context"的任务上凭空产生收益。
- 反序仍为 +2.000，只是成功的 pair 换成 `member-c+member-d` ⇒ **不是顺序侥幸**。

## §3 机制 lesion：增益随交接消失

| 项 | 值 |
|---|---|
| 审计规则下最佳 pair | `member-a+member-c`，增益 **+2.000**，`interleaved=4` |
| 冻结规则下同一面最佳 pair | 增益 **≤ 0**，`interleaved=0` |
| 四个单体成功率 | **全 0.0** |
| `gain_vanishes_without_handoff` | **true** |
| `all_singletons_fail` | **true** |

⇒ 增益**不能**归因于"某个成员本来就能做"，也**不能**归因于"任务本身平凡"。

## §4 冻结面 2 个改善的逐步归因（关键）

冻结面上被改变的是 `member-a+member-b` 与 `member-a+member-c`（各 0.25 → 0.50）。逐步轨迹显示两者都是**冻结规则结构性阻止、M4 才放行的真实交接**。

### 4.1 `member-a+member-b`：block-1（`patch_persist`）context 101/105/109

| 规则 | 轨迹 | 结果 |
|---|---|---|
| 冻结 | `a:read`✓ → `a:resolve`✓ → **无成员可绑定** | `all_members_exhausted` **失败** |
| M4 | `a:read`✓ → `a:resolve`✓ → **`b:read`✓ → `b:apply_patch`✓** | `goal_reached` **成功** |

`member-a` 用完自己的轮次后无法再绑定，**`member-b`（补丁专家）在冻结规则下一次机会都没有**。M4 让 b 进场并把补丁打完。这正是前几轮测得的"冻结规则 headroom"（`a+b` 丢 0.5）被回收。

### 4.2 `member-a+member-c`：block-2（`create_persist`）context 102/106/110

| 规则 | 轨迹 | 结果 |
|---|---|---|
| 冻结 | `a:read`✗ → `a:resolve`✗ → **无成员可绑定** | `all_members_exhausted` **失败** |
| M4 | `a:read`✗ → **`c:list`✓** → `a:read`✗ → **`c:create`✓** | `goal_reached` **成功** |

这条轨迹同时验证了 M4 的**"成功即解除封锁"**规则：`a` 失败后被封锁，`c` 成功后解除，`a` 重试仍失败，最后由 `c` 完成。**是真正的交错轨迹，不是单成员轨迹。**

### 4.3 诚实的反面细节

同两个 cell 在 block-1 的另一些 context 上，M4 让 `member-c` 进场后停在 `contract_intercepted:preview_WorkbenchConflictError`（`c:list`✓ → `c:create`✗）——**交接发生了，但被合同层拦下**。这是合同层的正常行为，不是伪影；也说明 M4 的效果受合同边界约束。

## §5 新引入的语义变化（需纳入门禁审查）

M4 引入一个新的停止原因 **`all_members_blocked`**（所有可绑定成员都在最近一次尝试中失败）。它出现在：
- 冻结面（36 次）与候选面（48 次）的测量中；
- 语义上不同于既有 `all_members_exhausted`（无成员可**绑定**）——新原因是"能绑定但都刚失败过"。

⇒ 落地前必须**单独预注册**并审查门禁语义（哪些门消费 `stop_reason`、新原因是否影响"安全拒绝"的判定）。本文**不改**任何门禁。

## §6 残余风险（不因审计通过而消失）

| # | 风险 | 处置 |
|---|---|---|
| 1 | 反序时成功的 pair 不同（`c+d` vs `a+c`） | 预注册中**冻结优先级定义**；把"两顺序都成立"作为验收项之一 |
| 2 | `all_members_blocked` 是新语义 | 门禁语义审查 + 预注册 |
| 3 | 规模：候选面仅 4 个 context | 扩到 ≥12 个 context 并报告区分面 |
| 4 | 只在 3 个种子偏移下验证 | 落地前扩到 ≥5 个 |
| 5 | 审计只覆盖既有 4 个成员族 | 成员集或模板族变化后须重跑 |

> **后续（加固轮，同日）**：风险 2/3/4 已在
> [M4 加固轮结果](M5_B0_M4_HARDENING_RESULT_20260913.md) 处理——规模扩到 12 context / 6 变体、
> 种子扩到 5 个、`all_members_blocked` 消费面清单化。**风险 1、5 仍开放（属 D5）**，
> 且加固轮把风险 3 的结论边界收窄为"仅表面稳健"（新登记 N1）。本表按审计当时状态原样保留。

## §7 纪律

- **未实施 M4**：不改 gate/runner/规则/冻结产物；反事实在命名空间副本中运行。
- 未注册任务、未训练候选专用模型；候选定义只存在于探针内。
- 未改动任何冻结报告/预注册/阈值；`growth_admitted=false`、`can_promote=false` 贯穿。
- **不声称"已实现协作"**：审计通过的是"**测量可信**"，不是"能力已落地"。
- 审计脚本只读；重跑结果一致。
