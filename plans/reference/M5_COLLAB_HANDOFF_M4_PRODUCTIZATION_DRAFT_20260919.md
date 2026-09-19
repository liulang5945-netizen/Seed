# M5 协作与自主选择：HANDOFF-M4 规则产品化包合同草案 v1

日期：2026-09-19。依据：用户在 P5.1h 结案后的缺口选择中选定「协作与自主选择——HANDOFF-M4 规则产品化」（问答记录「1」）。定位：[01 §3](../active/roadmap/01_SCOPE_AND_PHASES.md#3-m5-四轴与晋级缺口) 协作轴的已知欠账——**"该规则目前只存在于 gate 脚本，taiji/ 无对应实现 ⇒ 落地证明的是仪器，不是产品能力"**。本轮只做静态盘点与设计，训练另行审批。

## §1 目的、范围与非目标

**目的**：把 `m4_failure_handoff`（`rule_revision=1`，B0 WP-3 于 2026-09-15 落地进 gate 脚本的行为规则）实现进 `taiji/` 产品运行时的成员执行路径，使"失败让位选择＋按成员自身步数的 cue 语义"成为**产品行为**，并可产生同入口选择证据。

**非目标**：N1a binder 改动（跨内容结构，独立议题）；T2 纠错交接（需成员间证据通道，更大改动）；默认产品采用（M6）；R2 语言能力；修改 P5.2b gate 已冻结判据（REQUIRED_GAIN=1.65 / REQUIRED_K=5 / rule_revision=1 全部沿用）。

## §2 静态盘点（已核对）

**规则本体**（`scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py::_member_episode`，WP-3 落地 +21 行/2 处替换）：
- (a) **选择规则**：`chosen = bindable[0]` → 计算 `last_success_index` 与 `blocked`（自身最近一次尝试未执行的成员），取**首个未阻塞**候选；全阻塞 ⇒ 记 `"executed": False` 的终局步并 `finish("all_members_blocked")`——失败即让位，不再重试到 STEP_CAP；
- (b) **cue 规则**：`cues = tuple([cue] * (len(steps) + 1))` → 按 `own_steps`（该成员自己已执行步数）+1——后加入的成员不再被问到自己训练范围之外；
- 配套：`RULE_REVISION = 1`、`COMPOSITION_RULE = "m4_failure_handoff"`、revision-0 描述保留为带标签历史字段；revision-0 变体（m1a/m1b/m2a/m3）仍拒绝构造。

**落点现状（grep 已核）**：`m4_failure_handoff`/`bindable`/`chosen` 仅存在于 `scripts/training/` 的 gate/probe/audit 脚本；`taiji/` 与 `seed_platform/` **零命中**——产品运行时的成员执行路径没有该行为。

**冻结证据（产品化必须对齐的基准）**：WP-3 出口②③——落地后结构空间探针 11 格与封存扩面报告逐字段相同（三格 0→+2.000、交错 0→6、7/7 偏移为正）；反事实冻结面 regressions=0/improvements=2；`all_members_blocked` 停止原因消费面 17 文件/10 判断点（2026-09-15 重扫，`drift.clean=true`）。

## §3 产品化设计框架

1. **产品宿主选定（实现前冻结）**：静态映射 interaction-group 执行路径中成员选择与 cue 构造的产品等价点（候选：`taiji/interaction_group_online.py` / `interaction_group_learning.py` 的成员回合逻辑）；映射结果写入实现门报告，无等价点则先立最小产品执行入口（范围仍限于让位规则）。
2. **规则参数化**：产品实现带 `composition_rule`/`rule_revision` 字段（revision 0 = 字典序首成员，可复现历史行为；revision 1 = 让位规则），运行时显式选择，不做静默切换。
3. **仪器-产品对齐测试**：同一输入下产品实现与 gate 脚本 `_member_episode` 的选择序列/终止原因逐位一致（gate 作为规格的参照实现）。
4. **同入口选择证据**：产品路径上可触发失败让位与 `all_members_blocked` 终局，事件记录含 rule_revision；不再出现"只有仪器会做"的状态。
5. **回归**：revision 0 行为 + 既有 interaction/selection 测试全绿；无训练需求（规则是执行期选择逻辑，不涉参数更新——若实现中发现需要训练另行审批）。

## §4 条件推进

草案（本文件）→ 用户批准 → 实现门（零训练测试＋宿主映射报告）→ 同入口选择证据运行 → 结案记录。判定两态（规则在产品路径成立 / 不成立并记录原因）都如实入账。

## §5 结果去向与边界

成立 ⇒ 协作轴"产品同入口选择"欠账闭合一项，轴级共同门/批准仍另评；不成立 ⇒ 记录产品宿主的结构性障碍，欠账保留。全程不触碰默认产品采用、不重启内容绑定线、不修改 P5.2b 冻结判据。

## §6 结案（2026-09-19）

实现门＋同入口选择证据全部完成，**协作轴"产品同入口选择"欠账闭合**：

- 产品组件 `taiji/collab_handoff.py`（FailureHandoffPolicy revision 0/1 ＋ `execute_group_episode` 产品执行入口），9 门实现测试全绿（含 400 场景仪器-产品对齐 fuzz）；
- 同入口证据 `reports/taiji_collab_handoff_entry_evidence_20260919.json`：三场景确定性运行——handoff_then_goal（m0 失败→让位 m1→goal_reached）、all_members_blocked、all_members_exhausted——全部事件携带 rule_revision=1，五项检查全过；
- `taiji/` 从此拥有 HANDOFF-M4 的产品实现（此前只有 gate 脚本仪器）；后续 interaction 在线路径接入该组件即获得让位选择语义。
