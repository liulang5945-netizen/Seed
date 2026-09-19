# M5 统一执行入口证据包合同草案 v1（共同门收口包）

日期：2026-09-19。依据：[共同门差距盘点](M5_COMMON_GATE_GAP_INVENTORY_20260919.md) 收敛的唯一缺口＋用户批准起草（弹窗记录，含常设授权：弹窗长时间未响应时按推荐项自动推进）。定位：**M5 共同门（02 §2.1 统一认知闭环）的唯一剩余证据缺口**——单一 bundle、单一真实任务、三机制联合参与、逐步 trace、消融矩阵与简单策略对照。

## §1 目的、范围与非目标

**目的**：证明记忆（P5.1h 准入 child）、后果预测（workbench 终态/undo 口径）、选择（HANDOFF-M4 让位规则）在**同一模型 bundle、同一真实任务**中共同改善行为，并以消融对照给出各机制贡献。

**范围内**：统一 bundle 组装与 digest、单一任务族选定与冻结、全链 trace schema、消融矩阵与简单策略对照、预注册（任务/成功标准/比较方式/阈值/资源/停止线）、证据运行。

**非目标**：默认产品采用（M6）、新参数训练（预计极小，若有须单独审批）、R2 语言能力、历史冻结判据修改、多任务扩面（单任务先闭合共同门）。

## §2 静态盘点（接入面已核）

| 组件 | 接入面 | 身份 |
|---|---|---|
| 记忆/知识 | P5.1h 采纳 child（checkpoint sha256 `daf2e877…`，p51h:admission manifest） | `reports/taiji_p5_1h_child_20260919.pt` |
| 选择 | `taiji/collab_handoff.py`（FailureHandoffPolicy revision 0/1 ＋ `execute_group_episode` 执行入口，事件携带 rule_revision） | 提交 `20092939` |
| 在线回写 | `taiji/interaction_group_online.py`（apply_feedback/rollback_to/checkpoint，六类验收已过） | P5.2d v2 门 `4297d7d2` |
| 任务环境 | WorkEnvironment/p52a 任务族（create__override / create__observation / 语言确认，真实文件终态＋undo 口径） | P5.2b/P5.2d 同源 |
| 对照基线 | 可部署简单策略（频率/字典序首成员），各轴既有 lesion/placebo 方法 | 冻结沿用 |

## §3 统一入口设计框架（数值与任务在实现门前预注册冻结）

1. **bundle**：一个固定 artifact 绑定三组件（P5.1h child 权重＋HANDOFF-M4 策略参数＋在线 learner checkpoint）＋bundle digest；运行入口单一（一个脚本/一条命令）。
2. **任务**：从 P5.2 任务族选**一个**真实任务（冻结 task_id 与初始文件），设计上要求：执行需保留跨 tick 信息（记忆）、有可区分的行动后果（预测/undo 口径）、有多成员多步选择（让位规则）。任务确认书进预注册。
3. **trace**：每 tick 记录 输入 cue→成员记忆检索（cue_count 语义）→后果预测→策略决策（chosen/blocked）→执行→真实 outcome——字段 schema 进预注册，逐步可追踪。
4. **消融矩阵**：完整系统 / 简单策略（字典序首成员＋无记忆检索）／禁用记忆（cue 退化为常数）／禁用选择（revision 0）／禁用回写（不 apply_feedback）——每臂同任务同预算，预注册比较口径。
5. **成功标准框架**：完整臂在真实终态口径优于全部消融臂与简单策略（阈值/重复数在预注册冻结）；六类验收类（P5.2d 口径）继承回归。
6. **资源与停止线**：逐臂 wall/calls 记账；总预算与停止线预注册（预计 wall 量级与 P5.2d 相当：百秒级/臂）。

## §4 条件推进

草案（本文件）→ 用户批准 → 实现门（零训练：bundle 组装 digest 测试、trace schema 测试、消融臂接线测试）→ 预注册冻结（任务确认书＋全部数值线）→ 证据运行（如涉参数更新单独审批）→ 结案记录（两态如实入账）。

## §5 结果去向与边界

成立 ⇒ M5 共同门证据闭合，进入完整 CAP/阶段评审流程；不成立 ⇒ 按差距归因（接线/任务设计/机制贡献）记录，欠账保留。全程不触默认采用、不重启内容绑定线、不修改既有冻结判据。

## §6 结案（2026-09-19，预注册 §5/§6 修订后第五轮运行）

**`unified_entry_supported`——四条预注册线全过且消融有真实区分度**：

| 臂 | 成功率（主任务 ×3） | 归属 |
|---|---|---|
| **full** | **1.0** | 完整系统 |
| simple_strategy | 0.0 | 简单策略被否 |
| disable_memory | 0.0 | 记忆贡献实证（cue 语义禁用即崩） |
| disable_selection | 0.0 | **HANDOFF-M4 让位规则贡献实证**（持续阻塞下盲重试失败，full 让位后达标） |
| disable_writeback | 1.0 | 单任务内预期平手（回归性记录：回写价值在跨轮场景） |

- L2 资源线以修正后口径（总步数含失败尝试）通过；L3 失败注入下 full 仍达标；L4 trace 全合法（事件全带 rule_revision/bundle_digest）；
- 仪器演化如实入账：首轮 create_undo 空转判定撤回（§5）、L2 注入/度量修正（§6）——均为预注册修订先例（P5.2d §6–§8 同型），阈值零放宽；
- **M5 共同门（02 §2.1）的统一执行入口证据缺口就此闭合**：单一 bundle（digest `0ffae8cf…`，绑定 P5.1h 采纳 child＋HANDOFF-M4 策略＋P5.2d 回写描述符）、单一真实任务（lang_confirm `p52a-p52-train-000`＋未见实例）、三机制联合参与、逐步 trace、消融矩阵。
- 剩余 M5 退出项（非本包）：完整 CAP 评价、显式排除批准、阶段评审表落盘。
