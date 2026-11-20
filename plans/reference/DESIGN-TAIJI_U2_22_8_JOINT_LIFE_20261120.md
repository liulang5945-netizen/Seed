# 22.8 S0 预注册：同主体联合生命片段与全局收束（设计·待冻）

> 文档状态：[预注册草案] — 2026-11-20，主线队首 `22.8`。上游：[MAINLINE §3.4](../MAINLINE.md#34-228-同主体联合生命片段与全局收束-2280–2284)、[22.5.1 正式结案](RESULT-TAIJI_U2_22_5_1_FORMAL_CHECK2_20261120.md)（B 父 `child-20261112.pt`，`joint_gate_check2_pass / b_parent_earned`）、[22.4.2b 动作片段](RESULT-TAIJI_U2_22_4_2B_ACTION_20261120.md)（`action_consequence_pass`）、[22.6.0 C 任务合同](../manifests/taiji_u2_22_6_0_c_task_20261120.json)（`c_task_ready`）、[22.7 结案](RESULT-TAIJI_U2_22_7_WINDOW_20261120.md)。**未冻结、未运行、不占额度、不读 check/final。**

## 0. 唯一问题

同一可追溯主体（B 父谱系）在**连续 3 个会话 × 每会话 32 轮**的联合生命片段中，能否同时满足：

1. **中文任务**（joint24 颜色×位置、问答/表达）保持并改进；
2. **物理动作后果**（推方向→位置桶）在轨迹内被真实使用；
3. **反馈纠错**（一次真实反馈改变后，纠错净益与旧能力保持）；
4. **恢复**（会话间完整恢复，主体身份与谱系连续）；
5. **无提示活动**（C 任务 WAIT/OBSERVE 选择，无新提示时段不空转、不退化）；
6. 按 appearance / dynamics / occlusion / conflict / counterfactual 分账后，**无单项靠其它项代签**。

22.8 只回答这一联合问题；不新增 22.9，收束后输出 23 立项依据与九轴资格表。

## 1. 候选与身份（22.8.0 冻结项）

| 项 | 冻结值（S0 草案） |
|---|---|
| 主体 | B 父谱系 `child-20261112.pt`（22.5.1 `b_parent_earned`）；**不拼装其它 seed 的读数** |
| 结构 | Route-B 30 单位＋任务头/通路（复用 22.5.1 冻结实现）；如需新增头/通路，必须先过「中性出生/迁移差集」检查 |
| 初始状态 | B 父 bundle（参数/optimizer/RNG/cursor 摘要）；会话开始时从包恢复，会话结束时原子落盘 |
| 身份 | 同一父、同一 schema；每个 session 写 lineage 记录；不得以新初始化冒充连续主体 |

## 2. 会话与轮次协议

- **3 seed × 3 session × 32 轮**；每轮＝一个片段，按预注册顺序表抽取（固定 world/事件索引流，跨臂一致）。
- **片段类型与分母**（每 session 32 轮固定分配，读前冻结）：

| 类型 | 轮数/session | 任务 | 主要指标 |
|---|---|---|---|
| appearance | 8 | joint24（颜色×位置） | 24 类准确率 |
| dynamics | 8 | 22.4.2b 动作片段（N=3 推→位置桶） | 3 类准确率、动作边际（mask/permute 对照） |
| occlusion/conflict | 6 | 遮挡/冲突合法输入（缺失或矛盾通道） | 拒答/未知正确率、不误答率 |
| correction | 4 | 一次真实反馈改变后重测（4 类） | 纠错净益、旧保持 |
| no-prompt | 6 | C 任务（22.6.0 冻结 fixture） | 净效用（收益−成本）、空转/等待合理性 |

- **反馈与后果**：轮内动作/答案由真实世界（仿真）判定，反馈只在下一轮之前可用；不得用 check/final。
- **恢复**：每 session 结束与开始时各一次完整 bundle 恢复（独立进程语义）；恢复一致性按 state_dict/摘要逐位。

## 3. 门（读结果前冻结）

| 编号 | 条件 | 出口 |
|---|---|---|
| J1 中文任务 | appearance 准确率 ≥0.60 且 ≥多数类＋0.10，3 seed 同向 | `joint_task_verified` |
| J2 物理动作 | dynamics 准确率 ≥0.45 且动作边际 ≥0.05（mask/permute 各 ≥0.05），3 seed 同向 | `joint_action_verified` |
| J3 纠错 | correction 净益 ≥0.05 且 95% 下界>0；旧保持损伤 ≤0.02 | `joint_correction_verified` |
| J4 恢复 | 3/3 seed 跨会话恢复逐位一致、lineage 连续 | `joint_recovery_verified` |
| J5 无提示 | no-prompt 净效用 ≥最强固定策略＋0.05；空转率 ≤0.30 | `joint_idle_verified` |
| J6 分账 | 五项分账读数齐全，无任一项以另一项代签 | `joint_breakdown_complete` |
| 总判 | J1–J6 全过 ⇒ `joint_life_segment_verified`；部分过 ⇒ `joint_life_segment_partial`（列失败子项）；全负 ⇒ `joint_life_segment_negative` | —— |
| 兜底 | 恢复/身份失败 ⇒ `engineering_failure`；超帽 ⇒ `budget_stop`；未获 final 授权 ⇒ `final_not_authorized`（不视为失败） | —— |

## 4. 曝光与 `final`

- **check**：22.8 全局分配 **1 次**；仅用于 22.8.2 最后一次 rehearsal，读前冻结判据、只释放聚合计数；**首次访问即扣、不可撤销**。
- **final**：`final 128 world` **只在当次 `R-SEAL-1` 明确授权后**开启；S0 不生成、不读取；未获授权按 `final_not_authorized` 收口，不以其它分区替代。
- **封存面**：不提前生成 check/final 标签；不将 fit 数据混入训练。

## 5. 预算（22.8 计划 4,000＋余量 800＝4,800 CPU）

| 阶段 | 内容 | 帽 CPU |
|---|---|---|
| S1 冻结与 smoke | 候选摘要、片段数据流、3 session 轨迹、单轮 smoke（不训练/极小） | 250 |
| S2 3 seed × 3 session × 32 轮 | 会话内训练/推断＋恢复＋纠错＋无提示 | 3,000 |
| S3 check rehearsal | 1 次、只释放聚合计数 | 400 |
| S4 结账与 9 轴表 | 收束卡、23 立项依据、全局收束 | 200 |
| 合计 | | **3,850**（余 950 不自动花） |

## 6. 与既有证据的关系

- 22.4.2b（`action_consequence_pass`）与 22.6.0（`c_task_ready`）是**片段级**有界证据，本卡把它们放入同一主体连续轨迹，检验**联合**而非重复单卡；
- 22.5.1 已给 B 父 `joint_gate_check2_pass / b_parent_earned`，本卡不重跑 22.5.1，也不改写其判词；
- 22.7 结案 `growth_benefit_absent` 不阻断 22.8：本卡不需要生长增益，只用已验机制（中性出生/生命周期）保持身份。

## 7. 未冻结前必须清的四项（S1 前）

1. **片段数据流冻结**：五类片段各自的 world/事件索引、配对与顺序表、分母；读前 `criteria_sha256` 锁定；
2. **单主体承载方式**：B 父＋新任务头/通路是否需新增参数；若新增，须先过中性出生与迁移差集（不得伪称同一主体）；
3. **恢复与事务**：3 session 的 bundle 落盘/恢复路径、崩溃边界、lineage 记录格式；
4. **曝光账**：check 1 次与 final 授权的当次记录；任何提前读取按流程事故作废。

## 8. 不签

不签 final 通过、C/D/J 全任务、模型级成型、可独立接管交付；22.8 结案只输出范围限定结论与 23 立项依据。
