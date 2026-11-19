# 会话收口报告（2026-11-20，第 40–54 轮）：证据链、九轴、成本与未完成项

> 文档状态：[会话收口/只读汇总] — 2026-11-20。**零训练、零 check/final。** 本报告只汇总本会话新增证据，不改任何既有判词。目标**未完成**，主线继续。

## 1. 本会话关键结果（按重要性）

| # | 结果 | 判词 | 证据 |
|---|---|---|---|
| 1 | 生长机制三层缺陷修复（漏投影／双零死单元／单步探针） | 机制 clean | D-44/46/47 |
| 2 | 生长收益三次无混杂负结果（+1/+3 单位、早/中出生） | `growth_gain_absent` ×3；收益记 `growth_benefit_not_established_in_package_22` | D-47/48/49 |
| 3 | 仿真动作片段（N=3 推、位置桶、四臂反事实） | **`action_consequence_pass`**；`physical_transfer` 有界 verified | D-55 |
| 4 | 碰撞/多动力学族（自由推＋受控几何） | `collision_family_uncovered_with_evidence` | D-57/58/59 |
| 5 | 自动选择第二方法（成本敏感上下文策略） | `policy_gain_absent`；`autonomous_choice` negative | D-60 |
| 6 | 不确定性表达（集成＋温度缩放，256 世界） | **`calibration_already_adequate`**（ECE 0.028→0.021）；有界 verified | D-61/62 |
| 7 | 多会话连续性（3 独立进程） | **`multi_session_continuity_ready`** | D-63 |
| 8 | 跨会话出生与继承（30→33） | **`multi_session_birth_ready`** | D-65 |
| 9 | 跨会话休眠/退役/唤醒 | **`multi_session_lifecycle_ready`** | D-66 |
| 10 | 单元因果贡献（饱和/边界/早出生） | `unit_contribution_absent`×3（冗余非死单元） | D-67/68/69 |
| 11 | 22.6.3 递归相依设计两版（稀疏细化、多 seed） | `recursive_multiseed_absent`；uncovered | D-50 |
| 12 | 22.4.2 动作卡工程失败（fit 混训）＋通道路由缺陷 | `engineering_failure`；披露 groups=4 只覆盖 0–31 维 | D-53/54 |
| 13 | 22.7b 原卡（未投影协议） | `retention_breach`（归因漏投影） | D-42 |

## 2. 九轴当前值（同父 seed `20261112` 谱系）

| 轴 | 值 |
|---|---|
| engineering | verified |
| task | verified（`fit_face_only`；正式 check② 未结） |
| learning | verified（路由技能）／uncovered（内容积累） |
| retention | verified（严格 D breach 已披露） |
| autonomous_choice | **negative**（两法一致） |
| mature_growth | 结构机制 verified／收益 `growth_benefit_not_established_in_package_22`／单元贡献 negative（范围受限） |
| recursive_development | **uncovered** |
| physical_transfer | **verified（有界：推/位置桶）**；碰撞/遮挡与真机 uncovered |
| lifetime | **verified（有界：≥3 会话跨进程、出生/休眠/退役/唤醒）**；100 会话/24h uncovered |
| 附加 uncertainty_expression | **verified（有界：ECE 0.028→0.021）** |

## 3. 成本（本会话新增）

- 22.7b 系列（含超帽事故）≈1,800；22.7e/f/g ≈96×3；22.4.2/2b ≈68+105；碰撞两探针 ≈120；22.6.2b ≈22；22.6.4/4b ≈66；22.6.3 可行性 ≈150；多会话三探针 ≈60×3；单元贡献三探针 ≈113；合计 **≈3,200 CPU**（明细见各回执）。
- 阶段账仍为「超支永久记事故、不改帽」；每卡单价与帽单独定价。

## 4. 残留风险与未结项

1. **正式 check②** 未裁决（A 案 fit_face_only 维持；B 案需先冻 D 口径）；**B 父资格未正式化**。
2. `autonomous_choice` 两法负 → 需**在线闭环探索/部分观测历史**新设计。
3. `recursive_development` 需**稳健相依设计**（两次稀疏细化两版均失败）。
4. 生长收益需**等 FLOPs／未知最终容量／32→64 大跳跃/计算受限**条件才可重检。
5. 单元贡献在 joint24 上为**冗余**；需独特功能任务/连接级消融/功能分化。
6. 物理轴仅**仿真推/位置桶**；碰撞族构造可行但预算内不可学；真机未接。
7. 长跨度 lifetime（100 会话/24h）未测；22.8、23–30 仍**锁定**。

## 5. 待裁四项（见 [待裁汇总](MEMO-TAIJI_OWNER_DECISIONS_20261120.md)）

A 正式资格 A/B；B 22.8 与 23–30 准入；C 真机/多动力学族；D 阶段账处置。

## 6. 结论

**目标未完成。** 本会话把「证据链」系统化：两个能力卡有界通过（动作片段、不确定校准）、三个跨会话生命周期卡有界通过、生长机制缺陷修净但收益未建立、自动选择与单元贡献为范围受限负结果。后续优先级：正式 check② 决策 ＞ recursion/choice 新设计 ＞ 生长收益条件化重检 ＞ 真机前仿真扩展。
