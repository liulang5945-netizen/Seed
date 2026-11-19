# 包 22 阶段收口账（2026-11-20）：九轴资格向量、成本、残留风险与待裁项

> 文档状态：[阶段收口账] — 2026-11-20。范围：`22.5.0 / 22.5.1 / 22.6.x / 22.7.x` 已执行部分。**本账只汇总既有回执，不新训练、不读 check/final、不签新能力。** 同一主体＝已接受 v1 A 父 seed `20261112` 谱系（含 22.5.1 三训练子种子子体）。

## 1. 九轴资格向量（当前值；`verified` 均标注生效范围）

| 轴 | 当前值 | 依据 | 生效范围/限制 |
|---|---|---|---|
| `engineering` | **verified** | 22.3/22.4 已过；S0 身份隔离＋load 等价 3/3＋联合门 capsule 跨进程恢复；生命周期结构 13/13＋11/11 | 全程 preflight 与恢复；不签外部长时稳定性 |
| `task` | **verified（`fit_face_only`）** | 22.5.0 A 父挣得（记录 check 中位 `0.8906`）；22.5.1 fit 面 B `24/24`、D 相对 `22/24` 对 `9/24`、A `23/24`；未见模板无退化（饱和） | **非正式 check②**；B 父资格未正式化；架构门未跑 |
| `learning` | **verified（路由技能）/ uncovered（内容积累）** | 联合门相对冻结更新净益 `1.0`（LB `1.0`、3 seed 同向）；22.6.1 预测器 Huber 改善 `14.25%` | 跨会话探针 `cross_session_no_learning` 为天花板（非「学不会」）；**内容增量写入未证** |
| `retention` | **verified（有披露）** | A 保持 child `23/24` vs parent `23/24`（单侧上界 `0.0`）；D 同条件相对上界 `−0.375` | **严格 D 口径（父 A 真值）会判 breach**（`2/24`，已并列披露）；旧能力仅 fit 面 |
| `autonomous_choice` | **negative** | 22.6.0 fixture `c_task_ready`、22.6.1 预测器通过；**22.6.2 策略对照 `policy_gain_absent`**（learnable `0.6637` vs 最强固定 `0.83`，3 seed 全负） | 贪心用离线收益预测≠决策改进；需闭环反馈策略或风险校准（新卡） |
| `mature_growth` | **结构 verified / 收益 no-growth** | 22.7 结构生命周期 13/13＋11/11；硬度校准两轮无中间带 ⇒ **`growth_not_needed_in_scope`** | 单位图 32 单元 fit 已达 `1.0`；七臂比较未执行（未训练）；连接信用 `uncovered` |
| `recursive_development` | **uncovered** | 22.6.3「两次相依改进决策」因可教授面全饱和**受阻暂缓**（D-20261119-28） | 需新任务族（新谓词/混淆样本等） |
| `physical_transfer` | **uncovered** | `world.query` 动作语义未接入；无感知-动作-回执因果实验 | 需动作接口与 22.4.2 类片段设计 |
| `lifetime` | **verified（有界）/ uncovered（长跨度）** | 跨会话继续训练不退化＋继承参数逐位不变；独立进程重载逐位一致 | 仅 1 次会话边界、单进程尺度；多会话/24h 级未测 |

**主体身份备注**：所有 `verified` 来自同一父 seed `20261112` 谱系＋同一批子体；未拼 seed、未换父。

## 2. 成本（阶段账）

- 阶段帽 `24,000`；本阶段已知超支（D-20261119-15）：`≈35,250`（含 S0 诊断 ≈1,545、定位 93.5、联合门 S2–S4 ≈1,510、22.6/22.7 探针与校准 ≈600）。
- 本收口账新增成本 `0`（只读汇总）；未训练、未读 check/final。

## 3. 残留风险与未结项（按影响排序）

1. **正式 check② 未裁决**：A 案（`fit_face_only` 豁免）为当前工作路径，B 案（1 次 check 曝光、帽 540 CPU）可随时切换；**B 父资格未正式化**。
2. **D 口径歧义**：同条件参考通过、父 A 真值参考 breach；正式 check 前必须由 owner 冻结口径。
3. **S1 豁免残留**：旧 A 父生成链未重跑，仅 strict load＋记录 fit 复现；若日后怀疑谱系漂移需重开。
4. **架构优越性门未跑**：等算力配平前置未满足；不得以本账签架构优越性。
5. **22.6 策略负结果**：贪心预测器策略劣于最强固定；改进需闭环反馈/风险校准（新卡）。
6. **22.7 生长收益未证**：任务无中间带；`no_growth_needed_in_scope` 只证明「该范围不需要生长」，不是「生长有收益」。
7. **recursive_development / physical_transfer / 长跨度 lifetime** 仍 `uncovered`。

## 4. 待 owner 裁定（决策点）

1. 正式资格路径：A（`fit_face_only` 豁免，0 CPU）还是 B（1 次 check 曝光，帽 540 CPU）；若选 B，先冻 D 口径。
2. 是否投入**新任务族**（混淆样本/多属性联合/J 型序列）以重新打开 22.6.3/22.7 生长收益与递归发展（需新卡与预算）。
3. 22.8 与 23–30 长期路线是否准入（当前全部锁定）。
4. 阶段账处置：维持「超支永久记事故、不改帽」还是重冻阶段帽。

## 5. 建议（执行者意见，owner 可回溯）

- **先做 (1)+(2) 的零训练决策支持**：A/B 一页提案已备（D-20261119-14）；新任务族可用 C 池与 first-A 素材先做「中间带构造可行性」探针（零训练），确认头空间后再开卡。
- **22.8/23–30 不自动解锁**：本账九轴仍有 3 个 `uncovered`，按主线纪律不具备整包接受条件。
- 若两周内无法获得 owner 裁定，建议把「正式资格」「新任务族」「阶段账」三项冻结为显式待裁，继续只做零训练结构/接口证据，避免再次超支。

## 6. 更新（D-20261119-48）：生长机制线主机质已净、收益 uncovered

22.7e/f/g 三个无混杂负结果（前置门全过：中性出生、活单元、旧能力 0、恢复 exact、引用清理；净益均值 −0.0104/−0.0104/−0.0521，均 `growth_gain_absent`）。**mature_growth 轴更新为：结构/机制 verified ＋ 收益 `uncovered`（范围受限）**；详见 [生长线收口备忘](../reference/MEMO-TAIJI_U2_22_7_GROWTH_LINE_20261120.md)。不再以「22.7 硬度校准」单一理由描述该轴。

## 7. 终局更新（D-20261119-49）：生长收益裁决

`mature_growth` 最终记为：**结构/机制 verified ＋ 收益 `growth_benefit_not_established_in_package_22`**（负结果带条件，非「生长无用」）。重检前置条件见[裁决备忘](../reference/MEMO-TAIJI_U2_GROWTH_BENEFIT_VERDICT_20261120.md) §3；转向优先 22.6.3 递归发展或动作/物理轴。

## 8. 范围限制披露（D-20261119-54）：groups=4 的输入通道覆盖

`_BaseGraph` 按 `group % 6` 路由通道：22.5.1 之后用 `groups=4` 的卡（α r4、joint24、22.7b–g）输入只覆盖 48 维中的 0–31 维（goal/feedback 未路由）。任务可学性结果不作废，但属**特征覆盖不完整**的范围限制；动作轴须改用 `groups=6`。详见 D-20261119-54。

## 9. 更新（D-20261119-55）：physical_transfer 有界 verified

22.4.2b 动作片段通过（`action_consequence_pass`）：动作方向对后果预测有真实因果贡献（full 0.781 vs 删/置换动作维 0.41–0.45，配对 95% LB 0.0625>0）。`physical_transfer` 轴更新为 **verified（有界：仿真动作片段，N=3 推、位置桶、单主体）**；真机与多动力学族仍 `uncovered`。既有动作卡 22.4.2 的 `engineering_failure` 与 D-20261119-54 通道路由披露均保留。

## 10. 九轴终值（2026-11-20，D-20261119-56）

| 轴 | 值 |
|---|---|
| engineering | verified |
| task | verified(`fit_face_only`)；正式 check② 未结 |
| learning | verified(路由技能)／uncovered(内容积累) |
| retention | verified（严格 D breach 已披露） |
| autonomous_choice | negative（22.6.2） |
| mature_growth | 结构机制 verified／收益 `growth_benefit_not_established_in_package_22` |
| recursive_development | uncovered（22.6.3 两次相依纠正多 seed 不成立） |
| physical_transfer | **verified（有界：22.4.2b 仿真动作片段）**；真机/多动力学族 uncovered |
| lifetime | verified（有界）／长跨度 uncovered |

待裁四项与建议见 [待裁汇总](../reference/MEMO-TAIJI_OWNER_DECISIONS_20261120.md)。

## 11. 碰撞/多动力学族收口（D-20261119-59）

自由推与受控几何两个探针均未通过（后者数据可分 oracle≈0.93 但预算内模型未学到方向×几何合取）⇒ 该族记 `collision_family_uncovered_with_evidence`；`physical_transfer` 维持**有界 verified（22.4.2b 推/位置桶）**，真机与多动力学族仍 `uncovered`。

## 12. 不确定性表达（D-20261119-62，有界 verified）

22.6.4b 校准卡：joint24、groups=6、30 单位、3-seed 集成，test acc 0.9531、ECE 0.028→0.021、Brier 同步下降，判 `calibration_already_adequate`。**不确定性表达 = verified（有界：joint24 置信校准）**；连续中文表达、拒答/求助策略、跨任务校准仍 `uncovered`。

## 13. 多会话连续性（D-20261119-63，有界 verified）

三个**独立进程** session：后一 session 从 capsule 载入全新 graph 继续训练；3 seed 全部参数继承逐位一致、身份/谱系不变、准确率不降反升（如 0.4375→0.9062→0.9375）。`lifetime` 轴更新为 **verified（有界：≥3 会话跨进程）**；22.8 完整验收（≥3 session×32 轮 J 轨迹）与 100 会话/24h 级仍 `uncovered`。

## 14. 跨会话结构出生与继承（D-20261119-65，有界 verified）

3 个独立进程会话：session 2 活入口出生 3 单位（30→33，出生中性差 0.0），session 3 按 capsule spec 重建并继承全部参数；3 seed 全部参数逐位继承、newborn u0030–32 身份保持、准确率不降反升（如 0.4375→0.875→0.9375）。**「继承式出生＋跨会话延续」= verified（有界）**；分化/整合/休眠/退役跨会话、22.8 完整验收、真机与 100 会话级仍 `uncovered`。

## 15. 跨会话生命周期（D-20261119-66，有界 verified）

session 2 在训练后把 3 个新生儿置 2 dormant＋1 retired；session 3 载入后生命周期状态持久化，唤醒一个 dormant 单元：活动恢复（0.40–0.80）、logits 变化 0.57–0.70、参数 digest 不变、退役单元活动为 0、准确率不降反升。**「出生/休眠/退役/唤醒跨会话」= verified（有界）**；分化/整合跨会话、22.8 完整验收、真机与长跨度仍 `uncovered`。首版「训练前休眠致 p=0」设计缺陷已披露。

## 16. 单元因果贡献（D-20261119-67/68/69，范围受限 negative）

三探针一致：饱和任务下新生儿冗余；容量边界（12 单位）新生儿下降 0.03–0.06 不及旧单位 0.17–0.31；早出生（step 250、1,250 步后训）仍不稳（−0.03/−0.03/0.16，配对 LB<0）。⇒ **贡献子项 = negative（范围受限：joint24、偶发生长、当前容量）**；不改结构机制 verified 与生长收益判词。后续条件：新生儿承担独特功能的任务、连接级消融、或单元功能分化设计。

## 17. 跨会话持续学习（D-20261119-72，范围受限 negative）

joint24 旧能力（0.81–0.91）在动作任务上顺序微调 800 步后掉到 0.03–0.13（遗忘 0.69–0.88）；20% 旧任务重放只恢复到 0.16–0.34（遗忘仍 0.53–0.66），达不到 0.02 保持门。⇒ **「学习新能力并保持旧能力」在当前顺序微调协议下 = negative（灾难性遗忘）**。后继条件：50% 重放、参数隔离（独立读出＋新增单元冻结旧参）、EWC 类正则、或任务分隔下限；新卡预注册，不得同范围重试。

## 18. 持续学习结构性阻碍（D-20261119-73，零训练）

单读出架构（`RouteBGraph.readout`）是任务唯一出口：冻结旧单元会被新单元 `p` 列改变旧读出；冻结读出则新任务无判别方向 ⇒ **参数隔离不可实现**。`learning(新)＋retention(旧)` 组合在架构变更前记**范围受限 negative**；后继需任务专属/条件读出或多模块独立读出（新 pin、新卡、owner 裁定），不得靠重放比例或调门凑过。
