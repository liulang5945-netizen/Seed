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

## 7. 任务通路级结构与组合证据（第 55–66 轮新增，全部有界）

| 卡 | 结果 | 判词 |
|---|---|---|
| 多读出任务头（MLP 4→16→24） | 旧能力逐位保持 drop 0.0；新任务 0.4375/0.5625/0.5938 | `multihead_isolation_ready`（D-75） |
| 任务专用通路三任务跨会话 | 旧/新 drop 0.0；第三任务 1.0/1.0/1.0；继承 exact | `multihead2_ready`（D-78） |
| 四通路跨会话 | 旧 A/B/C drop 0.0；新 D 0.8125/0.75/0.7188 | `multihead3_ready`（D-79） |
| 五通路压力测试 | 旧 A–D drop 0.0；新 E（颜色）1.0/1.0/1.0 | `multihead4_ready`（D-83；226.23 CPU 超价 200 已披露） |
| 通路回收 | 删除隔离精确（其他通路逐位不变）、常数类基线塌陷、重初始化恢复 0.78/0.75/0.69 | `pathway_recycle_ready`（D-81） |
| 五通路干扰矩阵 | 停用任一通路其余四条 `others_max_delta=0.0`；重新载入逐位恢复 | `pathway_interference_ready`（D-84） |

**结论**：通路级结构机制「按需新增 → 叠加共存 → 回收/重生长 → 互不干扰 → 可逆停用」在单主体 5 任务范围内有界 verified；**不改**单元级生长收益判词（`growth_benefit_not_established_in_package_22`），也不代签共享单元分裂/整合、6+ 任务、真机或 22.8。

**成本**：本轮新增 ≈810 CPU（D-75 45.31、D-76 136.38、D-77 142.09、D-78 141.09、D-79 182.94、D-80 21.95、D-81 22.05、D-83 226.23、D-84 10.23；含超价 26.23）。

**残留项（未变）**：正式 check②／B 父资格／D 口径、`autonomous_choice` negative、`recursive_development`、单元级生长收益、共享单元分裂/整合、6+ 任务、真机/多动力学、100 会话/24h、22.8 与 23–30 准入。

## 8. 任务通路扩展至 8 条与 8×8 干扰矩阵（第 75–78 轮，有界 verified）

| 卡 | 结果 | 判词 |
|---|---|---|
| 七条通路（HEADS=7） | 旧 A–F drop 0.0；新 G 1.0/1.0/1.0 | `multihead6_ready`（D-93） |
| 七通路干扰矩阵 | 其余六条 `others_max_delta=0.0`、重新载入逐位恢复 | `pathway_interference_ready`（D-94） |
| 八条通路（HEADS=8） | 旧 A–G 全部 drop 0.0；新 H 1.0/1.0/1.0；门修正为 all_retentions_exact | `multihead7_ready`（D-95） |
| 八通路干扰矩阵 | 其余七条 `others_max_delta=0.0`；停用塌到常数类基线；重新载入逐位恢复 | `pathway_interference_ready`（D-96，10.00 CPU） |

**结论**：单主体、8 任务、8 头配置下，通路级「按需新增 → 旧任务逐位保持 → 互不干扰 → 可逆停用」完整成立（有界）；未触容量/干扰上限。**披露**：D-93 保持门曾漏检三条旧任务（已修正为覆盖全部旧任务），其数据同样为 0.0；交叉/颜色 mod 3 任务常数基线较高（0.75/0.3125），删除门余量小。

**未覆盖**：9+ 任务、单元级停用/分化（partial，D-88–90）、共享单元分裂/整合、正式 check②、真机/长跨度、22.8 与 23–30。

## 9. 通路扩展至 9 条与 9×9 干扰矩阵（第 79–82 轮，有界 verified）

| 卡 | 结果 | 判词 |
|---|---|---|
| 九条通路（HEADS=9） | 旧 A–H 全部 drop 0.0；新 I 1.0/1.0/1.0；继承 exact | `multihead8_ready`（D-99，397.05 ≤ 价 450） |
| 九通路干扰矩阵（9×9） | 其余八条 `others_max_delta=0.0`；停用塌到常数类基线；重新载入逐位恢复 | `pathway_interference_ready`（D-100，10.23 CPU） |

**结论**：单主体、9 任务、9 头配置下，通路级「按需新增 → 旧任务逐位保持 → 互不干扰 → 可逆停用」完整成立（有界），扩展曲线 3→9 任务一致、未触上限。**披露**：派生任务（象限/交叉/颜色 mod 3/parity）常数类基线较高（0.75/0.3125/0.53125），删除门余量小，后续复用应改口径。

**未覆盖**：10+ 任务、单元级分化（partial，D-88–90）、共享单元分裂/整合、递归发展（uncovered）、正式 check②（A 案维持）、真机/长跨度、22.8 与 23–30。

## 10. 九轴终值（第 84 轮）与剩余方向

| 轴 | 终值 |
|---|---|
| engineering | verified |
| task | verified（`fit_face_only`；正式 check② 未读；A 维持，B 零成本前置已冻结） |
| learning | 路由技能 verified＋**通路级新能力有界 verified（3→9 任务）**；内容积累 uncovered |
| retention | fit 面 verified（严格 D breach 披露）＋**通路隔离逐位保持（drop 0.0）** |
| autonomous_choice | **negative（两法一致）** |
| mature_growth | 结构机制 verified＋**通路级新增/回收/互不干扰/可逆停用有界 verified（3→9 条）**；单元级收益 `growth_benefit_not_established_in_package_22`；单元级分化 `partial` |
| recursive_development | **negative**（重复输入迭代 D-102＋结构稀疏细化 D-50）；顺序计算/显式循环形态 uncovered |
| physical_transfer | **有界 verified（推/位置桶）**；碰撞/遮挡/真机 uncovered |
| lifetime | **有界 verified**（≥3 跨进程会话、出生继承、休眠/退役/唤醒、通路跨会话新增/回收） |
| （附加）uncertainty_expression | 有界 verified（ECE 0.028→0.021） |

**规模成长分项判读**：单位/连接/状态/通信均有机制与有界证据；**有效功能在单元级未建立**、在通路级有界成立——**不得混写**。

**剩余可行方向（优先级）**：① 正式 check② 决策（A 维持 / B 六前置后一次性）；② 顺序计算式递归或显式循环模块（新设计）；③ `autonomous_choice` 在线闭环新设计；④ 共享单元分裂/整合；⑤ 真机/多动力学族；⑥ 100 会话/24h；⑦ 22.8 与 23–30 准入。

**总判定**：目标**未完成**；已有证据链覆盖结构、学习、保持、跨会话、通路级生长与不确定表达；`task` 未见泛化、单元级生长收益、递归、自动选择、真机与长跨度仍未闭合。

## 11. 工作记忆写回（D-20261119-112，有界 verified）

可学二分类两步任务（同向/反向）：oracle 1.0、自写回 1.0、零写回 0.603–0.638、shuffled 0.491 ⇒ `working_memory_writeback_ready`（48.94 CPU）。**显式写回**把 tick1 预测方向写入 tick2 输入即可完成「单 tick 不可能」的组合判断；对比 D-108 隐状态携带不可组合，收益来自写回通道本身。披露：标签不平衡（same 24.1%）、零写回低于多数类；后续加平衡采样。边界：单主体、二分类两步、手工注入；递归发展仍 negative（D-102/D-108），不改变任务轴。

## 12. 去污染严格写回控制（D-20261119-115，有界 verified 升级）

平衡 1:1＋invalid 标记＋配对前打乱行序＋独立置换 shuffled：oracle/自写回 1.0、零写回 0.492–0.523、shuffled 0.500–0.516、necessity ✓ ⇒ `working_memory_writeback_ready`（49.14 CPU）。只有正确写回带来满分；D-114 对照污染归因获证。**显式工作记忆写回**从「不平衡披露下有界」升级为「平衡＋严格控制下有界」；不含一般递归（D-102/D-108 negative）与任务轴变化。

## 13. v5 增量（第 98 轮）：通路 9 条、递归两形态、工作记忆写回、check② 前置

| 证据 | 结果 | 判词 |
|---|---|---|
| 九条任务通路 | 旧 A–H drop 0.0；新 I 1.0/1.0/1.0；继承 exact | `multihead8_ready`（D-99，397.05 ≤ 价 450） |
| 九通路干扰矩阵 | 其余八条 `others_max_delta=0.0`；重新载入逐位恢复 | `pathway_interference_ready`（D-100，10.23 CPU） |
| 重复输入递归 | tick 增益 0/−0.06/0；T=1 等算力基线更高 | negative（D-102） |
| 顺序计算递归 | tick1 0.54–0.63 学会中间预测、tick2≈机会且不优于控制 | negative（D-108，忠实测量） |
| 工作记忆写回（平衡＋严格） | oracle/自写回 1.0；零写回 0.492–0.523；shuffled 0.500–0.516 | **`working_memory_writeback_ready`（D-115，有界）** |
| check② 前置 | A 案维持；B 案零成本前置冻结（digest `050d524e…`） | 待 owner 裁定 |

**九轴增量**：`learning`＋＝通路级 9 任务新能力有界；`retention`＋＝通路隔离逐位保持；`lifetime`＋＝通路跨会话新增/回收/停用/重新载入；`recursive_development`＝negative（两形态）＋工作记忆写回有界 verified（单主体/二分类两步）；`mature_growth` 单元级收益仍未建立、单元级分化 `partial`。

**仍未闭合**：正式 check②（A/B）；`autonomous_choice`（两法负）；共享单元分裂/整合；真机/多动力学族；100 会话/24h；22.8 与 23–30 准入。

## 14. v6 总账（第 102 轮）：结构机制、各轴负结果、剩余优先级

**结构机制总账**

| 机制 | 状态 |
|---|---|
| 出生/继承 | 有界 verified（D-65/78/79） |
| 休眠/唤醒/退役/回收 | 有界 verified（D-66） |
| 通路级新增/回收/互不干扰/可逆停用 | 有界 verified（D-79/81/84/92/94/96/100） |
| 单元级分化 | **range-limited negative**（D-88–90/117–120；共享读出下结构不可达） |
| 单元级整合 | 合并算子无损；功能性整合未证（D-119/120） |

**各轴负结果与收口理由**

| 轴 | 结论 | 依据/条件 |
|---|---|---|
| 单元生长收益 | `growth_benefit_not_established_in_package_22` | D-49；需等 FLOPs/未知最终容量条件重检 |
| 单元贡献/信用 | negative（范围受限） | D-67–71：饱和冗余、边界不足、信用不可分 |
| 递归发展 | negative（重复输入 D-102、顺序计算 D-108）；工作记忆写回有界 verified（D-115） | 需顺序计算/显式循环或任务头路由 |
| 自动选择 | negative（两法一致） | D-60；需在线闭环/部分观测历史 |
| 碰撞/多动力学 | uncovered with evidence | D-57–59：构造可行、预算内不可学 |
| check② | A 案维持（`fit_face_only`，0 暴露）；B 案零成本前置冻结（D-98） | B 剩余四项待 owner 裁定 |

**剩余优先级**：① check② 决策（A/B）；② `autonomous_choice` 在线闭环；③ 顺序计算/显式循环；④ 真机/多动力学；⑤ 100 会话/24h；⑥ 22.8 与 23–30 准入。

**总判定**：目标未完成；结构机制、通路级组合、工作记忆写回已有有界证据；任务未见泛化、单元级分化/生长收益、递归、自动选择、真机与长跨度仍未闭合。

## 15. v7 增量（第 106 轮）：在线选择收口、工作记忆写回、结构轴收口、check② 前置

| 证据 | 结果 | 判词 |
|---|---|---|
| 在线选择 v2（12 箱/16 epoch/pooled 先验） | online 0.742/0.835/0.738 vs 最佳固定 0.84；超等待/随机、在线在学；all≥0.79 与 learning 未过 | `autonomous_choice_online_absent`；第三次 `policy_gain_absent`，**停止同范围**（D-124） |
| 工作记忆写回（平衡＋严格） | oracle/自写回 1.0；零写回 0.492–0.523；shuffled 0.500–0.516 | **有界 verified**（D-115） |
| 分裂/整合 | 分裂只复制（相关 0.98–0.99、可互替）；合并无损；共享读出下分化结构不可达 | range-limited negative（D-119/120 收口） |
| check② | A 案维持；B 零成本前置冻结 | 待裁定（D-97/98） |

**可裁事项**：① check② A/B（B 需补四项前置）；② 22.8 与 23–30 准入；③ 真机/多动力学投入；④ 阶段账处置；⑤ 是否授权「换环境/换问题」新设计（更大 C 池＋更宽策略集；顺序计算/显式循环），建议单卡 ≤300 CPU。

**判定**：目标未完成。现有环境/架构/预算内主要轴负结果已按出口收口；剩余推进依赖上述裁定之一。
