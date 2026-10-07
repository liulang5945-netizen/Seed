# PLAN-N1-02 · N1 基底归因判读（2026-10-07 判读入库；判据未调）

> **效力**：[预注册](PLAN-N1-02_attribution_prereg_20261007.md)（2026-10-07 冻结，无 owner 新决策点）的 §3 冻结判别规则按原文判读。**判定＝H-A（池化质量淹没序贯）在 P1 主因意义上＝非主因（not_primary）**，触发的是"非主因"的第二支（`F1 ≥ F1c + 0.20`）——序贯信号**在场且被跟随**，不是"被淹没到不驱动发射"。判据未调、未重跑凑线；预算＝1 配置 × 1 面 × 96 题单轮（另 1 次仪器机械修复，两次仪器返工均发生在任何面读数之前，按 S5 轮先例不计预算）。归因不是支线：本件是 PLAN-N1-01（S1 实施预注册）的输入，其重开前置（"基底归因实验另立预注册并出读数"）随本件达成。

## 1. 判据判读（冻结规则逐字对照，主群＝被困 never-LF 拖写代）

| 冻结条件 | 读数 | 是否满足 |
|---|---|---|
| H-A 成立需 `F2 − F1 ≥ 0.20` | 0.390498 | 满足（第一支） |
| H-A 成立需 **且** `F1 ≤ F1c + 0.10` | `F1 − F1c = 0.489032` | **不满足** ⇒ 合取不成立 |
| H-A 非主因 ⇔ `F2 − F1 ≤ 0.10` | 0.390498 | 不满足（第一支未触发） |
| H-A 非主因 ⇔ `F1 ≥ F1c + 0.20` | 0.489032 ≥ 0.20 | **满足 ⇒ 判 not_primary** |

⇒ 落点是"第一支成立、第二支否证"的形态：发射确实高比例跟着**最高质量字节**走（不是跟着随机），但**跟着轨迹序贯走的比例也远高于随机基线**（55% 对 6%）。序贯信号没有死，它是在每步约一半的场合赢不了质量。

## 2. 守卫判读（四条全部实跑，拒绝路径曾被真实触发过）

- **G1′ 行为同一性（判读前置，不过即整件作废）＝ 过**，两条独立读数：
  - 判读器内置基线校验：`behavior_baseline_check` = eaters 247 / never_lf 236 / stoppers 41，与 v37 基线全等（`BASELINE` 常量不符即 raise ⇒ rc=2 拒判）。
  - 逐位对照实跑（2026-10-07 本会话）：诊断件对 v37 件 `--subtree per_item` ⇒ **identical=true、behavior_diff_count=0**；schema_diff_count=288 全部是新增披露字段本身（首条 = `[0].surface_checks[0].trajectory_follow_v43`、`present_in_only_one_side`＝只存在于 v43 侧），即记录器只加披露、未动任何行为列 ⇒ **只读记录器成立**。
- **面指纹（fail-closed）**＝过：`format=taiji-a30-stop-failure-v43`、`trajectory_following=true`、`items_sha256=0541a3f4568a9c5b`（与 v37 基线同面），任一不符即拒判。
- **1:1 对齐＝过**：`trajectory_rows_total=71,937` ＝环内 observe 步总数（失配即拒判——缺陷件时代 73,728≠71,937 被这条守卫正确拒绝过一次，守卫**能为 false** 已被证明）。
- **与 S5 旗标互斥／默认路径逐位不变**＝过：开/关 `--trajectory-following` 各 1 件烟测，per_item identical=true（㊵-478②）。

## 3. 冻结读数（F1–F5 × 三群；单位＝预测步，取值直接抄自 verdict 件）

| 读数 | 主群 never_lf_eaters | 对照 all_eaters | 对照 stoppers |
|---|---|---|---|
| prediction_steps | 60,416 | 63,232 | 8,705 |
| event_absent_steps（电路未发言） | 11,520（19.1%） | 14,080 | 8,705（100%） |
| eligible_steps | 48,896 | 49,152 | 0 |
| **F1 轨迹跟随率** | **0.549141** | 0.549703 | null |
| **F2 质量命中率** | **0.939639** | 0.939687 | null |
| **F1c 随机基线** | **0.060109** | 0.060087 | null |
| **F2 − F1** | **0.390498** | 0.389984 | null |
| F1 − chance | 0.489032 | 0.489616 | null |
| **F3 末字节发射率** | **0.002066** | 0.002075 | null |
| F4 gate 中位 / p90 | 53.2277 / 72.4624 | 53.2255 / 72.4601 | null |
| F5 后继位置权重中位 | 0.237197 | 0.237687 | null |

- **stopper 群整体 `event_absent`（8,705/8,705）**＝提前停止的那批代跑在空库上（自停代根本没被告知内容就停），判别读数对它**无定义**，出 null 而非 0——这是数据性质，不是完整性违规；判读器只在**主群**判别集为空时才拒判。
- 主群 eligible 48,896 与下方资格集 48,707 之差 **189 步**（派生）＝事件内找不到 `codes[p]==prev` 的位置、无后继可跟随（`succ_byte=null`），同样置 null 不冒充 0。

## 4. 归因输入（判据之外的披露读数，2026-10-07 追加进判读器 `attribution_inputs_non_judgment` 字段；不参与判级）

追加后重跑判读器：**33 行新增、0 行删除**，git diff 证明 F1–F5 与 `HA_verdict` 一字未动，rc 仍为 0。主群资格集＝48,707 步（与 F1 同集合，比率可直接对照）：

| 披露读数 | 主群值 | 含义 |
|---|---|---|
| `succ_is_ev_top_rate` | 0.544213 | 54.4% 的步里，序贯后继**本身就是**证据最高质量字节 |
| `mass_hit_and_follow_rate` | 0.532429 | 其中 53.2% 被发射出去 ⇒ 派生比值 0.532429/0.544213 ≈ **0.978**：证据把正确后继放上顶时，解码几乎照发 |
| `succ_is_ev_top_but_not_emitted_rate` | 0.011785 | 只有 1.2% 的步是"证据给对了、解码没跟"（与上一行相加＝0.544214，对上 `succ_is_ev_top_rate` 0.544213，内部一致） |
| `mass_hit_and_not_follow_rate` | 0.40721 | 40.7% 的步发射的是"最高质量但非后继"字节 |
| 派生：顶质量非后继的步占比 | 1 − 0.544213 = **0.455787** | 其中 0.40721/0.455787 ≈ **89%** 被原样发射 ⇒ **失败主面在寻址/池化把非后继放上顶，不在解码无视证据** |
| `follow_rate_first_step` / `first_step_count` | null / 0 | 主群资格集里没有"首步"行（首步属 store 空的 `event_absent`），跟随率是**步深全程**的读数，非首步特例 |
| `event_length_median` / `p90` | 24 / 36 | 所选事件的字节长度 |

- **逐步损耗复合＝"走不到末字节"的算术形态**：每步 0.549 的位置保真，要连续走对 24 步（事件长度中位）的概率 ≈ 0.549^24 ≈ 6×10⁻⁷（派生算式，示意复合量级，非面读数）；实测 F3＝**0.002066**（每步发射末字节的占比）＝轨迹统计性死亡，与 S5 轮"209 条被困行里仅 5 行被终点分支动过、终点检测器无火可点"同一条事实的两种量法。
- **参数级预读（§2）与面层一致**：`gate_bias`/`copy_induce_bias` 均饱和 +2.5、gate 中位 53.2 ⇒ 门大开、诱导在场；F5 后继位置权重中位 0.237 ⇒ 序贯信号在位置权重里**确实占了一席**，但池化后不到一半的步能赢下 argmax。

## 5. 判级与路由（两条都如实写，判级不因叙述而改）

1. **字面路由（预注册 §3 冻结文本）**：not_primary ⇒ "转解码/训练层归因（§4 分流第二行'流畅但内容错误查依据'同型），S1 设计另议"。本条照登，且**下一件归因实验若要走，须另立预注册**（本件预算已用尽）。
2. **但 §4 的披露读数对这条路由加了必须如实登记的条件**：解码对证据是**忠诚**的（证据放顶即发射 ≈97.8%，"解码无视证据"这一形态被否证）；损失落在**每步约 45.6% 的场合寻址/池化把非后继字节放上顶**。因此对 PLAN-N1-01（S1）的设计约束输出为：**杠杆＝逐步位置保真（per-step position fidelity），而不是终点检测器、也不是字节质量池化上的局部调参**——§15.3 方案 A 的"逐位置证据＋回答可重新读取"方向与本读数相容（它保留位置维），S1 预注册须把判据写成**每步跟随率 × 事件长度复合**的形态，并解释它凭什么把 0.549 抬到能走完 24 步。
3. **P1 状态（累计两轮判读）**：S5 出口＝B（病在更深）；本轮把"更深"定位到**位置维在池化处的逐步丢失**（H-C"诱导缺席"已在参数级否证，H-B"证据从未参与"已否证，H-D 末字节可达性 F3＝0.2% 登记为设计约束）。P1"接口病"仍是部分成立——通路活着、信号活着、读出形态丢序。
4. **PLAN-N1-01 重开前置＝已满足**（03 与 ㊵-477 所载"基底归因实验另立预注册并出读数"达成）；S1 属产品改动，按 PLAN-N1-02 §1"判读后如需产品改动另走 PLAN-N1-01 并回 owner"——**预注册可由本轮起草，实施与开跑回 owner**。

## 6. 资产与复算命令

- 判读件（在库，已跟踪）：`reports/taiji_a30_stop_failure_self_v43_trajectory_follow_96_20261007.json`（format `taiji-a30-stop-failure-v43`，36 MB）＋verdict `reports/taiji_n1_trajectory_following_verdict_20261007.json`（status ok、`HA_verdict_on_primary_group=not_primary`、rc 0）。
- 仪器：探针 v43 旗标 `--trajectory-following`（只读记录器）＋判读器 [count_taiji_n1_trajectory_following.py](../../scripts/training/count_taiji_n1_trajectory_following.py)（fail-closed：面指纹／行为复现基线／1:1／轮步交叉核对，任一不符 rc=2 拒判）。
- **判读后机械收尾（㊵-480，与本件同一提交）**：本件在库三件 `.py`（探针 v43、判读器、S5 守卫单测）在㊵-478/㊵-477 未推送段里处于 **ruff/black 红态**（`statistics` 未用导入 1 处＋3 个文件排版不合黑），会被仓根 CI 的 `ruff check .`＋`black --check .` 两步拦下。收尾＝删该未用导入＋对三件跑 black，并证零语义变更：`ast.dump` 逐件比较格式化前后 **AST_EQUAL=True**（三件全中）；格式化后复跑判读器 ⇒ rc=0 且 verdict 件**逐位不变**（`diff` 空）；S5 守卫单测 5 passed；`audit_frozen_marker_formatter_conflict.py --no-write` ⇒ passed:true／failures:[]（冻结字面 marker 未被排版打断）；收尾后 `ruff check .`＝All checks passed、`black --check .`＝1476 files unchanged。探针记录的行号字段（`generation_loop_lines`、`caller`）取自 `taiji/model.py` 源码而非探针自身，故排版不改面件任何字段。
- 复算命令（仓根执行，两条都是本会话实跑过的原文）：
  - `python scripts/training/count_taiji_n1_trajectory_following.py --report reports/taiji_a30_stop_failure_self_v43_trajectory_follow_96_20261007.json --out-report reports/taiji_n1_trajectory_following_verdict_20261007.json` ⇒ rc 0，输出逐位复现 verdict 件（本会话复跑一致）。
  - `python scripts/training/compare_taiji_a30_report_identity.py --left reports/taiji_a30_stop_failure_self_v43_trajectory_follow_96_20261007.json --right reports/taiji_a30_stop_failure_self_v37_peakrun_circuitseedA_96_20261003.json --subtree per_item` ⇒ rc 0，identical=true／behavior_diff 0。
  - 面复跑命令（要重取面件时才需要）＝与 v37 基线同命令（checkpoint `a31_chunked_self/checkpoint.pt`、circuit `circuit-final.pt`、96 题、max_length 256、α=1.0）＋ `--trajectory-following`；零训练。
- **对北极星（§0）的意义**：本轮不产生能力增量，产生的是**通路诊断读数**——"经验→能力"通道里位置维的逐步流失率（F1 0.549/步 ⇒ 末字节到达 0.2%），这是 S1 验收必须打动的量。第 0 点对照不变（装机 13/72、主列 41→59、资格门控 X=0/9/27、复述严格命中 7/24）。
- 不变项：M6 收官、M7-CI 绿、M8 挂起；N1–N6 排序不变；N2（C6 已批）与 N3（甲乙并行已批）两泳道独立推进不受本件影响。台账：08 ㊵-479。
