# PLAN-N5-01 · 四层循环设计预注册（只冻设计不做跑）

日期：2026-10-09 · 线别：N5 自进化四层唤醒 · 母件：[09 §N5](../active/roadmap/09_NEXT_MAINLINE_PLAN.md)（L109-121）
批准：owner 第十次弹窗裁"**(a) 先做设计预注册**"（台账 08 ㊵-570④；本件即该选项的交付）

## 0. 为什么是"设计预注册"

㊵-570 现读证实：训练链只挂了"压强→决策"**前半**（`--growth-min-pressure` + 压强面），`AdaptiveResidualGrowthTrigger.last_decision` **无任何消费点**；四层循环的后半（候选生长→影子学习→贡献/保持验收→准入）的组件在产品里**存在但训练链未接线**：

| 组件 | 产品现状 | 所在 |
| --- | --- | --- |
| 压强→决策 | **已接线**（τ 两链真跑：β 36 步／circuit 33 步） | `train_seed_corpus.py` + `adaptive_residual_growth.py` |
| 候选生长 | 组件完备：`AdaptiveResidualGrowthCandidate`、`AdaptiveResidualShadow.from_parent_bridge` | `adaptive_residual_candidate.py`／`adaptive_residual_shadow.py`（1033 行） |
| 影子学习 | 组件完备：`shadow.learn(...)`／`forward`／`_validate_gain` | 同上 |
| 贡献/保持验收 | 组件完备：`candidate_utility`／`counterfactual_utility`／`candidate_gate` | 同上 |
| 准入/剪枝 | 组件完备：`AdaptiveStructuralGrowthController`／`StructuralAdmissionResult`／`AdaptiveStructuralPruningController` | `structural_growth.py`／`structural_validation.py` |
| 世界学习器 | 组件存在、**零真实调用**（§19.13 P4 痛点） | `WorldDynamicsLearner`（§6 四接口） |
| 训练链消费 | **不存在**（本件要设计的部分） | — |

⇒ N5 的"设计预注册"＝把**训练链如何消费 `Decision`** 冻成可评审文本；实施（产品/训练器代码）与跑面各自另批。

## 1. 四层循环的训练链接线设计（冻）

| 层 | 触发 | 组件接线 | 读数 |
| --- | --- | --- | --- |
| ① 压强→决策 | 已接线（`--growth-min-pressure`） | `AdaptiveResidualGrowthTrigger.observe` → `last_decision` | 压强面（N3 乙已有） |
| ② 候选生长 | `last_decision` 首次过阈 | `AdaptiveResidualShadow.from_parent_bridge(bridge_id=…)` 从父桥 birth anchor 生成影子；挂到 `substrate`（与 episodic 挂载同模式：训练器旗标 `--n5-shadow`，默认关） | 出生面：candidate_id/birth_anchor/gate/unit_count 现数 |
| ③ 影子学习 | 影子在场后持续 | 每批次调 `shadow.learn(...)`（与 `model.observe` 同数据源）；`record_counterfactual_parent_probabilities` 同步记录反事实 | 消费面：candidate_activity/residual_norm/eligibility_norm |
| ④ 验收→准入 | 到达冻结的验收窗 | `_validate_gain` + `candidate_utility` vs `counterfactual_utility` + `candidate_gate`；**同容量对照**＝同最终容量 fixed-large 与随机成长两臂（§9） | 贡献/保持面：验收窗首末对照；资源面：参数量/激活 |
| ⑤ 世界学习器 | 准入后首次 | `observe/propose/feedback` 三接口各一次真实调用（§6 合同），调用与结果全量落盘 | 世界面：三接口的输入输出件 |
| ⑥ 任务矩阵 | 影子准入前后 | 生长前后各跑一次冻结的任务矩阵（N4-01 寻址面＋N2 保持集七列），差值即"生长前后"读数 | 矩阵面：面件对照 |

## 2. 判据（冻；跑面时生效）

* J-N5-1（出生）：决策过阈后影子在场（`candidate_id` 非空、gate 现数）；
* J-N5-2（消费）：`candidate_activity > 0`（影子真的被数据驱动）；
* J-N5-3（贡献）：验收窗内 `candidate_utility > counterfactual_utility`（贡献超过反事实）；
* J-N5-4（保持）：验收窗末 `_validate_gain` 通过且保持集七列不跌（N2 的保持集口径沿用）；
* J-N5-5（世界）：WorldDynamicsLearner 三接口各有一次真实调用且有件；
* 任一不达 ⇒ 对应面如实出版负结果；**全系列不得只测损失下降**（§10 判据原型）。

## 3. 守卫（冻）

* G-N5a：`--n5-shadow` 默认关 ⇒ 训练行为逐位不变（信封键集不含影子键）；
* G-N5b：影子只在 `last_decision` 过阈后创建（压强不过阈 ⇒ 不创建）；
* G-N5c：对照两臂（fixed-large／随机成长）与主臂**同数据、同预算、同验收窗**；
* G-N5d：三对照缺任一 ⇒ 整件 `ran_not_measured`（§8.8 support/query 隔离纪律）。

## 4. 实施与跑的批次

* **实施**（训练器消费钩子＋接线＋契约测）：可自办（设计＝本件 §1）；
* **跑面**：长跑＋对照两臂＋任务矩阵 ⇒ **另批**（算力与时长随跑面预注册单独报）。

## 5. 末尾三行（发表资格前置）

* 预期全绿：G-N5a…d、J-N5-1/2/5（出生/消费/世界——这三个不依赖训练成效）；
* 预期可能不绿：J-N5-3/4（贡献/保持是假设本身，负结果照常出版）；
* 发表资格：跑面读数发表＝**另批**（对照两臂齐后）；本件签字只覆盖"设计冻结"。
