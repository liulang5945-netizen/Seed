# PLAN-N1-166：未见规则组合迁移确认

**状态：未执行、未生成题面、未消费确认面。N1-165原训练充分性门仅2/3通过；按原前置本计划不准入。2026-10-21起依用户优先级延后，先推进自然多轮对话训练与自由生成检查；日后重开须重新核对范围，不更改本计划的历史判据。**  
**日期：2026-10-20**  
**前置训练：** [N1-165中间剂量校准](PLAN-N1-165_transition_template_mid_dose_calibration_20261020.md)  
**问题边界：** [N1-163结案](PLAN-N1-163_execution_closeout_20261020.md) · [N1-164逐题归因](PLAN-N1-164_plan163_itemwise_pair_attribution_20261020.md)

## 要回答的问题

前序评估主要更换实体、材料值、问句表面、上下文长度或训练模板比例，底层规则仍与训练相同。N1-166改测训练中没有出现过的**规则组合**：单步转向作为熟悉原语控制，另测两步转向、三步转向以及方向修订后转向。题目同时改变实体、干扰和上下文位置，但不把换实体或换句式单独当作迁移证据。

有限确认集无法证明普遍智能。本计划若通过，只支持模型在两套冻结生成程序与所列组合上的迁移。

## 两套独立生成程序

N1-166A与N1-166B各生成一张独立face，使用不同随机种子和互不重叠的实体/材料值。两套builder不导入N1-153、N1-162或N1-165的face/schedule生成函数：A以当前观察记录语法表达规则；B由独立实现将同一潜在程序映射到另一套记录顺序和提问renderer。两套scorer分别从隐藏的潜在程序重新计算答案，不读取builder输出的expected-answer列。读取权限分离，12 endpoint输出全部冻结后才揭示标签并评分。

每张face按以下四个条件各96题/endpoint，共384题；每条件按8/16/32行上下文各32题，长度内平衡规则序列、目标位置和干扰实体距离。A/B合计每endpoint 768题。

1. **单步控制：** 4个初始方向×2种转向，平衡重复。
2. **两步转向组合：** 4个初始方向×4种左右转向序列，覆盖全部组合。
3. **三步转向组合：** 4个初始方向×8种左右转向序列，覆盖全部组合。
4. **方向修订后转向：** 先给初始方向，再修订当前方向，最后执行一次转向；完整交叉初始值、修订值及左右转向。

单步控制含已见问句形式与新renderer表面，各半；组合条件也均衡两个提问表面，但新表面作为独立分层报告。规则签名由潜在运算序列定义：训练schedule只有单步`turn`和材料类revision；`turn→turn`、`turn→turn→turn`、`revise(direction)→turn`必须在训练来源中零命中，否则拒绝生成。

## 测量endpoint

同一两张face由12个endpoint各查询一次，共**9,216条**自由生成查询：seed11/12/13各包括N1-162B共同父档、N1-163 surface replay control、N1-163全量preferred candidate、N1-165中间剂量candidate。每个endpoint先做fresh-process完整bundle恢复；同seed所有checkpoint的父SHA、累积更新、optimizer/RNG/history均绑定报告。训练门未通过的N1-163锚点保留作描述性参照，不进入迁移通过统计。

评价以自由生成严格方向答案为主，同时报告合法方向词率、字符边界停止、两个renderer和各类规则组合的逐题配对变化。train、dev和旧holdout都不查询，不训练、不更新、不写checkpoint。

## 预注册通过门

只有N1-165三个mid-dose endpoint均通过原N1-132训练充分性门（3/3），才生成N1-166A/B face。mid-dose候选必须在两个独立renderer分别满足：

- 每个seed的单步控制至少 **88/96** 严格正确。
- 每个seed的两步、三步和方向修订后转向各至少 **80/96** 严格正确。
- 每个seed、每类规则组合在8/16/32行各至少 **22/32**，避免总分掩盖单一长度失败。
- 同seedmid-dose candidate在每个renderer的三个组合条件上均高于N1-162B parent；相较N1-163 replay control的配对净差不得低于−8/96。逐题McNemar exact与区间一并报告，但不替代整数门。
- 两套face的来源排除、rule-signature零重叠、独立语义重算、无标签泄漏、checkpoint/source身份和重复查询审计全部通过。

这组门通过只称为“两套冻结生成程序上的组合迁移通过”；`can_promote=false`，不声称自然语言广泛泛化、跨领域推理或通用智能。

## 前置、成本与停止

- CPU≤12,000秒，包含12 endpoint独立恢复、A/B两face生成与审计、全部9,216条查询、独立语义评分及逐题归因。
- 生成前来源并集必须排除N1-149至N1-165全部训练题和已消费face中的实体/材料值；方向/左右转向等已见原语可重用，它们是迁移组合的原子算子，组合签名必须未见。
- 面、manifest、attempt、progress和result均write-once。source audit或语义审计失败则不查询；查询开始后不重抽、不修补、不重试。
- 不生成牺牲子集，不按早期模型成绩选择face；A/B两族一次冻结并一起执行。

## 产物

- `scripts/training/build_taiji_n1_plan166_compositional_transfer_faces.py`
- `scripts/training/run_taiji_n1_plan166_compositional_transfer_confirmation.py`
- `scripts/training/score_taiji_n1_plan166_transfer_family_a.py`
- `scripts/training/score_taiji_n1_plan166_transfer_family_b.py`
- `plans/manifests/n1_plan166_transfer_family_a_seed166101_20261020.json`
- `plans/manifests/n1_plan166_transfer_family_b_seed166202_20261020.json`
- `reports/taiji_n1_plan166_compositional_transfer_readiness_20261020.json`
- `reports/taiji_n1_plan166_compositional_transfer_result_20261020.json`
