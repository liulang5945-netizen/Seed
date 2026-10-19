# PLAN-N1-149：匹配弃答的上下文长度曲线诊断

**状态：依据用户授予的执行计划批准权已批准并冻结。**这是新的一次性diagnostic面，不作为确认门或能力晋升依据。

## 依据与目标

N1-148在32行配对面上显示known材料问题常被弃答，而unknown问题大多正确弃答。N1-149用40个固定目标/答案/问题场景，分别构造8、16、32行嵌套上下文，测量随着非目标干扰事实增加，known answer与unknown abstention如何变化。

## 冻结题面

- 固定seed126，共40个logical pair template；每个template构造8/16/32行三个长度，每个长度各一题known和一题unknown，共240题。
- 每题均为diagnostic split、d=1、实体长度10；known/unknown在同一长度内只改变位置0的一条事实：known放目标材料事实，unknown以近邻实体材料事实替代。所有长度条件的target entity、question和known答案保持相同；干扰事实按同一顺序从短到长追加。
- target行固定索引0，其他条件只增加尾部干扰行。40个pair的答案6/7字符各20个；同一pair在三个长度复用目标/答案值，pair之间及所有已消费源间无实体/材料交集。
- source exclusion继承N1-148的38,122实体名/15,288词值，并加入整张已消费v25 face新增1,320名/1,320值；预期并集39,442实体名/16,608词值。生成前核对v25面及结果SHA、sources与零交集。

## 固定读数与查询

- 同一12个N1-132/140/141冻结endpoint；先独立恢复完整checkpoint/Adam/RNG/history并核对哈希，失败则0查询。
- 每endpoint对240题各查询一次，总计2,880条。使用diagnostic split专用只读scorer，禁止重标holdout；source/out/checkpoint指纹和optimizer/history每endpoint前后守卫。
- 按endpoint/seed/长度/condition报告known精确回答、known字面弃答、unknown字面弃答、unknown误答；按logical pair列出8→16→32三长度结果。0训练/参数更新、0 dev/旧v3 holdout，不设置通过门或自动晋升。
- 每条query先落progress；一次失败或中断即封存，不重试、不补抽。

## 资源与产物

- CPU≤7,200秒，覆盖来源并集、题面合同、12端点readiness、2,880条查询与末端哈希核验。
- scripts/tests须先实现并提交，再生成面；readiness通过并提交后，才启动唯一一次查询。
- builder：`scripts/training/build_taiji_n1_v26_abstention_length_after_n148.py`
- diagnostic scorer：`scripts/training/eval_taiji_n1_v26_abstention_length_after_n148.py`
- 合同测试：`tests/taiji_native/test_n1_plan149_v26_abstention_length_contract.py`
- write-once face：`plans/manifests/n1_v26_abstention_length_seed126_20261019.json`
- write-once result：`reports/taiji_n1_v26_abstention_length_diagnostic_20261019.json`
