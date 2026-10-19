# PLAN-N1-154：fresh内容长度诊断逐题配对归因

**状态：已依据用户授予的执行计划批准权批准并冻结。**N1-153A已保存2,880条完整fresh-context逐题行；本计划只读取该封存JSON并做离线整数计数，不加载模型或checkpoint。

## 固定分析内容

- 输入必须是SHA-256=`af8853b4f4677b469164e3691e429476d02887adc5a47f9981ed5b976926d06a`的N1-153A完整result；核对六endpoint×480行、72个类别/长度分层各40行，seed/branch/length/item ID无重复且candidate/control配对完整。
- 按branch、context length、task kind汇总命中和unknown字面正确弃答。
- 对相同seed、item ID、kind和length配对`length_mixed`与`eight_row_control`，统计双方都对/都错及仅一边答对。
- 对每个branch、seed和kind，统计8→16、16→32、8→32的逐题错转对/对转错/保持数；分层与总计必须能精确回算至2,880行。
- 将结果绑定到N1-153A result、N1-150 result、N1-153/N1-153A计划及分析器/合同测试的SHA。

## 边界与产物

- CPU≤30秒；0模型加载、0checkpoint加载、0新查询、0训练/更新、0 dev/holdout/confirmation查询。
- 只描述本fresh synthetic diagnostic，不作真实世界能力或独立确认主张；不改变N1-132训练门、N1-133历史门或任何晋级状态，`can_promote=false`。
- 任一输入SHA、分层分母、paired ID或回算失败即停止，不重抽、不重试。
- Analyzer：`scripts/training/analyze_taiji_n1_plan154_fresh_context_itemwise_attribution.py`
- 合同测试：`tests/taiji_native/test_n1_plan154_fresh_context_itemwise_attribution_contract.py`
- 报告：`reports/taiji_n1_plan154_fresh_context_itemwise_attribution_20261020.json`
