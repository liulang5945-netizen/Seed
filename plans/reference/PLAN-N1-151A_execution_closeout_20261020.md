# PLAN-N1-151A 执行结案：N1-150逐题评分台账恢复

**状态：完整结束。**N1-151A只修复N1-151 preflight来源绑定返回值错误并使用新write-once产物路径；冻结评分范围未变。

## 执行结果

- 无查询readiness通过：496项来源/结果/进度/代码/计划/checkpoint SHA绑定不变；六枚N1-150 final checkpoint、192条v3 train任务、schedule及7,488条重复计分分母核验通过。CPU7.421875秒，0模型/checkpoint加载、0查询、0更新。
- 六个final full bundle由独立进程再次恢复6/6，fingerprint逐一与N1-150 final结果一致；CPU50.265625秒，累计CPU57.6875/3,600秒，0查询、0更新。
- 重复同一训练面评分7,488条：每endpoint 960条N1-132原门行+288条长度分层行。六个endpoint的原门整数摘要、factorial摘要及长度分类汇总都与N1-150结果逐项一致；评分前后模型/Adam/RNG/history fingerprint不变。
- 总计六endpoint×1,248行完整内嵌结果JSON；source/checkpoint哈希不变。评分CPU227.375秒，含 readiness 与restore前置的累计CPU285.0625/3,600秒。
- 0参数/optimizer更新、0 dev、0 holdout、0 confirmation，`can_promote=false`。这是同一train样本/同一checkpoint的逐题台账恢复，不构成独立泛化或能力验证。

## N1-151原前置失败记录

原N1-151 readiness（7.484375秒、494项来源哈希）已保留。其checkpoint前置错误地将N1-150验证器返回的source binding当作结果报告，导致六枚final checkpoint没有进入复核集合；错误发生在checkpoint加载前，0模型/checkpoint加载、0查询、0更新。失败回执见[报告](../../reports/taiji_n1_plan151_preflight_binding_failure_20261020.json)。N1-151A以独立的write-once路径修复并保留该失败记录，没有覆盖原N1-151产物。

## 结果与下一步

- [N1-151A逐题恢复结果](../../reports/taiji_n1_plan151a_train_score_ledger_recovery_result_20261020.json)，SHA-256：`7fb5ffa74b44ffa918d42a508c5d527991445367b3ec22457bb0d21a718ab6d6`。
- [readiness](../../reports/taiji_n1_plan151a_train_score_ledger_recovery_readiness_20261020.json)、[final恢复前置](../../reports/taiji_n1_plan151a_train_score_ledger_recovery_preflight_20261020.json)及[attempt](../../reports/taiji_n1_plan151a_train_score_ledger_recovery_attempt_20261020.json)保留分阶段证据。
- 唯一下一步：依据用户授予的执行计划批准权，执行[N1-152](PLAN-N1-152_length_itemwise_attribution_20261020.md)离线配对归因；只用本结果中的1,728条长度分层行，计算candidate/control同题对照及各分支8→16→32行变化，0模型/检查点加载、0查询、0更新，CPU≤30秒。
