# PLAN-N1-151A：N1-151 preflight来源绑定返回值修正

**状态：已依据用户授予的执行计划批准权批准并冻结。**N1-151已完成无查询readiness；final checkpoint preflight因实现错误停止在任何checkpoint加载之前。失败回执：[N1-151 preflight失败](../../reports/taiji_n1_plan151_preflight_binding_failure_20261020.json)。

## 根因与唯一变更

N1-150 runner 的 `_verify_preflight()` 返回顺序为 readiness、preflight、preflight-binding、train context、schedule。N1-151 preflight verifier 把第三项 binding 当作 N1-150 result，重新收集来源时因此漏掉六枚 N1-150 final checkpoint，和已冻结 readiness source set 不一致。

本修订只把 result 读取改为从其 write-once 报告路径读取，明确检查六枚final checkpoint摘要，再生成一套新的N1-151A readiness/preflight产物。保留原N1-151 readiness、原失败回执和N1-150所有文件；不覆盖、不重跑N1-150，也不重写已存在的N1-151 readiness。

## 冻结范围

- 仍重用N1-150同六枚final bundle、同192条v3 train父任务、同一schedule及source exclusions；逐题计分仍为每endpoint原门960条+长度分层288条，六endpoint共7,488条。
- 仍需6/6 final full bundle独立恢复通过后才评分；每endpoint的原门整数摘要和长度分类汇总必须与N1-150结果精确一致，模型/Adam/RNG/history评分前后指纹保持不变。
- CPU≤3,600秒；不训练、不更新、不写checkpoint、不生成新题、不查询dev/holdout/confirmation、不改变门或晋级状态，`can_promote=false`。N1-151A仍是train-only评分台账恢复，不作独立泛化结论。
- N1-151A代码和合同测试提交后，先用新的write-once路径做无查询readiness，再做final restore preflight，最后才执行一次固定7,488条评分；任一守卫失败即停，不重试。

## 修正批产物路径

- Readiness：`reports/taiji_n1_plan151a_train_score_ledger_recovery_readiness_20261020.json`
- Readiness attempt：`reports/taiji_n1_plan151a_train_score_ledger_recovery_readiness_attempt_20261020.json`
- Source binding：`reports/taiji_n1_plan151a_train_score_ledger_recovery_source_binding_20261020.json`
- Restore preflight：`reports/taiji_n1_plan151a_train_score_ledger_recovery_preflight_20261020.json`
- Preflight attempt/binding：同前缀下的`*_preflight_attempt_20261020.json`及`*_preflight_binding_20261020.json`
- Scoring attempt：`reports/taiji_n1_plan151a_train_score_ledger_recovery_attempt_20261020.json`
- 完整逐题结果：`reports/taiji_n1_plan151a_train_score_ledger_recovery_result_20261020.json`
