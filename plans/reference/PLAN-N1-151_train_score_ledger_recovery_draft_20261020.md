# PLAN-N1-151：N1-150 train-only 逐题评分台账补齐

**状态：已依据用户授予的执行计划批准权批准并冻结。**N1-150 的55,296次更新、7,488条计分、六个训练门及final bundle恢复均已完成。只读进度审计发现：progress JSONL保留了全部训练更新，但评分明细只保留最后一个 endpoint 的1,248行；前五个endpoint的6,240行在后续训练写入时被覆盖。N1-150结果JSON中的六端点聚合门和长度分层汇总完整。本计划只恢复同一组train-only逐题评分明细，不重训、不新增样本，也不把重复计分说成独立泛化验证。

## 固定对象与范围

- 只使用N1-150的六个final full bundle：seed11/12/13 × `length_mixed` / `eight_row_control`。每枚先做完整独立恢复检查，核对模型、Adam、RNG、history和fingerprint，再载入评分；不写入或更新任何checkpoint。
- 完全复用N1-150冻结的v3 train来源、192条父任务、schedule/context bank、N1-132 gate scorer及N1-150A评分选择：每endpoint原门960条，长度分层8/16/32行各96条，共288条。
- 共重复7,488条相同train-only计分（6×1,248）。逐题行直接写入一个write-once JSON结果文件；只用一个结果写入路径，不让训练/追加写入句柄并行作用于同一文件。
- 计分前后模型/Adam/RNG/history fingerprint须逐位相同；分母、task ID、类别、原门整数摘要和长度分类汇总须与N1-150 result逐项一致。任何来源、恢复、分母、摘要、SHA或CPU守卫失败即停止，不重试。

## 资源与边界

- 本计划CPU上限3,600秒，覆盖N1-150来源/计划重核、六个final bundle独立恢复、7,488条计分及末端SHA复核。
- 0训练、0 optimizer/参数更新、0新checkpoint；0 dev、holdout、confirmation查询，不改变任何gate或晋级状态。结果固定`can_promote=false`。
- 这7,488条是对同一训练面、同一checkpoint、同一题目与条件的重复计分，只为恢复逐题台账并核对N1-150聚合，不构成第二份确认结果，不作能力或泛化主张。

## 实施与产物

1. 实现独立的一次性恢复评分runner和合同测试，先核对N1-150结果、六枚final checkpoint及分母合同；提交实现后做无查询readiness。
2. readiness通过后，对六枚final bundle做独立完整恢复；6/6相等后才执行固定7,488条评分。
3. 每endpoint必须保留960条原门行及288条长度行；最终result内嵌全部7,488条逐题评分、六端点聚合和逐项对账结果。

- Runner：`scripts/training/run_taiji_n1_plan151_train_score_ledger_recovery.py`
- 合同测试：`tests/taiji_native/test_n1_plan151_train_score_ledger_recovery_contract.py`
- Readiness：`reports/taiji_n1_plan151_train_score_ledger_recovery_readiness_20261020.json`
- 恢复前置（`--out-report`；2026-10-23 标注：计划输出，未产出）：`reports/taiji_n1_plan151_train_score_ledger_recovery_preflight_20261020.json`
- Attempt（`--out-report`；2026-10-23 标注：计划输出，未产出）：`reports/taiji_n1_plan151_train_score_ledger_recovery_attempt_20261020.json`
- 完整逐题结果（`--out-report`；2026-10-23 标注：计划输出，未产出）：`reports/taiji_n1_plan151_train_score_ledger_recovery_result_20261020.json`
