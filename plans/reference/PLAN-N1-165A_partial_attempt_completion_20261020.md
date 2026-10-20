# PLAN-N1-165A：完成N1-165被中断的中间剂量批次

**状态：已按用户最新指示批准并冻结；完成本批后再转向日常对话训练。**  
**日期：2026-10-20**  
**原方案：** [N1-165](PLAN-N1-165_transition_template_mid_dose_calibration_20261020.md)

## 中断现场

用户先要求停止N1-165并转向日常对话训练；随后明确要求先完成手头批次再调整。原runner因此在外部中断，原attempt仍为`started`，planned result文件不存在，不把它报告为完成。

中断时：seed11 candidate已完成18,432更新、final bundle和1,248条train-only评分；seed12有7,564条完整training progress，但没有partial checkpoint；seed13未开始。进度文件末尾另有一条2,410字节的截断JSONL记录（SHA-256 `0d94ef091aeb5ddd3b037c3a5be14576283868c13d90eb5b220c846ab527a110`），对应额外一次已执行但未保存bundle、也未形成完整进度行的更新；恢复时只核对7,564条完整前缀，不把截断记录用于重放一致性判断。三枚zero checkpoint及独立恢复已通过3/3。原progress保留，SHA-256 `11c59e4348baec67f7c77ff79f8a3574789cb68bc945f8ed266e482652071abf`；原attempt SHA-256 `06742a1e8cd72d224cf67e3c6023064c9253592ebf1149f89dc8d69bd969b5d1`。

## 完成方式

- 保留seed11 final checkpoint、progress和评分；fresh-process独立恢复该checkpoint，并把原1,248条评分重算一次，用逐行字节/字段相等核验重建完整结果摘要。
- seed12从已验证zero bundle重放完整18,432步。对比前7,564步与中断attempt中完整持久化的逐步schedule身份和loss读数；必须精确相同，否则停止，不继续seed13。截断尾记录因缺少完整JSON字段而仅作为中断证据保留。
- seed13从已验证zero bundle完成冻结的18,432步。
- seed12/13均保存final full bundle并fresh-process独立恢复，完成各自960项N1-132原门与288项长度评分。
- 最终三个endpoint仍各自18,432有效更新、共55,296有效终点更新、3,744条唯一train-only评分。中断attempt的7,565步（7,564条完整进度加1条截断尾记录）未进入final model，均记为废弃计算。全系列实际参数更新操作为62,861次（中断attempt 25,997 + completion新增36,864）；completion为恢复seed11 summary额外重算1,248条train题，所以实际评分调用总数4,992（初次seed11评分1,248 + completion调用3,744），唯一评分集合仍3,744条。
- 完成后写出N1-165原计划result路径及N165A recovery result；原attempt、progress、zero preflight和seed11 checkpoint均不覆盖。

## 预算与停止

原N1-165执行最后回执为阶段CPU3,700.1秒/seed12 step7,168；考虑其后未记录的396步，按4,000秒保守计入关联预算。N165A completion CPU≤7,000秒；关联N1-162/163/164/N1-165及本completion CPU≤50,000秒。N165基线38,253.796875秒，加中断保守预留4,000秒和本批上限7,000秒，最坏总计49,253.796875秒。

任何parent/source/checkpoint SHA、seed12 replay前缀、门评分重算、最终恢复或CPU守卫不一致均停止并保留新失败回执；不生成新face，0 dev/holdout/confirmation查询、不晋级，`can_promote=false`。

## 产物

- recovery runner：`scripts/training/run_taiji_n1_plan165a_partial_attempt_completion.py`
- completion attempt/progress/result：`reports/taiji_n1_plan165a_partial_attempt_completion_{attempt,progress,result}_20261020.json`（progress为JSONL）
- 完成后N1主线再按用户指定优先转为中文日常对话材料和自由生成能力。
