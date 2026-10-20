# PLAN-R2-DIALOGUE-01B：修复有效restore worker并重跑冻结批次

**状态：已冻结；沿用PLAN-R2-DIALOGUE-01原数据、模型、训练和生成合同。**  
**基准方案：** [PLAN-R2-DAILY-DIALOGUE-01](PLAN-R2-DAILY-DIALOGUE-CALIBRATION_20261021.md)  
**retry1结案：** runner重复定义导致修复落在未调用函数，见[PLAN-R2-DIALOGUE-01A](PLAN-R2-DAILY-DIALOGUE-RETRY1_20261021.md)。

## 失败attempt与恢复证据

- 原attempt在训练前停止：formal更新0、生成查询0；失败为恢复worker在`from_checkpoint`之后访问不存在的`Taiji._observations`。CPU5.1875秒。attempt/result/readiness及zero checkpoint原样保留。
- retry1同样在训练前停止：formal更新0、生成查询0，CPU5.15625秒。修复代码误改了未被CLI调用的重复函数定义，实际worker仍读取`Taiji._observations`。
- 随后以修正的CLI worker对retry1已保存的zero bundle作fresh-process只读复核并通过：`complete=true`、`equal=true`、checkpoint digest `23455f0690ca4c7a3093ce0e88007e4a439f63f3ac562ffe2ffe921788f367c0`、CPU0.234375秒。复核报告[在此](../../reports/taiji_r2_daily_dialogue_retry1_restore_reaudit_20261021.json)，SHA-256 `cdce71e1c721b4aa6865647e3b5713d79e7ee09d7a209360479beec718d464c2`。

retry2启动前会逐项核验原attempt/result/readiness、retry1 attempt/result/readiness、两枚已写zero bundle及只读恢复报告的SHA-256。任何差异立即停止。

## retry2边界

- 唯一代码修复：删除未调用的重复worker，实际CLI恢复摘要读取`trainer.history`。不变更模型、语料、过滤、树级split、对照、epoch、12树生成样本、指标、120,000秒CPU上限或停止门。
- 正式retry2输出全部写入新命名空间：`reports/taiji_r2_daily_dialogue_20261021_retry2_*`与`data/checkpoints/r2_daily_dialogue_oasst2_20261021_retry2/`；retry1及其checkpoint保持原样。
- 两个arm都必须先完成512字节实际配置zero bundle保存和fresh-process恢复。任一失败时更新数保持0，不进入训练。
- retry2完成后仍只做48次自由生成，不读reference answer，不评exact-match，不查询dev，不晋级，`can_promote=false`。
