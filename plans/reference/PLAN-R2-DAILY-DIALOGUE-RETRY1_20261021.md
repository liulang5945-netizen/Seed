# PLAN-R2-DIALOGUE-01A：checkpoint恢复回执修复与单次重跑

**状态：已冻结；只恢复PLAN-R2-DIALOGUE-01被runner回执字段错误挡住的执行。**  
**基准方案：** [PLAN-R2-DAILY-DIALOGUE-01](PLAN-R2-DAILY-DIALOGUE-CALIBRATION_20261021.md)

## 首次attempt结案

首次attempt在训练开始前停止，CPU5.1875秒，`formal_parameter_updates=0`、`formal_generation_queries=0`。它已保存candidate的zero bundle，但fresh-process worker在`LanguageAlignmentTrainer.from_checkpoint`完成后，为报告访问不存在的`Taiji._observations`字段而退出；因此没有形成restore通过回执，也没有启动训练。checkpoint仍保留，SHA-256为`31804e42757737414f5139f94d99b97330dca7f2ccad5cc74c3b63f6261ddac5`。

首次attempt、失败result、readiness和zero checkpoint均保留，SHA分别为：

- attempt：`8e60098cdf43da29dc3a9d585f65cb43c9f62b002fe53ed02ace9f3d2b05021e`
- failed result：`e1ad7ee49ca9068193e305d4aa48948a36e85a7e64b3c0b75351450702e6b791`
- readiness：`e514951eef02530fad8dc6d1f28a9ff16544e52dd1786fceffcf375a22dadd0f`
- 首次attempt绑定的runner SHA-256：`4439ff25c73b6050fecb37424eb50c96b8ab203ee6041c3a83648ca06d151e20`。

## 重跑边界

- 只修复restore worker的状态摘要字段，使用Trainer持有的episode history；不再读取`Taiji._observations`。这是回执代码缺陷修复，不改变模型、数据、过滤、tree-level split、训练臂、epoch、自由生成样本数、评估指标、CPU上限或停止门。
- 重跑开始前重新核验基准方案、OASST2源文件与派生corpus、H3.7B parent、首次失败产物SHA。两个臂仍从同一个H3.7B parent开始；zero/final bundle独立写入新的`retry1`路径，不覆盖首次零bundle。
- retry产物使用`reports/taiji_r2_daily_dialogue_20261021_retry1_*`，checkpoint使用`data/checkpoints/r2_daily_dialogue_oasst2_20261021_retry1/`。首次attempt及文件继续保留。
- CPU≤120,000秒；source、zero/final fresh restore、更新数、tree split或CPU门不通过即停止并封存retry失败报告。此后不在同一retry目录重试，`can_promote=false`。
