# PLAN-N1-162F1 执行结案：N1-162F转录完整性复核

**状态：已完成，唯一保留项为历史 runner 来源漂移**<br>
**日期：2026-10-20**<br>
**冻结计划：** [PLAN-N1-162F1](PLAN-N1-162F1_transcript_integrity_revalidation_20261020.md)<br>
**复核报告：** [完整 JSON 报告](../../../reports/taiji_n1_plan162f_transcript_integrity_20261020.json)

## 结论

F1 对已经封存的 N1-162F attempt、progress 和 result 完成逐条离线复核。6,912个预期查询键全部唯一且齐全；question/answer 与训练输入重建一致；6,912条 `free_exact` 与输出类别均独立复算一致；progress 与 result 的6,912行逐字段匹配，3,456组双模板配对完整，144个endpoint摘要复算通过。九枚checkpoint的已记录哈希、独立恢复回执和fingerprint一致。

738项历史来源记录中，除F runner外737项当前哈希均相同，没有缺失源或其他漂移。唯一例外是F result记录的执行时runner SHA `10e1772f2c6bafc3ef3feb481a923dffa327f14b6fdb64d81c7074dfc3681999` 与当前文件SHA `a6faaa1405f3239a671534b9666cb52201d635edc613206c44ba326e78d4594c` 不同。F1确认转录内部一致及结果来源边界，但不能恢复旧runner源码，也不等于以旧runner重跑F。

本次CPU 12.375/120秒；0模型/checkpoint加载、0新推理/查询、0参数更新、0新测面；`can_promote=false`。N1-162F原输出保持不变。F1报告SHA-256：`e90341d6d8e93de7f8027322c10dc1b61b6ea2caa81531ae0b3f6915bcdebbc7`。

## 后续

N1-162B训练器固定PyTorch、OpenMP、MKL与OpenBLAS为单线程。是否提高线程数能缩短同一训练配方的墙钟时间尚未测量；按[PLAN-N1-162G](PLAN-N1-162G_training_thread_scaling_benchmark_20261020.md)先做同检查点、同训练槽位的线程吞吐对照，不改正式训练器、研究判据或能力状态。
