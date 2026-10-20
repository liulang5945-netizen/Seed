# PLAN-N1-162G1：线程吞吐基准前置校验修复

**状态：已依用户授予的执行计划批准权批准并冻结**<br>
**日期：2026-10-20**<br>
**父方案：** [PLAN-N1-162G](PLAN-N1-162G_training_thread_scaling_benchmark_20261020.md)<br>
**首次尝试：** [attempt](../../../reports/taiji_n1_plan162g_training_thread_scaling_attempt_20261020.json)、[空progress](../../../reports/taiji_n1_plan162g_training_thread_scaling_progress_20261020.jsonl)、[失败结果](../../../reports/taiji_n1_plan162g_training_thread_scaling_20261020.json)

## 首次尝试结论

N1-162G在source/schedule readiness后、首次checkpoint加载前停止：runner把训练模块误当作哈希校验模块，触发`AttributeError`。失败result报告CPU38.265625秒，`completed_trials=[]`，progress为空，0模型/checkpoint加载、0训练更新、0评分、0 checkpoint写入。report中的`source_hashes_unchanged=false`是因为校验函数未能被调用、source map仍为空，不表示源文件发生了变化。首次attempt、空progress及失败result均保留，不覆盖或冒充成功结果。

## 修复及执行边界

仅新增G1薄包装器：显式从N1-150 builder模块注入`verify_hashes`，将plan绑定切到本修复方案，并使用新的write-once输出路径。原N1-162G runner不修改。确认源码校验入口指向构建模块后，完整重做N1-162G冻结的同一测试：N1-162B seed11/surface_control checkpoint、同一schedule的第0–287步、线程数与交错顺序、warmup及计时更新数、10%重复收益解释规则均不变。

重试前先核对N1-162G失败记录、父方案/本修复方案SHA、checkpoint/schedule/N1-162B result SHA，并再次验证保存恢复回执及完整bundle fingerprint。每个线程repeat独立从原checkpoint恢复；所有更新仅作用于内存副本，checkpoint只读且结尾复核SHA。父尝试已用38.265625 CPU秒，因此G1本批剩余上限861.734375秒，使G+G1关联总量不超过900秒；不查询、不评分、不访问dev/holdout/confirmation，不改变正式runner、研究判据或晋升状态，`can_promote=false`。

G1的attempt/progress/result使用`plan162g1_training_thread_scaling`前缀单独保存。若修复后仍在首个更新前停止，封存G1失败，不循环修补；只有缺陷定位明确且无需改变测量设计时才继续报告既有成功结果。
