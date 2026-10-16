# 79 全fresh训练失败离线诊断（2026-10-16）

77 seed13 position已固定24576步、两恢复完整，终点167/95/165/94；lookup/revision/unknown均48/48，transition原23/48、改名21/48。原门未过，78不准入，v5未消费。77仍按冻结配方完成两个剩余控制，不按失败省掉它们，不补训/选中点/改门。

唯一后继为完整77封存后的纯日志诊断。独立CPU≤30秒；0模型加载/推理/优化器/参数/副本更新，0dev/v3holdout/v4/v5查询。只读77主报告和九无损分片、原v3训练manifest及已冻结协议。先绑定77完整终端报告SHA，核验九模型/18恢复/72副本步/221184更新及双层分片SHA/计数/来源/代码/预算守卫；失败门照实记录，不把训练充分性作为读取失败证据的前置。不得复用78只接受全部position充分性的入口来绕过其拒入。

固定分析：九模型五milestones的原四门、每类teacher/free逐行增/损；逐模型每6144步的完整loss与五组实际Adam位移、cap、clipped比例、scale、gradient统计，另末1024步均值；transition按原训练观察中的四种初始方向×left/right八格对齐，原题/改名逐格输出混淆及teacher/free一致，区分持续未学与后程遗忘。lookup/revision/unknown和两pair保留同分母作对照，不筛选困难样本或只报最好种子。统计仅定位相关性，不把梯度/位移相关当干预因果。

产物：scripts/training/diagnose_taiji_n1_fresh_training_logs.py、tests/taiji_native/test_native_fresh_training_log_diagnostic_contract.py、reports/taiji_n1_fresh_training_log_diagnostic_20261016.json及plans/reference裁决。独占写入、固定LF、完整来源SHA和CPU/零查询计数；先合成合同/静态检查/提交，再一次执行。无新训练或结构改动混入79。完成后据单一主要失效提出一个高上限机制/训练方案的新预注册，不追加77，不消费仍未见v5。

全项目保持/C4/同bundle三器官、N2–N6和绑定HEAD全量CI仍欠；失败本身不缩减目标，也不默认采用候选。
