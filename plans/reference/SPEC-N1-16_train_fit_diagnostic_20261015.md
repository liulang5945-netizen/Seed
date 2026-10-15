# SPEC-N1-16 · 15负结果后的train-only拟合诊断（2026-10-15）

在用户低成本自主授权内一次执行，独立CPU≤60秒、Torch单线程、零参数/optimizer更新，不能借此重评或拟合holdout。继承15完整负结果，不更改它的判据。

只读取15的9个trained checkpoint及原v2 manifest中train96项，每模型每项一次teacher-forced与一次自由生成，共864组。先核对正式报告SHA、manifest SHA、每档SHA及weights-only恢复；全部候选参数、来源自足，过程不加载母档或外部模型。每项新的audit会话只观察该train项真实材料，问题只读；gold仅TF损失和计分器可见。自由生成继续使用该模型实际position/每来源末态/cut路径，UTF8/cap64/boundary256不变。

保存逐项TF条件预测符号与目标长度、正确数、含boundary整串正确、CE、自由原始bytes/自然停止/严格exact；按seed/arm/kind汇总，明确teacher-forced条件正确和自由运行行为不是同一指标。每模型前后参数摘要与checkpoint文件SHA不变；没有新增dev/holdout模型查询。预算触边保存部分并判不完整，不减少96分母或更换检查点重跑。

报告名 `taiji_n1_native_evidence_train_fit_20261015.json`，独占写入reports。结果用于决定是先解决已见样本拟合、自由轨迹偏移还是未见内容泛化；不凭CE下降或梯度非零升级能力，不立即加参数/盲调学习率。下一步必须由实际分类确定并写回当前计划。
