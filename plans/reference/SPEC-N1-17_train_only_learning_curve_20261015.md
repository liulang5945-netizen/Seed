# SPEC-N1-17 · 单候选train-only学习充分性诊断（2026-10-15）

用户已授权此类低成本实验自主开展。16证明9模型尚未拟合已见样本，故本件先查剂量，不立即增容/更改优化器或重做原holdout。17是独立诊断协议，不延长或改写15的384步配方及负裁决。

固定从15 seed11 position、已完成384更新的trained checkpoint复制；原档SHA与参数不得改变，恢复完整Adam/RNG/会话来源。相同96train、相同顺序、Adam lr0.001、embedding16/hidden32/key16、相同逐位置读取/UTF8/cap64，固定追加3072次episode更新；无learning-rate sweep、无最佳seed选择。追加步0/384/768/1536/3072各对全部96train做一次TF与自由输出，480组，不查询dev或holdout。

独立累计CPU上限300秒（含两个独立恢复worker），Torch/BLAS单线程。任何追加更新前先保存17专用zero bundle并新进程weights-only核对live-parent参数/optimizer/来源/RNG/首步logits；固定终点再保存并独立恢复。新schema/protocol与15报告分离，独占新目录，原档不覆盖。CPU触边保留partial，不能减分母、假装完成或自动重复。

报告所有milestone的逐项读数及训练loss，不按训练拟合节点重新开启原holdout。它只能判读“同架构在多一点已批准资源下是否学得动已见任务”，不能声称逐位置胜过对照、未见能力或C4通过。代码/数据/原档hash及原档不变守卫记录，训练前后checkpoint/恢复检查必做。

若已见仍差，下一步查内容/目标的可识别性与误差竞争；若已见明显学得动，下一步在新的独立配对协议里核查训练充分性及未见泛化，而不沿用15旧holdout当新盲测。根据实际结果只选其中一个方向，不并开实验支线。
