# SPEC-N1-33 · 已批准32固定九模型续接合同（2026-10-15）

用户已在选择弹窗批准[32](PLAN-N1-32_fixed_paired_continuation_review_20261015.md)。31封存为训练前置失败，源报告SHA `b10fe3092c9fa6abd08f64fed48f02e59c4261a20c58a49a85f9702d0c5e06dd`，0 holdout/dev。新运行器/报告/档独立，不改31。

输入必须完整九模型/18恢复、6144更新/模型、源identity匹配且未有留出；核对每seed原始初始化历史三臂相同。每个输入按独立final SHA/参数及31协议/seed/arm/更新数恢复，参数、全Adam、RNG、来源、首/第二步与地址/辅助签名须与31完全一致。续接当前三臂参数不同是正常，不能重新初始化或重置优化器。

先CPU≤60秒工程预检：全部九输入签名、三seed position档各三臂零更新信用/前向及辅助相等、参数不变；独立新zero恢复。工程CPU含worker完整计入本批正式600秒总额，不另借旧预算。5相关工程测试通过（4.77秒），含原门/9216累计门、Adam/RNG与精确下一步、两协议拒混及篡改签名拒绝。

新正式九模型各固定追加3072（16轮），累计各9216，额外27,648；train从原6144后接train-000轮转，新增session避免来源覆盖。学习率/目标/模型/数据/三臂不变，每个续接zero/final独立恢复正式18worker。全train192×9、全九模型/恢复冻结和每个position原充分性门通过才首次v3 holdout32×9；dev不查。原14/30行为门保留，若仍不足0 holdout并停止，不循环追加。

新协议updates_per_model/total_updates计**追加**3072/27,648，同时明确starting6144/cumulative9216；training记录updates计该模型累计9216，并另报additional_updates3072。报告completed_updates计本批追加，cumulative_updates计九模型历史55,296加本批；zero恢复的completed_updates=6144表示零追加，不误称fresh零训练。

CPU≤600秒含工程报告已用CPU、正式父及18worker。触边保存partial与实际累计更新，不缩分母、不重跑；源码/数据/31报告/输入九档及新final档前后SHA守卫，冻结pre_holdout后才查询。工程/预算/来源缺口不放行。相关CI/全项目保持/联合能力债保留，不默认采用、不推送、不删除数据。
