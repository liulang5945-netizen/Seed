# PLAN-R2-DIALOGUE-07：同规模热启动与随机初始化配对续训

**状态：已按用户授予的执行计划批准权准入；与 R2-DIALOGUE-06 并发。**  
**日期：2026-10-21**  
**前置：** [R2-DIALOGUE-06 全轮自然对话预训练](PLAN-R2-DAILY-DIALOGUE-FULL-TURN-PRETRAINING_20261021.md)  

## 目标

直接检验“答复式训练得到的 R2 权重，能否迁移到自然多轮对话训练”。R2-DIALOGUE-06 使用 1M capacity fresh 初始化；本计划在旧 R2 checkpoint 的原生结构内配对比较热启动和同结构随机初始化，避免把初始化差异与模型规模差异混为一谈。

旧 N1 题面机制 checkpoint 属于不同任务模型，不作为本批语言生成模型的权重父档。N1-166 因 N1-165 原训练充分性门仅 2/3 通过仍不准入，本计划不更改其历史判据或状态。

## 冻结配对设计

两臂使用完全相同的 TaijiConfig 结构、全轮训练数据、训练顺序、优化器/训练器默认值、目标函数和更新量：

1. **warm-start**：从 R2-DIALOGUE-03R1 的 `dialogue_context_final.pt` 恢复完整 Taiji 权重；载入后重置本轮 LanguageAlignmentTrainer 的 corpus 游标与 history，在新语料上开始同一固定配方。
2. **fresh-same-capacity**：从同一 checkpoint 的序列化 TaijiConfig 构造相同形状的新模型，仅把初始化 seed 固定为 `20261023`；参数随机初始化，并启用与 warm-start 相同的 factorized response-plan readout。

每臂先独立保存并在新进程恢复 zero full bundle，再用副本完成单 episode 信用探针且证明父 zero bundle 不变；全部前置通过后才开始正式训练。两臂各在全部 3,762 条安全 OASST2 train-family 消息（508 families，user 与 assistant 两类目标）上训练 8 个完整 epoch，再在同一 525 条 train assistant episode 上校准 1 个 epoch。每臂计划 14,517,912 次全轮目标字节更新与 113,209 次 assistant 校准字节更新。

每臂 checkpoint、readiness、attempt、progress、result 使用独立 write-once 路径；zero、全轮 final、校准 zero、校准 final 均做 fresh-process 完整恢复。两臂共享只读的冻结语料与源文件，不共享或覆盖输出。

## 并发与资源

本计划两臂与仍运行的 R2-DIALOGUE-06 并发。启动前及执行中监测全机资源；若可用内存低于 8 GiB 或全机 CPU 连续采样超过 85%，停止启动新的工作进程，已运行进程按各自 checkpoint/attempt 合同处理。每个新 arm CPU≤120,000 秒，两臂合计≤240,000 秒；R2-DIALOGUE-06 原预算仍单独≤240,000 秒，总关联上限≤480,000 process-CPU 秒。该预算计入预检、保存/恢复和训练。

## 评价与边界

- 两臂训练中不查询 dev、holdout、confirmation，不加载 reference response，不做逐字答案评分，不自动晋升。
- 等 R2-DIALOGUE-06 与两臂均完成训练并产生 final restore 结果后，再对两臂进行一次自由生成。复用 R2-DIALOGUE-06 已保留的 39 个未查询 OASST2 dev-family prompt，作为同一输入下的第三、第四候选端点；只传 prompt/history，不读 reference。最终组合报告并列已有 R2-DIALOGUE-06 两端点的已保存生成行，不重跑它们。该复用只用于受限的同面配对比较，不视为独立确认或新泛化证据。
- 人工并排复核使用相同预选 12 families，评估中文可读性、是否贴合当前用户输入、历史连贯与重复；自由生成表面指标仅作描述。`can_promote=false`。
- 若 checkpoint 恢复、来源哈希、数据分母、更新量或资源守卫失败，保留 partial 证据并停止对应 arm，不修改原始 R2-DIALOGUE-06 文件，不静默重试。

## 预期产物

- 训练器：`scripts/training/run_taiji_r2_daily_dialogue_warmstart_matched.py`
- 对照评测器：`scripts/training/evaluate_taiji_r2_daily_dialogue_warmstart_matched.py`
- 契约测试：`tests/taiji_native/test_r2_daily_dialogue_warmstart_matched_contract.py`
- checkpoints：`data/checkpoints/r2_daily_dialogue_warmstart_matched_20261021/`
- readiness、attempt、progress、generation 与 result：`reports/taiji_r2_daily_dialogue_warmstart_matched_20261021_*`
