# PLAN-N1-155A 执行结案

**日期：2026-10-20**  
**状态：流程完整结束；训练门未全过，未生成 fresh face**  
**更正方案：** [PLAN-N1-155A](PLAN-N1-155A_runtime_views_handoff_correction_20261020.md)  
**上位方案：** [PLAN-N1-155](PLAN-N1-155_short_context_unknown_rebalance_and_fresh_diagnostic_20261020.md)

## 执行结果

schedule 的`views`透传更正后，六个 endpoint 共完成55,296/55,296次追加更新、7,488/7,488条 train-only 评分。六个 final full bundle 全部独立恢复，来源与 checkpoint 哈希未变。进度账本共62,784行：55,296条更新、5,760条原训练门评分、1,728条长度评分；JSONL逐行可解析，无缺行或坏行。progress SHA-256 为`83e951930272d30ab3b70a0164dba0ef628d8876a8138708be08a97d3a7acf73`。

计入首轮在零更新处停止的130.921875秒，累计 CPU 为9,325.21875/18,000秒。N1-155A结果`complete=true`表示冻结执行流程按停止门完成，不表示能力晋级；`can_promote=false`。

六个 endpoint 中，三个`unknown8_rebalanced` candidate 的原 N1-132训练门全部通过；三个`uniform_length_replay` control 中，seed11与seed13通过，seed12未通过。唯一失败为seed12 control 的 transition 30/48，令该 endpoint 的原始命中降为166/192、重命名命中167/192。由于上位方案要求六个 endpoint 全过才生成 seed128 fresh face，本轮按协议跳过该阶段：0 fresh 查询、0 dev、0 holdout、0 confirmation。

训练集汇总仅作描述性记录：candidate 原门命中合计574/576，control为544/576；unknown 原门为142/144与130/144。长度评分中的unknown正确数，candidate在8/16/32行分别67/72、69/72、71/72，control分别66/72、69/72、71/72。这些同训练集读数不能替代fresh泛化评估，也不能单独证明再平衡的因果作用。

完整逐 endpoint、checkpoint、评分及 CPU 明细见[结果报告](../../reports/taiji_n1_plan155a_unknown8_rebalance_result_20261020.json)。原始的`KeyError: 'views'`失败回执及0更新局部 checkpoint 保留；更正轮使用独立路径，没有覆盖或复用它们。

## 后续唯一队首

依据用户授予的计划批准权，已批准只读的[PLAN-N1-156](PLAN-N1-156_plan155a_train_score_itemwise_attribution_20261020.md)：对本轮已消费的7,488条 train-only 评分做逐题配对归因，定位seed12 control transition失分，并比较candidate/control的unknown长度变化；CPU≤60秒，不加载模型或 checkpoint、不做推理/新查询/训练、不访问dev/holdout/confirmation，不改变判据或晋级状态。
