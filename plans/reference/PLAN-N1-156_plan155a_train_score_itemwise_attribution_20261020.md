# PLAN-N1-156：N1-155A train-only 逐题配对归因

**状态：已依据用户授予的执行计划批准权批准，待实现与执行**  
**日期：2026-10-20**  
**来源执行：** [PLAN-N1-155A 结案](PLAN-N1-155A_execution_closeout_20261020.md)

## 背景与问题

N1-155A完成55,296次续训和7,488条train-only评分。三个`unknown8_rebalanced` candidate端点的原训练门均通过；seed12的`uniform_length_replay` control仅166/192，transition为30/48，导致六端点合取门失败，按上位方案未生成或查询seed128 fresh face。当前结果是配对训练集读数，不能据此判断未见泛化或宣称辅助机制因果效应。

本方案只对已经消费并保存于N1-155A progress账本的评分行做离线配对归因，重点定位seed12 control transition错误，并描述candidate与control的unknown答案在8/16/32行中的逐题变化。

## 冻结输入与范围

- N1-155A结果：`reports/taiji_n1_plan155a_unknown8_rebalance_result_20261020.json`，SHA-256 `e577d4dda753d016a46e11ab2e0202b2845206af107c54f87aff7f9fdf1ebb11`。
- 唯一逐题输入：`reports/taiji_n1_plan155a_unknown8_rebalance_progress_20261020.jsonl`，SHA-256 `83e951930272d30ab3b70a0164dba0ef628d8876a8138708be08a97d3a7acf73`。
- 预期事件分母固定为55,296条更新、5,760条原训练门评分、1,728条长度评分；归因只消费后两类共7,488行，训练更新行仅用于核对完整性。
- 原训练门按seed、`evaluation_set`、kind及题目/条件标识配对 candidate 与 control；长度评分按seed、context length、kind及题目ID配对。报告candidate-only正确、control-only正确、两边都正确、两边都错误，并列出方向性错转对与长度变化的逐题汇总。
- 在归因前验证上位结果完整、来源哈希稳定、账本SHA/行数/JSONL格式/事件分母准确；任一条件不符即停止并只写失败回执，不生成部分结论。

## 资源与边界

- CPU上限60秒。
- 不加载模型或 checkpoint；不推理、不查询、不训练、不更新参数。
- 不访问dev、holdout或confirmation；不生成新题、不接触其他测面。
- 不改变任何训练门、确认门、计划范围或晋级状态；结果仅是同训练集上的描述性逐题归因，`can_promote=false`。

## 输出

- 分析器：`scripts/training/analyze_taiji_n1_plan156_plan155a_train_score_itemwise_attribution.py`
- 契约测试：`tests/taiji_native/test_n1_plan156_plan155a_train_score_itemwise_attribution_contract.py`
- 报告：`reports/taiji_n1_plan156_plan155a_train_score_itemwise_attribution_20261020.json`

分析器、契约测试与本计划提交后，才运行唯一一次只读归因。报告通过完整性核对后，将在独立结案页记录逐题结果与唯一后续建议。
