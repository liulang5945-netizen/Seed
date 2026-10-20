# N1-165A 执行结案：完成被中断的中间剂量续训

**日期：2026-10-21**  
**状态：执行完整；原训练充分性门2/3通过，未晋级。**  
**方案：** [PLAN-N1-165A](PLAN-N1-165A_partial_attempt_completion_20261020.md) · **原批次：** [PLAN-N1-165](PLAN-N1-165_transition_template_mid_dose_calibration_20261020.md)

## 结论

N1-165A按冻结恢复方案完成了剩余训练和复核，不能把原中断attempt单独报告为完成。seed11的checkpoint与旧评分保留；seed12从已验证zero bundle重放，前7,564条完整进度逐条一致；seed13完成固定更新。末尾2,410字节截断JSON记录对应的额外一步没有进入任何final bundle。

三个终点各自18,432次有效更新，共55,296次；最终full bundle独立恢复3/3，来源哈希稳定。训练充分性门为seed11、seed12通过，seed13未通过，因此N1-165总体不通过该门，`can_promote=false`。本批没有生成新面，没有dev、holdout或confirmation查询。

## 数量与读数

| 项目 | 结案读数 |
|---|---:|
| 最终有效更新 | 55,296 |
| 中断与恢复期间物理更新操作 | 62,861（含7,565次未进入final bundle的seed12操作） |
| 唯一train-only评分项目 | 3,744 |
| 跨两次attempt的实际评分调用 | 4,992 |
| seed11旧评分复算 | 1,248/1,248逐行一致 |
| seed12中断前缀重放 | 7,564/7,564逐行一致 |
| zero bundle预检恢复 | 3/3 |
| final bundle独立恢复 | 3/3 |
| 训练充分性门 | 2/3通过；seed13未通过 |
| 恢复批次CPU | 5,483.6875秒 / 7,000秒 |
| N1-162/163/164/165A关联CPU上界 | 47,737.484375秒 / 50,000秒 |

三seed原门自由生成读数为seed11 191/192、seed12 188/192、seed13 178/192；这些是N1训练集门读数，不是日常对话能力评价。三枚final checkpoint SHA-256分别见[完整N1-165结果](../../reports/taiji_n1_plan165_transition_template_mid_dose_result_20261020.json)。恢复报告绑定的官方result正文SHA-256为`02fb635418739bbd2a67796bfb837a3c5c97cf35f70ca71b2ca9cd90151c16c9`；报告文件因Windows换行使用CRLF，正文哈希不含行尾。

## 留档与范围

- 恢复结果：[N1-165A result](../../reports/taiji_n1_plan165a_partial_attempt_completion_result_20261020.json)
- 恢复progress：[N1-165A progress](../../reports/taiji_n1_plan165a_partial_attempt_completion_progress_20261020.jsonl)
- 原中断attempt、progress、zero preflight与seed11 final均保留，不覆盖。
- 未执行的[N1-166组合迁移确认](PLAN-N1-166_compositional_transfer_confirmation_20261020.md)保持未消费，按用户最新优先级延后。

后续主线转向真实多轮中文对话材料和自由生成评价，见[PLAN-R2-DAILY-DIALOGUE-01](PLAN-R2-DAILY-DIALOGUE-CALIBRATION_20261021.md)。这只切换研究重点，不把N1-165未通过的训练门改写成通过，也不晋级任何模型。
