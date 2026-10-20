# R2-DIALOGUE-01 执行结案：OASST2多轮续训与自由生成检查

**日期：2026-10-21**  
**状态：配对续训与生成检查完整；对话语义质量未通过人工样例审阅；`can_promote=false`。**  
**冻结方案：** [PLAN-R2-DAILY-DIALOGUE-01](PLAN-R2-DAILY-DIALOGUE-CALIBRATION_20261021.md) · **执行修复：** [01B retry2](PLAN-R2-DAILY-DIALOGUE-RETRY2_20261021.md)

## 结论

本批把OASST2中文多轮对话树接入态极原生H3.7B byte-aligned语言模型，candidate保留最多3轮历史，对照使用相同用户/助手材料但只看当前句。两个臂各训练2个完整epoch，final bundle独立恢复均通过。模型生成UTF-8有效率均为100%，candidate多轮配置的end-marker率由父档8.3%升至75%，单轮对照为41.7%。

然而，生成样例仍主要是重复汉字和碎片拼接，多数无法读成贴题、连贯的日常回复。end-marker学会停止不等于学会自然交流；**本批未实现用户目标中的日常对话能力提升**。candidate与control在评价时也分别使用多轮与单轮输入，因此差异是续训与上下文配置的整体包比较，不能单独归因于训练材料或历史消费机制。保留R2结果作为负面/有限证据，不晋级、不改产品默认。

## 执行账本

| 项目 | 结果 |
|---|---:|
| OASST2中文整树划分 | train 525 / dev 62 / final 123 episode |
| 续训 | candidate和单轮control各2 epoch、各1,050 episode passes |
| plan-local response observations | candidate 226,418；control 226,418 |
| zero full-bundle fresh-process恢复 | 2/2通过 |
| final full-bundle fresh-process恢复 | 2/2通过 |
| final自由生成样本 | 12棵树，6有历史、6无历史；48次生成 |
| reference-answer/exact-match评分 | 0 |
| dev生成查询 | 0 |
| retry2 CPU | 1,787.15625 / 120,000秒 |
| 晋级 | `can_promote=false` |

代表生成读数：父档含/不含历史的边界率均1/12；candidate多轮为9/12，单轮control为5/12。candidate与control各有9种不同输出/12，重复输出率均25%。candidate平均输出244.5字节、重复4-gram率20.5%；单轮control平均339.8字节、重复4-gram率23.9%。这些是12个树级样本上的格式/重复描述指标，不代表语义对话质量。完整逐条输出见[不含参考回答的生成报告](../../reports/taiji_r2_daily_dialogue_20261021_retry2_generations.json)和[中文并列样例](../../reports/taiji_r2_daily_dialogue_20261021_retry2_generation_review.md)。

## 恢复和失败记录

- 首次R2 attempt与retry1都在正式训练前因restore worker摘要字段错误停止，正式参数更新与生成查询均为0；其attempt/result和zero checkpoint均保留，见[01A retry addendum](PLAN-R2-DAILY-DIALOGUE-RETRY1_20261021.md)。
- retry1 zero bundle随后通过只读fresh-process恢复复核，确认checkpoint本身有效；retry2修复实际CLI worker并以新命名空间完整执行，见[01B retry addendum](PLAN-R2-DAILY-DIALOGUE-RETRY2_20261021.md)。
- 执行前有两次临时checkpoint预检未返回结果，未留下正式模型产物且CPU未计量；它们已在retry2 attempt中标为unmetered，不能计入本批CPU结案值。

## 后续

按冻结方案的失败分流，下一步是只读归因本批生成输出，区分语言表示/解码停止/历史上下文消费三个症状；0模型加载、0训练更新、0生成查询，不读参考答案，也不自动追加本轮训练。归因后再冻结唯一的后续数据或模型修订方案。
