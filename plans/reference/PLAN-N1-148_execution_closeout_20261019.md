# PLAN-N1-148 执行结案：diagnostic split匹配弃答测量

**状态：执行完整；无通过门、无能力晋升结论。**diagnostic专用scorer保留`split="diagnostic"`完成12端点共960条paired query；来源/checkpoint哈希不变，0训练/参数更新/dev/旧v3 holdout，`can_promote=false`。

## 执行完整性

- seed125新面40对/80题、32行、d=1，known/unknown只在目标事实槽位不同；source exclusion并集38,122个实体名、15,288个词值，含整张已消费v24面，零交集。
- 12/12完整bundle readiness通过；query阶段12端点各80题，共960条。执行CPU1,091.672秒，累计1,232.922/3,600秒。逐题progress、attempt/result SHA和来源/checkpoint核对通过，12个endpoint均无错误。
- 逐题原始输出、teacher评分与paired summary见[完整diagnostic报告](../../reports/taiji_n1_v25_abstention_diagnostic_20261019.json)。摘要独立从960行重算一致。

## 配对结果

| endpoint family | known精确回答 | known输出`unknown` | unknown输出`unknown` | unknown误答 |
| --- | ---: | ---: | ---: | ---: |
| N1-132 parent | 22/120 | 98/120 | 118/120 | 2/120 |
| N1-140 block candidate | 30/120 | 90/120 | 119/120 | 1/120 |
| N1-141 candidate | 9/120 | 111/120 | 120/120 | 0/120 |
| N1-141 control | 31/120 | 89/120 | 120/120 | 0/120 |

N1-141 candidate按seed的known精确回答为1/40、2/40、6/40，unknown均40/40输出`unknown`。三seed合计9/120组为“known答对且unknown弃答”，111/120组对known与unknown都输出`unknown`。这支持当前诊断所见的长上下文known问题过度弃答模式；它不直接测量内部置信阈，也不构成能力或泛化确认。

## 后续

依据用户授予的执行计划批准权，已批准并冻结[PLAN-N1-149](PLAN-N1-149_matched_abstention_context_length_curve_diagnostic_draft_20261019.md)：用全新seed126、40个逻辑配对样例比较8/16/32行嵌套上下文，target固定索引0；共240题/endpoint、2,880条查询，CPU≤7,200秒。该diagnostic不训练、不重用v25题、不设置晋升门。

## 绑定证据

- N1-148冻结计划SHA-256：`35214c44aec6eb33f1ea57bcada0eabd597d69fa29528228f0a158e21e1ba966`
- v25 face SHA-256：`f63a086c62ac2ccb3b3e44c751735864be57244146c9499aed04b90fa2a748f9`
- readiness SHA-256：`8c99612ba9ef6850497f6541484a663fe0aca4e1c80066445a96e7ed8db30ca6`
- attempt SHA-256：`09722dc9b989194fcbdacd7a4a351f33f532a79ae19f3d7b87f7a85e6229a495`
- progress SHA-256：`4d60fd5807294eb0c93546b0a31ff4482ec3ad12ed2cf5d7c5488a29f26d5e91`
- 结果SHA-256：`468fec02ff4d50a7362660e0170065cda6f81f0139261ccbe44429da38b37f51`
