# PLAN-N1-147 执行结案：v24配对弃答诊断停止

**状态：一次性query attempt已启动但scorer拒绝diagnostic split；完成输出0条，不构成行为结果。**v24 seed124面已消费，不重跑、不把题面改标为holdout。

## 执行与失败边界

- v24面40对/80题、每题32行，准备阶段来源并集38,122个实体名/15,288个词值；题面合同通过。N1-147A修复入口下12/12完整bundle独立恢复，累计CPU143.125/3,600秒。
- 首endpoint N1-132/seed11/parent恢复后进入查询runner。计数器记1，但`diagnostic_rows_completed=0`，progress没有`query_row`事件；N1-144确认scorer在`measure()`开头拒绝`split="diagnostic"`，错误为`ValueError: confirmation holdout split and frozen arm required`。拒绝发生在`paired.project/observe`、teacher logits和生成之前，因此没有生成模型输出。
- attempt报告的bundle fingerprint在加载前记录；scorer finally路径没有产生fingerprint不一致错误。checkpoint哈希未变，0参数/optimizer更新、0 dev/旧v3 holdout查询。runner计数器和零输出状态见[完整结果](../../reports/taiji_n1_v24_abstention_diagnostic_20261019.json)，单条endpoint错误摘要由progress SHA绑定。
- 查询阶段CPU22.234秒，累计165.359/3,600秒，来源哈希复核不变；`complete=false`、`can_promote=false`。这次停止不判定known/unknown行为差异。

## 根因与后续

readiness本身已通过；失败是N1-147复用的holdout-only scorer不接受正确标记的diagnostic split。已批准的N1-147A只修复了哈希校验器绑定，未改变split语义。修正方案是保持diagnostic split，另实现具有该split专属守卫和完整bundle fingerprint检查的scorer。

依据用户授予的执行计划批准权，已批准并冻结[PLAN-N1-148](PLAN-N1-148_matched_abstention_diagnostic_scorer_revision_20261019.md)。它使用全新seed125题面；整张v24面纳入source exclusions，v24不再查询。

## 绑定证据

- N1-147计划SHA-256：`4d6c9799741d3290f5f1e6cda4325268ce086027e4d14a72f82a7f3662e09be8`
- v24题面SHA-256：`0622de968749c4f573007ad5bc5fcc8ee30a09ae660b3fecfd7db10e52ce7216`
- readiness SHA-256：`df330eed167e5acc3cdc5dc3466f2f0e7f28ecd30c2474204b87d77aa370fa6e`
- attempt SHA-256：`42c59e6a327a0b4ef96ae4a43f908c699095e56dd8a98f6a5f9da12da860850c`
- progress SHA-256：`f8f745c4886dead15993ddce8af38c08b20a0ae551de33ac7b3fe382b61f2dc5`
- result SHA-256：`bc02f88a6b6963bfe162e69f3832442ffa44019161fd3ca774b172fcde75a049`
