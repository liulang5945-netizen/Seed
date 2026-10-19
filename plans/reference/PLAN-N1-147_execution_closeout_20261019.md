# PLAN-N1-147 执行结案：v24匹配弃答诊断前置失败

**状态：attempt已启动但在scorer split守卫处停止；未形成任何逐题输出，不能作为行为读数。**按冻结的一次性规则，seed124 v24面不重跑；其全部实体/材料值保留在source exclusion并集。

## 执行事实

- v24题面已按合同一次生成：40组成对known/unknown、80题、每题32行；来源并集38,122个实体名、15,288个词值，source hash未变。
- N1-147A修复入口通过：12/12个完整bundle独立恢复，readiness CPU139.438秒，连同建面累计143.125/3,600秒。0参数更新、0 dev/旧v3 holdout查询。
- 查询attempt只进入首个endpoint。N1-144通用冻结确认scorer在`measure()`检查`split == "holdout"`时拒绝本计划正确标记的`diagnostic`题。attempt计数器在调用`measure()`前递增1，但已完成逐题行0；split守卫在observe、teacher logits和生成之前触发。progress仅有一个endpoint错误摘要，没有`query_row`事件。
- checkpoint字节哈希未变，参数更新/optimizer step均为0，来源哈希复核不变。执行CPU22.234秒，累计165.359/3,600秒。完整失败和progress SHA见[执行结果](../../reports/taiji_n1_v24_abstention_diagnostic_20261019.json)与[失败回执](../../reports/taiji_n1_v24_abstention_preflight_failure_20261019.json)。

这不是已知/未知行为比较结果，不能解释为模型能或不能弃答。失败归因是测量入口不接受diagnostic split。原v24面已消费，不修改split、不重试。

## 后续

依据用户授予的执行计划批准权，已批准并冻结[PLAN-N1-148](PLAN-N1-148_matched_abstention_diagnostic_scorer_revision_20261019.md)：用全新seed125面复做同一40对诊断，保持diagnostic标签，并采用独立只读scorer执行模型forward/generation与bundle指纹守卫。新面排除完整v24面；不训练、不更改问题配对、不设能力晋升门。

## 绑定证据

- N1-147计划SHA-256：`4d6c9799741d3290f5f1e6cda4325268ce086027e4d14a72f82a7f3662e09be8`
- v24题面SHA-256：`0622de968749c4f573007ad5bc5fcc8ee30a09ae660b3fecfd7db10e52ce7216`
- N1-147 readiness SHA-256：`df330eed167e5acc3cdc5dc3466f2f0e7f28ecd30c2474204b87d77aa370fa6e`
- N1-147 attempt SHA-256：`42c59e6a327a0b4ef96ae4a43f908c699095e56dd8a98f6a5f9da12da860850c`
- N1-147 progress SHA-256：`f8f745c4886dead15993ddce8af38c08b20a0ae551de33ac7b3fe382b61f2dc5`
- N1-147 result SHA-256：`bc02f88a6b6963bfe162e69f3832442ffa44019161fd3ca774b172fcde75a049`
