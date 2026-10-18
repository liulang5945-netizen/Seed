# PLAN-N02：V014 候选档逐轮回答追踪（待批准）

**状态：已批准并执行。** 本次只扩展了 V014 的逐轮观测；N02/C4 判据、训练配方与能力结论均未改变。

## 已完成的读数

- 母档半对：`reports/taiji_c4g1_replay_pi_V014_before_20261018.json`。
- 候选档半对：`reports/taiji_c4g1_replay_pi_V014_after_20261018.json`，SHA-256 `4cfd06e59828645f35d5b8db79583d93285baf91a804c226571b626c76610439`。
- 两半对的同题内容哈希均为 `c0031b5da5742a2c037139ab9cbf1c1a832d1b37d979e745f1138a7cb1530c65`，`base_sha256_unchanged=true`。
- V014 全量配对判读：`reports/taiji_n02_replay_item_moves_c4g1_20261018.json`。其中 penalty 0.5、1.0、2.0 的 V014 命中翻转、成句数、循环尾数和文本条数与母档半对逐项一致；penalty 0.0 没有命中翻转，故不出现在判读器的变化项列表。
- 母档在 penalty 0.5、1.0、2.0 的末轮回答均命中「浣熊」，penalty 0.0 未命中；四档 `answer_formed=false`。
- 候选档在四档惩罚下都未命中「浣熊」，末轮答案完全相同：21 字的「养一」循环；四档 `answer_formed=false`，`formed_texts=1`、`texts=3`。`cyclic_tail_texts` 依次为 1、2、2、2。候选 checkpoint 的 SHA-256 为 `da6954472c0411b55710d247aa9ba6d28a979213bb780e7aaeb0d0e3ff27c557`。
- 批准的逐轮复测：`reports/taiji_c4g1_replay_pi_V014_turn_trace_20261018.json`，SHA-256 `c3f63e6ab29c2aa12d3e08abc38206a66048a856e2676903af97a37ad4fba575`。同一 V014 内容哈希，penalty=1.0，母档/候选档各三轮，共 6 次生成；CPU 105.796875 秒（上限 180），两枚 checkpoint 前后 SHA 不变，更新/dev/holdout/confirmation 均为 0。
- 逐轮读数：母档依次为 11 字成句未命中、21 字循环未成句、21 字含「浣熊」但未成句且非循环尾；候选档为 22 字成句未命中、21 字循环未成句、21 字循环未成句且始终未命中。母档仅第 3 轮命中；候选档三轮均未命中。

这条逐轮读数把现象限定到已消费的单题：母档只在第 3 轮带出目标字，候选档没有一轮带出；两档第 1 轮已经不同（11 字对 22 字），第 2 轮都进入循环。它只能定位最早可观察差异在第 1 轮回答，不能证明因果机制，也不支持泛化能力主张。

## 唯一建议的下一步

停止继续查询 V014；按 [PLAN-N1-110 执行结案](PLAN-N1-110_execution_closeout_20261018.md) 对已封存训练 shard 做一次只读逐题归因，聚焦 seed13/position 的四条件联合门失败，并比较 seed11/12 与长度/条件的差异。此步不加载 checkpoint、不生成新答案、不训练、不读 dev/holdout；结果用于形成下一版 N 主线训练方案。

冻结范围：

- 只使用同一已消费题项 V014、同一 manifest、同一回路与 penalty=1.0；母档和候选档各新建一次 runtime，共 6 次回合生成。
- 记录两份 checkpoint、manifest、回路、测量脚本及输出报告的 SHA-256；题项内容哈希须保持上述值。
- 生成一次新的 write-once 报告；目标已存在即停止，不覆写。
- 独立 CPU 上限 180 秒；超限则停并保留失败证据。
- 零权重更新；不加入新题、不查询 dev/holdout、不改变预注册门、不作能力晋升或因果主张。
- 先通过 opt-in 输出合同测试，并确认两枚既有 checkpoint 的摘要与加载身份，再执行这六次生成；不另存或改写 checkpoint。

以上冻结范围已按 owner 批准执行；不得把单题逐轮结果外推为训练剂量或机制结论。
