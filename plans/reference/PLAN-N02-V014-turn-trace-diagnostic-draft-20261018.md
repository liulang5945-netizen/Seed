# PLAN-N02：V014 候选档逐轮回答追踪（待批准）

**状态：提案，尚未实施。** 本文只记录已完成的逐题半对读数，并提出一个新的、零训练测面。当前 N02/C4 判据、训练配方与能力结论均不变。

## 已完成的读数

- 母档半对：`reports/taiji_c4g1_replay_pi_V014_before_20261018.json`。
- 候选档半对：`reports/taiji_c4g1_replay_pi_V014_after_20261018.json`，SHA-256 `4cfd06e59828645f35d5b8db79583d93285baf91a804c226571b626c76610439`。
- 两半对的同题内容哈希均为 `c0031b5da5742a2c037139ab9cbf1c1a832d1b37d979e745f1138a7cb1530c65`，`base_sha256_unchanged=true`。
- V014 全量配对判读：`reports/taiji_n02_replay_item_moves_c4g1_20261018.json`。其中 penalty 0.5、1.0、2.0 的 V014 命中翻转、成句数、循环尾数和文本条数与母档半对逐项一致；penalty 0.0 没有命中翻转，故不出现在判读器的变化项列表。
- 母档在 penalty 0.5、1.0、2.0 的末轮回答均命中「浣熊」，penalty 0.0 未命中；四档 `answer_formed=false`。
- 候选档在四档惩罚下都未命中「浣熊」，末轮答案完全相同：21 字的「养一」循环；四档 `answer_formed=false`，`formed_texts=1`、`texts=3`。`cyclic_tail_texts` 依次为 1、2、2、2。候选 checkpoint 的 SHA-256 为 `da6954472c0411b55710d247aa9ba6d28a979213bb780e7aaeb0d0e3ff27c557`。

这说明候选档的目标字缺失在 penalty=0.0 时已经存在；现有读数不能说明它在哪一轮开始消失，也不能单凭这一题认定训练机制或泛化能力发生改变。

## 唯一建议的下一步

为现有 `scripts/training/measure_taiji_a30_repetition_penalty.py` 增加显式 opt-in 的 `--emit-turn-trace`，默认报告保持逐字节不变；随后只对 V014 做一次 penalty=1.0 的母档/候选档配对复测，保存三个对话回合各自的完整回答及其命中、成句和循环标记。这样可以定位「浣熊」是在候选档哪个回合开始缺失，且母档该档曾命中，可作同题锚点。

冻结范围：

- 只使用同一已消费题项 V014、同一 manifest、同一回路与 penalty=1.0；母档和候选档各新建一次 runtime，共 6 次回合生成。
- 记录两份 checkpoint、manifest、回路、测量脚本及输出报告的 SHA-256；题项内容哈希须保持上述值。
- 生成一次新的 write-once 报告；目标已存在即停止，不覆写。
- 独立 CPU 上限 180 秒；超限则停并保留失败证据。
- 零权重更新；不加入新题、不查询 dev/holdout、不改变预注册门、不作能力晋升或因果主张。
- 先通过 opt-in 输出合同测试，并确认两枚既有 checkpoint 的摘要与加载身份，再执行这六次生成；不另存或改写 checkpoint。

这是对既有 V014 题目的新逐轮观察面，需 owner 明确批准后才能实现和运行。完成后再决定是否值得提出训练剂量或机制变更。
