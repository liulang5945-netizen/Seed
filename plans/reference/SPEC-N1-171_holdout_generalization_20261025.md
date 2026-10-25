# SPEC-N1-171：留出泛化探针的选择、口径与判据

日期：2026-10-25。真实更新前冻结，承接[PLAN-N1-171](PLAN-N1-171_holdout_generalization_20261025.md)。本包只测「未参与更新的同分布 train 题能否被按题生成」，不是正式算法胜负、不是 L2、不是默认采用。

## 1. 选择与隔离不变量

1. 语料只来自冻结的 assistant train 加载函数（525 episode / 340 family），不读 dev/final/retention，不新增数据。
2. 每 family 至多 1 题，16 题来自 16 个不同 family；family 排序与同 family 内的取舍只由 sha256 决定，不含随机数。
3. 报告写入 16 题的 episode_id、family_id、秩、折、prompt SHA256、target 前 128 byte SHA256、prompt/target 字节数，以及整体 selection digest。
4. 每折的 4 个留出题与 12 个训练题在 prompt SHA256 与 target 前缀 SHA256 上都不得相交；不满足即工程失败，不静默改选择。
5. `dev_queries=0`；默认模型、旧 parent、旧 native 入口字节不变；`can_promote=false`。

## 2. 训练与评估口径

- 模型为 `ConditionalLanguageCandidateV2`（v2，参数 970,945），配置 `TaskCoreConfigV2()` 默认；每折新建、`bundle_id` 含折号，seed 169。
- 训练 64 次更新，每次使用该折 12 题的完整 batch（按秩固定顺序），target 取前 128 byte；AdamW lr 0.001、weight_decay 0、betas/eps 默认、clip 1；不采样、不换序、不搜索超参。
- 每折 0 步与 64 步评估同一 16 题；TF 与自由生成都只经 `next_logits`，无 renderer 补答案，UTF-8 合法字节掩码与 N1-170 相同。
- 自由生成固定 128 byte（遇 `<|end|>` 提前停止）；前缀匹配定义为生成的 target 前缀长度 / min(128, len(target))。
- 对照 A＝留出 prompt × 同折下一道留出题 target（末尾回绕）；对照 B＝留出 prompt × 训练 target 序号 `(3i+1) % 12`（i 为留出题在本折的序号）；对照 C＝留出 prompt × 对照 A target 的教师强制，只取 TF top-1/NLL。

## 3. 存档与恢复

- checkpoint 沿用 `taiji-conditional-language-trainer-v2` 格式：config、bundle、parameter revision、参数、AdamW、RNG、steps/曝光、TaskStateV2、cursor、digest。
- 每折保存 `fold{f}_final.pt`，另保存一份 0 步 `zero.pt`；全部 write-once，不覆盖 N1-170 存档。
- 每个存档在**新进程**恢复并核对：checkpoint digest 一致、参数 revision/steps/曝光一致、下一生成步 logits digest 与 15 byte 继续生成一致。

## 4. 判据与读数

- 前置 P：64 步时 12 个训练题 TF top-1 中位数 ≥0.90。
- 主判据 G：16 个留出题中 own 自由匹配 ≥0.20 的题数 ≥8，且这些题 own 匹配 ≥对照 A +0.15 且 ≥对照 B +0.15。
- 次判读 S：逐折 ≥0.20 题数、own/错配 TF 均值、A/B 均值、16 条自由输出中的不同输出数、训练题自由匹配分布；全部只报告。
- 出口：G 成立 `PASS_BINDING`；G 不成立且 P 成立 `FAIL_NO_GENERALIZATION`；G 不成立且 P 不成立 `INCONCLUSIVE_UNDERFIT`。任一出口都不改默认模型。

## 5. 失败与资源出口

- 工程失败（选择相交、存档/恢复、身份、来源变化）先修可逆实现；修复不得刷新本合同、不得改选择或判据。
- 费用上限 1,200 CPU / 1,800 wall（脚本内部 1,140/1,740 预留 60 秒发布）；触顶保存 partial 与真实费用，不续跑、不换种子/学习率/折数。
- 报告为 write-once；已存在即拒绝重跑。0 dev、0 reference、0 N1-166。
