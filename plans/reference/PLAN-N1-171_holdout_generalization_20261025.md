# N1-171：train-only 留出泛化探针（4 折轮换）

日期：2026-10-25，Asia/Shanghai。状态：**实施前冻结**。承接[N1-170 结案](PLAN-N1-170_execution_closeout_20261025.md)§7 的唯一后继建议。本包只回答一个问题：N1-170 的 v2 候选在**未参与更新的同分布 train 题**上，是否出现题-答绑定；不读 dev、不读 reference、不改默认模型、不新增预算。

## 1. VISION 条款对账

- VISION §81：decoder 必须实际读取共享语义与**被选中的证据**，由模型绑定对象/角色；不得由答案驱动的 renderer 补全。N1-170 只证明了已见 4 题的复现，没有证明「按题绑定」。
- VISION §89：**「loss 下降不等于自由生成或未见绑定有效」**。N1-170 的训练损失 5.553→0.301 nats、TF 0.9875 都属于该条明确排除的证据类型；本包用留出题的自由生成与错配对照替代 loss/TF 作为绑定证据。
- VISION §93「保持」要求独立面；本包不把留出题当保持面，也不把它写成 L2 或能力晋级。默认模型、旧 parent、已消费 dev 面均不变。

## 2. 证据：为什么现在就做留出探针

N1-170 在同一批 4 个 train 题上把 TF 从 0.681 升到 0.9875、自由匹配升到 0–78.1%，但同时暴露：

- 只有 4 个**已见**题、单 seed、target 首 128 byte，无法区分「记住 4 条序列」与「按 prompt 选回答」；
- `e888c23e` TF 0.992 但自由匹配 0，自由生成的是**另一题的已背回答**，说明题-答绑定不均匀。

在这两个缺口上继续加训练步数或加容量，都会得到不可归因的读数；因此先用**留出题 + 错配对照**判定绑定是否存在。

## 3. 设计：16 题、4 折轮换（对原建议的加强，不是放宽）

原建议是单次 12 训 / 4 留出。本包把它扩成**4 折轮换**：同一 16 题按确定性秩分成 4 折，每折 4 题留出、其余 12 题训练，四折各训一次独立候选。这样每个题都当过一次留出题，留出证据从 4 条升到 16 条；每折内部仍是严格的 12 训 / 4 留出，主判据比例仍是 50%（原 2/4 → 汇总 8/16），不降门。

确定性选择（先于运行冻结，运行报告写入全部哈希）：

1. train split 全量（525 episode / 340 family）按 N1-167/168 的冻结加载函数读取；
2. 逐 family 取**唯一**合格 episode：`1 ≤ prompt ≤ 512 byte`（逐 byte 证据窗口上限）、`128 ≤ target ≤ 384 byte`、response 含汉字、target 首 128 byte 无控制字节（`\t\n\r` 除外）；同一 family 多条合格时取 `sha256(episode_id)` 最小者；
3. family 按 `sha256(family_id)` 升序排列，取前 16；秩 0–15；折 `f = 秩 % 4`，每折留出 4 题（秩 ≡ f）；
4. 断言 16 个 episode/family/prompt/target 前缀各自唯一，且每折留出题与训练题无相同 prompt 或 target 前缀。

训练口径（每折独立，同一 v2、同一 seed/学习率，不搜索）：

- `ConditionalLanguageCandidateV2(TaskCoreConfigV2())`，bundle 名带折号；AdamW lr 0.001、wd 0、clip 1；
- 每步用该折**全部 12 题**（固定秩序）作一个 batch，target 取前 128 byte；
- 固定 **64 次更新**（与 N1-170 同批数；12 题/步 → 每步 1,536 target byte，每折 98,304 byte 曝光）。

评估（同一 `next_logits`，TF 与自由生成不分叉）：

- 每折在 0 步与 64 步对 16 题测：TF top-1/NLL/完整汉字、自由生成 128 byte 与前缀匹配；
- 64 步加三组对照，都用留出 prompt：A＝同折下一道留出题的 target；B＝训练题 target（按 `(3i+1) % 12` 取，确定性）；C＝留出 prompt 教师强制错配 target（只测 TF，不生成）；
- 另测 16 条自由输出中不同输出的个数，作为 prompt 是否改变输出的描述量。

## 4. 预注册判据（先于任何更新）

- **前置 P（可判性）**：64 步时 12 个训练题 TF top-1 的中位数 ≥0.90。达到才允许把阴性读数解释为「无泛化」；未达到记 `INCONCLUSIVE_UNDERFIT`，不称绑定失败。
- **主判据 G（题-答绑定）**：16 个留出题中，自由 128 byte 前缀匹配 **≥0.20 的题数 ≥8**，且这些题自身匹配同时比对照 A、对照 B 各高 **≥0.15**。满足称「已见分布内的题-答绑定」；不满足按 P 分档记阴性。
- **次判读 S（只报告，不设门）**：逐折 ≥0.20 的题数（4 个）、留出 own vs 错配 TF 的均值、对照 A/B 均值、不同自由输出个数、训练题自由匹配分布。

三档出口：G 满足 → `PASS_BINDING`；G 不满足且 P 满足 → `FAIL_NO_GENERALIZATION`（记忆/复现而非泛化）；G 不满足且 P 不满足 → `INCONCLUSIVE_UNDERFIT`。

## 5. 费用与停止线

- 上限 **CPU ≤1,200、wall ≤1,800**，落在 CC-2 剩余额度（1,418.047 CPU / 2,006.328 wall）内；不新增预算。
- 预计约 650 CPU（N1-170 实测 1.45 ms/target-byte × 4 折 98,304 byte ≈ 570 CPU + 评估/恢复）。触顶即停止，保存 partial/failure 与真实费用，不换种子、不加步数、不续跑。
- 0 dev、0 reference、0 旧 N1-166；`can_promote=false`；默认模型与 parent 字节不变。

## 6. 落地顺序

1. 新增 `scripts/training/calibrate_taiji_n1_plan171_holdout.py`（`--select-only` 先只读打印选择，再正式运行）；
2. 跑 4 折训练与评估，写 once-only 报告与逐折 checkpoint，全部在新进程恢复核对；
3. 写 [N1-171 结案](PLAN-N1-171_execution_closeout_20261025.md)，同步 [03](../active/roadmap/03_CURRENT_EXECUTION.md)、[10](../active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md) 与[黑板](delivery-runs/20261021-n-mainline/board.md)，提交。
