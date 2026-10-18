# PLAN-N1-111：seed13/position 权重—观察历史交叉归因（执行结案）

**状态：已停止，未发出模型查询。** Owner 于 2026-10-18 批准本方案。批准时方案快照 SHA-256：`2072eec6d9be795d6245826d3fe267ea5eeceebc79bb2fc6fd1bdec8ce238c74`。首次 runner 在恢复候选 checkpoint 时触发协议校验错误；依本文单次执行约束，不在 PLAN-N1-111 下重试。

## 依据与问题

- PLAN-N1-110 的训练门已失败并结案，不能续训、补训或查询 v11。
- 匹配的 run100 seed13/position 基线在 original/rename 四条件联合上均为 48/48；N1-110 length-balanced 候选为 5/48、4/48，且两个视图的成功父组没有交集。详情见 [N1-110 closeout](PLAN-N1-110_execution_closeout_20261018.md) 与只读 [training shard attribution](../../reports/taiji_n1_length_balance_failure_attribution_20261018.json)。
- 当前证据显示错误在 gold-prefix 与 free 评分里一致，且输出是合法方向词；但仅凭聚合训练报告无法判断差异跟随已学习权重，还是跟随存储的 43,008 条训练观察历史。

问题：同一固定 train transition 面上，position 四条件联合失败主要随 checkpoint 的网络参数变化，还是随 length-balanced 更新后存入的观察历史变化？本实验只作这一项机制归因，不作泛化或能力晋升主张。

## 固定设计

从两枚已冻结终点构造 2×2 交叉，不写回 checkpoint：

| 网络参数来源 | observation history 来源 | 角色 |
| --- | --- | --- |
| run100 baseline | run100 baseline | 原生基线锚点 |
| run100 baseline | N1-110 candidate | 只换记忆历史 |
| N1-110 candidate | run100 baseline | 只换网络参数 |
| N1-110 candidate | N1-110 candidate | 原生候选锚点 |

- 基线终点：`data/checkpoints/n1_hard_neighbor_training_v1_20261017/s13_position_final.pt`，SHA-256 `09df1d47eb490ef9afc5d563d08b40eca6b51951096875d016a67f133f7ed653`。
- 候选终点：`data/checkpoints/n1_hard_neighbor_length_balanced_training_20261018/s13_position_final.pt`，SHA-256 `d215a51d571976f216af43d0b7372f3cb702b6258237a7144252df4609f3b881`。
- 只使用现有 train split 的 48 个 transition 父项；每种状态组合跑 original 与 entity-renamed 两视图，各视图含 48 父项 × 4 条件。共 384 行/组合、1,536 行总计。
- 先运行两个原生锚点并逐项复现其已封存报告的 `joint_four_exact`（基线48/48、候选5/4）；任一不复现即停止，不运行两个混合组合。复现通过后才查询两种混合状态。
- 只在内存中交叉替换网络参数与观察历史；不优化参数，不写 checkpoint，不变更模型结构、训练目标、门槛或已消费题面。推理用现有确定性 argmax 路径。
- 输出按 checkpoint/manifest/runner SHA、状态组合、条件、方向×转向 cell、四条件联合数和逐题命中绑定；同时写 `model_loads`、`model_queries`、`parameter_updates` 与检查点前后摘要。
- CPU 上限 600 秒，单次 write-once 报告；目标已存在即拒绝。执行中断或来源不一致都停止，不重试、不补查询。
- 不读 dev/holdout，不造 v11 面，不跑确认，不作 promotion claim；`can_promote=false`。

建议产物：`scripts/training/diagnose_taiji_n1_position_weight_history_crossover.py`、专属合同测，以及 `reports/taiji_n1_position_weight_history_crossover_s13_20261018.json`。全部实现与合同测试通过、checkpoint/来源守卫就绪后，才执行一次诊断。

## 判读边界

- 若候选权重配基线历史仍失败，而基线权重配候选历史通过，证据将指向网络参数变化一侧；相反则指向历史内容一侧。
- 若两个混合都失败或都通过，结果说明存在交互或该 2×2 对照不足以分离来源；不得把单一混合结果包装成因果结论。
- 即使原生锚点复现且混合出现明显差异，该训练面结果也不替代新未见确认面；任何新训练 recipe、架构变化或新确认面均需另立冻结计划。

## 执行记录（2026-10-18）

- 合同测试 4 项通过；Ruff、Black、Python 编译通过；来源预检复现了原生锚点的封存门值：baseline 48/48、48/48，N1-110 candidate 5/48、4/48。预检未加载 checkpoint、未查询模型。
- 一次正式 runner CPU 9.703125 秒后停止。baseline checkpoint 完整恢复 1 枚；候选 payload 解码后，被 run100 专用 `load()` 拒绝：该加载器要求 `n1-hard-neighbor-fresh-paired-training-v1`，N1-110 candidate 使用 `n1-hard-neighbor-length-balanced-training-v1`。这是恢复契约不兼容，checkpoint/source hashes 未漂移。
- 查询与更新均为 0；两个 checkpoint 前后哈希一致；dev、holdout、confirmation 均为 0。两个 native anchor 与 hybrid 都没有运行。本报告字段 `checkpoint_loads=1` 统计完整恢复成功数；另见[计数澄清](../../reports/taiji_n1_position_weight_history_crossover_s13_20261018_restore_erratum.json)：实际两个 payload 均已解码，但只有 baseline 完整恢复。
- 执行报告：[`报告`](../../reports/taiji_n1_position_weight_history_crossover_s13_20261018.json)，SHA-256 `388e8ebe7717e5086e1dc8feb746c69854dad88411d152a10fe5c84f731d3a00`。

## 唯一建议的下一步

PLAN-N1-112 已按 owner 长期授权独立完成：同一题项、四格、1,536 条查询和判据，修正恢复契约后原生锚点全部复现。PLAN-N1-113/114 后续静态归因显示 reader/readout 是最大终态参数差异来源，且 candidate 的七字符材料更新带来更高 QA loss，但这些是描述性证据。PLAN-N1-111 的 `retry_allowed=false` 仍按原样结案，本次独立批次不是其重试。唯一下一步为裁定 PLAN-N1-115 五组参数全因子干预诊断。
