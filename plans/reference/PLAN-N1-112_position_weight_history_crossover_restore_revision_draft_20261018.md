# PLAN-N1-112：修正恢复契约后的权重—观察历史交叉诊断（待裁定）

**状态：已批准执行（依据长期授权，2026-10-18）。** 用户长期授权允许在不改计划范围、预注册判据或测量面的前提下自主推进。本修订保持同一两枚 checkpoint、48 个已消费训练父项、四条件、original/rename 视图及 1,536 条查询；只修复前次在任何模型查询前触发的协议加载器不兼容。PLAN-N1-111 仍按其 `retry_allowed=false` 结案，本次使用独立报告与 attempt 路径，不重开或覆盖前次记录。

## 失败依据

- PLAN-N1-111 合同测试、静态来源预检和两份 sealed train shard 核验均通过。执行读取了 baseline 与 N1-110 candidate 两个 checkpoint payload；baseline 完整恢复，候选在恢复时被拒绝，**model query=0**，没有执行任何 native anchor 或 hybrid 组合。
- 原执行报告：[`taiji_n1_position_weight_history_crossover_s13_20261018.json`](../../reports/taiji_n1_position_weight_history_crossover_s13_20261018.json)，SHA-256 `388e8ebe7717e5086e1dc8feb746c69854dad88411d152a10fe5c84f731d3a00`。计数澄清：[`restore erratum`](../../reports/taiji_n1_position_weight_history_crossover_s13_20261018_restore_erratum.json)。
- 根因是把 N1-110 checkpoint 交给 run100 专用 `load()`：后者要求协议 schema `n1-hard-neighbor-fresh-paired-training-v1`，候选报告和 payload 使用 `n1-hard-neighbor-length-balanced-training-v1`。checkpoint 与 source hash 未漂移。

## 拟执行的修订

- 保持 PLAN-N1-111 完全相同的四格：run100 baseline 与 N1-110 candidate 的网络权重来源 × observation-history 来源；只用 seed13/position 两枚既有终点和同一 frozen train transition manifest。
- 同一 48 个父项、四种条件、original/rename 两个视图；每格 384 条、总计 1,536 次生成。先测两个原生锚点，并逐行对比各自已封存的训练 shard；原生行或 joint-four 不完全复现则立即结束，不测 hybrid。
- 恢复时核验 checkpoint 固定 schema、seed/arm/update、payload protocol 与其自身报告 shard identity 相符；使用 `LocalRelationEvidenceCandidate.restore_bundle()` 恢复共同格式的候选 bundle，并恢复 Adam 状态后核对完整 bundle signature。不得调用只适用于 baseline 训练 protocol 的 `base.load()`。
- CPU 上限 600 秒。新报告采用 write-once 路径 `reports/taiji_n1_position_weight_history_crossover_s13_r1_20261018.json`；同时保留 attempt 文件。此提案本身仅供审批，preflight 通过且获准后才进行一次 runner 执行。
- 0 参数更新、0 optimizer step、0 checkpoint 保存、0 dev/holdout/confirmation 查询；checkpoint 前后哈希必须相同；不作泛化、能力晋升或 promotion 主张。若第二次 runner 仍在查询前失败，或原生锚点不符，不再重试。
- 判读仍使用已冻结的 44/48 两视图训练门：若 baseline 权重+candidate history 通过而 candidate 权重+baseline history 失败，读数指向权重变化；反向则指向历史内容；两格同过或同败只记为交互/未分离。不改门槛。

## 零查询实施准备

- r1 runner 已改为将 payload protocol 与 checkpoint 自身 sealed shard identity 核对，并通过 `LocalRelationEvidenceCandidate.restore_bundle()` 恢复模型；Adam 与完整 bundle fingerprint 核验仍在加载路径中。run100 专用 `load()` 不再用于候选恢复。
- 合同测试 9 项通过：除协议分歧接受/拒绝、四条件汇总与无训练调用外，还以合成模型 bundle 验证参数、观察历史和 Adam 状态恢复，并用 fake checkpoint reader 覆盖 `load_model()` 协议、签名与计数路径；无冻结 checkpoint 文件读取、模型推理或参数更新。Ruff、Black、编译均通过。
- `--validate-only` 仅检查冻结来源、已消费 factorial shards、前次 pre-query stop 和目录占用；将写入独立 readiness 报告 `reports/taiji_n1_position_weight_history_crossover_s13_r1_readiness_20261018.json`。预检设定为 0 checkpoint payload read、0 model load/query、0 update。
- 上述静态预检不解码 checkpoint。候选 bundle 的实际恢复、两枚 checkpoint 完整 signature 对账，以及模型查询均尚未验证，须在 PLAN-N1-112 获准后按同一次 runner 执行完成。

## 唯一建议的下一步

### 零查询预检结果（2026-10-18）

- `--validate-only` 通过，报告为 [`r1 readiness`](../../reports/taiji_n1_position_weight_history_crossover_s13_r1_readiness_20261018.json)，SHA-256 `b9ad33fb787b30bcc93f42ec3ea4d1c936f92eff35363a7d3436d67110d9a759`。
- 预检时读取的提案快照 SHA-256 为 `28235cd07e0bd540278f6c6f4cdf62dd08739fdc2abd11c251c236c751ae7844`；本执行记录在预检后追加，未改动冻结的诊断范围或判据。
- 核实 27 项 source guards、48 个原始与改名 transition 父项、两份 384 行 sealed shard，原生门值仍为 baseline 48/48、48/48，candidate 5/48、4/48。
- readiness 生成时合同测试 7 项通过；随后新增两项纯合成恢复测试，当前 9 项通过。readiness 报告绑定的 27 项 sources guards 仍全部与工作区匹配；checkpoint payload reads、model loads/queries、参数更新、optimizer steps、checkpoint saves 均为 0。Ruff、Black 与编译通过。
- 此预检只验证来源和设计准备度；它没有解码 checkpoint，因此未验证候选 bundle 的真实恢复，也没有产生 2×2 模型读数。

## 周边 N1 合同回归记录

- 运行 12 个相关 N1 合同文件，115 项通过、1 项失败：`test_n1_plan110_length_balanced_runner_contract.py::test_frozen_candidate_inputs_and_checkpoint_root_are_safe_before_preflight`。
- 失败来自该 preflight-only 测试在 2026-10-18 的已完成批次后仍要求 checkpoint 根目录、preflight 和训练报告不存在。当前 N1-110 报告完整，18 枚 zero/final checkpoint 均由封存报告标记为恢复相等；根目录文件集正好是该 18 枚。这里没有发现 r1 runner 回归。
- 该测试的 SHA-256 `8ba902d9d613b7c755ee19796443f2c15e3e0b21299d6a01c983171f658d4060` 被已结案 PLAN-N1-110 runner 固定来源守卫引用。尝试调整测试会被守卫拒绝；已将测试恢复到封存哈希，没有修改 N1-110 runner、checkpoint 或训练证据。本地失败记录保留，clean CI 环境结果尚未核验。

按本修订执行唯一一次 r1 runner。若原生锚点未逐行复现，即停止且不运行 hybrid；任何错误都不重试。此授权不包括训练、dev/holdout/confirmation 查询或能力晋升。
