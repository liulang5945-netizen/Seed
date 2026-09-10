# Seed / Taiji 唯一执行计划

> 修订：2026-09-11；实际代码/报告基线 37c1dfd8。本文覆盖所有旧文档中的执行许可和“下一步”。
> 本轮任务是根据新增结果修订方案；训练与实现按下述验收顺序在后续开发中执行。
> 研究依据：[本轮源码与结果复审](../../reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)；[历史执行记录](../../archive/history/20260910_result_review/EXECUTION_HISTORY.md)。

## 阶段收束：完成研究审计，不等于完成模型验收

本轮从 601413cd 的收束基线继续完成了 P0 等 replay validation-only 诊断、P1 失败审计、P1.1 数据契约修复、P2 小预算 validation pilot、P2.1 只读输出/行动链诊断、P2.2 安全 bridge canary、P2.3 recovery continuation 数据合同审计、P2.3 targeted learning pilot、P2.4 retention canary、P2.5 novel-composition probe、P2.6 novel K2 learning、P2.7 independent holdout generalization、P3.0 checkpoint/interrupt-resume contract、P3.1 single-cell state preflight、P3.2 K→G owner-transfer preflight 和 P3.3 G candidate data-signal canary；没有读取新的 sealed payload，也没有 promotion 成绩。代码/报告证据以 37c1dfd8、P0 报告、P1 v1/v2 报告、P2 pilot/P2.1/P2.2/P2.3/P2.4/P2.5/P2.6/P2.7/P3.0/P3.1/P3.2/P3.3 报告及本结果复审为准。

| 工作线 | 收束状态 | 后续处理 |
|---|---|---|
| 五类 K1/K2 学习器 | 实现资产保留；共 5,648 有效参数，不代表通用语言或完整认知能力 | 用作同父代持续学习基线，先不扩参 |
| fast/slow 与 replay | 等 replay 下与直接 continuation 等价；拆分独立贡献未证实 | replay 作为效果基线；FS 只保留为状态实现候选 |
| widened / 旧 parity | 当前合成路线关闭；错误计数结论撤回 | 保留失败证据和 XL 对照，不继续补次数或凑容量 |
| C-stage / scorecard | 报告入账完成；覆盖范围有限，can_promote=false | 不追加同质 formal；换成五类、同预算、同 artifact 验证 |
| S/G/K 连续整合 | P3.1 事件合同、P3.2 owner-transfer 和 P3.3 candidate data-signal 均通过；G 仍未 fit，K1/K2 继续冻结为 inherited candidate/evidence 来源 | 进入 P3.3 冻结 K 的 G-only 小步学习 Gate；禁止把 control-only、owner-transfer、candidate label 或 lineage 元数据当作 learned 能力 |
| Seed / IDE / provider / 插件 | 已有工程资产保留；本轮未重新验收客户端全链路 | 仅修阻塞主线的故障；新能力按 P5 的依赖解冻 |
| CI、临时目录与发布 | 不把历史局部测试当当前全仓通过 | 变更相关检查随步执行；发布另验收，不批量删除未知资产 |

长期核心目标不变：Taiji 拥有认知状态与行动选择，继承已有权重和学习状态持续成长，并可使用成熟技术。当前五类任务只是实验载体，不应被固化成架构能力上限。神经群体协作、开放式成长、跨域迁移和自主进化仍是待验目标，不能从模块存在或 checkpoint 数量推断完成。

### 下一阶段唯一交付目标

**P2.2 已完成，P2.3 数据合同/targeted learning、P2.4 retention、P2.5 novelty probe、P2.6 novel learning、P2.7 holdout generalization、P3.0 checkpoint/interrupt-resume contract、P3.1 single-cell preflight、P3.2 owner-transfer preflight 和 P3.3 candidate data-signal canary 已完成但模型仍未 promotion，下一步只做 P3.3 冻结 K 的 G-only 小步学习 Gate。** P0 已完成，P1.1 已通过数据 Gate，P2 pilot 已真实训练但未通过能力晋级；P2.1 证明连续 MSE 下降没有转成更高离散命中；P2.2 证明 typed abstention、根目录 recovery route 和世界对齐控制的工程合同可执行，但 recovery/alignment 成绩是 oracle control；P2.3 candidate-only fit 暴露了保持问题，P2.4 已用固定 rehearsal 保住旧类；P2.5 定位了新组合 K2 content 缺口，P2.6 学到该目标，P2.7 在 2 个全新 project、4 条全新 path 上达到 K2 content/Workbench `4/4`，局部泛化 Gate 通过；P3.0 证明 K1/K2 continuation 的 checkpoint 状态边界可恢复；P3.1 证明 S/G/K 事件合同可恢复；P3.2 证明 K 提供候选、G 持有选择且不读取外部运行时 target，但没有训练 G 或新增参数；P3.3 data-signal canary 用 40 条 train/10 条 validation 证明了候选、pair/abstain label、竞争干扰和 runtime 去标签合同成立，但尚未训练 G。P3–P5 是后续路线，不是当前并行待办。不以添加新器官、新 Gate、更大模型或训练次数代替输出、保持和可学习闭环验收。

- P0 已确定当前实现的效果基线：在 model17/course0、150 条 wake＋50 条固定 replay 上，FS 与 C-replay、FS-no-replay 与 C 的有效状态峰值差均为 `4.76837158203125e-7`，低于预先冻结的 `1e-5`；checkpoint preflight 通过。当前数据覆盖的验证类为 A/B/C，D/R 留给 P1。
- P1 v1 失败审计确认了根因：450 条 train 记录中每类表面 observation digest 为 90 个，但实际 K1/K2 mask-visible input 各只有 1 个；validation 缺 D/R、无 project 隔离。报告保留为失败证据，不覆盖。
- P1.1 已通过：修复后的 450 条 train + 10 条 validation 中，A/B/C/D/R 每类均有 2 个 K1/K2 visible input；course seed 改变可见序列；validation 覆盖五类，并与 train 在 project/template 上隔离。报告见 [P1 v2 数据契约审计](../../../reports/taiji_m5_k_p1_data_contract_audit_v2_20260910.json)，清单见 [P1 v2 manifest](../../manifests/taiji_m5_k_p1_data_manifest_v2.json)。model17/23/31 state_dict 仍相同，但本阶段只做单父代继承学习，该旁证不再作为 P1 Gate。
- P2 pilot 已完成：P1 v2 的 460 条记录重建为 0 mismatch；50 条均衡 wake（A/B/C/D/R 各 10）+ 10 条固定 replay；checkpoint 保存和独立进程恢复均通过。报告见 [P2 v2 pilot](../../../reports/taiji_m5_k_p2_validation_pilot_v2_20260910.json)，机械失败保留在 [P2 failure report](../../../reports/taiji_m5_k_p2_validation_pilot_failed_20260910.json)。
- P2 结果：frozen macro MSE `0.156574`；wake-only `0.029237`（Δ `-0.127337`）；wake-replay `0.030520`（Δ `-0.126053`）。但三臂宏观 semantic/transition goal/content 命中均为 `0.4`，R 类命中为 `0.0`；replay 相比 wake-only 反而使宏观和最坏类 MSE略差。因此只能确认连续输出拟合和 checkpoint 链路有效，不能确认离散输出、行动成功、抗遗忘或 replay 独立收益。
- P2.1 已完成只读诊断：10 条 validation 重新构建为 0 mismatch；三臂均为 5,648 有效参数，checkpoint 独立恢复通过。frozen/wake-only/wake-replay 的 K1→K2→planner→隔离 Workbench 成功率分别为 3/10、4/10、4/10；6/10 行因输入 confidence 低于 K1/K2 的 `0.55` floor 输出 `unknown`，不是 argmax 读出错；frozen 另有 1 条因 `stale_world_observation` 被 planner 拒绝。报告见 [P2.1 诊断](../../../reports/taiji_m5_k_p2_output_action_diagnostic_20260910.json)。
- P2.1 还确认既有 `READ_ONLY_ROUTES` 没有 `content:recover-target`→只读能力的路由。这个缺口不能用降低 confidence floor 或给缺失文件直接执行来掩盖；下一步必须先做安全 recovery bridge canary。
- P2.2 已完成：460 条清单重建 `mismatch_count=0`；三臂 6 条低证据行均生成可往返、不可执行的 typed abstention；`content:recover-target` 的 `workspace.list(path=".")` 在 2 条 R 控制行上全部通过且根目录约束成立；世界对齐控制 10/10 通过。恢复和对齐均标记为 `oracle_control`，不构成模型能力成绩。报告见 [P2.2 安全 bridge canary](../../../reports/taiji_m5_k_p2_2_safety_bridge_canary_20260910.json)。
- P2.2 重新划分了下一阶段指标：高证据可执行 cohort 只有 4/10 行；低证据 6/10 行的正确结果是安全 abstention；R 的真实“列举后重新观察并读取”连续数据尚未进入训练/验证合同。下一步必须先补齐这条可学习 continuation，再运行 targeted learning；不得降低 `0.55` floor，也不得把 oracle route 计入模型命中。
- P2.3 continuation 合同已通过：6 条 train、2 条 validation；初始 R 观察均为 `read_success=false`、Percept confidence `0.0`、下一步 `workspace.list(path=".")`；列举后的候选观察均为独立高证据 `confidence=0.99`、resolved 文件；K1/K2 示例往返、输入 digest、record digest、project/path/template 隔离均通过。manifest 和报告见 [P2.3 continuation manifest](../../manifests/taiji_m5_k_p2_3_recovery_continuation_manifest_v1.json) 与 [P2.3 contract report](../../../reports/taiji_m5_k_p2_3_recovery_continuation_contract_20260910.json)。
- P2.3 targeted learning 已完成：训练前 parent checkpoint 保存/独立恢复和三臂保存/独立恢复均通过；6 条 continuation candidate train、2 条 validation candidate 全程未把 validation 用于 fit。parent/targeted/reference 的 continuation validation 均为 2/2，说明没有新增能力；targeted 原五类 K1/K2 goal 命中为 1/4、3/4，parent 为 4/4、4/4，安全 abstention 仍为 6/6，`can_promote=false`。报告见 [P2.3 targeted pilot](../../../reports/taiji_m5_k_p2_3_targeted_learning_pilot_20260910.json)。
- P2.3 的失败归因固定为“candidate-only update interference”，不是数据合同失败：新 candidate 与 parent 的输出目标重复，训练没有可测增量，却改变了 B/C/D 的高证据语义读出。下一步必须用固定 50 条均衡 rehearsal 做 retention-preserving objective canary，不能继续单独增加 continuation fit 次数。
- P2.4 retention canary 已完成：P2 的 50 条 rehearsal digest、类别平衡和顺序全部复现；交错 50 rehearsal + 6 continuation 后，原五类 K1/K2 goal 命中仍为 4/4、4/4，Workbench 4/10，低证据安全 abstention 6/6，保存/独立恢复通过。continuation validation 仍为 parent 已有的 2/2，没有新增能力，`can_promote=false`。报告见 [P2.4 retention canary](../../../reports/taiji_m5_k_p2_4_retention_canary_20260910.json)。
- P2.4 的结论是保持目标已可用，但当前 recovery continuation 目标不是有效 novelty probe：它只重复了既有 `inspect-language` 输出。下一步禁止继续在这个目标上加 epoch；必须构造真正未见的 recovery 后语言/工具链组合，并先做 frozen parent validation-only 探针。
- P2.5 novel-composition probe 已完成：2 条 validation 候选均使用与 P1/P2 路径不重叠的 TypeScript＋可用 toolchain＋resolved＋inspect-language tuple；合同 digest、candidate path 隔离、checkpoint 保存和独立进程恢复均通过。frozen parent 在新组合上 K1 goal/content `2/2`、K2 goal `2/2`、K2 content `0/2`、Workbench `2/2`；两条 K2 均为 `ambiguous` 且 content 为 `None`，因此缺口定位为 K2 内容承接，不是识别、路由或 host 执行失败。报告见 [P2.5 novel-composition probe](../../../reports/taiji_m5_k_p2_5_novel_composition_probe_20260910.json)。
- P2.6 novel K2 learning 已完成：6 条 train candidate 与 2 条 disjoint validation candidate，固定 50 条 P2 rehearsal 按 P2.4 顺序交错；manifest 合同通过、validation 未 fit、参数未增长、训练前与三臂保存后独立恢复均通过。parent 新组合 K2 content `0/2`，rehearsal-only `0/2`，interleaved `2/2`；interleaved 的 P1 旧类 K1/K2 content `4/4`、安全 abstention `6/6`、Workbench `4/10`，不低于 P2.4 parent baseline。该结果证明“这个具体 K2 内容承接目标可学习且保持约束通过”，不证明开放泛化，`can_promote=false`。报告见 [P2.6 novel K2 learning](../../../reports/taiji_m5_k_p2_6_novel_learning_20260910.json)。
- P2.7 independent holdout generalization 已完成：4 条 holdout candidate 使用与 P1/P2.5/P2.6 全部 disjoint 的新路径，分属 2 个新 project；P2.6 learned arm 的已学 sanity K2 content `2/2`，holdout K1/K2 goal/content 均 `4/4`、Workbench `4/4`；frozen parent 同一 holdout K2 content `0/4`。P1 旧类相对 P2.4 baseline 不下降，参数量稳定，P2.6 checkpoint 独立恢复再次通过。报告见 [P2.7 holdout generalization](../../../reports/taiji_m5_k_p2_7_generalization_20260910.json)。这满足 P3 的局部泛化入场 Gate，但不自动授予 promotion。
- P3.0 checkpoint/interrupt-resume contract 已完成：固定 P2.6 `interleaved-rehearsal-novel` learned checkpoint 为 parent；uninterrupted、wake 中段中断恢复、replay 边界中断恢复的 worker/budget/RNG/stream digest/final cursor 全部一致；tampered cursor、wrong parent、missing lineage 全部拒绝，rollback 独立恢复通过。实际发生了 K1/K2 更新，但未新增 S/G worker、未增长参数，`can_promote=false`。报告见 [P3.0 checkpoint contract](../../../reports/taiji_m5_k_p3_0_checkpoint_contract_20260910.json)，清单见 [P3.0 manifest](../../manifests/taiji_m5_k_p3_0_checkpoint_contract_manifest_v1.json)。
- P3.1 single-cell state preflight 已完成：以 P3.0 continuation parent 为锚点，4 条 P2.7 holdout 形成 20 个五阶段事件；S/G/K owner mask、content digest、event/state chain、事件中点/阶段边界中断、两条独立恢复、rollback 和 tamper/base/manifest 拒绝全部通过。K 只读取 S evidence，G 只提供 control-only selection，最终 action 读取 G/K；K1/K2 checkpoint 未改变，`fit_called=false`、参数未增长、`can_promote=false`。报告见 [P3.1 report](../../../reports/taiji_m5_k_p3_1_single_cell_20260910.json)，清单见 [P3.1 manifest](../../manifests/taiji_m5_k_p3_1_single_cell_manifest_v1.json)。
- P3.2 owner-transfer preflight 已完成：在相同的 4 条 P2.7 holdout 上，K1 只提供 inherited goal/content candidate，GSelectionState 持有最终选择；K-only 与 owner-transfer 的 K1 selection、K2 output、safe abstention、Workbench 全部等价，外部 target 未进入运行时，参数未增长。事件/事件边界/case 边界独立恢复、trajectory digest 和 tamper/wrong-base/wrong-manifest/wrong-owner-mask 拒绝全部通过。报告见 [P3.2 report](../../../reports/taiji_m5_k_p3_2_owner_transfer_20260910.json)，清单见 [P3.2 manifest](../../manifests/taiji_m5_k_p3_2_owner_transfer_manifest_v1.json)。
- P2/P3.2 仍只允许使用冻结的 manifest、candidate/scorer/threshold/resource contract；每次训练前必须重新通过 checkpoint 保存/独立恢复 preflight。P3.3 必须冻结 P3.2 K parent，只训练 G；不得把外部 goal target、K1 已有输出或 owner-transfer 元数据冒充 learned G owner。
- 阶段完成必须同时给出训练基线、可重建数据清单、不可变候选、五类结果及失败分析。没有收益或保持失败也是可收束的研究结论，但不得因此解冻 P3 的能力整合或晋级。
- 按验收事件排期，不承诺缺乏运行时间依据的日历日期；同一时间只推进一个研究问题。

### P0 已完成：等 replay 机制归因

P0 的可重建入口为 [等 replay 诊断脚本](../../../scripts/training/eval_taiji_m5_k_p0_equal_replay_diagnostic.py)，结果见 [诊断报告](../../../reports/taiji_m5_k_p0_equal_replay_diagnostic_20260910.json)。它从同一 model17 v4 worker 快照派生 C、C-replay、FS、FS-no-replay，消费同一 150 条五类经历和同一 50 个 replay 索引，并逐 wake/replay/consolidate 记录轨迹与 A/B/C 验证类 MSE。结果完成“直接 continuation＋replay 是效果基线”的归因；它没有证明五类泛化、独立模型泛化、完整闭环或晋级。

因此不再以 FS 相对 C-replay 的效果差作为成长证据。FS 的剩余价值转为状态拆分、保存和恢复接口候选，P3 仍需独立进程中断续训验证。

### 自动推进与讨论边界

P1 合同完成后可做 P2 validation pilot。最终测试前冻结主指标、最低有意义收益、各类允许遗忘界和资源预算，注明各值的依据，禁止根据测试成绩放宽。

出现下列情况应提交已有成果并停在决策点：有效信号不足需要改变任务定义；公平对照后仍无收益需要改变学习机制；资源约束迫使缩减目标；或准备改变认知所有权/默认发布模型。讨论时给出证据、保留方案与替代方案的收益和代价，再更新唯一计划；不自动扩展训练规模或购买算力。

本轮已完成 P0/P1 validation-only 诊断、P1.1 修复、P2 小预算 pilot、P2.1 输出/行动链诊断、P2.2 安全 bridge canary、P2.3 continuation 数据合同/targeted learning、P2.4 retention canary、P2.5 novel-composition probe、P2.6 novel K2 learning、P2.7 independent holdout generalization、P3.0 checkpoint/interrupt-resume contract、P3.1 single-cell preflight、P3.2 owner-transfer preflight 和 P3.3 candidate data-signal canary；下一步只做 P3.3 冻结 K 的 G-only 小步学习 Gate，不读取 sealed、不扩 K、不进入结构成长或 promotion。

## 当前判断

K 信号空间已扩至五类；fast/slow＋真实 replay 已实现并通过基础机制测试。C-stage v2 的 D/R/A 小型评估中，FS 相对 C 平均 MSE 改善 0.001868，D/R 改善 0.002495，原报告四门通过。该结果包含额外 replay 的收益；跨模型独立性、全五类保持、等预算机制优势和 S/G/K 整合仍未验证。can_promote=false。

widened 合成路线已收束。先前“14,252 本轮更新”“v4 worker 三 seed 独立”“新 seed 即新课程”不再作为设计依据。P0 已把效果基线收敛为直接 continuation＋replay；P1 则证明当前课程的表面变化没有进入 K 的有效输入。旧 SGK v1 预注册待修订，暂停其 lineage-first 执行顺序。

P1.1 已把数据入口修复为可见状态优先的合同：A/B/C 通过语言证据的 resolved/ambiguous 变化，D 通过 header 语言证据的 ambiguous/resolved 变化，R 通过显式 recovery-language hint/no-hint 变化。P2 pilot 证明同一父代上的 K1/K2 连续 MSE 可显著下降，但离散 readout 命中不随之提升；这把问题从“有没有训练信号”推进到“输出阈值/目标/行动桥是否正确”。

## P3.0 已完成：checkpoint/interrupt-resume contract

P3.0 固定 P2.6 learned checkpoint 为 parent，把 P2.7 已通过的局部泛化能力放入可恢复的持续学习状态边界。它只验证 K1/K2 continuation，不把 S/G/K 联合成长写成已实现。

1. 固化 P2.2/P2.4 安全出口：confidence `<0.55` 的低证据样本只能产生 typed abstention；根目录 `workspace.list(path=".")` 仍是 host policy，不计模型 credit。此项已通过。
2. P3.0 已通过最小 continuation contract：parent、K1/K2 worker、wake/replay/consolidate phase cursor、experience/stream digest、RNG/采样状态、预算计数、origin/attached lineage 元数据均有内容地址；错误 parent、缺链和篡改 payload 均拒绝。
3. 三条可重建轨迹已通过：uninterrupted；wake 中段中断后恢复；replay 边界中断后恢复。每条都经过独立进程恢复，后续 worker/budget/RNG/stream digest/final cursor 与 uninterrupted 一致。
4. rollback 已通过：恢复到 P3 parent 后 K1/K2 source digest 保持一致。P3.0 没有新增 S/G worker、没有参数增长、没有把 lineage 元数据计作能力，`can_promote=false` 保持。

## P3.1 已完成：S→G→K 单 cell 状态整合预检

P3.1 已在 P3.0 continuation parent 上完成 4 条 P2.7 holdout 的接口回放，形成每例五阶段、共 20 个事件。schema、owner、读写 mask、事件 state chain、内容寻址、事件中点/阶段边界中断、独立恢复、rollback 和错误拒绝全部通过；报告见 [P3.1 report](../../../reports/taiji_m5_k_p3_1_single_cell_20260910.json)，清单见 [P3.1 manifest](../../manifests/taiji_m5_k_p3_1_single_cell_manifest_v1.json)。

关键边界必须保留：S 是 runtime evidence，G 是外部 goal/content selection 的 `control-only`，K 复用已有 K1/K2 learned worker；K readout 实际只读取 S evidence，最终 action 才读取 G/K。P3.1 没有调用 fit、没有新增参数，也没有证明 S/G 已经学习或 single-cell 优于 K-only；它只证明“状态可以正确接线、保存、恢复和回滚”。

## P3.2 已完成：S/K/G owner 边界迁移预检

P3.2 已在 P3.0 parent、P3.1 manifest 和同一 P2.7 holdout 上完成 owner-transfer 对照。K1 只提供 inherited goal/content candidate，`GSelectionState` 持有最终选择；K-only 与 owner-transfer 的 K1 selection、K2 output、safe abstention、Workbench 在 4/4 holdout 上完全等价，外部 target 未进入运行时，参数未增长。事件、事件边界、case 边界独立恢复，trajectory digest 一致，篡改 cursor、错误 base/manifest/owner mask 全部 fail-closed。报告见 [P3.2 report](../../../reports/taiji_m5_k_p3_2_owner_transfer_20260910.json)，清单见 [P3.2 manifest](../../manifests/taiji_m5_k_p3_2_owner_transfer_manifest_v1.json)。

边界结论是“选择所有权已可迁移且不破坏既有行为”，不是“G 已经学习”。P3.2 没有调用 fit、没有新增参数，也没有产生能力增益；因此下一步先建立并验证 P3.3 candidate-set data signal，随后才解冻冻结 K 下的 G-only 小步学习。

## 后续依赖顺序与验收

| 顺序 | 工作重点 | 进入下一步的条件 |
|---|---|---|
| P0 | 相同 replay 的机制归因 | **已完成**：两组轨迹均在 `1e-5` 内等价，checkpoint preflight 通过 |
| P1 | 真实学习信号与五类数据合同 | **已通过 P1.1**：五类各有至少 2 个 K1/K2 visible input，validation 五类覆盖且 project/template 隔离 |
| P2 | 五类学习及保持的独立验证 | **P2.7 局部跨项目/路径泛化通过**：holdout K2 content/Workbench `4/4`，旧类保持通过；已进入 P3.0 |
| P3 | 中断续训与 S/G/K 联合状态整合 | **P3.3 data-signal 已通过合同 Gate**：候选、标签隔离、独立恢复和 fail-closed 边界可复现；当前执行冻结 K 的 G-only 小步学习，S 仍非 learned |
| P4 | 结构成长必要性与收益验证 | 容量压力真实存在，增长收益优于强固定基线 |
| P5 | 知识来源、IDE、客户端、硬件发布 | 各项按所需模型能力与接口成熟度解冻 |

### P1：有效信号与评估数据（P1.1 已通过）

- 审计现有五类课程；按 K1/K2 typed mask 可见输入、目标张量、时序组合生成签名。分别统计经历数、唯一文件数、有效类数、模板族数。注释/文件名变化允许作为同分布扰动，不能充当新能力或新独立样本。
- train/validation/test 以项目或任务模板分组，完整覆盖 A/B/C/D/R；每类包含多个不同可见状态/组合，记录数量与重复率。测试设计应同时测同分布泛化和未见组合，不能每轮仅更换 seed。
- 父 worker 沿用现有权重即可做机制诊断；若主张跨模型泛化，则以实际初始化/训练流变化构建父 worker，并核验 state_dict 差异。所有独立性结论由实测决定，不强制为了 n=9 重建模型。
- 将 v1–v5 已读 sealed 登记为 consumed；后续开发用 validation。候选、scorer、阈值、资源预算和最终输入在读取下一份测试成绩前锁定。
- v1 失败合同保留为 [P1 失败审计](../../../reports/taiji_m5_k_p1_data_contract_audit_20260910.json)；修复后的合同状态为 `passed`，见 [P1 v2 审计](../../../reports/taiji_m5_k_p1_data_contract_audit_v2_20260910.json) 与 [P1 v2 manifest](../../manifests/taiji_m5_k_p1_data_manifest_v2.json)。P2 只能使用 v2 入口，不能用 v1 的一类一签名课程训练。

### P2：验证学习与保持

- 主要因果臂为 P0 选定流程与同新增训练预算的直接学习对照；F 测零更新漂移。FS 若仅是等价状态实现，无须继续宣称胜过同 replay 的基线。XL 作为容量参考，动态增长阶段再做最终容量对齐。
- 用五类宏平均和每类 MSE；D/R 弱类单列，A/B/C 旧强类非劣逐类检查，报告最坏模板结果。整体均值不能掩盖遗忘。
- 增加真实预测链：K1 预测→K2 多步状态→隔离 Workbench 执行。真实任务成功率/失败恢复与局部 MSE 分账；低置信度和 D/R 路径必须有可解释出口。
- 使用 validation-only pilot 确定样本量、最低有意义改善、数值容差和逐域非劣界。浮点噪声容差与“允许遗忘多少”分别定义；不能用 candidate 退化方差自动放宽所有门槛。
- 先保存候选和 presealed 合同，随后同一个 artifact 只读评分，不在第二阶段重训候选。记录代码版本、数据/参数/状态摘要和所有失败。
- 结果出口：收益/保持/资源通过→P3；无收益→回对应反馈或数据根因；数值等价→保留成本更合理的基线；机械错误→修复后重做技术预检。不得看测试成绩改当前版本阈值。
- P2 pilot 已执行上述最小预算和独立 checkpoint preflight；P2.1 完成了 K1→K2 级联与隔离 Workbench 诊断；P2.2 完成了 typed abstention、根目录 recovery 和世界对齐工程 canary；P2.3 continuation 数据合同/targeted learning、P2.4 retention canary、P2.5 novel-composition probe、P2.6 novel K2 learning、P2.7 holdout generalization、P3.0 checkpoint/interrupt-resume contract、P3.1 single-cell preflight、P3.2 owner-transfer preflight 和 P3.3 candidate data-signal canary 均已完成。当前模型结论从“只有连续 MSE 改善”推进为“一个具体 K2 content 目标在 rehearsal 保持约束下学习，并跨 2 个新 project/4 条新 path 泛化”；P3.0 已把 K1/K2 continuation 纳入可恢复状态边界，P3.1 已把 S/G/K 接线合同闭合，P3.2 已把选择所有权转给 G，P3.3 已证明 pair/abstain candidate set 有可训练信号但尚无 learned G 增益，下一步只做冻结 K 的 G-only 小步学习 Gate。

### P3：状态整合与同一父代连续课程（P3.3 data-signal Gate 已通过，G-only 当前）

P3.0 已用 [checkpoint/interrupt-resume contract](../../../reports/taiji_m5_k_p3_0_checkpoint_contract_20260910.json) 替代“先做 lineage 再补能力”的顺序：K1/K2 parent、worker、phase、replay stream、RNG、预算和 lineage 元数据已经可以内容寻址、独立恢复和回滚。P3.1 又把 S/G/K 的事件、owner mask、状态 digest 和恢复边界闭合，但没有把 control-only 的 S/G 说成 learned。P3.2 已证明 owner-transfer 不破坏已有行为：G 持有最终选择，K 提供候选，运行时不读取外部 target。当前 P3.3 必须冻结 P3.2 的 K，只为 G 建立真实学习信号和可恢复小步更新，不能回到原始从零训练，也不能重新授权暂停的 [SGK v1](../../reference/M4V2_SGK_PROMOTION_COURSE_PREREGISTRATION_20260910.md)。

1. 以 P3.0 parent、P3.1 manifest 和现有 K1/K2 worker 为唯一资产；S、K、G 的 schema、owner、输入输出 mask、attached lineage、checkpoint digest 和 rollback parent 必须独立可定位。
2. 把 K1 的共享语义表征与 goal/content 输出拆开记账：K 负责 evidence/world readout，G 负责 candidate selection/accept/reject；在拆分完成前，goal/content head 只能标记 `pending-owner-transfer`，不能同时计为 K 与 G 的能力。
3. 先实现无新增参数的 `GSelectionState` 和 owner-transfer adapter，外部 goal/content target 仅作 validation label；用同一 P2.7 holdout 运行 K-only 与 S/K/G 两臂，比较 K1/K2 goal/content、safe abstention、Workbench、延迟和 checkpoint 字节。
4. 在 observation、G selection、K readout、action 四个边界做独立进程恢复；验证事件/owner/worker/budget/RNG/logical digest 一致，篡改 G state、错误 parent、错误 mask 必须拒绝。P3.2 已满足该条件，下一步冻结 K 并进入 P3.3。

## 唯一下一步：P3.3 冻结 K 的 G-only 小步学习 Gate

P3.3 是本路线第一次让 G 产生真实 learned state，但不是把 K1 的已有 goal/content head 重新复制一份。目标是让 G 根据 K 提供的候选、证据和不确定性学习“选择/拒绝/安全 abstain”，而不是把外部目标直接喂给运行时。若候选集合没有可区分信号，必须停在数据/目标重设计，不用 identity fit 制造进步。

1. 冻结 P3.2 的 K1/K2 parent、候选生成、confidence floor、Workbench policy 和 owner-transfer manifest；训练前先做 parent checkpoint 保存、独立进程 restore、digest/lineage/rollback 预检，训练后对 G child 重复同一组检查。K 的 state_dict、参数计数和 digest 在 G 训练期间必须保持不变。
2. 建立独立的 G candidate-set contract：每条样本由 K 生成至少两个可审计候选（包括正候选、干扰候选或拒绝候选），记录 K evidence、candidate score、来源和最终 label；label 只在 fit/validation 评分时存在，绝不进入 inference input。P2.7 的 4 条 holdout 继续保持 untouched test，不得用于 G fit。
3. P3.3 data-signal canary 已通过：P1 重建出的 40 条 train + 10 条 validation 均有至少两个候选，目标覆盖完整，包含 `pair`/`abstain` 两类目标和竞争干扰项；project/path 隔离、runtime target 排除、P2.7 untouched test 和 P3.2 K checkpoint 独立恢复全部通过。候选集合当前主要是 K 提案与安全 abstain/reobserve 的二选一，不宣称已有性能增益；下一步才允许实现 G-only learner。
4. 在信号成立后只更新 G 的最小参数集，保留 K-only、zero-step owner-transfer 和 trained-G 三臂；比较 goal/content selection、safe abstention、Workbench、延迟、checkpoint 字节、G 参数数、K digest 以及旧类保持。所有指标先冻结再读取结果，不能根据结果放宽阈值。
5. Gate 必须同时满足：训练前后 checkpoint 可独立恢复；G 的验证选择/拒绝信号相对 zero-step 有可解释收益或至少不劣且归属确实转移；K 输出、旧类保持、安全出口、Workbench 不下降；错误 parent/manifest/owner mask/cursor 全部 fail-closed。任何一项失败都只修 G 边界或回滚，不扩训练轮数掩盖问题。
6. P3.3 不解冻结构增长、S 学习、CUDA、IDE/provider、客户端视觉或 promotion；只有 G owner 的真实小步收益和保持 Gate 通过后，才进入 P4 的容量压力与结构成长验证。

### P4：回归态极的长期目标——继承式结构成长

在固定容量持续学习和保持成立后，使用多任务干扰、容量扫描与长序列退化确定扩容压力。增长从同一模型继承有效权重和学习状态，新增结构零影响出生，并有可测 credit/活动/贡献。

对照至少包含最强固定容量学习流程、同最终有效容量的 fixed-large、动态增长及结构 lesion；同时比较质量、旧域保持和累计资源。不以 replica 翻倍、元数据 lineage 或单纯保存更多权重替代神经结构成长。是否引入成熟网络/表征组件，以项目需求和对照证据决策，避免原始从零重造与为避 Transformer 而降低能力。

### P5：外围开发顺序

1. 模型知识来源：Skill/MCP 文本、文档、成功/失败轨迹形成有来源语料；与普通数据同预算比较内化和未见任务收益。工具知识与宿主执行权限分离。
2. IDE/interaction-group/小型模拟：先支持模型产出类型化动作、识别文件语言、解释语言切换、preview/执行/undo；模拟和真实 Workbench 使用一致的动作合同。
3. 客户端插件热插拔与 provider watchdog：接口稳定后实现能力发现、兼容性、设备适配继承、故障隔离和回滚。语言 provider 可作语言器官，成绩不混入原生学习收益。
4. CUDA 待硬件具备后做 CPU 一致性与吞吐对照；产品视觉在接口稳定后统一 logo/托盘/任务栏、圆角、状态页和旧 HF 入口。严重客户端故障及本次变更引入的 CI 错误优先修复。

## 工程与文档纪律

每步完成后只更新本文当前状态和一份必要证据；归档调试流水，核心架构/需求持续留在 active。失败报告不覆写、不改绿。已否决 widened/旧 parity 不再获运行许可，产物保留可追溯性，确认无引用的临时产物才清理。

上次结果复审运行的 5 项 FS 基础测试通过；本次收束不重报为新增测试成绩。未运行新的研究训练，未宣称全仓 CI 通过。实现阶段按实际 workflow 运行相关 pytest、Ruff、B/SIM、Black、core mypy；脚本改变必须核对 CLI/输出合同与负例。计划/链接变更仅做 JSON、引用和 diff 校验，不新增镜像文档内容的测试。

文档仅保留一个执行入口，不再新增平行总计划：本文记录阶段状态、下一步和验收；结果复审记录证据解释；核心需求与架构常驻 active；已有历史流水保留在 archive。旧预注册即使留在 reference 也不重新获得执行许可。当前收束不移动有引用的研究资产，不覆盖旧报告或删除 checkpoint。根目录存在部分无读取权限的临时路径，未证明其为空或无用；后续清理须逐项验证绝对路径、引用和可恢复性，不能把它们报作已清理。

本轮 P2 pilot、P2.1、P2.2、P2.3 contract/targeted、P2.4 retention、P2.5 novel-composition、P2.6 novel-learning、P2.7 generalization、P3.0 checkpoint/interrupt-resume contract、P3.1 single-cell preflight、P3.2 owner-transfer preflight 和 P3.3 G candidate data-signal canary 报告与计划同步已提交本地 main。后续唯一入口是 P3.3 冻结 K 的 G-only 小步学习 Gate；不跳到结构成长、promotion 或外围路线，也不按历史“下一步”自动开跑。
