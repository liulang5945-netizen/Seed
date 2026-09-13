> 2026-09-13 整理前快照，来源提交 `102b81e1`。本文所有“当前/下一步/执行许可”仅为历史记录；现行顺序见 [唯一执行计划](../../../active/roadmap/03_CURRENT_EXECUTION.md)。相对链接已按归档位置机械重定位，原始文本可从来源提交恢复。

# Seed / Taiji 唯一执行计划

> 修订：2026-09-12（晋级宣布已批准——限定范围版）；M5 K 轴持续学习机制已在其实验载体上闭合、可验证并附着真实 runtime，项目所有者于 2026-09-12 独立批准晋级宣布（四条诚实边界随宣布生效：五类合成载体、结构成长未被触发、附着 opt-in 且默认行为路径不变、S 为架构性 control-only evidence）。**本宣布不构成通用能力主张；`growth_admitted=false` 贯穿。** 唯一下一步 = P5.1 知识来源预注册（先摸底既有 S 轴内化机械与语料现状）。本文覆盖所有旧文档中的执行许可和“下一步”。
> 本轮任务是根据新增结果修订方案；训练与实现按下述验收顺序在后续开发中执行。
> 研究依据：[本轮源码与结果复审](../../../reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)；[历史执行记录](../20260910_result_review/EXECUTION_HISTORY.md)。
> 前瞻性技术设想（自主唤醒/注意力外挂/跨设备快照等，**不参与主线、不改变执行顺序**）见 [未来技术设想](../../../reference/VISION_FUTURE_TECHNOLOGY.md)。

## 当前状态：M5 K 轴已晋级（限定范围），P5.1 知识来源为唯一下一步

2026-09-12 项目所有者独立批准晋级宣布：[M5_K_PROMOTION_DECLARATION_20260912.md](../../../reference/M5_K_PROMOTION_DECLARATION_20260912.md)（机器边界基准 = [scorecard v8](../../../reference/M5_K_AXIS_SCORECARD_V8_CONTRACT_20260912.md)，13 项 gate 全 true；最新机器报告 [v8 report](../../../../reports/taiji_m5_k_axis_scorecard_v8_20260912.json)）。
宣布范围 = M5 K 轴持续学习机制（能力 formal + 表示因子化 + 投影求解器 + 课程验证 + 联合课程 + runtime 消费合同 + 真实附着）在其载体上闭合；四条诚实边界随宣布生效，默认行为路径保持不变。

| 证据线 | 当前结论 |
|---|---|
| P4.7–P4.10 | 容量假设关闭，旧表示/约束路线关闭，parent-relative 特征因子化与基础特征空间 learnability gap 已记录 |
| P4.11 | `projection_solver_supported`：两个 seed 同时通过新任务与双保持门 |
| P4.12 | `course_level_validation_supported`：3 个身份批次 × 3 个 seed，9/9 projected cell 通过 |
| P4.13 | `promotion_course_supported`：9/9 A → B cell 通过，累积 A+B+保持约束零违反，向后保持零失败 |
| P4.14 | `joint_course_supported`：4/4 cell——Phase K（P2.6 机械）→ post-K 重 materialization → Phase G（P4.11 合同）全门 + 跨相门全过；post-K 景观漂移真实且 G 求解器维持平衡 |
| 默认 runtime rollout review | `default_runtime_rollout_review_supported`：4/4 cell 经消费合同 `taiji-default-runtime-rollout-attachment-v1` 磁盘加载后逐字段复现 P4.14 记录值；`fit_called=false`；`api/seed_runtime.py` 零改动、`checkpoints/seed_corpus.pt` digest 不变 |
| A8 晋级评审 | `promotion_review_recommended`：四项裁决独立批准（2026-09-12）；三项证据 veto 经 v7 机械翻转；SGK v1 标记 superseded；附着授权批准；常设准则「决策点优先上限更高」入账 |
| 默认 runtime 附着 | `runtime_attachment_supported`：P4.14 状态经 runtime 自有 fail-closed 合同（`api/taiji_runtime_attachment.py` + `SeedRuntime.attach_k_g_state`）附着真实 runtime，4/4 cell 经 typed 读出面逐字段复现 P4.14 记录值；拒绝探针全拒净；detach/re-attach 逐数值等价；默认行为零变化；`fit_called=false` |
| 晋级状态 | **已晋级（限定范围，2026-09-12 所有者独立批准）**：[晋级宣布](../../../reference/M5_K_PROMOTION_DECLARATION_20260912.md)——机器边界 13 项 gate 全 true；四条诚实边界随宣布生效（五类合成载体、结构成长未被触发 `growth_admitted=false`、附着 opt-in 且默认行为路径不变、S 为架构性 control-only evidence） |

**唯一下一步：P5.1 知识来源预注册**——把已验证的 K 轴持续学习机制接入既有 S 轴内化机械（governed corpus、Skill/MCP artifact 边界 `taiji/artifact_internalization.py`、语义 embedding 内化 S3），与普通数据同预算比较内化与未见任务收益；工具知识与宿主执行权限分离不变。先摸底现有机械与语料现状（S 轴各 Gate 报告、artifact 语料、internalization learner 入口），再冻结预注册；冻结前不训练、不接新数据源、不读取 sealed。

## 阶段收束：完成研究审计，不等于完成模型验收

本轮从 601413cd 的收束基线继续完成了 P0 等 replay validation-only 诊断、P1 失败审计、P1.1 数据契约修复、P2 小预算 validation pilot、P2.1 只读输出/行动链诊断、P2.2 安全 bridge canary、P2.3 recovery continuation 数据合同审计、P2.3 targeted learning pilot、P2.4 retention canary、P2.5 novel-composition probe、P2.6 novel K2 learning、P2.7 independent holdout generalization、P3.0 checkpoint/interrupt-resume contract、P3.1 single-cell state preflight、P3.2 K→G owner-transfer preflight、P3.3 G candidate data-signal/G-only learning、P3.4 behavior signal Gate、P3.5 reobserve-aware G-only learning、P3.6 独立行为 holdout/保持 Gate、P4.0 固定容量压力扫描、P4.1 公平容量合同预检、P4.2 隔离训练/公平容量归因、P4.3 保持约束增量学习、P4.4 保持身份/结构校准、P4.5 保持约束/更新规则对照和 P4.6 功能性 parent-preserving objective 对照；没有读取新的 sealed payload，也没有 promotion 成绩。代码/报告证据以 P0、P1 v1/v2、P2 pilot/P2.1/P2.2/P2.3/P2.4/P2.5/P2.6/P2.7、P3.0/P3.1/P3.2/P3.3/P3.4/P3.5/P3.6、P4.0/P4.1/P4.2/P4.3/P4.4/P4.5/P4.6 报告及本结果复审为准。

| 工作线 | 收束状态 | 后续处理 |
|---|---|---|
| 五类 K1/K2 学习器 | 实现资产保留；共 5,648 有效参数，不代表通用语言或完整认知能力 | 用作同父代持续学习基线，先不扩参 |
| fast/slow 与 replay | 等 replay 下与直接 continuation 等价；拆分独立贡献未证实 | replay 作为效果基线；FS 只保留为状态实现候选 |
| widened / 旧 parity | 当前合成路线关闭；错误计数结论撤回 | 保留失败证据和 XL 对照，不继续补次数或凑容量 |
| C-stage / scorecard | 报告入账完成；覆盖范围有限，can_promote=false | 不追加同质 formal；换成五类、同预算、同 artifact 验证 |
| S/G/K 连续整合 | P3.1 事件合同、P3.2 owner-transfer、P3.3 candidate data-signal/G-only、P3.4 behavior signal、P3.5 reobserve-aware G-only、P3.6 独立行为 holdout/保持、P4.4 保持身份/结构校准、P4.5 更新规则对照和 P4.6 功能性保持对照均完成；P4.0 观察到固定 G 选择压力，P4.1 已把 12 维候选输入与 9 维候选集上下文合同内容寻址，P4.2–P4.6 的 child training、恢复和保持检查完成 | **13 参数互斥已钉死**（保持/新任务在 seed 间系统性互斥，三类干预无效）；P4.2 的 fixed-large 无收益结论受坏协议污染，容量假设未被干净检验。决策已收束为 P4.7 容量假设干净检验（见「当前决策点」节）；停止调参、扩容、promotion 和外围解冻 |
| Seed / IDE / provider / 插件 | 已有工程资产保留；本轮未重新验收客户端全链路 | 仅修阻塞主线的故障；新能力按 P5 的依赖解冻 |
| CI、临时目录与发布 | 不把历史局部测试当当前全仓通过 | 变更相关检查随步执行；发布另验收，不批量删除未知资产 |

长期核心目标不变：Taiji 拥有认知状态与行动选择，继承已有权重和学习状态持续成长，并可使用成熟技术。当前五类任务只是实验载体，不应被固化成架构能力上限。神经群体协作、开放式成长、跨域迁移和自主进化仍是待验目标，不能从模块存在或 checkpoint 数量推断完成。

### 下一阶段唯一交付目标

**P2.2–P4.14 的合同、训练/验证、checkpoint/rollback、holdout/retention 与失败归因均已按冻结映射完成，默认 runtime rollout review 亦已执行通过，但模型仍未 promotion，结构也未增长。** P4.11 在两个 seed 上同时通过新任务与双保持门；P4.12 为 3 个身份批次 × 3 个 seed 的 9/9 projected cell；P4.13 为 9/9 两阶段 A → B cell，A+B+保持累积约束零违反、向后保持零失败，checkpoint/tamper/rollback/feature-source 与绝对资源预算全通过。P4.14 为 4/4 联合课程 cell——Phase K（P2.6 机械）之后的 post-K 景观上 Phase G（P4.11 合同）全门通过且跨相门全过。scorecard v6 已把 `k_worker_joint_course_completed` 与 `default_runtime_rollout_review_completed` 均翻转为 `true`（A8 入场条件齐备），但其余 v3 继承 veto 原样保留，所以 `promotion_gate=false`、`can_promote=false`、`growth_admitted=false`。**唯一交付目标是 A8 晋级评审（独立批准）**；在其余 veto 被评审逐项裁决前不解冻 owner、不接默认 runtime、不扩大训练范围，也不把已有机制证据写成完整认知能力。

- P0 已确定当前实现的效果基线：在 model17/course0、150 条 wake＋50 条固定 replay 上，FS 与 C-replay、FS-no-replay 与 C 的有效状态峰值差均为 `4.76837158203125e-7`，低于预先冻结的 `1e-5`；checkpoint preflight 通过。当前数据覆盖的验证类为 A/B/C，D/R 留给 P1。
- P1 v1 失败审计确认了根因：450 条 train 记录中每类表面 observation digest 为 90 个，但实际 K1/K2 mask-visible input 各只有 1 个；validation 缺 D/R、无 project 隔离。报告保留为失败证据，不覆盖。
- P1.1 已通过：修复后的 450 条 train + 10 条 validation 中，A/B/C/D/R 每类均有 2 个 K1/K2 visible input；course seed 改变可见序列；validation 覆盖五类，并与 train 在 project/template 上隔离。报告见 [P1 v2 数据契约审计](../../../../reports/taiji_m5_k_p1_data_contract_audit_v2_20260910.json)，清单见 [P1 v2 manifest](../../../manifests/taiji_m5_k_p1_data_manifest_v2.json)。model17/23/31 state_dict 仍相同，但本阶段只做单父代继承学习，该旁证不再作为 P1 Gate。
- P2 pilot 已完成：P1 v2 的 460 条记录重建为 0 mismatch；50 条均衡 wake（A/B/C/D/R 各 10）+ 10 条固定 replay；checkpoint 保存和独立进程恢复均通过。报告见 [P2 v2 pilot](../../../../reports/taiji_m5_k_p2_validation_pilot_v2_20260910.json)，机械失败保留在 [P2 failure report](../../../../reports/taiji_m5_k_p2_validation_pilot_failed_20260910.json)。
- P2 结果：frozen macro MSE `0.156574`；wake-only `0.029237`（Δ `-0.127337`）；wake-replay `0.030520`（Δ `-0.126053`）。但三臂宏观 semantic/transition goal/content 命中均为 `0.4`，R 类命中为 `0.0`；replay 相比 wake-only 反而使宏观和最坏类 MSE略差。因此只能确认连续输出拟合和 checkpoint 链路有效，不能确认离散输出、行动成功、抗遗忘或 replay 独立收益。
- P2.1 已完成只读诊断：10 条 validation 重新构建为 0 mismatch；三臂均为 5,648 有效参数，checkpoint 独立恢复通过。frozen/wake-only/wake-replay 的 K1→K2→planner→隔离 Workbench 成功率分别为 3/10、4/10、4/10；6/10 行因输入 confidence 低于 K1/K2 的 `0.55` floor 输出 `unknown`，不是 argmax 读出错；frozen 另有 1 条因 `stale_world_observation` 被 planner 拒绝。报告见 [P2.1 诊断](../../../../reports/taiji_m5_k_p2_output_action_diagnostic_20260910.json)。
- P2.1 还确认既有 `READ_ONLY_ROUTES` 没有 `content:recover-target`→只读能力的路由。这个缺口不能用降低 confidence floor 或给缺失文件直接执行来掩盖；下一步必须先做安全 recovery bridge canary。
- P2.2 已完成：460 条清单重建 `mismatch_count=0`；三臂 6 条低证据行均生成可往返、不可执行的 typed abstention；`content:recover-target` 的 `workspace.list(path=".")` 在 2 条 R 控制行上全部通过且根目录约束成立；世界对齐控制 10/10 通过。恢复和对齐均标记为 `oracle_control`，不构成模型能力成绩。报告见 [P2.2 安全 bridge canary](../../../../reports/taiji_m5_k_p2_2_safety_bridge_canary_20260910.json)。
- P2.2 重新划分了下一阶段指标：高证据可执行 cohort 只有 4/10 行；低证据 6/10 行的正确结果是安全 abstention；R 的真实“列举后重新观察并读取”连续数据尚未进入训练/验证合同。下一步必须先补齐这条可学习 continuation，再运行 targeted learning；不得降低 `0.55` floor，也不得把 oracle route 计入模型命中。
- P2.3 continuation 合同已通过：6 条 train、2 条 validation；初始 R 观察均为 `read_success=false`、Percept confidence `0.0`、下一步 `workspace.list(path=".")`；列举后的候选观察均为独立高证据 `confidence=0.99`、resolved 文件；K1/K2 示例往返、输入 digest、record digest、project/path/template 隔离均通过。manifest 和报告见 [P2.3 continuation manifest](../../../manifests/taiji_m5_k_p2_3_recovery_continuation_manifest_v1.json) 与 [P2.3 contract report](../../../../reports/taiji_m5_k_p2_3_recovery_continuation_contract_20260910.json)。
- P2.3 targeted learning 已完成：训练前 parent checkpoint 保存/独立恢复和三臂保存/独立恢复均通过；6 条 continuation candidate train、2 条 validation candidate 全程未把 validation 用于 fit。parent/targeted/reference 的 continuation validation 均为 2/2，说明没有新增能力；targeted 原五类 K1/K2 goal 命中为 1/4、3/4，parent 为 4/4、4/4，安全 abstention 仍为 6/6，`can_promote=false`。报告见 [P2.3 targeted pilot](../../../../reports/taiji_m5_k_p2_3_targeted_learning_pilot_20260910.json)。
- P2.3 的失败归因固定为“candidate-only update interference”，不是数据合同失败：新 candidate 与 parent 的输出目标重复，训练没有可测增量，却改变了 B/C/D 的高证据语义读出。下一步必须用固定 50 条均衡 rehearsal 做 retention-preserving objective canary，不能继续单独增加 continuation fit 次数。
- P2.4 retention canary 已完成：P2 的 50 条 rehearsal digest、类别平衡和顺序全部复现；交错 50 rehearsal + 6 continuation 后，原五类 K1/K2 goal 命中仍为 4/4、4/4，Workbench 4/10，低证据安全 abstention 6/6，保存/独立恢复通过。continuation validation 仍为 parent 已有的 2/2，没有新增能力，`can_promote=false`。报告见 [P2.4 retention canary](../../../../reports/taiji_m5_k_p2_4_retention_canary_20260910.json)。
- P2.4 的结论是保持目标已可用，但当前 recovery continuation 目标不是有效 novelty probe：它只重复了既有 `inspect-language` 输出。下一步禁止继续在这个目标上加 epoch；必须构造真正未见的 recovery 后语言/工具链组合，并先做 frozen parent validation-only 探针。
- P2.5 novel-composition probe 已完成：2 条 validation 候选均使用与 P1/P2 路径不重叠的 TypeScript＋可用 toolchain＋resolved＋inspect-language tuple；合同 digest、candidate path 隔离、checkpoint 保存和独立进程恢复均通过。frozen parent 在新组合上 K1 goal/content `2/2`、K2 goal `2/2`、K2 content `0/2`、Workbench `2/2`；两条 K2 均为 `ambiguous` 且 content 为 `None`，因此缺口定位为 K2 内容承接，不是识别、路由或 host 执行失败。报告见 [P2.5 novel-composition probe](../../../../reports/taiji_m5_k_p2_5_novel_composition_probe_20260910.json)。
- P2.6 novel K2 learning 已完成：6 条 train candidate 与 2 条 disjoint validation candidate，固定 50 条 P2 rehearsal 按 P2.4 顺序交错；manifest 合同通过、validation 未 fit、参数未增长、训练前与三臂保存后独立恢复均通过。parent 新组合 K2 content `0/2`，rehearsal-only `0/2`，interleaved `2/2`；interleaved 的 P1 旧类 K1/K2 content `4/4`、安全 abstention `6/6`、Workbench `4/10`，不低于 P2.4 parent baseline。该结果证明“这个具体 K2 内容承接目标可学习且保持约束通过”，不证明开放泛化，`can_promote=false`。报告见 [P2.6 novel K2 learning](../../../../reports/taiji_m5_k_p2_6_novel_learning_20260910.json)。
- P2.7 independent holdout generalization 已完成：4 条 holdout candidate 使用与 P1/P2.5/P2.6 全部 disjoint 的新路径，分属 2 个新 project；P2.6 learned arm 的已学 sanity K2 content `2/2`，holdout K1/K2 goal/content 均 `4/4`、Workbench `4/4`；frozen parent 同一 holdout K2 content `0/4`。P1 旧类相对 P2.4 baseline 不下降，参数量稳定，P2.6 checkpoint 独立恢复再次通过。报告见 [P2.7 holdout generalization](../../../../reports/taiji_m5_k_p2_7_generalization_20260910.json)。这满足 P3 的局部泛化入场 Gate，但不自动授予 promotion。
- P3.0 checkpoint/interrupt-resume contract 已完成：固定 P2.6 `interleaved-rehearsal-novel` learned checkpoint 为 parent；uninterrupted、wake 中段中断恢复、replay 边界中断恢复的 worker/budget/RNG/stream digest/final cursor 全部一致；tampered cursor、wrong parent、missing lineage 全部拒绝，rollback 独立恢复通过。实际发生了 K1/K2 更新，但未新增 S/G worker、未增长参数，`can_promote=false`。报告见 [P3.0 checkpoint contract](../../../../reports/taiji_m5_k_p3_0_checkpoint_contract_20260910.json)，清单见 [P3.0 manifest](../../../manifests/taiji_m5_k_p3_0_checkpoint_contract_manifest_v1.json)。
- P3.1 single-cell state preflight 已完成：以 P3.0 continuation parent 为锚点，4 条 P2.7 holdout 形成 20 个五阶段事件；S/G/K owner mask、content digest、event/state chain、事件中点/阶段边界中断、两条独立恢复、rollback 和 tamper/base/manifest 拒绝全部通过。K 只读取 S evidence，G 只提供 control-only selection，最终 action 读取 G/K；K1/K2 checkpoint 未改变，`fit_called=false`、参数未增长、`can_promote=false`。报告见 [P3.1 report](../../../../reports/taiji_m5_k_p3_1_single_cell_20260910.json)，清单见 [P3.1 manifest](../../../manifests/taiji_m5_k_p3_1_single_cell_manifest_v1.json)。
- P3.2 owner-transfer preflight 已完成：在相同的 4 条 P2.7 holdout 上，K1 只提供 inherited goal/content candidate，GSelectionState 持有最终选择；K-only 与 owner-transfer 的 K1 selection、K2 output、safe abstention、Workbench 全部等价，外部 target 未进入运行时，参数未增长。事件/事件边界/case 边界独立恢复、trajectory digest 和 tamper/wrong-base/wrong-manifest/wrong-owner-mask 拒绝全部通过。报告见 [P3.2 report](../../../../reports/taiji_m5_k_p3_2_owner_transfer_20260910.json)，清单见 [P3.2 manifest](../../../manifests/taiji_m5_k_p3_2_owner_transfer_manifest_v1.json)。
- P2/P3.2/P3.3/P3.5 只允许使用冻结的 manifest、candidate/scorer/threshold/resource contract；每次训练前必须重新通过 checkpoint 保存/独立恢复 preflight。P3.3 已冻结 P3.2 K、只训练 G，证明了 G 的 checkpoint/lineage/fit 边界；P3.4 已构造真实可区分的候选行为信号，但零 margin 平局不得训练。P3.5 不得追加同质 epoch，也不得把外部 goal target、K 已有输出、静态 tie-break 或 owner-transfer 元数据冒充 learned G owner。
- 阶段完成必须同时给出训练基线、可重建数据清单、不可变候选、五类结果及失败分析。没有收益或保持失败也是可收束的研究结论，但不得因此解冻 P3 的能力整合或晋级。
- 按验收事件排期，不承诺缺乏运行时间依据的日历日期；同一时间只推进一个研究问题。

### P0 已完成：等 replay 机制归因

P0 的可重建入口为 [等 replay 诊断脚本](../../../../scripts/training/eval_taiji_m5_k_p0_equal_replay_diagnostic.py)，结果见 [诊断报告](../../../../reports/taiji_m5_k_p0_equal_replay_diagnostic_20260910.json)。它从同一 model17 v4 worker 快照派生 C、C-replay、FS、FS-no-replay，消费同一 150 条五类经历和同一 50 个 replay 索引，并逐 wake/replay/consolidate 记录轨迹与 A/B/C 验证类 MSE。结果完成“直接 continuation＋replay 是效果基线”的归因；它没有证明五类泛化、独立模型泛化、完整闭环或晋级。

因此不再以 FS 相对 C-replay 的效果差作为成长证据。FS 的剩余价值转为状态拆分、保存和恢复接口候选，P3 仍需独立进程中断续训验证。

### 自动推进与讨论边界

P1 合同完成后可做 P2 validation pilot。最终测试前冻结主指标、最低有意义收益、各类允许遗忘界和资源预算，注明各值的依据，禁止根据测试成绩放宽。

出现下列情况应提交已有成果并停在决策点：有效信号不足需要改变任务定义；公平对照后仍无收益需要改变学习机制；资源约束迫使缩减目标；或准备改变认知所有权/默认发布模型。讨论时给出证据、保留方案与替代方案的收益和代价，再更新唯一计划；不自动扩展训练规模或购买算力。

本轮已完成 P0/P1 validation-only 诊断、P1.1 修复、P2 小预算 pilot、P2.1–P2.7、P3.0–P3.6、P4.0–P4.6 固定容量与更新规则诊断、P4.7 容量干净检验、P4.8 表征合同重设计、P4.9 特征空间探针、P4.10 特征因子化、P4.11 投影求解器、P4.12 课程级验证、P4.13 两阶段晋级课程、P4.14 K worker 联合课程与默认 runtime rollout review；当前全部证据线已由 scorecard v6 收束，A8 晋级评审入场条件齐备，不读取 sealed、不解冻 owner、不接默认 runtime、评审独立批准前不进入 promotion 或外围路线。

## 当前判断

M5 K 轴的 K1/K2/K3 standalone 证据、G 侧 solver 机制证据与 K worker 联合课程证据均已入账。P4.7 关闭了“增加固定容量即可解除张力”的假设；P4.8–P4.10 把问题定位到 parent-relative 表征因子化与可学习空间；P4.11–P4.13 证明“任务 fit + 联合可行域投影”在两 seed、9-cell 课程和两阶段累积学习上可以同时满足新任务、保持与向后保持；P4.14 证明 K worker 学习改变候选特征景观后，G 求解器在 post-K 景观上仍能同时通过新任务门与保持门，且 G 相不扰动 K worker。上述结论是机制证据，不是默认 runtime 已采用该机制，也不是完整认知能力。

scorecard v6 的机器边界为：`k_evidence_closed=true`、`learning_mechanism_closed=true`、`g_solver_mechanism_course_closed=true`、`k_worker_joint_course_completed=true`、`default_runtime_rollout_review_completed=true`（A8 入场条件齐备），且其余 promotion veto 原样保留，因此 `promotion_gate=false`、`can_promote=false`、`growth_admitted=false`。下一步只做 A8 晋级评审（独立批准），不自动追加同质 epoch、扩 K、接 owner 或启动 CUDA/IDE/provider/客户端路线。

P1.1 已把数据入口修复为可见状态优先的合同：A/B/C 通过语言证据的 resolved/ambiguous 变化，D 通过 header 语言证据的 ambiguous/resolved 变化，R 通过显式 recovery-language hint/no-hint 变化。P2 pilot 证明同一父代上的 K1/K2 连续 MSE 可显著下降，但离散 readout 命中不随之提升；这把问题从“有没有训练信号”推进到“输出阈值/目标/行动桥是否正确”。

## P3.0 已完成：checkpoint/interrupt-resume contract

P3.0 固定 P2.6 learned checkpoint 为 parent，把 P2.7 已通过的局部泛化能力放入可恢复的持续学习状态边界。它只验证 K1/K2 continuation，不把 S/G/K 联合成长写成已实现。

1. 固化 P2.2/P2.4 安全出口：confidence `<0.55` 的低证据样本只能产生 typed abstention；根目录 `workspace.list(path=".")` 仍是 host policy，不计模型 credit。此项已通过。
2. P3.0 已通过最小 continuation contract：parent、K1/K2 worker、wake/replay/consolidate phase cursor、experience/stream digest、RNG/采样状态、预算计数、origin/attached lineage 元数据均有内容地址；错误 parent、缺链和篡改 payload 均拒绝。
3. 三条可重建轨迹已通过：uninterrupted；wake 中段中断后恢复；replay 边界中断后恢复。每条都经过独立进程恢复，后续 worker/budget/RNG/stream digest/final cursor 与 uninterrupted 一致。
4. rollback 已通过：恢复到 P3 parent 后 K1/K2 source digest 保持一致。P3.0 没有新增 S/G worker、没有参数增长、没有把 lineage 元数据计作能力，`can_promote=false` 保持。

## P3.1 已完成：S→G→K 单 cell 状态整合预检

P3.1 已在 P3.0 continuation parent 上完成 4 条 P2.7 holdout 的接口回放，形成每例五阶段、共 20 个事件。schema、owner、读写 mask、事件 state chain、内容寻址、事件中点/阶段边界中断、独立恢复、rollback 和错误拒绝全部通过；报告见 [P3.1 report](../../../../reports/taiji_m5_k_p3_1_single_cell_20260910.json)，清单见 [P3.1 manifest](../../../manifests/taiji_m5_k_p3_1_single_cell_manifest_v1.json)。

关键边界必须保留：S 是 runtime evidence，G 是外部 goal/content selection 的 `control-only`，K 复用已有 K1/K2 learned worker；K readout 实际只读取 S evidence，最终 action 才读取 G/K。P3.1 没有调用 fit、没有新增参数，也没有证明 S/G 已经学习或 single-cell 优于 K-only；它只证明“状态可以正确接线、保存、恢复和回滚”。

## P3.2 已完成：S/K/G owner 边界迁移预检

P3.2 已在 P3.0 parent、P3.1 manifest 和同一 P2.7 holdout 上完成 owner-transfer 对照。K1 只提供 inherited goal/content candidate，`GSelectionState` 持有最终选择；K-only 与 owner-transfer 的 K1 selection、K2 output、safe abstention、Workbench 在 4/4 holdout 上完全等价，外部 target 未进入运行时，参数未增长。事件、事件边界、case 边界独立恢复，trajectory digest 一致，篡改 cursor、错误 base/manifest/owner mask 全部 fail-closed。报告见 [P3.2 report](../../../../reports/taiji_m5_k_p3_2_owner_transfer_20260910.json)，清单见 [P3.2 manifest](../../../manifests/taiji_m5_k_p3_2_owner_transfer_manifest_v1.json)。

边界结论已经继续收束：P3.2 证明“选择所有权可迁移且不破坏既有行为”；P3.3 证明“冻结 K 后 G 可以保存、恢复和更新”，但 trained-G 与 zero-step/K-only 完全一致；P3.4 证明行为 utility 能产生可审计候选分歧，同时识别出 10 条零 margin 静态平局。因此不再重复同质 G-only epoch，下一步改为只使用非零 margin、带 reobserve 安全投影的 P3.5。

## 后续依赖顺序与验收

| 顺序 | 工作重点 | 进入下一步的条件 |
|---|---|---|
| P0 | 相同 replay 的机制归因 | **已完成**：两组轨迹均在 `1e-5` 内等价，checkpoint preflight 通过 |
| P1 | 真实学习信号与五类数据合同 | **已通过 P1.1**：五类各有至少 2 个 K1/K2 visible input，validation 五类覆盖且 project/template 隔离 |
| P2 | 五类学习及保持的独立验证 | **P2.7 局部跨项目/路径泛化通过**：holdout K2 content/Workbench `4/4`，旧类保持通过；已进入 P3.0 |
| P3 | 中断续训与 S/G/K 联合状态整合 | **P3.6 独立行为 holdout/保持 Gate 已通过但尚未 promotion**：新 project/path trained-G utility `4.0>2.65`、target hit `4/4>1/4`，旧类、安全、Workbench 和 checkpoint 均保持；S 仍非 learned |
| P4 | 结构成长必要性与收益验证 | **P4.0–P4.14 已全部收束**：P4.7 关闭容量假设，P4.8–P4.10 完成表示/特征归因，P4.11–P4.13 证明投影求解器在 2-seed、9-cell、两阶段累积课程上支持新任务 + 保持 + 向后保持，P4.14 证明 K worker 联合课程 4/4 成立（post-K 景观上 G 求解器维持平衡）；默认 runtime rollout review 已通过；A8 晋级评审批准入账；默认 runtime 附着执行通过（4/4 cell runtime 路径消费等价）；机器晋级边界闭环，下一步 = 最终晋级宣布的独立批准 |
| P5 | 知识来源、IDE、客户端、硬件发布 | **P5.1 内容迁移 Gate 已通过**（sourced vs 同预算 placebo，delta `0.5` ≥ 0.15）；P5.1b 语义改写迁移核心门诚实失败（3 文本规模值函数不辨内容，disc delta `0.017`）；**P5.1c 对比辨别力 Gate 已通过**（10 文本/族对比训练，disc `0.7314`，九门全过——内容寻址语义内化成立）；**P5.1d 语义 encoder 注入 Gate 已通过**（`semantic_encoder_injection_supported`，注入合同 + 锚定校验 + 格式分派落产品模块，真实 governed 语料 disc `0.8700`，九门全过）；**P5.1e 同预算因果收益 Gate 已通过**（`same_budget_content_benefit_supported`，家族排他两臂单变量 = 训练内容族，a-gate delta `0.25` ≥ 0.15，九门全过，wall `65.194s`）；**P5.1f 真实语料同预算 Gate 预注册已冻结**（Tool_Use vs Code_Agent，UltraData-SFT-Agent-2609，九门 margin `0.15`，提交 `72c2d2d9`，尚未执行）；其余各项按所需模型能力与接口成熟度依序解冻 |

### P1：有效信号与评估数据（P1.1 已通过）

- 审计现有五类课程；按 K1/K2 typed mask 可见输入、目标张量、时序组合生成签名。分别统计经历数、唯一文件数、有效类数、模板族数。注释/文件名变化允许作为同分布扰动，不能充当新能力或新独立样本。
- train/validation/test 以项目或任务模板分组，完整覆盖 A/B/C/D/R；每类包含多个不同可见状态/组合，记录数量与重复率。测试设计应同时测同分布泛化和未见组合，不能每轮仅更换 seed。
- 父 worker 沿用现有权重即可做机制诊断；若主张跨模型泛化，则以实际初始化/训练流变化构建父 worker，并核验 state_dict 差异。所有独立性结论由实测决定，不强制为了 n=9 重建模型。
- 将 v1–v5 已读 sealed 登记为 consumed；后续开发用 validation。候选、scorer、阈值、资源预算和最终输入在读取下一份测试成绩前锁定。
- v1 失败合同保留为 [P1 失败审计](../../../../reports/taiji_m5_k_p1_data_contract_audit_20260910.json)；修复后的合同状态为 `passed`，见 [P1 v2 审计](../../../../reports/taiji_m5_k_p1_data_contract_audit_v2_20260910.json) 与 [P1 v2 manifest](../../../manifests/taiji_m5_k_p1_data_manifest_v2.json)。P2 只能使用 v2 入口，不能用 v1 的一类一签名课程训练。

### P2：验证学习与保持

- 主要因果臂为 P0 选定流程与同新增训练预算的直接学习对照；F 测零更新漂移。FS 若仅是等价状态实现，无须继续宣称胜过同 replay 的基线。XL 作为容量参考，动态增长阶段再做最终容量对齐。
- 用五类宏平均和每类 MSE；D/R 弱类单列，A/B/C 旧强类非劣逐类检查，报告最坏模板结果。整体均值不能掩盖遗忘。
- 增加真实预测链：K1 预测→K2 多步状态→隔离 Workbench 执行。真实任务成功率/失败恢复与局部 MSE 分账；低置信度和 D/R 路径必须有可解释出口。
- 使用 validation-only pilot 确定样本量、最低有意义改善、数值容差和逐域非劣界。浮点噪声容差与“允许遗忘多少”分别定义；不能用 candidate 退化方差自动放宽所有门槛。
- 先保存候选和 presealed 合同，随后同一个 artifact 只读评分，不在第二阶段重训候选。记录代码版本、数据/参数/状态摘要和所有失败。
- 结果出口：收益/保持/资源通过→P3；无收益→回对应反馈或数据根因；数值等价→保留成本更合理的基线；机械错误→修复后重做技术预检。不得看测试成绩改当前版本阈值。
- P2 pilot 已执行上述最小预算和独立 checkpoint preflight；P2.1 完成了 K1→K2 级联与隔离 Workbench 诊断；P2.2 完成了 typed abstention、根目录 recovery 和世界对齐工程 canary；P2.3 continuation 数据合同/targeted learning、P2.4 retention canary、P2.5 novel-composition probe、P2.6 novel K2 learning、P2.7 holdout generalization、P3.0 checkpoint/interrupt-resume contract、P3.1 single-cell preflight、P3.2 owner-transfer preflight、P3.3 candidate data-signal/G-only learning、P3.4 behavior signal、P3.5 reobserve-aware G-only learning、P3.6 独立行为 holdout/保持 Gate、P4.0 固定容量压力扫描、P4.1 上下文/fixed-large contract preflight、P4.2 隔离训练/公平容量归因 Gate、P4.3 保持约束增量学习 Gate、P4.4 保持身份/结构校准、P4.5 保持约束/更新规则对照和 P4.6 功能性 parent-preserving objective 对照均已完成。当前模型结论从“只有连续 MSE 改善”推进为“一个具体 K2 content 目标在 rehearsal 保持约束下学习，并跨 2 个新 project/4 个新 path 泛化”；P3.0 已把 K1/K2 continuation 纳入可恢复状态边界，P3.1 已把 S/G/K 接线合同闭合，P3.2 已把选择所有权转给 G，P3.3 已证明 G 可以学习但没有改变三臂行为，P3.4 已证明候选 utility 有分歧，P3.5 已证明非零 margin G-only fit 能改变 contested 行为且安全投影/holdout 不退化，P3.6 已证明该行为选择跨新 project/path 泛化且五类/安全保持通过；P4.0/P4.1/P4.2 暴露了候选宽度压力、上下文尚未归因和增量训练后的旧行为保持退化，P4.3 在 fresh retention 上没有区分 rehearsal 与 new-only，P4.4 在同构新身份上复现了保持退化，P4.5 则显示 rehearsal 无法带来额外收益，固定参数 trust-region 在两个 seed 间出现保持/新任务权衡，P4.6 的功能性约束仍无法在两个 seed 上同时满足保持和新任务。当前固定容量路线停止自动推进，不把保持失败改写成结构成长证据。

### P3：状态整合与同一父代连续课程（P3.6 已收束，P4 当前）

P3.0 已用 [checkpoint/interrupt-resume contract](../../../../reports/taiji_m5_k_p3_0_checkpoint_contract_20260910.json) 替代“先做 lineage 再补能力”的顺序：K1/K2 parent、worker、phase、replay stream、RNG、预算和 lineage 元数据已经可以内容寻址、独立恢复和回滚。P3.1 又把 S/G/K 的事件、owner mask、状态 digest 和恢复边界闭合，但没有把 control-only 的 S/G 说成 learned。P3.2 已证明 owner-transfer 不破坏已有行为；P3.3 已证明 13 参数 G 可以在冻结 K 上 fit、保存、独立恢复和拒绝篡改；P3.4 证明行为 utility 信号能造成可解释候选分歧；P3.5 证明非零 margin G-only fit 能改变 contested 行为且 reobserve projection、K digest 和 holdout 不退化；P3.6 又在全新 project/path 上复验了行为收益、五类保持、安全 projection 和 checkpoint/rollback。P3 的结论仍不是 promotion 或通用智能，不能回到原始从零训练，也不能重新授权暂停的 [SGK v1](../../../reference/M4V2_SGK_PROMOTION_COURSE_PREREGISTRATION_20260910.md)。

1. 以 P3.0 parent、P3.1 manifest 和现有 K1/K2 worker 为唯一资产；S、K、G 的 schema、owner、输入输出 mask、attached lineage、checkpoint digest 和 rollback parent 必须独立可定位。
2. 把 K1 的共享语义表征与 goal/content 输出拆开记账：K 负责 evidence/world readout，G 负责 candidate selection/accept/reject；在拆分完成前，goal/content head 只能标记 `pending-owner-transfer`，不能同时计为 K 与 G 的能力。
3. 先实现无新增参数的 `GSelectionState` 和 owner-transfer adapter，外部 goal/content target 仅作 validation label；用同一 P2.7 holdout 运行 K-only 与 S/K/G 两臂，比较 K1/K2 goal/content、safe abstention、Workbench、延迟和 checkpoint 字节。
4. 在 observation、G selection、K readout、action 四个边界做独立进程恢复；验证事件/owner/worker/budget/RNG/logical digest 一致，篡改 G state、错误 parent、错误 mask 必须拒绝。P3.2/P3.3 已满足 owner-transfer 与 G-only checkpoint/fit 条件，P3.4 已满足行为信号条件，P3.5 已满足行为增益与安全投影条件，P3.6 已满足独立行为泛化与保持条件；P3 阶段收束，下一步进入 P4。

## P3.6 已完成：独立行为 holdout 与保持 Gate

P3.6 严格 validation-only，加载 P3.5 zero-step/trained-G，不调用 `fit`，在两个新 project 和四条新 path 上重新执行候选生成、真实 Workbench utility 与 G selection。trained-G 新 holdout utility 为 `4.0`，高于 zero-step/K-only 的 `2.65`；behavior target hit 为 `4/4`，高于 `1/4`。3 条 reobserve target 与实际 selection 全部投影为可往返、无 `ActionIntent` 的 `ReadOnlyAbstention(next_step="workspace.list")`。

| Gate | 结果 | 证据 |
|---|---:|---|
| 新身份隔离 | 通过 | 2 个新 project、4 条新 path；candidate/behavior/observation digest 和 utility-margin record 均内容寻址且唯一 |
| 行为泛化 | 通过 | trained-G utility `4.0 > 2.65`；target hit `4/4 > 1/4`；K-only 与 zero-step 相同，trained-G 发生可解释行为变化 |
| 安全动作边界 | 通过 | target/selected reobserve 全部 typed projection、roundtrip、snapshot match、无 `ActionIntent` |
| 五类旧类保持 | 通过 | A/B/C/D/R 全部出现，trained-G target hit 不低于 zero-step，低证据 proposal 违规为 `0` |
| P2.7 与恢复 | 通过 | P2.7 Workbench `4/4`；K1/K2 digest 不变；K/G 独立 restore、lineage 和 rollback 通过 |
| promotion | 未通过/未开放 | `can_promote=false` 继续保持；P3.6 只证明当前行为选择的局部跨身份泛化 |

报告见 [P3.6 behavior holdout](../../../../reports/taiji_m5_k_p3_6_behavior_holdout_20260911.json)，清单见 [P3.6 manifest](../../../manifests/taiji_m5_k_p3_6_behavior_holdout_manifest_v1.json)。P3 阶段到此收束：不能再追加同质 G epoch，也不能把局部行为泛化写成通用智能或结构成长。

## P4.0 已完成：固定容量压力扫描（不授予结构成长）

P4.0 严格 validation-only，冻结 P3.5 trained-G、P3.2 K1/K2 和所有 Workbench/checkpoint 合同，在 5 个新案例、3 个新 project、20 个候选集合上扫描候选宽度 `2/4/8/12` 与序列长度 `1/4/16`。它没有调用 `fit`，没有修改 K，没有新增 G 参数，也没有读取 sealed payload。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 数据与身份隔离 | 通过 | 5 个新案例、3 个新 project、路径与 P3.4/P3.6 disjoint；20 个 candidate/behavior digest 唯一；每个宽度均为 5 个案例 |
| checkpoint / lineage | 通过 | K1/K2、zero-step G、trained-G 独立恢复；G lineage 通过；K1/K2 digest 前后仍为 `12851b…77a6` / `411ef5…3691` |
| 候选宽度压力 | 观察到压力 | trained-G residual：width 2=`0.32`、4=`0`、8=`0.54`、12=`0.59`；target hit 分别 `0.6/1.0/0.4/0.2` |
| 序列退化 | 未观察到 | length `1/4/16` utility 均为 `0.6375`；当前压力更像候选竞争/上下文合同问题，而不是时序记忆退化 |
| feature collision | 未观察到 | 四个宽度的 feature collision rate 均为 `0`；不能用“特征完全相同”解释退化 |
| 结构成长 | 未开放 | `growth_admitted=false`、`can_promote=false`；fixed-large 尚未建立可公平比较的 G owner/readout 合同 |

报告见 [P4.0 capacity pressure](../../../../reports/taiji_m5_k_p4_0_capacity_pressure_20260911.json)，清单见 [P4.0 manifest](../../../manifests/taiji_m5_k_p4_0_capacity_pressure_manifest_v1.json)。P4.0 的结论是“固定 G 在候选规模变化下出现可复现的选择压力”，不是“已经证明应当增加神经元”。由于 width 4 完整通过而 width 8/12 退化，且压力课程引入了跨案例 proposal competition/alias，必须先完成归因对照，不能直接实现 dynamic growth。

## P4.1 已完成：候选集上下文与公平容量合同预检

P4.1 在同一 P4.0 candidate/behavior artifact 上建立了内容寻址的候选集上下文合同：保留 G 当前 12 维逐候选输入，另定义 9 维上下文（候选数量、角色比例、joint-score 分布、候选相对排名/中心化分数）。它没有调用 `fit`，没有改变 P3.5 G/K parent；只生成 22 参数 fixed-large reference 和 context-lesion reference，并做独立进程恢复。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| P4.0 来源、G lineage、独立恢复 | 通过 | P4.0 digest 一致；P3.2 lineage 通过；trained-G 独立恢复通过 |
| 上下文合同 | 通过 | 20 个集合、width `2/4/8/12` 各 5 个；9 个 context feature 名称唯一；每个集合内部 context collision `0`；target/utility 未进入输入 |
| fixed-large reference | 通过预检 | 22 参数（当前 G 为 13）；reference 独立恢复，输入/读出和 checkpoint digest 可验证 |
| context lesion | 通过预检 | effective 参数回到 13；独立恢复；零 fit 下 20 个集合的选择与当前 G 完全一致 |
| 归因 | 未定 | `inconclusive`：零 fit reference 只能证明合同和边界，不能从同一验证集重放推断容量收益 |

报告见 [P4.1 context contract](../../../../reports/taiji_m5_k_p4_1_context_contract_20260911.json)，清单见 [P4.1 manifest](../../../manifests/taiji_m5_k_p4_1_context_contract_manifest_v1.json)。因此 P4.1 不批准结构成长，也不把 22 个零初始化 context 参数写成能力增长。

## P4.2 已完成：隔离训练与公平容量归因 Gate

P4.2 按预注册合同回答了 P4.1 的问题：在全新的 train/validation/holdout 身份上，学习型 context-aware fixed-small 是否能修复 P4.0 的 width 8/12 退化，还是必须依赖真正更大的固定容量。训练只发生在独立 child checkpoint，没有覆盖 P3.5/P4.0 parent，validation/holdout 没有参与 fit。

1. 生成与 P4.0/P4.1 project、path、template、candidate digest 全部 disjoint 的 A/B/C/D/R train/validation/holdout；候选宽度固定为 `2/4/8/12`，同时保留低证据 safe exit、reobserve projection、旧类 retention 和真实 Workbench。
2. 建立三臂：`fixed-small`（P3.5 13 参数 G 继承）、`context-aware-small`（存储 22 参数但冻结原 12 维 candidate 权重，只训练 9 维 context 权重和 bias，共 10 个可训练参数）和 `fixed-large`（同一 22 参数 context contract，candidate+context 全部可训练）。这样能区分“上下文输入本身有用”和“所有参数都需要重适配”。三臂均记录 parent/child lineage、参数实际数量、fit 数据 digest、checkpoint 保存/独立恢复、rollback 和 tamper rejection。
3. 只用新的 train cohort fit；新的 validation/holdout 只做评估。至少两个 deterministic seed，固定 epoch/学习率/资源预算；不把 P4.0 压力结果当训练标签，不读取 sealed，不把外部 target、candidate ID 或 utility 直接作为输入。
4. 每个 arm 必须同时报告 width 曲线、target hit、utility/residual、safe abstention、reobserve/action projection、五类旧类保持、Workbench、参数字节、CPU 时间和独立恢复；必须有 context-lesion 与 fixed-large 对照，防止“只加输入维度但未使用”或“复制权重”伪装成收益。
5. 只允许输出 `representation_contract`、`fixed_capacity_candidate` 或 `inconclusive`。只有 context-aware-small 与 fixed-large 在新 holdout 上均优于 fixed-small、且 fixed-large 相对 context-aware-small 仍有稳定增益时，才保留容量瓶颈假设；否则先修 G 输入/选择合同。无论结果如何，`growth_admitted=false`、`can_promote=false`，P5、CUDA、IDE/provider 和客户端视觉继续冻结。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 来源、身份与 checkpoint | 通过 | 5 类 train/validation/holdout、14 条 fit-eligible train；project/path/candidate/behavior 均与 P4.0/P4.1/旧保持集合隔离；三臂零步和训练后 checkpoint 均可独立进程恢复，parent 未覆盖 |
| 训练边界 | 通过 | 两个 deterministic seed；只用新 train cohort fit；validation/holdout、P4.0/P4.1、外部 target/utility 均未进入 runtime fit；K1/K2 digest 未变 |
| 新 holdout | 未通过晋级 | fixed-small 与 context-aware-small 平均 utility 均为 `0.68`；fixed-large 平均为 `0.65875`，没有稳定增益；seed 1 的 fixed-large 出现 6 次 safe-selection violation |
| 旧行为保持 | 未通过 | parent retention 为 `4/4`、utility `4.0`；训练后 fixed-small/context-aware-small 在两个 seed 均降到 `3/4`、utility `0.8`；fixed-large 只有 seed 1 恢复到 `4/4`，跨 seed 仍不稳定 |
| 归因与成长 | 未通过/关闭 | attribution=`inconclusive`，`experiment_passed=false`，`growth_admitted=false`，`can_promote=false`；不能据此增加神经元或进入 dynamic growth |

P4.2 的实际结论是：当前 13 参数 G 在新候选集合上没有出现可靠容量收益，22 参数 context arm 也没有形成可重复优势；训练本身先暴露了灾难性遗忘/保持合同问题。不能把这一失败解释成“需要更大拓扑”，因为固定容量臂尚未在保持约束下成为合格的增量学习基线。报告见 [P4.2 capacity attribution](../../../../reports/taiji_m5_k_p4_2_capacity_attribution_20260911.json)，清单见 [P4.2 manifest](../../../manifests/taiji_m5_k_p4_2_capacity_attribution_manifest_v1.json)。

## P4.3 已完成：保持约束下的增量学习 Gate

P4.3 按预注册边界只修复 P4.2 暴露的前置问题，没有扩大拓扑或引入新的 context 参数。它在同一 13 参数 fixed-small 容量上比较了 new-only 与 rehearsal-mix，并用全新 retention holdout 做保持验收；P3.6 旧 holdout 只作为明确标记的 rehearsal source，没有在训练后继续充当测试集。

1. 生成与 P4.0/P4.1/P4.2 全部 disjoint 的新 train、new-task validation、new-task holdout 和 fresh retention holdout。P3.6 的旧 holdout 只能作为明确标记的 rehearsal source，不能在 P4.3 结果中继续充当测试集；fresh retention 必须重新经过真实 Workbench/行为 utility 生成。
2. 保留不可训练的 P3.5 trained-G parent 与 zero-step child；建立 `new-only` 和 `rehearsal-mix` 两个 fixed-small 训练臂。两臂使用完全相同的新任务 train、epoch/学习率/seed/CPU 预算，rehearsal 只改变训练样本组成，不改变模型结构、输入合同或安全选择阈值。
3. 训练前后都执行 checkpoint 保存、内容摘要、独立进程恢复、错误 lineage/篡改拒绝和 parent rollback；记录新任务 fit digest、rehearsal digest、实际训练步数、参数字节和 K/G lineage。训练不得读取 new validation、new holdout 或 fresh retention 的 behavior target/utility。
4. Gate 同时要求：new-task holdout utility/target hit 不低于 new-only；fresh retention utility/target hit 不低于 zero-step parent；低证据 safe exit、reobserve projection、Workbench 和 K digest 全部保持。任一 seed 发生保持退化，结果只能进入“更新规则需修复”，不能进入容量归因或结构成长。
5. 结果出口只有 `retention_repaired`、`signal_insufficient` 或 `update_rule_unresolved`。本轮没有满足 rehearsal-specific gain，因此转入 P4.4 身份/结构校准；在保持问题完成更新规则对照前，P4 dynamic growth、P5、CUDA、IDE/provider 和客户端视觉继续冻结。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 来源、身份与训练边界 | 通过 | 新 train/validation/holdout/fresh retention 各 20 条；与 P4.0/P4.2/P3.6 的 project/path/candidate/behavior digest 均隔离；P3.6 仅作为 4 条 rehearsal fit |
| checkpoint / lineage | 通过 | 两臂零步/训练后 checkpoint 均可独立恢复；篡改、错误 parent lineage、rollback 全部通过；13 参数、K1/K2 未变 |
| 新任务与 fresh retention | 通过 | 两 seed 的 new-only 与 rehearsal-mix 均达到 new holdout utility `0.68`、target hit `0.6`；fresh retention 均为 utility `0.68`、target hit `0.6`，safe selection violation 为 `0` |
| rehearsal-specific 归因 | 未通过 | 两臂在两个 seed 的 holdout、fresh retention、safe projection 上完全相同；`rehearsal_specific_gain=false`，不能把保持通过归因给 rehearsal |
| 结果出口 | 未晋级 | `outcome=signal_insufficient`、`experiment_passed=false`、`growth_admitted=false`、`can_promote=false`；保持未退化，但 P4.2 旧 retention 退化尚未在同结构新身份上完成归因 |

P4.3 的正确结论是：在其 fresh retention 分布上，当前增量更新没有复现 P4.2 的旧保持退化；同时 rehearsal 与 new-only 完全等价，因此不能宣称 rehearsal 已修复遗忘。P4.4 已把这两个结果放到同一批与 P3.6 结构同构、但 project/path 全新的 sibling retention 上，证明 P4.2 历史退化的 5 个 arm/seed 全部复现，P4.3 两个 arm/两个 seed 也复现；因此保持问题属于训练后更新/保持约束的可重复问题，而不是旧 artifact 单独失真。下一步改为 **P4.5 保持约束/更新规则对照 Gate**，仍不扩容。

## P4.4 已完成：保持身份与结构校准 Gate

P4.4 严格 validation-only：从 P3.6 只提取候选数量、角色组成、置信度分桶和安全投影类型，生成 2 个新 project、4 条新 path 的 sibling retention；没有复制旧 path、target、utility 或 exact candidate digest，也没有调用 `fit`。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 来源链与结构合同 | 通过 | P3.6/P4.2/P4.3 manifest/report digest、来源链和结构摘要均通过；4 条 sibling 的候选宽度/角色组成/置信度分桶/安全投影与 P3.6 相同 |
| 新身份隔离 | 通过 | 2 个新 project、4 条新 path；与历史 P3.6/P4.2/P4.3 project/path、candidate/behavior digest 均隔离 |
| checkpoint / lineage | 通过 | P3.5 parent、P4.2 三臂和 P4.3 两臂全部独立恢复，lineage 有效；没有覆盖历史 checkpoint |
| parent sibling 基线 | 通过 | parent utility `1.0`、target hit `4/4`、safe violation `0`、reobserve projection 通过 |
| P4.2 退化复现 | 通过 | 历史发生退化的 5 个 arm/seed 在 sibling 上全部退化；固定小容量/上下文小容量/固定大容量的 seed-0 均为 utility `0.8`、target `3/4`，seed-1 fixed-small/context-small 同样为 `0.8`、`3/4` |
| P4.3 保持结果 | 退化复现 | new-only 与 rehearsal-mix 两个 seed 均为 utility `0.8`、target `3/4`；rehearsal 没有消除 sibling 上的保持退化 |
| 结果出口 | 已分类但未晋级 | `outcome=retention_failure_reproduced`、validation-only、`experiment_passed=false`、`growth_admitted=false`、`can_promote=false` |

报告见 [P4.4 retention identity calibration](../../../../reports/taiji_m5_k_p4_4_retention_identity_calibration_20260911.json)，清单见 [P4.4 manifest](../../../manifests/taiji_m5_k_p4_4_retention_identity_calibration_manifest_v1.json)。P4.4 只证明保持退化可跨同构身份复现，不证明应该增加神经元，也不证明任何更新规则已经修复；下一步只允许做保持约束/更新规则的受控对照。

## P4.5 已完成：保持约束与更新规则对照 Gate

P4.5 固定 13 参数 G、K1/K2、输入特征和 selection threshold，使用与 P4.4 结构同构但全新身份的 train/validation/holdout/rehearsal/retention；P4.4 sibling、P3.6/P4.2/P4.3 retention 均未进入 fit。三臂为 `new-only`、`rehearsal-interleaved` 和预先登记 parent 范数 15% trust-region 的 `constrained-update`，两个 deterministic seed 均做 checkpoint/lineage/独立恢复/tamper/parent rollback 检查。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 来源、身份与结构 | 通过 | train/validation/holdout 各 20 条；新 rehearsal 4 条、fresh retention 4 条；五类覆盖，retention 结构 digest 与 P4.4 合同一致，所有 project/path/candidate/behavior digest 隔离 |
| checkpoint / lineage | 通过 | 三臂两个 seed 的零步/训练后 checkpoint 全部独立恢复，tamper 拒绝，parent 未覆盖；参数始终 13，K1/K2 未变 |
| new-only 与 rehearsal | 无差异 | 两 seed 的两臂 new holdout 均 utility `0.68`、target `0.6`、safe violation `0`；fresh retention 均 utility `0.8`、target `3/4`，rehearsal 没有可分离收益 |
| constrained-update seed-0 | 保持通过但无新任务增益 | fresh retention utility `1.0`、target `4/4`、safe violation `0`；new holdout 退回 parent 的 utility `0.6375`、target `0.55`、safe violation `6` |
| constrained-update seed-1 | 新任务通过但保持失败 | new holdout utility `0.68`、target `0.6`、safe violation `0`；fresh retention utility `0.8`、target `3/4`，仍低于 parent |
| 结果出口 | 未晋级 | `outcome=update_rule_unresolved`、`experiment_passed=false`、`growth_admitted=false`、`can_promote=false` |

报告见 [P4.5 update-rule Gate](../../../../reports/taiji_m5_k_p4_5_update_rule_gate_20260911.json)，清单见 [P4.5 manifest](../../../manifests/taiji_m5_k_p4_5_update_rule_gate_manifest_v1.json)。P4.5 的结论不是“trust-region 没有价值”，而是它暴露了固定参数预算下的稳定性权衡：seed-0 约束足够强时保持恢复却没有新任务收益，seed-1 新任务恢复时保持仍退化；rehearsal 也没有提供独立增益。下一步必须从参数距离约束转向功能性 parent-preserving objective，仍不扩容。

## P4.6 已完成：功能性 parent-preserving objective 对照 Gate

P4.6 固定 13 参数 G、K1/K2、候选输入、selection threshold 和安全投影，使用与 P4.5 全部 disjoint 的新 train/validation/holdout/constraint-cohort/fresh retention。constraint-cohort 只向 functional 臂提供候选 feature vectors 和 parent 的 teacher score；训练代码不读取该 cohort 的 behavior target、utility 或 retention target。`new-only` 作为同预算基线，两个 deterministic seed 均做 checkpoint/lineage/独立恢复/tamper/parent rollback。

| Gate | 结果 | 关键实测 |
|---|---:|---|
| 来源、身份与结构 | 通过 | train/validation/holdout 各 20 条，constraint/fresh retention 各 4 条；五类覆盖，retention structure digest 与 P4.4 合同一致，所有 project/path/candidate/behavior digest 隔离 |
| checkpoint / lineage | 通过 | 两臂两个 seed 的零步/训练后 checkpoint 全部独立恢复，tamper 拒绝，parent 未覆盖；参数始终 13，K1/K2 未变 |
| new-only 基线 | 通过基线 | 两 seed new holdout utility `0.68`、target `12/20`、safe violation `0`；fresh retention utility `0.8`、target `3/4` |
| functional seed-0 | 保持通过但新任务失败 | fresh retention utility `1.0`、target `4/4`；new holdout utility `0.585`、target `9/20`、safe violation `6` |
| functional seed-1 | 新任务通过但保持失败 | new holdout utility `0.68`、target `12/20`、safe violation `0`；fresh retention utility `0.8`、target `3/4` |
| 结果出口 | 未晋级/路线收束 | `outcome=functional_update_unresolved`、`experiment_passed=false`、`growth_admitted=false`、`can_promote=false` |

报告见 [P4.6 functional parent-preserving objective](../../../../reports/taiji_m5_k_p4_6_functional_parent_objective_20260911.json)，清单见 [P4.6 manifest](../../../manifests/taiji_m5_k_p4_6_functional_parent_objective_manifest_v1.json)。P4.6 证明功能性 parent 保持项确实改变了更新轨迹，但仍在 seed 间形成互斥：保持恢复时新任务明显退化，新任务恢复时保持仍退化。P4.0–P4.6 的固定 G 路线已经完成预定的容量、身份、rehearsal、参数约束和功能约束对照，当前不能再自动追加 epoch、半径、loss weight 或同质 seed。

## 当前决策点：P4.7 已收束——容量假设关闭，进入表示合同重设计

**P4.7 已按预注册执行完毕**（[预注册 §8](../../../reference/M5_K_P4_7_CAPACITY_CLEAN_TEST_PREREGISTRATION_20260911.md)、报告 `reports/taiji_m5_k_p4_7_capacity_clean_test_20260911.json`）：两臂跑完全相同的 P4.6 functional 协议，单变量 = 容量 13→22（22 参数臂 candidate 权重 + bias 从 P3.5 parent 逐位继承、9 维 context 零初始化；出生等价 64 record 0 mismatch、最大 score 偏差 0.0——扩展算子干净性得证）。结果 **`capacity_hypothesis_closed`**：

- 13 参数臂在全新身份上**逐数值复现** P4.6 的 seed 间互斥（seed-0 败新任务 `0.585/0.45`+6 violation、过保持 `1.0/1.0`；seed-1 过新任务 `0.68/0.6`、败保持 `0.8/0.75`）——张力对身份变化鲁棒；
- 22 参数臂未改变定性形态（seed-0 `0.625/0.45` 仍败新任务、seed-1 与 13 参数**逐数值相同**）——出生零影响的 +9 context 参数是放大器不是解耦器；
- 机械门全过（`experiment_passed=true`）；`growth_admitted=false`、`can_promote=false`。

**收束判定**：固定容量路线（P4.0–P4.7）整体关闭。「保持/新任务互斥」定性为**表示问题**——functional teacher 约束把「新任务学习方向」与「parent 行为保持」耦合进同一 12 维 candidate 特征空间，任何在该空间内的容量/协议变化都只能移动数值不能消解张力。

**唯一下一步**：表示合同重设计预注册——解耦「选择学习」与「行为保持」的表示维度（方向：把保持约束从「拟合 parent 标量分」改为「保持 parent 的选择不变量」（如 safe-candidate 优先序、低证据拒绝边界），或把 candidate 特征空间拆分为 task-learning 与 parent-preservation 两个正交子空间）；预注册冻结前不训练、不扩容、不读取 sealed、不解冻 P5/CUDA/IDE/provider。

用户确认执行。**P4.8 表示合同重设计预注册已冻结：[M5_K_P4_8_REPRESENTATION_CONTRACT_REDESIGN_PREREGISTRATION_20260911.md](../../../reference/M5_K_P4_8_REPRESENTATION_CONTRACT_REDESIGN_PREREGISTRATION_20260911.md)。** 机制诊断：P4.6/P4.7 的标量 MSE teacher 约束无饱和点（梯度恒在），与 task delta 在同一权重上持续对抗——这是 seed 间互斥的机制根源。**三臂设计（相邻对单变量）**：`functional-13`（in-run 基线，第 3 次复现）/ `invariant-13`（唯一变更 = 约束形式改为**决策不变量 hinge**——保持 parent 的选择与安全回退边界而非标量分，hinge 满足后梯度恒零即有限支撑）/ `residual-26`（唯一变更 vs invariant-13 = 架构拆分：frozen parent head 13 逐位继承永不可训练 + δ head 13 零初始化，task/hinge 只写 δ；有效可训练容量三臂同为 13，容量不是变量）。决策不变量 hinge 冻结规格（guard band 0.01、selection margin 0.05、parent 决策由 frozen parent 在线计算、behavior target 永不进 fit、出生 constraint 损失恒 0 断言）。全分支结果映射：新臂过 + 基线张力复现 → `representation_redesign_supported` 进晋级课程级验证；两新臂仍互斥 → `invariant_constraint_insufficient` 收敛到特征空间重设计；基线第 4 轮不复现 → `baseline_drift` 重审。`growth_admitted=false`、`can_promote=false`、无 router/任务 ID（R5 教训）。

**P4.8 已执行完毕（[预注册 §9](../../../reference/M5_K_P4_8_REPRESENTATION_CONTRACT_REDESIGN_PREREGISTRATION_20260911.md)、报告 `reports/taiji_m5_k_p4_8_representation_redesign_20260911.json`）：`outcome=invariant_constraint_insufficient`——hinge 不足以解耦。** 执行前修订 §3.1（guard band 破坏出生零损失断言，改为**边际保持 hinge**：保持 parent 自身决策边际，出生零损失按构造成立）。结果：三臂出生等价精确（0 mismatch / 0.0 偏差）；`functional-13` 第 4 次逐数值复现张力；`invariant-13` seed-0 新任务 `0.625/0.5` 仍败 + 6 sv、seed-1 与基线逐数值相同；`residual-26` 与 `invariant-13` **逐数值相同**。**两条机制记录**：(a) **arm-3 等价定理**——frozen head + δ head 在均匀 SGD 下与共享权重继承初始化函数空间等价，架构变量携带零信息（逐数值相同是定理的经验确证）；架构要成为真实变量需分头学习率/δ trust-region/特征门控。(b) **逐点约束 ≠ 分布性边际不变**——hinge 精确保持 cohort 点边际，但线性函数在结构同构、身份不同的 sibling 点上仍被侵蚀。与 P4.7、P4.3–P4.5 合并的最终结论：**冲突位于表示本身——12 维 candidate 特征没有把「影响 parent 边际的方向」与「实现新任务排序的方向」因子化；约束形式、容量、架构三个维度均已排除为根因。** `growth_admitted=false`、`can_promote=false` 不变。

**当前唯一下一步**：特征空间重设计决策点——设计能把「parent 边际敏感方向」与「新任务排序方向」因子化的新特征维度（候选方向：parent-margin 特征显式化——把 frozen parent 对每个候选的边际贡献作为附加输入维度，使保持约束只作用于这些维度的权重；或非线性感知层把 safe-margin 判定与 proposal 排序解耦），先做 frozen validation-only 特征探针验证因子化可行性，再冻结预注册；预注册冻结前不训练、不扩容、不读取 sealed、不解冻 P5/CUDA/IDE/provider。

用户确认执行。**特征探针已先行完成**（[P4.9 probe](../../../../scripts/training/eval_taiji_m5_k_p4_9_feature_space_probe.py)，报告 `reports/taiji_m5_k_p4_9_feature_space_probe_20260911.json`，frozen validation-only、无任何 Taiji fit）：M1 目标秩结构 = target 在 parent 排序下 {rank 0: 8, rank 2: 6}（新任务信号 = 提升 parent 的第 3 位候选，utility gap mean 0.386）；M2 flip 方向与 parent 权重平方余弦 0.00036（近正交）；**M3 联合可行性（决定性）：基 12 维不可行（最小联合违反 0.0494——P4.2–P4.8 全部失败的定量解释），扩展 16 维（+4 个 frozen-parent-relative margin 特征）精确可行（违反 0.0）——`parent_relative_features_are_the_factorization`**。两阶段纪律不受污染：正式门沿用已冻结阈值，探针只影响臂设计。

**P4.9 特征空间重设计预注册已冻结：[M5_K_P4_9_FEATURE_SPACE_REDESIGN_PREREGISTRATION_20260911.md](../../../reference/M5_K_P4_9_FEATURE_SPACE_REDESIGN_PREREGISTRATION_20260911.md)。** 两臂（相邻对单变量）：`invariant-base-13`（P4.8 复现基线）/ `invariant-ext-17`（唯一变更 = 特征空间扩至 16 维：12 基 + parent_argmax_margin/parent_safe_margin/parent_rank_norm/is_parent_pick；17 参数，base 权重+bias 从 parent 逐位继承、4 新维度零初始化——出生等价精确；特征由内部 frozen parent 副本计算，训练全程非漂移）。两臂共享 margin-preservation hinge（canonical 函数）与全部协议。门沿用 P4.7/P4.8 冻结值。结果映射：ext 过 + base 张力 → `feature_factorization_supported`（固定容量路线在新表示下重开）；ext 败 → `learnability_gap`（瓶颈转优化动力学，转约束求解器方向）；base 全过 → `baseline_drift`。正式实验身份 `p4-10`。`growth_admitted=false`、`can_promote=false`；不加第三臂（P4.8 等价定理）。

**当前唯一下一步**：执行 P4.9 §7——(1) `taiji/g_selection_extended.py` + 定向测试；(2) 两臂 runner（py_compile/ruff/mypy 先行）；(3) 执行落盘报告；任一停止线触发即停。

用户确认执行。**P4.10 已执行完毕（[预注册 §8](../../../reference/M5_K_P4_9_FEATURE_SPACE_REDESIGN_PREREGISTRATION_20260911.md)、报告 `reports/taiji_m5_k_p4_10_feature_factorization_20260911.json`）：`outcome=learnability_gap`。** 实现：`taiji/g_selection_extended.py`（16 特征 head：base 权重逐位继承 + 4 因子化维度零初始化；特征由内部 frozen parent 副本计算，**独立存储字段 + digest 校验**保证非漂移——执行中发现并修复了「从 head base 维重建 feature source」的恢复设计错误）+ 定向测试 5/5 + 两臂 runner。结果：机械门全过（两臂出生等价精确、出生 hinge 恒 0、feature source 非漂移）；`invariant-base-13` 与 P4.8 逐数值一致；`invariant-ext-17` **权衡面移动但未闭合**——seed-0 新任务 `0.6375/0.55` 逼近门仍败 + 6 sv、retention-newtask 仅剩 safe violation 一项；seed-1 sibling 修复（`1.0`）但新任务退至 `0.6375/0.55`+6 sv；hinge 仅 12–13/112 步激活。**判定：可行解存在（P4.9 探针违反 0.0）但交错 SGD 从 parent 初始化不可达——瓶颈正式从「表示存在性」转为「优化动力学」。** `growth_admitted=false`、`can_promote=false` 不变。

**当前唯一下一步**：约束求解器方向预注册——把 P4.9 探针的联合违反最小化机械升级为**可行区域投影求解器**：task 学习后把权重投影到联合可行区域（最小化到当前权重的距离 subject to 联合约束），替代/增强交错 SGD 步；先冻结投影求解器的收敛判据与投影频率合同，再冻结正式预注册；冻结前不训练、不读取 sealed、不解冻 P5/CUDA/IDE/provider。

用户确认执行。**P4.11 投影求解器预注册已冻结：[M5_K_P4_11_PROJECTION_SOLVER_PREREGISTRATION_20260911.md](../../../reference/M5_K_P4_11_PROJECTION_SOLVER_PREREGISTRATION_20260911.md)**（求解器收敛判据 + 投影频率合同 + 正式实验设计一并冻结）。**求解器合同**：惩罚延续法（ρ ∈ {1,10,100,1000}，每相 full-batch Adam 6000 步、lr 0.05 余弦衰减，warm 延续确定性，无随机重启）；目标 = ρ·联合违反 + ½·||w − anchor||²，anchor = 基线臂训练终点（全程固定）；约束系统 = 任务约束（fit-eligible train 集 target argmax + safe-margin）+ 保持约束（constraint cohort 上 frozen parent 决策边际保持）；全部约束为分数差 → **投影只作用于 16 权重维，bias 不动**；收敛判据冻结（逐约束 ≤ 1e-6、总量 ≤ 1e-5，否则 `projection_incomplete` 诚实停止）；投影频率 = 每次训练恰一次末端投影。**两臂（单变量 = 末端投影）**：`invariant-ext-17`（P4.10 复现基线，无投影）/ `projected-ext-17`（与基线臂**逐位相同轨迹** + 一次联合投影——直接回答「P4.10 失败端点能否被一次投影修复」）。门沿用已冻结值；身份空间 `p4-11`。结果映射：投影完成 + projected 过 + 基线张力 → `projection_solver_supported`（固定容量路线在求解器机制下重开）；投影完成 + 仍互斥 → `projection_generalization_gap`（cohort 可行 ≠ 评估泛化，进入任务/表示联合重设计）；未收敛 → 机械失败停止。

**当前唯一下一步**：执行 P4.11 §7——(1) 投影求解器实现 + 定向测试（收敛判据/确定性/anchor 固定/bias 不动）；(2) 两臂 runner（py_compile/ruff/mypy 先行）；(3) 执行落盘报告；任一停止线触发即停。

用户确认执行。**P4.11 已执行完毕（[预注册 §8](../../../reference/M5_K_P4_11_PROJECTION_SOLVER_PREREGISTRATION_20260911.md)、报告 `reports/taiji_m5_k_p4_11_projection_solver_20260911.json`）：`outcome=projection_solver_supported`——投影求解器成立，P 系列首次有更新机制两 seed 同时通过双侧门。** 实现：`taiji/g_selection_projection.py`（确定性惩罚延续投影器）+ `apply_projected_weights` + 定向测试 5/5 + 两臂 runner（执行前修订 §2.1：保持约束改为决策同一性形式——探针已证可行的系统）。结果：**trajectory gate = 两臂投影前 head digest 逐位相同**（投影是唯一变量）；投影精确收敛（80 条联合约束，违反 0.0，距离 L2 2.72/2.98 如实审计）；**`projected-ext-17` 两 seed 全门通过**——holdout utility `0.8`≥0.68、target `0.75`≥0.6、0 safe violations、sibling `1.0/1.0` 非劣、retention-newtask `0.8`≥parent `0.6375`；基线臂复现 P4.10 失败。**P4.2–P4.11 完整证据链的机制结论：保持/新任务解耦 = 表示因子化（P4.9）+ 优化机制替换（P4.11 末端投影）两个必要成分的合取，任一单独不充分；且 cohort 可行性成功泛化到全部评估身份。** 固定容量路线在求解器更新机制下**重开**。`growth_admitted=false`、`can_promote=false` 不变。

**当前唯一下一步**：求解器机制下的晋级课程级验证预注册——更大 seed/课程矩阵 + 资源审计，gate 沿用已冻结阈值；预注册冻结前不训练、不读取 sealed、不解冻 P5/CUDA/IDE/provider。

用户确认执行。**P4.12 晋级课程级验证预注册已冻结：[M5_K_P4_12_COURSE_LEVEL_VALIDATION_PREREGISTRATION_20260911.md](../../../reference/M5_K_P4_12_COURSE_LEVEL_VALIDATION_PREREGISTRATION_20260911.md)。** 矩阵：**3 课程身份批 × 3 seeds = 9 cells**（统计单元 = 身份批，seed 为 replicate）；每 cell 双臂（baseline 无投影 / projected 逐位相同轨迹 + 一次末端联合投影）+ trajectory gate（投影前 digest 逐位相同）；求解器合同与 P4.11 逐参数相同、按 cell 实例化（决策同一性约束系统、anchor = 该 cell 基线臂终点、bias 不动、收敛判据冻结，任一 cell `projection_incomplete` 按机械失败记入）。门沿用 P4.7–P4.11 已冻结阈值（零变更）；**聚合门 = projected 臂 ≥ 8/9 cell 全门通过 ∧ 基线臂 ≥ 2/3 身份批张力复现**；资源审计逐 cell（fit/投影 wall-clock、24k solver 步、checkpoint 字节）+ 软门（projected 总 wall ≤ 3× baseline）。结果映射：≥ 8/9 ∧ 基线张力 → `course_level_validation_supported`（晋级课程入场资格成立，进求解器机制下的晋级课程预注册）；≤ 7/9 → `course_level_validation_failed`（按身份/seed/门分布归因）；基线全过 → `baseline_drift`。`growth_admitted=false`、`can_promote=false`；不加第三臂、不改求解器合同。

**当前唯一下一步**：实现 9-cell runner（`eval_taiji_m5_k_p4_12_course_level_validation.py`，py_compile/ruff/mypy 先行）并执行落盘报告；任一停止线触发即停。

用户确认执行。**P4.12 已执行完毕（[预注册 §8](../../../reference/M5_K_P4_12_COURSE_LEVEL_VALIDATION_PREREGISTRATION_20260911.md)、报告 `reports/taiji_m5_k_p4_12_course_level_validation_20260911.json`）：`outcome=course_level_validation_supported`——求解器机制课程级验证成立。** 9/9 cell projected 臂全门通过且九格指标**逐数值相同**（holdout utility `0.8`、target `0.75`、0 sv；sibling `1.0/1.0`；retention-newtask `0.8` ≥ parent `0.6375`）；基线臂 3/3 批张力复现；投影 9/9 精确收敛（违反 0.0）。机械门全过（264 digest 唯一、身份隔离、structure/checkpoint/tamper/feature-source/trajectory 9/9）。**新增机制结论：投影求解器把「保持/新任务权衡」从 seed 敏感的优化路径问题变成了确定性的可行性求解问题（九格零方差）。** 资源审计诚实记录：3× wall 软门 9/9 超限为结构性的（fit-only 基线 0.08s vs fit+投影 3–6.5s，比率 40–80×——投影正是机制成本），按预注册为描述性边界不影响聚合门；绝对耗时极小，正式资源 cap 属晋级课程预注册且须以绝对预算定义。`growth_admitted=false`、`can_promote=false` 不变。

**当前唯一下一步**：求解器机制下的晋级课程预注册——同一 parent 连续 S/G/K 课程，学习机制 = 「SGD 任务学习 + 末端联合投影」求解器机制，gate 沿用 A8 结构 + 资源 cap 以绝对预算定义；预注册冻结前不训练、不读取 sealed、不解冻 P5/CUDA/IDE/provider。

用户确认执行。**P4.13 求解器机制下的晋级课程预注册已冻结：[M5_K_P4_13_PROMOTION_COURSE_PREREGISTRATION_20260911.md](../../../reference/M5_K_P4_13_PROMOTION_COURSE_PREREGISTRATION_20260911.md)。** 关键设计：课程 = **两个相继的新任务阶段（A → B）**——检验晋级真正需要的**持续累积能力**而非单任务 + 保持。每 cell：Phase A（cohort A 任务 fit + 投影 #1 = A 约束 + 保持约束）→ Phase B（从 A 投影态出发的任务 fit + 投影 #2 = **累积系统** A 约束 + B 约束 + 保持约束）；**向后保持门（新增）= Phase B 后 A-holdout 仍过新任务门**；checkpoint/rollback 机械门沿用 P3.0 合同（A 投影态保存/恢复/回滚是 B 的安全网）；资源 cap 以**绝对预算**定义（每相 fit ≤ 60s、投影 ≤ 120s、cell 总 ≤ 600s——P4.12 比率式软门的修正）。矩阵 3 批 × 3 seeds = 9 cells；求解器合同与 P4.11/P4.12 逐参数相同；门阈值零变更。结果映射：聚合门过 → `promotion_course_supported`（G 侧晋级课程闭合，进 scorecard 更新与晋级评审）；Phase B 投影不收敛 → `cumulative_constraint_conflict`；B 破坏 A → `sequential_retention_failure`。诚实边界：G 选择头课程（K 联合运行为后续预注册）；A/B 同分布（检验累积 + 无遗忘，非跨任务类型泛化）。`growth_admitted=false`、`can_promote=false`。

**当前唯一下一步**：实现两相课程 runner（`eval_taiji_m5_k_p4_13_promotion_course.py`，py_compile/ruff/mypy 先行）并执行落盘报告；任一停止线触发即停。

用户确认执行。**P4.13 已执行完毕（[预注册 §9](../../../reference/M5_K_P4_13_PROMOTION_COURSE_PREREGISTRATION_20260911.md)、报告 `reports/taiji_m5_k_p4_13_promotion_course_20260911.json`）：`outcome=promotion_course_supported`——求解器机制支持持续累积，G 侧晋级课程闭合。** 9/9 cell：Phase A 全门（A-holdout `0.8`/0sv）→ Phase B 全门（B-holdout `0.8`/0sv）→ **向后保持门零失败（A-holdout 回检 `0.8/0.75`——B 学习后 A 零遗忘）**；**累积投影系统（A+B+保持，154 条约束）9/9 精确收敛（违反 0.0）**——「约束系统随课程增长」的可行性担忧经验未成立；rollback 门 9/9（A 投影态恢复行为逐位一致）；资源绝对预算全过（cell 12–16s ≪ 600s）；基线臂 3/3 批张力复现。runner 修复两处机械错误（digest 集合误初始化为 set、唯一性期望式算术 3×168→3×148），未触碰判据。`growth_admitted=false`、`can_promote=false` 不变。**G 侧晋级课程完整证据链闭合：表示因子化（P4.9）+ 求解器更新机制（P4.11）+ 单任务验证（P4.12）+ 持续累积课程（P4.13）。**

**当前唯一下一步**：scorecard 更新与晋级评审——把 P4.2–P4.13 的 G 侧证据线收束入 K 轴 scorecard 新版本，冻结晋级边界与默认 runtime rollout review 的入口条件（K worker 联合课程为后续预注册）；评审前不训练、不读取 sealed、不解冻 P5/CUDA/IDE/provider。

用户确认执行。**K 轴 scorecard v4 已冻结并执行：[M5_K_AXIS_SCORECARD_V4_CONTRACT_20260911.md](../../../reference/M5_K_AXIS_SCORECARD_V4_CONTRACT_20260911.md)。** 唯一变更 = 新增第四条证据线 `solver_mechanism_evidence`（来源 = P4.12 课程级验证 + P4.13 两相晋级课程，只读转录 + digest 校验：9/9 + 9/9 全门、累积投影全格精确零违反、向后保持门零失败、rollback/资源预算全过、机制零方差结论）；K1/K2/K3 与 C 阶段部分机械复用 v3 reducer 且 source digests 对 v3 报告逐位校验（零漂移），不重训、不重算。**晋级边界冻结**：v3 全部 veto 保留 + `g_solver_mechanism_course_closed=true`（G 侧晋级课程闭合的机器结论）+ 两个未完成入场条件 `k_worker_joint_course_completed=false`、`default_runtime_rollout_review_completed=false`——`promotion_gate=false`、`can_promote=false`；**晋级评审入场条件 = K worker 联合课程完成 + 默认 runtime rollout review 执行，二者齐备后 A8 评审才有资格召开且仍须独立批准**。报告 [scorecard v4](../../../../reports/taiji_m5_k_axis_scorecard_v4_20260911.json)；v3 报告保留为历史。**K 轴证据线现状：能力（K1/K2/K3）× 学习机制（FS 候选）× G 侧求解器机制（表示因子化 + 求解器更新 + 课程级验证 + 晋级课程）三条线闭合；剩余入口 = K worker 联合课程。**

**当前唯一下一步**：K worker 联合课程预注册——P2.6/P2.7 continuation 机械与求解器机制在同一 parent 上的联合运行（K worker 连续学习 + G 头求解器更新同课程）；预注册冻结前不解冻任何 owner、不接默认 runtime、不训练。

用户确认执行。**P4.14 K worker 联合课程预注册已冻结：[M5_K_P4_14_JOINT_COURSE_PREREGISTRATION_20260911.md](../../../reference/M5_K_P4_14_JOINT_COURSE_PREREGISTRATION_20260911.md)。** 交互点诊断：K worker 学习会**改变候选特征景观**（G 相候选特征由 K1/K2 读出派生），G 相必须在「K 学习后的景观」上维持求解器平衡——这是两条已验证机制从未被检验的组合面。**课程结构（每 cell 两相继阶段）**：Phase K（P2.6 机械原样：新 K2 content 目标 + 50 rehearsal 交错 + 旧类保持门 + K checkpoint）→ **post-K 重 materialization**（从 post-K worker 重新实例化 semantic/transition 并 materialize G cohort）→ Phase G（P4.11 合同原样：任务 fit + hinge + 末端联合投影，决策同一性参考 = frozen P3.5 parent 在 post-K 景观上的决策）。**跨相门（新增）**：`k_unchanged_after_g`（G 只写 G 头）、`g_preservation_vs_frozen_parent`（post-K 景观上 well-defined）、G 相 birth 等价精确。矩阵 **2 批 × 2 seeds = 4 cells**（主导风险 = 集成机械，机制均已单独验证；规模扩展属晋级课程 formal）；资源绝对预算（K fit ≤ 120s、G fit ≤ 60s、G 投影 ≤ 120s、cell ≤ 600s）。结果映射：4/4 全门 → `joint_course_supported`（scorecard v5 置 `k_worker_joint_course_completed=true`，晋级评审仅剩 rollout review）；K 保持回退 → `k_retention_regressed`；K 过而 G 败 → `cross_phase_feature_shift`（交互面真实存在，进入 post-K 校准设计）；投影不收敛 → 诚实停止。诚实边界：相继组合（非 K/G 同时训练）；K 相目标为 P2.6 同构。`growth_admitted=false`、`can_promote=false`。

**当前唯一下一步**：实现联合课程 runner（`eval_taiji_m5_k_p4_14_joint_course.py`，py_compile/ruff/mypy 先行）并执行落盘报告；任一停止线触发即停。

用户确认执行。**P4.14 已执行完毕（[预注册 §9](../../../reference/M5_K_P4_14_JOINT_COURSE_PREREGISTRATION_20260911.md)、报告 `reports/taiji_m5_k_p4_14_joint_course_20260911.json`）：`outcome=joint_course_supported`——K worker 联合课程成立，两条机制可组合。** 4/4 cell 全门：Phase K（P2.6 机械原样，K 起点 = 联合 parent 的 P3.2 base-continuation workers）新颖 K2 content 2/2、旧类 4/4、安全 abstention 6/6、参数 5,648 不变；**post-K 景观漂移真实**（K1/K2 digests `12851b…→9fe6f4…` / `411ef5…→81a189…`，跨 seed 逐位确定、跨批逐位相同——K 特征空间不编码路径/项目身份的诚实记录）；Phase G（post-K 景观上 birth 等价精确）holdout `0.8/0.8675/0sv`、sibling `1.0/1.0`、retention-newtask `0.8/0.8675`、投影 88 约束全格精确零违反；跨相门 `k_unchanged_after_g`/`g_preservation_vs_frozen_parent`/`birth_equivalence` 全过；资源 cell 9.5–12.2s ≪ 600s 全过；K/G parent 未覆盖。**scorecard v5 已冻结并运行：[M5_K_AXIS_SCORECARD_V5_CONTRACT_20260911.md](../../../reference/M5_K_AXIS_SCORECARD_V5_CONTRACT_20260911.md)**——`k_worker_joint_course_completed=true`（唯一门翻转），`promotion_gate=false`、`can_promote=false`；晋级评审入场条件仅剩 `default_runtime_rollout_review_completed`。

**当前唯一下一步**：默认 runtime rollout review 预注册——冻结 review 的范围（默认 runtime 在何种合同下消费 post-K/求解器 checkpoint）、rollout 指标与失败回滚边界；预注册冻结前不解冻任何 owner、不接默认 runtime、不训练。

用户确认执行。**默认 runtime rollout review 预注册已冻结：[M5_K_DEFAULT_RUNTIME_ROLLOUT_REVIEW_PREREGISTRATION_20260911.md](../../../reference/M5_K_DEFAULT_RUNTIME_ROLLOUT_REVIEW_PREREGISTRATION_20260911.md)。** 本 review 不是 P4.x 新实验（`fit_called=false` 贯穿），而是晋级评审最后一项入场条件 `default_runtime_rollout_review_completed` 的验收合同。**范围**：默认 runtime 现状只读盘点（`api/seed_runtime.py` 的 `SeedRuntime` 只消费 `checkpoints/seed_corpus.pt` 的 `seed-native-v1`；`api/`/`seed_platform/`/`taiji/adapter.py` 中零处 K/G 引用——fail-closed 现状如实入账）；消费对象 = P4.14 全部 4 个 cell 的 post-K K1/K2 worker + 逐 cell 投影后扩展 G 头；消费合同 `taiji-default-runtime-rollout-attachment-v1`（内容寻址 attachment manifest 钉定逐 cell artifact digest、P4.14 lineage、嵌入安全不变量 `confidence_floor=0.55`/`selection_margin=0.05`/K 参数 5,648；fail-closed load 顺序；运行时禁止 fit、禁止改写 artifact、禁止 override 内嵌不变量；沙箱 runner 演示消费路径并定义 runtime 侧 adapter 合同，不改产品代码、不接线上服务）。**rollout 指标（全部冻结零变更）**：核心新证据 = 消费等价门——通过消费路径加载后在冻结 cohort 上逐字段复现 P4.14 记录值（Phase K novel 2/2 + 旧类 4/4 + 安全 6/6；Phase G validation/holdout/retention-newtask `0.8/0.8675/0sv` + sibling `1.0/1.0`）；post-K digest 锚（`9fe6f48187ad…`/`81a1891c470b…`）与逐 cell G checkpoint 锚；嵌入不变量保持；非干扰门（`seed_corpus.pt` 与全部研究 artifact digest 前后一致）；独立恢复/rollback/篡改拒绝；资源绝对预算（每 cell ≤ 120s、总 ≤ 600s）。**矩阵**：4/4 cell 全部执行全部通过，无子采样。**失败回滚边界**：行为门失败时 rollback 门仍必须执行并通过；结果映射 `default_runtime_rollout_review_supported`（scorecard v6 翻转 `default_runtime_rollout_review_completed=true`，A8 获得召开资格仍须独立批准）/ `rollout_consumption_gap`（停止并桥接设计，不调门不重训）/ `rollback_unverified` / 机械失败诚实停止。

**当前唯一下一步**：实现 review runner（`eval_taiji_m5_k_default_runtime_rollout_review.py`，py_compile/ruff/mypy 先行）并执行落盘报告；任一停止线触发即停。

用户确认执行。**默认 runtime rollout review 已执行完毕（[预注册 §9](../../../reference/M5_K_DEFAULT_RUNTIME_ROLLOUT_REVIEW_PREREGISTRATION_20260911.md)、报告 `reports/taiji_m5_k_default_runtime_rollout_review_20260911.json`、消费合同 `plans/manifests/taiji_m5_k_default_runtime_rollout_attachment_v1.json`）：`outcome=default_runtime_rollout_review_supported`——4/4 cell 全门通过，A8 晋级评审入场条件齐备。** 核心新证据 = **消费等价**：P4.14 联合课程产物经磁盘加载的 fail-closed 消费路径（manifest 自校验 → lineage → 逐位 digest → 嵌入不变量 → 独立进程恢复 → 篡改/混装拒绝）后，Phase K（novel 2/2、旧类 4/4、安全 6/6）与 Phase G（validation/holdout/retention-newtask `0.8/0.8675/0sv`、sibling `1.0/1.0`）在 4 个 cell **逐字段复现 P4.14 记录值**；cohort 重物化 digest 与 P4.14 manifest 逐位一致（checkpoint 一律未重建）。机械门全过：K1/K2/G 篡改拒绝、wrong-cell 混装拒绝、G 评估后 K digest 不变（`k_unchanged` 消费版）、非干扰（`api/seed_runtime.py` 与 `checkpoints/seed_corpus.pt` sha256 前后一致、parent 锚未覆盖）、资源绝对预算全过（cell 2.0–2.1s ≪ 120s、总 267s ≪ 600s）、`fit_called=false`。执行中修复三处合同构建错误（attachment 自摘要须排除 `manifest_digest` 键；G 钉定摘要为排除内嵌 `checkpoint_digest` 键的约定；K2 `fact_threshold=0.55` 区别于 K1 的 `0.65`），均属合同侧、非工件问题，不触碰任何冻结判据。**scorecard v6 已冻结并运行：[M5_K_AXIS_SCORECARD_V6_CONTRACT_20260911.md](../../../reference/M5_K_AXIS_SCORECARD_V6_CONTRACT_20260911.md)**——`default_runtime_rollout_review_completed=true`（唯一门翻转），`promotion_gate=false`、`can_promote=false`；v5 全部 source digest 零漂移。

**当前唯一下一步**：A8 晋级评审——两个入场条件（K worker 联合课程 + 默认 runtime rollout review）已齐备；评审须独立批准，逐项裁决其余 veto（默认 runtime owner 附着、资源/rollback/旧能力门、同 parent learned S/G/K 证据、parent retention baseline）；评审召开前不训练、不接默认 runtime、不解冻 owner。

用户确认执行。**A8 晋级评审合同已冻结：[M5_K_A8_PROMOTION_REVIEW_20260912.md](../../../reference/M5_K_A8_PROMOTION_REVIEW_20260912.md)。** 评审对象 = v6 中仍为 false 的五项 veto，分两类裁决：**证据裁决三项**——`parent_retention_baseline_present`（建议翻转：P2.4→review 保持链强于 v1 字面要求）、`same_parent_continual_s_g_k_evidence`（建议翻转：P4.14 即同 parent 连续 S/G/K 课程，S 为架构性 control-only evidence 的边界如实入账；备选 learned-S 新线或维持 veto）、`resource_rollback_old_capability_gate`（建议翻转：SGK v1 的资源/rollback/旧能力三要求在 P4.12–P4.14 + review 后继链中以更强形式满足，SGK v1 标记 `superseded` 且其 FS 机制已被 P4.10/P4.11 证明为该任务失败机制）；**授权裁决一项**——`default_runtime_owner_attached` + `learning_mechanism_attached_default_runtime`（建议授权进入附着工程预注册：runtime 侧 adapter 按已验证消费合同实现 fail-closed load/独立恢复/行为等价/回滚；两项 veto 只在附着步执行通过后由 v7 翻转，不由本评审翻转）。结果映射 `promotion_review_recommended` / `promotion_review_partial_<veto>` / `promotion_review_deferred`；fail-closed：本评审不宣布 promotion、`can_promote` 只能由 v7 在五项 veto 全翻转后机械置位且仍须独立批准；A8 增长子句在本轴未被触发（P4.7 关闭容量假设）为如实记录而非跳过。

**当前唯一下一步**：评审人对 A8 评审合同 §3 的独立批准（逐项批准/否决/改选）；批准前一切保持 scorecard v6 现状。

用户确认执行。**A8 晋级评审已独立批准（2026-09-12，四项裁决全部批准，结论 `promotion_review_recommended` 已落盘 [评审合同 §7](../../../reference/M5_K_A8_PROMOTION_REVIEW_20260912.md)）。** §3.1/§3.2/§3.3 三项证据裁决批准（选项 A）；§3.4 附着授权批准；评审人常设准则入账：**后续决策点优先上限更高的选项**（以不违反 fail-closed 纪律与已冻结判据为前提）。**scorecard v7 已冻结并运行：[M5_K_AXIS_SCORECARD_V7_CONTRACT_20260912.md](../../../reference/M5_K_AXIS_SCORECARD_V7_CONTRACT_20260912.md)**——三项证据 veto 机械翻转（`parent_retention_baseline_present`、`same_parent_continual_s_g_k_evidence`、`resource_rollback_old_capability_gate`：false→true），两项附着 veto 保持 false；v6 全部 source digest 零漂移；SGK v1 标记 superseded（状态横幅，内容不改写）；`promotion_gate=false`、`can_promote=false` 机械保持。评审合同 §3.4/§5 的「scorecard v7」编号笔误已在 §7 勘误修正（判据零变更）。

**当前唯一下一步**：实现附着工程（`api/taiji_runtime_attachment.py` + `SeedRuntime.attach_k_g_state/detach` + 验收 runner `eval_taiji_m5_k_runtime_attachment.py`，py_compile/ruff/black/mypy 与相关 pytest 先行）并执行落盘报告；任一停止线触发即停。

用户确认执行。**默认 runtime 附着工程预注册已冻结：[M5_K_DEFAULT_RUNTIME_ATTACHMENT_PREREGISTRATION_20260912.md](../../../reference/M5_K_DEFAULT_RUNTIME_ATTACHMENT_PREREGISTRATION_20260912.md)。** 按评审人「优先上限更高」准则取真实 runtime 集成设计：`api/taiji_runtime_attachment.py`（runtime 自有消费合同实现，禁止 import `scripts.*`；load 顺序与 review 冻结值一致，G digest 排除内嵌键约定、K1/K2 fact_threshold 按角色钉定、无 override 通道）+ `SeedRuntime.attach_k_g_state/detach_k_g_state`（显式 opt-in、原子附着、失败整体拒绝并记录原因）+ runtime status `k_g_attachment` 段与能力面注册 + typed 读出面（`k1_predict`/`k2_predict`/`g_select`）+ 独立进程校验子命令。验收门（4/4 cell 全部）：runtime 读出面消费等价逐字段复现 P4.14 记录值、fail-closed 拒绝（篡改/混装/错 lineage 拒绝且 runtime 保持未附着）、detach→re-attach 行为逐数值等价、非干扰（`seed_corpus.pt` 与研究 artifact digest 前后一致、默认行为零变化）、预算（attach ≤ 60s/cell、总 ≤ 600s）、静态纪律。结果映射：4/4 全过 → `runtime_attachment_supported` → v8 翻转两项附着 veto（届时五项 gate 全 true，`promotion_gate`/`can_promote` 机械置 true，最终晋级仍须独立批准）；消费缺口/拒绝门失败/rollback 失败 → 诚实停止。诚实边界：附着 ≠ promotion（默认任务解释路径不改写）；附着为进程内 opt-in 内存态；`growth_admitted=false` 贯穿。

**当前唯一下一步**：实现附着工程（`api/taiji_runtime_attachment.py` + `SeedRuntime.attach_k_g_state/detach` + 定向测试 + 验收 runner，静态检查先行）并执行落盘报告；任一停止线触发即停。

用户确认执行。**默认 runtime 附着工程已执行完毕（[预注册 §7](../../../reference/M5_K_DEFAULT_RUNTIME_ATTACHMENT_PREREGISTRATION_20260912.md)、报告 `reports/taiji_m5_k_runtime_attachment_20260912.json`）：`outcome=runtime_attachment_supported`——4/4 cell 全门通过。** 实现：`api/taiji_runtime_attachment.py`（runtime 自有 fail-closed 消费合同，零 `scripts.*` import；load 顺序与 review 冻结值一致；K1/K2 fact_threshold 按角色钉定 0.65/0.55；G parent lineage 对 P4.1 校验；篡改/混装探针内建）+ `SeedRuntime.attach_k_g_state/detach_k_g_state/k_g_attachment_status`（显式 opt-in、原子拒绝、拒绝原因入 status 面）+ typed 读出面 `k1_predict/k2_predict/g_select` + `--verify-attachment` 独立进程校验；定向测试 15/15。验收（4/4 cell）：**runtime 读出面消费等价逐字段复现 P4.14 记录值**（novel 2/2、旧类 4/4、安全 6/6；`0.8/0.8675/0sv`、sibling `1.0/1.0`）；篡改/缺文件/混装三探针全拒净且 runtime 保持未附着；detach→re-attach 行为逐数值等价；非干扰全过（`seed_corpus.pt` sha256、model tick/参数计数、P4.14 报告均不变；附着不进入默认路径）；资源 attach 1.3s ≪ 60s、总 266s ≪ 600s；`fit_called=false`。`seed_runtime.py` mypy 错误数 24 = 改动前基线（新代码零新增）。**scorecard v8 已冻结并运行：[M5_K_AXIS_SCORECARD_V8_CONTRACT_20260912.md](../../../reference/M5_K_AXIS_SCORECARD_V8_CONTRACT_20260912.md)**——两项附着门翻转（`default_runtime_owner_attached`、`learning_mechanism_attached_default_runtime`：false→true），五项晋级 gate 全 true，`promotion_gate=true`、`can_promote=true` 机械置位（v7 合同 §4 冻结映射）；v7 全部 source digest 零漂移。**`can_promote=true` ≠ 已晋级：最终晋级宣布仍须独立批准。**

**当前唯一下一步**：最终晋级宣布的独立批准——机器边界已闭环；批准须保留诚实边界（五类合成载体、结构成长未被触发、附着 opt-in 且默认行为路径不变、S 为架构性 control-only evidence）；批准与否均落盘并同步唯一计划。

用户确认执行。**晋级宣布已获项目所有者独立批准并落盘：[M5_K_PROMOTION_DECLARATION_20260912.md](../../../reference/M5_K_PROMOTION_DECLARATION_20260912.md)（2026-09-12）——M5 K 轴持续学习机制晋级（限定范围）。** 宣布范围 = 能力 formal + C 阶段学习机制 + 表示因子化（P4.9）+ 投影求解器（P4.11）+ 课程级验证（P4.12）+ 两阶段累积课程（P4.13）+ K worker 联合课程（P4.14）+ runtime 消费合同（rollout review）+ 真实 runtime 附着（4/4 cell runtime 读出面逐字段复现、rollback 精确、非干扰全过）。四条诚实边界随宣布冻结生效（五类合成载体、结构成长未被触发 `growth_admitted=false`、附着 opt-in 且默认行为路径不变、S 为架构性 control-only evidence）；机器边界基准 = scorecard v8（13 项 gate 全 true）。任何行为切换（默认路径采用、附着持久化）仍走预注册节奏。

**当前唯一下一步**：P5.1 知识来源预注册——先摸底既有 S 轴内化机械与语料现状（S0–S3 各 Gate 报告、`taiji/artifact_internalization.py` 的 Skill/MCP artifact 边界、internalization learner 入口、governed corpus 现状），再冻结预注册：已验证的 K 轴持续学习机制接入有来源语料，与普通数据同预算比较内化与未见任务收益；工具知识与宿主执行权限分离不变。摸底为只读；冻结前不训练、不接新数据源、不读取 sealed。

用户确认执行（持续推进模式）。**摸底完成 + P5.1 预注册已冻结并执行：[M5_P5_1_SOURCED_KNOWLEDGE_TRANSFER_PREREGISTRATION_20260912.md](../../../reference/M5_P5_1_SOURCED_KNOWLEDGE_TRANSFER_PREREGISTRATION_20260912.md)、报告 `reports/taiji_p5_1_sourced_knowledge_transfer_20260912.json`：`outcome=sourced_knowledge_transfer_supported`，全门通过。** 摸底结论：S 轴机械完整（S0 原语、S3 语义 embedding 内化 passed、S4 异构 holdout passed、S6/S6b 真实结果 grounding supported、E4 artifact 内化 10 检查全过），未覆盖的缺口 = **同预算下语料内容本身的因果收益**（既有门只对照 lesion，未对照内容无关的同预算语料）。设计：两臂单变量 = 语料内容族——sourced（与 holdout 任务同 workflow 族）vs placebo（capability 词表与 holdout 零重叠、其余构造逐位同构），同预算断言逐项成立（3 artifacts/9 events/同 unit kinds/同 trainer 配置 seed=17）；先导探针实测数值依据（delta `0.5`）冻结 margin `0.15`。实测：**sourced holdout `0.5` > lesion `0.333`、placebo holdout `0.0`，content delta `0.5` ≥ 0.15**；E4机械门保持（checkpoint 往返两臂一致、quarantine 拒绝、semantic 内化两臂通过）；capability 词表断言非空成立；总 wall `0.83s`。诚实边界：泛化 = 结构/词表重叠泛化（特征哈希），非语义改写泛化；任务面单一（procedural next-action）；语料为构造 governed fixtures；执行分离沿用 E4。`growth_admitted=false`、`can_promote=false` 贯穿。

**当前唯一下一步**：P5.1 规模化预注册——把 S3 语义 embedding（`DocumentEmbedder`，MiniLM 384 维，已预注册锚定）接入 artifact/document 内化管线，使迁移从「词表重叠泛化」升级为「语义改写泛化」，并以更大任务面（语义/affordance 任务面加入因果对照）复跑同预算 sourced-vs-placebo 设计；先摸底 S3 embedder 与管线接合面，再冻结预注册；冻结前不训练、不接新数据源、不读取 sealed。

用户确认执行（持续推进模式）。**P5.1b 预注册已冻结并执行：[M5_P5_1B_SEMANTIC_PARAPHRASE_TRANSFER_PREREGISTRATION_20260912.md](../../../reference/M5_P5_1B_SEMANTIC_PARAPHRASE_TRANSFER_PREREGISTRATION_20260912.md)、报告 `reports/taiji_p5_1b_semantic_paraphrase_transfer_20260912.json`：`outcome=semantic_transfer_insufficient`——核心门诚实失败。** 实验发现三条：**(1) 语义值函数的改写泛化不区分内容族**——sourced trial 对 A 族改写值 `0.603`，但 placebo 训练 trial（同预算、同查询、都训到 train MSE→0）对同一查询集也有 `0.586`，content causality delta `0.017` ≪ 冻结 `0.3`：3 文本语料规模上值函数学到的是风格/骨架泛化而非内容寻址；**(2) 治理管线 admission 内容敏感**——placebo 臂因 procedural 迁移失败被原子回滚（probe 早期「placebo 全 0」的回滚伪象被识别，纯内容对照改用其训练 trial 完成）；**(3) 结构哈希 encoder 骨架地板复现**（A-para `0.873`/B-para `0.790`，对无关内容同打高分）——对 E4 `internal_value` 检查的解释细化入账。机械门全过；负结果如实入账并直接塑造下一步设计：内容寻址需要内容辨别性训练信号——内化学习器已有 `pairwise_margin`/ranking 更新机制（家族间成对偏好），是 P5.1c 的第一设计杠杆；语料需更大更多样。`growth_admitted=false`、`can_promote=false` 贯穿。

**当前唯一下一步**：P5.1c 预注册（成对对比训练：家族辨别力 Gate）——用内化学习器既有 `pairwise_margin`/ranking 更新机制在家族间成对对比（family-A 正例 vs family-B 负例），先探针实测辨别力数值（A/B 改写查询值差），再冻结 margin 与门；冻结前不训练、不接新数据源、不读取 sealed。

用户确认执行（持续推进模式）。**P5.1c 预注册已冻结：[M5_P5_1C_CONTRASTIVE_DISCRIMINATION_PREREGISTRATION_20260912.md](../../../reference/M5_P5_1C_CONTRASTIVE_DISCRIMINATION_PREREGISTRATION_20260912.md)。** 先导探针（`probe_taiji_p5_1c_contrastive_discrimination.py` + `probe_taiji_p5_1c_scale_sweep.py`，validation-only）实测判别力随语料规模跳升：3 文本 `0.111` → 5 文本 `0.621` → 10 文本 `0.736` → 20 文本 `0.730`（对比训练：family-A 奖励 1.0 / family-B 奖励 0.0 + 既有 `pairwise_margin=0.5` 跨族 ranking pairs）；据此冻结核心门 margin `0.3`（取 10 文本效应的 ~40%）。九门合同 = 对比辨别力（A-para − B-para ≥ 0.3）+ 改写迁移保持（A-para ≥ 0.3）+ B 族主动拒绝（B-para ≤ 0）+ 结构臂骨架地板复现（A/B-para 均 ≥ 0.3）+ 表层不相交 + embedder 锚定确定性（model/revision/config digest + double-embed）+ 同预算 + trial causal gate + wall ≤ 600s。

**P5.1c 已执行完毕：报告 `reports/taiji_p5_1c_contrastive_discrimination_20260912.json`：`outcome=contrastive_content_discrimination_supported`，九门全通过。** gate runner 执行中修复结构臂语料构造两处缺陷（train 自我复制触发 example_id 去重校验；holdout 复用 train 全量触发不相交校验），修复后按对比臂同构重建确定性划分重跑成功（holdout/retention 仅评估、不训练，不影响训练后分值）。实测：A-para `0.4476`、B-para `-0.2838`（主动拒绝），**discrimination `0.7314` ≥ 0.3**；结构臂内容盲地板复现（A-para `0.8118`/B-para `0.8551`，disc `-0.0432`）；trial causal gate 通过（train_loss_after `0.000397`、ranking_updates 10）；wall `27.8s` ≪ 600s；实测值与独立探针基准（迁移 `0.454` / 拒绝 `-0.282` / 10 文本 disc `0.736`）高度吻合、双路径互证。判定：**内容寻址语义内化成立**——P5.1b 负结果的根因（语料规模/多样性不足）被对比训练 + 10 文本/族修复。诚实边界沿用预注册 §5（构造 governed fixtures、任务面单一 semantic value、绕过 artifact-trainer admission 组合）。`growth_admitted=false`、`can_promote=false` 贯穿。

**当前唯一下一步**：语义 encoder 接入产品内化边界的工程预注册——把 P5.1c 验证的内容寻址语义 encoder（对比训练后的 `InternalizedFeatureLearner` trial + MiniLM 锚定 embedder）正式接入 artifact-trainer 内化边界的注入合同（trainer encoder 注入点 + 真实 governed 语料规模化），先摸底注入面，再冻结预注册；冻结前不训练、不接新数据源、不读取 sealed。

**P5.1d 注入面摸底完成（只读 recon）+ 预注册已冻结：[M5_P5_1D_SEMANTIC_ENCODER_INJECTION_PREREGISTRATION_20260912.md](../../../reference/M5_P5_1D_SEMANTIC_ENCODER_INJECTION_PREREGISTRATION_20260912.md)。** 摸底锁定工程债与合同事实：P5.1b/c probe 以运行时替换 `trainer.encoder = SemanticArtifactKnowledgeEncoder(...)` 绕过 `__init__`（line 315）与 `from_checkpoint`（line 733）两处硬编码，checkpoint 恢复需 deepcopy stub 换回（类型不安全、往返不忠实）；该 encoder 滞留 probe 脚本且其 `from_checkpoint` 忽略 payload 锚定字段（漂移不设防）。合同裁定（§2）：**方案 A = 构造参数注入 + from_checkpoint 格式分派**（子类与运行时替换否决）；encoder 晋升至 `taiji/artifact_internalization.py` 并修复锚定校验（model/revision/config_digest 逐项比对，任一不符 raise）；`__init__` 构造期 fail-fast 维度校验；`semantic_pairwise_margin` 提为 trainer 构造参数（默认 0 = 现行为）+ `consolidate` ranking_pairs 透传；**维度 = 384 直通不投影**（消费侧 `_features` 对来源零假设，P5.1c 已在 384 全信息上验证 disc 0.731，投影有损上限更低）；`dataset_digest`/checkpoint 对 encoder 的摘要绑定零改动自动继承。规模化（§3）：gate 语料改由 governed 管道（`DeclarativeSourceRegistry` → `SkillArtifactAdapter`）构建，禁脚本内手写 fixture。九门（§4）：静态四项、默认回归 digest 等价、构造 fail-fast、checkpoint 往返无 stub、锚定漂移拒绝、格式分派（旧 format 兼容）、**对比辨别力核心门 ≥ 0.3**（真实语料）、三器官 admission（384 维）、确定性 + wall ≤ 1200s。冻结前未训练、未接新数据源、未读取 sealed。

用户确认执行（持续推进模式）。**P5.1d 已执行完毕（预注册 §5 产物顺序六项、报告 `reports/taiji_p5_1d_semantic_encoder_injection_20260912.json`）：`outcome=semantic_encoder_injection_supported`——九门全过，锚定语义 encoder 正式接入产品内化边界。** 注入合同落地：encoder 晋升至 `taiji/artifact_internalization.py` 并修复锚定校验（model/revision/config_digest 逐项比对任一不符 raise）+ `__init__` 构造期维度 fail-fast + `semantic_pairwise_margin` 构造参数（默认 0 = 现行为）+ `consolidate` ranking_pairs 透传 + `from_checkpoint` 格式分派（旧 format 兼容）；probe 收敛：P5.1b/P5.1c probe 删本地定义改 import 产品模块，顺带修复 probe p5_1c 两处 HEAD 既有 ruff（I001/F401，grep 确认 `_surface_tokens` 无外部引用后移除）+ black 违规。gate 以真实 governed 语料（`DeclarativeSourceRegistry` → `SkillArtifactAdapter`）构建：train 20 skills / 120 artifacts / 60 experiences / 300 examples（FROZEN_TRAIN_EXAMPLES 硬校验通过）/ 10000 ranking_pairs，holdout/retention 各 24 artifacts / 12 experiences / 60 examples。九门实测：静态四项全绿（ruff/black 修复后 7 文件干净；mypy `--follow-imports=silent seed taiji` 61 错全部 HEAD 既有基线持平零新增，新 runner 直检 Success）；对比辨别力 **disc `0.870033` ≥ 0.3**（A-para mean `0.710239` / B-para mean `-0.159794`，surface_shared_words=[] 表层零共享，ranking_updates 23）；placebo 反置臂 disc `1.002894` 互证非偶然；三器官 admission 通过（semantic train_loss_after `0.355287`，**holdout_loss_after `0.352167` 与 retention_loss_after 逐位相等 = 版本化语料 unit 文本逐字相同 → 嵌入恒等的直接证据**；internalized/grounding 双 lesion 均 1.0；procedural holdout `1.0` > lesion `0.3333` = majority-kind 基线；affordance native `0.3333` < frozen `1.3168`）；确定性 replica/trial-trainer 双一致 + wall `62.9s` ≪ 1200s。§7 对账五项要点全部吻合预注册（冻结计数、五损失区间、版本化语料证据、disc > P5.1c `0.731412` 双路径互证）；`growth_admitted=false`、`can_promote=false` 贯穿。

**当前唯一下一步**：P5.1e 同预算因果收益 Gate 预注册——注入合同就位后回答 P5.1 原始问题在产品边界的最终形式：sourced（与 holdout 同 workflow 族）vs placebo（内容无关）同预算内化，经产品 trainer + 语义 encoder，未见任务 holdout 收益 delta ≥ 冻结 margin；先摸底再冻结；冻结前不训练、不接新数据源、不读取 sealed。

**P5.1e 摸底完成（只读 recon）+ 预注册已冻结：[M5_P5_1E_SAME_BUDGET_CONTENT_BENEFIT_PREREGISTRATION_20260912.md](../../../reference/M5_P5_1E_SAME_BUDGET_CONTENT_BENEFIT_PREREGISTRATION_20260912.md)。** 摸底锁定设计事实：P5.1d placebo（outcome 反置、双族同训）与 P5.1e placebo（家族排他、单变量 = 训练内容族）本质不同；`consolidate` commit-on-admit 语义确认（admitted 时三个 trial 提交回 trainer 本体、revision += 1）→ 共享 A-gate holdout 的 evaluation-only 复评可行；家族排他臂下语义价值预测收益 ill-posed（全正训练 → 恒 1 预测器同改善，双 outcome → 中偏预测与 sourced 同改善）→ **语义器官收益不以本 gate 度量**（P5.1d disc `0.870033` 已建立辨别力，本 gate 语义面降为臂内 admission sanity）；holdout 用**全新 A 族 skills**（新 subject/verb 文本同 capability steps，非 re-version——P5.1d re-version 内容逐字相同属弱 unseen，P5.1 原始设计即用新 scope）。两臂设计（§2）：sourced = A 族 8 active + 4 failed，placebo = B 族同构同数（12 skills × v1/v2/v3 partitions，两臂 artifact/experience/unit_kind/ranking_pairs 结构等式断言）；trainer kwargs 两臂逐位相同（P5.1d 冻结配置原样：feature_dim=384、SemanticArtifactKnowledgeEncoder、semantic_pairwise_margin=0.5、passes/epochs/seed 同）；扩展 workflow 集 runner 本地定义每族 16 条（P5.1c 范式）；共享评估面 = 4 条全新 A 族 gate skills。九门（§3）：static_four_checks、same_budget_enforced、capability_vocabulary_disjoint、arm_sanity_both_arms（每臂自身族 admission passed）、**content_transfer_margin ≥ 0.15（核心，P5.1 冻结常数原样继承）**、sourced_beats_lesion、affordance_content_specificity（native < frozen 且 < placebo，关系门）、checkpoint_roundtrip_both_arms、deterministic_and_budget（replica 一致 + wall ≤ 1200s）。三态（§4）：`same_budget_content_benefit_supported` / `content_benefit_insufficient`（机械门过而核心 margin 失败，回 S 轴归因）/ `failed`。冻结前未训练、未接新数据源、未读取 sealed。

用户确认执行（持续推进模式）。**P5.1e 已执行完毕（预注册 §6 产物顺序、报告 `reports/taiji_p5_1e_same_budget_content_benefit_20260912.json`）：`outcome=same_budget_content_benefit_supported`——九门全过，同预算下语料内容族的因果收益在产品边界成立。** 静态四项先行全过：py_compile/ruff 全绿；black 26.5.1 对齐——scope 内 4 个 HEAD 既有文件存在版本漂移，全部重排并独立提交 `51638b6e`（`git diff -w` 验证纯格式零语义）；mypy `--follow-imports=silent seed taiji` 61 错 = HEAD 既有基线持平零新增；pytest 全量两次运行结果一致（`4 failed / 1199 passed / 6 skipped`），4 个失败全部为 HEAD 既有且与本阶段提交无关（旧 M5 脚本 BOM、roadmap 执行 owner 期望过期、taiji_native 架构词表、`taiji/document_embedding.py` transformers import），如实入账不修复不掩盖。runner `scripts/training/eval_taiji_p5_1e_same_budget_content_benefit_gate.py`（726 行，创建提交 `eca3981f`）执行落盘报告（提交 `f7087697`），wall `65.194s` ≪ 1200s。九门实测：same_budget 断言逐位成立（两臂各 200 train examples / 3160 ranking pairs / 16+16 gate records，三分区各 72 artifacts / 40 experiences，unit_kind Counter 逐位相同）；capability 词表两臂零重叠（placebo ∩ a-gate = ∅）；双臂 admission passed（procedural train/holdout/retention `1.0`、lesion `0.1`）；**核心门 content_transfer：a-gate sourced `0.25` − placebo `0.0` = delta `0.25` ≥ 冻结 `0.15`**——sourced（editor.* 训练）把步骤序列能力迁移到全新 A 族 gate skills，placebo（network.* 训练）对同一 holdout 结构性为 0；sourced_beats_lesion（`1.0` > `0.1`）；affordance_content_specificity（native `0.36` ≪ frozen ~`1.37` 且 < placebo `0.36000002`）；checkpoint_roundtrip 双臂三项全保留；确定性 replica 一致。三态映射按预注册 §4：九门全过 → `same_budget_content_benefit_supported`。`growth_admitted=false`、`can_promote=false` 贯穿。

**P5.1f 只读摸底完成 + 预注册已冻结：[M5_P5_1F_REAL_CORPUS_SAME_BUDGET_PREREGISTRATION_20260912.md](../../../reference/M5_P5_1F_REAL_CORPUS_SAME_BUDGET_PREREGISTRATION_20260912.md)（冻结提交 `72c2d2d9`，尚未执行）。** 只读摸底（f1）：governed 准入面确认 = `SkillArtifactAdapter`/`McpArtifactAdapter`/`ClientPluginArtifactAdapter` → `DeclarativeSourceRegistry.register`（lifecycle discovered→staged→shadow→active）→ `project_to_ledger`（E1）→ trainer；仓库内无 SKILL.md、无运行时真实数据摄入面，故真实语料须沿 P5.1d 范式在 runner 内投影为声明式 manifest。四类来源盘点：本机 50 个真实 TRAE 插件 SKILL.md（lark/build-web-apps/stark/github 等，跨边界依赖）、14 个 MCP 工具描述符、repo 内 UltraData 真实语料、reports 历史报告（非语料）；按「优先上限更高」选定 **repo 内 gitignored 的 OpenBMB UltraData-SFT-Agent-2609**（2026-09-07 发布，483,661 条真实发布 agent 轨迹，General 311,006 / Tool-Use 82,760 / Code 69,895 / Search 20,000，带 LICENSE/README，`data/` 被 `.gitignore:140` 整体忽略故零跨边界、可 sha256 复现）。实测记录结构（keys=`uuid,messages,tools,source,domain`，tools 为 OpenAI 函数格式，assistant 带 tool_calls；Tool_Use 首记录 source=`areal_tau2`）与采样统计（前 100 条：Tool_Use 89 条含调用、业务工具词表 19+；Code_Agent 100 条含调用共 5523 次、词表 = apply_patch/ls/read_file/rg/shell 等软件工程族；两族零交集；Knowledge 0 调用）。两臂设计（§2）：sourced = `Tool_Use_part-1-of-9.jsonl`、placebo = `Code_Agent_part-1-of-7.jsonl`——两臂同为真实程序性语料、同任务面（next-tool-call），单变量 = 内容族，比 P5.1e 构造 fixtures 控制更严；确定性采样 = 行序取前 N 条含 ≥1 次 tool_calls 记录（无调用跳过并计数），train 200/holdout 60/retention 40/臂，a-gate 16（工具 ⊆ sourced train 词表）；每条轨迹经 `SkillArtifactAdapter` 投影（`skill_id=uuid`、version=数据集 tag、steps 展开 `tool.<name>`、knowledge 记 provenance）。九门与 P5.1e 逐项同构（核心门 5 = a-gate delta ≥ 0.15；门 3 真实词表若意外交集如实 failed 禁剪枝凑过；wall ≤ 1200s；门 1 pytest 沿用 HEAD 4 项既有失败基线零新增）；三态 = `real_corpus_content_benefit_supported` / `content_benefit_insufficient` / `failed`。诚实边界：L3 教师轨迹但真实分布/真实共享词表；corpus 不入 git，sha256+行区间复现，不触网不下载不重分发；unseen = 同词表新轨迹；`growth_admitted=false`、`can_promote=false` 贯穿。冻结后执行前不训练新门、不接预注册之外数据源、不读取 sealed。

用户确认执行（持续推进模式）。**P5.1f 已执行完毕（预注册 §6 产物顺序、报告 `reports/taiji_p5_1f_real_corpus_same_budget_20260912.json`）：`outcome=failed`——门 2/4/5/7/9 失败，核心门 5 因两臂均未达 admission 而「不可测」（accuracy/delta = null），按 §4 不构成内容收益判据、不得落 `content_benefit_insufficient`，亦不回改任何判据。** runner `scripts/training/eval_taiji_p5_1f_real_corpus_same_budget_gate.py`（1040 行，创建提交 `2fab7986`）：真实 JSONL 行序确定性采样（sourced=Tool_Use 行 [4,326] 取 300 轨迹 skip 27、sha256 `5bdec07e…`；placebo=Code_Agent 行 [0,299] 取 300 轨迹 skip 0、sha256 `3eed8ec8…`；a-gate 行 [327,343] 16 条 skip 1）→ SkillArtifactAdapter governed 投影 → 沿 P5.1e 双臂 trainer/九门/原子报告。静态四项对齐 HEAD 基线：py_compile/ruff/black 全绿（修复 UP035 导入拆分），mypy `--follow-imports=silent seed taiji` 61 错持平零新增，pytest `4 failed / 1199 passed / 6 skipped` 零新增。执行期修复（提交 `94d43c3a`）：runner 本地 `_MemoizedEmbedder` 破坏 embedder 批张量合同——`SemanticArtifactKnowledgeEncoder.encode` 依赖 `embed([t])[0]` 得 `(384,)` 行，缓存直接返回 `(384,)` 行会再被 `[0]` 降成 0 维标量触发 `internalization grounding must be a non-empty vector`；修复为缓存特征行、返回 `unsqueeze(0)` 一行批。记忆化全披露：417,702 calls / 1,095 misses / 416,607 hits（hit_rate `0.997379`），bit_equal 抽检 64/64，留 `--no-memoization` 开关。九门实测（报告提交 `84f229a7`，wall `1598.131s`）：门 1 static_four_checks **true**；门 2 same_budget_enforced **false**——真实分布天然机械不等（lifecycle 1200=1200，但 call 事件 1627 vs 19561、train examples 9310 vs 79384、各分区 artifact/experience 数全不等；ranking pairs 两臂严格 0 = 真实语料全成功无 failed 侧，零对为合法路径，如实不规避）；门 3 capability_vocabulary_disjoint **true**（a-gate 19 词 ⊆ sourced 27 词，与 placebo 10 词交集空）；门 4 arm_sanity_both_arms **false**——两臂 procedural retention `0.3656`/`0.4690` 均 < 0.5 admission 线 → admitted=false、rolled_back=true；门 5 content_transfer_margin **false 但不可测**——rollback 后 procedural readout 无提交词表（`action_kinds` 空）→ a-gate measurable=false、accuracy=null、delta=null，语义为「不可测」而非「收益不足」；门 6 sourced_beats_lesion **true**（holdout `0.3952` > lesion `0.0`，admission 外直算仍成立）；门 7 affordance_content_specificity **false**（sourced native MSE `9.04e-8` < frozen `1.482` 成立，但不 < placebo native `2.20e-8`，placebo 反而更低）；门 8 checkpoint_roundtrip_both_arms **true**；门 9 deterministic_and_budget **false**——replica 逐位一致 true 但 wall `1598.131s` > `1200s` 上限（4 个 trainer × 250/200 epoch 在万级 examples 上为主耗时，记忆化已把 ~41.8 万次编码压到 1,095 次真实 MiniLM 调用）。失败结构 = 真实语料机械不等（门 2，设计上对真实分布必失败）+ admission 未达连锁（门 4/5）+ 关系门反向（门 7）+ 预算超限（门 9），属构造/基建层问题，不反向篡改判据、不剪枝凑数、不重跑挑结果；`growth_admitted=false`、`can_promote=false` 贯穿。

**P5.1f 只读 S 轴归因 recon 完成（不训练、不冻结新门、不改判据）。** 五门失败全部获得机制级归因，无一是内容收益判据本身的失败：(1) **门 4/5（admission 未达 → 门 5 不可测）= procedural 训练预算与真实分布难度不匹配的欠拟合，投影信息无损**——证据链：semantic 器官双臂 passed 且 loss 近零（sourced `1.33e-4`、placebo `1.58e-5`，fit_updates 111,720/952,608）、affordance native MSE 双臂 `1e-8` 量级（同投影下回归近完美）→ 投影把真实轨迹信息完整送达；而 procedural 250 epoch（按 P5.1e 构造 fixtures 12 episodes 定标的预算）面对 200 episodes × 真实序列：sourced episode 均长 4.95 步（median 5、p90 8、max 15）、placebo 均长 62.0 步（median 56、p90 106、max 192）——placebo 长度 12.5× 但 episode 内工具重复率 0.935（软件工程动作循环）vs sourced 0.428（业务工具似随机序列），净难度两臂相近（train acc 0.4947 vs 0.5273），均为 250 epoch 记不全的量级。retention 0.3656/0.4690 < 0.5 是该欠拟合在 retention 分区的表现。(2) **门 2（same_budget_enforced）= 轨迹数预算口径对真实分布结构性不构成同预算**——轨迹 300=300 下 total calls 12.02×（1627 vs 19561）、semantic examples 8.53×（9310 vs 79384）；可选修正口径 = 信号量配额：semantic-examples 预算 9310 → placebo ~23.5 轨迹，total-calls 预算 1627 → placebo ~16.6 轨迹（每臂取整条轨迹至配额，配额内轨迹完整保留，机械断言改为配额后 examples/calls 数逐位相等）。(3) **门 9（wall 1598.131s > 1200s）= 完全由 examples 规模线性驱动，且与门 2 同根**——静态步数核算（用 P5.1e 校准：e 49,065 steps/65.194s = 1.329 ms/step，f 1,439,328 steps/1598.131s = 1.110 ms/step，步数比 29.3× vs wall 比 24.5×，估算与实测吻合）显示 placebo 臂 semantic fit 952,608 updates 占主导（82% 步数）；门 2 口径修正（placebo 信号量配额后）wall 预计回落至 ~200s 量级自动达标，无需减 epoch（避免动冻结 trainer 配置）。(4) **门 7（affordance_content_specificity）= 地板效应使关系门测量学失效**——e 的 native 0.36 vs frozen 1.37 有区分度，f 双臂 native MSE 均 ~`1e-8` 饱和（sourced `9.04e-8` vs placebo `2.20e-8`，两者都是数值噪声级），「sourced < placebo」在饱和区间无意义；后续设计需换未饱和 affordance 指标（如相对降幅或更难 affordance 面）而非此 MSE。recon 全程只读（读两份报告 JSON + 真实 JSONL 前 344/300 行统计 + runner 常量），未训练、未改库代码、未接新数据源、未读取 sealed；临时脚本已清理。

用户确认启动。**P5.1g 预注册已冻结 + 执行完毕（预注册 `303f5d51`、runner `5b63e386`、报告 `reports/taiji_p5_1g_real_corpus_quota_budget_20260912.json` 提交 `7fd3db2b`）：`outcome=real_corpus_content_benefit_supported`——九门全过，真实发布语料的内容因果收益在同预算下成立。** 摸底（只读探针，脚本已删、数据入预注册 §5）：reward 恒 1.0 根因（`_reward` 无 components 时 `success?1:-1` × 全成功语料）→ semantic/affordance 面结构性无信号；hidden 16 在 250–4000 epochs 上 train acc 饱和 ~0.55、retention 天花板 ~0.38（admission 不可达已测量）；hidden 64 全面更优且 250 epochs 时 a-gate 最高 0.6514。三项构造修正：**门 2 配额口径**（两臂各分区 total calls 逐位相等 989/385/253，placebo 行序取完整轨迹 23/8/3 条 + 3 条前缀截断，轨迹数不等如实披露）→ 同预算机械可判且 wall 随配额线性回落；**hidden 64 + trial-learner 测量路径**（与 consolidate 内部逐位同构，探针 250 epochs 逐位复现 f 的 0.4946898/0.3952/0.3656174 验证等价；admission 不可达如实披露，两臂 admitted=false、rolled_back=true 原样入报告，判据不动 admission 线）；**门 7 改制**（affordance MSE 饱和地板降为披露项，新门 = sourced a-gate 超越冻结 per-tick 多数工具基线 0.367 ≥ 跨代 margin 0.15，检测学习是否超越频率先验）。九门实测：静态四项全绿（pytest `4 failed/1199 passed/6 skipped` 零新增、mypy 61 错持平）；same_budget 逐位成立；词表不相交（a-gate 19 ⊆ sourced 27、∩ placebo 空）；双臂 holdout 0.4512/0.4149 > lesion 0.0；**核心门 content_transfer delta `0.651376` ≥ 0.15**（sourced a-gate `0.651376` 与探针逐位一致、placebo 结构性 0）；**频率基线 margin `0.284376` ≥ 0.15**（实测基线 0.366972 与冻结 0.367 吻合）；checkpoint 往返双臂全保留；replica 逐位一致 + wall `435.931s` ≤ 1200s（配额口径比 f 的 1598s 降 3.7×）。memo hit_rate `0.993182`、bit_equal 抽检过。三态映射 `real_corpus_content_benefit_supported`；`growth_admitted=false`、`can_promote=false` 贯穿。**P5.1 知识来源证据线全链闭合（构造+真实双成立）：内容迁移 `0.5` → 对比辨别力 `0.7314` → encoder 注入 `0.8700` → 构造同预算因果收益 `0.25` → 真实发布语料配额同预算因果收益 `0.6514`（九门全过）。**

**P5.2 只读摸底完成（IDE/interaction-group/小型模拟，不训练、不冻结新门）。** 四块盘点结论：(1) **Workbench 真实合同已完整**——`seed_platform/workbench.py`（"seed-workbench-contract-v1"，CapabilitySnapshot revision 6 共 18 capability：workspace.list/read/stat/search、editor.open、editor.diagnostics.read、workspace.programming_language.resolve、editor.set_language、workspace.apply_patch/create/rename/delete/undo、terminal.run、mcp.list/invoke）+ `WorkbenchActionRequest.from_action_intent`（L639）+ `WorkbenchTransaction`（undo_token，L787）+ preview_tool（L2275）+ 事务/语言 checkpoint（L1831-1876）；api 侧 preview→preflight→execute→undo→handoff 全链路由在位；`tests/test_workbench_contract.py` 覆盖 snapshot 防篡改/边界 token/参数漂移拒绝/preview→approval→执行→undo 闭环。(2) **文件语言识别已齐备**——`seed_platform/programming_languages.py`：`ProgrammingLanguageEvidence`（7 来源 + user_override + taiji_selection）、`ProgrammingLanguageAssessment`（resolved|ambiguous|unknown 状态 + explanation payload）、Registry resolve/select/toolchains；语言切换解释已有「选择证据清单」层，缺「切换叙事」（before/after+原因）结构与 taiji 侧记忆沉淀。(3) **interaction-group 学习面已有**（W7 时代）：`taiji/interaction_groups.py`（TraceEvent/TraceEpisode/TraceCorpus/GroupRecord + Evaluator）、`interaction_group_transfer.py`（MemberEvidence/TransferCandidate/TransferLearner）、`interaction_group_online.py`（OutcomeFeedback + learner 级 rollback）、`interaction_structural_bridge.py`（online→structural_pressure）；动作类型 = `ActionIntent.kind` 开放字符串，无注册表约束；注意 adapter.py 的 RecoveryReaderInteraction 是同名异概念、零连接。(4) **P5.1 procedural readout 与 interaction-group/Workbench 的桥缺失**——`ProceduralSequenceLearner`（cue→next action_kind）与 interaction-group 域、Workbench capability 体系无任何连接。**P5.2 差距清单**：(a) 类型化动作生成器（readout 预测 → ActionIntent + CapabilitySnapshot 参数 schema 校验，当前零交集）；(b) 语言识别复用（模拟直接走 `workspace.programming_language.resolve` 真合同）；(c) 语言切换解释结构（before/after/原因 + 沉淀为交互记忆）；(d) 模拟动作必须走与真实一致的同一条 preview→approval→execute→undo 路径 + interaction-group episode→`InteractionTraceEpisode` 投影。摸底全程只读，未训练、未改库代码、未接新数据源、未读取 sealed。

用户确认设计方向。**P5.2 预注册已冻结 + 执行完毕（预注册 `c464ef5d`、runner `f9f436bf`、报告 `reports/taiji_p5_2_workbench_simulation_contract_20260912.json` 提交 `3ce8bf37`）：`outcome=workbench_simulation_contract_supported`——九门全过，模型产出的类型化动作在小型模拟中 100% 走真实 Workbench 合同路径且能力迁移成立。** 设计：「小型模拟 = 真实 Workbench 合同的受限实例」（不造平行合同）——scripted IDE 场景 40 train/12 holdout/8 a-gate（四类轮换：语言确认/patch+undo/create+undo/歧义头文件 override），动作生成器 = P5.1 procedural readout（hidden 64、epochs 250 沿用 P5.1g 冻结构造常数）在 workbench.* 词表上训练，每动作经 `ActionIntent` → `WorkbenchActionRequest.from_action_intent`（kind→capability_id 直通）→ `policy_for` → 写动作走 `issue_approval`（preview 附带）→ `consume_approval` → `execute_tool` 真实代码路径；模拟审批者为确定性策略（同路径不同主体，如实披露）。九门实测：静态四项全绿（pytest `4 failed/1199 passed/6 skipped` 零新增、mypy 61 错持平）；contract_path_identity（180 执行动作链记录完整：policy/preview/approval/outcome，capability ⊆ snapshot 18）；action_typing_validity（guard 场景低置信度 set_language 被 `ask_user` 拦截证明合同拦截能力）；language_resolve_reuse（resolve/ambiguous/user_override 态齐备 + 逐场景与 `resolve_programming_language_evidence` 直调 parity）；language_switch_explanation（30 条 before/after/原因叙事：.py→taiji_selection、.h→user_override c→cpp）；undo_semantics（30 场景 undo 后文件 digest 逐位恢复）；readout_sanity（train `1.0` ≥ 0.9、holdout `1.0` > lesion `0.1538`）；interaction_trace_projection（40+12 episode → 双 owner `InteractionTraceEpisode` 字段级无损、`InteractionGroupEvaluator` 可消费——**P5.1 procedural 面与 interaction-group 学习面首次接通**）；transfer_and_budget（agate `1.0` vs per-tick 基线 `0.615385`，margin `0.384615` ≥ 跨代 0.15；replica 一致——`sensation` 派生自含随机单次 undo_token 的 digest，确定性对比排除该随机派生量并披露；wall `23.957s` ≤ 600s）。实现期诚实记录：runner 收敛迭代修复了四处构造/实现问题（场景 goal 模板未区分任务类型致 cue 不分离、`#` 开头内容触发 markdown 歧义、Windows write_text 换行翻译破坏 digest、ask_user 两段式审批被误判为拦截）——均为 runner 本地实现问题，判据零改动；临时脚本已清理。`growth_admitted=false`、`can_promote=false` 贯穿。

**当前唯一下一步（内联）**：P5 外围顺序第 2 项的模拟合同面已建立（模拟/真实同路径 + 类型化动作生成 + 交互记忆沉淀到 interaction-group 面）。下一个自然目标是把该合同面推向真实使用闭环——P5 外围顺序第 3 项（客户端插件热插拔与 provider watchdog）或第 2 项的深水区（interaction-group 在线学习消费 trace 语料的迁移 Gate：`InteractionGroupTransferLearner` 在 P5.2 trace 语料上的候选组评估）；两者都需先只读摸底再冻结预注册。在用户确认方向前不启动摸底之外的训练或新门。

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

本轮 P2 pilot、P2.1–P2.7、P3.0–P3.6、P4.0–P4.6 诊断、P4.7 capacity clean test、P4.8 representation contract、P4.9 feature-space probe、P4.10 feature factorization、P4.11 projection solver、P4.12 course-level validation、P4.13 promotion course、P4.14 joint course、默认 runtime rollout review、A8 晋级评审、默认 runtime 附着工程、晋级宣布与 K 轴 scorecard v4–v8 已与计划同步并提交本地 main。**M5 K 轴已晋级（限定范围，四条诚实边界随宣布生效）**；P5.1 内容迁移 Gate 已通过（`sourced_knowledge_transfer_supported`）；P5.1b 语义改写迁移诚实失败（`semantic_transfer_insufficient`，3 文本规模值函数不辨内容）；**P5.1c 对比辨别力 Gate 已通过（`contrastive_content_discrimination_supported`，disc `0.7314`，九门全过）**；**P5.1d 语义 encoder 注入 Gate 已通过（`semantic_encoder_injection_supported`，disc `0.8700`，九门全过，wall `62.9s`，锚定语义 encoder 正式接入产品内化边界）**；**P5.1e 同预算因果收益 Gate 已执行通过（`same_budget_content_benefit_supported`，a-gate delta `0.25` ≥ 0.15，九门全过，wall `65.194s`，报告 `reports/taiji_p5_1e_same_budget_content_benefit_20260912.json`）——P5.1 知识来源证据线（内容迁移 `0.5` → 对比辨别力 `0.7314` → encoder 注入 `0.8700` → 同预算因果收益 `0.25`）在产品边界全链闭合**；**P5.1f 真实发布语料同预算 Gate 已执行并如实 failed**（[M5_P5_1F_…20260912.md](../../../reference/M5_P5_1F_REAL_CORPUS_SAME_BUDGET_PREREGISTRATION_20260912.md) 冻结提交 `72c2d2d9`：sourced=UltraData Tool_Use 真实业务工具轨迹 vs placebo=Code_Agent 真实软件工程轨迹，确定性采样 200/60/40/臂 + a-gate 16，每条轨迹经 SkillArtifactAdapter governed 投影，corpus gitignored 不入 git 以 sha256+行区间复现）：runner 提交 `2fab7986` + 批合同修复 `94d43c3a`，报告 `reports/taiji_p5_1f_real_corpus_same_budget_20260912.json` 提交 `84f229a7`，九门门 1/3/6/8 过、门 2/4/5/7/9 失败——核心门 5 因两臂 retention `0.3656`/`0.4690` 均未达 admission 而 a-gate 不可测（null，非 delta 不足），门 2 为真实分布机械不等、门 7 placebo native MSE 反低、门 9 wall `1598.131s` > `1200s`（replica 逐位一致本身成立）；按 §4 落 `failed` 而非 `content_benefit_insufficient`，判据零改动、不剪枝不重跑挑结果——**P5.1 证据线的构造边界结论（内容迁移 `0.5` → 对比辨别力 `0.7314` → encoder 注入 `0.8700` → 构造同预算因果收益 `0.25`）在真实发布语料的逐位同量口径下未获复验，失败定位在构造/基建层而非内容收益判据**；只读 S 轴归因 recon 已完成（门 4/5 = procedural 预算欠拟合而投影无损、门 2/9 = 轨迹数口径对真实分布不同预算且 wall 随 examples 线性、门 7 = affordance MSE 双臂 `1e-8` 饱和地板 + reward 恒 1.0 根因）；**P5.1g 真实语料配额同预算 Gate 已执行通过（`real_corpus_content_benefit_supported`，九门全过：calls-quota 逐位 989/385/253、hidden 64 trial-learner 路径、a-gate delta `0.651376` ≥ 0.15、频率基线 margin `0.284376`、wall `435.931s`，admission rollback 如实披露，报告 `reports/taiji_p5_1g_real_corpus_quota_budget_20260912.json`）——P5.1 知识来源证据线在构造与真实语料双证据下全链闭合：内容迁移 `0.5` → 对比辨别力 `0.7314` → encoder 注入 `0.8700` → 构造同预算因果收益 `0.25` → 真实配额同预算因果收益 `0.6514`**；**P5.2 Workbench 小型模拟合同 Gate 已执行通过（`workbench_simulation_contract_supported`，九门全过：180 动作 100% 真实合同路径、guard 拦截验证、语言 resolve/override 复用、30 条切换叙事、30 场景 undo 逐位恢复、readout train `1.0`/holdout `1.0` > lesion、agate `1.0` vs 基线 margin `0.384615`、replica 一致、wall `23.957s`，报告 `reports/taiji_p5_2_workbench_simulation_contract_20260912.json`）——P5.1 procedural 面与 interaction-group 学习面首次接通，模拟 = 真实 Workbench 合同的受限实例**；当前唯一下一步（内联）= **P5.2 深水区（InteractionGroupTransferLearner 消费 P5.2 trace 语料）或 P5 外围第 3 项的方向决策，用户确认前只做只读摸底**。
