# 包 21：读出/发射的零训练分解（单一主线）

> 状态 [当前唯一主线·2026-11-07 立]。依据[全局梳理](SURVEY-PROJECT-STATE_20261107.md)与[包20关包报告](../../reports/taiji_package20_6_package_close_20261107.json)。**0 训练准入**：21.0–21.3 全部为零训练诊断/装配；只有 21.1–21.3 三条结构路径都走完且仍不能解释残差时，21.4 才允许先定价（不得先跑再补价）。confirm 全程封存、0 读取。

## 0. 唯一队首与唯一后继问题

**包 20 已关。21.1 零训练读出分解已结项，判 U 分支（2026-11-07）**——4/4 格 `template_forced` 内容位 <0.40 且相对 `natural` 增益 <0.05，`verdict=upstream_representation_is_first_wall`；CPU 1597.344 / wall 1610.581、0 Adam / 0 confirm / 0 写 checkpoint。报告＋manifest 已入库（[报告](../../reports/taiji_package21_1_readout_decomposition_20261107.json)、[manifest](../manifests/taiji_package21_1_readout_decomposition_20261107.json)），结项校准见[P0 结项卡](../../reports/taiji_package21_1_p0_closeout_20261107.json)。**包 21 已全局收束**（[收束卡](CLOSEOUT-PACKAGE21_GLOBAL_20261107.md)＋[机器卡](../../reports/taiji_package21_global_closeout_20261107.json)，六件齐、8 项结项条件全过）；21.3 判 `representation_axis_closed_for_this_graph`（范围限定判负），21.2/21.4 保持冻结、不自动转训练。**当前唯一待裁定＝下一轮唯一核心问题**（F 的按问句／反馈事件可分可寻址表达能力，以及是否需架构级改造）。本节状态标注不改 §2 任何冻结判据。详见[21.3 结项卡](../../reports/taiji_package21_3_closeout_20261107.json)。

**唯一后继问题（已由 21.1 回答，不自造第二问）**：整句恒 0 的第一次失败发生在**发射/读出**还是**反馈写读权威**？答案落在**上游表示/组合**——把句框喂对后内容位反而下降、整句仍恒 0，故 21.2（R 分支）冻结、21.3（U 分支）为唯一可能的下一步。

**为什么不是训练、不是 U1**：20.3b 已证 oracle F 替换仍 0/22、0 字节变化；20.2r 的 oracle 4.183% loss 同样 0 字节。21.1 的 U 分支进一步把首墙定位在表示/组合而非发射结构。U1/MGW1 是设计层（0 代码、0 参数、0 训练），且 U1 有 P1/P2 两个硬伤未修，**不得进入队列**。

## 1. 输入与冻结

| 项 | 冻结值 |
|---|---|
| 检查点 | `output/taiji_package20_4c_r2_checkpoints/{CF1,B_EH}_{20261105,20261106}_200.pt`（4 个，每个 0.072 GB；全目录 32 个均在） |
| 评测数据 | `taiji.rule_learning_20_1` dev world，索引 160–239；21.1 取固定 16 个索引（`160,165,170,…,235`，每 5 取 1），4 个 checkpoint 用同一集合 |
| 禁止面 | confirm 索引 240–319；Train1 训练索引 0–159 不用于本包读数 |
| 图构建 | `probe_taiji_package20_4c_r2_formal_comparison.build_graph(candidate, gate, arm)`，`addressed_store_enabled=False` |
| 身份 | 不改参数、不改图、不改 `DECODE_RULES`、不写 checkpoint；只读 `subject.state_dict()` |
| 目标字节 | `gold_bytes(world, turn)`；答案恒 27 字节（`ANSWER_BYTES`），无 EOS 停止位 |

## 2. 包 21.1：零训练读出分解（已结项，判 U 分支）

**三个臂（同一 checkpoint、同一 world 序列、同一随机性）**

| 臂 | 做法 | 测什么 |
|---|---|---|
| `natural` | 原样 `batched_turn` | 复现 20.4c_r2 dev 读数（基线） |
| `template_forced` | 只在**模板位**把回喂字节替换为 gold（`previous=gold[p−1]`，`p−1∈TEMPLATE_POSITIONS`） | **给对句框后，内容位能否解出** |
| `content_forced` | 只在**内容位**把回喂字节替换为 gold（`p−1∈DECISION_POSITIONS`） | **给对内容后，句框能否解出** |

**逐格读数**：模板位准确率、内容位准确率、决策位全对率、整句逐字节率、纠正面/反馈后首答分项（沿用 `accumulator()` 口径）。

**预注册判定（先落盘、跑完不改）**

- **R 分支（发射/读出是首墙）**：4 格中 ≥3 格满足 `template_forced` 内容位 ≥0.60 **且** 整句 ≥0.50 → 进 21.2。
- **U 分支（上游表示/组合是首墙）**：4 格中 ≥3 格 `template_forced` 内容位 <0.40 **且** 相对 `natural` 增益 <0.05 → 进 21.3。
- **I 分支（中间/不稳）**：其余 → 报告逐格读数，进 21.4 做**单项**定点诊断（只允许一个假设，不得并行开训练）。

**通过/不通过意味着什么**：R 分支说明「内容已经学到、缺的是结构化发射」——下一步是 21.2 加结构读出，**仍不训练**；U 分支说明「即使句框正确，内容仍错」——F/表示的功劳分配不是首要，21.3 先测表示可分性，**也不训练**；I 分支说明机制读数不稳，只做单点诊断。

**预算与闸**：CPU ≤ **4,800**、wall ≤ **7,200**、0 Adam、0 优化器步、0 confirm、0 写 checkpoint、0 新参数。CPU 到 4,320（90%）即停并如实报 `budget_stop`，不缩数据后重跑。

**停止条件**：任一 checkpoint 缺失或 sha 不符；任一 confirm 索引被触碰；图身份漂移；内存低于 200 MB。命中即停，报 `engineering_stop`，不补跑。

**产物**：`reports/taiji_package21_1_readout_decomposition_20261107.json`、`plans/manifests/…json`（双列 runner sha）、`…_progress.jsonl`（逐行 append＋fsync）；报告 write-then-rename。

## 3. 包 21.2：结构发射装配（仅 R 分支）

- **准入**：21.1 判 R 分支。
- **做法**：把 19.2r `ByteMatchRoute`／19.2w 停止边作为**任务级结构原语**接到 20 图上（0 可训练参数、0 权重迁移；只复用路由/掩码/停止逻辑，不搬 19.2v 权重），再跑 21.1 的三臂。
- **出口**：dev 整句 ≥0.80 且内容位不回退 → 结构上可解，残差归「学习如何被结构引导」；<0.80 → 结构发射不是充分修复，进 21.3。
- **上限**：CPU ≤ 6,000、0 Adam。
- **通过意味着什么**：Task1 式结构能力可以承担发射，训练只负责内容/反馈；**不通过意味着什么**：残差在内容表示，不能再用「加发射结构」掩盖。

## 4. 包 21.3：表示可分性诊断（仅 U 分支或 21.2 未过）

- **准入**：21.1 判 U 分支，或 21.2 结构发射后整句仍 <0.80。
- **做法（全部零训练）**：① 测 22 个纠正面 F 的逐询问键可分性（余弦分布、top1–top2 边距、秩）；② 用问句条件化键做闭式读出，与「窗口身份」和「标签打乱」两条最强对照比；③ 测逐轮写入与按 release 写入的 F 向量数差异。**不实现新图、不训练、不定价。**
- **出口**：闭式读出在纠正面 ≥0.60 且比最强对照高 ≥0.20 → 表示有权威，下一步是「把闭式读出变成图上可学读出头」的**定价**；否则表示轴在当前图/数据上判负（登记 `representation_axis_closed_for_this_graph`），转 21.4。
- **上限**：CPU ≤ 2,000、0 Adam。
- **预注册（2026-11-07 冻结，准入后但未执行）**：[冻结规格](SPEC-PACKAGE21_3_REPRESENTATION_SEPARABILITY_PREREG_20261107.md)＋[机读 manifest](../manifests/taiji_package21_3_prereg_20261107.json)（判据唯一机读源，附 `criteria_sha256`）。本节的门值、做法与预算**一字未改**；规格只裁定口径歧义（「最强对照」取 20.3a 已冻结的 per-seed max）、补四条对照并集与有效自由度闸门、钉死 4 checkpoint 最严判定，并登记「判负／不可判不自动转 21.4／训练」的转段冲突（见规格 §3.6）。判据先落盘、跑完未改；读数已按合同执行完毕，判 `representation_axis_closed_for_this_graph`（见[报告](../../reports/taiji_package21_3_representation_separability_20261107.json)与[结项卡](../../reports/taiji_package21_3_closeout_20261107.json)），不自动转 21.4 或训练。

## 5. 包 21.4：训练（最后手段，未准入）

- **准入**：21.1 判 I 分支，或 21.3 证明表示有权威但尚未在图上实现。
- **必须先做**：一次性定价（单位成本用 20.4b 的 56.906 CPU/更新、按真实图重测），写明 updates、seed 数、臂数、停止条件，落盘后才可开跑；超闸即停报，不得先跑再补价。
- **禁止**：以「多训一点」替代结构修复；以新臂/新剂量续命已关的包 20。

## 6. 退役与清理（本次演进后生效）

| 对象 | 处置 |
|---|---|
| 20.3a 每轮内容寻址写（`addressed_store_enabled`） | **退役为机制**：三闸已败（杠杆 8.7e-6、`byte_diff=0`、信息 0.4091）；保持默认关闭，后继不得直接启用；报告保留 |
| 20.2t 选项 3（批次写＋衰减累积） | 设计已判不过结构闸，**退役**，不再列为候选 |
| 20.x 诊断脚本（20.2r/2s/2t/20.3a/3b/4a/4b/4d） | 已并入 20.6；按既有归档惯例处理，证据字节不改 |
| `output/_peek_20_4c_r2.py` | 中间窥探脚本，**删除**；P0 核验 `ABSENT`，回执见[P0 结项卡](../../reports/taiji_package21_1_p0_closeout_20261107.json) §退役清单 |
| `output/taiji_package20_4_checkpoints`（1 个） | 20.4c 作废残留，标 `superseded`，不参与任何评测；P0 按账本 §98 出[旁挂声明](../manifests/taiji_package20_4_checkpoints_superseded_20261107.json)（含 `preflight.pt` sha256），可作冻结证据 |
| U1 / MGW1 规格 | **冻结为设计资产**：不占队列、不训练、不签能力；U1 在 P1/P2/P3/P4 修好前不得作为实现依据 |
| 能力卡 §7（数据前提三选项） | 标 `superseded`（18.0 已按方案 A 执行） |
| 账本 §15.1（起点门读数对象待裁定） | 标 `superseded`（19.2b / D16-5 已处理） |

## 7. 顺序与单一建议

`21.0 冻结 → 21.1（已结项，判 U）→ (U) 21.3（已结项，判负）/ (R) 21.2、(I) 21.4 冻结 → 包 21 全局收束（已完成）`。**只给一个下一步：由 owner 裁定收束卡给出的下一轮唯一核心问题。** 21.2–21.4 不预启动、不并行；准入 ≠ 执行 ≠ 里程碑关闭，判负不自动转训练；包 21 收束不等于 M 线关闭，未裁定前不新开包。

## 判据、出口与验收（索引）

本计划的冻结判据与出口集中在 §2（21.1 三臂的 R/U/I 预注册判定、预算与停止条件、产物规范）、§3（21.2 出口）、§4（21.3 出口）、§5（21.4 准入与禁止）、§6（退役与清理）。判据先落盘、跑完不改；改动须新报告并保留原块审计（D16-38）。五类有限出口：通过／判负／不可判／工程失败／预算停止，具名 `verdict` 写入机器报告。**本节只作索引，不改上述任何判据文字、门值或预算**（§0／§7 的 21.1 结项状态标注与 §2 标题括注属状态同步，不触及判据正文）；21.1 报告＋manifest 已于 2026-11-07 由 P0 结项校准入库（[P0 结项卡](../../reports/taiji_package21_1_p0_closeout_20261107.json)），可作冻结证据引用，此前「在途回执不得当冻结证据」的限定同时失效。21.3 的预注册判据另见[冻结规格](SPEC-PACKAGE21_3_REPRESENTATION_SEPARABILITY_PREREG_20261107.md)与[机读 manifest](../manifests/taiji_package21_3_prereg_20261107.json)；本节 §4 的冻结门值、做法与预算未因该预注册改动；21.3 的[结项卡](../../reports/taiji_package21_3_closeout_20261107.json)记录执行读数与判负出口。
