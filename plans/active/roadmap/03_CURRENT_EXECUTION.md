# Seed / Taiji 当前详细推进方案

> 当前裁决（2026-09-17，用户要求项目收束）：停止扩展及新训练，不自动接续。H3.7B已结项，H3.8 v1/v2按预承诺停止投入；最新P3b-v2进程已停止，现有checkpoint保留但缺完整结果报告，状态为“中断、未判定”，不是训练或能力通过。完整身份、证据和缺口见[项目收束记录](../../reference/PROJECT_CONSOLIDATION_20260917.md)。M5/R2未晋级，Mini后置，VISION不作为前置。

## 当前唯一下一步：现有P3b-v2产物只读核验（暂停，待用户重新启动）

2026-09-17结果整理已完成：[结果与结论总览](../../reference/PROJECT_RESULTS_AND_CONCLUSIONS_20260917.md)。原始H3.8 v1报告的逐seed优势为1/3（纠正历史摘要0/3）；`workspace_unused`为协议停止类别，不等于所有seed均无作用；“prefix不训练所以回答监督不存在、失败必然”不再作为已证根因。K轴限定成果、局部协作、工程修复与语言能力分别记账。本次仅整理，不解除暂停、不启动训练或评测。

先核验身份、恢复和目标格式是否符合冻结合同，再决定是否能按既定入口评分。不得追加训练、改阈值或用旧checkpoint冒充中断run的结果。本次仅收束，不自动执行此步骤。

## 历史工作包台账（以下旧“下一步”不再生效）

> 完成（2026-09-17，a24f8763）：两臂工程验证均 `passed_engineering_only`（[A legacy_target](../../../reports/r2_h3_7b_validation/legacy_target/report.json)、[B byte_aligned](../../../reports/r2_h3_7b_validation/byte_aligned/report.json)）——corpus digest 与冻结 fixture 一致、有效参数 277,970 ≤ 300k、checkpoint 往返与 factorized preflight 通过、零步与训练后 checkpoint 均新进程复核且一步续训 digest 与进程内探针一致、每臂 12 episodes/537 byte 更新无非法值；final/dev 按设计未评分，checkpoint .pt 留本地（digest 已入报告）。
>
> 完成（2026-09-17，768cca14）：合同「唯一后续」第一半——[可学习性与残余信用一致性检查](../../../reports/r2_h3_7b_validation/learnability_analysis.json)（只读）：两臂均 train 可学习（legacy surprise 5.619→4.138、byte_aligned 5.619→4.276，各 12/12 episodes 改善；byte_aligned 降幅略小 = 确定性监督坐标差异，非能力主张）；四相有限差分与 post-target-update-prior 测试现场复跑全过。剩余：冻结带明确问题与停止线的 dev 验证包预注册。
>
> 完成（2026-09-17，d6e15460）：**H3.7B 包闭合**——[dev 验证包](../../../reports/r2_h3_7b_validation/dev_legacy_target/report.json)（冻结于合同 Dev 验证包节，两臂各 2 epochs/1074 更新，preflight 与 fresh-process 全绿）⇒ **停止线 ③ 触发**：byte 对齐 16-byte 窗口几何在 dev 上无优势（surprise Δ −1.4342 vs legacy −1.4344；accuracy Δ +0.177 vs +0.207），几何问题回答为**负**；两臂 train→dev 迁移均非零（停止线 ② 未触发）。final 未读；dev 结果仅诊断修复效果。**修复包结论**：修复的价值在一致性本身（prior/信用/恢复），几何更换无 dev 收益；H3.7B 按停止线诚实收口，不扩预算。
>
> 完成（2026-09-17，9bd7df00）：**六维有界只读归因复审全部完成**——[归因总账](../../reference/M5_R2_H3_9_ATTRIBUTION_LEDGER_20260917.md)补齐 renderer/prefix representation/capacity-data 三维并与 H3.8 §1 合并。核心结论：监督目标是 response 自身的 count-sketch 且前缀零学习（`_prime` learn=False + `_count_sketch_chunk`），**prompt→response 内容依赖在监督信号中结构性缺席**——H3.7/H3.7B 全部负结果是该监督结构的必然输出；renderer 无跨 byte 可学习内容状态；数据处于记忆化区间。归因输入已完整，**H3.8 隔离原型（可训练 prefix 编码器 → 内容工作空间 → 可训练循环 renderer、真实下一 byte 交叉熵）的采用决策移交用户**（H3.8 §7：不能从继续推进推断为默认核心迁移授权）。
>
> 决策与冻结（2026-09-17，ea8d1ec6）：**用户答复「采用」**——[H3.8 隔离原型实现合同](../../reference/M5_R2_H3_8_ISOLATED_PROTOTYPE_CONTRACT_20260917.md)冻结：计算图逐条可测（可训练因果 prefix 编码器 → episode 内凝固的内容工作空间 W → 可训练循环 byte renderer，真实下一 byte 交叉熵含结束标记，v1 无辅助损失）；**五项实现门**（隔离清单/梯度数值差分/因果 mask 与 teacher-forced 等价/零步原子保存+fresh logits 一致+一步续训/接线合成检查）先于一切能力训练；数据合同（新语料五类对抗形状，旧 fixture 降级工程回归）与两臂匹配预算（基线 vs workspace）；数值预算在实现门通过后单独冻结。**唯一下一步：实现原型骨架 + 五门测试入套件。**训练授权沿用，dev 训练待实现门+数据合同+数值预算三事齐备后启动；`growth_admitted=false`、`can_promote=false` 贯穿。
>
> 完成（2026-09-17，a41a5684）：**H3.8 原型实现与五项实现门全部通过**——新模块 [sequence_workspace.py](../../../taiji/sequence_workspace.py)（14 项参数清单、105,057 参数；因果 prefix 扫描 → episode 内凝固 W（softmax 内容寻址）→ 可训练循环 renderer → 257 类 next-byte 交叉熵含结束标记；detach/mask/reset/超长拒绝四处全部标注；确定性 backprop trainer，optimizer 状态克隆防别名、完整恢复合同）+ [11 项门测试](../../../tests/taiji_native/test_sequence_workspace_contract.py)（含数值有限差分、fresh 进程一步续训 digest 一致、缺参数与篡改拒绝、打乱 prefix/清零 W 均改变输出）+ [门报告](../../../reports/r2_h3_8_implementation_gates_20260917.json)：五门 + scoped ruff/black/mypy 全绿，`outcome=passed`。未做能力训练、未动默认入口与既有 checkpoint。**下一动作（合同 §6 第 3 步）：数据合同（五类对抗形状、模板/实体/组合分离、独立 final）+ 小规模 train-only 可学习性检查 → 数值预算冻结成文**，然后才进入 matched dev。
>
> 完成（2026-09-17，90d24464 + 7d1829ac）：**合同 §6 第 3 步完成**——[数据合同](../../../reports/r2_h3_8_data_contract_20260917.json)（生成器 [build_taiji_r2_h3_8_corpus.py](../../../scripts/training/build_taiji_r2_h3_8_corpus.py)，fixture digest `5518ff50…` 已入库 `7d1829ac`）：train 25 / dev 16 / final 16，七项分离检查全真（实体池、模板措辞、(模板,实体) 组合、前缀集合三向不相交；每 split 五类对抗形状齐备）+ [train-only 可学习性](../../../reports/r2_h3_8_train_learnability_20260917.json)：原型新增合同要求的 **no-workspace baseline 臂**（独立清单 65,073 vs 105,057 参数；按名确定性初始化使两臂共享张量初值相同，差异仅归 workspace 路径）；两臂在冻结共同学习率 0.01 下均可学（workspace 5.566→0.349、baseline 5.560→0.303）；**报告内自带优化稳定性发现**：workspace 臂在 0.05 全预算下发散（loss 9.8552 > 起点 5.5663）而 baseline 容受该速率 ⇒ 共同 LR 冻结为 0.01 而非按臂调参。实现门报告已随基线臂门重跑，仍全绿。dev/final 未读。**下一动作：数值预算冻结成文（合同 §4：种子/步数/墙钟/存储上限/checkpoint 周期/恢复策略）→ 然后 matched dev 预注册（指标/阈值/统计规则在看到 dev 数据前冻结）。**
>
> 完成（2026-09-17，12381e1c + 55f7bf62）：**matched dev 已执行（预注册冻结 → 三 seed × 两臂 + 病灶列，75 s，12 次新进程 checkpoint 复核全绿，final 未读）⇒ `outcome=workspace_unused`**——[报告](../../../reports/r2_h3_8_matched_dev_20260917.json)：G-fit 过（两臂 train 0.81–0.95），但**病灶落差三 seed 为 −0.0283/−0.0472/+0.0472（不全为正）**、win 门 0/3、均值差 −0.0157（workspace 0.6006 vs baseline 0.6164）⇒ 零化 W 不持续损害 dev ⇒ **内容寻址路径收到梯度却从未成为承重路径**；closure-retrain 对照按预注册**不启动**（需先过 win 门）。**归因（计算图级）**：`renderer_start` 让 prefix 经 h0 直连 renderer 起点，renderer 全程可绕过 W —— 设计意图「renderer 用自身状态对 W 做内容寻址」当前只是**并列通道**而非**唯一通道**，故 W 可被训练忽略。**这是计算图修订信号，不是数据/epoch 信号**（合同 §4）。修订方向：① 单一前缀通道；② 参数对齐对照；③ 乘法门控（备选）。
>
> **H3.8-v2 闭合（2026-09-17，95547d0a → 881819c0 → 2edbcb67 → 12748e69）**：v2 修订预注册冻结（合同 §7：单一前缀通道——`start_vector` 学习常量起点、prefix 只能经 W 读入；dev 预注册 §7：判据/预算零改动，checkpoint 版本 1 拒读）→ 实现 + 单通道结构门（W 清零 ⇒ logits 逐位前缀不变，运行时复查先行）→ 五门实现门全绿 → v2 train 可学习性双绿（5.508→0.320 / 5.559→0.302；附带发现：0.05 稳定探针不再发散，v1 结论限定于 v1 图）⇒ matched dev v2（[报告](../../../reports/r2_h3_8_matched_dev_v2_20260917.json)）⇒ **仍 `workspace_unused`**：病灶 +0.0566/+0.0377/−0.0377（**2/3 转正** vs v1 的 1/3——修订在预测方向确实推进，但门要求逐 seed 全正）、win 均值 +0.022 < 0.03、boundary 全过。**按预承诺（合同 §7.6 / dev §7.4）收口：唯一通道图中病灶不全正无法用旁路解释 ⇒ 归因升级至 W 生成路径本身，H3.8 架构族停止追加预算，回到候选重议**（「两 seed 转正」观察随总账留档供重议）。R2 主线现处**候选重议决策点**。

> 更新：2026-09-17；R0证据与门禁审计已完成。本次决定将R2整模型语言能力设为当前主线；结构化R2入口、checkpoint前置、P1 developmental 对照、P2序列级只读评价、G1条件接口、H2/H3内部表示/读出审计、H3.1序列路径对照、H3.2 response-start候选读出、H3.3泛化控制与response-phase候选、H3.4逐位置条件信用审计、H3.5目标合同、H3.6-A target encoder plumbing及H3.6-B matched dev/bridge ablation均已完成；H3.6-B按预注册在final前负结果结项；本轮H3.7分解式回答工作空间与因果信用合同的隔离实现、preflight、三seed dev、消融与aggregate也已完成并在final前停止，训练权限不再扩展到同质补训或final读取，当前按固定归因顺序复审。
> 本文是唯一执行顺序来源；[01](01_SCOPE_AND_PHASES.md)管总阶段，[02](02_GATES_AND_CI.md)管晋级，[07](07_MINI_MODEL_DELIVERY.md)管整模型验收。
> 旧逐轮台账完整保留于Git的56a4c3e4:plans/active/roadmap/03_CURRENT_EXECUTION.md及各冻结报告。本次不改历史结果、不修改默认入口、不授权产品采用或架构切换；R2隔离smoke只用于验证新入口。

## 历史决策点：R2 候选重审议（后续已有P3b-v2工作，非当前执行指令）

> **2026-09-17 收敛**：H3.8（joint-sequence credit / workspace 机制）**已按预承诺结项，该架构族停止投入**
> —— v1 与 v2（单一前缀通道）两次 matched dev 均落 `workspace_unused`，lesion 非全正 ⇒
> 在单通道图里不能用 bypass 解释 ⇒ 归因升级到 **W-generation 路径本身**（见
> [H3.9 归因总账](../../reference/M5_R2_H3_9_ATTRIBUTION_LEDGER_20260917.md)）。
> **R2 主线因此停在「候选重审议」点**：下一步需要用户对候选方向做选择，而不是继续追加同质训练。
>
> 同日的 CAP-0 身份重锚（`2dec6145`）已把 git head / checkpoint sha256 / eval-set sha256 绑进报告
> 的 identity block ⇒「无法复用旧身份」的缺口闭合；但**语言缺陷在当前身份上依旧完全存在**
> （C 0/14、D 0/16、E 0/20，B/G 待人工复核，A/F/H not_executed）。
>
> 本文件仍是**唯一执行顺序来源**；[01](01_SCOPE_AND_PHASES.md) 管总阶段，
> [02](02_GATES_AND_CI.md) 管晋级，[07](07_MINI_MODEL_DELIVERY.md) 管整模型验收。

## 1. 项目位置与推进目标

当前仍在M5知识与身体研究。M0–M3限定资产和K轴批准继续继承；M4.V2结构成长未完成；M8为条件支线，不能阻塞CPU能力验证。

推进目标不再用实验编号计数，而是：

1. 同一训练态能够真实加载、原生生成，达到基本问答/上下文能力，并在同一入口展示核心能力。
2. 真实outcome更新改善后续行为，同时保持、幂等、恢复、撤销与预算成立。
3. 研究结项、能力轴晋级、M5退出、用户验收与发布分别审查，不互相替代。

不立刻交聊天演示。L3达标就交用户自由提问，不等待完整VISION或成长；模板回答、动作ID和外部代答不替代原生语言能力。

## 2. 最新证据与主张边界

| 对象 | 当前事实 | 边界与缺口 |
|---|---|---|
| K轴 | [限定晋级](../../reference/M5_K_PROMOTION_DECLARATION_20260912.md)已于9月12日批准 | 五类合成载体、opt-in、进程内范围；不代表真实语料准入、默认采用或成长 |
| B0/B1 | 评分/可达性审查、HANDOFF-M4 gate落地、B1表示门已完成 | 不再重开早期D1–D5；gate实现不等于产品核心实现 |
| B2-v4 | [冻结协议](../../reference/M5_B2V4_COLLABORATION_JUDGMENT_PREREGISTRATION_FROZEN_20260916.md)，28a1cf02记录九门通过；a+c增益2.0≥1.65，逐格6/6、交错和lesion成立 | collaboration_supported只指冻结机制＋v3选择程序＋create族内；轴未独立晋级、产品未采用、跨内容结构未证 |
| P5.2d v1 | [已提交报告](../../../reports/taiji_p5_2d_online_writeback_20260916.json)失败；成功反馈被10.0资源上限拒绝 | 找到信号不是完成在线学习；未触达验收不算通过 |
| P5.2d本地v2 | ~~未跟踪~~〔更新 2026-09-16〕§8 已同步并重跑，报告现随 commit f89d7a0d 入库（**重跑覆盖了 R0 审计引用的未跟踪 v2 报告文件**——审计结论以 R0 文档留痕为准，特此披露）；§7 预算校准（64）+ §8 仪器修正后 outcome 字段为 supported | **不晋级（R0 判定维持）**：A2/A3/A5 经修正仪器后可判且通过（record 集 1→1、重放拒绝+digest 不变、恢复 digest 逐字节一致），A4/A6 过；但 **A1 实质未变**——updated_pair 仍 b+c、后测收益 0，仅预测排序首位，与 R0「A1 假阳性」判定一致；选择校准（首次 admission 后残差不确定性超 maximum_uncertainty=2.0）是 A1 的真实缺口。并行债务已了结，不阻塞 R2 主线 |
| v2审查线索 | updated_pair仍b+c、六对预测全1.291667、后测收益0，但a1=true；a2/a3/a5/g1失败；恢复及rollback说明与细节需核对 | 〔更新 2026-09-16〕§8 同步重跑后：a2/a3/a5/g1 已可判且通过（g1 采样时机修正）；a1 的假阳性形态确认——根因 = 有界 select 未随准入改变（不确定性边界）+ 增益比较因选择未变而平凡成立；恢复与回滚细节已在 v2 报告 recovery_detail/rollback_detail 落盘并核对一致 |
| CAP默认入口 | [基线](../../reference/M5_CAP0_BASELINE_RESULT_20260915.md)：原tick=2入口去回显后C/D/E为0；B/G辅助判断待人工确认；〔更新 2026-09-17，2dec6145〕[身份重取](../../../reports/taiji_cap0_baseline_v1_20260917.json)绑定 git HEAD+checkpoint/评价集 sha256，当前身份复现 C 0/14、D 0/16、E 0/20 | ~~不能沿用旧身份~~ 身份缺口已闭合；B/G 人工盲审与 A05/H 标定仍待；语言缺口=完全缺失（当前身份确认） |
| P3b已提交材料 | [阶段结果](../../reference/M5_P3B_RESULT_20260916.md)：唯一共同tick17M，C/D差0，E差+0.05，未检测到该分辨率下效应 | 不等于分布无关或架构无效；J4 A/H及人工安全分支缺证 |
| P3b工作区终态 | treatment记录finished_at、campaign_stop=regressed；18M persistent；对照17M material；R0已核实两臂停止 | 数据分布效应只有一个共同tick，仍not_resolved；18M原始输出可观察但C/D/E未达标，保护隔离仍有DEBT-I7 |
| 本地heldout | 未跟踪reports/taiji_p3b_heldout_surprise_20260916.json为not_resolved | 先审协议与血缘；surprise不能替代对话评价 |
| 知识/身体 | P5.1g仍trial并回滚；Workbench合同资产存在 | 真实语料child未准入、身体全生命周期未结项 |
| CI | 9月16日查询run34869725409，d09dcc21，总体failure；3.10失败，3.12/Windows等通过 | 不是当前HEAD的CI；不称全绿，旧27项SystemExit不再视作未定位代码缺陷 |
| R2 aligned language seam | 结构化episode、response-only native readout、checkpoint digest/atomic save、zero/child恢复前置、paired诊断、static/slow/fast/fast_slow四臂、P2序列级只读评价、H3.1 beam、H3.2 response-start、H3.3 response-phase与泛化剖面、H3.4逐位置信用审计、H3.5表示合同、H3.6-A target encoder plumbing及H3.6-B零步前置均已完成；H3.6-B六个正式dev run与三组bridge ablation已完成；H3.7实现门、三seed dev、三组消融与aggregate已完成并在final前停止；〔更新 2026-09-17〕H3.7B 修复包三段闭合；H3.9 六维归因总账完成（监督结构性缺席 prompt→response 依赖）；H3.8 隔离原型经用户采用：实现+五门+数据合同+预算冻结+matched dev v1/v2 均 `workspace_unused`，按预承诺停止架构族追加 | H3.5-A/H3.6-B/H3.7/H3.7B几何/H3.8 v1/v2 六连负在 final 前收口；v2 病灶 2/3 转正（方向被证实但不稳定）留档；**R2 现处候选重议决策点（用户裁决）**；CAP-0 当前身份复现语言缺口完全缺失 |
| R2-G1 conditional response v2 | v2显式policy prefix、版本化checkpoint/corpus、train/dev/final输出碰撞指标和G1-S0/S1受控诊断已实现 | preflight通过；20 epoch混合训练的train collision=0.5、train exact=0.5，dev/final sequence criterion=0，paired改写敏感性=0；H2/H3审计显示train prefix context distinct=2/2、context L2=0.9694、next-byte probability L1=0.0771、argmax difference=0、recovery repeatable=true；输入已进入native state，但readout首选路径未形成可分离margin，不扩大同质训练 |

未提交结果转正须有：代码/协议版本、执行时间、产物hash、数据与模型血缘、原始结果、失败说明、独立输出路径。提交本身不证明有效；已见结果不得包装成前瞻预注册。

## 3. 总依赖与优先级

| 包 | 目标 | 硬前置 | 出口 |
|---|---|---|---|
| R0 | 结项与门禁一致性 | 只读清点、不改实验面 | 决策简报与逐门问题账 |
| R1 | 可信底座与主线硬前置 | R0封存；仅处理阻塞R2的加载、保存、隔离门 | 需要的隔离、加载、保存恢复证据 |
| R2 | 整模型语言能力当前主线 | R1相关硬门、新目标协议与训练授权 | L2，同bundle原始回答 |
| R3 | 在线适应闭环 | R0归因＋R1相关硬门 | 六类验收与后测收益 |
| R4 | 代表能力集成 | L2＋一项核心能力准入 | L3用户验收候选 |
| R5 | 独立扩面、知识准入、身体 | 各自设计批准和独立数据 | M5各轴候选评审 |
| R6 | M5退出、M6采用、M7发布 | 各轴和共同门齐备 | 明确范围的受控交付 |
| V | 未来高上限架构 | 单一VISION、独立合同决策 | 最小可证伪原型，不自动替代主线 |

R0已关闭状态不清问题。2026-09-16作出主线优先决策：R2先推进语言目标、信用分配与可验证训练设计；P5.2d修正仪器保留为并行债务，不再成为当前主线的前置。R1只处理实际阻塞R2的硬门，不无限清债；R3在线适应在R2形成可对话能力和必要隔离后再推进。R5不能无限推迟首次见模型，但M5整体退出仍须必达项。L3用户验收与M5退出不是同一个门。

## 4. R0：结项与门禁审查

**目标**：分清已结束、已失败、待解释、待批准。只读审查不启动runner。
**状态：已结项（2026-09-16）**。下列1–7项是本轮审计的范围与留痕，不是当前待执行清单；可复用的后续动作已经转入R1硬前置、R2主线或P5.2d并行债务。

1. 固定HEAD、差异清单、输出时间/hash与相关进程状态；仍变化文件只作带时间快照，不据此定稿。
2. P3b遵循[结项预案](../../reference/M5_P3B_RESULT_REPORTING_PLAN_20260915.md)补齐A–D表：两臂停止、共同tick、J1–J5、人工项、保护checkpoint前后摘要与未测维度。
3. 对protected_checkpoints_unchanged=false逐项定位影响对象：不忽略变化，也不无证据宣布全部实验作废。
4. P5.2d建“协议条文→计算函数→输入→原始值→布尔→主张”映射，优先审a1后测收益、a2保持、a3重复与过期分离、a5完整恢复、g1真实干预。
5. 核对预算64何时修改、是否执行前冻结、数据已曝光程度；不完整就标探索性，下一次另立独立验证。
6. 问题分类：仪器缺陷候选、机制缺陷候选、协议不一致、证据缺失。没有最小复现不定根因。
7. 决策包区分可直接修的实现错误、必须改合同的选择、新训练与预算授权。

**出口**：状态无矛盾；每个失败有证据位置和责任包；用户进行中文件不被改写。
**停止点**：保持定义、重复写入语义、预算、加载策略或学习机制变化，先说明选择、收益、风险再批准。

### R0审计结论（2026-09-16）

[完整证据与门禁审计](../../reference/M5_R0_EVIDENCE_GATE_AUDIT_20260916.md)已经完成。结论不是“在线机制通过”：预算64只让第4轮真实反馈进入learner；v2的A1通过是由“零收益也可通过、并列最高也可通过、未要求更新选择等于a+c”造成的假阳性。A2/A3/A5/G1分别受全局重拟合、stale优先级、恢复ID切片和rollback后计数影响，当前报告不能把它们归因为模型或库机制失败。A6的恢复/墓碑/环境undo有局部证据，但replay实际返回stale，不能按冻结协议宣布墓碑拒绝已通过。

P3b两臂已经结束：对照17M material停止，实验18M persistent停止；只有17M共同检查点，因此数据分布主效应仍不可判。实验18M确实产生了训练态原始输出，但C=0、D=0.0625、E=0.10，B/G待人工复核，A/F/H未测，不能触发L2/L3或用户验收。

R0关闭的是“状态不清”，不关闭P5.2d、不晋级在线轴、不授权第二次P3b。预注册§8是待实现的修订设计，不是本次v2的执行条件；历史v1/v2报告保持原样。

### 主线优先决策（2026-09-16）

用户决定先推主线。决策含义是：

1. 当前唯一主线转为R2整模型语言能力；第一步是冻结语言目标、数据结构、信用分配和整模型评价的设计，不是立即追加同质字节训练。
2. P5.2d v2的A1/A2/A3/A5/G1/A6仪器修正不被删除、不改写已有报告，也不继续占用当前主线；它登记为并行债务，只有在在线轴需要重新取证或资源明确时再执行。
3. P3b的真实输出证明训练态能够产生字符序列，但乱码与C/D/E低分说明“有输出”尚未等于“能回答”。因此R2必须同时解决编码合法性、回答条件化、上下文保持和结果信用，不能只把解码器或可读模板当作能力修复。
4. 在新目标协议冻结、checkpoint保存/恢复前置检查通过、训练权限明确之前，不启动新训练，不修改默认模型入口，不把VISION候选架构直接写入产品。

详细设计见[R2语言目标与信用分配设计](../../reference/M5_R2_LANGUAGE_TARGET_CREDIT_DESIGN_20260916.md)。

### Mini模型验收后置决策（2026-09-16）

用户明确要求先把主线推完，暂不推进Mini模型验收。这里的“后置”只改变当前执行顺序，不删除评价体系、不撤回L2/L3标准，也不把最终用户验收改写成可选项：

1. 当前工作集中在R2目标对齐训练、原生语言读出、成熟发展训练与运行期原生持续适应的架构落地。
2. 07整模型评价文档、CAP题集和L2/L3门继续作为后置验收合同保留，但不再作为当前R2实现的阻塞工作。
3. 当前不做Mini演示、不做用户自由提问、不反复跑CAP-0；只有R2训练链完成一个可加载的主线bundle后，才统一做后置验收。
4. P5.2d修正仪器仍是并行债务；Mini后置不意味着在线适应提前占用主线资源。

## 5. R1：可信实验底座

| 项目 | 具体实施要求 | 验收 |
|---|---|---|
| 产物隔离 | 优先处理DEBT-I7：测试默认进临时目录，生产checkpoint只读，前后hash守卫 | 隔离前不跑可能写产品状态的全量套件 |
| legacy加载 | 依[决策简报](../../reference/CAP0_LEGACY_LOADER_DECISION_BRIEF_20260915.md)批准显式迁移；现代缺件继续拒绝 | 迁移清单、来源完整、无静默随机初始化；加载成功不算语言成功 |
| 仪器同步 | 迁移与inventory/legacy-load新报告、DEBT-I5行号断言清理同批 | 测试实际调用新行为，不只读取历史报告 |
| 原子报告 | DEBT-I6：完整临时写后替换；复用前校验schema/tick/chain/题序/不学习标志 | 半写文件被识别；仅同tick快照可重评，缺快照拒绝 |
| 训练保存 | 零步保存→新进程恢复→同输入对比→学习一步→再保存恢复 | 模型/学习器/RNG/计数/pending/父子血缘/拒绝与tombstone均覆盖 |
| 资源与恢复 | 磁盘余量、最大checkpoint尺寸、中断注入；保留父快照 | 不覆盖准入模型；环境undo与模型rollback独立通过 |

加载选择：显式兼容迁移能恢复真实训练态，推荐；仅改外壳的重导出不能绕过结构不匹配；显式降级可诚实说明不能服务，但不解决能力。涉及核心兼容语义仍需决策，不能由本计划自动实施。

## 6. R2：整模型语言能力与高上限设计

继承[已有语言监督诊断](../../reference/M5_CAP0_P1_LANGUAGE_SUPERVISION_DIAGNOSIS_20260915.md)及P3a/P3b，不重做已完成P1。P3b不可判不足以证明架构失效，也不支持直接重复加训。当前R2的设计合同、根因排序、目标对齐方案、信用分配和晋级条件集中记录在[R2设计文档](../../reference/M5_R2_LANGUAGE_TARGET_CREDIT_DESIGN_20260916.md)。结构化入口已经落地为可执行代码和隔离预注册；本轮结果仍按“链路证据”和“能力证据”分账。

### 6.0 当前主线工作包

R2-D0设计合同已经转为实施，当前工作包分成四个连续出口，不把“脚本跑完”直接当成语言能力：

1. **D0结构化目标**：P2历史版本以`taiji-native-language-alignment-v1`固定system/context/history/user/assistant/end边界；G1当前版本升级为`taiji-native-language-alignment-v2`，在同一native边界中显式加入policy，JSONL仍按episode、family和split做内容寻址隔离。已完成版本化实现；实现位于`taiji/language_alignment.py`，P2预注册位于[M5 R2 aligned language pilot预注册](../../reference/M5_R2_ALIGNED_LANGUAGE_PILOT_PREREGISTRATION_20260916.md)，G1预注册位于[M5 R2-G1条件回答接口预注册](../../reference/M5_R2_G1_CONDITIONAL_RESPONSE_PREREGISTRATION_20260916.md)。
2. **P0保存恢复前置**：零步保存、新进程恢复、一次response更新、child恢复、parent保护和atomic save。已通过；任何正式pilot仍必须由入口自动重跑。
3. **plumbing smoke**：用四个受控episode验证训练入口、报告、原生读出和dev/final记录。已完成；受限UTF-8读出修复后dev/final合法率与无替换字符率均为1.0，但exact response为0，结论只能是S1与链路通过、S2未通过。
4. **paired目标诊断**：同一zero/child对照和原题/改写题四格读出已完成；输出对输入敏感且局部surprise下降，但exact response没有迁移，按停止规则不能直接扩大同质byte目标。
5. **R2-P1目标/读出升级**：把response-only字节信用接入已有fast/slow developmental F1状态，明确slow发展训练、fast运行期候选和边界replay巩固的边界；同一paired诊断下static/slow/fast/fast_slow四臂均已跑通，但未产生exact response收益，P1结论为“owner接线成立、目标不足”。
6. **R2-P2序列级目标与语义评价**：已固定回答边界、必需内容、禁止内容、未知策略和原始生成回传；`LanguageEpisode`支持`required_terms`、`forbidden_terms`和独立的`unknown_markers`，报告保存raw bytes hex、stop reason、sequence validity、semantic proxy和exact response。四臂只读核对通过，semantic规则没有进入fast/slow更新，避免把规则评分冒充语言能力。
7. **R2-G1条件回答接口/读出升级**：G1只把`unknown_policy`显式纳入同构prefix，`task_family`留作分层元数据，完成v2 response interface、版本隔离和训练样本之间的输出碰撞诊断；20 epoch有限过拟合仍出现50% train collision，dev/final sequence criterion为0，paired改写敏感性为0，接口升级未解决输出条件化。
8. **R2-H2/H3内部表示与读出owner审计**：已完成只读审计。20 epoch child 的train prefix context保持2/2 distinct，context pair L2=0.9694、cosine distance=0.0294，下一字节概率L1=0.0771、JS=0.00467，但argmax difference=0；原题改写的context L2=1.0004，probability L1=0.0797，恢复重复性为true。结论是输入条件已进入native predictive state，当前失败更接近readout margin/序列路径，而不是输入完全被压平。
9. **R2-H3.1原生读出/序列路径对照**：已完成只读对照。在同一protected native owner上保留greedy并加入bounded beam；beam改变了训练样本的候选输出，但只是把两个样本的正确/错误归属互换，collision和exact/sequence聚合均没有收益，恢复重复性成立。因此不能把搜索解码当作能力修复。
10. **R2-H3.2 response-start候选读出**：已完成隔离候选实现。候选只负责显式assistant boundary后的首字节，后续仍由protected predictive readout负责；候选可独立学习、保存、恢复和只读审计，不接`task_family` oracle、不改共享fabric。20 epoch smoke的train exact/sequence/top1均为1.0，dev/final exact/sequence/top1均为0，paired delta=0，preflight与恢复均通过；结论是首字节候选能拟合已见条件，但没有形成可迁移条件表示，不能进入L2。
11. **R2-H3.3-A泛化剖面**：已完成只读实现与同lineage执行。对H3.2 checkpoint记录train/dev/final的目标首字节覆盖、完整答案/答案前缀重合、任务族/策略覆盖、原生首字节top1/margin、context/probability digest和恢复重复性；结果为train首字节top1=1.0，dev/final=0，dev/final目标首字节均未在train出现，dev/final margin分别为-0.29182/-0.27589，training_performed=false，read-only与recovery均为true。结论是旧smoke同时存在输出支持集不足和条件迁移失败，不能据此直接定H5。
12. **R2-H3.3-B共享首字节支持的family-disjoint泛化与课程/容量对照**：已完成预算一致复测。dev/final首字节和unknown policy均在train有支持，但完整response/family未见；同一300k总预算中core=262,839、response-start候选=11,051、effective=273,890，`within_target=true`，preflight通过。10 epoch结果为response proxy=0.63730、train/dev/final exact与sequence均为0，response-start首字节top1为0.5/0.25/0，dev/final margin为0.06973/-0.02639，paired sensitivity=0.42857。结论是支持集混杂已排除，但整段response仍未形成可迁移读出。
13. **R2-H3.3-C phase-consistent完整response candidate**：已完成预算一致复测。使用同一v2输入合同、同一300k总预算和同一family-disjoint控制集，让`predictive_readout.response_phase`从assistant boundary开始承担整段response概率和局部学习；core=262,839、candidate=11,051、effective=273,890，`within_target=true`，preflight、checkpoint恢复和只读泛化均通过。10 epoch结果为response proxy=0.63730、train/dev/final exact与sequence均为0，phase首字节top1为0.5/0.25/0，dev/final margin为-0.10842/-0.18765，paired sensitivity=0.42857。结论是首字节/后续信用统一后仍未出现未见条件回答，不进入正式pilot。
14. **R2-H3.4条件信用可观测性审计**：已完成。固定预算一致的H3.3-B/C child checkpoint和同一family-disjoint控制集，逐回答位置记录teacher-forced目标概率/排名/熵/累计似然，并绑定free-generation的原始bytes、UTF-8合法性、end-marker停止、边界和输出碰撞；结果见`reports/taiji_r2_h3_4_conditional_credit_20260916.json`。B/C均为effective=273,890，preflight、恢复和只读成立；train/dev/final自由生成UTF-8、无替换字符、边界率均为1.0，但exact/sequence均为0，文本碰撞率为0.5/0.5/0.333。B首字节legal top1为0.5/0.25/0，C为0.5/0.25/0；continuation legal top1两者相同，为0.78431/0.37143/0.26667；end-marker位置三组均为1.0。结论是停止/边界信用不是主瓶颈，失败集中在条件响应起点与未见continuation的可迁移性；不进入正式pilot，不追加同质byte训练。
15. **R2-H3.5目标/数据/表示合同复审**：已完成。复审确认现有链路已经学会UTF-8/end-marker并保留条件状态差异，但`prefix state -> 单步byte局部误差`缺少贯穿回答的内容计划；继续同构byte训练、只换token粒度或先扩shared fabric都不能直接补上该机制。已冻结[分层回答计划与渲染合同](../../reference/M5_R2_H3_5_HIERARCHICAL_RESPONSE_PLAN_CONTRACT_20260916.md)：采用`prefix encoder -> response_plan_state -> plan-conditioned byte renderer`，运行时不读取task label/参考答案，计划与渲染分账、可消融、可保存恢复，并保持300k总预算和family-disjoint边界。当前到达核心架构决策后的实现入口，仍不构成S2/L2/Mini。
16. **R2-H3.5-A分层回答计划隔离候选**：已按授权完成三seed×两臂dev训练和parameter-matched消融，并按停止门在final前结项。六次preflight与预算均通过，control/treatment参数为273,890/276,610≤300k；control dev sequence三seed均为0.25，treatment为0/0.125/0.125，treatment dev surprise三seed均更高，exact与required coverage均为0。plan target cosine为0.007/-0.033/-0.089；移除plan bridge后三seedsequence均恢复0.25，但collision升高，说明hash plan没有形成可迁移内容几何且会干扰renderer。结果见`reports/taiji_r2_h3_5a_matched_dev_20260916.json`；final未读，不加epoch，不进入S2/L2/Mini。
17. **R2-H3.6计划目标几何与信用接口复审**：三种子无训练审计已完成，见`reports/taiji_r2_h3_6_plan_target_geometry_multiseed_20260916.json`和[H3.6合同](../../reference/M5_R2_H3_6_PLAN_TARGET_GEOMETRY_CONTRACT_20260916.md)。比较signed-hash、字符n-gram、native response state、train-only whitening及两种hybrid后，只有`train-whitened native response state + compositional char n-gram`在三个种子都保持0.583跨split最近邻策略匹配且最小样本距离≥0.212；纯native虽然部分种子更高但存在0.012级塌缩。几何选择已结项，不把离线最近邻分数写成语言能力。
18. **R2-H3.6-A target encoder plumbing与恢复 smoke**：已完成版本化`ResponsePlanTargetEncoder`。它绑定corpus digest和parent checkpoint digest，只用12条train样本拟合native均值/特征基/尺度，冻结后对24条样本生成32维等权hybrid target；本次fixture的model context为45维、有效白化秩为11。`reports/taiji_r2_h3_6_target_encoder_smoke_20260916.json`记录target payload磁盘保存恢复、H3.6 trainer target map保存恢复、24个target归一化和teacher checkpoint前后摘要一致；训练未发生。该plumbing已进入并支撑H3.6-B matched dev，结果由后续第20项记录。
19. **R2-H3.6-B预注册与零步前置**：已冻结[H3.6-B matched dev预注册](../../reference/M5_R2_H3_6B_MATCHED_RUN_PREREGISTRATION_20260916.md)及机读合同。control和treatment均使用response-plan candidate、32维plan、300k预算、同一fixture和seed 20260916完成`--preflight-only`；两者effective均为276,610≤300,000，zero-step round-trip、一次child更新恢复、parent保护和atomic save均通过。treatment额外写入encoder digest、encoder parent/corpus digest及train target map digest；两份证据见`reports/taiji_r2_h3_6b_control_preflight_20260916.json`与`reports/taiji_r2_h3_6b_treatment_preflight_20260916.json`。本包只证明目标链路、容量与恢复边界，不证明能力。
20. **R2-H3.6-B三seed matched dev与bridge ablation结项**：已按冻结合同完成control/treatment各三个seed、每臂10 epoch、每run 120 episodes；六个run均通过checkpoint保存恢复、parent保护、target lineage、native-only与final延迟门。control dev sequence为`0/0.125/0.125`，treatment为`0/0.25/0`，差值为`0/+0.125/-0.125`，均值均为`0.08333`；exact与required-term coverage六个run均为0，paired sensitivity六个run均为1.0。三个treatment child的只读bridge ablation sequence为`0.25/0.25/0.25`，collision从正常的`0`升为`0.75/0.75/0.125`，但没有撤销可确认的treatment核心序列收益。`reports/taiji_r2_h3_6b_matched_dev_result_20260916.json`判定`stopped_before_final`：seed方向不一致、无非代理序列改善、bridge因果门未成立；final不读、不追加同质epoch、不进入S2/L2/Mini，返回target/representation/readout设计复审。

21. **R2-H3.7分解式回答工作空间与因果信用合同冻结**：H3.6失败不是输入未到达，而是单一静态 plan、未学习的 bridge 和断开的回答级信用链共同导致。新合同采用 4 个 12 维 plan slots、总宽度48、每16个已生成 byte 切换消费槽；slot0保留全局回答头部结构，其余槽承载后续固定chunk；byte error显式更新 `plan_bridge` 和当前槽 planner rows。目标由 train-only、corpus/parent绑定的 UTF-8-safe chunk count-sketch生成，运行时不读标签或reference。control为同预算response-phase，treatment为factorized workspace，bridge/slot-credit均做只读消融。合同与机读版本见[M5 R2-H3.7合同](../../reference/M5_R2_H3_7_FACTORIZED_RESPONSE_WORKSPACE_CONTRACT_20260917.md)和`plans/reference/contracts/r2_h3_7_factorized_response_workspace_v1.json`；实现、回归测试、双臂checkpoint preflight已通过，证据为`reports/taiji_r2_h3_7_control_preflight_20260917.json`与`reports/taiji_r2_h3_7_treatment_preflight_20260917.json`，training_performed=false；现在进入三seed dev，final继续延迟。

22. **R2-H3.7首个seed正式matched dev与因果消融**：按冻结合同完成seed `20260917` 的control/treatment各10 epoch、120 train episodes、global step 5370；两份正式checkpoint和报告均落盘，effective parameters分别为273,890与277,970，paired/native-only/read-only与final延迟字段均通过。dev sequence criterion control/treatment均为0.25，teacher-forced mean surprise分别为2.73/3.55；单个seed尚不能判定三seed方向，也不能把surprise差异写成能力收益。treatment只读消融恢复校验首次暴露了评估器在替换readout实例后仍操作旧引用的问题，已修复并加入 transient slot-credit 跨episode恢复回归测试；修复后该seed消融正常为0.25、bridge=0.125、slot-credit=0.0，checkpoint_read_only=true。证据见`reports/taiji_r2_h3_7_control_dev_20260917_seed20260917.json`、`reports/taiji_r2_h3_7_treatment_dev_20260917_seed20260917.json`和`reports/taiji_r2_h3_7_ablation_20260917_seed20260917.json`；不读final，不追加epoch，唯一下一步是继续seed `20260918`与`20260919`的matched dev及各自消融。

23. **R2-H3.7第二个seed正式matched dev与因果消融**：按同一冻结合同完成seed `20260918` 的control/treatment各10 epoch、120 train episodes、global step 5370；effective parameters仍为273,890/277,970，checkpoint、parent/target lineage、native-only、paired与final延迟字段通过。dev sequence criterion control/treatment仍为0.25/0.25，teacher-forced mean surprise为2.749/3.321，exact与required-term coverage均为0；这些结果没有形成可交付能力。第二组只读消融 normal/bridge/slot-credit均为0.25，未显示可撤销的核心内容收益；结合首组的0.25/0.125/0.0，当前尚不能满足三seed一致方向或消融门。按预注册继续完成最后seed，而不是在部分矩阵上改写停止规则；seed `20260919`完成后必须先跑aggregate，失败则不读final、不追加epoch。

24. **R2-H3.7三seed aggregate结项**：seed `20260919`的control/treatment也完成10 epoch、120 train episodes、global step 5370；三组run全部通过checkpoint恢复、容量、target lineage、native-only、paired与final延迟门。aggregate显示control/treatment dev sequence均为`[0.25,0.25,0.25]`，逐seed差值为`[0,0,0]`；exact与required-term coverage六臂全为0，paired sensitivity均为1.0。UTF-8保持1.0，但boundary control=`[1.0,1.0,0.875]`、treatment=`[1.0,0.875,0.875]`，seed `20260918`出现treatment相对control退化。三组消融为normal=`[0.25,0.25,0.25]`、bridge=`[0.125,0.25,0.25]`、slot-credit=`[0,0.25,0.25]`；修正后的aggregate要求先存在真实 treatment-vs-control 内容增益再判“撤销核心收益”，因此两项消融门均为false，避免把并列分数下的随机变化误报为因果收益。`reports/taiji_r2_h3_7_matched_dev_result_20260917.json`判定`stopped_before_final`，停止原因是无三seed非代理内容方向、boundary退化、bridge/slot-credit未撤销核心收益；不读final、不追加epoch、不切默认入口、不进入S2/L2/Mini，回到冻结归因顺序：target sketch → slot phase schedule → bridge credit → renderer readout → native prefix representation → capacity/data。

25. **R2-H3.7冻结归因顺序的只读审计**：没有训练、没有读取final、没有修改任何child checkpoint；审计只用三份treatment checkpoint的dev prefix与已保存的train-bound encoder，并检查恢复前后 digest一致。`target_sketch`显示dev plan-target平均cosine=`0.24943`、平均L2=`3.46876`，四个slot平均cosine=`[0.20048,0.16450,0.12583,0.46729]`，因此target不是空或完全不对齐，但前三个slot明显弱于slot3；`slot_phase_schedule`相对phase0的平均概率JS=`[0,0.16811,0.52136,0.25835]`，phase1/2/3 argmax改变率=`[0.4167,1.0,0.7083]`，说明phase确实消费了不同概率面，不是静态plan未接入。bridge norm为`20.77084–21.54328`，但aggregate的bridge/slot-credit因果门仍为false；renderer的dev sequence/exact/required-term为`0.25/0/0`，prefix paired sensitivity为1.0，effective=277,970≤300,000。结论只到“target有弱且不均匀信号、phase链路工作、workspace→可迁移内容的credit/readout闭环未形成”，不把cosine或phase变化写成能力；审计证据见`reports/taiji_r2_h3_7_attribution_audit_20260917.json`，下一步进入架构设计讨论，不再执行H3.7同质训练。

不得复用P3b报告名或把`seed_beta.pt`当作可写目标。设计包的出口仍包括输入/上下文/回答边界/目标定义、训练与运行期分工、编码/语义/上下文/结果四层评价、checkpoint前置、L2/L3停止条件，以及失败后回到数据、目标、表示、读出或架构哪一层的判读规则；本轮已把其中的可执行保存恢复和S1读出部分先落地。

### 6.1 设计评审必须回答

| 层 | 问题 | 可证伪材料 |
|---|---|---|
| 目标 | 误差更新哪些参数，是否优化有上下文的答案内容 | 输入→状态→目标→更新→生成来源图，最小任务学习前后对照 |
| 表示/记忆 | 语义、顺序、跨轮信息如何保留，容量瓶颈在哪 | 上下文置换/重置、记忆消融、长度退化曲线 |
| 输出 | 真正生成来自哪些已训练状态，模板/约束解码贡献多少 | 同状态解码对照、原始回答、去回显评分 |
| 信用分配 | 远期答案错误如何影响先前状态与参数 | 延迟目标、干扰、旧能力保持，不只背固定样例 |
| 扩展性 | 更多数据/状态/预算能否改善能力 | 小规模预算曲线＋CAP读数；代理分与能力分开 |

**选择解释**：

- 当前合同内改进可复用资产、对比清晰，但必须有新机制假设，不无限重复语料训练。
- 成熟发展训练＋运行期原生适应有更高潜在上限，但要明确参数分工、训练方法、哪些现行约束改变、Taiji认知owner与连续状态如何保持；需独立架构批准和最小原型。
- 外部表达/工具可检验产品体验，只记E/T模式，不替代N模式原生语言出口，也不是当前优先解法。

**推荐原则**：采用“显式回答目标＋分层信用分配＋原生多时间尺度状态”的高上限路线，先用隔离pilot验证最小因果链，再决定是否进入成熟发展训练与运行期原生持续适应的完整VISION实现。若当前目标不能支撑L2，提出合同变更，不继续同质加训；已有因果/保持/状态资产保留，不全盘推倒。

### 6.2 训练与测量纪律

1. train/dev/最终未见集隔离。已曝光CAP题转回归；新泛化主张另冻结同类独立题及标准。
2. 执行前冻结主效应、配对单位、种子、预算、最大共同检查点数、停止规则、容忍区间来源和失败读法。
3. 分布对比先解决单臂停导致配对不足；安全急停优先，不足共同点判不可判，不能强迫不安全训练。
4. surprise、短token准确率、CAP对话分账；人工B/G、真实性A、性能H缺测不算通过。
5. 连续两次正式评审只有局部/代理分改善、整模型不动，暂停同质训练，进入目标/集成评审。
6. 新预算根据实测并注明同机竞争估算；历史符号速率不当未来承诺。

本轮R2-P1/P2/G1的具体读法：response-only训练使输入进入原生输出，受限读出把dev/final UTF-8合法率和无替换率保持在1.0；static/slow/fast/fast_slow均有原题与改写题输出变化，且所有preflight、fresh restore、parent保护和只读评分通过，但四臂exact response均为0→0。P2进一步显示dev/final回答边界、必需内容覆盖、未知策略和sequence criterion均未通过；G1 v2只显式加入运行时可提供的policy，20 epoch有限过拟合仍有50% train output collision、train exact=0.5，dev/final sequence criterion=0，paired改写敏感性=0。H2/H3只读审计进一步确认train prefix context与概率分布已经不同但argmax仍相同，且恢复重复性成立。结论是owner生命周期和输入状态链路成立，当前瓶颈转入readout margin/序列停止路径；不扩大同质byte训练或正式pilot。

**交付**：可加载bundle、训练/数据manifest、原始回答、学习曲线、保持矩阵、资源与恢复报告。
**出口**：L2达标；否则记录真实缺口，不降线、不宣布mini就绪。

## 7. R3：在线持续适应闭环

主命题是“真实反馈改善后续未见任务选择”，不是增加一次record。

1. 固定base lineage、在线/后测切分、更新预算与冻结对照；候选耗尽如实记录，不偷塞已见测试凑轮次。
2. 反馈准入、预测变化、行为收益三层独立报告；a1必须绑定后测估计量，不以applied替代。
3. 保持分行为保持与协议要求的预测/状态不变；旧门要求位级不变仍按旧门判，容许漂移须新协议和独立数据。
4. 同事件重复、不同事件过期parent、rollback后重放分开；event ID与lineage可审计。幂等是无二次更新还是显式拒绝先写合同。
5. 接收前、pending落盘后、更新后保存前、恢复后重投四处故障注入；新进程与连续运行比较选择/状态/账本。
6. online、frozen、等预算child同数据边界；反馈/更新消融验证因果，真实终态独立于模型自评分。
7. 64仅候选预算，单位、成功/失败轨迹成本、墙钟/episodes分别标定；新执行另需批准。

六类必达：新任务收益、旧任务保持、重复幂等、过期父状态拒绝、中断恢复、模型回退与环境撤销分离；数据/干预/预算共同门同时过。
**出口**：逐门原始证据一致、必达无缺测，才提在线轴评审。一轮写入不证明长期多轮适应；扩主张需另冻结序列。

## 8. R4/R5：集成、扩面与M5缺口

### 8.1 L3用户验收

顺序为L0身份恢复→L1真实输出/消融→L2基本对话→L3同入口代表能力＋安全/性能→用户自由提问。

F代表能力必须由同bundle实际提供。将研究gate迁入runtime需owner、持久化、权限与失败语义对齐，不能外跑脚本后把成绩拼回聊天。交付入口、模型卡、权重摘要、能力范围、成功/失败原始例答、重置恢复说明、评测与反馈表。人工验收分可运行/可信/有用；自动过线不等于用户接受。

### 8.2 独立结构协作

B2已结项，同族复现不反复计新能力。以[binder可行性](../../reference/M5_B0_WP6_BINDER_FEASIBILITY_20260915.md)为入口：

- T3缺多文件binder；T1还缺非平凡目标和早退语义；T2缺成员证据通道，不能一次binder修改宣称全闭合。
- treatment/oracle绑定权限公平；显式保留旧binder/旧机制，对照不指向同函数。
- 新内容结构、组合及split先冻结；create旧面仅回归。
- 独立扩面与贡献lesion完成后评审协作轴，不直接称跨域泛化。

### 8.3 知识、身体与长期轴

- P5.1h：真实语料来源/许可/去重、独立测试、污染检查、retention、child admission与回退；trial/admitted分列。
- 身体：注册/撤销、schema、权限、dry-run、真实执行、反馈归属、超时、补偿、幂等、审计；高风险真实动作另授权。
- M4成长：真实压力、同最终容量静态对照、出生/贡献lesion、保持与恢复；未触发记未测。
- 睡眠/梦境/玩耍、生命调节、多系统记忆、自主学习、跨域多模态仍归01总图，逐项登记收益/长期缺口；不消失，也不全部成为L3前置。

## 9. 何时晋级与交付

| 检查点 | 必需证据 | 权限与失败处理 |
|---|---|---|
| 研究结项 | 协议、原始结果、资源/限制、失败归因 | 负结果也可关闭包，缺口回轴级，不无限追加字母 |
| 轴晋级 | 独立范围＋保持/恢复/安全/预算＋CAP复评 | 所有者批准；缺哪门补哪门，新主张另冻结 |
| L3 | L2＋F＋A/G/H、同bundle恢复与来源 | 触发用户验收，不等于正式发布 |
| M5退出 | 知识、协作/选择、在线/身体及共同门、完整CAP | 范围排除须显式批准，不默许豁免 |
| M6采用 | admission、shadow/canary、隔离、降级、客户端能力一致 | 指定行为逐步采用；失败退回父版本 |
| M7发布 | 当前版本正式CI、安装、manifest、端到端、安全与回滚 | 批准指定版本，不用局部或历史绿灯代替 |
| M8 | 真设备同质量/恢复、端到端成本收益 | 单列范围，不阻塞CPU |

每包开工填写owner、授权、依赖、输入hash、交付、必达/探索门、资源上限、停止条件、判读者、出口。条件里程碑不是日期承诺，预算不足也不自动晋级。

## 10. 风险与维护

| 风险信号 | 控制措施 |
|---|---|
| 汇总true但原始收益0 | 审公式/输入/后测，不按布尔放行 |
| loader/binder/解码改了仍用旧baseline | 新chain新基线；历史不强行横比 |
| holdout反复曝光 | 转dev，新冻结最终集 |
| 测试写checkpoint、parent漂移 | 临时目录/hash守卫；受损证据逐项定位 |
| 入口并非被训练模型 | manifest绑定代码/入口/参数，缺件拒绝 |
| 两轮整模型不动 | 暂停同质训练，目标/信用分配评审 |
| 多任务争资源 | 独立目录和资源owner，不擅停他人进程 |
| 旧CI计数冒充当前 | 局部/全量/远端标各自提交日期 |
| 同质探针无限扩张 | 每包冻结预算/轮次，达到出口即结项 |
| VISION偷换合同 | 独立批准、原型、迁移/回退后才采用 |

[技术债](05_TECH_DEBT_REGISTER.md)按最新日期解释：旧27项SystemExit是环境守卫伪影，不是未定位架构缺陷；3.10数值敏感、仪器语义、隔离问题独立跟踪。本次仅运行不写checkpoint的文档/身份检查，不全量冒险重测。

清理的是活动页过时执行指令，不是历史失败证据；可再生产物确认无引用后才能定向清理。提交仅本任务文件，不接管用户未提交产物、不自动push。

唯一[未来VISION](../../reference/VISION_FUTURE_TECHNOLOGY.md)记录候选架构，本文管当前依赖与权限。每包结项同步01/02/03/07和首页，不追加互相矛盾的“最新状态”。

## H3.6-B结论：三seed dev与bridge ablation均已完成，final前负结果结项

H3.3的预算一致证据已经完成：H3.3-B response-start与H3.3-C response-phase在同一v2控制集、同一300k总预算、同一10 epoch和同一评价链上均通过preflight与恢复，但train/dev/final exact和sequence均为0；两者dev/final首字节top1均为0.25/0，paired改写敏感性均为0.42857。B的dev/final首字节margin为0.06973/-0.02639，C为-0.10842/-0.18765；统一整段response owner没有产生能力出口。由此H3.3结项，不再追加同质byte训练或继续堆叠边界候选。

H3.4已固定上述child checkpoint与数据完成审计。B的response-start owner与C的response-phase owner在训练后都给出相同的continuation surface：train/dev/final continuation legal top1为0.78431/0.37143/0.26667，平均legal rank为1.3137/14.8571/17.8667；end-marker位置legal top1三组均为1.0。两者自由生成的UTF-8合法率、无替换率、end-marker边界率均为1.0，但exact/sequence均为0，dev/final文本碰撞率仍为0.5/0.333。H3.4因此排除了“停止器或UTF-8读出是主要瓶颈”，也没有发现可直接支持S2的稳定未见margin。

H3.5复审已经完成。结论不是废弃现有主线，而是在已有原生状态、checkpoint、预算与评价底座上补上缺失层：assistant boundary由prefix state产生持久`response_plan_state`，byte renderer在整个回答中同时读取当前上下文与计划状态；计划目标、byte渲染、boundary和retention分别记账。运行时不得读取`task_family`、split、评分字段或参考response，现有UTF-8/end-marker路径保持不动。完整选择、数据课程、两臂、指标和停止规则见[H3.5分层回答计划与渲染合同](../../reference/M5_R2_H3_5_HIERARCHICAL_RESPONSE_PLAN_CONTRACT_20260916.md)。

H3.5-A最小candidate、plumbing smoke和matched训练冻结已经完成。v3控制集固定为12/8/4，覆盖共享起点不同内容、四类policy、history与context permutation；三个seed、两训练臂、10 epoch、300k预算、parameter-matched plan消融和final延迟权限均已写入可机读合同。训练入口的response-plan预算与`--defer-final`已验证，开发阶段不会读取final能力输出。

H3.5-A dev阶段六次matched运行已完成并触发停止：treatment在三个seed的dev sequence和surprise均不优于control，plan target cosine接近零或为负；plan bridge消融恢复sequence但增加collision。final保持未读，不能继续同一target加训，也不能把输出去碰撞当作内容能力。

H3.6三种子无训练geometry审计已完成。唯一入选方案是train-only whitened native response state与compositional n-gram等权混合：它的迁移下限和防塌缩距离同时优于signed-hash。H3.6-A target encoder plumbing已经完成：实现了train-only拟合、冻结均值/基向量/尺度、payload/digest、错误corpus/parent拒绝、教师checkpoint守卫，并已通过真实fixture reconstruction smoke；H3.6 target 已接入隔离的LanguageAlignmentTrainer，child checkpoint 可恢复同一份train target map。实现与报告见`taiji/response_plan_target.py`、`scripts/training/smoke_taiji_r2_h3_6_target_encoder.py`、`reports/taiji_r2_h3_6_target_encoder_smoke_20260916.json`。

H3.6-B matched dev预注册、训练入口的geometry选择、train-only encoder拟合、target payload血缘报告和`--preflight-only`已完成。control/treatment的零步前置均通过：effective=276,610≤300,000，checkpoint保存恢复、一次child更新、parent保护与target血缘检查均成立；正式报告明确`training_performed=false`，因此没有把前置的一次恢复性child update冒充正式能力训练。用户授权的control/treatment三seed正式dev已全部完成：六个run均为10 epoch、120 episodes，checkpoint与report均落盘；三组只读bridge ablation也已完成，aggregate判定为`stopped_before_final`。

H3.6-B按预注册停止规则结项：不读取H3.5-A或H3.6-B final，不追加同质epoch，不进入S2/L2/Mini，也不把surprise、collision或bridge影响写成模型能力。H3.7已把复审结论冻结为4-slot/48维/16-byte phase的分解式 workspace 与 renderer→bridge→slot credit 实现门；三seed正式dev、三组只读消融、aggregate和有界归因审计均已完成，aggregate在final前停止，明确不追加epoch、不读final、不切默认入口；当前进入需要讨论的新架构设计节点，H3.6-B与H3.7 child均作为只读失败证据。
