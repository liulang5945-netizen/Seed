# Seed / Taiji 当前详细推进方案

> 更新：2026-09-17；R0证据与门禁审计已完成。本次决定将R2整模型语言能力设为当前主线；结构化R2入口、checkpoint前置、P1 developmental 对照、P2序列级只读评价、G1条件接口、H2/H3内部表示/读出审计、H3.1序列路径对照、H3.2 response-start候选读出、H3.3泛化控制与response-phase候选、H3.4逐位置条件信用审计、H3.5目标合同、H3.6-A target encoder plumbing及H3.6-B matched dev/bridge ablation均已完成；H3.6-B按预注册在final前负结果结项；本轮已冻结H3.7分解式回答工作空间与因果信用合同，训练权限只用于该合同的隔离实现、preflight和dev验证，本地未提交观察与正式结果继续分账。
> 本文是唯一执行顺序来源；[01](01_SCOPE_AND_PHASES.md)管总阶段，[02](02_GATES_AND_CI.md)管晋级，[07](07_MINI_MODEL_DELIVERY.md)管整模型验收。
> 旧逐轮台账完整保留于Git的56a4c3e4:plans/active/roadmap/03_CURRENT_EXECUTION.md及各冻结报告。本次不改历史结果、不修改默认入口、不授权产品采用或架构切换；R2隔离smoke只用于验证新入口。

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
| P5.2d本地v2 | 未跟踪reports/taiji_p5_2d_online_writeback_v2_20260916.json：预算64，a+c反馈applied，outcome仍failed；[R0审计](../../reference/M5_R0_EVIDENCE_GATE_AUDIT_20260916.md)已逐门核对 | 预算校准信号可作局部事实；A1假阳性，A2/A3/A5/G1不可判，A6仅部分可用；协议和runner的§8修订仍未同步，不得晋级。修正仪器降为并行债务，不阻塞R2主线 |
| v2审查线索 | updated_pair仍b+c、六对预测全1.291667、后测收益0，但a1=true；a2/a3/a5/g1失败；恢复及rollback说明与细节需核对 | 必须逐门查证，不能认定只是预算问题，也不能未经诊断归罪核心架构 |
| CAP默认入口 | [基线](../../reference/M5_CAP0_BASELINE_RESULT_20260915.md)：原tick=2入口去回显后C/D/E为0；B/G辅助判断待人工确认 | 限定当时快照；默认checkpoint存在被测试写动的风险，不能沿用旧身份 |
| P3b已提交材料 | [阶段结果](../../reference/M5_P3B_RESULT_20260916.md)：唯一共同tick17M，C/D差0，E差+0.05，未检测到该分辨率下效应 | 不等于分布无关或架构无效；J4 A/H及人工安全分支缺证 |
| P3b工作区终态 | treatment记录finished_at、campaign_stop=regressed；18M persistent；对照17M material；R0已核实两臂停止 | 数据分布效应只有一个共同tick，仍not_resolved；18M原始输出可观察但C/D/E未达标，保护隔离仍有DEBT-I7 |
| 本地heldout | 未跟踪reports/taiji_p3b_heldout_surprise_20260916.json为not_resolved | 先审协议与血缘；surprise不能替代对话评价 |
| 知识/身体 | P5.1g仍trial并回滚；Workbench合同资产存在 | 真实语料child未准入、身体全生命周期未结项 |
| CI | 9月16日查询run34869725409，d09dcc21，总体failure；3.10失败，3.12/Windows等通过 | 不是当前HEAD的CI；不称全绿，旧27项SystemExit不再视作未定位代码缺陷 |
| R2 aligned language seam | 结构化episode、response-only native readout、checkpoint digest/atomic save、zero/child恢复前置、paired诊断、static/slow/fast/fast_slow四臂、P2序列级只读评价、H3.1 beam、H3.2 response-start、H3.3 response-phase与泛化剖面、H3.4逐位置信用审计、H3.5表示合同、H3.6-A target encoder plumbing及H3.6-B零步前置均已完成；H3.6-B六个正式dev run与三组bridge ablation已完成；H3.7合同已冻结，尚待实现门 | H3.4确认UTF-8/end-marker稳定但条件首字节与未见continuation不可迁移；H3.5-A/H3.6-B均在final前负结果结项。H3.7要求4-slot/48维/16-byte phase workspace与renderer→bridge→slot credit链，不追加旧target epoch，不读取旧final；只有实现preflight通过才启动H3.7 matched dev |
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

21. **R2-H3.7分解式回答工作空间与因果信用合同冻结**：H3.6失败不是输入未到达，而是单一静态 plan、未学习的 bridge 和断开的回答级信用链共同导致。新合同采用 4 个 12 维 plan slots、总宽度48、每16个已生成 byte 切换消费槽；slot0保留全局回答头部结构，其余槽承载后续固定chunk；byte error显式更新 `plan_bridge` 和当前槽 planner rows。目标由 train-only、corpus/parent绑定的 UTF-8-safe chunk count-sketch生成，运行时不读标签或reference。control为同预算response-phase，treatment为factorized workspace，bridge/slot-credit均做只读消融。合同与机读版本见[M5 R2-H3.7合同](../../reference/M5_R2_H3_7_FACTORIZED_RESPONSE_WORKSPACE_CONTRACT_20260917.md)和`plans/reference/contracts/r2_h3_7_factorized_response_workspace_v1.json`；当前只进入隔离实现和checkpoint preflight，preflight通过后才运行三seed dev，final继续延迟。

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

H3.6-B按预注册停止规则结项：不读取H3.5-A或H3.6-B final，不追加同质epoch，不进入S2/L2/Mini，也不把surprise、collision或bridge影响写成模型能力。H3.7已把复审结论冻结为4-slot/48维/16-byte phase的分解式 workspace 与 renderer→bridge→slot credit 实现门；当前唯一下一步是完成隔离实现和checkpoint preflight，只有实现门通过才启动授权的三seed matched dev，H3.6-B child继续作为只读失败证据。
