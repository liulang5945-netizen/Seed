# N主线执行证据与审批台账

> 文档状态：[状态] — 只登记执行证据、审批、费用与适用限制；统一编号、阶段状态和实际动作集中在[总计划](../../MAINLINE.md)。

更新：2026-10-31，Asia/Shanghai。本页原执行表已整合进总计划，不再发布第二套顺序或复制待办。以下按统一步骤关联证据，阶段记录只解释当时，不能把零步时“未训练”当作后续现状。

## 当前唯一下一步：见总计划的执行位置

直接按[总计划「执行位置」](../../MAINLINE.md#执行位置)推进。本标题仅保留导航兼容，不独立维护候选、状态、审批或下一动作。

## 2026-10-31 根目标与排期纠正

所有者明确连续生命活动、持续可塑性与规模生长共同构成递归进化，不是固定模型训练后另选的附加功能。根需求§3.0、架构§0.3、总计划09及后继阶段已据此修订；当前设计动作统一在总计划发布。已有R5系列仅作为局部固定容量配方证据，已有代码中独立的生命/睡眠/成长资产保留，但未据名称补签集成。

09.1静态共同内核合同可与R5c按原预算有限结案并行；本轮没有取消、停止、续跑或追加任何进程，也不把用户的目标澄清当作新服务/训练已经执行。实际生命/成长能力仍须在同一主体的状态/参数/拓扑和前后经历上验证。

## 2026-10-31 核心算法设计交付的进一步纠正

09.1必须先设计可实现、可反驳的核心神经算法，再推导运行对象/接口/保存合同。RNN/Transformer机制影响状态表达、传播及训练，不能被随意降为实现细节；目前已有局部算法与资产，尚无覆盖新根目标的统一核心算法完成/有效证据。设计判据见[架构§0.4](../TAIJI_NATIVE_ARCHITECTURE_V1.md#04-核心神经网络算法的设计交付20261031)，唯一动作仍由总计划发布；本记录不增加实验或修改冻结合同。

## 2026-10-31 09.1核心算法首稿与局部核对

所有者授权重新设计、不以当前候选兼容为约束。已交付[核心算法v1草案](../../reference/TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md)，主方案为多时间尺度可塑神经图。新文件包含实际方程与更新顺序，不是仅列运行接口。

[合成代数报告](../../../reports/taiji_core_algorithm_v1_math_checks_20261031.json)由[检查脚本](../../../scripts/training/check_taiji_core_algorithm_v1_math.py)生成，11项局部性质通过，绑定代码SHA/NumPy版本；无完整模型、无optimizer更新、无数据分区读取。它不代表完整信用、语言、生命活动或结构成长通过，正式执行与下一步仍由总计划发布。已有R5c等独立运行不因设计交付被停止/删除或追加。

## 同轮并行实验产物的状态边界

新算法设计收尾时只读核对R5c摘要：`FAIL_BEHAVIOR`、Q1 answer约0.4580、cut_drop=1.0；随后并行提交`01b693fa8`交付[正式评价](../../../reports/taiji_native_core_r5c_evaluation_20261031.json)及原实验结案。该实验与源码不是本轮新算法设计的执行成果；本轮不据摘要给出唯一原因或修改原门。新设计依据根目标及已发布历史，不把这一负结果当全架构充分反证。

## 2026-10-31 方程验证范围

所有者要求先调整主线，再验证调整核心方程。09.1增加原式反例、耦合数值/梯度核验和修订复测三个子项；本轮允许合成矩阵/自动微分的设计验证，不执行完整模型训练、正式数据评价或既有进程操作。旧11项数学核对仍只代表其原局部范围，新结果单独留存。

## 2026-10-31 v1.1方程验证与修订回执

按先调主线后验证的顺序，09.1.1–09.1.3完成本轮范围：[验证脚本](../../../scripts/training/validate_taiji_core_equations_v11.py)、[独立报告](../../../reports/taiji_core_equations_v11_validation_20261031.json)。原式无写入反例在200步内121次触发范数裁剪；v1.1近端擦除解决该不放大性质，另明确归一化尺度、整体损失下出生信用及变时长活动的TD/trace。新22项检查涵盖耦合状态/编码/记忆信用、出生前后和启动、有限差分/二阶/元导数及float32/64有限性；旧11项报告保持原位。

本轮0 optimizer步骤，但为验证导数计算两次合成内层更新表达式；没有完整模型、数据分区、真实任务训练/能力评价或完整主体恢复。局部数学保证与剩余全图/任务问题详见[算法v1.1 §21](../../reference/TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md#21-v11方程审查与验证记录)，不能把本报告当09.2通过。

## 2026-10-31 P0/P1复审与E1规格勘误

源码复核确认22项属于**合成耦合代理**：快更新只有硬许可/完整更新，没有连续chi；群间读取与h.mean输出不是正式Q/K/V及查询读出；q/key/value/fout/out等矩阵未开启梯度。旧报告按原条件通过的事实保留，正式全图与这些参数的信用尚未核验；不改脚本或报告SHA。

算法主稿E1补入f_i的F^T q多头读取、统一B平滑映射、P_R活动奖励/P_A准入/P_C最终确认及累计自适应上限；编码版本变更不再默认给旧H换revision即可兼容，首个短生涯以有界完整重放作迁移对照。小梯度只代表代理中可分辨的导数，float32实际更新、任务收益、持续状态保持及迁移成本仍待测。详见算法§6/10.3/12.2/21/23；排期只见总计划，本轮仅修订文档，不实施新模型或训练。

## 2026-10-31 E2边界与新颖性定位

所有者提出三条E1未覆盖边界。零优化器小矩阵复核：rank-one C按SVD请求4单元得到启动范数[0.029,0,0,0]；四点/seed=20261031的2000组内容盲门范数中位0.0347471，真门0.0538528；禁写范数约3.02000094，独立0.99衰减1000步为0.000130377。零期望指该四点全局C=0时的带符号梯度向量，并非范数。未复跑其他并行评估中的优化探针，也未更改其文件或旧11/22项证据。

主稿E2补有效秩限额与逐单元准入、内容门的随机/置换对照、chi冻结与陈旧状态合同；按原论文核对GDN2、DeltaNet/Gated DeltaNet、TTT、GradMax及可塑性先例，将五组比较按主张分工。现阶段不主张新基本算子或首次性，组合耦合的实际研究价值仍待证。执行与授权仍只见总计划，本记录不启动训练。

## 2026-10-31 v1.2生长信用决策与命题预审

所有者选定“生长信用与覆盖出生”作为研究增量，暂不添加自我修改预测/快记忆撤销组件。方向已进入算法§24和总计划；v1.1动力学/快记忆/表达公式与旧报告保留，生长候选明确区别于全局SVD基线，不据此操作训练进程或模型默认值。

[预审脚本](../../../scripts/training/check_taiji_growth_credit_v12_math.py)和[报告](../../../reports/taiji_growth_credit_v12_math_review_20261031.json)14项按声明条件通过，0优化器步骤、无数据分区。通过中包含原推广被反例推翻：rank=1但正交出边可有4行非零梯度；软门N=1秩和=2；新增残差能降聚合秩；H为0完整输入仍可有信用；随机门L_start可为1；共享簇覆盖仍可整体抵消。条件前向含真实门，6个入边坐标/三个eps与autograd一致，置零/翻符号负对照拒绝。均不代签09.2正式图或任务效力。

针对性查重发现GradMax、TINY、Low-Rank Learning by Design、GSS、Subspace Trap及Neurogenesis Deep Learning等直接相关先例；不能据当前证据声称“唯一未被覆盖”或确定新颖性等级。保留覆盖算子/耦合为候选贡献，第二阶段与延期范围按总计划决策，本台账不另发队首。

## 2026-10-31 R021正式前向与历史审计入库

按所有者要求，先冻结[六接口/凸分区门/覆盖工程界](../../reference/SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021_20261031.md)，再交付实际编码、情景/null读取、可学群间/快QKV、查询多头、动力学、连续chi与位置复制/生成。 [R021回执](../../../reports/taiji_core_reference_r021_forward_coverage_20261031.json)给出12项严格float64前向检查、10行方程覆盖矩阵和101个参数张量清单；backward_calls=0、optimizer_steps=0、data_splits_read=[]。小配置44,905参数只作同构工程夹具，不是默认模型规模或定价。

覆盖出生/噪声标定、实际学习信用、编码迁移和完整主体恢复都明列未实现/未验；不据前向通过签新模型有效。当前后继只见总计划，不在本台账新增队列。

五份v1.1/E1独立审计按[历史入库身份](../../manifests/taiji_core_v11_e1_audit_admission_20261031.json)与[原字节包](../../../reports/taiji_core_v11_e1_originals_20261031.tar.gz)保存。历史设计64fdf8576及原脚本/report绑定经核对；可读脚本加历史版本检查、两处静态修正及LF规范化，原报告不冒称绑定新兼容脚本，已知方法局限留在历史说明中。未重跑其300步玩具优化，亦非v1.2重新裁决。

入库维护检查：参考前向、架构/命名、单主线、报告索引行尾与凭据守卫合计50项通过；Ruff主规则/B-SIM、Black、298条相关内部链接及source/hash绑定通过。历史脚本在探针执行前拒绝当前v1.2设计，未创建输出或运行优化。只代表本地受影响检查，不宣称远端CI或能力门通过。

## 2026-10-31 长期目标表述收束

按所有者目标复核，统一补入内部状态连续、持续可塑、继承式新增/整合/回收、真实反馈改变负责机制，以及生长收益/主体恢复的按范围验收。“经验积累与认识纠正”区分真实历史和可修订认识；旧能力保护不保护已知错误。阶段方法与根目标分开，未把v1.2信用秩或R021写成必须成功的终极方案。总计划13只承接成熟与扩域，基础生命/学习/生长仍从09开始；本次仅改文字/计划，不推进09.2.2或更改训练权限。

## 2026-10-31 R022负责参数信用与实际数值判别（09.2.2）

先冻结[R022合同](../../reference/SPEC-TAIJI-CORE_CREDIT_R022_20261031.md)：三个声明负责目标（T_EXPR教师强制预测NLL、T_WRITE2两步快写链、T_WORLD_PROXY常数代理）与零类预声明，再交付[检查](../../../tests/taiji_native/test_living_core_credit_r022.py)与[报告](../../../reports/taiji_core_r022_credit_validation_20261031.json)。[R021回执](../../../reports/taiji_core_reference_r021_forward_coverage_20261031.json)所绑`taiji/living_core.py`字节经SHA核对未变；本报告`passed_r022_credit_only`、8/8检查通过，backward_calls=127、optimizer_steps=2（声明合成更新表达式，更新后参数即废弃）、data_splits_read=[]、费用54.61/900 CPU秒。

101个参数张量全部获得FD确认的信用或落入声明零类：87 credited＋12 world代理（真实世界损失仍属09.3/09.4）＋V birth_zero（初值U=0⇒V梯度精确0；声明两步Adam更新使U≠0后V解锁9.4e-4）＋`episodic.score.bias`判明为**softmax平移不变死参数**（I3的b_3：位置softmax对全体分数共同平移不变，值置换对照无任何效应；梯度为~3e-18浮点抵消残留，声明更新中累计|Δθ|≈1.2e-12≤lr·1e-9且更新后仍惰性）。该死参数是R021接口级结构事实，建议后续R021接口修订删除或标注；修订前保留原位、永不获信用。

读数：FD三eps（1e-4/1e-5/1e-6）扫描最优失配≤1.45e-10（atol 1e-8）；二阶方向导数100/100吻合、最大差1.9e-6；快写Frobenius裁剪分支未激活（余量14.78）；float64/32逐组比较98个已解析张量中位偏差2.6e-7、最大3.9e-6（门0.1）；置零/翻符号/许可allowed=False/表达后写排序/空view断路（含强制g_mix=1）负对照全过，coverage≡0/prior≡0状态条件零及其正对照全过，U激活对照成立；声明更新输出改变max|ΔP|=0.163，世界族Δθ精确为0（代理不进更新目标）。

执行中按合同纪律修订两处并落账：①发现死参数后先补录`inert_shift_invariant`零类（含值置换对照）再重跑，未改阈值放水；②置零对照阈值由`max(1e-8,1e-6·|L|)`修订为`max(1e-10,1e-8·|L|)`＋一阶预测回退——对照的语义是因果连通性，效应大小由FD确认的梯度负责；message_goal.weight实测ΔL=6.86e-6与一阶预测|g·θ|=6.88e-6几乎精确吻合，确认局部线性。本报告不签任务能力、生长、保持或恢复；当前唯一下一步见总计划（09.2.3）。

提交后按钩子要求重跑Mimosa深扫（focusFiles=本包三个代码/实现文件）：jobId `scan-job-mvwfho3z`，scanId `scan-2026-10-31T13-22-24.513Z-ef51fc8b7725`，**封印`sha256:6da506b4ecd7348849af483a998cf465c7df358c9cbb115b9746d35bbac86260`**，1,652项findings为既有普查基线、无新假设、离线advisory零匹配；同提交先跑单主线审计（ok）与全仓sha-drift普查（117,275份报告/5,216条sha声明，12条drift与3,174条missing均为历史checkpoint/pytest临时文件及并行清退会话删除所致，不涉本包证据链）。

## 2026-10-31 R023信用覆盖出生与有界迁移（09.2.3）

先冻结[R023合同](../../reference/SPEC-TAIJI-CORE_GROWTH_R023_20261031.md)，再交付[growth_credit.py](../../../taiji/growth_credit.py)、[state_migration.py](../../../taiji/state_migration.py)、[检查](../../../tests/taiji_native/test_growth_credit_r023.py)与[报告](../../../reports/taiji_core_r023_growth_birth_20261031.json)。`taiji/living_core.py`字节经SHA核对与R021/R022绑定一致；报告`passed_r023_birth_engineering_only`、8/8检查通过、backward_calls仪表计数落账、optimizer_steps=0、data_splits_read=[]、费用1.97/900 CPU秒。

交付事实：K=2门控信用构造（copy/generate混合门，图内量可审计）；逐(支路,群)128次置换噪声标定（每群独立零假设）；信号条件与max-min整数分配（R_max=8、枚举81态/群、tie-break=min rho→total rho→少用单元→稳定支路id，预算停实现）；`GatedGrowthCore`门控出生算子（`pre += Σ_c g_c·B_c·tanh(R_c x)`，镜像R021步进体＋唯一声明新增，父行不分门）；全局SVD基线对照臂（本夹具8单元/群 vs 覆盖分配1+1，判别性差异入账）；共享支路整体信用守卫。§10.3迁移：journal恒等重放逐位相等、编码变更shadow事务提交revision+1（非encoder参数逐字节不变、重放无参数突变/无奖励/无外部动作）、事件/字节预算拒绝保父。

读数：出生中性逐位成立；4个出生单元启动梯度与算子预测`B^T C_c`在1e-12一致（8.5e-4..2.5e-3＞1e-8）；L_start=1.0、独立方向比例1.0、kappa=0.998；P新列出生时刻梯度精确0；启动零信用准入失败→lesion回滚逐字节恢复；Hadamard反例（秩1但4行范数0.5）、`C₁=−C₂`整体抵消（全局基线拒绝＋共享守卫拒绝）、软门N=1秩和2、重放非单调（秩2→1按有效结果接受）全部按声明判定。

执行期发现（按纪律落账，不改阈值）：①§24.4新生单元是门控的（R021 §4"实际新支路和诊断必须使用同一个g"），门控前向为v1.2候选声明实现；②稠密`[n,m,p]`布局下逐群不对称出生需R/P零填充，padding单元梯度恒0且永不启动；③**冻结0.99置换噪声bar在本夹具不拒绝随机门**——真实门与随机门的σ₁/τ_noise余量同为1.03–1.05倍，时间平滑生涯的共同模结构使置换零假设膨胀；已如实记录校准不可分辨性，作为R021 §4标定协议owner决策输入（分块置换/去均值零假设等候选留其后），本合同不缩quantile。分配/出生/标定/迁移均不签任务收益、保持或恢复；唯一下一步见总计划（09.2.4完整主体恢复）。

提交后按钩子要求重跑Mimosa深扫（focusFiles=本包四个代码文件）：jobId `scan-job-mvwhkswo`，scanId `scan-2026-10-31T14-20-21.713Z-1875cac11fde`，**封印`sha256:5df80279f075d3494802b55453abeb409448b5d8f7bc20d54e37b3f8c8b2ed0e`**，1,652项findings为既有普查基线、无新假设、离线advisory零匹配；同提交单主线审计ok。

## 2026-10-31 R024完整主体恢复（09.2.4）

先冻结[R024合同](../../reference/SPEC-TAIJI-CORE_RESTORE_R024_20261031.md)，再交付[state_migration.py扩展](../../../taiji/state_migration.py)（显式step序列/断点续推/save_subject/load_subject/resume_subject/run_worker_job）、[worker CLI](../../../scripts/training/r024_restore_worker.py)、[检查](../../../tests/taiji_native/test_subject_restore_r024.py)与[报告](../../../reports/taiji_core_r024_subject_restore_20261031.json)。`taiji/living_core.py`字节经SHA核对与R021/R022/R023绑定一致；报告`passed_r024_restore_engineering_only`、6/6检查通过、optimizer_steps=3（声明合成更新表达式）、data_splits_read=[]、费用6.16/900 CPU秒。

主体bundle（`taiji-subject-bundle-r024-v1`，`weights_only=True`可读）覆盖：主体id/revision/step/config、出生结构与门值表（来自出生时父端点trace的审计量）、全部参数、H/F/nu_ema/coverage/prior/keys、声明Adam(lr=1e-2,betas=0.9/0.999)状态、资格迹替身（e←0.99e+mean‖x‖/√p，纯记录量）、RNG状态、生涯spec与游标、单调journal（fee/outcome/resolve/restore条目，seq严格递增）与未结结果。五个边界各保存并由**multiprocessing spawn独立解释器进程**续推：下一微步h/probabilities、下一次声明更新后参数/矩/更新后再微步与父端逐位相等；RNG探针、资格迹与费用按公式延续不清零；resolution以当前journal唯一闸实现幂等（旧快照重放拒绝重复领奖）；journal seq单调。

执行期工程事实（落账）：Windows下torch张量不能过进程队列，结果改走文件传递；spawn目标必须按名可导入（worker job直挂`taiji.state_migration.run_worker_job`）；"下一次更新"的比较要求父端同样执行续更新再逐位比对，bundle中保存的是上次更新后的参数与矩。声明更新为合成表达式（教师强制重放分支一步Adam），不签训练；迁移恢复通过明列**不替代语义保持**。09.2.1–09.2.4全部关闭；唯一下一步见总计划（09.3连续动力学与内容活动），训练/定价仍未准入。

提交后按钩子要求重跑Mimosa深扫（focusFiles=本包四个代码文件）：jobId `scan-job-mvwj70dh`，scanId `scan-2026-10-31T15-05-26.437Z-71a2162c069e`，**封印`sha256:919231ee5cd8d383802216b21b46c349b9b8676ffa81978c9ac162048fbcf345`**，1,652项findings为既有普查基线、无新假设、离线advisory零匹配；同提交单主线审计ok。

## 2026-10-31 R025活动循环与可追溯选择（09.3）

先冻结[R025合同](../../reference/SPEC-TAIJI-CORE_ACTIVITY_R025_20261031.md)，再交付[native_activity.py](../../../taiji/native_activity.py)、[检查](../../../tests/taiji_native/test_native_activity_r025.py)与[报告](../../../reports/taiji_core_r025_activity_dynamics_20261031.json)。`taiji/living_core.py`字节经SHA核对与R021–R024绑定一致；报告`passed_r025_activity_engineering_only`、5/5检查通过、无optimizer、data_splits_read=[]、费用0.16/900 CPU秒。

交付事实：六活动（interact/recall/imagine/explore/maintain/wait）循环——策略头（seeded init、未训练、hash前后一致）从主体自身状态量（池化N(h)、goal向量、nu特征（EMA重建＋unknown标志）、预测误差、未结数、队列数）读入→掩码softmax→确定性argmax，宿主只提供时钟/预算/合法mask（interact需队列、recall需store、wait恒合法）；触发量快照在执行前取并逐轮入单调journal（logits/probs/chosen_by/触发量/h范数/step_delta/imagined/store_size）。想象事件＝最后事件编码＋声明种子固定范数扰动，**不写store**（store_size逐轮不变落账）；wait无前向步（H逐位不变、时钟推进）；maintain限位复查显式no-op。无新消息6轮内journal逐轮decision且非wait活动产生状态变化；模型臂与固定阈值对照臂在同一轨迹决策序列分岔（工程分岔核验，不签优劣）；policy头经bundle extra逐位恢复。

声明读数：预测误差＝世界头对**池化后继表示**（mean_i N(h_i)）的读数——世界头从z预测主体级d维后继，与逐群h形状不同；真实世界损失、EMA目标与可学习活动信用属09.4。模型政策未训练，其选择由声明头决定；本报告不签"模型学会选择"、活动策略优劣或自主性。唯一下一步见总计划（09.4持续可塑性与结构出生）。

提交后按钩子要求重跑Mimosa深扫（focusFiles=本包三个代码文件）：jobId `scan-job-mvwjzj1g`，scanId `scan-2026-10-31T15-27-31.295Z-9dcd94362c0a`，**封印`sha256:11e603a8ef3f3d4708d0d71c4c9b5ad3b7b8ec1cd942e78c7b7899cbd7c61be6`**，1,652项findings为既有普查基线、无新假设、离线advisory零匹配；同提交单主线审计ok、三套件回归19项全过。

## 当前候选的审批依据

| 统一步骤 | 批准/边界事实 | 原始依据 |
|---|---|---|
| 02、06、07 | 用户于2026-10-30批准R2/R4与R5，R5现已按原门结案 | [R4页首及§9](../../reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md) |
| 08 | 用户于2026-10-30选择“批准，按合同执行”；该合同实现与训练授权已成立 | 提交`8780f858e`与[R5b §10执行记录](../../reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md#10-审批方式)；本次整理不实施 |
| 09–13 | 原R6能力接力不由R5/R5b批准自动授权 | 对照总计划各步所需新范围/合同，不借原批准越门 |

## 步骤01–06的交付证据

| 步骤 | 交付出处 | 读数与适用限制 |
|---|---|---|
| 01 | [R1合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md)、[世界审计](../../../reports/taiji_native_core_r1_world_audit_20261030.json)、[17案fixture审计](../../../reports/taiji_native_core_r1_public_fixture_audit_20261030.json) | 设计语义及公开反例，不是未见任务能力 |
| 02 | [R2合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md) | Q1三类架构组，Q2图内reader/decoder信用比较；架构包与单因素分账 |
| 03 | [R3记录](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)、[结构探针](../../../reports/taiji_native_core_r3_probe_20261030.json) | 53/53结构门，公开fixture、零步独立恢复；本阶段0更新，后续训练见07 |
| 04 | [修正旧图探针](../../../reports/taiji_native_core_r3b_baseline_20261030.json)、[R3 §6](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#6-r3b1-修正旧图基线20261030) | 22/22，固定分母17/17，仅value_bias可训3,341；unknown回current、解析失败丢弃等基线限制保留 |
| 05 | [普通基线探针](../../../reports/taiji_native_core_r3b2_ordinary_20261030.json)、[R3 §7](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#7-r3b2-普通端到端基线20261030) | 11/11，固定分母17/17，全文prompt、无硬解析门、120,097参数全可训 |
| 06数据 | [课程审计](../../../reports/taiji_native_core_r1_course_audit_20261030.json)、[分区身份](../../manifests/native_core_r1_split_v1.json) | 360历史取216/72/72；1,440 episodes、11,520 turns、5,760应答轮；反事实成对不跨分区；本台账不加载分区正文 |
| 06工程 | [零更新预检](../../../reports/taiji_native_core_r4_preflight_20261030.json) | reader/decoder等组有任务梯度，content_encoder无（持久latent边界）；0 optimizer updates，不称已学会 |
| 06定价 | [定价报告](../../../reports/taiji_native_core_r4_pricing_20261030.json) | 新核心/旧图/普通单价5.45/0.94/8.20 CPU秒/更新；3×50 pilot；零步/第25步恢复6/6；冻结512更新/run。仅该定价阶段未读query/holdout/sealed |

## 步骤07的结果、费用与限制

原始来源：[训练报告](../../../reports/taiji_native_core_r5_training_20261030.json)、[评价报告](../../../reports/taiji_native_core_r5_evaluation_20261030.json)、[原合同§9](../../reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md#9-r5-执行结果20261030)。固定分母评分器为[native_content_evaluator.py](../../../taiji/native_content_evaluator.py)。

| 读数 | 实际结果 | 解释边界 |
|---|---|---|
| 执行与恢复 | 15/15 run，7,680更新；恢复15/15；无NaN/无预算停止 | 完成训练不代表过能力门 |
| 整体结果 | `FAIL_BEHAVIOR` | 按原冻结门；不追改阈值 |
| 相对比较 | holdout约0.323，比最好基线高约0.169；Q2信用差约0.117 | 有真实但不足的相对收益，绝对行为仍不合格 |
| 关键失败项 | cut_drop约0.166，训练内插约0.329，收据约0.429 | 内容选择和生成路径值得优先检验；不能凭这些单项穷尽全部原因 |
| 父能力保持 | 报告`retention_parent=none` | 训练内插不代证旧能力保持 |
| CPU费用 | run CPU和42,108.75秒；训练进程总CPU42,224.15625秒；评价另6,529.890625秒 | 不将run和当全部实耗，不漏评价；完整ledger取原报告 |
| 数据使用 | 评价已读query/holdout，sealed未读 | 后继复用相同面是回归比较，不重新称独立确认 |

## 步骤08的证据入口

[已批准合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md)保留原候选、两项改动、9-run清单、门值、预算及审批。提交`dc7a43354`提供[native_content_core_v2.py](../../../taiji/native_content_core_v2.py)、[训练适配器](../../../taiji/native_content_training_v2.py)、评价适配及[11个定向测试定义](../../../tests/taiji_native/test_native_content_core_v2.py)；本轮只读代码与提交，不运行测试，不能据文件存在签工程门通过。

静态对照发现的待核项：批准合同按episode抽样，适配器按turn抽样且缩短自前缀路径的实际监督字节；50步pilot不覆盖257后分支，草拟额外分支探针需具体方法/更新账修订。原合同不在本轮改写，细项及处置顺序已写入总计划08。后续真实工程/定价/训练/评价回执到位后再追加身份与费用；并行草稿不计完成。

提交`b78bdfdce`追加[定价执行件](../../../scripts/training/price_native_content_core_r5b.py)、[训练执行件](../../../scripts/training/run_native_content_core_r5b.py)、[评价执行件](../../../scripts/training/evaluate_native_content_core_r5b.py)；本轮未运行；执行件存在不等于工程/训练完成。收尾核对时工作区出现`reports/taiji_native_core_r5b_pricing_20261030.json`，status为`r5b_priced`，登记60更新、约572.56 CPU秒、约59.00墙钟秒，零步/两阶段端点同进程和新进程恢复均记相等，query/holdout/sealed读取均记false。本次未提交该并行机器件；原50步与60步差异、实际日程分支及监督字节仍需按总计划08审计，不凭status补签全部前置门。

项目所有者已正式将[总计划](../../MAINLINE.md)定位为新的N主线。03继续仅登记证据与审批；09–13详细方案不因此成为已冻结训练合同或默认模型采用批准。

### 08.1 合同—实现对照表（2026-10-31 实施方交付，偏差已按原合同纠正）

「原要求→实现→处置→证据」四栏对照。原实现（`dc7a43354`）存在三处与批准合同的偏差，均已纠正；纠正后 12/12 定向测试＋13/13 R5 回归测试绿（本轮实测运行，非仅文件存在）。

| 合同要求（R5b §2.1/§5/§7） | 原实现 | 处置 | 证据（定向测试，本轮实测绿） |
|---|---|---|---|
| §2.1-A 复制轨：读取路径选记录，被选记录 value span 原始字节经指针分布进入字节分布，与词表 softmax 按 gate 混合 | `next_logits`：`_read` 选 top-1→span 字节 `copy_key/copy_query` 位置 softmax→`index_add` 成 257 维 copy 分布→`copy_gate(cat(updated,read))` 混合取 log | 一致，未改 | `test_copy_track_math_is_the_declared_mixture`（混合公式逐步复算 allclose） |
| §2.1-A/§5 通道不接触 gold；宿主伪造选择必须被拒 | teacher/answer/scheduled 全部要求 store 绑定；`content_read_cache` 校验每条 condition 记录的 `record_sha256`+`raw_span` 属于声明 store | 一致，未改 | `test_forged_condition_record_is_rejected` |
| §2.1-A copy 参数归 decoder 信用组（Q2 stop 仍恰为 reader+decoder） | `credit_group_of("copy_*")→decoder` | 一致，未改 | `test_credit_stop_freezes_copy_parameters` |
| 空 store/切断退化＝v1 逐字节 | 空 content_ids 直接返回 v1 raw logits | 一致，未改 | `test_copy_track_reduces_to_v1_without_content`、`test_cut_store_degenerates_to_v1` |
| §2.1-B 抽样单位＝**每 episode** 以 0.5 概率 | coin 按 `episode_id+turn_index` 逐应答轮抽样 | **偏差①纠正**：coin 去掉 `turn_index`，只按 (seed, update, episode_id) | `test_own_prefix_coin_is_pure_and_declared` |
| §2.1-B 教师强制**目标字节不变，仅换前缀来源** | 自前缀分支只监督后半段（监督字节缩短为 len−len//2） | **偏差②纠正**：scheduled 分支对**全目标**产出监督行，仅前半段的输入源换成自身贪心字节；曝光（监督字节数与目标）与纯 TF 分支逐字节一致；行数≠目标长度即抛错 | `test_scheduled_forced_conditioned_is_deterministic_and_byte_exact`（rows=len(target)）、`test_v2_adapter_loss_terms_byte_accounting`（两分支 supervised_bytes 相等） |
| §7/08.3 探针须证明**实际执行的分支**、日程位置与监督字节 | 原 phase B 强设 `schedule_update=300` 被 `R5BRunTrainer.update()` 每次重置回 `steps+1`（**phase B 实测的是纯 TF**；已产出的定价报告失真） | **偏差③方法修订**：新增 `R5BRunTrainer.schedule_position_override` 声明式探针机制，逐更新记账 `schedule_position/schedule_override_active/episodes_scheduled/scheduled_supervised_bytes`；定价脚本 phase A 断言 0 scheduled、phase B 断言 >0，否则工程失败 | `test_trainer_override_keeps_probe_on_declared_branch`；失真报告 `reports/taiji_native_core_r5b_pricing_20261030.json`（未提交）已删除，定价按修订方法在 08.3 重跑 |

runner（`b78bdfdce`）另补 `--run-id`（默认 20261031），避免与已停草稿的写一次目录冲突。

**并行草稿账（不计任何回执）**：按偏差纠正前的实现启动的 9-run 训练已按 08.1 停止，停在 `checkpoints/native_core_r5b/20261030/`（3 个端点 seed0/1/2＋q2_credit_on_seed0 零步，无报告）；耗 CPU 约 1.5 万秒，按 R5b §7 计入同一 120,000 CPU 秒总预算账。作废定价件 60 更新约 572.56 CPU 秒同样计入。预算余量约 10.4 万 CPU 秒，覆盖修订后重定价（≈0.1 万）、正式 9×512 训练（≈4.3 万）与评价（≈1 万）。

08.1 对照表不能代替 08.2 工程回执；下一步按总计划 08.2 交付信用/隔离/训练态恢复证据（含 256→257 边界与恢复后下一次更新）。

### 08.2 工程回执（2026-10-31 实测，`r5b_082_ok`）

[回执](../../../reports/taiji_native_core_r5b_082_engineering_20261031.json)、[探针脚本](../../../scripts/training/probe_native_content_core_r5b_082.py)、实现 `1315ff959`。五族门全过（无 failed_gates）：

- **信用**：同初态 credit_on/stop 前向逐位一致；逐 loss 组读数——on 侧 reader/decoder/core/attention/output/query_encoder 全部有梯度，stop 侧 reader+decoder 零梯度、冻结参数逐位不变、其余组确认位移（optimizer 真实生效）。
- **gold 与隔离**：宿主伪造 condition 记录（含可复制 span 的外来记录）被 store 绑定校验拒绝；跨 scope 读取为空、本 scope 恰 1 条。
- **退化与切断/救援**：带内容时复制轨激活（与 v1 分歧）；空 store 下 v2 与 v1 在**同一 cut condition** 下逐字节相等；救援答案与基线逐字节一致。
- **旧 v1 可加载**：R5 冻结端点 `q1_new_core_v1_seed0` 以原报告记录 digest（e8550592…）一致加载。
- **256→257 日程边界与训练态恢复**：连续 257 更新 vs 256 处 checkpoint 新进程恢复后下一更新——schedule 位置 256（0 scheduled）→257（2 scheduled，coin 按声明触发）；恢复后下一更新 loss 与连续 run **逐位相等**（0.3529213984619896）；恢复后端点 probe digest 与连续 run 端点一致（8ee2b741…）；cursor（run_id/seed/frozen_groups）、optimizer 步数、exposures 恢复记账在回执。

账目与作废记录：工程探针共 515 优化器更新、正式回执 4,707 CPU 秒/405 墙钟秒；此前三次脚本缺陷（函数序 NameError、GATES 键名 KeyError、写一次残留）与一次探针比较 bug（误把 V2 空 store 对 V1 带内容比较；实现无缺陷）各作废一跑，累计约 1.4 万 CPU 秒，连同作废回执（未提交已删）一并计入预算账。**累计已耗约 3.5 万/120,000 CPU 秒，余约 8.5 万**，覆盖 08.3 重定价（≈0.06 万）、08.4 正式 9×512（≈4.3 万）与评价（≈1 万）。08.2 是工程门，不代能力；query/holdout/sealed 未读。

### 08.3 定价与冻结（2026-10-31，按修订方法，`r5b_priced`）

[定价报告](../../../reports/taiji_native_core_r5b_pricing_20261030.json)（作废件已删后按修订方法重出）。分支证据齐备：phase A 50 更新 `scheduled_episodes=0`（纯 TF 区间符合声明）；phase B 10 更新经 `schedule_position_override=300` 声明式探针，`scheduled_episodes=19`、positions 逐更新记录全为 300、监督字节 4,336 逐更新记账——自前缀分支被真实执行并有账。**冻结单价 9.148 CPU 秒/更新**（两阶段中位数取大）；9×512 投影 ≈42,150 CPU 秒；零步/phaseA/phaseB 端点同进程＋新进程恢复 6/6 相等；定价成本 568.8 CPU 秒；query/holdout/sealed 未读。预算账：累计约 3.56 万/120,000 CPU 秒，正式训练后预计约 7.9 万，余量充足。

### 08.4 训练执行（2026-10-31，`r5b_training_complete`）

[训练报告](../../../reports/taiji_native_core_r5b_training_20261030.json)：9/9 run、4,608 更新、40,259 CPU 秒/0.97 墙钟小时；15 恢复门（零步＋端点×9）全过、无 NaN、无预算停止；每 run `scheduled_updates=256`（257–512 日程全程激活）；Q1 与 Q2 credit_on 端点 digest 逐 seed 复现（3/3）。credit_stop 臂 loss 停在 ≈1.1（reader+decoder 冻结，与设计一致）；credit_on 臂 4.6→0.11–0.19。

### 08.5 评价与结案（2026-10-31，`FAIL_BEHAVIOR`）

[评价报告](../../../reports/taiji_native_core_r5b_evaluation_20261030.json)（首次读 query/holdout，sealed 未读；基线按合同 §4 复用 R5 报告）。**判定：`FAIL_BEHAVIOR`**——failed_gates＝behavior 五门＋`retention.train_like_exact`＋`retention.receipt_exact`；比较/机制/成本/恢复门全过。逐门真数（holdout 3,456 轮，3 seed：0.525/0.484/0.506）：

| 门 | R5b 实测 | R5 对照 | 门槛 | 判定 |
|---|---|---|---|---|
| `answer_exact` | **0.505** | 0.323 | ≥0.80 | 败 |
| `current_query` | 0.352 | 0.173 | ≥0.85 | 败 |
| `previous_query` | 0.619 | 0.207 | ≥0.75 | 败 |
| `unknown_query` | 0.698 | 0.737 | ≥0.85 | 败 |
| `state_exact` | 0.750 | 0.750 | ≥0.90 | 败 |
| `cut_drop` | **1.000** | 0.166 | ≥0.50 | 过 |
| `rescue_gain` | 1.000 | 1.000 | ≥0.90 | 过 |
| 相关/无关交换改变差 CI 下界 | **0.990** | 0.020 | >0 | 过 |
| holdout−训练内插 | **0.009** | −0.006 | ≤0.10 | 过（自由生成差距已闭） |
| 训练内插 `answer_exact` | 0.497 | 0.329 | ≥0.85 | 败 |
| 收据 | 0.655 | 0.429 | ≥0.95 | 败 |
| holdout 净收益 | **0.352**（CI [0.332,0.371]） | 0.169 | ≥0.05 | 过 |
| Q2 信用差 | **0.390**（CI [0.370,0.410]） | 0.117 | ≥0.05 | 过 |
| 成本比 | 1.07 | 0.714 | ≤2.0 | 过 |
| 恢复 | 15/15 | 15/15 | 全过 | 过 |

**架构归因（按合同 §9 出口，不改门、不加剂量、不重跑）**：复制轨把「答案是否消费选中内容」从 R5 的旁路（cut_drop 0.166、收据 0.429）变为**因果已证实的消费**（cut_drop 1.000、交换差 0.990、收据 0.655）；自前缀日程把教师强制/自由生成差距基本闭合（内插−holdout 差 0.009 过线，内插绝对值 0.329→0.497）。剩余缺口是**绝对正确率**：答案现在真实经过选中内容，但选中/复制的值只有约一半正确（current 0.352 最弱），unknown 0.698 距 0.85 亦不足；`state_exact` 0.75 与 R5 持平（状态写路径与本次两改动正交，未受益也未受损）。按 R4 §4.6，R6 未准入；后继需另立有界合同针对「选中正确值」的读取精度，而非复用本门重跑。

费用：训练 40,259＋评价 4,769＋定价 569＋08.2 探针 4,707＋作废跑约 1.4 万＋并行草稿约 1.56 万 ≈ **8.01 万/120,000 CPU 秒**（合同 §7 预算内收案，非 budget_stop）。

### 08.5 后继裁定支持证据（2026-10-31，train-only 只读诊断，`diagnosis_complete`）

[诊断回执](../../../reports/taiji_native_core_r5b_selection_diagnosis_20261031.json)、[脚本](../../../scripts/training/diagnose_native_content_r5b_selection.py)。2,592 个采样 train 应答轮（216 episode×3 q1 端点，seed 20261030；query/holdout/sealed 未读、0 优化器更新）：**选择错误桶（选错记录/检索失败）为零——读取对正确记录的选择在 train 上已是 100%**；全部缺口在 `selection_correct_answer_wrong`＝0.435（选中正确内容后复制/表达执行产出错误字节；current 0.681、previous 0.376、unknown 0.282）。**裁定框架修正**：若立后继合同，靶点应是「选中正确内容后的复制/表达执行精度」（复制指针分布/gate/解码器执行），而非「选中正确值」的读取精度（该环节无错可选）。owner 裁定点保持开放：立后继合同（靶点依此修正）／全局收束转 09 静态方案／暂停保留现场。诊断成本 475 CPU 秒，计入预算账（累计约 8.06 万/12 万）。

### 08.5 后继裁定支持证据之二（2026-10-31，复制/表达失败分解，`diagnosis_complete`）

[回执](../../../reports/taiji_native_core_r5b_copyexpr_diagnosis_20261031.json)、[脚本](../../../scripts/training/diagnose_native_content_r5b_copyexpr.py)（q1 seed0 端点，同 216 episode 样本；train-only 只读，357 CPU 秒）。三个发现：①**纯教师强制 argmax 已不是有效探针**——TF 下 value 字节 argmax 仅 0.157、template 仅 0.038-0.053，与训练末段 loss 0.145 nats 矛盾的唯解释是自前缀日程刻意改变了输入分布，模型技能存在于自生成 regime（与 08.2 边界回执一致）；②**current 类错误 79% 在值字节之前的模板早期发散**（自生成时模板句式先错），previous 类 91% 在值字节本身；③unknown 全部为模板类发散。**对裁定①的含义**：后继合同若立，靶点＝生成侧复制/模板执行（双表面：current 的早期模板发散＋previous 的值字节拷贝），且设计冻结前需用与自前缀 regime 一致的探针（TF 探针无效）。owner 三选项裁定保持开放；诊断累计 ≈830 CPU 秒，预算账累计约 8.07 万/12 万。

### 08 后继裁定状态（2026-10-31，owner 已裁定：①批准 R5c）

owner 经弹窗裁定「① 批准 R5c」，[合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R5C_COPYEXPR_CANDIDATE_20261031.md)自 §5 生效；§5.1 设计冻结已批准前预完成（指针 CE 目标规则覆盖 91.67%、权重 0.3、梯度路径实测通过），直接进入实现。预算账：R5b 累计 8.07 万/12 万已收案；R5c 独立预算 60,000 CPU 秒/48h 自行记账。09 保持未准入。

### 08 后继裁定支持证据之三（2026-10-31，发散点归因，`diagnosis_complete`）

[回执](../../../reports/taiji_native_core_r5b_divergence_attribution_20261031.json)、[探针](../../../scripts/training/diagnose_native_content_r5b_divergence.py)（q1 seed0，同 216 episode 样本；270 CPU 秒，train-only 只读）。**429 个错误答案在首个分歧字节处 100% `guided_wrong`**——喂入完全正确的目标前缀后 argmax 仍选错字节且无恢复。机制判定冻结：训练信号问题（模型在这些位置自信学错分布），非推理漂移、非解码脆弱。R5c 草案 §2.1-D 已据此冻结为完整可批形态（C＝值字节辅助 CE 对表面 A；表面 B 预声明修订＝同预算内 current 类自前缀暴露加权）。裁定三选项保持开放；诊断累计 ≈1,100 CPU 秒，预算账累计约 8.07 万/12 万。

### 08 后继裁定支持证据之四（2026-10-31，R5c 设计冻结支持，`freeze_support_complete`）

[冻结回执](../../../reports/taiji_native_core_r5c_design_freeze_20261031.json)、[探针](../../../scripts/training/freeze_native_content_r5c_design.py)（train manifest 统计＋seed0 端点只读梯度探针，≈3.4 CPU 秒）。R5c 草案 §2.1-C 全部自由参数已预冻结：指针 CE 目标规则（value 子串单次出现 91.67% 覆盖、零出现 216 轮剔除、从不多次）、辅助权重 0.3、梯度路径实测通过。**草案至此零悬空项，批准即 Fully executable**；三选项裁定保持开放，未批准不执行 §5 任何一步。

### 08 裁定档案补充：预注册复审议程触发（2026-10-31）

`MEMORY.md`（本段原引的外部治理记录，仓库未提供可核验原件）的触发条件「同机制归因×3 → 路线复审议程」已满足：同一内容消费机制轴上连续三次架构归因指向同一结论——R5（答案未消费选中内容＋生成差距）→R5b（复制轨修复消费后绝对正确率仍短）→发散归因（训练信号问题，100% guided_wrong）。**此触发不替 owner 做裁定**，但为选项②补足实质：若转全局收束，路线复审议程应覆盖（a）复制/表达机制在该课程域的学习信号设计（辅助监督、暴露加权之外是否还有结构性缺口）、（b）state_exact 0.75 与两轮正交的独立路径、（c）课程域外泛化无证据的现状。裁定三选项与此议程一并呈报 owner。

### 08-R5c 执行与结案（2026-10-31，`FAIL_BEHAVIOR`，机制性负结果）

owner 批准后按 §5 顺序全链条走完：工程回执 `r5c_engineering_ok`（v3==v2 架构逐位一致、辅助 CE 梯度路径、信用/gold/边界恢复全过）→ 定价 `r5c_priced`（单价 10.46 CPU 秒/更新，A 相 0 scheduled/B 相 19 scheduled＋辅助项全触发）→ 训练 9/9（44,797 CPU 秒，恢复全过，辅助项逐 run 触发）→ 评价 [报告](../../../reports/taiji_native_core_r5c_evaluation_20261031.json) 按原门结案 **`FAIL_BEHAVIOR`**。

**机制性负结果**：指针辅助监督（权重 0.3）未修复复制/表达缺口，反而轻微变差——holdout answer_exact **0.458 vs 前身 0.505（delta −0.047）**；current 0.243（前身 0.352，更差）、previous 0.588（0.619）、unknown 0.758（0.698，更好）；收据 0.586（0.655）。内容消费机制保持完好（cut_drop 1.0、交换差 1.0、rescue 1.0），比较门仍过（+0.304 CI [0.285,0.323]）、Q2 信用差 +0.373、成本比 1.16。三 seed 0.507/0.442/0.425。**归因**：向选中记录值字节的指针辅助监督不改善且略损生成精度——「对指针分布直接加监督」这一假设被否定；缺口仍是训练信号问题，但辅助监督的施加位置/形式需重新假设。

费用：R5c 独立预算内收案——工程探针 5,160＋定价 651＋训练 44,797＋评价 5,066 ≈ **5.57 万/6 万 CPU 秒**，非 budget_stop。按合同 §7：不加剂量、不重跑、不改门；R6/09 保持未准入。预注册「同机制归因×3→路线复审议程」已触发且归因次数增至四次，全局路线复审为下一裁定节点。

## 继承的历史事实与限制

| 历史资产 | 已有事实 | 保留边界 |
|---|---|---|
| N175/N176 | N175 R为0/3，S未全判；N176固定1,360份生成与消费登记已完成，F2未解析 | 按原输出/hash/合同留存，不重复生成或借新计划代判 |
| N177.1–177.5 | 状态机、来源消费、结构化T2/T3、T1 producer与恢复已有有限工程证据 | 结构化Goal/状态不代证自然目标理解、环境学习或跨会话能力；默认Adapter未替换 |
| E0 | 15臂fresh-process 15/15，历史revision复算60/60，隔离/mask 8/8 | 仅固定train输入，不代签任务能力 |
| E2c/训练态E3 | 信用on/stop 12/12×2，零步恢复17/17，禁写组3/3 | 旧选择器无回答信用，真实链解析覆盖1/4单列 |
| F0a/F0b | F0a train/copy均1.0；F0b 12/12 run，实测4.68 CPU秒/更新，恢复24/24 | F0a不代泛化；F0b wall超96秒如实留账 |
| F1/F1b | 分别`FAIL_COPY_ONLY`/`FAIL_NO_STATE_EFFECT`，低分与费用不撤销 | 目标字节错位、时态接口、hidden旁路和指标混杂限制归因；旧cut/rescue不代正确性因果收益 |
| N178 | face2解析局部改善，22/29/24/64，非完整回答收益 | 采样seed不等于全参数独立初始化；面已用于选剂量，后继属开发诊断，不重置F累计停止线 |

详账见[阶段收束](../../reference/PROJECT_CONSOLIDATION_20261029.md)、[核心设计重审](../../reference/TAIJI_CORE_DESIGN_REVIEW_20261030.md)、[旧资产索引](../../reference/LEGACY_ASSET_INDEX.md)。历史表的处置不构成当前待办。

## 历史恢复、CI与资料归属

原逐轮正文可从[历史恢复登记](../../reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md#6-20261026活动计划重编)与Git恢复。历史R2中间checkpoint清理已有[逐项回执](../../../reports/taiji_r2_historical_checkpoint_prune_20261027.json)，该既有事实不授权本轮进一步删除。

历史fast lint修复及回执见收束记录；历史CI绿不代当前提交CI绿。每包只检查受影响范围，保留无关未提交工作。计划放plans，机器报告放reports，模型和数据沿受控根；当前排期全部回总计划，03不再复制执行清单。
