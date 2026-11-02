# 态极新的N主线：连续生命主体与递归进化落地计划

> 文档状态：[主线] — 本文件是项目当前N方向的唯一主线，集中维护统一编号、执行位置、阶段状态、落地工作项及出口。专项合同提供细则，其他计划不再重复排期。

更新：2026-10-31，Asia/Shanghai。新N主线发展同一持续活动、持续可塑且能按需规模生长的认知主体；共同内容学习是其组成。步骤01–13沿用统一编号。本轮冻结R021工程接口并交付正式参考前向，R022–R024完成09.2全部工程前置，R025完成09.3活动循环与可追溯选择；只运行合成工程检查，未执行任务训练、定价，不改既有冻结实验。

## 执行位置

**步骤08已于2026-10-31按原门结案：`FAIL_BEHAVIOR`（08.1对照→08.2工程回执→08.3定价→08.4训练9/9→08.5评价全部按序走完，费用8.01万/12万CPU秒）。** 08.1三偏差（逐turn抽样、监督字节缩短、定价分支失真）已按原合同纠正（`1315ff959`）；08.2五族门全过含256→257边界恢复逐位复现（`6b62d09eb`）；08.3按声明式分支探针重定价，单价冻结9.148 CPU秒/更新（`8af3d50f7`）；08.4训练9/9恢复全过；08.5评价（[报告](../reports/taiji_native_core_r5b_evaluation_20261030.json)）判负但归因清晰：复制轨使内容消费因果成立（cut_drop 0.166→1.000、交换差0.990、收据0.429→0.655）、自前缀闭合约束自由生成差距（内插−holdout 0.009过线），剩余缺口=绝对正确率0.505/0.80（current 0.352最弱、state 0.75与R5持平）。R6未准入；不自动加剂量、不重跑；费用与证据见[03](active/roadmap/03_CURRENT_EXECUTION.md)。步骤01–07保留原交付与`FAIL_BEHAVIOR`结案。

**09.2.2已于2026-10-31按[R022合同](reference/SPEC-TAIJI-CORE_CREDIT_R022_20261031.md)通过结案：`passed_r022_credit_only`（[报告](../reports/taiji_core_r022_credit_validation_20261031.json)，费用54.6/900 CPU秒）。** 101个参数张量在三个声明负责目标（T_EXPR教师强制预测NLL、T_WRITE2两步快写链、T_WORLD_PROXY合成代理）上全部获得FD确认的信用或落入声明零类：87 credited＋12 world代理＋V birth_zero＋`episodic.score.bias`判明为softmax平移不变的死参数（inert，建议后续R021接口修订处置）。FD三eps扫描最优失配≤1.45e-10、二阶方向导数100/100吻合（≤1.9e-6）、置零/翻符号/许可/排序/空view断路负对照与U激活对照全过、float32/64逐组中位偏差2.6e-7（98已解析张量）；声明两步Adam更新表达式落账Δθ/矩/ULP，输出改变0.163，V在U更新后解锁（9.4e-4），世界族与inert参数Δθ精确为0/可忽略。更新后的参数即废弃，不落盘、不结算成模型。

**09.2.3已于2026-10-31按[R023合同](reference/SPEC-TAIJI-CORE_GROWTH_R023_20261031.md)通过结案：`passed_r023_birth_engineering_only`（[报告](../reports/taiji_core_r023_growth_birth_20261031.json)，费用2.0/900 CPU秒）。** 在R021图残差P/R对上交付：K=2门控信用构造（copy/generate混合门，审计可溯）、逐(支路,群)128次置换噪声标定、信号条件、max-min整数分配（枚举81态/群、tie-break与预算停实现）、`GatedGrowthCore`门控出生（`pre += Σ_c g_c·B_c·tanh(R_c x)`，living_core.py字节不变）与全局SVD基线对照臂。反例电池8/8：中性逐位成立、4单元启动梯度与`B^T C_c`在1e-12精确吻合（8.5e-4..2.5e-3＞1e-8）、L_start=1.0、独立方向比例1.0、kappa=0.998、P新列出生时刻梯度精确0、启动零信用回滚、Hadamard/整体抵消（C₁=−C₂全局基线拒绝而共享支路守卫拒绝）/软门N=1秩和2/重放非单调按有效结果接受、迁移事务恒等重放逐位相等＋编码变更提交revision+1＋预算拒绝保父。**校准真发现**：冻结0.99置换bar在本夹具上不拒绝随机门（真实/随机门σ₁/τ_noise余量同为1.03–1.05倍，时间平滑共同模结构抬高位假设）——已按"不改阈值"纪律如实入账，作为R021 §4标定协议owner决策输入。稠密布局的逐群不对称出生以R/P零填充实现（padding单元梯度恒0）。

**09.2.4已于2026-10-31按[R024合同](reference/SPEC-TAIJI-CORE_RESTORE_R024_20261031.md)通过结案：`passed_r024_restore_engineering_only`（[报告](../reports/taiji_core_r024_subject_restore_20261031.json)，费用6.2/900 CPU秒）。** 主体bundle（`taiji-subject-bundle-r024-v1`，weights_only可读）覆盖主体id/revision/step、config、出生结构与门值表、全部参数、H/F/nu/coverage/prior/keys、声明Adam状态、资格迹替身、RNG状态、生涯spec与游标、单调journal与未结结果；五个边界（无新输入空闲、首次声明学习后、门控出生后、编码迁移提交后、未结结果挂起）各保存一次并由**multiprocessing spawn独立解释器进程**续推：下一微步h/probabilities、下一次声明更新后参数/Adam矩/更新后再微步输出全部与父端**逐位相等**；RNG探针、资格迹与费用按公式延续不清零；resolution以当前journal唯一闸实现幂等（旧快照重放不重复奖励）；journal seq严格单调。**09.2.1–09.2.4至此全部关闭。**

**09.3已于2026-10-31按[R025合同](reference/SPEC-TAIJI-CORE_ACTIVITY_R025_20261031.md)通过结案：`passed_r025_activity_engineering_only`（[报告](../reports/taiji_core_r025_activity_dynamics_20261031.json)，费用0.2/900 CPU秒）。** 新模块[taiji/native_activity.py](../taiji/native_activity.py)交付六活动（interact/recall/imagine/explore/maintain/wait）循环：策略头从主体自身状态量（池化N(h)、goal向量、nu特征、预测误差、未结数、队列数）读入→掩码softmax→确定性argmax选择，宿主只提供时钟/预算/合法mask；预测误差＝世界头对主体级后继表示的声明读数；想象事件不写store（journal逐轮imagined标志＋store_size不变）；wait无前向步、时钟推进、H逐位不变；maintain限位复查显式no-op落账；无新消息6轮内journal逐轮decision条目且至少一轮非wait产生状态变化；模型臂与固定阈值对照臂在同一轨迹上决策序列分岔（逐字记录，不签优劣）；策略头参数哈希前后一致（未训练）并经bundle extra逐位恢复。

**09.4已于2026-10-31按[R026合同](reference/SPEC-TAIJI-CORE_PLASTICITY_R026_20261031.md)完成同父主体六臂预注册比较：`passed_r026_preregistered_comparison`，预注册裁决＝`negative`（[报告](../reports/taiji_core_r026_plasticity_arms_20261031.json)，费用45.9/900 CPU秒）。** [native_development.py](../taiji/native_development.py)实现六臂（coverage_birth/gradmax_schedule/conditional_svd/random_gate/replay_only/capacity_random，每臂独立父副本、8步声明Adam更新、独立holdout/retention探针生涯）。结果：coverage_birth的ΔNLL_holdout（−0.1370）未优于对照（min=−0.1402，差+0.0032＞1e-3边际）；六臂全部超出0.05保持容差（0.144–0.148，声明更新幅度主导）；臂间holdout差≤0.0035。**诚实结论：在合成夹具与声明更新规模下，"覆盖/信用改善转为后继净收益"被预注册判据否定；出生方向在8步内未自我偿付，更新幅度本身破坏保持**。按合同不追加剂量/seed/臂；该负结果不证伪v1.2候选于真实条件，09.5按序进入。

**09.5已于2026-11-01按[R027合同](reference/SPEC-TAIJI-CORE_GATES_R027_20261031.md)通过结案：`passed_r027_gates_engineering_only`（[报告](../reports/taiji_core_r027_subject_gates_20261101.json)，费用3.5/900 CPU秒）。** R024主体bundle与R025活动循环按09.5清单合拢：统一主体生命序列上八次边界保存（交互后/无新提示两轮/睡眠等待/首次声明学习后/出生前/出生后/编码迁移提交后/未结Outcome），每次由spawn独立解释器进程恢复续推，活动轮决策logits与h、下一次声明更新后参数、resolution幂等全部与父端逐位一致。清单逐项核验落账：参数/拓扑（含增长结构）、optimizer/eligibility、RNG探针、游标（career cursor＋活动/账本seq单调）、记忆（store逐字节＋非法来源过滤带provenance）、编码revision（迁移提交revision+1）、预算（fee/probe延续不清零）、receipt（当前journal唯一闸）。query-only探针在detached副本上运行，subject状态逐位不变；外控触发（external_trigger）与内生轮（chosen_by）分账。执行期修正：门控子模型的迁移重放与活动轮必须携带门表（replay_transaction新增provider参数；fit门表对career外步按模回绕，声明仪器）。

**09.6.1 五项工程补齐已于2026-11-01按[09.6.1合同](reference/SPEC-TAIJI-CORE_ENGINEERING_20261101.md)结案：`failed_lesion`（[报告](../reports/taiji_core_r029_engineering_gaps_20261101.json)，26项检查中1项红）。** 五项缺口全部实现并各自过门，唯lesion的贡献判据按预注册分支判负——这是负结果结案，不是工程失败。交付：[constrained_learning.py](../taiji/constrained_learning.py)（§10.2受约束更新：active-set/KKT/互补松弛＋实际探针回退）、[activity_learning.py](../taiji/activity_learning.py)（活动价值头＋P_R探针奖励＋半马尔可夫资格迹，`TRACE_LAMBDA=0`为即时奖励夹具的声明值）、[native_task_fixtures.py](../taiji/native_task_fixtures.py)（5类确定性中文任务＋捷径/对调探针）、`growth_credit.lesion_trained`（只删新生单元、保留已训练父行）、`state_migration`的worker自报CPU与`retain_outputs`、`native_activity`的可选`model_action`。`taiji/living_core.py`字节未改。

读数（原理形状、09.6中文长生涯）：**§10.2约束更新**8步全接受（11次尝试、3次实际探针回退、ρ逐次减半），fit NLL降0.0307817（门≥0.010）、保持漂移0.0162904（门≤0.030）、ΔNLL_holdout−0.0172613、KKT残差1.94e-16、约束违反0.0；负对照Adam(lr=1e-3)漂移0.1111525复现超0.05。**代价必须与门一起读**：无约束对照fit降1.9725041、ΔNLL_holdout−0.3056635，收益是约束臂的64倍与18倍但突破G1；能声称的只是"约束更新是当前唯一同时满足G1与G2的配置"，不是它学得更好。**lesion**：出生对父函数逐位中性、未训练副本lesion后逐位回到父输出、训练后lesion逐位保留已训练父行且只移除新生单元（258→256，8单元/4组）三项全过；但训练收益0.0201929中移除新生单元只减少4.33e-7（占比2.1e-5，门≥1e-4）⇒判`lesion_does_not_remove_benefit`。**活动学习**：policy/critic摘要前后不同、learner状态逐位往返、暴露256未越上限、三臂核心参数摘要逐位相同；累计P_R训练臂0.0／未训练臂−0.0319197／固定阈值臂−0.0198944，优势+0.0198944判`activity_learning_advantage`——**但机制已查明是零成本空转**：训练臂32轮全选`maintain`、累计恰为0.0，优势数值等于对照臂烧掉的计算成本；成因是声明夹具只放一条外部消息、`interact`在32轮中最多合法一次，空转为结构性最优。报告以`advantage_is_idle_cost_avoidance`显式披露。**中文夹具**：应答轮分母25/28/10/25/16逐实例等于`career_step_specs`的round数；捷径尺有动态范围且无一满分（最新来源2/5、最早来源1/5、弃答2/5），记录反序后最新降为1、最早升为2证明顺序敏感；两个运行时模块源码中gold/category令牌计数全0。**子进程CPU**：09.6把`fresh_spawn_worker.cpu_seconds`记为**0.0**，09.6.1在同面bundle（77,304,316字节、optimizer state 123项）上实测子进程自报**0.8125 CPU秒**而父侧等待CPU为0.0——恢复成本此前不是被低估而是被记为零；八边界8次worker调用全部自报、0缺失（其中两次轻量调用受Windows 15.6毫秒时钟粒度限制自报0.0）。

**第三次独立否证结构生长收益**：R026六臂预注册`negative`、09.6的coverage−replay=−4.72e-7、09.6.1的新生单元贡献占比2.1e-5，三者同向。且09.6.1的lesion收益0.0201929与09.6在同lr下测得的coverage Δholdout−0.020193逐位吻合，说明两台仪器读的是同一现象：**在当前夹具与lr=1e-5下，覆盖出生没有可测贡献，观测到的全部收益来自继承父行的参数更新。** 该结论不得读作仪器故障，也不得据此反签结构有效。回归：R022–R027共33 passed、09.6.1新增15 passed；整树`ruff check .`与`black --check .`全绿（2371文件）、冻结标记审计passed；mypy核心面（CI基线0）本机实测44处错误，已按三组精确归因（分组单跑求和＝整树数）：A组20处/5文件与CI在上一次推送报的`Found 20 errors in 5 files`逐数吻合⇒**该门在本轮之前已红**；B组23处/5文件属R021–09.6参考图族，由本地领先origin/main的21个未推送提交带入（落在本轮改动文件的错误全在未改动行）；C组0处＝09.6.1三个新模块单跑`Success: no issues found in 3 source files`（原有1处`no-any-return`已修）。**本轮新增0处类型错误；A/B两组43处为既有债，未在09.6.1范围内结清，且推送这21个提交后CI仍会在该门红。**

**owner已于2026-11-01就09.6.1结案三裁**：①09.7＝**先改夹具再立项**（不花训练预算；活动面需要反复出现的外部触发——现行夹具32轮里`interact`最多合法1次，使最优解退化为零成本空转；生长面需要能让新生单元承担收益的任务）；②mypy核心面43处既有错＝**只登记为债、本轮不修**（[DEBT-I11](active/roadmap/05_TECH_DEBT_REGISTER.md)）；③本地领先`origin/main`的21个提交与09.6.1这一包＝**都不动，等owner排期**（不提交、不推送）。

**09.6.2 夹具改造已于2026-11-01交付，两项读数都是负结果，但性质不同。** 生长面：六格（lr∈{1e-5,1e-4}×steps∈{8,32,128}）全部发布，新生单元份额随步数**转负**（+2.15e-5→−2.76e-6→−7.13e-5），同期holdout收益从0.0202涨到0.3242——**「训练太慢」这一竞争解释被排除**：收益涨了16倍，但涨的全是继承父行，新生支路在长程训练后反而成了轻微拖累；唯一为正的格（1e-4/128）同一格保持漂移1.170（门0.030，超39倍）、holdout收益−0.4643，是发散不是可达。裁决`growth_fixture_no_reachable_cell`，按合同不改网格重测。活动面：夹具修好了——解析下界与脚本贪婪策略同为0.704、空闲0.0、单次交互实测净赚+0.044，确有0→0.704动态范围；但训练臂在声明种子与两个诊断种子上都是"交互一次后停在maintain/recall/imagine"（份额0.0627／−0.0253／0.0185）。机制随后查明：把到达改成每轮补一条后，同样三个种子给出**完全相同**的读数1.40807、32/32轮全选`interact`、份额1.00 ⇒ 失败源于`TRACE_LAMBDA=0`使「wait→到达→交互」这两步序列在信用上不可见，不是夹具、也不是"交互不划算"。

**09.6.3 出生惰性诊断已于2026-11-01交付（[报告](../reports/taiji_core_birth_inertness_20261101.json)，38.1 CPU 秒）。** 它又排除两个竞争解释：新生 R 行出生时恰为 0，8 步后长到 0.006730653（不是冻结的死参数）；它们拿到父行梯度范数的 6.13%（holdout）／5.32%（fit），初始 P 列范数是父列的 12.4%（不是小到可忽略）。在这三条都成立的情况下，新生支路对 holdout 的贡献仍只有 4.33e-7，同期 0.020192874 的收益全由继承父行承担 ⇒ 剩下的解释只有一条：**分支信用取的是 fit 窗口，它选的方向能让 fit 动，但学到的动作不迁移到 holdout**。这是可立新假设的中间量，本合同不据此提出也不批准任何新机制。

**09.6.3b 方向可达性上界已于2026-11-01交付（[报告](../reports/taiji_core_direction_oracle_20261101.json)，29.4 CPU 秒）。** 把出生方向换成神谕方向（holdout 梯度在各群 P 上的前几个奇异向量，只作可达性上界、不是可用方法），其余与 09.6.3 完全相同：收益 0.02019838、新生贡献 5.888e-6、份额 2.915e-4——**比 fit 信用方向高 13 倍且过了份额门（≥1e-4）**；但按 09.6.1 §2.3 的绝对门（贡献 ≥1e-4 NLL）仍差 17 倍。所以两个方向都成立、也都不足以单独下结论：**"任何方向规则都不行"被否证**（给对方向时新生单元确实变得有信息），**"方向规则是唯一瓶颈"也未证成**（给神谕方向时绝对贡献仍只有 5.9e-6）。

该 09 节点的历史去向（**已过期，不作当前指令**）：当时待裁的是是否按 09.6.3b 立一个针对方向规则的新机制假设；owner随后裁定按单点立项 09.7（已结案）。份额门已可达（神谕 2.9e-4），而绝对门（≥1e-4 NLL）即便用神谕也未达到。若不立新假设，则 09 按"结构生长在本参考图上份额可达但绝对收益不成立"收束，10–13 不依赖覆盖出生。活动面按 owner 裁定采用 `per_round=True` 逐轮供给，结论只能写"逐步选择"（1.40807／3 seed 一致）。09.6.1–09.6.3b 的交付作为既有基础，不重做。正式训练仍未批准。

**09.6.4 方向规则候选已于2026-11-01交付，判据自检失败（[报告](../reports/taiji_core_birth_direction_20261101.json)，128.2 CPU 秒）。** 规则是 fit 生涯内多个不相交子窗口的共识方向（不读评估集，AST 级不泄漏守卫）。四臂同轮：绝对贡献 共识K=2 **1.426e-4**／K=4 7.32e-5／fit-SVD基线 4.33e-7／**随机负对照 1.99e-5**；份额 7.014e-3／3.612e-3／2.146e-5／**9.836e-4**。合同第 3 条负对照如期起作用：**随机方向也过份额门**，所以预注册的份额判据不能把真规则与随机分开，H1 不得判为支持，状态按出口为 `metric_not_discriminating`。基线同轮复现偏差 0%；原字符串泄漏检查踩到模块自己的禁令句（自指误报），已改为 AST 标识符扫描。

**09.6.5 曾按 owner 的"改绝对贡献＋随机对照"重测，但种子轴是假的、结论作废**：`LivingCoreReference` 用内部固定种子初始化，全局 `torch.manual_seed` 不改变父模型——三个"种子"产出的模型摘要逐位相同（`921dc9981187406a`×3），共识臂三次读数也相同，所以"3 种子 ≥2 支持"只是同一配置跑三遍。旧报告按纪律保留、状态作废；其中仍然成立的只有"同一确定性主体上三个随机对照 1.988e-5／−1.720e-6／7.284e-6 均在绝对门下"。

**09.6.6 换真实变化轴已复现：`hypothesis_h1_supported`（[报告](../reports/taiji_core_birth_direction_parts_20261101.json)，357.2/1800 CPU 秒、0 失败检查）。** 变化轴改为规则自身的窗口数：parts=2 得 **1.426e-4**、parts=3 得 **1.343e-4**、parts=4 得 7.320e-5（未过）；随机对照最大 1.988e-5（差 5 倍），fit-SVD 基线在三格上逐位复现 4.332e-7，漂移全部 ≤0.030，泄漏扫描零命中 ⇒ 2/3 格过绝对门。**能声称的只有一句**：在当前参考图、合成中文生涯、确定性主体、8 步 lr=1e-5 下，把出生方向换成"fit 生涯多子窗共识"后，新生单元承担了 1.3–1.4e-4 的 holdout 改善，而随机方向最多 2.0e-5、旧规则 4.3e-7。

**09.6.7 稳定性面已交付：`platform_supported`（[报告](../reports/taiji_core_birth_direction_surface_20261101.json)，756.5/1800 CPU 秒、0 失败检查）。** 15 格全发布：8 格过绝对门、覆盖 parts {2,3,5,6}，保持漂移全在 0.0221–0.0223（门 0.030），随机对照全部低于门、基线逐位复现、泄漏零命中。两条规律值得单独记：效应随出生单元数**单调增强**（parts=5 时 7.99e-5 → 1.21e-4 → 1.74e-4），而 **parts=4 在三个单元数下全不成立**（3.24e-5／7.32e-5／9.50e-5）——恰是 09.6.3/09.6.4 用过的默认值，早期单点结论正好踩在最弱那一格上。

**09.6.8 净收益对照已交付：`net_benefit_supported`（[报告](../reports/taiji_core_birth_netbenefit_20261101.json)，66.2/600 CPU 秒、0 失败检查）。** 同一确定性父、同样 8 步、lr=1e-5：只重放臂 Δholdout **−0.020192402**（训练 21.688 CPU 秒），共识出生臂 **−0.020366326**（22.656 秒），**净收益 +1.739e-4**；出生构造 10.531 秒 ＋ 额外训练 0.968 秒 ＝ **11.50 CPU 秒**，即约 **6.6 CPU 秒／每 1e-4 NLL**（定价输入，合同不设成本门槛）。新生单元 lesion 贡献 1.7374e-4 与 09.6.7 同格逐位吻合，两臂保持漂移均 ≤0.030。**这是 09 线上第一份正的净收益读数**，也是"覆盖出生在当前参考图上没有可测贡献"这一持续结论的第一个反例——但它的范围必须钉死：单一确定性主体、合成中文生涯、8 步 lr=1e-5、单一参考图，parts=4 仍不成立，且不含任何任务能力。

唯一下一步是：**owner先审阅[算法重审](reference/TAIJI_ALGORITHM_REVIEW_20261101.md)，裁定[10.21修订提案](reference/SPEC-TAIJI-CORE_COMPOSITION_10_21_20261101.md#7-算法重审后的修订要求未批准)是否立项**。原冻结头方案保留为对照/上界设计，不直接按300步训练开工：10.19的16字节上限小于六个最短成功见证的18字节，0/6不能否证组合能力；须先冻结有oracle动态范围的生成/目标选择性复核，再审完整目标→条件读取/绑定→负责参数任务信用。旧≤4k/300步/600CPU不自动授予联合学习或提额；**未批准不实现新候选、不训练**。10.20原收案和11未准入状态保留，只收窄“缺组合算子”的因果措辞。目标现状地图仍见[03](active/roadmap/03_CURRENT_EXECUTION.md#20261101-目标条款证据对照现状地图不是收尾)。

2026-11-01重审补充：20份关键卡/当前实现的[阅读快照](../reports/taiji_algorithm_review_20261101.json)不新增模型试验。当前保持门/回退使用旧NLL绝对变化，测对称稳定性，不直接证明遗忘；旧8步边界只按旧冻结口径解释，前瞻合同区分真实损伤与稳定性。生长净收益保留，但正例实际为共识方向/常开门/固定数量，不给条件覆盖或自主生长整包签字；“只换margin”或“再加冻结小头”均不作为已证补救。源卡与原退出不改，本轮不动运行代码、权重、现有提交或推送。

08/R5c已经批准的有限实验按原预算/终点独立收束，结果作为内容/表达资产；本次不停止、恢复或追加进程，也不改冻结门。新核心设计不以R5c过门或旧模型兼容为前提，现有权重可不再采用；历史证据保留；09后续实现/计时/实验需按新算法规格与资源边界进入。本轮只修订计划，不启动持续服务或新训练。

## 1. 长期目标与当前主计算图

沿N主线，发展具有连续内部状态、持续可塑性与继承式规模生长能力的态极原生认知模型，使其自身可学习、可恢复的认知机制实际承担决策与表达。模型应能以自然中文可靠完成连续任务，在同一主体身份下跨会话积累经验、核验并纠正认识，从可核验的经验与反馈中改进负责机制，学习新能力并保持已核验且仍有效的旧能力；在授权和资源边界内，按需新增、整合或回收计算单元、连接与群体，继承已学参数、经验与有效功能，并根据干预的真实结果调整后续学习、活动和生长选择。必要时即使没有新提示，也能根据目标、状态和未结事项，自主选择继续、回忆与学习、求证、求助、休息或等待。

各阶段预先声明任务范围、数据与反馈边界、资源预算和收官条件，以身份与谱系可追溯的同一候选及其前后版本，对照合理基线，按范围验证未见任务表现、必要机制的实际贡献、反馈对负责模块的作用、旧能力保持、主体连续性与恢复一致性、结构生长与整合的真实收益、主动选择效果及全过程资源成本；成功、负结果、不可判、工程失败或预算停止均按预定出口结案。

权重、认知/生命状态、记忆、目标、拓扑和发展历史共同构成运行主体。递归进化、结构可塑性与规模生长绑定，规模可暂时不变，但不被永久写死。离线训练与冻结评价是研究/测量方式，不定义主体的发展终点；基础定义取[根需求§3.0](active/TAIJI_CORE_REQUIREMENTS.md#30-连续生命活动与递归进化的共同底座)。信用秩、覆盖出生、R021等是当前可替换的方法/工程候选，不写成长期目标必须成立的理论或永久实现。

本轮“新N主线”是项目执行方向的正式重定向，不是新核心已证明优越、原生认知已完成或默认模型已切换。旧N实验号与全部正负证据保留；旧N1–N6继续作为机制责任轴，本主线以步骤01–13调度。步骤08是一次有限候选比较，不能代替后续自然输入、经历选择、持续学习与主动决策的能力建设。

共同运行循环为：

```text
同一主体的状态/目标/生命调节/记忆/参数/拓扑
  → 选择活动：交互、预测、回放/巩固、想象、探索、结构维护、休息/等待
  → 执行活动并取得有来源的结果
  → 更新状态、负责参数与必要结构
  → 同一主体的后续状态与发展判断
```

外部消息是事件入口之一。下列内容子图嵌入这个周期，不再被当作整个模型：

```text
原始输入及来源 → 可学习事件/内容编码 → 版本化持久内容状态
query/目标＋时间意图 → 内容读取 → 随生成前缀逐步消费内容的表达/行动
合法任务反馈 → 本候选声明的负责参数与状态
```

- 结构化解析作为辅助视图、监督和基线；未解析内容可带来源/状态进入候选存储，不自动变成可信事实。
- 任务世界的实体/属性/合法取值清单用于生成与评分，不成为自然输入的宿主语义查表器或唯一准入门。字节编码集合与任务域清单分开。
- 当前/历史/未知的时间语义、来源、权限及版本不能被软学习绕过；旧值查询不得悄悄改用当前值。
- 读取与内容条件表达均需实际任务信用；编码/写入若冻结或跨持久latent断图，必须单列，不宣称已由回答目标塑形。
- 同一bundle保存参数、store、编码revision、optimizer、RNG与cursor等适用状态；训练前验证完整保存及独立进程恢复。

具体神经计算不由上述流程图代替：[算法主稿](reference/TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md)§4–13提供v1.1底座，§24给出当前v1.2生长信用与覆盖出生候选。保留递归图、快/慢更新和表达公式，不要求继承旧GRU/TSK权重；本轮不以新增组件分散研究增量，也不将外围调度当核心算法。设计与局部数学预审不等于算法已实证有效。

## 2. 编号与阅读规则

**执行编号只用「步骤01–13」；子项用「08.1」这样的十进制层级。** 下表是唯一执行队列，后文沿用同一编号。已完成步骤不重排、不重跑；新增工作只在所属步骤下追加子项，或经方案修订新增后继步骤。

R1、R3b、R5b等保留为原合同/实验身份，不再另作排期编号；N1–N6是机制责任轴，CC/VX是职责与历史映射，L2–L5是验收等级，文件名03/10是兼容路径，均不与执行步骤混用。原合同文件、报告、审批、数据hash和checkpoint身份不改名。

只需先读本文件；做到某一步再打开该行所链合同。未执行工作的详细卡按[08](#08-复制轨候选本轮实际执行包)、[09](#09-连续生命与递归进化共同内核)、[10](#10-自然目标后果预测与有限行动)、[11](#11-自然中文连续任务与最小用户验收)、[12](#12-跨会话纠错反馈学习与旧能力保持)、[13](#13-有边界主动选择与条件扩域)排列。证据细账见[03台账](active/roadmap/03_CURRENT_EXECUTION.md)，能力采用条件见[07](active/roadmap/07_MINI_MODEL_DELIVERY.md)。[10](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)和[旧重构方案](reference/PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)仅保留兼容导航/设计依据，不维护第二张执行表。

## 3. 唯一执行队列

| 步骤 | 要完成的工作 | 当前状态 | 本步完成条件与依据 |
|---|---|---|---|
| 01 | 统一任务、时间、字节与评分语义 | 已完成：设计与公开fixture | [R1合同](reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md)：current/previous/unknown、来源/版本、首字节监督、全部应答轮固定分母一致 |
| 02 | 定义新核心、两基线及信用比较 | 已批准并已用于首轮比较 | [R2合同](reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md)：Q1架构包与Q2图内信用分开，同数据/权限/资源 |
| 03 | 实现新核心与零步恢复 | 结构交付完成，后已参加步骤07 | [R3记录](reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)：53/53结构检查；不把结构通过写成学会 |
| 04 | 接入修正旧图基线 | 结构交付完成，后已参加步骤07 | [R3b-1记录](reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#6-r3b1-修正旧图基线20261030)：22/22，固定分母，声明冻结语言头/仅bias可训等限制 |
| 05 | 接入普通端到端基线 | 结构交付完成，后已参加步骤07 | [R3b-2记录](reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#7-r3b2-普通端到端基线20261030)：11/11，全文输入、无硬解析门、全参数可训 |
| 06 | 物化课程、测成本并冻结比较合同 | 已完成：`r4_priced` | [R4合同](reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md)：216/72/72历史分区，恢复6/6，冻结512更新/run及原数值门 |
| 07 | 执行首轮有界比较并结案 | 已结案：`FAIL_BEHAVIOR` | [R5结果](reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md#9-r5-执行结果20261030)：有相对净收益，绝对行为/内容依赖/内插等门未全过；不自动追加训练 |
| **08** | **实现并检验复制轨候选** | **已两轮按预定出口结案：R5b `FAIL_BEHAVIOR`＋R5c `FAIL_BEHAVIOR`（2026-10-31，结果为内容/表达资产）；[R5c评价](../reports/taiji_native_core_r5c_evaluation_20261031.json)已入账** | [R5b合同](reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md)（answer 0.505/0.80，内容消费因果成立，选择零错误）＋[R5c合同](reference/SPEC-NATIVE-CONTENT-CORE_R5C_COPYEXPR_CANDIDATE_20261031.md)（指针辅助监督机制性负结果：0.458 vs 前身 0.505，current 更差；内容消费与比较门保持）；同面回归，不称新独立确认 |
| **09** | **连续生命与递归进化共同内核，包含自然内容/经历能力** | **09.2关闭；09.3活动循环（R025）；09.4六臂预注册negative（R026）；09.5恢复门已过（R027）；09.6定价已交付；09.6.1五项工程缺口已实现并按预注册出口结案`failed_lesion`（第三次独立否证结构生长收益）；09.6.2夹具改造已交付：生长网格六格全否（第四次否证，且排除"训练太慢"）、活动夹具修好并查明失败源于`TRACE_LAMBDA=0`的两步信用（逐轮供给变体3/3种子达上界）；09.6.3出生惰性诊断已交付：新生单元**确实被训练**（R行0→0.00673、拿到父行6.1%梯度）但贡献仍4.33e-7 ⇒ 剩余解释指向"信用方向对holdout无用"；09.6.3b神谕方向上界：份额2.915e-4（过份额门，比fit方向高13倍）但绝对贡献5.888e-6（差绝对门17倍）⇒ 方向规则确是瓶颈之一、但不是唯一；09.6.4方向规则候选：份额判据自检失败（随机也过门），但绝对贡献上共识1.426e-4对随机1.99e-5；09.6.5改绝对判据但**种子轴是假的**（父模型跨全局种子逐位相同）⇒结论作废；09.6.6换真实轴（窗口数）复现：2/3格过门（1.426e-4／1.343e-4，parts=4为7.32e-5）⇒`hypothesis_h1_supported`；09.6.7扩15格稳定性面：**8格过门、覆盖parts{2,3,5,6}**⇒`platform_supported`（薄平台，最大1.737e-4；parts=4三档全不成立）；09.6.8净收益对照：共识出生比不出生多改善**1.739e-4** holdout，代价出生11.50 CPU秒（≈6.6秒/每1e-4 NLL）⇒`net_benefit_supported`；09.6.9换第二评估生涯复核：两条生涯净收益+1.739e-4／+1.903e-4⇒`replicated_on_second_holdout`；09.7单点32步片段：净收益升至7.187e-4／8.147e-4但**两臂漂移0.084752／0.085631破门**⇒`retention_breach`；09.7b定位可持续点：**仅8步在门内**（净收益1.739e-4／1.903e-4），16步起两臂同步越线⇒卡住的是训练预算不是出生；owner两裁＝生长面不按原假设立项／活动面逐轮供给且只称逐步选择；owner三裁＝先改夹具／先补净收益／接受8步结案且转入09.8；09.8全局收束`closeout_passed_engineering_only`（六件齐备、继承135/135逐位、spawn恢复逐位、bundle 77.6MB不入git）；队首＝owner裁定09.9/10准入（T3合同尚无立项依据）；训练未准入** | 同一主体连续活动、学习与结构出生/有效接入/保持/恢复；v1.2增量以信用覆盖与重放—生长假说为主，不把前向通过、秩或活单元数当能力 |
| 10 | 自然目标、后果预测与有限行动 | **10.1环境已冻结：`passed_10_1_environment_only`（三条设计判据全过、四条捷径均不能满分）；10.2绑定契约与规则基线完成：`passed_10_2_binding_contract_only`（五类反例全入分母、缺条件与歧义强制拒答）；10.3契约完成但首测判 `planner_no_advantage`（任务太浅、贪心已最优）⇒ 加做 10.1b 给世界加容量1，重测通过 `passed_10_3_prediction_contract_only`（规划在2格严格优于贪心）；10.4 第一份模型侧选择读数 `model_below_random`（主体2.81=随机2.81<贪心3.86，19/19 全选杯子句）；10.5 换目标条件化接口后 `contrast_below_floor`（两动作 contrast 差仅8.018e-5＜1e-3，目标效应是共模的）；10.6 动作特定构造 `action_gap_below_floor`（动作差4.804e-4，比10.5大6倍仍不到门、方向相反）；10.7 一次有界训练 `training_below_chance_plus`（lr=1e-4 损失降37%但留出仍3/6、漂移0.0322破门；机制＝学到的是常量偏置，训练只换了常量）⇒ **四条路都没造出目标条件化选择，P2 仍未满足**；10.11 收束卡 `closeout_10_evidence_complete`（九项复核全过、`p2_satisfied=false`，收口决定权仍在 owner）；10.7a 两臂权重回落逐位复现（最大绝对差 0.0、命中 3/6 一致）；10.7e 目标→动作读出留出 **6/6**、对调对照 **0/6**（1,024 参数、35.8 秒，主体逐位冻结）；**owner 当日裁定"收口 10＋立项读出实验"**——10 以负结果＋这条夹具级正读数收口，11 仍未准入；**10.20 能力线机检收案**（18 份卡，唯一声称已被 10.7e-1 撤回，机制＝单跳 copy 有、组合没有）；**owner 同日再裁：接受负结果、转候选设计** | 只有声明行动/规划能力时启用；不成为首次L3的普遍前置；10.1 只验任务本身，未涉及模型 |
| 11 | 同一候选的自然中文连续任务与最小验收 | **准入合同已交付：`step11_not_admitted`——P1／P2／P4 不成立，P3 已成立（11.7 安全读数补齐）** | 按07满足L2＋至少一项代表能力＋安全/恢复/性能，才触发L3范围的用户验收；[准入报告 v3](../reports/taiji_core_l3_admission_11_v3_20261101.json)（v1／v2 保留；v3 把 10.4–10.7 四张模型侧读数纳入 P2 扫描面，结论不变）逐项带来源字段：**P1** 07 裁决仍是 `FAIL_BEHAVIOR`；**P2** 四条模型侧读数全为负且均在册（`model_below_random`／`contrast_below_floor`／`action_gap_below_floor`／`training_below_chance_plus`）；**P3** 恢复与性能有读数，安全项由 11.7 卡支撑（20 次打分逐位可重复、0 写／0 socket／0 子进程）；**P4** 无候选可追溯到 09.8 的同一主体摘要 |
| 12 | 跨会话纠错、持续学习与保持的长跨度验收 | 后继范围，未准入；基本学习从09开始 | 在09同一主体基础上检验多session/分布变化/长期保持，按L4范围举证，不把参数学习推迟到本步才开始 |
| 13 | 自主发展策略与大规模成长/扩域验收 | 后继范围，未准入；生命与成长基础从09开始 | 逐步扩大主体可自主选择的干预/结构范围，对照策略及同最终容量模型；不是首次加入生命或结构成长 |

宏观关系是：01–08保留局部研究与资产；09建立连续生命/可塑性/结构发展的共同内核，10–11验证它的任务和中文表达能力，12–13扩大持续学习、内生发展与规模范围。连续状态、学习和结构生长贯穿共同内核，不再排成“静态模型学好→最后可选加生命”。成熟自主策略和长跨度能力仍分阶段举证，既不删除目标，也不一次要求全部长期机制完成。

## 4. 顺序执行工作卡

### 01–06 已完成的前置交付

保留上述合同、结构探针、课程审计和定价报告，不因改编号重新执行。课程首版为12实体×3属性×每属性4值，每槽最多3合法版本；时间意图为current/previous/unknown；未解析内容可带来源/状态入store，硬解析作辅助与基线。完整语义以步骤01合同为准。

三类架构组是新核心、修正旧图和普通端到端；新图内credit_on/stop是单因素比较。R5中的内容编码器未获得任务梯度，读取/表达组有梯度；不将前向可达或非零梯度等同于未见任务收益。定价期未读query/holdout，步骤07评价已消费这两个面，二者按时点分账。

### 07 首轮比较的结论与保留边界

15/15 run、7,680更新完成，按原门结案`FAIL_BEHAVIOR`。holdout answer_exact约0.323，较最好基线高约0.169，Q2信用差约0.117；但cut_drop约0.166、训练内插约0.329、收据约0.429，绝对行为和相应机制/内插门不足。原结果、阈值和费用均保留，不能写成通过，也不能写成完全没有收益。

这些读数支持优先检验内容消费和自由生成路径；它们没有单独证明全部正确回答都走了旁路，也没有证明增加更新永远无效。步骤08同时改变复制轨与训练前缀日程，只能先判架构包收益，不能将收益全部归给其中一个改动。

R5报告的`retention_parent=none`：其中“保持门”实际含训练内插探针，**尚无相对父模型的旧能力保持证据**。运行CPU、训练进程总CPU和评价CPU分别落账，不能漏掉评价与恢复成本。

### 08 复制轨候选：本轮实际执行包

**前置与边界。** 候选`new_core_v2`继承`new_core_v1`，合同已于2026-10-30获用户批准，原始记录见[R5b §10](reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md#10-审批方式)。`dc7a43354`已提交核心、训练/评价适配和11个测试定义，`b78bdfdce`已提交专用执行件；文件及测试定义的数量不代替实际通过回执。本轮只把余项写实，不改两项候选变化、9-run配置和冻结门。

| 子步骤 | 按序实施 | 必须交付及验收 | 未满足时停在哪里 |
|---|---|---|---|
| 08.1 合同对齐 | 对照批准合同逐项核对复制轨、日程、每episode抽样、目标对齐与实际监督字节；将代码差异列成“原要求/实现/处置/证据”四栏 | `native_content_core_v2.py`、`native_content_training_v2.py`与合同的对照记录；先按原合同纠正实现偏差。若需改变监督含义、候选或曝光，则先形成具体合同修订 | 实现提交不等于本项闭合；存在未决差异不得进入标定/正式训练 |
| 08.2 工程与恢复 | 执行公开反例、逐loss信用/禁写组、同初态on/stop前向和完整训练态恢复；特别覆盖日程256→257边界及恢复后下一次更新 | 实际探针/测试回执、代码SHA；模型/store/optimizer/RNG/cursor/日程种子/已用预算恢复；无gold、空store退化、切断与救援、跨scope隔离及旧v1可加载证据 | 任一必要工程门失败，停在具体接口；不以已有测试文件补签通过 |
| 08.3 定价与冻结 | checkpoint先通过，再按批准方法进行50更新train-only pilot；单位成本与正式资源投影一起落账 | 真实CPU/wall、分支计数、监督字节、恢复开销、预算余量；未测路径明确标未测，不能凭phase名称认定已测 | 新增日程分支探针先修订具体方法和更新账；批准总上限不等于可任意追加pilot |
| 08.4 训练与评价 | 准备v2专用加载和9-run清单：Q1候选3＋Q2两臂6，seed0/1/2、512更新、batch4；训练后才读批准评价面 | 单写者run-root锁、原子checkpoint、可恢复游标/资源账、每run零步/端点恢复；基线端点/报告hash复用；全部应答轮固定分母及原门逐项读数 | 无运行器、错版本、泄漏、NaN、保存/恢复失败或触限，按原合同停；不回退旧v1/15-run命令凑执行 |
| 08.5 结案与独立性决策 | 按原R4判定顺序裁决全部必需门，报告相对收益、内容消费、自前缀效果范围、父保持缺项及总成本 | 成功/负结果/不可判/工程失败/预算停止；给出结构约束、信息/信用链、因果与竞争原因、一个后继；同步本表及03 | 失败先全局收束，不自动加更新/seed/数据；PASS_Q1也不能直接采用或进入09训练 |

**已识别而未修复的实施核对项。** 以下来自静态读取，仅列待办，由实施方处理：

- 日程：批准合同写“每episode以0.5概率”，当前适配器按`episode_id＋turn_index`逐应答轮抽样；自前缀路径还缩短实际受监督目标字节。必须先核对抽样单位和目标监督含义，不将“目标数组未变”当作同曝光。推荐先使实现回到原合同；确需改变时另记批准修订，保留原版本与成本。
- 定价：50步均在纯教师强制区间，不能宣称实测了257步后的增耗。当前定价执行件另有10步探针；收尾出现的工作区机器件也登记了60更新与`r5b_priced`，CPU约572.56秒。它与原50步pilot描述的差异须补齐具体方法/更新账修订，不能把status当作已获全部方法批准。该机器件的分阶段字段未给出实际分支计数和监督字节；验收须补这些读数与实际日程位置。仅写`self_prefix`标签或临时赋值不足，trainer可能在update时重设日程位置。
- 运行器：旧R5脚本固定v1/15run及v1加载链。现已提交的v2执行件须验收加载、清单、恢复及原门覆盖后才登记正式命令；未存在的运行器不写成可执行命令。Q1与Q2重复credit_on端点只能算复现，不能扩大独立样本数。

**资源与裁决不变。** 原合同为50更新pilot、正式9×512更新、总上限120,000 CPU秒/48墙钟小时、单位成本停止因子1.5，包含pilot/训练/评价。具体门仍逐字见[R5b §6–7](reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md#6-数值门沿用-r4-4145-原值不改)及R4，本页不重抄阈值。

**08到09的判门。** R5b复用已用于选方向的query/holdout，只支持同面回归结论；若正向到足以建议进入新自然经历范围，08.5须同时完成新确认面/污染审计与父能力保持协议的决策。可把独立验证并入09首个冻结任务，而非再开无任务意义的探针。没有有效父候选或关键内容机制证据时，09只推进独立静态方案，不执行依赖性训练；sealed不因本页出现而获准读取。

### 09 连续生命与递归进化共同内核

**对象与交付。** 主体身份、内部状态、参数、拓扑、经历和发展账本共同延续；自然内容/时间绑定与经历选择是其任务活动。先把已训父模型或明确标记的原型初态接入同一运行周期，不将每轮`fresh_bundle`的离线训练例当作生命连续性。09.1首稿已交付且不依赖R5c过门，后续实现/实验则另按该合同冻结范围与资源；旧固定任务可作对照和保持集，但不是主体的全部世界。

| 子步骤 | 实施与代码落点 | 交付/验收与停止 |
|---|---|---|
| 09.1 核心神经算法设计 | 设计一个统一候选：单元/连接、状态更新与传播、内源驱动、共同表征、预测/任务信用、快慢可塑性与记忆、结构出生/整合、活动策略。明确变量维度、方程/精确步骤、反馈来源和一个完整时步；随后才推导主体id/schema/API/保存 | 算法设计书、实际计算/信用图、初始化与成长迁移、资源复杂度、最小反例及竞争解释；必须能直接导向最小实现。接口清单/未定义的advance函数/调度流程不算算法；草案存在也不算已证实有效 |
| 09.1.1 原式审查与反例（本轮完成） | 核对所有状态、归一化、时间、门、目标及平均尺度；先在原式上构造合法反例，分清有界/不放大/收敛/任务效力 | 保存复现参数与来源，不把一次局部passed扩大成全图稳定或能力结论 |
| 09.1.2 耦合代理数值与梯度核验（原范围完成） | 小维度多群/多步合成代理做自动微分与有限差分，检查h/F/部分参数、新单元启动、无写入行为、变时长TD/资格迹 | 只用合成输入、0优化器更新；缺连续chi与正式Q/K/V/查询读出，多组读取/输出矩阵不在梯度检查内；原22项通过仅限代理，不读实验数据或签正式全图 |
| 09.1.3 修订与复测（局部范围完成，v1.2预审已补） | 保留E1/E2与旧11/22项；v1.2预审明确独立方向秩界、硬/软门差异、覆盖分配、重放非单调及相关先验 | 新14项0优化器步骤，包含原推广命题反例；不签新理论/任务效力。GradMax、TINY、梯度秩/重放先例纳入差异审查，正式全图仍待09.2 |
| 09.2 最小计算与持续状态参考实现 | 按下列09.2.1–09.2.4逐项推进，贯通实际计算、信用、一次出生、坐标迁移及主体恢复；新对话不重置主体 | 只交字段、代理MSE、总梯度norm或恢复权重均不足；所有工程前置通过后才定价/申请正式训练 |
| 09.2.1 方程与参考图对齐（R021范围完成） | [工程合同](reference/SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021_20261031.md)六接口先冻结；实现可学群间/快QKV、两层情景/null、查询多头、连续chi、位置复制/生成及world前向。覆盖矩阵逐条指向源码行号、101个参数张量、状态与检查 | 12项float64前向oracle在atol/rtol=1e-12通过；无backward/optimizer/数据分区，非h.mean或固定读取矩阵。只签参考前向，未实现学习/出生/迁移/完整恢复均在回执明列 |
| **09.2.2 信用与数值判别（R022范围完成，2026-10-31）** | [合同](reference/SPEC-TAIJI-CORE_CREDIT_R022_20261031.md)声明三负责目标与零类；[检查](../tests/taiji_native/test_living_core_credit_r022.py)在R021实际图检查连续chi、正式QKV/查询/全部101个声明可学参数的一阶/二阶导数；FD三eps扫描＋尺度匹配容差、置零/翻符号/许可/排序/空view断路负对照、U激活对照、float64/32逐组比较；[报告](../reports/taiji_core_r022_credit_validation_20261031.json)逐张量落账 | 87 credited＋12代理＋V birth_zero（实际更新解锁）＋1 inert（`episodic.score.bias`平移不变死参数，留待R021接口修订）；无意外断图、无低于噪声未分辨；FD最优失配≤1.45e-10、二阶≤1.9e-6、fp32中位2.6e-7；两步声明Adam更新Δθ/矩/ULP落账，输出改变0.163，不签任务收益。参考前向的requires_grad已逐项代之以确认信用；真实任务仍需后续有界比较 |
| 09.2.3 信用覆盖出生与坐标迁移（R023范围完成，2026-10-31） | [合同](reference/SPEC-TAIJI-CORE_GROWTH_R023_20261031.md)＋[growth_credit.py](../taiji/growth_credit.py)/[state_migration.py](../taiji/state_migration.py)：K=2门控信用、逐(支路,群)标定、max-min分配、`GatedGrowthCore`门控出生与全局SVD基线；§10.3 journal恒等重放/编码变更提交/预算拒绝；反例电池8/8（[报告](../reports/taiji_core_r023_growth_birth_20261031.json)） | 中性逐位、启动1e-12交叉验证（L_start=1.0、独立方向1.0、kappa=0.998）、P新列零梯度、零信用回滚、Hadamard/抵消/软门/非单调反例按声明判定；**校准真发现＝冻结0.99置换bar不拒随机门（余量1.03–1.05倍）**，阈值未改、留R021 §4 owner决策；分配/出生/迁移不签任务收益 |
| 09.2.4 完整恢复及下一更新（R024范围完成，2026-10-31） | [合同](reference/SPEC-TAIJI-CORE_RESTORE_R024_20261031.md)＋[state_migration扩展](../taiji/state_migration.py)（bundle schema/断点续推/worker job）＋[worker CLI](../scripts/training/r024_restore_worker.py)：五边界保存、spawn独立进程续推、声明更新表达式、resolution幂等闸、单调journal；[报告](../reports/taiji_core_r024_subject_restore_20261031.json) | 下一微步/下一次更新/更新后再微步与父端逐位相等；RNG探针、资格迹与费用延续不清零；旧快照重放不重复奖励（resolution以当前journal唯一闸）；journal seq单调。迁移恢复通过不替代语义保持；09.2.1–09.2.4全部关闭后按序进入09.3，不自动开训 |
| 09.3 连续动力学与内容活动（R025范围完成，2026-10-31） | [合同](reference/SPEC-TAIJI-CORE_ACTIVITY_R025_20261031.md)＋[native_activity.py](../taiji/native_activity.py)：六活动循环、模型所有策略头（掩码softmax＋argmax，触发量逐轮入journal）、想象不写store、wait/maintain语义、固定阈值对照臂、bundle携带policy恢复；[报告](../reports/taiji_core_r025_activity_dynamics_20261031.json) | 无新消息6轮journal逐轮decision＋非wait状态变化；两臂序列分岔核验（不签优劣）；策略头未训练（hash前后一致）；预测误差=世界头对池化后继的声明读数（真实世界损失属09.4）。不签模型学会选择或自主性 |
| 09.4 持续可塑性与结构出生（R026范围完成，2026-10-31） | [合同](reference/SPEC-TAIJI-CORE_PLASTICITY_R026_20261031.md)预注册六臂同父比较＋[native_development.py](../taiji/native_development.py)（声明反馈更新/四类出生臂/独立holdout-retention探针）；[报告](../reports/taiji_core_r026_plasticity_arms_20261031.json) | **预注册裁决negative**：coverage_birth未优于replay_only/capacity_random（差+0.0032＞1e-3）；六臂全部超0.05保持容差（声明更新幅度主导）；臂间差≤0.0035。按合同不追加剂量；负结果不证伪真实条件下的v1.2候选；新容量与功能贡献未分开成立 |
| 09.4.1 预期自我修改研究 | 仅在09.4基础证据与保持/恢复成立后，冻结世界模型预测出生/巩固/重放的旧损伤、新收益和成本的最小测试 | 当前09.4保持全臂超限，本子项**不触发**，待决策 |
| 09.5 主体工程与恢复门（R027范围完成，2026-11-01） | [合同](reference/SPEC-TAIJI-CORE_GATES_R027_20261031.md)＋[native_activity扩展](../taiji/native_activity.py)（env序列化/来源过滤/门控索引）＋[worker activity_round模式](../taiji/state_migration.py)：八次边界保存×spawn独立恢复、清单逐项核验、query-only隔离、外控/内生根因分账；[报告](../reports/taiji_core_r027_subject_gates_20261101.json) | 活动轮决策与h、下一次声明更新、resolution幂等逐位一致；非法来源剔除带provenance；fee/probe不清零；seq单调；query-only探针subject状态逐位不变。不签语义保持或自主性 |
| 09.6 生命片段定价与冻结（范围完成，裁定见09.6.1） | [合同](reference/SPEC-TAIJI-CORE_PRICING_20261101.md)＋[定价脚本](../scripts/training/verify_taiji_living_fragment_pricing_r028.py)：原理形状中文长生涯、活动轮、出生全链、迁移、保存/新进程恢复与六臂实测；学习率校准冻结`lr=1e-5`；[报告](../reports/taiji_core_r028_life_fragment_pricing_20261101.json) | 定价`passed_r028_pricing_engineering_only`、190.5/1800 CPU秒；冻结六臂、3 seed、G1–G7数值门、P_R=0/P_A=18/P_C=18与1800 CPU/3600墙钟预算。合同明列功能约束更新、活动策略学习、lesion、自然任务生成器和子进程CPU五项未实现缺口；**训练准入仍由owner裁定** |
| 09.6.1 五项工程补齐（范围完成，按预注册出口`failed_lesion`结案） | [合同](reference/SPEC-TAIJI-CORE_ENGINEERING_20261101.md)＋[验证器](../scripts/training/verify_taiji_r029_engineering_gaps.py)＋[测试](../tests/taiji_native/test_r029_engineering_gaps.py)：五项缺口全部实现，`living_core.py`字节未改；[报告](../reports/taiji_core_r029_engineering_gaps_20261101.json) | 26项检查1项红＝lesion贡献判据。约束更新8步全接受、fit降0.0307817、漂移0.0162904、KKT1.94e-16、负对照漂移0.1111525；lesion三项结构门全过但新生单元只占训练收益2.1e-5（4.33e-7对0.0201929）⇒第三次独立否证结构生长收益；活动学习优势+0.0198944经查明为零成本空转（`advantage_is_idle_cost_avoidance`）；中文夹具5类分母绑实际应答轮数、捷径尺有范围且无一满分；09.6把恢复CPU记为0.0、实测子进程自报见报告。§4重新定价：约束更新8.830 CPU秒/接受步（无约束的3.53倍），09.7三档1104/1250/**1984（破G7的1800达184秒）**；**09.7是否执行与按哪档预算执行由owner裁定，训练未准入** |
| 09.6.2 夹具改造（范围完成，两项读数：`growth_fixture_no_reachable_cell`＋`failed_activity_fixture`） | [合同](reference/SPEC-TAIJI-CORE_FIXTURES_20261101.md)＋[到达夹具](../taiji/native_activity_fixture.py)＋`run_round`的`arrivals`扩展＋[测试](../tests/taiji_native/test_fixture_arrivals.py)＋[验证器](../scripts/training/verify_taiji_fixture_arrivals.py)；[v1报告](../reports/taiji_core_fixture_arrivals_20261101.json)（首跑，含一条我写错的门，保留）+[v2报告](../reports/taiji_core_fixture_arrivals_v2_20261101.json)（正式读数） | **生长**：六格全发布，新生份额随步数转负（+2.15e-5→−2.76e-6→−7.13e-5），同期holdout收益0.020→0.324全归父行；唯一正份额格（1e-4/128）漂移1.170、收益−0.464＝发散 ⇒ 第四次独立否证，且**排除"训练太慢"**，不改网格重测。**活动**：夹具已有0→0.704动态范围（脚本策略0.704、空闲0.0、单次交互净赚+0.044），但训练臂在3/3种子上"交互一次后停手"（份额0.0627/−0.0253/0.0185）；逐轮供给变体3/3种子达**1.40807、32次交互** ⇒ 失败源于`TRACE_LAMBDA=0`的两步信用，非夹具。v2重测生长网格与v1逐位相同。**唯一下一步＝owner裁定09.7活动面声明方式（等待链+λ>0／逐轮供给且不称顺序能力），训练未准入** |
| 09.6.3 出生惰性诊断（范围完成） | [脚本](../scripts/training/diagnose_birth_inertness.py)＋[报告](../reports/taiji_core_birth_inertness_20261101.json)：原理形状、coverage_birth子、lr=1e-5、8步；测出生/训练后的R、P范数，出生时holdout/fit梯度在父行与新生行上的分配，以及lesion贡献 | 新生R行出生恰为0、8步后0.006730653（不是死参数）；拿到父行梯度范数的6.13%（holdout）/5.32%（fit）；新生P列初始为父列的12.4%（不是小到可忽略）；贡献仍4.33e-7、父行承担0.020192874。⇒ 排除"没梯度/太小"，剩余解释只有**信用方向对holdout无用**。本合同只出诊断，不提也不批新机制 |
| 09.6.3b 方向可达性上界（范围完成） | [脚本](../scripts/training/diagnose_direction_oracle.py)＋[报告](../reports/taiji_core_direction_oracle_20261101.json)：把出生方向换成holdout梯度的神谕奇异向量（仅可达性上界，非可用方法），其余同09.6.3 | 收益0.02019838、新生贡献**5.888e-6**、份额**2.915e-4**：比fit信用方向高13倍且**过份额门**，但按09.6.1的绝对门（≥1e-4 NLL）仍差17倍 ⇒ **"任何方向都不行"被否证，但"方向是唯一瓶颈"也未证成**。新假设只能针对方向规则，且必须预先声明瞄准份额门还是绝对门 |
| 09.6.4 方向规则候选（范围完成，判据自检失败`metric_not_discriminating`） | [合同](reference/SPEC-TAIJI-CORE_DIRECTION_20261101.md)＋[规则](../taiji/birth_direction.py)（fit生涯内多个不相交子窗口的共识方向，AST级不泄漏守卫）＋[验证器](../scripts/training/verify_taiji_birth_direction.py)＋[报告](../reports/taiji_core_birth_direction_20261101.json)：四臂同轮（共识K=2/K=4、fit-SVD基线、随机负对照） | 绝对贡献：共识K=2 **1.426e-4**／K=4 7.32e-5／基线4.33e-7／**随机1.99e-5**；份额：7.014e-3／3.612e-3／2.146e-5／**9.836e-4**。随机方向也过份额门 ⇒ **预注册份额判据无判别力**，按出口判`metric_not_discriminating`（不是支持H1）。基线同轮复现通过（偏差0%）。合同外观察只登记：绝对贡献上共识K=2过1e-4且为随机7.2倍，但单种子单轮、且合同排除绝对门，不作声称 |
| 09.6.5 绝对贡献判据（结论作废） | [合同](reference/SPEC-TAIJI-CORE_DIRECTION_ABS_20261101.md#4-执行更正20261101同日)＋[报告](../reports/taiji_core_birth_direction_abs_20261101.json)：改用绝对贡献≥1e-4且随机对照<1e-4 | **种子轴是假的**：`LivingCoreReference` 用内部固定种子，全局`manual_seed`不改父模型，三种子摘要逐位相同（`921dc9981187406a`×3）、共识臂读数三次相同（1.426e-4×3）⇒`hypothesis_h1_supported`作废，旧报告保留不删。仍有效的只有"同一确定性主体上三个随机对照1.988e-5／−1.720e-6／7.284e-6均在门下" |
| 09.6.6 真实变化轴复现（范围完成，`hypothesis_h1_supported`） | [合同](reference/SPEC-TAIJI-CORE_DIRECTION_REPLICATION_20261101.md)＋[验证器](../scripts/training/verify_taiji_birth_direction_parts.py)＋[报告](../reports/taiji_core_birth_direction_parts_20261101.json)：变化轴改为规则窗口数`parts∈{2,3,4}`，每格共识＋2随机对照＋基线 | parts=2 **1.426e-4**、parts=3 **1.343e-4**、parts=4 7.320e-5（未过）；随机对照最大1.988e-5（差5倍）；基线三格逐位复现4.332e-7；漂移全≤0.030；泄漏零命中。⇒ 2/3格过绝对门。**薄边际（仅1.4倍）**，不可外推：未测成本偿付、单一确定性主体、单一参考图、非任务能力 |
| 09.6.7 稳定性面（范围完成，`platform_supported`） | [合同](reference/SPEC-TAIJI-CORE_DIRECTION_SURFACE_20261101.md)＋[验证器](../scripts/training/verify_taiji_birth_direction_surface.py)＋[报告](../reports/taiji_core_birth_direction_surface_20261101.json)：`parts∈{2..6}`×`units∈{1,2,3}`共15格全发布，预注册门槛"≥6格且≥3个parts" | **8格过门、覆盖parts{2,3,5,6}**；保持漂移全0.0221–0.0223（门0.030）；随机对照全在门下、基线逐位复现、泄漏零命中。效应随单元数单调增强（parts=5：7.99e-5→1.21e-4→1.74e-4）；**parts=4三档全不成立**（3.24e-5／7.32e-5／9.50e-5），恰是09.6.3/09.6.4的默认值。薄平台（最大1.737e-4＝门的1.74倍）；成本偿付未测、单一确定性主体、非任务能力 |
| 09.6.8 净收益对照（范围完成，`net_benefit_supported`） | [合同](reference/SPEC-TAIJI-CORE_NETBENEFIT_20261101.md)＋[验证器](../scripts/training/verify_taiji_birth_netbenefit.py)＋[报告](../reports/taiji_core_birth_netbenefit_20261101.json)：同一确定性父、同8步lr=1e-5，只重放 vs 共识出生（parts=5,units=3） | 只重放 Δholdout −0.020192402／训练21.688秒；共识出生 −0.020366326／22.656秒；**净收益 +1.739e-4**；出生构造10.531＋额外训练0.968＝**11.50 CPU秒**（≈6.6秒/每1e-4 NLL，定价输入、非门槛）。lesion贡献1.7374e-4与09.6.7同格逐位吻合；两臂漂移≤0.030。**09线首份正净收益**；仍受限于单一确定性主体、合成夹具、8步lr=1e-5、parts=4不成立、无任务能力 |
| 09.6.9 跨评估生涯复核（范围完成，`replicated_on_second_holdout`） | [合同](reference/SPEC-TAIJI-CORE_NETBENEFIT2_20261101.md)＋[验证器](../scripts/training/verify_taiji_netbenefit_second.py)＋[报告](../reports/taiji_core_netbenefit_second_20261101.json)：同两臂在HOLDOUT与一条新声明的第二评估生涯上各测pre/post；RETENTION仍只作保持探针，不当评估面 | HOLDOUT净收益**+1.739e-4**（与09.6.8逐位相同，一致性门通过）、SECOND_HOLDOUT **+1.903e-4** ⇒ 净收益不是单条评估生涯的产物。两条都是合成夹具、单一确定性主体、8步lr=1e-5；parts=4仍不成立；单价约6.6 CPU秒/每1e-4 NLL（出生构造10.5秒为主）；无任务能力 |
| 09.7 有界联合片段（范围完成，`retention_breach`） | [合同](reference/SPEC-TAIJI-CORE_JOINT_FRAGMENT_20261101.md)＋[验证器](../scripts/training/verify_taiji_fragment_09_7.py)＋[报告](../reports/taiji_core_fragment_09_7_20261101.json)：单点候选parts=5/units=3跑32步分四检查点，两臂同父同步 | 净收益单调增长：8→32步 HOLDOUT 1.739e-4→**7.187e-4**、第二生涯1.903e-4→**8.147e-4**；但**两臂都破保持门**（0.084752／0.085631，门0.030）⇒按出口结案`retention_breach`。**破坏保持的是32步总训练量**（两臂仅差0.0009），不是出生。8步一致性门通过 |
| 09.7b 可持续点定位（范围完成，`sustainable_horizon_found`＝8步） | [合同](reference/SPEC-TAIJI-CORE_HORIZON_20261101.md)＋[验证器](../scripts/training/verify_taiji_fragment_horizon.py)＋[报告](../reports/taiji_core_fragment_horizon_20261101.json)：同两臂逐检查点记保持漂移 | 8步 0.022070／0.022269（门内，净收益1.739e-4／1.903e-4）；16步 0.043447／0.043861、24步 0.064／0.065、32步 0.0848／0.0856 全部越线。⇒ **仅8步在门内**，净收益随步数近线性涨（8→32约4.1–4.3倍）但要吃到必须动保持门或lr，二者均不在已批准范围 |
| 09.8 全局收束（范围完成，`closeout_passed_engineering_only`） | [合同](reference/SPEC-TAIJI-CORE_CLOSEOUT_09_20261101.md)＋[验证器](../scripts/training/verify_taiji_closeout_09.py)＋[能力/资源卡](../reports/taiji_core_09_closeout_20261101.json)：六件收束物（bundle／活动轨迹／父子映射／参数继承／原始输出／卡片）＋三项一致性 | **bundle 77,609,636 字节**（`output/taiji_core_09/subject_bundle.pt`，`.pt` 已被 gitignore:62 忽略、不入 git）；父 3,311,259 参数→子 3,323,547（+12 新生单元，4 群×3）、growth_parent_units=256；**参数继承 135/135 逐位相等**；spawn 恢复逐位相同；活动轨迹 8 轮全 `interact`（逐轮供给下的正确行为，不判偏好）；原始输出两条 query 的 top-1 均为字节 229（原样记录、不评分）。435.2/900 CPU 秒、0 失败检查 |

**研究范围收束。** v1.2不主张新快权重算子、新架构门类或首次擦写解耦；新颖性仍待与已有秩/功能梯度/重放工作比较。可撤销快记忆、按来源定位影响及治理重建扩展留至09.5之后的版本决策，本包不新增实现；基本权限、来源标记、合法mask、单调journal及原恢复边界不删除。第二阶段预测性自我修改不抢占当前参考图/生长信用任务。

**R021工程冻结。** 诊断与覆盖门使用同一softmax凸分区（非负、每样本和为1）；K≤4、R_max=8、枚举≤6561状态/1CPU秒/2墙钟秒。数值秩/128次置换噪声标定、绝对信用下限与检查窗条件、未标定/全mask/预算超限停止均见工程合同§4。尚未实现或完成这些后继标定，不因前向回执准入出生。工程界不构成任务效应门或永久模型上限。

**自然内容范围。** 首包可沿原12实体任务域，加未列名指称/别名作为单列泛化臂，清单只用于生成和评分。当前/前代含同值不同版本、相关更正、无关干扰、来源冲突/未知和陌生指称/改写均保留。时间意图歧义与缺资料未知分账；runtime不得读取gold槽、时间类别或选中ID。当前适配器借这些标签的有限实验范围不变，但不当作已实现自然生产路径。

**资产与新增。** 复用native_content_core的来源/版本/内容读取、受控复制/生成和保存资产，旧adapter/homeostasis/sleep/adaptive_residual提供经核对的动态、重放、结构与血缘部件；它们尚未因此集成进新核心。拟以`taiji/native_lifecycle.py`承接主体状态/活动循环、`taiji/native_development.py`承接合法可塑更新与结构变更，原计划的native_content_binding承接自然内容输入；均为拟建职责，不是现有可执行入口。实际模块可在合同内合并，禁止仅为目录对齐堆同名器官。

### 10 自然目标、后果预测与有限行动

**启用规则。** 仅当选定任务需要行动/规划时进入；首次11若以记忆作代表能力，可记录本步“不适用”直接审11，不新增全T3前置。首包推荐一个确定状态、动作和回执的封闭模拟环境，不开真实外部副作用；至少包含一类真正需要两步的任务，否则只声称一步选择。

| 子步骤 | 实施与交付 | 验收/停止 |
|---|---|---|
| 10.1 范围及模拟环境（范围完成，`passed_10_1_environment_only`） | [合同](reference/SPEC-TAIJI-CORE_T3_GOAL_20261101.md)＋[环境](../taiji/native_goal_world.py)（2对象×3位置、`move(object, place)`、每步≥2合法动作、动作ID按声明种子逐步打乱）＋[测试](../tests/taiji_native/test_goal_world.py)＋[验证器](../scripts/training/verify_taiji_goal_world_10_1.py)＋[报告](../reports/taiji_core_goal_world_10_1_20261101.json) | 四条夹具（f1/f2 共享初态、f1+f3 与 f2+f4 各共享 Goal）；穷举最优 f1/f2/f4=0.99、f3=1.0（已满足、0 步）；**换 Goal 后最优动作不同**（f1 是杯子→窗台、f2 是印章→木桌）；四条捷径 `first_legal`/`lowest_id`/`fixed_object`/`cycle` **都至少在一格拿不到最优、且没有一条在全部格最优**；动作ID逐步重排、终态评分只读终态与步数。**未涉及任何模型；不签任务能力，10.2/10.3 未准入** |
| 10.1b 任务加深（范围完成，`passed_10_1b_environment_deepened`） | [合同](reference/SPEC-TAIJI-CORE_T3_GOAL_10_1B_20261101.md)＋`native_goal_world` 的 `occupy` 参数与 g1/g2 夹具＋[验证器](../scripts/training/verify_taiji_goal_world_10_1b.py)＋[报告](../reports/taiji_core_goal_world_10_1b_20261101.json) | 位置容量1（目标位被占需先腾开），默认 `occupy=False` **旧路径逐位不变**（f1..f4 仍 0.99/0.99/1.00/0.99、捷径仍不通吃）；加深后三条 10.1 判据仍成立；规划器 [0.99,0.98,1.00,0.99,0.98,1.00] 对贪心 [0.99,−0.04,1.00,0.99,−0.04,1.00]，**g1 上贪心确被阻挡**；不涉及模型 |
| 10.2 自然Goal绑定（契约与基线完成，`passed_10_2_binding_contract_only`） | [合同](reference/SPEC-TAIJI-CORE_GOAL_BINDING_20261101.md)＋[规则基线](../taiji/goal_binding.py)＋[测试](../tests/taiji_native/test_goal_binding.py)＋[验证器](../scripts/training/verify_taiji_goal_binding_10_2.py)＋[报告](../reports/taiji_core_goal_binding_10_2_20261101.json)：五类公开反例（改写／否定／合取／缺条件／歧义）全部入分母 | 6 夹具全计分母：改写的两种说法绑定出同一条件集；否定极性 −1 且 `satisfied` 判读正确；合取含两条条件；**缺条件与歧义都拒答并给出原因、条件为空（不得猜）**；绑定器 AST 无 `Goal`/`gold`/`terminal_state`/`reward`/`score` 标识符。**本步交付的是规则基线，模型侧绑定未交付**；不签任务能力 |
| 10.3 预测及选择（契约完成；首测 `planner_no_advantage`，加深后重测通过） | [合同](reference/SPEC-TAIJI-CORE_GOAL_PLANNER_20261101.md)＋[参考规划器](../taiji/goal_planner.py)＋[测试](../tests/taiji_native/test_goal_planner.py)＋[验证器](../scripts/training/verify_taiji_goal_planner_10_3.py)＋[报告](../reports/taiji_core_goal_planner_10_3_20261101.json)：五条臂（规划/贪心/随机/扰动模型/错目标） | 规划器与穷举最优逐位相同（0.99/0.99/1.00/0.99），但**与贪心完全打平、随机在3/4也最优** ⇒ `planner_no_advantage`。扰动模型臂每步预测均与实际不符且终态降到 −0.04 ⇒ 预测栏**有判别力**；错目标臂至少一格低于规划器 ⇒ 绑定条件被用上。首测结论是任务太浅（3位置、单对象目标，贪心已最优）⇒ 加做 10.1b；加深后重测 `reports/taiji_core_goal_planner_10_3b_20261101.json` 为 `passed_10_3_prediction_contract_only`：规划器 [0.99,0.98,1.00,0.99,0.98,1.00] 对贪心 [0.99,−0.04,1.00,0.99,−0.04,1.00]，2 格严格更优、其余持平；不签模型能力 |
| 10.4 模型侧选择（范围完成，`model_below_random`） | [合同](reference/SPEC-TAIJI-CORE_SUBJECT_SELECTION_20261101.md)＋[选择器](../taiji/subject_action_scorer.py)＋[测试](../tests/taiji_native/test_subject_action_scorer.py)＋[验证器](../scripts/training/verify_taiji_subject_selection_10_4.py)＋[报告](../reports/taiji_core_subject_selection_10_4_20261101.json)：09.8 同一候选（父摘要逐位吻合）对每个合法动作的中文句子算条件 NLL、取最低 | 主体 **2.81**／贪心 **3.86**／随机 **2.81** ⇒ `model_below_random`；仪器四项全过。机制：**19 次选择 19 次都选杯子句、0 次印章句**，NLL 差稳定 0.0646–0.0650 ⇒ 固定字面偏好而非目标条件化决策。`capability_passed=false`，**P2 仍未满足**；不声称任务能力 |
| 10.5 目标条件化接口（范围完成，`contrast_below_floor`） | [合同](reference/SPEC-TAIJI-CORE_GOAL_CONDITIONED_20261101.md)＋[对比度选择器](../taiji/goal_conditioned_scorer.py)＋[验证器](../scripts/training/verify_taiji_goal_conditioned_10_5.py)＋[报告](../reports/taiji_core_goal_conditioned_10_5_20261101.json)：同一句在匹配/错配目标下的 NLL 差，字面先验相减抵消 | 杯子→窗台 **+5.725e-4**、印章→窗台 **+6.527e-4**，**差仅 8.018e-5＜1e-3 底座** ⇒ `contrast_below_floor`；零对照恰为 0、同候选与确定性通过。机制：目标**确实推动分数**（两句都 +6e-4 量级）但**推得一样多**＝共模位移，不能区分动作。判据两稿均被我否决（2 个候选下"翻转"与"方向"都恒真），终稿只判幅度+零对照 |
| 10.6 动作特定构造（范围完成，`action_gap_below_floor`） | [合同](reference/SPEC-TAIJI-CORE_ACTION_VERDICT_20261101.md)＋[verdict 打分器](../taiji/action_verdict_scorer.py)＋[验证器](../scripts/training/verify_taiji_action_verdict_10_6.py)＋[报告](../reports/taiji_core_action_verdict_10_6_20261101.json)：`prefix = 局面+目标+动作句`，比「是/否」似然 | 杯子 **−0.143546567**、印章 **−0.143066181**，**动作差 4.804e-4＜1e-3**、方向相反 ⇒ `action_gap_below_floor`；答案可分（差约0.143）、零对照与确定性通过。三条免训练接口横向：10.4 恒定字面偏好／10.5 共模 8.0e-5／10.6 动作进前缀 4.8e-4（放大6倍仍不过门）⇒ **P2 未满足** |
| 10.7 第一次有界训练（范围完成，`training_below_chance_plus`） | [合同](reference/SPEC-TAIJI-CORE_TRAINING_10_7_20261101.md)＋[训练模块](../taiji/goal_conditioned_training.py)＋[测试](../tests/taiji_native/test_goal_conditioned_training.py)＋[验证器](../scripts/training/verify_taiji_goal_training_10_7.py)＋[报告](../reports/taiji_core_goal_training_10_7_20261101.json)：12 个合成样本、按局面 6/6 划分、是非题监督、两个 lr 都发布 | 未训练 3/6；lr=1e-5 损失 11.2069→11.0475、留出 3/6、漂移 0.0026；lr=1e-4 损失→**7.0997**、留出仍 **3/6**、漂移 **0.0322 破门**。机制：命中模式 [F,T,F,T,F,T] → 翻转 [T,F,T,F,T,F]，即**训练把常量偏好从印章换成杯子**，不是学到条件化。权重仅落 `output/taiji_core_10_7/`、`weights_published=false` |
| 10.7a 权重可重放（范围完成，`weights_replay_bitwise`） | [验证器](../scripts/training/verify_taiji_training_replay_10_7a.py)＋[测试](../tests/taiji_native/test_training_replay_10_7a.py)＋[报告 v2](../reports/taiji_core_training_replay_10_7a_v2_20261101.json)（[首版](../reports/taiji_core_training_replay_10_7a_20261101.json)保留，换绑原因同 10.11） | 由声明种子重建同一主体、`strict=True` 载入两臂权重、重算 12 个留出判定：**两臂逐位相同**（最大绝对差 0.0）、命中 3/6 与卡一致、权重 26,633,881／26,634,022 字节与卡相等、父摘要一致、40.6 CPU 秒 ⇒ 已保存的 10.7 权重可回落并复现其发表读数；**不表示**该臂学会了条件化选择，不训练、不发权重 |
| 10.7b 梯度对齐诊断（范围完成，`training_groups_pull_together`） | [脚本](../scripts/training/diagnose_training_direction_10_7b.py)＋[测试](../tests/taiji_native/test_training_direction_10_7b.py)＋[报告](../reports/taiji_core_training_direction_10_7b_20261101.json)：读 10.7 六个训练样本的 per-sample 梯度（**不更新参数**）＋已保存的 lr=1e-4 更新 | 按正确对象分两组：**两组均值梯度余弦 0.999977**；任意两样本梯度余弦中位数 0.999977、**残差份额 1.75e-5**；‖均值梯度‖／平均‖梯度‖＝0.99999 ⇒ 一阶上六个样本几乎是同一个目标。Δθ 与均值梯度余弦 −0.048（Adam 是符号型更新，高维近似正交属预期，不作失败信号）、‖Δθ‖＝4.439。⇒ **条件化信号比共模小 5 个数量级**；"再多训几步"造不出它，③ 必须显式减掉共模。不训练 |
| 10.7c 残差结构诊断（范围完成，`residual_group_structure`） | [脚本](../scripts/training/diagnose_residual_structure_10_7c.py)＋[测试](../tests/taiji_native/test_residual_structure_10_7c.py)＋[报告 v2](../reports/taiji_core_residual_structure_10_7c_v2_20261101.json)（[首版](../reports/taiji_core_residual_structure_10_7c_20261101.json)保留：并列划分漏计，p 由 0.05 更正为 0.1） | 把 10.7b 的 1.75e-5 残差按"正确对象"分组：**组解释份额 F＝0.6459**（随机划分期望 0.2），组内残差余弦 0.5226／0.5087（门 0.5，**刚过线**）；10 个去重 3/3 划分里真实划分最高、次优仅 0.2561 ⇒ 穷举 p＝0.1（该设计下限）。⇒ 残差**不是噪声**：条件化信号在参数空间里存在，只是被共模压了约 4 个数量级。因此 ③ 的第一候选是**减共模的更新/目标**，加新通路留作它失败后的第二步——**这条推断已被 10.7d 否证**（只换目标，平均更新只剩 0.56%）。不训练 |
| 10.7e 目标→动作读出（范围完成，`readout_binding_supported`） | [合同](reference/SPEC-TAIJI-CORE_READOUT_10_7E_20261101.md)＋[读出头](../taiji/goal_action_readout.py)＋[验证器](../scripts/training/verify_taiji_readout_10_7e.py)＋[测试](../tests/taiji_native/test_goal_action_readout.py)＋[报告](../reports/taiji_core_readout_10_7e_20261101.json) | 主体全冻结（参数与留出损失逐位相同）；只训 1,024 参数秩 4 双线性头（U 小随机、V 零 ⇒ 起点全 0 分、6/6 平局）；6 个训练样本 300 步后**留出 6/6**、末损失 0.0；标签对调对照 **0/6**；35.8 CPU 秒。边界：留出 4 个正确动作里 2 个训练没见过（跨局面＋部分跨动作）；头只吃编码、不读循环状态；**状态刻意不叫 `passed_*`**——这是夹具级绑定，不是 11 要求的代表能力，P2 仍不满足。**全动作排名探针（10.7e-1）把强度收窄**：留出 top-1 **3/6**、训练 top-1 **1/6** ⇒ 6/6 只证明"对特定干扰项的判别"，不证明全局选择 |
| 10.7e-1 全动作排名（范围完成，`readout_rank_below`） | [脚本](../scripts/training/probe_readout_rank_10_7e1.py)＋[测试](../tests/taiji_native/test_readout_rank_10_7e1.py)＋[报告](../reports/taiji_core_readout_rank_10_7e1_20261101.json)：**不训练**，只把 10.7e 保存的头拿来对"2 对象×3 位置＝6 个动作"整体打分 | 留出 **top-1 3/6**（平均名次 2.0，机会 1/6）、**训练 top-1 仅 1/6**、36.2 CPU 秒 ⇒ 这个头**连自己训练目标的全局第一名都排不出来**：10.7e 的 6/6 是"正确动作 > 它被训练比较的那一个干扰项"，不是"从动作集里挑对"。⇒ 若要声称"会选择"，训练目标本身就得是 6 选 1；判据 `readout_rank_below` 已按预注册落在门内。原样主体的同题基线是 **2/6**（正好等于最佳常量；该跑超预算判 `budget_stop`，只作非正式参考）⇒ 头**没有显示出高于先验的选择能力** |
| 10.7f 六选一选择（范围完成，`readout_selection_absent`） | [合同](reference/SPEC-TAIJI-CORE_SELECTION_10_7F_20261101.md)＋[读出头](../taiji/goal_state_readout.py)＋[验证器](../scripts/training/verify_taiji_selection_10_7f.py)＋[测试](../tests/taiji_native/test_goal_state_readout.py)＋[报告](../reports/taiji_core_selection_10_7f_20261101.json) | 主体冻结；1,536 参数头读"目标＋局面＋动作"，6 路交叉熵、300 步：**训练损失降到 5e-06（把 6 个样本完全记住）**，但**留出 top-1 只有 2/6**、与未训练的常量选择（2/6）**持平**；目标错配对照 0/6；39.3 CPU 秒 ⇒ **能背不能迁**：训练集拟合到零误差，留出没有任何超出常量的收益。夹具级、不构成 P2 |
| 10.7g 特征可加性（范围完成，`features_additive`） | [脚本](../scripts/training/probe_feature_additivity_10_7g.py)＋[测试](../tests/taiji_native/test_feature_additivity_10_7g.py)＋[报告](../reports/taiji_core_feature_additivity_10_7g_20261101.json)：对 6 个动作句做 2×3 两向分解，**不训练** | 对象＋位置主效应占 **99.987%** 的能量（随机设计基线 0.6），两个对象的主效应范数都是 0.7668 ⇒ 编码器里的动作向量**几乎完全可加**，10.7f 的失败不是特征纠缠；34.6 CPU 秒 |
| 10.7h 几何读出＋夹具直通（范围完成，`geometry_does_not_select`） | [脚本](../scripts/training/probe_geometry_select_10_7h.py)＋[报告](../reports/taiji_core_geometry_select_10_7h_20261101.json)：零参数读出（方向取自编码器主效应、目标用 `goal_vector`），**不拟合、不训练** | 留出 **0/6**、训练 4/6、换目标对照 0/6，33.5 CPU 秒 ⇒ `goal_vector` 与动作空间对不上。**同时查出**：12/12 样本的 `goal` 与正确动作句**逐字节相同** ⇒ 这套夹具的捷径是**复读题面**，而候选没有复读通道。继续加头／加样本之前，必须先改目标措辞 |
| 10.8 目标改写与复读捷径（范围完成，`paraphrase_still_copyable`） | [改写模块](../taiji/goal_paraphrase.py)＋[验证器](../scripts/training/verify_taiji_goal_paraphrase_10_8.py)＋[测试](../tests/taiji_native/test_goal_paraphrase_10_8.py)＋[报告](../reports/taiji_core_goal_paraphrase_10_8_20261101.json)：三种不用"把…放到"的措辞 × 6 局面 × 2 对象＝36 个目标 | 与任一动作句**逐字节相同 0/36**、**实体唯一确定 36/36**，但**字符二元组重叠匹配器 36/36 全对** ⇒ **光改措辞去不掉复读捷径**：只要目标点名对象与位置，重叠匹配就赢。要真正排除必须让目标**不点名实体**（按属性／关系描述）＋给主体词汇表，那是更大的夹具改动。无模型、0.0 CPU 秒 |
| 10.9 属性描述目标＋词汇表（范围完成，`attribute_fixture_fair`） | [夹具](../taiji/native_goal_attributes.py)＋[验证器](../scripts/training/verify_taiji_goal_attributes_10_9.py)＋[测试](../tests/taiji_native/test_goal_attributes_10_9.py)＋[报告](../reports/taiji_core_goal_attributes_10_9_20261101.json)：目标改为不点名实体（`让圆口的器皿到靠窗的台面。`），属性→实体映射做成词汇表 | 12 个目标：**实体名出现 0/12**、**词汇表唯一解析 12/12**、**字符重叠匹配器 2/12（机会 1/6）** ⇒ 夹具公平：答案仍由目标唯一决定，但只能经词汇表拿到。本步不接模型（词汇表怎么交给主体是下一步的事） |
| 10.10 词汇表可用性（范围完成，`glossary_not_used`） | [脚本](../scripts/training/probe_glossary_use_10_10.py)＋[报告](../reports/taiji_core_glossary_10_10_20261101.json)：把词汇表拼在目标前，6 个目标 × 2 动作的成对读数，**不训练**；对照臂＝词表映射反向 | 真词表 **0/6**、反向词表 **0/6**（两臂都恒取同一个错动作）⇒ 词汇表对未训练主体**毫无影响**；494.4 CPU 秒（预算 900，未超）。这是新夹具上的**地板读数**：把表塞进 prompt 这条便宜路不通 |
| 10.11 词汇表走 records（范围完成，`records_channel_live`） | [模块](../taiji/native_goal_glossary.py)＋[验证器](../scripts/training/verify_taiji_glossary_records_10_11.py)＋[测试](../tests/taiji_native/test_glossary_records_10_11.py)＋[报告](../reports/taiji_core_glossary_records_10_11_20261101.json) | 词表进 `records`、不进 prompt：prompt 内单跳重叠 **0/6**、records 改动 NLL **12/12**（通道活）、成对命中 **6/6**。**6/6 随即被 10.11b 打掉**：通道活、映射未被用上 |
| 10.11b 映射置换对照（范围完成，`control_also_wins`） | [脚本](../scripts/training/probe_glossary_control_10_11b.py)＋[报告](../reports/taiji_core_glossary_control_10_11b_20261101.json) | 只把词汇表映射换成**真置换**：真映射 **6/6**、错映射 **6/6** ⇒ 两臂 token 集合相同、只有配对变而命中不变 ⇒ 选择由**与映射无关的不变量**驱动。10.11 的 6/6 **不可归因于读表**，若只跑 10.11 就会误报 |
| 10.12 词汇表是否到达循环状态（范围完成，`records_reach_state`） | [脚本](../scripts/training/verify_taiji_records_state_10_12.py)＋[测试](../tests/taiji_native/test_records_state_10_12.py)＋[报告 v2](../reports/taiji_core_records_state_10_12_v2_20261101.json)（[v1](../reports/taiji_core_records_state_10_12_20261101.json)保留：import 排序后重发） | 同一目标、真映射 vs 置换映射，取"读完全部 prompt 补丁之后"的 `state.h`（gated 路径）：**6/6 不同**、与无 records 也 **6/6 不同**、中位相对距离 **0.002951**；33.0 CPU 秒 ⇒ 词表确实进到循环状态，读数头若吃 `state.h` **原理上**能看到映射。必要条件成立，不等于会被用上 |
| 10.13 状态特征六选一（范围完成，`state_selection_absent`） | [读出头](../taiji/state_action_readout.py)＋[验证器](../scripts/training/verify_taiji_state_selection_10_13.py)＋[测试](../tests/taiji_native/test_state_action_readout.py)＋[报告 v2](../reports/taiji_core_state_selection_10_13_v2_20261101.json)（[v1](../reports/taiji_core_state_selection_10_13_20261101.json)保留：参数计数手算错） | 公平夹具、词表只走 records；特征＝读完目标后的 `state.h`（512 维）＋动作编码；**2,560 参数**（v1 误记为 4,608，v2 改从模块取数并加守卫）、300 步、主体逐位冻结、38.4 秒。三臂：A 真→真 **2/6**、B 真→置换 **2/6**、C 置换→真 **2/6** ⇒ 与未训练常量持平；10.12 已证状态里带着映射，**6 个样本却教不会头把它变成选择**。读数线按负结果收束 |
| 10.14 读出线收束卡（范围完成，`readout_line_closed_negative`） | [验证器](../scripts/training/verify_taiji_readout_closeout_10_14.py)＋[报告](../reports/taiji_core_readout_closeout_10_14_20261101.json)：只读 20 份卡，**不重跑、不训练** | 20/20 在册；**唯一的能力声称是 10.7e，且已被 10.7e-1（`readout_rank_below`）撤回**；权重未发布、默认未替换；**全线 CPU 2,996.6 秒** ⇒ 读出线按负结果结案。不表示主体永远不能选择，不改 11 |
| 10.15 宽夹具（范围完成，`wide_fixture_ready`） | [夹具](../taiji/native_goal_world_wide.py)＋[验证器](../scripts/training/verify_taiji_wide_fixture_10_15.py)＋[测试](../tests/taiji_native/test_wide_fixture_10_15.py)＋[报告 v3](../reports/taiji_core_wide_fixture_10_15_v3_20261101.json)：4 对象 × 5 位置，目标不点名实体、词表桥接 | **120 局面 / 480 样本 / 9,600 个 logit 约束**；实体泄漏 **0/480**、词表解析 **480/480**、与正确动作共享二元组 **0/480**、复读匹配器 **24/480 = 5%**（机会线）。v1/v2 的两处泄漏（"地上的织物"→"到地"、"左边的匣子"→"到左"）由本步自己抓出并改掉 |
| 10.16 宽夹具二十选一（**作废**：`control_not_null`） | [脚本](../scripts/training/verify_taiji_wide_selection_10_16.py)＋[报告](../reports/taiji_core_wide_selection_10_16_20261101.json) | 480 样本按**局面** 80/40 切：三臂留出 **100%** ⇒ 不是能力——480 个样本只有 **20 个不同目标**，按局面切分**一个目标都没留出**，头把 20 个目标背下来了 |
| 10.17 按对象留出（范围完成，`wide_selection_absent`） | [脚本](../scripts/training/verify_taiji_wide_object_split_10_17.py)＋[报告](../reports/taiji_core_wide_object_split_10_17_20261101.json) | 整个 `黄色球` 不进训练（15 个目标／5 个动作）：训练 360、留出 120，三臂 **0%**（最佳常量 20%），38.5 秒 ⇒ **见过的目标背到 100%、没见过的对象 0%**：卡点是**组合泛化**，不是样本量 |
| 10.18 从 records 复制实体名（范围完成，`records_copy_inconclusive`） | [脚本](../scripts/training/verify_taiji_records_copy_10_18.py)＋[报告](../reports/taiji_core_records_copy_10_18_20261101.json)：prompt＝`<描述词>是`、target＝`<实体>`，只量 NLL、不训练；对照＝把 target 换成另一个实体 | 带 records 比不带便宜 **5/5、平均 +3.114 nat**；对照成立 **4/5**；34.0 CPU 秒 ⇒ **主体会用 records 说出实体名**（模型自己的机制，不是字符串表）。按预注册对照未满分 ⇒ `inconclusive`，不升级成 supported |
| 10.19 生成式选择（范围完成，`generation_absent`） | [脚本](../scripts/training/verify_taiji_generation_10_19.py)＋[报告](../reports/taiji_core_generation_10_19_20261101.json)：不训练，prompt＝`目标是{描述}到{描述}。我应该`、records＝词汇表、贪心解码 16 字节 | 命中 **0/6**（不带 records 也 0/6），34.9 CPU 秒。**生成内容说明机制**：带 records 时 6 个目标**全部**吐出同一句 `台是木桌。`（复读一行词表后 EOS）；不带 records 时是 16 字节二进制噪声 ⇒ 主体**有单跳 copy、无组合** |
| 10.20 能力线收束卡（范围完成，`capability_line_closed_negative`） | [验证器](../scripts/training/verify_taiji_capability_closeout_10_20.py)＋[报告](../reports/taiji_core_capability_closeout_10_20_20261101.json)：只读 18 份卡，不重跑、不训练 | 18/18 在册；**唯一能力声称 10.7e 已被 10.7e-1 撤回**；三条负锚点（10.13／10.17／10.19）全为负；copy 证据（10.18）在册；所有带 `backbone_frozen` 的臂全 true；这些卡合计 **2,945.8 CPU 秒**。机制句写进卡：**单跳 copy 有、组合没有** |
| 10.7d 成对目标探针（范围完成，`margin_isolates_differential`） | [脚本](../scripts/training/diagnose_margin_objective_10_7d.py)＋[测试](../tests/taiji_native/test_margin_objective_10_7d.py)＋[报告](../reports/taiji_core_margin_objective_10_7d_20261101.json)：同一候选上比两种目标的梯度——10.7 的平摊损失 vs 成对 margin `softplus(-(verdict_other−verdict_correct))`，都不更新参数 | 平摊支复算 F＝0.645858（与 10.7c 逐位相同，交叉校验通过）；margin 支六个梯度**成对相反**（两两余弦中位数 −0.9995、组内 0.9997、F＝0.9998），但**个体范数 0.0121、均值范数仅 6.8e-05（比值 0.56%）**、与平摊均值夹角余弦 −0.0047 ⇒ 平均更新几乎为零。⇒ **"只换损失"不行**：两组对共享参数的要求相反，一阶更新要么共模、要么相消；要承载条件化必须让某组参数的梯度**逐样本不同**（输入条件的读出，其梯度是外积）。不训练 |
| 10.8 执行及结果信用 | 接入receipt、幂等、awaiting_outcome、取消与恢复；合法Outcome只更新指定预测/选择/表达模块，路由与实际更新分别记 | 错intent/伪造/重复回执拒绝；未知结果不能生成“成功”结算；取消/模型回滚不撤销已发生动作 |
| 10.9 工程、定价与合同 | 在分发前/待回执/回执后保存并独立恢复；冻结动作上限、rollout深度、参数/反馈责任、资源和门 | 来源/回执/恢复任一不成立，不做正式任务比较；真实环境扩展另定授权与资源 |
| 10.10 验证及收束 | 未见目标/状态/动作组合、实际终态、说明一致性、旧语言/记忆保持、恢复/总成本合取 | 预测错回world，预测对选择错回executive，动作对说明错回decoder；一个主要缺口结案，不重做全部底座 |
| 10.11 收束依据与负结果卡（范围完成，`closeout_10_evidence_complete`） | [合同](reference/SPEC-TAIJI-CORE_CLOSEOUT_10_20261101.md)＋[验证器](../scripts/training/verify_taiji_closeout_10.py)＋[测试](../tests/taiji_native/test_closeout_10.py)＋[复算守卫](../tests/taiji_native/test_closeout_10_recompute.py)＋[报告](../reports/taiji_core_10_closeout_v2_20261101.json)（v1 保留：[首版](../reports/taiji_core_10_closeout_20261101.json)，其验证器绑定写于 black 格式化之前，v2 为现行绑定；两次九项读数逐项相同）：只读复核 10.4–10.7 四份报告与 11 准入，不产生新读数 | 九项复核全过：同一候选摘要、四项出口原文在册、无能力声称、命中序列为交替形（未训练与 lr=1e-5 均 `FTFTFT`、lr=1e-4 `TFTFTF`，各 3/6）、保持门单侧破门、损失比 ≤0.70、权重不出库、11 仍 `step11_not_admitted`、预算在册。`p2_satisfied=false`、`stage_close_decision=owner_pending`；**收口决定权在 owner**，不训练 |

**子项编号说明（2026-11-01）。** 本步已执行的四个模型侧读数占用 10.4–10.7（其合同与报告文件名沿用交付时的 `10.4`…`10.7` 后缀，按§2不改名）；未开工的三张工程卡原为 10.4–10.6，与已交付读数重号，现顺延为 10.8–10.10，内容不变。

**复用与缺口。** `joint_action.py`与`world_learning.py`提供结构化目标/预测/选择资产，`joint_task_state.py`提供动作生命周期；它们尚不是新核心自然Goal生产链。`joint_learning.py`的信用路由目前只读，不能当实际学习器。拟将新核心桥接与自然Goal绑定放`taiji/native_content_action.py`；只在任务需要时实现，不复制整套旧系统。

### 11 自然中文连续任务与最小用户验收

**输入。** 已有受限能力证据的同一父候选，记忆或所选行动作为代表项。目标是在同一可加载入口完成自然任务，不用多个runner的不同模型成绩拼接成整模型。

| 子步骤 | 实施与交付 | 验收/停止 |
|---|---|---|
| 11.1 隔离加载 | manifest绑定候选ID、代码/配置、参数与状态schema、字节/分段编码、来源和依赖；提供同一主体独立加载，显式声明冻结评测/受控学习/持续运行模式 | 冻结评价禁写，持续运行按主体的活动/学习许可推进；新进程缺件拒绝，无隐式fit/下载/随机回退；不直接替换产品默认 |
| 11.2 单一任务入口 | 统一observe/answer/clarify/reset/save/load及适用act/outcome；先保存原始模型输出，再单列合法解码约束和产品加工 | 研究与候选入口在同可见信息下行为一致；显示层不得用模板补写正确内容，失败真实呈现 |
| 11.3 连续中文任务与评分 | 问答、材料任务、指令、追问、更正、澄清、未知；来源/关系/表达分组，任务内保持多轮；全应答轮与整任务终态双账 | 固定答案规则评分，自由回答按预声明量表并保留人工核验；辅助解析失败不单独否决正确回答；未知与过拒分开 |
| 11.4 当前候选仪器 | 接真实性A、所选代表机制F及当前设备H；核对来源、scope、缺权重、原始生成、重复运行、中断恢复、峰值内存及p50/p95 | 历史阈值审适用性；未测记未测，不能借别的checkpoint补签；不把所有长期H提前为任意原型前置 |
| 11.5 正式资源/能力协议 | 沿07原L2/代表项/安全/恢复/性能框架，冻结任务集、盲审、设备资源与阈值。若需新语料/更新范围，则在训练前另定合同 | 07建议起点B/C/D/E各20项、G至少20项及H至少30次混合运行仍按原建议/冻结身份处理；不能看成绩后改线，也不能借“集成”补训 |
| 11.6 CAP与最小验收 | 同bundle合取L2＋至少一项代表能力＋安全/恢复/性能，提供能力卡、原始对话、加载说明、限制及恢复包 | 达标才冻结最小L3范围供用户验收；不等待全T3/L4/L5，验收不等默认采用；失败回一个主要缺口 |
| 11.7 候选仪器读数（范围完成，`instrument_measured_quiet`） | [合同](reference/SPEC-TAIJI-CORE_INSTRUMENT_11_7_20261101.md)＋[验证器](../scripts/training/verify_taiji_instrument_11_7.py)＋[测试](../tests/taiji_native/test_instrument_11_7.py)＋[报告 v2](../reports/taiji_core_instrument_11_7_v2_20261101.json)（首版是修复前的失败跑，保留） | 20 次打分：第 1／第 20 次返回值逐位相同；p50 **0.120 s**／p95 **0.125 s**；建主体把峰值工作集从 217 MB 抬到 1.483 GB（+1.267 GB），20 次打分**再抬 0**；评分期间 0 写文件、0 socket、0 子进程、0 `os.system`；34.8 CPU 秒。⇒ 只在这台机器、这条打分路径上成立；不表示"主体安全"、不覆盖训练/生长/恢复路径、不外推。并入 11 门后 P3 翻真，准入仍 `step11_not_admitted` |
| 11.1a 候选 manifest 与隔离加载（范围完成，`isolated_load_verified`） | [manifest 模块](../taiji/candidate_manifest.py)＋[验证器](../scripts/training/verify_taiji_candidate_manifest_11_1.py)＋[报告](../reports/taiji_core_candidate_manifest_11_1_20261101.json) | 绑定候选摘要（135 张参数张量，`feb0cbcb…`）、6 个模块 SHA、bundle 字节与 SHA、config 与状态 schema、三种模式（冻结评测禁写）；本进程与**子进程**各自重建同一身份；负对照两枚（改模块 SHA、给 bundle 加 1 字节）**均被拒**；0.5 CPU 秒。不测行为、不签能力 |
| 11.2a 单一任务入口（部分完成，`entry_contract_verified`） | [入口模块](../taiji/candidate_session.py)＋[验证器](../scripts/training/verify_taiji_task_entry_11_2.py)＋[报告](../reports/taiji_core_task_entry_11_2_20261101.json) | `observe/answer/clarify/reset` 建在 11.1a 上：**raw 先于 decoded**、decoded 只能是 raw 顶字节（不得自填）、clarify 的 `content=None`、reset 逐位回初态、两会话重放 raw 逐位相同、`frozen_eval` 拒绝 `learn`；1.2 CPU 秒。**save/load/outcome/act 未交付** |
| 11.2b 入口往返（范围完成，`entry_roundtrip_verified`） | [验证器](../scripts/training/verify_taiji_task_entry_roundtrip_11_2b.py)＋[报告](../reports/taiji_core_task_entry_roundtrip_11_2b_20261101.json) | 快照存取后同一答案**逐位相同**、续一次观察后仍**逐位相同**；`outcome` 记录结果且**状态不变**（冻结模式不学习）；`act` 被拒并给 `no_action_channel`（该候选没有动作通路）；外来候选的快照被拒；1.2 CPU 秒 |
| 11.3a 固定答案任务套件（范围完成，`task_suite_floor_measured`） | [套件](../taiji/native_task_suite_11_3.py)＋[验证器](../scripts/training/verify_taiji_task_suite_11_3.py)＋[报告](../reports/taiji_core_task_suite_11_3_20261101.json) | 7 项（问答／材料／追问／更正／指令／澄清／未知），一次会话顺序提问、raw-first 入口、16 字节贪心：**可评分 5 项全 0**；澄清与未知 2 项**不计分母**（入口没有模型驱动的澄清/弃答通道，已点名）；2.2 CPU 秒 |
| 11.3b 弃答门（范围完成，`abstain_gate_degenerate`） | [验证器](../scripts/training/verify_taiji_abstain_11_3b.py)＋[报告](../reports/taiji_core_abstain_11_3b_20261101.json) | 弃答只看原始分布（top 概率 <0.5 或熵 >3.0 nat），**不用文本模板**：**7/7 全部弃答**（含 5 个可答项），命中 0；1.5 CPU 秒 ⇒ 通道存在但**不携带信息**；要它有用须先在**独立**标定集上定 τ／H，在同一批 7 项上标定就是拟合 |
| 11.3c 信度可分性（范围完成，`confidence_not_separable`） | [验证器](../scripts/training/verify_taiji_confidence_separable_11_3c.py)＋[报告](../reports/taiji_core_confidence_separable_11_3c_20261101.json)：只读 11.3b 卡，不重跑 | 可答组 top 概率 **[0.009005, 0.009137]**、应弃组 **[0.009092, 0.009106]**——区间**重叠**；熵两边都 **5.4960–5.4961**（均匀参考 5.549）⇒ **没有任何阈值能分开**：不是门没调好，是分布近乎恒定、信号本身不存在 |
| 11.4a 候选仪器台账（范围完成，`instrument_facets_inventoried`） | [验证器](../scripts/training/verify_taiji_instrument_inventory_11_4.py)＋[报告](../reports/taiji_core_instrument_inventory_11_4_20261101.json)：只读 7 张卡，不重跑 | 10 项面：**7 项有读数**（来源／缺权重／原始生成／重复运行／峰值内存／p50-p95／代表机制 F）；**3 项点名未测**——**中断恢复**仅快照往返（未做进程 kill）、**真实性 A** 未做、**设备 H** 未采集。"未测记未测"，不拿空白当零 |
| 11.4b 设备档（范围完成，`device_profile_measured`） | [验证器](../scripts/training/verify_taiji_device_profile_11_4b.py)＋[报告](../reports/taiji_core_device_profile_11_4b_20261101.json)＋台账 [v2](../reports/taiji_core_instrument_inventory_11_4_v2_20261101.json) | **24 逻辑 CPU／12 torch 线程／无 CUDA／48 GiB**（`GetPhysicallyInstalledSystemMemory`；`GlobalMemoryStatusEx` 在本机返回 0，已换源并写进卡）。并入台账后 **11.4 v2：8 项有读数，剩 2 项未测**——中断恢复仅快照往返、真实性 A 未做 |
| 11.4c 中断恢复（范围完成，`interrupt_recovery_verified`） | [验证器](../scripts/training/verify_taiji_interrupt_recovery_11_4c.py)＋[报告](../reports/taiji_core_interrupt_recovery_11_4c_20261101.json)＋台账 [v3](../reports/taiji_core_instrument_inventory_11_4_v3_20261101.json) | 子进程观察→写快照→**`os._exit(0)` 硬退**；全新进程从该快照续跑，答案与父进程**逐位相同**；两子进程 rc=0；1.1 CPU 秒 ⇒ 快照在硬退出瞬间是完整的。台账 v3：**9/10 有读数，只剩真实性 A** |
| 11.4d 仪器收口（范围完成，`instrument_closed_pending_realism_and_capability`） | [验证器](../scripts/training/verify_taiji_instrument_closeout_11_4d.py)＋[报告](../reports/taiji_core_instrument_closeout_11_4d_20261101.json) | 9/10 有卡、唯一缺 **真实性 A**；代表机制 F＝`capability_line_closed_negative`；卡里写死 **`candidate_usable=false`** ⇒ 仪器面齐**不等于**候选可用，11 仍 `step11_not_admitted` |

**代码边界。** 复用`api/seed_runtime.py`和`taiji/adapter.py`的运行适配边界，拟新增隔离候选封装`taiji/native_content_candidate.py`。现有chat有产品处理且默认learn开启，新入口须显式区分运行模式并保留原始输出：固定能力评价关闭学习，持续主体模式保留获准的状态推进与可塑活动，不能一律冻结整个主体；本方案不在当前阶段改产品默认。具体命令只在真实入口交付后登记。

### 12 跨会话纠错、反馈学习与旧能力保持

**目标与输入。** 在09已经建立的持续状态、合法学习和结构可塑性基础上，扩展多session、分布变化、长期保持与污染治理；本步是长跨度L4范围验收，不是首次接入参数学习。只把新事实写入store、只恢复参数或降低训练loss，均不等于持续参数学习。

| 子步骤 | 实施与交付 | 验收/停止 |
|---|---|---|
| 12.1 冻结经验流 | 父bundle/旧能力集、任务变化、session边界、support/query/保持分区；先留原预测，再按release_at释放合法反馈 | 每条反馈含id/source/owner/scope/task/prediction revision/允许接收组/许可撤销状态；未来标签及保持答案不可见。缺一责任边界不训练 |
| 12.2 持久准入与撤销 | 显式跨session携带许可、冲突/过期/撤销、幂等反馈cursor；迟到反馈绑定原版本；来源信任由授权验证，不凭caller一句experienced | 跨owner/scope、重复/乱序反馈、撤销后恢复、imagined/unknown结果反例；不得用默认cross_session=True当完整许可 |
| 12.3 实际反馈更新 | 在原只读路由之外实现接收/应用更新，登记有效参数delta、禁写组和反馈→负责模块链；编码器若更新，按09重编码与revision策略处理旧latent | 路由存在、梯度非零和参数改变分别报告；如本包仅改reader/decoder则声明范围，不能包装成全部核心已会学 |
| 12.4 恢复与定价 | 零更新、首次合法更新、session切换、撤销/过期边界分别保存并独立恢复；包含store、反馈cursor、预算、optimizer/RNG、参数及encoder revision | 恢复不能复活撤权、清零费用、重复学已消费反馈或重发已结动作；任一不一致为工程失败，先不正式比较 |
| 12.5 同资源有界流比较 | 至少比较同记忆冻结参数与同记忆获准更新；按主张加反馈置换/负责模块冻结。同父初态、反馈时序及数据权利 | 更新组在隔离query优于同记忆冻结组，父保持不越预注册容差，才支持参数学习；新事实记忆和新规则/表达学习分别统计 |
| 12.6 长期读数与收束 | 按独立流/来源估不确定性；适应速度、累计错误、变化后恢复、父保持、污染/来源撤销、峰值存储及全成本；当前候选A/F/H和长期窗口冻结 | 正负/不可判/工程/预算有限结案；退化保留父版本，回更新/巩固，不自动增流长度/剂量或拿短期H代长期健康 |

**工程复用。** `native_content_core.py`提供带来源/时间/版本的store与重编码，`joint_task_state.py.new_session`提供显式carry引用的协议资产，`joint_learning.py`提供只读责任映射。`NativeRunTrainerV1`当前逐episode新建store，不能直接当持续经验流执行器；`continual_evaluation.py`、`continual_checkpoint.py`及`seed_platform/training_checkpoint.py`仅复用适配后的保持/游标/血缘资产。拟新增流更新核心`taiji/native_content_adaptation.py`，不强套旧多worker结构。

结案必须能区分六种结果：状态记忆有效但更新无额外收益；反馈没到负责模块；更新生效但新query无收益；新收益伴父退化；旧latent失配；来源/恢复使结果不可判。以这些事实选择一个后继，不能用“终身学习系统已接通”代结论。

### 13 有边界主动选择与条件扩域

**首包选择。** 在09已有基本内源活动、学习及成长闭环上，扩大“继续/求证/求助/等待”的自主策略与可干预范围；按一个真实缺口扩展成长或其他能力，不同时铺开全部多模态。无新提示仍须有有效目标、合法可选行为和剩余预算；Seed负责时钟、权限、资源和执行，不替模型预选正确策略。

| 子步骤 | 实施与交付 | 验收/停止 |
|---|---|---|
| 13.1 主动任务合同 | 冻结信息足够、缺证据、能力不足、待回执、权限不足、资源不足等情境及正确处理；候选顺序随机、ID不编码正确性 | 明确模型选择与宿主拒绝/执行边界；首包为受控模拟，无授权不联系外部人或执行真实副作用 |
| 13.2 模型选择入口 | 读取Goal、证据/记忆、不确定性、未结Outcome、能力与资源，输出选择及来源；宿主仅提供合法候选mask | 固定Homeostatic阈值/if-else作基线，不能称模型学会主动性；拟新增`taiji/native_content_autonomy.py`承接实际可学习选择 |
| 13.3 结果信用 | 选择绑定intent/参数revision/所消费内容/receipt/后续任务终态；各选项成本预声明，合法结果更新负责选择模块 | 未知/待回执保持pending，重复或迟到结果不重复奖惩，预测不替真实Outcome；求助答案来源另列，不伪造成模型知识 |
| 13.4 边界与恢复 | 验等待无忙循环，冷却/重试/最大决策数/剩余预算可恢复；撤权停止新执行，缺回执不反复发新intent | 越权、无限循环、账本/幂等失败即停止；模型回滚不撤销外部既成动作 |
| 13.5 策略比较 | 同目标/环境初态，对照固定调度、随机合法和必要的简单外控策略；比任务完成、累计收益、无效求证/打扰、合理等待、旧保持及总成本 | 主动采集导致信息分布不同要记录实际获得量；录制流回放仅证离线策略，主动探索效力需受控交互证据 |
| 13.6 收束与单项扩域 | 按冻结门结案主动选择范围；若有真实瓶颈，再选更大规模成长、多模态或更复杂调度中的一个合同，继承09已验的基础生长 | 扩大成长范围仍加同最终容量对照/消融/迁移回滚，多模态加同事件交换/单模态控制；不自动宣称开放自主发展或同时开全部长期方向 |

这里后置的是成熟自主策略、更大规模和扩域验收。基础连续状态、可塑学习与实际结构生长/贡献从09共同内核开始设计、实现和检验，不能等到13才首次接入；继承、信用、保持与恢复贯穿后继阶段。R5c已按原固定容量范围结案，保留结果，不追加上述成长实验。

复用`executive.py`的可更新utility/候选选择与合法Outcome资产，`homeostasis.py`只提供状态和固定规则对照。模型自述“我决定等待”、定时唤醒或守护进程运行，不能代替选择贡献及后续任务收益证据。

### 所有未执行工作共用的实施账与资源冻结

以下是09–13各包必须填写的实施字段，不另立编号队列。08沿原批准合同；09–13的方案已细化，但尚不具备正式实验的数值/资源准入，不能自动沿用512更新、120,000 CPU秒或R5门。

| 必填交付 | 落地要求 |
|---|---|
| 候选与变化 | 父bundle/代码SHA/配置/schema、已验能力、继承/变更/新增/不支持清单；原始生成与动作归属明确 |
| 数据与分母 | task/source/history/stream分组、经历拟合/活动奖励P_R/准入P_A/独立确认保持P_C身份与hash；全部应答轮和任务终态；训练/运行/评分三视图隔离。奖励和准入反馈均记累计自适应暴露与次数上限，不以简单样本拆分代替组隔离 |
| 数量和成本 | 精确来源组/样本/episode/turn/目标字节数，seed/每臂更新/反馈/动作/存储量；定价pilot自身先限更新/反馈/CPU/wall与数据可见性，再测单位成本；正式总账包含工程/恢复/评价开销 |
| 比较与门 | 主任务端点、合理基线、仅为科学上界的oracle、必要因果/信用控制、最小有用效应和区间、父保持容差、来源/恢复/成本硬门；正式结果前冻结 |
| 执行与恢复 | 单写者run-root、原子checkpoint、模型/store/optimizer/RNG/cursor/预算、适用反馈与动作receipt；最早保存点先独立恢复再训练 |
| 产物与CI | 源码沿taiji，测试沿tests/taiji_native，运行器沿scripts/training，报告沿reports，数据/身份沿plans/manifests与受控根；实际CLI存在后才登记命令。只跑受影响检查，保存同提交CI证据 |
| 有限出口 | 每个正式包一轮冻结比较；新增seed/剂量/方法/数据需具体修订，不因未过门自行追加。输出架构归因、竞争解释、实际费用和一个后继 |

决策顺序：先在各步的合同/静态设计子项完成上述草案并过工程门；在定价前冻结train-only pilot自身的更新/反馈数、CPU/wall、保存停止线及数据可见性，未填实不得运行；再依据实测冻结正式比较的数量、预算和效应门，最后进入有界比较。定价失败不自动追加。无法从当前证据确定的参数，不在本轮编造为冻结数值；缺字段即正式投入未准入。设计、工程、获准训练、范围通过、产品采用五种状态分别记，不用方案存在代替完成。

## 5. 统一验收、失败回流与停止

三类未解命题继续分别记账：方法能否获得未见净收益及投入退出点；工程接通能否转为联合学习；观测是否覆盖当前候选并能裁决。旧producer无回答梯度、TSK已有边局部更新与共享192→48读出的信息限制都是特定路径的事实，不能外推全模型必败。详细证据见[核心重审](reference/TAIJI_CORE_DESIGN_REVIEW_20261030.md)。

每个后继包在投入前写清：父候选与变化、任务/覆盖/数据权限、信息路径、loss到参数/状态的信用路径、基线与因果控制、旧能力、来源/恢复、CPU/wall/产物成本、数值门、累计停止线及唯一后继。训练之前先验checkpoint保存与独立恢复；梯度非零、loss下降和合同数量不替代行为收益。

| 实际出口 | 必须保留的结论 | 回流规则 |
|---|---|---|
| 范围内通过 | 只证明声明任务及所测机制/成本 | 先审下一步范围和合同，再决定是否投入；不自动扩面/采用 |
| 行为或泛化不足 | 已见拟合、未见任务与资源分别报告 | 选一个主要缺口提出有判别力的候选，不默认加轮 |
| 内容/信用无有效贡献 | 正确性、状态依赖、信用路径和旁路分账 | 查表示→持久化→选择→表达及竞争原因，暂停依赖此机制的扩训 |
| 旧能力/来源/恢复失败 | 新任务收益不能冲销退化或越权 | 保留父版本，回更新约束或具体工程边界 |
| 不可判 | 点名缺失的效应量、分母、身份或控制 | 先修判别设计，限定补证，不无限观察 |
| 工程失败/预算停止 | 不算科学反证；实际费用全保留 | 按原合同停止，不自动清账、重跑或提额 |

2026-10-31根目标纠正：前面关于“先换表达骨干”“以后条件加入生命/成长”的排期解释已撤出当前默认顺序。RNN/Transformer的状态表达、信息传播和优化性质参与决定核心算法；后继必须先完成统一神经算法设计，不能只靠外围生命周期流程拼装。算法选择同时服从持续主体、可塑性、内源活动及继承式结构生长要求。此前有界配方仍按原证据解释；其冻结权重快照只能是研究对照/测量切片，不能成为递归进化主体的最终形态。细节和纠偏记录见[核心重审§13](reference/TAIJI_CORE_DESIGN_REVIEW_20261030.md#13-20261031-根目标纠偏连续生命主体与递归进化)。

原N177 F合同累计重审计数与费用不因改名、parser改善或新错误发现清零。新候选同样要有预先写明的有限出口；步骤08.5和每次失败后均先作项目级收束，防止局部补洞脱离总目标。

2026-10-31历史驱动复核：算法明确归类为带快权重与结构可塑性的多时间尺度递归图网络候选；不是标准GRU或Transformer堆栈，也未形成公认新类别。旧M4跨情境utility/lesion差异提示优先比较条件化出生支路；四点合成反例已显示全局C可为0而条件启动梯度非零。09.2须覆盖该盲区，但v1.1默认公式和旧22项报告不变；此为修改候选而非已证任务收益，见[算法§22](reference/TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md#22-算法类别与历史驱动的修改意见20261031)。

2026-10-31验证方法说明及E1复核：现有22项只核验合成耦合代理，正式连续chi、Q/K/V与查询输出缺失，多组矩阵未验梯度，具体差集见[算法§21](reference/TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md#21-v11方程审查与验证记录)。09.2先做方程对应，再验小信号容差/eps负对照、float32实际更新与完整状态迁移/恢复；探针分为P_R/P_A/P_C并限制自适应次数。编码更新后的H/F语义与全历史重算代价仍未解决，不能以revision字段或恢复一致代签保持。规格勘误、证据和待验证项分别记，原脚本/报告/训练权限不变。

## 6. 旧方向、组件与延期项

| 资产 | 当前用途与边界 |
|---|---|
| N177 E/F、N178 parser、旧V4 | 失败/诊断证据或修正旧图基线来源；不自动追加剂量、seed、teacher或旧sealed重评 |
| TSK、identity、精确复制、GRU、原状态容器 | 可继承和比较的组件；按实际任务路径验证，不因名字保留或整体推倒 |
| N1–N6、CC、VX、VISION | 机制/职责/长期设计参考；从总计划选取，不发布另一套工单 |
| M0–M8、R2日对话、UI生命页 | 按原批准范围保留证据与发布面；不能拼成绩或变成当前研究前置 |
| A/F/H仪器与历史阈值 | 可复用仪器，当前候选覆盖/适用性/自动门另验；CAP H06和长期跨度不得借历史短期读数代签 |
| 硬件优化、激进重连、共享场替换、多模态 | 仅在实际瓶颈与新合同成立时进入对应步骤，不并列启动 |

新核心的旧资产复用/重实现/基线/数据/权重及删除条件统一见[旧算法资产分流§5](reference/LEGACY_ASSET_INDEX.md#5-新核心对旧算法资产的具体处理20261031)。09.2只按窄依赖表引入已审计的通用实现或适配协议；不强制继承旧权重、旧类层次或旧训练配方，不把旧通过结果转移为新能力。

完整旧资产路径见[LEGACY_ASSET_INDEX](reference/LEGACY_ASSET_INDEX.md)。旧文“当前/下一步”仅解释写作时点；已有独立授权运行按其原合同结案，本次不对进程做停止、恢复或追加。

## 7. 文件职责与维护

| 文件/目录 | 只负责什么 |
|---|---|
| 本文件 | 唯一总计划：编号、顺序、当前状态、实际执行位置、工作卡和出口 |
| [03台账](active/roadmap/03_CURRENT_EXECUTION.md) | 审批、报告、费用与限制的证据出处；队首标题仅静态跳转本页，不另发动作 |
| [10兼容入口](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md) | 旧章节跳转与编号解释；不复制状态表 |
| R1–R5b专项合同 | 原候选、数据、参数责任、冻结门/预算及历史执行记录；按步骤查阅 |
| [核心需求](active/TAIJI_CORE_REQUIREMENTS.md)、[架构规范](active/TAIJI_NATIVE_ARCHITECTURE_V1.md)、[07](active/roadmap/07_MINI_MODEL_DELIVERY.md) | 长期约束、认知所有权、能力采用规则；不维护执行队列 |
| [02](active/roadmap/02_GATES_AND_CI.md)、[05](active/roadmap/05_TECH_DEBT_REGISTER.md)、[08](active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md) | CI、技术债、上游重放等运营 |
| `reports/`、`plans/manifests/` | 唯一机器证据、受控数据身份；不因整理更改旧报告或分区 |

每步完成：先对照真实源码/报告/批准，全局核对本表与执行位置；再补03证据引用和相应合同执行记录，检查受影响链接/CI并提交本包。索引只导航，不抄状态。只删去活动文档中失效的重复指令；原失败、唯一报告、冻结件和checkpoint保留，不批量清理无关工作。

`plans/active/`文档状态标签、索引覆盖和单主线守卫继续生效；03的唯一“当前唯一下一步”标题保留为兼容跳转，本页“执行位置”是实际任务唯一来源。文档包核对编号、链接、锚点和差异，不为整理计划启动模型或研究训练。

## 8. 整合记录

2026-10-30第一次整合：将MAINLINE、03、10和旧重构方案中重复的排期收敛至本页，统一步骤01–13及08.1–08.5；同步R5b已批准事实；区分阶段零步快照与后续已训练事实、同面回归与独立确认、训练内插与父能力保持。原R/N/CC/VX/L身份及冻结条款继续可追溯。

2026-10-30后续定位：项目所有者将本件确定为新的N主线；沿用01–13，补全08余项和09–13逐子步落地卡、真实代码复用/新增缺口、输入/信用/恢复/对照/资源/退出及转段责任。依据`dc7a43354`与`b78bdfdce`将实现/执行件提交与工程验收分开，不触碰并行执行件或既有冻结门。

2026-10-30项目级收束：完成源码/脚本/测试/报告/计划/客户端与根目录的归属盘点，归档11文件、删除26文件及旧空目录，释放108,750,241逻辑字节；详见[整理记录§8](reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md#8-20261030-项目级整理收束)。当前R5b、冻结证据、数据/权重及有消费者的旧路径保留，执行位置不变。

2026-10-31递归骨干讨论：补入核心重审§10，确认当前GRU组件与内容系统变体的区别、外部递归模型证据及其迁移限制、同面实验归因和后继骨干对照判据；同步已核验的R5c批准/实现事实，不新增研究执行授权。

2026-10-31非RNN替代讨论：补入核心重审§11，明确优先表达骨干对照、潜在收益/代价、信用与持久状态独立性，以及现行Transformer block限制的采用决策；不变更现有有界包或自动开训。

2026-10-31规模生长讨论：补入核心重审§12，区分已有连接、固定容量学习、重连和真正继承式扩容；将成长兼容设计前置到后继骨干合同，同时保持正式自主成长的范围与授权边界。

2026-10-31根目标纠偏：按所有者明确要求，将连续生命活动、持续可塑性和按需规模生长统一为主体基础；09.1共同内核合同取代先换固定表达骨干的默认设计队首，12/13改为长跨度与成熟自主发展验收。原01–13编号、冻结实验和有效运行授权保留，本轮不执行新服务/训练。

2026-10-31核心算法纠偏：09.1交付明确为核心神经算法设计书，不再以主体接口/调度合同代替；根需求和架构§0.4同步要求可实现的状态、学习、记忆与生长耦合规则。当前尚无该统一算法已完成或有效的结论，不改原实验/授权。

2026-10-31算法首稿交付：所有者允许不沿用当前模型；新增单一[核心算法v1草案](reference/TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md)，选定可塑神经图和明确耦合规则，完成11项局部合成代数核对。后继09.2验证完整参考微步、信用与一次出生/恢复，不把局部数学检查签成模型或自主成长通过。

2026-10-31方程审查开工：按所有者顺序先调整主线，将原式反例、耦合自动微分核验及v1.1修订列入09.1.1–09.1.3；09.2参考实现与正式训练不抢跑。

2026-10-31方程审查收束：原式复现快记忆无写入放大、近零RMS放大，以及整体loss平均/变时长信用的定义缺口；修订为v1.1并通过22项局部与耦合核验。原报告不覆盖，下一步仍需真实主体参考图和完整恢复，不能据数值检查签能力通过。

2026-10-31旧算法资产处置：索引新增具体代码与权重/数据/证据的分流表，区分实现复用、语义适配、重新实现、基线留存和审计后清理；本轮不搬删源文件或模型，旧消费者迁移在物理退役前完成。

2026-10-31验证差集收束：采纳P0/P1复审，撤回“按正式方程耦合已核验”的过宽措辞；E1补齐读取/归一化规格及探针、状态迁移要求，09.2细分为四个顺序工程交付。当前入口为09.2.1正式图逐项对齐，未执行新模型实现/训练或覆盖旧证据。

2026-10-31 E2与新颖性收束：采纳有效秩限额/逐单元准入、随机内容盲门方差及chi冻结语义三条；小矩阵复算确认相应边界，未训练模型。新颖性定位为已有算子组合与稳定化候选，组合贡献需实证，近端零目标步不等同完整RLS。补五组主张相关比较及随机门负对照，保持09.2.1入口，不新增研究投入授权。

2026-10-31 v1.2方向决策：研究增量集中生长信用与覆盖出生；14项局部预审反驳通用活单元数上界、软门秩和样本上界、重放单调增秩、仅凭H秩或共享簇覆盖判容量等推广。主稿§24给出条件正确的命题、覆盖分配、诊断与可失败预测，并登记GradMax/TINY/梯度秩/GSS/Subspace Trap等直接先例；新颖性未定。09.4后再审预期自我修改，09.5后再议精确撤销扩展，当前唯一入口仍09.2.1，不新增组件/训练。

2026-10-31 R021工程交付：六接口与软门/覆盖界先冻结，再实现正式参考前向，12项oracle/10行覆盖矩阵通过，101个参数张量归属登记；无反向传播或优化器更新。五份v1.1/E1独立审计入库，原字节单独封存并记录只读版本映射/已知方法局限，不重跑玩具优化或改旧报告。执行入口推进到09.2.2，未越过出生/恢复/定价门，不宣称有效新模型成立。

2026-10-31 R022信用判别交付：先冻结[R022合同](reference/SPEC-TAIJI-CORE_CREDIT_R022_20261031.md)（三负责目标、零类、FD/eps/二阶判别、负对照、fp32对比与两步声明Adam更新表达式），再交付检查与write-once报告；09.2.2按`passed_r022_credit_only`结案。执行中发现并补录两个结构性事实：`episodic.score.bias`（I3的b_3）为softmax平移不变死参数（值置换对照无效应，建议后续R021接口修订删除或标注）；U初值全零使置零对照空转（改为激活对照）。对照阈值按"连通性由对照负责、效应大小由FD确认的梯度负责"的语义修订并落账一阶预测。101张量全部获得确认信用或声明零类，无意外断图；合成更新后参数即废弃。执行入口推进到09.2.3，不授予训练/定价/出生。

2026-10-31 R023覆盖出生交付：先冻结[R023合同](reference/SPEC-TAIJI-CORE_GROWTH_R023_20261031.md)，再交付[ growth_credit.py](../taiji/growth_credit.py)（信用/标定/分配/门控出生/全局基线/共享支路守卫）与[state_migration.py](../taiji/state_migration.py)（§10.3 journal恒等重放、编码变更事务、预算拒绝），反例电池8/8按`passed_r023_birth_engineering_only`结案。三个执行期发现按纪律落账：①§24.4新生单元是门控的（"实际新支路和诊断必须使用同一个g"），门控前向以`GatedGrowthCore`实现、living_core.py字节不变；②稠密布局下逐群不对称出生需R/P零填充（padding单元梯度恒0）；③**冻结0.99置换噪声bar在本夹具不拒绝随机门**（真实/随机门余量同为1.03–1.05倍，时间平滑共同模抬高位假设）——阈值未改，作为R021 §4标定协议owner决策输入。执行入口推进到09.2.4，不授予训练/定价。

2026-10-31 R024主体恢复交付：先冻结[R024合同](reference/SPEC-TAIJI-CORE_RESTORE_R024_20261031.md)（bundle schema、五边界、spawn独立进程、声明更新表达式、resolution幂等闸），再交付state_migration扩展（显式step序列/断点续推/save/load/resume_subject/worker job）与worker CLI，六项检查按`passed_r024_restore_engineering_only`结案。执行期事实：Windows下torch张量不能过进程队列（结果改走文件传递）；spawn目标必须按名可导入（`run_worker_job`直挂target）；bundle声明Adam状态使"下一次更新"逐位可比较；resolution以当前journal唯一闸拒绝旧快照重放重复领奖。**09.2.1–09.2.4全部关闭**，执行入口推进到09.3连续动力学，不授予训练/定价；迁移恢复通过明列不替代语义保持。

2026-10-31 R025活动循环交付：先冻结[R025合同](reference/SPEC-TAIJI-CORE_ACTIVITY_R025_20261031.md)，再交付[native_activity.py](../taiji/native_activity.py)（六活动算子/策略头/触发量/固定阈值对照）与五项检查，按`passed_r025_activity_engineering_only`结案。执行期声明：预测误差＝世界头对**池化后继表示**的读数（世界头从z预测主体级d维后继，与逐群h形状不同——真实世界损失与EMA目标属09.4）；触发快照在执行前取（队列pop后计数才不失真）；模型政策在未训练下的选择由掩码softmax＋argmax决定，测试只断言可追溯性与合法性、不断言具体选择。想象不写store、wait无前向步、maintain显式no-op均逐轮入journal；策略头参数hash前后一致。执行入口推进到09.4持续可塑性与结构出生，不授予训练/定价。

2026-10-31 R026六臂预注册比较交付：先冻结[R026合同](reference/SPEC-TAIJI-CORE_PLASTICITY_R026_20261031.md)（预注册主终点/保持探针/净收益判据/预算判停），再交付[native_development.py](../taiji/native_development.py)（声明反馈更新/覆盖出生/GradMax固定日程/仅条件SVD/随机门/仅重放/容量匹配随机臂）与四项检查，按`passed_r026_preregistered_comparison`结案、**预注册裁决＝negative**。执行期事实：①保持探针必须独立于更新数据（fit journal的NLL下降是学习不是保持——合同在正式运行前修正定义）；②六臂必须各自持有独立父副本（replay_only原地更新共享parent曾污染后续臂——修复为每臂deepcopy）；③在声明更新规模下六臂全部超出0.05保持容差（0.144–0.148），coverage_birth未优于对照（差+0.0032＞1e-3边际），臂间holdout差≤0.0035——出生方向8步内未自我偿付，更新幅度本身破坏保持。按合同不追加剂量/seed/臂；负结果不证伪真实条件下的v1.2候选。09.4.1预期自我修改不触发。执行入口推进到09.5主体恢复门，不授予训练/定价。

2026-11-01 R027恢复门合拢交付：先冻结[R027合同](reference/SPEC-TAIJI-CORE_GATES_R027_20261031.md)，再交付native_activity扩展（env序列化/来源合法性过滤/轮级门控索引）、worker `activity_round`模式（门表兼容对值对与字典两种格式、career外步按模回绕）与两项检查，按`passed_r027_gates_engineering_only`结案。执行期事实：①门控子模型的迁移重放与活动轮必须携带门表——replay_transaction新增provider参数，worker对career外步按模回绕门表（声明仪器）；②R024 provider重构时曾丢失None守卫致spawn子进程崩溃，回归立即捕获并修复。八次边界（交互/无新提示/睡眠等待/首次学习/出生前后/迁移提交/未结Outcome）全部spawn独立恢复逐位一致；清单十二项核验落账。执行入口推进到09.6定价与冻结——**owner裁定节点：训练准入在09.6合同上裁定**。

2026-10-31长期目标文字收束：保留自然中文、连续任务、反馈学习、旧能力与有限出口，明确连续状态/持续可塑/继承式真实规模生长；将“参与”收紧为认知机制实际承担决策与表达，并把生长整合、主体连续性/恢复及全过程成本纳入按范围验收。阶段方法不写入根目标，13明确为成熟与扩域，不再后置基础生长。执行位置和实验权限不变。
