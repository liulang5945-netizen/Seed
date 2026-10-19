# Seed / Taiji 计划与架构入口

> **2026-10-19 N1队首**：N1-139已完整执行，55,296次更新、7,488条train-only计分、12/12 bundle恢复，来源哈希不变；累计CPU 3,493.89/3,600秒。原训练门5/6通过；seed13候选transition从48/48降至28/48，候选增强上下文840/864、两行控制411/864，未查询dev/holdout，can_promote=false。结案见[PLAN-N1-139](reference/PLAN-N1-139_execution_closeout_20261019.md)，逐题归因见[报告](../reports/taiji_n1_plan139_itemwise_retention_context_attribution_20261019.json)。用户已批准[PLAN-N1-140](reference/PLAN-N1-140_baseline_retention_balanced_matched_continuation_draft_20261019.md)：每父项baseline 24/48、六种增强条件各4/48，六分支55,296次更新、7,488条train-only计分、CPU≤7,200秒。N1-140已完整结案：55,296次更新、7,488条train-only计分，zero/final bundle独立恢复各6/6，来源及12枚checkpoint哈希均通过；累计CPU 2,553.406/7,200秒。N1-132原训练门5/6通过，唯一失败为seed13 candidate transition 19/48（重命名14/48）；candidate增强上下文662/864、control 411/864，未查询dev/holdout，can_promote=false。详见[N1-140中文结案](reference/PLAN-N1-140_execution_closeout_20261019.md)、[完整结果](../reports/taiji_n1_plan140_baseline_retention_balanced_matched_continuation_result_20261019.json)与[逐题归因](../reports/taiji_n1_plan140_itemwise_retention_context_attribution_20261019.json)。用户已批准并冻结[N1-141](reference/PLAN-N1-141_interleaved_context_schedule_matched_continuation_draft_20261019.md)：仅交错candidate slot顺序，其余配方、55,296次更新、7,488条train-only计分和原门不变，CPU≤7,200秒。runner与契约测试已提交（5/5通过）；来源/schedule readiness通过，2,304个transition slot及交错比例核验通过，274项来源哈希不变；CPU4.359秒，0模型/checkpoint加载、更新及查询。N1-141已完整结案：55,296次更新、7,488条train-only计分，zero/final恢复各6/6，累计CPU2,581.750/7,200秒；N1-132原训练门6/6通过。candidate增强上下文834/864、control411/864；seed13 candidate transition从N1-140的19/48恢复至48/48，29个旧错全修复、无新增错。未查询dev/旧holdout，can_promote=false。详见[N1-141结案](reference/PLAN-N1-141_execution_closeout_20261019.md)、[完整结果](../reports/taiji_n1_plan141_interleaved_context_schedule_matched_continuation_result_20261019.json)与[逐题归因](../reports/taiji_n1_plan141_itemwise_retention_context_attribution_20261019.json)。依据用户授予的执行计划批准权，唯一下一步为已批准冻结的[N1-142 v21新题确认](reference/PLAN-N1-142_v21_fresh_confirmation_after_n141_draft_20261019.md)：160题、12端点、先范围后确认，最多1,920查询/CPU3,600秒。当前实现source union、face builder和评分runner；未生成新面或查询，全部前置通过后才消费v21面。

> 历史 N1-123 进展记录保留作追溯；当前状态以本页首条与[03当前执行](active/roadmap/03_CURRENT_EXECUTION.md)为准。本页只同步N1研究队首，不改写项目整体M0–M8状态。

> **历史快照（2026-10-19 N1-126阶段，已由上方N1-135状态取代）**：v14全上下文位置面已结案，candidate position38/60、cut37/60，确认门未通过；只读审计发现当前模型已包含问题对齐及关系残差机制，错误主要是lookup误拒答和transition错方向。该时点建议为[PLAN-N1-126](reference/PLAN-N1-126_source_pruned_saved_error_diagnostic_draft_20261019.md)，0训练、180查询、CPU≤600秒。本段仅保留历史记录，不代表当前队首。

> **2026-10-15 主线进度**：[PLAN-N-01](reference/PLAN-N-01_development_revision_20261014.md) · [C2 校准预注册](reference/PLAN-N1-07_C2_consumer_calibration_prereg_20261014.md) · [C2 预算裁决](reference/PLAN-N1-07_C2_calibration_budget_adjudication_20261015.md)。C1完整面通过；C2单器官校准112/112臂完成，累计601.91秒CPU；选择8/16，七臂严格命中均0/16。完整结果见 [C2裁决](reference/PLAN-N1-07_C2_calibration_result_adjudication_20261015.md)；状态流诊断48对/96查询已完成，CPU155.08秒，见 [08裁决](reference/PLAN-N1-08_C2_state_flow_adjudication_20261015.md)。09补零位置48臂也已完成，CPU77.5秒、三臂严格命中仍0/16；当前唯一建议为 [10核心学习路线评审](reference/PLAN-N1-10_native_evidence_credit_route_review_20261015.md)。低成本诊断无需逐步审批。已批准长期连续推进；当前唯一队首以 [03](active/roadmap/03_CURRENT_EXECUTION.md) 为准。

> **2026-10-14 收纳导航**：[项目地图](../docs/README.md) · [全项目评估](reference/PROJECT_WIDE_ASSESSMENT_20261014.md) · [整理记录](reference/PROJECT_ORGANIZATION_20261014.md)。本页下方旧阶段摘要保留追溯；当前状态与执行顺序以 [03](active/roadmap/03_CURRENT_EXECUTION.md) 为准，架构与模型主线见 [09](active/roadmap/09_NEXT_MAINLINE_PLAN.md)。M5/M6 已批准收束，勿把下方历史“当前仍为 M5”当作现状。

> 更新：2026-09-18。用户澄清要补的是像多模态、恶意信息一样遗漏的完整方向：本轮唯一VISION §23新增能力/风险版图及八项实际方案，根需求/架构/01/07同步入口。§20多模态、§21恶意信息继续保留，§22归位为运行机制细化，不再充当大方向查漏的回答。推荐设计不等于架构采用或新运行授权，局部实验仍按02 §2.5有条件准入；最新研究状态与唯一下一步只看03及有效合同/裁决。

## 项目位置

先读[当前结果与结论总览](reference/PROJECT_RESULTS_AND_CONCLUSIONS_20260917.md)：按正向成果、负结果、不可判结果、真实输出和结论边界整理；运行收束状态见[收束记录](reference/PROJECT_CONSOLIDATION_20260917.md)。

当前仍为M5/R2原生语言。D1测量底座可继承；本次续写核对的[D8/v3 matched报告](../reports/r2_d8_matched_dev_20260918.json)多字值完整率均值0.3621达到K1，但M4/flip为0，整体outcome仍failed、未准晋级，不能抹去复制正信号或写成完整语言通过。最新逐包裁决见[03](active/roadmap/03_CURRENT_EXECUTION.md)及对应有效合同；若摘要与新报告不同步，先核对身份/裁决，不用旧“待启动”覆盖已发生结果。首页不另设活动队首；旧672结果仍按[历史复核](reference/M5_DISTILLATION_TOMBSTONE.md)保留。

当前核心缺口是同任务中的共同表示、记忆消费、世界预测、规划选择与学习信用未形成统一能力证据。原P3b的数据分布效应只有一个共同检查点，不能判定；P5.2d后继报告虽有仪器修正，A1真实收益仍未闭合，不以预算放宽或汇总布尔代替。

## 怎样用这套计划指导开发

| 你要回答的问题 | 主要入口 | 应得到的指导 |
|---|---|---|
| 项目整体还缺什么，何时晋级？ | [01 §2–6](active/roadmap/01_SCOPE_AND_PHASES.md) | M0–M8状态、能力缺口、依赖及交付，未完成的长期目标不被局部研究隐藏 |
| 像多模态、恶意信息一样，还有哪些大方向被忽略？ | [VISION §23](reference/VISION_FUTURE_TECHNOLOGY.md#23-能力与风险版图补全被局部主线遮蔽的八个大方向)、[01 §4.1](active/roadmap/01_SCOPE_AND_PHASES.md#41-能力与风险版图补漏20260918用户澄清) | 常识、抽象推理、空间物理、社会认知、目标对齐、正常数据风险、创造/生成、多语言；区分未展开与已有原则，给方案及归属 |
| 下一段开发具体要做什么？ | [03 §5](active/roadmap/03_CURRENT_EXECUTION.md)、[01 §6.4–6.5](active/roadmap/01_SCOPE_AND_PHASES.md) | 活动包与局部准入看03；转段条件及数据/模型/集成/评价交付看01，不把整条路线当一次授权 |
| 现有设计受限时有哪些高上限选择？ | [VISION §15–19](reference/VISION_FUTURE_TECHNOLOGY.md) | 五类备选及推荐双通道组合；具体计算/训练、取舍与否决，不能按模块名称或实现难度排名 |
| 七项项目缺口怎样实际解决？ | [VISION §19](reference/VISION_FUTURE_TECHNOLOGY.md#19-项目缺口的实际解决方案推荐核心学习配方演进与交付) | 核心/数据/学习/反馈/消费者/交付逐项映射，依赖与当前R2优先项明确，不自动开多条主线 |
| 多模态怎样形成同一个模型？ | [VISION §20](reference/VISION_FUTURE_TECHNOLOGY.md#20-多模态共同认知专用编码共享事件跨模态学习与验收) | 专用编码、共享对象/事件、区域/时间引用、对齐课程与跨模态干预；不以OCR/ASR或答案拼接代替理解 |
| 自主学习遇到恶意信息怎么办？ | [VISION §21](reference/VISION_FUTURE_TECHNOLOGY.md#21-自主学习的信息治理不可信输入分层准入抗污染与恢复)、[02 §2.6](active/roadmap/02_GATES_AND_CI.md) | 观察/相信/学习/行动分权，来源继承、候选隔离、快慢准入、污染撤销与回退；不承诺过滤绝对安全 |
| 长期运行还有哪些机制缺口？ | [VISION §22](reference/VISION_FUTURE_TECHNOLOGY.md#22-长期运行的补全设计不确定性信念修订记忆预算技能与持续评价) | 何时求证、冲突/历史信念、有限记忆、技能与目标、私有状态隔离、自主调度和更新/恢复；同一压力序列检验共同收益 |
| 怎样证明越用越会，而不是测试时偷学答案？ | [07 §4.4](active/roadmap/07_MINI_MODEL_DELIVERY.md#44-冻结能力与在线学习分账20260918设计补充)、[VISION §22.8](reference/VISION_FUTURE_TECHNOLOGY.md#228-长期评价冻结能力和在线学习必须分账) | 固定能力测试保持冻结；在线流先记录预测再释放合法反馈，只评价后续适应，另测旧能力与资源成本 |
| 什么算完成，何时给用户模型？ | [02 §2–3](active/roadmap/02_GATES_AND_CI.md)、[07](active/roadmap/07_MINI_MODEL_DELIVERY.md) | 规划/实验/能力/阶段/采用分别记账，L3触发后置用户验收 |
| 如何从研究runner接成一个模型？ | [07 §5.2](active/roadmap/07_MINI_MODEL_DELIVERY.md)、[VISION §17.7](reference/VISION_FUTURE_TECHNOLOGY.md) | 权重/编码身份、会话映射、真实消费者、恢复与同bundle评价；候选接入不等于默认采用 |

先看目标与缺口，再确定交付和选择设计，不从算法名字倒推项目必须做什么。本轮仅完善这些规划；项目研发的当前范围和停止条件沿用03与冻结合同，不因文档更新撤销旧成果或自动放行新训练。

## 文档职责与证据入口

| 文档 | 职责 |
|---|---|
| [根需求](active/TAIJI_CORE_REQUIREMENTS.md) | CR-1–CR-10长期目标；成熟能力与原型、方法选择分开 |
| [架构与规则](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | §0.2四层规则；§4.6–4.7统一任务/行动合同；§6.3–6.4学习与恢复；§8.5规则实施清单 |
| [01 总阶段地图](active/roadmap/01_SCOPE_AND_PHASES.md) | M0–M8、长期目标与能力轴缺口 |
| [02 晋级与发布](active/roadmap/02_GATES_AND_CI.md) | 实验、轴晋级、阶段退出、发布各自门槛 |
| [03 详细推进方案](active/roadmap/03_CURRENT_EXECUTION.md) | 当前规划模式、R2能力路径、开发包输入/交付/验证/退出，以及唯一活动入口 |
| [R2设计依据复核](reference/M5_DISTILLATION_TOMBSTONE.md) | 课程与六seed结果、固定回答/重复样本风险、归因收紧与证据缺口 |
| [R0证据与门禁审计](reference/M5_DISTILLATION_TOMBSTONE.md) | P3b终态、P5.2d v2逐门可用性与下一版仪器修正边界 |
| [CAP-0旧权重加载器决策与落地](reference/CAP0_LEGACY_LOADER_DECISION_BRIEF_20260915.md) | v8/v9检查点被判"格式不支持"的根因、M2-2i迁移落地及**行为未变**证明（逐维同分＋100题输出逐字节相同）；只动恢复分支的 legacy 一侧，镜像分支与迁移后校验未放宽 |
| [P3b双臂结项与二次战役决策](reference/M5_DISTILLATION_TOMBSTONE.md) | 双臂逐tick读数、"不可判"的定量理由、A′成对未见集NLL的实测成本与分辨率；是否再开战役是用户机时授权三选一 |
| [R2语言目标与信用分配设计](reference/M5_R2_LANGUAGE_TARGET_CREDIT_DESIGN_20260916.md) | 历史语言目标与信用假设、实现来源；不覆盖后继证据或03执行裁决 |
| [R2-H3.5分层回答计划合同](reference/M5_DISTILLATION_TOMBSTONE.md) | 回答计划与byte渲染分层、v3数据课程、预算、恢复、消融和最小可证伪对照 |
| [R2-H3.5-A候选预注册](reference/M5_DISTILLATION_TOMBSTONE.md) | 32维plan、监督编码、运行时oracle禁令、checkpoint前置和smoke出口 |
| [R2-H3.5-A matched预注册](reference/M5_DISTILLATION_TOMBSTONE.md) | v3控制集、三seed对照、plan消融、final延迟权限和停止门 |
| [07 整模型验收](active/roadmap/07_MINI_MODEL_DELIVERY.md) | 真实输出评价、L3触发最小用户版本 |
| [05 技术债](active/roadmap/05_TECH_DEBT_REGISTER.md) | 隔离、仪器、恢复与CI；按证据范围及明确后继修订解释 |
| [唯一完整VISION](reference/VISION_FUTURE_TECHNOLOGY.md) | 大方向查漏先读§23；§15–19架构/学习方案；§20多模态；§21信息治理；§22长期运行机制 |
| [06 历史决策](active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md) | 旧C/A/B路线依据，不是当前执行授权 |
| [仓库可清理内容评估 2026-09-27（v2）](reference/DISTILLATION_TOMBSTONE_20261007.md) | 只读扫描按重建代价切三档：零重建 1.86G / CPU+网络 7.3G / GPU | 机时 52G（模型检查点+实验产物统一裁决，含不可删锚点）；git 历史无需瘦身的否定结论 |
| [旧执行计划快照](archive/history/EXECUTION_BEFORE_RULE_REPAIR_20260917.md) | 原逐轮台账与当时判断，保留追溯；全部旧“下一步”失效 |
| [R2逐轮诊断快照](archive/history/R2_DIAGNOSTIC_QUEUE_BEFORE_DESIGN_OPTIONS_20260917.md) | 七次后继追加的旧队首与当时解释；当前结论由设计依据复核限定 |

## 推进原则

统一能力主张以[认知闭环验收原则](active/roadmap/02_GATES_AND_CI.md#21-统一认知闭环验收原则20260917用户确认)为准：同一模型、同一真实任务，记忆/世界模型/规划的联合行为贡献；原生语言与行动能力分账，不以模块或界面代替。本条不覆盖具体包的执行/停止状态。

1. 本轮只完善开发指导，仓库研究主线保持R2，最新阶段裁决与唯一入口见03。必要局部实验按[02 §2.5](active/roadmap/02_GATES_AND_CI.md)有条件准入：明确判别价值、对照、累计预算及停止/返回动作；已授权且未触发停止线可按约定推进，不逐次全架构审批。当前已停止配置不会因本轮规则修订自动重开，A–E也不依次试遍。
2. 最小任务族属于方法而非主线：直接服务R2判别的小型文本实验可在批准包内使用；需要新Workbench环境或新增行动/适应能力的方案保留VISION，未批准不执行、不成为R2前置。讨论设计不自动改变执行顺序。
3. 基本对话、同bundle代表能力、安全/恢复/性能达到L3才交用户自由验收；不立即赶演示，也不等待完整VISION/M8。
4. M5退出仍须知识、协作/选择、在线/身体与共同门齐备；排除项显式批准。
5. 历史失败保留，新结果不追改旧阈值。新训练、核心合同、默认行为与资源投入需相应批准；训练前必须先通过checkpoint保存/恢复前置检查。
6. 不可妥协边界、能力合同、工程合同、实验选择分层管理；规则写明目的、范围、实现状态和验证方式。允许版本化替换方法，不事后放松确认门，不把设计文本当已实现能力。

9月16日查询的最新远端CI run34869725409总体失败（3.10腿），对应d09dcc21而非当前HEAD。局部通过不等于远端全绿。
