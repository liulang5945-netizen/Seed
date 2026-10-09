# plans/reference 索引（自动生成 2026-10-07，蒸馏收束后 49 件＋N 系列 30 件）

留存政策（owner 2026-10-07）＝核心设计与相关内容留档参考；实验过程文档蒸馏后删除（已删 A 支线 13 件＋M5 族 158 件＋第二批 29 件，墓碑见 [M5_DISTILLATION_TOMBSTONE.md](M5_DISTILLATION_TOMBSTONE.md) 与 [DISTILLATION_TOMBSTONE_20261007.md](DISTILLATION_TOMBSTONE_20261007.md)，git 历史可回溯）。项目收束见 [M0_M7_PROJECT_CONSOLIDATION_20261006](M0_M7_PROJECT_CONSOLIDATION_20261006.md)，下一轮主线见 [../active/roadmap/09_NEXT_MAINLINE_PLAN.md](../active/roadmap/09_NEXT_MAINLINE_PLAN.md)。

## M7 当前（CI/发布面，活跃）

- [M7_RELEASE_SURFACE_READINESS_20261006](M7_RELEASE_SURFACE_READINESS_20261006.md)
- [M7_SCOPE_SHIFT_PREREG_20261006](M7_SCOPE_SHIFT_PREREG_20261006.md)

## M6 收官与产品线

- [M6_EXIT_READINESS_REVIEW_20261003](M6_EXIT_READINESS_REVIEW_20261003.md)
- [M6_FRONTEND_DSH_WORKSPACE_DOCK_PLAN_20260921](M6_FRONTEND_DSH_WORKSPACE_DOCK_PLAN_20260921.md)
- [M6_SCOPE_EXCLUSION_APPROVAL_20261003](M6_SCOPE_EXCLUSION_APPROVAL_20261003.md)
- [M6_SCOPE_EXCLUSION_APPROVAL_20261006](M6_SCOPE_EXCLUSION_APPROVAL_20261006.md)
- [SPEC-M6-01_replay_injection_fold_prereg_20261001](SPEC-M6-01_replay_injection_fold_prereg_20261001.md)
- [TAIJI_G5_READINESS_20260926](TAIJI_G5_READINESS_20260926.md)
- [UI_LIFE_UX_SPEC_20261003](UI_LIFE_UX_SPEC_20261003.md)
- [UI_LIFE_VISUAL_DIRECTION_20261003](UI_LIFE_VISUAL_DIRECTION_20261003.md)

## N 主线与项目收束

- [PLAN-N1-00_s5_endpoint_falsification_prereg_20261007](PLAN-N1-00_s5_endpoint_falsification_prereg_20261007.md)（N1 第一步，S5 便宜证伪预注册，判据先冻；owner 2026-10-07 弹窗批后同日开跑）
- [PLAN-N1-00_S5_ADJUDICATION_20261007](PLAN-N1-00_S5_ADJUDICATION_20261007.md)（S5 判读：J1 不成立 ⇒ 病在更深，P1 降级为部分成立；S1 顺延重审）
- [PLAN-N1-02_attribution_prereg_20261007](PLAN-N1-02_attribution_prereg_20261007.md)（N1 基底归因预注册：完整计算图＋参数级预读＋F1–F5 与三带判别规则先冻；无 owner 新决策点，零训练只读诊断）
- [PLAN-N1-02_ATTRIBUTION_ADJUDICATION_20261007](PLAN-N1-02_ATTRIBUTION_ADJUDICATION_20261007.md)（归因判读：H-A 判 **not_primary**——序贯信号在场且每步 54.9% 被跟随；损失面＝逐步位置保真（45.6% 步顶质量非后继）；S1 设计约束输出，PLAN-N1-01 重开前置达成）
- [PLAN-N1-01_s1_readout_prereg_20261007](PLAN-N1-01_s1_readout_prereg_20261007.md)（S1 实施预注册：H-S1a 硬序贯位置掩码（零训练可证伪）与 H-S1b 再激活重放（要训练）两假设分工；主判据 J-S1a＝never-LF 拖写行 ≤118 先冻、守卫四条、预算 3 支面；**不开跑，待 owner 批**）
- [PLAN-N1-01_ADJUDICATION_20261008](PLAN-N1-01_ADJUDICATION_20261008.md)（S1a 阶段0 判读：J-S1a **不成立**（225＞118），但 F1 0.549→0.747、F3 ×20.8 ⇒ 走第三出口"读取可修、但不充分"；守卫四条＋反向守卫全过；出口文本含糊缺陷登记 DEBT-G46）
- [PLAN-N2-01_consolidation_powerup_prereg_20261007](PLAN-N2-01_consolidation_powerup_prereg_20261007.md)（N2 巩固通电预注册：C6 六决策点现状核对（①②④⑤已在库，⑥面板行**已核＝缺**，落点三段已点名，**真未做＝organs/learn 从未在真实长跑启用**）；合取判据 J-N2a 旧能力保持 ∧ J-N2b 新收益 vs 不学对照；守卫含**回退路径必须真演示一次**；改权重半径待 owner 批）
- [PLAN-N2-01_ADJUDICATION_20261008](PLAN-N2-01_ADJUDICATION_20261008.md)（N2 第一次通电判读：加速点① 在最小剂量档判**不成立**——CAP-0 严格命中 9→12 过、复述面 2.0 档成句 25→23 未过、J-N2b 治疗 −14.4628 ≤ 对照 −14.2791；G-N2-1 回退**双重实证**通过；当场撞出 **DEBT-G47 巩固产物装不回来**；⑥ 面板行一并落地）
- [PLAN-N3-03_tau_definition_20261008](PLAN-N3-03_tau_definition_20261008.md)（N3 乙前置定义件：τ 是"这张面、这个装配、这条加权和"的阈值；实测五维里两维在今天训练链上结构性恒零（`fast_slow_conflict` 挂载即 `fast_is_zero`、`activity_saturation` 需 gate>0），故默认 `minimum_pressure=0.70` 在退化复合量上算术不可达（上界 0.425）；装配三选一（含"为什么不选降 τ"的出处）＋一条硬事实：学习模式不入档，每次 load 后必须重施加；本件不冻数值、不开跑）
- [PLAN-N3-04_developmental_assembly_prereg_20261008](PLAN-N3-04_developmental_assembly_prereg_20261008.md)（N3 乙步骤二预注册：判据先冻不开跑。开头先更正 owner 弹窗里我方给错的选项形状——丙单独只解 fast_slow_conflict，activity_saturation 要 bridge gate>0 才有活动可饱和（adaptive_residual_bridge.py:101-107 与 :145 的提前退出），故冻成丙 与 丙＋乙 两臂等 owner 点；主判据＝在场性（该维整场非零）、次判据＝现取 τ 且必须有动态范围证明；面内自述五列硬要求，其中 mode_reapplied_after_load 把不入档那条变成机检；预算按 ㊵-491 标定实测外推。**2026-10-08 仪器已落地（㊵-496）并撞到一条改命令形状的产品前置**：`migrate_f1_to_developmental_synapses()` 与 `readout_utf8_position_input` 互斥（`taiji/model.py:1124-1135` 响亮拒绝，两者同开＝静默空转）⇒ 五支面必须带 `--no-readout-position`，读数限定为"放行装配＋位置输入关闭下的数"（§5ter）；烟测（只验仪器）见两维都活、`pressure` max 0.711851 ⇒ ㊵-484③ 的 0.425 上限不再成立，但 `should_propose` 全 0 ⇒ 触发不只由 pressure 决定；契约测 15 件入库。**同日 §5ter 再更正一次（㊵-497）：我先前那句"带上 `--no-readout-position` 就行"不成立**——两条在训链的权重带 `position_weight`，带着该旗标被 `taiji/organs.py:925` 拒、不带则被 `taiji/model.py:1124-1135` 拒 ⇒ **冻结的链进不了这档装配**（判据／守卫／面数未改，只有热启动源作废）；可执行替代＝出厂链 `seed_beta.pt`（不带位置列，rc=0 实证、4,000 符号外推每支 ≈46 秒），换链换的是分布 ⇒ 待 owner 点）

- [PLAN-N3-04_ADJUDICATION_20261008](PLAN-N3-04_ADJUDICATION_20261008.md)（乙步骤二 5 支面判读：**在场性成立、阈值不成立**——`fast_slow_conflict` zero_share 0.00025、`activity_saturation` 0.0（四张覆盖率 1.0）⇒ ㊵-484③ 的"两维整场恒零／上限 0.425 够不着 0.70"被推翻（`pressure` max 0.7016~0.7092），但 `should_propose` 总数 0、最长连续段 0 ⇒ 无动态范围，**τ 不得发表**（规则值 0.70 恰等默认阈）；口径否证支 beta 两支触发、circuit 两支缺同链控制臂（面数缺口）；同链两档 gate 差极小而链间差大；我自己的 G-N3b-1 与 G-N3b-2 在同一臂上互斥，登记 DEBT-G51 并并列写出两种读法）
- [PLAN-N3-05_holdout_probe_rebuild_prereg_20261008](PLAN-N3-05_holdout_probe_rebuild_prereg_20261008.md)（DEBT-G49 "零命中重建"预注册，判据先冻不开跑：验收式 A-1..A-4（复用 `_disjointness`、反例已知 5/35、字节数落同档、形状同类、同机两趟都 0）＋预先指定的选择规则（无一通过就判"造不出"，不许放宽凑过）；形状＝**加新列 `holdout_surprise_v2`、旧列与旧常量不动**（保住历史读数的复算入口）＋件里自带两把重合度尺；守卫含"打分器只读"；实施回 owner 批）
- [PLAN-N2-02_second_powerup_dosewindow_prereg_20261008](PLAN-N2-02_second_powerup_dosewindow_prereg_20261008.md)（N2 第二次通电预注册：owner 批第二次改权重并指定同时修读数窗口。两件必须另立的原因写进 §0——①第一次 J-N2b 用整篇打分而剂量只有 64 符号（不同窗⇒分子天然被稀释），②保持侧恰好跌破 1 项时旧判据无定义。本件把保持侧写成单值式（任一列跌幅 ≥1 即判有代价，并预先指定拖写行数越少越好），新收益改成剂量同窗的 Δquality>0 ∧ Δaccuracy≥0（一正一负记 not_resolved，不许挑尺子）；新守卫 G-N2-5 产物可载双向（睡过的档要能装回来、没睡过的档摘要不得被改动），G-N2-6 明令禁止再拿对齐件代答装载性。剂量与第一次同档，预算 ≈45-60 分钟本机 CPU）
- [PLAN-N2-02_ADJUDICATION_20261008](PLAN-N2-02_ADJUDICATION_20261008.md)（N2 第二次通电判读：三条合取＝**J-持久化 成立／J-N2a' 不成立／J-N2b' 不成立**。正面交付＝原生候选档可被 `SeedRuntime.load` 读回且摘要逐位同，反向那半（未睡的母档）同绿 ⇒ **DEBT-G47 在真实产品链上结清**；保持侧 14 列里 7 列跌（CAP-0 E 1→0、复述命中两档各 −1、成句四档 −6..−11），主列 56→47 按预先指定方向记不跌破；新收益窗内 −1.1086 ≤ 0 判不成立，整篇旁证 +0.2047 与之反号故只披露不判级。另记两条：回退面与巩固前读数逐位同＝"同一份权重跑两遍 CAP-0 逐位同"的第二次实例（⇒ 1 项粒度不是噪声），以及 run-1 的 after 建立在对齐件上、与 run-2 不同源不可互代）
- [PLAN-N2-03_reset_attribution_prereg_20261008](PLAN-N2-03_reset_attribution_prereg_20261008.md)（N2 归因实验预注册：**零改权重**的单变量分离——Arm R＝装载母档后**只调一次** `reset_dynamics(episode_id="wake-after-sleep")` 再存盘，恰好复现 run-2 两族变化里的"情节/工作记忆被清空"那一族而**参数族严格为零**；J-N6-归因三档互斥（`attrib_R≥0.5` 的列数 ≥5 判收束致损／2..4 判 `partially_resolved`／≤1 判权重致损），主列不入归因分数只作旁证；§3ter 记一处自己的口径更正——把 `content_digest(checkpoint())` 当"权重摘要"导致首跑红，G-N6-1 改按参数族零差并要把清掉的情节态叶子如实自述）
- [PLAN-N2-03_ADJUDICATION_20261008](PLAN-N2-03_ADJUDICATION_20261008.md)（N2 归因判读：判 **`cost_from_weight_update`**——run-2 跌破的 7 列在收束臂上**一列都没复现**（`attrib_R` 全为 `−0.0`，臂档读数与巩固前逐列相等：E 1/严格命中 6,6/成句 24,24,24,25），`n_reproduced=0/7`、无列被剔除；四条守卫全绿（参数族 changed/dropped/added 全 0、臂档可载、三面同题集、取数函数从 run-1/run-2 判读器导入）；主列旁证 56→57 同样没复现 run-2 的 47。顺量到一条比归因更通用的事实：**这七列读数不消费情节/工作记忆那两族状态** ⇒ 以后"掉一格"不许再用"刚收束把态清了"解释。去向＝保持集照 §8.3（要改权重，回 owner），且**取消一条反对 DEBT-G47 修法甲的论据**但不重开该裁）
- [PLAN-N3-06_gate_attribution_replay_prereg_20261008](PLAN-N3-06_gate_attribution_replay_prereg_20261008.md)（N3 乙"六道分项闸里哪一道在拦"预注册：**零算力、零改产品码、零新面**——把在库压强面的观测用 `AdaptiveResidualGrowthPressure.create` 重构后喂回产品自己那支 `AdaptiveResidualGrowthTrigger`，逐行核对重放 `should_propose`（A-1）与 `pressure_digest`（A-2）才出版；J-N3c 三档互斥（恰一道恒假＝指认它／≥2 道＝"合取不能唯一指认"／六道都有真步而触发为零＝转持续性侧）。§0 先记一条口径更正：闸比的是 `pressure_ema`（五个 EMA 的加权和 :388-395），面里记的是原始加权和 ⇒ ㊵-499 那句"max 0.7092 越过 0.70"是原始口径 ⇒ 入册 DEBT-G53）
- [PLAN-N3-06_ADJUDICATION_20261008](PLAN-N3-06_ADJUDICATION_20261008.md)（该件判读：四张面一致判 **`single_gate_blocking`**＝合成量那道闸（阈 0.70 对 EMA 整场上界 **0.67813／0.679165／0.685877／0.680911**，`meets_pressure` 真步数 **0/3,999**），其余五道**没有一道恒假**（最紧的 `activity_saturation_ema` 也有 50–104 步真）；锚点四张面全对（A-1 与 A-2 各 3,999 行 **0 失配**、A-3 反例支实跑 rc=2"199 条低于样本下限 500"）。反事实块（跑前未冻、已在 §3bis 自报）＝摘掉合成量后五道同时为真的最长段 8–12 步 ≥ `required_pressure_steps=3` ⇒ **持续性不是拦阻者**；同一冻结规则套到 EMA 口径给 beta 链 **0.65**（六道合取 36/34 步真、最长 6 ⇒ 会触发）、给 circuit 链 **0.70**（0 步 ⇒ 不会）⇒ τ 是链上的量、跨链不可搬，且 `would_have_triggered` 只在首次触发点之前有效）
- [PLAN-N3-07_tau_ema_caliber_definition_20261008](PLAN-N3-07_tau_ema_caliber_definition_20261008.md)（N3 乙 τ 定义**口径升版**，零算力、不冻数、不动常量：τ 是 `pressure_ema` 的阈值而非原始加压量的阈值，τ 的完整名字加第六元"口径"＝（设备,链路,checkpoint,装配,生成预算,**口径**）。含一条**恒等式**（权重为常量且四支 EMA 初值 0 ⇒ Σwᵢ·EMAᵢ ＝ EMA(Σwᵢxᵢ)，所以旧面可重放出 EMA 口径、但**不许**把原始 p90 当 EMA p90）、一条 **warmup 显式处置**要求（`ema_rate=0.25` ⇒ 离渐近值 1% 内需 ≈16 步，推导非实测；且连续段不被 `reset_dynamics` 清）、一条**规则粒度缺陷**（两链 EMA p90 集合不重叠但间距只有 0.007085／最大跨链差 0.009236，而格宽 0.05 是它的 5.4–7.1 倍 ⇒ 0.65/0.70 的分岔是取整方向，不是链差的量度；据此同日更正 PLAN-N3-06 判读 §5.2）。旧件 PLAN-N3-03 不删不改写，冲突以本件为准）
- [PLAN-N3-08_face_self_report_prereg_20261008](PLAN-N3-08_face_self_report_prereg_20261008.md)（DEBT-G53 修法② 预注册，判据先冻不开跑：把**六道 `minimum_*`＋`ema_rate`＋五支 EMA 初值**写进 face 头、把 decision 自带的十个键（五支 EMA＋`pressure_ema`＋连续段＋预算＋`reasons`）写进每一行观测，格式升 v2 且读写两侧同时认 v1/v2。主判据 J-N3c-自述单值式＝重放仪在新面上的 `assumed_from_product_defaults` 长度必须 **== 0**（今天四张面都是 6）且 `threshold_self_reported_in_face` 为真的道数必须 **== 6**（今天是 1）；兼容锚＝一枚在库 v1 面改前改后判读件**逐位不变**（`cmp`）；守卫五条含"零抄写"（写侧不得出现阈值字面量，插一个写死值必须让测红）、"版本集机检一致"、"缺披露即 rc=2 而不是回落产品默认"——**回落会静默重建本件要消灭的那个假设**。跑面那一档（4 在线面＋1 守卫臂）会更新权重 ⇒ 另批 owner。**同日第一档已实施并判读（㊵-508）**：两条判据都成立——新 v2 面 `assumed_from_product_defaults` 归 **0** 且自述 **6/6**（v1 是 6 缺／1 报），四张在库 v1 面判读件逐位不变（19,154 B／sha256 前缀 `2ee018515e01d9c0`），契约测 7→**12 passed**（含"插一个写死阈值必须被抓到"的反例支），烟测面副读数用新自述面**独立复现了 ㊵-484 的"两维恒零"**）
- [PLAN-N3-10_arm_b_shape_prereg_20261008](PLAN-N3-10_arm_b_shape_prereg_20261008.md)（N3 甲乙臂形状＝DEBT-G52 的落地件，owner 裁"先定形状再两臂一起跑"；零算力、不跑。**冻形状时读出一条比"没写实现"更硬的东西**：暴露量 `S ≈ U（唯一篇数）·L（每篇平均符号数）·R（每篇被访问次数）` ⇒ "符号数同 ＋ 数据不扩（U 变）"必然让 `R` 跟着变，所以原先记的两条"天然实现"（截语料⇒S 不等／抬 epochs⇒R 成第二自变量）**不是两个可选坑，是同一约束的两种违约**。冻死的是可执行形状 B＝`--max-unique-documents K`（篇池循环重用，默认关逐位不变）＋面内自述三轴 `unique_documents`/`document_visits`/`mean_revisits`（缺任一即 rc=2、不许由命令行反推）＋一条能为假的机器判据（**两臂 `mean_revisits` 相等＝U 没变＝仪器坏，响亮失败**）＋读数边界 `interpretation_limit`（B 只答"等符号暴露下容量是否抬出噪声带"，**不许**答"封顶在容量还是数据"）。更干净的 C＝等数据量变参数量（三轴全对齐）登记为**待 owner 升版重冻**那一格，本件不替 owner 选；锚点逐条核过（取篇处只有 `train_seed_corpus.py:103-128`、外层 `:606`、预算支 `:677-679`、身份 `:344`，且明写不动 `:131-175` 的 answer 并列支））
- [PLAN-N3-09_ADJUDICATION_20261008](PLAN-N3-09_ADJUDICATION_20261008.md)（该件判读：四张治疗面全部 **`freezable`**（β 0.65／circuit 0.66；合取真步数与最长段 36/6、34/6、33/7、16/**3**），**J-N3e-跨链分辨力成立**；核心锚点＝每行"六道合取"与该行 `decision_reasons` 对表 **5×3,999 行 0 失配** ⇒ 面内重算与产品自述同源。升版的全部理由在"旧 0.05 格对照那一列"：circuit 两档旧格给 **0.70** 而该面合成量上界只有 0.685877／0.680911 ⇒ 旧规则冻的是全场不可达的阈；**β 链新旧同值 0.65 ⇒ 升版对 β 是中性改动**（如实报）。两处限定已点名：`circuit_gate100_v2` 最长段**恰好 == 3**（边界通过）、`guard_off_v2` 的 `five_gate_window_composite_ceiling` 为 **None** ⇒ 默认关装配上 τ 定成任何数都不触发＝"出厂件零进化"的算术版且件内自带证据。**守卫臂那格判出的是本预注册自己的缺陷**：§3 写的臂形状（不带 `--no-readout-position`）与 §5 要对标的 ㊵-498 基线（带）不同源 ⇒ G-N3e-2 按字面不可能为真，记 `not_established` 并入册 **DEBT-G56**，remedy＝补跑一支带该旗标的默认关臂（≈46 秒，在原批文数量内）；顺带买到一条通用事实：**位置输入的开与关会产生两张不同的面**（同 seed 同预算下 3 个字段每行都不同、`online_accuracy` 0.21280→0.25031）。新测 7 passed（四条拒判支＋判级两支各自可达）；台账 08 ㊵-510）
- [PLAN-N3-09_tau_rule_upgrade_prereg_20261008](PLAN-N3-09_tau_rule_upgrade_prereg_20261008.md)（N3 乙 τ **冻结规则升版**预注册，owner 第五次弹窗裁"先升规则再跑 v2 面"：四处逐字冻死——①母量改 `pressure_ema`（v2 面自带的 `decision_pressure`，不许再用原始 `pressure`）；②warmup 先剔 `K=17` 条（`log(0.01)/log(0.75)≈16.0078` 上取整，推导非实测）并自述 `warmup_dropped`/`warmup_share`；③格宽 **0.05→0.01**（两链 p90 间距只有 0.007085–0.009236 ⇒ 0.05 格会吞掉链差并把 circuit 链送到其可触发上界 0.685877 之上）；④**动态范围由"先测的流程要求"升成硬验收**——候选 τ 必须让六道合取在同面有 ≥`required_pressure_steps` 的连续段，否则判 `not_freezable_at_this_grid` 并出版"必须 ≤ X"那条上界 ⇒ 新规则下"冻出一个够不到的阈"不可能再被报成成功。另配会为自己判假的 **J-N3e-跨链分辨力**（两链候选值若仍相同即判"0.01 格仍粗"、本件主张被否证，不许当场换统计量）。跑面范围已点名＝4 张 v2 面＋1 支默认关守卫臂、出厂链、4,000 符号）
- [PLAN-N2-04_retention_set_prereg_20261008](PLAN-N2-04_retention_set_prereg_20261008.md)（N2 第三件改权重实验预注册＝保持集，owner 裁"先出预注册、判据先冻不开跑"。**现状核对是一条新的结构缺口**：按词扫 `L_keep`/`keep_set`/`retention set` 在 `taiji/**`、`seed_platform/**`、`seed/**` 里**命中 0** ⇒ "保持集"在产品里不是对象、VISION §8.3 那句是设计意图；现成最接近的 `taiji/structural_lineage.py:41-63` 保的是**结构血统不是行为读数** ⇒ 队首决策＝保持切片住哪一层（甲＝进 `sleep_pass` 的 `spec.datasets`，能改代价本身但要产品码批文／乙＝只在巩固前后对表，零产品码改动只买"知道跌没跌"）。保持集成员＝run-2 那七列所属两张面原样复用（CAP-0 E 维＋复述 ×24 的命中两档与成句四档），分离条件机检化为"两侧材料 `overlap == 0`，取不到即整件不判"；判据两条合取（J-N2c-保持＝七列逐列跌幅 == 0／J-N2c-收益＝沿用 N2-02 剂量同窗式，一正一负记 `not_resolved`）；守卫六条含"改权重批文与产品码批文两条不互相代答")
- [PLAN-N3-01_r4_hooks_prereg_20261007](PLAN-N3-01_r4_hooks_prereg_20261007.md)（N3 乙：R4 生长协议接钩子——**压强阈先在线面冻结**；trainer 按词 grep 命中 0 ⇒ **训练面上今天没有压强读数**，但**生产者已在产品路径**（`Seed._record_adaptive_residual_growth_pressure`，`taiji/model.py:1069-1106`；bridge/trigger 未挂载时返回 None）⇒ 第一步＝**只读挂载＋记录 pressure**（`gate` 留默认 0.0＝不放行生长；**这仍是一次会更新权重的普通训练跑**，"零改权重"的说法已于 ㊵-483 在件内更正）；J-N3b 五面齐备＋同容量对照。**2026-10-08 步骤一已跑：否证支触发 ⇒ 阈值不冻，并测出默认阈在本链算术不可达（㊵-484）**）
- [PLAN-N3-02_scaling_probe_prereg_20261007](PLAN-N3-02_scaling_probe_prereg_20261007.md)（N3 甲：参数×10 单点——先用[在库件复算仪](../../scripts/training/audit_taiji_n3_plateau_baseline.py)把"平台"钉成锚点（**两档斜率均在自身噪声带内、holdout 反而变差**），再谈算力；×10 两臂（数据扩/不扩）判据与守卫已冻，**算力从未单独批过 ⇒ 不起跑**）
- [PLAN-N4-01_addressing_measurement_prereg_20261008](PLAN-N4-01_addressing_measurement_prereg_20261008.md)（N4 首件＝**测量口径预注册，判据先冻、零跑**。落盘前四条代码事实把 N4 自己的定价依据否证了（＝**DEBT-G58**）：①`EpisodicMemoryStore.retrieve`（`taiji/episodic_memory.py:60-89`）对非空库**无条件返回前 `limit` 条**（只在 `limit <= 0` 或库空时给 `()`）⇒ "寻址失败率"作为零命中占比**恒等于**"空库占比"，09:105 那句"空库/寻址失败"把一件事写成了两件事；②产品侧唯一的"变空"通道是可见性过滤 `_recovery_memory_is_readable`（定义 `taiji/adapter.py:10838`，用处 `:10950-10957`）⇒ 真正可分的是四态 `store_absent`／`store_empty`／`all_hits_gated`／`emitted`，而今天**四态塌成同一个 `()`、零计数零原因键**；③`attach_episodic_memory` 的非测试调用点全仓 **5 处且全在 `scripts/training/eval_taiji_p{3,4,6}*.py`**，`api/**` 与产品默认装配零挂载 ⇒ 默认形态下 `episodic_ids` 恒空；④最接近的仪器 `eval_taiji_p4_episodic_recall.py:60-71` 用 `torch.eye` 造线索，答的是"库与读路通不通"，**不答**"相似经历互相干扰多深"。⇒ 母量冻成唯一一条 `wrong_top1_rate`（top-1 取错记录数 ÷ 带真值配对的查询数，分母由仪器现数）＋`true_rank_p1_rate`，噪声带沿用 PLAN-N3-02 §0 的自取形状；判据三条（J-N4-1 尺有动态范围／J-N4-2 稀疏编码收益 > 噪声带且 p1 不降／J-N4-3 保持四格逐格单报），**出口写成三条互斥且数值钉死**（DEBT-G46 修法的落地），第三条明令"尺没有动态范围就整件不判、不许当场换母量续命"；守卫四条各能为假，G-N4-3 要求挂载态由面自述、凡引用产品主张必须点名"挂载链"。§4 的基线现算**要跑一次只读面 ⇒ 回 owner 批**，耗时与件尺寸本件未测、不引外推；对 N1 的依赖未满足（J-S1a 不成立）已写明。措辞门自扫 rc=0、`criterion_lines=13`、含糊用词 0 处；台账 08 ㊵-524。**〔2026-10-08 同日升版〕本件 §2 的『噪声带』条款已由 [PLAN-N4-02](PLAN-N4-02_band_floor_prereg_20261008.md) 替代（带下界 `1/(n//5)`、`paired_min=100`、出口加第四条 `ruler_too_coarse`），本件其余条款继续有效、旧件不删**）
- [PLAN-N4-02_band_floor_prereg_20261008](PLAN-N4-02_band_floor_prereg_20261008.md)（N4 的**带下界与样本下限升版件**，只替代 PLAN-N4-01 §2 的"噪声带"那一条，其余条款继续有效、旧件不删。升版的理由全部来自仪器自己的 CLI 实跑：母量是**逐查询 0/1** 序列，段宽 `size = max(1, paired//5)` ⇒ 带的**下界**是 `1/(paired//5)`，实测四档 `paired=8/10/20/100` 对应下界 **1.0/0.5/0.25/0.05**，而实测带在 `paired>=10` 的三档都停在 **0.5**（因为冒烟夹具是"前半全对、后半全错"的构造序，段均值必出 `0→0.5→1` 台阶）⇒ `n=8` 那档 J-N4-2 要求"降幅 > 1.0"＝**算术上不可能成立**，这条在跑面之前就查出来了（DEBT-G59）。冻下来的新条款：①判读只用面内实测带，`noise_band_floor` 只用于"这次能不能分辨"的自检，不许拿下界冒充实测带；②**配对查询下限 `paired_min=100`**（`size=20` ⇒ 下界 0.05，正是本系列的分辨目标量级），不足即走新增的**出口④`ruler_too_coarse`**（互斥第四条，不判 J-N4-1 的否证、不许当场换统计量）；③**查询顺序由 `queries_sha256` 钉死**，比两臂的带必须同一个 sha，换顺序＝换尺；④守卫 **G-N4-5**：面缺 `paired_queries`/`block_size`/`noise_band_floor`/`noise_band_adjacent_block_max` 任一键 ⇒ rc=2（仪器已出版四键，两档测各钉一次）。§6 点出新缺口：满足 `paired>=100` 的带真值题集**还不存在**（现成材料只有 consolidated 三枚小文件＋16 条 workbench 记录），跑面仍回 owner、耗时未测不引外推。措辞门自扫 rc=0、`criterion_lines=11`、含糊用词 0 处；台账 08 ㊵-525／DEBT-G59）
- [PLAN-N5-01_four_layer_loop_design_prereg_20261009](PLAN-N5-01_four_layer_loop_design_prereg_20261009.md)（N5 **四层循环设计预注册**＝owner 第十次弹窗裁 (a) 的交付：§0 现读接线表、§1 六层接线设计（影子 from_parent_bridge→learn→验收→准入→WDL 首调→任务矩阵）、§2 判据 J-N5-1…5 先冻、§3 守卫 G-N5a…d、实施可自办跑面另批）
- [PLAN-N4-03_s4_sparse_coding_prereg_20261009](PLAN-N4-03_s4_sparse_coding_prereg_20261009.md)（N4 **S4 稀疏编码臂预注册**＝判据先冻不开跑：H-N4b 假设、WTA(RP(cue)) 编码冻结 32→256 top-16、面仪器 `--cue-encoding` 旗标、判据 J-S4-1…3、对照臂＝稠密基线 0.06 同材料 sha 钉死；§2b 披露 J-S4-2 退化为精确匹配冒烟）
- [PLAN-N4-04_product_default_mount_design_prereg_20261009](PLAN-N4-04_product_default_mount_design_prereg_20261009.md)（N4 **产品默认挂载设计预注册**：判据五条先冻、不开跑，动 `taiji/` 之前须先过反例探针）
- [PLAN-N5-02_shadow_utility_adjudication_prereg_20261009](PLAN-N5-02_shadow_utility_adjudication_prereg_20261009.md)（N5 **影子通电后的贡献验收**：六条判据先冻、含 shadow_inert 独立否证支，写于 G 跑读数之前）
- [PLAN-N5-03_ruler_usable_formula_prereg_20261009](PLAN-N5-03_ruler_usable_formula_prereg_20261009.md)（N5 **判据升版**：给 `ruler_usable` 补机械公式，接管 PLAN-N5-02 的 J-N5b-4 那半句）
- [PLAN-N5-04_retention_rebase_prereg_20261009](PLAN-N5-04_retention_rebase_prereg_20261009.md)（N5 **保持侧换底**：从头臂上 `before=0` 使 J-N5b-5 成恒真式 ⇒ 改成「同一基件续训、只动 `--n5-shadow-gate` 一个变量」的双臂，before 面复用 20261008 已入库件；判据/守卫/失败出口先冻）
- [PLAN-N5-05_position_input_wiring_prereg_20261009](PLAN-N5-05_position_input_wiring_prereg_20261009.md)（N5 **甲路线实施预注册**：把 `readout_utf8_position_input` 接进发育 F1 bank 学习链；owner 弹窗 #16 三择一裁的甲。判据先冻，`J-N5e-2` 要求做功证据——只删 `raise` 判 `not_wired`）
- [PLAN-N4-05_product_write_semantics_prereg_20261009](PLAN-N4-05_product_write_semantics_prereg_20261009.md)（N4 **产品写入路径已在、缺的是回合驱动**：五条判据先冻，含"两档 cue 不可比"与"不许拿训练侧口径冒充产品档"）
- [PLAN-N4-06_perturbation_strength_and_ruler_prereg_20261009](PLAN-N4-06_perturbation_strength_and_ruler_prereg_20261009.md)（N4 **扰动强度与"尺可用"判据先冻**：两档角度、`ruler_usable` 公式、方向性预注册；冻在跑之前）
- [PLAN-N4-07_angle_pair_upgrade_prereg_20261009](PLAN-N4-07_angle_pair_upgrade_prereg_20261009.md)（N4 **升版：只换角度对 (0.70,1.05)**，J-N4f-2/3 公式一字不改；旧篇与 `ruler_unusable` 旧裁定保留并标取代关系）
- [PLAN-N1-03_forced_predecessor_prereg_20261008](PLAN-N1-03_forced_predecessor_prereg_20261008.md)（N1 第三件＝**区分两条出口的强制前驱对照预注册，判据先冻、不开跑**。缘起是 PLAN-N1-01 判读件 §4 把"读取可修但不充分"（第三出口）与"收回'病在读取丢序'、转解码/推导层"（exit-2）**并列登记成不能同时成立**——那正是 DEBT-G46 的含糊出口本体，所以本件的任务不是再买一个缓解读数，而是把两条出口分开。可执行形状由四条代码锚点定死：`taiji/copy_circuit.py:285-292` 的后继位置式（`codes[i-1] == prev_byte`）、`:322-338` 的 `prev_byte=None ⇒ 允许位置 0`、探针 `probe_taiji_a30_stop_failure.py:1333` 把 `prev_byte` 作为**参数下发**（⇒ 强制档有落点而不动产品码）、`:152/:164/:27` 的 `emitted_is_argmax`/`boundary_is_argmax`/`argmax_mismatch_steps` 已在报（旁证不必新建）。两臂只差"前驱是模型自己发射的字节 vs 真值序列第 k−1 字节"，**现成的 `--oracle-selector` 是选择侧 oracle（`:1109-1407`），不许拿来代答路径侧**（A2 那颗真 oracle 只否证过选择侧：选对事件后位置随机仍 19–21/104）。母量两条必须同面出版＝`end_byte_arrival_rate`（现行 F3 口径）＋`step_position_fidelity`（现行 F1 口径）；**本件明令不用相邻段噪声带**（量级差 20 倍一档，带尺会把结论推给取法），判据三条互斥且绝对线钉死：`forced ≥ 0.90 ∧ self ≤ 0.20` ⇒ 推导层复合；`forced ≤ 0.20` ⇒ 训练层信号缺失；`0.20 < forced < 0.90` ⇒ **整件不判并升版加档**，不许当场换母量或改用缓解行数替任一支续命。守卫四条（单变量含"两侧都不开 oracle-selector"、题集/α/max_length/checkpoint sha 逐位同源、首字节 `prev_byte=None` 语义不许漂、回退计数两分母原样披露）。§4 的 A 档（探针加 `--force-true-predecessor`＋契约测）零算力可自办，B 档跑面回 owner。措辞门自扫 rc=0、`criterion_lines=14`、含糊用词 0 处；台账 08 ㊵-531）
- [PLAN-N3-12_j_n3a_anchor_upgrade_prereg_20261008](PLAN-N3-12_j_n3a_anchor_upgrade_prereg_20261008.md)（N3 甲判据**对照锚升版件**＝DEBT-G64 的修法①，判据先冻、不开跑、**签字前不生效**。动因是甲臂真件出数：acc 五段 `0.280905/0.31394/0.3183/0.33464/0.352`、斜率 **+0.071095 > 自取噪声带 0.033035**、`holdout_surprise_v2` 末段−首段 **−0.483725**（改善）⇒ 原 J-N3a 的第一项与"保"项都达线，只有第二项不达——因为它的锚 `0.594120+0.02` 取自**自答链/另一份语料**的 2M 档（PLAN-N3-02 §0），而甲臂吃 `dialogue_extended_clean.jsonl` ⇒ 差 −0.262 是**结构不可越**，照字面合取会把口径缺陷报成"容量不解除"并据此关掉 09 §4 的容量路线。升版只做一件事：**把锚搬到与臂同源的那条链**＝新增一支 `--scale 2`、同语料、同 seed、同 250,000 预算、同喂法的对照臂，门槛 +0.02 与合取形状逐字保留；旧锚降级为历史背景数（照旧出版、不参与判决）。守卫四条含 `G-N3g-1 两臂 corpus_fingerprint 必须相等否则 rc=2`（即 G64③ 要的显式披露）、`G-N3g-3 窗口为零的收尾行必须剔除并披露剔除数`（G63 形状）、`G-N3g-4 负结果照常出版且不许回退旧锚再判一次`。§1 明写 §8.7 前置：当前 `sequence_length` 取不到 ⇒ 判读必为 `ran_not_measured`（补这一列属取数侧改动，不动周期行键集）。§4 定价**未测不引外推**（×2 档每 tick 成本与 ×10 不成比例，`--scale` 乘尺寸、边数近似二次）。措辞门自扫 rc=0；台账 08 ㊵-539/540、DEBT-G64）
- [PLAN-N3-13_section87_sequence_length_source_prereg_20261008](PLAN-N3-13_section87_sequence_length_source_prereg_20261008.md)（N3 甲 §8.7 **第四项取值面升版**＝同源复算件（零训练跑量）：判据 C-1…C-4 先冻（同指纹／同预算／重放篇数＝自述篇数／吃满预算），两臂实测三条守卫全 True ⇒ 判读器 rc 从 2 变 0、`J_N3a` 出 `not_holds`；**但这个否定完全由 G64 那条跨链锚决定 ⇒ PLAN-N3-12 签字前不可发表**，§5 另记"本轮前缀分布≠整库分布"）
- [HANDOFF_N_MAINLINE_20261009](HANDOFF_N_MAINLINE_20261009.md)（N 主线**停靠交接件**：已收口项与证据锚点、待 owner 五件、下一格的精确入口（先给 `wall_clock_ms` 补断言）、本轮门与本机坑（autocrlf 幻影脏・pnpm 隐式 install 改锁文件・并行会话在飞件让 doc-sync/hygiene 红），§5 明确"目标未完成"的审计口径）
- [DISTILLATION_TOMBSTONE_20261007](DISTILLATION_TOMBSTONE_20261007.md)
- [M0_M7_PROJECT_CONSOLIDATION_20261006](M0_M7_PROJECT_CONSOLIDATION_20261006.md)

## M5 蒸馏层（20 件收口/批准/宣言/设计＋墓碑）

- [M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914](M5_B0_CLOSEOUT_AND_PLAN_REVISION_20260914.md)
- [M5_CAP0_EVAL_SET_FROZEN_20260915](M5_CAP0_EVAL_SET_FROZEN_20260915.md)
- [M5_CAP_FH_CLOSEOUT_20260920](M5_CAP_FH_CLOSEOUT_20260920.md)
- [M5_CAP_H_THRESHOLD_FREEZE_20260920](M5_CAP_H_THRESHOLD_FREEZE_20260920.md)
- [M5_COMMON_GATE_GAP_INVENTORY_20260919](M5_COMMON_GATE_GAP_INVENTORY_20260919.md)
- [M5_CORRECTION_20260920_A_SURFACE_FORM_FLIPS](M5_CORRECTION_20260920_A_SURFACE_FORM_FLIPS.md)
- [M5_DEFAULT_SUBSTRATE_SWITCH_CONTRACT_20260920](M5_DEFAULT_SUBSTRATE_SWITCH_CONTRACT_20260920.md)
- [M5_DISTILLATION_TOMBSTONE](M5_DISTILLATION_TOMBSTONE.md)
- [M5_EXIT_APPROVAL_20260920](M5_EXIT_APPROVAL_20260920.md)
- [M5_EXIT_READINESS_REVIEW_20260920](M5_EXIT_READINESS_REVIEW_20260920.md)
- [M5_K_PROMOTION_DECLARATION_20260912](M5_K_PROMOTION_DECLARATION_20260912.md)
- [M5_P3B_V2_PILOT_CLOSURE_20260917](M5_P3B_V2_PILOT_CLOSURE_20260917.md)
- [M5_POST_P5_2_REVIEW_20260913](M5_POST_P5_2_REVIEW_20260913.md)
- [M5_POST_ROUTE_A_REVIEW_20260913](M5_POST_ROUTE_A_REVIEW_20260913.md)
- [M5_R2_ARCH_COPY_CIRCUIT_PROJECT_20260925](M5_R2_ARCH_COPY_CIRCUIT_PROJECT_20260925.md)
- [M5_R2_ARCH_LEVEL_PROPOSAL_20260925](M5_R2_ARCH_LEVEL_PROPOSAL_20260925.md)
- [M5_R2_CONTENT_BINDING_CALIBRATION_STOP_20260919](M5_R2_CONTENT_BINDING_CALIBRATION_STOP_20260919.md)
- [M5_R2_D8_ADJUDICATION_B_20260918](M5_R2_D8_ADJUDICATION_B_20260918.md)
- [M5_R2_D8_CLOSEOUT_NEXT_PLAN_20260919](M5_R2_D8_CLOSEOUT_NEXT_PLAN_20260919.md)
- [M5_R2_LANGUAGE_TARGET_CREDIT_DESIGN_20260916](M5_R2_LANGUAGE_TARGET_CREDIT_DESIGN_20260916.md)
- [M5_R2_T8_MECHANISM_DESIGN_20260925](M5_R2_T8_MECHANISM_DESIGN_20260925.md)

## M4/V2 蒸馏层（3 件综述与收口）

- [M4V2_R4_CLOSURE_DECISION_20260909](M4V2_R4_CLOSURE_DECISION_20260909.md)
- [M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09](M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md)
- [M4_V1_V2_RESULT_REVIEW_2026_09_10](M4_V1_V2_RESULT_REVIEW_2026_09_10.md)

## R2/决定/设计/项目级

- [CAP0_LEGACY_LOADER_DECISION_BRIEF_20260915](CAP0_LEGACY_LOADER_DECISION_BRIEF_20260915.md)
- [DECISION-A30_emission_timing_gate_20261002](DECISION-A30_emission_timing_gate_20261002.md)
- [FRONTEND_TS_VS_HARNESS_ADOPTION_DECISION_BRIEF_20260921](FRONTEND_TS_VS_HARNESS_ADOPTION_DECISION_BRIEF_20260921.md)
- [PLAN-B-03_cortex-split_20260925](PLAN-B-03_cortex-split_20260925.md)
- [PROJECT_CONSOLIDATION_20260917](PROJECT_CONSOLIDATION_20260917.md)
- [PROJECT_RESULTS_AND_CONCLUSIONS_20260917](PROJECT_RESULTS_AND_CONCLUSIONS_20260917.md)
- [REPO_SECRET_REMEDIATION_20260919](REPO_SECRET_REMEDIATION_20260919.md)
- [TAIJI_D2_BACKEND_CHANNEL_DESIGN_20260927](TAIJI_D2_BACKEND_CHANNEL_DESIGN_20260927.md)
- [TAIJI_G4_LIFE_SYSTEM_SPEC_20260923](TAIJI_G4_LIFE_SYSTEM_SPEC_20260923.md)
- [TAIJI_HARNESS_ADOPTION_BRIEF_20260922](TAIJI_HARNESS_ADOPTION_BRIEF_20260922.md)
- [TAIJI_RESEARCH_REVIEW_2026_09_06](TAIJI_RESEARCH_REVIEW_2026_09_06.md)

## 其他

- [README](README.md)
- [VISION_FUTURE_TECHNOLOGY](VISION_FUTURE_TECHNOLOGY.md)

