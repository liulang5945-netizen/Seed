# 从这里开始

> 文档状态：[当前唯一入口·2026-11-07 21.3 已结项判负；待裁定全局收束]。**唯一主线**的状态、父身份、依赖与执行位置在[MASTER-PLAN](MASTER-PLAN.md)，本页只导航。

**一句话状态**：19.5 原训练窗口保留判负，19.2b–w 已整体关闭。Task1 受限选择/绑定复制在 19.2v 独立确认 **188/192＝0.9792**（正文/选择/起点/来源全对，4 例末端模板外溢）；19.2w 停止修复另有合成/dev 证据，未封存复核。执行包 20 已按计划 §10 第 2 行**关包**，判定 **`package20_closed_current_learning_candidate_not_established`**——"当前学习候选未成功＋F 额外贡献未证"：20.4 两 seed 上两臂方向相反（1105 利 B_EH、1106 利 CF1），F 净贡献门槛（点差 ≥0.05 且配对下界 >0）不成立；20.4d 共用批次 pooled Δ **−0.028064**（CI **−0.064706…+0.010539**）、两臂不可分辨；绝对水平远低于强捷径，**整句恒 0**。20.5 准入不成立、**不执行**；confirm 封存、**record 保持 0 读取**；**无自动 20.x 补丁链**。成本 measured **50,573.637 CPU**（提案 B=24,000 的 **2.1072×**、48,000 量级闸的 **1.0536×**，超 **2,573.637**；含声明估计 52,893.637＝1.102× 闸）；正式 800 updates＋作废 25＝**825 updates / 825 Adam**。能力账两条并列不混淆：**Task1 由结构承担、已测范围内训练权重贡献 0**；**反馈学习未立**（负结果，不等于不可能）。详见[阶段20计划](reference/PLAN-PACKAGE20_RULE_LEARNING_CORRECTION_20261105.md)与[20.6 关包报告](../reports/taiji_package20_6_package_close_20261107.json)。

**队首（唯一）**：包 21.3 表示可分性诊断已结项，判 representation_axis_closed_for_this_graph（0 训练、4/4 checkpoint 读数一致、CPU 144.984）；窗口身份对照高于主探针、绑定口径 margin 为负、distinct_f_vectors=2，表示轴在当前图/数据上关闭。按预注册不自动转训练：当前唯一待裁定是是否发布包 21 全局收束卡；21.2 与 21.4 保持冻结，不训练、不并行、不实现 U1、不定价。

**21.3 结项证据**：[报告](../reports/taiji_package21_3_representation_separability_20261107.json)＋[执行 manifest](manifests/taiji_package21_3_representation_separability_20261107.json)（双列 runner sha 一致且等于入库字节）＋[结项卡](../reports/taiji_package21_3_closeout_20261107.json)。主探针 0.6364，**窗口身份对照 0.6818**、绑定 margin **−0.0455**；20.2s 遗留 mean 口径 margin +0.1182 会给 `inconclusive`，但按预注册**不绑定、未事后切换**。F 余弦中位 ~1.0、秩 2、有效自由度 2（<3），配对 bootstrap 如实报退化。0 训练 / 0 confirm / 0 写 checkpoint。**范围口径**：当前图/数据上的关闭，不是「表示不可能有权威」；判负不自动转 21.4 或任何训练包。

**21.3 预注册（冻结，读后未改）**：[冻结规格](reference/SPEC-PACKAGE21_3_REPRESENTATION_SEPARABILITY_PREREG_20261107.md)＋[机读 manifest](manifests/taiji_package21_3_prereg_20261107.json)（`criteria_sha256=02b727f9…`）＋[特征卡](manifests/taiji_package21_3_feature_card_20261107.json)（读数前落盘）。判据先落盘、跑完未改。

**21.1 结项证据**：[报告](../reports/taiji_package21_1_readout_decomposition_20261107.json)＋[manifest](manifests/taiji_package21_1_readout_decomposition_20261107.json)（双列 runner sha 一致、4 个 checkpoint sha 逐一吻合）＋[P0 结项卡](../reports/taiji_package21_1_p0_closeout_20261107.json)。三臂与冻结判据见[包21计划](reference/PLAN-PACKAGE21_READOUT_DECOMPOSITION_20261107.md) §2，判据先落盘、跑完未改；预算 CPU ≤4,800 实际用 1597.344。confirm 继续封存，旧 confirm 不得复用。**范围口径**：U 分支是准入条件，不是 21.3 完成，也不是里程碑关闭。

**诊断链**：20.2r／20.2s／20.2t／20.3a／20.3b／20.4a／20.4b／20.4d 均为**诊断、非计划步骤**，结论已并入 20.6。

**继续读**：[MASTER-PLAN](MASTER-PLAN.md)（唯一权威主线：状态表、依赖、权威源、冲突消解、盘点）。**当前包**：[包21 读出分解](reference/PLAN-PACKAGE21_READOUT_DECOMPOSITION_20261107.md)。**上一轮梳理**：[全局梳理 2026-11-07](reference/SURVEY-PROJECT-STATE_20261107.md)。**详细开发计划**：[后续开发详细计划 2026-11-07](reference/PLAN-DEVELOPMENT_DETAIL_20261107.md)（分节风险／替换方案／结项条件；不发布队首）。
**轨道**：[ROADMAP-NEXT](reference/ROADMAP-NEXT_20261104.md)（19/20/21 包：步骤、判据、预算、分支）。
**已裁定**：[MASTER-PLAN §5](MASTER-PLAN.md)（M2=①、逐字节门采纳、19 直推、任意切换降级）。
**判据与定理**：[账本](reference/INVARIANT-LEDGER_20261103.md)。**读数边界**：[能力卡](reference/CAPABILITY-CARD_20261103.md)。**流水**：[03](active/roadmap/03_CURRENT_EXECUTION.md)。

## 旧锚点（兼容入口，内容已下沉）

<a id="1-长期目标与当前主计算图"></a>
- **1 长期目标与当前主计算图** → [MASTER-PLAN §0](MASTER-PLAN.md)
<a id="3-唯一执行队列"></a>
- **3 唯一执行队列** → [MASTER-PLAN §2](MASTER-PLAN.md)
<a id="r3b-旧基线合同"></a>
- **R3b 旧基线合同**（原合同身份，不发布指令）→ [R3 记录 §6–§7](reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)
<a id="4-顺序执行工作卡"></a>
- **4 顺序执行工作卡** → [10 工作包细则](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)
<a id="5-统一验收失败回流与停止"></a>
- **5 统一验收/失败回流与停止** → [MASTER-PLAN §5](MASTER-PLAN.md)＋[10](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)
<a id="09-连续生命与递归进化共同内核"></a>
- **09 连续生命与递归进化共同内核** → [长期发展 V2](active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md)
<a id="执行位置"></a>
- **执行位置** → [03 当前执行](active/roadmap/03_CURRENT_EXECUTION.md)
<a id="本轮规划边界与下一轮立项"></a>
- **本轮规划边界与下一轮立项** → [MASTER-PLAN §1](MASTER-PLAN.md)
<a id="根目标覆盖与研究准入边界"></a>
- **根目标覆盖与研究准入边界** → [核心需求 L1–L14](active/TAIJI_CORE_REQUIREMENTS.md)

原正文（405 行）逐字节归档：[archive/20261104_plan_consolidation/MAINLINE_FULL_HISTORY_20261104.md](archive/20261104_plan_consolidation/MAINLINE_FULL_HISTORY_20261104.md)。
