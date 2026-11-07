# 从这里开始

> 文档状态：[当前唯一入口·2026-11-07 包 21 启动]。**唯一主线**的状态、父身份、依赖与执行位置在[MASTER-PLAN](MASTER-PLAN.md)，本页只导航。

**一句话状态**：19.5 原训练窗口保留判负，19.2b–w 已整体关闭。Task1 受限选择/绑定复制在 19.2v 独立确认 **188/192＝0.9792**（正文/选择/起点/来源全对，4 例末端模板外溢）；19.2w 停止修复另有合成/dev 证据，未封存复核。执行包 20 已按计划 §10 第 2 行**关包**，判定 **`package20_closed_current_learning_candidate_not_established`**——"当前学习候选未成功＋F 额外贡献未证"：20.4 两 seed 上两臂方向相反（1105 利 B_EH、1106 利 CF1），F 净贡献门槛（点差 ≥0.05 且配对下界 >0）不成立；20.4d 共用批次 pooled Δ **−0.028064**（CI **−0.064706…+0.010539**）、两臂不可分辨；绝对水平远低于强捷径，**整句恒 0**。20.5 准入不成立、**不执行**；confirm 封存、**record 保持 0 读取**；**无自动 20.x 补丁链**。成本 measured **50,573.637 CPU**（提案 B=24,000 的 **2.1072×**、48,000 量级闸的 **1.0536×**，超 **2,573.637**；含声明估计 52,893.637＝1.102× 闸）；正式 800 updates＋作废 25＝**825 updates / 825 Adam**。能力账两条并列不混淆：**Task1 由结构承担、已测范围内训练权重贡献 0**；**反馈学习未立**（负结果，不等于不可能）。详见[阶段20计划](reference/PLAN-PACKAGE20_RULE_LEARNING_CORRECTION_20261105.md)与[20.6 关包报告](../reports/taiji_package20_6_package_close_20261107.json)。

**队首（唯一）**：**包 21.1 零训练读出分解**——在 20.4c_r2 的 4 个末步 checkpoint 上，把「句框」与「内容」分开强制（`natural`／`template_forced`／`content_forced` 三臂，dev 16 world），一分二回答：整句恒 0 的第一次失败发生在**发射/读出**还是**反馈写读权威**。预算 CPU ≤4,800、0 Adam、0 训练、0 confirm；判据先落盘（[包21计划](reference/PLAN-PACKAGE21_READOUT_DECOMPOSITION_20261107.md)）。该包只给一个下一步；未过前不开 21.2–21.4、不实现 U1、不定价。confirm 继续封存，旧 confirm 不得复用。

**诊断链**：20.2r／20.2s／20.2t／20.3a／20.3b／20.4a／20.4b／20.4d 均为**诊断、非计划步骤**，结论已并入 20.6。

**继续读**：[MASTER-PLAN](MASTER-PLAN.md)（唯一权威主线：状态表、依赖、权威源、冲突消解、盘点）。**当前包**：[包21 读出分解](reference/PLAN-PACKAGE21_READOUT_DECOMPOSITION_20261107.md)。**上一轮梳理**：[全局梳理 2026-11-07](reference/SURVEY-PROJECT-STATE_20261107.md)。
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
