# 从这里开始

> 文档状态：[当前唯一入口·2026-11-05 阶段收束]。**唯一主线**的状态、父身份、依赖与执行位置在[MASTER-PLAN](MASTER-PLAN.md)，本页只导航。

**一句话状态**：19.5原训练窗口保留判负，19.2b–w已整体关闭。Task1受限选择/绑定复制在19.2v独立确认188/192＝0.9792；19.2w停止修复另有合成/dev证据，未封存复核。执行包20 rev2的**20.1**（规则夹具/同形参考图/泄漏探针，`ready`）、**20.2**（`ready`：dL/dF 0.068832、135 张量会话冻结下全无梯度、B_EH 0 写、三路径恢复 90/90）、**20.2r**（`under_powered_f_channel`：提交字节重放四项误差 0.0、F 五档杠杆 max **0.00795%** < 1%、五档 byte_diff 0、oracle 诊断 4.183% 但 0 字节变化）、**20.2s**（`inconclusive`：F 闭式探针决策位留一 **0.6818**、最强对照 0.5091、差 0.1727；22 个纠正面 F 只有 **2 个不同向量**，0.6818 与"仅按反馈窗口猜"完全相同）与**20.2t**（纯设计闸，推荐每轮内容寻址；+0 参数、+266,273 状态元素≈2.03 MiB；未实施、未定价）、**20.3a**（按选项 2 实施内容寻址写并按预注册三闸判 `option2_failed`：结构闸过、杠杆闸 `leverage_max` **8.716e-6** < 0.01 且五档 `byte_diff` 0、信息闸 **0.4091** < 0.60 且低于最强对照 **0.5909**；0 训练 / 224.94 CPU）与**20.3b**（`authority_weak`：44 个决策位 gold 全在 3 名外、deficit 中位 3.376e-4＝spread 的 0.46%、模板位 11–13/12 已错；统一增益首次翻位 g=1e4 且 0/22 答对；闭式岭回归 9/22、方向一致率 0.682、W 秩 1、49,152 参数；0 训练 / 56.92 CPU）均已交付；0 训练。学习收益、纠正和学后保持仍待检验。详见[阶段20计划](reference/PLAN-PACKAGE20_RULE_LEARNING_CORRECTION_20261105.md)与[20.2t 备忘](reference/DESIGN-20_2t_WRITE_GRANULARITY_20261105.md)。

**队首（唯一）**：**20.3b 已执行并判 `authority_weak`（有翻位但 9/22 < 11；统一增益首次翻位 g=1e4 且 0/22 答对）；不修寻址、不改 τ/α、20.3 定价未准入，正式训练未准入，旧 confirm 保持封存。**

**继续读**：[MASTER-PLAN](MASTER-PLAN.md)（唯一权威主线：状态表、依赖、权威源、冲突消解、盘点）。
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
