# Seed / Taiji 计划入口

> 文档状态：[索引] — 本页只导航，不定义方向、不发布队首。

正式持续 Goal：**沿 N 主线，把态极发展成由自身认知机制驱动、能通过自然中文可靠完成连续任务、跨会话积累和修正经验、从反馈中学会新能力并保持旧能力的原生认知模型；以同一可恢复版本的未见表现、旧能力保持和实际成本验收每轮进步。** Seed 负责产品与运行承载。

## 阅读顺序

1. [唯一主线](MAINLINE.md)：方向、主计算图、执行队列、旧方向处置。**任何计划冲突以本文件为准。**
2. [03 当前执行](active/roadmap/03_CURRENT_EXECUTION.md)：当前状态、唯一队首、预算与停止事实。
3. [10 持续能力执行计划](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)：工作包依赖、出口、失败回流。
4. 当前合同：[重构方案](reference/PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)、[R1 语义](reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md)、[R2 基线](reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md)、[R3 实现](reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)。
5. 规范约束：[核心需求](active/TAIJI_CORE_REQUIREMENTS.md)、[原生架构](active/TAIJI_NATIVE_ARCHITECTURE_V1.md)。
6. 旧资产：[LEGACY_ASSET_INDEX](reference/LEGACY_ASSET_INDEX.md)。旧 N177/F/N178、VISION、T2/T3、09 机制词典都在这里被明确标注为基线/组件/历史，不再产生当前指令。
7. 运营：[02 门禁/CI](active/roadmap/02_GATES_AND_CI.md)、[05 技术债](active/roadmap/05_TECH_DEBT_REGISTER.md)、[07 整模型交付](active/roadmap/07_MINI_MODEL_DELIVERY.md)、[08 重放手册](active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md)。

## 维护规则

- 方向上只能有一份文件：[MAINLINE](MAINLINE.md)。新方向只能作为“候选”进入其第 3 节，不允许另建并列主线。
- 只有 03 可以写“当前唯一下一步”；其他文档不得发布队首。
- `plans/active/` 每份文档前 10 行必须有 `> 文档状态：[...]`；取值限主线、状态、工作包、合同、运营、规范、组件、历史、证据、索引。
- 旧文档里的“当前/下一步/唯一动作”只解释写作时点；当前调度只看 MAINLINE 与 03。
- 冻结 SPEC、审批、原始输出和 Git 身份不因收敛改名或删除；只降级其调度权。
- 目录与产物按[目录规范](../docs/FOLDER_STRUCTURE_RULES.md)管理；训练前先验证 checkpoint 保存/恢复。

## 守卫

`scripts/training/audit_plan_single_mainline.py` 与 `tests/taiji_native/test_plan_single_mainline_contract.py` 检查：唯一队首标题、active 文档状态标签、索引覆盖、旧文档不得自称主线。违反时测试失败。

## 历史

2026-10-23 的[全项目历史整理](reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md)保留旧路径与 Git 恢复身份；更早的逐轮状态在 `plans/archive/`。历史失败、审批与排除项继续有效，旧队首只解释当时时点。
