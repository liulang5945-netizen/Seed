# Seed / Taiji 计划入口

> 文档状态：[索引] — 本页只导航，不定义方向、不发布队首。

正式持续Goal：**发展同一持续活动、持续可塑、能按需新增单元/连接/群体并继承经验与能力的态极原生认知主体；无新提示时仍可按目标与状态活动、学习、巩固、休息或等待。自然中文任务、跨会话经验与旧能力保持在这一主体上逐阶段验证。** 连续生命与规模生长是递归进化的基础，不是权重训练结束后再选的附加项；Seed负责运行承载。

## 按顺序推进只读一份

[MAINLINE 新N主线落地计划](MAINLINE.md)已由项目所有者正式确定为新的N主线，统一方向、步骤01–13、未完成工作的逐子步实施卡、依赖、决策点及有限出口。直接从[执行位置](MAINLINE.md#执行位置)进入，按[唯一执行队列](MAINLINE.md#3-唯一执行队列)推进；状态与后继只在总计划维护。**任何计划冲突以总计划为准。**

R1、R3b、R5b 等保留为原合同和实验身份，N1–N6 保留为机制责任轴；它们都不是另一套执行编号。原文档中的数字文件名也不代表新的推进顺序。

## 需要核验时再查

- [核心算法v1主稿](reference/TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md)：09.1的具体方程、信用、结构出生、时序和反证，不另发执行队列。

- [03 证据与审批台账](active/roadmap/03_CURRENT_EXECUTION.md)：核验报告、批准、预算、恢复与历史限制；只跳转总计划的执行位置。
- [10 兼容导航](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)：保留旧章节入口，转到总计划，不另排工作包。
- [合同与证据索引](reference/README.md)：按总计划步骤查原 SPEC；[重构设计依据](reference/PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)不再承担调度。
- [核心需求](active/TAIJI_CORE_REQUIREMENTS.md)、[原生架构](active/TAIJI_NATIVE_ARCHITECTURE_V1.md)：核验认知职责与长期约束。
- [活动文档索引](active/PLAN_INDEX.md)、[旧资产索引](reference/LEGACY_ASSET_INDEX.md)：查运营、机制组件与历史材料。

## 维护规则

- 方向、执行编号、阶段状态和实际动作只维护在[MAINLINE](MAINLINE.md)；新方向先进入其第 3 节，不另建并列主线。
- 03 保留唯一的“当前唯一下一步”入口标题并静态跳转总计划；索引、兼容页和原设计不复制队首或阶段状态表。
- `plans/active/` 每份文档前 10 行必须有 `> 文档状态：[...]`；取值限主线、状态、工作包、合同、运营、规范、组件、历史、证据、索引。
- 旧文档里的“当前/下一步/唯一动作”只解释写作时点；当前调度只看总计划。
- 冻结 SPEC、审批、原始输出和 Git 身份不因收敛改名或删除；只降级其调度权。
- 目录与产物按[目录规范](../docs/FOLDER_STRUCTURE_RULES.md)管理；训练前先验证 checkpoint 保存/恢复。

## 守卫

`scripts/training/audit_plan_single_mainline.py` 与 `tests/taiji_native/test_plan_single_mainline_contract.py` 检查：唯一队首标题、active 文档状态标签、索引覆盖、旧文档不得自称主线。违反时测试失败。

## 历史

2026-10-23 的[全项目历史整理](reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md)保留旧路径与 Git 恢复身份；更早的逐轮状态在 `plans/archive/`。历史失败、审批与排除项继续有效，旧队首只解释当时时点。
