# Seed / Taiji Native v1：范围与阶段

> 本文件定义长期范围和阶段退出条件，不记录逐次实现日志，也不决定当前下一步。2026-08-29 以前的完整 P0–P7 推进记录见 [SEED_NATIVE_PHASE_HISTORY_20260829.md](../../archive/history/SEED_NATIVE_PHASE_HISTORY_20260829.md)。

## 1. 项目目标

Taiji 是 Seed 中拥有认知状态、学习、世界模型、记忆、规划和结构发展的原生架构。它站在成熟技术之上复用 embedding、attention、状态空间、图计算、优化器、RL、检索、编译器、数据库、provider 和 CUDA，但重新安排所有权：成熟技术是器官、算子或训练手段，不能成为隐藏的第二 cognition。

目标不是：

- 从 byte、神经元或计算机原语重新发明一切；
- 为了“不同于 Transformer”而拒绝成熟方法；
- 用生物名称硬编码神经元角色；
- 把模型参数增加、全量重训或 UI 动画称为自进化；
- 让语言 provider、prompt、前端或 Legacy 选择工具、拥有记忆或伪造 Taiji 状态。

## 2. 固定设计原则

1. **能力优先。** 先定义需要形成的能力、状态和因果闭环，再选择成熟技术或自研机制。
2. **Owner 唯一。** 每份可变认知状态只有一个 owner；跨层通过版本化 DTO、event 和 checkpoint 交换。
3. **真实闭环。** 输入必须绑定来源，行动必须经过真实环境，学习必须来自真实 Outcome。
4. **渐进成长。** 保留旧能力，在原 checkpoint 上局部更新、巩固、增长、剪枝和回滚；不默认从零重训。
5. **资源治理。** 稀疏、容量和规模是预算策略，不是表达能力的硬编码上限。
6. **证据分层。** S0 模拟、S1 replay/sandbox、S2 真实 Workbench/client 不得相互冒充。
7. **可恢复。** 任何训练、结构或装配改变必须保存、恢复、继续并保留 lineage/tombstone。
8. **高上限但可证伪。** 优先选择能支持开放域、长期学习和身体成长的方案，同时要求 red proof、holdout、lesion 和 rollback。

## 3. P0–P7 阶段与当前状态

| 阶段 | 长期职责 | 当前基线 | 仍未闭合 |
|---|---|---|---|
| P0 架构定基线 | 身份、所有权、Transformer/Legacy 边界、核心对象 | 已完成 | 随新发现持续校准，不得另建第二架构 |
| P1 合同与兼容骨架 | versioned state/DTO、native checkpoint、兼容迁移 | 已完成基线 | schema 演进仍须向后兼容 |
| P2 感知与时间抽象 | 学习型 assembly、关系、时序、多尺度状态 | 窄 Gate 已通过 | 开放模态、长期复杂输入与规模曲线 |
| P3 世界状态与 workspace | entity/event/relation/affordance、预测、选择性路由 | 窄 Gate 和真实 Workbench 基线已通过 | 更长 horizon、开放 schema 和复杂因果干预 |
| P4 多系统记忆 | working/episodic/semantic/procedural、巩固、遗忘 | 窄 Gate、容量/干扰和 checkpoint 基线已通过 | 开放域长期记忆治理与 R5A 内化 |
| P5 执行认知 | goal、planning、imagination、replan、executive credit | 有限闭环和 grounded multi-step 基线已通过 | 长程自治、写入治理和真实长期任务 |
| P6 生成与行动 | ContentPlan、语言/工具效应器、provider、安全 fallback | native-readable、provider watchdog、工具 Outcome 基线已通过 | 外部 artifact S2、语言质量、R5B 效应器注册 |
| P7 持续发展与规模化 | homeostasis、结构提案、成长/剪枝、跨区协作、CUDA | 小型结构 Gate、ledger、回滚和 CPU profile 已有 | R5A/B/C 开放域闭环、长期保持、真实 CUDA/多模态 |

“基线已通过”只表示对应窄 Gate 存在，不表示阶段在开放域上永久完成。新能力必须进入其 owner 和阶段，不通过叠加脚本或 UI 绕开阶段合同。

## 4. 通用阶段退出门槛

每个阶段或子阶段只有同时满足以下条件才可关闭：

- owner、输入、输出、状态转移和失败语义已冻结；
- 参数、非参数状态、资源和外部 artifact 在 checkpoint 中有明确归属；
- red proof 能稳定失败，绿实现不靠跳过分支或伪造 fixture；
- train/holdout 隔离，关键能力有 lesion，跨 seed 或独立任务切片结果可复现；
- 恢复后能继续一步，lineage、预算、结构和 tombstone 等价；
- S0→S1→S2 按能力风险逐级完成；未达到 S2 时声明中明确限制；
- 相关 Python/前端/API/桌面/安全/CI 门禁全部实际执行，没有被 `needs` 或环境早退隐藏；
- 实现事实、manifest、当前下一步和报告在同一提交收口。

## 5. 当前后续阶段映射

当前详细开发不重新发明 P0–P7，而是在 P4–P7 的交界完成 W7-R5：

- **R5A 知识内化**属于 P4/P5：真实 Outcome 形成可恢复的选择/记忆参数，经过可删性 lesion 后才允许移除描述性外挂。
- **R5B 效应器成长**属于 P6：能力包形成可撤销身体装配，Taiji 仍通过 structured affordance 选择。
- **R5C 结构自进化**属于 P7：长期错误、容量和遗忘触发局部结构候选，经过 shadow/holdout/lesion/rollback 后进入稳定模型。
- **语言/provider 成熟**属于 P6，不得插回 P5 决策。
- **CUDA**属于 P7 的执行加速与数值一致性验证，不决定认知架构。

具体依赖、模块和 Gate 见 [04_EXECUTION_PLAN.md](04_EXECUTION_PLAN.md)，即时动作只看 [03_CURRENT_EXECUTION.md](03_CURRENT_EXECUTION.md)。

## 6. 暂停与恢复的独立线

- R3 Windows shell：页面证据完成，真实任务栏/托盘/通知/DPI 因工具无法激活窗口而 `tool-blocked`。
- R4 CUDA：当前主机 CPU-only，保持 `hardware-blocked`。

两条线不从路线删除；条件具备后单独补证并提交。它们未完成时不能发布相应声明，但不冻结与之无依赖的 R5 CPU/native 工作。

## 7. 永久禁止的捷径

- 用固定 action/intent/神经元类型表替代学习与资源治理；
- 用 provider、prompt 或前端维护认知状态或工具选择；
- 用 CPU 测试宣称 CUDA；
- 用 S0 模拟宣称真实自治；
- 在没有 checkpoint roundtrip 时启动长训或结构突变；
- 把执行器删除称为知识内化；
- 用全量从零训练作为持续成长的唯一迭代方式；
- 为赶进度修改 Gate 让错误变绿，或在 CI 红时继续叠加功能。
