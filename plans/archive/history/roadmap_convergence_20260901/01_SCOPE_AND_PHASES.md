# Seed / Taiji Native v1：范围与阶段

> 本文件定义长期范围和阶段退出条件，不记录逐次实现日志，也不决定当前下一步。2026-08-29 以前的完整 P0–P7 推进记录见 [SEED_NATIVE_PHASE_HISTORY_20260829.md](../../archive/history/SEED_NATIVE_PHASE_HISTORY_20260829.md)。2026-08-31 交付顺序阶段 P0–P8 的完整推进记录见 [62_POST_S51_PROJECT_CONVERGENCE_20260831.md](../../archive/history/62_POST_S51_PROJECT_CONVERGENCE_20260831.md)。

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

## 3. P0–P7 能力阶段与当前状态

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

## 6. 统一执行原则

1. 每一阶段必须闭合一个真实用户旅程，不能只因为某个内部 ledger 多了字段就继续切下一片。
2. provider 可以做语言编码/表达和候选语义证据，但不能拥有 goal、tool choice、memory、policy 或最终 ActionIntent。
3. 新结构只有在现有权重、路由、记忆和策略已不足，并通过 holdout、lesion、资源、旧能力保持与 rollback 后才能准入。
4. 训练或在线学习前，先通过 checkpoint 保存、进程关闭、恢复、继续；失败时禁止开始训练。
5. 新成长路径使用 verified artifact；历史证据兼容必须显式声明原因，不允许静默升级或伪造 measurement facts。
6. CUDA 与 Windows shell 证据继续保留，但硬件/工具不可用时不阻塞 CPU/native 主线，也不得宣称通过。
7. CI 依用户当前决定暂缓；暂缓不等于通过。恢复 CI 时集中收口全部累积问题，在全绿前不继续发布功能声明。
8. 已完成过程进入 archive；核心需求、当前架构、当前缺口和未完成 Gate 留在 active/reference。

## 7. 交付顺序阶段 P0–P8 与未退出 Gate

本节的 P0–P8 是**交付顺序阶段**，与第 3 节的**能力阶段 P0–P7** 是两个不同维度：第 3 节按认知能力归属划分 owner，本节按用户旅程闭合顺序划分交付批次。已完成阶段的逐条过程记录见 [62_POST_S51_PROJECT_CONVERGENCE_20260831.md](../../archive/history/62_POST_S51_PROJECT_CONVERGENCE_20260831.md)，报告与测试证据中的 `P2-13`、`P5-1`、`P6-1d`、`P7-1` 等编号均属本节维度，不改名。

P0–P2 的交付目标是闭合下面这条自然语言到可读输出的完整链路：

```text
用户输入
  -> Observation / Percept
  -> Taiji Goal 与任务约束
  -> 当前 WorldState / Workbench affordances
  -> Taiji Plan / ActionIntent
  -> preview / policy / approval
  -> IDE 或工具执行
  -> EnvironmentOutcome
  -> World / Memory / Learning
  -> ContentPlan / LanguageOrgan
  -> 可读输出
```

### P3：语言器官与 provider 生产化

目标：让语言层稳定充当“嘴巴/语言接口”，同时保持 Taiji 认知所有权。

- `native-readable` 继续作为无外部 provider 时的真实默认；`structured-stub` 仅可显式调试。
- 用相同 ContentPlan 对 native-readable 与外部 provider 评估可读性、约束保持、事实遗漏、幻觉和延迟，禁止比较两个不同认知结果。
- 在 packaged client 中验证真实 Qwen provider artifact 内容寻址、版本轮换、watchdog、previous→native 降级、cooldown、重启重绑与失败通知；确定性集成 seam、后端真实加载和浏览器字段已通过，真实 Qwen 安装包现场与质量 Gate 仍未验收。
- provider 输出必须经过约束检查；不得写 Taiji memory、改变 intent、调用工具或绕过 policy。
- P7-1 质量基线已实测但未通过：当前 Qwen2.5-0.5B-Instruct 清晰案例通过率 `0.2857`，模糊请求未达到高歧义；因此当前 artifact 只能作为实验/回退 provider，生产语义入口必须等待更强 artifact 通过同一 Gate。

退出 Gate：真实客户端完成至少一次成功轮换、一次失败回退和一次 checkpoint 重启；语言后端变化不改变同一任务的工具决策与事实约束。

### P4：interaction-group、学习与自进化收益 Gate

目标：从“机制可以运行”升级为“能力确实增长”。provider watchdog、interaction-group 和小型模拟 Gate 都在本阶段继续，不搁置。

- interaction-group：在预注册任务上比较最强单体、稠密平均、随机分组与学习分组，只有组合完成单体不能完成的任务才声明 `1+1>2`；P4-6 已增加异质成员画像和未见组合关系 transfer，P4-7 已增加三轮 future 对照，P4-8 已增加真实 Outcome 写回/拒绝/重启/rollback，但不把有界归纳或受控在线更新外推成开放域智能。
- 长期收益：P4-7 已在真实 Workbench future holdout 上和只调权重/路由/记忆对照比较；后续必须扩展任务域与连续反馈，而不是继续堆叠一次性任务族或合成 reward。
- 在线成长：P4-8 已验证 Outcome 能在显式 Gate 后写回 relation learner 并可重启/rollback；P4-9 已验证在线 interaction evidence 只有在独立 holdout/retention、arbitration、shadow validation、admission 和 rollback 之后才能抵达结构变化；P4-10a 已验证首次结构扩容可在两个未见三动作 Workbench context 上超过固定容量的权重/路由/记忆对照；P4-10b 已验证第二个独立在线周期将容量从 2→3→4 并在未见四动作任务上保持收益；P4-11 已验证 editor+MCP 跨域结构收益与旧 workspace 保留；P4-12 已验证 terminal 三域治理、资源/审批、失败停止与 checkpoint 恢复。自然语言主线当前转向 P2-13 产品 API/前端接线，避免后端 Gate 与真实用户旅程脱节。
- 小型模拟 Gate：只用于快速验证状态转移、credit 和 rollback，不替代真实 Workbench longitudinal Gate。
- R5A：证明知识内化后可降低外部 artifact 依赖，同时旧任务不退化；未过独立删除评审前不物理删除来源。
- R5B：只有真实新工具能力通过 shadow、approval、resource 和 disposer Gate 后才进入 registry；不从任意源码直接注册 executor。
- R5C：先比较权重/路由/记忆调整；只有持续容量不足时才申请结构增长。结构变化必须在未见任务上产生净收益，并通过 lesion、资源和 rollback。
- 睡眠/玩耍/稳态：接入同一真实任务纵向证据，验证不同模式的因果作用，避免成为外围 scheduler 展示。

退出 Gate：至少一个长期任务族同时证明新能力收益、旧能力保持、资源边界、因果 lesion 和 checkpoint continuation；“自进化”声明以该证据为起点，不以新增参数数量为依据。

### P5：工程模块化与 Legacy 残留退役

目标：降低大文件和永久兼容层对架构上限的限制，但不做无证据的大重写。

- 按 owner 拆分 `taiji/adapter.py`：checkpoint/state、provider、recovery、structural lifecycle、workbench projection 分离；adapter 只保留组合与兼容 facade。
- 拆分 `api/seed_runtime.py`：runtime lifecycle、Workbench orchestration、structural artifact bridge、persistence/status 分离。P5-1 已先把自然语言 Workbench plan/approval/execute 协议移至 `api/natural_language_workbench.py`，P5-2 已将语义 grounding engine 移至 `api/workbench_grounding.py`，P5-3 已将 request/preflight/execution boundary 移至 `api/workbench_execution.py`，均保留 runtime facade。
- 拆分 `seed_platform/workbench.py`：capability registry、policy/preflight、language/editor、workspace executor、terminal/MCP executor 分离。
- `taiji/contracts.py` 按 observation/world/goal/action/memory/language 划分，但保持版本化导出和 checkpoint 迁移。
- 为 HF/GGUF/Transformer 残留建立退役清单：前端 live 路径保持禁止；隐藏 API tombstone 只保留一个明确迁移窗口；`gguf_path`、`download_hf` 等配置在迁移测试通过后删除。
- Legacy NeuroPlex 保持离线 benchmark，不允许重新成为 Seed 启动默认或 Taiji runtime 依赖。

退出 Gate：模块依赖方向可静态检查；旧 checkpoint 可显式迁移；前端、OpenAPI 和运行配置不再暴露模型格式切换；行为回归不因拆分改变。

### P6：客户端真实性与桌面体验

目标：让客户端展示真实 Taiji，而不是重复入口、固定文案或与实现不一致的状态。

- 生命状态只保留一个主入口和一个详情视图；全局页面不再顶置重复状态条。
- 多维状态采用真实 runtime projection，明确“已测事实、估计值、不可用”，不造假百分比。
- 侧边栏无需滚动即可访问主功能；IDE、知识、训练、生命和设置名称与当前 Taiji 能力对齐。
- Windows 应用图标、任务栏、托盘、通知弹窗统一使用同一 Taiji logo；窗口圆角、DPI、键盘导航和 reduced-motion 一并现场验证。
- HF/GGUF/Transformer 格式切换不得重新出现在产品 UI。

退出 Gate：packaged `Seed.exe` 在真实 Windows 桌面完成窗口、任务栏、托盘、通知、DPI 和关键用户旅程取证；前端字节与安装包内容一致。

### P7：集中 CI、仓库和发布收口

目标：按用户决定，在功能主线完成一轮后统一处理累积 CI，而不是把“未运行”写成“通过”。

- 先跑与变更相关的定向 Gate，再跑 Python lint/type/test、前端 lint/test/build、OpenAPI、Legacy-off、checkpoint、Windows/package 全矩阵。
- 对所有失败按根因分组，一次只修一个根因，直到无 skipped/允许失败的关键 Gate。
- 审计 `main`、attached worktree、local refs、`origin/main`；保存有价值的未提交变化后再收束到只保留 main。
- 生成发布 manifest，绑定 commit、前端字节、Seed.exe、checkpoint format、报告 digest 和能力声明。

退出 Gate：CI 全绿、工作树和 refs 收敛、发布包与 main 同源、远端同步状态通过实时核验。

### P8：CUDA 独立恢复线

当前保持 `hardware-blocked`，不删除、不伪装完成。真实 CUDA 主机到位后按固定顺序执行：同一 workload CPU profiler → CUDA profiler → CPU→CUDA→CPU checkpoint → 数值/结构/预算一致性 → 热点证据评审。只有 profiler 证明现有算子是瓶颈，才实现 fused/sparse kernel。

## 8. 暂停与恢复的独立线

- R3 Windows shell：页面证据完成，真实任务栏/托盘/通知/DPI 因工具无法激活窗口而 `tool-blocked`。
- R4 CUDA：当前主机 CPU-only，保持 `hardware-blocked`。

两条线不从路线删除；条件具备后单独补证并提交。它们未完成时不能发布相应声明，但不冻结与之无依赖的 R5 CPU/native 工作。

## 9. 永久禁止的捷径

- 用固定 action/intent/神经元类型表替代学习与资源治理；
- 用 provider、prompt 或前端维护认知状态或工具选择；
- 用 CPU 测试宣称 CUDA；
- 用 S0 模拟宣称真实自治；
- 在没有 checkpoint roundtrip 时启动长训或结构突变；
- 把执行器删除称为知识内化；
- 用全量从零训练作为持续成长的唯一迭代方式；
- 为赶进度修改 Gate 让错误变绿，或在 CI 红时继续叠加功能。

- 不把外部 Skill 文本、MCP 返回或插件说明直接拼接为训练真值。
- 不允许模型生成源码、shell、依赖或 manifest 后自动安装/执行。
- 不允许 provider、Skill、MCP 或 frontend 拥有 Taiji Goal、ActionIntent、policy 或结构 admission。
- 不允许 holdout、retention、evaluator 预期答案进入训练。
- 不允许同一 trial 同时改变 Taiji 认知和客户端插件后声称单一因果收益。
- 不通过硬编码操作词表、神经元角色、任务 ID 或 prompt 分支提高 Gate。
- 不因本机无 CUDA 而停止原生正确性、checkpoint 和 CPU 小规模训练。
- 不把插件数量、参数数量或神经元数量本身当成智能增长。
