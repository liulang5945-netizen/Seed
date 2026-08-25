# Seed / Taiji Native v1 开发路线（2026-08）

状态：**当前唯一执行路线**

更新时间：2026-08-25（Taiji 架构重新定基线）

## 1. 目标与纠正后的边界

Taiji 是完整原生认知架构；Seed 是项目、产品和运行时。路线不再以“把 Transformer 各功能换成更原始的神经元算子”为目标，而以项目原始需求中的异质协作、`1+1>2`、自适应激活、身体—生命闭环、睡眠/玩耍、持续学习、自我成长，以及可学习表征、世界模型、记忆、目标、推理、规划和生成的完整闭环为目标。

```text
Seed runtime hosts Taiji
Taiji owns perception → cognition → learning → action
TSK-v8 is a reusable kernel, not the completed architecture
Legacy NeuroPlex is a frozen offline comparison
```

长期目的见 [TAIJI_CORE_REQUIREMENTS.md](TAIJI_CORE_REQUIREMENTS.md)，权威设计见 [TAIJI_NATIVE_ARCHITECTURE_V1.md](TAIJI_NATIVE_ARCHITECTURE_V1.md)。旧 raw-byte substrate 路线完整保存在 [旧路线归档](../archive/history/SEED_DEVELOPMENT_ROADMAP_PRE_NATIVE_V1_20260825.md)，其中 R1–R5 不再是当前研究执行顺序。

## 2. 当前事实

### 2.1 可以保留的成果

- `taiji/` 已与 `neuroplex`、`transformers` 解耦，具备独立命名空间、状态和 checkpoint。
- TSK-v8 已有 raw-byte codec、持续 predictive fabric、局部学习、情景原型、行动闭环和 K 系列机制证据。
- `CapacityPolicy`、参数预算、CPU/CUDA device 语义和恢复测试可以成为新架构的资源治理基础。
- Seed 原生 API、训练入口、桌面产品和 Legacy 开关已有可复用工程基线。
- S1 产品体验验收已完成；阶段报告和归档体系有效。

### 2.2 不再成立的声明

- raw byte one-hot 不是 Taiji 的完整输入表示，只是文本器官边界。
- `sensor → fabric → episodic field → byte motor` 不是完整认知架构。
- 固定 fan-in、固定区域比例和 257 动作不是 Taiji 身份。
- “没有 embedding/attention/optimizer”不是智能或原生性的充分条件。
- N0–N11/M5–M7 只证明 kernel 机制，不证明概念、推理、语言或 AGI。
- Seed 不再拥有认知器官、目标和自我模型；这些属于 Taiji。

### 2.3 P1 当前实现状态（2026-08-25）

P1 已在当前分支落地为兼容纵切片：

- `taiji/contracts.py` 定义版本化的 `Observation`、`PerceptEvent`、`WorkspaceState`、
  `WorldState`、`MemoryState`、`GoalState`、`PlanState`、`ActionIntent`、`Outcome`，
  并提供可恢复的 `CognitiveState` 与 `NativeCheckpoint`；
- `taiji/adapter.py` 的 `TSKV8Adapter` 保留 TSK-v8 公开 API，同时把观察→感知事件→
  工作空间/世界/记忆摘要→行动意图→环境结果接入 Taiji-owned v1 状态；
- `Seed.architecture` 是正式入口，`Seed.substrate` 仅保留为旧调用方的兼容别名；
- Seed 的 `seed-native-v1` 旧 checkpoint 仍可读取，新 checkpoint 增加 Taiji v1 原子信封，
  认知状态由 Taiji 序列化和恢复；
- 旧 TSK-v8 行为、参数和 kernel checkpoint 仍保持原路径，不把 adapter 的桥接摘要宣称为
  完整 Taiji 智能。

P1 的剩余工作只限于回归门禁与边界维护；下一阶段进入 P2 学习型感知与时间抽象。

## 3. 执行原则

1. 能力合同先于模块命名和代码目录。
2. 每阶段交付一个可执行纵切片，禁止只创建空框架。
3. raw input、内部表征和最终输出必须分层，不能共享一个 byte alphabet 冒充认知。
4. 每个新增能力都要有 holdout、损伤实验和失败标准。
5. development training 与 lifetime learning 分开报告；辅助优化不能冒充自主在线学习。
6. CUDA 优化只针对已证明必要的 v1 算子，不绑定旧 kernel 拓扑。
7. Legacy 继续冻结，不进入 Taiji forward，也不因本轮重构立即删除。
8. 优先复用成熟的 embedding、路由、状态空间、图计算、优化器、强化学习、检索和 CUDA 方法；只有它们无法满足持续状态、因果闭环、终身学习或资源约束时才自研替代。

## 4. 阶段顺序

| 阶段 | 目标 | 退出结果 |
|---|---|---|
| P0 | 架构重新定基线 | 权威身份、目标架构、旧 kernel 和旧路线边界清晰 |
| P1 | v1 合同与兼容骨架 | 新状态/事件/所有权合同可执行，TSK-v8 行为和 checkpoint 不回退 |
| P2 | 学习型感知与时间抽象 | 从 byte 流形成可变长度 assembly，并在未见组合上迁移 |
| P3 | 世界状态与工作空间 | 对象/事件/关系持续存在，选择性路由支持多步任务 |
| P4 | 情景、语义与程序性记忆 | one-shot 经历可巩固成可迁移概念和技能 |
| P5 | 目标、推理、想象与规划 | model-based rollout 在未见目标上优于 reactive baseline |
| P6 | 原生语言与工具行动 | 内容计划经表达器官稳定生成可读语言和结构化工具调用 |
| P7 | 持续进化、多模态与规模化 | 内生调节/结构成长、保持旧能力、跨模态迁移、真实 CUDA 与资源治理闭环 |
| P8 | 产品原生化与公开测试 | Seed 默认发行只承载真实 Taiji v1 能力，达到发布门槛 |

阶段严格按 P0 → P8 推进。产品安全修复可以并行，但不得用 UI、Agent 壳或 Legacy fallback 伪造尚未完成的 Taiji 能力。

## 5. P0：架构重新定基线

### 工作项

- 把 Taiji 定义为完整认知架构，把 Seed 定义为产品/运行时。
- 新建 Taiji Native Architecture v1 权威合同。
- 将当前精确方程规范和旧路线移入 archive，定位为 TSK-v8 历史/兼容证据。
- 更新 README、计划入口、边界测试和能力声明。
- 停止旧 R4 语言长训、旧 R5 机制叠加和针对固定 fan-in 的 CUDA kernel 工作。

### 退出门槛

- active 文档不存在“Seed 是认知模型主体”“TSK-v8 是完整 Taiji”的冲突。
- 旧文档链接可追溯，归档中的旧“下一步”全部失效。
- 架构/命名边界测试通过，`main` 干净并有单一提交。

## 6. P1：v1 合同与兼容骨架

### 工作项

1. 定义 `Observation`、`PerceptEvent`、`WorkspaceState`、`WorldState`、`MemoryState`、`GoalState`、`PlanState`、`ActionIntent` 和 `Outcome` 的版本化协议。
2. 建立 Taiji 顶层认知状态和原子 checkpoint envelope；Seed 只保存产品元数据并委托 Taiji checkpoint。
3. 将现有 `Taiji` 类明确包装为 `TSK-v8` compatibility adapter，不立刻移动全部源码。
4. 增加所有权测试：`seed/` 不得新增概念记忆、规划、语言模型或 teacher policy；`taiji/` 仍不得导入 Legacy/Transformer。
5. 建立一个最小纵切片：观察进入、状态推进、产生意图、执行、接收 outcome；初期可调用 kernel，但所有接口使用 v1 语义。

### 退出门槛

- 现有 kernel 测试、Seed API 和 checkpoint 恢复无回退。
- v1 状态能保存、恢复和确定性续跑。
- 认知所有权由 AST/contract tests 强制，而不是只写在计划中。

## 7. P2：学习型感知与时间抽象

### 工作项

- 保留 byte codec，新增可学习局部特征和可变时长 assembly。
- 用预测稳定性、边界惊讶、重复和上下文区分学习 chunk，不提供人工 token 答案。
- 支持低层 byte 纠错通路与高层事件通路并存。
- 建立 A1 数据：未见词形、未见组合、边界扰动、随机 chunk 和 byte-only 对照。

### 退出门槛

- 高层表示相对 byte-only 基线提升 holdout 预测/压缩和未见组合迁移。
- chunk lesion 显著降低跨边界泛化，但不破坏无损 byte 回退。
- assembly 数量和长度由数据形成，不写数据集词表或答案映射。

## 8. P3：世界状态与工作空间

### 工作项

- 建立实体、属性、关系、事件和 affordance 的分布式动态绑定。
- 引入容量受限、可学习的选择性路由与广播。
- 把预测目标从 next byte 扩展到下一事件、状态变化和行动后果。
- 在对象持续性、关系交换、时间打乱和干预任务上预注册 A2/A3。

### 退出门槛

- 未见组合与干预预测显著优于频率/反应式基线。
- workspace 路由 lesion 和关系绑定 lesion 产生可解释损失。
- 世界状态可从 checkpoint 恢复，不依赖外部 Python 事实表。
- 异质群体的学习型路由在至少一类组合任务上显著优于最强单体、稠密平均和随机/固定路由，形成 A3 的 `1+1>2` 证据。

## 9. P4：多系统记忆

### 工作项

- 分离 working、episodic、semantic、procedural 四类记忆职责。
- 把当前 `EpisodicField` 接入 fast episodic 角色，而非复制成所有记忆。
- replay 从情景重现升级为结构抽取、关系重组和技能压缩。
- 建立新组合、相似事件干扰、来源追踪和长期遗忘评测。

### 退出门槛

- one-shot 情景保持与跨 episode 语义迁移同时成立。
- 删除 episode 身份后仍保留可迁移规律，删除语义巩固后迁移显著下降。
- 10²→10⁴ episode 扩展给出容量、干扰和资源曲线。
- fatigue/curiosity/stress 等 homeostatic state 能在无外部硬编码日程下正确触发探索、专注、睡眠或休息；sleep/play/dream lesion 分别产生预注册的能力损失。

## 10. P5：执行认知

### 工作项

- 建立目标层级、价值、不确定性、冲突和进度状态。
- 世界模型支持带 provenance 的 imagined rollout。
- 规划器比较真实可执行候选，不直接产生漂亮文本作为替代。
- 自我监控预测成功率并触发搜集信息、重规划或请求外部决策。

### 退出门槛

- 在未见目标、延迟奖励和环境干预中优于 reactive policy。
- world-model、rollout、goal-memory 和 value lesion 均有独立效应。
- 内部置信度对外部成功率校准，并能改善资源分配。

## 11. P6：语言与工具生成

### 工作项

- 区分内容计划、表达计划和最终 byte 编码。
- 从 `ActionIntent` 生成语言或结构化工具调用；byte motor 只做末端 codec/回退。
- 训练和评测同时覆盖语义保持、可读性、上下文、工具参数正确性和执行结果。
- 建立 direct-next-byte、无内容规划、无 workspace 和随机表达对照。

### 退出门槛

- 语言输出在人类盲测、语义一致性和多轮目标保持上达到预注册门槛。
- 工具调用不仅格式正确，而且真实执行成功并把 outcome 回写认知状态。
- 生成不依赖 Legacy 路由或外部 teacher 的运行时决策。

## 12. P7：持续进化、多模态与规模化

- 新任务学习后测量旧能力保持、适应速度和结构增长。
- 把异质专门化、结构重连/剪枝、assembly 分化和能力缺口驱动的资源申请变成 Taiji 内生发展过程；结构变化必须可回滚。
- 将好奇、疲劳、压力、安全和资源需求纳入持久 homeostatic state，真实控制探索、睡眠、玩耍和学习预算。
- 增加图像/音频/身体器官，共享世界状态而非晚期答案拼接。
- `CapacityPolicy` 转为资源治理器，支持可塑拓扑和按层预算。
- 对 v1 实际热点做 CPU/CUDA profiler，再决定 fused/sparse kernel。
- 报告吞吐、显存、能耗、数值一致性和 checkpoint 跨设备恢复。

## 13. P8：产品与发布

- Seed UI/API 只展示已通过 Gate 的 Taiji 能力。
- S2 安全、覆盖率和门禁继续完成；S3 打包、版本、更新和回滚在 v1 API 稳定后收口。
- 默认发行不安装 Legacy 重依赖；Legacy 只保留离线对照和显式兼容构建。
- 发布物包含模型卡、数据卡、能力 Gate、失败边界和恢复方式。

## 14. 持续门禁

- Taiji/Seed/Legacy 所有权 AST 测试；
- v1 state/checkpoint schema 和确定性恢复；
- TSK-v8 K 系列回归；
- 当前阶段 A Gate 的 holdout、lesion 和跨 seed 结果；
- 数据 manifest、实验注册、代码 commit 和训练 lineage；
- planned/actual learned state 与资源预算；
- 后端、前端、桌面、Legacy-off 启动和安全门禁。

辅助训练结果必须标记 `native-assisted`；只有不依赖辅助 teacher 决策且能继续终身学习的路径才能标记 `native-local`。A0–A9 的目的追溯和 Gate 定义以 Taiji v1 架构文档为准。

## 15. 停止项

在 P2 通过前：

- 不续跑旧 16M→100M raw-byte 长训；
- 不为 TSK-v8 继续增加认知补丁；
- 不写绑定固定 fan-in 的自定义 CUDA kernel；
- 不用增加神经元数量替代学习型抽象；
- 不删除 Legacy 对照；
- 不把旧 N/M 通过记录宣传为完整智能进展。

## 16. 当前唯一下一步

**实施 P2：在保留 byte 无损回退的前提下，加入学习型局部特征、可变时长 assembly 和时间抽象，并用未见组合 holdout 验证它确实超越 byte-only 基线。**
