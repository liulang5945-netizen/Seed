# Taiji Native Architecture v1

状态：**目标架构与后续实现的最高权威合同**

决策日期：2026-08-25

当前实现状态：尚未完成。仓库顶层 `taiji/` 当前运行的是 Taiji Substrate Kernel v8（TSK-v8），它是本架构可复用的低层实验内核，不是完整 Taiji。

P1 兼容纵切片已完成：版本化 v1 合同、Taiji-owned `CognitiveState`/`NativeCheckpoint`、
TSK-v8 adapter 和 Seed 所有权门禁已经进入代码。该纵切片只证明合同与闭环可恢复，
不改变“TSK-v8 不是完整 Taiji”的能力声明。

P2 已开始：`LearnedPerception` 提供连续局部特征、递归预测误差、可配置的下一观测
预测训练和可变时长 assembly，并随 v1 adapter checkpoint 保存。A1 合同还要求完成
assembly 迁移、边界扰动与 random chunk lesion 证据。加入 future-window assembly
目标后的最新报告显示未见组合 gain 为 `-0.0088`、marker score delta 为 `+0.0222`、
marker rate delta 为 `+0.0098`、random-chunk drop 为 `+0.0077`，仍未达到加强后的
Gate 要求。随后加入自监督 assembly consistency/contrastive 目标，最新结果仍为未见
组合 gain `-0.0225`、marker score delta `+0.0184`、marker rate delta `+0.0401`、
random-chunk drop `+0.0048`，没有超过 Gate。连续目标失败说明问题已从实现细节升级
为 P2 组合关系定义。当前已完成合同重定：`AssemblyRelationCorpus` 以共享 atom、
不重叠 ordered pair、boundary/random controls 作为 A1 relation v1 数据边界，真实
manifest 已生成；pair provenance 只用于评测，不进入模型。小规模 relation subgate
已在三个 seed 上通过，但完整 P2 仍需更大 atom pool、独立语料分区和旧 byte-level
对照；当前仍不能宣称 Taiji 已拥有通过完整 Gate 的学习型抽象能力。

扩展验证已完成：在 `dialogue16` 和 `shared16` 两个独立语料分区、16 atoms、240
ordered-pair 规模上，slot binding、boundary consistency 和 random binding lesion
的跨 seed 最小值分别达到 `+0.9375/0.9841/+0.6875` 与 `+0.9219/0.9811/+0.6406`。
这只关闭结构性 A1 relation subgate，不等于自然语言、世界模型或完整 Taiji 已实现。

## 0. 本次纠正

此前路线把“原生”误解为“从最原始信号开始，并排除一切与 Transformer 功能相似的抽象”。这导致：

- raw byte one-hot 被当作 Taiji 的内部表示，而不只是文本器官的边界编码；
- 固定区域、固定 fan-in、固定群体与单 byte motor 被当作完整架构；
- “没有 tokenizer/embedding/attention/optimizer”成为目标本身；
- N0–N11/M5–M7 的低层机制门槛被误用为智能能力门槛；
- Seed 被定义成模型主体，Taiji 被降成 substrate，认知所有权发生倒置。

纠正后的定义是：

> **Taiji 是完整的原生认知架构和模型。Seed 是承载、训练、服务、产品、设备与插件运行时。**

“原生”只要求 Taiji 自己拥有表征、状态、记忆、学习、推理、目标、行动与生成合同，不依赖 Transformer hidden state、teacher logits 或外部模型在运行时替它思考。它不要求把所有高级信息处理都降回 one-hot、固定突触或单步 byte 预测。

### 0.1 从归档恢复的项目原始目的

新架构不是从一张白纸发明通用 AI，而是服务 [Taiji 核心目标与设计依据](TAIJI_CORE_REQUIREMENTS.md) 中长期有效的 CR-1–CR-10：唯一认知主体、异质协作、自适应激活、结构成长、身体因果闭环、生命调节、睡眠/梦境/玩耍、多系统记忆、自主改进和跨域多模态组合。

核心需求常驻 active；归档只保存原始讨论、实验与旧实现。本文负责把这些“为什么”转换为架构职责和 A0–A9 Gate，不能自行改写项目使命。

## 1. 身份与所有权

```text
Seed project / product runtime                     seed/, api/, frontend/, desktop/
  ├─ distribution, lifecycle, device and resource management
  ├─ dataset, experiment, checkpoint and evaluation orchestration
  ├─ API/UI/plugin/tool/environment adapters
  └─ hosts one Taiji cognitive architecture

Taiji native cognitive architecture               taiji/
  ├─ perception and learned abstraction
  ├─ multi-timescale predictive world state
  ├─ workspace, attention-like routing and working memory
  ├─ episodic, semantic and procedural memory
  ├─ goals, value, reasoning, imagination and planning
  ├─ hierarchical action and communication generation
  └─ developmental + lifetime learning

Taiji Substrate Kernel v8                          current taiji/ implementation
  └─ raw-byte codec + predictive fabric + episodic prototype + byte motor

Legacy NeuroPlex                                  neuroplex/
  └─ frozen Transformer comparison; never enters Taiji cognition
```

认知机制必须落在 `taiji/`。`seed/` 可以调度训练、设备、数据、环境和插件，但不能保存一个隐藏的概念图、规划器、语言模型或 teacher policy，再把结果包装成 Taiji 输出。

## 2. 正向设计原则

### 2.1 能力优先，而不是反 Transformer 优先

Taiji 必须正面回答以下问题：

1. 如何从连续多模态输入中形成可复用的特征、事件、对象和概念？
2. 如何维持带不确定性的世界状态，并预测干预后的变化？
3. 如何在工作记忆中选择、组合和广播当前相关信息？
4. 如何把一次经历沉淀成可迁移的语义和技能？
5. 如何形成目标、比较方案、模拟后果并执行多步计划？
6. 如何把内部意图生成语言、工具调用或身体动作？
7. 如何在训练后继续从真实交互中学习而不灾难性遗忘？

只要这些能力由 Taiji 自己实现，选择性路由、连续 embedding、内容寻址、并行优化或可学习编码都不是身份回退。禁止的是把 Transformer block、外部 teacher 隐状态或 Legacy 决策接进认知路径，而不是禁止有用的计算功能。

### 2.2 原始输入只存在于器官边界

byte、像素、波形、传感器值和工具响应是接口格式，不是认知本体。每种器官必须把原始输入转换成带时间、模态、置信度和来源的 `PerceptEvent`；随后由 Taiji 学习形成更长时间尺度的 assembly、事件与概念。

文本可以从 UTF-8 byte 开始，但必须学习可变长度 chunk、形态、词语、短语和语义结构。允许同时保留 byte 回退通路；禁止把手写词表或固定 tokenizer ID 当作唯一内部真相。

### 2.3 连续状态与多时间尺度

Taiji 保留持续状态和事件驱动更新，但不能只有单一 tick 和记忆 trace。至少区分：

- 感觉时间尺度：毫秒到局部片段；
- 工作时间尺度：当前任务、语句、交互轮次；
- 情景时间尺度：一次经历和跨 episode 关联；
- 语义时间尺度：稳定概念、关系与技能；
- 发展时间尺度：结构成长、巩固与长期价值变化。

### 2.4 稀疏和局部是资源策略，不是表达能力上限

稀疏连接、局部可塑性和事件驱动执行是优先设计，但拓扑必须允许学习、重路由、增长与剪枝。固定 fan-in 可以作为 kernel 的参考执行格式，不能成为 Taiji 永久只能看到固定坐标子集的架构身份。

### 2.5 闭环行动是真实因果，不等于 byte 自回归

语言 byte 回灌只是一个通信器官的闭环。Taiji 的动作必须由目标和世界模型产生，经过环境改变下一观察；语言、工具调用和身体动作都是 `ActionIntent` 的不同器官实现。

### 2.6 站在巨人的肩膀上，而不是从零重造计算机科学

Taiji 的底层逻辑必须围绕项目需求重新组合，但没有必要把成熟方法视为污染。可以直接继承经过验证的数学、算法、工程和训练经验：

- dense/sparse linear algebra、卷积、递归、状态空间模型和图计算；
- learned embedding、位置/时间编码、归一化、残差和门控；
- attention/content addressing 作为选择性路由算子；
- BPE、Unigram、byte fallback 或神经 chunker 作为文本器官候选；
- autograd、optimizer、mixed precision、distributed training 和 CUDA kernel；
- predictive coding、Hebbian/STDP、eligibility trace、reinforcement learning 和 model-based planning；
- ANN/vector search、图索引和数据库作为明确边界下的外部长期知识设施；
- 公开数据集、标准 benchmark 和成熟安全/评测方法。

采纳一个成熟组件不等于照搬其宿主架构。Taiji 可以使用 attention 算子而不是 Transformer 堆栈，可以使用 embedding 而不把固定 token 序列当作唯一世界表示，可以用 optimizer 做发展训练而不放弃终身局部学习，也可以用检索设施而不把检索结果冒充内部记忆。

每个外部机制按四个问题判断：

1. 它解决 Taiji 哪个明确能力缺口？
2. 它的状态和决策是否由 Taiji 拥有、可保存、可损伤、可替换？
3. 移除它后损失能否通过预注册实验测量？
4. 它是否把 Transformer/外部模型重新变成运行时认知主体？

前三项有明确答案且第四项为否，就应优先复用成熟实现；只有现有方法违反 Taiji 的持续状态、因果闭环、终身学习或资源目标时，才设计新机制。

## 3. 完整架构

```text
Observation streams
  text / image / audio / tool / body / environment
                 │
                 ▼
L0 Organ adapters and raw codecs
                 │  Observation
                 ▼
L1 Learned perceptual hierarchies
  features → variable-duration assemblies → events
                 │  PerceptEvent
                 ▼
L2 Multi-timescale predictive dynamics
  persistent latent state + prediction error + uncertainty
                 │
        ┌────────┴────────┐
        ▼                 ▼
L3 Workspace          L4 Memory system
  selective routing     working / episodic / semantic / procedural
  binding / focus        consolidation / retrieval / forgetting
        └────────┬────────┘
                 ▼
L5 World and self model
  entities / relations / causes / affordances / self-state
                 │
                 ▼
L6 Executive cognition
  goals / value / reasoning / imagination / planning / monitoring
                 │  ActionIntent / CommunicationIntent
                 ▼
L7 Hierarchical decoders and effectors
  language / tools / body / environment action
                 │
                 └────────────── environment feedback ──────────────┐
                                                                    └→ L0
```

所有层都属于 Taiji。它们可以以多个异步区域实现，不要求七个顺序神经网络层；上述分层定义的是职责和可验证接口。

`Homeostatic/Developmental Regulation` 是贯穿 L1–L7 的横向系统：维持需求、唤醒、疲劳、压力、好奇、可塑性、睡眠/玩耍模式和结构预算。它不能被放回 Seed scheduler 作为外部脚本。

## 4. 核心状态合同

Taiji v1 的完整认知状态至少包含：

| 状态 | 含义 | 关键约束 |
|---|---|---|
| `PerceptState` | 当前多模态特征、assembly 与事件边界 | 由数据学习，不以 byte one-hot 作为全局表示 |
| `PredictiveState` | 多时间尺度隐状态、预测与不确定性 | 可持续推进，可被真实观察纠正 |
| `WorkspaceState` | 当前焦点、绑定对象、任务上下文与广播内容 | 容量受限，路由可学习且可 lesion |
| `WorldState` | 实体、关系、事件、因果与 affordance 的分布式表示 | 支持未见组合与反事实预测 |
| `MemoryState` | 工作、情景、语义和程序性记忆 | 快写慢学，检索与巩固可独立损伤 |
| `GoalState` | 目标层级、约束、价值、进度和未决承诺 | 不由 UI prompt 临时替代 |
| `PlanState` | 候选行动、模拟轨迹、风险和选择依据 | 行动前可评估，结果后可归因 |
| `SelfState` | 能力、置信度、资源、历史和当前身体/工具状态 | 内部判断必须预测外部结果 |
| `HomeostaticState` | 好奇、疲劳、压力、安全、资源需求和唤醒模式 | 必须影响真实行为与学习，不是 UI 数字 |
| `DevelopmentState` | 专门化、结构成长/剪枝、技能成熟度和能力缺口 | 结构变化受预算、证据和回滚约束 |
| `LearningState` | eligibility、可塑性资源、发展阶段和结构预算 | checkpoint 可恢复且更新可审计 |

状态之间通过版本化事件合同通信，不共享无定义的大向量。接口类型是工程合同；其内部特征、assembly 和关系必须由学习形成。

## 5. 各层设计

### 5.1 L0：器官适配器

- 文本器官保留 byte codec 作为无损输入输出边界。
- 图像、音频、工具和身体状态使用各自的采样与时间同步合同。
- 所有观察携带 `modality/timestamp/source/provenance/confidence`。
- 器官不负责概念理解，不调用隐藏的外部语言模型替 Taiji 解释输入。

### 5.2 L1：学习型感知与时间抽象

- 从局部预测稳定性、重复、边界惊讶和跨模态共现中形成 assembly。
- assembly 可跨可变时长，并允许组合成更高层事件；不存在固定“一 byte 一认知 tick”的永久限制。
- 训练必须证明学到的表示能迁移到未见序列和未见组合，而不只提升训练 byte accuracy。
- byte、像素等低层通路保留为纠错和精确重建支路。

### 5.3 L2：多时间尺度预测动力学

- 当前 predictive fabric 可作为快速动力学候选内核。
- 区域需要显式不确定性、跨时间尺度预测和可学习路由。
- 预测对象从“下一个 byte”扩展为下一特征、事件、状态转移和行动后果。
- 局部误差是主要学习信号之一，但不能要求所有长期 credit 都在一个 tick 内解决。

### 5.4 L3：工作空间与选择性路由

- 通过竞争、门控、共振或内容寻址选择当前相关 assembly。
- 支持对象—属性—关系的临时绑定和跨区域广播。
- 这是 Taiji 自己的动态路由机制；不复制 Transformer block，也不因其功能类似 attention 就禁止它存在。
- 工作空间容量、路由稀疏度和维持时长必须可测、可损伤、可按资源预算调整。

### 5.5 L4：多系统记忆

- `working`：维持当前推理变量、目标和未完成操作。
- `episodic`：一次经历的时序、来源、行动与结果，支持 one-shot 写入。
- `semantic`：跨经历提取稳定概念、关系、因果和统计结构。
- `procedural`：可执行技能与策略，不等同于答案缓存。
- replay/sleep 负责重组和巩固，但必须通过新组合迁移证明形成了语义，而不是复制情景。

当前 `EpisodicField` 只作为 fast episodic 原型，不再代表完整记忆系统。

### 5.6 L5：世界模型与自我模型

- 世界状态表达实体、持续性、关系、事件、因果、空间/时间和可行动性。
- 表示可以是稀疏分布式 assembly、动态关系场或混合结构；不能依赖外部 Python 字典作为真正认知记忆。
- 自我模型维护能力、资源、工具、身体、置信度和行为后果，不是固定 persona 文本。
- 必须能回答“若执行动作 A 会发生什么”，并在干预数据上修正因果预测。

### 5.7 L6：执行认知

- 目标形成：从内在需求、用户任务和环境约束产生目标层级。
- 推理：在 workspace 中组合世界状态、记忆和约束。
- 想象：让世界模型无外部动作地展开候选轨迹，并明确标记 imagined provenance。
- 规划：比较候选轨迹的价值、风险、成本和可逆性。
- 监控：预测成功率，检测冲突、失败和需要外部信息的边界。

推理不以“输出一段看似合理的 byte”作为完成；内部计划必须能预测实际行动结果。

### 5.8 L7：生成与行动

- `ActionIntent` 先描述目标、对象、参数、约束和预期结果，再由器官编码为 byte、工具调用或身体动作。
- 语言生成至少区分内容规划、表达规划和最终 byte 编码。
- byte motor 可保留为最末端 codec/回退器官，不能继续直接承担全部认知输出。
- 每次执行都生成可追踪的 pending action，真实 outcome 回写世界模型、记忆和 credit assignment。

### 5.9 横向系统：异质协作、生命调节与发展

- Taiji 的基本计算群体允许异质：不同感受野、时间尺度、学习规则、能力画像和资源成本。
- 专门化不是预先写死 `zh/en/code/math` 域标签，而是由经验、路由使用、预测优势和目标贡献形成；可用成熟 MoE/router 方法作为初始实现。
- workspace 必须证明协作产生单一群体不具备的能力，而不只是加权平均更高分。
- homeostatic state 决定何时探索、专注、休息、睡眠、玩耍或请求资源；调节作用进入路由、学习率/可塑性、阈值、记忆写入和行动价值。
- development state 依据持续能力缺口和资源预算触发 assembly 分化、连接增长/剪枝、区域扩容或技能固化；任何结构变化都要可回滚和跨 seed 验证。
- sleep 负责重放、去干扰和语义巩固；dream 运行带 `imagined` provenance 的世界模型；play 主动选择高信息增益但安全的行为。三者不是普通文本生成模式。

### 5.10 v1 首选参考技术栈

以下是首个可执行版本的唯一推荐基线，不是要求每个机制永久固定。替换必须用同一 Gate 证明更好。

| 能力 | v1 首选成熟机制 | Taiji 化方式 |
|---|---|---|
| 文本边界 | UTF-8 byte fallback + trainable byte embedding + causal convolution/SSM patcher | 不从 one-hot 直接进入全脑；学习可变时长 patch。SentencePiece/Unigram 分段只可作为 `native-assisted` 辅助视图和基线，不是唯一内部真相 |
| 图像/音频感知 | patch/conv encoder 与时频前端 | 每种器官形成 `PerceptEvent`，在共享事件/世界空间对齐，不直接拼接最终答案 |
| 持续动力学 | gated recurrent state-space blocks（SSM）+ TSK predictive dynamics 对照 | 保留跨 tick 状态和多时间尺度；用 holdout/lesion 决定 TSK 局部预测内核是否进入正式区域 |
| 专门化与协作 | sparse MoE-style experts/regions + learned router + load-balance/协作增益目标 | expert 是 Taiji-owned 专门化区域，不是完整 Transformer 成员；路由同时读取内容、不确定性、目标和资源 |
| 工作空间 | 少量 latent slots + content-addressed cross-attention/gating | attention 是路由算子，不是堆叠 Transformer 身份；容量受限、可持续、可 lesion |
| 世界模型 | object/event-centric latent state + relational graph dynamics | 维护实体持续性、关系和干预后果；图结构由观察学习，不用外部事实字典代替 |
| 情景记忆 | 持久 event log + learned embedding/index + ANN 检索 | 数据可落盘/索引，但事件、provenance、检索 query 和使用决策属于 Taiji；TSK `EpisodicField` 作为无外部槽对照 |
| 语义/程序记忆 | replay consolidation + prototype/graph/weight updates | 从多经历提取规律和技能，必须在新组合上优于情景复读 |
| 规划 | latent world-model rollout + goal-conditioned actor/value；必要时有限 beam/tree search | imagined state 明确标记，不把语言 chain-of-thought 当世界模拟；真实 outcome 校准 model/value |
| 语言效应器 | 内容计划条件化的 compact SSM 或 Transformer decoder | 允许成熟 decoder 作为器官；它不能拥有世界模型、目标和最终行动决策，移除后内部 ActionIntent 仍成立 |
| 发展训练 | PyTorch autograd/optimizer、mixed precision、distributed/CUDA | 用于形成 Taiji-owned 参数；训练模式和 teacher/辅助信号透明标记 |
| 终身学习 | eligibility trace、neuromodulation、episodic write、replay、局部/低秩快速可塑参数 | 在线适应不要求全模型 BPTT；长期 credit 可跨事件保存，睡眠期再整合到慢参数 |
| 结构进化 | router usage/能力缺口/干扰/资源共同驱动的 expert split/merge、edge grow/prune | 初期由安全控制器提出并回滚，成熟后逐步内生；禁止按预设领域名直接创建专家 |

这套基线刻意吸收 Transformer、SSM、MoE、图网络、向量检索和 model-based RL 的成熟成果，但认知中心是 Taiji 的持续世界状态、异质协作、生命调节与发展闭环。

## 6. 学习体系

Taiji 同时拥有两条学习平面。

### 6.1 发展训练

用于形成感知层级、世界模型、语义记忆、语言器官和通用技能。允许批处理、并行模拟和 CUDA；可使用 autograd/optimizer 作为阶段性实验工具，但必须满足：

- 不以 Transformer hidden state、teacher logits 或外部模型决策作为 Taiji 的运行时依赖；
- 报告明确区分 `native-local`、`native-assisted` 和 `evaluation-only`；
- assisted 初始化不能被冒充为 Taiji 已具备自主终身学习能力；
- 每个辅助训练机制都必须有移除后果和迁移到原生学习规则的计划。

因此，“是否使用 optimizer”不再是身份判据；真正判据是认知能力是否由 Taiji 参数与状态承载，以及离线训练后能否继续原生适应。

### 6.2 终身学习

运行时以预测误差、局部 eligibility、奖励/新颖性/不确定性调制、结构可塑性和 replay 为主：

- 快速局部学习适应当前环境；
- 情景记忆 one-shot 保存真实经历；
- 语义巩固跨经历抽取稳定结构；
- 程序性学习把成功计划压缩为技能；
- homeostasis、遗忘和结构预算防止无界增长。

长期 credit assignment 可以跨 tick、episode 和 imagined rollout 保存 eligibility；“局部”不等于“只能看相邻一个 tick”。

## 7. 输入、输出与智能形成

Taiji 的智能不由某个单独神经元或权重产生，而来自四个闭环同时成立：

1. **表征闭环**：感知误差促使系统形成能压缩和预测输入的 assembly/概念。
2. **世界闭环**：行动改变环境，真实结果纠正因果世界模型。
3. **记忆闭环**：经历被检索、重组并巩固成可迁移的语义和技能。
4. **目标闭环**：目标驱动信息选择与计划，结果反过来修正价值、自我判断和策略。

输出不是直接从输入映射出的答案，而是：

```text
当前世界状态 + 相关记忆 + 目标/约束
  → workspace 组合与候选推理
  → imagined outcomes
  → plan/action intent
  → language/tool/body decoder
  → environment outcome
  → learning
```

## 8. 硬编码治理

### 8.1 允许固定

- 序列化格式、dtype、协议版本、因果事务顺序和安全边界；
- 物理接口事实，例如 byte 值范围、图像通道和设备能力；
- 为可复现实验声明的初始资源上限。

### 8.2 必须由学习或资源治理决定

- assembly、概念、关系、技能和语义边界；
- 路由权重、记忆索引、目标优先级和行为策略；
- 可塑拓扑的增长、重连、剪枝和区域资源分配；
- 各时间尺度的有效跨度和工作空间内容。

### 8.3 `CapacityPolicy` 的新角色

`CapacityPolicy` 只做资源治理：根据参数、内存、设备和延迟预算给出上限。它不再规定 Taiji 必须有三层固定比例区域，也不把 fan-in 比例、memory unit 数或 257 维 motor 当作认知结构。

### 8.4 禁止项

- 数据集问题到答案的固定映射；
- cue 专属永久槽、隐藏 prompt、样例分支；
- 通过改阈值掩盖失败机制；
- 把 Python list/dict 中的外部记忆称为 Taiji 学习；
- 为了“非 Transformer”而禁止必要的可学习抽象或选择性路由。

## 9. 执行与 CUDA 边界

CUDA 适配围绕架构算子，而不是围绕旧固定 fan-in 代码永久优化：

- 稀疏 gather/scatter 与局部可塑性；
- 事件批处理和异步多时间尺度更新；
- 动态路由、assembly 激活和 segmented reduction；
- 记忆检索、重放和并行 imagined rollout；
- CPU/CUDA 状态与 checkpoint 一致性。

在 L1–L7 的最小纵切片证明之前，不写绑定旧 TSK-v8 拓扑的自定义 CUDA kernel。

## 10. 当前代码的重新定位

| 当前组件 | 保留价值 | v1 定位 |
|---|---|---|
| `ByteSensor` | 无损文本边界 | L0 text codec，不再是全局表示 |
| `TaijiFabric` | 持续预测状态与局部误差实验 | L2 快速动力学候选 kernel |
| `EpisodicField` | one-shot 分布式情景实验 | L4 fast episodic 原型 |
| `ByteMotor` | byte 概率与因果 pending action | L7 最末端 text codec/回退器官 |
| `Taiji` 当前类 | 可执行闭环与 checkpoint 基线 | `TSK-v8` compatibility adapter |
| N0–N11/M5–M7 | kernel 因果与回归证据 | K 系列 kernel 回归，不再作为智能进展门槛 |

旧精确方程与张量合同保存在 [TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md](../archive/implementation/TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md)。继续维护它只允许修复回归、checkpoint 兼容和作为新架构的候选内核对照，不再在其上直接叠加“概念模块”“推理模块”补丁。

## 11. 能力反证门槛

| Gate | 必须证明 | 关键对照 |
|---|---|---|
| A0 所有权 | Seed 不承载隐藏认知；Taiji 可独立完成认知纵切片 | 移除 Seed 业务逻辑后能力不变 |
| A1 学习型抽象 | 从 byte/感知流形成可变时长 assembly，迁移到未见组合 | 固定 byte one-hot、随机 chunk、无 chunk lesion |
| A2 世界状态 | 保持对象/事件并预测行动干预结果 | 频率基线、无关系绑定、时间打乱 |
| A3 自适应协作 | 异质群体按不确定性/能力动态路由，协作完成单群体不能完成的任务 | 最强单体、稠密平均、随机/固定路由、无 workspace |
| A4 情景→语义 | 多次经历形成可迁移概念/关系，而非情景复读 | replay lesion、episode ID lesion、新组合 holdout |
| A5 生命调节 | fatigue/curiosity/stress 等内生状态正确改变探索、学习、睡眠和资源使用 | 固定调度、随机 drive、无调质、sleep/play lesion |
| A6 目标与规划 | 未见目标上 imagined rollout 改善真实成功率，自评能预测结果 | reactive policy、随机 rollout、world/self-model lesion |
| A7 原生生成 | 内部意图稳定生成可读语言/工具动作并保持语义 | 直接 next-byte kernel、内容规划 lesion |
| A8 持续进化 | 新任务后保留旧能力，并根据缺口完成可回滚的分化/增长/剪枝 | frozen topology、随机增长、无 replay/homeostasis |
| A9 多模态具身 | 不同器官共享世界状态，行动改变感知并产生跨模态迁移 | 单模态、晚期答案拼接、外部编码器/Agent 替代 |

每个 Gate 必须预注册数据、随机种子、指标、损伤组和失败条件。K 系列 kernel 测试继续全绿只是进入实验的前提，不是 Gate 通过。

## 12. 目标代码边界

迁移完成后的逻辑布局：

```text
taiji/
  contracts/       Observation, PerceptEvent, Goal, Plan, Action, state protocol
  perception/      modality organs, learned chunking and assemblies
  dynamics/        multi-timescale predictive state and routing
  workspace/       focus, binding and broadcast
  memory/          working, episodic, semantic, procedural
  world/           entity/event/relation/causal/self models
  executive/       goals, reasoning, imagination, planning and monitoring
  learning/        local plasticity, modulation, consolidation, structure policy
  effectors/       language, tool and body decoders
  kernel_v8/       current implementation during compatibility migration
```

目录不是一次性重命名任务。先建立合同和最小纵切片，再逐项迁移，避免只有空模块和漂亮分层。

## 13. 当前唯一实现入口

P1 已完成，P2 relation subgate 已收口。P3 的对象/事件/affordance/行动/结果合同、可恢复 `TaijiWorldState`、结构化对象/关系/时间打乱、多步 episode 窄 Gate、`TSKV8Adapter` transition lineage、runtime prediction record、error-driven online correction 和最小 WorkspaceRouter 已落地；A3 静态组合与 `assemble → commit` world-outcome 窄 Gate 已通过，P4 working/episodic memory 最小真实经历、cue-conditioned one-shot recall、additive semantic consolidation、multi-factor/noisy semantic Gate、容量/干扰曲线、standalone procedural consolidation、procedural runtime ownership、多步 procedural robustness 和 homeostatic/sleep-play Gate 已通过，P5 单步 goal-planning、imagined rollout/replan trigger 和实际 replan/calibration Gate 已通过，下一入口是 delayed reward、环境干预与 reactive/value/world-model lesion。当前结果仍是小型数值关系/两步/保持检索能力，不等于一般语义巩固、长程规划或通用智能。
