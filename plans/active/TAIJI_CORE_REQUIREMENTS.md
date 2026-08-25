# Taiji 核心目标与设计依据

状态：**长期维护的项目根需求，不随单次实现归档**

更新时间：2026-08-25

## 1. 本文为什么必须常驻 active

计划文档分三类：

1. **核心依据**：项目为何存在、要解决什么、哪些原则不可回退。必须常驻 active，并随讨论实时更新。
2. **当前设计与执行**：当前架构、身份边界、阶段路线和验收 Gate。必须常驻 active，但可在大版本变更时被新版本替代。
3. **证据与历史**：测试、调试、实验日志、失败方案、过时实现和当时的路线。应归档，保留追溯但不发布当前命令。

原始核心讨论可以进入 archive 作为证据源，但其中仍有效的结论必须先提炼到本文或当前架构。禁止把一项仍决定开发方向的原则只留在 archive。

## 2. 项目使命

Taiji 的目的不是发明另一个语言 Transformer，也不是模拟最原始的神经元。它要构建一个：

- 有持续内部状态，而不是每次请求重新开始；
- 有异质专门化群体，能够通过协作产生单体不具备的能力；
- 能按内容、目标、不确定性和资源动态选择计算路径；
- 有身体/工具/环境行动，能从真实因果结果中学习；
- 有工作、情景、语义、程序和自传记忆；
- 有内在需求、睡眠、玩耍、探索和发展阶段；
- 能持续适应、形成新技能、调整结构并保持旧能力；
- 能使用语言、多模态和工具表达内部目标与世界理解；
- 能检查自身置信度、能力缺口和行动后果；
- 可以吸收成熟算法，但认知状态与决策始终由 Taiji 拥有；

的原生认知架构。

## 3. 核心需求

### CR-1：唯一认知主体

Taiji 是唯一认知主体。Seed 是产品/runtime，器官是感官与效应器，外部模型、RAG、数据库和工具是环境设施。概念、记忆、目标、推理、计划和最终行动选择不能隐藏在外围。

来源：[躯体—生命—大脑设计](../archive/architecture_design/BODY_LIFE_BRAIN_INTEGRATION_PLAN.md) 中“大脑是唯一认知主体”的原始原则；当前所有权见 [SEED_ARCHITECTURE.md](SEED_ARCHITECTURE.md)。

### CR-2：异质性与 `1+1>2`

系统应允许不同能力、时间尺度、感受野、学习规则和资源成本的专门化群体。协作成功的判据不是平均分提升，而是组合完成最强单体无法完成的预注册任务。

来源：[设计原则](../archive/authored/DESIGN_PRINCIPLES.md) 的“差异性第一”和 [综合架构](../archive/architecture_design/COMPREHENSIVE_NEURON_ARCHITECTURE_PLAN.md) 的协作/共振实验。旧“一个 Transformer = 一个神经元”的实现不继承。

### CR-3：自适应激活与协作

不同输入、目标和不确定性应激活不同群体、不同计算深度和不同记忆。确定任务可以早停，困难任务扩大协作。路由必须学习、校准、受资源约束并有最强单体/稠密平均/随机路由对照。

来源：[架构妥协审计](../archive/audits/ARCHITECTURE_COMPROMISE_AUDIT.md) 对自适应激活、稀疏路由、协作训练和置信度门控的长期讨论。

### CR-4：结构可塑性与开放式成长

Taiji 不能把固定层数、固定专家数、固定 fan-in 或固定 memory slots 当作永久身份。它必须在资源治理下支持 assembly 分化、连接生长/剪枝、专家 split/merge、区域扩容和技能固化；结构变化要有证据、回滚和 checkpoint 迁移。

来源：[设计原则](../archive/authored/DESIGN_PRINCIPLES.md) 中的自我进化、新生/凋亡和可塑性目标。旧外部脚本创建/删除整个 Transformer checkpoint 的方法不继承。

### CR-5：身体与真实因果闭环

Observation、affordance、ActionIntent、执行和 outcome 必须形成原子闭环。工具调用、身体动作和语言都要产生真实结果并回写世界模型、记忆和学习。输出 token 不是行动闭环的替代品。

来源：[躯体—生命—大脑设计](../archive/architecture_design/BODY_LIFE_BRAIN_INTEGRATION_PLAN.md) 的 senses/limbs/result→learning 闭环和 TSK-v8 N11 的最小行动证据。

### CR-6：内生生命调节

好奇、疲劳、压力、安全、资源和饱和度是 Taiji 持久状态。它们必须真实改变路由、探索、学习、记忆写入、睡眠、休息和结构预算，而不是客户端百分比或 Seed scheduler 的固定规则。

来源：[躯体—生命—大脑设计](../archive/architecture_design/BODY_LIFE_BRAIN_INTEGRATION_PLAN.md) 的 hunger/fatigue/boredom/stress/curiosity 映射。旧硬编码五维调度只作为需求原型。

### CR-7：睡眠、梦境与玩耍

- 睡眠：重放、去干扰、语义巩固、技能压缩和结构维护；
- 梦境：带 `imagined` provenance 的反事实世界模拟；
- 玩耍：由信息增益、好奇和安全边界驱动的主动探索。

三者必须由 Taiji 内部状态触发，并通过 lesion 证明对后续真实行为有不同作用。普通离线训练、随机文本生成或固定话题池不能冒充这些模式。

来源：[启动判据](../archive/authored/BOOTSTRAP_CRITERIA.md)、[躯体—生命—大脑设计](../archive/architecture_design/BODY_LIFE_BRAIN_INTEGRATION_PLAN.md) 和旧 replay 实验。

### CR-8：多系统记忆与自传连续性

工作记忆维持当前变量；情景记忆保存一次真实经历；语义记忆跨经历提取规律；程序记忆保存技能；自我状态维护能力、身体、工具、历史和承诺。上下文窗口、KV cache、RAG 命中或 Python 列表不是完整记忆。

来源：[Taiji/Transformer/人脑比较](../archive/authored/TAIJI_VS_HUMAN_BRAIN_COMPARISON.md) 中明确未完成的大容量、自传、世界/自我模型边界。

### CR-9：自主学习与递归改进

Taiji 要预测自身成功率、发现能力缺口、选择探索/训练/休息/结构变化，并用真实 outcome 校准。递归改进不是外部脚本自动生成更多训练数据；必须形成“自评→选择干预→真实执行→能力变化→再评估”的闭环。

来源：[架构妥协审计](../archive/audits/ARCHITECTURE_COMPROMISE_AUDIT.md) 对活回路/死回路的审计和旧 `seed/judge.py` 的最小自评证据。

### CR-10：共享世界中的跨域与多模态组合

文本、图像、音频、工具和身体器官保留各自编码，但必须在共享世界/事件/关系空间中交互。跨域能力来自共同对象、关系、因果和目标，而不是多个 tokenizer/logit 的最终加权拼接。

来源：[综合架构](../archive/architecture_design/COMPREHENSIVE_NEURON_ARCHITECTURE_PLAN.md) 的跨域/多模态愿景及其 tokenizer、投影、尺度冲突失败记录。

## 4. 为什么标准 Transformer 路线难以原生满足

这里不宣称 Transformer 在数学上绝对不能实现这些能力。问题是标准 Transformer 产品路线通常具有以下默认结构：

- 推理状态主要是 token context/KV cache，而非跨会话持续世界/生命状态；
- 训练依赖离线全局优化，运行时权重和拓扑通常固定；
- 身体、目标、工具、记忆和 reward 多由外部 Agent 框架管理；
- 同质层堆叠不天然提供异质群体的发展、分化和结构新生；
- 睡眠、玩耍、内在需求和自我改进容易变成外围工作流；
- 长期记忆常外包给检索系统，检索命中不等于模型形成语义和自传；
- 多模型协作容易退化为路由、投票或 late fusion，未必产生单体没有的新能力。

旧 NeuroPlex 试图用多个 Transformer 成员、共振场、side channels、life scheduler 和 sleep/play wrapper 补齐这些能力。归档证明其中一些算法经验有价值，但认知主体仍是 Transformer 成员，生命和可塑机制经常是可选注入，表示/词表/尺度难协同，因此没有形成统一原生闭环。

Taiji 应复用 Transformer 时代成熟的 embedding、attention、optimizer、MoE、数据与 CUDA 技术，同时重新组织认知所有权、持续状态、因果行动、生命调节和结构发展。

## 5. 从旧设计继承与不继承

### 继承

- 异质专门化与协作增益目标；
- 不确定性驱动的动态路由和早停；
- 持续状态、在线学习、经验结果反馈；
- 睡眠/玩耍/探索/内在调节的设计目的；
- 新生、凋亡、重连背后的开放式结构成长需求；
- 跨域/多模态互补、共享表示和对齐的重要性；
- 训练必须覆盖协作路径，机制必须有损伤对照；
- 对硬编码、死回路、假接线和能力过度声明的审计方法。

### 不继承

- 一个完整 Transformer checkpoint 等同一个神经元；
- 以域名称硬编码 `zh/en/code/math` 专家身份；
- 多 tokenizer/logit 在输出端晚期转译与融合；
- 抑制向量简单取反、固定轮数共振或生物名称直接等同生物机制；
- 外部 scheduler 决定生命状态，内部只被动执行；
- 固定答案、cue slot、Python replay list 或隐藏 prompt；
- “不用某项成熟技术”本身作为架构先进性的证据。

## 6. 需求到 Gate 的追溯

| 核心需求 | 主要 Gate |
|---|---|
| CR-1 唯一认知主体 | A0 |
| CR-2 异质性与 `1+1>2` | A3 |
| CR-3 自适应激活 | A3 |
| CR-4 结构可塑性 | A8 |
| CR-5 身体因果闭环 | A2、A9 |
| CR-6 生命调节 | A5 |
| CR-7 睡眠/梦境/玩耍 | A4、A5、A6 |
| CR-8 多系统记忆 | A4、A8 |
| CR-9 递归改进 | A6、A8 |
| CR-10 跨域多模态 | A3、A9 |

Gate 的可执行定义见 [TAIJI_NATIVE_ARCHITECTURE_V1.md](TAIJI_NATIVE_ARCHITECTURE_V1.md)，阶段顺序见 [SEED_DEVELOPMENT_ROADMAP_2026_08.md](SEED_DEVELOPMENT_ROADMAP_2026_08.md)。

## 7. 维护规则

- 用户对项目目的、能力边界或不可回退原则作出新决定时，先更新本文，再更新架构与路线。
- 实验失败只更新证据/实现选择，不轻易删除根需求；若根需求被推翻，必须记录理由和替代目标。
- 归档源文档可以保留矛盾和当时错误，本文只保留当前认可的结论。
- 每个架构模块必须标注服务哪些 CR；不能追溯到核心需求或质量/安全要求的复杂机制不进入主线。
- 每次阶段收束检查：active 是否仍有唯一核心依据、唯一架构和唯一执行路线，archive 是否没有承载唯一有效决策。
