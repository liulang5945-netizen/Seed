# Seed / Taiji 架构方向决策

> 初始决策：2026-08-21
>
> 重大修订：2026-08-25
>
> 当前决策：**Taiji 是完整原生认知架构；Seed 是项目、产品和运行时；TSK-v8 是可复用 kernel，不是完整 Taiji。**

项目长期目的与不可归档的根需求见 [TAIJI_CORE_REQUIREMENTS.md](TAIJI_CORE_REQUIREMENTS.md)。本文只维护身份、技术采纳和不可回退边界。

## 0. 规范词表

| 规范名 | 指代 | 当前代码/文档事实 |
|---|---|---|
| **Seed** | 项目、产品、分发与运行时 | `seed/`、`api/`、`frontend/`、`desktop/`；承载 Taiji，不拥有隐藏认知 |
| **Taiji** | 完整原生认知架构与模型 | 目标合同为 [TAIJI_NATIVE_ARCHITECTURE_V1.md](TAIJI_NATIVE_ARCHITECTURE_V1.md)；当前尚未完整实现 |
| **Taiji Substrate Kernel v8（TSK-v8）** | 当前可执行低层研究 kernel | 顶层 `taiji/` 现有 byte/fabric/memory/motor 实现；精确历史规范见归档 |
| **Legacy NeuroPlex** | 冻结的 Transformer 基线 | `neuroplex/`；只用于离线比较和显式兼容，不进入 Taiji cognition |
| **`taiji.*` 历史别名** | Legacy NeuroPlex 的旧 pickle/import 路径 | 只在受控兼容和 `scripts/archive/` 中解释，不指当前 Taiji |
| ~~态极~~ | Legacy NeuroPlex 的旧中文称呼 | 冻结历史可保留，新代码与新文档不使用 |

“Taiji Predictive Fabric（TPF）”从完整架构名降为 TSK-v8 中 predictive dynamics 的历史/候选内核名，不再代表 Taiji 全部。

## 1. 被纠正的旧决策

以下旧表述失效：

- “Seed 是模型主体，Taiji 是底层 substrate”；
- “raw-byte sensor → predictive fabric → episodic field → byte motor 是完整 Taiji”；
- “Taiji 必须禁止 tokenizer、embedding、attention、optimizer 和 autograd”；
- “固定稀疏连接、局部单 tick 更新和单 byte motor 是不可回退身份”；
- “N/M kernel 机制通过等于 Taiji 智能能力前进”。

错误根源是用“反 Transformer”定义 Taiji，而没有用项目需要的认知能力正向定义 Taiji。这会让系统越独立越原始。

## 2. 不可回退的新边界

1. Taiji 拥有感知表征、持续状态、工作空间、世界/自我模型、记忆、目标、推理、规划、行动、生成和学习。
2. Seed 只拥有产品/runtime/训练与设备调度，不得实现隐藏认知后把结果包装成 Taiji。
3. raw byte、像素和波形只属于器官边界；Taiji 必须学习高层 assembly、事件、概念和关系。
4. Taiji 可以复用成熟算法和工程，包括 embedding、attention-like routing、SSM、图计算、optimizer、RL、检索和 CUDA。
5. 复用成熟组件不能把 Transformer/Legacy/外部 teacher 重新变成运行时认知主体。
6. 终身学习、真实行动因果和 checkpoint 自足继续是原生性的核心要求。
7. TSK-v8 作为 compatibility/kernel 基线冻结扩张，只做回归、兼容和候选算子验证。
8. Legacy NeuroPlex 继续冻结；Taiji 不导入它，它也不反向依赖 Taiji。

## 3. 原生性的判据

一个能力可以称为 Taiji 原生能力，当且仅当：

- 状态与参数属于 Taiji checkpoint；
- 输入来自版本化 Observation/Memory/Goal 合同；
- 决策由 Taiji 内部状态产生，而不是外部模型返回；
- 结果能通过真实 outcome 进入 Taiji 学习；
- 能力可被独立损伤、替换和测量；
- Seed/Legacy 被移除后，认知纵切片仍可运行。

因此，使用 PyTorch optimizer 训练 Taiji-owned encoder 不自动破坏原生性；运行时调用 Legacy LM 生成计划则一定破坏原生性。

## 4. 站在巨人肩膀上的采纳规则

优先采用成熟方法，只在证据表明其不满足目标时自研。每项采纳必须记录：

| 问题 | 必须回答 |
|---|---|
| 能力缺口 | 它解决感知、记忆、路由、规划、生成或执行中的什么问题？ |
| 所有权 | 参数、状态和决策是否在 Taiji 内？ |
| 因果证据 | 移除/替换后有什么可测损失？ |
| 运行依赖 | 是否需要 Transformer/teacher 在运行时继续思考？ |
| 资源边界 | CPU/CUDA、内存、延迟和扩展行为是否符合产品目标？ |

允许“借算法”，禁止“借认知主体”。Taiji 的创新可以来自系统级组织、持续学习和因果闭环，不要求每个矩阵运算都从零发明。

## 5. TSK-v8 的保留边界

当前 `taiji/` 已证明的事实仍然有效：

- 不导入 `seed`、`neuroplex` 或 `transformers`；
- 有持续状态、局部预测更新、情景原型和 pending action/outcome；
- checkpoint、CPU/CUDA device 语义与 N0–N11/M5–M7 kernel 回归可复现；
- raw-byte codec 可无损接入文本流。

但它只证明一个非 Transformer 低层闭环能运行，不证明学会语言、概念、世界模型、推理或 AGI。旧精确状态方程和门槛保存在 [TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md](../archive/implementation/TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md)。

P1 前不大规模搬动当前源码；先加 v1 合同与 adapter，再决定哪些 kernel 组件进入 perception/dynamics/memory/effectors。

## 6. Transformer 与 Legacy 边界

Taiji 不以“逐功能替换 TransformerBlock”为设计主轴。Transformer 仅是一个离线比较对象。

当前被替代的 Legacy `TransformerBlock` live 消费点继续封闭为三处：

| 消费点 | 性质 |
|---|---|
| `neuroplex/resonance/neuron.py` | Legacy 基线内部 |
| `scripts/training/train_tinystories.py` | 离线 Transformer 对照 |
| `scripts/training/train_tinystories_field.py` | 离线 field 对照 |

允许研究 attention、embedding 或 optimizer，不允许新增对 `neuroplex/layers.py::TransformerBlock` 的正式消费者。前者是复用成熟算法，后者是把被替代的认知主体重新接回产品。

## 7. 包与 checkpoint 迁移

- `taiji/` 继续作为正式认知命名空间。
- 当前 `Taiji` 类和 `taiji-native-v8` checkpoint 进入 TSK-v8 compatibility line。
- v1 建立新的认知 state/envelope，明确嵌套或迁移 kernel payload，不静默猜测格式。
- `seed-native-v1` 在过渡期继续可读；最终 Seed envelope 只保存产品元数据和完整 Taiji checkpoint 引用/载荷。
- 历史 pickle alias 继续由受控兼容代码处理，不能污染 `sys.modules` 中的正式 Taiji。

## 8. 能力声明

当前可以声明“Taiji v1 目标架构已定基线，TSK-v8 kernel 可执行”。不能声明“完整 Taiji 已实现”。

完整能力必须依次通过 A0–A9：所有权、学习型抽象、世界状态、自适应协作、情景到语义、生命调节、目标规划、原生生成、持续进化和多模态具身。旧 K/N/M 结果只作为 kernel 回归证据。

## 8.1 神经元、共振与规模自进化的重新定位（2026-08-26）

本节固定“神经元架构”与“自进化”的正确含义，避免把原生误解成“每个神经元都手写”或“只要扩大参数量就会进化”。

### 神经元的实现单位

Taiji 使用 PyTorch/CUDA 可执行的向量化神经元群体，而不是为每个神经元创建一个 Python 对象。一个神经元的计算状态至少包括活动、阈值/适应、时间常数、局部资格迹和稳态调制；一个突触包含稀疏连接、效能权重和可塑性状态。神经元群体被组织成局部区域，区域再通过工作空间、世界状态和记忆接口协作。

概念更新为：

```text
activity[t+1] = local_input
              + sparse_synaptic_input
              + recurrent_state
              + memory/homeostatic_modulation
              - adaptive_threshold

synapse[t+1] = synapse[t]
             + local_activity_trace
             + prediction_error
             + outcome/reward
             + structural_policy
```

这不是要求 Taiji 使用某个固定公式，而是要求“神经活动、突触可塑性和外部结果”成为可追踪的不同状态。当前 TSK-v8 已有向量化区域、稀疏连接、持续状态和局部更新；显式细胞类型、相位共振和自主拓扑增长仍属于 v1 后续实现，不得提前宣称完成。

### 神经元如何区分

神经元的身份不应由 `if neuron_id == ...` 这样的语义查表决定，而应由四类可观测特征共同形成：

1. **动力学特征**：时间常数、阈值、适应速度、是否具有持续活动或瞬时响应；
2. **连接特征**：输入/输出的稀疏拓扑、连接方向、局部回路和跨区域路由；
3. **响应特征**：对时间模式、对象关系、预测误差、目标和奖励的选择性；
4. **可塑性特征**：它在什么局部信号下增强、抑制、保持或改变连接。

因此可以存在兴奋型、抑制型、整合型、记忆型、预测型、门控型、调制型和运动输出型群体，但这些是动力学和职责的初始先验，不是固定知识表。训练后应通过响应曲线、连接图和因果 lesion 识别其实际分工；同一类神经元也允许因经历形成不同子类型。

### 神经元如何“共振”

共振不是所有神经元同时输出相同值，也不是一个神秘的智能开关。在 Taiji 中应定义为：一组具有相互促进的时间活动、预测关系和相位/节律一致性的神经元群体，在共同任务上下文中被稳定放大，并能够向工作空间广播。

目标机制由五部分构成：

- 局部循环连接，让活动保留多个时间步；
- 兴奋/抑制平衡，避免所有区域无条件放大；
- 不同时间常数和传导延迟，形成对不同时间尺度的选择性；
- eligibility trace、预测误差和奖励，把“共同出现”与“结果有效”区分开；
- 全局或区域级 neuromodulation，只在目标、注意、压力、疲劳和新奇度允许时放大活动。

概念上可用群体相干度判断是否形成 assembly：

```text
coherence(group) = magnitude(mean(exp(i * phase_of_each_unit)))
```

相干度高、预测误差下降、目标贡献为正的群体可以进入工作空间；相干度高但结果错误的群体必须被抑制或重组。当前代码已有递归状态、局部更新、工作空间路由和稳态调制，但还没有完整的显式相位/延迟共振内核；这应通过共振 assembly 的正向 Gate、随机 chunk lesion 和跨任务迁移来验收。

### 规模扩大与自进化

必须区分三个概念：

- **训练**：改变已有参数和突触效能；
- **个体发展**：在一个 Taiji 实例生命周期内改变记忆、路由、容量和拓扑；
- **生物学进化**：跨多个个体/代际改变可遗传结构。Taiji 当前首先实现前两者，不能把一次扩大模型称为达尔文进化。

Taiji 的规模扩张应至少有四个层级：

```text
容量扩张：更多神经元、状态槽位和记忆容量
连接扩张：新增突触、重路由、跨区域桥接
模块扩张：形成新的专用区域或认知器官
时间扩张：从短时感知发展到长期记忆、技能和自我模型
```

扩张不能只由“模型变大”触发。目标增长策略应根据持续预测误差、反复失败、任务新颖度、资源瓶颈和已有模块冗余度决定：先调整已有突触和记忆；只有确认现有容量无法解释或完成任务时，才申请新容量；新增结构经过 holdout、因果 lesion、资源预算和 checkpoint 回滚验证后才能纳入主网络。长期无贡献或重复的连接/模块应被剪枝或合并。

生物脑也不是单调增大：发育期会经历神经发生、突触过量生成、经验选择和大规模剪枝；成年系统依靠稀疏激活、模块复用、抑制控制和能量预算提高效率。Taiji 应借鉴这种“增长—竞争—巩固—剪枝”的循环，而不是把参数规模当作智能的充分条件。

### 与人脑的边界对照

| 层面 | 人脑 | Taiji 当前/目标 |
|---|---|---|
| 物理单元 | 细胞体、树突、轴突、突触、胶质和代谢系统 | 张量状态、稀疏突触、区域状态、资源与稳态调制 |
| 神经元类型 | 兴奋/抑制、感觉、投射、局部中间神经元、调制神经元等 | 当前以区域/算子为主；目标是动力学先验 + 学习后的群体分化 |
| 信号 | 脉冲、膜电位、递质、延迟和节律 | 连续活动/事件、可选脉冲化算子、递归状态、稀疏路由和时间常数 |
| 共振 | 局部回路、脑区同步、相位锁定、节律和抑制控制 | 目标为循环群体 + 相干度 + 工作空间门控；显式相位机制尚未完成 |
| 学习 | Hebbian/STDP、奖励调制、皮层可塑性、睡眠巩固 | 局部/在线更新、世界预测误差、奖励、记忆巩固和可恢复 checkpoint |
| 记忆 | 海马情景记忆、皮层语义记忆、程序和工作记忆 | 工作、情景、语义、程序记忆已形成原型，开放域巩固仍有限 |
| 规划 | 前额叶—基底节—海马—感觉运动环路 | 目标、价值、世界模型、想象 rollout、重规划和执行反馈 |
| 稳态 | 下丘脑、脑干、激素、睡眠、能量和身体反馈 | homeostasis、curiosity、fatigue、stress、sleep/play 的软件调制 |
| 规模成长 | 发育、突触生成/剪枝和代际进化 | 当前容量策略和 checkpoint；需求驱动拓扑增长/剪枝是后续目标 |
| 身体闭环 | 身体是认知的一部分，行动改变感觉输入 | 通过 Environment/Outcome/WorldState 合同建立具身闭环，仍是窄世界实验 |
| 能耗约束 | 极强的能量、空间、传导和代谢限制 | 资源预算、稀疏化、容量策略和 CUDA 执行；尚不等同生物能耗 |

因此，Taiji 不需要逐项复制人脑的生物细节，应该复制人脑在系统层面的关键原则：异质单元、稀疏协作、持续状态、多时间尺度、预测与行动闭环、稳态调制、记忆分工、增长与剪枝。它必须保留计算工程可验证性，不把“像人脑”当作免于评测的理由。

后续所有“神经元架构”和“自进化”实现，必须同时回答：新增了什么状态、由什么反馈驱动、如何证明它被使用、如何在 lesion 中失效、如何 checkpoint/rollback，以及它是否真的提高了未见任务迁移，而不是只增加了参数量。

## 9. 当前唯一入口

状态已滚动更新：四步连续重规划、3/4/5 步变量 episode、不同失败位置、after-state relation 变化、executive-to-world prediction train/holdout 与 no-online-update calibration control、runtime calibration trace 的多步连续性/恢复、world-model planner projection/replan lesion、跨 seed 的 3/4/5 步 world-dynamics imagined rollout、imagined-to-real execution、runtime recovery state、recovery transfer、world-error calibration policy、normalized world-error contract 及 schema-scale transfer contract Gate 均已通过；核心对象术语表、状态转移图以及 `Assembly`、`Event`、`Concept`、`SelfState`/`DevelopmentState` 最小版本化合同已落地，且已接入真实 runtime lineage：观测生成 assembly/event，Outcome 写回 episodic 血缘并更新 self/development，跨 episode 语义巩固生成 concept，native checkpoint 可在巩固前保存并恢复，source/semantic lesion 已有回归覆盖；多信号 Concept Gate 也已通过：latent、world object/relation 和 Outcome 共同参与，跨 schema/未见对象支持集增长及三类信号 lesion 均有回归覆盖；`ConceptFormationOrgan` 已从 `TSKV8Adapter` 提取为 Taiji 自有语义器官，拥有独立 concept registry、容量/塑性/剪枝控制与 checkpoint，adapter 已把 `ConceptMatch` 接入 MemoryState 与 planner concept prior；concept transfer Gate 已通过 schema 1/2/4/8 的未见任务规划迁移、容量干扰、三类证据 lesion、器官 checkpoint 与 native runtime checkpoint recovery；concept sequence Gate 已通过时间顺序 action sequence、反转序列对照、变量 schema scale、concept lesion、adapter checkpoint 和失败 replan；状态条件 suffix Gate 已通过真实 WorldTransition 的 after-state/prediction-error/outcome trace、部分执行后的剩余 suffix 检索、错误状态与完全错位动作 fail-closed、环境 after-state 保留，以及 organ/native checkpoint recovery；变量 horizon / 分支塑性 Gate 也已通过同一 Concept 内共享前缀分支竞争、分支特有真实反馈只更新对应 trace、trace lesion 与 checkpoint recovery；trace capacity / selective branch Gate 也已通过 trace_capacity=1/2/4 的容量曲线、分支增量加入、单一 trace lesion 及剩余 identity checkpoint recovery；online branch birth Gate 也已通过连续真实 transition 链、新 trace_id、重复抑制、失败 feedback、adapter 发布、settle_action episode buffer 与 native checkpoint continuation；branch attribution Gate 也已通过多个同时激活 Concept 的唯一 owner 归属、低置信度/近似平分/owner lesion fail-closed、真实 settle_action buffer 和 native checkpoint continuation；当前唯一下一步是把 branch birth 接入 DevelopmentState 结构成长预算与 rollback 合同。

执行顺序只看 [SEED_DEVELOPMENT_ROADMAP_2026_08.md](SEED_DEVELOPMENT_ROADMAP_2026_08.md)。P1 已完成首个兼容纵切片，P2 relation subgate 已在两个独立语料分区通过，P3 world-state/action-outcome 合同、可恢复 state store、结构化对象/关系/时间打乱、多步 episode 窄 Gate、adapter transition lineage、runtime prediction record、error-driven online correction、最小 workspace 路由/lesion 以及 A3 静态/world-outcome 窄 Gate 已落地；P4 working/episodic memory、cue-conditioned one-shot recall、additive semantic consolidation、multi-factor/noisy semantic Gate、semantic runtime/checkpoint ownership、容量/干扰曲线、standalone procedural skill、多步 procedural robustness、procedural runtime ownership 和 homeostatic/sleep-play Gate 已通过，P5 单步 goal-planning、imagined rollout/replan trigger、实际 replan/calibration 与 delayed reward/intervention Gate 已通过，P6 structured content/expression/tool-call codec、tool execution/outcome、tool failure/replan、unseen-tool/parameter transfer、cross-organ expression consistency、learned content selection、runtime content-selection ownership、online content credit assignment、holdout content transfer、text organ codec、terminal language-organ boundary、backend registry/training contract、external decoder realization/lesion、Qwen provider smoke、Taiji-owned realization validator/fallback、runtime semantic constraint/feedback、language fallback/replan、train/holdout provider baseline、rollbackable provider trainer、trained-provider safety integration、provider artifact/loader、Seed client provider startup、frontend client observability、client input-boundary、P7 executive contract、executive environment-loop、candidate synthesis、affordance feature transfer、affordance online-credit、contextual grounding、world-grounding lineage、end-to-end grounding transfer、grounded multi-step environment 和 grounded multi-step train/holdout 窄 Gate 已通过；当前唯一入口是扩展状态条件 suffix 到变量 horizon、同前缀分支竞争、trace lesion 和实际 outcome/error 增量更新，继续禁止固定 action/intent 表。
