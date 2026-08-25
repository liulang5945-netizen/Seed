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

## 9. 当前唯一入口

执行顺序只看 [SEED_DEVELOPMENT_ROADMAP_2026_08.md](SEED_DEVELOPMENT_ROADMAP_2026_08.md)。P1 已完成首个兼容纵切片，P2 relation subgate 已在两个独立语料分区通过，P3 world-state/action-outcome 合同、可恢复 state store、结构化对象/关系/时间打乱、多步 episode 窄 Gate、adapter transition lineage、runtime prediction record、error-driven online correction、最小 workspace 路由/lesion 以及 A3 静态/world-outcome 窄 Gate 已落地；P4 working/episodic memory、cue-conditioned one-shot recall、additive semantic consolidation、multi-factor/noisy semantic Gate、semantic runtime/checkpoint ownership、容量/干扰曲线、standalone procedural skill、多步 procedural robustness、procedural runtime ownership 和 homeostatic/sleep-play Gate 已通过，P5 单步 goal-planning、imagined rollout/replan trigger、实际 replan/calibration 与 delayed reward/intervention Gate 已通过，P6 structured content/expression/tool-call codec、tool execution/outcome、tool failure/replan、unseen-tool/parameter transfer、cross-organ expression consistency、learned content selection、runtime content-selection ownership、online content credit assignment、holdout content transfer、text organ codec、terminal language-organ boundary、backend registry/training contract、external decoder adapter realization/lesion、Qwen provider smoke、Taiji-owned realization validator/fallback、runtime semantic constraint/feedback、language fallback/replan、train/holdout provider baseline、rollbackable provider trainer、trained-provider safety integration、provider artifact/loader、Seed client provider startup、frontend client observability、client input-boundary、P7 executive contract、executive environment-loop、candidate synthesis、affordance feature transfer、affordance online-credit、contextual grounding 和 world-grounding lineage 窄 Gate 已通过；当前唯一入口是完成 `WorldAffordanceGroundingProducer → LearnedAffordanceFeatures → ExecutiveController` 端到端对象/关系绑定 holdout，并验证 producer lesion 的能力下降，禁止固定 action/intent 表。
