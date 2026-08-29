# Seed 产品与运行时架构

> 修订日期：2026-08-28
>
> 纠正：Seed 是项目、产品和运行时，不再被定义为 Taiji 之上的认知模型主体。Taiji 是完整原生认知架构。

## 1. 所有权

```text
Seed project/runtime                              seed/, api/, frontend/, desktop/
  ├─ product identity and distribution
  ├─ process, device, resource and lifecycle management
  ├─ datasets, experiments, evaluation and release
  ├─ API/UI/plugin/tool/environment adapters
  └─ hosts Taiji through its public architecture contract

Taiji native cognitive architecture              taiji/
  ├─ perception and learned representation
  ├─ predictive world/self state and workspace
  ├─ working, episodic, semantic and procedural memory
  ├─ goals, reasoning, imagination and planning
  ├─ language/tool/body action generation
  └─ developmental and lifetime learning

Frozen comparison runtime                        neuroplex/
  └─ Legacy Transformer baseline; never enters Taiji cognition
```

Seed 可以决定“在哪台设备运行、加载哪个 checkpoint、使用什么数据、连接哪个环境、如何展示结果”，不能决定“这个概念是什么、下一步如何推理、目标是什么、该输出什么”。后者全部是 Taiji 的认知责任。

## 2. 当前代码事实与目标事实

| 维度 | 当前代码 | Taiji v1 目标 |
|---|---|---|
| 顶层入口 | `seed.model.Seed` 通过 `TSKV8Adapter` 承载 Taiji v1 合同 | Seed runtime 启动一个完整 Taiji architecture |
| Taiji 能力 | P1–P7 已建立感知、世界状态、工作空间、记忆、规划、结构生长、`ActionIntent/ToolCall/Outcome` 等研究合同与窄 Gate；产品执行平面尚未闭合 | 感知→世界模型→记忆→执行认知→生成→真实工具/身体 outcome 的完整闭环 |
| checkpoint | `seed-native-v1` 保留旧 `substrate` 载荷，并增加 Taiji v1 原子信封 | Seed 保存产品元数据，认知状态由 Taiji checkpoint 完整拥有 |
| 输入 | UTF-8/raw byte 训练路径 | 多模态 Observation，经 Taiji 学习型感知形成内部表征 |
| 输出 | 产品聊天已接语言器官；结构化工具执行只在 Taiji 测试环境中闭环，未接 Seed 工作台 | ActionIntent 经语言、工具或身体效应器执行，并把真实结果回写 Taiji |

现有 API 和 checkpoint 不立刻破坏。P1 通过 compatibility adapter 保留行为，同时把新认知合同放到 Taiji 所有权下。

当前实现入口是 `Seed.architecture`；`Seed.substrate` 仅是历史兼容别名。adapter 的职责是把
TSK-v8 的旧 byte/fabric/action 信号映射到版本化 v1 合同，不把这个映射误报为完整 Taiji。

## 3. Seed 允许拥有的内容

- 安装、进程、设备、显存、并发和生命周期；
- 数据集 manifest、训练任务、实验注册、评测和报告；
- checkpoint 文件管理、版本下载、校验、回滚和发布；
- API、桌面、前端、移动端和远程连接；
- 工具/环境的协议适配、权限、审计、超时和安全边界；
- 用户设置、工作区、知识文件、日志和可观测性；
- Legacy-off/Legacy-on 构建和离线对照调度。

这些设施可以向 Taiji 提供 Observation、affordance、resource budget 和真实 outcome，但不能替 Taiji 形成隐藏决策。

## 4. Seed 禁止拥有的隐藏认知

- tokenizer/embedding/语言模型输出被包装成 Taiji 思考结果；
- Seed 内的概念图、事件 K/V 表、答案缓存或 persona prompt 作为真实记忆；
- Seed/Agent 层规划好动作，再让 Taiji 只做打分或文案生成；
- Legacy hidden state、teacher logits 或外部模型决策进入 Taiji forward；
- Python replay list、RAG 检索结果或工作流状态被冒充为 Taiji 内生学习。

外部知识库和工具可以使用，但必须作为带 provenance 的 Observation 进入 Taiji；是否相信、组合和执行由 Taiji 决定。

## 5. 成熟技术的采纳边界

Seed 可以提供 PyTorch、CUDA、数据库、向量索引、分布式训练、数据处理和标准评测。Taiji 可以采用成熟的 embedding、attention-like routing、状态空间、图计算、optimizer 和强化学习算法。

判据不是“它是否曾在 Transformer 中使用”，而是：

- 是否解决 Taiji 的明确能力需求；
- 认知状态和决策是否仍由 Taiji 拥有；
- 是否可保存、可替换、可损伤、可测量；
- 是否引入外部模型的运行时认知依赖。

## 6. 包依赖合同

```text
seed / api / clients ──public runtime API──> taiji
              │                              X
              └── optional offline ──> neuroplex

taiji ─X─> seed / neuroplex / transformers
neuroplex ─X─> seed / taiji
```

`taiji/` 不导入 `seed`、`neuroplex` 或 `transformers` 的边界继续保留。它可以依赖通用数值/系统库，但新增依赖必须说明其认知职责与替换界面。

## 7. 产品能力声明

当前产品只能声明：

- 有可执行、非 Transformer 的 TSK-v8 研究 kernel；
- 已验证持续状态、局部学习、情景原型和行动闭环；
- 正在按 Taiji Native Architecture v1 建设完整认知层。

不能再把 byte-cycle accuracy、N0–N11/M5–M7 或旧 800K/16M 训练写成“Taiji 已完成智能架构”。这些是 kernel 证据。

完整目标见 [TAIJI_NATIVE_ARCHITECTURE_V1.md](TAIJI_NATIVE_ARCHITECTURE_V1.md)，执行顺序见 [SEED_DEVELOPMENT_ROADMAP_2026_08.md](SEED_DEVELOPMENT_ROADMAP_2026_08.md)。

## 8. 产品执行平面与工作台边界

2026-08-28 的产品链路审计确认：Taiji 内部已经有
`ActionIntent → ContentPlan → ToolCall → TaijiToolEnvironment → Outcome` 合同，但 Seed 产品并没有把这条合同接到自带
IDE、文件系统、终端、诊断器和 MCP。
因此“内核会形成结构化工具动作”和“客户端里的模型能自主使用工作台”是两个不同事实，当前只完成前者。

### 8.1 三种“语言”必须分开

| 规范字段 | 含义 | 所有者 | 禁止混用为 |
|---|---|---|---|
| `natural_language_backend` | 将 Taiji-owned `ExpressionPlan` 表达为人类语言的末端器官 | Taiji 器官合同；Seed 装载 provider | Python/JavaScript 等编程语言 |
| `programming_language_id` | 当前文件的 Monaco/LSP/运行器语言，例如 `python`、`rust` | Seed 工作台 capability；Taiji 可选择动作 | 语言 provider、模型类型 |
| `artifact_format` | `taiji-native` checkpoint、外部 provider artifact 或导入/导出适配格式 | Seed 训练/发布层 | Taiji/Transformer 认知架构切换 |

外部语言 provider 只负责“嘴巴”，不能因此获得文件、终端或 IDE 权限。编程语言选择是可执行、可撤销的工作台动作，
必须从文件内容、扩展名、项目 manifest、LSP 可用性和用户约束形成证据；不能靠前端硬编码下拉框冒充 Taiji 的自主判断。

### 8.2 目标执行链

```text
Taiji goal/world/self state
  -> ActionIntent / ToolCall
  -> Seed Workbench Capability Registry
  -> policy + approval + budget + freshness Gate
  -> WorkbenchEnvironment
       -> workspace read/write/patch
       -> editor open/set-language/diagnostics
       -> terminal run/test/build/debug
       -> MCP/plugin adapters
  -> typed execution result + after-state + audit
  -> Taiji Outcome / Observation / memory / online credit

Frontend IDE <- subscribes to the same execution/audit state; it is not the execution authority
```

Taiji 不直接点击 Vue/Monaco DOM，也不把任意自然语言翻译成未审计 shell。Seed 提供版本化 capability、参数 schema、
权限、预算、超时、事务、撤销和真实执行；Taiji 根据自己的目标和世界状态选择能力，并从真实 outcome 学习。

### 8.3 当前已确认的产品断点

- `api/seed_runtime.py::SeedRuntime.chat()` 只生成文本，没有消费 Taiji 的 `ToolCall`；
- `api/app.py` 把工作台、Agent、MCP、RAG 和插件路由全部挂在 `legacy_available()` 后，原生 Seed 启动时反而没有工作台 API；
- `frontend/src/composables/useWorkspaceBridge.js` 声称打通聊天、IDE 和终端，但当前无任何组件调用；
- `frontend/src/components/MonacoEditor.vue` 的语言列表、扩展名推断和切换全是页面内状态，后端、checkpoint、
  capability snapshot 和 Taiji `SelfState` 均不可见；
- `seed_platform/runtime_service.py` 在原生模式明确上报空工具列表，而前端仍宣称已能工具调用和自主探索；
- 工作台运行/工程脚手架、Agent/ReAct、MCP、插件、记忆和 RAG 的现行实现仍直接导入 NeuroPlex，不能作为 Taiji 原生执行平面；
- 设置页仍允许把正式产品热切换到 Cortex，训练页仍展示 GGUF/LoRA 发布动作，均与“Legacy 仅离线对照、
  Taiji native artifact 为正式产品格式”的方向冲突。

这些断点统一按总路线第 16 节的 Workbench Closure 路线处理；在真实纵切片通过前，客户端不得再把“工具存在”“可自主执行”
或“支持某格式”当作产品已完成能力。

## 9. Legacy 边界

Legacy NeuroPlex 继续作为冻结的 Transformer 离线对照和显式兼容扩展。现阶段不删除：产品壳仍有部分懒加载依赖，同预算比较也需要稳定对照。

冻结意味着不再向 Legacy 增加认知功能，只允许安全、兼容和行为保持修复。默认产品最终达到 Legacy-off；是否从仓库移除必须等 Taiji v1 通过语言/工具 Gate、产品迁移和对照归档后再决定。

## 10. 当前唯一边界动作

当前路线入口只看 [SEED_DEVELOPMENT_ROADMAP_2026_08.md](SEED_DEVELOPMENT_ROADMAP_2026_08.md)，当前唯一下一步只看其指向的 [03_CURRENT_EXECUTION.md](roadmap/03_CURRENT_EXECUTION.md)。
本文件只固定所有权和执行平面边界，不另设实现顺序。
