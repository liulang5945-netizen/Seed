# Seed 产品与运行时架构

> 修订日期：2026-08-25
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
| 顶层入口 | `seed.model.Seed` 包装一个 `Taiji` 实例 | Seed runtime 启动一个完整 Taiji architecture |
| Taiji 能力 | TSK-v8 byte/fabric/episodic/motor kernel | 感知→世界模型→记忆→执行认知→生成完整闭环 |
| checkpoint | `seed-native-v1` 外壳嵌套 kernel checkpoint | Seed 保存产品元数据，认知状态由 Taiji checkpoint 完整拥有 |
| 输入 | UTF-8/raw byte 训练路径 | 多模态 Observation，经 Taiji 学习型感知形成内部表征 |
| 输出 | byte motor/generation | ActionIntent 经语言、工具或身体效应器执行 |

现有 API 和 checkpoint 不立刻破坏。P1 通过 compatibility adapter 保留行为，同时把新认知合同放到 Taiji 所有权下。

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

## 8. Legacy 边界

Legacy NeuroPlex 继续作为冻结的 Transformer 离线对照和显式兼容扩展。现阶段不删除：产品壳仍有部分懒加载依赖，同预算比较也需要稳定对照。

冻结意味着不再向 Legacy 增加认知功能，只允许安全、兼容和行为保持修复。默认产品最终达到 Legacy-off；是否从仓库移除必须等 Taiji v1 通过语言/工具 Gate、产品迁移和对照归档后再决定。

## 9. 当前唯一边界动作

P1 必须把上述所有权变成可执行测试：Seed 只调度，Taiji 拥有认知状态与 ActionIntent，TSK-v8 通过 compatibility adapter 保持当前行为。
