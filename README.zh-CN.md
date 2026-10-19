# Seed — Taiji 原生认知架构的运行时

[English](README.md) · [项目地图](docs/README.md) · [贡献指南](CONTRIBUTING.md)

**Seed 是一个探索持续运行、能够从经历中学习的认知系统的研发项目。** 项目的认知架构 **Taiji（太极）** 研究如何让感知、记忆、世界模型、推理和行动协同工作，使系统在与环境交互中逐步发展。Seed 为这套架构提供运行时、实验工具、API 和产品界面。

项目的长期目标是让系统在多次交互之间延续知识和内部状态，从行动后果中学习，并在保留已有能力的同时获得新技能。这个仓库同时承载认知机制的实现，以及验证和使用这些机制所需的工程设施。

## Taiji 在探索什么

Taiji 的设计围绕五个相互关联的方向展开：

- **持续性**：保留内部状态，并根据新观察修订对世界的认识。
- **分工明确的记忆**：用工作记忆维持当前任务，用情景记忆保存经历，用语义记忆提取共同知识，用程序记忆积累技能。
- **专门化与协作**：让具有不同表征和学习规则的异质群体共同处理问题，并按任务、不确定性和资源预算选择计算路径。
- **从行动中学习**：把语言、工具和环境交互连接到可观察的结果，让真实反馈影响后续决策。
- **终身发展**：通过学习、重放、巩固和结构适应获得新能力，同时保持已有能力。

这些是正在研究的架构目标。当前实现包含实验内核和限定范围的原型，完整架构及通用对话能力仍有待实现和验证。[核心需求](plans/active/TAIJI_CORE_REQUIREMENTS.md)说明项目动机，[架构设计](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md)定义目标机制与接口。

## 项目如何组成

**Taiji 负责认知**：学习型表征、状态、记忆、目标、推理和行动选择。**Seed 负责承载**：进程、设备、数据集、实验、checkpoint 管理、API 和环境适配。**Taiji Harness 负责产品交互**：web 与桌面客户端、交互界面及打包分发。

目标交互闭环如下：

```text
观察输入 → 感知 → 状态、记忆与推理 → 行动
   ↑                                 │
   └──────── 环境反馈与学习 ──────────┘
```

Python 实现包含 TSK-v8 底层内核，以及学习型读取、记忆、规划和发展学习等原型。不同组件采用不同学习方法：底层内核使用局部预测误差学习，部分序列原型使用优化器和自动微分。[代码 Wiki](TAIJI_CODE_WIKI.md)提供机制到实现的对应关系。

## 仓库里有什么

| 位置 | 用途 |
| --- | --- |
| [`taiji/`](taiji/) | 原生认知机制与研究原型 |
| [`seed/`](seed/)、[`seed_platform/`](seed_platform/) | Python API、运行时服务、持久化与谱系管理 |
| [`api/`](api/) | 产品集成所用的后端接口 |
| [`taiji-harness/`](taiji-harness/) | 独立 TypeScript workspace 中的 web 与桌面客户端 |
| [`scripts/`](scripts/)、[`instruments/`](instruments/)、[`tests/`](tests/) | 实验执行器、测量工具与验证 |
| [`plans/`](plans/)、[`reports/`](reports/) | 研究设计、决策与实验证据 |
| [`docs/`](docs/)、[`design/`](design/) | 开发文档与视觉资产 |
| [`neuroplex/`](neuroplex/) | 冻结的历史对照与兼容代码 |

## 快速开始

在仓库根目录执行以下命令，需要 **Python 3.12+**。基础包依赖 PyTorch；客户端有独立的安装流程。

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/seed/test_project_identity.py -q
```

下面的小型 Python 示例展示字节学习与内存中的 checkpoint 恢复：

```python
from seed import Seed

model = Seed()
model.learn_bytes(b"abcdabcdabcdabcd", epochs=200)
checkpoint = model.checkpoint()
restored = Seed.from_checkpoint(checkpoint)
print(restored.generate(b"a", length=8))
```

这是一个小型研究示例，展示的是字节学习接口，不能据此推断通用语言理解能力。

开发后端时：

```bash
python -m pip install -e ".[dev,legacy]"
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

客户端安装与构建见 [Taiji Harness 贡献指南](taiji-harness/CONTRIBUTING.md)。开发检查和贡献方式见 [Seed 贡献指南](CONTRIBUTING.md)。

## 近期成果

截至 2026-10-19，N 主线正在研究学习型读取、实体绑定与状态转移。最近完成的 N1-141 实验中，六个端点全部通过原训练门，zero/final checkpoint 均独立恢复成功。这属于训练集证据，新题泛化仍待验证，当前没有 N 系列模型获准晋升。成果细节见[实验结案](plans/reference/PLAN-N1-141_execution_closeout_20261019.md)，实时进展见[当前执行计划](plans/active/roadmap/03_CURRENT_EXECUTION.md)。

## 进一步了解

- [项目地图](docs/README.md)：按目的查找文档。
- [核心需求](plans/active/TAIJI_CORE_REQUIREMENTS.md)：长期目标与设计原则。
- [架构设计](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md)：认知组件及其合同。
- [活动计划](plans/active/PLAN_INDEX.md)与[研究报告](reports/README.md)：当前工作和实验证据。
- [目录规范](docs/FOLDER_STRUCTURE_RULES.md)与[仓库卫生规则](docs/REPO_HYGIENE_RULES.md)：贡献和产物的存放约定。

## 许可证

[Apache License 2.0](LICENSE)。
