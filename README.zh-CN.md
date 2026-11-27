# Seed — 态极原生认知模型的运行时

[English](README.md) · [项目站点](https://taij-seed-liulang.traesite.cn/) · [项目地图](docs/README.md) · [贡献指南](CONTRIBUTING.md)

**态极（Taiji）是本项目正在研发的原生认知模型。** 它的架构研究如何让感知、记忆、世界模型、推理和行动协同工作，使模型在与环境交互中逐步发展。**Seed** 为态极提供运行时、实验工具、API 和产品界面。

项目的长期目标是让模型在多次交互之间延续知识和内部状态，从行动后果中学习，并在保留已有能力的同时获得新技能。这个仓库同时承载认知机制的实现，以及验证和使用这些机制所需的工程设施。

## Taiji 在探索什么

Taiji 的设计围绕五个相互关联的方向展开：

- **持续性**：保留内部状态，并根据新观察修订对世界的认识。
- **分工明确的记忆**：用工作记忆维持当前任务，用情景记忆保存经历，用语义记忆提取共同知识，用程序记忆积累技能。
- **专门化与协作**：让具有不同表征和学习规则的异质群体共同处理问题，并按任务、不确定性和资源预算选择计算路径。
- **从行动中学习**：把语言、工具和环境交互连接到可观察的结果，让真实反馈影响后续决策。
- **长期发展**：通过学习、重放、巩固和结构适应获得新能力，同时保持已有能力。这是研究目标，尚未被实验证明。

这些是正在研究的架构目标。当前实现包含实验内核和限定范围的原型，完整架构及通用对话能力仍有待实现和验证。[核心需求](plans/active/TAIJI_CORE_REQUIREMENTS.md)说明项目动机，[架构设计](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md)定义目标机制与接口。

## 项目如何组成

**Taiji 负责认知**：学习型表征、状态、记忆、目标、推理和行动选择。**Seed 负责承载**：进程、设备、数据集、实验、checkpoint 管理、API 和环境适配。**Taiji Harness 负责产品交互**：web 与桌面客户端、交互界面及打包分发。

目标交互闭环如下：

```text
观察输入 → 感知 → 状态、记忆与推理 → 行动
   ↑                                 │
   └──────── 环境反馈与学习 ──────────┘
```

Python 实现现已包含两条核心路径：当前主线正在预训练与合龙的 **态极原生纯字节有状态神经群底座**（`taiji/native_cognitive_backbone.py`、`taiji/native_runtime.py`：纯 UTF-8 字节、异质时间尺度持久神经群、精确 `u2_operators.fp1` KKT 快记忆、SVD 中性出生与五态单元生命周期、U2 §7 复制头、9 态外部动作台账与 5 平面结果分流）以及早期 `TSK-v8` 局部稀疏底层内核。[代码 Wiki](TAIJI_CODE_WIKI.md)与[项目站点](https://taij-seed-liulang.traesite.cn/)提供机制到源码的完整映射。

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

使用有界 `N36 / N37` 交付运行时与 `§15` 运营状态机 CLI：

```bash
python scripts/taiji_n36_takeover.py build --bundle-dir ./taiji_bundle --seed 6801
python scripts/taiji_n36_takeover.py chat --bundle-dir ./taiji_bundle --utterance "请把红色杯子放到桌上。"
python scripts/taiji_n37_ops.py init-ops --bundle-dir ./taiji_bundle --initial-state Qualified
```

客户端安装与构建见 [Taiji Harness 贡献指南](taiji-harness/CONTRIBUTING.md)。开发检查和贡献方式见 [Seed 贡献指南](CONTRIBUTING.md)。

## 近期成果

截至 **2026-11-27**，有界主线（`≤N30 → N31 → N32 → N33 → N34 → N35 → N36 → N37`）已在 `bounded_closed_vocab_and_multimodule_v1` 范围内完成全部收口（**`check` / `final` 读取 0 次**）：
- **已确认有界能力（`N33–N35`）**：百会话五模块联合耐久（`N34.5.90`）、封闭词表中文多轮指代、两族合成物理因果迁移、在线 Beta 可信度学习、学习式不确定性校准与在位生长（`N34.9.90`、`N35.1.91` 8/9 门 3/3 通过）。
- **有界交付与 `§15` 运营状态机（`N36 / N37`）**：独立可接管运行包（[CLOSEOUT-N36](plans/reference/CLOSEOUT-N36_20261126.md)，`D1..D5` 5/5 过）与 8 态运营状态机、6 步日常影子升级及三层变更分流（[CLOSEOUT-N37](plans/reference/CLOSEOUT-N37_20261126.md)，`O1..O6` 6/6 过）。
- **进行中的 `N38` 态极原生纯字节底座预训练与统一运行时**：`taiji-native-byte-group-v2`（`native_byte_mid`，1,455 万参数起可自发生长、纯 UTF-8 字节 `BYTE_VOCAB=259`、零 BPE、零 Transformer、4 异质持久神经群、`u2_operators.fp1` KKT 快记忆、SVD 中性出生与五态单元生命周期、U2 §7 复制头、动态宏字节词表挂接与统一 [TaijiNativeRuntime](taiji/native_runtime.py)）正在通过批处理 `bmm` 加速跑 24 小时开放中文 CPU 预训练（`float32` 稳态 `~5,000+ bytes/sec`）；`mid_18m` Transformer 仅作为外部对照基线留存（已留存 `best_val_loss = 5.0442`）。

开放词表自然中文、真机传感器/执行器、24 小时现场试运行、主体内学习式自主策略（`VX5`）与 `check / final` 分区仍明确标为**未验证 / 不签署**。全局收束见 [CONSOLIDATION_20261126](plans/reference/CONSOLIDATION_20261126.md) 与 [主线总纲](plans/MAINLINE.md)。

## 进一步了解

- [项目站点](https://taij-seed-liulang.traesite.cn/)：态极与 Seed 架构原理与真实源码地图交互概览。
- [项目地图](docs/README.md)：按目的查找文档。
- [核心需求](plans/active/TAIJI_CORE_REQUIREMENTS.md)：长期目标与设计原则。
- [架构设计](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md)：认知组件及其合同。
- [活动计划](plans/active/PLAN_INDEX.md)与[研究报告](reports/README.md)：当前工作和实验证据。
- [目录规范](docs/FOLDER_STRUCTURE_RULES.md)与[仓库卫生规则](docs/REPO_HYGIENE_RULES.md)：贡献和产物的存放约定。

## 许可证

[Apache License 2.0](LICENSE)。
