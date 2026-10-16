# Seed — Taiji 原生认知架构的运行时

[English](README.md) · [项目地图](docs/README.md) · [当前执行](plans/active/roadmap/03_CURRENT_EXECUTION.md)

Seed 托管并对外提供 **Taiji**：探索持续状态、记忆、局部学习和行动的研究架构。仓库包含 Python 研究运行时，以及 Taiji Harness 产品外壳。

## 当前状态

活动研究为 **N 主线**，N1 正在研究学习型读取、实体绑定和状态转移。已有结果属于限定实验；训练、未见评估、保持和消费者行为分别记账，尚不足以证明通用对话、AGI 或自主进化能力。

截至 2026-10-16，run86c 九个 fresh 模型已完成，工程、来源、分片、checkpoint 与恢复核验通过。三个 position 种子中有一个未过冻结训练门，因此 v5 确认面仍未消费，候选不能晋升。下一项建议是零查询离线轨迹诊断，当前待批准。详见[最终裁决](plans/reference/EXP-N1-86_detached_factorial_execution_adjudication_20261016.md)、[当前执行计划](plans/active/roadmap/03_CURRENT_EXECUTION.md)和[全项目评估](plans/reference/PROJECT_WIDE_ASSESSMENT_20261014.md)。产品阶段交付与研究能力验收分别判断。

## 架构与学习

- `taiji/`：原生认知机制和研究原型。
- `seed/`：兼容 API；`seed_platform/`：存储、谱系和运行时服务；`api/`：后端接口。
- `taiji-harness/`：独立 TypeScript workspace 中的客户端与产品外壳。
- `neuroplex/`：历史对照及兼容代码。

底层基座采用局部预测误差学习，部分发育训练序列原型使用优化器及 autograd，学习主张必须对应具体实现与实验。长期目标见[核心需求](plans/active/TAIJI_CORE_REQUIREMENTS.md)与[架构合同](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md)，实现细节见[代码 Wiki](TAIJI_CODE_WIKI.md)。

## 快速开始

需要 Python 3.12+ 与 PyTorch。以下只演示小型字节学习 API，不代表通用聊天模型。

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/seed/test_project_identity.py -q
```

```python
from seed import Seed

model = Seed()
model.learn_bytes(b"abcdabcdabcdabcd", epochs=200)
checkpoint = model.checkpoint()
restored = Seed.from_checkpoint(checkpoint)
print(restored.generate(b"a", length=8))
```

完整 Python 检查使用 `python -m pytest tests -q`；部分测试需要可选依赖或本地产物。局部通过不代表完整 CI 健康。

## 产品外壳

```bash
python -m pip install -e ".[dev,legacy]"
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

客户端 workspace 的启动方式见 [Taiji Harness 贡献说明](taiji-harness/CONTRIBUTING.zh.md)，本地提供者路由为 `/api/chat/stream`。

## 仓库导航

| 位置 | 唯一职责 |
| --- | --- |
| `taiji/`、`seed/`、`seed_platform/`、`api/` | 模型机制、运行时与后端 |
| `taiji-harness/` | 产品客户端与打包 |
| `scripts/`、`instruments/`、`tests/` | 可复用任务、测量与验证 |
| `plans/` | 需求、活动调度、冻结协议和决策 |
| `docs/` | 开发说明与项目导航 |
| `reports/` | 机器可读研究证据 |
| `design/` | 品牌资产与生成工具 |
| `data/`、`checkpoints/`、`taiji_data/` | 本地数据集和模型状态，迁移前须保护 |
| `output/`、`outputs/`、`artifacts/` | 历史运行及归档路径；已跟踪JSON归入 `reports/legacy/` |

遵守[目录规范](docs/FOLDER_STRUCTURE_RULES.md)和[仓库卫生规则](docs/REPO_HYGIENE_RULES.md)。训练须遵循对应冻结协议及预算，并先通过 checkpoint 保存恢复检查。历史失败实验保留为证据。

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。
