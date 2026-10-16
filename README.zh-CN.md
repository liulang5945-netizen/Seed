# Seed — Taiji 原生认知架构的运行时

[English](README.md) · [项目地图](docs/README.md) · [当前执行](plans/active/roadmap/03_CURRENT_EXECUTION.md)

Seed 托管并对外提供 **Taiji**：探索持续状态、记忆、局部学习和行动的研究架构。仓库包含 Python 研究运行时，以及 Taiji Harness 产品外壳。

## 当前状态

活动研究为 **N 主线**，N1 正在研究学习型读取、实体绑定和状态转移。已有结果属于限定实验；训练、未见评估、保持和消费者行为分别记账，尚不足以证明通用对话、AGI 或自主进化能力。

截至 2026-10-17，run88c 的三个 fresh position 种子已在"末周期学习率衰减"（前 18432 步 0.001、末 6144 步 0.00025）配方下跑完，并由一台零算力复核仪从封存分片独立复算出：三种子全部通过冻结的六训练门；至 18432 步（含该里程碑）为止的 loss、五组更新读数与各 960 行评分逐项复现 run86c 基线；正式 6 次加预检 3 次独立恢复均 equal、110 枚来源守卫未变、CPU 3385.578125 秒在 4800 秒上限内。训练充分不等于能力：候选仍不可晋升、`TRAIN_SHA` 保持 None、v5 确认面仍未消费；判读件同时记下反向读数——同一衰减使 seed13 少 1 题，而被修好的失败原本只出现在三个种子中的一个。两条对照臂批次随后均收口并核验，九枚同协议终端也按已冻绑定对 v5 **只查询了一次**：四门合取**不成立**——`advantage_cut` 假（position 79/96 对 cut **80**/96）、`each_position_kind` 假（seed13 的 lookup 0/8、revision 0/8），`each_position_seed`（31/32/16）与 `advantage_terminal`（对 33/96，差＋46）为真。只读已封存的 288 行还能看到：position 与 cut 在 **79/96** 个未见 (seed,item) 上逐字输出相同，且"只有 position 对"为 **0** ⇒ **位置读出臂在未见证据上没有一次做到 cut 做不到的事**，失分按 seed 与题目类分布而与读取臂无关。该面已消费、不可重跑；另有一格 `sources_unchanged` 是我自己那台绑定仪的已知假阴性（批次存 CRLF 归一摘要、收尾按原始字节复核，149 枚守卫真实改动 0 枚＝DEBT-G147），故这次执行也不声称工程完整。owner 随后在三条预注册出口里裁定**甲**，甲已零查询跑完：一台只读仪把那 288 行逐题归因后出版 `reports/taiji_n1_confirmation_error_attribution_run89.json`，读数是 **`selection_side=96/96`、`compounding_only=0`**（每一行自由生成失败处，教师强制也没选中目标序列），错形只有 `first_symbol_wrong=69` 与 `prefix_then_diverge=27` 两类非零，**`transition` 类 72 行零错**，而三臂同错的 16 组全部落在 seed13 的 8 个 lookup 与 8 个 revision 上——这 16 组里两两输出相同的组数为 **0**，即三枚终端在同题同错时错成三个不同的串。这些是"下一步该测哪个变量"的关联读数，不是机制、不是臂差、不是任何能力格。详见[甲路逐题错因归因判读](plans/reference/EXP-N1-91_run91_error_attribution_adjudication_20261017.md)、[v5 未见确认判读](plans/reference/EXP-N1-90_v5_confirmation_adjudication_20261017.md)、[run88c 整批评判](plans/reference/EXP-N1-88_run88c_batch_adjudication_20261017.md) 与[当前执行计划](plans/active/roadmap/03_CURRENT_EXECUTION.md)。产品阶段交付与研究能力验收分别判断。

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
