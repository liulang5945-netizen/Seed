# Seed — Taiji 原生认知架构的运行时

Seed 是训练、评估、部署并托管 **Taiji** 的项目、产品与运行时。Taiji 是一个**原生认知架构**——从在线预测编码机制构建，而不是 Transformer 的包装。内核从**局部预测误差**中学习（无反向传播、无注意力矩阵、无上下文窗口、运行时无教师模型）；在内核之上，Taiji 拥有自己的表征、持续状态、记忆、目标、规划与行动选择，同时在合适处刻意复用成熟算法（embedding、SSM、MoE 式路由、优化器、检索）。

不吹不黑：当前可执行代码是 **Taiji Substrate Kernel v8（TSK-v8）**——一个字节级预测编码研究内核。它是可用的基座，不是完整的认知架构：内置能力已被验证，语言级智能仍在训练中（见[现状](#现状)）。

## 语言 / Language

- **简体中文**：本页为中文项目介绍
- **English version**: [README.md](README.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![tests](https://img.shields.io/badge/tests-900%2B-green.svg)](.github/workflows/ci.yml)

## 架构

### Taiji 是什么

Taiji 的目标是一座**完整的原生认知架构**（合同见 [Taiji 原生架构 v1](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md)，需求见 [Taiji 核心目标](plans/active/TAIJI_CORE_REQUIREMENTS.md)）：

- **唯一认知主体。** Taiji 端到端拥有认知状态与决策路径。运行时没有任何 Transformer hidden state、教师 logits 或外部模型替它思考；Seed 是承载/产品运行时，外部模型与工具是环境设施。
- **持续、多时间尺度的状态**，而不是每次请求重新开始——感觉、工作、情景、语义、程序与发展尺度并存。
- **身体 → 真实因果闭环。** 观察变为内部 `PerceptEvent`；行动以 `ActionIntent → WorldAction` 改变环境；真实 `Outcome` 回写世界校准、记忆写入、信用与学习。
- **异质专门化协作，而非单一大而同的网络。** 感受野、时间尺度、学习规则各异的神经群体相互协作；门槛是只有**组合**才能解出的预注册任务（`1 + 1 > 2`）。
- **复用工具箱，拥有心智。** 学习到的 embedding、SSM 块、attention-作为-路由、MoE 式专家、优化器与检索都允许作为机制——不变量是 Taiji 的持续状态、记忆、目标与决策始终由 Taiji 拥有、可保存、可损伤、可替换。

### 分层总览

```mermaid
flowchart LR
    obs[文本 / 图像 / 音频 / 工具 / 身体] --> L0[L0 器官适配器 + 编解码]
    L0 --> L1[L1 学习型感知层级<br/>特征 → assembly → 事件]
    L1 --> L2[L2 多时间尺度预测动力学]
    L2 --> L3[L3 工作空间<br/>选择性路由 + 绑定]
    L2 --> L4[L4 记忆系统<br/>工作 / 情景 / 语义 / 程序]
    L3 --> L5[L5 世界与自我模型<br/>实体 / 关系 / 因果]
    L4 --> L5
    L5 --> L6[L6 执行认知<br/>目标 / 推理 / 规划]
    L6 --> L7[L7 解码器与效应器<br/>语言 / 工具 / 身体]
    L7 --> fb[环境反馈] --> L0
```

贯穿 L1–L7 的横向**稳态 / 发展调节**系统，从内部状态驱动好奇、疲劳、压力、睡眠/玩耍与结构预算——而不是来自某个 UI 调度器。

### 记忆与认知状态合同

Taiji 的状态是版本化、可观测的：`PerceptState`、`PredictiveState`、`WorkspaceState`、`WorldState`、`MemoryState`、`GoalState`、`PlanState`、`SelfState`、`HomeostaticState`、`DevelopmentState`、`LearningState`。记忆是多系统——**工作**（当前变量）、**情景**（一次性真实经历）、**语义**（跨经历提炼的稳定概念）与**程序**（技能）——不是上下文窗口、KV cache、一次 RAG 命中，也不是某个 Python 列表。

### 学习：两个平面

1. **发展期训练** —— 批量离线形成感知层级、世界模型、语义记忆与语言器官。使用优化器/蒸馏时显式标记为 `native-assisted`；原生内核同时运行自己的局部 delta 规则（自 2026-08-26 起 `taiji/` 内不再存在 `backward()`）。
2. **终身学习** —— 运行时通过局部预测误差、资格迹、奖励/新颖性调制、情景写入、回放与结构可塑性持续适应，不灾难性遗忘。

### 当前可执行内核（TSK-v8）

`taiji/` 今天运行的是基座内核：原始字节编解码 + 预测 fabric + 分布式情景场原型 + 字节运动器官，接入一个可观测、可 checkpoint 的闭环。不依赖 Transformer，PyTorch 仅作张量执行引擎。更新只发生在固定扇入边上——没有稠密结构掩码、没有注意力矩阵、没有上下文窗口、没有优化器、没有 `backward()`：

```math
u_t^r = \mathrm{Bound}\left(\lambda_u u_{t-1}^r + \alpha_g (D^r)^T e_{t-1}^{r} + \alpha_T \hat a_t^r + \alpha_c c_t^r\right)
```

```math
\Delta D^r=\eta_D\, e_{t-1}^{r}(q_{t-1}^r)^T, \qquad
\Delta T^r=\eta_T\,(a_t^r-T^r q_{t-1}^r)(q_{t-1}^r)^T
```

情景存储是一个共享分布式场：写入一个事件激发一个重叠的 engram 群体 `h`——不追加任何行、键或值：

```math
h^{event}=\phi(Qs+\gamma_e(Aa+Oo+r\rho+Tt+Ee+Pp)), \qquad
\Delta W^{mem}=\eta_m g(h^{event}-W^{mem}h^{cue})(h^{cue})^T
```

（张量形状、更新顺序与状态合同见 [TSK-v8 规范](plans/archive/implementation/TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md)。）

## 能力

项目有一种**对抗式验证文化**：下面的每项能力都由已提交、带病变对照的验证装置测量——固定种子、明确基线（随机 / 冻结父模型 / 简单规则 / 仅哈希）、holdout+retention 只读、全新进程重开 checkpoint 并比对摘要，该失败就如实 `failed`。[M0 五项能力契约](plans/manifests/taiji_foundation_baseline_v1.json) 冻结了"什么算数"。

### 已验证的内核机制（可复现、已提交）

| 能力 | 结果 |
|---|---:|
| 在线字节循环预测（无反向传播） | 0% → **94.12%** 准确率；平均惊奇度 −98.02% |
| 自由生成 | `a → bcdabcda`，全部 8 步精确 |
| 歧义消解（N7） | 100% vs 一阶模型 50% |
| 延迟 / trace 记忆（N8） | 仅 trace 100%；移除 trace 或动态状态 → 50% |
| 无教师长程自由运行（N9） | 128 个动作全部精确 |
| 稀疏迁移（N10） | 相对稠密前向差异 ≤ 2.98e-8；存量为 98.59% |
| 行动信用（N11） | 100% vs 随机 50%、无行动学习 57.5% |
| 情景场一次性写入（M5） | 8 个情景存入一个共享场，零逐事件槽位；召回 87.5% vs 对照 25% |

### 结构成长与协作（已过 Gate）

- 区域**生长 / split / merge / prune** 与连接剪枝均通过 holdout、预算、trial 往返与逆回滚 Gate；结构提案来自真实预测误差/资源信号，永不来自预先写死的意图表。
- **跨区协作学习器**依据实测的预测误差迁移与资源状态选择显式跨区连接（三 seed 门禁）。

### 能力反证门槛 A0–A9（"聪明"的合同）

每个 Gate 都必须在**未见**数据上、对照显式基线证明机制本身有效；训练分数不能充当通过证据（定义见[架构合同](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md#11-能力反证门槛)）：

| Gate | 必须证明 | 进展 |
|---|---|---|
| A0 所有权 | 移除 Seed 决策逻辑后 Taiji 仍能完成认知纵切片 | 合同与纵切片已闭合 |
| A1 学习型抽象 | 可变时长 assembly 迁移到未见组合 | 关系 subgate 闭合；完整 assembly 未闭合 |
| A2 世界状态 | 保持实体/事件并预测干预后的变化 | 狭窄世界 Gate 已闭合 |
| A3 自适应协作 | 异质群体击败任何单一群体 | 工作空间基础已建；完整 Gate 未闭合 |
| A4 情景→语义 | 概念迁移，而非情景复读 | 原型与运行时所有权在；巩固未闭合 |
| A5 稳态调节 | 驱动探索/学习/睡眠，而非 UI 数字 | 原型已门禁；广度未闭合 |
| A6 目标与规划 | 想象 rollout 改善真实成功率 | 单步规划 Gate 已闭合 |
| A7 原生生成 | 内部意图 → 可读语言/工具动作 | 结构化生成已闭合；流畅性在训练中 |
| A8 持续进化 | 旧能力存活；受治理的生长/剪枝 | 成长 Gate 已闭合；B5 在训练中 |
| A9 具身 | 器官共享世界状态；跨模态迁移 | 合同在；完整 Gate 未闭合 |

### 统一能力评估（M0）与当前训练（M1）

M0 建造了**测量机器**和一个可信零点（`status=failed` 是设计使然）：

- B1 字节/组合预测 —— 在 1 MiB 规模上仍不如 unigram 基线。
- B2 延迟记忆 —— 召回存在，但对 memory-lesion 无因果增益。
- B3 世界转移 / B4 目标驱动行动 —— task 级信号在 pilot 规模通过（world error → ~1e-5、goal success 0.5 → 1.0），尚未到 foundation 规模。
- B5 持续学习 —— 延续已验证；backward transfer 仍为负。

M1 随后在 CPU 上开始训练这个闭环（课程 F1→F5、三个固定种子、内容寻址数据、原子 `parent/last/best` checkpoint、全新进程只读复核）。F1 字节预测在 1 MiB 规模下降 holdout BPB 约 30%；F3 世界/行动课程达到阶段 Gate；**记忆课程是当前前沿**：原生关联基座被自己的数据契约判定为不适合，一级 **identity key/value 器官**已晋升为默认开启（15/15 门禁），而 foundation 规模的判定又显示该器官在干扰下寻址仍失败——三次反证探针锁定根因；修复进行中（M1-65）。证据在 `reports/` 下与编号计划一一对应：[唯一执行计划](plans/active/roadmap/03_CURRENT_EXECUTION.md)。

## 现状

- 已完成并提交：基座内核与验证链（900+ 测试）、结构成长 Gate、M0 测量机器、M1 训练管线（F1–F5、三 seed）、记忆数据契约、identity 器官 v2 默认开启。
- 进行中：M1-65（抗干扰记忆寻址）——当前唯一下一步。
- 诚实边界：这是一个训练中的学习机制原型，**不是**完整的认知架构，不是语言模型，也不构成任何 AGI 主张。乱码输出是预期的内核行为。

## 快速开始

```bash
python -m pip install -e ".[dev]"
python scripts/training/verify_taiji_native_v7.py        # 基座回归链
python -m pytest tests -q                                # 900+ 已提交测试
```

Seed 运行时兼容 API：

```python
from seed import Seed

model = Seed()
model.learn_bytes(b"abcdabcdabcdabcd", epochs=200)

print(model.score_bytes(b"abcdabcdabcdabcd"))
print(model.generate(b"a", length=8))

checkpoint = model.checkpoint()
restored = Seed.from_checkpoint(checkpoint)
```

训练入口（CPU）：`scripts/training/train_taiji_foundation.py`、`train_taiji_memory.py`、`train_taiji_world_action.py`、`train_taiji_joint.py`。

## 产品外壳

Seed 以自包含 Windows 桌面构建交付（双入口 `Seed.exe` + `SeedBackend.exe`）：双击即拉起后端、激活原生运行时，并在数秒内于 `http://127.0.0.1:8000` 提供 Web UI——聊天、训练面板、生命状态雷达图、IDE 工作区与 Agent 配置。开发模式：

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000   # 后端 + Web UI
python desktop/main.py                                       # 桌面壳
cd frontend && npm run dev                                   # 前端开发服务器
```

环境开关：`SEED_PORT`（默认 8000）、`SEED_HOST`（默认 127.0.0.1）、`SEED_RUNTIME=1`（启动时激活 Seed 原生运行时）。历史 beta 证据见 `reports/seed_public_beta_release_20260823.md`。

## 源码结构

```text
taiji/                  原生认知架构（不 import seed/neuroplex/transformers）
├── fabric.py           预测性循环 tick
├── sparse.py           固定扇入突触、局部更新
├── memory.py           分布式情景编码 / 补全 / 回读
├── identity_organ.py   一级可训练 key/value 记忆器官
├── organs.py           原始字节感受器、稀疏感受器库、奖励感知运动器官
├── foundation_tasks.py B1–B5 能力适配器（M0 契约）
├── foundation_training.py  联合 F1–F5 训练、checkpoint、谱系
└── model.py            observe / learn / score / generate / checkpoint

scripts/training/       verify_* 链、train_taiji_* 入口、eval_taiji_m1_*
tests/taiji_native/     内核回归 + 所有权契约
reports/                每个里程碑的编号、已提交证据
plans/active/           核心需求 · 架构 v1 · 唯一执行计划
```

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。