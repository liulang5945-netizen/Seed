# 态极唯一主线

> 文档状态：[主线] — 本文件是方向与执行队列的唯一来源。只有[03 当前执行](active/roadmap/03_CURRENT_EXECUTION.md)可以发布“当前唯一下一步”；其他任何文档不得自称主线，也不得把旧文的“下一步”当作当前指令。

2026-10-30 由用户要求做项目级收敛后建立。此前 R1/R2/R3 都只改了局部，导致 N177/F/N178、VISION、T2/T3、09 机制词典与共同内容核心同时像“当前主线”。本文件把方向收敛为一条，把其余内容明确降级为基线、组件、运营或历史。

## 1. 长期目标（不变）

沿 N 主线，把态极发展成由自身可学习、可恢复的认知机制参与决策与表达的原生认知模型，使其能以自然中文可靠完成连续任务、跨会话积累和纠正经验、从可核验的经验与反馈中学习新能力并保持旧能力；各阶段预先声明范围与收官条件，以同一可恢复候选及其前后版本、对照合理基线，按范围验收未见任务表现、必要机制贡献、旧能力保持、恢复与真实成本。

目标不变，变的只是当前实现路径：只保留一条可比较、可恢复、可结案的主计算图。

## 2. 唯一主计算图

```text
原始中文输入
  → 可学习的事件/内容状态（连续向量 + 原始引用 + 来源 + 时间 + 版本）
  → query / 时间意图条件读取
  → 每步重新消费内容的生成
  → 合法任务反馈回到声明的负责模块
```

必须同时满足：

- **不是唯一硬门**：结构化解析只作辅助视图、审计和辅助监督；解析失败的内容仍可带状态进入存储与读取路径。
- **不是整句固定偏置**：生成每一步都重新读取当前内容与解码前缀，而非给冻结语言模型加同一个 bias。
- **不是 current-only**：`current/previous/unknown` 必须进入读取器；旧值查询不得回退到当前值。
- **不是无信用旁路**：任务 loss 至少到达声明的 reader 与 decoder 组；辅助解析 CE 单列；query-only 路径必须真正隔离事实字节。
- **不是不可恢复的实验**：参数、版本 store、cursor、任务状态在同一 bundle 内零步保存，独立进程恢复后同一输入产生同一读取与 logits。

当前实现：[native_content_core.py](../taiji/native_content_core.py)；接口与读数见[R3 记录](reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)。

## 3. 唯一执行队列

| 顺序 | 阶段 | 当前状态 | 出口/边界 |
|---|---|---|---|
| R1 | 统一语义合同 | 已完成（设计级） | 时间/byte/固定分母/cut-rescue/来源/fixture 统一 |
| R2 | 候选与三类基线合同 | 已写，待用户评审 | 新核心/修正旧图/普通基线，Q1 与 Q2 分离 |
| R3 | 新核心隔离实现与零步恢复 | 已完成（结构门） | 53/53 结构检查；未训练，不称学会 |
| **R3b-1** | **修正旧图基线** | **已完成（结构门）** | 22/22 检查；固定分母 17/17；仅 value bias 可训练 |
| **R3b-2** | **普通端到端基线** | **已完成（结构门）** | 11/11 检查；固定分母 17/17；全部参数可训练 |
| R4 | train-only 定价与合同冻结 | 当前阶段（准入待 R2 评审） | 单位成本、数据/保持/数值门/预算/停止线 |
| R5 | 一次有界独立比较 | 待 R4 | Q1 架构包 + Q2 信用单因素；正负均结案 |
| R6 | 条件能力接力 | 仅在 R5 有净收益后 | T2/T3/自然连续任务/跨会话，按 07 分别验收 |

规则：

- 只能按序推进；跳级必须在本表留下书面修订与理由。
- R2 用户评审未完成前不进入 R4 训练/定价与 R5 评价；R3b 的隔离实现可在评审并行。
- 任何新方向先作为“候选”写进本表，不允许另建一份并列主线文档。
- 每阶段结束：更新本表状态、[03](active/roadmap/03_CURRENT_EXECUTION.md) 与[10](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)，并在对应合同记录证据。

## 4. 旧方向处置（均不是当前主线）

| 旧方向/资产 | 处置 | 允许的用途 | 不允许的用途 |
|---|---|---|---|
| N177 E1/E2/F0a/F0b/F1/F1b | 历史基线/证据 | 作为 `corrected_old_v1` 的来源与失败对照 | 作为新候选续推、追加剂量或重读 sealed |
| N178 parser 覆盖 | 诊断资产 | 解释旧失败中的解析缺口 | 自动扩量、替代主计算图或重置跨包计数 |
| N177 五模式/T1 producer/V4 | 组件与旧基线 | 修正语义后接入 `corrected_old_v1` | 冻结语言旁固定 bias 继续当主候选 |
| T2 情景查询、T3 目标/行动 | 条件后继 | R5 有净收益后按 R6 进入 | 当前排期、当前训练或当前能力主张 |
| CC-5/6/7、跨会话、自主扩域 | 条件后继 | 对应合同成立后进入 | 以“计划已写”当作已实施 |
| VISION 42 条/VX0–VX7 | 长期设计储备 | 取用具体条款作设计输入 | 发布当前工单或第二套主线 |
| 09 N1–N6 机制词典 | 组件参考 | 查机制责任与历史映射 | 作为“当前主计算图”或当前队首 |
| M0–M8、R2 日对话、UI 生命页 | 冻结资产/发布面 | 原批准范围与证据引用 | 重启旧路线、拼成绩或覆盖新验收 |
| 01 阶段地图、06/11/12 与旧交接 | 历史 | 追溯当时状态 | 作为下一步依据 |

统一规则：旧文档里的“当前”“下一步”“唯一动作”只解释其写作时点；当前调度只看第 3 节和 03。

## 5. 当前代码面

| 类别 | 路径 | 使用规则 |
|---|---|---|
| 当前主线 | `taiji/native_content_core.py`、`tests/taiji_native/test_native_content_core.py`、`scripts/training/probe_native_content_core_r3.py` | 只按 R3b/R4 合同扩展；改动走测试与探针 |
| 旧基线来源 | `taiji/joint_state_consumption*.py`、`conditional_language_v4.py`、`joint_input_producer.py`、`joint_t1_trainer.py`、`scripts/training/run_taiji_n1_plan177_f_f1.py`、`evaluate_taiji_n1_plan177_f_f1_holdout.py` | 只用于 `corrected_old_v1`；不在此路径继续叠局部头 |
| 数据与身份 | `plans/manifests/native_core_r1_public_fixture_v1.json`、`plans/manifests/` | 公开面可复用；sealed 面按原边界 |
| 运营与门禁 | `scripts/training/audit_*.py`、`check_plans_markdown_links.py`、`02`/`05`/`07`/`08` | 维护质量，不定义方向 |
| 历史实验 | 其余 `scripts/training/`、`reports/` 旧件 | 保留证据，不再新增同类 runner |

## 6. 文档层级（只读这些）

| 角色 | 文档 |
|---|---|
| 方向（唯一） | 本文件 |
| 状态与队首（唯一） | [03 当前执行](active/roadmap/03_CURRENT_EXECUTION.md) |
| 工作包与依赖 | [10 持续能力执行计划](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md) |
| 当前合同 | [重构方案](reference/PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)、[R1](reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md)、[R2](reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md)、[R3](reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md) |
| 规范约束 | [核心需求](active/TAIJI_CORE_REQUIREMENTS.md)、[原生架构](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) |
| 运营 | [02 门禁/CI](active/roadmap/02_GATES_AND_CI.md)、[05 技术债](active/roadmap/05_TECH_DEBT_REGISTER.md)、[07 交付](active/roadmap/07_MINI_MODEL_DELIVERY.md)、[08 重放](active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md) |
| 旧资产 | [LEGACY_ASSET_INDEX](reference/LEGACY_ASSET_INDEX.md) |
| 入口 | [plans/README](README.md)、[PLAN_INDEX](active/PLAN_INDEX.md) |

其余文档一律视为历史、证据或冻结合同，不再产生当前指令。

## 7. 维护规则（防止再次混乱）

1. 方向和队列只改本文件；新方向只能作为“候选”进入第 3 节。
2. 只有 03 能写“当前唯一下一步”。
3. `plans/active/` 下每份文档前 10 行必须有 `> 文档状态：[...]`，取值限：主线、状态、工作包、合同、运营、规范、组件、历史、证据、索引。
4. 旧文档不得被当前工作包引用为“下一步”；只能引用为证据、基线或背景。
5. 每阶段结束先更新本表与 03，再更新 10 和合同；不允许只更新报告。
6. 旧方向的脚本与计划不删除，但必须标记历史；不新增 legacy runner、不重跑旧 sealed。
7. 本规则由[单主线审计脚本](../scripts/training/audit_plan_single_mainline.py)与对应测试强制执行。

## 8. 本次收敛的决策记录

- 2026-10-30：用户指出仅做 R1/R2/R3 局部调整不足，要求整个项目收敛。据此建立本文件；将 09、VISION、N177、N178、T2/T3 与旧 roadmap 明确降级；重写 `plans/README.md`、`PLAN_INDEX.md`、`reference/README.md`；新增旧资产索引与单主线守卫。
- 保留：长期目标、N1–N6 机制身份、旧正负证据、冻结批准与来源/恢复纪律。
- 退出默认假设：辅助解析收敛会产生有用语义；继续补局部头/加剂量会自然拼成完整认知能力。
