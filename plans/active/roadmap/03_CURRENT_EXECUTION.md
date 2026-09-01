# Seed / Taiji 模型优先统一开发计划

> 计划基线：2026-09-01。本文是当前项目唯一执行计划、唯一优先级表和唯一“下一步”来源。架构合同仍由 `plans/active/` 下的核心需求与架构文档负责；历史 `W/R/P/C/E/S` 编号只作为已有提交、测试和报告的追溯标签，不再决定开发顺序。

## 1. 路线纠偏与当前事实

当前路线正式从“先完善持续进化和外围器官”纠正为“先证明基础能力，再立即进入模型训练”。

现有代码已经证明 checkpoint、局部突触更新、程序记忆、世界转移、结构候选、回滚、经验账本、Skill/MCP artifact 投影和客户端能力隔离等机制可以运行；这些结果是训练基础设施，不是通用能力证据。默认 Taiji 约有 `146,889` 个可更新标量，现有关键学习报告主要使用 2～8 条训练经验或几十条人工构造样本，不能证明自然语言理解、稳健规划、跨任务泛化或长期自主成长。

因此，项目对当前模型采用以下统一口径：

- Taiji 是**原生学习机制原型**，尚未证明具有稳定基础认知能力；
- E1～E7 形成的 ledger、checkpoint、内化、客户端隔离和归因机制全部保留，但降为训练底座；
- E8 已完成的 bounded replay 采样合同冻结保留，直到真实训练 checkpoint 通过基础能力评估后再接续；
- Qwen/provider 只作为实验语言器官或训练教师，不拥有 Taiji 的 Goal、Memory、WorldState、Plan、ActionIntent 和结构准入；
- Skill/MCP 可以贡献受治理的知识、示例和真实 Outcome，但在模型具备基础学习能力前不继续扩展真实第三方连接；
- 当前 CPU 主机足以完成最小能力基线、训练管线、微型训练和 checkpoint 验证；CUDA 保留为后续规模线，不阻塞模型优先主线。

本轮纠偏后的单一目标是：

> 先得到一个在独立未见数据上确实优于随机、冻结、规则和哈希基线，并能保存、恢复、继续训练的 Taiji checkpoint；随后才允许讨论持续进化、结构扩大和客户端身体扩展。

## 2. MiniMind 的参考边界

参考项目：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)。MiniMind 的价值是可复现的小模型训练工程，不是 Taiji 的目标架构。

Taiji 采用以下训练纪律：

| MiniMind 经验 | Taiji 采纳方式 |
|---|---|
| 预训练 → SFT → 偏好/Agent 训练的阶段顺序 | 改写为原生基础训练 → 世界/行动训练 → 语言与指令对齐 → 持续学习；禁止基础能力未形成就跳到 RL、Agent 或自进化宣传 |
| mini/full 分级数据与统一 JSONL | 建立 smoke、pilot、foundation 三档数据 manifest；每条数据记录来源、许可、目标、分区、provenance 和 taint 状态 |
| 独立的训练、推理和评估入口 | 新建 Taiji foundation trainer、独立 evaluator 和 checkpoint canary，不再把大量一次性 `eval_taiji_*` 当作训练主线 |
| 保存模型、优化状态、随机状态、epoch/step 并断点续训 | 扩展为保存 Taiji 权重/突触、局部可塑性状态、可选优化器、调度器、RNG、数据游标、目标权重、资源账本和 parent/child lineage |
| 训练损失、学习率、吞吐和定期权重保存 | 同时报告训练曲线、独立 holdout、retention、遗忘、恢复时间、checkpoint 大小和 CPU 成本 |
| 训练后独立推理 | 每个晋级 checkpoint 必须在全新进程中恢复，再运行五项基础能力和可读输出 canary |

MiniMind 官方 [预训练器](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py) 和 [SFT 训练器](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py) 在训练中执行反向传播、梯度裁剪、优化器更新和定期 checkpoint，并支持从模型、优化器、scaler、epoch 和 step 恢复。Taiji 可以复用这类成熟训练工程，但不能照搬以下内容：

- 不把 MiniMind/Qwen 的 Transformer block 变成 Taiji 大脑；
- 不把纯 next-token loss 当作 Taiji 智能的全部目标；
- 不照搬其 GPU 规模、词表、超参数或数据阈值；
- 不把 teacher/provider 的回答直接记成未经验证的世界事实；
- 不要求 Taiji 从零重训一个与成熟生态隔绝的新 tokenizer。

Taiji 采用“站在巨人肩膀上”的双边界：原始 byte 输入继续作为无损器官通道；成熟 tokenizer、embedding 或语言模型可以在训练期作为语义教师和语言器官，但其输出必须经过 provenance、约束和 holdout 隔离，Taiji 核心仍拥有持续状态、记忆、世界模型、目标、规划和行动选择。

## 3. 唯一优先级与阶段顺序

从本计划生效起只使用 `M0～M8` 作为当前开发顺序。任何历史阶段编号不得插队。

| 顺序 | 阶段 | 状态 | 主要产物 | 允许进入下一阶段的条件 |
|---|---|---|---|---|
| 0 | M0 CPU 五项基础能力真实性基线 | **已完成（M0-0/M0-1/B1/B2/B3/B4/B5/M0-3/M0-4）** | 数据合同、对照 evaluator、checkpoint preflight、基线报告 | 测量链可信且能保存/恢复；模型得分可以失败，但失败必须被如实记录 |
| 1 | M1 Taiji foundation 训练管线与首次 CPU 训练 | **当前进行（M1-4：F3 世界与行动 pilot）** | 原生 trainer、数据流水线、训练曲线、首个 child checkpoint | 未见数据相对父 checkpoint/对照有稳定净提升 |
| 2 | M2 世界—行动—语言后训练 | 待开始 | 世界预测、行动信用、ContentPlan/语言蒸馏和受控 SFT checkpoint | 任务成功、事实约束、旧能力保持同时通过 |
| 3 | M3 综合能力晋级与真实 Workbench 验证 | 待开始 | 独立评测套件、真实 Workbench longitudinal report、晋级 checkpoint | 至少一个真实任务族获得可重复净收益 |
| 4 | M4 持续学习、自进化和结构成长 | 冻结等待 M3 | bounded replay 接线、多周期保持、结构候选与单项回滚 | 真实 checkpoint 连续学习收益大于固定容量/weight-only 对照 |
| 5 | M5 Skill/MCP 数据飞轮与客户端身体 | 冻结等待 M4 | 知识内化、经验回流、IDE/Workbench 身体、客户端插件准入 | 认知收益与客户端执行收益可消融归因，权限和回滚闭合 |
| 6 | M6 语言 provider 与产品体验收口 | 冻结等待 M5 | provider watchdog、语言切换、HF 残留退役、桌面/视觉现场 | 产品能力与真实 Taiji 状态一致，packaged client 现场通过 |
| 7 | M7 全量 CI、仓库和发布收口 | 每阶段局部执行，最终集中验收 | 全矩阵 CI、发布 manifest、main/origin 收敛 | 阻塞 CI 全绿且发布包与 checkpoint 可追溯 |
| 8 | M8 CUDA 与规模化 | `hardware-blocked` | profiler、跨设备 checkpoint、稀疏/融合优化 | 真实 CUDA 主机上的收益和数值一致性通过 |

每次只允许一个主阶段处于“当前进行”。安全修复、相关定向测试和文档事实同步可以随主阶段执行，但不得借“并行支线”重新扩张方向。

估算日程按单开发者和当前 CPU 环境计算：M0 约 3～5 个开发日，M1 约 7～14 个开发日加实际训练墙钟时间，M2 约 7～14 个开发日，M3 约 4～7 个开发日，M4/M5 各约 1～2 周，M6 约 1～2 周，M7 约 3～7 个开发日；估算只用于排程，不替代 Gate。训练墙钟时间由资源 preflight 后写入 manifest，不提前虚构日期。

## 4. 当前进度与唯一下一步

M0-0 已完成并通过 `reports/taiji_m0_checkpoint_preflight_20260901.json`：父 checkpoint 落盘后关闭原进程，由全新 Python 进程恢复；恢复后的下一步预测和模型摘要与同状态期望值一致；child checkpoint 再次成功落盘。该门禁验证了训练前最关键的“能保存、能恢复、能继续”，但不代表五项能力已经形成。

M0-1 已完成并通过 `plans/manifests/taiji_foundation_baseline_v1.json`、`taiji/foundation_evaluation.py` 和 `reports/taiji_foundation_baseline_20260901.json` 的契约 canary。五项能力、四类对照、三 seed、四分区、样本下限、holdout 只读和报告字段已统一；当前报告为 `not_evaluated`，因为真实任务适配器尚未运行，这个失败/未评估状态是预期且必须保留的。

M0-2 已完成：适配器复用现有 Taiji perception、memory、world、action 和 continual-learning 接口，统一输出真实 measurement；没有调用 provider、前端或 MCP，holdout 阶段均保持只读。

M0-2 的 B1 适配器已完成并通过 smoke 证据 `reports/taiji_m0_b1_smoke_20260901.json`：真实 Taiji 在 3 个 seed 上优于 random 和 frozen-parent，但仍落后于 unigram 对照，故 B1 当前为失败而非晋级；holdout 更新数为 0。B1 smoke 只证明测量链和适配器可运行，不替代 manifest 规定的 foundation 数据量。

B2 适配器也已完成并通过 `reports/taiji_m0_b2_smoke_20260901.json`：Taiji episodic memory 能在训练分区写入并在 holdout/retention 只读召回，但三个 seed 中都没有稳定超过 memory lesion，故 B2 判失败。这说明当前记忆路径还没有形成可归因的因果增益，不能把“召回结果存在”当作记忆能力已经成立。

B3 适配器已完成并通过 `reports/taiji_m0_b3_smoke_20260901.json`：三 seed 的最差 holdout transition error 约为 `0.124`，优于冻结父模型最强对照约 `0.158`，retention 未明显退化且 holdout 更新数为 0；但 smoke 样本仍不足以形成 foundation 晋级结论。

B4 适配器已完成并通过 `reports/taiji_m0_b4_smoke_20260901.json`：动作器从冷启动状态出发，训练只用 cue→action→reward 的原生动作突触更新；三 seed 的 holdout success rate 均为 `1.0`，无奖励归因消融组均为 `0.5`，retention 为 `1.0`，且 holdout 更新数为 0。该结果只证明 B4 的因果测量链在 smoke 规模上成立，不替代 manifest 的 foundation 样本下限。

B5 适配器已完成并通过 `reports/taiji_m0_b5_smoke_20260901.json`：phase-B 确实从 phase-A 的 checkpoint 继续，holdout 更新数为 0，并记录了 replay 与无 replay 的旧能力变化；当前 smoke 的最差 backward transfer 为负，说明 Taiji 现有序列学习仍会遗忘，B5 任务保持失败而不是伪造通过。这是进入 M1 的直接训练目标：先改善持续学习与旧能力保持，再扩大训练规模。

M0-3/M0-4 已完成：统一矩阵报告为 `reports/taiji_foundation_baseline_20260901.json`，checkpoint gate 为 `passed`，整体 `status=failed`、`can_promote=false`。B1 使用真实中文数据的 foundation 分区（train `1,048,576`、holdout `131,072`、retention `131,072`），最差 seed 为 `6.497 BPB`，仍高于 unigram `5.942`，所以 F1 必须先改善 byte/边界/组合预测。B2 的 `0.75` 没有超过 memory lesion，说明记忆写入尚未形成稳定因果增益；B3 task-level 通过但只有 `8/4/4` 样本，B4 task-level 通过但只有 `32/16/16` 样本，二者都不能晋级 foundation；B5 最差 backward transfer 为 `-0.244`，但 phase-B checkpoint continuation 已被验证，旧能力遗忘是 F5 的直接目标。

M0 的零点已经足够可信，不能因 capability failed 继续扩大外围建设。M1-0～M1-3 已完成并进入 **M1-4：F3 世界与行动 pilot**；首轮固定 F1→F2→F3→F4→F5 顺序，优先目标是让 child checkpoint 在 B1/B2/B5 上出现可归因净提升，同时保持 B3/B4 不退化。

本步只允许创建或修改以下 owner：

- `plans/manifests/taiji_foundation_baseline_v1.json`：五项能力、数据分区、对照、seed、资源和报告 schema；
- `taiji/foundation_evaluation.py`：不带训练副作用的统一 evaluator；
- `taiji/foundation_tasks.py`：原生 B1～B5 任务适配器；当前已完成 B1、B2、B3、B4、B5；
- `tests/taiji_native/test_foundation_evaluation.py`：数据泄漏、对照、checkpoint 和失败口径 red；
- `tests/taiji_native/test_foundation_tasks.py`：原生任务适配器的真实状态和 holdout 只读回归；
- `scripts/training/eval_taiji_foundation_baseline.py`：CPU 基线入口；
- `reports/taiji_foundation_baseline_<date>.json`：首次真实性报告；
- `reports/taiji_m0_b1_smoke_<date>.json`：B1 smoke 真实性证据。
- `reports/taiji_m0_b2_smoke_<date>.json`：B2 smoke 真实性证据。
- `reports/taiji_m0_b3_smoke_<date>.json`：B3 smoke 真实性证据。
- `reports/taiji_m0_b4_smoke_<date>.json`：B4 smoke 真实性证据。
- `reports/taiji_m0_b5_smoke_<date>.json`：B5 smoke 真实性证据。

开始写 evaluator 前先核对已登记的 OpenAPI snapshot 漂移，确保现有 CI 基线没有被误当作本步新增失败。M0 报告生成后，不因模型分数低而回到外围建设；只要 checkpoint 和测量链可信，就立即进入 M1，低分直接成为首轮训练目标。

## 5. M0：CPU 五项最小能力验证方案

### 5.1 公共数据与对照规则

五项能力共用以下规则：

1. `train`、`holdout`、`retention`、`lesion/control` 按来源、文档、环境族或 episode 隔离，禁止随机拆散同一轨迹造成泄漏。
2. smoke fixture 只验证代码；能力报告不得再使用 2～8 条样本作为最终证据。首轮 baseline 至少达到：序列任务 `1 MiB` 训练字节并各有独立 `128 KiB` holdout/retention；其余任务每项至少 `1,000` 条训练样本和各 `200` 条 holdout/retention。若 CPU preflight 证明预算不可承受，只能缩短序列长度或分批流式读取，不能缩到失去统计意义。
3. 固定至少三个 seed，报告均值、标准差和最差 seed；单一有利 seed 不晋级。
4. 每项同时运行 random、frozen-parent、简单规则/统计、hash-only 和完整 Taiji。规则基线强于 Taiji 时必须如实失败。
5. evaluator 不得调用 provider、前端、MCP executor 或训练接口；holdout 期间任何权重、记忆、游标或 RNG 非预期变化都判失败。
6. 报告同时记录参数数、实际更新标量数、峰值内存、CPU 时间、吞吐、checkpoint 字节和恢复耗时。

### 5.2 五项能力

| 能力 | 输入与未见划分 | 核心指标 | 必需反证 |
|---|---|---|---|
| B1 序列预测与组合泛化 | 原始 byte 文本/结构流；按文档和组合模式留出 | bits-per-byte、next-byte accuracy、长短序列稳定性 | n-gram/Markov、随机 chunk、边界扰动、frozen parent |
| B2 延迟记忆与关联召回 | 新 key/cue、事件、延迟和干扰项；留出 key 与组合 | recall accuracy、延迟退化曲线、干扰后保持 | memory lesion、顺序打乱、只看最后输入、容量对照 |
| B3 世界状态与因果转移 | `state + action -> next_state/outcome`；按环境族留出 | transition error、校准、反事实动作区分 | 静态复制、忽略 action、打乱 action、frozen parent |
| B4 目标驱动行动与信用分配 | goal、affordance、action、Outcome；留出目标—能力组合 | success、regret、失败停止、纠正样本利用率 | 随机动作、最频动作、无 Outcome credit、无 memory |
| B5 连续学习与旧能力保持 | 按 A→B→C 顺序训练并保留独立旧任务集 | forward transfer、backward transfer、forgetting、恢复后延续 | 无 replay、无 consolidation、顺序反转、parent checkpoint |

B1 证明“输入中存在可学习规律”；B2 证明“状态可以跨时间保存并被 cue 取回”；B3 证明“模型区分行动造成的后果”；B4 证明“模型能围绕目标选择行动并使用结果更新”；B5 证明“学习不是每次从头开始”。五项都不等于通用智能，但它们是进入真实训练和自进化讨论的最低事实基础。

### 5.3 M0 交付顺序

1. M0-0：审计现有 checkpoint、参数计数、数据游标和已知 CI 基线；磁盘 checkpoint roundtrip 失败时先修复。**已完成。**
2. M0-1：冻结 manifest、JSONL/trajectory schema、分区方法和 baseline 实现；先写会失败的泄漏与副作用测试。**已完成。**
3. M0-2：实现统一 evaluator 和五项适配器；复用现有 `taiji/evaluation.py`、memory/world/action 接口，但不继续扩大一次性脚本。**已完成。**
4. M0-3：运行三 seed CPU baseline，输出一份能力矩阵，不用五份互相矛盾的“passed”报告。**已完成。**
5. M0-4：基于失败曲线冻结 M1 的首轮训练目标、资源预算和模型 tier，然后直接启动训练。**已完成，进入 M1。**

M0 的退出条件不是“五项全绿”。M0 的目标是得到可信零点；如果五项全部失败，只要数据、对照和 checkpoint 可信，也必须进入 M1，而不是继续做插件、UI 或 Gate 外围。

## 6. M1：验证后立即进入的训练方案

### 6.1 训练系统

新增统一入口 `scripts/training/train_taiji_foundation.py` 和 owner `taiji/foundation_training.py`。训练器必须支持：

- `smoke`：分钟级验证读取、更新、保存、恢复和报告；
- `pilot`：当前 CPU 可承受的正式训练，产生可比较 child checkpoint；
- `foundation`：数据和模型规模可扩展的完整配置，当前硬件不强行运行；
- `--resume`：从保存的 dataset cursor 和学习状态继续，不重复消费或跳过数据；
- `--eval-only`：全新进程只读 checkpoint，禁止训练副作用；
- 定期保存 `last`、`best-holdout` 和显式里程碑 checkpoint，不覆盖唯一可恢复父版本。

checkpoint 至少包含：模型/突触状态、结构 revision、局部可塑性状态、可选 optimizer/scaler/scheduler、RNG、epoch/step、数据 manifest/digest/cursor、目标权重、指标曲线、资源账本、parent/child lineage 和代码 revision。训练前必须先做“保存 → 结束进程 → 恢复 → 继续一步 → 再保存”的真实磁盘 canary。

### 6.2 数据流水线

使用统一 JSONL/trajectory 容器，但不把不同目标强行压成聊天文本：

```text
sample_id / source / license / objective / partition
payload / target_or_outcome / provenance / taint / content_digest
```

数据分三档：

- `foundation-smoke`：仓库内最小可复现样本，只测管线；
- `foundation-pilot`：CPU 首训数据，覆盖 byte 流、记忆 episode、world transition 和 goal/action/outcome；
- `foundation-full`：后续大规模开放数据与真实 Seed 经验，不在当前 CPU 阶段强行下载或训练。

MiniMind 的公开数据格式和清洗思路可作为文本阶段参考，但任何外部数据进入 Seed 前必须核对许可、来源、去重、语言比例、长度分布和污染。Skill/MCP 说明和调用结果分别标记为 `knowledge` 与 `experience`，不直接拼接 secret、执行源码、holdout 答案或 provider 幻觉。

### 6.3 两时间尺度学习

Taiji 不把“原生”误解成“禁止成熟优化方法”。训练采用互补的两时间尺度：

- **发展期离线学习**：在可微模块上使用成熟优化器、梯度裁剪、课程学习和教师蒸馏，承担从较大数据中形成表征、世界预测和行动策略的任务；
- **运行期局部学习**：保留突触局部预测、Outcome credit、记忆写入和受控结构可塑性，承担部署后的增量适应。

二者写入同一版本化 Taiji checkpoint，但必须能分别 lesion。离线梯度不把 Transformer 变成核心；在线局部学习也不能因为“更像生物”而免除独立 holdout。若某个现有稀疏模块尚不可微，先保留其局部学习，并通过接口接入发展期目标；不得为追求统一优化器重写整个 Taiji。

### 6.4 首轮训练课程

M1 按固定顺序执行：

1. F1 感知与预测：byte/边界/组合预测，目标是降低独立 holdout BPB，而不是生成漂亮文本。
2. F2 记忆与时间：延迟召回、干扰保持、episode/provenance 绑定。
3. F3 世界与行动：state/action/outcome 预测、目标选择、失败停止和 credit。
4. F4 联合短训：在同一 checkpoint 上混合 F1～F3，并用 retention 防止单项训练互相覆盖。
5. F5 首次晋级：全新进程恢复最佳 checkpoint，重新跑 M0 五项矩阵。

模型规模不按愿望预设。先训练默认 micro tier，再根据“holdout 仍改善但容量曲线持续受限”提出 pilot tier；只有固定容量相对 weight/memory/route 调整持续失败，才允许增加区域、神经元或连接。参数增加本身不是收益。

M1 晋级至少要求：三个 seed 中 child checkpoint 在预注册主指标上稳定优于 frozen parent 和最强简单基线；至少四项不退化，一项出现明确净提升；checkpoint 恢复后结果一致；训练/holdout 无泄漏；CPU 成本处于 manifest 预算。未达到时继续同一阶段修数据、目标或学习规则，禁止绕到外围功能。

### 6.5 M1 已执行边界与当前冻结决策

M0 的 CPU 实测 foundation 矩阵约耗时 42 分钟，说明当前机器只能先做短 pilot，不能直接启动 foundation 全量训练。M1-0 固定如下边界：

- 模型从 `TaijiConfig` 的 `micro` tier 开始；不扩神经元规模、不接 CUDA、不引入 Transformer/provider；
- 训练顺序固定为 F1→F2→F3→F4→F5，每个阶段结束保存 `last`、阶段里程碑和可恢复 cursor；
- F1 使用 `data/simple_zh/dialogue_extended_clean.jsonl` 的受控 train 分区，pilot 只取可在当前 CPU 完成的窗口；F2～F4 使用结构化本地 corpus，不把 goal/world/outcome 压成聊天文本；
- 第一优先修复 F1 的 unigram 反超、F2 的 memory-lesion 无差异和 F5 的负 backward transfer；B3/B4 只要求短训不退化，不能用它们的 smoke 通过抵消前述失败；
- 每个 pilot seed 都执行磁盘 `save→新进程 restore→继续一步→再 save`，并把数据 manifest、cursor、parent/child digest、代码 revision 和指标曲线写入同一训练报告；
- M1-0 的唯一退出条件是产生可恢复、可比较的 child checkpoint；是否晋级由 F5 重新运行 M0 五项矩阵决定，不由训练 loss 单点决定。

M1-0 已完成：`taiji/foundation_training.py` 和 `scripts/training/train_taiji_foundation.py` 已实现 content-addressed JSONL 分区、cursor 分块训练、`parent/last/best-holdout` 原子 checkpoint、`--resume` 和 `--eval-only`。`reports/taiji_m1_smoke_20260901.json` 的 smoke 训练使用 `4096/1024/1024` 字节，parent holdout 为 `9.025 BPB`，best holdout 降到 `6.864 BPB`；新进程 eval-only 保持 checkpoint 只读，回归测试为 `12 passed`。训练报告现在在落盘前写入 `report_path`，避免报告内容与终端输出不一致。这只是管线可用性证据，不代表模型已经达到 M1 晋级标准。

M1-1 已完成：`reports/taiji_m1_pilot_20260901.json` 记录了 micro/seed 11 的 F1 pilot，parent holdout 为 `9.063 BPB`，best/final holdout 为 `6.384 BPB`，retention 为 `6.435 BPB`；16 个 cursor chunk 完成，parent/last/best checkpoint 均落盘，新进程 eval-only 报告确认 `checkpoint_read_only=true`。这证明 F1 pilot 管线产生了真实改善迹象，但单 seed 不足以晋级。

M1-2 已完成：固定 `partition_seed=11` 后，三个 pilot 报告的 dataset digest 均为 `370e9edc…`，每个 seed 的 16 个 chunk 都完成，final holdout 相对 parent 均改善：seed 11 为 `9.063→6.384 BPB`，seed 29 为 `8.738→6.395 BPB`，seed 47 为 `9.397→6.246 BPB`；三个 last checkpoint 均通过新进程 eval-only，且 holdout/retention 只读。F1 pilot 已具备稳定净改善迹象，但尚未运行完整 M0 重测，因此不宣称 M1 晋级。

M1-3 已完成：新增 `MemoryTrainingRun` 与 `scripts/training/train_taiji_memory.py`，以结构化 `MemoryEpisode(cue, action, outcome, provenance)` 接入原生 episodic field，不把记忆目标压成聊天文本。F2 smoke 的 8 episode 结果为 `0.750→0.625`，且 lesion 为 `0.750`，按预注册口径保留为失败证据，说明样本过小不能支撑记忆因果增益。扩大到固定 corpus digest `53ed3250…` 的 64 episode pilot 后，seed 11/29/47 的 parent→final recall 分别为 `0.5625→0.671875`、`0.484375→0.53125`、`0.53125→0.703125`；三者 final 均高于各自 memory-lesion，retention 与 final 一致，holdout/retention 更新数均为 `0`。三个 last checkpoint 均在全新进程中 eval-only 复核，报告 `checkpoint_read_only=true`。F2 pilot 达到本阶段的“有因果净增益、无 retention 退化、可恢复”条件，但这不等于 M1 总体晋级，仍需 F3/F4/F5 和完整 M0 重测。

当前唯一下一步是执行 **M1-4 F3 世界与行动 pilot**：沿用 F2 的可恢复 checkpoint 边界，建立结构化 `state/action/outcome/goal` corpus，把世界转移预测、目标动作选择、失败停止和 Outcome credit 接入真实 Taiji world/action 路径；先跑 8～16 条 smoke 验证 checkpoint/cursor/holdout 只读，再固定同一 corpus 分区跑三个 seed 的 64 条 pilot。F3 完成前不进入 F4，也不切换到 provider、Workbench、Skill/MCP 或客户端外围。

## 7. M2～M8 的开发日程与外围任务安置

### M2：世界—行动—语言后训练

在 M1 checkpoint 上继续，而不是重新初始化：

- 用真实但受控的 Workbench trajectory 扩大世界模型和行动信用；
- 用成熟语言模型/embedding 作训练教师，将语义对齐到 Taiji 的 Percept、Goal、WorldState、ContentPlan，而不是让 provider 直接选择工具；
- 进行 SFT-like 指令对齐，先做解释、问答、澄清和 ContentPlan，再做需要 approval 的 IDE 行动；
- Qwen2.5-0.5B 继续作为失败基线；更强 provider 只有通过相同质量 Gate 才能成为语言器官候选；
- DPO/RLAIF/Agent 训练保持关闭，直到监督阶段输出稳定且 evaluator 可验证事实和行动结果。

### M3：综合能力晋级

- 将小型模拟 Gate 保留为快速回归，不再当作真实能力结论；
- interaction-group 在训练后重新评估，并与最强单体、随机组、稠密平均、weight-only、memory-only 和 route-only 比较；
- 在真实 Workbench 留出项目完成至少一个纵向任务族；
- 同时报告语言可读性、工具决策、世界预测、恢复、遗忘、资源和 lesion；
- 只有 M3 晋级 checkpoint 才能成为 Seed 默认认知候选。

### M4：持续学习、自进化与结构成长

- 把已完成的 bounded replay 接入真实训练 `consolidate()`，而不是人工微型经验；
- 连续多个 checkpoint 周期验证净能力收益、旧能力保持、污染隔离和 rollback；
- Skill/MCP/Workbench 的 correction、failure、success 按 provenance 进入 replay；
- 先尝试 weight、route、memory 和 learning-rule 调整，持续容量失败后才允许结构增长；
- 每次结构变化只改变一项，做 no-change、weight-only、memory-only、route-only、structure-only 消融。

### M5：Skill/MCP 数据飞轮与客户端身体

- Skill/MCP 文本作为知识语料，真实调用作为经验语料；内化后必须关闭外部来源做 deletion/lesion 评审；
- MCP connector、executor、permission、resource 和 UI 由 Seed 客户端 capability 继承，不写入 Taiji 神经网络；
- 恢复 E6-5b 前必须明确第三方目标、transport、网络、凭据引用、owner、approver 和撤销责任；未提供时继续保持 `connection_attempted=false`；
- 客户端插件热插拔只负责 Vue route/sidebar/panel/command/settings/visualization 与 Workbench capability；桌面根壳仍采用安全重启更新；
- 完成“模型规划 → Workbench preview/approval → IDE 执行 → Outcome 回流”，包括依据文件证据自主选择/切换 IDE language；
- 同一次试验不得同时改变 Taiji checkpoint 和客户端插件后把收益归给一方。

### M6：provider、Legacy 和产品体验收口

- provider watchdog、真实 artifact rotation、cooldown、previous/native fallback 和重启重绑在此阶段完成；
- 清除产品 live 路径残余 HF/GGUF/Transformer 格式切换，保留必要迁移 tombstone 的明确期限；
- 生命状态只保留一个产品入口，多维状态全部来自真实 runtime projection；
- 侧边栏、IDE、训练、知识、Agent 和设置与实际能力对齐；
- Windows 窗口圆角、任务栏、托盘、通知 logo、DPI、键盘导航和 reduced-motion 完成 packaged `Seed.exe` 现场取证；
- 视觉美化只能表达真实状态，不能用动画或百分比掩盖模型能力缺口。

### M7：CI、仓库和发布

CI 不是最后才运行的支线：每个 slice 都运行相关 pytest/lint/type/API/frontend/build/checkpoint 检查，禁止新增红线；M0、M1、M3、M5、M6 结束各跑一次阶段全量矩阵。M7 只负责最终集中收口：

- 修复全部剩余 Python、OpenAPI、前端、Legacy-off、checkpoint、Windows/package 问题；
- 确认没有关键 job 被 skip 或允许失败；
- 审计并收束 main、worktree、refs 和 origin/main；
- 发布 manifest 绑定 commit、数据 digest、训练配置、checkpoint、报告、前端字节和 `Seed.exe`。

当前已知 `test_openapi_snapshot` 漂移必须在 M0-0 显式核对新增 API 后修复或回退，不能通过静默放宽测试处理。

### M8：CUDA 与规模化

当前保持 `hardware-blocked`，但不从计划删除。真实 CUDA 主机可用后执行：同一 workload CPU profiler → CUDA profiler → CPU→CUDA→CPU checkpoint → 数值/结构/预算一致性 → 热点优化。只有 profiler 证明瓶颈后才实现 fused/sparse kernel；不能用 CPU 结果宣称 CUDA 已适配。

## 8. 全阶段阻塞规则

出现以下任一情况必须停止当前 slice，先修复：

- checkpoint 不能保存、关闭进程恢复或继续一步；
- holdout/retention 被训练消费，或数据 digest/partition 不可追溯；
- evaluator、provider、前端或 prompt 暗中提供预期答案或最终 ActionIntent；
- 训练指标只在 train 改善，未见数据不改善；
- 新增 CI 失败、OpenAPI 漂移未评审、关键 job 被 skip；
- 需要真实外部凭据、联网执行、第三方 MCP 激活或不可逆删除但没有明确授权；
- 需要改变 Taiji/Seed/provider 所有权边界；
- 用增加参数、神经元、插件、测试数量或动画效果替代能力收益。

每个 slice 固定交付：实现、定向测试、结构化 report、计划事实同步、`git diff --check`、提交。归档中的历史“下一步”全部失效。

## 9. 历史成果的重新定位

- E1～E4：保留为数据、checkpoint 和内化基础设施；其微型 Gate 不再代表基础智能。
- E5～E7：保留为客户端边界、权限和因果归因基础设施；在 M5 前不继续扩展真实连接。
- E8：bounded replay 采样已完成，主训练接线推迟到 M4。
- E9/R4：映射到 M8，保持硬件阻塞。
- 旧 P/C/R/S 系列：完成记录进入 Git、report、manifest 和归档快照，不再拥有执行优先级。
- Legacy NeuroPlex/TinyStories Transformer 训练脚本：只作冻结对照和训练工程参考，不能充当 Taiji foundation trainer。

计划历史完整快照见 [roadmap_convergence_20260901](../../archive/history/roadmap_convergence_20260901/README.md)。
