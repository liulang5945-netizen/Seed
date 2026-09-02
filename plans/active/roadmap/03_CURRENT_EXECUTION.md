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
| 1 | M1 Taiji foundation 训练管线与首次 CPU 训练 | **当前进行（M1-6：F5 首次晋级）** | 原生 trainer、数据流水线、训练曲线、首个 child checkpoint | 未见数据相对父 checkpoint/对照有稳定净提升 |
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

M0 的零点已经足够可信，不能因 capability failed 继续扩大外围建设。M1-0～M1-5 已完成并进入 **M1-6：F5 首次晋级**；首轮固定 F1→F2→F3→F4→F5 顺序，优先目标是让 child checkpoint 在 B1/B2/B5 上出现可归因净提升，同时保持 B3/B4 不退化。

本步只允许创建或修改以下 owner：

- `plans/manifests/taiji_foundation_baseline_v1.json`：五项能力、数据分区、对照、seed、资源和报告 schema；
- `taiji/foundation_evaluation.py`：不带训练副作用的统一 evaluator；
- `taiji/foundation_tasks.py`：原生 B1～B5 任务适配器；当前已完成 B1、B2、B3、B4、B5；
- `tests/taiji_native/test_foundation_evaluation.py`：数据泄漏、对照、checkpoint 和失败口径 red；
- `tests/taiji_native/test_foundation_tasks.py`：原生任务适配器的真实状态和 holdout 只读回归；
- `scripts/training/eval_taiji_foundation_baseline.py`：CPU 基线入口；
- `scripts/training/eval_taiji_f5_promotion.py`：从 F4 child 全新进程恢复并执行 M0 晋级覆盖审计；
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

M1-4 已完成：新增 `WorldActionTrainingRun` 与 `scripts/training/train_taiji_world_action.py`，将 `WorldDynamicsLearner` 的原生局部 world update 与 Taiji kernel 的 goal→action→reward credit 放入同一个可恢复 checkpoint。F3 smoke 的 world holdout error 为 `2.0576→0.0514`，goal success 为 `0.5→1.0`；固定 64 条 world + 64 条 goal pilot 后，seed 11/29/47 的 world error 分别为 `1.8871→0.0000137`、`0.1570→0.0000742`、`0.4283→0.00000795`，goal success 均为 `0.5→1.0`，credit-lesion 均为 `0.5`，world transition rejection 为 `0`。三个 last checkpoint 均由全新进程 eval-only 复核，报告 `checkpoint_read_only=true`；F3 pilot 达到阶段退出条件，但不代表 M1 总体晋级。

M1-5 已完成：`JointTrainingRun` 与 `scripts/training/train_taiji_joint.py` 将 F1 byte prediction、F2 delayed memory、F3 world/action 串到同一 checkpoint lineage，checkpoint 同时保留模型、世界学习器、三个数据 digest、四个 cursor、parent lineage 和指标历史。正式 F4 pilot 矩阵见 `reports/taiji_m1_f4_pilot_matrix_20260901.json`：固定 partition/corpus 下 seed 11/29/47 的 sequence holdout BPB 分别为 `8.0056→5.2096`、`8.0056→5.4118`、`8.0056→5.4045`；memory recall 分别为 `0.5→0.65625`、`0.5→0.53125`、`0.5→0.65625`；world holdout error 均下降到 `1.37e-5`、`7.42e-5`、`7.95e-6`；goal success 均为 `0.5→1.0`；retention 同步改善，credit-lesion 均为 `0.5`，holdout updates 与 world transition rejections 均为 `0`。三个 last checkpoint 均在全新进程中 eval-only 复核且 `checkpoint_read_only=true`，因此 F4 达到阶段 Gate；但这只是联合短训通过，不代表 M1 总体或 M0 五项晋级。

F4 过程中发现并修复了一个真实恢复错误：float32 范数边界的 1 ulp 舍入会让 `SparseSynapses.load_payload()` 在恢复时二次改写权重，破坏 checkpoint digest 和只读评估。`ab4b079` 增加了容差幂等边界与回归测试；seed 11 从修复前保存的 `last.pt` 继续完成，随后三 seed 的新进程 eval-only 均通过。这次修复是 checkpoint 正确性的基础修复，不是模型能力收益。

M1-6 的 F5 首轮审计已完成，报告为 `reports/taiji_m1_f5_promotion_20260901.json`。三个 F4 `best-holdout.pt` 均在全新进程恢复；sequence、memory、world、goal 四类联合指标在 seed 11/29/47 均相对各自 parent 改善，`checkpoint_read_only=true`。但报告状态必须是 `blocked`：F4 pilot 只有 B1 `16384/4096/4096` 字节、B2 `64/64/64`、B3 `64/32/32`、B4 `64/32/32`，均未达到 manifest 的 foundation 下限；B5 没有可审计的 phase-A→phase-B 连续学习分区；同时 F4 child 上尚未重算完整 M0 controls。四项指标的 pilot canary 不能替代完整能力晋级。

为避免把扩大数据误当成旧 pilot 的 `resume`，已补充显式 continuation 接口：`JointTrainingRun.from_continuation_checkpoint()` 和 `train_taiji_joint.py --continue-from` 会校验源 checkpoint digest，重新计算扩展数据上的 parent，清零新课程 cursor，并保存 `continuation_source_checkpoint_digest` lineage。报告 `reports/taiji_m1_f5_continuation_canary_20260901.json` 已用 seed 11 的 F4 best checkpoint 完成 smoke canary：源 lineage、扩展 parent、全新 `parent/last/best-holdout`、训练后恢复和 eval-only 均通过；final sequence BPB `5.4676→5.0399`、memory recall `0.6667→0.8333`、world error `0.007176→0.0000183`、goal success 保持 `1.0`，且 `holdout_updates=0`、world rejection 为 `0`。这只证明 continuation 链路正确，不改变 F5 的能力 Gate 状态。

为进入真实覆盖量，`scripts/training/train_taiji_joint.py` 已开放 `--profile foundation`，默认使用 manifest 规定的 B1 `1,048,576/131,072/131,072` 字节预算、B2/B3/B4 各 `1,000` 条训练样本，并把默认 checkpoint 间隔提高到 `256`，避免 CPU 上每个小步都重复扫描大 holdout。该入口只扩大数据覆盖，不扩大 Taiji 神经元规模、不接 Transformer/provider；正式运行仍需先完成磁盘 checkpoint canary，并把三 seed 的训练耗时和磁盘占用写入报告。

F5 的 continuation 现在还支持显式 `--replay-corpus/--replay-profile/--replay-epochs`。回放阶段有独立 `phase=replay`、cursor、epoch 和 corpus digest，和新 phase-B 的训练状态一起写入同一 checkpoint；普通 `--resume` 会严格校验 replay digest，避免恢复时悄悄换 replay 数据。continuation 回归测试已覆盖 replay 执行、磁盘恢复和 eval-only 只读。这使 B5 的“phase-A→phase-B→replay”成为训练链上的真实阶段，而不是只在 evaluator 里模拟。正式 full-coverage 首次启动发现 F2 合成 corpus 的 cue 生成器仍按 `65+index` 递增，在 `count=1,000` 时越过 byte sensor；该数据合同错误已修复为有界且可重复的 cue 周期，并加入 foundation-scale 回归测试，未产生有效训练 checkpoint。

replay CLI canary `reports/taiji_m1_f5_replay_canary_20260901.json` 已通过：从 F4 seed 11 best 恢复，完成扩展课程后实际进入 `phase=replay`，回放 digest 为 F4 pilot 数据 digest `370e9edc…`，并生成 parent/last/best 三类 checkpoint；全新进程的恢复与只读评估保持通过。canary 仍是 smoke 规模，只证明课程和 checkpoint 链路，不能替代 full-coverage 能力测量。

首次 full-coverage seed 11 启动前的父 checkpoint 保存通过，B1 已实际完成 `1,048,576` 字节；随后发现默认每 `256` 步做完整 holdout 扫描会在 CPU 上产生数分钟停顿。进程在安全的 `memory_cursor=301/1000` checkpoint 处停止，未丢失训练状态；新增 `metric_interval`，将“每步保存 last”和“低频计算指标”分离。恢复时可提高 metric 间隔，仍在每个阶段末测量并保留 full Gate 所需的最终指标。

继续验证恢复流程时又发现：普通 `--resume` 已携带 checkpoint 中的 `parent_metrics`，但构造器仍无条件重算一次大 holdout。现在改为有已保存 parent metrics 时直接复用，只有新课程 continuation 必须重新测量扩展数据 parent；这不改变 Gate 数值，只去除 resume 的重复 CPU 扫描。该修复提交前会重新跑训练恢复回归。

seed 11 的 full-coverage continuation 已完成，报告为 `reports/taiji_m1_f5_full_seed11_20260901.json`，独立进程复核为 `reports/taiji_m1_f5_full_seed11_eval_only_20260901.json`。B1/B2/B3/B4 训练覆盖分别为 `1,048,576/1,000/1,000/1,000`，replay 实际消费 `16,384` 字节，`world_transition_rejections=0`、`holdout_updates=0`；训练报告与 eval-only 的 8 项指标和 checkpoint digest 完全一致。seed 11 的扩展 parent→final 为 sequence `5.3649→4.8648 BPB`、memory `0.570→0.536`、world `1.57e-5→3.16e-8`、goal `1.0→1.0`，所以 memory 退化使该 seed 不能单独晋级，但它证明 full-coverage 课程和恢复链路可运行。

seed 29 在 world `648/1000` 保存时暴露 Windows 读者锁竞态：`last.pt` 与较新的 `last.pt.tmp` 均通过独立加载和 digest 校验，训练进程退出但没有 checkpoint 损坏。`JointTrainingRun.save()` 已增加短指数退避重试，避免 eval-only/状态读取造成的瞬时 `PermissionError`；恢复将优先使用已验证的 `last.pt.tmp`（world `649/1000`、global step `1713`），再继续同一课程。

seed 29 的 full-coverage continuation 已完成，报告为 `reports/taiji_m1_f5_full_seed29_20260901.json`，独立进程复核为 `reports/taiji_m1_f5_full_seed29_eval_only_20260902.json`。B1/B2/B3/B4 覆盖分别为 `1,048,576/1,000/1,000/1,000`，replay 实际消费 `16,384` 字节，`world_transition_rejections=0`、`holdout_updates=0`；训练报告和 eval-only 的 checkpoint digest 均为 `2fd8d98d…`，8 项指标逐项一致，且 `checkpoint_read_only=true`。seed 29 的扩展 parent→final 为 sequence `5.4686→4.8460 BPB`、memory `0.489→0.502`、world `6.6895e-4→2.9632e-8`、goal `1.0→1.0`，全部主指标不退化并有净收益；但仍需 seed 47 和三 seed 聚合 Gate，不能单独宣称 M1 晋级。

seed 47 的 full-coverage continuation 已完成，报告为 `reports/taiji_m1_f5_full_seed47_20260901.json`，独立进程复核为 `reports/taiji_m1_f5_full_seed47_eval_only_20260902.json`。B1/B2/B3/B4 覆盖分别为 `1,048,576/1,000/1,000/1,000`，replay 实际消费 `16,384` 字节，`world_transition_rejections=0`、`holdout_updates=0`；训练报告和 eval-only 的 checkpoint digest 均为 `6f548721…`，8 项指标逐项一致，且 `checkpoint_read_only=true`。seed 47 的扩展 parent→final 为 sequence `5.4107→4.8173 BPB`、memory `0.585→0.493`、world `7.9789e-6→3.0604e-8`、goal `1.0→1.0`；sequence/world/goal 有净收益，但 memory 退化，因此该 seed 不能单独晋级。

在 seed 47 完成后的代码审计中确认：此前“联合训练保存已重试”的记录与实现不一致，重试只存在于旧的 `FoundationTrainingRun.save()`，`JointTrainingRun.save()` 仍会在首次 `PermissionError` 时失败。新增的回归测试先在旧实现上复现失败，再验证联合训练保存的五次有限指数退避；该修复已通过 `tests/taiji_native/test_foundation_training.py` 全部 `7 passed`、ruff 和 `git diff --check`，不改变 checkpoint 内容或原子替换语义。

三个 full-coverage seed 的聚合审计已完成，报告为 `reports/taiji_m1_f5_full_promotion_20260902.json`，状态必须记录为 `blocked`。B1 覆盖为 `1,048,576/131,072/131,072`，B2/B3/B4 覆盖分别达到 `1000/1000/1000`、`1000/500/500`、`1000/500/500`；三个 seed 的训练报告与独立 eval-only 在 checkpoint digest 和 8 项联合指标上逐项一致，`checkpoint_read_only=true`、`holdout_updates=0`、`world_transition_rejections=0`，且 replay history 真实执行。phase-A 旧能力审计显示 memory holdout recall 在三个 seed 均下降，backward-transfer score 分别为 `-0.140625`、`-0.015625`、`-0.0625`，所以 F5 不能晋级；同时 B5 专用 task-unit holdout/retention 和 random/simple-rule/hash-only/full M0 controls 仍未注册或重算。

当前唯一下一步是 **M1-7：修复 memory 保持与 B5 Gate 口径**：先为 phase-A replay 建立独立、可计数的 B5 train/holdout/retention 分区和 no-replay counterfactual，再调整回放/记忆学习规则，使三 seed 的旧 memory 能力不低于 parent；之后重新执行 full-coverage 训练、只读复核和完整 M0 controls。该 Gate 未通过前不进入 M2，也不切换到 provider、Workbench、Skill/MCP 或客户端外围。

M1-7 的第一片已完成：新增 `plans/manifests/taiji_continual_memory_v1.json`、`ContinualMemoryCorpus`/`ContinualMemoryTask` 和 `scripts/training/eval_taiji_b5_memory.py`。B5 现在有显式 phase-A train/holdout/retention、phase-B 干扰与 novel-cue 控制、replay train，并实际比较 no-replay counterfactual 与 replay 的旧能力变化；所有 holdout/retention 读操作保持 `holdout_updates=0`，语料 digest 可复现。canary 报告 `reports/taiji_m1_b5_memory_canary_20260902.json` 为 `failed`：phase-A 旧 recall `0.75`，no-replay 后 `0.50`，replay 后 `0.625`，replay 相对 no-replay 有 `+0.125` 因果增益，但仍有旧记忆回退，说明仅有当前事件重放还不足以恢复 memory。该失败是下一片 memory replay 接线的输入，不进入 full retrain。

M1-7 的第二片已完成：`JointTrainingRun` 与 `scripts/training/train_taiji_joint.py` 新增可校验的 `replay_memory_corpus`、epoch/cursor/digest 和 `phase=replay-memory`，byte replay 后会真正重放 phase-A memory episodes；普通 resume 会拒绝 replay memory digest 不一致。canary `reports/taiji_m1_f7_memory_replay_canary_20260902.json` 及独立复核 `reports/taiji_m1_f7_memory_replay_canary_eval_only_20260902.json` 均通过链路校验，memory recall `0.75→0.875`，checkpoint digest `e0033f12…`，`checkpoint_read_only=true`；相关训练/任务测试共 `13 passed`。这证明接线有效，但还不代表 foundation full-coverage 的三 seed B5 已通过。

当前唯一下一步是 **M1-8：用 memory replay 重跑三 seed full-coverage F5**：从三个 F4 `best-holdout.pt` 沿原 lineage 继续，保持 B1/B2/B3/B4 foundation 配置与 byte replay，同时加入 phase-A memory replay（初始 count `64`），重新生成三 seed child、独立 eval-only 和 B5 backward-transfer/no-replay 审计。结果未满足旧 memory 不退化前，不进入 M2，也不切换到 provider、Workbench、Skill/MCP 或客户端外围。

M1-8 的三 seed full-coverage 重跑与独立 eval-only 已完成，训练报告分别为 `reports/taiji_m1_f8_full_memory_replay_seed11_20260902.json`、`reports/taiji_m1_f8_full_memory_replay_seed29_20260902.json`、`reports/taiji_m1_f8_full_memory_replay_seed47_20260902.json`，复评报告使用同名 `_eval_only` 后缀。三组均满足 B1 `1,048,576/131,072/131,072`、B2 `1000/1000/1000`、B3/B4 `1000/500/500`，checkpoint digest 与复评指标逐项一致，`checkpoint_read_only=true`、`holdout_updates=0`、`world_transition_rejections=0`，并真实写入 byte replay 与 `phase=replay-memory` history；但 phase-A old memory 仍分别下降 `-0.140625`、`-0.03125`、`-0.015625`，因此不能晋级。聚合报告为 `reports/taiji_m1_f8_memory_replay_promotion_20260902.json`，状态 `blocked`。

M1-8 的 B5 专项 foundation 审计报告 `reports/taiji_m1_b5_memory_foundation_20260902.json` 是旧测试合同下的历史结果，不能再作为当前 Gate 依据：它把同一 cue 绑定到 phase-A/phase-B 的相反 action，属于不可判定的输入输出合同，且旧配置中的 lifetime confidence decay 会让写入越多回忆越弱。

M1-9 已完成第一轮修复但仍未晋级。`TaijiConfig.memory_confidence_decay` 的 native 默认改为 `0`，保留非零值仅作 legacy 兼容；memory replay 学习比例进入配置、PendingExperience、checkpoint 和 report；joint CLI 现在拒绝 partial/unrelated replay，启用 replay 时必须使用与当前 memory course 完全相同的 corpus digest，并记录 `exact-current-memory-corpus` 关系。相关 focused tests 为 `13 passed`，代码与 checkpoint 语义检查通过。

B5 合同同步改为 disjoint cue，测量共享容量干扰而不是同一输入的矛盾答案。有效 foundation 报告为 `reports/taiji_m1_m9_b5_disjoint_foundation_20260902.json`，三 seed 的 replay causal gain 分别为 `0.665/0.560/0.725`，old backward transfer 分别为 `0.050/0.005/-0.020`，但 replay 后 phase-B new recall 全部降为 `0`，总体 Gate 仍为 `blocked`。joint exact-memory canary `reports/taiji_m1_m9_exact_memory_canary_v2_20260902.json` 记录了 corpus 关系正确，但 memory recall 仍从 `0.75` 降至 `0.6875`。因此当前可以确认：生命周期衰减是一个规模硬编码问题，重复写回又是一个会覆盖新读出的巩固问题；不能继续用 full retrain 掩盖这两个问题。

当前唯一下一步改为 **M1-10：实现非破坏性 memory consolidation**：把 replay 从“再次写入完整 episode”改成可验证的 reactivation/consolidation 路径，分别控制 association、action readout、new-skill readout 的更新；新增 no-replay、replay、repeated-replay、readout-only、association-only 和 no-change 对照，并要求旧 holdout/retention 与 phase-B new holdout 同时不退化。M1-10 canary 未通过前不重新跑三 seed foundation，不进入 M2，也不切换到 provider、Workbench、Skill/MCP 或客户端外围。

M1-10 的第一轮诊断已完成：固定 action encoder 作为 recall 解码参照的对照，在 disjoint B5 canary 上仍只能把 old recall 从 `0.125` 提到 `0.25`，无法同时保留 phase-B new recall，因此不接入主路径。当前应重构的是共享低维 `readout_receptors → action_readout` 的塑性边界：下一片只实现一个可迁移的局部 readout/consolidation 原型，保留现有 readout 作为 compatibility fallback，并用同一 B5 corpus 做 readout-only、association-only、no-change 消融；没有同时满足 old/new retention 前，不再扩大 meta 维度或重跑 foundation。

M1-10 的 replay ablation 已完成并被 Gate 阻断，报告为 `reports/taiji_m1_b10_replay_ablation_canary_20260902.json`。在相同 disjoint B5 corpus、三个 seed、replay scale `0.5` 下比较了 shared/local action decoder 与 all/association/readout 学习目标：shared/all 是最强候选，但 phase-B new recall 仍被压到 `0/0/0.125`；readout-only 只能部分保留旧能力，association-only 不能恢复新读出，local decoder 也没有带来可晋级收益。因此没有任何组合满足 old holdout、retention 和 new holdout 同时不退化，`can_promote=false`。本轮新增的局部 decoder 已纳入 checkpoint、参数预算、架构契约和旧 checkpoint 兼容路径，但保持 shared decoder 为默认，不把失败原型伪装成主路径。

当前唯一下一步为 **M1-11：受保护的 cue-selective readout consolidation 原型**：在不增加基础模型宽度、不开启 full foundation 的前提下，为记忆 cue 建立可检查的局部 action payload 路由，使 replay 只更新与被重激活 cue 绑定的读出，不覆盖其他 cue 的共享 action evidence；保留 shared/local fallback，加入 cue collision、old/new holdout、repeated-replay、no-change 和 checkpoint round-trip Gate。只有三 seed 的旧能力与新能力都不退化，才允许把该路径接入真实 foundation replay；否则继续停留在 M1 诊断，不进入 M2。

M1-11 的 cue-selective 首轮原型已完成，但 Gate 仍阻断。`reports/taiji_m1_b11_cue_selective_ablation_canary_20260902.json` 显示该模式没有改善三 seed 的 old/new 联合保持；单条记忆可读出，多条记忆下仍发生退化。随后完成的 pattern 诊断排除了“训练 pattern 与查询 pattern 不一致”这一主因：write→query cosine 最低仍为 `0.921`，但 cue population 的 pairwise cosine 最高达到 `0.659～0.814`，且 action readout 的有效 fan-in 对单个 cue 最低只有 `4～7/32`。因此当前瓶颈是 cue code 的可分性和局部支持覆盖，而不是继续调 replay 学习率；`cue_selective` 保持实验模式，不成为默认路径。

当前唯一下一步为 **M1-12：cue-binding population 与支持覆盖原型**：在不建立 Python cue→answer 表、不过早扩大 foundation 的前提下，为 cue pattern 增加固定容量、可审计的竞争性 binding population，并让 action payload 的物理支持从该 population 均衡取样；用同一 B5 corpus 验证 cue 间分离度、readout support、old/new holdout、repeated-replay、no-change 和 checkpoint round-trip。若分离度提高但能力仍退化，再转向 replay provenance 的双时间尺度读出；在此之前不重跑 full foundation、不进入 M2。

M1-12 的独立 binding diagnostic 已完成，报告为 `reports/taiji_m1_b12_binding_diagnostic_canary_20260902.json`。固定 sparse projection 加 top-k competition 只使部分 seed 的 cue 平均相似度小幅下降，跨 phase 最高 cosine 仍为 `0.610～0.773`；它没有改变现有 action readout 的物理支持覆盖，也没有通过 old/new 能力 Gate。因此该 binding 方案不接入主模型，不能把随机映射当作 cue 分离的解决方案。

当前唯一下一步为 **M1-13：replay provenance 双时间尺度 protected readout**：保留 waking fast readout 与 replay slow readout 两条可回滚的本地塑性路径，按 cue-local activity 和 provenance 分别更新，在 recall 时以可测的 cue familiarity/支持度进行融合；加入 repeated-replay、no-change、old/new holdout、cue collision、checkpoint round-trip 及参数预算 Gate。若 slow path 仍无法在不损伤 phase-B 的情况下恢复 phase-A，则停止继续增加 memory decoder，转入基础 cue population 的结构性重设计；在 Gate 通过前不重跑 full foundation、不进入 M2。

M1-13 的 dual readout 原型与 provenance 隔离已完成，但 Gate 仍阻断，报告为 `reports/taiji_m1_b13_dual_readout_ablation_canary_20260902.json`。replayed 写入确实不会修改 fast action readout，且 checkpoint/参数预算回归通过；但三 seed 的 old/new 联合能力仍未通过。只改变 slow read gain 的受控检查从 `1` 到 `8` 仍未通过，说明继续调融合增益不能替代 cue 分离和支持覆盖。该 dual 模式保留为实验候选，默认 shared 不变。

当前唯一下一步为 **M1-14：可审计的结构性 cue slot allocation**：停止继续叠加 action decoder，改为让 cue population 在固定容量内通过可验证的 slot 竞争和占用/释放规则形成低碰撞绑定，并把 slot 选择、碰撞率、支持覆盖和可逆性写入 checkpoint/report；只在 B5 的 repeated-replay、no-change、old/new holdout 和 checkpoint round-trip 全部通过后，才允许把 slot route 接到 slow readout。否则保留现有 native memory，不重跑 full foundation、不进入 M2。

M1-14 的 `CueBindingBank` 独立原型和 B5 诊断已完成，报告为 `reports/taiji_m1_b14_slot_binding_canary_20260902.json`。它只保存 cue prototype 与 slot 使用状态，不保存 action/answer 表；三 seed 的 32 个 cue 在 128 个固定容量内占用 32 个 slot，跨 phase collision 为 `0`，checkpoint round-trip 后 query slot mismatch 为 `0`，并通过 release/读不写单测。该结果只准许把 binding 接入实验 slow readout，不准许晋级能力或重跑 foundation。

当前唯一下一步为 **M1-15：slot-routed slow action readout 实验**：将已通过结构 Gate 的 slot code 作为 dual slow readout 的输入，保持 fast action readout 与默认 shared 完全不变；加入 slot support-balanced topology、repeated-replay、no-change、old/new holdout、collision、checkpoint round-trip 和参数预算对照。只有 slow slot route 在三 seed 上恢复旧能力且不损伤新能力，才考虑替换当前 dual cue-local slow path；否则回滚实验 route，保留 CueBindingBank 作为独立结构组件。

M1-15 的 slot-routed slow action readout 已完成 canary，但 Gate 阻断，报告为 `reports/taiji_m1_b15_slot_dual_ablation_canary_20260902.json`。slot binding 没有碰撞，但接入 slow readout 后三 seed old/new 仍未同时保持；只改变 slow read gain 为 `4/8/16` 也未通过。按收敛规则已删除 `slot_dual` 对主 memory 的接入和额外参数膨胀，只保留 `CueBindingBank` 及其结构诊断报告，避免失败实验成为架构残留。

当前唯一下一步为 **M1-16：slot-aware repeated-replay/no-change Gate**：不再添加新的 action decoder，先把 CueBindingBank 接入一个隔离的实验 evaluator，完整测量 repeated replay、no-change、slot release/reallocate、old/new holdout、cue collision、支持覆盖与 checkpoint round-trip；实验结果只写 report，不改变默认 Taiji checkpoint。只有在这些条件与能力保持同时通过后，才允许重新评审是否接入 slow readout；否则冻结 memory 结构，转向基础 cue encoder 的训练数据/可分性改造。

M1-16 的 slot lifecycle Gate 已完成，报告为 `reports/taiji_m1_b16_slot_gate_canary_20260902.json`。三 seed 的 repeated replay same-slot rate 为 `1.0`，no-change 查询全部路由且 state preserved，release 后读为 unbound、重新学习回到释放 slot，checkpoint round-trip query mismatch 为 `0`，cross-phase collision 为 `0`。这些结果证明 CueBindingBank 的状态机和可逆性成立；M1-15 已证明把它直接接进 action readout 仍不能通过 old/new 能力 Gate，因此 slot bank 继续保持独立，不进入默认 checkpoint。

M1-17 的 cue encoder 可分性训练实验已完成并失败。隔离 probe 在不使用 action/outcome 标签的条件下对固定 cue encoder 做了 winner、最近邻跨 cue 排斥和 homeostasis 更新；三 seed canary 均出现同一塌缩：seed 11/29/47 的训练后最大跨 cue cosine 为 `0.999849/0.999748/0.999868`，active support 全为 `128/128`，slot 占用均从 `96` 降为 `1`，`can_promote=false`。这说明当前自选 winner + 局部 anti-Hebbian 规则产生了正反馈，不能作为 cue population 学习规则。probe、评估器、测试和临时报告已删除，默认 Taiji checkpoint、fast/slow action readout 与 cue encoder 均未被改动。

当前唯一下一步为 **M1-18：冻结结构并进入原生训练基线**：保留当前固定 cue encoder、shared action decoder 默认路径与已验证的 checkpoint lineage，把失败的 cue plasticity、slot-routed readout 和新增 decoder 明确列为禁止接入项；先执行全新模型的 checkpoint save→load→eval-only canary，确认训练前保存可逆，再用现有 native joint-training 入口做三 seed 小规模基线，报告 sequence、memory、world、goal、checkpoint digest 和五项 M0 control，不修改架构参数。只有基线报告证明保存/恢复稳定且训练确实改善可读指标，才允许在真实训练数据上扩大课程；若 memory/B5 仍失败，下一片只针对训练信号和数据课程定位，不再增加外围结构。

M1-18 已完成并未晋级 M1。训练前 checkpoint preflight 报告 `reports/taiji_m1_b18_checkpoint_preflight_20260902.json` 为 `passed`，全新进程恢复后的下一步、模型 digest 与 child checkpoint 均一致。默认 shared action decoder、固定 cue encoder、无 replay/slot/probe 的 native joint smoke 已在 seed 11/29/47 完成，训练报告和独立 eval-only 报告逐项一致，`checkpoint_read_only=true`；sequence holdout BPB 分别为 `8.0056→5.5823/5.7423/5.7330`，memory recall 为 `0.625/0.625/1.0`，world error 均下降，goal success 均为 `1.0`。统一 M0 smoke 矩阵 `reports/taiji_m1_b18_m0_matrix_20260902.json` 仍为 `failed`：B1 `7.2191` 仍差于 simple-rule `6.0767`，B2 `0.75` 未形成稳定因果优势，B5 backward transfer 为 `-0.2442`；B3/B4 的 smoke metric 通过但样本量未达到 foundation 下限。因此本片只证明 native 训练和恢复链路可继续，不能宣称五项能力或 M1 晋级。

M1-19 已完成，报告为 `reports/taiji_m1_signal_diagnostics_20260902.json`，没有晋级项。B1 在同一 dataset digest 下比较了 chunk/stream 与 boundary/no-boundary 四种课程；三 seed 的 holdout 均值仅在 `5.60309～5.60472 BPB` 之间变化，最佳 `chunked_without_boundary` 也没有形成足以解释当前失败的收益，说明 byte boundary 不是主因。B5 在同一 disjoint corpus 上比较了 `all/association/readout` 三种 replay target 与 `0.05/0.10/0.25/0.50/1.0` 五档强度，15 组全部 `status=failed`；最接近的是 `readout + 1.0`，三 seed old backward transfer 最低为 `0.125`，但 new-after 三者均为 `0`，是恢复旧记忆同时抹掉新记忆的交换，不是可迁移训练规则。全程 `architecture_unchanged=true`、shared decoder、holdout/retention 只读；M1-19 的临时 smoke 报告已清理。

M1-20 已完成但 Gate 阻断，报告为 `reports/taiji_m1_interleaved_rehearsal_20260902.json`。在固定 disjoint phase-A/phase-B corpus、三 seed、shared decoder、架构不变的条件下，比较 `no_replay`、`posthoc`、`interleave_every_1`、`interleave_every_4`；四种课程的 replay 次数分别为 `0/16/16/16`，所有记录的 checkpoint fresh-process round-trip 均通过，但没有任何 schedule 同时满足 old holdout、old retention 不低于 parent 且 new holdout 不低于 no-replay。每步交错没有稳定恢复 old；posthoc 仅部分 seed 取得 old gain；周期交错也无法跨 seed 同时保护 old/new。此前发现的 replay scale 传递遗漏已修正，最终报告使用配置中的 `replay_memory_learning_scale=0.25`；因此该结果可作为有效诊断。结论是 replay 时序本身不是充分条件，不能接入 foundation，也不能进入 M2；未改变默认 Taiji checkpoint、shared decoder 或 slot/cue plasticity 实验边界。

M1-21 已完成但 Gate 阻断，报告为 `reports/taiji_m1_replay_admission_20260902.json`。评估已修正为每个 phase-B episode 后立即决定下一条 phase-A replay，避免把 phase-B 结束后的结果误当成 admission 效果；三 seed 比较 `no_gate`、`familiarity_gate`、`conflict_reject_gate`，三种策略分别实际改变了 replay admitted/rejected（熟悉度门控为 `9/13/16`、`13/3/16`、`16/0/16`，冲突门控为 `15/14/12`，对应 seed 11/29/47），所有 checkpoint fresh-process round-trip 与 read-only persistent state 均通过，但没有任何策略晋级。更重要的是，三 seed 的 phase-B-only `no_replay` control 的 old holdout 已为 `0/0/0`；replay gate 无法在此之后恢复 old，new holdout 仍可保持 `0.625～0.875`。因此当前主损伤来自 phase-B 新 episode 对 shared memory 的直接写入，而不是 replay admission；未改变默认 Taiji checkpoint、decoder、cue encoder 或 slot bank。

M1-22 已完成但 Gate 阻断，报告为 `reports/taiji_m1_phase_b_write_ablation_20260902.json`。在关闭 replay、固定 shared decoder 和三 seed 的条件下，加入可复现默认 baseline `1.0` 后比较 `all`、`association`、`readout` 与 `0.05/0.10/0.25/0.50/1.0` 局部写入强度；所有 checkpoint fresh-process round-trip 与 read-only persistent state 均通过，但没有候选同时保住 phase-A old holdout/retention 并让 phase-B new holdout 严格超过 no-write。`all@1.0` 能产生 new recall `0.5/0.625/0.875`，但 old holdout 为 `0/0/0`；`readout@1.0` 只得到 new `0.375/0.625/0.625`，old 仍为 `0.5/0/0.375`；低强度与 association 倾向保住 old，却没有 new gain。结论是只调整 learning target 或标量强度不足以解决 old/new 冲突，未改变默认 checkpoint 或主路径。

M1-23 已完成但 Gate 阻断，报告为 `reports/taiji_m1_support_mask_20260902.json`。在关闭 replay、固定 shared decoder、三 seed 和 `mask_fraction=0.5` 的条件下，cue-conditioned support mask 确实把 action readout 的有效 fan-in 大约减半，并把跨 phase 逐 cue edge Jaccard 从 shared 的 `0.7895/0.8343/0.8062` 降到 `0.3911/0.3876/0.4230`（seed 11/29/47）；但 child old/new 仍未同时保留，shared 与 cue-mask 均无晋级策略，所有 checkpoint fresh-process round-trip 和 read-only persistent state 均通过。union edge ratio 因固定 fan-in union 饱和为 `1.0`，已补充 pairwise Jaccard 作为实际隔离指标。结论是减少 support overlap 本身不是能力收益，未改变默认 checkpoint、decoder、cue encoder 或 slot bank。

M1-24 已完成但 Gate 阻断，报告为 `reports/taiji_m1_support_alignment_20260902.json`。在关闭 replay、固定 shared decoder、`mask_fraction=0.5` 和三 seed 条件下比较 `shared`、`write_mask_only`、`read_mask_only`、`aligned_mask`；四条路径均能独立改变对应的有效 fan-in 和 action evidence，write/read pairwise Jaccard 也按预期分别下降，但没有任何路径同时保住 old holdout、old retention 并让 new holdout 严格超过 no-write。`write_mask_only` 仍出现跨 seed old 崩溃，`read_mask_only` 仅部分减轻读取注入，`aligned_mask` 没有额外能力收益；所有 checkpoint fresh-process round-trip 与 read-only persistent state 均通过。结论是支持 mask 的写入/读取错位不是根因，继续微调 mask 或 replay 没有依据，默认 checkpoint 与 shared decoder 保持不变。

当前唯一下一步为 **M1-25：显式 cue representation/capacity redesign 的最小结构原型**：基于 M1-23/24 的 support 与 cosine 证据，冻结现有默认 checkpoint 和生产路径，在隔离 evaluator 中只引入一个可审计的 cue-conditioned identity route，将同一 cue 的 action evidence 写入固定容量的局部 identity support，并保留 shared path 作为 compatibility control；不使用 Python cue→answer 表、不接收 holdout 反馈、不做 full foundation。原型必须报告 identity collision、per-cue support、old/new holdout、repeated replay、no-change、parameter/edge budget、checkpoint digest 与 fresh-process round-trip；只有三 seed 同时通过能力与结构 Gate，才允许讨论接入 native memory，否则删除原型并保留现有 checkpoint。

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
