# Taiji 当前研究审视与路线修订依据

> 日期：2026-09-06。审计代码基线：main / `4c45d56`；另检查了当时未提交的 M2-2af 草案和路线更新。
>
> 本文是事实与决策依据，不提供第二套执行顺序。唯一计划见 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md)。本轮不实施模型修复、不启动新训练、不改写历史实验报告。

## 1. 结论

项目已经有真实学习成果：中文 byte 预测优于 unigram、带干扰的关联记忆、简单世界转移与动作奖励学习，以及可恢复的 checkpoint 和增量 memory 容量。但近几轮工作大量集中在边界、registry、报告接线；模型表征、语义与未见组合的训练证据仍薄弱。

核心问题不是“再训一次还是永远不训”，而是明确每一次续训的输入、被训练参数和能力出口。自然成长也需要更新既有参数、加入新容量并巩固旧知识；保存两份 readout 和人工选择分支只是其中一种隔离方法。

本次重审发现足以改变下一步顺序的代码证据。active 零收益不能直接触发容量改造；先校正评分，随后用真实续训和强对照判断瓶颈。原有原生认知目标保持，成熟学习算法仍应作为候选，以模型所有权与能力证据判断。

## 2. 项目覆盖与证据范围

| 平面 | 本次检查的入口 | 已知事实与边界 |
|---|---|---|
| 根需求/架构 | active 的核心需求、Native v1、Seed architecture、roadmap | 目标包含持续状态、异质协作、身体、成长；不是只做字节预测器 |
| 原生训练 | `taiji/model.py`、`organs.py`、`foundation_training.py`、joint runner | 训练主要作用于 kernel 与独立 world learner；不能推定 adapter 全部器官已训练 |
| 原生认知层 | `taiji/adapter.py`、perception/world/executive 等模块入口 | TSKV8Adapter 承载更多器官；训练覆盖与产品加载仍需逐 owner 对齐 |
| 评估/数据 | foundation evaluator/tasks、B5 stream、M2ad、M2af 草案 | 已确认错 owner、holdout 重复与 Gate 目标混用 |
| 工程与客户端 | CI workflow、pyproject、实现参考和 Seed 架构合同 | 有现成 Workbench、权限、provider/客户端底座；本轮未做客户端现场和全仓 CI |
| 实验报告 | M2j/v/u/w/x/y/ad 等 JSON、两个实际 checkpoint | 交叉核对状态、指标与谱系；并非重新跑完全部历史训练 |

本轮实做：无训练的 owner 调用追踪、B5 重复计算、checkpoint 只读加载与参数统计。历史报告里已有的测试数不当作本轮新执行的测试结果。未核查远端最新 CI 状态，不宣称 origin/main 与本地同步。

中断实验在用户切换到重审后仍有 PID 32132 存活；已确认命令路径后停止，未发现目标 M2af 报告。该次中断不产生能力结论。未改写模型 checkpoint。

## 3. 关键发现

### F01：active 评分实际使用 protected，能力比较失效

代码链：

- `scripts/training/eval_taiji_m2ad_active_continuation.py::_score_read_only` 调用 `model.score_bytes(data)`，未传 scope。
- `taiji/model.py::score_bytes` 调用 `observe(..., readout="predictive")`，没有传 `_predictive_readout`。
- `observe` 在未覆盖时选择 `self.predictive_readout`，即 protected。
- `generate` 和带边界的 `learn_bytes` 则显式解析 active readout。
- M2af 草案直接复用了同一评分 helper，因此“改用 phase-C”不会修复量尺。

只读诊断在临时内存模型中，把 active bias 设为明显偏向 A，追踪 `BytePredictiveReadout.probabilities` 的对象身份：

| 调用 | protected 次数 | active 次数 | 附加观察 |
|---|---:|---:|---|
| `score_bytes(b"AAAA")` | 6 | 0 | checkpoint 前后相同 |
| `generate(b"A", 4, boundary=active, ...)` | 0 | 6 | 生成 AAAA，owner 为 predictive_readout.active |

没有调用训练，未写 checkpoint。复现可用 `unittest.mock.patch.object` 包装该方法，按 `self is model.predictive_readout` 与 `self is model._active_predictive_readout` 计数。

**结论：** M2ad 报告的 0.0 gain 和 0.0 retention delta 是两个 protected 分支的比较，不能证明 active 没学会、已饱和或训练数据重复导致无益。“父代已见 B，所以无增益”的原解释不成立。registry/隔离/保存检查仍可保留其各自意义，active 能力部分必须标为无效并重新测量。原始 JSON 保留，纠正通过新版本报告体现。

### F02：旧 B5 的新任务 holdout 是训练流的完整子串

`eval_taiji_foundation_baseline.py::_b5_phase_b_stream` 使用：

```python
bytes(32 + ((index * 37 + seed + offset) % 224) for index in range(length))
```

train 为 length=4096 / offset=0，holdout 为 length=200 / offset=1。37 与 224 互素，该序列周期为 224；偏移量 1 只是在同一个周期上平移。

本轮对 seed 11、29、47 都得到：

```text
train.find(holdout) == 109
train[:-224] == train[224:]  # True
```

**结论：** 这个课程可以测对特定周期流续训时的遗忘压力，不能作为新序列泛化证据。由它得到的旧任务遗忘现象并不会自动消失，但“新任务表现”与整体持续学习声明需要降级。不能仅改变 offset、ID 或 split 名称来修正。

### F03：统一 Gate 把已有能力、保持和本轮增量混成一个条件

`_evaluate_loaded_b4` 要求：

```python
worst_native > max(baseline_metrics.values())
```

baseline 包含直接 frozen parent。当 parent 和 child 均为 1.0，该 Gate 必然失败。M2v 正式报告中 B1≈4.384、B3≈3.62e-8、B4=1.0 因持平/略差而失败；它记录的是“相对这轮 parent 没有新增收益”，不等于原能力不存在。

当前 retention 的部分判断还把同一模型的 retention 分区与 holdout 分区比较，而不是同一旧任务上的 parent/child 配对，可能混入分区难度差异。应逐任务追溯，而非一次性把所有 failed 改成 passed。

**修订原则：**

- 绝对能力：固定问题定义、基线/认证祖先与明确阈值；
- 旧能力保持：同一 old split 上的直接父代与 child 配对；
- 本轮增量：只要求预注册的新目标获得有效改善；
- 整体候选：量尺合格 + 必要能力达标 + 旧能力保持 + 至少一个本轮目标改善。

历史 manifest/report 不原地放宽。新语义以新版本并列报告，保留 B5 真正负结果。

### F04：任务量扩大并不等于任务多样性扩大

`train_taiji_world_action.py` 中：

- world corpus 改变 position，但动作始终是 push amount=1，目标始终 position+1；
- goal corpus 使用 `cue=65+index%2` 对应两种 action，数量增至 1000 仍只是反复两个条件；
- `build_world_learner` 用 train+holdout+retention 构造 schema。需要区分公开任务 schema 与从测试提取的标签/统计量，不能默认其独立。

B2 的 `DelayedMemoryCorpus` 则明确要求查询指向已写入的 key；这是合理的记忆测量设计，不应机械判为泄漏。但通过已存关联检索不能宣称形成了语义抽象。identity 槽内最近邻 value 也是记忆实现，不等同于整个网络学会组合推理。

**结论：** 保留这些窄课程作机制和回归；泛化课程另加入未见规则、目标/对象组合、动作语义变化及更长依赖。数量、独立性与难度分别记录。

### F05：task-aware 分支管理尚未解决 task-free 学习与路由

M2x 的 centroid 路由 active 使用率为 0；M2y 的 novelty 路由使新任务 active 使用率约 62.7%，但 new BPB 约 6.463，仍差于 active-only 的约 3.426。

M2z～2ac 改用显式 Workbench task boundary：它适合实际任务生命周期和权限，但由任务元数据指定 generation scope。两个问题不能混为一谈：

- 已知任务身份时如何可靠选择某个已训练分支；
- 无 task label 时模型如何发现变化、利用新旧知识并决定更新。

进一步，registry 记录了 boundary digest，但 `_predictive_readout_for_scope` 只接收 scope；传入 token 的合法性和它是否匹配已挂载分支是不同检查。R0 应加入 foreign-boundary 测试验证这一合同，不能仅凭元数据存在认定闭合。

**结论：** 保留 task-aware 机制为受控基线与产品边界。自主路由、共享知识、分支整合和有预算的容量增长放回明确研究目标，不能用任务 ID 的切换作为替代证明。

### F06：下一轮脚本没有满足长跑训练的基本条件

M2af 未提交草案的具体问题：

- 继承 F01 的错误评分；
- 在 `learn_bytes` 之后才首次保存，未执行训练前 persistence preflight；
- fresh process 仅检查 digest，没有恢复后能力复测；
- 唯一 active 模型文件作为 probe 被删除，没有保留最终训练产物和续训游标；
- 单 seed 的任意正 gain 加容差就可令 `can_promote=true`，与多 seed Gate 不一致；
- loader 固定要求 sequence-only child，无法直接使用已训练 memory-growth 后代。

**处理：** 草案转存为不可执行的 [.py.txt 归档](../archive/history/research_review_20260906/M2AF_DRAFT_NOT_VALIDATED.py.txt)，主线重用稳定 trainer/measurement 设施重构。此举只收束上一轮由助手创建的草案，不删除历史模型或报告。

### F07：当前实验可能从更老的分支继续，不能代表全项目最新模型

本轮只读加载了两份实际 checkpoint：

| 指标 | seed11 sequence child | seed11 identity-growth child |
|---|---:|---:|
| training_phases | sequence | memory |
| 模型格式 | taiji-native-v10 | taiji-native-v10 |
| identity capacity | 128 | 512 |
| parameter_tensors 标量 | 193,586 | 476,978 |
| 参数张量字节 | 774,344 | 1,907,912 |
| joint 文件字节 | 18,771,087 | 42,908,047 |
| outer digest | 9533f476…2decef | 9524f35a…a48210 |

来源分别为 `output/taiji-m2-f5-seed11-private-context-20260905/last.pt` 与 `output/taiji-m2s-seed11-identity-generation-20260905/last.pt`。

这些计数含接口报告的模型参数，不涵盖所有 checkpoint 张量/记忆/结构索引、adapter 认知器官和单独 world learner；joint 文件还包含谱系等内容。不能混同参数存储与总内存/磁盘，也不能将 128-slot 分支实验视为 512-slot 后代已通过。

**结论：** 每轮先明确直接父代与认证祖先，沿正确谱系继承。不同研究臂不能未经迁移和复测就手工合并 checkpoint。

### F08：原生身份约束与成熟训练技术之间仍有未解决张力

Native v1 明确允许成熟 embedding、attention、optimizer、递归/状态空间组件；但 2026-08-26 的实现收口又把 `taiji/` 的 autograd 全部移除，`verify_taiji_native_v7.py` 的 AST 合同禁止 `backward`、`topk`、`MultiheadAttention` 等属性。

这说明“站在巨人肩膀上”尚未完整落实到发展训练的选择空间。手写 LocalAdam/梯度可能与成熟实现数学等价；换一种写法不会自动得到更优信用分配、效率或上限。

**本轮决策：** 将禁止算子的实现约束重新列为需要证据检验的设计选择，允许在隔离对照中比较成熟训练方式。当前代码/测试不修改；若 R2 证据支持迁移，先明确 checkpoint、运行时依赖、发展/在线状态、等价性与安全边界，再修改正式合同。不得把“移除 autograd”当作智能性的成绩。

## 4. 研究主张的保留与限制

| 主张 | 当前可说 | 还缺什么 |
|---|---|---|
| 神经元网络 | 有持续动力学、稀疏连接和局部学习的原生实现 | 表征能力、跨群体协作及任务泛化的独立归因 |
| 在原有上成长 | 已有续训、identity 增量容量与 lineage | 多轮真实能力净收益、容量/成本控制、压缩与自主选择 |
| 不依赖 Transformer 大脑 | Taiji 核心有独立运行路径 | 所有产品/语义器官的训练覆盖和 native-only 对照 |
| 语言学习 | 有真实中文 byte 统计收益 | 超过强字符对照的长程/语义能力与稳定可读输出 |
| 硬件优势 | 参数和状态可以准确统计 | 等质量、等数据/训练时间的吞吐与内存比较 |
| Skill/MCP 进化 | 数据来源与客户端能力合同已有底座 | 模型内化收益、去来源后保持和真实执行结果闭环 |

## 5. 技术参考与采纳边界

以下是研究候选依据，不是“这些论文已经替 Taiji 证明成功”，也不构成并行实现清单：

- [Net2Net](https://arxiv.org/abs/1511.05641) 研究通过保持函数的变换将既有网络知识迁移到更宽/更深网络。可参考“先保留旧函数，再训练新增容量”的思想；能否用于当前稀疏/状态/记忆结构需单独证明。
- [Progressive Neural Networks](https://arxiv.org/abs/1606.04671) 利用既有网络与横向连接进行任务迁移，提供保留旧知识和增加能力的参考。Taiji 仍需测分支成本、任务身份假设和最终知识整合，不能直接宣布已实现开放成长。
- [EWC](https://arxiv.org/abs/1612.00796) 通过限制对旧任务重要权重的更新减少遗忘，可作为有预算的稳定性对照；不能承诺无容量上限或彻底消除遗忘。
- [Mamba](https://arxiv.org/abs/2312.00752) 提供输入相关的选择性状态空间序列建模思路。可作为 Taiji 时间表征候选来源；论文的硬件和性能结果不能外推到本机 CPU 或当前实现。
- [MiniMind 官方仓库](https://github.com/jingyaogong/minimind) 提供预训练、SFT、保存恢复与独立推理的训练工程参考。复用阶段化训练纪律，不照搬其参数规模、GPU 耗时和认知主体。

推荐的研究方向是可持续训练的 Taiji 表征与时间核心、受控快速可塑状态、可巩固记忆及必要时继承式增长。是否选择具体 GRU/SSM/局部规则由相同任务上的实测决定。完整阶段/验收/失败出口只在统一计划中维护。

## 6. 本轮交付边界

已完成研究复核、量尺问题最小复现、路线修订、旧日志归档和入口同步。没有修复 score_bytes、改变 Gate 实现、修改训练规则或运行新课程。下一次实现按统一计划的 M2.R0 开始。

原始报告继续保存，因此自动工具若直接读取旧 `status=passed` 仍需由后续 R0 的报告版本/失效登记修正；本轮文档纠正不能被当作已完成程序修复。
