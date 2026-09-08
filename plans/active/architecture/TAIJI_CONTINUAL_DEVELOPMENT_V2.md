# Taiji 继承式成长架构 v2

> 状态：当前 M4 架构设计，2026-09-09 生效。
>
> 本文只定义“如何在原有能力上继续生长”的架构与验收语义；唯一执行顺序仍由 [03_CURRENT_EXECUTION.md](../roadmap/03_CURRENT_EXECUTION.md) 决定。M4.R0～R12 的历史报告不改写，修订后的证据解释见 [M4 固定容量证据链复盘](../../reference/M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md)。

## 1. 决策

M4 v1 关闭的是一组 **TSK-v8 F1 next-byte continuation 实验**，没有关闭 Taiji 的继承式成长方向。M4 v2 不再把“连续覆盖同一读出层 + 每阶段 BPB 不得有任何正向波动”当作开放式成长，也不把另挂一个未进入主路径的结构网络当作已经生长。

新的唯一设计是：

1. 现有 checkpoint 成为慢知识的父代，不重新初始化；
2. 在线经历先写入可回滚的快速可塑状态和情景记忆；
3. 睡眠/巩固用有来源的重放把可迁移部分写入慢参数；
4. 当“固定容量已经无法同时吸收新能力并保持旧能力”有直接证据时，才创建零影响的 shadow 结构候选；
5. 候选经训练、路由、损伤和保持 Gate 后才并入主路径；失败回到父代；
6. 整个过程不向模型提供人工任务 ID、领域名或答案路由表。

这是一套连续发展系统，不是每任务一个冻结 head，也不是每轮从头训练一份模型。

## 2. M4 v1 暴露的设计错误

| 设计点 | 仓库中的实际做法 | 为什么不能回答 A8 成长问题 | v2 修正 |
|---|---|---|---|
| 学习对象 | 主要评估 `BytePredictiveContext` / `BytePredictiveReadout` 的 next-byte BPB | F1 是文本低层候选 kernel，不能代表概念、世界模型、技能和行动能力 | 同时使用低层预测、结构化技能和真实结果三类量尺；能力归属到具体 owner |
| 连续成长 | 在固定张量上连续局部更新，或克隆一个 active readout | 覆盖权重和人工分支都不是结构生长 | 引入快/慢可塑状态、真实 replay、shadow 增长和 learned routing |
| 扩容对照 | R2 新建零读出槽并把旧/新概率固定 50/50 混合 | 固定混合会稀释旧函数；没有表征扩宽、路由学习或候选成熟期，因此负结果不能否定容量压力 | 候选初始输出门为 0，保证函数不变；只学习 residual 与 router，验证后逐步开放 |
| 主路径 | `AdaptiveNeuronNetwork` 由显式 `step_cross_region_network()` 单独调用 | checkpoint 中“存在结构网络”不等于普通观察、预测或行动会消费它 | 结构区域必须进入同一认知 forward/credit 路径；旁路仅作为 shadow 对照 |
| owner 一致性 | R7 更新 protected context+readout；R10/R12 在 Workbench boundary 下只更新 active readout、冻结 context | 跨轮比较混入了 owner、forward 和分支语义变化 | 同一实验族固定 owner 图；改变 owner 必须单列因果实验，不能借用旧对照 |
| 巩固 | 当前 preservation 只让 active readout 在当前输入上接近 protected 概率 | 这是静态 teacher 约束，不是旧经历重放、重要性估计或快慢权重整合 | 旧经历由 episodic ledger 提供；慢写入由 replay、importance 和验证共同决定 |
| 课程 | C/C2/C3 是同一语料的随机 record-disjoint 分区；R12 则一次性完全替换语料 | 前者主要测同分布续训，后者把数据密度、文体和领域突变混在一起；两者之间缺少自然过渡 | 分为同分布、渐进混合、技能组合三条轴；不再用“全量替换”代表自然成长 |
| Gate | 任一 seed 的任一旧 C BPB `> 0` 即失败；新能力主要只看 C3；A 保持未进入 formal Gate | frozen 对照天然零退化；没有非劣界、置信区间和完整累计能力矩阵；会把 `+0.002 BPB` 与灾难性遗忘同等处理 | 预先校准非劣界，报告均值/最坏域/置信区间；所有历史能力都进入累计矩阵 |
| 指标语义 | R10 的 `active_a_retention_bpb` 是绝对值，却由通用聚合器附加 `degradation_seed_count`；R12 的 `+1.62` 是相对另一训练臂的差值 | 绝对 BPB 不能用 `>0` 判断退化；臂间差值不能直接称为相对父代遗忘 | 每个指标声明方向、基线、单位和 delta；缺父代基线时禁止使用“遗忘”措辞 |
| 自主性 | active branch 由外部 Workbench boundary 指定 | 证明了权限隔离，不证明模型能识别上下文并自主选择知识 | boundary 只控制副作用授权；认知路由必须从内容、状态、不确定性和资源学习 |

## 3. 哪些证据仍然有效

M4 v1 不是无效劳动，以下基础可直接继承：

- record-level 排除、数据 digest、source lineage 和报告互链可信；
- 原子 checkpoint、fresh-process restore、只读 scorer 和 owner 差分已经闭合；
- 更新尺度显著改变新数据吸收与旧分布表现，说明稳定—可塑冲突真实存在；
- 固定 preservation 项在 R10 的 readout-only 路径上确实减小参数位移，但没有通过原 Gate；
- abrupt UltraData replacement 相比同配置 simple_zh continuation 显著恶化 A 分布与周期保持，说明突然换分布是高风险干预；
- 现有结构 proposal、预算、lineage、checkpoint、lesion、rollback 资产适合作为 v2 的安全外壳。

以下旧结论降级：

- “更新规则轴已穷尽”降为“已测试的标量 scale 与静态 readout preservation 候选已结束”；
- “容量压力不成立”降为“零读出槽 + 固定平均没有收益”；
- “分布切换主导遗忘”降为“abrupt corpus replacement 是当前实验中的最大臂间差异”；
- “cycle3 是架构固有难度”撤销；当前只有一个课程顺序、三模型 seed 和 byte 边际统计，不能形成固有性结论；
- “M4 连续成长已完成”撤销；只完成了 v1 量尺与若干反例。

## 4. v2 核心状态

### 4.1 发展型突触

每个进入 M4 主路径的可塑连接至少拥有下列可保存状态：

| 状态 | 含义 | 更新时机 |
|---|---|---|
| `slow_weight` | 已巩固、跨经历稳定的知识 | 发展训练或通过 Gate 的睡眠巩固 |
| `fast_delta` | 当前环境中的快速适应残差 | 在线 wake 学习；可衰减、清零或回滚 |
| `eligibility` | 哪些连接对近期预测/行动有因果资格 | 每个相关事件更新，跨 tick 保存 |
| `importance` | 改动该连接对已掌握能力的风险 | replay/保持探针估计，不由参数绝对值代替 |
| `usage` / `age` | 路由使用、成熟度和长期闲置 | 运行时与睡眠维护 |
| `plasticity` | 当前允许的写入强度 | 由误差、新颖性、稳态和资源共同调制 |

有效连接为 `slow_weight + gate * fast_delta`。旧 checkpoint 迁移时令 `slow_weight=旧权重`、`fast_delta=0`，第一步输出必须逐位等价；因此改架构不等于重新训练。

### 4.2 发展型区域

一个可成长区域由稳定 trunk、若干 residual population、learned router 和局部状态组成。新增单元/区域遵循：

1. **提议**：持续 residual error、快速状态饱和、重要性冲突和 replay 干扰共同形成容量压力；单纯参数占用率不够；
2. **零影响出生**：新候选拥有独立身份与 checkpoint lineage，但输出门为 0，父代函数不变；
3. **shadow 成熟**：候选只学习父代残差，不参与真实行动选择；
4. **路由验证**：内容/状态/不确定性路由必须优于随机、固定和平均路由；不得读取任务 ID；
5. **准入**：候选带来未见样本/任务收益，旧能力满足非劣，lesion 会失去对应增益，资源预算可接受；
6. **合并或回滚**：无独立贡献则合并/剪枝；任何失败恢复父 checkpoint、拓扑和发展状态。

### 4.3 记忆与巩固

受控 replay 不是把旧字节前缀再次拼进训练流。v2 replay 单元是带 `episode/tick/source/observation/action/outcome/owner` 的真实经历：

- wake：新经历先写 `fast_delta` 与 episodic memory；
- sleep：按信息增益、旧能力覆盖和来源许可采样真实记录；
- consolidation：在 replay 与当前经验的联合证据上把可迁移结构写入 `slow_weight`；
- semantic/procedural extraction：只有新组合或新情境能复用时才称为语义/技能；
- rollback：若巩固后累计能力下降，恢复慢权重，保留失败证据而不污染父代。

protected checkpoint 继续作为实验对照和灾难恢复点，但不再冒充生物式慢知识机制。

### 4.4 自主发展控制

`DevelopmentState` 的决策输入限定为模型可获得的信号：预测残差、校准不确定性、能力缺口、router 负载、fast/slow 冲突、replay 收益、资源和真实 outcome。外部脚本可以执行安全审批与实验调度，不能硬编码“这是代码任务，所以创建 code expert”。

第一阶段由确定性安全控制器提出候选；只有在候选选择本身通过校准 Gate 后，才把干预选择逐步交给 Taiji。这避免把自动脚本误称为自进化。

## 5. 站在成熟技术上的训练边界

Taiji 原生性由认知所有权决定，不由“是否调用 autograd”决定：

- **发展训练**：允许 PyTorch autograd、AdamW、成熟 embedding/SSM/attention/MoE、批处理和未来 CUDA，用于形成 Taiji-owned 参数；
- **终身学习**：以快速低秩/稀疏局部状态、eligibility、情景写入和受控 replay 为主，不能要求运行时每次全量反向传播；
- **睡眠巩固**：可以使用小批量优化，但输入、目标、重要性、写入 owner 和准入决定必须属于 Taiji 的可审计状态；
- **外部 teacher**：只允许标注为 `native-assisted` 的发展阶段辅助；运行时移除后 Taiji 的世界、目标和行动能力仍须成立。

当前全仓 AST 的 autograd 禁令在 v2 实现前保持不变；是否放开必须由隔离分支的同量尺对照决定，不能静默绕过，也不能把手写等价梯度当作更高上限的理由。

## 6. 新课程与量尺

### 6.1 三条课程轴

| 轴 | 要回答的问题 | 课程 |
|---|---|---|
| S：同分布续训 | 能否继续吸收同一环境的新记录 | 同源 record-disjoint，多 course seed 和顺序 |
| G：渐进分布 | 能否像自然经历一样适应环境变化 | 旧/新来源按 `80/20 → 60/40 → 40/60 → 20/80` 过渡；另保留 abrupt replacement 作为压力上界 |
| K：技能组合 | 是否真的增加能力而非只改字节统计 | 结构化语义 → 世界转移 → Workbench 只读意图 → 隔离执行；每阶段都有未见组合和真实 outcome |

S/G 负责低层稳定—可塑性，K 才是 A8 的主要能力证据。任何候选必须先过便宜 S smoke，再过 G，最后才进入 K；不能用 S 的 BPB 代替 K。

### 6.2 累计能力矩阵

每个阶段结束后都评分所有既有能力，记录：

- 相对同一父 checkpoint 的新能力增益；
- 每个旧域/旧技能的绝对分数和 delta；
- average forgetting、worst-domain forgetting、backward/forward transfer；
- fast/slow owner 写入、router 使用、候选 lesion、checkpoint 恢复；
- 参数、内存、训练字节、延迟和能耗代理成本。

BPB 只能在同一数据域内比较。跨语料必须分别对各自 frozen-parent baseline 归一化，不能把一个语料上的高 gain 解释为学习成功。

### 6.3 非劣 Gate

旧 Gate 的精确 `delta <= 0` 保留为“完美保持”观测项，不再是唯一生死线。正式 Gate 在实验前从 frozen parent 的独立 holdout block/course-seed 波动中校准 `epsilon`，并同时要求：

1. 技术、owner、source、checkpoint/fresh restore、只读和副作用 Gate 全通过；
2. 新能力相对父代及 fixed-capacity baseline 的置信下界为正；
3. 旧能力平均值满足非劣，且任一关键域不得超过灾难性遗忘上限；
4. 累计效用高于 frozen、固定容量、random growth 和等参数预分配对照；
5. growth lesion 移除新增收益，router lesion 显著恶化组合任务；
6. 不提供任务 ID 的路由仍通过；
7. 收益/参数、收益/时间和峰值内存没有越过预注册预算。

`epsilon` 必须来自量尺稳定性并设置上限，不能在看见候选结果后调大。精确零退化、平均非劣和灾难上限三项全部报告，避免用均值掩盖单域崩溃。

## 7. 实现边界：复用、改造、停止使用

### 7.1 直接复用

- `SparseSynapses` 的稀疏存储与局部更新；
- `LocalAdam` 和现有 owner/lesion 工具作为对照实现；
- `AdaptiveNeuronRegion` / `AdaptiveNeuronNetwork` 的稳定身份、拓扑迁移和 checkpoint；
- structural evidence、proposal、budget、validation、lineage、rollback 资产；
- episodic/semantic/procedural memory 和 `DevelopmentState` 合同；
- M4 v1 的数据 digest、fresh restore、只读 scorer 与资源遥测。

### 7.2 必须改造

- 给可塑连接增加 fast/slow/importance/eligibility 状态和向后兼容迁移；
- 把 adaptive network 接到实际 observation→workspace/prediction→credit 主路径；
- 将 occupancy-only capacity pressure 改成 residual/conflict/utility pressure；
- 增加零输出 shadow gate、learned router 和 matched-capacity baselines；
- 用经历 ledger 驱动 replay，取消当前输入上的静态 logits preservation 作为主巩固方案；
- 将量尺升级为多轴课程与累计能力矩阵。

### 7.3 停止作为晋级依据

- active/protected readout branch 数量；
- fixed 50/50 输出平均；
- 单一 byte marginal JS；
- 单 course order、单一 partition chain；
- 绝对 BPB 上的 `> 0 = degradation`；
- checkpoint 变大、参数发生变化、proposal 被创建等技术事实。

这些资产可保留做兼容、权限隔离和反事实，不再决定 A8 是否完成。

## 8. v2 里程碑与停止线

具体先后以执行计划为准，设计依赖固定如下：

1. **M4.V2.R0 量尺合同**：实现课程 manifest、累计 scorecard、指标方向/基线 schema、非劣界校准和旧报告兼容审计；不训练；
2. **M4.V2.R1 零变化迁移**：现有 F1 checkpoint → fast/slow 突触状态，`fast=0`，输出、score、owner 和 fresh restore 精确一致；
3. **M4.V2.R2 快适应—慢巩固**：先在 S/G 课程比较 slow-only、fast-only、fast+真实 replay+consolidation；失败则不扩容；
4. **M4.V2.R3 主路径结构桥**：一个 zero-gated adaptive residual region 进入真实 forward/credit，未开门时逐位等价；
5. **M4.V2.R4 shadow 生长**：用直接容量压力提议、训练、验证、lesion、准入/回滚；与等参数预分配和 random growth 对照；
6. **M4.V2.R5 自主路由**：移除 evaluator task ID，证明 learned router 在 G/K 课程形成专门化与组合增益；
7. **M4.V2.R6 A8 formal**：三模型 seed × 至少三 course seed/order，完成 S/G/K 累计能力、资源和 checkpoint 矩阵。

停止线：

- R0 未能消除指标语义歧义，不准训练；
- R1 保存或等价迁移失败，不准写任何新权重；
- R2 没有稳定—可塑收益，不准用扩容掩盖学习规则失败；
- R3 未进入实际主路径或候选关闭时不等价，不准称结构成长；
- R4 不优于等参数预分配/random growth，不准进入自主路由；
- 任一阶段出现来源泄漏、旧能力灾难性退化或不可回滚，恢复父代并归档候选。

## 9. 当前唯一设计出口

M4.V2.R0 已完成量尺合同与语义审计；R1 也已完成 checkpoint-compatible fast/slow 零变化迁移，并在真实 v10 joint-training checkpoint 上通过源不变、outer/inner digest、fresh restore 和 `fast=0` smoke，仍保持 `can_promote=false`。当前唯一下一步是 R2 的 S/G 快适应—慢巩固 canary：允许写 developmental state，但不允许结构扩容或跳过 slow-only/fast-only/真实 replay 对照。不得跳到 R3 结构桥、M5 外围或客户端美化。
