# Seed / Taiji 唯一执行计划

> 修订：2026-09-09。
>
> 当前主线：**M4.V2 继承式成长重构**。M4.R0～R12 已作为 F1 fixed-capacity 实验证据归档，但其对“容量、分布和全部学习规则”的过度外推已撤销。当前架构依据为 [Taiji 继承式成长架构 v2](../architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md)，历史证据边界见 [M4 固定容量证据链复盘](../../reference/M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md)。
>
> 本文是唯一执行顺序和唯一“下一步”来源。其他 active 文档定义长期需求或架构，不发布并行任务；archive/reference 中的旧“下一步”全部失效。

## 1. 当前事实基线

### 1.1 项目定位

Taiji 是唯一认知主体，Seed 是产品/runtime、Workbench、设备、权限和发布载体。Taiji 可以复用 embedding、SSM、attention、MoE、autograd、optimizer、CUDA 和成熟语言器官，但持续世界状态、记忆、目标、路由、发展选择与行动决定必须由 Taiji 拥有。

目前是有真实小规模训练和多个窄能力 Gate 的研究原型，不是成熟语言大模型，也不是已经完成自主进化的系统。

### 1.2 已完成且仍有效

| 范围 | 有效成果 | 不得外推为 |
|---|---|---|
| M0/M1 | checkpoint、局部学习、字节预测、延迟记忆、简单世界转移与行动基线 | 通用语言、推理或完整神经认知架构 |
| M2 测量与训练 | record-disjoint 数据、owner 路由、read-only score、fresh restore；结构化事实/Goal/ContentPlan、多步状态转移、双 owner 组合与保持 | 开放域语义、共享表征联合优化或自然语言能力 |
| M3 Workbench | project-disjoint observation、task-state sequence、只读 ActionIntent、preview/approval、隔离执行与 undo | 未经授权的真实客户端自治写入 |
| M4 v1 技术链 | 三模型 seed 的数据 lineage、原子保存、fresh restore、owner 差分、资源测量和可重复 F1 cascade | CR-4/A8 结构成长已完成或已被否决 |
| 结构资产 | `AdaptiveNeuronRegion/Network`、proposal、budget、validation、lesion、rollback、lineage | 已自动进入普通 observation/decision 主路径 |
| 产品外围 | provider artifact/watchdog、Skill/MCP、插件、客户端与 CI 的既有实现 | 模型本体能力已随外围功能增长 |

### 1.3 M4 v1 实验的有效读法

- R7 scale `0.5` 的 C3 新分布 gain 均值 `+0.152757 BPB`、3/3 为正；C cycle3 平均 delta `−0.005050 BPB`，但 seed11/47 分别 `+0.002126/+0.009497`，因此未通过当时预注册的 exact-zero all-seed Gate。Gate 结果保留，不能把它外推成“方法没有稳定—可塑信号”。
- R10 证明的是 readout-only 路径上固定 logits preservation 不通过 Gate；它没有检验真实经历 replay、importance 或 fast/slow consolidation。
- R2 的容量臂是“零 readout slot + fixed 50/50 概率平均”，没有 learned router、表征扩宽或零影响成熟期；负结果不再作为容量压力的总否决。
- R7 更新 protected context+readout，R10/R12 在 active Workbench boundary 下只更新 readout。跨轮不能把差异全归因到 consolidation 或数据。
- R12 的 A `+1.62 BPB` 是 UltraData 臂相对 simple_zh continuation 臂的差值，不是报告内直接计算的 parent→child forgetting；它只支持“abrupt replacement 高风险”。
- C/C2/C3 的 record-level disjoint 有效，但三者仍来自同一语料；R9 的 byte marginal JS 不能代表高阶序列、语义或技能边界。

## 2. 本次架构决定

1. **M4 v1 结束，M4 v2 重开。** 不新增 R13，也不沿旧 Gate 直接跑混合语料。
2. **先量尺、后模型。** 先让每个指标都声明方向、父基线、单位、delta 和能力 owner；缺基线不得写“遗忘”。
3. **在原 checkpoint 上迁移。** 旧权重成为 `slow_weight`，新增 `fast_delta=0`；迁移边界输出必须等价，不重新开始训练。
4. **快适应与慢巩固分开。** wake 写 fast/episodic，sleep 用真实经历 replay、importance 与验证写 slow；protected snapshot 只是对照/恢复点。
5. **结构必须进入主路径。** `AdaptiveNeuronNetwork` 只有被 observation→prediction/workspace→credit 实际消费，才算成长 substrate；显式旁路调用只算 shadow 实验。
6. **增长必须函数保持。** 新 population/region 以输出 gate=0 出生，学习 residual 和 router，通过 Gate 后开放；固定平均或每任务新 head 禁止作为正式方案。
7. **路由不看任务 ID。** 外部 boundary 继续控制权限，不控制认知知识选择；路由从内容、状态、不确定性、目标和资源学习。
8. **成熟技术可用于发展训练。** 原生身份按认知所有权判断。当前 AST 禁令在正式对照与合同迁移前保持，不能静默绕过。

## 3. 唯一执行顺序

| 顺序 | 阶段 | 状态 | 完成条件 |
|---|---|---|---|
| 0 | M0/M1 基线 | 已完成 | 历史能力按原边界保留，不重跑整段 |
| 1 | M2 可信训练 | 已完成当前范围 | owner/data/checkpoint/语义与多器官窄 Gate 可复用 |
| 2 | M3 Workbench 因果边界 | R0～R5 已完成，真实客户端写入仍需单独授权 | 当前不扩大副作用范围 |
| 3 | M4 v1 fixed-capacity F1 | R0～R12 已归档 | 候选与报告保留；不再发布旧方向下一步 |
| 4 | **M4.V2 继承式成长** | **当前阶段** | 依次完成 R0～R6；禁止跳级 |
| 5 | M5 知识与身体 | 冻结等待 M4.V2.R4 | Skill/MCP 成为有来源经历；硬件 capability 由客户端继承，认知/执行收益分开归因 |
| 6 | M6 provider 与产品收口 | 冻结等待 M4.V2.R4 | provider watchdog、HF 残留、客户端真实能力对齐、桌面体验一致 |
| 7 | M7 CI/发布 | 每轮相关检查，阶段末全量 | 无新增 CI 退化；release manifest 绑定代码/数据/model/package |
| 8 | M8 CUDA | 本机硬件阻塞，不阻塞 CPU 主线 | 有真实设备后做同 checkpoint 数值/收益/成本对照 |

### 3.1 M4.V2 里程碑

| 里程碑 | 单一研究问题 | 主要交付 | 退出 Gate |
|---|---|---|---|
| R0 量尺合同 | 当前结论是否由指标语义和 Gate 偏差造成 | course manifest、metric spec、cumulative scorecard、非劣界校准、legacy report audit | 能识别绝对值/delta、owner 混用、缺父基线、跨域 BPB 和 exact-zero 偏差；不训练 |
| R1 零变化迁移 | 旧 checkpoint 能否无损进入 fast/slow 状态 | versioned developmental synapse payload、migration、fresh restore | `slow=old/fast=0`；输出、score、owner、参数计数与恢复在容差内一致 |
| R2 快—慢学习 | 真实 replay consolidation 是否改善稳定—可塑性 | wake fast update、episodic sampling、sleep slow write、rollback | S/G 课程优于 slow-only/fast-only/旧 preservation；不过则不扩容 |
| R3 主路径结构桥 | adaptive region 能否真实参与计算且关闭时不改变父代 | zero-gated residual region、forward/credit 接线、owner trace | gate=0 等价；打开后有可测贡献；关闭/lesion/restore 完整 |
| R4 shadow 生长 | 容量压力能否触发有价值的新结构 | residual/conflict pressure、shadow train、admission/rollback | 优于 fixed-capacity、random growth、等参数预分配；保持与资源 Gate 通过 |
| R5 自主路由 | 无任务 ID 时能否形成专门化与协作 | content/state/uncertainty/resource router、load/credit trace | 优于最强单体、固定/随机/平均路由；router lesion 破坏组合能力 |
| R6 A8 formal | 是否展示可重复、受控资源的连续净成长 | 3 model seeds × ≥3 course seeds/orders 的 S/G/K 矩阵 | 新能力置信下界为正、旧能力非劣且无灾难域、结构 lesion 有因果、可回滚 |

依赖是严格的：`R0 → R1 → R2 → R3 → R4 → R5 → R6`。R2 失败时回到学习/记忆设计，不允许用增长掩盖；R3 失败时不能运行 R4；R4 未胜过 matched-capacity baselines 时不能称自进化。

## 4. 当前唯一下一步

M4.V2.R0、M4.V2.R1、R2 与 R3 已完成当前阶段的量尺、迁移、快/慢学习和第一条主路径结构桥 Gate；R2 候选被接受，R3 canary 通过；R4 pressure 已由 R3 bridge 的真实 predictive observation tick 驱动，candidate artifact 已被固化，独立 shadow materialization、candidate-only causal training smoke 和首个短 S/G 五臂对照已通过技术 Gate，但 pressure-driven growth 没有稳定优于 R3 fixed-capacity，candidate lesion 在 S/G 方向不一致。随后补上的逐 tick activity、candidate residual、credit、projection update 和 trace digest 证明候选确实被激活并持续收到 causal credit；把候选 projection 的学习资格从瞬时 activity 扩展到已有的 candidate eligibility trace 后，候选残差可测增强，G lesion 增益变为正，但 S lesion 仍为负，仍不能宣称增长有效。审计发现首轮五臂没有严格共享同一 parent/owner/训练边界，已在本轮修正为同一 `pressure_parent`、同一 parent-bridge 学习边界和同一课程预算；candidate-only smoke 独立保留并通过。校正后的复跑中 pressure growth 相对 fixed-capacity 在 G 改善约 `0.04233`、S 退化约 `0.00033`，相对 fixed-large 仍未占优，S lesion 仍为负。固定边界的 3 model-seed × 3 course-seed/order 矩阵证明 pressure growth 已相对 fixed-capacity 稳定改善（G `9/9`、S 均值改善约 `0.04915`），但相对 fixed-large 在 S 为 `0/9` 占优，问题从 credit 断路收敛为出生选择质量不足。单-anchor pressure birth 已修复 S 但专门化 G；top-2 birth mixture 的 9-cell 结果为：相对 fixed-capacity 的 S `9/9`、G `8/9` 改善，S lesion `9/9` 为正但 G lesion 仅 `2/9` 为正且均值为负，相对 fixed-large 的 S `8/9`、G `4/9` 占优。utility trace 证明 canonical G 的平均 residual·feedback 约 `0.0409` 不能解释 G holdout lesion `-0.00555`；第一轮 gate 的近似 credit 已证明不足。exact counterfactual、direct exact gate credit、dual-state、three-source gate 与 aligned pressure birth 已按 canonical→9-cell 完成：aligned output 使 formal G lesion 均值从约 `-0.00199` 收敛到 `-0.000008`，`7/9` cell non-worse，S 仍 `9/9` 正且均值 `+0.07515`；但 pressure 对 fixed-large 的 G 仍平均劣化 `0.00287`，所以这是“功能伤害基本消除”而不是“结构增长晋级”，整体保持 `can_promote=false`。当前唯一下一步是 **M4.V2.R4：对 aligned anchor mixture 做 deterministic birth homeostasis，使 incoming/recurrent/output 三类 candidate mapping 的有效范数与 anchor mixture 的加权二阶范数一致，再跑 canonical→9-cell**；不改变 gate credit、pressure、parent、课程、预算或对照，不进入 R5 learned router。

R0 的实现与审计产物如下：

- `taiji/continual_evaluation.py`：完成 versioned/content-addressed 的 metric spec、课程 manifest、累计 snapshot/scorecard；绝对值、parent delta、comparison delta、owner 和 read-only 输入 digest 已分离；缺少 parent baseline 时非劣判定 fail-closed；跨域 raw BPB 不可隐式聚合；phase 标签不进入 model context。
- `scripts/training/audit_taiji_m4v2_measurement_contract.py`：只读重审 R7/R10/R12，生成 `reports/taiji_m4v2_measurement_audit_20260909.json`；R10 A 被标为 absolute BPB，R12 `+1.62` 被标为 arm-vs-arm，历史 exact-zero Gate 保留但不再作为全局架构否决。
- `tests/taiji_native/test_continual_evaluation.py`：13 项定向测试通过；目标模块 ruff 与 `git diff --check` 通过；没有训练、checkpoint 权重或旧 JSON 改动。

R0 的技术结论是“量尺语义审计通过”，不是模型晋级：审计报告仍保持 `can_promote=false`。

### 4.1 R1：零变化迁移（已完成）

本步骤只迁移已有 F1 checkpoint 的表示，不写入新学习权重，不改变 forward：

- 新建 versioned developmental-synapse payload：`slow_weight=old checkpoint`、`fast_delta=0`，并保存 eligibility/importance/usage/age/plasticity 元数据的默认值；
- 对旧 F1 checkpoint 做迁移、保存、fresh restore、二次迁移四路一致性比较；
- 固定同一输入流和同一 owner graph，比较输出 logits、loss/BPB、参数计数、checkpoint digest、评估输入 digest；
- `taiji/developmental_synapse.py`：完成 F1 两个稀疏 owner 的 versioned bundle；旧 edge topology/weight 进入 `slow_weight`，`fast_delta=0`，eligibility/importance/usage/age/plasticity 显式保存；有效 forward 为 `slow + fast`。
- `scripts/training/migrate_taiji_f1_checkpoint.py`：支持裸 `taiji-native-v10` 和外层 `taiji-native-joint-training-v1`；源文件不覆盖，外层 digest 重算，磁盘 save/load/fresh restore 逐项校验。
- `taiji/model.py` / `taiji/organs.py`：R1 overlay 只读接入 F1 predictive context/readout；默认无 overlay 路径不变，挂载态写入明确拒绝。
- 真实 `output/taiji-m2-f5-seed11-private-context-20260905/last.pt` smoke 通过：`source_unchanged=true`、`source_checkpoint_digest_valid=true`、`restored_outer_digest_valid=true`、`fresh_restore_matches=true`、`fast_is_zero=true`；smoke 产物已清理。
- `tests/taiji_native/test_developmental_synapse.py`：磁盘迁移、forward/score/generate 等价、fresh restore、只读停止线均覆盖。

R1 完成条件：

1. 旧 F1 权重、topology、owner graph 无损可寻址；
2. `slow=old/fast=0` 时输出、score、generate、参数计数和 restore 在容差内等价；
3. source checkpoint 不被覆盖，outer/inner digest 都能验证；
4. R1 overlay 不允许写入，未通过 R2 前不启动真实 fast/slow 学习。

### 4.2 R2：快适应—慢巩固（实现合同与单 CPU smoke 已完成）

本步骤第一次允许写 developmental state，但仍不扩容、不接入新的 adaptive region：

- 先做 S 同分布 smoke，再做 G 渐进混合 canary；每个 checkpoint 都生成绝对能力、parent delta、comparison delta 和资源记录；
- 固定同一 parent/owner graph，比较 slow-only、fast-only、fast + 真实经历 replay + consolidation；禁止把旧字节前缀或静态 preservation logits 当作经历 replay；
- wake 只写 fast/episodic evidence，sleep/replay 才允许写 slow；记录 eligibility、importance、usage、age、plasticity 的变化；
- 通过 calibrated epsilon、worst-domain catastrophe、fresh restore、rollback 和 read-only Gate 后，才允许保留 R2 候选；失败则恢复 R1 parent，不运行 R3/R4。

已完成的实现与单 CPU smoke：

- `taiji/developmental_synapse.py`：发展突触提供 `slow_weight + fast_delta` 的固定拓扑 forward/backproject、fast/slow 局部更新、eligibility/importance/usage/age 更新、慢巩固和边界约束；另有 versioned 的真实 wake replay event/buffer。
- `taiji/model.py` / `taiji/organs.py`：R1 overlay 继续默认只读；R2 只能通过显式 mode 进入，`fast`/`slow`/`fast_slow` 的写入 owner 明确，replay 只消费实际 wake 局部 credit trace，静态 preservation logits 被拒绝；checkpoint 保存 developmental state 与 replay evidence，fresh restore 强制回到 read-only。
- `scripts/training/eval_taiji_m4v2_r2_canary.py`：固定同一 parent、同一 owner graph，对比 `slow_only`、`fast_only`、`fast_replay`；S 后接 `80/20 → 60/40 → 40/60 → 20/80` G 课程，生成 manifest、scorecard、parent delta、资源和恢复/回滚 Gate。
- `reports/taiji_m4v2_r2_sg_canary_20260909.json`：单 CPU smoke 的三臂合同 Gate 全通过，S/G 的本次样本相对 parent 均为 lower-surprise 改善；这只证明实现边界和一次小课程闭合，不证明正式晋级。

### 4.3 R2.Formal：多 course-seed/order S/G Gate（已完成）

只复用 R2 的实现，不改 owner、不扩容、不接 R3。已先以 parent 独立、等比例、等长度 block 的波动校准 epsilon，再运行 3 个 course seed/order 的 S/G 短课程；每个 arm 都保留 slow-only、fast-only、fast+real replay+consolidation 对照。正式报告为 `reports/taiji_m4v2_r2_formal_sg_20260909.json`：初版因把超过上限的 `0.4629` 波动截断为 `0.05` 而被 fail-closed；修正校准 block 后 `max_deviation=0.0176304`，epsilon=`0.0176304`，所有 Gate 通过。replay 相对每个 course 的最强 fixed-capacity arm 在 S/G 均不劣且严格更优，旧 F1 owner 未被写入，fresh restore/rollback/read-only 全通过。该结果接受 R2 候选，但不代表 R3 已通过，更不代表 A8 晋级。

### 4.4 R3：主路径 zero-gated adaptive residual bridge（已完成）

R3 只实现一个最小结构桥，不创建多个专家、不引入任务 ID 路由、不接 Skill/MCP 或客户端外围。候选从当前 `observation → predictive context/readout → credit` 路径获得输入和误差；稳定 F1 trunk 保持原 owner；新增 residual population 以输出 `gate=0` 出生；forward、credit、checkpoint、lesion 和 rollback 消费同一 owner graph。

已完成的实现与 Gate：

- `taiji/adaptive_residual_bridge.py`：以 `AdaptiveNeuronRegion` 为底层的稀疏 residual population，稳定 unit identity、显式 `predictive_context` 输入源、gate、lesion、局部 credit、runtime state 和 versioned payload；recurrent fan-in 对自连接约束做了小容量边界裁剪。
- `taiji/model.py`：把 bridge 接入 native predictive forward；gate=0 时不 step、不写 credit、不改变原预测；显式 `learn_adaptive_residual_bridge` 允许在结构 admission 试验中冻结成熟 F1 context/readout，只把同一 causal error 送入新桥；checkpoint、identity lineage、参数清单、reset、fresh restore 和 rollback 均保留 bridge。
- `taiji/__init__.py`：导出 bridge 的 format/version/类型，保持类型边界可寻址。
- `scripts/training/eval_taiji_m4v2_r3_bridge.py`：先做 checkpoint preflight，再做 zero-gate 等价、active residual、local credit、mature F1 owner 不变、lesion、fresh restore 和 rollback canary；报告为 `reports/taiji_m4v2_r3_bridge_canary_20260909.json`。
- `tests/taiji_native/test_m4v2_r3_bridge.py` 与 `test_m4v2_r3_bridge_canary.py`：覆盖默认无 bridge、gate=0 主路径等价、显式 bridge credit、owner freeze、lesion、fresh restore、rollback 与报告 Gate。

R3 canary 结果：所有 12 项 Gate 通过；`gate=0` 输出与 parent 完全一致，active residual `L1=1.30631685`，概率最大变化 `0.00535537`，local credit 改变 bridge payload，mature F1 两个 owner 未变，fresh restore/lesion/rollback 通过。该结果只接受“结构候选已进入主路径”的技术事实，不接受结构成长或 A8 晋级，`can_promote=false` 保持不变。

### 4.5 R4：shadow 生长与 matched-capacity 对照（阶段收束，未晋级）

R4 要回答的不是“能否再创建一个区域”，而是“固定容量已出现可观测压力时，按真实压力出生的候选是否比随机增长和从一开始就等量预分配更有用”。只允许沿 R3 bridge 这一个结构入口做第一轮 shadow growth；不引入 learned router、不接 Skill/MCP、不接客户端外围。

实施顺序固定为：

1. **压力合同与主路径观测（已完成）**：`taiji/adaptive_residual_growth.py` 已定义 residual error、fast/slow conflict、activity saturation、utility gap、resource state 的 versioned/content-addressed observation；`AdaptiveResidualGrowthTrigger` 提供 EMA、连续压力、预算、parent digest、重复 evidence 和 checkpointable decision。`Taiji.observe(readout="predictive")` 只在同一 native observation→prediction/credit tick 已产生 prior error 且 R3 bridge 作为当前 owner 时记录 pressure；没有 task ID，没有 synthetic evaluator 路由，也不自动改拓扑。decision digest、last pressure、trigger checkpoint 和 model fresh restore 已闭合。
2. **候选提议（已完成）**：`should_propose=true` 的 decision 已固化成 content-addressed zero-impact candidate artifact，在 bridge 上按稳定身份描述一个最小 unit，保存 parent/source digest、proposal evidence、topology diff、结构预算和 pending topology proposal；artifact 创建没有改变 parent 函数、bridge unit 数量或 forward/credit substrate。
3. **shadow materialization（已完成）**：从 candidate artifact 复制 bare parent 到独立 shadow 实例，在独立 checkpoint 中应用 pending topology proposal；旧 unit 的 incoming/recurrent 支持、权重、runtime state 逐项保持，新增 unit 初始化可寻址，shadow 的 identity-preserving sparse projection 对齐原输出，新增 unit projection edge 初始为零，shadow 为 `gate=0`；fresh restore 后 candidate/parent digest、bridge identity 和 shadow payload 一致。materialization 失败不会改变原 parent 且不消耗正式结构预算。
4. **shadow 训练（最小 Gate 已完成）**：先保存 shadow bare checkpoint，在 fresh restore 后只打开 shadow gate，冻结成熟 F1 trunk 与 parent readout，只给 shadow residual/projection/候选 region credit 做一次最小更新；已经验证候选 projection 会在真实活动 tick 改变、旧 region 前缀不被写入、再次保存/fresh restore/续步有效，失败可回滚到 bare shadow。该 Gate 只证明训练边界，不证明能力收益。
5. **短课程与准入对照（已完成首轮，但结果未通过晋级）**：已用同一 parent、同一 S/G 课程、同一训练预算比较 frozen parent、R3 fixed-capacity bridge、pressure-driven growth、random growth、同等最终参数量的 fixed-large；五臂都完成 checkpoint preflight、短训练/holdout、candidate lesion、fresh restore、rollback 和资源记录。报告中 pressure-driven growth 的 S/G 结果没有稳定优于 R3 fixed-capacity，candidate lesion 的方向不一致，因此只接受“对照管线闭合”，不接受“生长有效”。
6. **candidate credit/activation 修订（已完成一轮，未晋级）**：已增加逐 tick candidate activity、projection credit、candidate-vs-parent residual、eligibility 和 lesion trace，并把候选 projection 的资格改为使用已有 candidate trace；成熟 F1 trunk、pressure contract、课程和对照数量未改变。新报告显示 pressure candidate 的平均 candidate/parent residual ratio 约 `0.00981`，G lesion delta `+0.00474`，S lesion delta `-0.00027`；候选 credit 通路有效，但跨 S/G 不一致，因此 R4 仍未通过。
7. **matched-capacity 边界校正（已完成）**：五臂现在统一从 `pressure_parent` 进入；fixed-capacity 与 growth efficacy 都使用 parent bridge 可学习边界；candidate-only 版本单独作为冻结 parent 的 shadow smoke。校正后技术 Gate 全通过，但 pressure growth 只在 G 改善、S 轻微退化，且相对 fixed-large 未占优，不能晋级。
8. **多 seed/order 复测（已完成，未晋级）**：固定校正后的五臂、eligibility 规则、pressure contract、课程内容和预算，完成 `3 model seeds × 3 course seeds/orders = 9` 个 cell；9/9 的 pressure growth G 相对 fixed-capacity 改善，S 均值改善 `0.04915`，但 pressure 相对 fixed-large 在 S 为 `0/9` 占优，故不能把收益归因于压力驱动出生已经优于等量预分配。正式报告为 `reports/taiji_m4v2_r4_shadow_formal_20260909.json`。
9. **pressure-conditioned candidate birth（已完成一轮，未晋级）**：候选 artifact 仍保持零影响和 content-addressed；shadow materialization 根据同一 `pressure_parent` 的 native activity+eligibility 选择单 anchor，再升级为固定 fan-in/预算内的 top-2 activity+eligibility birth mixture；anchor identities、归一化权重和 parent digest 可 fresh restore 重算。两轮都证明 S 能力可改善，但 G 的 candidate lesion 不稳定，不能晋级。
10. **candidate utility/selectivity 诊断（已完成）**：逐 tick 已保存 candidate residual、同一 causal feedback、candidate utility、candidate/parent residual ratio 和 S/G phase。canonical 中 pressure candidate 的 G utility 均值为正但 G holdout lesion 为负，证明必须引入上下文选择性，不能用全局 utility 标量直接开关。
11. **candidate output selectivity gate（已完成一轮，未晋级）**：增加了从 parent eligibility trace 到 candidate gate 的 1×fan-in 稀疏 projection；所有 growth/fixed-large 臂均携带 gate 以保持参数匹配，gate bias/weights、utility baseline 和 runtime state 可 checkpoint，parent freeze/restore/rollback 全通过。但 gate 值仍在约 `0.50` 附近，G lesion 仍不稳定，因此不继续调学习率。
12. **exact counterfactual gate credit（已完成一轮，未晋级）**：shadow 在 candidate forward 后保存同一 readout/evidence 下的 candidate-off 概率，下一 observation 用真实 target 在 shadow learn 之前计算 `loss_off - loss_on`；9-cell 技术 Gate 全通过，但 candidate lesion 的 G 只有 `2/9` 正，不能晋级。
13. **direct exact gate credit（已完成一轮，未晋级）**：exact delta 已直接用于连续 gate 的 local update，EMA 只保留为诊断；pressure 训练期 G gate 均值约 `0.46`，但 S/G holdout gate 都约 `0.42`，G lesion 仍为负，说明瓶颈转为 gate 输入表征不足，不能晋级。
14. **dual-state context gate（已完成一轮，未晋级）**：gate projection 的输入改为 parent instantaneous activity 与 parent eligibility 的拼接；所有 growth/fixed-large 臂使用相同新 topology，技术 Gate、checkpoint、restore、rollback 全通过。canonical holdout 的 S/G gate 约 `0.451/0.429`，formal G lesion 仍仅 `2/9` 正，说明局部双状态略有区分但不能支撑未见 G 泛化。
15. **three-source context gate（已完成一轮，未晋级）**：gate 输入扩展为 parent bridge input context + instantaneous activity + eligibility；所有 growth/fixed-large arm 使用相同 topology，技术 Gate、checkpoint、restore、rollback 全通过，但 canonical holdout S/G gate 约 `0.419/0.422`，formal G lesion 仅 `2/9` 正，说明继续扩 gate 输入不能修复出生功能未对齐。
16. **aligned pressure birth output（已完成一轮，部分通过）**：pressure anchor/mixture 在 materialization 时同步初始化 candidate→context output projection，使候选输出映射继承同一 anchor mixture；formal G lesion 均值约 `-0.000008`、`7/9` non-worse，S lesion `9/9` 正，但 pressure 相对 fixed-large 的 G 仍平均劣化 `0.00287`，只能接受“功能伤害基本消除”，不能晋级。
17. **birth homeostasis（已完成一轮，部分通过）**：对 aligned anchor mixture 的 incoming/recurrent/output candidate mapping 做 deterministic weighted-second-moment norm matching，避免新单元因稀疏截断和 mixture 权重而出现过强或过弱的有效范数；保持三源 gate、direct exact credit、parent、pressure、课程、预算、projection 和 fixed/random 对照不变，完成 canonical→9-cell。formal 中 G candidate lesion 均值 `+0.003081`（`6/9` 胜出、`9/9` 不劣），S 均值 `+0.080387`（`9/9` 胜出）；相对 fixed-capacity 的 G/S 均为 `9/9` 改善。它证明 homeostasis 修复了出生映射的量级失配并恢复候选的可观测因果贡献，但相对同等最终参数量的 fixed-large，G 均值仍为 `+0.001002`、仅 `4/9` 胜出，不能证明压力驱动生长优于从一开始预分配。
18. **因果 Gate 与 R4 收束（已完成判定，未晋级）**：candidate lesion、fresh restore、rollback、owner 隔离、matched parent boundary 和 9-cell 技术 Gate 全通过；但 fixed-large 的 G 非劣/优效条件未通过，故只能接受“候选确实产生因果增益且 homeostasis 降低了新增伤害”，不能接受“压力驱动结构生长已优于等量预分配”。`can_promote=false` 保持不变，默认能力继续使用 R3/pressure parent，homeostasis 只保留在 shadow 实验路径，不进入 R5 learned router。

R4 的最小交付 `pressure/proposal → shadow materialize → shadow train → validate → lesion/admit/rollback` 已形成可复现 CPU canary 和 versioned report；birth homeostasis 已把 G candidate lesion 从此前的负均值修复到正均值，但相对 fixed-large 的 G 仍未达 Gate，因此 R4 以“技术闭合、结构不晋级”收束。后续只允许做收束审计、默认路径保护和计划证据整理，不得继续用 gate 输入、出生缩放或新语料无边界试探来掩盖 fixed-large 反证；R5 自主路由、Skill/MCP/provider/客户端外围继续冻结，直到新的架构决策明确解除停止线。

R4 收束审计已完成其研究边界部分：R4 定向 6 项测试通过，homeostasis 只存在于独立 shadow，materialize 不改变默认 parent checkpoint/forward，版本一致性检查通过，本轮涉及文件的 ruff、B/SIM 和 black 约束通过。全量 native 回归得到 `627 passed, 17 failed, 15 errors, 1 skipped`；15 个 error 主要是本机历史 pytest 临时目录锁权限，切换到仓库可写 basetemp 后代表性测试可正常执行；剩余失败集中在既有 context/delayed memory、Workbench neutral baseline、interaction structural gate 和 synapse longevity 基线，不属于本轮 R4 代码路径。tracked source 的 ruff、B/SIM 与 core mypy 在本轮已分别收敛到 0，但全量 native 回归仍未绿，不能把仓库 CI 写成已通过。

M7 CI 基线收敛第一批已完成：tracked source 的主 ruff 门禁从 9 项降为 0，安全自动修复涉及的 14 个脚本/平台/测试文件已通过 `py_compile`、ruff、black diff 和 `git diff --check`；R4 代码路径与默认 parent 未改变。随后完成 B/SIM 显式契约审计，`51 → 42 → 0`，所有 `zip` 改动都基于已确认的等长输入，未使用 unsafe 批量改写；受影响模块回归 `23 passed`。本轮完成 core mypy 棘轮：`mypy --follow-imports=silent seed taiji` 从 `175 errors / 24 files` 收敛到 `0 errors / 95 files`，同时通过源码范围 ruff、B/SIM、py_compile；与类型边界相邻的回归分组共 `85 passed`，Qwen provider 的 `tmp_path` 组受本机 pytest 临时目录权限阻塞，代码未出现断言失败。

**当前唯一下一步**：进入既有 native 回归归因，先在仓库可写 basetemp 下复现 `context/delayed memory` 准确率失败，判断是当前行为回归、过期阈值还是测试环境噪声；保持 R4 shadow/默认 parent 不变，不通过放宽阈值消除失败，确认根因后再修复并回归，R5 learned router 及 Skill/MCP/provider/客户端外围继续冻结。

R0 完成条件（已满足）：

1. 所有合同 versioned、content-addressed、checkpoint-independent 且 Python 3.10 兼容；
2. 旧 JSON 不改写，audit 生成新的版本化报告；
3. 技术结论、历史 Gate 结论和架构结论分字段输出；
4. 对 R7/R10/R12 的重审结果与本计划 §1.3 一致；
5. 定向 pytest、ruff、`git diff --check` 通过；
6. 仍保持 `can_promote=false`。

R1 通过后的唯一下一步是 R2 S/G canary；R2 formal 通过后的唯一下一步是 R3 主路径结构桥；R3 canary 通过后的 R4 先把 bridge pressure 接入真实 tick，再固化 zero-impact candidate，经过独立 shadow materialization 后才允许做一次最小训练步。R4 未通过则恢复 R3 parent，不得用 R5 路由、客户端外围或新语料掩盖容量/准入失败。

## 5. v2 课程与 Gate

### 5.1 课程轴

- **S 同分布续训**：同源 record-disjoint，多 course seed/order；检验吸收新记录。
- **G 渐进分布**：旧/新来源按 `80/20 → 60/40 → 40/60 → 20/80` 过渡；abrupt replacement 只作为压力上界。
- **K 技能组合**：结构化语义 → 世界转移 → Workbench 只读意图 → 隔离执行；检验累计技能和真实 outcome。

S/G 的 BPB 只证明低层预测变化，K 是 A8 的主要能力证据。正式候选按 `S smoke → G canary → K canary → multi-seed/order formal` 推进。

### 5.2 必须同时存在的对照

1. frozen parent；
2. fixed-capacity slow-only；
3. fast-only + rollback；
4. fast + real replay + consolidation；
5. random growth；
6. 与成长后参数量相同、从开始就预分配的 fixed-large；
7. 候选 growth + learned router；
8. growth/router/replay lesion。

缺 matched-capacity baseline 时，参数增多带来的收益不能归因于“生长方式”；缺 lesion 时，候选结构存在不能归因于能力。

### 5.3 正式晋级条件

- 所有技术/来源/owner/checkpoint/read-only/副作用检查全过；
- 新能力相对父代和最强 fixed-capacity baseline 的置信下界为正；
- 旧能力平均非劣，任一关键域不越灾难上限；
- exact-zero 作为额外观测报告，但非劣 `epsilon` 必须在看候选结果前由父代 block/course 变异校准并封顶；
- 无 evaluator task ID 的路由通过；
- 新结构 lesion 消除其增益，rollback 恢复父代函数；
- 收益/参数、收益/时间、峰值内存均在预算内；
- `can_promote` 只由 aggregate 计算，单 seed 永远不能晋级。

## 6. M5～M8 解冻顺序

| 工作 | 当前允许 | 解冻点 |
|---|---|---|
| Skill/MCP 语料 | 维护已有 provenance/schema，不扩大采集 | R4 后作为 K 课程的新经历来源；去来源 holdout 证明内化 |
| MCP 硬件与客户端插件 | 修阻塞 bug、保留 capability/ledger | R4 后验证身体能力继承、热插拔、撤销与认知 checkpoint 独立回滚 |
| provider watchdog / language organ | 保持可运行与显式回退 | R4 后做真实 artifact 轮换、故障恢复与语言效果；不替模型思考 |
| interaction-group | 保留已有实验资产 | R5 作为 learned router/协作对照，不提前扩系统 |
| 小型模拟 Gate | 继续用于快速回归 | 每个 M4.V2 里程碑先 smoke；模拟绿不替代真实 K 课程 |
| HF/Transformer 残留 | 保持 native/organ 类型边界 | M6 清理产品混淆与迁移 tombstone，不因名字做破坏性删除 |
| 视觉/桌面 | 只修崩溃和阻塞使用的问题 | M6 统一 logo、托盘/通知/任务栏、圆角/DPI、导航和生命状态入口 |
| CUDA | 记录 CPU 吞吐与内存 | M8 有真实硬件后验证同 checkpoint 等质量加速 |

## 7. 研究、CI 与提交纪律

- 每轮开始检查 working tree、现行 checkpoint、报告、active plan 和用户已有改动；不覆盖无关改动。
- 每轮只检验一个主假设，预先写输入、允许写入的 owner、基线、反证、预算、停止线和报告 schema。
- 任何训练前先在目标目录做初态保存、fresh-process 恢复、一次小更新、再保存/恢复/续步一致性；失败不训练。
- 旧报告不可改绿；更正通过新 audit/versioned report 表达。
- 定向测试在每个提交前执行；触及公共模块时运行相邻回归，阶段末再跑全量 CI。不得把“以后统一修 CI”当作接受当前新增错误的理由。
- 代码需满足 Python 3.10、ruff、black/mypy 现行配置和 `git diff --check`；只报告实际执行的检查。
- 每个完整单元精确 `git add` 并本地提交；不自动 push。训练 artifact、checkpoint、原始报告不随意删除。
- 核心讨论与当前设计留 active；证据解释放 reference；重复日志和失效执行顺序放 archive。本文保持约 30 KiB 内，已完成明细不再回填正文。
- 本计划不授权购买算力、访问新外部服务、真实客户端写入或发布。

## 8. 权威索引

- [Taiji 核心需求](../TAIJI_CORE_REQUIREMENTS.md)：CR-1～CR-10，长期不变目标。
- [Taiji Native Architecture v1](../TAIJI_NATIVE_ARCHITECTURE_V1.md)：完整认知架构与 A0～A9。
- [Taiji 继承式成长架构 v2](../architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md)：M4 v2 的状态、主路径、学习与结构准入设计。
- [M4 fixed-capacity 证据复盘](../../reference/M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md)：R0～R12 数值、资产及复审后的解释边界。
- [研究审视](../../reference/TAIJI_RESEARCH_REVIEW_2026_09_06.md)：M0～M3 的事实、失效结论和复现。
- [实现事实](../../reference/IMPLEMENTATION_STATUS_2026_08.md)：代码能力边界；更新晚于该文档的事实以本计划和版本化报告为准。
