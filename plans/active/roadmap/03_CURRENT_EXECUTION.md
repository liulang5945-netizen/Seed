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

M4.V2.R0 与 M4.V2.R1 已完成，当前唯一下一步是 **M4.V2.R2：快适应—慢巩固 S/G canary**。

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

### 4.2 R2：快适应—慢巩固（下一步）

本步骤第一次允许写 developmental state，但仍不扩容、不接入新的 adaptive region：

- 先做 S 同分布 smoke，再做 G 渐进混合 canary；每个 checkpoint 都生成绝对能力、parent delta、comparison delta 和资源记录；
- 固定同一 parent/owner graph，比较 slow-only、fast-only、fast + 真实经历 replay + consolidation；禁止把旧字节前缀或静态 preservation logits 当作经历 replay；
- wake 只写 fast/episodic evidence，sleep/replay 才允许写 slow；记录 eligibility、importance、usage、age、plasticity 的变化；
- 通过 calibrated epsilon、worst-domain catastrophe、fresh restore、rollback 和 read-only Gate 后，才允许保留 R2 候选；失败则恢复 R1 parent，不运行 R3/R4。

本步骤不修改模型权重、不运行长训练、不引入新语料。建议实现位置：

- `taiji/continual_evaluation.py`
  - `MetricSpec`：名称、方向（higher/lower）、单位、父基线、关键域与灾难上限；
  - `ContinualCourseManifest`：phase、source/dataset digest、训练预算、holdout、course seed/order；phase 标签只给 evaluator，不注入模型；
  - `ContinualEvaluationSnapshot`：每个 checkpoint × 所有累计能力的绝对值、父代 delta、对照 delta、owner 和 read-only digest；
  - `ContinualScorecard`：average/worst forgetting、forward/backward transfer、非劣判定和资源归一化。
- `scripts/training/audit_taiji_m4v2_measurement_contract.py`
  - 只读加载 R7/R10/R12；
  - 明确标出 R10 A 指标是 absolute BPB、R12 `+1.62` 是 arm-vs-arm；
  - 拒绝 R7/R10 owner 图被当作同一对照；
  - 把 exact-zero 结果保留为历史 Gate，同时生成“缺少 epsilon/course-order，不能晋级也不能作全局否决”的新 verdict。
- `tests/taiji_native/test_continual_evaluation.py`
  - higher/lower 指标方向反例；
  - absolute 值不能被当成 forgetting delta；
  - 缺 parent baseline fail-closed；
  - 跨域 raw BPB 拒绝直接合并；
  - course phase 不泄漏给模型；
  - scorecard round-trip/content digest；
  - exact-zero 与 calibrated non-inferiority 同时报告。

R0 完成条件（已满足）：

1. 所有合同 versioned、content-addressed、checkpoint-independent 且 Python 3.10 兼容；
2. 旧 JSON 不改写，audit 生成新的版本化报告；
3. 技术结论、历史 Gate 结论和架构结论分字段输出；
4. 对 R7/R10/R12 的重审结果与本计划 §1.3 一致；
5. 定向 pytest、ruff、`git diff --check` 通过；
6. 仍保持 `can_promote=false`。

R1 通过后的唯一下一步是 R2 S/G canary；R2 未通过则回退学习/巩固规则，不得用结构扩容或客户端外围掩盖失败。

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
