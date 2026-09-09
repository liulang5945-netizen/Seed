# M4.V2.R6 / A8 promotion formal 预注册：同一 parent 的连续净成长

> 注册日期：2026-09-09。前置输入为 M4.V2 R0～R4 的真实状态、M5 K 轴 v2 scorecard 和当前 Taiji continual architecture v2。本文只冻结 promotion course 的入口条件、同一 parent 合同、课程矩阵、量尺和停止线；在入口 Gate 完成前不得运行 R6 formal，也不得把 K1/K2/K3 standalone shadow 接到默认 runtime。

## 1. 为什么必须另立 promotion course

M5.K1/K2/K3 formal 已分别证明：

- 结构化语义、world transition、只读 Workbench 和真实 outcome/dependency 在受控课程中可以形成可复现组合链；
- 三条 evidence 的 A/B/C 因果分离在各自 formal 矩阵上成立；
- 但三个 learner 都是 `learn=False` 的 standalone shadow，没有同一 parent checkpoint、持续训练后的旧能力保持、资源等价、rollback 和默认 runtime owner。

因此 K scorecard 的 `k_evidence_closed=true` 不能直接转化为 A8 的 `can_promote=true`。R6 只回答更高层问题：**同一个 Taiji parent 在连续 S/G/K 课程中，能否在受控资源内增加新能力，同时保持旧能力并可回滚。**

## 2. 当前入口状态与不可越过的前置条件

### 2.1 已有事实

- R0：量尺、source digest、累计 scorecard 合同已存在；
- R1：旧 checkpoint 到 fast/slow 的零变化迁移已通过；
- R2：快/慢学习与真实 replay 的 S/G Gate 已通过；
- R3：zero-gated adaptive residual bridge 已进入 native observation→prediction/credit 主路径，关闭时与 parent 等价；
- R4：pressure/candidate/lesion/rollback 影子链和技术 Gate 已闭合，但压力驱动 growth 尚未稳定胜过等参数 fixed-large，因此 `can_promote=false`；
- R5：learned router 尚未解冻；
- M5 K 轴：K1/K2/K3 standalone evidence 已由 v2 scorecard 收束，`promotion_gate=false`。

### 2.2 R6 formal 的入口 Gate

R6 runner 只能在以下条件全满足后实现/运行：

1. R4 的结构增长候选经过独立决策：要么在新预注册中证明其相对 matched fixed-large 的结构性收益，要么明确将 R6 设为 fixed-capacity parent 的连续能力 promotion，不把失败的 R4 growth 写成成功；
2. 若 R6 使用 router，先完成独立 R5 canary/formal 的 router/lesion/无任务 ID Gate；若不使用 router，必须在 R6 preregistration addendum 中冻结“为何不需要 router”的边界，不能静默删掉路由变量；
3. 存在一个可 fresh restore 的、带 owner graph、source manifest、resource manifest 和 rollback token 的 parent checkpoint；
4. K1/K2/K3 能力必须有 Taiji-owned runtime adapter 和统一 owner lineage，不能把 standalone learner 对象直接挂到默认 runtime；
5. parent 在独立 S/G/K holdout 上先完成 baseline repeat，校准波动 `epsilon` 并设置预注册上限；看见 candidate 结果后不得调大 epsilon；
6. checkpoint preflight 能证明：parent、candidate、fixed-capacity、random-growth/fixed-large 对照都能保存、fresh restore、digest 校验和 rollback；
7. 当前 CI/native 账本没有与本课程相关的真实失败；Windows 临时目录 ACL 只能作为环境噪声记录，不能被写成 Gate 通过。

入口不满足时，R6 只允许做 contract、adapter schema、checkpoint preflight 和无训练 smoke，不允许写新模型权重或宣布 A8 结果。

## 3. parent、owner 和课程边界

### 3.1 同一 parent 合同

所有 arms 从同一个 immutable parent checkpoint digest 开始，并共享：

- 相同的 Taiji owner graph、semantic/world/action memory 初态和 source manifest；
- 相同的 S/G/K train/holdout 划分、课程顺序、训练步数上限和资源预算；
- 相同的 parent baseline repeat、checkpoint preflight、fresh restore 和 rollback protocol；
- 相同的真实 Workbench/S6B outcome boundary；失败 outcome 不能被拒绝后从统计中删除。

任何 arm 只能写入自己的 candidate namespace。parent namespace 在 frozen arm 中必须字节不变；candidate 发生回滚时恢复 parent，并保留失败 artifact、lineage 和 resource audit。

### 3.2 K 能力的 native adapter

M5 K evidence 只作为冻结的 benchmark/课程定义来源，不能直接把 `StructuredSemanticLearner` 或 K3 projector 当默认认知 owner。R6 必须先提供一个 Taiji-owned adapter，至少声明：

- observation/event schema、world facts、goal/content/action lineage 的 owner；
- K1/K2/K3 能力对同一 parent checkpoint 的输入/输出 boundary；
- fast/slow/replay 写入位置、importance/eligibility、checkpoint format 和 rollback；
- Workbench outcome、dependency digest、capability snapshot 与 parent tick 的审计链；
- adapter 关闭或 lesion 时的等价/退化预期。

adapter 先以 shadow/controlled runtime 运行，必须通过 same-parent preflight；未通过前不允许 default runtime dispatch。

## 4. Formal 矩阵和 arms

### 4.1 矩阵

固定 `3 model seeds × 3 course seeds/orders = 9 cells`：

- `model_seed = (17, 23, 31)`：parent 初始化/模型随机性；
- `course_seed = (0, 1, 2)`：同一课程内 source order、组合顺序和 holdout 内容变体；
- 阶段顺序固定为 `S → G → K`，不允许通过把 K 先训练来制造 transfer；
- course order 只改变每阶段预注册的 record/episode 顺序，不改变 train/holdout 边界；
- 每个 cell 串行、独立 workspace、独立 report，任何异常停止该 cell 并报告 harness/environment。

### 4.2 最小 arms

每个 cell 至少包含以下共享 parent 的 arms：

1. **frozen-parent**：不写新能力，测自然波动和 parent retention；
2. **matched-fixed-capacity**：同一参数/资源预算，只允许现有 owner 适应，不新增结构；
3. **candidate-continuation**：使用预注册的 Taiji-owned fast/slow/replay/structure/router continuation；
4. **random-growth 或 fixed-large**：与 candidate 最终资源/参数量匹配的最强容量对照；
5. **lesion arms**：candidate lesion、router lesion、必要时 replay/feedback lesion，分别验证收益来源。

不得用“没有 candidate 的空臂”替代 matched fixed-capacity；不得用最终参数量不匹配的 fixed-large 作为唯一对照。

## 5. 量尺和 formal Gate

### 5.1 新能力

K 的新能力必须来自 formal 前冻结的未见组合和真实 outcome 任务；至少包含 K1/K2/K3 的 successor holdout，而不是只重放 M5 report。报告：

- candidate 相对 frozen-parent、matched-fixed-capacity 和最强容量对照的 paired delta；
- 9-cell 的均值、置信下界、min/mean/max 和每个语言/任务域的绝对分数；
- K3 outcome admission、dependency lineage 和 lesion 的独立字段。

Gate：新能力相对 parent 与 matched fixed-capacity 的预注册置信下界都必须 `> 0`；相对 fixed-large 不得用均值掩盖结构性退化。

### 5.2 旧能力 retention

每个阶段结束后重新测全部既有 S/G/K holdout：

- 平均旧能力 delta `≥ -epsilon`；
- 任一关键域 delta 不得超过灾难性遗忘上限；
- exact zero-retention 作为观测项保留，不替代 epsilon/non-inferiority Gate；
- 报告 average forgetting、worst-domain forgetting、backward/forward transfer；
- parent、candidate、fresh-restored candidate 的结果必须可对齐到同一个 source/metric digest。

`epsilon` 由 frozen parent 的独立 baseline repeat 校准，必须有上限；没有 parent baseline 时 retention 判定 fail-closed。

### 5.3 结构、资源和可回滚

必须全部通过：

- candidate/adapter 关闭时与 parent 输出逐位或合同容差等价；
- candidate lesion 会移除新增收益；router lesion 会显著损害需要路由的组合任务；
- checkpoint preflight、fresh restore、owner/source digest、rollback、failed artifact retention 全部通过；
- 参数/内存/训练字节/推理延迟/能耗代理不超过预注册预算；
- 新能力收益/参数、收益/时间不劣于 fixed-large 的预注册下界；
- 无任务 ID、无硬编码 domain label 的路由仍能通过；
- 无 provider、联网、客户端写入、CUDA 隐变量。

### 5.4 晋级判定

只有在 `9/9` cells 同时通过新能力、旧能力、结构因果、资源、checkpoint/rollback 和副作用 Gate 时，才允许记录 `promotion_gate=true`。即使通过，仍需单独的默认 runtime rollout review；formal 通过不自动接入生产路径。

任一核心 Gate 失败，`can_promote=false`，保留 parent，候选回滚并进入归因/新预注册；不通过调阈值、挑 seed、删除失败域或把 standalone K report 当作补偿。

## 6. 停止线

- parent preflight/fresh restore 失败：停止，不训练；
- R4/R5 入口未满足：停止 R6 execution，只做 contract/schema/smoke；
- 新能力提升但旧能力出现灾难域：停止，回到 replay/路由/owner 设计；
- candidate 不优于 matched fixed-capacity/fixed-large：不称自进化，不扩展外围变量；
- lesion 不影响收益：判定结构或反馈没有被主路径真实消费；
- 资源超预算、rollback 不可用或 source lineage 断裂：停止并回滚 parent；
- 仅 Windows pytest temp ACL 失败：标记环境阻断，单独修复可复现 runner，不改变模型判据。

## 7. 产物和唯一后续动作

- preregistration：本文件；
- parent adapter/preflight：另立实现任务和 versioned checkpoint contract；
- formal runner：只有入口 Gate 完成后才创建，不提前写训练代码；
- report：`reports/taiji_m4v2_r6_a8_promotion_formal_20260909.json`；
- 当前唯一允许的下一步：先实现 same-parent adapter 的 checkpoint preflight/smoke（不训练、不接 default runtime），并根据结果更新入口 Gate；未通过前保持所有 K shadow owner detached。

## 8. adapter preflight 执行记录（2026-09-09）

已实现 `taiji/continual_k_adapter.py` 和
`scripts/training/eval_taiji_m4v2_r6_k_adapter_preflight.py`，报告为
`reports/taiji_m4v2_r6_k_adapter_preflight_20260909.json`。本轮只验证入口
合同，不产生模型训练权重，也不修改默认 runtime：

- 13/13 checks 通过：parent digest、K3 dependency projection/lineage、prefit
  checkpoint、candidate checkpoint、fresh restore、同一 parent 绑定、隔离
  candidate namespace、explicit rollback、rollback 后 dependency 保留和
  rollback checkpoint roundtrip；
- `training_performed=false`、`default_runtime_attached=false`、
  `candidate_promoted=false`、`cuda_required=false`；
- rollback record 保存与 staged candidate 完全一致的 `trial_id`，失败候选
  回滚后恢复 parent namespace，同时保留 K3 projection 证据；
- `KContinualAdapter` 仍只是 Taiji-owned shadow/preflight boundary，不能把
  本轮 13/13 当成 R6 formal 或 A8 promotion。R4 structural growth、R5
  router、same-parent baseline epsilon、matched fixed-large 和完整资源/旧能力
  retention Gate 仍未满足，因此 `can_promote=false` 不变。

本轮同时新增 `tests/taiji_native/test_m4v2_r6_k_adapter.py`，3 个 adapter
回归测试通过。下一步仍只能在本合同入口满足后进入 R6 course；当前不得训练、
接默认 runtime 或引入 provider/MCP/client/CUDA 变量。

## 9. admission audit 执行记录（2026-09-09）

已完成只读入口审计：`scripts/training/audit_taiji_m4v2_r6_admission.py`，
报告为 `reports/taiji_m4v2_r6_admission_audit_20260909.json`。审计只读取并
content-address 了 R4 formal、R5 formal、K 轴 scorecard v2 和 R6 adapter
preflight，没有重跑训练或篡改任何判据。

审计结果为 `status=passed`（审计本身完整），但 admission 是
`blocked_shadow_only`，`can_start_r6_formal=false`、`can_promote=false`：

- 已确认：R4 technical evidence、R5 report 的拒绝结论、K1/K2/K3 evidence
  closed、R6 adapter 13/13 preflight，以及 R5 resource caps 本身无超限；
- 未满足：R4 structural growth admission（对 fixed-large 的 G non-worse
  只有 4/9）、R5 router 或明确 no-router addendum、same-parent retention
  baseline、完整 S→G→K course、全 arm 的 resource/rollback/old-capability
  Gate 和相关 CI/native ledger 的可核验清洁状态；
- 因此 R4/R5 失败资产继续作为可回滚 shadow，K evidence 不能直接成为默认
  owner，R6 adapter 也不能接入 default runtime。

当前只允许冻结一份 **R6 fixed-capacity parent admission addendum**：明确
R4/R5 均不晋级、R6 不使用 router 的边界、parent baseline/epsilon 的校准
协议，以及完整 Gate 的先后顺序。addendum 通过前不训练、不写新模型权重。

## 10. baseline/preflight 执行记录（2026-09-09）

已实现并运行 `scripts/training/eval_taiji_m4v2_r6_parent_baseline_preflight.py`，
报告为 `reports/taiji_m4v2_r6_parent_baseline_preflight_20260909.json`。
本轮结果需要分成两个层次理解：

- **机械 Gate 通过**：固定 model seed `17/23/31` × course seed `0/1/2`，每
  cell 3 个 repeat seed `401/503/607`；S/G parent repeat、checkpoint/fresh
  restore、rollback、owner/source/resource manifest 和 6 个声明 arm 的
  checkpoint roundtrip 全部通过（9/9 checks）；所有 arm 都明确
  `training_performed=false`、`candidate_promoted=false`；
- **完整 R6 baseline 尚未完成**：当前只测得 S/G，重复波动为 0，按冻结公式
  取 epsilon floor `0.01`。K parent retention 没有被 standalone K1/K2/K3
  报告伪造，报告明确 `k_parent_retention_available=false`、
  `baseline_complete=false`，`can_start_r6_formal=false`。

因此本轮不是 R6 formal admission，也不是训练许可；它只证明 CPU 上 parent 和
各类 shadow/adapter artifact 可以安全保存、fresh restore 和 rollback。下一步
唯一允许的动作是单 cell controlled adapter smoke，用同一 parent 建立 K
输入/输出和 retention 观测；仍不得接 default runtime、不得训练 formal、不得
引入 provider/MCP/client/CUDA。

## 11. controlled adapter smoke 执行记录（2026-09-09）

已把 `KAdapterInput`、`KAdapterOutput`、`KAdapterExchange` 加入
`taiji/continual_k_adapter.py`，并运行
`scripts/training/eval_taiji_m4v2_r6_adapter_controlled_smoke.py`。报告为
`reports/taiji_m4v2_r6_adapter_controlled_smoke_20260909.json`，单 cell
`model_seed=17 / course_seed=0` 的 16/16 smoke checks 通过：

- 输入已经类型化为 observation/world/goal/content/source manifest digest；
- 输出已经类型化为 action/outcome/dependency/projection digest，并要求
  same-parent echo、scope echo、input echo 和完整 lineage；
- candidate stage、fresh restore、explicit rollback 后 typed exchange 保留
  全部通过；
- 同一个 parent fresh restore 前后 S/G old-capability 观测 delta 为 `0.0`，
  在 epsilon `0.01` 内；
- `fixture_outcome_only=true`，`k_learner_training_performed=false`、
  `k_learner_owner_attached=false`、`execution_performed=false`、
  `default_runtime_attached=false`，因此 `baseline_complete=false`、
  `can_start_r6_formal=false` 仍保持。

这一步解决的是 K 的 native input/output/lineage 边界，不是把 K1/K2/K3
standalone learner 变成默认模型。下一步必须先冻结 **K learner-owner
attachment contract**：adapter 作为唯一 candidate/rollback owner，K1 semantic
和 K2 transition 作为受控 subordinate worker，K3 作为 deterministic outcome
projection；确认 checkpoint/owner graph 后才允许任何 K learner training。

## 12. K worker attachment preflight 执行记录（2026-09-09）

已完成 owner attachment contract 的实现与单 cell preflight：

- `taiji/k_worker_manifest.py` 固化 K1/K2/K3 worker manifest、输入/输出合同
  digest、同一 parent/source/resource 的 bundle 和 owner graph digest；
- `KContinualAdapter` 现在只能把完整 `KWorkerManifestBundle` 挂到匹配的
  parent/owner/source/resource/candidate namespace，并把 bundle 写入 joint
  checkpoint；fresh restore、typed exchange ledger 和 rollback 都保留该
  bundle；
- `scripts/training/eval_taiji_m4v2_r6_k_worker_attachment_preflight.py` 已
  只读扫描/恢复真实 artifact，并在恢复后执行 K1/K2 typed probe、K3 projector
  状态校验、same-parent echo、S/G epsilon retention 与 stage/rollback；
- 相关回归为 **18 passed**，新增 manifest/bundle tamper、缺失 artifact 和
  adapter restore 覆盖；目标模块 Ruff、py_compile、定向 mypy 通过。

报告为：`reports/taiji_m4v2_r6_k_worker_attachment_preflight_20260909.json`。
本机当前没有 K1 semantic、K2 transition 或 K3 outcome projector 的真实可恢复
artifact，故报告为 `status=artifact_missing`；parent fresh restore/rollback
通过，但 `can_start_r6_formal=false`、`can_promote=false`。本轮没有随机初始
化、没有训练、没有 default runtime/provider/MCP/client/CUDA 变量，也没有把
standalone K formal report 当作 worker checkpoint。

因此 R6 formal 入口仍然关闭。下一步唯一允许动作是建立可保存、可恢复且带同一
parent/source/resource lineage 的 K1/K2 worker artifact 生成管线，再用本
attachment preflight 验收；K3 只恢复真实 projector checkpoint，不训练。
