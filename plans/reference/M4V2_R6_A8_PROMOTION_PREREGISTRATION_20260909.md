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

## 13. K worker artifact 与 attachment Gate 执行记录（2026-09-09）

已完成上述动作。`scripts/training/build_taiji_m4v2_r6_k_worker_artifacts.py`
复用冻结的 M5.K2 课程，在 CPU 上生成三类真实 artifact：K1 semantic
`training_steps=2080`、K2 transition `training_steps=5040`、K3 deterministic
projector `training_steps=0`。训练前保存/回读 Gate 与训练后 fresh restore Gate
均通过，结果见 `reports/taiji_m4v2_r6_k_worker_artifact_build_20260909.json`。

重新运行 `scripts/training/eval_taiji_m4v2_r6_k_worker_attachment_preflight.py`
后，`reports/taiji_m4v2_r6_k_worker_attachment_preflight_20260909.json` 为
`status=passed`：K1/K2/K3 同 parent bundle、owner graph、typed exchange、S/G
retention、candidate stage、rollback 和 joint checkpoint 全通过；
`attachment_gate_passed=true`、`can_start_k_controlled_canary=true`，但
`can_start_r6_formal=false`、`can_promote=false`。artifact 仍是本机 ignored
checkpoint，不作为 standalone formal evidence，也没有接 default runtime。

执行中曾发现 preflight 与 builder 使用不同 parent episode id，导致 digest
不一致；已统一复用 R6 baseline parent factory，并加入回归测试。该修复是
lineage 身份修复，不是放宽 Gate 或重算指标。

**当前唯一下一步**：在这组已挂接 artifact 上运行 single-cell controlled K
canary，完整记录 S→G→K typed lineage、真实 outcome、candidate namespace 更新
和失败 rollback；在 canary 通过前不扩大 formal 矩阵、不接 default runtime、不
引入 MCP/provider/client/CUDA。

## 14. controlled K canary 执行记录（2026-09-09）

已实现并运行 `scripts/training/eval_taiji_m4v2_r6_k_worker_controlled_canary.py`，
报告为 `reports/taiji_m4v2_r6_k_worker_controlled_canary_20260909.json`。
首轮失败被定位为隔离 Workbench root 未覆盖 `get_setting("workspace_path")`，
导致执行器读取默认 workspace；按已通过的 M5.K2 runner 模式补齐 root selector
后，未改模型、artifact 或判据，重跑通过。

最终 single-cell 结果 `status=passed`：K1/K2 typed result 均 `resolved`；
`typescript_05.ts` 真实 `workspace.read` 成功、reward `1.0`；K3 接受并应用
真实 outcome/dependency projection；adapter exchange 包含 K1/K2/K3 manifest
lineage；joint checkpoint、candidate stage、explicit rollback 和 S/G retention
全部通过。全程 `training_performed=false`、`candidate_training_performed=false`、
`candidate_promoted=false`，provider/MCP/client/CUDA/default runtime 未接入。

这只证明已挂接 artifact 能完成 single-cell native K 闭环与原子回滚，不能替代
跨 seed formal、长期自进化或 promotion evidence；`can_start_r6_formal=false`、
`can_promote=false` 继续固定。

## 15. formal runner 输入合同与失败归因已冻结（2026-09-09）

已新增 [M4V2_R6_FORMAL_RUNNER_INPUT_CONTRACT_20260909.md](M4V2_R6_FORMAL_RUNNER_INPUT_CONTRACT_20260909.md)，将本轮 single-cell 证据的边界落实为 formal runner 合同：

- causal 轴明确区分“真实 outcome 被消费”与 A/B/C/lesion 因果证据；
- resource 轴补齐 wall-clock、peak working set、更新步数、写入字节、参数字节和
  inference trace，canary 的一次 `elapsed_seconds` 不再被当作 resource Gate；
- side-effect 轴要求 parent/candidate namespace、checkpoint/rollback、typed
  lineage、工作区写入和外部集成前后账本；
- formal 输入必须是显式 content-addressed manifest，固定 model/course/repeat
  seed、S→G→K、CPU 和每个 model seed 的 parent/worker/course registry；禁止隐式
  `_parent()`、目录猜 artifact 或按结果重选 seed；
- 失败按 `input_contract → lineage → checkpoint_restore → course_harness →
  environment_blocker → worker_resolution → workbench_outcome → projection →
  causal → retention → resource → side_effect → aggregate` 优先级归因，保留
  cell/arm/step/digest，不能用笼统异常覆盖首因。

当前合同状态仍是 `can_start_r6_formal=false`：本机只有 model-17 的 flat worker
artifact，三份 parent checkpoint registry、model-23/31 worker bundle 和 manifest
preflight 尚未完成。下一步只实现 manifest preflight 并补齐三份 parent registry，
不运行 full formal course、不接 default runtime。

## 16. formal input manifest preflight 执行记录（2026-09-09）

已实现并运行 `scripts/training/eval_taiji_m4v2_r6_formal_input_manifest_preflight.py`，
生成 `plans/manifests/taiji_m4v2_r6_formal_input_v1.json` 和三份本机 ignored parent
checkpoint：

- `model_17.pt`、`model_23.pt`、`model_31.pt` 均完成 atomic save、fresh restore，
  checkpoint digest、owner/source/resource manifest digest 全部一致；
- course registry `0/1/2`、baseline repeat `401/503/607`、`S→G→K`、五类 arm、
  CPU-only resource contract、closed side-effect contract 和 failure contract 全部
  通过；
- report `reports/taiji_m4v2_r6_formal_input_manifest_preflight_20260909.json`
  为 `status=blocked_input`，唯一阻断是 `worker_registry` 尚未包含三个 model
  seed 的 K1/K2/K3 bundle；runner 没有从目录猜 artifact，也没有运行 formal task。

因此 parent checkpoint 训练前/运行前保存 Gate 已经闭合，但 formal input 尚未 ready，
`can_start_r6_formal=false`、`can_promote=false` 不变。下一步只生成 model-17/23/31
各自的 content-addressed K worker bundle、挂接到 manifest 并重新执行 preflight；不
运行 full formal course、不接 default runtime。

## 17. 三 seed worker registry 与 manifest Gate 执行记录（2026-09-09）

已按 input contract 为 model 17/23/31 分别生成 K1/K2/K3 bundle，并重新执行
manifest preflight：

- 三个 model seed 各有独立 candidate namespace、parent checkpoint digest、worker
  source/resource digest 和 bundle/owner graph digest；
- K1 semantic、K2 transition 的训练前 save/restore 和训练后 restore 均通过，K3
  deterministic projector fresh restore 通过；
- `plans/manifests/taiji_m4v2_r6_formal_input_v1.json` 的 parent/worker/course
  registry 完整，报告 `reports/taiji_m4v2_r6_formal_input_manifest_preflight_20260909.json`
  为 `status=passed`、`formal_input_ready=true`；
- 仍没有运行 S→G→K full formal，没有计算 causal/resource/retention/all-arm
  side-effect Gate，`can_start_r6_formal=false`、`can_promote=false` 不变。

下一步只实现消费该 manifest 的 formal runner per-cell ledger 和结构化失败归因；
静态/输入 Gate 通过前不执行 9-cell course、不接 default runtime。

## 18. Formal runner 入口层 preflight 执行记录（2026-09-09）

已实现 `scripts/training/eval_taiji_m4v2_r6_formal.py`，并在不执行课程的模式下
重新验证 manifest/input Gate。报告 `reports/taiji_m4v2_r6_formal_preflight_20260909.json`
生成 `9×5=45` 个 `not_started` per-cell/per-arm ledger row，所有 row 预留 causal、
resource、retention、side-effect、checkpoint 和 failure 字段；
`status=input_ready`、`formal_input_ready=true`，但 `course_executed=false`、
`can_start_r6_formal=false`、`can_promote=false`。

该入口层只证明 formal runner 的输入闭合和 ledger 形状，不是 R6 formal 结果。下一步
在一个固定 cell 上实现五 arm 的 S→G→K execution/measurement layer，首个失败按
冻结的优先级记录后停止，不直接扩展到 9 cells。

## 19. 首个 single-cell execution 与 fixed-large 阻断（2026-09-09）

已运行 `scripts/training/eval_taiji_m4v2_r6_formal_single_cell.py`，固定
`model_seed=17 / course_seed=0`，报告为
`reports/taiji_m4v2_r6_formal_single_cell_20260909.json`：

- candidate 真实 `workspace.read` 成功，K3 accepted dependency projection、
  typed lineage、stage/rollback、S/G retention 通过；
- 同一真实 outcome 的 K3 lesion 精确返回 `outcome_feedback_lesioned`，adapter
  在 projection 未 accepted 时 fail-closed 拒绝 stage，lesion Gate 通过；
- parent/matched 只完成 detached K 的 S/G/checkpoint control；没有把它们当成 K
  能力结果；
- fixed-large 只有 R6 旧 structural shadow preflight，没有 K-task-equivalent
  executor，报告按 `input_contract / fixed_large_k_control_required` 阻断，整体
  `status=blocked_controls`。

该结果是正确的设计阻断，不是模型失败，也不是阈值失败。下一步先实现与 candidate
同输入/同资源口径的 K fixed-large control，再重跑该 single cell；未完成前不进入
9-cell formal。

## 20. fixed-large control 边界修订（2026-09-09）

`fixed-large` 已从“旧 R4 structural shadow”修订为 K-task-equivalent native control，
合同见 [M4V2_R6_FIXED_LARGE_K_CONTROL_PREREGISTRATION_20260909.md](M4V2_R6_FIXED_LARGE_K_CONTROL_PREREGISTRATION_20260909.md)：两个独立 K1/K2 replica、固定 arithmetic ensemble、worker-training task seed `3/4`、formal holdout `0/1/2` 不重叠、同一 typed input/output 和真实 Workbench。该修订不改变 candidate 主阈值，也不引入 Transformer/provider/router。

旧 structural shadow 只作历史证据，不得进入 R6 K aggregate。下一步只实现 native
fixed-large artifact/checkpoint preflight 和 model17/course0 comparator；完成前
`can_start_r6_formal=false`、`can_promote=false`。

## 21. Native fixed-large artifact 与 checkpoint Gate（2026-09-09）

已实现 `scripts/training/build_taiji_m4v2_r6_k_fixed_large_ensemble.py`，为
`model_seed=17` 生成两个独立 native K1/K2 replica（worker-training task seed `3/4`），
并写入 `checkpoints/taiji_k_fixed_large/model_17/taiji_r6_k_fixed_large_ensemble.pt`。
报告为 `reports/taiji_m4v2_r6_k_fixed_large_ensemble_build_model_17_20260909.json`，
其关键 Gate 全部通过：

- `ensemble_width=2`、参数量 `9666`，固定 arithmetic ensemble 没有 learned router、
  task-id 分支、Transformer、provider 或外部执行依赖；
- 两个 replica 的训练前 checkpoint save/restore、训练后 fresh restore、ensemble
  fresh restore 和 K3 fresh restore 均通过；
- worker-training task `3/4` 与 formal holdout task `0/1/2` 的路径、semantic input
  digest、transition input digest 交集均为空；
- artifact、ensemble、owner/resource/source manifest 均 content-addressed，
  `optimizer_state_present=false`，`can_start_r6_formal=false`、`can_promote=false`。

## 22. Native fixed-large single-cell comparator（2026-09-09）

已实现 `scripts/training/eval_taiji_m4v2_r6_k_fixed_large_canary.py` 并接入
`eval_taiji_m4v2_r6_formal_single_cell.py`。`model_seed=17 / course_seed=0` 的
fixed-large arm 与 candidate 使用同一 holdout 输入、语言 registry、read-only planner、
隔离 Workbench 和 K3 outcome projection；报告
`reports/taiji_m4v2_r6_k_fixed_large_controlled_canary_model_17_20260909.json` 与
`reports/taiji_m4v2_r6_formal_single_cell_20260909.json` 均通过：

- fixed-large `typescript_05.ts` 真实 read、K1/K2 typed result、K3 dependency、
  exchange lineage、checkpoint、stage/rollback 和 parent retention 通过；
- 切断一个 K1 replica 的 fact head 后，branch lesion 对 fact map 可观测，且没有
  污染完整 ensemble 主路径；
- 五臂 single-cell 当前为 `status=passed`，但 `course_executed=false`、
  `can_start_r6_formal=false`、`can_promote=false` 继续固定；旧 structural shadow
  仍不进入 R6 K aggregate。

下一步只补齐 candidate/fixed-large 的 paired peak-resource、checkpoint-write、
parameter/inference 和 side-effect measurement，使单 cell 的资源/因果 ledger 完整；
完成并预注册前不扩大到 9 cells、不运行 full formal、不接 default runtime。

## 23. Single-cell paired resource ledger（2026-09-09）

已按同一 `process_rss_before_after_lower_bound` 方法补齐 candidate 与 fixed-large 的
paired resource ledger，并重跑 `model_seed=17 / course_seed=0`：

- 两个 K 单步都为 `1/1`，`measurement_complete=true`，inference trace 都为 `1`，
  training update 都为 `0`，side-effect/rollback Gate 通过；
- candidate K1/K2 为 `4833` parameters / `19332` bytes，checkpoint write 为
  `35403` bytes；fixed-large 宽度 `2` 为 `9666` parameters / `38664` bytes，
  checkpoint write 为 `125829` bytes；
- peak RSS、wall-clock、参数字节、checkpoint 字节和 inference trace 已记录
  `fixed-large - candidate` paired delta；该 delta 只用于资源对照，不事后改变任何
  promotion 阈值；
- `reports/taiji_m4v2_r6_formal_single_cell_20260909.json` 当前为 `status=passed`，
  但 `course_executed=false`、`can_start_r6_formal=false`、`can_promote=false`。

下一步冻结 9-cell formal 的资源 validity/aggregate 合同（包括同一 CPU 口径、无效
cell、peak resource、checkpoint/parameter/inference 和 paired delta 的聚合规则），
合同冻结前不执行 9-cell full formal。

## 24. 9-cell resource validity / aggregate 合同冻结（2026-09-09）

资源合同已独立冻结：[M4V2_R6_RESOURCE_AGGREGATE_PREREGISTRATION_20260909.md](M4V2_R6_RESOURCE_AGGREGATE_PREREGISTRATION_20260909.md)。它沿用输入 manifest 的 CPU、peak `1.25×`、wall-clock `1.5×` 上限，固定每 arm 的 RSS、参数、checkpoint、inference、训练步数、device/resource digest 字段，规定缺失/超预算 cell 不得被删除或用均值补齐，并冻结 9-cell 的 mean/min/max 与一侧 95% Student-t lower bound 聚合方式。

该合同不改变 candidate 主 Gate，也不把 fixed-large 事后改造成晋级阈值；fixed-large 仍是 strongest-capacity paired control。当前 full formal 仍关闭，下一步只补齐 model23/31 的 fixed-large artifact、fresh restore 和 formal input registry。

## 25. Three-seed fixed-large registry closure（2026-09-09）

model23/31 的 native fixed-large 已完成构建并通过与 model17 相同的 checkpoint、
source/holdout non-overlap、CPU/resource manifest、ensemble/K3 fresh-restore Gate。
`fixed_large_registry` 已由 preflight 从 artifact 自动重建并写入 formal input manifest；
三 seed registry、parent/worker/course registry 和 manifest digest 全部通过，formal
runner 也已生成带 fixed-large artifact digest 的 9×5 not-started ledger。

这只表示输入闭合，不表示运行或晋级：model17/course0 的真实 single-cell 仍是唯一
已执行 cell，`course_executed=false`、`can_start_r6_formal=false`、`can_promote=false`
保持。下一步先实现单 cell execution contract 的正式 ledger 写入和失败归因，再以同一
runner 扩展 8 个 cell；不接 default runtime、provider、MCP、client 或 CUDA。

## 26. Five-arm resource ledger closure（2026-09-09）

model17/course0 的 single-cell 已对五个 arm 统一执行 resource Gate：
`frozen-parent`、`matched-fixed-capacity`、`candidate-continuation`、`fixed-large`、
`lesion` 的 `resource_gate` 全部为 `true`，每个 arm 均列出 CPU、resource digest、
peak-RSS 方法、参数/字节、checkpoint path/bytes、inference trace 和 training steps。
candidate/fixed-large paired delta 与 K3 lesion 结果保持通过，且没有引入新的外部变量。

该结果只闭合首个 cell 的资源/side-effect ledger，不把 single-cell 冒充 9-cell formal。
下一步把已验证的五臂执行合同抽成 formal runner 的可复用 cell executor，先重放并做
字段级一致性检查，再扩展 model17 的其余 course seed，最后才进入完整 3×3 aggregate；
`can_start_r6_formal=false`、`can_promote=false` 不变。

## 27. Reusable executor replay（2026-09-09）

`eval_taiji_m4v2_r6_formal_single_cell.py` 现提供 `run_cell(model_seed, course_seed, …)`，
默认包装入口不变。两个连续的 model17/course0 执行在同一 input manifest 下产生相同
`execution_contract_digest`，说明 parent/worker/fixed-large 选择、五臂结构、K3 lesion、
rollback、side-effect 和 failure contract 没有因重放漂移；资源时间/RSS 仍按合同允许变化。

下一步是把 formal runner 的一个 `not_started` row 接到这个 executor，完成单 cell 的
显式 ledger 写回与 failure attribution；不直接跳到 9-cell，也不解冻 default runtime、
provider、MCP、client 或 CUDA。

## 28. First formal ledger row execution（2026-09-09）

`--execute-cell` 已消费且只消费 `model17/course0` 一个 `not_started` row；formal
execution report 将五臂真实结果、资源 Gate、side-effect、checkpoint/rollback、
execution digest 和 failure attribution 写回，结果 `executed_passed`，其余 8 rows
仍未启动。该入口继续强制 `can_start_r6_formal=false`、`can_promote=false`，没有接入
default runtime、provider、MCP、client 或 CUDA。

下一步按预注册矩阵只执行 `model17/course1`，验证 course seed 变化不会造成 executor
隐式使用 model17/course0 数据或 parent；不并发扩大、不跳过失败 row。

## 29. Monotonic ledger / course1 execution（2026-09-09）

runner 已加入 prior execution ledger digest 校验与不可覆盖规则；`model17/course1`
成功追加，course0 的 `executed_passed` 记录保持不变，当前 execution report 只保留
两个已执行 row，另外 7 个仍为 `not_started`。course1 的五臂 K/resource/rollback/
side-effect contract 通过，未接入任何外围系统。

下一步只推进 `model17/course2`，完成 model17 的 3-course slice；在该 slice 闭合前不
扩大到 model23/31，不运行 aggregate 或 promotion。

## 30. Model17 three-course slice closure（2026-09-09）

model17/course2 已通过并追加，model17 的三个 course row 均为 `executed_passed`，其余
model23/31 row 仍未启动。formal runner 同时冻结了 9-cell 的前序执行顺序，禁止跳过
失败或未执行的 row；这只是 execution order/ledger Gate，不是 promotion Gate。

下一步只执行 `model23/course0`，继续复用同一 parent/worker/fixed-large/resource/rollback
合同；在六个 model23/31 row 完成前不计算 9-cell aggregate、不接默认 runtime。

## 31. Model23/course0 execution（2026-09-09）

model23/course0 已成功追加，candidate/fixed-large/lesion 的 K 因果边界和五臂资源、
rollback、side-effect contract 全部通过，execution order Gate 没有跳过 model17 slice。
当前只推进 model23/course1；完整矩阵和 promotion 仍关闭。

## 32. Model23/course1 execution（2026-09-09）

model23/course1 已通过并追加，累计五个 cell row 为 `executed_passed`；candidate/
fixed-large/lesion 因果与五臂 resource/rollback/side-effect contract 均未出现漂移。
下一步只执行 model23/course2，完成第二个 model slice 后再进入 model31，不计算 aggregate。

## 33. Model23 three-course slice closure（2026-09-09）

model23/course2 已通过，model23 三个 course row 完成，累计 6/9 row 为
`executed_passed`；当前进入 model31/course0，仍不开放 aggregate、default runtime 或
promotion。

## 34. Model31/course0 execution（2026-09-09）

model31/course0 已通过并追加，累计 7/9 row 完成；candidate/fixed-large/lesion 与资源、
rollback、side-effect contract 均保持通过。下一步只执行 model31/course1，仍不计算
aggregate 或 promotion。

## 35. Model31/course1 execution（2026-09-09）

model31/course1 已通过并追加，累计 8/9 row 完成；仅 course2 未启动。下一步执行最后
一个 row，完成后才进入预注册的 aggregate 计算，promotion 仍关闭。

## 36. Nine-cell execution ledger closure（2026-09-09）

model31/course2 已通过，9/9 cell row 均为 `executed_passed`；五臂 resource、causal
lesion、rollback 和 side-effect ledger 全部闭合。下一步只做预注册 aggregate 计算和
统计 Gate，aggregate 完成前不改变 `can_promote=false`，不接 default runtime/provider/
MCP/client/CUDA。

## 37. Nine-cell aggregate audit and control-design blocker（2026-09-10）

aggregate runner 已按冻结合同完成 9×5 arm 的 content-addressed 重读和统计，报告为
reports/taiji_m4v2_r6_formal_aggregate_20260909.json。9/9 execution row 均通过；
candidate K 成功率为 1.0，lesion 为 0.0，candidate-lesion 差值为 1.0，
candidate floor、K3 lesion、lineage、rollback 和 side-effect Gate 均通过。candidate
相对 matched 的 peak mean 为 1.1091×，低于 1.25×，但 wall mean 为 6.5769×
（范围 5.9264×–6.8290×），9 个 cell 全部超过预注册 1.5×。

更关键的阻断不是模型表现，而是 matched-fixed-capacity 的实际实现：它没有 attach K
worker，参数数和 inference trace 都为 0，故 frozen-parent/matched 的新增能力为
null，与 candidate 的两个 0.25 paired-delta Gate 都不可计算。聚合器不把 detached
当作 0 分，也不删除或替换 cell；当前 status=blocked_aggregate，
can_start_r6_formal=false、can_promote=false。

这确认了控制命名与实验语义发生偏移。下一步唯一工作是新增一份控制修订预注册并实现
真正 same-capacity matched K：同一 K bundle、同一 restore/输入/trace 和参数/内存预算，
但禁止 feedback/output admission 与参数更新，得到可观测的 0 能力对照；随后只重跑
matched/candidate/lesion 的资源与能力 paired slice。禁止事后放宽资源阈值、把 fixed-large
改作 promotion threshold，或接入任何外围 runtime。

## 38. Matched-control revision first-cell evidence（2026-09-10）

新的 control revision manifest 已内容寻址生成，digest 为
9221b0920413e7d470821dc66b7daa1a2769ba2502c6b6af1801eb5ad1555ae7，旧
blocked_aggregate 证据不被覆盖。model17/course0 revised five-arm cell 已通过：
frozen-parent admission baseline=0.0、matched no-feedback=0.0、candidate=1.0，
两个原本缺失的 capability delta 都为 1.0；matched 与 candidate 参数/trace 相同，
wall=1.2724×、peak=1.0024×，均在既有预算内。

matched 只执行 K1/K2/K3 内部路径并在 output admission 前停止，不 bind dependency
projection、不写 candidate stage、不创建 exchange、不更新参数；因此这是实际 same-
capacity control，而不是将 detached null 改成 0。当前仍保持
can_start_r6_formal=false、can_promote=false。下一步只执行新 revision 的
model17/course1，成功后再按顺序扩展，不接默认 runtime、provider、MCP、client/CUDA。
