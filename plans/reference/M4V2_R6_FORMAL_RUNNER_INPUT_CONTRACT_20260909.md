# M4.V2.R6 formal runner 输入合同与失败归因（2026-09-09）

> 本文件冻结 R6 formal runner 的输入边界、证据字段和失败归因规则。它不是
> formal 结果，也不授权训练、晋级或接入 default runtime。只有所有入口 Gate
> 满足后，runner 才能按本合同读取输入并执行 9-cell course。

## 1. 冻结结论

当前 single-cell controlled K canary 已证明：

1. K1 semantic 和 K2 transition 可以从真实 worker artifact 恢复并产生 typed
   result；
2. 只读 `workspace.read` 被 planner 接受后，隔离 Workbench 返回真实成功 outcome；
3. K3 可以消费该 outcome，写入同 tick world dependency，并把 projection、
   dependency 和三类 worker manifest 放入 typed exchange lineage；
4. joint checkpoint、candidate namespace、explicit rollback 和 S/G 旧能力
   retention 在一个 cell 上闭环；
5. 本轮没有 candidate training、default runtime、provider、MCP、client、CUDA
   或 promotion side effect。

这条证据仍然**不能**推出：

- K 新能力相对 matched fixed-capacity 的因果增益；
- K3 feedback 相对 no-feedback/lesion 的因果增益；
- 9 cells、跨 model/course seed 的稳健性；
- candidate training 后的旧能力 retention、资源等价或长期自进化。

因此 formal runner 必须同时产出三条独立证据轴：

| 证据轴 | canary 当前状态 | formal 必须新增 |
| --- | --- | --- |
| 因果 | 真实 outcome 被消费，但没有对照和 lesion | 同 cell 的 frozen/matched/candidate/lesion/fixed-large arms，逐 cell 比较，不能用均值抵销失败 |
| 资源 | 只有一次 `elapsed_seconds`，且 adapter `training_steps=0` | 每 arm 的 wall-clock、peak working set、训练更新步数、写入字节、candidate 参数字节和 inference trace 数量 |
| 副作用 | single-cell 的 default/external=false，stage/rollback 通过 | 每 arm 的 namespace、checkpoint、runtime attachment、外部集成和 rollback ledger 前后快照 |

## 2. Runner 只能消费显式输入清单

正式 runner 的唯一入口是：

```text
plans/reference/M4V2_R6_FORMAL_RUNNER_INPUT_CONTRACT_20260909.md
        ↓
taiji-m4v2-r6-formal-runner-input-v1.json
        ↓
R6 formal runner
```

runner 不得在没有清单时隐式调用 `_parent()`、隐式生成 worker、从目录猜测
artifact、从 report 猜测 checkpoint，或在看到结果后重新选择 seed/arm。清单路径
使用相对仓库根的 POSIX 风格路径，报告只保存相对路径和 digest，避免把本机绝对
路径当作实验身份。

### 2.1 顶层合同

输入清单的最小格式固定为：

```json
{
  "format": "taiji-m4v2-r6-formal-runner-input-v1",
  "version": 1,
  "experiment_id": "m4v2-r6-a8-promotion-formal-v1",
  "device": "cpu",
  "model_seeds": [17, 23, 31],
  "course_seeds": [0, 1, 2],
  "baseline_repeat_seeds": [401, 503, 607],
  "phase_order": ["S", "G", "K"],
  "parent_registry": [],
  "worker_registry": [],
  "course_registry": [],
  "resource_contract": {},
  "side_effect_contract": {},
  "failure_attribution_contract": {}
}
```

顶层合同有以下不可变约束：

- `model_seeds`、`course_seeds`、`baseline_repeat_seeds` 必须分别严格等于
  `[17, 23, 31]`、`[0, 1, 2]`、`[401, 503, 607]`；
- `phase_order` 严格为 `S → G → K`；每个 candidate arm 都从对应 cell 的同一
  immutable parent 分叉；
- `device` 严格为 `cpu`；CUDA、provider、MCP、client、network 不是隐含输入；
- 清单自身必须有 content digest，runner 运行开始后不得修改；
- 任一 registry 缺失、重复、越界或 digest 不匹配，归因为
  `input_contract`/`lineage`，不得归因为模型能力。

### 2.2 parent registry

每个 model seed 必须有一个实际落盘、可 fresh restore 的 parent entry：

```json
{
  "model_seed": 17,
  "checkpoint_path": "checkpoints/taiji_r6_parents/model_17.pt",
  "checkpoint_digest": "sha256...",
  "owner_graph_digest": "sha256...",
  "source_manifest_digest": "sha256...",
  "resource_manifest_digest": "sha256...",
  "namespace": "taiji:parent:model-17",
  "fresh_restore_digest": "sha256..."
}
```

`checkpoint_digest` 是清单中 parent 的身份，不允许 runner 以运行时重新构造的
对象替代它。fresh restore 必须在任何 arm 运行之前完成；恢复后的 checkpoint
digest、owner/source/resource manifest 必须全部与 entry 相等。parent namespace
在整个 formal course 中只读，candidate 只能写自己的 namespace。

当前真实状态：R6 baseline factory 已能确定性地产生 model 17/23/31 的 parent
摘要，但尚未形成 formal 所要求的三份明确 parent checkpoint registry；因此
formal runner 不能仅凭现有 `_parent()` 直接启动。

### 2.3 worker registry

每个 model seed 必须有一个固定的 K1/K2/K3 worker bundle，且三类 artifact 必须
同时存在：

```json
{
  "model_seed": 17,
  "parent_checkpoint_digest": "sha256...",
  "candidate_namespace": "taiji:k:candidate:model-17",
  "bundle_digest": "sha256...",
  "workers": {
    "k1.semantic": {
      "path": "checkpoints/taiji_k_workers/model_17/taiji_r6_k1_semantic.pt",
      "artifact_digest": "sha256...",
      "worker_checkpoint_digest": "sha256..."
    },
    "k2.transition": {
      "path": "checkpoints/taiji_k_workers/model_17/taiji_r6_k2_transition.pt",
      "artifact_digest": "sha256...",
      "worker_checkpoint_digest": "sha256..."
    },
    "k3.outcome_projection": {
      "path": "checkpoints/taiji_k_workers/model_17/taiji_r6_k3_outcome_projection.pt",
      "artifact_digest": "sha256...",
      "worker_checkpoint_digest": "sha256..."
    }
  },
  "source_manifest_digest": "sha256...",
  "resource_manifest_digest": "sha256..."
}
```

worker bundle 必须满足当前 `KWorkerManifestBundle` 合同：精确包含 K1/K2/K3、
同一 parent/source/resource、相同 owner graph 和 candidate namespace。artifact
必须在 runner 读结果前固定；同一个 model seed 的三个 course cells 必须消费同一
bundle，不能按 cell 重新训练或挑选 artifact。当前已生成的 flat model-17
artifact 只覆盖一个 bundle；它可以作为迁移输入，但不是完整 formal registry。

### 2.4 course registry

每个 course seed 必须有固定的 course entry：

```json
{
  "course_seed": 0,
  "course_label": "r6-order-0",
  "course_variant_seed": 0,
  "source_manifest_digest": "sha256...",
  "holdout_digest": "sha256...",
  "phase_order": ["S", "G", "K"],
  "workspace_contract": {
    "root_isolation": true,
    "split_isolation": true,
    "read_only_routes_only": true
  }
}
```

course input 必须在同一 cell 的所有 arms 中字节/摘要一致。workspace root、
registry、语言集合、train/holdout 分离和 observation digest 都是 course input
的一部分；任何路径选择错误、跨 split digest 泄漏、holdout 类型未进入词表或
隔离 root 未生效，归因为 `course_harness`，不得把它记成 worker/模型失败。

## 3. Formal arms 与输入/输出边界

每个 9-cell 必须按以下固定 arm 集合运行：

| arm | K worker | 目的 | 允许写入 |
| --- | --- | --- | --- |
| `frozen-parent` | 不挂接 | parent noise/retention baseline | 不写 parent；只写独立报告 |
| `matched-fixed-capacity` | 不消费 K 增长边界 | 同资源、同容量的 continuation 对照 | 仅独立 candidate namespace |
| `candidate-continuation` | 完整 K1/K2/K3 | 声明的 Taiji K continuation | 仅自己的 candidate namespace |
| `fixed-large` | 固定的大容量对照 | 区分“新增能力”与单纯容量收益 | 仅独立 candidate namespace |
| `lesion` | candidate boundary 被预注册 lesion | 验证收益是否真正依赖 K boundary | 仅独立 candidate namespace |

`candidate-continuation` 和 `lesion` 必须共享 parent、course、资源预算和输入
holdout；lesion 只能切断预注册的 K boundary/feedback，不能同时改变 corpus、
seed、worker artifact 或阈值。`matched-fixed-capacity` 与 `fixed-large` 的参数
和资源预算必须在读取 candidate 结果之前写入清单。

每个 arm 的最小 per-cell 输出必须包括：

```json
{
  "cell": {"model_seed": 17, "course_seed": 0},
  "arm": "candidate-continuation",
  "parent_checkpoint_digest": "sha256...",
  "worker_bundle_digest": "sha256...",
  "phase_rows": [],
  "new_capability": {},
  "old_capability_retention": {},
  "causal": {},
  "resource": {},
  "side_effects": {},
  "checkpoint_ledger": {},
  "failure": null
}
```

## 4. 三类必须独立记录的 evidence

### 4.1 Causal evidence

canary 中 `real_workbench_success=true` 和 `k3_dependency_applied=true` 只证明
真实 outcome 被链路消费，不是因果对照。formal 需要对每个 cell 记录：

- candidate、matched、lesion 的同任务逐项成功率和 paired delta；
- K3 outcome/dependency projection 是否被下一步 world/依赖实际读取；
- lesion 后新增能力是否下降到预注册阈值以下；
- frozen-parent 与 fixed-large 的绝对能力、旧能力和资源记录；
- 每一个失败 task 的输入 digest、worker manifest digest、projection digest 和
  outcome digest。

主判据沿用 addendum：candidate K holdout `>=0.75`，candidate 相对
matched-fixed-capacity 和 frozen-parent 的 paired delta 各 `>=0.25`，且 lesion
必须破坏声明的新增收益；任意 cell 失败即 formal gate 失败。

### 4.2 Resource evidence

canary 的 `elapsed_seconds` 仅用于诊断，不足以进入 resource Gate。formal 每个
arm/phase 必须记录：

- `wall_clock_seconds`；
- `peak_working_set_bytes`；
- `training_update_steps`、`candidate_parameter_bytes`、`checkpoint_write_bytes`；
- `inference_trace_count`；
- `device`、线程/worker 数和 resource manifest digest。

判据固定为：peak working set 不超过 matched parent 的 `1.25x`，wall-clock 不超过
`1.5x`；预算在输入清单中冻结，不能用 candidate 结果反向扩大。

### 4.3 Side-effect evidence

每个 arm 必须在 pre/post/rollback 三个时点记录：

- parent namespace、candidate namespace、active namespace；
- parent/candidate/rollback checkpoint digest；
- typed exchange、worker bundle 和 lineage 是否保留；
- default runtime、provider、MCP、client、network、CUDA 是否为 false；
- workspace 写入路径集合及其 digest；formal 的 read-only course 不得产生业务
  工作区写入；
- failure artifact 是否先落盘，再执行 rollback；rollback 后 parent digest 和
  旧能力输出是否恢复到合同容差。

canary 的 stage/rollback 通过只能作为该字段族的 single-cell 先验，不能代替
9-cell all-arm side-effect Gate。

## 5. 失败归因合同

### 5.1 统一 failure record

任何 `status != passed` 的 cell/arm/phase 都必须输出一个 failure record；异常
不能被一个笼统的 `blocking_reason` 吞掉：

```json
{
  "class": "course_harness",
  "phase": "K",
  "cell": {"model_seed": 17, "course_seed": 0},
  "arm": "candidate-continuation",
  "step": "workspace.read",
  "is_model_evidence": false,
  "is_environment_blocker": false,
  "stop_line": true,
  "recoverability": "runner_fix_required",
  "exception_type": "FileNotFoundError",
  "message": "isolated workspace root did not reach Workbench executor",
  "evidence_digests": ["sha256..."],
  "parent_checkpoint_digest": "sha256...",
  "worker_bundle_digest": "sha256..."
}
```

字段规则：

- `message` 只描述可复现事实，不写“模型笨”“模型聪明”等结论；
- `is_model_evidence=true` 只允许用于已通过 input/lineage/checkpoint/course/
  environment Gate 后的 worker resolution、真实 outcome、causal、retention 或
  aggregate 失败；
- `is_environment_blocker=true` 只用于 OS/ACL/依赖/设备等 runner 无法控制的阻断；
- `stop_line=true` 表示本次 formal 必须停止，不能跳过该 cell；
- 每条 record 必须携带足够 digest，能回到 parent、worker、course 和具体 step。

### 5.2 固定分类和归因优先级

runner 按以下优先级给首次失败分类；前一层失败时，不得把同一事件下沉为模型
能力失败：

1. `input_contract`：清单缺失、版本/矩阵/路径/字段非法；
2. `lineage`：parent、owner graph、source/resource、worker bundle 或 namespace
   digest 不一致；
3. `checkpoint_restore`：训练前保存、fresh restore、joint restore 或 rollback
   digest 不一致；
4. `course_harness`：workspace/split/schema/语言词表/observation 或输入 digest
   构造错误；
5. `environment_blocker`：Windows ACL、权限、依赖、设备或临时目录不可用；
6. `worker_resolution`：合同合法后 K1/K2 输出类型、状态或必需字段无法解析；
7. `workbench_outcome`：intent 已被合法接受，但真实执行 outcome 不符合任务合同；
8. `projection_contract`：真实 outcome 无法被 K3 接受、应用或形成依赖 lineage；
9. `causal_gate`：A/B/C/lesion 对照没有达到预注册因果阈值；
10. `retention_gate`：旧 S/G/K 能力低于 epsilon 或 rollback 后未恢复；
11. `resource_gate`：working set、wall-clock、更新步数或写入字节超预算；
12. `side_effect_gate`：发生未经授权的 namespace/runtime/external write 或 rollback
    污染；
13. `aggregate_gate`：单 cell 通过但 9-cell 汇总/最差域/全矩阵条件不通过；
14. `unexpected`：不匹配以上类别的异常，必须保留原始异常类型和 digest，禁止
    静默重试后覆盖首次事实。

canary 首轮的 workspace root 问题按该规则属于 `course_harness`，不是
`workbench_outcome` 或 `worker_resolution`；修复后真实 read 成功的结果才可作为
后续 causal course 的输入。

## 6. Runner 的停止与晋级边界

formal runner 的退出状态只反映本次合同执行是否完整：

- `status=blocked_input`：输入合同/lineage/checkpoint/environment 在正式执行前
  阻断；
- `status=failed`：已执行的 cell/arm 有 Gate 失败或出现可归因异常；
- `status=passed`：9/9 cells 的所有 formal Gate 全通过，但仍只表示
  `promotion_gate=true` 的形式证据完整；
- `can_promote` 在当前项目阶段仍固定为 `false`，必须经过单独 runtime rollout
  review，不能由 formal runner 自动改为 true。

即使 runner 产生 `status=passed`，也不得接入 default runtime，不得删除 R4/R5
shadow，不得自动引入 provider/MCP/client/CUDA。formal 只验证 fixed-capacity
Taiji-owned continuation 的证据闭环。

## 7. 实施顺序

1. 先生成三份 content-addressed parent checkpoint，并对三份 parent 做 fresh
   restore；
2. 将 K worker builder 改成按 model seed 生成独立 bundle，生成前后都通过
   checkpoint Gate；
3. 生成并校验本合同的 input manifest；manifest 校验失败时不启动 course；
4. 实现 formal runner 的 per-cell ledger 和上述 failure record；先跑 input/
   lineage/checkpoint/course preflight，再运行任何 K task；
5. 仅在 preflight 全部通过后串行执行 9 cells，最后才计算 causal/resource/
   retention/side-effect aggregate；
6. formal 结果提交后再由独立 review 决定是否进入 runtime rollout；本合同不改变
   `can_promote=false`。


## 8. 输入 manifest 执行记录（2026-09-09）

已完成本合同的输入 Gate：

- model 17/23/31 的 parent checkpoint 已实际落盘并 fresh restore，三份 digest、
  owner/source/resource manifest 一致；
- 复用现有 K worker builder，分别以 learner seed 17/23/31 生成独立的 K1 semantic、
  K2 transition、K3 deterministic projection bundle；每个 bundle 都绑定对应
  parent digest 和 `taiji:k:candidate:model-{seed}` namespace；
- 三个 bundle 的 K1/K2 训练前 checkpoint save/restore、训练后 restore、K1/K2
  typed probe、K3 restore 和 bundle digest 校验均通过；
- `plans/manifests/taiji_m4v2_r6_formal_input_v1.json` 已包含完整 parent/worker/course
  registry；`reports/taiji_m4v2_r6_formal_input_manifest_preflight_20260909.json`
  为 `status=passed`、`formal_input_ready=true`；
- 该过程没有执行 formal course、没有接 default runtime，也没有 provider/MCP/client/
  CUDA side effect。

输入 Gate 通过不等于 R6 formal admission：`can_start_r6_formal=false`、
`can_promote=false` 继续固定。下一步是实现只消费该 manifest 的 formal runner
per-cell ledger 和结构化 failure record；runner 必须先重新执行 manifest/fresh
restore，再允许任何 S→G→K task，且首个 Gate 失败即停止。


## 9. Formal runner 入口层执行记录（2026-09-09）

已实现 `scripts/training/eval_taiji_m4v2_r6_formal.py` 并运行入口 preflight：

- runner 重新读取并校验 manifest/input preflight，而不是调用隐式 `_parent()` 或
  目录扫描；
- 生成了 `9 cells × 5 arms = 45` 个 `not_started` ledger row，每行预留
  `phase_rows`、新能力、旧能力 retention、causal、resource、side-effect、
  checkpoint ledger 和结构化 failure 字段；
- report `reports/taiji_m4v2_r6_formal_preflight_20260909.json` 为
  `status=input_ready`、`formal_input_ready=true`，但明确
  `course_executed=false`、`training_performed=false`、`can_start_r6_formal=false`、
  `can_promote=false`；
- 静态校验：`py_compile` 和目标脚本 Ruff 均通过；没有执行任何 S→G→K task，没
  有 default runtime/provider/MCP/client/CUDA side effect。

入口层已经冻结，下一步进入真正的 execution layer：先在一个预注册 cell 上实现
五 arm 的 S→G→K phase ledger、真实 Workbench outcome、resource/side-effect
measurement 和首个失败归因；单 cell 未通过前不扩大到 9 cells。

## 10. 首个 single-cell execution 结果（2026-09-09）

已实现并运行 `scripts/training/eval_taiji_m4v2_r6_formal_single_cell.py`，固定
`model_seed=17 / course_seed=0`。该 runner 复用冻结 manifest，执行 candidate 与
K3 lesion 的同路径真实 Workbench read，并回放 parent/matched controls；没有运行
完整 S→G→K course。

结果见 `reports/taiji_m4v2_r6_formal_single_cell_20260909.json`：

- `candidate-continuation`：K1/K2 typed result、`typescript_05.ts` 真实
  `workspace.read`、K3 accepted projection、dependency 应用、stage/rollback 和
  S/G retention 通过；单步 K task success 为 `1/1`；
- `lesion`：使用同一输入和真实成功 outcome，K3 以精确 reason
  `outcome_feedback_lesioned` 拒绝，adapter 因没有 accepted projection 而拒绝
  stage，说明反馈边界确实 fail-closed；
- `frozen-parent`、`matched-fixed-capacity`：只完成 parent/S/G 控制与 checkpoint
  retention，K worker 按 arm 合同保持 detached，不能当作 K 新能力结果；
- `fixed-large`：现有资产只有 R6 旧 structural shadow preflight，没有与 K
  candidate 同任务、同输入、同资源口径的 fixed-large executor。runner 将该缺口
  归为 `input_contract / fixed_large_k_control_required`，报告整体为
  `status=blocked_controls`，没有用 structural shadow 冒充 K 因果对照。

因此这一步闭合了 candidate/lesion 的单步边界，但没有关闭 R6 formal Gate。
下一步必须先设计并实现 K-task-equivalent fixed-large control，再重跑同一 single
cell；在该对照可比较前不扩大矩阵、不运行 9-cell formal。

fixed-large 的具体定义已冻结到 [M4V2_R6_FIXED_LARGE_K_CONTROL_PREREGISTRATION_20260909.md](M4V2_R6_FIXED_LARGE_K_CONTROL_PREREGISTRATION_20260909.md)：使用两个独立 native K1/K2 worker replica、固定 arithmetic ensemble、formal holdout 外的 worker-training slices `3/4`，不使用 learned router 或外部模型。旧 R4 structural shadow 继续保留为历史 shadow，不再作为 R6 K control。

## 11. Native fixed-large control 与 single-cell rerun（2026-09-09）

旧 structural shadow 的 input-contract 阻断已经按独立预注册合同修正。新的
`taiji-k-fixed-large-ensemble-v1` artifact 由两个 native K1/K2 replica 组成，宽度固定
为 `2`，以 arithmetic score-map ensemble 产生普通 typed K1/K2 result；K3 仍是同一
deterministic outcome projector。builder 的 checkpoint/source/holdout Gate 通过，且
worker-training task `3/4` 与 formal holdout `0/1/2` 的路径及输入 digest 无交集。

single-cell 重跑后，`fixed-large` 与 candidate 在同一真实 `typescript_05.ts`
Workbench read 上均通过 K1/K2→planner→execution→K3；fixed-large 的 K1 branch lesion
可观测，五臂报告变为 `status=passed`。这只关闭了 fixed-large 的可比性阻断，不关闭
formal/promotion：当前 candidate 的 resource `measurement_complete=false`，fixed-large
仍缺少与 candidate 同方法的 peak working-set 记录，paired resource/side-effect
aggregate 尚未形成。因此 `course_executed=false`、`can_start_r6_formal=false`、
`can_promote=false` 保持。

## 12. Single-cell paired resource ledger closure（2026-09-09）

随后已补齐并验证 candidate/fixed-large 的同方法 resource ledger。两臂均有
`measurement_complete=true`、`process_rss_before_after_lower_bound`、参数字节、
checkpoint write、inference trace 和 wall-clock；single-cell report 记录了
`fixed-large - candidate` 的逐字段 delta，且两臂 K success 都为 `1/1`。这只证明
资源字段闭合和本 cell 可比较，不能推断 9-cell aggregate；因此 formal runner 仍不
执行 course，`can_start_r6_formal=false`、`can_promote=false` 不变。

下一步是预注册 9-cell 的资源 validity/aggregate 规则，包括 CPU 口径、无效 cell、
peak resource、checkpoint/parameter/inference 和 paired delta 的处理；规则冻结前
不得运行 full formal。

## 13. 9-cell resource validity / aggregate contract（2026-09-09）

资源合同已冻结到 [M4V2_R6_RESOURCE_AGGREGATE_PREREGISTRATION_20260909.md](M4V2_R6_RESOURCE_AGGREGATE_PREREGISTRATION_20260909.md)：每个 arm 必须真实记录统一 CPU/RSS 方法、参数/字节、checkpoint write、inference trace、训练步数和 resource digest；matched-fixed-capacity 提供 `1.25×` peak、`1.5×` wall-clock reference，缺失/超预算 cell 不可删除或均值填补，9-cell 输出 mean/min/max 与一侧 95% Student-t lower bound。

当前 `model17/course0` 已验证字段闭合；由于 model23/31 尚无对应 fixed-large
ensemble registry，`can_start_r6_formal=false` 继续固定，full formal 不得启动。

## 14. Three-seed fixed-large registry closure（2026-09-09）

已为 `model_seed=23/31` 生成与 model17 同合同的 native fixed-large ensemble，并将
三份 artifact 纳入 `plans/manifests/taiji_m4v2_r6_formal_input_v1.json` 的
`fixed_large_registry`。preflight 不读取 builder report 作为真相，而是从每个 artifact
重建并核对：artifact/ensemble/source/resource/owner digest、parent digest、CPU 与无
optimizer、width=2、worker task `3/4`、formal holdout `0/1/2`、K3 fresh restore 和
source/holdout path/semantic/transition non-overlap。

`reports/taiji_m4v2_r6_formal_input_manifest_preflight_20260909.json` 现为
`status=passed`、`formal_input_ready=true`；formal runner 现为 `status=input_ready`，
其 9×5 ledger 为 `not_started` 且每个 cell 已记录对应 fixed-large artifact、ensemble
checkpoint digest。model17/course0 single-cell 通过，candidate/fixed-large paired
resource measurement 仍闭合；`course_executed=false`、`can_start_r6_formal=false`、
`can_promote=false` 继续固定。

下一步转入 execution layer：先实现**单个预注册 cell 的可复用五臂 S→G→K execution
contract**，让 model17/course0 在 frozen ledger 上逐臂写入 baseline、K capability、
causal lesion、resource、side-effect、checkpoint/rollback 和 failure attribution；
该 execution cell 通过前不扩大到其余 8 cells，也不接 default runtime/provider/MCP/client/CUDA。

## 15. Model17/course0 five-arm resource contract closure（2026-09-09）

single-cell runner 已将 frozen-parent、matched-fixed-capacity、candidate-continuation、
fixed-large、lesion 五个 arm 统一写入 resource object：CPU/device、各自
`resource_manifest_digest`、wall-clock、peak RSS 及固定测量方法、worker/candidate 参数
计数与字节、checkpoint path/bytes、inference trace、training update steps 和
`measurement_complete`。五个 arm 的 `resource_gate` 均为 `true`，缺失任一 arm 字段会
直接生成 `resource_gate` failure，不再只检查 candidate/fixed-large paired subset。

重跑结果仍为 `status=passed`，candidate/fixed-large K success 为 `1/1`，K3 lesion
边界可观测，所有 side-effect/rollback 约束保持关闭；这闭合了首个 cell 的资源 ledger，
但尚未运行完整 S→G→K course，也不代表 9-cell aggregate 或 promotion。下一步是把该
五臂执行合同抽为 formal runner 可复用的单 cell executor，先以同一 model17/course0
重放并逐字段比对，再开放其余 8 cells。

## 16. Reusable cell executor replay closure（2026-09-09）

single-cell 脚本已抽出公开的 `run_cell(model_seed, course_seed, ...)` executor，
`run_single_cell()` 仅保留 model17/course0 的兼容默认入口；executor 从显式 manifest
按 model/course seed 取 parent、candidate worker 和 fixed-large registry，不再把 seed
写死在执行逻辑中。

executor 另生成 replay-stable 的 `execution_contract_digest`，只覆盖五臂结构化结果、
causal/side-effect/failure 和 Gate，不覆盖 wall-clock/RSS 等允许波动的资源字段。以
同一 manifest 连续执行 model17/course0 两次，canonical 与 replay 均 `status=passed`，
五臂 `resource_gate` 全为 true，两个 digest 均为
`8d670a310c08a81e024b3c8f2256ba5afa4808491871e67637205d6156c5d22c`，字段级合同一致。

这闭合了首个 cell executor 的重放一致性，但仍没有执行 9-cell course。下一步让 formal
runner 以该 executor 消费一个 `not_started` ledger row，先完成 model17/course0 的
显式 ledger 写回与 failure attribution，再扩大矩阵；不接 default runtime/provider/
MCP/client/CUDA。

## 17. First formal ledger row execution（2026-09-09）

formal runner 已新增受控 `--execute-cell` 入口：它先重新通过 input/runner preflight，
只允许消费预注册的 `model17/course0` `not_started` row，再调用同一个 `run_cell`，将五
臂 `status/phase_rows/new_capability/retention/causal/resource/side_effects/checkpoint`
和 failure attribution 写回该 row。结果见
`reports/taiji_m4v2_r6_formal_execution_20260909.json` 与
`reports/taiji_m4v2_r6_formal_cell_model_17_course_0_20260909.json`：该 row 为
`executed_passed`，五臂均有 resource/side-effect/checkpoint ledger，execution digest
为 `8d670a310c08a81e024b3c8f2256ba5afa4808491871e67637205d6156c5d22c`；其余 8 个 row
保持 `not_started`。

这仍不是完整 course 或 aggregate：`course_executed=false`、`can_start_r6_formal=false`、
`can_promote=false`。下一步只开放下一个已预注册 row `model17/course1`，继续用同一
executor 和同一 failure/resource contract；若该 cell 任何一臂失败，保留 row/报告并按
优先级归因，不用其他 cell 抵销。

## 18. Monotonic execution ledger and course1 closure（2026-09-09）

formal execution report 现在在执行前读取上一份同 manifest digest 的 ledger；只有目标
row 仍为 `not_started` 才允许写入，已执行 row 不可覆盖。`model17/course1` 已用同一
executor 完成并追加到 execution report：course0 与 course1 均为 `executed_passed`，
course1 五臂 resource Gate 全部通过，execution contract digest 为
`59a02e15451660abaf8ee62f8e7300ec3e011ac5b0c74d47e6bfa2bcb21c9dc7`，其余 7 个 row
仍为 `not_started`。

当前仍不是完整 3×3 formal：`course_executed=false`、`can_start_r6_formal=false`、
`can_promote=false`。下一步只执行 `model17/course2`，验证第三个 course seed 后再决定
是否开放 model23/31；任何失败都按 row 停止线处理。

## 19. Model17 three-course slice closure and fixed order（2026-09-09）

`model17/course2` 已追加成功，三个 model17 row 均为 `executed_passed`，course2 五臂
resource Gate 全部通过，execution contract digest 为
`ec45d6ee3caadb8bdfb0f3e7e6af2636ea5141e190355d585bba533865824a56`，execution report
没有 failure，model23/31 六个 row 仍为 `not_started`。

runner 现在固定执行顺序
`17/0 → 17/1 → 17/2 → 23/0 → 23/1 → 23/2 → 31/0 → 31/1 → 31/2`，并在消费目标 row
前检查所有 predecessor 为 `executed_passed`；prior ledger 的 manifest digest 与 row
status 均不满足时立即阻断。当前仍不运行 aggregate，下一步只开放 `model23/course0`。

## 20. Model23/course0 execution（2026-09-09）

`model23/course0` 已通过固定前序顺序并成功追加：candidate/fixed-large K success
均为 `1/1`，lesion 为 `0` 且五臂 `resource_gate` 全部为 true；execution contract
digest 为 `d9b35e04f4b1136e944a3fb3088412564bda2a384ef9145bacda2ebd6115e103`。累计 ledger
现在有 4 个 `executed_passed` row，`model23/course1/2` 与全部 model31 row 仍为
`not_started`，没有 aggregate 或 promotion 意味。

下一步只执行 `model23/course1`；继续要求 prior manifest digest、前序 row、fixed-large
registry 和五臂 resource/side-effect/checkpoint contract 全部通过。

## 21. Model23/course1 execution（2026-09-09）

`model23/course1` 已按固定 predecessor order 成功追加：candidate/fixed-large K
success 均为 `1/1`，lesion 为 `0`，五臂 `resource_gate` 全部为 true，execution contract
digest 为 `e0e03bb18b94a53bcf4cf9c91436819a21347b476df17adb771dcaf8b1c127c3`。累计 ledger
有 5 个 `executed_passed` row，model23/course2 与 model31 三个 course 仍为
`not_started`，不提前计算 aggregate。

下一步只执行 `model23/course2`，继续沿用 prior ledger、固定 parent/fixed-large 和五臂
resource/side-effect/checkpoint contract。

## 22. Model23 three-course slice closure（2026-09-09）

`model23/course2` 已成功追加，execution contract digest 为
`b09fe600293131f3690e942dd28e36f3b581149592dec46854900a49f746806d`；candidate/fixed-large
均为 `1/1`，lesion 为 `0`，五臂 `resource_gate` 全部为 true。累计 6 个 row 为
`executed_passed`，model31 的 `course0/1/2` 仍为 `not_started`，没有计算 aggregate。

下一步只执行固定顺序中的 `model31/course0`，继续使用 model31 的 content-addressed
parent/worker/fixed-large registry 和 prior ledger，任何失败停止在该 row。

## 23. Model31/course0 execution（2026-09-09）

`model31/course0` 已通过固定顺序并成功追加，execution contract digest 为
`26918f09462b84e8fa39e4ddb7d9c0ee1a96fecc0699cca62149341be58de791`；candidate/fixed-large
均为 `1/1`，lesion 为 `0`，五臂 `resource_gate` 全部为 true。累计 7 个 row 为
`executed_passed`，model31/course1/2 仍为 `not_started`，aggregate 尚未运行。

下一步只执行 `model31/course1`，继续使用 prior ledger 与 model31 content-addressed
输入，不改变任何 Gate 或外部变量。

## 24. Model31/course1 execution（2026-09-09）

`model31/course1` 已成功追加，execution contract digest 为
`8595f247af189b5a71f0d903b77fe40c654c5517e7e801a85942cbc0683fdfef`；candidate/fixed-large
均为 `1/1`，lesion 为 `0`，五臂 `resource_gate` 全部为 true。累计 8 个 row 为
`executed_passed`，仅 model31/course2 仍为 `not_started`，aggregate 尚未运行。

下一步只执行最后的 `model31/course2`，完成 9-cell ledger 后再做预注册 aggregate
calculation；在 aggregate 计算前不改变任何主 Gate 或 promotion 状态。
