# M4.V2.R6 K learner-owner attachment contract

> 版本：v1，冻结日期：2026-09-09。本文承接
> [M4V2_R6_FIXED_CAPACITY_ADMISSION_ADDENDUM_20260909.md](M4V2_R6_FIXED_CAPACITY_ADMISSION_ADDENDUM_20260909.md)
> 与 typed adapter smoke。它只定义 K1/K2/K3 如何成为同一个 Taiji-owned
> controlled owner graph 的 subordinate worker；不代表已经训练、已经接入默认
> runtime 或已经完成 A8 promotion。

## 1. 当前真实资产与问题

当前代码中已经存在三类能力资产，但它们的 ownership 不同：

| worker | 真实实现 | 当前 checkpoint/版本 | 职责 | 是否训练 |
|---|---|---|---|---|
| K1 semantic | `StructuredSemanticLearner` / `StructuredSemanticResult` | `taiji-structured-semantic-training-v1`, version `2` | `PerceptEvent → world/Goal/ContentPlan` typed semantic output | 有本地 fit，但目前只是 standalone shadow |
| K2 transition | `StructuredSemanticTransitionLearner` / `StructuredSemanticTransitionResult` | `taiji-structured-semantic-transition-v3`, version `3` | `WorldState + PerceptEvent → predicted next world/Goal/ContentPlan` | 有本地 fit，但目前只是 standalone shadow |
| K3 outcome | `OutcomeDependencyProjector` / `OutcomeDependencyProjection` | `taiji-outcome-dependency-projection-v1`, version `1` | observed `WorldEvent → typed outcome/dependency projection` | 无 optimizer，确定性 projection |

M5.K1/K2/K3 的 formal report 证明了各自课程的 evidence，但没有证明这三个
对象已经共享一个 parent、共享一个 candidate namespace 或拥有默认 runtime。
R6 的任务不是再复制三套 learner，而是建立一个能审计它们的唯一 owner。

## 2. 唯一 owner graph

```text
immutable fixed-capacity parent
          │
          ▼
KContinualAdapter  ── candidate namespace / checkpoint / rollback / promotion veto
    │        │        │
    │        │        └── K3 OutcomeDependencyProjector (deterministic observed-world worker)
    │        └────────── K2 StructuredSemanticTransitionLearner (subordinate predictive worker)
    └─────────────────── K1 StructuredSemanticLearner (subordinate semantic worker)
          │
          ▼
typed KAdapterInput → typed KAdapterOutput/KAdapterExchange
          │
          ▼
existing planner / Workbench boundary (controlled shadow only)
```

### 2.1 adapter 是唯一 candidate owner

只有 `KContinualAdapter` 可以：

- 绑定 parent checkpoint、owner graph、source/resource manifest；
- 打开或关闭 candidate namespace；
- 接收 K worker manifest；
- 保存输入、输出、dependency projection 和 rollback record；
- 决定 candidate 是否能被 staged、fresh restore、rollback 或最终拒绝。

K1/K2/K3 worker 不得自行创建 candidate namespace、写 parent、提交 rollback 或
接管默认 dispatch。worker 的 digest 变化必须通过 adapter 的 candidate checkpoint
被观察；无法对齐 digest 时 fail-closed。

### 2.2 subordinate worker 的最小职责

- K1 只负责从 typed percept 输入生成 semantic result；不能直接执行 Workbench，
  不能解释 outcome dependency；
- K2 只负责在 typed world/event 上做预测 transition；预测 world 与 observed
  world 必须保留来源字段，不能把 prediction 当成真实 outcome；
- K3 只负责把真实、已审计的 `WorldEvent` 投影为 observed-world/dependency
  witness；不能更新 K1/K2 权重，也不能越过 adapter 改写 parent；
- 现有 planner 继续是 `Goal/ContentPlan → ActionIntent` 的安全 owner，
  Workbench 继续是实际执行和 outcome owner；R6 adapter 只编排并审计它们，
  不重造第二套 planner、executor 或 world store。

## 3. typed input/output 映射

### 3.1 输入

`KAdapterInput` 是唯一进入 K controlled boundary 的 envelope，当前已实现：

- `parent_checkpoint_digest`：同一 parent 的硬约束；
- `observation_digest`：来自 `PerceptEvent`/Workbench observation 的稳定摘要；
- `world_digest`：输入时刻的 `WorldState` 摘要；
- `goal_digest`、`content_plan_digest`：已有 semantic/transition owner 的结构化
  目标和内容摘要；
- `source_manifest_digest`、`episode_id`、`tick`：数据和时间血缘。

原始动态 digest、临时路径、provider 文本、网络 payload、客户端写入结果和
CUDA 状态不能被当作可泛化事实；它们只能作为外部 audit 的受限 metadata。

### 3.2 输出

`KAdapterOutput`/`KAdapterExchange` 是唯一输出边界，当前已实现：

- action digest：对应现有 planner 产生的 `ActionIntent`；
- outcome signature：对应真实 `WorldEvent` 的 canonical outcome；
- dependency/projection digest：对应 K3 projection；
- `success`、scope、input echo、parent echo 和完整 lineage。

typed exchange 通过不等式条件才算可消费：

```text
output.parent == input.parent == adapter.parent
output.input_digest == input.input_digest
output.dependency_digest == bound_projection.dependency_digest
output.projection_digest == bound_projection.projection_digest
scope_id ∈ output.lineage
```

任意条件不满足，adapter 必须在执行或训练前拒绝；不能把执行器返回的失败当作
lineage 校验。

## 4. worker checkpoint 与 owner manifest

worker attachment 不能只保存一个模型 state dict，必须保存下列内容：

```text
KWorkerManifest
  worker_id: k1.semantic | k2.transition | k3.outcome_projection
  worker_kind: semantic | predictive_transition | deterministic_projection
  checkpoint_format / version
  worker_checkpoint_digest
  owner_digest(s)
  source_corpus_or_projection_digest
  input_contract_digest
  output_contract_digest
  parent_checkpoint_digest
  candidate_namespace
  training_steps
  optimizer_state_present (K3 must be false)
```

adapter 的 `owner_graph_digest` 必须由排序后的 worker manifest digest、adapter
contract version 和 parent digest 共同计算，不能由一个手写字符串代替。worker
清单还必须说明：

- K1 的 fact feature masks/readout exclusions；
- K2 的 typed transition input masks；
- K3 的 projection scope/lesion state；
- 所有 worker 的 source split、task/course seed 和资源 manifest。

## 5. checkpoint、训练和 rollback 顺序

### 5.1 训练前

1. fresh restore immutable parent，校验 parent digest；
2. 从真实 K1/K2 checkpoint 构造 worker manifest；K3 从 projector checkpoint 构造
   deterministic manifest；
3. 校验 worker source/corpus/projection digest、contract version 和 owner digest；
4. 组合 owner graph digest，写入 adapter preflight checkpoint；
5. fresh restore adapter + workers，再逐一比较 parent/worker/adapter digest；
6. 只有全部通过，才允许 controlled K canary 申请写 candidate namespace。

当前仓库已完成 parent/adapter/typed exchange 的 preflight，但尚未完成真实
K1/K2 worker manifest attachment；因此不能跳到第 6 步。

### 5.2 candidate training

candidate training 只能发生在 adapter candidate namespace：

- K1/K2 的更新必须由预注册 learner course 驱动，更新数和 replay 预算固定；
- K3 不训练，只消费真实 outcome projection；
- parent worker checkpoint 在 candidate 训练期间只读；
- 每个阶段结束保存 worker + adapter joint checkpoint，并记录 owner graph digest；
- 任何 old-capability、resource、lineage 或 checkpoint Gate 失败，先保存
  failed artifact，再由 adapter rollback 到 parent。

### 5.3 rollback

rollback 的原子边界是整个 owner graph，不是某个 worker：

```text
K1 worker + K2 worker + K3 projection state + KAdapter exchange ledger
                         ↓ rollback
immutable parent owner graph + retained failure artifact/record
```

不能只回滚 K1 而留下 K2 candidate，也不能保留 candidate exchange 在 parent
namespace。已观察的失败 outcome 可以留在 audit/experience ledger，但不能作为
parent 的可执行权重或默认 owner。

## 6. controlled attachment Gate

在任何 K learner training 前，必须先通过一个 single-cell attachment preflight：

- K1/K2 checkpoint 可 fresh restore，owner digest 与 manifest 一致；
- K3 projector checkpoint、scope、lineage 和 lesion 状态一致；
- 三个 worker 的 parent digest、source manifest、contract digest 一致；
- `KAdapterInput → worker outputs → KAdapterExchange` 通过同一 parent echo；
- old S/G holdout retention 在 epsilon 内；
- candidate stage/rollback 后 worker 与 adapter exchange ledger 一致；
- `training_steps` 仍为 0，`default_runtime_attached=false`，CUDA/provider/MCP/
  client/network 均未出现。

这一步是“能安全接上”，不是“能力已经变强”。只有 attachment preflight 通过，
才可以另立 K controlled canary；canary 仍必须沿用 `S → G → K`，并由 R6
addendum 的 new-capability/retention/resource/lesion Gate 判定。

## 7. 明确禁止的捷径

- 不把 K1/K2/K3 standalone formal report 的 digest 当作同一 parent worker
  checkpoint；
- 不把 K3 observed outcome projection 合并进 K2 prediction head 的训练事实；
- 不让 planner 或 Workbench 直接写 K worker state；
- 不用一个 `owner_graph_digest` 常量掩盖 worker 缺失；
- 不因为 typed smoke 16/16 通过就宣布 K learner 已连接或 R6 formal 已解冻；
- 不接 learned router、provider、MCP、客户端写入或 CUDA 隐变量。

## 8. 执行前结论（历史记录）

当前真实结论：typed exchange boundary 已通过 controlled smoke，parent S/G
retention 观测已存在；但真实 K1/K2 worker checkpoint 尚未绑定到 adapter，K
parent retention 仍未闭合，`can_start_r6_formal=false`、`can_promote=false`。

本节结论已由 §9 的 attachment preflight 更新。

## 9. attachment preflight 执行记录（2026-09-09）

已实现并运行：

- `taiji/k_worker_manifest.py`：K1/K2/K3 manifest、合同 digest、同 parent
  bundle 和 owner graph 的 content-addressed contract；
- `taiji/continual_k_adapter.py`：`KWorkerManifestBundle` 原子挂接、joint
  checkpoint/fresh restore、exchange ledger 与 rollback 后保留；
- `scripts/training/eval_taiji_m4v2_r6_k_worker_attachment_preflight.py`：
  真实 artifact restore、K1/K2 typed probe、K3 projector restore、source/
  resource/parent/contract/owner digest 校验、typed exchange、S/G retention
  与 stage/rollback Gate；
- `tests/taiji_native/test_m4v2_r6_k_worker_manifest.py`：manifest/bundle
  tamper、同 parent、adapter restore 及缺失 artifact 回归。

报告：`reports/taiji_m4v2_r6_k_worker_attachment_preflight_20260909.json`。
本机没有可恢复的真实 K1 semantic、K2 transition、K3 outcome projector
artifact，因此结果必须且确实为 `status=artifact_missing`；parent fresh
restore/rollback 机械检查通过，但 `can_start_r6_formal=false`、
`can_promote=false`。本轮没有随机初始化、没有训练、没有把 standalone
formal report 冒充 worker checkpoint，也没有接入 default runtime、provider、
MCP、client 或 CUDA。

## 10. 当前结论与唯一后续动作

attachment contract 已实现，但真实 worker owner graph 仍未形成。下一步只能
建立**可保存、可恢复、带同一 parent/source/resource manifest 的 K1/K2 worker
artifact 生成管线**，并由本 §6 preflight 先验收后才允许任何 controlled K
canary；K3 仍只从真实 projector checkpoint 恢复，不训练。生成 artifact 期间
不得接 default runtime，不得用随机初始化或报告 digest 代替 checkpoint。
