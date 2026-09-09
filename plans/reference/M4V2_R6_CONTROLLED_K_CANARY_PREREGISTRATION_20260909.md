# M4.V2.R6 controlled K canary contract（2026-09-09）

## 1. 目的与边界

本 canary 只验证已通过 attachment Gate 的真实 K1/K2/K3 worker artifact 能否
在同一个 immutable parent 上完成一条真实闭环：

```text
K1 semantic result
      ↓
K2 predicted world / transition result
      ↓
read-only Workbench intent + real execution
      ↓
K3 typed outcome/dependency projection
      ↓
KAdapterInput → KAdapterOutput → KAdapterExchange
      ↓
candidate namespace stage → explicit rollback
```

它不是 K formal、不是 A8 promotion、不是默认 runtime rollout，也不重新训练
worker。K1/K2 只读取已保存 checkpoint；K3 只恢复 deterministic projector。
`can_promote=false` 固定。

## 2. 固定实验对象

- parent：R6 baseline factory，`model_seed=17`，parent digest 必须与 artifact
  manifest 完全一致；
- worker source：已生成的 M5.K2 course artifact，`task_seed=0`、
  `learner_seed=17`；
- workspace：进程私有、仓库可写的临时 Workbench root；
- holdout：M5.K2 的第一条未见三步 episode，只执行首步作为 single-cell
  canary，避免把一次 transport canary误报为 multi-step formal；
- resource：CPU only；不接 provider、MCP、client、network 或 CUDA；
- adapter：必须绑定完整 K1/K2/K3 worker bundle，K3 scope 作为 dependency
  scope，不能使用 fixture-only projection 或 standalone report digest。

## 3. 必须观测的 Gate

1. K1 fresh restore 后返回 `StructuredSemanticResult`；
2. K2 fresh restore 后返回 `StructuredSemanticTransitionResult`；
3. planner 只消费 K1/K2 typed result，产生被 Workbench capability snapshot
   接受的 read-only intent；
4. Workbench 真实执行成功，outcome 从真实执行结果构造；
5. K3 projection 接受同 tick 的真实 `WorldEvent`，并保留 dependency/lineage；
6. `KAdapterInput`、`KAdapterOutput`、`KAdapterExchange` 对 parent、input、
   outcome、dependency、projection、worker manifest 的 digest echo 全部通过；
7. exchange checkpoint fresh restore 后 ledger 与 worker bundle 一致；
8. candidate namespace stage 后 fresh restore，随后 explicit rollback 回到
   parent namespace，worker bundle 与 exchange ledger 不丢失；
9. 同一 parent 的 S/G old-capability 观测 delta 在已校准 epsilon 内；
10. adapter `training_steps=0`，本轮 `candidate_training_performed=false`、
    `candidate_promoted=false`，无默认 runtime side effect。

## 4. 停止线

任一 Gate 失败即 `status=failed`，不调阈值、不重训、不更换 parent、不扩大
episode；保存报告并保持 `can_promote=false`。成功只说明 native K 闭环和原子
rollback 已经可执行，不能证明跨 seed 泛化、长期自进化或 R6 formal 能力。

## 5. 产物与唯一后续动作

- runner：`scripts/training/eval_taiji_m4v2_r6_k_worker_controlled_canary.py`；
- report：`reports/taiji_m4v2_r6_k_worker_controlled_canary_20260909.json`；
- canary 通过后，唯一允许下一步是审阅 single-cell 因果/资源/side-effect
  证据并冻结 R6 formal runner 的输入合同；canary 未通过则回到失败归因，不能
  直接进入 formal。

## 6. 执行记录（2026-09-09）

首轮执行因 Workbench 隔离 root 没有同步覆盖 `get_setting("workspace_path")`
而把真实读取发到了默认 workspace，`typescript_05.ts` 返回 `not_found`；这
是 harness 环境边界错误，不是 worker 能力结果。对照 M5.K2 runner 补齐 root
selector 覆盖后，未改变模型、artifact、阈值或判据，重新执行通过。

最终报告：`reports/taiji_m4v2_r6_k_worker_controlled_canary_20260909.json`。
结果为 `status=passed`：

- K1 `StructuredSemanticResult` 与 K2 `StructuredSemanticTransitionResult` 均
  `resolved`；
- `typescript_05.ts` 经真实 Workbench `workspace.read` 成功，真实 reward 为
  `1.0`；
- K3 接受同 tick outcome，并把 dependency fact 写入 enriched world；
- exchange parent echo、三类 worker manifest lineage、joint checkpoint、
  candidate stage、explicit rollback、S/G retention 全部通过；
- `training_performed=false`、`candidate_training_performed=false`、
  `candidate_promoted=false`，provider/MCP/client/CUDA/default runtime 均未接入。

因此该结果只闭合 single-cell native K 闭环和原子回滚，不证明跨 seed 泛化、
长期自进化或 R6 formal promotion；`can_start_r6_formal=false`、
`can_promote=false` 继续保持。

## 7. 证据审阅与 formal 输入边界（2026-09-09）

审阅结论：

- 因果证据为部分通过：真实 Workbench outcome 被 K3 projection 和 dependency
  lineage 消费，但本 canary 没有 matched control、frozen parent 或 lesion，不能
  证明新增能力的因果增益；
- 资源证据仅为诊断：记录了 `elapsed_seconds` 和 adapter `training_steps=0`，没有
  formal 所需的 peak working set、wall-clock 对照、写入字节或 inference trace；
- 副作用证据在 single-cell 范围通过：default/provider/MCP/client/CUDA 均未接入，
  candidate stage/explicit rollback 保留 typed exchange；但不能代替 all-arm
  9-cell ledger。

上述边界已冻结到 [M4V2_R6_FORMAL_RUNNER_INPUT_CONTRACT_20260909.md](M4V2_R6_FORMAL_RUNNER_INPUT_CONTRACT_20260909.md)。首轮 workspace root 错误按
`course_harness` 归因，不归因为 worker 或模型失败；后续 formal 统一使用显式
parent/worker/course registry 和结构化 failure record。formal 入口仍关闭。
