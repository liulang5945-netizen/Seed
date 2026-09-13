# M5 K 轴 scorecard v6：默认 runtime rollout review 入账与 A8 入场条件齐备

> 注册日期：2026-09-11。前置：[默认 runtime rollout review 预注册 §9](M5_K_DEFAULT_RUNTIME_ROLLOUT_REVIEW_PREREGISTRATION_20260911.md)（`default_runtime_rollout_review_supported`——4/4 cell 全门通过）与 [scorecard v5 合同](M5_K_AXIS_SCORECARD_V5_CONTRACT_20260911.md)。本 v6 在不重训、不重算任何 cell 的前提下新增第六条证据线 **`rollout_review_evidence`**（默认 runtime rollout review，只读转录 + digest 校验），并把 `default_runtime_rollout_review_completed` 置为 `true`。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 目的与唯一变更

- **唯一变更**：新增 `rollout_review_evidence` 证据线（来源 = review 报告 + attachment manifest，只读转录 + digest 校验）；K1/K2/K3、C 阶段学习机制、G 侧求解器机制与 K worker 联合课程部分机械复用 v5 reducer（`build_v5` 原样调用），且对 v5 报告记录的 source digest 逐位校验——任何漂移即失败。
- **入场条件齐备**：`default_runtime_rollout_review_completed=false → true`（唯一门翻转）。晋级评审（A8）两个入场条件（`k_worker_joint_course_completed=true` + `default_runtime_rollout_review_completed=true`）**全部满足**，A8 评审获得召开资格；评审本身仍须独立批准。
- **晋级边界不变**：v3 继承的全部 veto（`parent_retention_baseline_present=false`、`default_runtime_owner_attached=false`、`same_parent_continual_s_g_k_evidence=false`、`resource_rollback_old_capability_gate=false`、`learning_mechanism_attached_default_runtime=false`）原样保留——因此 `promotion_gate=false`、`can_promote=false` 机械保持。这些 veto 是 A8 评审要逐项裁决的对象，不是本 scorecard 的可翻转载体。

## 2. 固定输入与内容寻址

- K1/K2/K3 formal reports、C 阶段 formal v2 报告、v2/v3/v4/v5 scorecard 报告、P4.12/P4.13/P4.14 报告：与 v5 完全相同的输入（v5 报告为 digest 一致性校验基准）；
- **默认 runtime rollout review 报告**：`reports/taiji_m5_k_default_runtime_rollout_review_20260911.json`（校验 `format=taiji-m5-k-default-runtime-rollout-review-v1`、`status=completed`、`outcome=default_runtime_rollout_review_supported`、`experiment_passed=true`、`can_promote=false`、`growth_admitted=false`、`fit_called=false`、4/4 cell `passes_all`、四组 gate（consumption/restore/mechanical/non_interference）全 true、wrong-cell mixing 被拒绝、runtime inventory 无 K/G 引用）；
- **attachment manifest**：`plans/manifests/taiji_m5_k_default_runtime_rollout_attachment_v1.json`（`format=taiji-default-runtime-rollout-attachment-v1`，`manifest_digest` 与 review 报告记录一致）。

## 3. rollout_review_evidence 字段（只转录，不重算）

- `source_report` / `attachment_manifest` / `attachment_manifest_digest`；
- `consumption_contract`：内容寻址逐 cell artifact digest（钉自 P4.14 报告）、P4.14 lineage、嵌入安全不变量（confidence floor、selection margin、参数计数）、fail-closed load 顺序、禁止项（fit / 改写 artifact / override 不变量 / 重建 checkpoint）；
- `runtime_baseline`：`api/seed_runtime.py` 现状只读盘点——零 K/G 引用、`checkpoints/seed_corpus.pt` sha256 前后一致；
- `matrix`：4/4 P4.14 cell 全部执行全部通过，无子采样；
- `consumption_equivalence`：Phase K（novel 2/2、旧类 4/4、安全 6/6）与 Phase G（validation/holdout/retention-newtask `0.8/0.8675/0sv`、retention-sibling `1.0/1.0`）在**磁盘加载消费路径**上逐字段复现 P4.14 记录值；
- `mechanical_summary`：digest/不变量、篡改拒绝、wrong-cell 混装拒绝、独立进程恢复（K 经 pilot verifier、G 经 P4.13 extended verifier）、行为 rollback、G 评估后 K worker digest 不变、非干扰、cohort 重物化 digest 与 P4.14 manifest 一致、绝对资源预算全过（cell ~2.0–2.1s ≪ 120s、总 267s ≪ 600s）；
- `fit_called=false`、`training_performed=false`；
- `default_runtime_rollout_review_completed`（机器结论）= true。

## 4. promotion veto（v5 全部保留 + 单一门翻转）

v5 全部 veto 原样保留；唯一翻转：`default_runtime_rollout_review_completed: false → true`。

**A8 晋级评审状态（冻结，更新）**：两个入场条件齐备——A8 评审**有资格召开**；评审须独立批准，并逐项裁决其余 veto（默认 runtime owner 附着、资源/rollback/旧能力门、同 parent learned S/G/K 证据、parent retention baseline）。`promotion_gate = all(promotion_gates)` 机械保持 `false`；`can_promote = false` 固定。

## 5. 不做的事

- 不重训、重跑或修改任何 formal 判据；不触碰 review 报告与 P4.14 的 frozen 语义（只读已落盘报告）；
- 不把 K/G 机制接入默认 runtime、不修改 `api/`/`seed_platform/` 产品代码、不解冻任何 owner；
- 不把 rollout review 的消费等价写成默认 runtime 已采用该机制（那是 A8 之后的工程与审批决策）；
- 不修改 v5 报告与更早的历史产物。

## 6. 产物与唯一后续动作

- auditor：`scripts/training/audit_taiji_m5_k_axis_scorecard_v6.py`；
- report：`reports/taiji_m5_k_axis_scorecard_v6_20260911.json`；
- 路线图同步 + 独立提交；v5 报告保留为历史、不可覆盖；
- 唯一后续动作：**A8 晋级评审**（入场条件已齐备；评审独立批准，逐项裁决其余 veto；评审召开前不训练、不接默认 runtime、不解冻 owner）。

## 7. 执行记录（2026-09-11，v6 已运行）

1. 机械纪律：`py_compile` / `ruff` / `black` / `mypy --follow-imports=silent` 通过后运行 auditor；v5 source digests（K1/K2/K3 + C 阶段学习机制 + P4.12 + P4.13 + P4.14）重算后与 v5 报告逐位一致（漂移校验零触发）。
2. 报告 `reports/taiji_m5_k_axis_scorecard_v6_20260911.json`：8 份 source digests（v5 全部 + `DEFAULT_RUNTIME_ROLLOUT_REVIEW`）；`rollout_review_evidence` 完整转录 review 报告（`default_runtime_rollout_review_supported`、4/4 cell、消费等价逐字段复现、机械/恢复/非干扰全过、`fit_called=false`）。
3. 晋级边界更新：`default_runtime_rollout_review_completed=false → true`（唯一门翻转）；其余 veto 原样保留；`promotion_gate=false`、`can_promote=false`。A8 晋级评审入场条件齐备。
4. v5 报告与全部历史产物未覆盖；无 owner 解冻、无默认 runtime 接入、无训练。
