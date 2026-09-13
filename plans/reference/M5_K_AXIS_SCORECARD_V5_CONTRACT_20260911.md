# M5 K 轴 scorecard v5：K worker 联合课程入账与晋级入场条件收敛

> 注册日期：2026-09-11。前置：[P4.14 预注册 §9](M5_K_P4_14_JOINT_COURSE_PREREGISTRATION_20260911.md)（`joint_course_supported`——4/4 cell 全门通过）。v4 scorecard 已收束 G 侧求解器机制证据线并冻结晋级边界；本 v5 在不重训、不重算任何 cell 的前提下新增第五条证据线 **`joint_course_evidence`**（P4.14 K worker 联合课程，只读转录），并把 `k_worker_joint_course_completed` 置为 `true`。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 目的与唯一变更

- **唯一变更**：新增 `joint_course_evidence` 证据线（来源 = P4.14 报告，只读转录 + digest 校验）；K1/K2/K3、C 阶段学习机制与 G 侧求解器机制部分机械复用 v4 reducer（`build_v4` 原样调用），且对 v4 报告记录的 source digest 逐位校验——任何漂移即失败。
- **入场条件收敛**：`k_worker_joint_course_completed=false → true`；晋级评审两个入场条件仅剩 **`default_runtime_rollout_review_completed=false`**。`can_promote=false` 不变；`promotion_gate = all(promotion_gates)` 机械保持 `false`（其余 veto 原样保留）。

## 2. 固定输入与内容寻址

- K1/K2/K3 formal reports、C 阶段 formal v2 报告、v3/v4 scorecard 报告：与 v4 完全相同的输入（v4 报告为 digest 一致性校验基准）；
- P4.12 / P4.13 报告：与 v4 相同；
- **P4.14 报告**：`reports/taiji_m5_k_p4_14_joint_course_20260911.json`（校验 `format=taiji-m5-k-p4-14-joint-course-v1`、`status=completed`、`outcome=joint_course_supported`、`experiment_passed=true`、`can_promote=false`、`passing_cells=4`、`phase_k_failed_cells=0`、`phase_g_failed_cells=0`、`cross_phase_failed_cells=0`、`incomplete_projection_cells=0`）。

## 3. joint_course_evidence 字段（只转录，不重算）

- `source_report`：P4.14 报告名与 outcome；
- `course_structure`：每 cell 三段——Phase K（P2.6 机械原样：p4-14 新颖 K2 content cohort + 50 条 P2 rehearsal 交错）→ post-K 重 materialization（G cohort 从 post-K worker 实例化）→ Phase G（P4.11 合同原样：任务 fit + hinge + 末端联合投影）；
- `matrix`：2 身份批 × 2 seeds = 4 cells，全部通过；
- `phase_k_summary`：新 K2 content validation 2/2、旧类 K1/K2 content 4/4、安全 abstention 6/6、参数 5,648 不变、K checkpoint 独立恢复；
- `phase_g_summary`：holdout target `0.8` / utility `0.8675` / safe violations `0`；sibling `1.0/1.0`；retention-newtask `0.8/0.8675`；投影 88 约束全格精确零违反；birth 等价精确；
- `cross_phase_gates`：`k_unchanged_after_g` / `g_preservation_vs_frozen_parent` / `birth_equivalence` 全部通过；
- `landscape_shift_note`：post-K K1/K2 digests 相对 pre-K 已移动（交互面真实存在），G 求解器在漂移后景观上维持平衡；post-K worker 跨批逐位相同（K 特征空间不编码路径/项目身份——诚实记录）；
- `k_worker_joint_course_completed`（机器结论）= true。

## 4. promotion veto（v4 全部保留 + 单一门翻转）

v4 全部 veto 原样保留；唯一翻转：`k_worker_joint_course_completed: false → true`。

**晋级评审入场条件（冻结，更新）**：`k_worker_joint_course_completed=true`（已满足）+ `default_runtime_rollout_review_completed`（未满足）——后者完成后 A8 晋级评审才有资格召开；评审本身仍须独立批准。`promotion_gate = all(promotion_gates)` 机械保持 `false`；`can_promote = false` 固定。

## 5. 不做的事

- 不重训、重跑或修改任何 formal 判据；不触碰 P4.12/P4.13/P4.14 的 frozen 语义（只读已落盘报告）；
- 不把 K worker 证据与 G 选择头证据混成单一分数（分域保存）；
- 不把 solver 机制或 K worker 接入默认 runtime；不修改 v4 报告与更早的历史产物；
- 不解冻任何 owner、不读取 sealed。

## 6. 产物与唯一后续动作

- auditor：`scripts/training/audit_taiji_m5_k_axis_scorecard_v5.py`；
- report：`reports/taiji_m5_k_axis_scorecard_v5_20260911.json`；
- 路线图同步 + 独立提交；v4 报告保留为历史、不可覆盖；
- 唯一后续动作：**默认 runtime rollout review 预注册**（晋级评审两入场条件的最后一项；预注册冻结前不解冻任何 owner、不接默认 runtime、不训练）。

## 7. 执行记录（2026-09-11，v5 已运行）

1. 机械纪律：`py_compile` / `ruff` / `mypy --follow-imports=silent` 通过后运行 auditor；v4 source digests（K1/K2/K3 + C 阶段学习机制 + P4.12 + P4.13）重算后与 v4 报告逐位一致（漂移校验零触发）。
2. 报告 `reports/taiji_m5_k_axis_scorecard_v5_20260911.json`：7 份 source digests（v4 全部 + `P4_14_JOINT_COURSE`）；`joint_course_evidence` 完整转录 P4.14（`joint_course_supported`、4/4 cells、Phase K 绝对门 + P2.6 保持门全过、Phase G holdout `0.8/0.8675/0sv`、投影 88 约束全格精确零违反、跨相门三过、post-K 景观漂移诚实记录）。
3. 晋级边界更新：`k_worker_joint_course_completed=false → true`（唯一门翻转）；其余 veto 原样保留；`promotion_gate=false`、`can_promote=false`。晋级评审入场条件仅剩 `default_runtime_rollout_review_completed`。
4. v4 报告与全部历史产物未覆盖；无 owner 解冻、无默认 runtime 接入、无训练。
