# M5 K 轴 scorecard v4：求解器机制证据收束与晋级边界冻结

> 注册日期：2026-09-11。前置：[P4.13 预注册 §9](M5_K_P4_13_PROMOTION_COURSE_PREREGISTRATION_20260911.md)（`promotion_course_supported`——G 侧晋级课程闭合）与 [结果复审 §38](M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)。v3 scorecard 已收束 K1/K2/K3 standalone 证据线与 C 阶段学习机制结论；本 v4 在不重训、不重算任何 cell 的前提下新增第四条证据线 **`solver_mechanism_evidence`**（P4.12 课程级验证 + P4.13 两相晋级课程，只读转录），并冻结晋级边界。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 目的与唯一变更

- **唯一变更**：新增 `solver_mechanism_evidence` 证据线（来源 = P4.12 + P4.13 报告，只读转录 + digest 校验）；K1/K2/K3 部分与 C 阶段学习机制部分机械复用 v3 reducer（`build_v3` 原样调用），且对 v3 报告记录的 source digest 逐位校验——任何漂移即失败。
- **冻结晋级边界**：G 侧持续学习机制证据闭合 ≠ promotion。`can_promote=false` 不变；晋级评审的**入场条件冻结**为三项（§4），全部满足前不解冻任何 owner、不接默认 runtime。

## 2. 固定输入与内容寻址

- K1/K2/K3 formal reports：与 v2/v3 完全相同的三份；
- C 阶段 formal v2 报告：`reports/taiji_m4v2_c_stage_formal_v2_20260910.json`（v3 的学习机制证据线来源）；
- v3 scorecard 报告：`reports/taiji_m5_k_axis_scorecard_v3_20260910.json`（digest 一致性校验基准）；
- **P4.12 报告**：`reports/taiji_m5_k_p4_12_course_level_validation_20260911.json`（校验 `format=taiji-m5-k-p4-12-course-level-validation-v1`、`status=completed`、`outcome=course_level_validation_supported`、`experiment_passed=true`、`can_promote=false`、`projected_pass_cells=9`、`baseline_tension_batches=3`、`incomplete_projections=0`）；
- **P4.13 报告**：`reports/taiji_m5_k_p4_13_promotion_course_20260911.json`（校验 `format=taiji-m5-k-p4-13-promotion-course-v1`、`status=completed`、`outcome=promotion_course_supported`、`experiment_passed=true`、`can_promote=false`、`projected_pass_cells=9`、`baseline_tension_batches=3`、`incomplete_projection_b_cells=0`）。

## 3. solver_mechanism_evidence 字段（只转录，不重算）

- `source_reports`：P4.12 / P4.13 报告名与 content digest；
- `p4_12_course_level_validation`：outcome、9-cell 矩阵（3 批 × 3 seeds）、projected 通过 9/9、基线张力 3/3 批、零方差声明（九格逐数值相同）、资源软门结构性超限的诚实记录；
- `p4_13_promotion_course`：outcome、两相结构（A → B 累积投影 154 约束）、9/9 全门、**向后保持门零失败**（A-holdout 回检 `0.8/0.75`）、rollback 门 9/9、资源绝对预算全过；
- `mechanism_conclusion`：**保持/新任务解耦 = 表示因子化（P4.9）+ 优化机制替换（P4.11 末端投影）的合取，任一单独不充分；机制表现确定性（零方差）**；
- `solver_attached_default_runtime = false`。

## 4. promotion veto（v3 全部保留 + 晋级入场条件冻结）

v3 全部 veto 原样保留（parent retention 缺失、standalone shadow、default runtime 未附着、无同 parent 连续 S/G/K 证据、无资源/rollback/旧能力门、学习机制候选已选、机制未接默认 runtime）。新增：

- `g_solver_mechanism_course_closed`（= true，本证据线的机器结论——G 侧晋级课程闭合）；
- `k_worker_joint_course_completed`（= false——K worker 联合课程未预注册未运行）；
- `default_runtime_rollout_review_completed`（= false——默认 runtime rollout review 未执行）。

**晋级评审入场条件（冻结）**：上述三项中后两项完成后，A8 晋级评审才有资格召开；评审本身仍须独立批准。`promotion_gate = all(promotion_gates)`、`can_promote = false` 固定。

## 5. 不做的事

- 不重训、重跑或修改任何 formal 判据；不触碰 P4.12/P4.13 的 frozen 语义（只读已落盘报告）；
- 不把 G 选择头证据与 K worker 证据混成单一分数（分域保存）；
- 不把 solver 机制接入默认 runtime；不修改 v3 报告与更早的历史产物。

## 6. 产物与唯一后续动作

- auditor：`scripts/training/audit_taiji_m5_k_axis_scorecard_v4.py`；
- report：`reports/taiji_m5_k_axis_scorecard_v4_20260911.json`；
- 路线图同步 + 独立提交；v3 报告保留为历史、不可覆盖；
- 唯一后续动作：**K worker 联合课程预注册**（P2.6/P2.7 continuation 机械与求解器机制在同一 parent 上的联合运行；预注册冻结前不解冻任何 owner、不接默认 runtime、不训练）。

## 7. 执行记录（2026-09-11，v4 已运行）

1. 机械纪律：`py_compile` / `ruff` / `mypy` 通过后运行 auditor；v3 source digests（K1/K2/K3 + C 阶段学习机制）重算后与 v3 报告逐位一致（漂移校验零触发），K1–K3 与 C 阶段评分段与 v3 逐字段相同。
2. 报告 `reports/taiji_m5_k_axis_scorecard_v4_20260911.json`：6 份 source digests（K1/K2/K3 + `C_STAGE_LEARNING_MECHANISM` + `P4_12_COURSE_LEVEL_VALIDATION` + `P4_13_PROMOTION_COURSE`）；`solver_mechanism_evidence` 完整转录 P4.12（`course_level_validation_supported`、9/9 cells、3/3 批张力、零方差声明）与 P4.13（`promotion_course_supported`、9/9 cells、累积投影 154 约束全格精确零违反、向后保持门零失败、rollback/资源预算全过）。
3. 晋级边界冻结：`promotion_gates` 保留 v3 全部 veto 并新增 `g_solver_mechanism_course_closed=true`（本证据线机器结论）与两个未完成入场条件 `k_worker_joint_course_completed=false`、`default_runtime_rollout_review_completed=false`；`promotion_gate=false`、`can_promote=false`。
4. v3 报告与全部历史产物未覆盖；无 owner 解冻、无默认 runtime 接入、无训练。
