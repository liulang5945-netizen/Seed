# M5 K 轴 scorecard v3：学习机制收束入账与晋级边界冻结

> 注册日期：2026-09-10。前置：[C 阶段 formal v2 预注册 §7](M4V2_C_STAGE_FORMAL_V2_PREREGISTRATION_20260910.md)（四门全过，FS 收束为 K 相位默认学习机制候选）。v2 scorecard 已闭合 K1/K2/K3 standalone 证据线；本 v3 在不重训、不重算任何 cell 的前提下，把 C 阶段学习机制结论作为第四条证据线收束入账，并冻结「FS 候选 → 晋级」的边界条件。

## 1. 目的与唯一变更

- **唯一变更**：新增 `learning_mechanism_evidence` 证据线（来源 = C 阶段 formal v2 报告，只读转录）；K1/K2/K3 部分机械复用 v2 reducer（`build_scorecard` 原样调用），且三份 source digest 必须对 v2 报告记录值逐位校验——任何漂移即失败，不静默接受。
- **冻结晋级边界**：learning mechanism candidate ≠ promotion。FS 的「候选」地位是 C 阶段 §5 冻结映射的如实转写；晋级仍被 v2 的全部 runtime/parent 边界门挡住，另加机制附着门。v3 不解冻任何 owner。

## 2. 固定输入与内容寻址

- K1/K2/K3 formal reports：与 v2 完全相同的三份（`taiji_m5_k1_skill_composition_formal_20260909.json` / `taiji_m5_k2_multistep_formal_20260909.json` / `taiji_m5_k3_outcome_dependency_formal_20260909.json`）；
- C 阶段 formal v2 报告：`reports/taiji_m4v2_c_stage_formal_v2_20260910.json`（校验 `format=taiji-m4v2-c-stage-formal-v1`、`status=passed`、`verdict.formal_passed=true`、`can_promote=false`、G1–G4 全 passed、sealed v5 sha256 `c7680ad5…34b5d`、task_seed=151）；
- v2 scorecard 报告：`reports/taiji_m5_k_axis_scorecard_v2_20260909.json`（仅作 digest 一致性校验基准，不参与评分）。

## 3. learning_mechanism_evidence 字段（只转录，不重算、不新增统计）

- `source_report` / `formal_status` / `sealed_task_seed` / `sealed_sha256` / `epsilon_cat` / `epsilon_ni`；
- `gates_passed`：G1（v2 sealed 口径）/ G2 / G3 / G4 逐门布尔；
- `fs_mechanism_status = "default-learning-mechanism-candidate"`，来源注记 = 预注册 §5 冻结结果映射；
- `descriptive`：sealed 课程均值（C/FS）、弱类课程均值（D/R）、FS 弱类胜出课程列表——全部直接取自 formal 报告 `course_aggregation`；
- `fs_attached_default_runtime = false`。

## 4. promotion veto（v2 全部保留 + 新增两项）

v2 的五项 veto 原样保留（parent retention 缺失、standalone shadow、default runtime 未附着、无同 parent 连续 S/G/K、无资源/rollback/旧能力门）。新增：

- `learning_mechanism_candidate_selected`（= true，唯一新增的「通过」项——C 阶段 formal v2 的机器结论）；
- `learning_mechanism_attached_default_runtime`（= false，FS 未接入默认 runtime）。

`promotion_gate = all(promotion_gates)`、`can_promote = false` 固定。

## 5. 不做的事

- 不重训、重跑或修改 K1/K2/K3 formal 判据；不触碰 C 阶段 sealed 语义（只读已落盘报告）；
- 不把 MSE delta 与 success ratio 混合成单一分数（C 阶段量尺是 sealed MSE delta，与 K1–K3 的 success ratio 分域保存）；
- 不把 FS 接入默认 runtime；不修改 v2 报告与更早的历史产物。

## 6. 产物与唯一后续动作

- auditor：`scripts/training/audit_taiji_m5_k_axis_scorecard_v3.py`；
- report：`reports/taiji_m5_k_axis_scorecard_v3_20260910.json`；
- 路线图同步 + 独立提交；v2 报告保留为历史、不可覆盖；
- 唯一后续动作：晋级课程预注册——同一 parent 的连续 S/G/K 课程，学习机制 = FS fast/slow+replay 候选，含资源等价、rollback、旧能力非劣门；该预注册冻结前不解冻任何 K shadow owner、不接默认 runtime、不训练。

## 7. 执行记录（2026-09-10，v3 已运行）

1. 机械纪律：`py_compile` / `ruff check` 通过后运行 `audit_taiji_m5_k_axis_scorecard_v3.py`；K1/K2/K3 三份 source digest 重算后与 v2 报告记录值逐位一致（漂移校验零触发），K1–K3 评分段与 v2 逐字段相同。
2. 报告 `reports/taiji_m5_k_axis_scorecard_v3_20260910.json`：4 份 source digest（K1/K2/K3 + `C_STAGE_LEARNING_MECHANISM` = `190202fe…cc80`）；`learning_mechanism_evidence` 完整转录 C 阶段 formal v2（status=passed、sealed task_seed=151、G1–G4 全 true、FS 弱类 3/3 胜出、epsilon_cat=0.01 / epsilon_ni=0.003846）。
3. 晋级边界冻结：`promotion_gates` 保留 v2 全部 veto 并新增 `learning_mechanism_candidate_selected=true`（唯一通过项）与 `learning_mechanism_attached_default_runtime=false`；`k_evidence_closed=true`、`learning_mechanism_closed=true`、`promotion_gate=false`、`can_promote=false`。
4. v2 报告与全部历史产物未覆盖；无 owner 解冻、无默认 runtime 接入、无训练。
