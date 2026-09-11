# M5.K P4.12 预注册：求解器机制下的晋级课程级验证（3 身份 × 3 seed 矩阵）

> 冻结日期：2026-09-11。前置：[P4.11 预注册 §8](M5_K_P4_11_PROJECTION_SOLVER_PREREGISTRATION_20260911.md)（`projection_solver_supported`——P 系列首次两 seed 全门通过）与 [结果复审 §36](M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)。本文把求解器机制从「单身份 × 两 seed」扩展到**3 课程身份 × 3 seed = 9 cell 矩阵**并加资源审计，验证机制鲁棒性；这是求解器机制下的晋级课程级验证。门沿用 P4.7–P4.11 已冻结阈值（零变更）；冻结后按 §7 顺序执行。

## 1. 可证伪假设

P4.11 证明：在单身份批上，「SGD 任务学习 + 末端联合投影」两 seed 全门通过，且 cohort 可行性泛化到全部评估身份。

**假设**：该机制跨**独立课程身份**与**fit-order seed** 鲁棒——3 个全新身份批 × 3 个 seed 的 9 个 cell 中，projected 臂在 ≥ 8/9 cell 上同时通过新任务门与双分布保持门，且基线臂（无投影）在 ≥ 2/3 身份批上复现保持/新任务张力。若 ≤ 7/9，机制鲁棒性不成立，回机制归因。

## 2. 矩阵与 cell 定义

- **矩阵**：3 课程身份批（`p4-12` 身份空间，batch 0/1/2，各自完整的数据 cohort 与 task-seed 空间）× 3 seeds (0,1,2) = **9 cells**；统计单元 = 身份批（n=3），seed 为 replicate；
- 每 cell：`ExtendedGSelectionLearner.from_parent_learner` 出生 → baseline 臂（invariant_fit 8 epochs，无投影）+ projected 臂（**逐位相同轨迹** + 一次末端联合投影）→ 该批 4 个 split 上评估；
- 每 batch 的数据：train / validation / holdout / retention-newtask 各 20、constraint 与 retention-sibling 各 4（P4.4 结构合同逐行同构）；身份与 P4.1–P4.11 全部 manifest 及 P4.9 探针报告隔离；
- **trajectory gate 每 cell 强制**：两臂投影前 head digest 逐位相同（同轨迹确证——投影是唯一变量）。

## 3. 求解器合同（与 P4.11 逐参数相同，按 cell 实例化）

- 惩罚延续 ρ ∈ {1, 10, 100, 1000}，每相 full-batch Adam 6000 步、lr 0.05 余弦衰减至 0.005，warm 延续，无随机重启；
- 约束系统：决策同一性形式（P4.9 探针已证可行）——任务约束（该批 fit-eligible train 集 target argmax + safe-margin）+ 保持约束（该批 constraint cohort 上 frozen parent 决策同一性）；
- anchor = 该 cell 基线臂训练终点的 16 权重维；bias 不动；feature source 不动；
- **收敛判据（冻结）**：逐约束违反 ≤ 1e-6 且总量 ≤ 1e-5 → `projected`；任一 cell `projection_incomplete` → 该 cell 按机械失败记入，不调参重试（若 > 1 cell 不完整则整体 `status=failed` 停止）。

## 4. 门（沿用已冻结值；零变更）

- **逐 cell 新任务门**（holdout）：utility ≥ `0.68 − 1e-9`、target ≥ `0.6 − 1e-9`、safe violations == 0、reobserve projection 通过；
- **逐 cell 保持门**（retention-sibling 与 retention-newtask 双集）：utility / target 非劣于该批 in-run parent、safe == 0、reobserve 通过；
- **聚合门**：projected 臂 ≥ **8/9** cell 全门通过 ∧ 基线臂在 ≥ **2/3** 身份批上张力复现（至少一个 cell 触发新任务或保持门失败）；
- **机械门**：全部 cell 的 checkpoint 独立进程恢复 + tamper 拒绝 + parent 未覆盖 + K1/K2 digest 不变 + feature source 非漂移 + trajectory gate；参数 17/17 精确。

## 5. 资源审计（逐 cell 记录，聚合报告）

- fit wall-clock、投影 wall-clock（分相）、求解器总步数（24,000/cell）、checkpoint 字节、参数 17；
- **资源软门**：projected 臂总 wall-clock（fit + 投影）≤ **3×** 基线臂 wall-clock（逐 cell）；超限 cell 记入失败归因，不影响聚合门（描述性资源边界，正式资源 cap 属晋级课程预注册）。

## 6. 结果映射（全分支；执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `course_level_validation_supported` | 聚合门通过（≥ 8/9 ∧ 基线 ≥ 2/3 张力）；若 8/9 则报告必须含失败 cell 的逐门归因 | **求解器机制课程级验证成立**——固定容量路线在求解器机制下具备晋级课程入场资格；进入求解器机制下的晋级课程预注册（同一 parent 连续 S/G/K 课程，gate 沿用 A8 结构） |
| `course_level_validation_failed` | ≤ 7/9 cell 通过 ∧ 基线张力复现 | 机制鲁棒性不成立——按失败 cell 的身份/seed/门分布归因（哪些约束族、哪些类），回求解器或表示设计 |
| `baseline_drift` | 基线臂 ≥ 2/3 身份批全过 | 张力为数据敏感 → P4.8–P4.11 结论全部重审 |
| 机械失败 | 任一 cell `projection_incomplete` > 1、checkpoint/身份门失败 | `status=failed` + 归因，停止 |

`growth_admitted=false`、`can_promote=false` 贯穿（晋级课程预注册是下一份合同的入口，不直接解冻任何 runtime）；dynamic growth、P5、CUDA、IDE/provider 继续冻结；不加第三臂、不改求解器合同、不读取 sealed。

## 7. 产物顺序

1. `scripts/training/eval_taiji_m5_k_p4_12_course_level_validation.py`（9-cell runner：3 身份批复用 P4.11 数据机械、per-cell 双臂 + 投影 + 资源审计；py_compile/ruff/mypy 先行）；
2. 执行产出 `plans/manifests/taiji_m5_k_p4_12_course_level_validation_manifest_v1.json` + `reports/taiji_m5_k_p4_12_course_level_validation_20260911.json`；
3. 路线图/记录文档同步 + 独立提交。

## 8. 执行记录（运行后补）
