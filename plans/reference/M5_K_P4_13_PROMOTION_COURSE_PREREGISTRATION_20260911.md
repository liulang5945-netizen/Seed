# M5.K P4.13 预注册：求解器机制下的晋级课程（两相继阶段 + 累积约束投影）

> 冻结日期：2026-09-11。前置：[P4.12 预注册 §8](M5_K_P4_12_COURSE_LEVEL_VALIDATION_PREREGISTRATION_20260911.md)（`course_level_validation_supported`——9/9 cell 全门、九格零方差）与 [结果复审 §37](M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)。本文把求解器机制从「单任务 + 保持」扩展到**晋级真正需要的持续累积形态**：同一 parent 上两个相继的新任务阶段（A → B），约束系统逐阶段累积；gate 沿用已冻结阈值 + 资源 cap 以绝对预算定义；rollback 机械门沿用 P3.0 checkpoint 合同。冻结后按 §8 顺序执行。

## 1. 可证伪假设

P4.11/P4.12 证明：单任务 cohort 上「SGD 任务学习 + 末端联合投影」两 seed / 九格全门通过且零方差。

**假设**：该机制支持**持续累积**——阶段 B（新 cohort，从阶段 A 投影态出发）学习后，**累积约束系统**（A 任务约束 + B 任务约束 + 保持约束）的投影能让 **A 与 B 两个阶段的门同时通过**（B 学会、A 不遗忘、保持不破）。若累积系统不可行（投影不收敛）或 B 破坏 A，则持续累积需要约束优先级设计或任务重设计。

## 2. 课程结构（每 cell 内两相继阶段）

- **Phase A**：cohort A（train/validation/holdout-A/retention-newtask 各 20 + constraint/sibling 各 4，身份 `p4-13a`）→ 任务 fit（SGD 8 epochs，P4.10 原样机械）→ **投影 #1**（约束 = A 任务约束 + 保持约束；anchor = A fit 终点）→ 评估 A-holdout + sibling + retention-newtask；
- **Phase B**：cohort B（train/validation/holdout-B，身份 `p4-13b`，与 A 及全部历史隔离）→ 任务 fit **从 A 投影态出发**（同机械）→ **投影 #2**（约束 = **A 任务约束 + B 任务约束 + 保持约束**——累积系统；anchor = B fit 终点）→ 评估 B-holdout + **A-holdout 回检** + sibling + retention-newtask；
- **checkpoint/rollback 机械门**（P3.0 合同）：Phase A 投影态 checkpoint 保存 + 独立进程恢复 + 行为等价；回滚验证 = 从 A checkpoint 恢复的 learner 与 A 投影态行为逐位一致（B 失败时回滚路径可用性的机械证明）。

## 3. 机制合同（与 P4.11/P4.12 逐参数相同）

- 投影求解器：惩罚延续 ρ ∈ {1,10,100,1000}、每相 6000 步 Adam + 余弦 lr、warm 延续、无随机重启；收敛判据逐约束 ≤ 1e-6 / 总量 ≤ 1e-5；投影只作用于 16 权重维、bias 不动；
- 任务 fit：SGD 8 epochs、lr 0.15、order_seed=seed（task MSE one-hot，与 P4.10–P4.12 同机械）；
- 保持约束参考：frozen P3.5 parent 决策同一性（决策同一性形式，探针已证可行）；
- 特征：扩展 16 维 φ，frozen feature source 计算（非漂移，独立字段 + digest 校验）。

## 4. 矩阵、门与资源（绝对预算）

- **矩阵**：3 身份批 × 3 seeds = **9 cells**，每 cell 跑完整 A → B 课程；统计单元 = 身份批（n=3）；
- **逐相门（frozen 阈值零变更）**：
  - 新任务门（A-holdout 与 B-holdout 分别）：utility ≥ `0.68`、target ≥ `0.6`、safe violations == 0、reobserve 通过；
  - 保持门（sibling 与 retention-newtask，Phase B 后双查）：utility / target 非劣于 in-run parent、safe == 0；
  - **向后保持门（P4.13 新增）**：Phase B 后 **A-holdout 仍过新任务门**——A 的能力在 B 学习后不遗忘；
- **聚合门**：projected 臂 ≥ **8/9** cell 上「Phase A 全门 ∧ Phase B 全门（含 A 回检）」∧ 基线臂（无投影）≥ 2/3 批张力复现；
- **资源 cap（绝对预算，冻结）**：每相 fit ≤ `60s`、每相投影 ≤ `120s`、每 cell 课程总 wall ≤ `600s`（P4.12 实测 fit ≈ 0.08s、投影 ≈ 3–6.5s，headroom ≥ 10×）；超限 cell 记失败归因；
- 机械门：checkpoint 独立恢复 + tamper 拒绝 + parent 未覆盖 + K1/K2 digest 不变 + feature source 非漂移 + trajectory gate（每相两臂投影前 digest 逐位相同）+ 参数 17/17。

## 5. 结果映射（全分支；执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `promotion_course_supported` | 聚合门通过 | **求解器机制支持持续累积**——G 侧晋级课程闭合；进入 scorecard 更新与晋级评审（A8 讨论；默认 runtime rollout review 仍是独立门槛） |
| `cumulative_constraint_conflict` | Phase B 投影不收敛（≥ 2 cell）| A+B+保持联合系统不可行——两阶段需求在特征空间冲突；进入约束优先级/任务重设计 |
| `sequential_retention_failure` | 投影收敛 ∧ Phase B 破坏 A-holdout（≥ 2 cell）| 累积投影的 cohort 约束不足以覆盖向后保持——需把 A 保持点显式纳入 Phase B 约束系统 |
| `baseline_drift` | 基线臂 ≥ 2/3 批全过 | 张力数据敏感 → 重审 |
| 机械失败 | checkpoint/身份/结构门失败或孤立投影不收敛 | `status=failed` + 归因，停止 |

`growth_admitted=false`、`can_promote=false` 贯穿（本课程闭合的是 G 侧晋级课程；promotion 评审与默认 runtime 解冻是独立后续）；dynamic growth、P5、CUDA、IDE/provider 继续冻结。

## 6. 边界（诚实声明）

- 本课程是 **G 选择头的晋级课程**（已验证机制的scope）；K worker 的连续学习（P2.6/P2.7 机械）与 G 课程的联合运行属后续预注册，不在本文范围；
- Phase A/B 为同分布（五类压力结构）的不同身份 cohort——检验的是「相继累积 + 无遗忘」，不是跨任务类型泛化；
- 不读取 sealed；不加第三臂；不改求解器合同；不扩到 9-cell 以外的矩阵。

## 7. 与既有教训的对齐

- **P4.12 教训**：资源 cap 以绝对预算定义（本预注册 §4），不得使用比率式软门；
- **P4.8 教训**：逐点约束的泛化边界由评估身份门实测（A-holdout 回检是新增的向后保持维度）；
- **P3.0 教训**：checkpoint/rollback 机械先行（Phase A 投影态的保存/恢复/回滚是 Phase B 的安全网）；
- **P4.11 等价定理**：不加「frozen+δ」第三臂。

## 8. 产物顺序

1. `scripts/training/eval_taiji_m5_k_p4_13_promotion_course.py`（9-cell 两相课程 runner：数据 3 批 × 2 相、双臂、累积投影、资源审计、checkpoint/rollback 机械门；py_compile/ruff/mypy 先行）；
2. 执行产出 `plans/manifests/taiji_m5_k_p4_13_promotion_course_manifest_v1.json` + `reports/taiji_m5_k_p4_13_promotion_course_20260911.json`；
3. 路线图/记录文档同步 + 独立提交。

## 9. 执行记录（运行后补）
