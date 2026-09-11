# M5.K P4.9 预注册：特征空间重设计（parent-relative margin 特征因子化）

> 冻结日期：2026-09-11。前置：[P4.8 预注册 §9](M5_K_P4_8_REPRESENTATION_CONTRACT_REDESIGN_PREREGISTRATION_20260911.md)（`invariant_constraint_insufficient`——约束形式/容量/架构三维度均排除为根因）与 [结果复审 §34](M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)。设计探针 [P4.9 probe](../../scripts/training/eval_taiji_m5_k_p4_9_feature_space_probe.py)（报告 `reports/taiji_m5_k_p4_9_feature_space_probe_20260911.json`，frozen validation-only、无任何 Taiji fit）已先行测量因子化可行性，本文据其结果冻结正式实验的臂、门与映射；执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 探针测量结果（设计输入，2026-09-11）

- **M1 目标秩结构**：14 个 fit-eligible 新任务集上，behavior target 在 frozen parent 排序下的位置为 {rank 0: 8, rank 2: 6}——新任务信号 = 「把 parent 排第 3 位的候选提为首位」；utility gap（target − parent pick）mean `0.386`、max `0.9`。
- **M2 冲突几何**：flip-needed 均值方向与 parent 权重方向的平方余弦 = `0.00036`（近正交）——冲突不是正面顶撞，而是基空间内两约束系统的不可同时满足。
- **M3 联合可行性（决定性）**：在「新任务 target 全部胜出 + sibling 上 parent 决策完整保持」的联合约束系统上：
  - **基空间（12 维）不可行**：最小联合违反 `0.0494`（newtask-argmax 族 `0.0413`、sibling-argmax 族 `0.0081`）——不存在任何 12 维线性打分器同时满足两侧；P4.2–P4.8 的全部失败由此得到定量解释；
  - **扩展空间（16 维 = 12 基 + 4 个 frozen-parent-relative 维度：parent_argmax_margin / parent_safe_margin / parent_rank_norm / is_parent_pick）可行**：最小联合违反精确 `0.0`——存在线性打分器同时满足两侧。
- **判定**：`parent_relative_features_are_the_factorization`——parent-relative margin 特征正是缺失的因子化维度。两阶段纪律不受污染：正式门沿用 P4.7/P4.8 已冻结阈值，探针只影响臂设计。

## 2. 可证伪假设

扩展空间中存在满足双侧约束的线性解（探针已证存在性）；**假设**：从 parent 初始化出发、在相同交错协议（task + margin-preservation hinge）下，SGD 训练能够到达该可行解——即特征因子化 + 有限支撑约束足以在训练动力学层面解耦保持与新任务。若训练仍失败，则瓶颈从「表示存在性」转为「优化动力学」，重设计转向约束求解器方向。

## 3. 臂（两臂，相邻对单变量）

| 臂 | 特征空间 | 参数 | 保持约束 | 单变量 |
|---|---|---|---|---|
| `invariant-base-13` | 基 12 维 | 13 | margin-preservation hinge（P4.8 原样） | —（in-run 基线，第 2 次复现） |
| `invariant-ext-17` | **扩展 16 维**（12 基 + 4 个 parent-relative） | 17（16 权重 + bias） | margin-preservation hinge（同一 canonical 函数） | vs 基线：仅特征空间 |

- **扩展 learner**（`taiji/g_selection_extended.py` 的 `ExtendedGSelectionLearner`）：单个 `nn.Linear(16, 1, bias=True)`；base-12 权重与 bias 从 P3.5 parent **逐位继承**、4 个 parent-relative 维度**零初始化**——出生分数 = parent 分数精确（新维度贡献为零），出生等价门照旧。特征 φ 由**内部 frozen parent 副本**计算（特征在训练全程是非漂移的环境常量）；select/fit 时按集合在线计算 φ。
- parent-relative 维度定义为（记 s_p = frozen parent 对候选打分、safe 为同一 `_safe_candidate`）：`parent_argmax_margin = s_p(x) − max_{c≠x} s_p(c)`、`parent_safe_margin = s_p(x) − s_p(safe)`、`parent_rank_norm = 1 − rank_p(x)/(n−1)`（rank 按 parent 分数 + 决策规则同序 tiebreak）、`is_parent_pick = 1{x = parent argmax}`。
- 两臂共享：同一 train fit 集、同一 constraint cohort（P4.4 结构合同、全新身份）、同一交错协议（epochs=8、lr=0.15、order_seed=seed、`training_steps += 2`）、两 deterministic seeds (0,1)、同一 hinge 参考源（frozen P3.5 parent 的决策与边际——canonical `margin_preservation_hinge`）。

## 4. 数据与身份

- 正式实验身份空间 `p4-10`（探针占用 `p4-9`）；train / validation / holdout / retention-newtask 各 20、constraint 与 retention-sibling 各 4（P4.4 结构合同逐行同构），与 P4.1–P4.9 全部 manifest 隔离；constraint cohort 的 behavior target/utility 不进入 fit；retention 不进入 fit；不读取 sealed。

## 5. 门（沿用已冻结值；探针未触碰任何门）

- 新任务门（holdout）：utility ≥ `0.68 − 1e-9`、target ≥ `0.6 − 1e-9`、safe violations == 0、reobserve projection 通过；
- 保持门（retention-sibling 与 retention-newtask 双集）：utility / target 非劣于 in-run parent、safe == 0、reobserve 通过；
- 机械门：两臂两 seed 零步/训练后 checkpoint 独立进程恢复 + tamper 拒绝 + parent 未覆盖 + K1/K2 digest 不变；ext 臂出生等价精确（0 mismatch、最大偏差 0.0）+ 出生 hinge 损失恒 0 + **特征源 frozen 副本训练前后 digest 不变**；参数 13/17 精确。

## 6. 结果映射（全分支；执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `feature_factorization_supported` | ext 臂两 seed 同时过新任务 + 双保持门 ∧ base 臂张力复现 | **特征因子化假设成立**——固定容量路线在新表示下重开；进入新表示下的晋级课程级验证预注册 |
| `learnability_gap` | ext 臂 ≥1 seed 仍互斥 ∧ base 臂张力复现 | 可行解存在但交错 SGD 不可达——瓶颈转为优化动力学；重设计转向直接约束求解（可行解构造/约束投影更新） |
| `baseline_drift` | base 臂两 seed 全过 | 互斥为数据敏感 → 全部结论重审 |
| 机械/身份失败 | 任一 | `status=failed` + 归因，停止 |

`growth_admitted=false`、`can_promote=false` 贯穿；dynamic growth、P5、CUDA、IDE/provider 继续冻结；不加第三臂（P4.8 等价定理已证明「frozen+δ 共享 lr」是空转臂，不再纳入）、不改 task fit 形式、不扩到 9-cell。

## 7. 产物顺序

1. `taiji/g_selection_extended.py`（ExtendedGSelectionLearner：16 维 φ + 继承初始化 + canonical hinge fit）+ 定向测试（出生等价精确/φ 非漂移性/hinge 委托/往返 tamper/父代容量拒绝）；
2. `scripts/training/eval_taiji_m5_k_p4_10_feature_factorization.py`（两臂 runner，py_compile/ruff/mypy 先行）；
3. 执行产出 `plans/manifests/taiji_m5_k_p4_10_feature_factorization_manifest_v1.json` + `reports/taiji_m5_k_p4_10_feature_factorization_20260911.json`；
4. 路线图/记录文档同步 + 独立提交。

## 8. 执行记录（运行后补）
