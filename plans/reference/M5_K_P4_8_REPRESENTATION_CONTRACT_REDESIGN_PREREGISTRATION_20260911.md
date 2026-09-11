# M5.K P4.8 预注册：表示合同重设计（决策不变量保持 + 残差架构）

> 冻结日期：2026-09-11。前置：[P4.7 预注册 §8](M5_K_P4_7_CAPACITY_CLEAN_TEST_PREREGISTRATION_20260911.md)（容量假设关闭——互斥不随容量消失）与 [结果复审 §32–§33](M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)（互斥定性为表示耦合问题：functional teacher 把「新任务学习方向」与「parent 行为保持」耦合进同一 12 维 candidate 特征空间）。P4 验收问题不变（固定容量持续学习须同时满足保持与新任务），关闭的是旧表示合同；本文冻结新表示合同的两步单变量检验。冻结后按 §8 顺序执行。

## 1. 可证伪假设

P4.6/P4.7 的保持约束是**标量 MSE teacher 匹配**：在 constraint cohort 上把 student 的逐 candidate 分数拉向 parent 的逐 candidate 分数。该约束无饱和点（MSE 梯度恒在），与 task delta 在同一组权重上持续对抗——这是 seed 间互斥的机制根源。

**假设**：把保持约束从「拟合 parent 标量分」改为「保持 parent 的**决策不变量**」（hinge 形式，有限支撑——决策区域被保持后梯度恒零），互斥即被解耦：student 在 parent 分布上只需保持 parent 的选择（含安全回退边界），分数尺度自由留给新任务学习。进一步地，把「parent 函数」与「适应函数」架构拆分（frozen parent head + 零初始化 δ head）使保持约束只需约束 δ 的序变，降低冲突面。

## 2. 三臂（相邻对单变量）

| 臂 | 架构 | 可训练层 | 保持约束 | 单变量 |
|---|---|---|---|---|
| `functional-13` | 13 参数继承可训练（P4.6 复现） | 全部 13 | 标量 MSE teacher（P4.6 原样） | —（in-run 基线，第 3 次复现） |
| `invariant-13` | 13 参数继承可训练（同上） | 全部 13 | **决策不变量 hinge**（§3） | vs functional-13：仅约束形式 |
| `residual-26` | frozen parent head（13，逐位继承、永不可训练）+ δ head（13，零初始化） | 仅 δ 13 | 决策不变量 hinge（作用于总分，梯度只写 δ） | vs invariant-13：仅架构拆分 |

- 三臂共享：同一 train fit 集、同一 constraint cohort、同一交错协议（每步 task delta 后接一个 constraint 步，`training_steps += 2`）、epochs=8、lr=0.15、order_seed=seed、两 deterministic seeds (0,1)。
- `residual-26` 参数核算：26 = 13 frozen + 13 trainable；**有效可训练容量 13**，与另两臂同——架构是变量，容量不是（P4.7 已排除容量）。
- task fit：`functional-13`/`invariant-13` 对总分拟合 one-hot（P4.6 原样）；`residual-26` 对总分拟合 one-hot 但 delta 只写 δ head（总分 = parent head + δ head）。

## 3. 决策不变量 hinge（冻结规格）

对每个 constraint-cohort record（parent 决策由 frozen parent 的 `select` 规则在线计算，**不读 behavior target/utility**）：

- 记 student 总分为 s(·)，parent 决策为 c*，safe 候选为 safe（同一 `_safe_candidate` tiebreak，输入属性非学习量），guard band `δ_m = 0.01`，selection margin `m = 0.05`（learner 现值）。
- **parent 选了 proposal p\***（选择规则要求 p\* 为 argmax 且超出 safe-margin）：
  `L = max(0, δ_m − (s(p*) − max_{c≠p*} s(c))) + max(0, (m + δ_m) − (s(p*) − s(safe)))`
- **parent 选了 safe s\***（选择规则要求无 proposal 越过 safe-margin）：
  `L = Σ_{proposal p} max(0, s(p) − s(s*) − (m − δ_m))`
- 实现：`error_i = ∂L/∂s(c_i)`（激活项取 ±1），经 `apply_linear_delta` 写入可训练层（方向已对源码核实：`weight -= rate · errorᵀ @ inputs`，推高分数用负 error）。hinge 满足后 error 恒零——**有限支撑是与 MSE 的本质区别**。
- 出生一致性：三臂在零 fit 时 constraint 损失恒为 0（parent 决策天然处于保持区域内）——作为机械门断言。

## 4. 数据与身份（P4.7 harness 复用，全新身份）

- train / validation / holdout / retention-newtask：各 20 records（5 类 × 宽度 2/4/8/12）；constraint cohort 与 retention-sibling：各 4 records（P4.4 结构合同逐行同构）；
- 全部 project/path/candidate/behavior digest 与 P4.1–P4.7 manifest 隔离（`p40_p48_` 路径前缀、task seeds 48000–48500）；
- constraint cohort 的 behavior target/utility 不进入 fit；retention 不进入 fit；不读取任何 sealed payload。

## 5. 门（全部在执行前冻结）

- **新任务门**（逐臂逐 seed，holdout）：utility ≥ `0.68 − 1e-9`、target hit ≥ `0.6 − 1e-9`、safe violations == 0、reobserve projection 通过（P4.3/P4.5/P4.6/P4.7 四轮稳定的 frozen new-only 基线）；
- **保持门**（逐臂逐 seed，retention-sibling 与 retention-newtask **双集都必须**）：utility / target hit 非劣于 in-run parent 同 split 实测、safe violations == 0、reobserve projection 通过；
- **机械门**：三臂两 seed 零步/训练后 checkpoint 保存 + 独立进程恢复 + tamper 拒绝 + parent 未覆盖 + K1/K2 digest 不变；`residual-26` 另加：出生等价精确（0 mismatch、最大偏差 0.0）+ **frozen head 训练前后 digest 逐位不变**；出生 constraint 损失 == 0 断言；
- 参数计数冻结：13 / 13 / 26（13 frozen + 13 trainable）。

## 6. 结果映射（全分支；执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `representation_redesign_supported` | 任一新臂两 seed 同时过新任务门 + 双保持门 ∧ `functional-13` 张力复现 | **表示合同重设计成立**。active ingredient 判定：仅 `residual-26` 过 → 架构拆分是关键；两新臂都过 → 约束形式已足够。→ 进入晋级课程级验证预注册（固定容量保持+新任务基线首次成立，P4 结构成长问题以新表示合同重开） |
| `invariant_constraint_insufficient` | 两新臂在 ≥1 seed 上仍互斥 ∧ `functional-13` 张力复现 | hinge 不足以解耦 → 12 维特征空间本身无法同时表示两种排序 → 特征空间重设计决策点（新特征维度/非线性感扣），固定容量路线在现表示下关闭 |
| `baseline_drift` | `functional-13` 两 seed 全过（第 4 轮不复现张力） | 互斥为数据敏感 → 容量/表示结论全部重审 |
| 机械/身份/checkpoint 失败 | 任一 | `status=failed` + 归因，停止 |

`growth_admitted=false`、`can_promote=false` 贯穿所有分支；dynamic growth、P5、CUDA、IDE/provider 继续冻结；不加第四臂、不改 task fit 形式、不扩到 9-cell。

## 7. 与既有关键教训的对齐

- **P4.4 教训**：保持验收双分布（sibling = 保持分布为主门，newtask = 新任务分布自身的灾难检查）；
- **P4.7 教训**：扩展算子出生零影响必须实证（residual 的 δ=0 出生等价是同一标准）；容量不是变量（有效可训练容量三臂同为 13）；
- **P4.6 教训**：constraint cohort 只读输入特征与 parent 在线决策，behavior target/utility 永不进入 fit；
- **R5 教训**：不引入 router/任务 ID——三臂均无任务条件化，保持与新任务的区分完全由数据分布与约束形式承担。

## 8. 产物顺序

1. `taiji/g_selection_residual.py`（ResidualGSelectionLearner：frozen head + δ head、决策不变量 hinge fit、checkpoint/独立恢复）+ 定向测试（出生等价精确、hinge 方向与有限支撑、frozen head 不可变、往返/tamper、决策保持语义）；
2. `scripts/training/eval_taiji_m5_k_p4_8_representation_redesign.py`（三臂 runner，py_compile/ruff/mypy 先行）；
3. 执行产出 `plans/manifests/taiji_m5_k_p4_8_representation_redesign_manifest_v1.json` + `reports/taiji_m5_k_p4_8_representation_redesign_20260911.json`；
4. 路线图/记录文档同步 + 独立提交。

## 9. 执行记录（运行后补）
