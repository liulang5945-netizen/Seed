# M5.K P4.11 预注册：可行区域投影求解器（约束求解器更新机制）

> 冻结日期：2026-09-11。前置：[P4.9 预注册 §8](M5_K_P4_9_FEATURE_SPACE_REDESIGN_PREREGISTRATION_20260911.md)（`learnability_gap`——扩展 16 维空间存在精确可行解但交错 SGD 不可达）与 [结果复审 §35](M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)。本文把 P4.9 探针的联合违反最小化机械升级为**可行区域投影求解器**，并将其收敛判据、投影频率合同与正式实验设计一并冻结；冻结后按 §7 顺序执行。

## 1. 可证伪假设

P4.10 证明：扩展空间存在精确可行解（联合违反 0.0），但交错 per-set SGD 从 parent 初始化不可达（两 seed 均无法同时过双侧门）；hinge 仅在 12–13/112 步激活且只作用于 cohort 点。

**假设**：若在 P4.10 的训练终点之上施加**一次联合可行区域投影**——最小化到该终点权重的距离 subject to 联合约束系统（新任务约束 + 保持约束）——则投影后的 learner 将同时通过新任务门与双分布保持门。即：瓶颈确实在优化路径而非解的存在性或泛化性。若投影完成（违反达标）但门仍失败，则「cohort 可行 ≠ 评估身份泛化」的墙在精确可行解下依然成立，属于任务/表示设计层面的更深问题。

## 2. 投影求解器合同（冻结）

- **约束系统**（与 P4.9 探针 M3 同构，按本轮数据实例化）：
  - 任务约束：每个 fit-eligible train set 上 target 严格 argmax（`w·(x_t − x_c) ≥ 1e-6`）；target 为 proposal 时另加 safe-margin（`w·(x_t − x_safe) ≥ 0.05 + 1e-9`）；
  - 保持约束：每个 constraint-cohort set（P4.4 结构合同、全新身份）上 frozen parent 决策同一性——parent 选 proposal 时 argmax（`≥ 1e-6`）+ safe-margin（`≥ 0.05 + 1e-9`）；parent 选 safe 时所有 proposal 侵入 ≤ `0.05 − 1e-9`；
  - 全部约束为分数差（bias 消去）→ **投影只作用于 16 个权重维，bias 不受约束且保持训练终点值**。

### 2.1 修订（2026-09-11，执行前——未实现求解器、未 materialize 数据）

原 §2 把保持约束写为「边际保持形式」（b = parent 自身边际）。实现前核查发现：P4.9 探针 M3 建立可行性的保持约束是**决策同一性形式**（b = 固定阈值 ε/0.05）；边际保持形式与任务约束的联合可行性从未被测量（它严格更强——parent 边际可能薄于其决策阈值所需的富余），直接实现会冒无谓的 `projection_incomplete` 风险。修订为**决策同一性形式（探针已证可行）**；边际保持语义仅保留在基线臂的 SGD hinge 步中（P4.10 原样，不受影响）。假设、门、映射、臂数零变更——投影假设「从 P4.10 端点到达可行区域」以探针已证的那个可行区域为定义域才是良构的。
- **特征空间**：扩展 16 维 φ（12 基 + 4 个 frozen-parent-relative 维度），特征由 feature source（frozen、digest 守护）计算——与 P4.10 完全一致。
- **求解器**：惩罚延续法（无 scipy 环境的确定性替代）——
  - 目标：`F_ρ(w) = ρ · Σ_i max(0, b_i − a_i·w) + ½·||w − w_anchor||²`，anchor = 基线臂训练终点权重（投影目标，全程固定）；
  - ρ 延续序列 `{1, 10, 100, 1000}`，每相 full-batch Adam 6000 步、lr `0.05` 余弦衰减至 `0.005`， warm start 自上一相解（第一相 warm start = anchor）；
  - 无随机重启（warm 延续确定性）；
  - **收敛判据（冻结）**：最终解逐约束违反 ≤ `1e-6` 且总违反 ≤ `1e-5` → `projected`；否则 `projection_incomplete` → 实验按机械失败停止（诚实停止，不调参重试）。
- **投影频率（冻结）**：**每次训练恰一次末端投影**——基线臂完整跑完 P4.10 协议（交错 task+hinge SGD，8 epochs）后，对其终点权重施加一次投影。不做逐步/逐 epoch 投影（P4.10 已证逐点机制不足；末端单次投影是对「到达可行解」的最小干预）。
- **距离审计**：报告 `||w* − anchor||` 的 L1/L2/∞ 与逐维位移——投影移动量的完整诊断。

## 3. 臂（两臂，单变量 = 末端投影）

| 臂 | 训练 | 末端投影 | 单变量 |
|---|---|---|---|
| `invariant-ext-17`（P4.10 复现基线） | 交错 task+hinge SGD，8 epochs | 无 | —（in-run 基线，第 2 次复现） |
| `projected-ext-17` | **与基线臂逐位相同的轨迹**（同 seed、同数据、同 hinge 步） | **一次联合投影**（§2 求解器） | vs 基线：仅末端投影 |

- 两臂共享：同一 train fit 集、同一 constraint cohort、同一协议超参、两 deterministic seeds (0,1)、同一特征源（frozen parent 副本）。
- 投影从基线臂**同 seed 的训练终点**出发——直接回答「P4.10 的失败端点能否被一次投影修复」。

## 4. 数据与身份

- 身份空间 `p4-11`（`p40_p411_` 路径前缀、task seeds 51000–51500）；train / validation / holdout / retention-newtask 各 20、constraint 与 retention-sibling 各 4（P4.4 结构合同逐行同构）；与 P4.1–P4.10 全部 manifest 及 P4.9 探针报告 digest 隔离；constraint cohort 的 behavior target/utility 不进入任何 fit 或求解器（求解器只读候选特征与 frozen parent 决策/边际）；retention 不进入 fit；不读取 sealed。

## 5. 门（沿用已冻结值）

- 新任务门（holdout）：utility ≥ `0.68 − 1e-9`、target ≥ `0.6 − 1e-9`、safe violations == 0、reobserve projection 通过；
- 保持门（retention-sibling 与 retention-newtask 双集）：utility / target 非劣于 in-run parent、safe == 0、reobserve 通过；
- 机械门：两臂两 seed checkpoint 独立进程恢复 + tamper 拒绝 + parent 未覆盖 + K1/K2 digest 不变 + feature source 非漂移；出生等价精确；参数 13/17 精确；投影收敛判据达标（§2）。

## 6. 结果映射（全分支；执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `projection_solver_supported` | 投影完成 ∧ projected 臂两 seed 同时过新任务 + 双保持门 ∧ 基线臂张力复现 | **求解器更新机制成立**——固定容量路线在「SGD 任务学习 + 可行区域投影」下重开；进入该机制下的晋级课程级验证预注册 |
| `projection_generalization_gap` | 投影完成 ∧ projected 臂 ≥1 seed 仍互斥 ∧ 基线臂张力复现 | **cohort 可行 ≠ 评估身份泛化**——精确可行解下逐点墙依然成立；固定容量路线在现任务设计下关闭，进入任务/表示联合重设计决策点 |
| `projection_incomplete` | 求解器未达收敛判据 | 机械失败，`status=failed` 停止 |
| `baseline_drift` | 基线臂两 seed 全过 | 互斥为数据敏感 → 全部结论重审 |

`growth_admitted=false`、`can_promote=false` 贯穿；dynamic growth、P5、CUDA、IDE/provider 继续冻结；不加第三臂、不改基线臂协议、不扩到 9-cell。

## 7. 产物顺序

1. 求解器实现（`taiji/g_selection_projection.py` 的纯函数投影器 + `ExtendedGSelectionLearner` 的投影后恢复接口）+ 定向测试（收敛判据/确定性/anchor 固定/bias 不动/birth 不受影响）；
2. `scripts/training/eval_taiji_m5_k_p4_11_projection_solver.py`（两臂 runner，py_compile/ruff/mypy 先行）；
3. 执行产出 `plans/manifests/taiji_m5_k_p4_11_projection_solver_manifest_v1.json` + `reports/taiji_m5_k_p4_11_projection_solver_20260911.json`；
4. 路线图/记录文档同步 + 独立提交。

## 8. 执行记录（2026-09-11，已运行，投影求解器成立）

1. 实现顺序：`taiji/g_selection_projection.py`（纯函数投影器）+ `ExtendedGSelectionLearner.apply_projected_weights` + 定向测试 5/5（简单系统收敛到可行侧/确定性/anchor 已可行则不动/不可行系统诚实失败/extended learner 投影应用 + bias 与 feature source 不动 + 往返）+ mypy 干净后运行两臂 runner。执行前修订 §2.1（保持约束从边际保持形式改为**决策同一性形式**——探针已证可行的系统；边际保持语义仅保留在基线臂 SGD hinge 中）。
2. 报告 `reports/taiji_m5_k_p4_11_projection_solver_20260911.json`：`status=completed`、`experiment_passed=true`（机械门全过：两臂出生等价 0 mismatch / 0.0 偏差、出生 hinge 恒 0、**trajectory gate = 两臂投影前 head digest 逐位相同**（同轨迹确证）、feature source 非漂移、身份与 P4.1–P4.10 隔离、参数 17/17 精确）。
3. 结果（两 seed，frozen 门）：
   - **投影精确收敛**：80 条联合约束，max violation = total violation = **0.0**（两 seed）；投影距离 L2 `2.72/2.98`、L∞ `2.19/2.52`（诚实记录，有限 ρ 的可行侧富余包含在内）；
   - `invariant-ext-17` 基线：与 P4.10 同构失败（两 seed holdout `0.6375/0.55`+6 sv；sibling `1.0` 过）；
   - **`projected-ext-17` 两 seed 全门通过**：holdout utility `0.8`（≥ 0.68）、target `0.75`（≥ 0.6）、safe violations `0`；retention-sibling `1.0/1.0`（非劣于 parent）；retention-newtask `0.8`（≥ parent `0.6375`）。
4. **判定（按 §6 冻结映射）：`projection_solver_supported`——可行区域投影求解器成立。** 这是 P 系列首次有更新机制在两 seed 上同时通过新任务门与双分布保持门：「SGD 任务学习 + 末端联合投影」的更新机制修复了 P4.10 的 learnability gap，且泛化到评估身份（holdout/sibling/retention-newtask 均为全新身份）。固定容量路线在求解器更新机制下**重开**。`growth_admitted=false`、`can_promote=false` 不变。
5. 唯一下一步 = 求解器机制下的晋级课程级验证预注册（更大的 seed/课程矩阵 + 资源审计，gate 沿用已冻结阈值）。
