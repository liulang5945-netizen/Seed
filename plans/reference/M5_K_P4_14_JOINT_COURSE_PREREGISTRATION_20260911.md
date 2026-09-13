# M5.K P4.14 预注册：K worker 联合课程（K continuation × G 求解器机制同 parent 联合运行）

> 冻结日期：2026-09-11。前置：[scorecard v4 合同 §7](M5_K_AXIS_SCORECARD_V4_CONTRACT_20260911.md)（`k_worker_joint_course_completed=false` 为晋级评审两个未完成入场条件之一）与 [结果复审 §38](M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)。本文冻结两条已验证机制——**K worker continuation（P2.6/P2.7 机械）**与 **G 头求解器机制（P4.9–P4.13 机械）**——在同一 parent 上的联合课程；gate 阈值沿用各自已冻结值（零变更）；冻结后按 §8 顺序执行。

## 1. 可证伪假设

两条机制各自已验证：K worker continuation 在保持约束下学习新 K2 content 目标并跨项目泛化（P2.6/P2.7）；G 头求解器机制（表示因子化 + 末端联合投影）在新任务与保持双门上零方差通过（P4.12/P4.13）。但二者从未在同一 parent 的同一课程中**相继运行**——而联合的交互点是真实的：**K worker 学习会改变候选特征景观**（G 相候选特征由 K1/K2 读出派生），G 相必须在「K 学习后的景观」上维持求解器平衡。

**假设**：两条机制可组合——Phase K（K worker 学习新 K2 content 目标、保持旧类）之后的特征景观上，Phase G（求解器机制）仍能同时通过新任务门与保持门，且 G 相不扰动 K worker（K 参数在 G 相冻结）。

## 2. 课程结构（每 cell 内两相继阶段，同一 parent）

- **Phase K（K worker continuation，P2.6 机械原样）**：新颖 K2 content 目标学习——6 条 disjoint train candidate + 2 条 disjoint validation candidate（身份 `p4-14` Phase K cohort），固定 50 条 P2 rehearsal 按 P2.4 顺序交错；K1/K2 局部更新、参数不增长、K3 不变；**K checkpoint 保存 + 独立进程恢复**；
- **重 materialize**：从 **post-K worker** 重新实例化 semantic/transition learner，materialize Phase G 的候选 cohort（G 相候选特征反映 K 学习后的景观——这是联合课程的交互面）；
- **Phase G（G 头求解器机制，P4.11 合同原样）**：`ExtendedGSelectionLearner`（base 权重从 frozen P3.5 G parent 逐位继承 + 4 因子化维度零初始化；特征源 = frozen parent 副本，非漂移）→ 任务 fit（SGD 8 epochs + margin-preservation hinge）→ **末端联合投影**（任务约束 + 决策同一性保持约束，惩罚延续求解器，收敛判据冻结）→ G checkpoint。

## 3. 门（沿用各自已冻结阈值，零变更）

- **Phase K 门**（P2.6 冻结值）：新 K2 content 目标 validation `2/2`；旧类 K1/K2 content `4/4`；安全 abstention `6/6`；K checkpoint 独立恢复；参数不增长；
- **Phase G 门**（P4.7+ 冻结值）：holdout utility ≥ `0.68`、target ≥ `0.6`、safe violations == 0、reobserve 通过；保持门（sibling 与 retention-newtask）非劣于 in-run parent；投影收敛判据（逐约束 ≤ 1e-6 / 总量 ≤ 1e-5）；
- **跨相门（联合课程新增）**：
  - `k_unchanged_after_g`：G 相全程 K1/K2 digest 不变（G 只写 G 头）；
  - `g_preservation_vs_frozen_parent`：G 相保持参考 = frozen P3.5 G parent 在 **post-K 景观**上的决策同一性（well-defined：parent 头对任意特征可打分）；
  - G 相 birth 等价（post-K 景观上 vs frozen parent 决策）精确成立。

## 4. 矩阵与资源

- **矩阵**：2 身份批 × 2 seeds = **4 cells**。理由：两条机制均已单独验证（P4.12/P4.13 九格零方差、P2.6/P2.7 门通过），本课程的主导风险是**集成机械**（post-K 重 materialization、跨相状态传递），首次联合运行以集成验证为先；若成立，规模扩展属晋级课程 formal；
- **资源（绝对预算，冻结）**：Phase K fit ≤ `120s`、Phase G fit ≤ `60s`、Phase G 投影 ≤ `120s`、cell 总 ≤ `600s`；K checkpoint / G checkpoint 字节记录；
- 机械门：全部 checkpoint 独立进程恢复 + tamper 拒绝 + parent（G 侧 P3.5 checkpoint 与 K 侧 pre-K workers）未覆盖。

## 5. 结果映射（全分支；执行后禁止调门）

| 分支 | 条件 | 判定 |
|---|---|---|
| `joint_course_supported` | 4/4 cell：Phase K 全门 ∧ Phase G 全门 ∧ 跨相门 | **两机制可组合**——scorecard v5 将 `k_worker_joint_course_completed=true`；晋级评审两入场条件仅剩 rollout review；进入默认 runtime rollout review 预注册 |
| `k_retention_regressed` | Phase K 保持门失败（≥ 1 cell） | K continuation 机械在联合语境下回退——回 K 侧机制归因（P2.6 机械未变，检查联合课程数据差异） |
| `cross_phase_feature_shift` | Phase K 全门 ∧ Phase G 失败（≥ 1 cell） | **K 学习的特征漂移破坏 G 求解器平衡**——联合的交互面真实存在；进入跨相特征漂移处理设计（G 相约束系统纳入 post-K 校准） |
| `projection_incomplete` | G 相投影不收敛（≥ 2 cell） | post-K 景观上联合约束系统不可行——诚实停止 |
| 机械失败 | checkpoint/身份门失败 | `status=failed` + 归因，停止 |

`growth_admitted=false`、`can_promote=false` 贯穿（本课程闭合 scorecard v4 的入场条件之一；promotion 评审与默认 runtime rollout review 是独立后续）；dynamic growth、P5、CUDA、IDE/provider 继续冻结；不加第三臂、不改任一机制的已冻结合同。

## 6. 边界（诚实声明）

- 本课程检验的是**两机制的相继组合**（K 学习改变景观 → G 在新景观上求解），不是 K/G 同时训练（那需要约束系统的跨头联合求解，属后续设计）；
- K 相的新颖目标 = P2.6 同构（新 K2 content 组合），非新任务类型；
- G 相为单阶段（P4.13 已闭合两相 G 侧课程）；
- 不读取 sealed；不重算任何历史 formal。

## 7. 与既有教训的对齐

- **P2.6 教训**：novel 目标须与 P1/P2.5 全部 disjoint，rehearsal 保持约束不可省略；
- **P4.8 教训**：逐点约束的泛化由评估身份门实测（本课程 G 相 holdout 为全新身份）；
- **P4.10 教训**：feature source 独立存储 + digest 校验（frozen parent 副本在 K 漂移景观下仍是非漂移锚）；
- **P4.12 教训**：资源 cap 绝对预算定义。

## 8. 产物顺序

1. `scripts/training/eval_taiji_m5_k_p4_14_joint_course.py`（两相联合 runner：Phase K P2.6 机械 + post-K 重 materialization + Phase G 求解器机械 + 跨相门；py_compile/ruff/mypy 先行）；
2. 执行产出 `plans/manifests/taiji_m5_k_p4_14_joint_course_manifest_v1.json` + `reports/taiji_m5_k_p4_14_joint_course_20260911.json`；
3. 路线图同步 + scorecard v5（若 `joint_course_supported` 则 `k_worker_joint_course_completed=true`）+ 独立提交。

## 9. 执行记录（2026-09-11，P4.14 已运行）

1. **Runner**：`scripts/training/eval_taiji_m5_k_p4_14_joint_course.py`（py_compile / ruff / black / mypy --follow-imports=silent 通过后执行）。K 相起点 = P3.2 base-continuation workers（联合 parent 的 K worker，digest 对 P4.2 manifest 逐位校验）；G 相 parent = frozen P3.5 checkpoint（lineage 断言 + 独立进程恢复）。P4.11/P4.13 机械（`_task_constraints`/`_preservation_constraints`/`_save_extended_checkpoint`/extended tamper）直接复用 P4.13 模块原函数；Phase K 机械复用 P2.6 原函数（`_interleaved_stream`/`_fit_stream`/`_save_arm`/`_score_arm`/`_chain_row`）。
2. **报告** `reports/taiji_m5_k_p4_14_joint_course_20260911.json`，manifest `plans/manifests/taiji_m5_k_p4_14_joint_course_manifest_v1.json`：**`outcome=joint_course_supported`，4/4 cell 全门通过**（`experiment_passed=true`）。
   - Phase K（每 cell）：新颖 K2 content validation 2/2（parent 零样本已 2/2——P2.7 泛化的延续，非新任务类型，符合 §6 诚实边界）；旧类 K1/K2 content 4/4、安全 abstention 6/6；参数 5,648 不变；K checkpoint 独立恢复 + tamper 拒绝全过；post-K worker 跨 seed 逐位确定。
   - **景观漂移真实**：post-K K1/K2 digests 相对 pre-K 移动（k1 `12851b…→9fe6f4…`、k2 `411ef5…→81a189…`）；post-K worker 跨批逐位相同（K 特征空间不编码路径/项目身份——诚实记录，批身份差异在 G 侧评估身份）。
   - Phase G（post-K 景观）：birth 等价精确（0 mismatch / 0.0 偏差）；holdout target `0.8` / utility `0.8675` / safe violations `0`；sibling `1.0/1.0`；retention-newtask `0.8/0.8675`；投影 88 约束全格精确零违反；rollback/tamper/feature-source 全过。
   - 跨相门：`k_unchanged_after_g`（G 相后 K checkpoint digest 不变）、`g_preservation_vs_frozen_parent`（保持参考 = frozen parent 在 post-K 景观的决策同一性）、`birth_equivalence` 全过。
   - 资源绝对预算：cell 9.5–12.2s ≪ 600s，0 违规；checkpoint 字节已记录；K/G parent（pre-K workers 与 P3.5 checkpoint）未覆盖。
3. **scorecard v5**：[M5_K_AXIS_SCORECARD_V5_CONTRACT_20260911.md](M5_K_AXIS_SCORECARD_V5_CONTRACT_20260911.md) 冻结并运行——`k_worker_joint_course_completed=true`（唯一门翻转），`promotion_gate=false`、`can_promote=false`；晋级评审入场条件仅剩 `default_runtime_rollout_review_completed`。
4. `growth_admitted=false`、`can_promote=false` 贯穿；无 owner 解冻、无默认 runtime 接入、无 sealed 读取。唯一下一步 = **默认 runtime rollout review 预注册**。
