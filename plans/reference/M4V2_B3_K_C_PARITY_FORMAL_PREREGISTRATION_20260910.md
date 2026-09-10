# M4.V2 B3-K C-entry parity formal 预注册（sealed 评分判据冻结）

> 冻结日期：2026-09-10。前置：[caliber revision §6/§7](M4V2_B3_K_C_PARITY_CALIBER_REVISION_20260910.md)（等价类根因、v2 修复 9/9、课程独立性修正）与 [parity v2 manifest](../manifests/taiji_m4v2_b3_k_c_parity_v2.json)。本文在看任何 sealed 结果之前冻结 formal 的输入、评分语义、判据与停止线；冻结后实现 runner 并运行。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 可证伪假设

> 在诚实等预算 parity 下（v2 artifacts：双臂同流、600 新增步、38,664 字节、课程独立），candidate（widened 双通道，K3-anchored 乱序 + 正序）对未见 sealed 任务的泛化改善（相对 frozen parent 的 combined-MSE delta）**优于** strong control（fixed-large 同构 ensemble，正序 + 逆序）——即学习规则的差异（异构信号通道合成 vs 同构 replica 概率平均）在同等预算下产生可测的泛化差异。

## 2. 冻结的评分输入（digest 固定）

| 输入 | 位置 | 校验 |
|---|---|---|
| Sealed test artifact | `plans/manifests/taiji_m4v2_b3_k_c_sealed_test_v1.json` | 文件 sha256 `89896556d1d3dc86fb21a8c9303d7fdf1b83ecc5a7f22777586c7fe22539d78f`；内部 `artifact_digest=1c9e3853…`；3 episodes；v2 训练未接触 |
| v2 widened artifacts（9 cell） | `checkpoints/taiji_k_candidate_c_entry_parity_v2/model_{17,23,31}/course_{0,1,2}/taiji_c_entry_parity_v2_widened.pt` | digest 必须与 v2 build report `artifact_digests.widened` 逐 cell 一致 |
| v2 fixed-large ensembles（按课程 3 份） | `checkpoints/taiji_k_fixed_large_c_entry_v2/.../taiji_c_entry_parity_v2_ensemble.pt` | digest 与 v2 build report 一致；**预期恰 3 份互异（每课程一份）**——若 9 份互异或 1 份则课程独立性失败，停止 |
| Parent workers | `checkpoints/taiji_k_workers/model_{17,23,31}` | digest 与 v2 build report 的 parent 一致（K1/K2 权重跨 model 逐位相同为已知事实） |
| v2 build report | `reports/taiji_m4v2_b3_k_c_parity_v2_build_20260910.json` | `status=passed`、`course_independence.gate_passed=true` 为运行前提 |

## 3. 评分语义（冻结）

1. **指标**：6 分量 MSE（k1 fact/goal/content、k2 transition/goal/content）在 holdout experiences 上的均值 + `combined_mse`（6 分量的均值）；`delta_vs_frozen = after − before`（负 = 改善）。与 v1 formal 同口径，不重定义。
2. **candidate readout（诚实等价声明）**：widened 双通道共享同一推理输入，logit 域通道和**数学上等价于权重相加**（fact→goal→content 每级的 head 输入对两通道相同）。实现为两通道 state_dict 相加成单 learner 后走标准 predict；该等价性写入报告，比较解释为「两个不同顺序训练运行的权重合成 vs 概率平均」。
3. **fixed-large readout**：replica 概率算术平均（既有 `_fixed_large_loss_score`），不改变。
4. **validation 集**：沿 v1 程序重建（validation episodes、holdout split、与 train/sealed 输入不交），仅用于 G1 质量门与 G2 灾难界的**派生**——派生完成后即冻结数值写入 runner 输入，然后才允许读 sealed。
5. **sealed 读取次数**：v1 formal 已读一次；v2 训练未接触 sealed 内容。runner 记录 `sealed_read_count`，诚实报告。

## 4. 判据（全部在看 sealed 前冻结）

**技术门（先于评分，任一失败即停）：**

- T1 全部输入 digest 校验通过；v2 build report `status=passed` 且课程独立性门通过；
- T2 评分键集无漂移（6 分量键完全一致）；无 NaN/Inf。

**主判据（sealed，统计单元 = course，n=3）：**

- **G1 质量门（validation）**：candidate 的 combined delta 在 ≥ 2/3 课程上 < 0（证明学到了东西）；
- **G2 灾难界**：`epsilon_cat = max(0.01, 3 × std(9 个 cell 的 validation candidate combined delta))`（population std），在 sealed 读取前由 validation 派生并冻结；sealed candidate combined delta 在**每个**课程上 ≤ `+epsilon_cat`；
- **G3 主判据**：sealed 上 candidate combined delta < fixed-large combined delta 在 **≥ 2/3 课程**成立，**且** 3 课程均值上 candidate < fixed-large；
- **G4 描述性（非门）**：9 个 per-cell 结果、每课程的 6 分量分解、candidate−fixed-large 的逐课程差值，全部如实呈现。

**结果映射：**

- G1∧G2∧G3 全过 → 「等预算下学习规则差异成立」的证据闭合；进入 K 轴 scorecard v3 与 §5.3 晋级讨论（`can_promote` 只由 aggregate 计算）；
- G3 失败 → 如实报告「candidate 未胜过 strong control」；保留最强 fixed-capacity 基线，回学习规则设计归因；不调阈值、不换课程重跑；
- G1/G2 失败 → 学习本身或稳定性失败，回 B 阶段归因。

## 5. 停止线

- 任何 digest/键集校验失败 → 停，不评分；
- **sealed 读取后禁止任何判据/阈值/输入调整**——第二次 sealed 读取只允许发生在判据实现 bug 的修复 + 新预注册之后；
- epsilon 派生若出现异常（std 不可用、validation 缺失）→ 停，不临场替代；
- 不引入 provider/联网/真实客户端写入/default runtime。

## 6. 产物

- Runner：`scripts/training/eval_taiji_m4v2_b3_k_c_parity_formal.py`（先 py_compile/ruff/mypy）；
- 报告：`reports/taiji_m4v2_b3_k_c_parity_formal_20260910.json`（format `taiji-m4v2-b3-k-c-parity-formal-v1`，含 frozen-input 校验、validation epsilon 派生记录、sealed 逐 cell/逐课程结果、G1–G4 判定、`can_promote=false`）；
- 路线图执行记录 + 独立提交。
