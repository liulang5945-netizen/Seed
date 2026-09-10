# M4.V2 C 阶段正式比较预注册（fast/slow+replay 机制，类分解判据）

> 冻结日期：2026-09-10。前置：[B3 K-phase pilot](M4V2_B3_K_PILOT_PREREGISTRATION_20260910.md)（机制门 12/12；replay 收益是类选择性的：D/R 弱类 FS 优于 C、整体略负）、[C-entry 证据线收束](M4V2_B3_K_C_PARITY_V3_PREREGISTRATION_20260910.md)（fixed-large 概率平均 = strong arm）、[信号空间扩展](M5_K_SIGNAL_SPACE_EXPANSION_PREREGISTRATION_20260910.md)（5 类 v4 workers）。本文在 materialize 全新 sealed v4 之前冻结 C 阶段正式比较的全部判据；冻结后实现 runner 并运行。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 可证伪假设

> B3 pilot 显示 fast/slow+replay 的收益是**类选择性**的（父代弱类 D/R 改善、整体略负）。C 阶段假设：**在多课程 sealed 评分上，fast/slow+replay（FS）对父代弱类（D/R）的泛化改善显著优于直接 continuation（C），且整体表现对 C 非劣**——即 replay 机制的价值真实存在且可按类分解测量。若弱类优势在 sealed 上不成立，则 replay 机制在 K 相位无正式价值，B 阶段以 continuation 为默认学习机制收束。

## 2. 臂与基底（零重训，全部复用 v4 artifacts）

| 臂 | 构成 | 容量 | 预算（new_update_steps） |
|---|---|---|---|
| F frozen | v4 parent workers 原样 | 5,648 参数/实例 | 0 |
| C continuation | parent + 150-experience 流上的直接局部 delta | 5,648 | 300 |
| **FS candidate** | parent + wake（同流 delta 写 fast）+ sleep（b=50 真实经历 replay 写 slow + consolidate） | 5,648 | 300 wake + **100 replay（单列）** |
| XL fixed-large（**次要-bar**） | 2 个同构 replica（正序/逆序）概率平均 | 11,296（2×，**容量差异如实记录**） | 600（每 replica 300） |

- 基底：`checkpoints/taiji_k_workers_v4/model_{17,23,31}/`；课程独立性设计延续（3 课程 × 类块排列）；9 cell，**统计单元 = course（n=3）**，model 维度为 replicate；
- XL 臂为次要-bar（容量 2× 混杂，不做主判定），回答「机制臂距离 strong arm 多远」；
- 零重训：全部臂复用 v4 parity build 的 artifacts（digest 对 build report 校验）。

## 3. 评分输入（冻结）

- **Sealed v4**（全新 materialize）：`materialize` task_seed 未用过（≠0/19/47/73），3 episodes 首 file 覆盖 **D（.h）/ R（missing）/ A（python）**——弱类分解量尺要求 D/R 类在 sealed 中有代表；不交性对 train/validation fixture 与 sealed v1/v2/v3 全部强制；materializer 与 v3 同构；
- **Validation**：沿既有程序重建（holdout episodes；仅用于 G1/G2 的 epsilon 派生）；
- **v4 artifacts**：widened/fixed-large v4 build report 的 digest 逐 cell 校验（FS 不需要 widened 通道——FS 是单实例机制，其 wake/sleep 在 pilot 中已完成，artifact 由 v3 runner 式的 FS 训练段重新生成并落盘，digest 固定）。

**实现说明**：FS 臂的 wake+sleep 训练在 v4 formal runner 内重放（pilot 脚本的 FS 段函数化），训练前 checkpoint 落盘、训练后 fresh restore 校验；训练超参与 pilot 完全一致（semantic 2.0 / transition 0.2 / 每 experience 1 epoch / b=50）。

## 4. 判据（全部在看 sealed v4 前冻结）

**技术门**：T1 全部输入 digest 校验 + v4 build report 状态与课程独立性门；T2 评分键集无漂移、无 NaN；T3 两阶段纪律（pre-sealed 产物先落盘）。

**G1 学习门（validation）**：C 与 FS 的 combined delta 在 ≥ 2/3 课程上 < 0（两个学习臂都学到东西）。

**G2 灾难界**：`epsilon_cat = max(0.01, 3 × population std(9 个 validation candidate combined deltas))`，pre-sealed 冻结；C 与 FS 的每课程 sealed delta 均 ≤ +epsilon_cat。

**G3 弱类主判据（类分解）**：定义 `weak_class_delta(arm, course)` = 该课程 sealed 上 D/R 类 experiences 的 combined delta 均值。**FS 的 weak_class_delta < C 的 weak_class_delta 于 ≥ 2/3 课程**。

**G4 整体非劣**：`epsilon_ni = max(0.002, 3 × population std(9 个 validation paired (FS − C) combined deltas))`，pre-sealed 冻结；FS 整体 sealed delta ≤ C 整体 sealed delta + epsilon_ni 于 ≥ 2/3 课程。

**G5 描述性（非门）**：XL fixed-large 的弱类/整体 delta（容量 2× 如实标注）；逐 cell 全量结果；FS replay 成本（100 步）。

**结果映射**：G1∧G2∧G3∧G4 全过 → 「replay 机制的类选择性价值在 sealed 上成立」，FS 成为 K 相位的默认学习机制候选，进 K 轴 scorecard v4 与晋级讨论；G3 失败 → replay 机制无正式价值，**continuation（直接局部 delta）收束为 K 相位默认学习机制**，B 阶段闭合；G4 失败（弱类优势存在但整体灾难性劣化）→ replay 设计回炉（抽样量/巩固率），单变量修订后再预注册。

## 5. 停止线

- sealed v4 materialization 不交性失败 → 换 task seed 重 materialize（不触模型）；
- digest/键集失败 → 停；
- **sealed v4 读取后禁止任何判据/阈值/readout/臂构成调整**；
- epsilon 派生异常 → 停；
- 不引入 provider/联网/真实客户端写入/default runtime；`can_promote=false` 固定。

## 6. 产物顺序

1. `materialize_taiji_m5_k_v4_sealed_test_v4.py` + `plans/manifests/taiji_m5_k_v4_sealed_test_v4.json`；
2. FS 训练段从 pilot 函数化 + `scripts/training/eval_taiji_m4v2_c_stage_formal.py`（四臂、两阶段、G1–G4；先 py_compile/ruff）；
3. pre-sealed 产物 + 正式报告 `reports/taiji_m4v2_c_stage_formal_20260910.json`；
4. 路线图执行记录 + 独立提交。
