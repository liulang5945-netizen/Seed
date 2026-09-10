# M4.V2 S/G/K 连续课程晋级 formal 预注册（promotion course v1）

> 冻结日期：2026-09-10。前置：[scorecard v3 合同 §6/§7](M5_K_AXIS_SCORECARD_V3_CONTRACT_20260910.md)（唯一后续动作 = 本文）、[C 阶段 formal v2 §7](M4V2_C_STAGE_FORMAL_V2_PREREGISTRATION_20260910.md)（FS 收束为 K 相位默认学习机制候选）、[B3 K-phase pilot](M4V2_B3_K_PILOT_PREREGISTRATION_20260910.md)（FS 机制门 12/12）、[A8 promotion 预注册 §3–§5](M4V2_R6_A8_PROMOTION_PREREGISTRATION_20260909.md)（晋级门结构：新能力置信下界 / 旧能力 retention / 资源-rollback-副作用）、[R6 资源聚合同](M4V2_R6_RESOURCE_AGGREGATE_PREREGISTRATION_20260909.md)（peak/wall 口径）。本文冻结同一 parent 连续 S/G/K 课程的假设、合同、臂、判据、epsilon、结果映射与停止线；冻结后按 §9 顺序执行，顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 可证伪假设与诚实边界

**假设**：同一 Taiji parent（P0）经连续 S→G→K 课程——S/G 相位以 fast/slow+真实 replay 演化父代、K 相位以 FS（fast/slow+replay）continuation 演化 worker——在预注册资源内：(i) K 相位在 fresh sealed v6 上产生真实改善且 FS 弱类更优；(ii) 全程旧能力保持（父代 S holdout 在 G 后 ≥ −epsilon_SG；worker validation ≤ +epsilon_cat）；(iii) 逐 phase joint checkpoint 可 fresh restore、可显式 rollback、资源在冻结预算内。

**诚实边界（冻结，防夸大）**：K worker 与父代参数不相交（K worker 由 adapter 输入驱动，不读父代权重；K 课程特征由数据派生）。S/G 后的父代演化**不改变** K worker 的输入分布。因此本 formal 的证据目标是**连续 lineage 上的整合证据**——joint checkpoint（父代+worker 同账）、双轴 retention、资源与 rollback——**不是**「S/G 学习迁移到 K」的权重耦合声明。任何把二轴结构写成迁移效应的表述都违反本节。

## 2. 同一 parent 合同与 lineage re-binding

- P0：`checkpoints/taiji_r6_parents/model_{17,23,31}.pt`（digest 已在 formal input manifest 注册）；所有臂从同一 P0 出发，parent namespace 在 frozen 臂中字节不变。
- S/G 相位：P0 →（S phase fast/slow+replay）→ P1 →（G-08-02/06-04/04-06/02-08 四 blend 相位）→ P2；全程 candidate namespace、逐 phase joint checkpoint（含 fast/slow 状态、course cursor、RNG、owner lineage——B2 preflight 机械）。
- K 相位：v4 worker bundle **lineage re-binding** 到 P2——bundle 扩展记录 `origin_parent_digest=P0`、`attached_parent_digest=P2` 与完整 S/G chain digests；K continuation experiences 绑定 `parent_checkpoint_digest=P2`。adapter 现行 fail-closed 的 parent 一致性校验全部保留，re-binding 是显式加法合同扩展，先实现后训练。
- **re-binding 的诚实理由**：worker 权重独立于父代权重，re-binding 是 lineage 元数据操作；把 v4 worker 在 P2 上重建会把「连续成长」混淆为「worker 重建」，并摧毁 v4 证据链的可比性。拒绝该路线。

## 3. 课程矩阵与预算

- 矩阵：3 model seeds（17/23/31）× 3 course seeds = **9 cells**；统计单元 course（n=3），model 维度为 replicate（已知三 parent 的 K worker 权重互异，v4 重建后独立性成立）。
- **S/G 课程**：结构与字节预算 = B3 pilot probe `course_manifest`（digest `a5e1025f…`）同构（S: 19/14；G 四相位 94–106/11）；course seed 只变块内 record 顺序，train/holdout 边界不变。S/G replay 预算单列：每相位 replay events ≤ train bytes（实测如实报告）。
- **K 课程 = 全新 registry**：course seeds **(3,4,5)**，150 experiences 5 类平衡（A/B/C/D/R 各 30），experience 身份铸造延续 parity v2 机制（首 file 注释行，类不变而 digest 唯一），anchored 排列类模式守卫延续；K 预算 C-control = 300 新增步、FS = wake 300 + replay 100 单列（C-stage 口径）。**fresh registry 的理由**：若复用 C-stage 课程与 worker 初态，K 相位结果将与 C-stage 逐位相同（worker 不读父代权重），G1/G2 沦为空转门；fresh registry 使机制判据成为对新增据的真预测。
- **sealed v6**：全新 materialize，task_seed 取未用值（排除 {0,19,47,73,101,151} 与新 K registry 身份空间），3 episodes 首 file 覆盖 D/R/A，与 v1–v5 sealed、train/validation、新 K registry 强制不交。

## 4. Arms（每 cell 4 臂，S/G 相位在三活跃臂间逐位共享）

1. **frozen-parent**：全课程零更新——epsilon 校准 + 自然波动基线；
2. **FS candidate**：S/G fast/slow+replay（B3 pilot 证实的父代最优机制）+ K FS continuation；
3. **C-control**：与 FS candidate 逐位相同的 S/G 演化，仅 K 相位用直接 continuation——机制单变量；
4. **replay-lesion**：与 FS candidate 唯一差异 = K 相位 sleep 去 replay（consolidate 清 fast 但不写 slow replay 更新）——验证 replay（而非 fast/slow 分相本身）是弱类收益来源。

不做 fixed-large / matched-fixed-capacity：容量问题 C-entry 已闭合（fixed-large 概率平均 = strong arm）、机制比较 C-stage 已闭合（FS 候选）；本 formal 的证据目标是连续 lineage 整合，不重开已闭合维度，也不复活被否决的路线。

## 5. 量尺与判据（两阶段纪律：pre-sealed 先冻结）

**pre-sealed 产物**（sealed v6 读取前落盘）：frozen-parent 的 S holdout baseline repeat（repeat seeds 401/503/607 × 3 model seeds）→ `epsilon_SG = max(0.01, 3×std)`（跨 model seed 取 max，单一冻结值）；9 个 validation FS delta 的 population std → `epsilon_cat = max(0.01, 3×std)`。

- **G1 K 新能力（sealed 主测量，C-stage v2 操作化）**：FS 与 C-control 的 sealed combined delta < 0 于 ≥2/3 课程；
- **G2 机制复制**：FS 弱类（D/R）sealed delta < C-control 于 ≥2/3 课程——C-stage 结论对 fresh registry 的真预测；
- **G3 父代 retention**：G 完成后 S holdout mean_surprise delta ≥ −epsilon_SG（FS/C/lesion 三臂同查；frozen 臂 provide 校准；S 后 G holdout 提升为描述性）；K 相位后父代 S/G 精确零变化作为观测项记录（父代不被 K 写入），不替代 G3；
- **G4 worker retention**：FS/C/lesion 的 K validation combined delta ≤ +epsilon_cat（灾难界）；
- **G5 资源与 rollback**：worker 参数字节 FS=C=lesion 精确相等；FS/C/lesion peak RSS ≤ 1.25× frozen 基线；checkpoint write bytes ≤ 1.05× C-control；逐 phase joint checkpoint fresh restore + 显式 rollback（含跨课程回 P0）+ 失败 artifact 保留全过；wall-clock 与 S/G replay 消耗如实报告（replay 为声明预算差，不设 wall cap）。

量尺分离：父代轴用 `score_bytes` mean_surprise（R2 口径），worker 轴用六分量 structured-loss MSE（C-stage 口径）；两者分域保存，不平均成单一分数。

## 6. 结果映射（全分支，sealed 读取后禁止调门）

- **G1∧G2∧G3∧G4∧G5 全过** → `same_parent_continual_s_g_k_evidence=true`、`resource_rollback_old_capability_gate=true`、`parent_retention_baseline_present=true`；scorecard v4 收束入账；`can_promote=false` 保持——晋级讨论需独立 default-runtime rollout review（v3 合同 §4 边界不变）；
- **G1 败**（sealed 无改善）→ FS 在连续 lineage 无效，回机制归因（fresh registry 难度 vs C-stage）；
- **G2 败** → C-stage 机制结论不迁移到 fresh registry，机制选择重开预注册；
- **G3 败** → 父代遗忘超界，回 S/G replay/consolidation 设计；
- **G4 败** → worker 灾难界破，回 K 课程/巩固设计；
- **G5 败** → 资源或 rollback 机械失败，停止，不调阈值。

## 7. 停止线

- parent preflight / fresh restore 失败：停止不训练；
- re-binding 合同定向测试不过：停止；
- sealed v6 materialize 不交校验失败：停止；
- 任一 cell 执行异常：停该 cell 并按 harness/environment 归因，不静默重试；
- sealed v6 一旦读取，禁止任何判据/阈值/臂调整；`can_promote=false` 固定。

## 8. 不做的事

- 不接 default runtime、不引入 provider/MCP/client/CUDA/真实客户端写入；
- 不重训 v4 workers 的 M5 课程、不重算任何历史 formal；
- 不把 S/G/K 绝对分数平均成「智能分数」；
- 不用 MSE delta 冒充执行成功率（真实执行链证据属 R6 revised formal 线，两线不合流）；
- 不伪造父代 retention（S/G retention 用真实 holdout 实测，K 轴不写父代 F1/memory owner）。

## 9. 产物顺序

1. lineage re-binding 合同扩展（`k_worker_manifest.py` / `k_continuation.py` / adapter）+ 定向测试（tamper、chain 完整性、fail-closed 保留）；
2. promotion course manifest v1（content-addressed：S/G 相位 + fresh K registry + 预算声明）+ 逐 phase joint checkpoint preflight（扩 B2 机械至父代+worker 同账）；
3. 单 cell smoke（model17/course0）——机械门 only，不读 sealed v6；
4. pre-sealed 产物（baseline repeat + epsilon 派生）+ sealed v6 materialize；
5. 9-cell formal runner → `reports/taiji_m4v2_sgk_promotion_formal_20260910.json`；
6. 路线图执行记录 + 独立提交；scorecard v4 在 formal 判定后另行收束。
