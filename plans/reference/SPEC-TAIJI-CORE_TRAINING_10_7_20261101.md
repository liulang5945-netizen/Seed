# 10.7：第一次有界训练——目标条件化选择（不发布权重）

> 文档状态：[合同] — owner 已于 2026-11-01 批准「一次训练」。这是本主线**第一次**训练授权，范围严格限定在下面的合成任务与预算内；训出的权重**只用于验收、不发布、不改产品默认**。

日期：2026-11-01。候选为 09.8 同一主体（父模型摘要须逐位吻合）。

## 1. 数据与任务（全部合成、无外部语料）

- 世界同 10.1b（2 对象 × 3 位置、容量 1）。
- 样本 =（局面, 目标）对。局面枚举所有 6 种两对象占位；每个局面只有 1 个空位，因此恰有 2 个目标（各自命名一个对象放到该空位）有正确动作，其余 4 个目标无解、不入样本。共 **12 个样本**。
- **训练/留出按局面划分**：3 个局面（6 样本）训练，另 3 个局面（6 样本）留出。同一局面不跨集，避免记忆泄漏。
- 监督信号：动作句进前缀的是非题（同 10.6 构造）。正确动作的答案取「是」、错误动作取「否」。
- 损失：`NLL(是 | prefix_correct) + NLL(否 | prefix_wrong)`，按样本平均。

## 2. 训练范围与判据

- 网格：`lr ∈ {1e-5, 1e-4}` × `32 步`，Adam，单线程。**两个 lr 都跑、都发布**，不挑。
- **主判据**：至少一个 lr 在**留出 6 样本**上选择正确 ≥ **5/6**，且该配置下 `RETENTION` 漂移 ≤0.030 ⇒ `training_learned_selection`。
- **未学到**：留出正确 ≤4/6 ⇒ `training_below_chance_plus`（4/6 与 5/6 之间记 `training_partial`）。
- **保持**：任一配置使漂移 >0.030 ⇒ 该配置记 `retention_breach`，不得作为候选。
- **同候选**：训练前父模型摘要等于 09.8 卡片。
- **不发布**：报告必须写 `weights_published=false`、`default_model_replaced=false`；训练只落盘到 `output/taiji_core_10_7/`（`.pt` 已被 gitignore 忽略）。

## 3. 预算与出口

- 预算：≤600 CPU 秒 / 1200 墙钟秒；超限 `budget_stop`。
- 出口：`training_learned_selection`／`training_partial`／`training_below_chance_plus`／`retention_breach`／`candidate_identity_mismatch`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`taiji/goal_conditioned_training.py`、`tests/taiji_native/test_goal_conditioned_training.py`、`scripts/training/verify_taiji_goal_training_10_7.py`、`reports/taiji_core_goal_training_10_7_20261101.json`、MAINLINE/03 同步。
- **通过只表示**在 12 个合成样本的 6 个留出样本上，这次有界训练把选择正确率推到 ≥5/6；**不表示**自然中文连续任务能力、不表示 P2 满足（P2 还要求与随机/贪心对照比较）、不表示 L3、不表示训练准入常态化。**这一次授权不外推。**
