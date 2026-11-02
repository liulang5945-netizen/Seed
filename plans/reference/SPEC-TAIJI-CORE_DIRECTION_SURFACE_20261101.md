# 09.6.7：出生方向规则·稳定性面合同（不授予正式训练）

> 文档状态：[合同] — owner 已于 2026-11-01 裁定「先扩稳定性面」。本合同只冻结超参数网格与"平台判据"；**不批准 09.7 正式训练**。

日期：2026-11-01。规则与臂实现沿用 09.6.4/09.6.6，不重写；`taiji/living_core.py` 字节保持不变。

## 1. 要回答的问题

09.6.6 在窗口数轴上 2/3 格过绝对门（parts=2 → 1.426e-4、parts=3 → 1.343e-4、parts=4 → 7.320e-5），边际只有 1.4 倍。**一个贴着门线的点不能作为 09.7 的依据**；本合同问的是：这条效应在超参数面上是**平台**还是**尖峰**。

## 2. 网格与判据（先声明、后测量）

`parts ∈ {2,3,4,5,6}` × `units_per_group ∈ {1,2,3}`，共 **15 格，全部发布**，不做事后挑选。每格只跑共识臂；对照另跑。

1. **平台判据**：15 格中**至少 6 格**满足 `绝对贡献 ≥ 1e-4` 且 `retention_drift ≤ 0.030`，**且**这些格覆盖至少 **3 个不同的 `parts` 值** ⇒ `platform_supported`。
2. **尖峰判据**：过线格数 ≥1 但不足 6，或覆盖的 `parts` 值 <3 ⇒ `spike_only`（记录分布，不作 09.7 依据）。
3. **全不过** ⇒ `no_cell_passes`。
4. **随机对照**：每 `units ∈ {1,2,3}` 跑 2 个随机种子，**全部** `< 1e-4`；任一过线 ⇒ `metric_not_discriminating`（整轮作废，不判平台/尖峰）。
5. **基线**：fit-SVD 基线在 1 单元 2 单元 3 单元各复现一次，三者相对 4.33e-7 的比值都落在 `[0.2, 5]`；否则 `baseline_face_mismatch`。
6. **不泄漏**：AST 标识符扫描零命中；否则 `leak_detected`。

## 3. 预算与出口

- 预算：≤1800 CPU 秒 / 3600 墙钟秒；超限 `budget_stop`。
- 出口：`platform_supported`／`spike_only`／`no_cell_passes`／`metric_not_discriminating`／`baseline_face_mismatch`／`leak_detected`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`scripts/training/verify_taiji_birth_direction_surface.py`（复用 09.6.4 的臂实现）、`reports/taiji_core_birth_direction_surface_20261101.json`、MAINLINE/03 同步。
- **通过只表示**"在该确定性主体、该参考图、合成中文生涯、8 步 lr=1e-5 下，共识方向规则在声明的超参数面上多数格过绝对门"。不表示成本偿付、不表示任务能力、不表示跨参考图外推，也不表示 09.7 准入。**不记录主体多样性以外的复现**：主体是确定性的，本合同的变化轴只有超参数。
