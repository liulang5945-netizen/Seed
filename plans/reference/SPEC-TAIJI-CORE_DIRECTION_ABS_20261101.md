# 09.6.5：出生方向规则·绝对贡献判据合同（不授予正式训练）

> 文档状态：[合同] — owner 已于 2026-11-01 裁定「改绝对贡献＋随机对照」。本合同替换[09.6.4](SPEC-TAIJI-CORE_DIRECTION_20261101.md)的判据口径；09.6.4 的份额门结论保持原样、不重判。**不批准 09.7 正式训练**。

日期：2026-11-01。`taiji/living_core.py` 字节保持不变。规则与四臂实现沿用 09.6.4，不重写。

## 1. 为什么换口径

09.6.4 的份额门（`newborn_share ≥ 1e-4`）被自己的负对照否掉：随机方向也拿到 **9.836e-4**。份额的分母是约 0.02 的总收益、几乎不随臂变化，比值主要反映噪声。同一轮里**绝对贡献**才显示出分离度：共识 K=2 **1.426e-4**、随机 **1.99e-5**、fit-SVD 基线 4.33e-7。

## 2. 预注册判据（先声明、后测量）

三个声明种子 `SEED ∈ {20261101, 20261112, 20261113}`，每个种子跑三臂：共识 K=2、随机方向、fit-SVD 基线。

1. **主判据**：共识臂在**至少 2 个种子**上满足 `新生儿绝对贡献 ≥ 1e-4` 且 `retention_drift ≤ 0.030`。
2. **随机对照**：随机臂在**全部 3 个种子**上绝对贡献 `< 1e-4`。任一种子过线即判 `metric_not_discriminating`（主判据作废）。
3. **基线复现**：fit-SVD 基线在 3 个种子上绝对贡献的几何均值落在 `4.33e-7 × [0.2, 5]` 区间；否则判 `baseline_face_mismatch`。
4. **不泄漏**：AST 标识符扫描（09.6.4 已实现）必须为零命中，否则 `leak_detected`。

三个种子是预注册的，不做事后挑选；若主判据只在 1 个种子上成立，判 `hypothesis_h1_not_replicated` 并如实记录分布。

## 3. 预算与出口

- 预算：≤1800 CPU 秒 / 3600 墙钟秒；超限 `budget_stop`。
- 出口：`hypothesis_h1_supported`（≥2/3 种子过线且对照全在门下）／`hypothesis_h1_not_replicated`（1/3）／`hypothesis_h1_refuted`（0/3）／`metric_not_discriminating`／`baseline_face_mismatch`／`leak_detected`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`scripts/training/verify_taiji_birth_direction_abs.py`（复用 09.6.4 的臂实现，不复制）、`reports/taiji_core_birth_direction_abs_20261101.json`、MAINLINE/03 同步。
- **通过只表示**"一个不泄漏的方向规则在 3 个声明种子的多数上让新生单元承担 ≥1e-4 的 holdout 收益改善"。不表示任务能力、不表示生长收益已能偿付自身成本、不表示 09.7 准入，也不得据此声称 09.6.1 的绝对门在其他夹具上成立。

## 4. 执行更正（2026-11-01，同日）

首轮报告 `reports/taiji_core_birth_direction_abs_20261101.json` 写出的 `hypothesis_h1_supported` **作废，不得引用**。原因是本合同的种子轴是假的：`LivingCoreReference` 用内部固定种子初始化，全局 `torch.manual_seed(...)` 不改变父模型——三个"种子"产出的模型摘要逐位相同（`921dc9981187406a` ×3），所以"3 个种子上 ≥2 支持"实际是**同一配置跑了三遍**，共识臂的三次读数也因此完全相同（0.000142628236 ×3）。

该报告里仍然有效的只有一件事：**在同一确定性主体上，三个不同随机对照给出 1.988e-5 / −1.720e-6 / 7.284e-6，均低于绝对门**，而共识臂 1.426e-4、基线 4.33e-7。它证明了"共识臂不只是随机噪声"，但**不构成复现**。

按纪律：旧报告保留原处不删、状态标为作废；复现改在[09.6.6](SPEC-TAIJI-CORE_DIRECTION_REPLICATION_20261101.md)用真实变化轴（规则的窗口数）重做。
