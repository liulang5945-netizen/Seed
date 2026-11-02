# 09.6.6：出生方向规则·真实变化轴复现合同（不授予正式训练）

> 文档状态：[合同] — 用于替换[09.6.5](SPEC-TAIJI-CORE_DIRECTION_ABS_20261101.md#4-执行更正2026-11-01同日)里失效的种子轴。**不批准 09.7 正式训练**。

日期：2026-11-01。`taiji/living_core.py` 字节保持不变。规则、四臂实现与泄漏守卫全部沿用 09.6.4/09.6.5，不重写。

## 1. 为什么重做

09.6.5 用"三个种子"当复现轴，但 `LivingCoreReference` 由内部固定种子初始化，全局 `torch.manual_seed` 不改变父模型：三个"种子"的模型摘要逐位相同，共识臂三次读数也逐位相同（0.000142628236 ×3）。所以那次运行只是**同一配置重复三遍**，`hypothesis_h1_supported` 已作废。

本合同把变化轴换成**规则自身的窗口数 `parts ∈ {2, 3, 4}`**：这是真实作用于出生方向的自由度，不是重复采样。

## 2. 预注册判据

三格 `parts ∈ {2, 3, 4}`，每格跑共识臂、随机对照（两个随机种子 `{20261101, 20261112}`）与 fit-SVD 基线。

1. **主判据**：共识臂在**至少 2 格**满足 `新生儿绝对贡献 ≥ 1e-4` 且 `retention_drift ≤ 0.030`。
2. **随机对照**：所有随机对照都必须 `< 1e-4`；任一过线判 `metric_not_discriminating`。
3. **基线复现**：fit-SVD 基线在 3 格上的绝对贡献相对 4.33e-7 的偏差均落在 `[0.2, 5]` 倍内，否则 `baseline_face_mismatch`。
4. **不泄漏**：AST 标识符扫描零命中，否则 `leak_detected`。
5. **主体确定性声明**：合同显式记录"父模型跨全局种子逐位相同"，因此本合同的复现**只**指窗口数轴，不得被读成跨主体或跨任务的复现。

## 3. 预算与出口

- 预算：≤1800 CPU 秒 / 3600 墙钟秒；超限 `budget_stop`。
- 出口：`hypothesis_h1_supported`（≥2/3 格过线且对照全在门下）／`hypothesis_h1_not_replicated`（1/3）／`hypothesis_h1_refuted`（0/3）／`metric_not_discriminating`／`baseline_face_mismatch`／`leak_detected`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`scripts/training/verify_taiji_birth_direction_parts.py`（复用 09.6.4 的臂实现）、`reports/taiji_core_birth_direction_parts_20261101.json`、MAINLINE/03 同步。
- **通过只表示**"共识方向规则在声明的窗口数上、对同一确定性主体，能让新生单元承担 ≥1e-4 的 holdout 改善且保持漂移不超 0.030"。不表示绝对收益能偿付出生成本、不表示任务能力、不表示 09.7 准入。
