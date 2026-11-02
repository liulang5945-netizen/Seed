# 09.6.8：出生净收益对照合同（不授予正式训练）

> 文档状态：[合同] — owner 已于 2026-11-01 裁定「先补净收益对照」。本合同只冻结"出生 vs 不出生"的同父同预算对照与其判据；**不批准 09.7 正式训练**。

日期：2026-11-01。`taiji/living_core.py` 字节保持不变。方向规则与臂实现沿用 09.6.4/09.6.7。

## 1. 要回答的问题

09.6.7 证明共识方向让新生单元承担 1.4–1.7e-4 的 holdout 改善（相对**同一次训练后把新生单元摘掉**）。但那还不是净收益：真正的对照是**根本不出生**——同一父、同样 8 步、同样 lr。本合同量这一条。

## 2. 两臂与判据（先声明、后测量）

- **A 臂 `replay_only`**：父模型直接训练 8 步（lr=1e-5，声明更新），无出生。
- **B 臂 `consensus_birth`**：共识方向出生（`parts=5`、`units_per_group=3`，即 09.6.7 最高格 1.737e-4 的配置），再训练同样 8 步。
- 两臂同一父（确定性主体）、同一 fit 生涯、同一 holdout/retention 探针。

判据：

1. **净收益**：`ΔNLL_holdout(replay) − ΔNLL_holdout(birth) ≥ 1e-4` ⇒ `net_benefit_supported`。（ΔNLL 取 post−pre，负为改善；差值 ≥1e-4 表示出生臂比不出生臂多改善至少 1e-4。）
2. **保持**：两臂 `retention_drift` 都 ≤0.030；任一超限 ⇒ `retention_breach`。
3. **成本入账**：出生本身的 CPU（构造方向＋`gc.birth`）、两臂各自的 8 步训练 CPU、参数增量（新生行/列数）全部记录，并给出 `extra_cpu_per_1e4_nll`（每 1e-4 净收益额外花多少 CPU 秒）。**本合同不设成本门槛**，只把它作为 09.7 定价的输入。
4. **一致性**：B 臂的 lesion 贡献必须与 09.6.7 同格（parts=5, units=3）的 1.737e-4 相差 ≤20%，否则 `arm_face_mismatch`。

## 3. 预算与出口

- 预算：≤600 CPU 秒 / 1200 墙钟秒；超限 `budget_stop`。
- 出口：`net_benefit_supported`／`net_benefit_refuted`（净收益 <1e-4）／`retention_breach`／`arm_face_mismatch`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`scripts/training/verify_taiji_birth_netbenefit.py`、`reports/taiji_core_birth_netbenefit_20261101.json`、MAINLINE/03 同步。
- **通过只表示**"在该确定性主体与合成中文生涯上，出生臂比不出生臂多改善 ≥1e-4 的 holdout NLL，且保持不超限"；不表示任务能力、不表示跨参考图外推、不表示 09.7 准入。成本读数只作定价输入。
