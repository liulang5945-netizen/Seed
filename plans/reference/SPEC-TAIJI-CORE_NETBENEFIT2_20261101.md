# 09.6.9：净收益跨评估生涯复核合同（不授予正式训练）

> 文档状态：[合同] — owner 已于 2026-11-01 裁定「先换第二个 holdout」。本合同只冻结第二评估生涯与跨生涯判据；**不批准 09.7 正式训练**。

日期：2026-11-01。`taiji/living_core.py` 字节保持不变。两臂实现与方向规则沿用 09.6.4/09.6.7/09.6.8。

## 1. 要回答的问题

09.6.8 的净收益（+1.739e-4）只在 `HOLDOUT` 一条评估生涯上测过。本合同在同两臂上再加**一条新的、结构同形但题材不同**的评估生涯，问这条净收益是不是只在单一夹具上成立。

**为什么第二条不复用 `RETENTION`**：`RETENTION` 同时是保持漂移门的探针，用它当评估生涯会把"被评估的面"和"被守的门"合成同一个，读数无法分账。

## 2. 第二评估生涯（声明）

题材与 `HOLDOUT`（绿色钥匙/门后→口袋）不同，结构同形：prompt 为同长度量级的中文重复材料、target 为短句重复、`records` 与 `RAW_NU/KNOWN_NU` 结构一致。它在**本合同中首次声明**，此前没有任何测量用过它，因此不是"挑出来的好面"。

## 3. 判据（先声明、后测量）

两臂同 09.6.8（`replay_only` 与 `consensus_birth`，parts=5、units=3、8 步、lr=1e-5），每臂在 `HOLDOUT`、`SECOND_HOLDOUT`、`RETENTION` 上各测 pre/post。

1. **主判据**：两条评估生涯上**都**满足 `net_benefit ≥ 1e-4`（`net_benefit = Δ_first − Δ_birth`，负为改善）。
2. **夹具特异**：只有 `HOLDOUT` 满足、第二条不满足 ⇒ `fixture_specific`（记录分布，不得作为 09.7 依据）。
3. **都不过** ⇒ `net_benefit_refuted`。
4. **保持**：两臂在 `RETENTION` 上的漂移都 ≤0.030；否则 `retention_breach`。
5. **一致性**：出生臂在 `HOLDOUT` 上的净收益必须与 09.6.8 的 +1.739e-4 相差 ≤20%，否则 `arm_face_mismatch`（证明本轮的臂与上一轮同源）。
6. **成本**：出生构造、两臂训练、两条评估生涯的探针成本分别入账。

## 4. 预算与出口

- 预算：≤600 CPU 秒 / 1200 墙钟秒；超限 `budget_stop`。
- 出口：`replicated_on_second_holdout`／`fixture_specific`／`net_benefit_refuted`／`retention_breach`／`arm_face_mismatch`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`scripts/training/verify_taiji_netbenefit_second.py`、`reports/taiji_core_netbenefit_second_20261101.json`、MAINLINE/03 同步。
- **通过只表示**"净收益在同一确定性主体、同两臂、两条结构同形的合成中文评估生涯上都 ≥1e-4"；不表示任务能力、不表示真实语料上成立、不表示 09.7 准入。**第二生涯仍是合成夹具**，不是"未见任务"。
