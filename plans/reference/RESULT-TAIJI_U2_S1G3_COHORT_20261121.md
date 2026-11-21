# 结果卡：cohort 发育式整合（`s1g3_partial`，196.59 CPU，零 check）——结构正确、功能贡献不足

> 文档状态：[结果卡/结构] — 2026-11-21（D-20261121-237）。上游 [S1-G/G2](RESULT-TAIJI_U2_S1G_GROWTH_FIX_20261121.md)。**零 check/final、只读 fit。**

## 1. 协议

- 48 单元主体（8 群×6、13 维内部读出），先训 P-D 350 步至 base 0.87–0.89；
- `grow_antithetic_cohort_` 生长**双生 cohort**（冻结成熟图、冻结出生锚点、exact zero birth）；
- **immature 期**用 `immature_cohort_logits`（正 twins）辅助 CE 训练 500 步（λ=0.5），随后 `promote_cohorts_to_mature_()`；
- lesion = promote 后准确率 − 将 cohort 置 `dormant` 后准确率；旧技能用 CN/P24 专用头测；capsule_v2 冷载对比；
- 2 seed、≤800 CPU、零 check。

## 2. 读数（[回执](../../reports/taiji_u2_s1g3_cohort_20261121.json)）

| 指标 | seed 20261120 | seed 20261121 | 门 | 判 |
|---|---|---|---|---|
| base | 0.891 | 0.867 | — | — |
| final（cohort mature） | 0.922 | 0.859 | — | — |
| **lesion（dormant 差）** | **0.031** | **−0.008** | ≥0.20 | **✗** |
| 出生中性 diff | **0.0** | **0.0** | ≤1e−6 | ✓ |
| 旧技能损伤 | **0.0** | **0.0** | ≤0.02 | ✓ |
| capsule_v2 恢复 diff | **0.0** | **0.0** | ≤1e−6 | ✓ |

- **`s1g3_partial`**；196.59 CPU、零 check/final。

## 3. 判读（如实）

1. **结构正确性保持**：antithetic cohort 的出生中性、旧技能保持、胶囊兼容全部成立（与 S1-G 一致）；
2. **功能贡献不足**：cohort 成熟后 lesion 仅 0.031 / −0.008，**远低于 0.20 门**——cohort 的 delta 在读出中权重很小，500 步发育训练未能让它承担功能；
3. **与旧线一致**：22.7 曾以等 FLOPs 判定 `growth_benefit_absent`；本次新核的三种生长路径（裸 `grow_units_neutral_`、两阶段新生儿、antithetic cohort 发育）**均未实现 ≥0.20 的功能贡献**；
4. **结构线结论（有界）**：**「可中性生长、可精确恢复、旧技能保持」成立；「新单元/群体承担可测功能贡献」未成立**；
5. 不签结构收益/生长收益/check/final/C/D/J；**23-S 仍不解锁**。

## 4. 收束与后继

- **结构整合线（S1-G/G2/G3）以负结果收束**：不再开同范围生长卡（单范围 ≤2 次修订铁律已用满）；
- S1 线总成本 ≈**2,223 CPU**（S1 255 + S1b 319 + 诊断 304 + S1-R 257 + S1-R2 593 + S1-G 298 + S1-G3 197）；零 check/final；
- 待 owner 裁定：① `23-S` 窄合同（中文＋四类纠错＋遮挡物理，P-B 用 0.875 有界值或专用大主体）；② 暂停新核主线按有界能力收束；③ 另行立项生长机制（需新架构/新证据，不在本次范围内）。
