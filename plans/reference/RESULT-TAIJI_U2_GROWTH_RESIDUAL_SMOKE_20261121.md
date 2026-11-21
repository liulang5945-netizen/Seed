# 结果卡：残差生长烟测（growth_residual_partial；131.36 CPU；1 seed；零 check）

> [结果卡] — 2026-11-21（D-20261121-262）。上游 [联合生长更正](RESULT-TAIJI_U2_GROWTH_JOINT_SMOKE_20261121.md)。

## 1. 协议
受保护 72 单元主体（P-D 训练 800/700/600 步）→ +12 中性单位 → 旧 P-D 头冻结，新单位专用残差头（24→64→4）＋仅新单位行可训 700 步；combined = 旧 logits + 新 logits。

## 2. 读数（回执 reports/taiji_u2_growth_residual_smoke_20261121.json）
old_only 0.6719；combined 0.7188（+0.0469）；**lesion_new 0.0469**；判 `growth_residual_partial`；131.36 CPU、零 check。

## 3. 判读
1. 新单位可被专用残差头读取，带来 **+4.7%** 提升——首个「新单位对组合输出有非零可测贡献」的信号；
2. 但远低于 0.20 门：新单位只能承担很小的残差，功能仍主要由旧单位承载；生长收益**未成立**；
3. 限制：1 seed；旧 P-D 基线较低（0.672，受训练步数缩减影响）；不签生长/协同/check/final。

## 4. 后继
增加新单位容量（+24 或整群）、延长残差训练、或让残差头同时看旧+新（旧路低 LR）；目标 lesion ≥0.20。
