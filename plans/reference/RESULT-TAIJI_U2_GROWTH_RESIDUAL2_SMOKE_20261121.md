# 结果卡：大容量残差生长烟测（growth_residual2_partial；144.78 CPU；1 seed；零 check）

> [结果卡] — 2026-11-21（D-20261121-263）。上游 [残差生长](RESULT-TAIJI_U2_GROWTH_RESIDUAL_SMOKE_20261121.md)。

## 1. 协议
受保护 72 单元（P-D 0.8125）→ 中性生长 **+24**（12 群×2）→ 旧 P-D 头冻结、新单位专用残差头（48→64→4）＋新单位行可训 **900 步**；combined=旧+新 logits。

## 2. 读数（回执 reports/taiji_u2_growth_residual2_smoke_20261121.json）
old_only 0.8125；combined 0.8750（+0.0625）；**lesion_new 0.0625**；判 `growth_residual2_partial`；144.78 CPU、零 check。

## 3. 判读
1. 容量 12→24、训练 700→900 步后，新单位贡献从 0.047 升到 **0.0625**，方向一致但增幅有限；
2. 仍**远低于 0.20 门**：新单位只能承担小残差，生长收益仍未成立；
3. 限制：1 seed；残差头仅见新单位；旧头冻结；不签生长/协同/check/final。

## 4. 后继
若继续：残差头同时见旧+新（旧路低 LR）、或新单位按组 specialization（每群不同角色）、或接受「生长收益在当前协议下不可得」为负结论并写入收束。
