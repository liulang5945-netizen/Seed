# 结果卡：生长绑定烟测（`growth_binding_partial`；128.5 CPU；1 seed；零 check）

> [结果卡] — 2026-11-21（D-20261121-259）。上游 [生长收益烟测](RESULT-TAIJI_U2_GROWTH_BENEFIT_SMOKE_20261121.md)。

## 1. 协议
1 seed；受保护 72 单元主体（P-D 0.8359）→ 中性生长 +12 单位 → **新 P-D 头仅在“新单位专用 mask”下训练** 700 步（旧单位行冻结、旧 D 路由不参与）；评测 new-only / old-only / combined 三路由。

## 2. 读数（[回执](../../reports/taiji_u2_growth_binding_smoke_20261121.json)）
| 路由 | P-D |
|---|---|
| 生长前基线（旧 24 单位） | 0.8359 |
| **new-only（仅新 12 单位）** | **0.8906** |
| old-only（旧 24 单位，生长后） | 0.7031 |
| combined（旧＋新） | 0.7344 |
| lesion(new) = combined − old-only | **0.0312** |

判 `growth_binding_partial`；128.5 CPU、零 check。

## 3. 判读
1. **新 12 单位可独立承载 P-D 达 0.8906** —— 新容量具备可学习/可承载功能，这是此前 lesion 0.0 时未见的正向信号；
2. 但**组合增益未成立**：combined 0.734 < new-only 0.891，且 lesion 仅 0.031；新头在 new-only 路由上训练，不能直接与旧单位组合（路由/表示不兼容）；
3. 结论：生长单位**可被强制绑定并承载功能**，但「生长带来总能力提升/协同」仍需联合训练协议（旧+新一起优化/蒸馏），不是简单拼接。

## 4. 后继
联合生长协议：新单位与旧单位共同训练（旧单位低 LR/保护）、读出混合路由；验证 combined ≥ max(old,new) 且 lesion ≥0.20。
