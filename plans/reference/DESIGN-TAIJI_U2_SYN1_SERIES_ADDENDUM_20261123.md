# SYN-1 系列预注册补记（SYN-1b…SYN-1e 变体；1 seed、≤300 CPU/卡、零 check）

> [预注册·补记] — 2026-11-23（D-20261123-322…325）。主设计见 [SYN-1](DESIGN-TAIJI_U2_SYN1_20261123.md)。本文件记录四个变体的预注册口径；各卡判据清单已先行冻结（`taiji_u2_syn1b/…/syn1e_criteria_20261123.json`，各有 `criteria_sha256`），训练前不得改门。

## 变体一览

| 卡 | 决定号 | 与主卡的唯一差异 | 动机（读前） |
|---|---|---|---|
| SYN-1b | D-322 | 联合训练样本 48→2048、联合头 400 步 | SYN-1 用 48 样本训练（train 0.979/test 0.231）⇒ 规模不足导致记忆 |
| SYN-1c | D-323 | 同上＋**逐块状态标准化**（训练批 mean/std） | SYN-1b 诊断：CN 状态 std≈486、P27≈0.45，混入同一 96 维输入致优化失败（线性探针 0.267≈机会） |
| SYN-1d | D-324 | 同上＋联合头 1600 步、lr 5e-3 | SYN-1c train 0.822/test 0.719：训练不足/泛化间隙 |
| SYN-1e | D-325 | 同上＋联合训练样本 8192、头 batch 128 | SYN-1d train 0.904/test 0.742：过拟合于 2048 样本 |

## 判据（四变体相同，读前冻结）

| 门 | 判据 |
|---|---|
| G0 联合任务 acc | ≥0.80 |
| G1 CN 块休眠 drop | ≥0.40 |
| G2 P27 块休眠 drop | ≥0.40 |
| G3 单块头 acc | ≤0.35 |
| G4 单位级休眠（各 12）drop | ≥0.10 |
| G5 capsule | ≤1e-6 |
| G6 预算/纪律 | ≤300 CPU、`criteria_sha256` 读前校验、零 check/final |

**出口**：全过⇒`syn1*_verified`；G0 不过⇒`syn1*_low_joint`；G1/G2 不过⇒`not_synergistic`；G3 不过⇒`single_block_sufficient`；G4 不过⇒`no_unit_causality`；其余照旧。**不调阈、不改门。**

## 披露

- 变体只改训练规模/标准化（读前声明的实现修正），不追溯修改已判定的卡；
- 联合任务为人工构造组合标签；只训练读出（块冻结）；单训练 seed；不签 24-S/29/final/C/D/J；零 check/final。
