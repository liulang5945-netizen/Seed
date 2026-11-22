# XFER-1b 预注册：跨会话组合规则迁移——150 步测量窗（避免饱和；1 seed、≤400 CPU、零 check）

> [预注册] — 2026-11-23（D-20261123-308）。上游 [XFER-1](RESULT-TAIJI_U2_XFER1_20261123.md)（D-307）：`xfer1_no_transfer`，但读数显示**天花板效应**——warm 臂在会话 B 第 25 步即达 1.000，fresh 臂到第 225 步才 1.000；在 250 步主读数处两臂同为 1.000 ⇒ G2 增益判据被天花板吞掉（0.000）。本卡不改任务/种子/门，只把会话 B 的**测量窗**从 250 步缩到 150 步（fresh 仍在 ~0.58，未饱和），使「首会话迁移」可测。判据清单：[`taiji_u2_xfer1b_criteria_20261123.json`](../manifests/taiji_u2_xfer1b_criteria_20261123.json)（`criteria_sha256` 读前冻结）。**按常设授权执行，待 owner 回溯。**

## 1. 与 XFER-1 的差异（唯一）

- 会话 A：seen 对 600 步（同）；
- 跨会话：capsule_v2 保存→加载→应用到同候选副本（前向差 ≤1e-6，同）；
- 会话 B：unseen 对训练 **150 步**（原 250），每 25 步记录曲线；`B_warm`/`B_fresh` 均取第 150 步；
- 判据 G0–G3 与 XFER-1 完全相同（阈值未改）。

## 2. 判据（读前冻结，同 XFER-1）

| 门 | 判据 |
|---|---|
| G0 会话 A 学会 | `A_end` ≥0.80 |
| G1 迁移水平 | `B_warm` ≥ `A_end` − 0.05 |
| G2 迁移增益 | `B_warm` − `B_fresh` ≥0.05 |
| G3 跨会话保持/身份 | `A_after_B` drop ≤0.05；capsule 前向差 ≤1e-6 |
| G4 预算/纪律 | 单 seed、≤400 CPU、`criteria_sha256` 读前校验、零 check/final |

**出口**：全过⇒`xfer1b_verified`；G2 不过⇒`xfer1b_no_transfer`；G1 不过（G2 过）⇒`xfer1b_level_low`；G3 不过⇒`xfer1b_retention_breach`；G0 不过⇒`xfer1b_baseline_insufficient`；超帽/工程异常照旧。**不调阈、不改门。**

## 3. 披露

- 本卡是 XFER-1 的**测量窗修订**（新卡新哈希），不是对 XFER-1 判词的回溯；XFER-1 的 `xfer1_no_transfer` 与天花板读数原样保留；
- 合成受控中文；单 seed；不签 24-S 正式包/29/final/C/D/J；零 check/final。
