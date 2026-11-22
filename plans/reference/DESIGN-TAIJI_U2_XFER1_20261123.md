# XFER-1 预注册：跨会话组合规则迁移（会话 A seen 颜色对 → 会话 B 新颜色对；1 seed、≤400 CPU、零 check）

> [预注册] — 2026-11-23（D-20261123-307）。objective 条款「在同一主体身份下跨会话积累经验」目前只有保持/纠错（24-S）证据，**跨会话迁移（非恢复）未测**。判据清单：[`taiji_u2_xfer1_criteria_20261123.json`](../manifests/taiji_u2_xfer1_criteria_20261123.json)（`criteria_sha256` 读前冻结）。**本文零训练、零 check；按常设授权执行，待 owner 回溯。**

## 1. 主体与任务

- `Subject72`（72 单元）；CN 路由 0:24、标准 CN 头；单任务训练（其余头冻结）；
- **CN 组合任务**：8 类颜色名二元组（引擎 `cn_split`）；`seen` = 非 HOLDOUT 颜色对（会话 A 用），`unseen` = HOLDOUT 新颜色对（会话 B 用），两者不相交；
- **会话 A**：seen 对训练 **600 步**；A 末在 seen 测试集（256，独立种子）评分 `A_end`；
- **跨会话**：A 末 `capsule_v2` 保存→加载→应用到同候选副本（前向差 ≤1e-6，身份/恢复一致性），即会话 B 起点；
- **会话 B**：unseen 对训练 **250 步**（每 25 步记录曲线）；`B_warm` = 250 步后 unseen 测试 acc；
- **无 A 基线**：同 seed 全新 `Subject72`（同初始化，无 A 经验）训练同样 250 步，`B_fresh`；
- **保持**：B 训练后 A seen 测试 acc（`A_after_B`），drop = A_end − A_after_B。

## 2. 判据（读前冻结）

| 门 | 判据 |
|---|---|
| G0 会话 A 学会 | `A_end` ≥0.80 |
| G1 迁移水平 | `B_warm` ≥ `A_end` − 0.05 |
| G2 迁移增益 | `B_warm` − `B_fresh` ≥0.05 |
| G3 跨会话保持/身份 | `A_after_B` 相对 A_end drop ≤0.05；capsule 前向差 ≤1e-6；B 起点恢复差 ≤1e-6 |
| G4 预算/纪律 | 单 seed、≤400 CPU、`criteria_sha256` 读前校验、零 check/final |

**出口**：全过⇒`xfer1_verified`；G2 不过⇒`xfer1_no_transfer`；G1 不过（G2 过）⇒`xfer1_level_low`；G3 不过⇒`xfer1_retention_breach`；G0 不过⇒`xfer1_baseline_insufficient`；超帽⇒`budget_stop`；工程异常⇒`engineering_failure`。**不调阈、不改门。**

## 3. 披露

- 合成受控中文（模板化 8 类颜色对），非开放中文；「会话」= 同一主体上的两段训练，不是真实对话；
- seen/unseen 划分沿用既有引擎 HOLDOUT，未改动；
- 单 seed 试点；不签 24-S 正式包/29/final/C/D/J；零 check/final。
