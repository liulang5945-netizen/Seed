# LC-1 预注册：单位生命周期——可逆休眠/唤醒、经保持验证的退役与回收再利用（1 seed、≤400 CPU、零 check）

> [预注册] — 2026-11-23（D-20261123-306）。objective 神经结构要求「支持继承式出生、分化、整合、**可逆休眠**及**经保持验证的退役/回收**」；既有 S 线只验过出生/生长（F1c/F1e）与 capsule 恢复，**休眠/唤醒/退役/回收未执行过**。本卡在 F1e/UQ-1 同款候选上补齐该条款。判据清单：[`taiji_u2_lc1_criteria_20261123.json`](../manifests/taiji_u2_lc1_criteria_20261123.json)（`criteria_sha256` 读前冻结）。**本文零训练、零 check；按常设授权执行，待 owner 回溯。**

## 1. 主体（同一候选 seed 20261120，结构与 UQ-1 正式运行一致）

- `Subject72`（72 单元、12 群×6，`cross_group_edges=True`）；p24 块 24:48、p27 块 48:72；
- **分块隔离**：训练前把两块入边分别限制在本块＋输入（w/write_state 跨块列清零），输入混用在 p24 后冻结（沿用 D-305 的 wd=0 修正：Adam weight_decay 会穿过被掩码零梯度行）；
- 每任务 48 维专用头（p24u 读 24:48、p27u 读 48:72）；p24 训 500 步、p27 训 600 步；测试集与训练流不同种子（各 256）。

## 2. 生命周期实验（同一候选上顺序执行）

1. **基线**：A0=p24 测试 acc、B0=p27 测试 acc、capsule 前向差；
2. **休眠**：p27 块整体置 `dormant` → 测 p27（应失能）与 p24（应不变）；p24 块 12 单元置 `dormant` → 测 p24 剂量反应（应下降）与 p27（应不变）；
3. **唤醒**：恢复 `mature` → 两任务 acc 应与基线**逐值相同**（可逆性），capsule 差 ≤1e-6；
4. **退役（经保持验证）**：p27 块置 `retired` → 验证 p24 保持（drop ≤0.001）、p27 失能、参数仍在 capsule 中可恢复（差 ≤1e-6）、lifecycle 计数正确；
5. **回收再利用**：把退役 p27 块参数重新初始化（新随机种子）＋`unit_generation +=1`＋置 `immature`，作为**新任务通路**（修订来源纠正，8 次交互、96 批、标签反馈）训练，再提升 `mature`；测新任务 acc、p24 保持、代际谱系、capsule、promotion 前向不变；并将回收块置 dorm/anew 验证新任务对回收块**因果依赖**。

## 3. 判据（读前冻结）

| 门 | 判据 |
|---|---|
| G0 基线胜任 | A0 ≥0.70 且 B0 ≥0.70 |
| G1 休眠有效 | p27 休眠后 acc ≤0.30 且 p24 drop ≤0.01；p24 半块休眠后 p24 drop ≥0.05 |
| G2 可逆唤醒 | 唤醒后 acc 与 A0/B0 逐值相等（≤1e-6）且 capsule ≤1e-6、promotion 前向 ≤1e-9 |
| G3 退役保持 | 退役 p27 后 p24 drop ≤0.001、p27 acc ≤0.30、capsule 差 ≤1e-6、lifecycle retired 计数=24 |
| G4 回收再利用 | 新任务 8 交互 acc ≥0.75；p24 保持 drop ≤0.001；回收单元 generation 增量=1；capsule ≤1e-6；回收块休眠后新任务 acc ≤0.30 |
| G5 预算/纪律 | 单 seed、≤400 CPU、`criteria_sha256` 读前校验、零 check/final |

**出口**：全过⇒`lc1_verified`；G2 不过⇒`lc1_dormancy_irreversible`；G3 不过⇒`lc1_retirement_breach`；G4 不过（其余过）⇒`lc1_recycle_partial`；G1 不过⇒`lc1_dormancy_weak`；G0 不过⇒`lc1_baseline_insufficient`；超帽⇒`budget_stop`；工程异常⇒`engineering_failure`。**不调阈、不改门。**

## 4. 披露

- 合成面；`dormant/retired` 的单位状态被 `active_mask` 置零（休眠期不保留逐样本状态）；「可逆」指**参数/功能/身份**可逆（唤醒后逐值恢复），不指休眠期状态保留——本卡同时记录该边界；
- 回收为实验层重初始化＋代际计数，不声称生物分化；回收会放弃被回收块原任务能力（p27），旧能力保持只对**未回收块（p24）**验证并如实披露；
- 单 seed 试点；不签 24-S 正式包/25-S/final/C/D/J；零 check/final。
