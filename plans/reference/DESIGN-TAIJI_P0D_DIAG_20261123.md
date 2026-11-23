# P0-D 预注册说明：快慢双时间尺度与迁移验证（压力任务；1 seed、零 check、探索诊断）

> [预注册说明] — 2026-11-23（D-20261123-337）。判据在读前写入脚本 docstring 与本文件（运行前已存在脚本与门）。**探索诊断，不出能力结论；零 check/final。**

## 1. 设置（P0-B/P0-C 压力任务）

base p24 1200 步；零化旧单元对新传感器 32:36 的接线；生长 24 随机出生新生单元；**固定随机适配器**把传感器 one-hot 路由到 base 可传输的通道组（P0-C：路由即够）。新生单元状态 48 维；线性读出。

## 2. 四阶段

1. **快通路（FAST）**：新线性读出仅训练 **8** 次交互（batch 96，真实标签）→ acc_fast；
2. **巩固（SLEEP）**：带标签真实回放，新生单元＋读出训练 **200** 次交互 → acc_sleep；
3. **迁移（TRANSFER）**：撤掉快通路——读出重置为**新的线性头**，仅再训 8 次交互 → acc_transfer（≥0.75 说明功能已进入单元）；
4. **重置单元（RESET）**：新生单元恢复到出生快照、新线性头 8 次交互 → acc_reset；drop = acc_transfer − acc_reset（≥0.15 说明单元承载功能）。

## 3. 判据（读前冻结）

| 门 | 判据 |
|---|---|
| G0 FAST | acc_fast ≥0.75 |
| G1 SLEEP | acc_sleep ≥0.90 |
| G2 TRANSFER | acc_transfer ≥0.75 |
| G3 RESET DROP | ≥0.15 |
| G4 旧能力保持 | 旧 p24 drop ≤0.01 |

**出口**：全过⇒`p0d_verified`；G0 不过⇒`p0d_fast_failed`；G1 不过⇒`p0d_sleep_failed`；G2 不过⇒`p0d_no_transfer`；G3 不过⇒`p0d_units_not_needed`；G4 不过⇒`p0d_retention_breach`。**不调阈、不改门。**

## 4. 披露

单 seed、合成面；适配器固定随机（P0-C 结论）；巩固使用真实标签回放（非蒸馏）；探索诊断；不签 24-S/25-S/29/final/C/D/J。
