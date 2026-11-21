# S1 预注册：同主体单一谱系多能力联合生命流演练（零 check / 零 final）

> 文档状态：[预注册/待跑] — 2026-11-21（D-20261121-230）。上游 [MAINLINE §3A](../MAINLINE.md#3a-新核心执行合同唯一活动s1-与-23-s28-s)、[S0 阶段证据](RESULT-TAIJI_U2_HOLISTIC_ALGORITHM_VERIFY_20261121.md)。**未运行、零训练、零 check/final。**

## 1. 目的

把 S0 的分轴闭环（D-225/227/228）收束到**同一 `SelectivePersistentGraph` 主体、同一条连续生命轨迹**上，验证多能力**联合**成立而非分轴拼凑；这是 `23-S..28-S` 正式合同的唯一准入前置。

## 2. 主体与协议

- **主体**：单一 `SelectivePersistentGraph`（`per_sample_state=True`、`capsule_v2`），**同一实例顺序**走 P-A→P-F；阶段间不重置、不换体、不 fork 评估体。
- **数据**：全部合成受控面（沿用 S0 数据生成器），`train/fit` 分离；**零 `check` / 零 `final`**。
- **种子**：3 个（`20261120/21/22`），逐 seed 独立重跑全生命流。
- **预算**：`<= 2,000 CPU / 3 seed`（含冷载恢复）；超帽 `budget_stop`。

## 3. 阶段与门（读前冻结，不得读后调）

| 阶段 | 内容 | 主门 |
|---|---|---|
| **P-A** | 32 维合法通道、零位置嵌入、双向/最小对立中文 | `masked_bi >= 0.95`；`masked_adv >= 0.90` |
| **P-B** | 2 实体 5 会话流、四类纠错（过时/授权误导/矛盾/来源撤销） | `overall >= 0.95`；`unprotected <= 0.40`；`gain >= 0.20` |
| **P-C** | known→ACT / amb→QUERY / unres→WAIT | 三态路由各 `>= 0.80`；效用 − 最强固定 `>= 0.05`；`brier <= 0.20` |
| **P-D** | 4 类一维遮挡摩擦物理 | 准确率 `>= 0.80`；拔除探测消融降 `>= 0.20` |
| **P-E** | 两代中性生长（16→24→32） | 终态 `>= 0.90`；出生中性 `<= 1e-6`；可逆损伤 `>= 0.20`；**旧技能损伤 `<= 0.02`** |
| **P-F** | `capsule_v2` 冷载恢复 + `t=20` 断点重连 | 逐位 `<= 1e-6`；重连后已学能力 `>= 0.90` |

- **总门**：P-A..P-F 全过 + 旧技能损伤 `<= 0.02` + 全程零 `check/final` ⇒ **`s1_verified`**；
- 任一主门不过 ⇒ `s1_partial`（缺口单列，不得同卡调阈）；任一门因天花板/任务不可学 ⇒ `presentation_failure`；超帽 ⇒ `budget_stop`；工程失败 ⇒ `engineering_failure`。

## 4. 交付

- `reports/taiji_u2_s1_lifeline_20261121.json`（逐 seed/阶段读数、CPU、门）；
- `RESULT-TAIJI_U2_S1_LIFELINE_20261121.md` + D 号；
- 通过后：`23-S.0` 解锁（新 pin、重定价、owner `training_authority`）。

## 5. 不签

不签 `check/final/C/D/J`、模型成型、开放中文、真机、结构线 25–27 重开；`S1` 只是 23-S 准入前置。
