# 22.5.W.1.1 设计卡：可学习工作记忆门控（DMS 父获取；非长期/情景/联想记忆）

> 卡号：`22.5.W.1.1`｜日期：2026-11-17｜状态：**冻结但不再作为目标记忆轴**（owner 2026-11-17：目标记忆定位是**情景/联想记忆**；W.1.1 只是**工作记忆前置**，**不训练**；设计/价格保留为历史与前置证据）｜性质：22.5.W 工作记忆父候选的训练卡，不是长期/情景/联想记忆，也不是 U/V 读路径延续。
> **定位（owner 2026-11-17 校正）**：只学一个 **4 槽有界工作记忆**的写入/保持/读出（单 episode、delay 0–8）；不做跨会话存储、巩固、提取、遗忘或来源证据，**不承诺无限上下文**；长期记忆另立 [22.5.X 情景与联想记忆轴](DESIGN-TAIJI_U2_EPISODIC_ASSOCIATIVE_20261117.md)。
> 上游：[W.1 结构记忆结果](RESULT-TAIJI_U2_W1_MEMORY_20261117.md)（`structural_memory_viable`：normal 1.0、lesion 0.53125、delay 0–8）｜[DMS 可行性](RESULT-TAIJI_U2_W_DMS_FEASIBILITY_20261117.md)（原图 `task_not_learnable`）。

## 1 单问题

结构 oracle 证明"有 4 槽持久状态就能在 D=4 后读出样本"；本卡问：

> **把固定写入换成可学习门控、基础图与编码冻结，只训门控与读出，能否在未见组合上取得 DMS 工作记忆父？**

## 2 机制

- `LearnedMemoryGraph`（`taiji/u2_unit_memory_readout.py`）：
  - 状态 `m`（4 槽样本）与 `p`（4 槽测试行属性存在性）；
  - `g_w = sigmoid(W_g [input_summary, m, p] + b_g)`；`c = tanh(W_c [input_summary, m, p])`；`m' = (1-g_w)m + g_w c`；
  - 测试阶段写入 `p = rgb[:, :4].max(0)`（固定存在性）；
  - 读出 `memory_readout([m', p, m'*p]) → 2`（same/different）。
- **中性出生**：`W_g=0`、`b_g=-10`、`W_c=0`、`memory_readout=0` ⇒ 门近关、`m=p=0`、读出 logits 0；基础图行为不变。
- **冻结基础图与编码**；只训 **5 个张量**：`write_gate.weight/bias`、`write_candidate.weight/bias`、`memory_readout.weight`（约 **476 标量**，无其他新增）。
- 数据：`unit-input-dms-v1`，train 64／fit 32（fit 为未见属性×位置组合）；3 seed；2000 更新/seed；Adam lr 0.01。
- 任务损失：DMS 二分类 CE；不加额外监督。

## 3 门与出口

- **父门**（3 seed 全过）：fit（未见组合）**≥0.90**、train **≥0.95**、工作记忆 lesion（样本阶段置空）**≤0.60**。
- 工程：中性出生、可训恰 5 张量、checkpoint 往返、独立恢复、成本 ≤实测帽。
- 出口按序：`budget_stop`→`engineering_failure`→`parent_not_acquired`→`memory_lesion_failed`→`parent_acquired`。**只签 DMS 工作记忆父资格**，不签自然中文/能力/check②/22.6，也不签长期/情景/联想记忆。

## 4 验证与定价（已完成，零训练）

- **smoke** [报告](../../reports/taiji_u2_w11_learned_memory_smoke_20261117.json)：verdict `smoke_passed`、**21/21**、`uncovered=[]`；逐 seed 中性出生读出零、门零、可训恰 5 张量、可训有梯度/冻结无梯度、checkpoint 往返、数据 64/32。
- **定价** [报告](../../reports/taiji_u2_w11_learned_memory_price_20261117.json)：per_update 实测；`measured_cpu = 3×2000×per_update =` **35.1562**；帽 **70.3125 CPU／140.625 wall**（×2.0，设计帽 600／1200 内）。定价自身 36 个用后即弃更新。
- **契约** [manifest](../manifests/taiji_u2_w11_learned_memory_20261117.json)：`status=frozen`、`training_authority=null`、`criteria_sha256=ac268d9f6ef8906d…`、6 枚 source pin；`--run` 授权前抛 `missing_training_authority`。
- **契约测试**：[test_w11_learned_memory.py](../../tests/taiji_native/test_w11_learned_memory.py) **11 passed**（契约冻结/帽、pin、可训恰 5 张量、中性出生、梯度覆盖、checkpoint 往返、run 授权守卫、价格与帽、smoke 全绿、编码/分区、route 响亮拒绝）。

## 5 当前状态与唯一后继

**当前**：设计＋定价冻结；**0 训练**、`training_authority=null`。**唯一后继**：owner 训练授权（3 seed×2000 更新、只训 5 张量、帽 **70.3125 CPU／140.625 wall**）；授权后 `--run`→`--recover`→`--verify`，按 §3 出口结案；授权前 0 训练新增、0 能力、0 父；check② 不读、22.6 不解锁。
