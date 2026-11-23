# 设计说明：单元级原始输入适配器（extra input adapter）

> [设计说明] — 2026-11-23（D-20261123-354）。代码：`taiji/u2_selective_units.py`（`SelectiveNeuralUnits.attach_extra_` / `forward(..., extra=)` / `SelectivePersistentGraph.forward(..., extra=)`）；smoke 回执 [report](../../reports/taiji_adapter_smoke_20261123.json)。

## 1. 动机

P0-B/P0-Q 证明基座编码会**抑制未用通道组**（32:36、40:44）。若新证据来源（如中文修订标记）只存在于这些通道，旧表示与新生的 **write_in（读 base 特征）** 都读不到它——任务要么不压力（放进字符空间，P0-O 首试），要么不可学（放进抑制组，P0-B）。需要一条**单元级、可学习的原始输入通路**，让新生单元绕过 base 特征的抑制，同时不改变旧单元行为。

## 2. 数学定义（逐单元）

对单元 $i$，原写门输入为
$u_i(t) = W^{\text{in}}_i x(t) + W^{\text{st}}_i z(t) + b_i$，
加入原始输入 $e(t) \in \mathbb{R}^{D_e}$（如通道 32:36 的 one-hot）后
$u_i(t) \leftarrow u_i(t) + V_i\, e(t)$，
其中 $V \in \mathbb{R}^{N \times D_e}$ 为**逐单元**可学习矩阵，写门 $g_i(t)=\sigma(u_i(t))$ 不变。
**零初始化** $V=0$ 时前向与原来逐值相同（smoke E0 差 0.0），不改变任何既有行为。

## 3. 生命周期与保护

- 参数形状 $(N, D_e)$，按单元行切片；`mask_newborn_grads` 已把 `extra_proj.weight` 纳入 `UNIT_ROW_PARAMS`（旧单元行梯度置零）⇒ 只有新生单元能学习新通路。
- `budding_snapshot` / `load_snapshot` / capsule 的逐单元行切片对新增参数天然适用。
- 适配器可随单元休眠/唤醒（参数保留，功能随单元活动掩码）。

## 4. Smoke 结果（`adapter_smoke_verified`）

压力任务（出生 `write_in`/`r` 置 0，新生单元原本读不到任何输入）中，仅经原始适配器传入 4 类 one-hot：
- 零初始化前向差 **0.0**；
- 40 次交互训练 → **1.000**（write_in/r 仍为零，功能完全来自新通路）；
- 清零适配器权重 → 0.258（**drop 0.742**）⇒ 功能由新通路承载。

## 5. 边界

零初始化 + 线性投影；未做多通道/非线性适配器；不改变 base 图；不签 24-S/25-S/29/final/C/D/J。
