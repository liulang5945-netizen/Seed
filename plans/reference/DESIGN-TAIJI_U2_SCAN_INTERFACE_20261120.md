# 设计草案：batch×time 分离的扫描接口（22.X，零训练设计稿）

> 状态：[设计草案/待实现] — 2026-11-20。解决 D-20261119-105 披露的工程限制：`RouteBGraph/NeuralUnits` 的 `inputs [T,G,D]` 把 **batch 当 time**、状态在批内样本间连续携带，无法表达「每样本独立 T 步递归」。本接口为**新模块**，不改核心 `taiji/u2_neural_units.py`。

## 1. 接口契约

新增 `taiji/u2_scan.py`：

```
class UnitScan:
    def __init__(self, graph): ...          # 包住 RouteBGraph（只读引用 units/readout/query/state_mix）
    def run(self, channels: dict[str, Tensor[B,T,8]], gates: Tensor[B,T,N],
            state: dict | None) -> dict     # activity [B,T,N], readout [B,T,out], state
```

- 逐 tick 更新：`drive = x_t @ r.T`、`incoming = (g_t*z) @ (w*edge_mask).T`、`z = c*z + (1-c)*tanh(v)`、`a = (1-eta)*a + eta*z`；
- 读出：`projected = p @ (state_mix0*activity + state_mix1*a*g)`，`direct @ query.T`，`readout(direct)`（与核心 forward 同一公式）；
- 状态 `{z,a}` 形状 `[B,N]`，**样本之间零交互**；`None` 时按 `graph.initial_state()` 广播初始化。

## 2. 验证点（实现卡必须先过，零 check）

1. **T=1 等价**：`run(B=1,T=1)` 与 `graph.forward` 单步输出差 ≤1e-6；
2. **批次独立**：同两样本拼接 run 与分别 run，逐元素差 ≤1e-9（无跨样本泄漏）；
3. **顺序不变**：batch 置换后按逆置换取回，逐样本结果不变；
4. **状态携带**：同一样本 tick1 输入不同时 tick2 活动不同（状态确实跨 tick）。

四条全过才允许重跑 D-104 的四门顺序递归卡；否则记实现缺陷并修正。

## 3. 预算与门

- 实现＋验证 ≤60 CPU（零训练）；随后重跑顺序递归卡 ≤200 CPU（3 seed、四门同 D-104，不调阈值）。
- 回归：现有 T=1 路径（pretrain 脚本）结果不变；pytest 6 项与报告条件门保持绿。

## 4. 风险与出口

- 风险 1：与核心公式不一致（读出/门控顺序）→ 验证点 1 抓住，修正后重跑。
- 风险 2：性能过慢（逐 tick Python 循环）→ 限制 T≤4、B≤64；记录 CPU。
- 风险 3：状态初始化语义与 `graph.initial_state()` 不同 → 用同函数广播。
- 出口：验证通过⇒实现卡结案，重跑顺序递归；顺序递归仍负⇒如实记 negative 并收口该形态。
