# 22.5.2 Q2：Route-A 与 Route-B 唯一前向与迁移差集（源码级，零训练）

> 文档状态：[证据/设计差集] — 2026-11-20，22.5.2 S0/S1 交付物。依据 `taiji/u2_neural_units.py`（`_BaseGraph`、`RouteAGraph`、`RouteBGraph`）与 P3.9 冻结实现（19/19 检查）。**本文只列源码事实，不由行为读数代替；不签架构优劣。**

## 1. 共享核心（两路线相同）

| 组件 | 形状/作用 |
|---|---|
| 合法输入编码 | 6 个通道各 `Linear(8,8)`（初值单位阵）→ `input_projection` 8×8 → 按组复制成 `[T,G,8]` |
| 单位群 | `N=30`（groups=6×5）；`r[N,8]` 驱动、`w[N,N]`＋`edge_mask` 入边、`c/eta/b[N]` 衰减/速率/自抑制、`p[8,N]` 单位到 hidden 的投影 |
| 持续状态 | `z,a[N]`＋`H[G,8]`；`internal_tick` |
| 生命周期/门 | `active_gate(gates)` 与 `unit_activity = gate*z`；两路线共用 |
| 单位递推（逐 tick） | `drive=(x*r).sum(-1)`；`incoming=(gate*z)@w.T`；`v=drive+incoming-b*a`；`z=c*z+(1-c)tanh(v)`；`a=(1-eta)a+eta*z` |

## 2. Route-A 唯一前向（`RouteAGraph`）

```
per tick t:
  drive_g[t,g,h] = einsum(w_g[g,h,d], x[t,g,d]) + contribution[t,g,h]   # 单位贡献经 p 汇入 H
  H_t = (1 - ell[g]) * H_{t-1} + ell[g] * tanh(drive_g[t,g,:])
readout = h_readout(mean_t H_t)
```

- **A 独有参数**：`w_g[G,8,8]`（384）、`ell[G]`（6）、`h_readout[8,out]`；
- **读出路径**：`units → p → contribution → H → h_readout`；`H` 在读出路径上，**切断/扰动 H 会改变输出**；
- **状态语义**：`z/a` 与 `H` 是两层持续状态（H 为固定中介）。

## 3. Route-B 唯一前向（`RouteBGraph`）

```
per tick t:
  gated_a = a_trace * gate
  state_view = mix0 * activity + mix1 * gated_a          # mix = state_mix (2,)
  projected = p.T @ state_view                            # [hidden]
  direct[t] = projected @ query.T                         # [query_slots]
readout = readout_linear(direct)                          # 逐 tick
```

- **B 独有参数**：`query[query_slots,8]`（默认 4×8）、`state_mix[2]`、`readout[query_slots,out]`；
- **读出路径**：`units → gated activity → p → query → readout`；**H 不在读出路径**，`cut_h` 后输出逐位相同（P3.9 已验证）；
- **状态语义**：`z/a` 是唯一持续计算状态，`H` 为 scratch。

## 4. 迁移差集（参数/状态逐项）

| 项 | 共享可迁移 | A 独有 | B 独有 | 迁移规则 |
|---|---|---|---|---|
| `units.*`（`r,p,c,eta,b,w,edge_mask`） | ✅ 同名同形，可逐值复制 | — | — | 复制后必须重跑合法输入门与梯度覆盖；**`p` 的功能语义不同**（A 经 H，B 直读），不得声称行为等价 |
| 输入编码 | ✅ `input_encoders`＋`input_projection` | — | — | 直接复制 |
| 状态 | `z,a` 可复制 | `H` 在读出路径 | `H` 仅 scratch | A→B：保留 z,a,H；B→A：H 置零后由 A 前向重算 |
| 读出参数 | — | `w_g,ell,h_readout` | `query,state_mix,readout` | 跨路线无逐值映射；只能随机初始化（新研究候选）或另行学习线性映射（未做不假设） |
| 身份/谱系 | 需新图 schema 与恢复规格 | — | — | 无映射时必须登记**新身份**，不得伪称同一主体连续成长（路径备案 §3.1） |

**结论（差集）**：两路线共享单位群与状态递推，差异集中在**读出组织**（A：单位贡献→固定 H 中介→线性读；B：门控活动→直接投影→查询）。迁移只对 `units.*`、输入编码与 `z/a` 成立；读出参数**不可逐值迁移**，跨路线行为等价**未被证明也不假设**。

## 5. 对 FLOPs 估计器的输入

- 共享逐 tick：`drive N*D`＋`incoming N*N`＋`contribution hidden*N`；
- A 追加逐 tick：`w_g G*8*8`＋`H 更新 G*8`；读出一次 `hidden*out`；
- B 追加逐 tick：`p 投影 hidden*N`＋`query query_slots*8`＋`readout query_slots*out`；
- 估计器须把上述逐项列出、按 multiply-add=1 FLOP 计、反向按 2× 前向计；不把门/加法逐项漏计。

## 6. 边界

- 本文不读 check/final、不训练、不签任何能力；
- 行为差异只在 22.5.2 预注册门（Δ≥0.05、95% 下界>0）下报告；未过则记 `inconclusive`；
- P3.9 的 `compute_parity_unavailable` 窗不因本文被改写，只作为教训引用。
