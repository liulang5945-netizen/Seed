# 读出算子缺陷与实现规格（U2 身份绑定）

> 日期：2026-11-18｜状态：**缺陷已定位，修补待实现**｜范围：`taiji/u2_subject.py` 的读出通路 ＋ `taiji/u2_pixel_read.py`
> 用途：在 VSCode（有 C/C++ 环境）里实现用。本文只写"问题＋契约＋验收"，不含训练计划。

## 1 缺陷：绑定发生在提问之前

`binding(self, state, tokens, coordinates)`（`taiji/u2_subject.py:446`）**没有 request 参数**。它在 `correct()` 里、请求到达**之前**执行，把 25 个模态 token 压成 `U2Config.objects = 3` 个槽（`u2_subject.py:53`）。

之后 `branch_context`（约 `:1210`）走 `h = read_h(state, goal)` → `o = read_o(state, h)` → `joint_readout([h,o,fast])`。也就是说：**问题只能在这 3 个已合并的槽上加权，无法回到 token 层。**

### 证据三联

| 观测 | 读数 | 说明 |
|---|---|---|
| patch 级（16 token＋patch 中心坐标）外挂头，收敛后 | train 1.000、fit **0.510**（3 体） | 槽/粗粒度下区分不出同列实体 |
| 像素级（256 像素＋精确 x/y）同一匹配算子 | fit 中位 **0.9948**（5 种子，2000 步） | 绑定推迟到提问时刻即可解 |
| `binding` 签名 | 无 request | 绑定与问题无因果通路 |

### 历史

`u2_subject.py` 15 个提交中仅 1 个与读出相关（`133bb99c9` binding v2），改的是"token 怎么并进槽"，不是"问题怎么选 token"。v4→v7 的迭代（metadata、独立 request、correction adapter）全在外围。

## 2 算子契约（已在头级验证到 0.9948，按此实现）

```
输入
  blocks : (16, 48)   observations["rgb"]        # 4x4 patch，块内 4x4 像素 x 3 通道
  coords : (16, 8)    coordinates["rgb"]         # patch 中心 (x,y) 在前两列
  request: (L,) long  请求字节（不喂 one-hot）

展开
  raster   = blocks.view(4,4,4,4,3).permute(0,1,3,2,4,5).reshape(16,16,3)
  pixel_xy = 每个 patch 中心 + 固定子偏移 (-3/32,-1/32,+1/32,+3/32)
             patch 顺序 row-major(4x4)，块内 (i,j) → 像素 (prow*4+i, pcol*4+j)

匹配（余弦 + 可学温度，这是稳定性的来源）
  k   = L2norm(Linear(3→16)(pixel))                      # (256,16)
  txt = Embedding(256,16)(request) + pos[0:L]            # (L,16)
  q   = L2norm(Linear(32→16)([txt.mean(0), txt.max(0)])) # (16,)
  s   = (q · k) * sqrt(16) * exp(log_temp)               # log_temp 是标量参数，初值 0
  s_null = q · L2norm(null)                              # null 是 (16,) 参数，初值 0
  w   = softmax([s, s_null])[:-1]                        # (256,)

读出（线性坐标矩）
  mass = Σw;  mx = Σw·x;  my = Σw·y
  f = [mx/mass, my/mass, mass, max(w), mx, my]           # (6,)
  delta = Linear(6→d)(f)                                 # d = config.d，**零初始化**

注入
  context = joint_readout([h,o,fast]) + delta
```

**必须保持：`delta` 的最后一层零初始化** —— 出生时对父代严格中性（实测 Δ=0.0）。

## 3 接线点（三处，缺一不可）

| 位置 | 作用 |
|---|---|
| `U2Config.readout: str = "patch"` | 开关，只接受 `patch` / `pixel-salience`（已加） |
| `correct()` 内 `tokens = self.encode(...)` 之后 | 存 `pixel_blocks`（detach）、`pixel_coordinates`、`pixel_grid` 到 result |
| `begin_utterance()` | 算一次 `pixel_delta` 存进 child（栅格/坐标/请求一句话内不变） |
| `branch_context()` | `context += state["pixel_delta"]`（无则回退现算） |

## 4 验收（全过才算接线成功）

1. **出生中性**：`readout="patch"` 与 `readout="pixel-salience"`（同权重）首步 log 概率 **Δ 严格 0.0**，支撑集一致（注意掩掉 `-inf`，直接相减会得 NaN）。
2. **红方向**：`out.weight += 1e-3` 后 Δ ≠ 0（否则中性测试是装饰）。
3. **可学**：一次 backward 后 `out` 梯度非零；上游（pixel/text/byte/position）在 **1–3 步优化后**非零。第 0 步上游必为 0，这是零初始化注入的数学必然，不要判成缺陷。
4. **缓存等价**：缓存 `pixel_delta` 与强制重算的首步 log 概率 maxdiff 0.0。
5. **恢复一致**：save → load → 再评估，行级 digest 相同。

## 5 已踩过的坑（都会静默失败）

- `state.get("pixel_grid") or pixel_grid(...)`：张量多元素时布尔求值抛 `Boolean value of Tensor is ambiguous`，必须写显式 `is None`。
- **bundle 优化器口径**：`load_learner` 重建单个 param group，任何两组学习率的优化器都会在读取时抛「参数组数量不同」。要分组学习率必须先升级 bundle schema。
- **零初始化导致第 0 步上游梯度为 0**，已如 §4.3 所述。
- `accept_byte` 原本每字节重算 `goal()`（三层卷积），已改为按 utterance 缓存；改动后 forward 逐位不变，但**梯度图共享方式变了**（同一次 goal 被多次复用，数学上等价）。
- 本机 `torch.compile` **不可用**（inductor 缺 `cl.exe`）；`backend="eager"` 只追踪不融合。VSCode 那边有 C/C++ 环境的话，优先试 inductor，这是 3–8× 的唯一低风险来源。
- 改 `u2_subject.py` 会让 **55 条 `source_pins`（35 个 manifest）失配**，旧 bundle 只能只读、不能 resume。

## 6 两条修补路线（推荐 A）

**A（已实现，待训练验证）：绑定推迟到提问时。** 像素显著度读，问题直接与 256 像素比对，不经过 3 个槽。优点：不需要新增槽，不随实体数膨胀；已在头级到 0.9948。

**B（未实现）：槽随实体数生长。** 保留槽结构，但让 `objects` 成为可增长的量，绑定后仍能在槽内保留实体身份。这与"按需生长"直接对应，但要解决：新槽出生时如何中性、槽身份如何可追溯、槽数增长时旧 bundle 如何兼容。

## 7 一句结论

**算法迭代了 8 轮，读出算子只改过 1 轮，且那一轮不在这条关键路径上。** 22.5.0.A 的 3 体 fit 是第一个真正要求"从观测里选出被问的那个实体"的门，因此它是第一次把这个缺陷照出来的门——不是新出现的问题。
