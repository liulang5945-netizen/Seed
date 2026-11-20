# 设计草案：专用工作记忆槽（22.X，零训练设计稿）

> 状态：[设计草案/待实现] — 2026-11-20。依据 D-132/133/134：显式循环＋写回的组合形态有深度收益（0.91–0.98）但**单通道均不充分**（状态消融 0.70–0.81、零写回降 0.046–0.083）⇒ 需要**专用记忆架构**让写回通道可独立承载跨步信息。

## 1. 机制定义

新模块 `taiji/u2_memory.py`（不改核心）：

```
class MemorySlots(nn.Module):
    # K 个 d 维寄存器；每步由单位活动经 write head 生成候选
    write(activity, gate) -> memory      # memory ← (1-g)⊙memory + g⊙candidate
    read(memory) -> [B, K*d]             # 拼接到读出输入
```

- **写入**：`candidate = tanh(W_w @ activity)`，`gate = sigmoid(W_g @ activity)`（逐维软门控，可学习）；
- **读出**：`direct` 与 `memory_read` 拼接后进 readout（`readout` 输入维度 +K*d）；
- **重置语义**：首步 memory=0；每步可被 gate 覆盖（学习决定保留/覆盖）；
- **不依赖** `z/a` 隐状态做记忆：评估时可**把隐状态 z/a 每步清零**，只靠 memory 槽跨步传递。

## 2. 任务与协议

- 三步动作序列 `same13=(d1==d3)`，最终步读出预测（同 D-132/134）；
- 训练：CE(step1 方向)＋CE(最终 same13)；
- 臂：self（正常）、**state-ablated**（每步后 z/a 清零，只靠 memory）、zero-memory（清空槽）、shuffled 顺序、oracle（真实 d1 写入槽）。

## 3. 门（预注册）

1. `self ≥ 0.85`（正常路径）；
2. **单通道充分性**：state-ablated `≥0.85`（memory 槽独立承载）；
3. **记忆必要性**：zero-memory 比 self 低 `≥0.15`；
4. **顺序依赖**：shuffled 顺序低 `≥0.15`；
5. **oracle 参考**：`≥0.90`。

五门全过 ⇒ `memory_slot_ready`（有界：单主体、三步、K=2 槽）；否则按失败子条件 partial/negative。

## 4. 预算与出口

- 新 pin：`u2_memory.py`；训练 1,500 步、3 seed、零 check、**价 ≤250 CPU**、阈值不调；
- 出口：全过记有界 verified；门 2 不过记 `memory_single_channel_insufficient`；其余 partial/negative；停止前记录失败子条件。

## 5. 与既有形态区别

| 形态 | 记忆载体 | 单通道充分性 |
|---|---|---|
| D-108 隐状态 | `z/a` | 否（读出不可组合） |
| D-115 写回（两步） | 输入特征维 | 有界 verified（两步） |
| D-132/134 组合 | `z/a`＋写回 | 组合成立、单通道否 |
| **本设计** | **专用寄存器（可学习软写入）** | 待测 |

## 6. 判读边界

即使通过也只证明「K=2 专用记忆槽支撑三步 same13」有界成立，不等于一般多步推理；不改变任务轴/check②。
