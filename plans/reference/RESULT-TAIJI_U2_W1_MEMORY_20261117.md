# 22.5.W.1 结构记忆探针结果（`structural_memory_viable`）

> 卡号：`22.5.W.1`｜日期：2026-11-17｜正式判定：**`structural_memory_viable`**｜性质：**零训练结构 oracle**，不是学习结果、不是能力声明。
> 设计：[DESIGN-TAIJI_U2_W1_MEMORY_20261117.md](DESIGN-TAIJI_U2_W1_MEMORY_20261117.md)｜模块：[taiji/u2_unit_memory_readout.py](../../taiji/u2_unit_memory_readout.py)｜探针：[probe_u2_w1_memory_20261117.py](../../scripts/training/probe_u2_w1_memory_20261117.py)｜报告：[reports/taiji_u2_w1_memory_20261117.json](../../reports/taiji_u2_w1_memory_20261117.json)。
> 纪律：**0 图优化器步、0 权重写入、只读合成 train/fit、封存 0**；实耗 **16.1875 CPU／16.485 wall**（自设帽 120／240 内）。

## 1 机制

`MemoryRouteBGraph` 在单位核心 B 图上加一个 **4 槽样本记忆 `m`**：

- **写入**：sample 阶段 `m = rgb[:, :4].mean(0)`（样本属性 one-hot）；
- **保持**：delay／test／query 阶段 `m` 不变；
- **读取**：query 阶段特征 `[m, present, m*present]`（12 维），线性/凸 CE 头 → 2 类；
- **无新参数、无梯度穿过写入**：这是手写结构 oracle，只证信息路径存在。

## 2 读数（3 个出生图 seed；fit 32）

| seed | normal fit | lesion fit | test-only fit | 正对照 | 置换标签 |
|---|---|---|---|---|---|
| 2026111701 | **1.0** | 0.53125 | 0.53125 | 1.0 | 0.0 |
| 2026111702 | **1.0** | 0.53125 | 0.53125 | 1.0 | 0.0 |
| 2026111703 | **1.0** | 0.53125 | 0.53125 | 1.0 | 0.0 |

**延迟扫描（normal／lesion）**：delay **0/1/2/4/8** 全部 **1.0／0.53125**——样本在任意延迟后仍可线性读出；不写入/不呈现样本时回到机会带。

**仪器**：编码 oracle 100%、分区 disjoint、标签各半、`m` 正常时逐位等于样本 one-hot、lesion 时为零、参数不变、正对照 1.0、置换 0.0；`uncovered=[]`。

## 3 判读

- **信息路径存在**：只要图里有一个 4 槽持久状态，样本在 D=4（乃至 D=8）后仍可被线性读出；当前单位核心 B 图缺的正是这个状态（W 可行性 normal/lesion/test-only 同值 0.56–0.66）。
- **不是学习结果**：写入/保持/读取是手写规则，**0 图优化器步**；它证明"可解"，不证明"模型能学会"。
- **范围限制**：合成 DMS、`unit-input-dms-v1`、3 个出生图 seed；不证明自然中文/多模态/物理认知，不证明可学习门控一定能学到。

## 4 能签与不能签

**能签**：结构记忆路径可行；DMS 在 D=4 下可由该路径线性读出；lesion/test-only 控制有效；仪器与成本。**不能签**：模型已经会记忆；可学习门控一定能过；任何能力/父/check②/22.6；结构 oracle 不是能力证据。

## 5 唯一后继

写 `22.5.W.1.1` **可学习门控**的冻结契约＋定价（零训练设计，训练另请授权）：

- 固定写入 → `g_w = sigmoid(W_g [input_summary, m])`、候选 `c = tanh(W_c [input_summary, m])`、`m' = (1-g_w)m + g_w c`；读出 `[m, present, m*present] → 2`；
- 中性出生（`g_w=0`、`m=0`、读出零初始化）；只训门控/读出，值侧/编码冻结；3 seed×2,000 更新以内；
- 父门：fit ≥0.90、train ≥0.95、未见组合 ≥0.85、记忆 lesion ≤0.60、独立恢复、成本 ≤实测帽；
- 先定价，再请 owner 授权训练；授权前 **0 训练新增**、0 能力、0 父；check② 不读、22.6 不解锁。
