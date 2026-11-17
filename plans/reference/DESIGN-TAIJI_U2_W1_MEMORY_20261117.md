# 22.5.W.1 持久记忆机制设计卡（结构 oracle ＋可学习门控）

> 卡号：`22.5.W.1`｜日期：2026-11-17｜状态：**结构探针完成＝`structural_memory_viable`**（normal 1.0/1.0/1.0、lesion 0.53125、delay 0–8 全保持；见[结果卡](RESULT-TAIJI_U2_W1_MEMORY_20261117.md)），**待 owner 裁决 W.1.1 可学习门控设计＋定价**｜性质：22.5.W 父获取的机制前置，不是训练卡。
> 上游：[DMS 可行性](RESULT-TAIJI_U2_W_DMS_FEASIBILITY_20261117.md)（`task_not_learnable`；normal/lesion/test-only 同值 0.56–0.66，delay=0 也不行 ⇒ 状态不保留样本）。
> 纪律：本卡只做**零训练结构探针**；训练另立 `22.5.W.1.1` 并先定价。

## 1 单问题

DMS 的样本在 sample→delay→test→query 链后没有留在可读状态里，**delay=0 也不行**。本卡问：

> **如果在单位图里加一个最小的持久记忆状态 `m`（写入-保持-读取），样本能否在 D=4 后线性可读、lesion 是否塌到 0.60 以下？**

这是**结构存在性**问题，不是"模型已经学会记忆"；先证明路径存在，再谈学习。

## 2 机制

### 2.1 结构 oracle（本卡探针用，固定规则、0 新参数）

- 新增状态 `m ∈ R^4`（样本属性 one-hot 槽），初始为零；
- **写入**：当 question 阶段标记为 `sample`（`question[:,0]=1`）时，`m = rgb[:, :4].mean(0)`（样本属性 one-hot 在 16 列上取均值）；
- **保持**：delay／test／query 阶段 `m` 不变；
- **读取**：在 query 阶段，特征 `[m, present, m*present]`（12 维；`present` 为测试行各属性是否出现，`m*present` 是"样本属性在测试行出现"的合取证据）；
- 线性/凸 CE 头（2 类）在 train 上拟合、fit 评分；**图参数 0 更新**。
- 这是**手写记忆 oracle**，只证"信息路径存在"；它不代表模型能自己学会写入/保持/读取。

### 2.2 可学习门控（下一卡 `22.5.W.1.1`，本卡不定价）

- 把固定写入替换为 `g_w = sigmoid(W_g [input_summary, m])`，候选 `c = tanh(W_c [input_summary, m])`，`m' = (1-g_w)·m + g_w·c`；读取头 `[m, present, m*present] → 2`；
- 中性出生：`g_w=0`、`m=0`、读出零初始化 ⇒ 出生不改变旧行为；
- 只训 `W_g/W_c/读出`，值侧/编码冻结；3 seed；父门见 §4。

## 3 结构探针（本卡执行）

- 数据/编码复用 `22.5.W` 的 `unit-input-dms-v1`（train 64／fit 32、属性×位置 disjoint、标签各半、编码 oracle 100%）。
- 3 个出生图 seed；每 seed：
  - normal（写入样本）→ `m` 应为样本 one-hot；特征头 fit 应 **≥0.95**；
  - lesion（样本阶段置空，不写入）→ `m=0`，头应 **≤0.60**；
  - test-only（无 sample/delay）→ `m=0`，头应 **≤0.60**；
  - 延迟扫描 0/1/2/4/8：normal 不塌、lesion 不升。
- **仪器**：编码 oracle 100%、分区 disjoint、标签各半、图参数不变、正对照 1.0、置换标签 ≤0.50；`uncovered=[]`。
- **预算**：自设帽 **120 CPU／240 wall**；0 图优化器步、0 权重写入、封存 0。
- **出口**：`structural_memory_viable`（3/3 normal ≥0.95 且 lesion/test-only ≤0.60）／`memory_path_failed`／`instrument_invalid`／`budget_stop`。

## 4 训练卡 `22.5.W.1.1`（只在结构探针通过后立项）

- 单变量：固定写入/保持 → 可学习门控；值侧与编码冻结；只训门控与读出。
- 父门：fit ≥0.90、train ≥0.95、未见组合 ≥0.85、记忆 lesion ≤0.60、独立恢复、成本 ≤实测帽。
- 预算：先定价（结构探针给每更新代理价）；3 seed × 2,000 更新以内；不对未测剂量承诺。
- 出口：`parent_acquired`／`parent_not_acquired`／`memory_lesion_failed`／`engineering_failure`／`budget_stop`；只签父资格。

## 5 能签与不能签

**能签**（本卡）：结构记忆路径是否存在；DMS 在 D=4 下是否能被该路径线性读出；仪器与成本。**不能签**：模型已经会记忆；可学习门控一定能学到；任何能力/父/check②/22.6；结构 oracle 不是能力证据。

## 6 当前状态与唯一后继

**当前**：设计完成；**0 训练**。**唯一后继**：结构探针（`MemoryRouteBGraph`，固定写入/保持/读取）→ 按 §3 出口分类 → 若 `structural_memory_viable`，写 `22.5.W.1.1` 冻结契约＋定价，请 owner 授权训练；若 `memory_path_failed`，关父获取线或换机制。裁决前 0 训练新增、0 能力、0 父；check② 不读、22.6 不解锁。
