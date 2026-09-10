# M4.V2 B3-K C-entry parity 口径修正与修复路线（冻结）

> 冻结日期：2026-09-10。前置：[widened-candidate 设计](M4V2_B3_K_C_WIDENED_CANDIDATE_DESIGN_20260910.md)与构建报告（8/9，model31/course1 零差分）。本文冻结两件事：(1) 更新步数的诚实口径；(2) 零差分的唯一修复路线。冻结前不进入 sealed 评分、不训练、不改 distinct 门。

## 1. 口径发现（机器证据）

| checkpoint | training_steps | 含义 |
|---|---:|---|
| parent K1 / K2（model17） | 2080 / 5040 | worker artifact 构建时的既有训练 |
| fixed-large replica0 K1 / K2 | 2083 / 5043 | **新增仅 3+3 = 6 步**（3 train experiences × 1 epoch） |
| widened 每通道 K1 / K2 | 2083 / 5043 | **新增仅 3+3 = 6 步**（同流） |

结论：parity 硬门的「14,252 实际更新步数」中 **14,240 是 parent 继承步数**；两臂的真实新增均为 **12 步/cell**（fixed-large 2 replicas × 6；widened 2 channels × 6）。旧口径把继承与新增混在一个计数器里，使「fixed-large 训练 14,252 步 vs candidate 6 步」的表述夸大了两臂学习预算差异的真实语义。两臂真实新增步数**恰好相等（12 = 12）**——widened candidate 在诚实口径下已经与 fixed-large 同预算。

## 2. 新口径（冻结）

1. **分离计量**：`inherited_steps`（parent checkpoint 的 `training_steps` 起点，逐 worker）与 `post_training_steps`（训练后计数器）分开记录；`new_update_steps = post − inherited`，逐 worker × 逐通道/replica 报告。
2. **双臂同口径重述**：fixed-large 与 widened candidate 的 parity 比较只使用 `new_update_steps`；旧口径的 14,252 保留为历史观测值，不覆盖旧 JSON。
3. **只读重述审计**：从现有 checkpoint 的 training_steps 反推 fixed-large 与 widened 的新增步数，生成版本化重述报告（`taiji-m4v2-b3-k-c-parity-caliber-revision-v1`）；不重跑训练。
4. **硬门替换**：`new_update_steps` 双臂精确相等替换旧的 total-14,252 门；参数字节 ±1% 门、逻辑 checkpoint 门、distinct 门不变。

## 3. 修复路线（冻结，唯一选择）

**选择：更强的新增预算课程——每 cell 引入 150 个 parent 未见过的新 experiences（新 task_seed workspace 变体生成新文件内容），双臂消费同一流。**

- **对零差分对症**：model31/course1 零差分的根因是「parent 在 3 个旧 experience 上已收敛 → delta≈0」。新 experiences 必然产生非零 delta（parent 从未见过）→ 顺序通道的分化恢复，distinct 门可诚实通过。
- **双臂预算自动相等**：每 experience 每实例 K1+K2 各 1 步；fixed-large 2 replicas × 2×150 = 600 新增步/cell；widened 2 channels × 2×150 = 600 新增步/cell——**任何 n 下双臂新增自动相等**，冻结 `n_new = 150`（新增 600 步/cell：K1 300 + K2 300）。
- **不引入失败 episode**：失败准入语义是第二个变量，与本轮修复正交；留作后续可选增强，避免单变量纪律被破坏。
- fixed-large 按 manifest 的 `candidate-upward` 与「do_not_shrink_fixed_large」原则同步消费新课程（不重训 parent；replica 从同一 parent 出发消费同一新流）。

## 4. 停止线

- 重述审计发现新增步数双臂不等 → 停，归因 harness 计数，不放宽；
- 新课程下仍出现零差分 cell → 停，归因新 experience 的生成多样性（workspace 变体是否真实改变 percept 特征），不通过调学习率掩盖；
- 新 experiences 不得触碰 holdout/sealed 词汇边界（train/holdout input digest 泄漏检查沿既有语料合同）；
- sealed 评分在修复后的 formal 预注册确认前不读取；`can_promote=false` 固定。

## 5. 产物顺序

1. 只读口径重述审计报告（含 §1 数字链）；
2. 新增预算课程 manifest v2（n_new=150，双臂同流，K3-anchored 排序见证延续）；
3. 双臂重跑 widened/fixed-large build → 三硬门 + distinct 门核验；
4. 通过后才进入修复后的 C-entry parity formal 预注册（sealed 评分判据另行冻结）。

## 6. 归因修正（2026-09-10，机器证据推翻 §3 的原归因）

本节由用户质疑触发，经四步诊断实验证伪原归因并钉死真机制。**§3 中「对症」段的原归因（「parent 在课程语料上已收敛 → delta≈0 → 顺序通道无信息」）是错的**，修复路线 n_new=150 的机制理由需要如下修正（方向不变，理由更换）。

### 6.1 证据链（全部可复现）

1. **证伪「收敛」**：build 报告中 model31/course1 的通道内 delta norms 为 K1 `1.9645e-01`、K2 `3.0298e-02`——非零且与其他 cell 同量级；单步 fit 的 loss 亦非零（fact_loss `0.0190`）。parent 没有收敛，梯度一直存在。
2. **delta 等价类**：course1（multiset A+A+B）的 exp0 与 exp1 从同一 parent 出发单步 fit 后的 state_dict digest **逐位相同**（`b1462293e7b6617c`），但二者 input_digest 不同（`4e79e1cd…` vs `39cc349a…`）——差异只在掩码不可见特征（byte_length 等）。K1.1 类型化绑定下，两个 A 型 episode 的**掩码可见学习信号完全等价**：fact head 只读因果特征列（掩外列权重恒零），goal/content head 读 teacher-forced 真值 facts——两者可见输入与目标相同。
3. **决定性序列实验**（model31 parent，course1）：序列 `(exp0, exp1, exp2)` 与 anchored 排列 `(exp1, exp0, exp2)` 的**每一步中间 digest 完全相同**，最终权重逐位一致；控制组 `(exp0, exp2, exp1)` 则每步分化。即 exp0/exp1 对**任意当前权重**可逐位互换。
4. **model31/course1 的 anchored 排列恰为 `[1,0,2]`**——只交换这对等价 experience。「非恒等排列」≠「非等效排列」：`anchored_permutation` 的防恒等兜底只检查字典序恒等，不检查 delta 等价类上的等效性。这是零差分的直接机制。

### 6.2 修正后的根因陈述

> model31/course1 的通道差分精确为 0，不是因为 parent 无梯度，而是因为课程 target multiset `A+A+B` 中两个 A 型 experience 构成 **delta 等价类**（类型化绑定使它们在掩码可见语义下是同一个学习信号），而该 cell 的 anchored 排列 `[1,0,2]` 恰好只交换等价类成员——排列在浮点上等效恒等。缺陷在**排列选择的等价类盲区**，不在课程预算，也不在 parent 状态。

### 6.3 衍生事实（同样须上报）

- **三个 model seed 的 parent K worker 权重完全相同**（state_dict diff norm = `0.0`；checkpoint digest 的差异全部来自 metadata）。因此 9-cell 矩阵的 model 维度对 K workers 没有独立性——真正的变化源只有 course_seed（课程与排列）。这触碰 C 阶段合同「重复同一已训练 worker 的运行不能充作独立模型训练样本」的红线：当前矩阵的 9 个 cell 不是 9 个独立模型样本。parity formal 的统计解释必须按此修正（独立样本数实际为 3 个 course）。
- 8/9 的通道差分（`4e-3~9e-3`）是真实的顺序效应信号（控制组证实），但幅度仅为 delta norm 的 2.5%–7.5%——在线局部 delta 的顺序效应是二阶小量，该设计的分化能力天然有限，distinct 门只能证明「非退化」，不能证明「强分化」。

### 6.4 对修复路线的修正

n_new=150 的方向保留，但机制理由更换为：**新 experiences 的掩码可见学习信号必须互异（打破等价类），且排列空间 150! 下「等效恒等」概率可忽略**。同时新增两条硬前提：

1. manifest v2 必须声明「新 experiences 的目标 multiset 不含重复类型」——否则 A+A+A… 会重新制造等价类（与 git 历史上 `make K target diversity tensor-aware` 的教训同源）；
2. parity formal 的统计单元改为 course（独立样本 n=3），model 维度降级为排列见证因子；或者补建互异的 per-model parent K workers（成本更高，另行决策）。

`anchored_permutation` 的等价类盲区修复（排列约束：至少移动一个非等价成员）作为 harness 防线保留，但它治标；课程目标互异才治本。
