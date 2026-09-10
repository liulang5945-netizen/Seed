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
