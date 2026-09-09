# M4.V2.R5 预注册：资源归一化的条件模块化

> 起草日期：2026-09-09。本文是对 [M4V2_R4_CLOSURE_DECISION_20260909.md](M4V2_R4_CLOSURE_DECISION_20260909.md) §4 唯一推荐假设的指标/对照/停止线预注册草案，**确认前不写实现、不训练**。确认后本文即为 R5 canary/formal 的唯一判定依据，不得追溯修改。执行顺序仍以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 可证伪假设

> 在相同累计更新预算、峰值内存与训练时间约束下，按内容/状态条件分配可塑容量的模块（conditional module），在未见课程组合 transfer 上优于等量最终参数的预分配（fixed-large），且旧能力保持不劣。

若 conditional module 在资源归一化条件下不能胜过 fixed-large，则"结构增长的价值来自条件分配而非最终参数量"这一假设被否决，M4 结构增长主线收束。

## 2. 臂与对照

| 臂 | 内容 | 角色 |
|---|---|---|
| A fixed-capacity | R3/pressure parent 原样，无新增参数 | 基线 |
| B fixed-large | 等量最终参数从一开始预分配 | **最强对照，不可删除** |
| C conditional module | 内容/状态路由的条件模块；zero-gated、独立 shadow、checkpointable、可回滚 | 候选 |

约束（三条臂全部适用）：

1. 同一 parent owner boundary、同一课程预算、同一 3×3 matrix（model seeds `71/83/97` × course seeds `101/202/303`，与 R4 相同）；
2. 路由输入只允许内容特征、状态（context/activity）、不确定性（surprise）、目标与资源信号；**禁止 evaluator task ID、phase 标签、文件名或任何评估侧泄漏**；
3. C 臂保持 R4 已确认的安全外壳：zero-gated 出生、shadow materialization、candidate-only training、fresh restore、lesion、rollback；
4. 每臂生成绝对能力、parent delta、comparison delta 与资源记录（沿用 `taiji/continual_evaluation.py` 三分离语义）。

## 3. 资源归一化协议

每个 cell 记录并约束以下三项，任一超限该 cell 判无效，不得进入统计：

1. **累计更新预算**：候选/模块的参数更新次数与浮点操作量不超过 B 臂等量参数的同步预算；
2. **峰值内存**：峰值工作集字节不超过预设上限（以 R4 formal 的 1.25 倍为初始上限，预注册后固定）；
3. **训练墙钟时间**：单 cell 训练时间不超过 B 臂均值的 1.5 倍（预注册后固定）。

资源记录进入 versioned report；"更少的早期活跃容量"以 C 臂活跃（非零梯度接收）参数字节数的时间曲线与 B 臂对比来度量。

## 4. 指标（全部预注册，统计只用 versioned report）

1. **未见组合 transfer（主指标）**：9-cell 的 G holdout mean surprise delta（相对 A），C 相对 B 要求 `non-worse ≥ 7/9` 且 mean 更优；S holdout 非劣 `9/9`；
2. **route lesion（因果证明）**：关闭路由后 C 臂 G/S 退化必须显著（G lesion 为负均值），证明路由被真实使用而非常量旁路；
3. **resource-normalized utility**：Δsurprise per 单位累计更新预算与 per 单位活跃容量字节，C 与 B 并排报告；
4. **旧能力保持**：A retention 与 S holdout 三臂 non-worse；
5. **未见 K 组合**：课程序列外的第四 phase（K）作为 transfer 探针，只报告不作为晋级必要条件。

## 5. 停止线

1. **canary 先行**：单 model-seed 小课程 CPU smoke 通过全部技术 Gate（owner 隔离、route lesion 可观测、checkpoint/fresh restore/rollback、资源记录完整）才允许 9-cell formal；任一失败不跑 formal；
2. **formal 判定**：§4 主指标 + 旧能力保持全部满足 → 可宣布"条件模块化优于等量预分配"，进入下一版架构讨论（仍非 A8 晋级）；任一不满足 → 假设否决，M4 结构增长主线收束，剩余路线转 M5 外围；
3. **全程 `can_promote=false`**：晋级讨论是架构层决策，不是本轮能力晋级；
4. **反证不可掩盖**：不调 Gate 输入、不改 birth scale、不引入新语料或外围系统来回避 B 臂反证；
5. **默认路径保护**：R4 shadow 与 conditional module 均不进入默认 parent，失败可整体回滚。

## 6. 与既有资产的衔接

- 复用 R4 的安全外壳、lesion 工具、9-cell 矩阵与 `continual_evaluation` 三分离量尺；
- 复用 M4 固定容量证据链复盘（[M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md](M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md)）的 C2/C3 结论作为本假设的动机：规则轴穷尽、分布主导——条件模块化的价值主张恰好是"按内容/状态分配容量以降低分布适应成本"；
- 量尺审计（`taiji_m4v2_measurement_audit_20260909.json`）已把 R10/R12 历史数值的语义（absolute BPB vs arm-vs-arm）修正，本实验的 delta 全部按新语义计算。

## 7. 确认点（当前唯一下一步）

1. §3 资源上限的初始数值（内存 1.25×、墙钟 1.5×）是否接受；
2. §4 主指标的阈值（`non-worse ≥ 7/9` 且 mean 更优）是否接受；
3. §5 的出口二分（晋级讨论 vs 主线收束转 M5）是否接受。

三项确认后，R5 进入实现顺序：先 conditional module 的 zero-gated shadow 外壳（不改默认路径），再 route learner 的最小实现，再 canary——与 R4 相同的交付节奏。
