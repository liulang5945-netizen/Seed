# M4.V2.R4 收束与下一版容量假设

> 记录日期：2026-09-09。本文是 R4 formal 的证据解释与架构停止线，不是新的晋级 Gate，也不改写原始报告。执行顺序仍以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 决策

R4 以“技术闭合、结构不晋级”收束：

- R4 的候选确实进入了 native predictive observation/credit 主路径；shadow materialization、checkpoint、fresh restore、rollback、owner 隔离和 lesion 因果链成立。
- birth homeostasis 修复了候选 mapping 的量级失配，pressure candidate 相对其 parent 的因果贡献为正。
- 但在同一 parent、同一课程预算、同一 3×3 model/course seed 矩阵下，pressure-driven growth 没有稳定优于从一开始就等量预分配的 `fixed-large`，尤其在 G 课程上仍平均劣化。
- 因此 `can_promote=false` 保持；默认能力仍使用 R3/pressure parent，R4 candidate 仍是独立 shadow；不得进入 R5 learned router。

这不是“结构成长无效”的总论断，而是对当前候选、当前路由、当前单一 residual population、当前资源预算和当前 Gate 的精确否决。

## 2. 可复核证据

来源：

- [R4 shadow canary](../../reports/taiji_m4v2_r4_shadow_canary_20260909.json)
- [R4 shadow formal](../../reports/taiji_m4v2_r4_shadow_formal_20260909.json)
- [M4 fixed-capacity evidence review](M4_FIXED_CAPACITY_EVIDENCE_REVIEW_2026_09_09.md)

formal 矩阵固定为 model seeds `71/83/97` × course seeds `101/202/303`，共 9 cells；所有 arm 共享 parent owner boundary、课程预算和 pressure contract，技术 Gate 全部通过。

| 比较 | G 结果 | S 结果 | 解释 |
|---|---:|---:|---|
| pressure candidate lesion | mean `+0.003081`；`6/9` better，`9/9` non-worse | mean `+0.080387`；`9/9` better/non-worse | 候选对自身结果有因果贡献 |
| pressure vs R3 fixed-capacity | `9/9` better；mean surprise delta `-0.019313` | `9/9` better；mean `-0.086353` | 生长候选不是空壳，能改善固定 parent |
| pressure vs fixed-large | 仅 `4/9` better/non-worse；mean delta `+0.001002` | `7/9` better/non-worse；mean delta `-0.020601` | 不能证明压力驱动方式优于等量预分配，G 反证仍在 |

这里的 delta 方向是 mean surprise 越低越好；所有统计来自 versioned report，不能用改阈值或挑单 seed 重新解释。

## 3. 已确认的设计事实

### 3.1 可以保留

1. `AdaptiveResidualBridge` 已真实接入 `observation → predictive context → credit`，不是旁路 demo。
2. 零影响出生、独立 shadow、candidate-only training、fresh restore、lesion 和 rollback 的安全外壳成立。
3. `eligibility + direct counterfactual credit + aligned output + birth homeostasis` 能把候选从“无因果/量级失配”修复到“有因果贡献”。
4. default parent 没有被 shadow candidate 污染；失败时仍可恢复原 checkpoint。

### 3.2 不得继续外推

1. 候选有因果贡献 ≠ 压力驱动生长已经优于固定大容量。
2. R4 结果 ≠ 自主进化完成、A8 完成或通用智能。
3. 继续调 gate 输入、birth scale、mixture 权重或新语料，不能替代 fixed-large 反证。
4. R5 learned router、Skill/MCP 内化、provider/client 外围或 CUDA 不能作为 R4 失败的遮蔽物。

## 4. 唯一推荐的下一版研究假设

如果继续研究，下一版应把问题从“再造一个 residual 单元”收敛为：

> **增长的价值是否来自按内容/状态分配可塑容量，而不是来自最终多了同样数量的参数？**

这意味着下一版必须预注册一个“资源归一化的条件模块化”实验，而不是继续微调当前 R4：

- 固定-large 仍保留为最强最终容量对照，不能删除；
- pressure growth 只有在相同累计更新预算、峰值内存和训练时间约束下，凭借更少的早期活跃容量或更好的未见 G/K transfer 获得优势，才有独立价值；
- 路由输入只能来自内容、状态、不确定性、目标和资源，禁止 evaluator task ID；
- 必须增加 route lesion、resource-normalized utility、未见组合 holdout 和旧能力保持；
- 任何新结构仍必须 zero-gated、shadow、checkpointable、可回滚；
- 在该实验的指标、对照和停止线预注册前，不写代码、不训练、不把它命名为 R5 晋级。

这是“继续研究的单一出口”，不是对当前失败结果的追溯性改 Gate。当前 R4 Gate 和报告保持原样。

## 5. 当前冻结边界

- `can_promote=false`；
- R4 shadow candidate 不进入默认 parent；
- R5 learned router 冻结；
- Skill/MCP/provider/client/CUDA 不引入新变量；
- 旧报告和 checkpoint 不改写、不删除；
- CI 已知代码门禁通过，Black 全量和精确 npm ci 的本机限制只作为环境证据记录。

下一步需要在这条“条件模块化 + 资源归一化”假设上完成明确的指标/对照/停止线决策；在决策确认前，继续写实现会造成路径漂移。
