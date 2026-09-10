# M4.V2 B3-K C-entry widened-candidate 路线设计（冻结）

> 冻结日期：2026-09-10。前置：[C-entry capacity-parity audit](M4V2_B3_K_C_CAPACITY_PARITY_AUDIT_20260910.md)（`blocked-current-comparison-confounded`）与 [capacity-parity manifest](../manifests/taiji_m4v2_b3_k_c_capacity_parity_v1.json)（`frozen-preflight`，9/9 fixed-large cell 通过输入核验，candidate design `pending`）。本文冻结 widened-candidate 的路线设计；设计冻结后按 manifest 生成 artifact 并机器核验三份硬门，核验通过前禁止训练、sealed formal、promotion、调学习率、复活 R5/旧结构增长路线，或接入 default runtime/provider/MCP/client/CUDA。

## 1. 目标与硬约束（全部来自已冻结的 parity contract）

| 硬门 | 当前 candidate | fixed-large（保留，不缩小） | widened-candidate 目标 |
|---|---:|---:|---|
| 参数字节（±1%） | 19,332（4,833 参数） | 38,664（9,666 参数） | **38,664 ±1%** |
| 实际参数更新步数（精确相等） | 6 | 14,252 | **14,252** |
| 逻辑训练 checkpoint 发射数（相等） | 2 | 9 | **9** |

不变量：同一 frozen parent（3 个 model seed digest 不变）、同一 C-entry course digest、同一 train/holdout 隔离、同一 target-tensor multiset、K3 deterministic projector 不变、`candidate_updates_must_be_distinct`、fresh restore/rollback、无 default runtime attachment。

## 2. 冻结的路线：单实例双通道分解（supervised + K3-feedback）

**拓扑不变**：仍是一个 native K1 learner + 一个 native K2 learner + 不可变 K3（不复制 worker）。**加宽方式**：每个可学习头从单 delta 通道扩展为**双 delta 通道**，forward 输出为两通道之和：

| Worker | 头 | 通道 1（supervised） | 通道 2（k3-feedback） |
|---|---|---|---|
| k1.semantic | fact_head / goal_head / content_head | episode 监督 delta（现行 fit 语义，全部 train experience） | K3 admitted outcome/dependency 事件的投影 delta（同一 tick 的 K3 projection 输出） |
| k2.transition | transition_head / goal_head / content_head | 同上 | 同上 |

- 两通道**形状与 supervised 通道完全相同**，因此 worker 参数精确 ×2：4,833 → **9,666 参数 = 38,664 字节（ratio 0.0%）**，无需凑数、无自由宽度参数。
- **update 语义不同且真实（§4 修订后的通道定义）**：实现核查发现 K3 projection 事实（`outcome::*`/`dependency::*`）不在 K worker 现有 13-fact 词汇内，扩展词汇会使 K2 交叉段形状膨胀（13→17 facts → 参数 ~15.9k ≫ 9,666），破坏精确 ×2。按 §4 修订条款（不参考 validation/sealed 表现）把通道 2 的定义从「K3-feedback 目标」修订为「**K3-anchored permuted-order 通道**」：通道 1 消费正序 experience 流（与 fixed-large replica_0 相同）；通道 2 消费由 **K3 projection digest 派生的确定性乱序** experience 流（每 tick 的 update 记录携带 projection digest 作为排序见证）。非退化依据：在线局部 delta 对经历顺序敏感（M5.S6 formal 已实证 seed 顺序导致 margin 剧变），两条顺序不同的流从同一 parent 出发必然产生不同的最终权重；distinct 证据门（差分范数）机器核验。K3 语义从「目标源」如实降级为「排序见证」。
- **readout 语义不同**：fixed-large 的 readout 是「同构 replica 的概率算术平均」；widened candidate 的 readout 是「单 bundle 内双头 **logit 相加**后再归约（sigmoid/softmax）」——logit 域合成 vs 概率域平均，且两通道经历顺序不同。owner/update/readout 三条路径均不同。
- **学习规则仍是无 optimizer 的局部 delta**：v1 合同的 `optimizer_state_present=false` 保持；新增的只是 update 视图。因此 **K continuation 合同以 widened 变体升 v2**（`taiji-k-continuation-widened-v2`，双通道 update 语义 + receipt 扩展），遵循合同自身的版本演进条款，不静默修改 v1。

## 3. 训练预算与 checkpoint 政策（与 fixed-large 精确对齐）

- **更新步数**：fixed-large = 每 replica（K1 2,083 + K2 5,043 = 7,126 步）× 2 replicas = 14,252。widened candidate = 每 tick **两通道各一次真实头更新**：7,126 ticks × 2 通道 = **14,252 次实际参数更新**，精确相等。每次更新都改变至少一个参数（零 delta 的 tick 仍计入但必须在 receipt 中报告零更新计数——若零更新占比过高则该设计不通过 distinct 证据门）。
- **逻辑 checkpoint**：与 fixed-large 相同的 episode/phase 边界，每 cell 发射 **9 个**逻辑训练 checkpoint（当前 candidate 的 2 个来自 pilot 预算，非政策上限）；prefit/fresh-restore/rollback/audit 副本单独列账，不计入 9。
- **课程**：同一 course digest、同一 3 train episodes × 3 model seeds、同一 target-tensor multiset；课程 seed 必须改变实际经历顺序/组合（沿 B1 冻结）。

## 4. 参数核算闭环

实现时以参数核算表逐头列出（K1 三头 + K2 三头 × 2 通道），机器核验 `worker_parameter_count == 9666`、`worker_parameter_bytes == 38664`。因为双通道与 supervised 通道同形状，核验是确定性的；若实现发现形状偏差（如某头在 v2 中另有结构），只允许在「不参考 validation/sealed 表现」的前提下调整通道定义，并重新走本文冻结。

## 5. 诚实性风险与 distinct 证据要求

1. **退化风险**：若 feedback 子集在训练中与全集重合且两通道 error 相同，双通道可能收敛到近相同的权重——届时 candidate 实质上退化为「同 error 双头」，失去对照意义。**硬性要求**：逐 cell 报告两通道的参数差分范数（per head）与反馈子集覆盖率；差分范数为零或反馈子集覆盖率 < 50% 的 cell 判为 distinct 证据失败，停止并归因，不进入 sealed 评分。
2. **feedback 信号定义冻结**：k3-feedback 通道的 error 信号 = K3 `OutcomeDependencyProjection` 在该 tick 的 projection delta（内容寻址、admission 后）；不得在看过 validation/sealed 结果后改定义。
3. **禁止**：调学习率（局部 delta 的既有学习率沿 B1 冻结）、复制 replica、把 fixed-large 的 ensemble 改成 candidate 的一部分、读 sealed formal 结果、为凑参数添加无 update 语义的死参数（所有新增参数必须被 14,252 步真实更新覆盖）。

## 6. 停止线

- 参数核算不命中（>1%）→ 回设计（单变量修订），不硬凑；
- distinct 证据失败（§5.1）→ 停止，归因 feedback 信号质量（K3 projection 信息量）与数据视图设计，不放宽；
- 更新步数/checkpoint 数不精确相等 → harness 缺陷，修复后重跑核验；
- 任一 fresh restore/rollback/parent-unchanged/K3-unchanged Gate 失败 → 停止；
- 以上任何失败都完整报告，不因失败缩短证据链。

## 7. 验收产物（顺序）

1. `taiji/k_continuation.py` 合同 v2（双通道语义 + receipt 扩展）+ 定向测试；`SEMANTIC_TRANSITION_VERSION`/`SEMANTIC_TRAINING_VERSION` 若结构变更则 bump；
2. widened-candidate artifact builder（按 manifest `candidate_format: taiji-k-c-entry-parity-candidate-v1`）+ 参数核算表报告；
3. 三硬门机器核验报告（参数字节 / 14,252 更新步 / 9 checkpoint + distinct 证据 + fresh restore/rollback/parent/K3 不变）；
4. 通过后才进入同预算的 C-entry parity formal 训练与 sealed 评分（formal 前另冻结判据确认记录）；
5. `can_promote=false` 全程固定；promotion 仍要求 candidate 在 9/9 sealed cell 胜过 strong control 并通过既有 technical/resource/quality Gate。
