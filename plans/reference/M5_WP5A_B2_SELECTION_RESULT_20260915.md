# M5 WP-5a / B2 选择门结果：机械门全过，选中组合零收益（负结果）

日期：2026-09-15。状态：**已执行，负结果**。
预注册：[WP-5a B2 选择预注册（冻结版）](M5_WP5A_B2_SELECTION_PREREGISTRATION_FROZEN_20260915.md)；
估计目标与 G1–G6 继承[路线 B 冻结版](M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md)（binder v1）。
产物：[runner](../../scripts/training/eval_taiji_b0_b2_selection_gate.py) /
[报告](../../reports/taiji_b0_b2_selection_20260915.json) /
[契约测试](../../tests/taiji_native/test_b0_b2_selection_contract.py)（10 passed）。

## §1 一句话结论

**仪器可信、机械门全过，但选择器选中的组合产出零任务收益。**
B2 选中的是 `member-b+member-c`（预测交互最高 0.141802），而**唯一真正产生增益的是
`member-a+member-c`（+2.000）**——后者在同一表示下的预测值更低（0.083829）。
⇒ 结论：**该表示的"能区分"（B1 已证 rank 4）没有转化为"选得对"**；
H1/H2 均不成立，协作轴本轮**未获收益证据**。

## §2 执行事实

| 项 | 值 |
|---|---|
| outcome | `failed`（detail `no_task_benefit: G4 and G5 both failed while mechanical gates passed`） |
| 候选面 | 3 格 × 6 context × 11 cells × 2 repeats = **396 episodes**（全 factorial 一次取得） |
| fit 面 | 8 个非候选结构格 step 0（176 episodes，与 B1 逐字节同一构造） |
| 域内 holdout | 同 8 格 step 1（176 episodes） |
| 总 episodes | **748** |
| wall | **31.2 s**（cap 60 s，G6 过） |
| 规则版本 | `rule_revision=1`、`m4_failure_handoff` |
| 可复现 | 两次运行**稳定字段哈希相同**（`8c78ef23…`，剔除计时/commit） |
| 恢复/篡改 | fresh-process 恢复**复现同一选择**；篡改拒绝 |
| 稳定性 | 成员重命名 + fit 顺序置换下**选择不变** |

## §3 G1–G6 判定

| 门 | 结果 | 依据 |
|---|---|---|
| G1 干预真实性 | **过** | 零步非 baseline episode = 0，`interventions_happened=true` |
| G2 数据隔离 | **过** | fit task_ids ∩ candidate task_ids = ∅（结构断言） |
| G3 排序稳定性 | **过** | 重命名 + 顺序置换后选择不变 |
| **G4 任务门（H1）** | **不过** | `U(selected)=−1.0` vs `U(best_fixed_singleton)=−1.0` ⇒ 增益 0，未超过可部署对照 |
| **G5 协作门（H2）** | **不过** | 逐格增益 `0.0 < required 1.65`；lesion 归因不成立（0 个成功 context）；所选 pair 自身交错数 **0** |
| G6 预算门 | **过** | 31.2 s ≤ 60 s |

## §4 关键数字

**逐格（三格数字相同——表示不条件化于 context，故三格同一选择）**

| 项 | 值 |
|---|---|
| 所选 pair 逐格增益 | `0.0` / `0.0` / `0.0` |
| `member-a+member-c` 逐格增益 | **`+2.000`**（唯一正增益） |
| 其余四对 | `0.000` |
| 四个单体 mean outcome | 全部 `−1.0`（组合专属面，单体按设计全败） |
| `required` / `required k` / `available k` / `n` | `1.65` / **5** / **6** / **6** |
| `ceiling_gain` / `max_clearable_reference` | `2.0` / `1.85` |

**选择细节**：`utility=0.141802`、`uncertainty=0.604566`、`resource_cost=4.75`、`observations=5`。

**对照臂**（全部来自同一次 factorial 执行，零额外成本）

| 臂 | 值 |
|---|---|
| `selected_pair`（b+c） | 增益 `0.0`、mean outcome `−1.0` |
| `best_fixed_singleton`（可部署 H1 对照） | mean outcome `−1.0`，**四单体并列垫底**（`tied_at_floor=true`） |
| `best_observed_fixed_pair`（诊断列，不可部署） | `member-a+member-c`，增益 `+2.000` |
| `no_learning`（回退固定 a+b） | 增益 `0.0` |
| `random_pair`（seed 17） | `member-b+member-d`，增益 `0.0` |
| `additive_train_only` 排名首位 | `member-a+member-d`（**也不含真正增益的 pair**） |
| `a_count` | 对全部 pair 预测相同，无判别力 |

## §5 机制归因：为什么选中的是零收益组合

1. **表示不条件化于 context**：三候选格得到**同一个**选择（预注册 §2.2 已冻结该边界，报告
   `context_conditioned=false` 且显式声明**不是逐格路由**）。
2. **训练语料里没有"create × 语言"这种联合必需结构的正例**：fit 只用 8 个非候选结构格，
   而联合必需只存在于候选 create 行（§4 隔离条款要求如此）。表示因此只能靠**加性/表面特征**
   外推，学到的是"谁在 train 面更常出现"而非"谁补上缺失子句"。
3. 结果就是：**预测最高的是 b+c（create 专家 + 另一成员），而真正补上语言子句的是 a+c**。
   B1 报告已如实披露 `rank_correlation=0.270` 且 `best_predicted ≠ best_actual`；
   **B2 把这条披露转化成了下游后果**——排序相关性为正，但 top-1 选错，收益为零。

## §6 三态归属与**预注册缺口**（如实登记）

- 冻结三态表（路线 B §10 / WP-5a §3）只定义四种：`collaboration_supported`、
  `routing_only`（G4 过 G5 不过）、`representation_ineffective`、`failed`（**任一机械/合同门失败**）。
- 本轮实际落在 **(G4 不过, G5 不过) 且机械门全绿**——**冻结表没有对应标签**。
- 处置：记为 `outcome=failed` 并附 `outcome_detail`，同时在报告写入
  `preregistration_gap` 字段，**不新造第五态、不改三态表**。⇒ 该缺口需计划修订补标签。

## §7 界限：本轮**不**能主张什么

- **不能**主张协作：G5 不过，`collaboration_supported` 不成立；`interleaved` 是所选 pair 的
  **0**（报告中"任意 pair 交错 18"是另一列，已在 `interleaved_note` 中显式区分，防止被误读）。
- **不能**主张路由收益：G4 不过，连"覆盖更广"也不成立。
- **不能**把 `member-a+member-c` 的 `+2.000` 当作"选择器成功"：它是**诊断列**，
  来自同一 factorial 的事后读取，不是可部署策略。
- **不能**把本轮负结果当作"任务不可达"：`required k=5 ≤ available k=6`、`feasible=true`，
  任务本身可达（与 N1/WP-1.5 一致）。
- **不能**因为 G4 不过就说"单体更强"：四单体在该面**全部失败**（−1.0），
  G4 不过是因为**双方都垫底**（tie），不是因为对照赢了。

## §8 下一步

按路线 B §7 的出口条款，本轮属**"表示不足 ⇒ 回 B1"**：
表示能区分但排序方向不足以挑出产生增益的组合。可选路径（需决策，见计划修订）：

1. **B1 迭代**：让排序目标对齐任务效用（当前拟合目标是 train 面交互，与 create 行联合必需结构不同分布）；
2. **CAP-0**：按 2026-09-15 修订，工作包结项点触发整模型能力基线；
3. **WP-6（binder）**：与本次负结果**无因果关系**（本轮失败在排序，不在可表达性），不构成理由。

**唯一下一步由计划决定，本文只落账结果。**
