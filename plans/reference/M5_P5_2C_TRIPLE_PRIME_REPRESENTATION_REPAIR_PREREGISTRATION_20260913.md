# M5 P5.2c‴ 预注册：能力面 profile 表征修复（路线 A）

- **状态**：冻结（执行前不得修改判据）
- **编号**：P5.2c‴（路线 A：profile 表征修复）
- **前身**：P5.2c′（`transfer_no_gain`）、P5.2c″（`transfer_signal_constant`）
- **决策来源**：用户决策「先 C 后 AB」；路线 C 已完成，本轮进入路线 A
- **纪律**：`growth_admitted=false`、`can_promote=false` 贯穿；不得反向改判据、放宽判据凑通过、覆写前身报告、重跑挑结果

---

## 0. 本文修什么、不修什么

**本文只修 profile 表征（representation），不修测量仪器，不修 pair 关系项形式。**

- **不做**：不改变 P5.2c″ 的证据构造（仍持有 `a+c`、`b+d` 两对，仍 `train=144 / holdout=88 / removed=32`）。仪器已由 P5.2c″ 修好，本轮**必须保持完全一致**，否则无法归因。
- **不做**：不引入新的 pair 关系项形式（那是路线 B）。本轮 pair 特征式的**形式家族不变**，只改**输入 profile 的信息含量**。
- **边界声明**：
  - 若本轮为**正结果**，只可声称「**在该 profile 表征下**，两个互补的未见组合产生了可测量且超过最强对照的联合增益」；**不得**声称「组合迁移问题已解决」、「群体的联合能力已确立」或「可迁移到产品」。
  - 若本轮为**负结果**，只可声称「**在该 profile 表征下**仍不可行」；**不得**声称「迁移不可行」、「组合无联合增益潜力」或「已排除表征假设」。

---

## 1. 缺陷根因（本轮执行前已由代码复现，非推理）

### 1.1 实测事实（P5.2c″ 报告，`member_evidence`）

```
contributions = { member-a: 0.5, member-b: 0.5, member-c: 0.5, member-d: 0.5 }
contribution_uniform = true
```

四个成员 **contribution 完全相等**。

### 1.2 特征式与秩塌缩（代码复现）

`_pair_features` 返回：

```
( 1.0 , (c₁ + c₂) / 2 , c₁ · c₂ )
```

当 `c₁ = c₂ = 0.5` 时，**任何** pair 的行都是 `(1.0, 0.5, 0.25)`：

| 观测 pair | realized interaction | 特征行 |
|---|---|---|
| `a+b` | −0.5 | (1.0, 0.5, 0.25) |
| `a+d` | −0.5 | (1.0, 0.5, 0.25) |
| `b+c` | 0.0 | (1.0, 0.5, 0.25) |
| `c+d` | −0.5 | (1.0, 0.5, 0.25) |

**设计矩阵秩 = 1，而列数 = 3。** 6 个 pair 塌成同一行。

### 1.3 拟合结果（真实观测记录上复现）

以 P5.2c″ 的真实观测记录（`a+b:−0.5`、`a+d:−0.5`、`b+c:0.0`、`c+d:−0.5`）拟合：

```
coefficients = ( -0.375 , 0.0 , 0.0 )
residual_rmse = 0.216506
```

**第 2、3 列权重恒为 0**，learner 对全部 6 个 pair 给出同一个预测 `−0.375`。

进一步的最小复现（含 `b+c` 与 `b+d` 的差异目标）显示：只要 contribution 全等，无论 `interaction` 如何取值，`_fit` 都返回近似常量解，**预测与真实目标的相关性为 0**。

### 1.4 两个独立的致命点

| # | 亏损类型 | 说明 |
|---|---|---|
| ① | **值亏损** | `contribution = mean(singleton − baseline)` 是**跨全部 context 的池化标量**。block/能力面结构在 `build_member_evidence` 里就被平均掉了。4 成员全等 ⇒ 秩 1。 |
| ② | **形式亏损** | `InteractionGroupMemberEvidence` 只携带 `contribution / recovery_effect / resource_cost / observations / context_count`，**全是标量**；pair 特征只有「和」「积」两式。**能力面的交集/并集信息不存在于任何一列**。 |

### 1.5 真实结构本可学（关键反证）

按 block 的 `realized_pair_gain`（P5.2c″ `family_coverage` 实测）：

| pair | blk0 | blk1 | blk2 | blk3 |
|---|---|---|---|---|
| `a+b` | **2** | 0 | 0 | 0 |
| `a+c` | **2** | 0 | 0 | 0 |
| `a+d` | **2** | 0 | 0 | 0 |
| `b+c` | 0 | **2** | **2** | 0 |
| `b+d` | **2** | **2** | 0 | 0 |
| `c+d` | 0 | 0 | **2** | 0 |

**这张表是结构清晰、完全可分的**：`a` 系打 {0}、`b+c` 打 {1,2}、`b+d` 打 {0,1}、`c+d` 打 {2}。

⇒ **数据里有信号，表征把它丢了。** 这直接排除了「该 cohort 无联合增益潜力」的解释，也是本轮唯一正结果的**可行性上界依据**。

---

## 2. 唯一结构性改动

### 2.1 改动内容（单点，可回滚）

**让 profile 携带「能力面」——即该成员在哪些 block 上有效——并让 pair 特征表达能力面的重叠度。**

具体形式（**冻结**）：

1. `InteractionGroupMemberEvidence` 新增字段 **`surface`**：`tuple[int, ...]`，升序、去重，元素为 block 索引。
   - 派生方式**冻结**：在 `build_member_evidence` 中，对每个 context 计算 `singleton_outcome − baseline_outcome > 0`（严格大于 0）时记该 context 的 block 为「有效 block」；`surface` = 所有有效 block 的升序去重元组。
   - **block 的定义冻结**：`block = task_index % 4`（P5.2c″ 已验证与模板精确对应）。
   - 若某成员无任何有效 block，`surface = ()`，**必须有明确语义**（见 §2.3）。
2. `_pair_features` 在**保持原三列**的基础上追加**能力面重叠度列**（**冻结**）：

   ```
   ( 1 ,
     (c₁ + c₂) / 2 ,
     c₁ · c₂ ,
     |S₁ ∪ S₂| / 4 ,          # 覆盖广度：联合覆盖了多少 block（归一化）
     |S₁ ∩ S₂| / 4 ,          # 重叠度：冗余程度
     |S₁ ⊕ S₂| / 4 )          # 对称差：互补程度
   ```

   其中 `S₁`、`S₂` 为两成员的能力面集合，`⊕` 为对称差，分母 `4` 为 block 总数（**冻结**）。

   **归一化理由（冻结）**：使新列与 `contribution` 同量级（0–1），避免 ridge 正则对不同尺度的列施加不等惩罚。

3. `InteractionGroupMemberEvidence.version` 由 `1` → **`2`**。
   - `version=1` 的 checkpoint **必须拒绝**（fail closed），不得静默补默认 `surface`。
   - `model_revision` 由 `1` → **`2`**。

### 2.2 为什么「只改 contribution 的量」不够

若只把 `contribution` 改成非等值标量（例如按 block 加权），特征秩会立刻恢复满秩（**值亏损消除**），但：

- 互补对与冗余对若**恰好**拥有相同的 `contribution` 统计量，仍不可分；
- 更根本地，`contribution` 是**标量**，它在定义上无法表达「两个成员覆盖的 block 集合是否相交」——**形式亏损仍在**。

因此本轮**必须同时**做值修复与形式修复，二者缺一不可。这也直接决定了路线 B 的边界：pair 关系项的**形式家族**在本轮扩展了（新增集合列），路线 B 若仍需要做，动机将不同（可能针对三元组或学习式关系项）。

### 2.3 `surface = ()` 的语义（冻结，防误用）

空能力面成员**不得**被解释为「无效成员」。冻结语义：

- `surface = ()` 表示该成员在**观测到的 context 上未取得任何严格正向的单体增益**。
- 这可能是能力缺失，也可能是**该 cohort 的 context 覆盖不足**（本例 block-3 对所有成员 0 成功即属后者）。
- 因此：`surface = ()` 的成员**不参与**能力的正向断言，**但必须仍在候选枚举中**，且其存在必须在报告中单独列出计数。

**禁止**：把 `surface = ()` 当作「可以直接剔除该成员」的依据 —— 那会把 block-3 不可达这一**环境/契约问题**误判为**成员能力问题**。

---

## 3. 证据构造：与 P5.2c″ 逐项一致（不得改动）

| 项 | 值 | 说明 |
|---|---|---|
| 成员集 | `member-a/b/c/d`（4 个） | **不扩容**（理由见 P5.2c″ §2.3，已被实证） |
| 持有 pair | `a+c`（`P₁*`，索引 `(0,2)`）、`b+d`（`P₂*`，索引 `(1,3)`） | 与 P5.2c″ 完全一致 |
| 观测 pair | 4 对（含冗余对 `a+d`） | 必须保留，使 learner 能观察「部分组合无超额收益」 |
| train episodes | **144** | 8 context × 11 cell × 2 重复 − 32 移除 |
| holdout episodes | **88** | 4 context × 11 cell × 2 重复 |
| 移除 episode | **32** | 两对持有组合的联合 `(T,T)` cell，从**全部 8 个 train context** 整体移除 |
| singleton 保留 | 每成员 **16** 条 | 保证「未见」而非「无支持」 |
| 上下文预算 | 900 s | 与 P5.2c″ 一致 |
| MARGIN | 0.15 | **冻结** |
| 校准阈值 | 符号一致率 ≥ 0.5 / 绝对误差中位数 ≤ 0.35 | **冻结** |

**一致性强制**：本轮 runner **必须**复用 P5.2c″ 的 `_build_corpus` / `_entry_audit` 语义，并在入场审计中增加一条**与 P5.2c″ 的构造等价性断言**（`train=144 / holdout=88 / removed=32` 三项全等），不等价即 `blocked_at_entry_audit`。

---

## 4. 对照

### 4.1 对照集（沿用 P5.2c″ 的 5 个 margin-bearing 对照）

| 编号 | 对照 | 说明 |
|---|---|---|
| C1 | `lesion_learner` | 移除新 profile 列（置 0）后的同一 learner |
| C2 | `strongest_singleton` | 逐 context 最强单体（**主要对手**） |
| C3 | `no_learning` | 无学习，取固定组合 |
| C4 | `random_combination` | 随机组合（固定种子） |
| C5 | `train_only_simple_regression` | train-only 简单回归 |

**C1 是本轮的特有关键对照**：它直接度量「新增的能力面列本身贡献了多少」，而非「整个 learner 有多强」。C1 必须与主对象使用**同一** profile 对象（只把新列置 0），否则不可归因。

### 4.2 新增必报分账（本轮特有）

| 指标 | 说明 |
|---|---|
| `coefficient_norm_by_column` | 逐列系数绝对值 —— 直接检验新列是否被真正使用（预期：**非零**）。若新列系数为 0，则表征修复**未生效**，属机械失败 |
| `feature_rank` | 设计矩阵的数值秩。**必须 > 1**；等于 1 即秩塌缩未解决 |
| `prediction_distinctness` | 6 个 pair 的预测值中的**不同值个数**。等于 1 即预测仍为常量 |
| `surface_per_member` | 每成员实测能力面（**本轮重新实测，不得引用任何旧值**） |
| `surface_pair_kinds` | 两持有对的 `DISJOINT`/`EQUAL`/`PARTIAL` 分类（**本轮重新实测**） |

---

## 5. 九门（冻结）

| # | 门 | 判据 |
|---|---|---|
| 1 | `static_checks` | ruff 全仓 + py_compile 通过 |
| 2 | `evidence_admissibility` | 入场审计通过，`conditions = []` |
| 3 | `unseen_combination_identity` | 两持有对均 `in_observed_records=false`、`joint_count_in_train=0`、`joint_count_in_holdout=8` |
| 4 | `unseen_context_holdout` | 4 未见 context 严格隔离 |
| 5 | `label_opaqueness` | 持有由不透明索引决定 |
| 6 | `real_execution` | 真实执行，`interventions_happened=true`，非 baseline 零步 = 0 |
| 7 | `rejection_recovery` | 拒绝探针 9 项全 true；恢复探针 `selection_reproduced=true` |
| 8 | `prediction_binding_and_calibration` | 绑定先于评分；符号一致率 ≥ 0.5；绝对误差中位数 ≤ 0.35 |
| 9 | `representation_and_transfer` | **见下**（本轮扩展） |

### 门 9 拆分（冻结）

门 9 要求**全部**成立：

```
(a) representation_effective := ( feature_rank > 1 )
                              and ( coefficient_norm_by_column[-3:] 不全为 0 )
                              and ( prediction_distinctness > 1 )
(b) margin_cleared           := any_held_out_positive
                              and ( object_gain > strongest_control_gain + MARGIN )
(c) collaboration_holds      := 至少 1 个持有对的 mean_gain_vs_strongest_single > 0
(d) replica_consistent       := true
(e) total_wall               <= 900 s
```

**新增 (a) 的理由（冻结）**：若表征修复**没有生效**（新列权重被正则压到 0、或因数据仍然秩亏），那么增益门未过**不能**被解读为「表征修复无效」——它连被测都没测到。因此必须先证明修复生效，再谈增益。**这是本轮防止错误归因的核心机制。**

---

## 6. 三态结论（冻结）

| 结论 | 条件 |
|---|---|
| `failed` | 任一机械门（1–8）未过 |
| `representation_repair_ineffective` | 机械门全过、门 9 未过，且 **(a) 不成立** ⇒ 表征修复未生效，本轮**不作能力结论**，须先修实现 |
| `transfer_signal_constant` | 机械门全过、(a) 成立、门 9 未过，且 `any_held_out_positive=false` |
| `transfer_no_gain` | 机械门全过、(a) 成立、门 9 未过，且 `any_held_out_positive=true` 但 margin 未清 |
| `unseen_combination_transfer_supported` | 全部门过 |
| `blocked_at_entry_audit` | 入场审计未过（含 §3 的构造等价性断言） |

**`representation_repair_ineffective` 是本轮新增态**，与 `failed` 的区别：实现是「成功跑完」的，机械门全过，但修复本身没起作用。**不得**把它并入 `transfer_no_gain` 或 `transfer_signal_constant`。

---

## 7. 已知代价（如实披露，不得隐藏）

1. **`surface` 的引入引入了「按 block 分块」这一结构先验**。本 cohort 中 block 与模板精确对应，所以该先验是**正当的**；但它**不可**外推到 block 与模板不对应的场景。报告中必须显式声明 `surface` 依赖 `block = task_index % 4` 这一映射。
2. **`surface` 的粒度是 block 级，不是 task 级**。同一 block 内的任务差异被忽略。
3. **`|S₁ ∪ S₂| / 4` 等三列引入了共线性风险**（三者的分母相同、彼此线性相关：`|∪| = |∩| + |⊕|`）。**恰好**满足 `∪ − ∩ − ⊕ = 0`，即三列**恒线性相关**。因此设计矩阵的秩仍会受 ridge 保护，但**逐列系数不可单独解释**。报告必须声明：只可解释**预测**，不可解释**单个系数**。
   - 保留三列（而非只留一列）的理由：ridge 对不同列的惩罚不同，三列并存给求解器更多自由度；但**这一选择的代价必须披露**。
4. **仍不解决 block-3 不可达**：区分面仍为 `0.75`，所有增益统计须按此折算理解。
5. **恢复效应仍未测量**，记为 0，不编造数据。
6. **`P₁*` 结构性指定**，对照 C4 与正确答案重合，须从 margin 排除。

---

## 8. 纪律

- 本轮**不得**修改或覆写 P5.2c、P5.2c′、P5.2c″ 的任何预注册与报告。
- 本轮**必须**保留 P5.2c″ 作为干净基线（同 144/88 划分、同两持有对、同 5 对照）。
- 阈值、对照、三态判据在本文冻结后**不得**改动。
- 不得因归因结论把前身 gate 追认为通过。
- `growth_admitted=false`、`can_promote=false` 贯穿。
- 临时探针用毕即删。
- `version=1` checkpoint 必须 fail closed，不得静默补默认 `surface`。

---

## 9. 执行前已完成的代码复现（留证）

本轮在执行前已完成以下复现（**非推理**，脚本用毕即删）：

| 复现项 | 结果 |
|---|---|
| 特征行在 uniform contribution 下是否全等 | **是全等**，`(1.0, 0.5, 0.25)` |
| 真实观测记录上的拟合系数 | **`(-0.375, 0.0, 0.0)`**，第 2/3 列权重为 0 |
| residual_rmse | `0.216506` |
| 6 个 pair 的预测 | **全部 −0.375**（常量） |
| 按 block 的真实增益是否可分 | **完全可分**（§1.5 表） |

**结论**：缺陷已定位到代码行级（`_pair_features` 第 507–518 行、`build_member_evidence` 第 232–248 行、`_ridge_fit` 第 552 行仅对 `index >= 1` 施加 ridge），且数据侧存在可学信号。本轮的正结果是**有可行性上界的**，不是盲目尝试。

---

## 10. 修复后实测的残余局限（**执行前发现，如实披露**）

表征修复实现后，在**同一批真实观测记录**上复测，确认修复**机械生效**：

| 门 9(a) 指标 | 修复前 | 修复后 |
|---|---|---|
| `feature_rank` | 1 | **2** |
| `coefficients` | `(-0.375, 0, 0)` | `(-0.4672, 0, 0, 0.0820, -0.0820, 0.1639)` |
| 表面列是否非零 | 无此列 | **非零** |
| `prediction_distinctness` | **1**（全 −0.375） | **2** |
| `a+c` vs `a+d` 排序 | 相等 | **`a+c` > `a+d` ✓** |
| `b+d` vs `a+d` 排序 | 相等 | **`b+d` > `a+d` ✓** |

### 10.1 残余局限（必须在报告中显式声明）

修复**能**区分「冗余 vs 互补」，但**不能**在互补对之间排序：

| pair | 能力面 | ∪ | ∩ | ⊕ | 特征表面部分 |
|---|---|---|---|---|---|
| `a+b` | {0} ∪ {1} | 2 | 0 | 2 | (0.5, 0.0, 0.5) |
| `a+c` | {0} ∪ {2} | 2 | 0 | 2 | (0.5, 0.0, 0.5) |
| `b+c` | {1} ∪ {2} | 2 | 0 | 2 | (0.5, 0.0, 0.5) |
| `b+d` | {1} ∪ {0} | 2 | 0 | 2 | (0.5, 0.0, 0.5) |
| `a+d` | {0} ∪ {0} | 1 | 1 | **0** | (0.25, 0.25, 0.0) |

**四个互补对的特征向量完全相同**。因为当前三列只捕获「覆盖了多少 block」与「重叠多少」，**不捕获「覆盖了哪些具体 block」**。

### 10.2 这对结论解读的硬约束

1. **若本轮正结果**：只能声称「learner 学会了**避开冗余组合**」，**不能**声称「learner 能识别最优互补组合」。因为 `a+c`、`b+d`、`a+b`、`b+c` 在它眼里**不可分**。
2. **若本轮负结果**：**不能**声称「表征修复无效」——修复已由门 9(a) 证明生效。负结果只能说明「**block 计数级别的**能力面信息不足以支撑迁移增益」。
3. **禁止**在执行后把本节的局限删除或弱化；若需改进（例如加入 per-block 指示列），必须**另立预注册**，不得在本轮内修改特征式。

**本节的存在理由**：若不做此披露，一旦本轮出正结果，极易被读成「表征问题已解决、组合迁移已具备」。实际它只解决了「值亏损 + 冗余/互补不可分」这一层，**block 身份信息仍未被表征**。
