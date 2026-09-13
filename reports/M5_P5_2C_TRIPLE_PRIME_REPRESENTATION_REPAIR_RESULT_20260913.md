# M5 P5.2c‴ 路线 A 结果报告：profile 表征修复

- **预注册**：`plans/reference/M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_PREREGISTRATION_20260913.md`（冻结）
- **runner**：`scripts/training/eval_taiji_p5_2c_triple_prime_representation_repair_gate.py`
- **原始报告**：`reports/taiji_p5_2c_triple_prime_representation_repair_20260913.json`
- **回归测试**：`tests/taiji_native/test_p5_2c_triple_prime_representation_repair_gate.py`（23 passed）
- **提交**：`949223bb`
- **结论**：`transfer_signal_constant`
- **growth_admitted=false，can_promote=false**（贯穿，未变）

---

## 1. 一句话结论

**表征修复本身成功了**（`representation_effective=true`：秩 1→2、新列系数非零、
预测区分度 1→2），**但迁移增益仍然为零** —— 三个新增列只能区分「冗余 vs 互补」，
无法区分「哪个互补更好」，因此两个未见互补对依旧拿不到任何增益。

这是一个**比 P5.2c″ 更精确的负结果**：它把「表征坏了」与「能力不存在」这两件事
分开了 —— 前者已被证伪，后者仍未被动摇。

---

## 2. 门禁结果

| 门 | 结果 |
|---|---|
| 1 static_checks | ✅ |
| 2 evidence_admissibility | ✅ |
| 3 unseen_combination_identity | ✅ |
| 4 unseen_context_holdout | ✅ |
| 5 label_opaqueness | ✅ |
| 6 real_execution | ✅ |
| 7 rejection_recovery | ✅ |
| 8 prediction_binding_and_calibration | ✅ |
| 9 representation_and_transfer | ❌ |

机械门（1–8）**全过**，故结论为 `completed` 而非 `failed`——这一点很重要：
它排除了「接线错了 / 语料没构造对」这类解释。

### 入场审计（entry_audit）

`passed=true`，无任何未过条件。**新增条件**`corpus_carries_multi_block_pair_gain` 通过：

```
pair_coverage_by_block:
  member-a+member-b: {0: 3}
  member-a+member-c: {0: 3}
  member-a+member-d: {0: 3}
  member-b+member-c: {1: 3, 2: 3}   ← 两个 block
  member-b+member-d: {0: 3, 1: 3}   ← 两个 block
  member-c+member-d: {2: 3}
multi_block_pairs = ['member-b+member-c', 'member-b+member-d']
```

### 构造等价性

```
expected == actual == {train_episodes: 144, holdout_episodes: 88,
                       removed_episode_count: 32, observed_pair_count: 4}
matches = true
```

与 P5.2c″ **逐项一致**，因此增益差异只能归因于表征改动。

---

## 3. 表征修复：成功（这是本轮的核心成果）

| 指标 | P5.2c″（修复前） | P5.2c‴（修复后） |
|---|---|---|
| `feature_rank` | 1 | **2** |
| `feature_width` | 3 | 6 |
| `coefficients` | `(-0.375, 0.0, 0.0)` | `(-0.0246, 0.0, 0.0, -0.0615, +0.0615, -0.1230)` |
| `surface_columns_any_nonzero` | —（无此列） | **true** |
| `prediction_distinctness` | 1 | **2** |
| 6 对预测 | **全为 −0.375（常量）** | 分两组：冗余对 `a+d` = −0.0246；其余 5 对 = −0.1168 |
| `residual_rmse` | 0.216506 | 0.306515 |

`representation_effective = true`（门 9(a) 三项全过）。

**根因确认（执行前已在预注册 §1 代码级复现）**：
`contribution = mean(singleton − baseline)` 是跨全部 context 的池化标量，
4 个成员全部等于 0.5，于是旧特征式 `(1, (c₁+c₂)/2, c₁·c₂)` 对**每一对**都产出
同一行 `(1.0, 0.5, 0.25)` ⇒ 设计矩阵 3 列而秩 1 ⇒ 只能拟合出常量。

修复后 `contribution` **依然全部是 0.5**（`contribution_uniform=true`）——
这正是本轮的要点：**信号不是从 contribution 来的，而是从新增的 surface 来的**。
4 个成员的 surface 实测为 `a={0}, b={1}, c={2}, d={0}`（`surfaces_all_equal=false`）。

---

## 4. 为什么增益门还是没过

```
object (被选中)  = member-a + member-c     gain = -0.5
required         = strongest_singleton(1.5) + MARGIN(0.15) = 1.65
margin_cleared   = false
collaboration_holds = false
any_held_out_positive = false            ← 两对均为 0/4 context 优于最强单体

两对场景实测:
  member-a+member-c: mean_gain_vs_strongest_single = -0.5
  member-b+member-d: mean_gain_vs_strongest_single =  0.0
```

最强对照是 **`strongest_singleton`（逐 context 最强单体）= 1.5**。
逐 context 看得很清楚：context 108/109/110 上总有一个单体拿到 +2.0，
而两对组合在同样 context 上分别只拿到 0.0 / -2.0 / 0.0 ——
**组合没有创造出单体之上的增量**，在 context 110 上甚至是**负交互**。

### 关键：为什么修复没能救回来

预注册 §10 已**在执行前**披露了这一残余局限，本轮实测完全落在预测上：

```
四个互补对的特征行完全相同：(union=0.5, overlap=0.0, symmetric=0.5)
仅冗余对 a+d 不同：          (union=0.25, overlap=0.25, symmetric=0.0)
```

三列计的是「覆盖了多少」（基数），**不携带「覆盖了哪些」**（身份）。
于是 `feature_rank=2` 只够把「冗余对」与「互补对」分成两组，
组内 5 个互补对**仍然不可分辨** —— 而两个未见对恰好都在组内。

**因此本轮结果只能断言**：表征修复让「避开冗余组合」变得可学（`a+d` 被单独压到
−0.0246，比其他对高），**不能**断言「能识别出最优互补组合」。

---

## 5. 严谨性说明：`representation_repair_ineffective` 为何未被触发

预注册新增了第四个三态结论 `representation_repair_ineffective`，其存在理由正是防止
**错误归因**：若修复没生效，增益门失败**不能**读作「表征修复无效」——因为那根本没被测到。

本轮 `representation_effective=true`，故该结论不触发，落入 `transfer_signal_constant`。
这个分支结构使「没测到」与「测了没用」被机械区分开，是本轮可归因的前提。

---

## 6. 本轮修过的三个接线 bug（如实披露）

均为**门禁接线错误**，非结论操纵；修复依据是 P5.2c″ 的既有实现，未改任何判据：

1. **`_entry_audit` 被误委托**：父模块版本读的是父模块自己的 `family_coverage` 全局，
   本轮新增条件会被套用到本 gate 从未构造过的数据上。
   → 本地实现，显式接收本 gate 的 `family_coverage`。
2. **`surface_of` 引用了不存在的键**（`KeyError: 'context_id'`）：原始 matrix episode
   只有 `task_id`，无 `context_id`。
   → 改为与 P5.2c″ **逐字一致**的 `(episode_id, success, stop_reason, resource_cost)`。
3. **`executed_entries` / `provenance` / `safety_violation` 取层级错误**：
   这三项挂在 **step** 上而非 episode 上，导致 `real_execution` 门假性失败。
   → 按父模块实现改回逐 step 遍历。

另修 `_select_sum_heuristic`（C5 对照）参数：它按**profile contribution**排序，
故须传 profiles 而非 learner，且 `resource_budget` 为必填关键字参数。

上述 bug 一度让 `real_execution` 与 `representation_and_transfer` 双双失败；
若未排查即记录，会得出完全错误的负面结论。

---

## 7. 已知代价与残余局限（禁止执行后删除）

1. 三列恒线性相关（`|∪| = |∩| + |⊕|`）⇒ **单个系数不可单独解释**，只能看预测。
2. 只覆盖 block 粒度，不覆盖「哪个 block」⇒ 互补对组内不可分（本轮结论的直接约束）。
3. 未解决 block-3 可达性：`discriminating_fraction` 仍 **0.75**
   （block-3 66 条 episode 全部失败，60 条 `contract_intercepted`，
   `uniformly_unreachable=true`）—— 属测量仪器层面，本轮声明不修。
4. `contribution` 仍是池化标量（`contribution_uniform=true`）——
   本轮有意不改，以证明「新增 surface 确实提供了独立信号」。
5. C4 `fixed_combination` 与正确答案按构造重合，已排除在 margin 对照集之外。
6. 未见面对基数仅 2，校准统计（`sign_match_rate=1.0`, `median_AE=0.25`）**不是独立样本**。

---

## 8. 声称边界

**本轮支持**：
- 池化标量 profile 是 P5.2c″ 常量预测的**充分原因**（代码级复现 + 修复后秩与区分度回升）。
- 在**该表征**下，两个未见互补对不产生超越最强单体的增益。
- 语料中确实存在 multi-block 联合增益结构（入场条件已证），故这不是「无信号」问题。

**本轮不支持**：
- ❌ 「未见组合迁移不可行」—— 表征的第 2 项亏损（形式/身份）仍在。
- ❌ 「组合选得不好」—— 障碍在表征能表达的区分度，不在选择策略。
- ❌ 「表征问题已解决」—— 仅「避开冗余」可学，「识别最优互补」仍不可。
- ❌ 任何 `growth_admitted` / `can_promote` 含义。

---

## 9. 下一步

按已决策的「先 C 后 AB」，路线 A 至此完成。**路线 B（pair 关系项形式）须新预注册**。
其动机已因本轮而**收窄且更明确**：不再是「修 contribution 的量」，
而是**让表征能表达 block 身份**（如按 block 分解的 per-block 特征或 surface 的
位置敏感编码），使五个互补对不再共享同一特征行。

**路线 C 的既有结论不受影响**，P5.2c、P5.2c′、P5.2c″ 的预注册与报告**未被修改或覆写**，
任何阈值**未被放宽**，本轮未重跑挑结果。

---

## 10. 附带事故：.git 结构性损坏（已在同一次提交中修复）

运行本 gate 后准备提交时 git 报 `fatal: not a git repository`。实测根因：
`.git/refs/` 整个目录树缺失；第二个 pack 只剩 `.idx` 而 `.pack` 丢失；
`.git/index` 的 cache-tree 含失效 sha1（20 个 blob 报 missing）；
reflog 含 237 条指向已丢失对象的条目。

后果：HEAD 停在 `d360513d`，其后 6 个提交对象全部丢失，远端亦仅到 `c7bbd389`。

**内容零丢失**：逐一核验 6 个提交涉及的 14 个文件，全部完好存在于工作区。
修复动作：建回 `refs/` 目录树 → `git read-tree HEAD` 重建索引
（missing blob 20→0）→ 重建 `logs/HEAD` → 把损坏 reflog **移出 .git** 存档
（留在 `.git/logs` 下 fsck 会持续解析）→ 备份整个 `.git` 与 5 个交付文件到
`E:/Seed-backup-gitstate-20260913-183929/`。

结果：`git fsck --connectivity-only` **exit 0**，仅余 6 个 dangling commit（既有状态）。
25 个路径已提交为 `949223bb` 并推送至 `origin/main`，历史现已落到远端。

**遗留警示：在确认无需回溯历史前，不要执行 `git gc` / `git prune`。**
