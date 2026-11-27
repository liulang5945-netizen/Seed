# M5 P5.2c 入场审计 — P5.2b 结果的结构性缺陷与 P5.2c 阻塞

日期：2026-09-13。类型：**入场审计（gate-blocking）**。结论：**P5.2b 的 admitted group 不成立；P5.2c 不具备入场条件，必须停止。**
上游：[P5.2c 预注册](../plans/reference/M5_P5_2C_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md)（本审计由该预注册 §5 门 2/3 的实际执行触发）。
关联：[P5.2b 报告](taiji_p5_2b_group_causal_corpora_20260913.json)、[P5.2b 预注册](../plans/reference/M5_P5_2B_GROUP_CAUSAL_CORPORA_PREREGISTRATION_20260913.md)、[推进方案 §7](../plans/active/roadmap/03_CURRENT_EXECUTION.md)。

## 1. 触发路径（为何在 P5.2c 实现前就发现）

P5.2c 实现按冻结预注册接线后，`select(candidate_member_sets, unseen_only=True)` 返回 `None`：**候选面上不存在任何「未见组合」**。4 个成员共有 `C(4,2)=6` 个 pair，而 `train_only_candidates` 恰好返回了全部 6 个——`_pair_features` 只支持 pair（三元组 raise `interaction transfer relation currently supports pairs only`），故未见组合集合为空。

这不是 learner 的问题，而是**上游 trace 的构造缺陷**：`train_only_candidates` 在每个 context 上都估出了 6 个 pair，意味着**每个 context 都必须齐备全部 4 个 factorial cell**。带着这个必然后果回到矩阵，暴露出真正的病根。

## 2. 根因：block-0 context 的 episode 无事件，fallback 到「目标已达成」

P5.2a `_validation_tasks()` 的 `lang_confirm` 模板（`index % 4 == 0`，即 context `100/104/108`）**目标状态 == 初始状态**：`goal_files = initial` 且 `goal_language = {name: "python"}`，而文件本身是可推断为 Python 的 `.py`——语言 selection 在 `restore_language_state(None)` 后由 `programming_language.resolve` 语义自动成立。

后果链（实测，`_execute_matrix` 264 episodes）：

| context | episodes | zero-step（无任何动作） | 其中非 baseline | success |
|---|---|---|---|---|
| 100 / 104 / 108（block 0） | 22 各 | **20 / 22** | **19 / 22** | 21 / 22 |
| 101/105/109（block 1） | 22 各 | 0 | 0 | 6 |
| 102/106/110（block 2） | 22 各 | 0 | 0 | 6 |
| 103/107/111（block 3） | 22 各 | 0 | 0 | **0** |

**全矩阵非 baseline 零步 episode 合计 57 个（19 × 3 context）**，全部集中在 block 0。

block-0 context 中，除 `none`（1 步失败）与 `member-a`（2 步真实执行）之外的 **9 个干预 cell 连一步都没执行**就判成功：`_member_episode` 在 tick 循环开头即 `if p52a._goal_reached(environment, task): return finish("goal_reached")`，于是 `steps=[]` → `_project` 产出**空事件 episode**。

实测（context 100/104/108 的 11 个干预 cell × 2 重复，逐条一致）：
```
active=none              steps=1 success=False calls=[]
active=member-a          steps=2 success=True  calls=['member-a']
active=member-b          steps=0 success=True  calls=[]      <-- 未执行
active=member-c          steps=0 success=True  calls=[]      <-- 未执行
active=member-d          steps=0 success=True  calls=[]      <-- 未执行
active=member-a-member-b steps=0 success=True  calls=[]      <-- 未执行
active=member-a-member-c steps=0 success=True  calls=[]      <-- 未执行
active=member-a-member-d steps=0 success=True  calls=[]      <-- 未执行（即 admitted group）
active=member-b-member-c steps=0 success=True  calls=[]      <-- 未执行
active=member-b-member-d steps=0 success=True  calls=[]      <-- 未执行
active=member-c-member-d steps=0 success=True  calls=[]      <-- 未执行
```

## 3. 被污染的具体结论

P5.2b 报告 admitted 的唯一 group 为 `member-a+member-d`：

```
contribution = -0.6667   interaction = 0.2222   holdout_interaction = 0.2222
```

该 `interaction = 0.2222` 的构成，完全落在 block-0 的**空事件 cell** 上：

- `(F,F)` baseline：`outcome = -1.0`（无决策来源 → `all_members_exhausted`，1 步）
- `(T,F)` `member-a` 单体：`outcome = +1.0`（真实执行 2 步达成）
- `(F,T)` `member-d` 单体：`outcome = +1.0`，但**零步、无任何动作、无事件**
- `(T,T)` `member-a+member-d`：`outcome = +1.0`，但**零步、无任何动作、无事件**

`interaction = pair − first − second + baseline = 1 − 1 − 1 + (−1) = -1 × ...` 的逐 context 均值落在 `0.2222`，其**唯一来源是「`member-d`/pair cell 未执行却记成功」**。这不是「超出单体的联合增益」，而是**目标状态在干预生效前已满足**导致的伪成功。

**因此：P5.2b 的 1 对 admitted group 应被判定为无效证据**；5 对 `low_confidence` 拒绝反而未受影响（拒绝在方向上是保守的）。

补充：block-0 的 `contribution = -0.6667`（组合成本）同样由该伪成功结构生成，不可用于任何结论。

## 4. 同时证伪的更强命题

P5.2b 报告 `gates.real_execution = true`、`cell_completeness = true`——两门都**未检出**这一问题：

- `real_execution` 的判据是「存在已执行动作且 provenance 合规且安全违规 0」，对**「整个 cell 零执行」**无能为力：`executed_entries` 非空即通过，空 cell 只是不加分。
- `cell_completeness` 的判据是 episode 计数 `== 12 × 2 × 11`，对**内容为空**的 episode 同样无能为力。
- 二者都只检查**是否存在**，不检查**干预是否真的发生了**。这是判据设计缺陷，不是执行疏漏。

由此，P5.2b 的九门应重述：**门 6（real_execution）与门 3（cell_completeness）判据不足，P5.2b 的 `group_causal_corpora_supported` 结论不成立**。

## 5. P5.2c 的阻塞判定（按冻结预注册 §7 出口）

预注册 §5 门 2/3 的实际执行返回：

- 门 3 `unseen_combination_identity`：**不可满足**——4 成员下 pair 面被 6/6 占满，未见组合集合为空。因此**本预注册的设计在当前成员数量下结构性不适配**，不是执行失败。
- 门 2 `evidence_admissibility`：形式可过，但**其所消费的 train-only 证据已被 §3 污染**（block-0 的伪成功进入了 `teacher-forced` 之外的估计池）。

按预注册 §7「机械失败回合同」「无因果信号回 P5.2b」，本 gate 走**最保守出口**：

> **P5.2c 停止。先修 P5.2b：存在真实可干预的 cell 与真正的未见组合，再重新预注册。**

具体需要修的三件事（按优先级）：

| 优先级 | 缺陷 | 修法方向 |
|---|---|---|
| P0 | block-0 context（`lang_confirm`）目标 == 初始状态，导致 cell 未执行即判成功 | 排除或重定义该模板：validation context 必须是**非平凡持久目标**（P5.2a 已对 undo 类做过同样排除，但漏了 `lang_confirm` 的「语言 selection 自动成立」路径） |
| P0 | 九门不检查「干预是否真的发生」 | `real_execution` 增加**非空事件断言**（每个非 `(F,F)` cell 必须 ≥1 事件）；`cell_completeness` 增加**零步 episode 计数 = 0（除 `(F,F)` 外）** |
| P1 | 未见组合面为空（4 成员 → 6 pair 全观测） | 扩充成员数（≥5，使 pair 面 > 已观测数）或改 hinge 为「留出一个 pair 不估计」的 train/holdout 划分 |

## 6. 纪律声明

- 本审计**不修改** P5.2b 报告、不修改 P5.2b 预注册；结论以新文档追加，原报告与预注册保留为失败证据。
- 本审计**不重跑** P5.2b 以挑结果；`0abf463f` 的产物与提交保持原样。
- P5.2c 预注册**保持已冻结状态**，但其执行被本审计阻塞；预注册中「4 成员 / 6 pair」的设计前提已在本文披露为不成立，任何后续 P5.2c 执行必须**新预注册**。
- `growth_admitted=false`、`can_promote=false` 不变。
- 本审计的探测脚本为临时脚本，用毕即删；报告不覆写不改绿。
