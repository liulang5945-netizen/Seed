# 放权决策台账（DECISION-LEDGER）

> 用途：记录**由执行者自行决定**的越界决策，一笔一条，可逐条复核。授权来源统一为：**用户 2026-11-09 授权（放权，须留档）**。
> 覆盖范围：①预算超限；②动封存集或改写已冻结判据；③开新训练窗口；④方向二选一（按“上限更高的方案”取舍）。其余纯实现修复、同口径复测、文档同步不在此立账。
> 纪律：留档在决策当时写，不事后补；每笔独立提交（不推送）；索引行追加在里程碑简报末尾。
> 关联：判据与读数见 [SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md](SPEC-TAIJI-ADDRESS-SYSTEM_20261108.md) §20、[INVARIANT-LEDGER_20261103.md](INVARIANT-LEDGER_20261103.md) §47、[CAPABILITY-CARD_20261103.md](CAPABILITY-CARD_20261103.md) U35。

## 索引

| 编号 | 类型 | 一句话 | 提交 |
|---|---|---|---|
| D-20261109-01 | 判据改写（冻结记录事后修正，判据本身未变） | 仅格式重跑后改写 manifest 的 `runner_sha256`／`freeze_note`，判据与语料未动、读数逐项相同 | 见本笔提交 |

---

## D-20261109-01 冻结记录事后修正（判据未变）

**1. 编号／时间／类型**：D-20261109-01｜2026-11-09（与决策同一会话，记录时间晚于决策约数分钟）｜判据改写类——实际是“冻结 manifest 的记账字段事后改写”，**判据与语料均未变**。

**2. 动了什么（逐字）**：文件 `plans/manifests/taiji_contig_20261109.json`，两处。

- `runner_sha256`
  - 原值：`3aca50ce…`（**完整摘要已不可得**：该字节版本从未入库，其报告已被覆盖；仅保留前缀）
  - 新值：`b804a5d460fb71ebb0187c89c872a661f120e0d4352f2e6d5bb9609b9f118da3`
- `freeze_note`
  - 原值（逐字）：`harness-only revision: fixture classification and the unknown-gate definition changed; address layer, corpus, ranking and thresholds unchanged`
  - 新值（逐字）：`harness-only revision: fixture classification and the unknown-gate definition changed; address layer, corpus, ranking and thresholds unchanged. Runner reformatted in-process (black Mode line_length=100, py312; formatting only) and readings re-taken under the same criteria.`
- 同时**新增**增量块 `post_reading_corrections`（不覆盖原判据，只并列保存原值与新值、原因、比对读数、回退路径）。

**3. 触发原因**：`cf030848a` 首次提交时 pre-commit 的 in-process black 闸判“would reformat”，因为本轮新增的 harness 代码未按 `line_length=100`／py312 排版。格式化**只改空白与换行**，但会改动文件摘要，导致冻结的 `runner_sha256` 不再描述将入库的文件——即“报告的双列 runner sha 与仓库字节不一致”。为保住“报告由恰是该 runner 产出”的可复核性，格式化后按同一判据整跑复测。读数出处：[taiji_contig_20261109.json](../../reports/taiji_contig_20261109.json) 的 `gates`／`write`／`length_buckets`／`fallback`。

**4. 代价**：

- 失去的保证：“冻结后 manifest 逐字节不动”的强保证被打破一次（两个记账字段被改写）。
- 变化的分母／分类：**无**。未知集分母仍 550，`unknown_answerable` 仍 1，回退分母仍 9，`answerable_total` 仍 1,101。
- 旧结论可比性：**全部可比**——重跑读数逐项相同：入表 542/550、`<64 B` 121/129＝0.937984、旧口径 1/550、新口径 0/550、回退 9 条、独立复核违反 0、判定 `continuity_restored`。
- 预算：复测 **46.235 CPU／7.131 wall**（上限 200），**未超限**；该次重跑属“作废重跑”性质，**不计入有效预算**，单列于此。

**5. 替代方案（及否掉理由）**：

- 用 `--no-verify` 直接提交未格式化 runner——**否决**：违反“不得跳过 hook”。
- `--freeze` 升 `freeze_revision 5`——**否决**：判据与语料未变，升版会制造“判据第二次变更”的假象，且会丢掉 `criteria_changed_revision=4` 的记录。
- 不提交 runner、只提交报告与文档——**否决**：报告将引用仓库中不存在的 sha，证据链断裂。

**6. 撤回路径**：

- 回到上一轮（rev 3、runner `3089e1fb…`）：`git revert cf030848a`。
- 回到 rev 4 格式化前字节：**不可行**（该版本从未入库）；可恢复的是读数与判据（本就不变），manifest 原值已逐字保存在 `post_reading_corrections.fields[].before`。
- 若要撤销本笔的台账登记：删除 `plans/manifests/taiji_contig_20261109.json` 的 `post_reading_corrections` 键并 revert 本笔提交。

**7. 授权来源**：用户 2026-11-09 授权（放权，须留档）。

**8. 对已有结论的影响**：U35 与 §20、§47、能力卡的**所有旧读数仍然可比**（判据、语料、地址层、排序、阈值均未变，读数逐项相同）；**没有读数作废**。唯一降级的是“manifest 冻结后不可变”这一形式性保证，已在本笔与 manifest 的 `post_reading_corrections` 中并列留痕。
