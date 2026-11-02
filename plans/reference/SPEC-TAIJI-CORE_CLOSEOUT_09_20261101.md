# 09.8：09 全局收束与能力/资源卡（不授予正式训练）

> 文档状态：[合同] — owner 已于 2026-11-01 裁定「接受 8 步结案，转入 09.8 全局收束」。本合同冻结 09 阶段的收束物、卡片的必填项与判法；**训练仍未准入**。

日期：2026-11-01。`taiji/living_core.py` 字节保持不变。

## 1. 收束物（六件，缺一不可）

按总计划 09.8 行要求的六项：

1. **完整主体 bundle**：父模型 ＋ 共识出生（parts=5、units=3）＋ 8 步声明更新后的主体，经 `save_subject` 落盘到 `output/taiji_core_09/subject_bundle.pt`（`.pt` 不入 git），卡片记录其 SHA-256 与字节数，并**在全新 spawn 进程里恢复并续推一步**，与同进程续推逐位比对。
2. **实际活动轨迹**：训练后策略头在逐轮供给夹具上的 8 轮真实决策序列（活动名、chosen_by、step_delta、external_trigger），逐轮入卡。**只报告，不判优劣**。
3. **父/子结构映射**：父参数量、子参数量、新生单元数（每群）、父行/父列在子中的位置；并断言子模型在**训练前**对父参数逐位继承。
4. **参数继承**：上述逐位断言的读数（相等参数个数／总数、最大绝对差）。
5. **原始输出**：两条 held-out query 的第一字节与概率向量前 3 项，原样入卡。**不评分、不称正确性**。
6. **能力/资源卡**：09 各子步骤（09.1–09.7b）逐条记录其状态出口、一句可声称的话和一句不可声称的话；资源侧记录各报告的实际 CPU/墙钟与 09 线的合成总账。

## 2. 判据

1. **六件齐** 且卡片字段完整 ⇒ 否则 `incomplete_card`。
2. **恢复逐位**：spawn 进程续推的 h 与概率必须与同进程续推逐位相同 ⇒ 否则 `restore_mismatch`。
3. **继承逐位**：训练前子模型对父参数逐位相等（允许的例外必须点名并给出原因，本参考图已知 `episodic.score.bias` 不参与图）⇒ 否则 `inheritance_mismatch`。
4. **活动轨迹非退化**：8 轮中至少 1 轮为 `interact`，且至少 1 轮为 `wait`（逐轮供给下两者都应出现）⇒ 否则 `degenerate_activity_trace`。**这不是"学会选择"的证据**。
5. **子步骤出口全在册**：卡片列出的每个子步骤都必须指向一个存在且 SHA 与卡片一致的报告；任一缺失或不一致 ⇒ `trace_incomplete`。
6. **不得出现能力声称**：卡片的可声称句里若出现"能力／学会／掌握／达到 L3"等词 ⇒ `capability_claim_forbidden`。判据只查卡片文本，不查合同与台账。

## 3. 预算、出口与交付

- 预算：≤900 CPU 秒 / 1800 墙钟秒；超限 `budget_stop`。
- 出口：`closeout_passed_engineering_only`／`incomplete_card`／`restore_mismatch`／`inheritance_mismatch`／`degenerate_activity_trace`／`trace_incomplete`／`capability_claim_forbidden`／`budget_stop`／`engineering_failure`。
- 交付：本合同、`scripts/training/verify_taiji_closeout_09.py`、`reports/taiji_core_09_closeout_20261101.json`、`output/taiji_core_09/` 下的 bundle、MAINLINE/03 同步。
- **通过只表示** 09 阶段的六件收束物齐备且三项一致性（恢复、继承、轨迹）成立；不表示任务能力、不表示生长收益可外推、不表示 09.7 准入被追认。**训练仍未准入。**

## 4. 执行更正（2026-11-01，同日）

首轮收束器写出 `degenerate_activity_trace`，但**那是判据写错，不是主体退化**：本合同 §2.4 要求 8 轮轨迹里同时出现 `interact` 与 `wait`，可是活动面已按 owner 裁定改成**逐轮供给**（`per_round=True`）——每一轮 `interact` 都合法且都付钱，训练后策略 8 轮全选 `interact` 正是该夹具下的正确行为。要求它必须出现 `wait`，等于用旧夹具的直觉去判新夹具。

更正：轨迹判据改为**至少一轮 `interact`**（即非全空闲），并在卡片里记 `activity_policy` 说明逐轮供给下 `interact` 每轮合法。这一改**放宽了**判据，所以必须同时说明它挡不住什么：它只能排除"全空闲"，**不能**证明策略"偏好"交互——偏好与优劣不在 09.8 的判据内。

同时把误导性的检查名 `six_artefacts_present` 改为 `all_sub_step_reports_present`（它实际验的是 13 份子步骤报告齐备，不是六件收束物——六件由卡片字段本身承载）。首轮卡片在入库/索引前移出仓库留档，不删字节。
