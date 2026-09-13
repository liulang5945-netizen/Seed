# M5 K 轴 scorecard v7：A8 评审结论入账与证据 veto 机械翻转

> 注册日期：2026-09-12。前置：[A8 晋级评审合同 §7](M5_K_A8_PROMOTION_REVIEW_20260912.md)（`promotion_review_recommended`——四项裁决独立批准）与 [scorecard v6 合同](M5_K_AXIS_SCORECARD_V6_CONTRACT_20260911.md)。本 v7 在不重训、不重算任何 cell 的前提下新增 `promotion_review_evidence` 证据线（评审结论只读转录 + 合同文档 sha256 校验），并把三项证据 veto 机械翻转为 `true`。执行顺序以 [03_CURRENT_EXECUTION.md](../active/roadmap/03_CURRENT_EXECUTION.md) 为准。

## 1. 目的与唯一变更

- **唯一变更**：新增 `promotion_review_evidence`（来源 = A8 评审合同 §7 批准结论 + SGK v1 superseded 状态横幅，只读转录 + sha256 记录）；全部证据线机械复用 v6 reducer（`build_v6` 原样调用），v6 source digest 逐位校验零漂移。
- **唯一门翻转（3 项，均为已批准的证据裁决）**：`parent_retention_baseline_present`、`same_parent_continual_s_g_k_evidence`、`resource_rollback_old_capability_gate`：`false → true`。
- **保持不变（fail-closed 设计）**：`default_runtime_owner_attached=false`、`learning_mechanism_attached_default_runtime=false`（待附着工程步执行并通过自身验收门后由**下一个** scorecard 翻转——评审合同 §7 勘误已修正原文的 v7 编号笔误）；因此 `promotion_gate=false`、`can_promote=false` 机械保持。

## 2. 固定输入与内容寻址

- K1/K2/K3、C 阶段、v2–v5 scorecard、P4.12/P4.13/P4.14、rollout review、attachment manifest：与 v6 完全相同（v6 报告为 digest 一致性校验基准）；
- **A8 评审合同**：`plans/reference/M5_K_A8_PROMOTION_REVIEW_20260912.md`（校验存在性标记：`promotion_review_recommended`、四项裁决全部批准、常设上限更高准则）；
- **SGK v1 合同**：`plans/reference/M4V2_SGK_PROMOTION_COURSE_PREREGISTRATION_20260910.md`（校验 superseded 横幅标记；sha256 记录入证据线）。

## 3. promotion_review_evidence 字段（只转录，不重算）

- `source_contract` / `source_contract_sha256` / `superseded_sgk_contract` / `superseded_sgk_contract_sha256` / `decision_date=2026-09-12` / `outcome=promotion_review_recommended`；
- `dispositions`：四项裁决的批准口径逐项转写（P2.4→review 保持链、P4.14 同 parent S/G/K 课程 + S 架构边界、SGK v1 后继取代映射、附着授权待执行）；
- `standing_decision_principle`：后续决策点优先上限更高选项（以不违反 fail-closed 纪律与已冻结判据为前提）。

## 4. promotion veto（v6 全部保留 + 三项翻转）

`promotion_gate = all(promotion_gates)` = false（两项附着 veto 仍 false）；`can_promote = false` 固定。**v8 翻转条件（冻结）**：附着步按其预注册执行完毕、4/4 cell 全门通过后，v8 翻转两项附着 veto——届时五项 gate 全 true，`promotion_gate` 机械置 true，`can_promote` 机械置 true，**最终晋级决定仍须独立批准**。

## 5. 不做的事

- 不重训、重跑、不触碰任何 frozen 报告语义；不改产品代码、不接默认 runtime（附着预注册批准前的 fail-closed 边界不变）；
- 不把评审结论写成 promotion：v7 只转录已批准的裁决；
- 不修改 v6 报告与更早历史产物；SGK v1 文档只加状态横幅，内容不改写。

## 6. 产物与唯一后续动作

- auditor：`scripts/training/audit_taiji_m5_k_axis_scorecard_v7.py`；
- report：`reports/taiji_m5_k_axis_scorecard_v7_20260912.json`；
- 路线图同步 + 独立提交；v6 报告保留为历史、不可覆盖；
- 唯一后续动作：**默认 runtime 附着工程预注册已冻结**（[M5_K_DEFAULT_RUNTIME_ATTACHMENT_PREREGISTRATION_20260912.md](M5_K_DEFAULT_RUNTIME_ATTACHMENT_PREREGISTRATION_20260912.md)）；执行前不训练、不改产品代码路径以外的默认行为、不读取 sealed。

## 7. 执行记录（2026-09-12，v7 已运行）

1. 机械纪律：`py_compile` / `ruff` / `black` / `mypy --follow-imports=silent` 通过后运行 auditor；v6 source digests 重算后与 v6 报告逐位一致（漂移校验零触发）；翻转前断言三项 veto 在 v6 中确为 false（转录防呆）。
2. 报告 `reports/taiji_m5_k_axis_scorecard_v7_20260912.json`：9 份 source digests（v6 全部 + `A8_PROMOTION_REVIEW`）；`promotion_review_evidence` 完整转录批准结论与 superseded 映射。
3. 门翻转核验：v6→v7 恰好三项翻转，两项附着门保持 false；`promotion_gate=false`、`can_promote=false`。
4. v6 报告与全部历史产物未覆盖；无 owner 解冻、无默认 runtime 接入、无训练。
