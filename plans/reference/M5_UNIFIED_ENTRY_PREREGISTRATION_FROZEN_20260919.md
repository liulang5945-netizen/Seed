# M5 统一执行入口证据包预注册（冻结版）

日期：2026-09-19。状态：**已冻结**（本文提交即冻结）。合同：[统一执行入口证据包合同草案 v1](M5_UNIFIED_ENTRY_EVIDENCE_PACKAGE_DRAFT_20260919.md)（用户授权起草＋常设授权：弹窗超时按推荐项自动推进）。本文件冻结任务、数值线、资源与停止线；**冻结后证据运行前不调任何数值**。

## §1 任务冻结与三机制必要性

**主任务**：`p52a.build_all_tasks()["train"]` 中模板族 `create_undo` 的**首个任务**（文件序确定性选取，task_id 进报告）。
**未见实例**（迁移证据）：同族第二个任务（文件序次位）。

**三机制必要性论证（预注册）**：
- **记忆**：create_undo 要求跨 tick 记住已执行步骤（own_steps cue 语义）＋ P5.1h child 的语料工具知识支撑成员可绑定面；禁用记忆（cue 常数化）⇒ 成员被查询到训练范围外 ⇒ bind 失败上升；
- **后果预测/真实后果**：执行产生真实文件终态（可核验），undo 语义要求记住 undo_token——预测与真实结果分开记录；
- **选择**：两步以上任务中成员失败后必须让位（revision 1）而非字典序重试（revision 0）——禁用选择（revision 0）⇒ 资源浪费/失败路径。

消融矩阵的因果预期（冻结）：disable_memory ⇒ bind 失败率上升；disable_selection ⇒ resource_cost（executed actions）不降反升或任务失败；disable_writeback ⇒ 多轮间无改进（单任务内由 step 间记忆承载，此项为回归性记录）。

## §2 数值线（冻结）

| 项 | 冻结值 |
|---|---|
| 臂 | full / simple_strategy / disable_memory / disable_selection / disable_writeback（五臂） |
| 每臂重复 | 主任务 ×3 repeats ＋ 未见实例 ×1（迁移记录） |
| STEP_CAP | 8（沿用执行入口默认） |
| **L1 共同门主线** | full 臂成功率 ≥ 每个消融臂与简单策略臂的成功率（≥ 比较在 3 repeats 的多数决上） |
| **L2 让位资源线** | full 臂在含失败注入场景的 executed actions ≤ disable_selection 臂（HANDOFF-M4 核心预测：让位省资源） |
| **L3 失败注入线** | 预注入 m0 首选失败的场景中，full 臂让位后仍 goal_reached（disable_memory 臂预期失败，如实记录） |
| **L4 trace/安全线** | 全臂 trace schema 合法（逐事件 tick/kind/rule_revision/bundle_digest）、无 contract_intercepted 之外的越权、bundle digest 逐臂一致 |
| 资源 | 每臂 wall ≤ 120s；总 sweep ≤ 600s；零训练预算（成员 readout 复用 P5.2b 冻结 trial-learner 路径，250 epochs 属仪器口径——若实现判定需重训成员，单独呈批） |
| 停止线 | 任一臂非确定/crash ⇒ 记录并停；不看结果调线 |

## §3 运行与产物

1. runner：`scripts/training/run_taiji_unified_entry_evidence.py`（组装 bundle→逐臂×重复执行→四线判定→报告落盘 `reports/taiji_unified_entry_evidence_20260919.json`）。
2. 判读：L1–L4 全过 ⇒ 共同门证据闭合（`unified_entry_supported`）；部分过 ⇒ `partial`＋归因；机制类失败 ⇒ failed＋欠账保留。两态如实入账，不自动晋级。
3. 预算闸门：成员 readout 训练需要时，runner 拒跑并呈批（同 P5.1h 先例）。

## §4 纪律

负结果不改绿；报告不覆写；冻结后不改本文数值；trace/成员身份与 bundle digest 绑定贯穿；临时脚本用毕即删。
