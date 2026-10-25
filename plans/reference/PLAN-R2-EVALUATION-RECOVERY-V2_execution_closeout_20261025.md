# R2 评价工程恢复 v2 执行结案

日期：2026-10-25，Asia/Shanghai。状态：**CLOSED_COMPLETE**（R1–R6 全部满足；用户当日确认 48 行判读：四端不可读、不贴题、机械重复，有历史行不连贯）。用户当日采纳[消费边界 v2](delivery-runs/20261021-n-mainline/artifacts/04-evaluation-consumption-boundary-v2.md)，实施计划与判据预注册见[05计划](delivery-runs/20261021-n-mainline/artifacts/05-evaluation-recovery-v2-plan.md)。原 CC-1 与恢复 v1 的 `CLOSED_FAILURE`、原始证据、已消费资格均不变。

## 1. 实际执行

- 工程准入：`admission_001` 在准入保护区计算处停止（0 恢复、0 题面，树已退出，70.703125 CPU／71.328 wall）；修正保护区后 `admission_002` 在真实 Windows Job/IPC/token/NTFS 下完成**四端只读真实恢复**（345.578125 CPU／347.719 wall），未进入 selector。
- 正式消费：`formal_001` attempt `1b679707823e5cf69dfc6721537fdd75a055066465ae85ed`。四端恢复 receipt 全部 `complete=true/equal=true`，`episode_count=525`、`global_step=113209`、参数 971,427（1M 全轮）与 278,010（其余三端）。
- **唯一耐久 marker 在 selector 之前发布**（`published_at_utc=2026-10-25T05:58:20Z`），此后未重试、未换名、未补题、未再次读题。
- 生成：`query_attempts=156`、`persisted=156`、逐端点 39、`reuse=0`；0 参数/优化器更新；`can_promote=false`。face SHA256 `be674a18…`，attempt/result/generations/ledger 摘要见[执行审计](../../../reports/taiji_r2_evaluation_recovery_v2_execution_audit_20261025.json)。

## 2. 能力读数（描述性，不设新通过门槛）

四端点（**含 parent 基线**）在冻结 decoder 下均输出控制/停止标记串与无词法结构的汉字拼接，无可理解句子。48 行判读工作表中：中文可读性 48/48 不可读，贴题 48/48 不贴题，有历史的行不连贯，四端均为机械重复；端点级格式统计见[判读表](PLAN-R2-EVALUATION-RECOVERY-V2_human_review_20261025.md)。

关键含义：

1. 这不是某一训练臂的相对退化——**基线本身已不可读**，因此本轮无法给出「配方 A 优于配方 B」的排序；能确认的是四条路径都没有产生可用中文对话表达。
2. 这是**能力负结果**，不是工程失败：156 条输出、四端恢复、来源/费用与发布均闭合。
3. 该读数只覆盖**冻结 decoder/发射结构**下的可见行为，不能单独证明模型内部没有可用表示。若 decoder 与训练目标格式不一致，同一 checkpoint 也可能被系统性地误读；这属于 CC-2 归因，不在本包内改判据、重跑或再读 dev 面。

## 3. 与预注册判据对账（R1–R6）

| 条件 | 结果 |
|---|---|
| R1 三臂到原终点 | 已完成（三 result、三 readiness、六阶段 final 与独立恢复），本包不重训 |
| R2 身份、终态与费用闭合 | 端点 SHA/源码/预算修订链与窗口账本一致；无活 writer、无未结 reservation |
| R3 四端点准入 | `admission_002` 真实四端恢复通过；正式 attempt 内四 receipt 齐全 |
| R4 一次固定 W5b | 156 attempts＝outputs＝persisted，逐端 39，reuse=0，沿原冻结题面与 decoder 一次 |
| R5 固定人工判读 | 48 行、四维、依据与不可判项已逐行填写，责任判读人于 2026-10-25 确认全部初判；四端不可读/不贴题/机械重复，有历史行不连贯 |
| R6 结案落盘并提交 | 本文件、执行审计、03/05/黑板/10 同步与提交 |

## 4. 保持覆盖

本包不包含独立的旧能力保持探针：四端点均在冻结 decoder 下不可读，无法在该面上分离「保持」与「从未形成」。保持覆盖仍由 CC-4/CC-6 按其准入条件度量；本包不宣称 L2/L3/L4 通过或旧能力已保持。

## 5. 实际成本（同一评价窗口，CPU/wall 各 14,400 秒）

| 项 | CPU 秒 | wall 秒 |
|---|---:|---:|
| 准入 001（工程失败） | 70.703125 | 71.328 |
| 准入 002（四端真实恢复） | 345.578125 | 347.719 |
| 正式一次（发布前实测） | 651.703125 | 966.015 |
| v2 合计 | 1,067.984375 | 1,385.077 |
| 窗口剩余（含历史尾账预留与 v1 已测） | 9,847.53125 | 12,740.687 |

正式阶段的 result/cost receipt 与发布成本按冻结语义单列：`cpu/wall` 为 finalization 起点实测，result/review/cost receipt 的发布成本由 300 秒 finalization reserve 覆盖，不宣称其已被测量。窗口账本的每条事件见 `…_recovery_v2_window_ledger.jsonl`（运行态 JSONL 按仓库忽略规则留在本机，摘要与 SHA 在审计中）。

## 6. 证据与已知瑕疵

- 机器事实位于 `reports/taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_paired_eval_recovery_v2_admission_00{1,2}_*` 与 `…_formal_001_*`；版本化[执行审计](../../../reports/taiji_r2_evaluation_recovery_v2_execution_audit_20261025.json)保留关键文件 SHA、marker、计数与费用。
- 160 个 worker token claim（`…_formal_001_attempt.json.*.claim.json` 等）为运行态一次性凭证，留在本机不入库；其文件名与 SHA 已完整列入执行审计的 `evidence_sha256`，可据此核对每次任务授权。
- surface 命名沿用 extension 家族：`review` 文件扩展名为 `.json`，内容为 Markdown 展示稿；不因此重跑或改历史绑定。
- 机器合同 r1 因准入 001 证明 120 秒/端保护区不可行而由 r2 取代；r1/r2 均为不可变证据，正式阶段只用 r2（marker 绑定其 SHA）。`--validate-only` 双预检多耗的约 65 CPU 秒已计入窗口，不影响结论。
- 执行完成后仅对 v2 provider 与新增契约测试做 black 空白重排（行为不变）；执行时快照 SHA 由合同 r2 与审计固定，**不刷新 r2、不重跑**。因此当前两个文件字节与 r2 的 source 列表存在纯格式差异，任何后续 preflight 都会按 source 漂移拒绝——这是预期的封存而非缺陷。

## 7. 出口与唯一后继

责任判读人已确认 48 行判读，R5 完成，本包按 **CLOSED_COMPLETE** 收口：能力读数为「四端不可读、无配方排序」，不追加训练、不补题、不再读 dev 面。

唯一后继建议：进入 **CC-2 归因**——先用训练侧（非 dev）输入检验同一 checkpoint 在训练目标 decoder 下能否产生可读中文，以区分「发射/解码结构失配」与「prompt→response 依赖从未进入监督」。该诊断不得复用已消费的 dev 面，也不改本轮判据；开工前按 N 纪律补三件套（VISION 对账、PLAN-Nxx、SPEC 判据预注册）。
