# PLAN-N1-94 · v6 确认子集的一次性查询与四门判级（先冻，待范围检验通过后才发射）· 2026-10-17

本件在 **v6 题面还不存在**的时候写死，目的是让"签完参数之后"没有任何一步需要临场判断。它承接 PLAN-N1-93 的产物：题面分成牺牲／确认两半，牺牲半先做动态范围检验（`verify_taiji_n1_v6_range_check.py`），只有 `range_check_passed=true` 才允许本件对**确认半那 20 题**发起一次性查询。

## 0. 发射前置（缺任何一条 ⇒ rc=2，attempt 不写，题面不算消费）

- P1 已签署：`SIGNED_DISTRIBUTION` 来自 owner 对 PLAN-N1-93 §1 的签字，题面 `plans/manifests/n1_fresh_confirmation_v6_20261017.json` 在库且与 `reports/taiji_n1_v6_face_construction_checks_20261017.json` 的 `face_sha256` 同值；
- P2 尺已量过：`reports/taiji_n1_v6_range_check.json` 存在、`adjudication.range_check_passed=true`、`confirm_subset_queried=false`、且该件里的 `subsets_queried` 只含 `"sacrifice"`；
- P3 九枚端点在库且字节未变：走 90 那台族绑定（`family_receipts()` → `endpoints()`），三批报告与两份批核验件任一缺失或摘要不符即拒；
- P4 三族共享守卫一致：`cross_family_guard_conflicts()` 为空；
- P5 消费纪律：本件写靶（报告与 attempt）此前不存在。

## 1. 输入与绑定

题面＝v6 确认子集（20 枚 episode，id 清单取自题面 `subsets.confirm`，本件不自己切分）；端点＝run88c position ×3 ＋ 89a terminal ×3 ＋ 89b cut ×3；生成上限＝沿用被测执行器里的实现值并在报告里披露其住所在何处（DEBT-G150 的教训：`protocol.max_generated_bytes` 无消费处 ⇒ 本件必须写"实际生效的那一枚在哪"）；每次装载后逐枚重读端点字节；判据与门值以 `sha256` 钉进报告的 `protocol`。

## 2. 判据（四门照抄 85／90 的**原绝对值**，分母随题面缩放的方案列在旁边供 owner 另裁；未经签字一律按左列执行）

| 门 | 本件执行值 | 比例照搬会得到 | 说明 |
| --- | --- | --- | --- |
| `each_position_seed` | ≥ 10/20 ×3 枚 | ≥ 16/32 → 同比例 10/20 | 这一条比例与绝对值同形（半数），无需另裁 |
| `each_position_kind` | ≥ 2/5 ×4 类 ×3 枚 | ≥ 2/8 → 同比例 1.25→2/5 | 取整后与绝对值一致 |
| `advantage_terminal` | ≥ **12/60** | ≥ 7.5→8/60（同比例） | 本件取**不缩放**的原绝对值，即更严的一侧 |
| `advantage_cut` | ≥ **12/60** | ≥ 8/60（同比例） | 同上；理由＝90 那轮证明"领先量"最容易被放松，宁可一开始就取严 |

四门为**合取**；`advantage_*` 与 `each_position_*` 分别点名，只涨绝对门不写成臂差，只大臂差不算过。发射前若 owner 未对右列另行签字，任何人（包括我）不得改用右列。

## 3. 一次性消费与失败路径

首题之前独占写 `reports/taiji_n1_transition_factorial_confirmation_v6_attempt.json`，写入即该 20 题已消费：中断、部分行、判红都不允许重跑、重抽、换 seed、改线值或复用该面；报告只在六件齐时判工程完整，否则如实 `stopped`／`interrupted` 并保留 partial 行与查询计数。`complete=false` 只允许由**真实**的六件缺失造成——摘要口径混用那一格（DEBT-G147）在本件里预先处理：来源守卫同时出版 `sha_raw` 与 `sha_json_normalized` 两枚并各自命名，判据只取 `sha_raw`，另一枚仅作披露，避免把归一差异读成"工程不完整"，也避免反过来把真实改动抹平。

## 4. 出口（预先写好，读数出来后才点名哪条为真）

- 四门全绿 ⇒ 仅支持"N1 的 C3 在**新面**上未见确认这一格可判"，仍 `can_promote=false`，不声称 C4／三器官／联合能力；
- 任一门假 ⇒ 收回两句并点名出处：①"位置读出臂在未见集上领先两臂"（EXP-N1-90 已收回一次，本件若再判红则永久失效，不得再以换题面重试）；②"换更难的新面即可打开 C3"——这一句是本件独有的新句，判红即收回，且必须同时说明"题集难度不是未证变量"；
- `complete` 而非 `behavioral_pass` ⇒ 面已消费，此后任何"未见能力"主张必须另立题集并先经 owner；
- 三条出口互斥由取值保证，本件不自选下一格。

## 5. 预算与仪器复用（不重抄生成链）

执行器＝复用 85 的 `evaluate` 与 90 的族绑定（`load_with_family_protocol`、`cross_family_guard_conflicts`、`endpoints`），新文件只写：v6 题集读取、P1–P5 前置、判据的分母适配与两口径摘要披露。查询数＝20 × 9＝180 行，上限 ≤180 秒 CPU 单批（牺牲半实测的每秒成本先出，再决定是否收缩端点顺序）；零训练、零远端动作、零外部配额。契约测 `tests/taiji_native/test_n1_run94_v6_confirmation_contract.py` 必须走：P1–P5 各一支"缺件必须 rc=2 且 attempt 不落盘"、四门判级的真假两支与"恰好等于线"那一支、两口径摘要不同源时必须点名差异数（不许静默全绿）、模式互斥与单枚写靶拒绝。

## 6. 现在不做什么

题面未落盘、范围检验未跑 ⇒ 本件此刻只允许存在为文本；发射顺序＝owner 签 §1(93) → `--execute-signed` 落题面 → 范围检验 → 达线后本件才允许 `--execute-approved`。在此之前任何一次对确认子集的查询都视为违规消费。
