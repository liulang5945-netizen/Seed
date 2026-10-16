# PLAN-N1-90 九枚跨三族绑定与 v5 未见确认执行件（2026-10-17，判据先冻）

本件只做一件事：把 PLAN-N1-89 §4 的九枚终端绑定写成可执行、可否证的取数件，并把 [PLAN-N1-85](PLAN-N1-85_transition_factorial_frozen_confirmation_20261016.md) 的四门判据原样搬到九枚"同学习率日程、只差读取臂"的检查点上执行一次。判据数值一个不改、不新增、不换测面；训练充分性结论在 [EXP-N1-88](EXP-N1-88_run88c_batch_adjudication_20261017.md)，本件不重复声称。

## 1. 输入（三族九枚，逐枚按字节点名）

- position 三枚＝`reports/taiji_n1_transition_factorial_lr_decay_run88c.json` 族登记的 `s{seed}_position_final.pt`；
- terminal 三枚＝`reports/taiji_n1_transition_factorial_lr_decay_run89a.json` 族登记的 `s{seed}_terminal_final.pt`；
- cut 三枚＝`reports/taiji_n1_transition_factorial_lr_decay_run89b.json` 族登记的 `s{seed}_cut_final.pt`；
- 三族批件必须各自 `complete=true` 且带同族批核验件（run88c 用 `..._run88c_batch_verification.json`，89a/89b 用 `..._run89{a,b}_batch_verification.json`），核验件里 `execution.model_loads=0`、`parameter_updates=0`、未见查询各 0；
- 三族 `protocol.learning_rate_schedule` 必须同为 0.001 起、`final_cycle_start=18432`、`final_cycle_lr=0.00025`、`adam_moments=continuous_unmodified`；三族 `code_identity.frozen_runner_sha256` 必须同为 `38018dbe08b9eb4c35d0b5c97196727b34f94efee68d68113e80269b8d3c9491`；三族 train 源摘要、curriculum 与评分视图摘要必须逐枚相同 ⇒ 九枚之间唯一自变量是读取臂；
- 控制器 SHA 允许不同（run88 与 run89 两代编排件），差异必须在本件报告里逐枚点名，不许写成"九枚同一控制器"；
- 题集＝既有 v5 `plans/manifests/n1_fresh_confirmation_v5_20261016.json`，SHA `22ba37976bc6eaaaf2425b84714a6efa6e95e7e5aa05f32eb3a7c0c10d9ffc2d`，32 题、四类各 8；显式 holdout 路径，不给 oracle 地址/解析器/条件标签。

## 2. 执行形状（复用 85 冻结执行器，不重抄生成链）

执行件 `scripts/training/eval_taiji_n1_transition_factorial_confirmation_run89.py` 只做绑定与身份，其余全部走 85 那台执行器的函数本体：`measure`/`evaluate`/`adjudicate`/`audit` 一律 import 复用（同 §4 的 `readiness`/`identity`/`PROTOCOL`/`REPORT`/`ATTEMPT` 以进程内补丁替换成九枚形状），因此确认会话前缀仍是件里那句 `confirmation85-`，本件如实登记这一处实现名与批次名不同名，不借此声称"跑的是 84 单族收据"。

每枚模型：完整 `torch.load(weights_only=True)` → `train.load` → 全 bundle 指纹等于检查点自带 `expected_signature.bundle` 且等于 `candidate` 指纹；每题后清临时会话；九枚全程 0 参数更新、0 优化器步、0 副本签名步，逐枚取 `paired.fingerprint` 前后不变（变了即 `confirmation changed parameters…` 响亮失败）。总上限 180 秒 CPU（85 原值），串行单线程。

## 3. 判据（85 §原14判据，数值一字不改）

- `each_position_seed`＝三枚 position 每枚 `free_exact ≥ 16/32`；
- `each_position_kind`＝每枚 position 四类各 `≥ 2/8`；
- `advantage_terminal`＝position 三枚总分 `≥` terminal 三枚总分 `+12/96`；
- `advantage_cut`＝position 三枚总分 `≥` cut 三枚总分 `+12/96`；
- 合取：`behavioral_pass` 只在上述四条同时为真时成立；`288` 行、`9` 次完整加载、`cpu_seconds ≤ 180`、`sources_unchanged`、`code_unchanged` 六件齐才判 `complete`，`passed = complete ∧ behavioral_pass`。

不许互换的三把尺：单枚 `free_exact`（成句）不等于四类分项之和，四类分项不等于 `strict hit`；总分只在同 32 题、同臂分母内比较。

## 4. 一次性消费与失败路径

首题之前独占写 `reports/taiji_n1_transition_factorial_confirmation_run89_attempt.json`，写入即 v5 已消费：中断、部分行、判红都不允许重跑、重抽、换 seed、改线值或复用该面。报告只在六件齐时判工程完整，否则如实 `stopped`/`interrupted` 并保留 partial 行与查询计数。

- 四条全绿 ⇒ 只支持"N1 的 C3 未见确认这一格可判"，仍 `can_promote=false`，不替换默认模型，不声称 C4/C5/三器官结项；
- 任一臂门失败 ⇒ 按 PLAN-N1-89 §7 第二行收回"position 在未见集上领先两臂"与"衰减配方支持未见迁移"两句，并点名 EXP-N1-88 提交 `2d89c3595` 只保留训练充分性那半句；训练充分结论不撤；
- 若 `complete` 而非 `behavioral_pass` ⇒ 面已消费，此后任何"未见能力"主张必须另立题集并先经 owner，本件不许把它读成"再跑一次"的许可；
- `advantage_*` 与 `each_position_*` 分别点名：只涨绝对门不许写成臂差，只大臂差不够，四门是合取。

## 5. 前置与预算（现取，不估）

前置＝89a 与 89b 两批各自的核验件为真；本件在 §1 全部通过前不得加载任何模型（`readiness` 先行，缺任一枚即响亮失败、attempt 不写、v5 不消费）。执行预算 ≤180 秒 CPU 单批，与两批训练预算各自独立；零外部配额、零远端动作。

〔2026-10-17 落盘前的零算力自测，逐条实跑而非推断〕①拒入支在真实代码上出 **rc=2** 且点名缺件（`terminal batch report is missing: taiji_n1_transition_factorial_lr_decay_run89a.json`），不是 traceback 的 rc=1；②v5 面现读＝32 题、四类各 8、`split` 全为 `holdout`、id 全唯一，字节 SHA `22ba37976bc6eaaa…` 等于 76 准入件里那枚封条，`face.validate` 不抛 ⇒ §1 的题集那一半已为真；③§4.2 那句"三族同源"今天可机检：run88c 报告 110 枚守卫、89a 与 89b 预检各 107 枚，**三族交集 98 枚且逐枚摘要相同（不一致 0 枚）**，交集里含 `n1_paired_questions_20261015.json`／`n1_native_evidence_curriculum_v3_20261015.json`／`n1_fresh_transition_factorial_20261016.json`／v5 题集本体与九枚 fresh 零态档；④跨族装载补丁在真实 89a 零态档上验过：该档 `protocol.schema` 与 84 默认 protocol 不同名，`load_with_family_protocol` 仍装载成功且把 `train.PROTOCOL` 复原、`_observations` 为 0。合同测 18 passed（含 rc=2 与"拒入时 attempt 不落盘"两支）。

## 6. 出口

本件只有一格产出：`reports/taiji_n1_transition_factorial_confirmation_run89.json` 与判读件 `plans/reference/EXP-N1-90_*`。发表资格前置不变——未见读数之外仍需保持集、C4 净增益、三器官各自行为证据；本件不结 N1，也不开 N2/N3/N5。
