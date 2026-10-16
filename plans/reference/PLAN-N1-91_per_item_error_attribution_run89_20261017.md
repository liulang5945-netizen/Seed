# PLAN-N1-91 · 已封存 288 行的逐题错因归因（甲路，零查询）· 2026-10-17

owner 于 2026-10-17 在 EXP-N1-90 §6 的三条出口里裁定**甲**：不再花机时、不再开新面，只用已经封存的确认报告做一次逐题错因归因，把"位置读出臂在未见集上与 cut 无差别"这条否证换成下一格的机制假设。本件是那条裁定的预注册，跑在任何归因读数之前落盘。

## 0. 本件不许声称什么（写在最前面）

本件的产出是**关联读数**，不是能力读数，也不是臂差读数。四条例句在本件里永久无效：①不得由任何分类计数得出"某臂更强"；②不得由错形分布得出"机制已查明"——错形只指认下一步该测哪个变量，指认不了成因；③不得把 `teacher_exact` 与 `free_exact` 的差异读成"训练不足"或"训练充分"，两者量的不是同一件事；④不得引用本件作为 C3、C4 或三器官任何一格的证据。发表资格前置不变（见 EXP-N1-90 §7）。

## 1. 输入与绑定（枚枚点名，缺件即响亮失败）

只读两枚在库件：`reports/taiji_n1_transition_factorial_confirmation_run89.json`（已消费的 v5 确认报告，288 行＝9 枚终端 × 32 题）与 `plans/manifests/n1_fresh_confirmation_v5_20261016.json`（题集本体，32 枚 episode）。四条绑定守卫先跑，任一失败则 rc=2、不写任何输出：

- B1 行数与形状：`rows` 恰 288 行，`arms={cut,position,terminal}`、`seeds={11,12,13}`、`splits={holdout}`、每 (seed,arm) 恰 32 行、`id` 集合与题集 32 枚一致；
- B2 题集同源：报告 `confirmation_seals.v5` 记录的题集摘要等于题集本体的现取摘要（按报告登记的 `evidence_sha256_policy` 口径，两枚各自命名并披露，不许只报其一）；
- B3 聚合复算：由 `rows` 逐行重算的 `free_exact` 三臂合计必须与报告自己的 `aggregate_exact`（position 79／terminal 33／cut 80）逐项相等，九组 `summary[*][*].free_exact` 同样逐项相等；
- B4 字段语义锚：对每一行，`output_hex` 解码后的符号序列必须与 `generated_steps`、`stop_reason` 自洽——`boundary` 行满足 `generated_steps == 符号数 + 1`（末位是 256，不写进 hex），`cap` 行满足全体 `generated_steps` 同值且该值是全表最大观测步数（本件不去猜上限住在哪个常量，只把观测到的同值与上限键名一并披露）；`target_symbols` 末位必须是 256。不自洽的行数必须为 0，否则本件的分类轴失效，当场拒绝。

## 2. 分类轴（预注册，名字冻死，跑完不许改轴）

**轴一 · 三臂一致性**（按 `id` 对同一 seed 的三枚终端配对，288 行归并为 96 组三元组）：`all_three_right`／`all_three_wrong`／`exactly_one_right`（点名是哪一臂）／`exactly_two_right`（点名缺哪一臂）。

**轴二 · 错形**（对每一行 `free_exact=false` 的行，按 `output_hex` 解码序列与 `target_symbols` 去掉末位 256 后那段核心序列的首个分歧位置判）：`empty_output`（输出零符号）／`first_symbol_wrong`（输出非空但分歧在 index 0）／`prefix_then_diverge`（公共前缀长 ≥1 且双方都还有余符）／`truncated_prefix`（输出是核心的真前缀）／`overshoot`（核心是输出的真前缀且输出更长）／`equal_output_not_exact`（输出与核心逐符相同却不判对，预期 0 枚，出现即说明 `free_exact` 另有条件）／`no_boundary_cap`（`stop_reason=cap`）。同一行可命中 `no_boundary_cap` 与另一条形态类，计数表须两列并立，不许相加；六条形态类必须互斥且完备，落不进任何一类的行数记 `unclassified` 并作为合格条件之一（必须为 0）。

**轴三 · 选择侧还是续跑侧**（每行两个读数同行）：`teacher_exact=true 而 free_exact=false` ⇒ `compounding_only`；`teacher_exact=false` ⇒ `selection_side`（并按 `teacher_tokens_correct` 与目标长度之差给出 `tokens_missing` 分布）。这一轴是本件的主要产出：它把"A 支线的旧 oracle 结论（选对事件后位置随机仍 19–21/104 ⇒ 瓶颈在选择侧之外）"在 N1 原生题面上重做一遍。

**轴四 · 分布表**：kind（`lookup`／`revision`／`transition`／`unknown`）× seed × arm 的 free 与 teacher 两列计数，全部由 `rows` 现算，且必须与 `summary` 一致（B3 已守）。

**轴五 · 同错输出率**：三臂同错的行里，两两 `output_hex` 相等的比例（分臂对给出），以及三臂全同的比例。这一轴量的是"三枚终端是不是在同一点上以同一种方式错"，它是臂间可替换性的直接证据。

## 3. 出口（预先写好，读数出来后不许另立）

- 若 `selection_side` 占 free 错行的多数 ⇒ 下一格是**读通路侧**：位置读出与 cut 在证据选择上没有差别，要测的变量是"哪条通路提供了候选"，不是"位置怎么排"；本件据此提出但不自选，仍需 owner 裁。
- 若 `compounding_only` 占多数 ⇒ 下一格是**续跑侧**：打分本身对、自由生成崩，要测的变量是生成预算与边界符，与本件 §0③ 的禁令同时生效。
- 若轴五三臂同错且输出全同的比例高 ⇒ 三枚终端在这 32 题上近乎可互换，则任何"读出臂"方向的题都应当先做**可互换性检验**再谈能力，本件把这条写进 09 的队首候选。
- 若 `unknown` 类的错集中在 `first_symbol_wrong` ⇒ 弃权表达（答 `unknown`）本身是短板，与位置读出无关，属输出头那一格。
四条出口互斥性由读数保证：轴三两支互斥、轴五与轴一不冲突。本件不自选出口，只出版读数与点名到哪条出口为真。

## 4. 一次性、失败路径与预算

仪器 `scripts/training/attribute_taiji_n1_confirmation_errors_run89.py` 只读那两枚件，输出单枚 `reports/taiji_n1_confirmation_error_attribution_run89.json`，写靶已存在即拒绝（单枚纪律沿用 89 批核验仪 `verify_...run89_batch.py:100` 那条）；`--readiness` 档只走 §1 四条守卫不写盘。零查询、零训练、零远端动作；预算 ≤60 秒 CPU 单批，实测用后即报。守卫测 `tests/taiji_native/test_n1_run91_error_attribution_contract.py` 必须在真实 288 行上双支都走（分类器对一个手工构造的错形样本必须能为假），并把 §1 四条守卫各做一支"挪一格必须红"的负对照。

判绿只认汇总行：本件的合格条件是 B1–B4 四条守卫全真＋轴一五类计数之和＝96＋轴二各类计数之和＝free 错行总数（`no_boundary_cap` 单列除外，按 §2 两列并立）＋`incoherent_rows=0`。任一不满足 ⇒ 本件记 `stopped` 并保留 partial 行，不许把 partial 读成结论。
