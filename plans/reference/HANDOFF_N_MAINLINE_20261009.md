# N 主线交接（2026-10-09 03:36Z · 轮次预算将尽时的停靠件）

**为什么要有这份件**：本轮预算见底（37/40），按本仓纪律（`queue-head-is-plan-doc-not-my-tasklist`／
"预算不够时把交接写进 plans/reference 而不是会话总结"），把"已收口／待裁／下一格的精确入口"落到可被
下一个人（或恢复后的我）**机检**的位置。所有数字都指向已入库的台账行与件，不在这份件里新造读数。

## 1. 已收口（有件、有门、有台账行）

| 项 | 终态一句话 | 证据锚点 |
| --- | --- | --- |
| N3 甲（×10 单点 scaling 探针） | 判据链**全仪器化**并判出 `J_N3a′=not_holds`（甲末段 0.352 对 对照臂 0.33966+0.02=0.35966，差 +0.01234） | 件 `reports/taiji_n3a_j_n3a_prime_20261009.json`；台账 08 ㊵-548／549／552；预注册 [PLAN-N3-12](PLAN-N3-12_j_n3a_anchor_upgrade_prereg_20261008.md)／[PLAN-N3-13](PLAN-N3-13_section87_sequence_length_source_prereg_20261008.md)（两件已由 owner 第八次弹窗裁"两件都签"） |
| §8.7 取数面 | 五项在真件上**全部由训练器自述供数**（`sequence_length_source="exit_record"`），旧产物的补数走同源复算件（守卫 C-1…C-4） | ㊵-545／546；`scripts/training/recompute_taiji_n3a_sequence_length.py` |
| N2 乙档（保持集） | 判出 **`cost_persists`**：七列全跌（`cap0:E 1→0`、两档严格命中 6→5、`well_formed` 各跌 6／7 条），分离机检 `measured` 且 `windows_found_in_corpus=0`；收益侧仍标 `not_judged_here` | 件 `reports/taiji_n2_04_retention_pair_20261009.json`；台账 ㊵-559；口径裁定＝owner 第九次弹窗 (乙-1)（条目短于 `NGRAM_BYTES=24` 不参与包含判定，逐条出版 `skipped_short_detail`） |
| ×2 同语料对照臂 | 跑完且吃满预算，墙钟实测 `elapsed_seconds=1070.09` | `output/n3a_control_x2/progress_exit.json`；㊵-548 |
| 仓库卫生 | DEBT-G63 四条修法全部结清（①③④ 有测、②在判读器）；G65／G66 两条**长红**修掉并留下"先红一次再放宽"的同族纪律 | ㊵-544／550／551／556／561；`tests/taiji_native/test_g63_*`、`test_g14_*` |

## 2. 待 owner（只有你能定；我不代选）

1. **N4 的材料出处**：真值记录只能在**挂载链的在跑状态**里现生（四枚在库 checkpoint 实测无
   `episodic_memory` 键，㊵-537 取证）。计时已不构成约束——寻址面 100 查询 `wall_clock_ms=798.608`、
   端到端 ≈2.2 s/次（㊵-560／561）。要裁的是"从哪条挂载/训练跑里现生材料"，那是**改权重范畴**。
2. **N1-03 的前置（产品码）**：给 `Taiji.generate()` 加默认 `None` 的逐字节喂入参数（DEBT-G62）。
3. **N5 四层真跑／N6**：尚未起跑，各自要先预注册、判据先冻。
4. **推送授权**：本轮实测 `origin/main..HEAD`＝**4 笔未推**（㊵-558／559／560／561）；上一次的授权
   只覆盖 `51dadcd70..501cbb983` 那 13 笔，我按"单次授权不外推"没有替这批决定。
5. **一条命名/口径债**（低优先）：`reports/taiji_n4_face_cost_quote_20261009.json` 名字带 `cost_quote`
   而内容是寻址面读数＋计时列——要么承认现名（引文里说清"件内含 `wall_clock_ms`"），要么改名并同步引用。

## 3. 下一格的精确入口（可自办、零批文、零改权重）

1. **给 `wall_clock_ms` 补断言**（最小单元，先做这个）：在
   `tests/taiji_native/test_n4_01_addressing_surface_contract.py` 加一支——跑 `main` 一次，断言件里
   `"wall_clock_ms" in payload`、`isinstance(payload["wall_clock_ms"], float)`、`>= 0`，
   并且**面读数的键集只许多不许少**（沿用 `test_n3_02_sequence_length_column_contract.py` 里
   `required - set(record)` 那个写法）。现状：10 passed 未回退，但**没钉这一列**（㊵-561④ 自报）。
2. **N3 甲的"发表资格"回写**：PLAN-N3-13 §6 的三行前置里，第二合取项现在**有值了**（对照臂到位）——
   把该件的状态列按 `table-status-column-goes-stale` 那条纪律就地打日期戳，并核对 03 队首那句是否还新。
3. **仓库修复档**：G57②③、G55 两级守卫、G54 默认改 3（**需同步旧夹具的判据行数**）、G48②、G61②、
   G64③ 默认化（G67 的 (甲) 点名能力已随 ㊵-559 落地）。

## 4. 门与本机注意（下一个人会踩的）

- 仓根 `E:/Seed` **没有 lefthook 配置** ⇒ `git commit` 不自动跑门。本轮每笔都显式跑：
  `ruff check .`／`black --check .`／B-SIM 棘轮／相关 pytest 册／五枚 md verify 叶子／
  `doc-sync`（43 叶）／`hygiene`（18 叶）。**最后一次实测**：`18 passed`（N4 面测＋两册文档守卫）、
  五枚 md 叶子各 rc=0；`hygiene` 与 `doc-sync` 在 03:14–03:24 期间红在**并行会话**的
  `packages/client/ui-primitives/src/markdown/parse.ts`（TS2769）等文件上，我按 pathspec 只提自己动过的文件，
  没有替别人重生成入库产物。
- `git checkout HEAD -- <件>` 会把盘上文本翻成 **CRLF 而 `git diff --numstat` 看不见**（本轮实测 326 处）。
  动任何被 checkout 复位过的件之前，先按字节 `count(b"\r\n")`。
- `pnpm run <门>` 的隐式 install 会改写被跟踪的 `taiji-harness/pnpm-lock.yaml`（本轮两次发生）。
  没加包就不要把锁文件漂移带进提交（备份到仓外→`git checkout HEAD --` 复位）。
- 台账挑号必须在拼接那一刻全文 `count`（本轮㊵-544～561 连续，无撞号）。

## 5. 完成审计口径（避免误判"做完了"）

N 主线的成功判据是**每项能力有冻结判据＋仪器化读数＋可发表/不可发表的边界声明**，不是"跑了很多档"。
按此：N1／N2／N3 三项**已具备仪器化判读**（本文 §1），N4 卡在材料、N5／N6 **未开工**，
因此**目标未完成**，这份件是停靠点而不是结项书。
