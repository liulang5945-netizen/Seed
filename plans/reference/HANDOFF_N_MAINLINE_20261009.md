# N 主线交接（2026-10-09 03:36Z · 轮次预算将尽时的停靠件）

**为什么要有这份件**：本轮预算见底（37/40），按本仓纪律（`queue-head-is-plan-doc-not-my-tasklist`／
"预算不够时把交接写进 plans/reference 而不是会话总结"），把"已收口／待裁／下一格的精确入口"落到可被
下一个人（或恢复后的我）**机检**的位置。所有数字都指向已入库的台账行与件，不在这份件里新造读数。

## 1. 已收口（有件、有门、有台账行）

> **2026-10-09 补（owner 第九/十次弹窗后）**：N3 甲收口（`J-N3a′=not_holds`，判据全仪器化，㊵-548/549/570）；N2 乙档收口（**同跑配对** `cost_persists`＋同窗收益成立 ⇒ 合取不成立，㊵-566/567；跨跑拼接版降为旁证）；N4 进入有真读数状态（`wrong_top1_rate=0.06` 挂载面基线，㊵-575；`--episodic-mount` 实施＋守卫 ㊵-574）；N5 设计预注册落盘（[PLAN-N5-01](PLAN-N5-01_four_layer_loop_design_prereg_20261009.md)，owner 裁 (a)，㊵-576）——实施格（训练器消费钩子）是下一格，入口见 §3。

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

1. ~~给 `wall_clock_ms` 补断言~~ **已完成**（2026-10-09：断言已落在
   `test_n4_01_addressing_surface_contract.py` 的 `test_wall_clock_column_is_published_and_face_keys_do_not_shrink`，
   复用 `_clean_fixture`，N4 面测 11 passed——本条最初列入时为待办，同日已结清）。
2. **N3 甲的"发表资格"回写**：PLAN-N3-13 §6 的三行前置里，第二合取项现在**有值了**（对照臂到位）——
   把该件的状态列按 `table-status-column-goes-stale` 那条纪律就地打日期戳，并核对 03 队首那句是否还新。
3. **N5 实施格（下一格，可自办）**：按 [PLAN-N5-01](PLAN-N5-01_four_layer_loop_design_prereg_20261009.md) §1 接训练链消费钩子——`last_decision` 过阈 ⇒ `AdaptiveResidualShadow.from_parent_bridge(config, bridge_payload, candidate)` 生成影子 ⇒ 流循环内 `shadow.learn(...)`。**逐处 Edit 不再批量拼接**（573 教训：>5 处锚点必失控）；候选构造器参数与桥 payload 形状需先现读 `adaptive_residual_candidate.py`。
4. **仓库修复档**：G57②③、G55 两级守卫、G54 默认改 3（**需同步旧夹具的判据行数**）、G48②、G61②、
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

## ㊵-613 补：#8「N5 保持侧」探源结果（2026-10-09 晚，只读、零跑）

- **可执行的下一步不是跑，是先升版**：`J-N5b-5` 需要同一对臂上的 before/after 四张面件，但 G/H 两臂 `base_ticks=0`（从头训练、退出才写件）⇒ 没有「before」这个可评对象。旧七列基线住在 `checkpoints/seed_a31self_with_circuit.pt`＋copy circuit 面上，与 N5 臂不同源 ⇒ 按 G-N2c-3 只能拒判。**不许**把 `--no-circuit` 档读成同一张面。
- **已定位的取数面（供升版件引用，命令面已实测可达）**：
  - `python scripts/training/eval_taiji_cap0_baseline.py --checkpoint <臂> --report <面> --dimensions E`
  - `PYTHONUTF8=1 python scripts/training/measure_taiji_a30_repetition_penalty.py --checkpoint <臂> --chain raw_masked --limit 24 --penalties 0.0,0.5,1.0,2.0 --out-report <面>`（注意：这台件的 `--help` 在 GBK 控制台会 `UnicodeEncodeError`）
  - 判读：`python scripts/training/adjudicate_taiji_n2_04_retention_pair.py --cap0-before … --cap0-after … --replay-before … --replay-after … --consolidation-corpus <巩固语料> --retention-manifest plans/manifests/cap0_eval_set_v2.json --out <裁定件>`
- **另一处要如实登记的缺口**：两臂 `n5_shadow` 块实测 8 枚键，㊵-593 说要补的 `shadow_branch_hits`／`shadow_materialized` 读回 `None`（＝没进过块），所以「影子被喂过」目前只有结果侧读数支撑，过程侧计数器未落地。
- **欠两笔债（列数未核对，故未硬插 05）**：① 产品档寻址尺 θ_small 落几何盲区属**扰动设计教训**；② 在场计数器承诺未兑现。

## ㊵-620 补：把本件里**已过期的处方**逐条点名作废（读这份件的人只看这一节就够）

> 立这条节的理由＝本仓自己那条纪律：**交接清单里过期的若是处方／命令，比过期的结论更贵**——
> 结论过期只误导认知，处方过期会让人白花机器时间。下面每条都带本件的定位，原句**不删**（历史保留），只标作废与现行。

1. **§2 第 4 项「4 笔未推」＝已过期读数**。现行以 `git rev-list --count origin/main..HEAD` **现取为准**
   （本件落笔时实测 `15`，提交后再 +1）。且推送已由 owner 裁「你自己在终端跑」⇒ 助手侧不再尝试。
2. **§3 第 3 项「N5 实施格＝接消费钩子」＝已完成，别再当入口**。转发（`Seed.observe` 条件转发）、
   当场消费（`_record_pressure` 内 propose＋materialize，删掉符号流轮询旧块）、
   信封填充顺序（先填 `n5_shadow`／`episodic_memory` 再 `atomic_save`）、通电旋钮 `--n5-shadow-gate`
   四件都已入库，并有测钉住（`test_n5_03`／`test_n5_04`／`test_n5_05`／`test_n5_08`）。
3. **上面「㊵-613 补」里那三条命令：现在照抄必然失败，两处实测原因**——
   ① `--pressure-record` 的真实签名是 `type=Path`（`train_seed_corpus.py:1126-1132`），**要一个 JSONL 路径**，
   当裸旗标写在 argparse 阶段就 `parser.error`（㊵-616 实测两臂各 rc=2，零训练发生）＝**DEBT-G70**；
   ② 保持侧被两道响亮拒绝夹死：`taiji/model.py:1284→:1131` 拒绝「位置输入＋发育 F1 迁移」共存，
   而 `Seed.restore`（`seed/model.py:240`）又拒绝与基件 config 不符的形状，
   `checkpoints/*.pt` 全集 13 枚**每枚**都带 `readout_utf8_position_input=true` ⇒
   在 owner 裁甲／乙／丙之前，那四张 after 面件**取不出来**＝**DEBT-G71**。
   ⇒ 这三条命令**保留作形状参考**（出件方与判读器都是对的），但**标「未验可跑」**，别据此排期。
4. **§5 的「N4 卡在材料、N5／N6 未开工」＝已过期状态**。现行：N4 已有**产品档**寻址读数
   （120 条由产品自己的 `settle_action` 写入，寻址尺经 PLAN-N4-07 升版后判 `angle_sensitive`，㊵-612）；
   N5 已通电并有**过程侧在场读数**（㊵-618：治疗臂 `shadow_forward_hits=11735`／`shadow_learn_hits=11735`，
   对照臂同一块内三枚全 0）；N6＝债册已随本轮 G70…G73 继续清。**但能力读数仍为零新增**——
   这条边界声明不许被上面任何一条替换。
5. **「欠两笔债未插 05」这句也已过期**：05 现按真实列形状（5 列）正式插了四行——
   `G70` 冻结命令是散文不是可执行物、`G71` 保持侧产品层互斥（归 owner 择一）、
   `G72` 恒真键 `shadow_materialized`（同轮已撤换）、`G73` θ_small 扰动设计债（升版已解）。
6. **下一格真实入口（不依赖任何批文、零算力、属 `scripts/`＋测）**：让 N5 判读器把过程侧读数**机械出版**——
   `scripts/training/adjudicate_taiji_n5_shadow_gate.py` 目前只断言 5 枚键（`REQUIRED_KEYS`），
   对新落盘的 `shadow_forward_hits`／`shadow_learn_hits`／`shadow_branch_hits` **视而不见**。
   落点＝在该件的 `judge_arm()` 里按 `'k' in block` 判在场、按 `.get()` 取值并加进输出，
   同步 `tests/taiji_native/test_n5_07_shadow_gate_adjudication_contract.py` 加一条
   「块内带计数器 ⇒ 读数出现在输出；不带 ⇒ 标 `unverified_missing_face`」的双向测（**两支都要走**）。
   做完它，`J-N5b-1` 才同时覆盖「结果侧五枚键」与「过程侧三枚计数」，下一轮正式跑无需人工抄数。
