# 交接：N2／C4 这一腿在材料维地板出版之后（2026-10-18 04:48Z 落笔）

## 0. 三个锚点（**落笔时现取**，下一次开工的第一个动作是重取这三行再复用本页）

- `git log --oneline -1` → `6d7869e9a feat(taiji): name the rehearsal retention cost at item level - it is answer loss, not sentence loss`（**05:42Z 刷新**：本页首次落笔后本腿又做完 §1 第 8、9 两格；本窗口累计 16 笔提交，从 `713cedd50` 起）
- `git rev-list --count origin/main..HEAD` → **120**（05:42Z 现取；这个数每提交一次就变，引用前必须重取。owner 本窗口裁"暂不推"，推送一事不许自行执行）
- 别人在飞的件（**不要提交、不要 checkout、不要抢它的写靶**）：`scripts/training/run_taiji_n1_length_balanced_training.py`、`tests/taiji_native/test_n1_plan110_length_balanced_runner_contract.py`、`scripts/training/run_taiji_n1_native_evidence_paired.py`（现取为 ` M`）；另有 6 枚 `reports/taiji_n5_07_*.json` 是 autocrlf/stat 幻影（`git diff --numstat` 空、盘与 HEAD 的 `b'\r'` 都是 0）。
- 本腿自己的写靶状态：`reports/taiji_n2b_powerup_20261008_c4g1.json`／`…_c4g2.json`／`…_c4gb1.json` 三枚都已入库；**同 tag 复跑会被驱动的 `_refuse_completed_target` 拒**（DEBT-G169），要再跑必须换 tag。

## 1. 本窗口做成了什么（每条都有件或提交为证）

1. **材料态受控恢复端到端兑现**（DEBT-G170 第②③步结清）：`--material-snapshot` 目录里的手写基线在两趟跑完（含抛错支）后都把 `data/consolidation/state.json` 还原成快照内容；并且新加了"**快照只收一次**"守卫（提交 `713cedd50`）——第二趟用同一目录＝回到第一趟的起点，这是配对能叫配对的前提。
2. **一对同材料重复买到的是确定性，不是地板**（`reports/taiji_n02_repeat_determinism_c4g_20261018.json`，提交 `1df3bd6b3`／`328708b48`）：c4g1 与 c4g2 的 `mother_digest`／`treated_digest`／候选再装载摘要／四张窗口与整件读数／`surprise_modulation`／三枚增益列**逐位相同**，而 `pass_id`／`started_utc`／`seconds` 三列都不同（1586.9 对 1601.8）。⇒ 通电是**材料集的确定函数**；按旧定义填 floor=0 会让任何正 Δ 判过。地板仪那条"逐位相同即拒"的守卫就是抓住这一点的设计（合同测 18 passed）。
3. **地板换成材料维并第一次取到可辩护的数**（`reports/taiji_n02_noise_floor_material_c4g_20261018.json`，提交 `35f65f276`／`6b168dcde`）：A 睡全库最新 200 条、B 睡最旧那批（同 `code_identity 8165ccbb…`、同 `mother_digest 3d812715…`、同 pool 200／k 8／dose 64／来源配比 32:168:0，材料指纹与末段权重都不同）。**Δq(A)=+0.8949958795371575，Δq(B)=−0.3483618319703936 ⇒ floor=1.2433577115075511**（accuracy 那列 floor=0.029296875）。
4. **C4 同面判级第一次四条全出数**（`reports/taiji_n02_c4_net_gain_with_material_floor_c4g1_20261018.json`）：`status=cost_persists`；`J_N02_1_gain.verdict=not_resolved`（`0 < Δ ≤ 地板`、`passed=null`）；`J_N02_2=cost_persists`；`J_N02_3=true`；`J_N02_4=true`；`ruler_precondition.state=present`。
5. **保持侧代价落到了现行候选档**（提交 `512bfaca3`／`2c8b5197d`）：四张同面读数 280.1／262.1／346.9／336.0 秒全 rc=0，配成 `reports/taiji_n2_04_retention_pair_c4g1_20261018.json`＝**3 列跌**（`replay_strict_hits:1.0` 6→5、`…:2.0` 6→5、`replay_well_formed:2.0` 25→23），而 `replay_well_formed:0.0` **24→26（涨）**。⇒ 旧档 n2c1 那"四列全跌全是 well_formed"的形状**不是现行机制**，别再引用它当现行。
6. **两枚新仪器进门**：`emit_taiji_n02_noise_floor_from_pair.py`（地板仪，五列准入＋两档维度）与 `emit_taiji_n02_repeat_determinism.py`（确定性件），合同测现取 **18＋6＝24 passed**；`black`／`ruff`／六项台账与文书门 rc=0（`audit_taiji_ledger_cited_reports_are_tracked` 仍有 1 枚外来红，见 §5）。
7. **判据升版已签完并落文**：`PLAN-N-02 §9`（9.1–9.7，提交 `43db4c5db`／`388833136`），owner 两次弹窗：一签"改成材料维配对"、二签"反选账本"＋"两批都要超过地板才算真"。§9.3 里留了**甲案被预飞否证**的原地更正（未投影条数实测 0 ⇒ 甲案的 B 会 12 秒 fail-closed；预飞约 2 秒，省掉一趟 26.5 分钟）。
8. **保持侧代价的机制普查（零算力，提交 `622aedbce`）**：新只读仪 `census_taiji_n02_retention_output_shape.py` 对母档／候选档的 cap0 两张件做逐条对量（先核 `eval_set_sha256`＋`chain`＋`declared_mode`＋`trained_during_eval`，再核**件自述的检查点摘要＝盘上 `.pt` 字节摘要**：`7470e2ea…`／`da695447…`）。结论两条否证：**①cap0 这一面没掉分、净涨 1 项（D 6→7）**，所以"巩固让模型普遍答不好"这句不成立；**②三条分数翻转项（D03／D04 涨、D09 跌）的输出一律 255／256 字节、前后逐位相同** ⇒ "掉分是因为变短"在 cap0 上被当场否证——D09 掉的是期望子串 `42`，候选档吐的是 `2` 的连续重复段（"换了字"而不是"变短"）。同时形态确实在动：99/100 条文本被改写、循环样输出 61→40、字节总量 −18%、`replacement_chars` 0→0（不是编码坏）。合同测 9 passed／1.55s，三支判读值各有真走过的一支，测还抓住过两条仪器自身缺陷（默认拿读数件 sha 比检查点 sha ⇒ 真件第一跑就被自己拒；少一维时消息点名"找到了哪几维"而不是"缺了哪一维"）。

9. **保持侧代价已经点到题（05:41Z，提交 `89e2d427e`／`7bd3f0fe8`／`6d7869e9a`）**：复述仪加了 opt-in `--emit-per-item`（默认出版形状与旧推导逐字节相同，且**用真面复现证了一次**——同一对 checkpoint 重取两趟，聚合列与 04:12Z 那两张件逐档全等），新判读器 `adjudicate_taiji_n02_replay_item_moves.py` 出版 `reports/taiji_n02_replay_item_moves_c4g1_20261018.json`：`verdict=hit_only`——四档罚分共 5 次命中翻转，**每一次的 `texts`／`formed_texts` 都原地不动**，其中 **V014 独占 0.5／1.0／2.0 三档**、且它在 pen0.0 母档本来就没命中；pen0.5 是"丢 2 得 2、净 0"（聚合列把它完全藏住）。⇒ 代价是**答不出期望子串**（不是"不成句"），且高度集中在一题 + 换血，不是普遍退化。合同测现取 6＋11 passed（含"四个判读值都可达且等于词表"那一支）。

## 2. 结论边界（可说／不可说）

- **可说**：①现行配方（`sleep_pass.run(organs=True, learn=True)` 现行默认四参数、64 字节窗、k=8）的窗口增益**落在它自己的材料维噪声带内**（0.895 ≤ 1.243），且**符号随材料翻**（最旧那批是 −0.348）；②同材料重复是确定的，可复现性门为真；③保持侧在现行档确实掉分（3／7 列），来源与保持集字节级交集为 0（116 窗全测），两侧同题集同偏移；④旧那对 2026-10-08 件的极差 3.5916 是**材料差**，DEBT-G168 当年列的第三种可能（版本差）不再是首选解释；⑤代价**不是普遍输出退化**——cap0 那一面净涨 1 项，而三条分数翻转项的字节长度相同（㊵-1125），所以它是复述面题集分布特有的，且"变短"这一支在 cap0 上已被否证；⑥复述面的跌已**点到题**并判成 `hit_only`（㊵-1126）：四档共 5 次命中翻转，每一次的 `texts`／`formed_texts` 原地不动，V014 独占三档、且它在 pen0.0 母档本来就不命中 ⇒ 是"期望子串答不出"，且高度集中而非普遍。
- **不可说**：**"C4 净增益成立"**（合取里 J-1 不可判、J-2 假）；**"通电没有增益"**（A 是正的，符号不稳不是零效应）；**"地板＝0"**（那是同材料档的确定性，不是噪声）；**"两批 Δ 相减＝效应量"**（是极差）；**"现行代价形状＝well_formed 四罚档全跌"**（那是旧档）；**"巩固让模型普遍答得更好/更差"**（cap0 净＋1 只覆盖 cap0 那一面，复述面才是负的，两面不同源）；**"掉分是因为输出变短"**（cap0 上字节相同；复述面上逐题 `texts` 也原地不动）；**"代价是成句能力退化"**（`hit_only`：掉命中的题成句条数没动；pen2.0 那两条成句跌住在别的题上，与命中不同题）；**"循环文本减少所以输出更干净"**（聚合层为真、题级可反向：V014 的 `cyclic_tail_texts` 在 0.5／1.0 由 1→2）；**"两列 strict_hits 各跌 1 ⇒ 有两题坏了"**（pen0.5 是"丢 2 得 2、净 0"，聚合列把 4 次翻转全藏住）；**"C5 可开工"**（C4 未成立，C5 硬前置没满足）。
- **一次性仪器口径**：c4g1／c4g2／c4gb1 三个 tag 都已消费，同 tag 复跑会被拒；再跑必须换新 tag，且 `--material-snapshot` 要指向新目录（否则 capture-once 会回到**旧基线**——那正是配对要的，但别误当成"新的一天"）。

## 3. 下一格（owner 已裁方向：**先治保持侧代价**）

**队首（本腿）**：把"巩固为什么让保持侧掉分"当**能力问题**定位。**两步已做完**：cap0 那面净涨 1 项且翻转项字节相同（㊵-1125）；复述面已点到题并判成 `hit_only`（㊵-1126）⇒ 问题现在收窄成一个具体对象：**为什么 V014 的期望子串在巩固后不出现了，而同一题在 pen0.0 母档本来也不出现。**

1. **下一步＝看 V014 的原文对照（秒级，不需新预算）**：`per_item` 现在只有计数、没有原文，所以要么(a)给 `--emit-per-item` 再出一档"答案首若干字节＋长度"（同一函数里加键，默认仍逐字节不变，配一条新合同测），要么(b)先只取那一题把成本压到秒级——题位已机检：`plans/manifests/r2_copy_surface_extension_v3_position_random.json` 的 `dimensions.X.items` 里 `V014` 的下标是 **13**（现取 `ids.index('V014')==13`，全集 104 题），故
   `python scripts/training/measure_taiji_a30_repetition_penalty.py --checkpoint checkpoints/seed_n2b_mother_20261008_c4g1.pt --limit 1 --offset 13 --emit-per-item --out-report reports/taiji_c4g1_replay_pi_V014_before_20261018.json`
   （after 侧换 candidate）。看清"母档答里有的子串在候选档答里被什么取代"之后，才谈得上修法。
2. **修法候选（要 owner 批文的那一格）**：如果 V014 的失败形态是"巩固把该子串挤成循环尾巴"（现有旁证：V014 在 0.5／1.0 的 `cyclic_tail_texts` 1→2 **涨**，而聚合 `cyclic_texts` 四档全降 ⇒ 聚合方向会掩盖题级反向变化，引用"循环减少"必须限定在聚合层），最小干预是**减剂量**（`organs.max_symbols`）或改夜选权重——两者都属射程/剂量变更 ⇒ 要 owner 批、且双臂同窗，不许用改判据把它抹平。
3. **可选（不挡路）**：给 B 档候选也补同面保持侧四张读数——命令与本轮逐字相同，只把 checkpoint 换成 `…_c4gb1.pt`、落点名换成 `…_c4gb1_…`；单价现取 cap0 ≈262–280 秒／趟、replay ≈337–347 秒／趟（四张 ≈20 分钟）。今天的 `cost_persists` 只覆盖 A 这一档。

**并行（不归本腿，别抢）**：03 的"当前唯一下一步"仍是并发会话的 PLAN-N1-110 readiness 那一格（它的 write-once 件 `reports/taiji_n1_length_balanced_training_20261018_runner_readiness.json` 至今不在盘上，也正是那枚外来红的来源）。

## 4. 待 owner 裁的事（本腿只列真的需要他裁的）

- **C4 之后往哪走**：J-1 已被材料维地板判成 `not_resolved`。要么承认"这一档增益不可复现"并换剂量/换机制（＝改射程），要么把 J-1 的口径从"单批 Δ>地板"改成 §9.5⑴ 的两批合取（那要先补 DEBT-G173 的仪器）。**我没有自自动判据。**
- **DEBT-G160**（CPU 预算用绝对 `time.process_time()`）：修法要重封三条链的件，代价是重出版摘要。
- **DEBT-G162／G164**：本窗口 owner 已裁"都先维持红，等逐条裁"，所以两条仍挂着，不许我代松门槛。
- **推送**：113 笔未推，现取数每次都会变。

## 5. 债与仪器缺口（新增／更新）

- **DEBT-G171 结清**（地板换维并取到数）；**DEBT-G174 已结清**（`--emit-per-item` 落地＋两趟重取＋`hit_only` 判读件，见 §1 第 9 条；剩下的"要原文"是它的延伸而非新债，入口在 §3 第 1 条）；**DEBT-G172** 待开工（判读器不校验四条是否同一候选档）；**DEBT-G173** 待开工且已扩一条同族缺陷：`gain_judgement` 把 `floor_source` 写死成"同代码同面重复读数极差"，而 20261018 的地板是材料维 ⇒ 那句话与件内 `floor_dimension` 不同源；改法是读地板件的维度列，**不重跑任何一趟**（数值不受影响）。
- 普查仪自身的两条缺陷由它自己的合同测抓出并已改（05:10Z，提交 `622aedbce`）：①第一版拿**读数件文件**的 sha 去比**检查点**的 sha，结构上不可能相等 ⇒ 真件第一跑就被自己拒（DEBT-G155 那一族的第四次）；②少一维时拒绝消息点名的是"找到了哪几维"而不是"缺了哪一维"，读的人要反推 ⇒ 已改成 `missing: G` 并把这句话钉进测。
- 外来红一枚（不是我造成的）：`audit_taiji_ledger_cited_reports_are_tracked` 现取 `violated`，唯一 offenders＝`reports/taiji_n1_length_balanced_training_20261018_runner_readiness.json`（并发腿的 write-once 件，03 与 08 里共 4 处引用，早于本窗口存在——已按 `git show HEAD:` 对账）。它 `malformed_citations` 两条也是历史省略式写法，本轮未增。
- `tests/seed/test_platform_boundary.py::test_source_face_is_the_git_face_not_the_whole_disk` 仍红（`.seed_test_tmp/` 残留目录），是登记过的常态红，本轮未处置。

## 6. 复现这一页读数的注意事项（都踩过）

- **不许对 `code_identity` 钉住的 `.py` 用 `git checkout`／`git restore`**：autocrlf 会把盘上字节翻成 CRLF，摘要当场变；要复位用 `git -c core.autocrlf=false show HEAD:<path>` 按字节写回。
- 两枚材料基线的落点在**仓外**：`C:/Users/23747/AppData/Local/Temp/n1_v6_sign/material_reset_c4g/`（A：投影账本清空）与 `…/material_reset_c4gB/`（B：反选账本，同目录里 `baseline_manifest.json` 记着 168 枚 chosen digest、扫描条数 269、库容 1808、与 A 交集 0）。**这两枚目录不在版本控制里**，机器一清就没了——下次要复用先把它们复制到仓外备份或重跑构造脚本 `build_c4gB_baseline.py`（也在 TEMP，非在库件）。
- `data/consolidation/state.json` 我在落笔前**还原成窗口开始前的原始账本**（`passes=10`，1288 条列表／1276 枚去重），与 `material_reset_c4f/sleep_state.json` 逐字节相同 ⇒ 产品运行时状态没被我留下人工账本。三趟跑写出的语料仍在 `data/consolidated/`（4 枚 `corpus-20261018*`，`.gitignore` 覆盖）。
- 长跑用 `Start-Process` 起（`nohup &` 会被工具回收），完成判据读日志里那行 `report -> …`，不读进程存在性；墙钟报价只认件自述的 `seconds`（本窗口我把"一对 ≈3 分钟"报错了 8 倍，真值 1586.9／1614.2 秒／趟，已写进 08 ㊵-1122 并更正 ㊵-1119 那句"≈80 秒"）。

## 7. 停靠动作（本页写完即停）

- 工作区没有本腿的未提交件：本窗口所有代码／件／台账／计划都在上述 11 笔提交里；`git status` 里剩下的脏项全是 §0 列出的别人在飞件与幻影件。
- 门：`black --check`／`ruff check`／`audit_taiji_debt_table_column_shape`（173 行）／`audit_taiji_ledger_marker_uniqueness`（748 行、duplicates 空）／`audit_taiji_tracked_report_line_endings`（holds）／`census_taiji_debt_row_effective_status`／`check_plans_markdown_links`／`audit_taiji_prereg_exit_wording`（criterion_lines 36 ≥ 地板 5、unpinned 0）均 rc=0；唯一 rc=1 是 §5 那枚外来红。
- 未推：113 笔，owner 裁"暂不推"，我不执行 `git push`。
