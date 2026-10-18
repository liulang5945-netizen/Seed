# PLAN-N-02 · C4「净增益」包的预注册：交付四件各落到哪台真仪器、哪个分母，以及今天缺的究竟是哪一件（判据先冻、零算力、不起跑，2026-10-17）

## 0. 本件是什么、不做什么

* C4 在工作包表里同时挂在 N2/N3/N5 名下（[PLAN-N-01](PLAN-N-01_development_revision_20261014.md) §3 表 C4 行），它是**阶段包**而不是某一 N 项的实施附件 ⇒ 编号沿用 PLAN-N-01 的跨项惯例记作 `PLAN-N-02`，不占 N2/N5 序列号，也不新增里程碑编号。
* 本件只冻三样东西：**交付四件到仪器/件/分母的映射**、**判据与守卫**、**前置的真值表**。它不起任何跑、不改产品码、不改权重、不重开已判格（N2-02/03/04/05、批次 C 的 N5-20…24、N1-88/89/90/91/92 的结论一律继承）。
* 立件时机说明（不是插队）：C4 的现行前置为假（§5 逐条给出处），所以本件与 PLAN-N1-93/94 同形——**在有数据之前把判据与出口写死**，等 owner 在 N1 那条线上裁定后，C4 不需要任何临场判断。

## 1. 现状核对：C4 的四件交付物，今天哪件有真面、哪件没有

PLAN-N-01 §3 给 C4 的合格交付原文是**四件**：`新能力/旧能力矩阵`、`来源与保持`、`资源与质量曲线`、`同运行代码的可比重复或独立验证`。逐件现取：

| 交付件 | 今天有没有真面 | 出处（仪器／件／面） | 还缺什么 |
| --- | --- | --- | --- |
| ① 新能力/旧能力矩阵 | **两侧都有真面**，缺的是"同一次实验里两侧同时读" | 旧能力：CAP-0 冻结集 `plans/manifests/cap0_eval_set_v2.json`（100 项、逐维）＋复述 ×24（`scripts/training/measure_taiji_a30_repetition_penalty.py`）；新能力：批次 C 两臂配对器 `scripts/training/audit_taiji_n5_22_cap0_pairing.py`（J-N5u-1/2/3 载体）与通电读数 `reports/taiji_n2b_powerup_20261008_n2c1.json`（现读字段名：`j_n2b_prime_delta_quality = 2.4829673663245373`、`j_n2b_prime_delta_accuracy = 0.0`；㊵-567） | 缺**一次同时读两侧的运行**，其前置是 §5 的 P1/P2，不缺货 |
| ② 来源与保持 | **已出值，且是负结果** | `scripts/training/adjudicate_taiji_n2_04_retention_pair.py:348-356`：保持侧出 `J_N2c_retention ∈ {retention_holds, cost_persists}`，收益侧在 `:351` 写死 `J_N2c_gain = "not_judged_here"`（**不代答**）；`:300-304` 是 `separation_unverified`＋`verdict = not_judged` 出口。两枚件的现读：`reports/taiji_n2_04_retention_pair_20261009.json`＝跨跑拼接、`columns_compared 7`、**7 列全跌**（`cap0:E`、`replay_strict_hits:1.0/2.0`、`replay_well_formed:0.0/0.5/1.0/2.0`）；`..._n2c1_20261009.json`＝同一次通电前后两态、**4 列跌且全是 `replay_well_formed` 四档**（`cap0:E` 与 strict_hits 两列 delta=0）；两件 `verdict` 均 `cost_persists` | 已结清到 B 档；再往前要**甲层**（保持切片进 `spec.datasets`）＝产品码批文，PLAN-N2-04 §6 已交回 owner |
| ③ 资源与质量曲线 | **没有现行真面——四件里唯一缺设计的一件** | 仓内唯一名字带 resource 的仪器是 `scripts/training/eval_capability_resource.py`，其 docstring 自称"deterministic R5B-L3 resource and rollback canary"，`cpu_ms` 是**入参**（`_candidate(..., cpu_ms=...)`）而不是测量 ⇒ 它评的是注册表/回滚语义，**不能回答"质量随资源怎么变"**。历史面：`probe_taiji_p5_1c_scale_sweep.py`（语料尺寸×族判别，docstring 自述 validation-only）与 `eval_taiji_p4_semantic_scale.py` 属**已收束的 M 主线**，按 03 横幅"历史读数与下一步不得引用为现行" | 缺的是**口径设计**：资源轴取什么（更新数／CPU 秒／语料字节／记忆占用）、质量指标取哪一列、采样点几档、每档几次重复、分母怎么算。**这不花批文，是本件 A 档可自办的一格** |
| ④ 同运行代码的可比重复或独立验证 | **有形状与纪律** | 噪声地板配对件 `reports/taiji_n5_10_noise_floor_retention_pair_20261011.json`；读数纪律两条已在库：同一份权重跑两遍逐位相同 ⇒ **1 项粒度不是噪声**，以及"两轮 after 不同源不可互代"（㊵-495/496/567） | 缺的是把这条纪律**接进 C4 判据**（本件 §3 的 J-N-02-1 因此写成"Δ 必须与地板比"） |

一句结论：**C4 不是"缺仪器"，是缺一次同时满足 ①②④ 的运行，而 ③ 还缺口径设计。** 后者今天就能做且不花钱，前者被 §5 的前置挡住。〔㊵-1039 ③ 的口径设计已落在 [PLAN-N-03](PLAN-N-03_c4_resource_quality_curve_prereg_20261017.md)，并纠正了本件一处低估：我原写"③ 只缺设计不缺跑"，PLAN-N-03 §1 用三条现读否证了"拿已有检查点就能画曲线"（三批 `totals.completed_updates` 全 = 73728 ⇒ 无资源梯；九枚端点给的是 seed 轴＝噪声地板；跨族比较同时换了学习率配方与语料 ⇒ 混杂），所以**设计零算力、画点要一次有界训练**（价按现读 0.041640 秒/更新算到字段级 ≈7445 秒 ≈2.07 CPU 小时，拆两批）。〕

## 2. 本件先冻三条方法学前置（每条都是这条线付过代价买来的，写成硬条件而不是提醒）

* **P-尺（先测尺，再测能力）**：任何用来声称净增益的对比面，出版前必须自带**逐端点动态范围列**，且范围检验只跑在**不用于声称**的那一半子集上；不达线 ⇒ 出版 `range_not_met` 并禁止进入合取判级。出处：PLAN-N1-93 §3、[EXP-N1-95](EXP-N1-95_v6_preflight_range_sweep_20261017.md) §2/§10（同一族面在五个难度取值 × 三次抽样上把可用端点钉在 5/9，说明"饱和的尺"会伪装成"能力已到上限"）。
* **P-门（判级只从判门函数出）**：判据的算术必须由仪器自身的判门函数出版；**禁止**在判读件、台账或对话里手对齐数字。出处：DEBT-G153（本轮我把手算的"每 seed 与 12 比"当成判级，发布了一句随后被自己否证的"互斥"）与记忆 `verdict-arithmetic-must-be-instrumented`。
* **P-分母（分子集合／分母集合／聚合层级三件一起写）**：任何线值都必须写明"分子取自哪个集合、分母是多少、是逐 seed 还是合计"。本件自身示范：`advantage` 类一律写 `12/60 合计` 或 `≥x/20 逐 seed`，不写裸数字。出处：DEBT-G152（PLAN-N1-93 §3 那句"6/20"的分母其实应是 15）、`aggregate-column-two-questions-before-use`。

## 3. 判据（先冻；四条合取，任一非真即不得声称净增益）

* **J-N-02-1 新能力收益**：同运行、同种子、同材料剂量下，治疗臂相对对照臂的新能力读数差 `Δ = treated − control > 0`，**且 `Δ` 必须超过同配置重复实测的噪声地板**；`0 < Δ ≤ 地板` ⇒ 记 `not_resolved`（不判增益也不判无增益），`Δ ≤ 0` ⇒ `no_gain`。地板必须来自**本次同代码同面**的重复读数，不许引用别的批次的极差。
* **J-N-02-2 旧能力保持**：保持集**逐列** `after − before ≥ 0` 全部成立（口径照 PLAN-N2-04 §3 的 `retention_holds`；任一列跌 ≥1 ⇒ `cost_persists`）。列集与件名固定为 §1② 那一组，**不许**事后替换成"另外的保持口径"。
* **J-N-02-3 可比性（同运行代码）**：两侧必须同题集同偏移（CAP-0 `eval_set_sha256` 相同、复述三面 `items/item_offset/first_item` 相同）；来源摘要必须双口径并报并拆出 `convention_only_count`／`real_change_count`，**只有 `real_change_count == 0` 才算工程完整**（DEBT-G147 那格假阴性的复发路径已被拆掉，本条把它写成判据而不是纪律）。
* **J-N-02-4 分离机检**：巩固材料与保持集的**字节级交集 `overlap == 0`** 必须真取到数；取不到 ⇒ `unverified` 且整件不判（PLAN-N2-04 §2 同条，禁止用"两侧题集不一样吧"代答）。
* **合取与分流**：`J-1 ∧ J-2 ∧ J-3 ∧ J-4` 才谈"净增益机制成立"，这也是 C5 的硬前置（PLAN-N-01 §3 的 C5 行原文："C4 至少一个净增益机制成立"）。三条预写分流照 PLAN-N-01 C4 行的"停止与返回"列，不自选：代价持续 ⇒ 回到更新范围/保持（＝N2 甲层，要产品码批文）；容量疑问 ⇒ 可比容量×数据设计；贡献缺失 ⇒ 停止准入与调度扩展。
* **交付件 ③ 的处置（现在就定死，避免它变成事后加的尺）**：资源与质量曲线在本包里**只作描述性出版，不设阈、不进合取**——因为今天没有现行面与口径（§1③）。它的口径设计件必须先于任何一次曲线跑（A 档），且顺序按 09 §2 第 24 行原文"**先证明新能力收益和旧能力保持，再评规模、经验流与巩固速率**"执行。

## 4. 守卫（六条，任一不符即整件不判；每条都要有能为假的负对照）

* **G-N-02-1 面齐**：①②④ 三面缺任一 ⇒ `not_judged`（今天这一条**就是红的**，见 §5）。
* **G-N-02-2 同源自证**：`eval_set_sha256` 与复述偏移任一不同 ⇒ `ran_not_measured`，不许作差。
* **G-N-02-3 地板在场**：J-1 使用的噪声地板必须随件出版（重复读数次数与极差值），缺失 ⇒ 拒判。
* **G-N-02-4 尺先于能力**：动态范围列缺失或 `range_check_passed != true` ⇒ 只许出版 `range_unverified`，四门全绿也不算 C4。
* **G-N-02-5 判级来源**：件里的判级字段必须由判门函数写；判读件/台账若出现与该函数输出不一致的手算结论 ⇒ 该格记未清（P-门）。
* **G-N-02-6 写靶一次性**：默认写靶不得指向已入库件，件已存在即响亮拒绝；重跑必须幂等于"只写自己那份新件"（本仓 `instrument-default-output-must-not-target-tracked` 那条实测债的形状）。

## 5. 前置真值表（现取，逐条给出处；这一节决定本件今天不起跑）

| 前置 | 内容 | 当前真值 | 出处 | 谁能让它真 |
| --- | --- | --- | --- | --- |
| P1 | C2 生成消费前置 | **已判（甲档口径＝行为贡献）**〔2026-10-18 ㊵-1108 就地改判：owner 弹窗签 [PLAN-N-04 §8](PLAN-N-04_c2_verifiable_criterion_prereg_20261018.md) 甲档，C2 的合格列改为位移列合取，读数 `option_a_conjunction_holds=true`（oracle-correct 6 > zero-replay 0 且对照＝0；件 `reports/taiji_n1_c2_ruler_range_20261018_r2.json`，零重测、`model_queries=0`、`parameter_updates=0`）。严格命中 0/16 那一句**没有被修好**，它只是从合格列转为披露列，且 `can_be_cited_as_capability_falsification=false` 的收回句继续生效。本行从此只说明"C2 判得动了"，不涨任何能力读数〕〔㊵-1099 就地更正：本行的"假"降级为**不可判**——同一枚封存件的 `summary` 里 `oracle-correct`（上界诊断臂）严格命中也是 **0/16**，而换剂量干预的 `taiji_n1_c2_padding_placement_final_20261015.json` 出版 `raw_changes_from_tail` oracle **6**／wrong 3／zero **0**（各 16）⇒ 机制侧位移量得到、严格命中这把尺在该面**没有动态范围**。按本件 §2 的 P-尺纪律，无范围的尺不能出能力否证；改判据或改阶梯口径归 owner，我不自改。详见 [完成度对照 §4](N_MAINLINE_COMPLETION_AUDIT_20261017.md)〕 | 消费者校准 112/112 臂跑完，七臂严格命中各 **0/16**（09 §3.2，㊵-876）"机制接入不等于能力通过" | 需要另立的消费侧包，不是本件；**先补一把有范围的 C2 尺**（或 owner 裁"严格命中"这条判据本身）〔2026-10-18 ㊵-1108：这一句已被走掉——尺改成按列出版（复核仪 v2），判据由 owner 签甲档取位移列；剩下的不再是"补尺"而是 C4 自己那四条〕 |
| P2 | C3 学习信用／未见确认可判 | **未定**〔㊵-1099 就地更正：右侧那句"等 EXP-N1-95 §11 四裁"已过期——四裁与 PLAN-N1-93 §1 签署已于 2026-10-17 落地（08 ㊵-1042），v6 面已消费并落**出口 ⑵**（`retry_allowed=false`）；其后 v9／v10 两张面也已消费出值，现行腿＝PLAN-N1-110 length-balanced 重训。C3 的"未定"因此改由两件事决定：v11 四门读数，以及 P1 那把尺是否先修出范围〕 | N1 现队首＝owner 在 [EXP-N1-95](EXP-N1-95_v6_preflight_range_sweep_20261017.md) §11 的甲／乙／丙／丁里裁 | owner 裁定 |
| P3 | 新能力面选定（按主失效类型取一个有界包） | **选定①＝N2 权重写入范围**〔2026-10-18 owner 弹窗裁；发射单已写在 §6ter，但 P5 改权重批文未给 ⇒ **"选定"不等于"已跑"**〕 | PLAN-N-01 C4 行"按主失效类型选择一个有界包"；主失效今天有两支候选（N2 权重写入范围 vs N1 选择侧＋seed13 零检索） | 裁定后由我落 §6 的 A 档设计 |
| P4 | 资源与质量曲线的口径 | **已备（口径设计件，不是"跑过一次曲线"）**〔2026-10-18 就地改判：本件 §3 末条把交付件③定死为"只作描述性出版、不设阈、不进合取，且口径设计件必须先于任何一次曲线跑"——那件已入库＝[PLAN-N-03](PLAN-N-03_c4_resource_quality_curve_prereg_20261017.md)，故 P4 从"缺件"改"已备"；**这不等于曲线有数**，真曲线仍等 P3／P5 的包落地后一次现行跑〕 | §1③（合成 canary 不算面；M 语料扫描不得引用为现行）＋PLAN-N-03 | 口径已定（我可自办的那半已办完）；跑一次要等 P3 选包与 P5 批文 |
| P5 | 改权重批文 | **无** | PLAN-N2-04 §4 G-N2c-6 两条批文不互相代答 | owner |
| P6 | 剂量与预算写死 | **已写（§6ter 第 3 条；手法＝由件自述回读，不手填数字）**〔2026-10-18：不引入新旋钮，治疗量取自 `sleep_pass.run(..., organs=True, learn=True)` 现行默认；算力那栏只有台账既有估计"≈45–60 分钟"、本轮未复测，发射后一律以件自述为准〕 | 沿用 `run()` 现行默认四参数，本件**不放大任何一档**（PLAN-N2-04 §2） | 落 §6 A 档时一并写死 |

## 6. 预算分档与"谁能动"

| 档 | 内容 | 算力 | 批文 |
| --- | --- | --- | --- |
| **A（现在可做）** | 资源与质量曲线的**口径设计件**（轴／指标／分母／采样档／重复次数）＋ 本件判据落进一台**只读判读器**（消费 §7 的出口枚举）＋ 其契约测（六条守卫各配能为假的负对照） | 零训练、秒级 | 自办〔**两半都已落地**：判读器见 §6bis（㊵-1038），曲线口径见 [PLAN-N-03](PLAN-N-03_c4_resource_quality_curve_prereg_20261017.md)（㊵-1039）〕 |
| B | 一次同时读 ①②④ 的有界包（治疗/对照双臂，剂量同 run-1/2） | 与 N2 B 档同量级（≈45–60 分钟 CPU） | **会改权重 ⇒ 回 owner** |
| C | 甲层保持切片进 `spec.datasets`（真·候选版本隔离，才可能改"代价"本身） | 与 B 同量级 | 产品码批文＋改权重批文，两条都要 |

### 6bis. A 档的判读器已落地并跑过一次（㊵-1038 现取）

仪器 `scripts/training/adjudicate_taiji_n02_c4_net_gain.py`（只读三枚已封存件，`models_loaded=0`、`parameter_updates=0`、`writes_any_face=false`、`is_capability_claim=false`），合同测 `tests/taiji_native/test_n02_c4_net_gain_reader_contract.py` **7 passed**（四条判据各有真假两支、"恰好等于地板"那一支单独钉住、分离守卫跑过一次**已知-good** 并配两条负对照、面被换字节必须响亮拒绝、写靶一次性）。

拿今天的真件跑一次，出版 `reports/taiji_n02_c4_net_gain_readiness_20261017.json`，读数就是 C4 的**输入契约缺口清单**（不是任何能力主张）：

* `status = range_unverified`，`ruler_precondition.state = absent` ⇒ **增益面缺 `usable_endpoints` 那一列**（P-尺）。这一条挡住合取进入判级，正是设计意图。
* `J_N02_1_gain.verdict = not_judged`：`delta_quality = 2.4829673663245373` 在场，但 `noise_floor = null` ⇒ 缺**与增益同量纲**的地板字段 `noise_floor_delta_quality_abs`；现成的 N5-10 噪声地板件给的是**计数级**极差，量纲不同，本件按守卫拒绝代答（G-N-02-3）。
* `J_N02_2_retention.verdict = cost_persists`：4 列跌（全是 `replay_well_formed:0.0/0.5/1.0/2.0`），照实出版。
* `J_N02_3_comparability.passed = true`、`J_N02_4_separation`＝`measured`／`overlap 0`／`windows 116` ⇒ **可比性与分离这两条今天就是绿的**，不需要新实验。

一句边界：本件的判级字段是仪器的输出，判读件与台账引用 C4 状态时**必须引用这份件里的值**，不得手对齐（P-门）。另记一次自己的缺陷：第一版把分离守卫写成 `status == "ok"`，而封存件的成功词汇是 `"measured"`——那条守卫会惩罚它自己要测的成功行为，是"新写的守卫必须先跑一次已知-good"的又一次实测；发现时该产物尚未入库，已删除并以修正后的仪器重出（无读者被误导面）。

### 6ter. B 档选包＝①（owner 2026-10-18 弹窗裁）的发射单：**本轮只写不发**，缺一句改权重批文

现取四件（全部本轮实测，不是沿用台账旧数）：

1. **母档已换 ⇒ 双臂必须同运行。** 今天 `DEFAULT_CHECKPOINT` 解析到 `checkpoints/seed_a31self_with_circuit.pt`（`.gitignore:62` 的 `*.pt` 忽略它 ⇒ 不进版本库），用**该仪器自己的 `_digest()`** 现取 `3d812715bd766b2cd0db19f81e728cd20cfabd4156ce6f2bd9e5e8cf5905b02f`、`tick=273`；而封存件 `reports/taiji_n2b_powerup_20261008_n2c1.json` 自述 `mother_digest=3e00bafc…`、`mother_tick=273`。**同 tick 不同权重** ⇒ 2026-10-08 那一对 `j_n2b_prime_delta_*` 与今天再跑出来的数**不可跨母档相减**（这是 §2 的"跨链搬 τ"同一族）。结论：J-N-02-1 的 Δ 只能在**同一次运行**里由治疗臂与对照臂同母档读出，历史件只作"形状与出口的先例"引用。
2. **落点必须带 tag。** `run_taiji_n2b_powerup.py` 的默认落名是 2026-10-08 那批（现取在册 4 枚：`taiji_n2b_{powerup,guard_default}_20261008{,_n2c1}.json`），它有 `--run-tag` 正是为 G-N2c-2"治疗档落新名"而设。⇒ 发射命令固定为两条，tag＝`c4r1`：`python scripts/training/run_taiji_n2b_powerup.py --phase guard2 --run-tag c4r1`（对照臂，件自述 `digest_before==digest_after`＋`guard_pass`）与 `python scripts/training/run_taiji_n2b_powerup.py --phase powerup2 --run-tag c4r1`（治疗臂，出版 `j_n2b_prime_delta_quality`／`j_n2b_prime_delta_accuracy`＝本件判读器的 `--gain` 输入）。**默认空 tag 会去覆写在册件，禁止。**
3. **剂量写死方式（P6）。** 本件不引入任何新剂量旋钮：治疗量完全来自 `phase_powerup2()` 里那一次 `sleep_pass.run(runtime=…, reason="n2-powerup-2", organs=True, learn=True)` 的**现行默认**，与 §5 P6 那句"沿用 `run()` 现行默认四参数、不放大任何一档"一致；写死的手法＝**发射后由件自述的 `mother_tick`/前后摘要与窗内配对列**回读，而不是我在提案里手填一个数字。若这些默认在发射前被别人改过，本单作废重签（同 [[frozen-criterion-stays-when-secondary-moves]]）。
4. **两把前置尺（缺任一则本件不判，不去改判据）。** ①噪声地板：判读器 `--noise-floor` 必须给**同量纲**的 `noise_floor_delta_quality_abs`（`floor_quality()` 只认同名键，不同量纲的极差不作地板，G-N-02-3）；没有地板 ⇒ J-N-02-1 落 `not_resolved`，**不得**写成"无收益"。②动态范围：`--gain-face` 必须出版 `usable_endpoints`，否则 `range_column` 报 `state=absent`——**这一条今天就是要红的**：今天的 N2B 件里没有逐端点范围列。〔2026-10-18 已按"补仪器、不动判据"结清这一格：新增只读仪 `scripts/training/verify_taiji_n02_gain_ruler_range.py`（读那两枚 N2B 封存件，合同测 7 条全绿），出版 `reports/taiji_n02_gain_ruler_range_20261018.json`，**现读 `usable_endpoints=1`、`range_verdict=single_channel_only`**（`by_source`＝interactions 200／constraints 0／workbench_capabilities 0），双臂分母各 **8** 项，`j_n2b_prime_delta_quality=2.4829673663245373` 而 `j_n2b_prime_delta_accuracy` **恰为 0.0**。⇒ `--gain-face` 从此有真件可指；但同一把尺先把上限说清：**来源通道只有一条活着、accuracy 那列没有位移**，所以 B 档即便跑完，跨来源归因与逐列保持能声称的幅度也受这一列限制。〕

算力与批文：§6 表里 B 档"≈45–60 分钟 CPU"是**台账既有估计，本轮未复测**，发射后一律以件自述为准，不拿估计当预算；**改权重批文（P5）仍未给**，owner 这次裁的是"选①"，故本单停在纸面，等一句"可跑 powerup2"。跑完后按 §7 四条互斥出口判读，判红即收回 §7 点名的句子。



## 7. 出口（四条互斥，先写死）＋ 判红即收回的句子

* ⑴ 四条件全真 ⇒ 才可写"C4 有一个净增益机制成立"，且仍 `can_promote=false`、C5 另立件。
* ⑵ J-1 真而 J-2 假 ⇒ 写"这一档买到的是代价换收益"，并按 PLAN-N-01 C4 行分流回"更新范围/保持"（＝C 档要批文）。
* ⑶ J-1 假或 `not_resolved` ⇒ 写"该剂量档无收益或不可判"，**不得**把"面太难／太易"写成原因，除非 §2 的 P-尺列在场。
* ⑷ J-3 或 J-4 假 ⇒ 整件不判，只出版工程缺口（这条的红是仪器红，不是能力红，两句不许混）。
* 不许写：**"C4 已开工"**（今天是规格层）；**"仪器写好了所以更接近完成"**（分层口径见记忆 `explain-ordering-and-separate-instrumentation-from-capability`）；**"资源曲线显示……"**（P4 缺件）；**"M 的语料规模扫描说明……"**（不得引用为现行）。
* 发表资格前置：本件的合取结论只在四条件同枚件在场、判级字段来自判门函数、且 `real_change_count == 0` 三条同时成立时才可出版。

## 8. 不变项

M6 收官、M8 挂起；`TRAIN_SHA=None`、`can_promote=false`、v5 面已消费（`retry_allowed=false`）、v6 题面未生成未查询；`taiji/**` 与 `seed/**` 本件未动；产品默认位（含 `sleep_pass` 的 `organs/learn`）不改；N1 队首仍是 owner 的四裁，本件不与之竞争队首。
