# PLAN-N5-20 批次 C 的长跑规格与定价（DEBT-G116 修法② 的冻结件；判据先冻，实施只动 `scripts/`）

冻结时刻：2026-10-12（㊵-738）。执行体＝`scripts/training/train_taiji_r2_readout_retrain.py`（现读 507 行，
`run_arm()` 在第 107 行，臂定义在 `ARMS`，进度流 `progress.jsonl`、摘要 `surface_digests`、续跑守护在 107-175 段）。
本件冻的是**长跑要产出什么、怎么算成、以及什么读数才算"长出权重"**——不是预算：预算二裁⑵ 已给（19 h 单臂＋12 h 对照/复算档）。

## §0 现状（都是本会话实测，可复算）

* producer 已在 #72 落地：`Taiji.record_generated_answer(...)`（来源固定 `generated`、奖励固定 `0.0`、
  **默认零调用者**），契约册 `test_n5_30_generated_producer_contract.py` 六支全绿（㊵-736）。
* 长跑器仍不碰情节：对 `provenance`／`settle_action`／`memory.write`／`episodic` 四个关键词命中数＝**0**（㊵-734 复跑同值）。
  ⇒ 批次 C 缺的正是**挂接**，不是机时。
* 情节写入的落账时机是**下一次观察**（㊵-736 §7 的实测）：只 `settle`/`record` 不 observe，计数与权重都不动。
  任何挂接都必须把"喂完之后走一步"写进循环，否则跑出来的计数会恒 0 而被误读成"没长"。

## §1 挂接形状（三句，第一句是硬约束）

1. **只有一个新旗标**：`--generated-feed-every N`，默认 **0＝关闭**。默认关闭是"不动现有三臂语义"的唯一形状——
   不新增臂、不改 `ARMS` 里任何既有臂的 spec、不改续跑守护的判定。
2. **只调产品入口**：挂接体必须调 `substrate.record_generated_answer(...)`（producer 在 `Taiji` 一侧），
   随后**照现有节奏走一步观察**让写入落账；不许在跑器里自己拼情节 payload 或直调 `EpisodicField.write(`。
3. **不引入非零奖励**：generated 的喂入永远零奖励（§19.6）。跑器自述里出版 `generated_reward_sum`，它必须恰为 `0.0`。

## §2 判据（冻五条，每条写明"什么情况下它为假"）

* **J-N5r-1 关闭即无副作用**：`--generated-feed-every 0`（或不传）跑完，`generated_write_count` 必须为 **0**
  且 `generated_row_weight`（见 J-N5r-2 定义）必须恰为 **0.0**。为假的情形＝任何一项非零 ⇒ 挂接漏进了默认路径，
  本件全部作废重冻，而不是"把非零解释成无害"。
* **J-N5r-2 长出权重才算成**：打开喂入的一段跑之后，必须同时出版
  ①`generated_write_count > 0`（被走到）、②`generated_row_weight > 0`，其定义钉死为
  `provenance_readout.edge_weight[generated 行].abs().max()`（**行内绝对值最大**，不用均值/和，避免被大量零边稀释成假 0）。
  为假的情形＝①真②假：写进去了但那一行没有任何权重变化 ⇒ 说明该行的可写边没被走到（`local_update` 只写已存在的边，
  见 ㊵-720 的"活槽位"结论），必须点名是拓扑没长还是学习率/衰减把它抹平，不许把②改小阈值蒙过去。
* **J-N5r-3 定价只认本次实测**：跑器自述必须出版 `measured_ticks_per_second` 与 `eta_hours_at_requested_budget`，
  且长跑预算只能由它推。参考上界（**不是**定价来源）：㊵-673 三臂 150k ticks 实测 932.063／775.959／775.876 秒
  ⇒ 约 161–193 ticks/秒 ⇒ 19 h ≈ 1.1×10⁷–1.3×10⁷ ticks。为假的情形＝件里出现的 ETA 来自引用而不是本次读数。
* **J-N5r-4 双臂只差一个变量**：治疗臂（喂 generated）与对照臂（同 base、同 corpus、同预算、不喂）之间
  只允许 `generated_feed_every` 一个差异，跑器自述必须把两臂的这个旗标值与其余关键字段并排出版。
  为假的情形＝两臂还差 corpus/seed/base sha/`--fresh` 中任一项 ⇒ 差异不可归因，两臂同 0 或同非 0 都不算结论。
* **J-N5r-5 跑完必须能在装机面重取**：新落的两臂件必须能通过在册双臂普查器
  `scripts/training/measure_taiji_disk_load_face_two_arms.py`（㊵-730）读出"可载入＋`receptors_inherited` 自述"。
  为假的情形＝新件在普查器上拒载 ⇒ 批次 B 的迁移支没覆盖到长跑器写出的形状，先修载入再谈能力。

## §3 守卫（两条，各带负对照）

* **G-N5r-1 不许第二条生成链**：跑器文件内不得出现 `memory.write(` 或直接构造 `EpisodicWrite`；判定＝AST 数
  该文件里属性名为 `write` 且挂在情节场对象上的调用，必须为 0；负对照＝喂一段含 `model.memory.write(...)` 的
  合成源码，计数必须为 1（同族教训见 ㊵-736 §7 的 `ast.walk` 一项）。
* **G-N5r-2 关旗标不改变既有臂的进度语义**：`progress.jsonl` 每行仍必须带 `session_id`（㊵-736 之前踩过的
  "跨进程追加导致相邻行差算出负速率"），且新增键只许出现在 `run_report.json` 的自述块里，
  不许塞进既有必需键的冻结块。为假的情形＝任一既有必需键被新键挤掉或复用语义。

## §4 批次与批准边界

| 批次 | 内容 | 半径 | 谁能做 |
|---|---|---|---|
| A | 本件＋台账 | 仅 `plans/` | 我可自办 |
| B | 跑器挂接＋两支自述键＋契约册（短程出数，预算按 §2 的速率现定） | `scripts/training/`＋`tests/` | 我可自办（四裁/五裁已给的 producer→定价顺序之内） |
| C | 19 h＋12 h 长跑与能力主张 | 训练跑（墙钟超出本轮） | **等 owner 排窗**，起跑前须先出 B 的短程读数 |

## §5 判据载体在哪（措辞门要求的一节）

执行体＝`tests/taiji_native/test_n5_31_readout_retrain_generated_feed_contract.py`（号现取：`test_n5_*` 用到 30），
`J-N5r-1`／`J-N5r-2`／`J-N5r-4` 走"小配置＋短预算"的进程内跑（不起真长跑），`J-N5r-3`／`J-N5r-5` 走跑器自述与
普查件的字段一致性；`G-N5r-1`／`G-N5r-2` 为 AST/键集守卫。

## §6 落笔即钉的三行

* **预期绿**：`J-N5r-1`（关闭无副作用）、`J-N5r-2`（喂入后两枚键同正）、`J-N5r-3`、`J-N5r-4`、`J-N5r-5` 五支＋两条守卫。
* **仍红时收回哪句**：若 `J-N5r-2` 的②为 0 ⇒ 当场收回"批次 C 已可起跑"，并把结论改写成"generated 行没有可写边"这一条新缺陷
  （它属 DEBT-G104 的续，需要结构判据而不是加预算）；若 `J-N5r-1` 红 ⇒ 收回"默认不动"，挂接回退；
  若 `J-N5r-5` 红 ⇒ 收回"长跑件可复算"，批次 B 先修载入形状。
* **发表资格前置**：本件全绿之前，「`generated` 已长出权重」不许说；全绿之后，短程读数只授权**方向**；
  任何**幅度**主张必须来自批次 C 的真长跑，且写清是哪两臂、哪张面、什么预算。


## §7 挂接落地与两条形状更正（2026-10-12 ㊵-740；§1–§6 原文一字不动）

* **§1 第 2 句按字面走不通**：冻的入口 `record_generated_answer(...)` 骑在 `settle_action` 上，而 A 臂（读出头）
  整段循环从不 `act()`，于是"喂完之后照现有节奏走一步"之前先缺那枚 pending 动作——㊵-739 因此退回半截跑器并
  登记 DEBT-G117。六裁⑴ 选甲＝新增**无动作**入口 `record_generated_prediction(...)`，挂接改调它。
  这不是放松：§1 第 2 句的两条硬约束（只调产品入口、不许在跑器里自己拼情节 payload 或直调 `EpisodicField.write(`）
  照旧成立，`G-N5r-1` 现读仍为 0。
* **挂接撞到的三处产品约束**（全是响亮错，没有一处静默）：①`act()` 要求先有一次动作读出的观察（A 臂没有）；
  ②同一段活动动力学回合内不许切换读出 ⇒ 喂入后的那一步必须沿用**该臂自己的** `readout`；
  ③`learn_predictive_readout` 要求 `learn=True` ⇒ 落账那一步把全部 `learn*` 键置假、只保留 `readout`，
  也就是"喂入自己不动学习面"。契约册
  `test_the_flush_keeps_the_arms_readout_and_turns_every_learn_off` 钉的就是②③两支。
* **第一次双臂出数**（本次实测；同一 base `checkpoints/seed_a31self_with_circuit.pt`、
  摘要 `d6169a358eaee6d194d4795e3167a7bcbb42dfde57b92aa1199bcebbed89699b`、同臂 A、`--symbols 400`、
  `--skip-symbols 11199800`，两跑只差 `--generated-feed-every` 0 对 50）：
  对照臂 `feeds=0`、`generated_write_count=0`、`generated_row_weight_max_abs=0.0`、245.075859 符号/秒；
  治疗臂 `every_symbols=50`、`feeds=8`、`generated_write_count=8`、`generated_row_weight_max_abs=0.276523322`、
  `generated_reward_sum=0.0`、206.739309 符号/秒。⇒ **`J-N5r-1` 真、`J-N5r-2` 两半都真（被走到＋那一行长出了权重）、
  `J-N5r-4` 成立**；`J-N5r-5` 未测（它要的是真长跑落件，临时件不入册，所以本件现状＝四支有值＋一支未测）。
* **`J-N5r-3` 冻错了一枚键名**（新登 DEBT-G118）：件里要 `measured_ticks_per_second`，跑器原生出版的是
  `measured_symbols_per_second`（该跑器的预算旋钮是 `--symbols`）。本次两臂在同一窗口里 `symbols_consumed` 与
  `absolute_tick` 同为 400，数值口径不因此改变，但"该键存在"这一半按字面为假，不许蒙过去。
* **定价只认治疗臂**：19 h ≈ 206.739309 × 68 400 ≈ **1.41×10⁷ 符号**；12 h ≈ **8.93×10⁶ 符号**。
  ㊵-739 那枚 248.552647 与本次对照臂 245.075859 相差 1.4%，只有两个样本，所以两臂速率之差（−15.6%）
  **不许**被说成"喂入的代价"；它在这里只被当作定价的保守除数。§6 那句"任何幅度主张必须来自批次 C 的真长跑"
  仍然有效——以上是速率，不是能力。
* **§1 第 1 句的"只有一个新旗标"仍然成立**：`ARMS` 未增未改、续跑守护判定未改，新键只进 `run_report.json`
  的 `generated_feed` 自述块（`G-N5r-2` 现读：`progress.jsonl` 每行仍带 `session_id`，既有必需键一枚没被挤掉）。
