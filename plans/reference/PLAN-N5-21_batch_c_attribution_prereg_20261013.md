# PLAN-N5-21 批次 C 的配对差能不能归因（J-N5t-1…4 先冻；零新机时，全部读数取自在盘产物）

冻结时刻：2026-10-13（㊵-769）。执行体＝两台**已在册**仪器＋一台新仪器：

* 已在册：`scripts/training/audit_taiji_batch_c_arm_parity.py`（㊵-759，`taiji-batch-c-arm-parity-v1`）——判配置面／旗标面／结果面，**不判**指标差与权重差；
* 已在册：`scripts/training/compare_taiji_n5_arm_tensors.py`（DEBT-G92／㊵-663，今日升版 `taiji-n5-arm-tensor-comparison-v2`，见 §0 第四条）；
* 新增：`scripts/training/audit_taiji_batch_c_stream_parity.py`（本件 §3 点名，判据 J-N5t-1 的载体）。

本件冻的是**"什么读数才算把配对差归因给一个变量"以及"差异要住在哪一面"**，在任何对照臂终盘读数取到**之前**。

## §0 现状（全是本轮实测，可复算）

1. 配对差事实面（㊵-758 ⑦／㊵-761 更正后）：同点 `symbols_consumed=2,000,000` 处
   `cumulative_mean_surprise` 治疗臂 **7.1682** 对 对照臂 **6.1814**、`cumulative_online_accuracy` **0.272093** 对 **0.279056**。
   方向：治疗臂更"惊"、更不准。**这是两个点，不是一条曲线**——曲线由本件 J-N5t-1 的仪器出版。
2. 治疗臂终盘自述（`output/n5_20_batchC_treated/A/run_report.json` 现读）：`status=completed`、`arm=A`、
   `skip_symbols=11199800`、`symbols_budget=symbols_consumed=8900000`、`fresh=true`、`resumed_from=null`、
   `expected_write_surface=realised_write_surface=["predictive_readout"]`、`write_surface_ok=true`、
   基座 `checkpoints/seed_a31self_with_circuit.pt`（sha `d6169a358eaee6d194d4795e3167a7bcbb42dfde57b92aa1199bcebbed89699b`）、
   `generated_feed={every_symbols:50, feeds:178000, generated_write_count:178000, generated_row_weight_max_abs:1.379441381, generated_reward_sum:0.0, measured_symbols_per_second:241.952872}`。
   ⇒ 速率与 ETA 两枚键住在 `generated_feed` 块**里**，不在顶层（`J-N5r-3` 的键名错位即 DEBT-G118，引用要点名路径）。
3. 进度流行程结构（同臂 `progress.jsonl` 现读）：**36 行**、边界从 `250,000` 每 `250,000` 一格到 `8,900,000`、
   末行 `final=true`、`session_id` 全集为单值 `A-2026-10-12T16:31:35Z`、`window_symbols` 首行 249,999 余行 250,000。
   源码依据（`train_taiji_r2_readout_retrain.py`）：行只在 `consumed % progress_every == 0` 时落＋收尾一次，
   而 `window_symbols = seen - mark[0]`、`seen` 只在 `prior_prediction is not None` 时涨 ⇒
   **`window_symbols` 是模型依赖量**，不是流的边界。
4. 今日现取的两台仪器读数（都是本件的设计前提，不是结论）：
   * `measure_taiji_disk_load_face_two_arms.py --only 治疗臂 checkpoint.pt`（㊵-768）⇒ 治疗档 `loaded=true`／`receptors_inherited=true`，
     对照档（退回一枚来源枚举）refused `ValueError|synapse payload shape does not match architecture`。
   * `compare_taiji_n5_arm_tensors.py 基座 ↔ 治疗臂`（`reports/taiji_n5_arm_tensor_base_vs_batch_c_treated_20261013.json`，
     `status=refuse_name_sets_differ`、rc=2）⇒ 治疗臂信封比基座**多两枚** `memory.last_event`（`substrate.*` 与 `taiji.kernel.*` 各一）；
     **8 枚**形状不齐，全在 `memory.provenance_encoder.{edge_weight,pre_index}` 与 `memory.provenance_readout.{edge_weight,pre_index}`（两根各四），
     形状对为 `[384, 4] vs [384, 5]` 与 `[4, 96] vs [5, 96]` ⇒ **DEBT-G104 批次 B 的受体通道重映射第一次在长跑器产出上有字节级出处**；
     62 枚差异槽里 54 枚可比，delta 榜前四是 `perception.dynamic.symbol_exposure`（729998）与
     `identity_organ.bank.visits`（177995）这类**计数器**而不是权重。
5. **一条结构性读数限制（本件为什么要借第三台仪器）**：跑器的面账本是 `SURFACES = ("motor", "predictive_readout")`
   （`scripts/training/readout_retrain_spec.py:37`），**不含 `memory.*`** ⇒ 喂入真正写的那一面在跑器自己的
   `realised_write_surface` 里不可见；`write_surface_ok=true` 只授权"账本内那两面动了"，不授权"只有那两面动了"。
   基座↔治疗臂面里 `identity_organ.bank.visits` 的 177,995 与喂入数 178,000 只差 5，**这句在基座↔臂面上不构成归因**
   （对照臂同样跑满 8.9M 符号，也会动计数器）——它只在 J-N5t-2 的**臂↔臂**差里才有资格被引用。

## §1 本件要答的一句话

批次 C 那枚配对差能不能归因给"喂 `generated`"这**一个**变量，以及差异住在哪一面。归因需要三条并立：
同流（J-N5t-1）、同旗标外配置（已由并排仪判）、同跑器码版（已由并排仪按起跑时刻回查 `git rev-list`）。

## §2 判据（冻四条，每条写明"什么情况下它为假"）

* **J-N5t-1 输入流同序同界**：两臂 `progress.jsonl` 必须**行数相等**，且逐行按行序对齐时
  `symbols_consumed` 与 `absolute_tick` **同值**；两臂各自的 `session_id` 只许一枚值（多枚＝跨进程追加，历史踩过）。
  为假的情形＝任一行边界不同或行数不同（含 `stopped_by_request` 早停）⇒ **配对作废**，
  §0 第 1 条那两个点退回成"两个臂各自的点"，本件全部机制句收回。
  **出版但不判**：`window_symbols`、`session_symbols`、`elapsed_seconds`、速率——前二是模型依赖量（§0 第 3 条），
  后二是计时量；把它们当混杂去判会造出一条必然为假的判据。
  本条只证"流同"，**不证**"实现同"：浮点累加次序与线程不在本条覆盖范围，引用时必须带上这句。
* **J-N5t-2 差异要住在具名的三类面上**：`compare_taiji_n5_arm_tensors.py --checkpoint 治疗 --checkpoint 对照` 必须
  `status=measured`，且差异集（`differing_slots_names`）的每一枚都要落进下面三类之一：
  ①`memory.provenance_encoder.*`／`memory.provenance_readout.*`（喂入直接写的那一面）、
  ②`identity_organ.bank.*`（producer 走的计数与 bank）、
  ③跑器账本内的 `motor.*`／`predictive_readout.*`。
  为假的情形分两支且**分开收回**：出现**第四类**名字 ⇒ 差异有第三条通路，机制句收回，只保留"两臂读数不同"；
  差异集为**空**（`verdict=identical`）⇒ surprise 差不可能由权重解释，改判成"两臂权重逐位相同、差只在计量口径"，
  并把它升格成一条独立缺陷（计量口径没对齐）。
* **J-N5t-3 比得成才谈得上判**：两臂各自的 `mirror_consistency` 必须 `agree_within_arm == mirrored_slots`
  （任何一臂内 `substrate.*` 与 `taiji.kernel.*` 不一致＝存盘路径分叉，比"两臂有差"更响亮，全部结论作废）；
  臂↔臂那一趟的 `shape_mismatch_slots` 必须为 **0**——两臂由同一版跑器、同一基座写出，形状不齐说明并排仪漏判了配置面，
  先修判据再谈差异。名集不齐（`refuse_name_sets_differ`）的那趟**不许**当 J-N5t-2 的判据载体。
* **J-N5t-4 幅度句另需一枚噪声地板**：本件只授权"方向＋定位"。任何**幅度**主张（包括把 7.1682−6.1814 写成"涨 0.9868"这类读法）
  必须先有一枚**同配置重跑**的噪声地板（同 base、同 corpus、同预算、同旗标值再跑一臂），那要新机时 ⇒ 属 owner 排期。
  本件冻结时即写明：**该前置未满足**；未满足之前 §1 那句只许以"方向＋定位"的形式出现。

## §3 判据载体在哪（措辞门要求的一节）

* J-N5t-1＝新仪器 `scripts/training/audit_taiji_batch_c_stream_parity.py` 的 `verdict`，
  契约册＝`tests/taiji_native/test_n5_34_batch_c_stream_parity_contract.py`（号现取：`test_n5_*` 用到 33）。
  合成两臂进度流钉四支：同界为绿／把一行边界挪一格必须红／少一行必须红／`window_symbols` 不等**不得**把 verdict 拖红。
* J-N5t-2／J-N5t-3＝已在册比较器 `compare_taiji_n5_arm_tensors.py` 的件，其测为
  `tests/taiji_native/test_n5_18_arm_tensor_compare_contract.py`（今日 9 支全绿，含 DEBT-G129 那两支：形状不齐必须点名、
  delta 表与形状名表必须是差异槽的一个划分）。
* 读数件＝`reports/taiji_batch_c_stream_parity_20261013.json`（J-N5t-1 的发表件，等对照臂终盘后一次成型）。

## §4 半径

只动 `plans/`＋`scripts/training/`（新增一枚仪器）＋`tests/`＋一枚新读数件。**不**动 `taiji/`、**不**起新跑、
**不**在对照臂在飞期间读它的产物（进度流是追加式、`checkpoint.pt` 每 `checkpoint_every` 重写，读到半写信封只会造出假读数）。

## §5 落笔即钉的三行

* **预期绿**：`J-N5t-1`（两臂同流是跑器的硬设计，§0 第 3 条给了源码依据）与 `J-N5t-3`（两臂同版同基座）。
* **仍红时收回哪句并点名**：`J-N5t-1` 红 ⇒ 收回"批次 C 是一变量实验"，并点名是哪一行边界差；`J-N5t-3` 的镜像那支红 ⇒
  收回两臂全部比较结论并新登一条存盘分叉缺陷；`J-N5t-2` 出现第四类 ⇒ 只收回机制句，配对读数本身仍出版。
* **发表资格前置**：`J-N5t-1`＋`J-N5t-2`＋`J-N5t-3` 三支同绿之前，「喂 `generated` 改变了 X，因而 surprise 变高」
  这句**不许**发表；三支同绿之后，只授权**方向与定位**，幅度句仍被 `J-N5t-4` 挡住。

## §6 落笔后同日更正（2026-10-13 ㊵-770；§2 原文一字不动，判据单位改窄，**臂↔臂读数尚未取过**）

**要更正的那半句**：§2 的 `J-N5t-2` 把"差异集里出现第四类名字 ⇒ 机制句收回"写在了**全部差异槽**上，
而我列的三类面漏掉了**计量面**。查过源码，这一漏会让一条本来能判真的判据**必然误报**：

* 喂入那一步不止写情节，它还**多走一次 observe**（`scripts/training/train_taiji_r2_readout_retrain.py:120-127`：
  `record_generated_prediction(...)` 之后紧跟 `substrate.observe(answer_symbol, **flush_kwargs)`，且 `learn*` 位全关）。
* observe 顶起计量面：`taiji/perception.py:876` 的 `self._symbol_exposure[symbol].add_(1.0)`
  （存盘名＝`perception.dynamic.symbol_exposure`），`taiji/cue_binding.py:58` 的 `self.visits = torch.zeros(self.capacity, ...)`
  按绑定/观察递增。⇒ 治疗臂比对照臂多 **178,000 次**观察（＝`generated_feed.feeds`），两臂的计量面**按构造必然不同**。
* 这与 §0 第 4 条的实测形状吻合：基座↔治疗臂的 delta 榜前两名就是这两枚计数器（729,998 与 177,995）；
  而 177,995 与 `feeds` 只差 5——**这句不是归因**（基座↔臂那一趟把 8.9M 次普通观察也算进去了），
  它只在臂↔臂那一趟里才有资格被引用，且引用时只许说"计量面之差与喂入次数同量级"。

**改窄后的判据单位**（三条，全部挂在 `J-N5t-2` 的名字下生效）：

1. 差异槽按**名字**分成三档出版（机械规则，不靠人读）：
   `计量面` ＝ 名字以 `.symbol_exposure` 或 `.visits` 结尾；`拓扑面` ＝ 名字以 `.pre_index` 结尾
   （`taiji/adaptive_residual_shadow.py:372-382` 把它当 long 型槽位索引用，所以它是**结构**而不是权重，也不是计数器）；
   其余＝`权重面`。`J-N5t-2` 的"第四类"判定**只看权重面与拓扑面**；计量面**只出版不判**，
   并且件里要写明"两臂计量面按构造必然不等"这句（否则下一轮有人拿它当混杂未对齐的证据）。
2. **反方向的诱惑也钉住**：若权重面与拓扑面差异集为空、只有计量面不等 ⇒ 读数是"喂入只动了计量"，
   `J-N5t-2` 走原文"差异集为空"那一支（措辞不许升格成"权重没变所以喂入无效"，因为喂入本来就不许动学习面——
   那正是 `PLAN-N5-20` §1 第 3 句要的）。
3. **这条更正本身要可核对**：`compare_taiji_n5_arm_tensors.py` 现在没有这三档分列，
   所以 §3 的载体在取臂↔臂读数**之前**必须先把分档做成加性字段＋一支负对照（同一条差异集，三档相加＝总差异槽），
   否则我就是在用一台还没这格列的仪器引用这格列。

**性质声明**：本节改的是**判据单位与分档**（把必然不等的计量面从判决里摘出去），不是阈值、不是结论强度；
`J-N5t-1`／`J-N5t-3`／`J-N5t-4` 与 §5 那三行一字未动，§2 原文留在原地作历史。

### §6.1 同一轮再更正一次（2026-10-13 ㊵-770，仍**未取任何臂↔臂读数**）：`J-N5t-2` 的效力要降级

上面第 1 条把计量面摘出去是对的，但**剩下的两类仍然不够支撑"机制定位"**。把真面的分档跑出来才看清：
基座↔治疗臂那一趟 62 枚差异槽分成 权重 54／拓扑 4／计量 4，而"权重 54"里点名的前几枚是
`substrate.memory.association.edge_weight`、`substrate.identity_organ.bank.prototypes`、
`substrate.identity_organ.bank.occupied`、`substrate.cognitive_state.{percept.features,workspace.broadcast,world.latent}`
——**都不在我 §2 列的三类面里**。

原因不是名字起错，而是**可达面比 §2 写的宽**：

* 喂入那一步 `observe(answer_symbol, learn*=False)` 是一次真观察 ⇒ 情节场、绑定库、认知状态这类
  "观察会写"的面在两臂之间**按构造**可以不同；
* 普通步的 `learn*` 是**开着的**：治疗臂多了 178,000 条经验，它后续每一步的更新对象与幅度都可能不同
  ⇒ 连 `predictive_readout.*` 都没有"应当相等"的先验。

⇒ **按 §2 原文跑，`J-N5t-2` 会在一个完全正常的结果上误报"出现第四类"**（把可解释的差异读成不可归因）。
现在把它降级——方向是**变弱**，不是放宽：

1. `J-N5t-2` 只保留一条**冻结面自证**，它是这台仪器真正有判别力的那一半：
   A 臂的 spec 明写"运动面冻结，`predictive_readout` 单点写入"，所以对**每一臂自己**必须
   `run_report.digests_before["motor"] == run_report.digests_after["motor"]`；并且臂↔臂那一趟里，
   **冻结集**＝名字含 `.motor.` 的那批（本轮真面现读 **10 枚**：`substrate.motor.*` 与 `taiji.kernel.motor.*` 各 5 枚——
   `receptors.channel`／`receptors.polarity`／`synapses.pre_index`／`synapses.edge_weight`／`bias`）差异槽必须为 **0**。
   刻意**不含** `*.state.motor_context` 与 `*.state.motor_probabilities` 那 4 枚：它们是逐步状态不是权重，
   两臂不同是按构造预期的，把它们算进冻结集会让这条判据必然为假。
   为假的情形＝冻结面在两臂之间动了 ⇒ 存在第二条通路，两臂之差不可归因，机制句与方向句一起收回。
   （这一支**能为假**：它判的是"账本内声明冻结的那一面"，不是"所有没点名的面"。）
2. 其余面（`memory.*`／`identity_organ.*`／`cognitive_state.*`／`predictive_readout.*`／计量与拓扑）**只出版、不定位**；
   任何"差异住在被喂的那一行"这类句子在 `J-N5t-2` 下**不发表**。
3. 定位句的前置另立一件（新登 **DEBT-G131**），形状先写死在这里：**以基座为共同参照跑两趟 base↔臂**
   （比较器已经出版 `differing_slots_names`，两趟各一枚），再取"只在治疗臂出现的差异名集"并按前缀归组。
   刻意**不**从跑器的 `digests_before`／`digests_after` 推面集——那两枚键只覆盖 `SURFACES` 那两面
   （＝DEBT-G130 记的同一处失明），从它推出"面集"会把 memory/association/bank 全吞掉；
   新仪器只做两趟件之间的**集合差**，不重抄比较链（[[dont-reimplement-the-chain]]）。
   在它出数之前，`J-N5t-4` 的幅度前置继续挡着。
4. §2 原文与 §6 第 1 条一字不动，都在原地可查；改在任何臂↔臂读数之前。
