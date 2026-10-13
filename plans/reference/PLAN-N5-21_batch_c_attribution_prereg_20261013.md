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
