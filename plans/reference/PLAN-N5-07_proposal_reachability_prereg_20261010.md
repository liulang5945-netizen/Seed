# PLAN-N5-07 实施预注册（2026-10-10）：甲路＝"先让提议真发生"的三段前置

> owner 弹窗 #19 第①裁：N5 保持侧的下一手取**先让提议真发生**（更强结论），不取"把影子支从
> `should_propose` 解耦"。本件是那一句的**实施形状**，判据先冻、不开跑。
> 台账：[08 ㊵-654](../../active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md)；债：[DEBT-G84](../../active/roadmap/05_TECH_DEBT_REGISTER.md)；
> 上游件：[PLAN-N5-04](PLAN-N5-04_retention_rebase_prereg_20261009.md)（判据一字不动）、
> [PLAN-N5-04 判读](PLAN-N5-04_ADJUDICATION_20261010.md)、[PLAN-N3-08](PLAN-N3-08_face_self_report_prereg_20261008.md)、
> [PLAN-N3-09](PLAN-N3-09_tau_rule_upgrade_prereg_20261008.md)。

## §0 为什么现在必须另立一件（不是重跑一次双臂）

三条现读证据，全部零算力、都在盘上可复算：

1. **双臂塌成一枚权重**：㊵-654 的两臂逐张量比对 `DIFFERING=0 of 238`，两臂 `n5_shadow` 整块缺席，
   9,727 条观测里 `decision_should_propose` 真值 **0**，`decision_pressure` 上界 **0.5571776111896318**。
2. **在库四张在线面是 v1，冻不出合法 τ**：`freeze_taiji_n3_09_tau_rule.py --face
   output/n3_04/beta_gate100/pressure.jsonl` 现跑 rc=2，理由原文＝"format 是
   `taiji-n3-pressure-face-v1` ⇒ 本件规则只认面内自述口径，v1 面缺六道阈与 EMA"。
   ㊵-505 那句"β 链 0.65 会触发"是在 v1 面上**假定产品默认值**重放得来的，按 PLAN-N3-08 的主判据
   （`assumed_from_product_defaults` 长度必须 == 0）**不得**当结论搬用。
3. **唯一的 v2 面跑在"两维恒零"的装配上**：`output/n3_08_smoke/pressure.jsonl`（4,999 条有效观测）
   的收尾自述＝`fast_slow_requested=false`／`bridge_gate_actual=0.0`／`learning_mode=read_only`，
   `decision_fast_slow_conflict_ema` 与 `decision_activity_saturation_ema` 的 max 都是 **0.0000**；
   该面按规则算出的候选 τ＝**0.42**（`pressure_ema` max 0.423982）但
   `six_gate_longest_run=0` ⇒ `not_freezable_at_this_grid`。
   **这条红要归位给装配，不归位给产品**——owner 2026-10-08 已明令"不在两维恒零的配置上调低默认阈凑触发"。

⇒ 所以"让提议真发生"不是把某个数改小，而是**在四维真活着的装配上重新取得 τ**。
基座为什么换到出厂 beta 链：㊵-497 已查实本件原先冻的在训链不可执行（`--no-readout-position`
带着它撞 `organs.py:925`、不带它撞 `model.py:1124-1135` 的互斥拒绝），而 ㊵-504 已把链裁成
`seed_beta.pt` 与 `seed_beta_with_circuit.pt` 两枚出厂系档——**换链换的是分布**，因此本件的结论
只回答"出厂系装配上通电有没有保持代价"，不回答"在训链上 τ 定在哪"。

## §1 不动的东西

- PLAN-N5-04 的 `J-N5d-1/2/3` 与 `G-N5d-1..5` **一字不改**；本件只是它的前置件（前置未达成时
  `J-N5d-1` 维持"无读数"，本件不代答）。
- 产品默认位不动（`taiji/config.py` 不改），六道阈只进命令行（沿用 `G-N5d-3`）。
- 判读器与配对仪器一律**不重造**：`adjudicate_taiji_n2_04_retention_pair.py`、
  `adjudicate_taiji_n5_shadow_gate.py`、`freeze_taiji_n3_09_tau_rule.py`、
  `run_taiji_n5_retention_lane.py` 都是现成的；本件不加新尺。

## §2 三段的形状（参数写死，等第①段批准）

- **第①段＝跑 PLAN-N3-08 的正式 v2 面**：出厂链两枚（`seed_beta.pt`／`seed_beta_with_circuit.pt`）
  × bridge gate 两档（`0.25`／`1.0`）＝**4 支在线面 ＋ 1 支默认关守卫臂**（与 ㊵-504 的面数同形），
  每支 `--max-symbols 4000`、`--readout predictive`、带 `--developmental-fast-slow`，
  放行臂带 `--developmental-bridge-gate`，`with_circuit` 两份额外带 `--no-readout-position`
  （PLAN-N3-04 §5ter 的互斥约束，读数须带这条限定）。
- **第②段＝零算力现冻 τ**：对第①段每支面跑 `freeze_taiji_n3_09_tau_rule.py`，取
  `tau_candidate_v2` 与 `dynamic_range_at_candidate`。
- **第③段＝双臂**：基座取第②段里**通过 J-N5f-2** 的那条链，`--growth-min-pressure` 写该链现冻值，
  每臂 `--max-symbols 60000`，唯一变量 `--n5-shadow-gate 1.0` vs 缺席，其余参数两臂逐字相同；
  由 `run_taiji_n5_retention_lane.py` 执行（它现在带两道花钱前的门）。
  三条已长在训练器里的既有纪律本件直接继承、不重造：`--growth-min-pressure` 的 `dest` 是
  `growth_minimum_pressure`，它**只换 `minimum_pressure` 一枚**，其余五道阈与 `ema_rate` 仍取产品默认
  （`:1190` 帮助原文），缺省 `None` 时连 policy 参数都不给；挂载后从 `trigger.policy` **读回实际值**，
  请求值 ≠ 读回值 ⇒ "面作废（τ 必须由实际生效值定义）"（`:752-757`），并把
  `minimum_pressure_requested` 写进面头（`:826`）。owner 2026-10-08 第六次弹窗已裁"τ 按链分别给值"。

## §2bis 第①段的参数表（**形状未跑**——本件落库时一条面都没起，逐条由批文冻结）

| 面 | 基座 | `--developmental-bridge-gate` | 位置输入 | `--max-symbols` | 守卫 |
| --- | --- | --- | --- | --- | --- |
| `beta_gate025` | `checkpoints/seed_beta.pt` | `0.25` | 开（默认） | `4000` | 放行臂 |
| `beta_gate100` | `checkpoints/seed_beta.pt` | `1.0` | 开（默认） | `4000` | 放行臂 |
| `circuit_gate025` | `checkpoints/seed_beta_with_circuit.pt` | `0.25` | `--no-readout-position`（PLAN-N3-04 §5ter） | `4000` | 放行臂 |
| `circuit_gate100` | `checkpoints/seed_beta_with_circuit.pt` | `1.0` | `--no-readout-position` | `4000` | 放行臂 |
| `beta_guard_off` | `checkpoints/seed_beta.pt` | 不给 | 开（默认） | `4000` | 默认关对照臂 |

- 五支都带 `--developmental-fast-slow`、`--readout predictive`、`--pressure-record <面目录>/pressure.jsonl`，
  并各自点名 `--checkpoint`／`--progress` 到 `output/n5_07/<面名>/`（**互不相同的落点**，㊵-650 那条
  "两臂不许指向同一档"的教训同样适用于多面）。
- 第①段**不带** `--growth-min-pressure` ⇒ 面内自述的 `minimum_pressure` 就是产品默认 0.70，
  J-N5f-2 冻出的 τ 是"应当放在哪"，不是"已经生效的是多少"；第③段才把该 τ 作为命令行值生效，
  并由训练器读回复核（请求≠读回 ⇒ 面作废）。
- 命令逐条在起跑前写进批文并**当场跑一次 `--help` 校验旗标名**（㊵-593／㊵-638④ 两次都是把带参旗标
  写成裸旗标、照抄即被 argparse 拒），未做这条校验就不许起面。


## §3 判据（先冻，四条）

- **J-N5f-1（面自述合格）**：第①段每支面的收尾自述里六道 `minimum_*` 与 `ema_rate` **全部由面内自述给出**，
  且逐行闸表与自述一致——**〔2026-10-10 ㊵-656 就地更正取数键，语义未变〕** 原判据句写的
  `assumed_from_product_defaults` 与"自述为真的道数"是**重放仪**（PLAN-N3-06／N3-08 族）的输出键，
  而本段唯一的执行者是 `freeze_taiji_n3_09_tau_rule.py`，它对"零假定"的证明是三个键：
  `policy_self_reported`（六枚 `minimum_*` ＋ `required_pressure_steps` ＋ `ema_rate` 齐）、
  `gate_table_rows_checked == observations`、`gate_table_mismatches == 0`。
  四张在线面实测各 3,999 行 checked／**0 mismatches** ⇒ 语义（自述必须为真、不许回落默认值）达成，
  写错的是键名——这是我按别台仪器的输出形状预注册了一条取数句，属 [[red-attribution-needs-a-fixed-face]]
  的同族缺陷，按日期登记而不是悄悄改。任一支不满足 ⇒ 该支不得进第②段，本件按 §6 出口 E1 处理。
- **J-N5f-2（τ 可冻）**：第②段对该链的放行臂面出版 `tau_candidate_v2` 为数值，且
  `dynamic_range_at_candidate.passes == true`、`six_gate_longest_run ≥ 3`，其中
  `required_pressure_steps == 3` 必须**从面内自述读回**（不许抄常量）；`passes == false` ⇒ 第③段不起。
- **J-N5f-3（提议真发生＝本件的在场性判据）**：第③段两臂各自的压强面里 `decision_should_propose`
  为真的行数 **≥ 1**；两臂 `n5_shadow` 块的 `missing_self_report_keys` 长度 **== 0**；
  治疗臂 `shadow_branch_hits ≥ 1`；控制臂 `shadow_learn_hits == 0` 且其 `gate == 0.0`。
  四条是**合取**，缺任一条 ⇒ 双臂仍无对象，收回"通电有读数"这句。
- **J-N5f-4（不代答保持侧）**：本件**不**给 `J-N5d-1` 填值；仅当 J-N5f-3 成立后，四张 after 面才交给
  未改动的判读器，按 N5-04 §2 的原式判，且两臂 verdict 各自抄录。

## §4 守卫（fail-closed，六条）

- **G-N5f-1 τ 不得跨链搬**：τ 的六元组（面／装配／加权和／链／口径／自述来源）必须与要用它的那条链
  逐元一致；来自另一条链或 v1 面的数值出现在命令里 ⇒ 拒绝起跑（PLAN-N3-03 §4 ＋ ㊵-654 的同款教训）。
- **G-N5f-2 六道阈全取自面内自述**：判读时出现 `assumed_from_product_defaults` 非空 ⇒ rc=2，
  不许回落到产品默认。
- **G-N5f-3 默认位不动**：`taiji/config.py` 与产品默认位一字不改，τ 只进命令行。
- **G-N5f-4 单变量机械执行**：双臂由 `run_taiji_n5_retention_lane.py` 的 `lineage_refusals` 核对
  （同底 sha／同语料指纹／摘掉逐臂落点旗标与该枚门旗标后 argv 逐位相同），两臂落点指向同一档 ⇒ 拒。
- **G-N5f-5 花钱前的两道门不许绕过**：`--only` 只用于重跑**已经过了门**的那一步；用 `--only` 绕过
  G-N5d-4 在场性门或材料溯源门来"先出个数" ⇒ 本件视为违规，读数不出版。
- **G-N5f-6 warmup 只在 τ 计算里剔**：`K = 17` 条 warmup 的剔除发生在分位数计算内；面行与面件本身
  不删不改（面件的逐行完整性由既有锚点守）。
- **G-N5f-7 只动一枚是可冻性的边界，不是缺陷**：训练器只把 `minimum_pressure` 交给命令行，其余五道
  阈与 `ema_rate` 恒取产品默认 ⇒ J-N5f-2 的"可冻"必须在**那五道取面内自述默认值**的前提下判；
  任何"把另外几道也调低就能触发"的推算**不得**写进本件的结论（owner 2026-10-08 已裁
  "不在两维恒零的配置上调低默认阈凑触发"，本条是它在单旋钮形状上的延伸）。

## §5 预算（全部取实测，不外推）

- 第①段（**〔2026-10-10 ㊵-656 用实测替换外推〕**）：出厂链 4,000 符号五支面的完成记账
  ＝`28.79`／`28.44`／`29.39`／`28.61`／`21.16` 秒（`reached_budget` 五支全 `true`）
  ⇒ 合计 **136.39 秒 ≈ 2.27 分钟**。
  原先写的"≈46 秒／支、五支≈4 分钟"是用 ㊵-497 的 200 符号 2.316 s **线性外推**得到的，比实测高约 1.6 倍
  （外推为何偏高本件不猜成因，只登记一条：预算数要取无负载机器上的直测）；
  各面 `.pt` 摘要前 12 位＝`e413b6a4915f`（beta_gate025）／`ed6e52f21bc6`（beta_gate100）／
  `f48a7bb2b167`（circuit_gate025）／`ad0344d7244e`（circuit_gate100）／`d167fd085adf`（守卫臂）。
  磁盘按同形状已实测档位估 ＝ `output/n3_04/`＋canary 合计 **233.7 MB**（㊵-504 登记值，含
  `*.pt` 231,690,070 B、`.jsonl` 13,303,443 B）。
- 第③段：双臂各 **450.294 s**／**453.186 s**（㊵-654 实测）⇒ ≈ **15 分钟**；
  每臂 `.pt` **22,277,577 B**、压强面 **11,980,813 B**／**11,980,840 B**（实测）。
- 第②段：零算力（读 JSONL）。

## §6 失败出口（数值合取式，不许两读）

- **E1（第①段面不合格）**：`assumed_from_product_defaults` 长度 ≥ **1** 或自述为真的道数 < **6** ⇒
  本件停在第①段，收回"τ 已现冻"，并把不合格的道数写进台账（不重跑第③段）。
- **E2（τ 冻不出）**：对该链全部放行臂面，`passes == false` 的支数 **==** 支数 ⇒ 第③段不起，
  结论写"该装配上六道合取在本档暴露量内无动态范围"，并按 09 §4 分流转"换暴露量"或"换装配"（都要另立件）。
- **E3（提议仍不发生）**：第③段两臂 `decision_should_propose` 真行数合计 **== 0** ⇒ 双臂无对象，
  收回"J-N5d-1 有读数"，并登记一条"τ 现冻值仍够不到"的读数（含 τ 值与 pressure EMA 上界）。
- **E4（臂塌回来）**：J-N5f-3 四条件里任一条不满足 ⇒ 本件按"单臂"处理，`J-N5d-1` 仍记无读数，
  **不许**用"两臂七列相同"当稳健性证据（㊵-654 的形状）。
- **E5（可冻性只能靠动另外五道换来）**：某支面在其余五道取面内自述默认值时 `passes == false`，
  而把任一道阈下调后才 `passes == true` ⇒ 该 τ 判**不可冻**（G-N5f-7），第③段不起，
  并把"需要动第几道、下调到多少才够"作为**读数**登记，而不是作为**许可**。

## §7 待 owner 点（只有一格）

- **待批＝第①段跑面**（会更新权重、属扩样，PLAN-N3-08 件内已写明"跑面那一档仍等 owner"）。
  第②段零算力不需批；第③段的机时已被 #19 第①裁覆盖，但它的参数（基座、`--growth-min-pressure`
  现冻值、`--max-symbols 60000`）要在起跑前写进批文，不事后追改。

## §8 预期与收回声明（落笔即钉）

- 预期为绿的成员：**J-N5f-1**（仪器已按 PLAN-N3-08 §5bis 落地并烟测过）、**G-N5f-4／G-N5f-5**
  （跑道器现在就把它们当硬门，㊵-654 实测 rc=2 时四张面一步没跑）。
- 仍红时收回的句子并点名提交：若 **J-N5f-2** 红（`passes == false`），收回"通电有对象"这句，
  并点名是 ㊵-654 登记 DEBT-G84 的那笔提交引入的门把该形状暴露出来的（门没有错，臂是真的塌）。
  若 **J-N5f-3** 红，收回"N5 自进化唤醒已判"，且 09 §2 N5 节的开工前置维持未达成。
- 发表资格前置：任何"保持代价"的方向性结论，必须同时具备 ①双臂 `n5_shadow` 块在场且
  `missing_self_report_keys` 长度 == 0，②治疗臂 `shadow_branch_hits ≥ 1` 且控制臂
  `shadow_learn_hits == 0`／`gate == 0.0` 的对照，
  ③τ 来自同链同装配的 v2 面现冻值——三条缺一不发表。

## §3bis 第①②段实测结果（2026-10-10 ㊵-656；跑面批文＝owner 弹窗 #20 第①格"五支全跑"）

| 面 | `pressure_ema` max | 候选 τ（0.01 格） | 0.05 格对照 | 六道为真步数 | 最长连续段 | `gate_table` | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `beta_gate025` | 0.678104 | **0.64** | 0.65 | 26 | 4 | 3,999 checked／**0 mismatch** | `freezable` |
| `beta_gate100` | 0.680159 | **0.64** | 0.65 | 21 | 4 | 3,999／0 | `freezable` |
| `circuit_gate025` | 0.685877 | **0.66** | 0.65 | 33 | 7 | 3,999／0 | `freezable` |
| `circuit_gate100` | 0.680911 | **0.66** | 0.65 | 16 | 3 | 3,999／0 | `freezable` |
| `beta_guard_off`（默认关装配） | 0.422163 | 0.41 | — | **0** | 0 | 3,999／0 | `not_freezable_at_this_grid` |

- **J-N5f-1 达成**：五支面全部 `format = taiji-n3-pressure-face-v2`，`policy_self_reported` 六枚
  `minimum_*` ＋ `required_pressure_steps` ＋ `ema_rate` 齐，逐行闸表与自述一致（每支 3,999 行、0 失配）。
- **J-N5f-2 达成，且 τ 确实按链分开**：β 链 **0.64**、circuit 链 **0.66**（两档 gate 各自独立复算，
  同链两档给出同一个 τ，链间差 0.02 ＝ 2 个格宽）。这把 ㊵-505 那条"同一规则搬到 EMA 口径给 β 链 0.65
  会触发、circuit 链 0.70 不会"的**推得值**换成了**面内自述实测值**，也兑现 owner 2026-10-08 第六次弹窗
  "τ 按链分别给值"那句——此前它没有可执行形状，因为只有 v1 面（被本规则当场 rc=2 拒收）。
- **反向对照在场**：守卫臂（不带发育学习模式、bridge 未放行）的候选 τ 0.41 下六道为真步数 **0** ⇒
  同一台仪器在"两维恒零"的装配上判不可冻，在活装配上判可冻 ⇒ 本段的判别力不是恒真（[[guard-must-be-able-to-fail]]）。
- **读数的限度（先写在这里，别等 §9）**：这些 τ 全部冻在**出厂系冷底**上（`seed_beta.pt` 727,923 参数／
  `seed_beta_with_circuit.pt` 771,227 参数）。压强是预测失败的度量，所以"压强够得到闸"与"底子已经有能力"
  在同一条链上是有冲突的——本件第③段跑之前，必须先用 §9 的动态范围前置把这一点量成数，而不是跑完才发现
  保持侧七列全住在地板上。

## §9 保持侧这把尺的动态范围（第③段起跑**之前**先量，不事后补）

基线两面（同底 `seed_beta.pt`，零训练取得：`reports/taiji_n5_07_base_beta_cap0_20261010.json`
＋`reports/taiji_n5_07_base_beta_replay24_20261010.json`）现读：

| 列 | 基线值 | 有没有下行动态范围 |
| --- | --- | --- |
| `cap0:E`（简单指令与推理，严格真命中） | **3**／20 项机检（归一 0.15） | 有，但只有 3 项的粒度 |
| `replay_strict_hits:1.0` | **0**／24 | **没有**——已在地板上，跌无可跌 |
| `replay_strict_hits:2.0` | **0**／24 | **没有**——同上 |
| `replay_well_formed:0.0` | 34 | 有 |
| `replay_well_formed:0.5` | 51 | 有 |
| `replay_well_formed:1.0` | 54 | 有 |
| `replay_well_formed:2.0` | 60 | 有 |

- **七列里 5 列可判、2 列结构上不可判**（`strict_hits` 两档基线为 0）。⇒ 第③段出读数后，
  这两列的"没跌"**不许**写进"旧能力保持住了"的证据链，它们是地板不是证据
  （[[fake-zero-quantile-check-the-tail]]／[[gate-surface-criterion-resolution]] 同族）。
- 与 ㊵-654 那一对（with_circuit 底）相比，`cap0:E` 的粒度从 1 项升到 **3 项**——这是换底的**收益**：
  1 项粒度的跌曾被两次复核为"真实行为差而非噪声"，3 项至少让"跌 1 项"不再等于整维清零。
- 代价也要照实说：`strict_hits` 两列在冷底上归零，意味着**甲路对"复述是否仍能严格命中"这一维没有判别力**。
  若第③段只在 `well_formed` 四列出跌，结论只能写成"成句形状有代价、严格命中不可判"，
  **不许**写成"七列中五列跌、两列保持"。
- 本节的三条判定全部来自基线面自身，与第③段的 after 面无关 ⇒ 它在结果出来之前就已经把
  "什么算读数、什么不算"钉死了，这正是它要起的作用。
