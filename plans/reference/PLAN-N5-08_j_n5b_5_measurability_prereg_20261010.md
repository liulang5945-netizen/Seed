# PLAN-N5-08 预注册（2026-10-10）：给 `J-N5b-5` 定取数式与"噪声地板前置"（零算力，判据先冻不开判）

> 触发点：㊵-659 查明 `adjudicate_taiji_n5_shadow_gate.py:326` 把 `j_n5b_5` 写成硬编码字面量
> `"unverified_retention_lane_not_run"`，而保持侧跑道器已两次产出配对件 ⇒ 那句"未跑"为假（DEBT-G89）。
> 本件**不改** `J-N5b-5` 的判据本体（[PLAN-N5-02](PLAN-N5-02_shadow_utility_adjudication_prereg_20261009.md) §2 :37 原文照旧），
> 只补它缺的那一层：**从哪些键、按什么式子、在什么前置下**才允许出值。
> 台账：[08 ㊵-659](../../active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md)；债：DEBT-G89；
> 上游：[PLAN-N5-04](PLAN-N5-04_retention_rebase_prereg_20261009.md)、
> [PLAN-N5-07](PLAN-N5-07_proposal_reachability_prereg_20261010.md)＋[其判读](PLAN-N5-07_ADJUDICATION_20261010.md)。

## §1 为什么现在写它，而不是直接去改那一行

因为**数已经能算了，而我现在还不能发表它**：治疗臂对控制臂的逐列差（今日两枚配对件）是
`cap0:E` 2 对 1、`well_formed` 四档 19 对 15／32 对 35／34 对 42／36 对 32。
按 `J-N5b-5` 的字面（"相对对照臂下降为 0 才算保持"）那两档 32 对 35、34 对 42 就是"下降 3 与 8" ⇒ 判"未保持"。
但两臂的权重差今天分不清来自哪：影子学习，还是 60,000 步累积的浮点非确定性（㊵-656②、owner #21 第①裁的第三档尚未跑）。
**在这种状态下把 `j_n5b_5` 接上线，产出的就是一个把噪声读成效应的判据。** 所以顺序只能是：先冻结取数式与前置（本件），
再跑分岔档，最后才允许出值。直接把 `:326` 改成"去读配对件"＝无冻结取数式的改判据，本件明令禁止。

## §2 取数式（键名逐字，全从既有件读，不新造尺）

输入＝两份 `*_retention_pair.json`（`format: taiji-n2-04-retention-pair-v1`）：治疗臂与控制臂各一份，
且两份的 `cap0_before_path`／`replay_before_path` 必须**同一对基线件**（不同源 ⇒ 整件拒判）。
〔2026-10-10 ㊵-661 实施时收紧：只比路径名会把「同名不同字节」的两张面读成同源 ⇒ 本器改比配对件
自己出版的 `cap0_before_sha256`／`replay_before_sha256`，并同读两份件的 `guards.G_N2c_3_*` 结论（§4 G-N5g-1）。
摘要比对是**加严**不是放宽：路径不同源的件在旧写法下会假过。〕

- 逐列取 `per_column[列].after`，定义 `drop(col) = control.after − treated.after`（列集合＝判读器自己出版的
  `criteria.columns_compared` 那 7 枚，**不另列一份**）。
  〔2026-10-10 ㊵-661 现读纠正：**这句的键名写错了，且写错的方式很贵**——`criteria.columns_compared` 是**整数 7**
  （计数），列名住在 `per_column` 的键里。照原句实施第一次就在真件上抛 `TypeError: 'int' object is not iterable`。
  现行取法＝列名集合取 `sorted(per_column)`，再与 `criteria.columns_compared` **互相印证**（不相等 ⇒
  `column_count_mismatch`＋rc=2），所以"7 列"这句话仍是件里读的、不是我自己数的。判据本体未动。〕
- **地板列先剔除**：`drop` 与 `保持` 的判定只允许发生在 `baseline_headroom(col) = base.after 字段所在的该列基线值 > 0`
  的列上。基线值取该列的 `per_column[col].before`（两份件里同一个数），等于 0 的列一律记
  `not_judgable_floor` 并**成对出版**，不得进入分母（㊵-659 的实测形状：`replay_strict_hits:1.0`／`:2.0`
  两列基线即 0 ⇒ 若把它们算进"保持"，两个 0 会伪装成 2 项证据）。
- 判定分母＝`judgable = columns_compared − not_judgable_floor`，本件今日形状是 `7 − 2 = 5`。

## §3 判据（先冻，三条；每条都是数值合取式）

- **J-N5g-1（保持侧合取项本身）**：当且仅当对**全部** `judgable` 列都有 `drop(col) ≤ 0`，
  `j_n5b_5` 才允许填 `holds`；任一列 `drop(col) ≥ 1` ⇒ 填 `cost_persists`；
  `judgable < 3` ⇒ 填 `not_judgable_below_min`（列数不足时不出方向，今日 `judgable = 5` 不触发这一支）。
- **J-N5g-2（噪声地板前置，本件的新增部分）**：`j_n5b_5` 出 `holds` 或 `cost_persists` 的**硬前置**＝
  存在一支与双臂同码同参、不带 `--n5-shadow-gate` 的第三档（㊵-657④ 那条命令），且已出版
  `noise_floor = max over cols of |replicate.after − control.after|`。规则：任一 `|drop(col)| ≤ noise_floor` 的列
  必须记 `not_resolved` 并从判定分母里移出（**移出要红**：`judgable − not_resolved ≥ 3` 才允许出值，
  否则整枚填 `not_resolved_insufficient_range`）。前置未到位 ⇒ 本件明令 `j_n5b_5` 只能填
  `unverified_noise_floor_missing`，**不许**填方向。
- **J-N5g-3（两臂 verdict 不一致时不许挑轻的）**：若两臂 `verdict` 不同（一 `retention_holds` 一 `cost_persists`），
  按 PLAN-N5-04 §4 走 `arm_dependent`，本件不新增判级词、也不得取对自己有利的一臂（:60 原文沿用）。

## §4 守卫（四条，各自能为假）

- **G-N5g-1 同源不许重造**：两份件的 `G_N2c_3_cap0_same_source`／`G_N2c_3_replay_same_source` 必须都是
  `ok`（判读器已算，本件只读它的结论），任一非 `ok` ⇒ 整件 rc=2，`j_n5b_5` 保持 `unverified`。
- **G-N5g-2 地板列不许静默入分母**：`not_judgable_floor` 数组必须**成对出版**（列名＋该列基线值），
  缺该数组或数组长度 ≠ 基线为 0 的列数 ⇒ rc=2（反例支：给一列基线 0 却不出版 ⇒ 必须红）。
- **G-N5g-3 零假定**：本器**不得**自带任何阈值字面量。`line`（0.02）、`required_pressure_steps`、
  `noise_floor` 一律读自输入件；写侧出现阈值数字字面量 ⇒ 契约测红（沿用 `test_n3_06` 那族的 `HARDCODED_POLICY_LITERAL` 思路，不重造）。
  〔2026-10-10 ㊵-661 实施时把范围说清（这句按字面扫全器会当场红在我自己已有的东西上）：扫描测的范围＝
  本笔新增的 `judge_retention_lane`，允许的整数只有本件 §3 写死的两个计数（`judgable ≥ 3` 的 `3`、地板定义的 `0`，
  外加布尔语义的 `1`／`2`），且**不许有浮点**。全器扫的话 `:372` 的 `--line` 默认值 `0.02` 是命中项——
  它是 PLAN-N5-02／03 冻的过线界、可被旗标覆盖、不属本笔新增，**没被我搬进件里** ⇒ 另登记 **DEBT-G90**，
  不靠缩范围把它蒙过去。扫描测同时断言同一台仪器在 `main` 上抓得到 `0.02`，否则那两条断言是空扫。〕
- **G-N5g-4 前置未到位就是未判**：第三档不存在（或其件里 `measurement_complete` 非 true）时，
  `j_n5b_5` 只能是 `unverified_noise_floor_missing`，`j_n5b_6` 只能是 `not_adjudicable_*`；
  **不许**因为"数已经能算"就直接出 `holds`／`cost_persists`。

## §5 实施与预算

- 零算力：读四份既有 JSON（两份配对件＋两份 `progress_exit.json`）＋一份第三档件。
- 改动面＝`scripts/training/adjudicate_taiji_n5_shadow_gate.py`（把 `:326` 的字面量换成取数函数，
  新增 `--retention-treated`／`--retention-control`／`--noise-floor-arm` 三枚可选旗标；不给旗标时
  行为与今天**逐位相同**，仍出 `unverified_*`——这条兼容锚由测钉住）＋契约测。
- **零产品码**（`taiji/**`、`seed/**` 不动），零权重。

## §6 出口（三条互斥）

- 三条判据全部可判且 `j_n5b_5` 出了值 ⇒ 才允许把 `J-N5b-6` 从 `not_adjudicable` 挪进四元合取判定
  （其值仍由各判据决定，本件不预定方向）。
- `noise_floor ≥ max|drop|`（即所有列的臂间差都不超过同配置两档的漂移）⇒ 记
  `not_resolved_insufficient_range`，并收回"N5 保持侧门的方向已判"这句，转 09 §4 分流的"换暴露量"分支
  （暴露量档＝把每臂从 60,000 提到能让 `bands` 收窄的位置，另立一件）。
- 第三档跑出来与两臂逐位相同（即 `noise_floor = 0`）⇒ 前置自动满足，J-N5g-1 直接可判，
  且这句"确定性成立"要带"在 60,000 tick 上"一起说（不外推到别的暴露量与装配）。

## §7 落笔即钉的三行

- 预期为绿的成员：`G-N5g-1`（两份今天的件本来就 `ok`）、`G-N5g-2`（反例支在测里造）、
  以及"不给新旗标时逐位不变"那条兼容锚。
- 仍红时收回的句子并点名提交：若 `j_n5b_5` 停在 `unverified_noise_floor_missing`，
  收回"N5 合取已判"与"保持侧门有方向"两句，并点名是 ㊵-659 登记 DEBT-G89 的那笔提交（`1a692e739`）
  把占位符暴露出来的——占位符不是 bug 的全部，"能算"和"可发表"是两件事。
- 发表资格前置：任何 `holds`／`cost_persists` 的对外表述，必须同时给出
  ①`judgable` 的列名与 `not_judgable_floor` 的列名，②`noise_floor` 的数值与它来自哪一档，
  ③两臂 `verdict` 的各自抄录——三条缺一即视为未判。

## §8 实施与第一次读数（2026-10-10 ㊵-661，零算力、零产品码、零权重）

- 实施面＝`scripts/training/adjudicate_taiji_n5_shadow_gate.py` 新增 `judge_retention_lane()`／`_read_pair()`／
  `_same_source_failures()` 与三枚可选旗标；契约测＝`tests/taiji_native/test_n5_17_j_n5b_5_retention_lane_contract.py`
  **18 passed**（与既有判读器测合跑 33 passed）。每条分支都有正反例：地板剔除、`judgable < 3`、
  噪声档缺失／过大／不在同底／缺列、两臂 `verdict` 不一致走 `arm_dependent`、守卫非 `ok`、列数对不上（整数与列表两种）。
- **兼容锚已证**：不给三枚旗标时按 ㊵-659 那条命令重跑，输出与已封存的
  `reports/taiji_n5_07_shadow_gate_full_20261010.json` **逐字节相同**（`rc=1`、`j_n5b_5` 仍是那句占位原话、
  不出 `retention_lane` 块）⇒ 已入库读数没被本笔追认或改写。
- 第一次接上线的真件读数＝`reports/taiji_n5_08_jn5b5_first_read_20261010.json`：
  `j_n5b_5 = unverified_noise_floor_missing`，`not_judgable_floor`＝`replay_strict_hits:1.0`／`:2.0`（两列基线 0），
  `judgable = 5`，逐列差 `cap0:E −1`／`0.0 −4`／`0.5 +3`／`1.0 +8`／`2.0 −4`，`rc=1`。
  **这一枚不发表任何方向**（§7 前置②缺 `noise_floor`），它证明的是取数式在真件上能走通、且符号变号那三列是仪器算出来的。
- `holds` 那一支在测里造出来了（逐列差全 ≤ 0 ＋ 第三档逐位复现 ⇒ `noise_floor = 0`）——
  占位符时代它永远出不来，所以这条绿灯是**新能力**不是新结论。
- 本笔之后 `j_n5b_5` 距出值只差一件事＝同码同参第三档（㊵-657④ 那条命令，owner 终端）。
  `J-N5b-6` 的合取仍停在 `not_adjudicable_until_2_3_4_5_are_all_measured`：这一格里 `2`／`3`／`4` 已有读数
  （`shadow_learned`／`pairing_valid_on_quadruple`／`ruler_unusable`），`5` 现在是"前置未到位"而不是"没接上"。

## §9 第三档到位后的出值（2026-10-10 ㊵-663，判据与取数式一字未动）

- 第三档由 owner 按弹窗 #22② 跑完：418.55 秒、`exit_reason=max_symbols_reached`、`reached_budget=true`，
  训练器自述 `checkpoint_sha256=c5d0e7de5aa83fb3…` 与独立计算的文件摘要逐位一致。
- 它的配对件＝`reports/taiji_n5_08_determinism_pair_20261010.json`。**两件事必须如实记**：
  ① 第一次产件被 `adjudicate_taiji_n2_04_retention_pair.py` 以 rc=2 拒收（`status=separation_unverified`、
  不出版 `per_column`）——这道 fail-closed 是配对件自己的纪律，本器不绕过它；
  ② 该臂没有属于自己的夜间产出，判读时引用的是**控制臂那份**材料，属**显式声明的替换**
  （与 ㊵-656⑥ 的 `substituted` 同性质），不得读成"第三臂消费了这份材料"。
- **出值**＝`reports/taiji_n5_08_jn5b5_with_noise_floor_20261010.json`：`j_n5b_5 = cost_persists`、
  `noise_floor = 0`、`resolved` 五列、`not_resolved` 空、`rc=1`（来自 `j_n5b_4=ruler_unusable`）。
  §7 三条发表前置随之齐：为此本件把 `arm_verdicts` 升为**仪器的必需披露键**（前置③的机械化），
  取数式、地板列定义、`judgable ≥ 3`、rc 语义一字未改；不给三枚旗标时输出仍与 ㊵-659 封存件逐字节相同（重跑复核过）。
- §6 的三条出口在此**只落到第三支的一半**：`noise_floor = 0` 满足"第三档与两臂逐位相同"那一句，
  所以可说的是「**在 60,000 tick、这套装配、这条链上**保持侧的臂间差不小于测量地板（地板＝0，故逐列差全是可分辨的）」；
  不可说的是"确定性普遍成立"（未外推到别的暴露量／装配），也不可把 `cost_persists` 读成"门让能力变差"——
  它读的是"治疗臂相对对照臂未保持"，而收益侧（`J-N5b-4`）仍是分辨率陈述。
- 本件的债（DEBT-G89）随此结清。**同形第二枚占位符**在 `j_n5b_6`（`:518`）上暴露出来 ⇒ 另立 **DEBT-G93**，
  合取式须由 **PLAN-N5-09** 先冻再接（本件不替它下结论）。
