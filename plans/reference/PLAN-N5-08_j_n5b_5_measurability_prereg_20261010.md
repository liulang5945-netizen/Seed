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

- 逐列取 `per_column[列].after`，定义 `drop(col) = control.after − treated.after`（列集合＝判读器自己出版的
  `criteria.columns_compared` 那 7 枚，**不另列一份**）。
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
