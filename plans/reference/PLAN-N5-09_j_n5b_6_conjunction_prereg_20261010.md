# PLAN-N5-09 预注册（2026-10-10）：冻 `J-N5b-6` 四元合取的取数式（零算力，判据先冻不开判）

> 触发点：㊵-663 查明 `adjudicate_taiji_n5_shadow_gate.py:518` 把 `j_n5b_6` 写成字面量
> `"not_adjudicable_until_2_3_4_5_are_all_measured"`，而四支里 2／3／5 今天都有值、4 有分辨率陈述
> ⇒ 那句"尚未全部测得"不再反映测况（DEBT-G93，与已结清的 DEBT-G89 同形）。
> 本件**不改** `J-N5b-6` 的判据本体（[PLAN-N5-02](PLAN-N5-02_shadow_utility_adjudication_prereg_20261009.md) §2 :39-41 原文照旧），
> 只补它缺的那一层：**四支各自的取值如何归成三态、合取在什么条件下取哪个字面量**。
> 台账：[08 ㊵-663](../../active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md)；债：DEBT-G93；已接的前件：PLAN-N5-08。

## §1 为什么现在写它，而不是直接把 `:518` 改成条件式

因为**合取的语义还没定**，而它决定 N5 能不能收口。PLAN-N5-02 §2 :40 只说"缺任一支 ⇒ 明确写『不成立／未判』并点名缺哪支"，
这句话里藏着三件没冻的事：(i) 「不成立」与「未判」用哪个字面量、二者何时互换；(ii) `ruler_unusable` 属于哪一态；
(iii) "点名缺哪支"要不要进件。先改码就是把这三件留给一次已经花过钱的颜色不同的读数去解释——
DEBT-G89 的教训正是"能算"与"可发表"是两件事（㊵-660），本件因此先冻后接。

## §2 三态归类（逐支点名取值；全部读自既有键，不新造尺、不新造判级词）

- **J-N5b-2**（通电且学到）：`shadow_learned` ⇒ **established**；`shadow_inert`／`not_powered` ⇒ **not_established**；
  `unverified_missing_face`／`ran_not_measured` ⇒ **unmeasured**。
- **J-N5b-3**（配对成立）：`valid` ⇒ **established**；`invalid` ⇒ **not_established**；`unverified` ⇒ **unmeasured**。
- **J-N5b-4**（收益侧过线）：`holds` ⇒ **established**；`not_holds` ⇒ **not_established**；
  **`ruler_unusable` ⇒ not_established**（依据 PLAN-N5-02 §4 :52 原文"结论只能是『这把尺答不了这个问题』"——
  它限制的是**结论**，不是把这一支升成"已成立"；把它记成 unmeasured 才是发明，故明令禁止）；
  `unverified_*` ⇒ **unmeasured**。
- **J-N5b-5**（保持侧）：`holds` ⇒ **established**；`cost_persists` ⇒ **not_established**；
  `arm_dependent` ⇒ **not_established**（PLAN-N5-04 §4 的判级词，它表达的是"两臂不同结论"，不是"没测"）；
  `unverified_*`／`not_judgable_below_min`／`not_resolved_insufficient_range` ⇒ **unmeasured**
  （后两者是"这一格这次判不了"，属前置未到位，不是阴性结果）；
  结构失败那一族（`faces_not_same_source`／`column_sets_differ`／`baseline_differs`／`column_count_mismatch`／
  `noise_floor_columns_incomplete`／`unverified_same_source_guard_not_ok`）⇒ **unmeasured**，且它们已由取数式抬到 rc=2，
  合取不许把它们读成"阴性结果"。

- **G-N5h-5 未列出的值不许有默认归类**：四支里任何一枚取到本件 §2 **未列举**的字面量 ⇒ `j_n5b_6 = unverified`
  且 `unmeasured` 数组里点名那支＋本器 rc 抬到 2（**白名单式，不写 else 分支兜底**）。
  理由：默认归类就是拿一条没冻过的规则去决定 N5 收不收口——那正是本件要避免的事的第二种形态。

守卫（每条都要能为假）：
- **G-N5h-1 不许新造判级词**：合取的输出字面量只能取自 `established`／`not_established_missing=[…]`／`unverified_missing=[…]`
  三形，且「不成立／未判」两个词沿用 PLAN-N5-02 §2 :40 原文，不得替换成"部分成立""有条件成立"一类新词。
- **G-N5h-2 点名必附**：凡 `not_established` 或 `unmeasured` 的成员都要进件（`missing`／`unmeasured` 两个数组），
  空数组时键仍在（缺键与空数组是两件事，前者是仪器坏）。
- **G-N5h-3 未判不许被算成不成立，也不许反过来**：两支中任一为 `unmeasured` ⇒ 合取走 `unverified_missing`，
  即便另有支已经 `not_established`（**这条是本件最容易被写错的一格**：混合态下若走 `not_established`，
  等于把"还没测"混进"测了且否证"，那正是 DEBT-G89 里那句假陈述的成因）。
- **G-N5h-4 兼容锚（本笔起草时自纠过一次形状错，原文留此为证）**：我起初把它写成"不接线时输出与已封存件逐字节相同"——
  合取一旦接上就**不存在**"不接线"那条分支，那句话会是一条**永远不为假**的守卫（比没守卫更坏）。
  现行形状改成人能复核、也能为假的三条：(a) 四枚成员键 `j_n5b_2`／`j_n5b_3`／`j_n5b_4`／`j_n5b_5` 的取值与接线前**逐位相同**
  （测里用同一组输入跑接线前后两版对照）；(b) 已入库的封存件**不被本笔重生成**（文件摘要不变，测里钉
  `reports/taiji_n5_08_jn5b5_with_noise_floor_20261010.json` 的 sha 与本笔开始时现读一致）；
  (c) 新增键只加不改：`j_n5b_6` 之外新增 `conjunction` 块，既有顶层键名一个不删。

## §3 判据（三条，数值合取式；先冻不开判）

- **J-N5h-1**：当且仅当四支全 **established** ⇒ `j_n5b_6 = established`，此时才允许写 PLAN-N5-02 §2 :39 那句"影子对该母量有贡献"。
- **J-N5h-2**：无 `unmeasured` 且至少一支 **not_established** ⇒ `j_n5b_6 = not_established`，
  件里必须给 `missing` 数组；今日按 §2 归类的预期成员＝`J-N5b-4`（`ruler_unusable`）与 `J-N5b-5`（`cost_persists`）。
- **J-N5h-3**：存在 `unmeasured` ⇒ `j_n5b_6 = unverified`，件里给 `unmeasured` 数组，
  且**不得**同时给 `missing`（两词共现＝这句话在同时说两件互斥的事）。

## §4 出口（三条互斥）

- 四支全 established ⇒ 允许发表"有贡献"，并转 09 §1.2 五级演进第①级；本件不预定这一支会不会发生。
- `not_established`（含今日预期的那一形）⇒ 按 PLAN-N5-02 §2 :38 "收益与代价要同时发表"出版两半，
  N5 的出口交付（09 §2 N5 ④"重启条件即在此满足**或证伪**"）以**证伪**这一路结清，并转 §4 分流表归因。
- `unverified` ⇒ N5 不收口，且必须在 03/09 写明缺的是哪一支的前置；**不许**用 `J-N5d-*` 或 CAP 分区的旁证顶替（PLAN-N5-02 §2 :41）。

## §5 实施与预算

- 零算力、零产品码、零权重：改动面＝`scripts/training/adjudicate_taiji_n5_shadow_gate.py`
  （把 `:518` 的字面量换成 `judge_conjunction()`，输入是本器自己已经算好的四枚键）＋契约测。
- 测必须双向：混合态走 `unverified`（G-N5h-3 的反例支）、四支全真走 `established`、
  含 `ruler_unusable` 但不含 `unmeasured` 走 `not_established`，以及兼容锚（不接线逐字节不变）。
- 今日两臂的读数已齐 ⇒ 本件落地后**同一趟**即可出版 `j_n5b_6` 的第一个真值，不需要新跑权重。

## §6 落笔即钉的三行

- 预期为绿的成员：`G-N5h-2`（两个数组都在）、`G-N5h-4`（兼容锚）、`J-N5h-2`（今日数据应给 `not_established`，
  `missing` 恰为 `["J-N5b-4", "J-N5b-5"]`）。
- 仍红时收回的句子并点名提交：若 `j_n5b_6` 落不进三形之一（例如 `ruler_unusable` 被归类测出别的态），
  收回"N5 合取已有值"与"保持侧代价已发表"两句，并点名本笔实施提交；DEBT-G93 保持未修。
- 发表资格前置：任何 `j_n5b_6` 的对外表述必须同时给出 ①四支各自原值（不翻译）、②归类后的三态、
  ③`missing`／`unmeasured` 数组原文——三条缺一即视为未判，且**不许**把 `unverified` 念成"实验失败"。

## §7 实施与第一次读数（2026-10-10 ㊵-665，零算力、零产品码、零权重）

- 实施＝`judge_conjunction()`／`classify_branch()`／白名单 `BRANCH_VALUE_CLASS`，`:518` 的字面量换成三形之一；
  判级词全部取自 PLAN-N5-02 §2 :40 原文（「不成立／未判」），**未新造词**（G-N5h-1 成立）。
- **第一次读数**＝`reports/taiji_n5_09_j_n5b_6_first_read_20261010.json`：
  `j_n5b_6 = not_established`、`missing = [J-N5b-4, J-N5b-5]`、`unmeasured = []`、`unknown_values = []`、`rc = 1`。
  这与本件 §3 J-N5h-2 与 §6 在**跑之前就写死**的预期成员逐字相符 ⇒ 属于验证，不是把结论搬来对齐数据。
- 四枚成员原值（不翻译）＝`J-N5b-2 shadow_learned`／`J-N5b-3 valid`／`J-N5b-4 ruler_unusable`／`J-N5b-5 cost_persists`；
  三态＝`established`／`established`／`not_established`／`not_established`（§6 前置①随件出版）。
- 契约测＝`tests/taiji_native/test_n5_19_j_n5b_6_conjunction_contract.py` **14 passed**，其中三形各正反例、
  混合态走 `unverified`（G-N5h-3 反例支）、白名单外的值不静默归类（G-N5h-5）、缺键记 `None` 不猜零、
  旧封存件 sha 钉死（G-N5h-4b）。定向七文件合跑 **94 passed**。
- **两处既有测的期望值随新值集重推**（`test_n5_07::test_missing_metric_file…` 与
  `test_n5_17::test_no_retention_flags…`）：原断言"缺任一支不许给出有贡献"一字未减，
  改成钉 `unverified` **且**要求 `unmeasured` 点名缺哪支——比字面量断言更强。
  过程中我自己先按想象中的输出写了一次 `unmeasured`，跑出来才重推成"3 无面／4 无件／5 无旗标各自都算未测"
  （[[feedback-recompute-dont-hand-carry]] 的又一次实例，记在这里而不是悄悄改掉）。
- **本件的连带影响**：PLAN-N5-08 §5 那条"不给旗标时逐位相同"的兼容锚自此结构上不可能成立，
  已在那件里带日期就地降级为键级三条（原文不删）。
- §4 三条出口落到**第二条**：`not_established` ⇒ 按 PLAN-N5-02 §2 :38 同时发表收益与代价两半，
  09 §2 N5 出口"重启条件在此满足**或证伪**"走证伪这一路；第一条（established）未发生，
  第三条（unverified）今日不触发（`unmeasured` 为空）。
