# PLAN-N5-22 批次 C 两臂的**能力面**配对读数（J-N5u-1…4 先冻；零新机时，只跑评测不训练）

冻结时刻：2026-10-13（㊵-773）。**冻在任何一臂的 CAP 读数取到之前**——两臂终盘 `checkpoint.pt` 此刻一枚还没配齐
（对照臂 PID 25584 在飞，现取 06:37:23Z 处 `4,500,000/8,900,000`）。

执行体＝已在册仪器 `scripts/training/eval_taiji_cap0_baseline.py`（`taiji-cap0-baseline-v1`，
健康面 `taiji-cap0-health-v3`），评测集＝`plans/manifests/cap0_eval_set_v2.json`（`cap0-eval-set-v2`）。
本件不新增仪器、不动 `taiji/`、不跑训练——它冻的是**"两臂的能力差要满足什么才算读数成立"**。

## §0 现状（全部本会话现读，可复算）

1. 该仪器的件形（读自已入库的 `reports/taiji_n5_07_base_beta_cap0_20261010.json`）：
   `identity.{git_head, checkpoint_sha256, eval_set_sha256}`、`declared_mode`、`trained_during_eval`、
   `chain.{relax_legacy_guard, constrained_decode}`、`dimensions{A..H}`、`min_lines`、`scoring_discipline`（5 条）。
   每枚被驱动维度＝`{name, item_count, error, seconds, tally, items}`，
   `tally` 的键**逐字**是：`machine_scored_items`／`machine_scored_correct`／`machine_normalised`／
   `pending_human_review_items`／`untested_items`。
2. 那份件里的**实测形状**（β 基座、同一评测集）：C 机检 14 题／对 0，D 16／1，E 20／3，
   **B 与 G 的 `machine_scored_items` 都是 0**（20 题全在 `pending_human_review_items`），
   A／F／H 是 `status="not_executed"`（`note` 原文："本 runner 未覆盖该维度；按 07 §5 记 not_executed，不记 0 也不记通过"）。
   ⇒ **这台仪器能机检配对的维度只有 C／D／E**；B／G 要人工复核（＝owner 侧），A／F／H 不在本件射程。
3. 单趟耗时形状：B 26.26s、D 76.98s、E 30.75s（件里 `seconds` 逐维出版）⇒ 五维一趟分钟级，
   两臂 ×2 仍是零**训练**机时，不需要 owner 排窗；但它是 CPU 活 ⇒ 与长跑臂并发会污染两臂自取速率（J-N5r-3 的定价来源），
   所以取数排在对照臂退出之后。
4. 已知口径陷阱（不是本件新发现，引用要带）：`cap-strict-hit-denominator` 那条记录着
   "严格复制口径下 E 维可命中数恒 0"，而**这台仪器的 E 维在 v2 评测集上机检 20 题、对 3**——两者不是同一把尺
   （前者是 `score_taiji_r2_copy_strict_cap.py` 的"期望词须在更早用户轮逐字出现"）。
   ⇒ 本件绝不用绝对分作跨档比较，只用**同分母逐题翻转**（见 J-N5u-3）。

## §1 取数命令（点名为两臂，禁用默认写靶）

```
python scripts/training/eval_taiji_cap0_baseline.py --checkpoint output/n5_20_batchC_treated/A/checkpoint.pt --report reports/taiji_n5_22_cap0_batch_c_treated_20261013.json
python scripts/training/eval_taiji_cap0_baseline.py --checkpoint output/n5_20_batchC_control/A/checkpoint.pt  --report reports/taiji_n5_22_cap0_batch_c_control_20261013.json
```
两趟都**不传** `--relax-legacy-guard`／`--constrained-decode`／`--copy-circuit`（链路默认，`chain` 两枚键都应为 false），
`--health` 与 `--f-live-evidence` 本件不开（A／F／H 不在射程，开了就把"未测"写成"跑了但没结果"）。

## §2 判据（冻四条，每条写明"什么情况下它为假"）

* **J-N5u-1 配对有效性**：两枚件的 `identity.eval_set_sha256` 必须逐字相等、`identity.git_head` 必须相等、
  `chain` 两枚键必须相等、`trained_during_eval` 必须两枚都是 false、被驱动的每枚维度 `error` 必须是 `null`。
  为假的情形＝任一项不等 ⇒ **两臂差不可配对**，本件全部结论降级为"两份各自的能力读数"，
  并且必须先查是哪一个键不齐（评测集被改过／中途有人提交换过代码面／链路旗标不一致），不许用"差得不大"混过去。
* **J-N5u-2 每枚维度先过动态范围**：一枚维度进入判决的条件是**两臂的 `machine_scored_items` 都 > 0 且相等**
  （同分母）。按 §0 第 2 条的形状，**预期 B／G 出局**（机检数 0，全在待人工复核），
  C／D／E 预期入场；若某维两臂分母不等，该维出局并把不等出版——
  分母不等时比"对题数"是把两把尺相减（[[aggregate-column-two-questions-before-use]]）。
* **J-N5u-3 判定只看逐题翻转，不看绝对分**：对每枚入场维度，按 `items` 里同一题号逐一配对，
  统计 `flips_treated_only`（治疗臂对、对照臂错）与 `flips_control_only`（反向）。四条**互斥**分支：
  `capability_gain`＝存在某维 `flips_treated_only − flips_control_only ≥ 3` **且** 无任何维 `≤ −3`；
  `capability_loss`＝镜像（某维 `flips_control_only − flips_treated_only ≥ 3` 且无任何维 `≥ +3`）；
  `not_resolved`＝两方向都不到 3；
  `no_eligible_dimension`＝J-N5u-2 把 C／D／E 全部挡在门外（那时**只许**说"本面无机检配对的维度"）。
  为假／为危险的情形＝有维同时命中 gain 与 loss 两支 ⇒ 那不是"挑一支",而是**方向不一致**，
  按原文记 `direction_inconsistent` 并整档不判（区间互斥要预先想清楚，见 [[frozen-criterion-stays-when-secondary-moves]] 第 1 条）。
  **引用单位不可拆**：任何 `capability_gain`／`capability_loss` 句必须与
  ①`J-N5r-4`（配置面只差一枚旗标，并排仪判）＋②`J-N5t-1`（两臂同流，流同界仪判）**一起**引用；
  单独引能力句就是把它交给那两个未锁住的变量（[[arm-moves-two-variables-price-the-free-middle]]、
  [[judge-a-training-change-on-its-own-face]]）。
* **J-N5u-4 幅度与机制都另需前置**：本件最多产出"方向"（哪一维翻了几题）。
  ①**幅度**句（含百分数、"涨了几成"）需要同配置重复臂（噪声地板，要新机时＝owner，＝`J-N5t-4` 同一前置）；
  ②**机制归属**句需要 `DEBT-G131` 那台集合差仪器出数（`PLAN-N5-21` §6.1 第 2 条）；
  ③ B／G 的**人工复核**是 owner 侧动作，不授权我用 `pending_human_review_items` 的数量去推任何能力句
  （"在场≠过门"，[[cap-gate-no-proxy-lighting]]）。
  为假的情形＝以上任一前置未满足就发表对应句子——这一支不需要读数就能判违规，所以它写在最前面也有效。

## §3 判据载体在哪（措辞门要求的一节）

* J-N5u-1／J-N5u-2＝**新配对器**（本件的实施格）`scripts/training/audit_taiji_n5_22_cap0_pairing.py`，
  契约册 `tests/taiji_native/test_n5_36_cap0_pairing_contract.py`（号现取：`test_n5_*` 用到 35）；
  两枚真件在配对器里只被读、不被重算（不重抄评测链，[[dont-reimplement-the-chain]]）。
* J-N5u-3 的逐题翻转需要 `items` 的**题号与对错标记**——现读那份件的 `items` 是 20 元素列表，
  元素字段名**必须**在实施时用真件现取（本件刻意不写死键名：
  按另一台仪器的输出形状套键名，本仓错过两次，代价见 [[frozen-criterion-stays-when-secondary-moves]] 末两条）。
* 读数件＝`reports/taiji_n5_22_cap0_batch_c_pairing_20261013.json`。

## §4 半径

`plans/`（本件）→ 实施格＝`scripts/training/` 一枚新配对器＋`tests/` 一册＋两枚评测读数件与一枚配对读数件；
不动 `taiji/`、不动跑器、不起新训练。评测在对照臂退出**之前不许开跑**（§0 第 3 条的争用理由）。

## §5 落笔即钉的三行

* **预期绿**：`J-N5u-1`（同一评测集、同一 HEAD、默认链路、`error` 全 null）与 `J-N5u-2` 的入场集＝{C, D, E}；
  `J-N5u-3` **不预判方向**——三臂历史上第一次真喂 generated 的能力后果未知，
  `not_resolved` 是完全可能的合法出口（读平就记读平，不许换成 B／G 的人工面去"找东西说"，
  [[frozen-criterion-stays-when-secondary-moves]]）。
* **仍红时收回哪句并点名**：`J-N5u-1` 红 ⇒ 当场收回"两臂能力可比"，并点名失配的键；
  `J-N5u-2` 只剩 0 枚入场维 ⇒ 收回全部能力句，改记"本面无机检配对"，并把它升格成一条新债（尺的覆盖面问题）；
  `J-N5u-3` 落到 `direction_inconsistent` ⇒ 收回 gain／loss 两句，只出版逐维翻转表。
* **发表资格前置**：`J-N5u-1`＋`J-N5u-2`＋`J-N5u-3` 三支齐绿之前，「批次 C 让能力变了」这句不许发表；
  齐绿之后只授权**方向**，且必须与 `J-N5r-4`／`J-N5t-1` 同引；幅度句与机制句分别被 `J-N5u-4` 的①②挡着。


## §3.1 载体键名现取（同轮，㊵-773；这不是判据变更，是把 §3 留白那格填上）

按"键名必须在真件上现取"的规矩（本文件主旨的最后两条），我读了已入库的
`reports/taiji_n5_07_base_beta_cap0_20261010.json`，
`dimensions.D.items[]` 的键**逐字**是：

```
id, family, load_ok, machine_precheck, matched, expected_contains,
pending_human_review, prompt_echo_stripped, raw_last_output, score, tick, turns, verdict_text
```

题号形如 `D01`…`D20`（`len(items)`＝20＝`item_count`），机检对错住在 **`score`**（0/1），
是否待人工复核住在 **`pending_human_review`**（bool），装载是否成功住在 **`load_ok`**（bool）。

⇒ 配对器（`audit_taiji_n5_22_cap0_pairing.py`）的入场规则据此钉死为三句，都是**能红**的：

1. 一题进入配对必须**两臂同 `id` 都在**、两臂 `load_ok` 都为真、两臂 `pending_human_review` 都为假。
   任一句不成立 ⇒ 该题记 `excluded_ids` 并出版原因；**不**把它当"0 分"参与翻转（那是把没测到写成测到，
   [[probe-output-must-be-verified-present]]）。
2. 某维入场题数为 0 ⇒ 该维不参与 `direction`，出版 `no_items_in_dimension`；全部入场维都为 0 ⇒
   `direction="no_eligible_dimension"`，本档只出"本面无机检配对"这一句。
3. 两臂 `scoring_discipline`（5 条）必须逐字相等——它是测面的定义的一部分；
   不等＝两臂用的不是同一把尺，按 `J-N5u-1` 失配处理（配对的资格先没，谈不上方向）。
