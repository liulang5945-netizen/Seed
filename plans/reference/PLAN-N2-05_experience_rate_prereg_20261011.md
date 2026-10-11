# PLAN-N2-05 每单位经验的增益（速率本体读数）——判据先冻、零算力、不动权重（2026-10-11）

## §0 结的是哪一条

`09_NEXT_MAINLINE_PLAN.md` §2 **N2 出口交付**原文要两样：①一次真实长跑＋离线巩固的完整读数（已由
[PLAN-N2-02 判读](PLAN-N2-02_ADJUDICATION_20261008.md) 交掉：三条合取＝持久化成立／`J-N2a′` 不成立／
`J-N2b′` 不成立）；②**能力增量／单位经验（速率本体读数）**——这一样到今天**没有成列**：分子分母都在
盘上，缺的是一个先冻住的除法与它的符号限定。㊵-676 的完成度对照把它列为"未兑现-可自办"，本件做掉它。

**本件零算力、不改权重、不动产品码**：只做一次已封存读数的除法，并按判据把话说死。

## §1 分子与分母（逐字取自同一枚件的同一次跑）

件＝`reports/taiji_n2b_powerup_20261008.json`（第二次通电，按 PLAN-N2-02 冻结装配跑的对照双臂）：

* 分子 `j_n2b_prime_delta_quality = −1.1086209986387523`，它自己等于
  `material_treated_window.quality_mean − material_control_window.quality_mean`
  （−14.173620701828504 对 −13.064999703189752，本件复算并断言逐位相符）；
* 分母候选三枚，**必须并排出版**：`organs.stats.cycles = 8.0`、`organs.max_symbols = 64`、
  `organs.texts = 8`。本档 `cycles == texts`，原因是批文里 `cycles_per_text = 1` ⇒
  **这是剂量形状的巧合，不是两条独立分母**，必须连着这句一起出版。

## §2 判据（冻；三条，都是数值式，不新增任何阈值字面量）

* **J-N2e-1 速率可算**：`rate_cycles = Δquality / cycles`、`rate_symbols = Δquality / max_symbols`、
  `rate_texts = Δquality / texts` 三枚必须同时出版；分母 ≤ 0 ⇒ 记 `not_computable_nonpositive_dose`
  并 rc=2，**不许出现除零后的小数**。
* **J-N2e-2 分母不许事后挑**：三枚速率必须并列出版且**不得只发表其中最好看的一枚**；若某两枚
  分母本档相等（此例 `cycles == texts`），必须出版 `degenerate_denominators` 点名它们。
* **J-N2e-3 符号限定（这一条是本件的主要内容）**：分子为负 ⇒ 这一列只能以
  **"每单位经验的代价"**出版，**不许**写成"能力增益速率"；并且分子所在的那条判据
  `J-N2b′` 已判不成立 ⇒ **速率不改变该判定，也不得被用来把它救成"其实有方向"**。

## §3 守卫（三条，各自能为假）

* **G-N2e-1 同源**：分子与三枚分母必须来自**同一枚件**；跨件搬分母（例如拿 `window_bytes` 顶替
  `organs.max_symbols`）即判 `mixed_sources` 并拒绝出版。
* **G-N2e-2 复算分子**：`Δquality` 必须能在件内由两臂 `quality_mean` 现减复算，逐位不符 ⇒ rc=2。
* **G-N2e-3 剂量自述在场**：`organs.texts`／`organs.cycles_per_text`／`organs.max_symbols` 任一缺失
  ⇒ 记 `dose_self_report_absent`，不得由命令行反推（DEBT-G70 那一族的老雷）。

## §4 判据载体在哪（措辞门要求的一节）

执行体＝新仪 `scripts/training/summarize_taiji_n2_experience_rate.py`（读数件
`reports/taiji_n2_05_experience_rate_20261011.json`）。它**不做任何统计推断**：只读那枚件、复算减法、
做三个除法、按 §2/§3 出版判定；契约测
`tests/taiji_native/test_n2_05_experience_rate_contract.py` 同时钉"正常档三枚并列"与"分母为 0 必须
响亮拒绝"，两档都要能为假。本件不重抄任何生成链（分子来自既有判读器已出版的键）。

## §5 出口（三条互斥）

* 三枚速率出版 ∧ `J-N2e-3` 的限定语随读数一起写 ⇒ N2 出口交付第②样**兑现**（负方向），
  完成度对照里那一格从"未兑现"改为"已判定＝每单位经验是代价"；
* 若出版时漏掉符号限定（只报数不报向）⇒ 本件视为未兑现，收回"N2 的速率读数已交付"这句；
* 若 `G-N2e-2` 复算不符 ⇒ 件内两臂 `quality_mean` 与分子不同源，本件不开口，转登债。

## §6 落笔即钉的三行

* **预期为绿的成员**：`J-N2e-1`（分母全为正）、`G-N2e-1`／`G-N2e-2`（同一枚件、减法可复算）。
* **仍红时收回的句子并点名**：任一守卫红 ⇒ 收回"每单位经验的代价已成列"，点名是本件的
  同源/复算守卫生效，而不是 N2 的判定变了。
* **发表资格前置**：对外写这枚速率必须同时给 ①三枚分母版本、②`degenerate_denominators` 的点名、
  ③"分子为负 ⇒ 这是代价不是增益"的限定语与它所属的 `J-N2b′ 不成立` 判定；三条缺一即视为未交付。
