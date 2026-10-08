# PLAN-N3-08 · 把"六道阈＋五支 EMA＋拒绝原因"写进压强面自述（DEBT-G53 修法②，2026-10-08 起草，**判据先冻、不开跑**）

> **来源与效力**：[PLAN-N3-07](PLAN-N3-07_tau_ema_caliber_definition_20261008.md) §5 末句——"六道分项闸的其余五道阈＋`ema_rate` **今天没有任何面自述过** ⇒ **修法② 不落地，τ 就没有'面内自述'的发表资格**"；
> 以及 [PLAN-N3-06 判读](PLAN-N3-06_ADJUDICATION_20261008.md) §5 那条"重放读法只证'这批在库面上能定位'，不证'未来的面自带定位'"。
> 本件改的是**仪器的记录面**（`train_seed_corpus.py` 的 `--pressure-record` 那一段），**不改产品内核、不改任何常量、不放行生长**。
> 成熟度＝前置补齐件（不是判据实验）：它的产出是"下一张面能不能被自己解释"，不产出能力结论。

## 0. 为什么必须做（本轮量出来的两个缺口）

1. **自述缺口**：face 头只有三道阈（`train_seed_corpus.py:535-539`＝`minimum_pressure`／`required_pressure_steps`／`growth_resource_cost`），
   另外 **四道 `minimum_*` ＋ `ema_rate` ＋ 五支 EMA 的初值**在件里取不到 ⇒ PLAN-N3-06 只能"取产品默认"并逐条标
   `assumed_from_product_defaults`（六条）。**假设一旦为假，整件指认就作废**，而今天没有任何读数能证明它不为假。
2. **口径缺口**：面每行记的 `pressure` 是**原始加权和**，而闸比的是 `pressure_ema`（`adaptive_residual_growth.py:388-395`、:437-444）⇒
   面自己**说不出**六道闸各自过没过，"是哪一道在拦"只能靠外部重放回答。DEBT-G53 的正式修法是把这条口径差**关进件内**。

## 1. 实施形状（先写死，跑前不解释）

* **写侧**（`train_seed_corpus.py:528-553` 的 face 行、`:556-567` 的 `_record_pressure`）：
  * face 头的 `policy` 段补齐**六道 `minimum_*`** ＋ `required_pressure_steps` ＋ `growth_resource_cost` ＋ **`ema_rate`**，
    并新增 `ema_initial` 段自述五支 EMA 的初值（四支 `0.0`、`resource_state_ema` 为 `1.0`，锚 :359-363）；
    **全部读自 `trigger.policy` 与产品类，仪器内不许出现任何一个抄写的阈值字面量**；
  * 每行 `kind=pressure` 补记产品 decision 自带的十个键：`pressure`（＝`pressure_ema`）、`residual_error_ema`、
    `fast_slow_conflict_ema`、`activity_saturation_ema`、`utility_gap_ema`、`resource_state_ema`、
    `consecutive_pressure_steps`、`structural_budget`、`resource_cost`、`reasons`——**只从 `observe` 的返回值读，零重算**；
  * 格式名升 `taiji-n3-pressure-face-v2`（旧 v1 名保留给在库四面）。
* **读侧**（两支都要同时认 v1/v2，且新增一条"版本集与写侧一致"的机检）：
  `count_taiji_n3_pressure_thresholds.py:73`（face 头解析）与 `replay_taiji_n3_06_gate_attribution.py:88`；
  夹具 `tests/taiji_native/test_n3_04_developmental_flags_contract.py:230` 同步 v1/v2 两支。
* **不动**：`AdaptiveResidualGrowthPolicy` 任何常量、`AdaptiveResidualGrowthTrigger.observe`、`reset_dynamics`、
  默认位、产品件写靶（面一律落 `output/`）。

## 2. 判据（先冻；两条都是"面自己能不能解释自己"，不是能力结论）

* **J-N3c-自述**（主判据，单值式）：在新面上跑 PLAN-N3-06 那台重放仪，输出里
  `assumed_from_product_defaults` 的长度必须 **== 0**（今天四张面都是 **6**）
  且 `per_gate.*.threshold_self_reported_in_face` 为真的道数必须 **== 6**（今天是 **1**）。
  两条各自能为假；任一不符 ⇒ 判"修法② 未达成"，**不许**用"反正重放也能算出来"代答。
* **J-N3c-兼容锚**：一枚**在库 v1 面**（`output/n3_04/beta_gate025/pressure.jsonl`）在改前与改后的读侧代码下，
  判读件必须**逐位不变**（`cmp` 无差）。这条防的是升版把四张已判读面变成不可复算——
  正是 DEBT-G35 那一族（对的路径、找不回的字节）在仪器侧的形状。

## 3. 守卫（五条，任一不符即整件不判）

* **G-N3c-1 默认关逐位不变**：不开 `--pressure-record` 的臂，进度行八个既有键（`ticks`／`window_ticks`／`online_accuracy`／
  `mean_surprise`／`holdout_surprise`／`base_ticks`／`ticks_at_exit`／`reached_budget`）与改前同参臂**逐位同**。
  **本条不许继承任何他旗标的证明**（㊵-496 的既有纪律），要实测。
* **G-N3c-2 旧列取值不动**：同一枚面上、同一 tick，新面与旧面**共有的原始六列**（五个信号＋`pressure`）必须逐位同——
  新键只能**加**，改一个旧键的取值＝换尺。
* **G-N3c-3 零抄写**：写侧新增的每个阈值/初值都必须来自 `trigger.policy` 或产品类的读取；
  契约测里加一条**源码守卫**——`train_seed_corpus.py` 的压强段内不得出现字面量 `0.7`／`0.55`／`0.4`／`0.35`／`0.25`（`ema_rate` 的那个 0.25 只能来自 `policy`）
  与 `0.30/0.20` 权重（加权式归产品所有）。守卫要能为假：插一个写死的阈值进临时副本必须让测红。
* **G-N3c-4 版本集机检一致**：写侧输出的格式名必须落进读侧接受的集合，且集合里每个成员都有对应夹具；
  只在一侧加版本 ⇒ 响亮失败（DEBT-G53 的根因之一就是"读侧写侧各自演进"）。
* **G-N3c-5 缺披露即拒判**：v2 面若缺 `reasons` 或任一支 EMA ⇒ 重放仪 rc=2 并点名缺哪个键
  （**不是**回落到"取产品默认"——回落会静默重建本件要消灭的那个假设）。

## 4. 预算（分两档，第二档另批）

| 档 | 内容 | 代价 | 批文 |
|---|---|---|---|
| 本件现在做 | 写侧加披露＋读侧双版本＋G-N3c-3/4/5 的契约测＋**一支 `--smoke` 面**（落 `output/`，验旗标被走到、面能自述） | 零算力（秒级冒烟），不训练不写产品件 | 属仪器改动，随本件自办 |
| 之后另批 | 4 张在线面 ＋1 支默认关守卫臂（形状照 PLAN-N3-04 §4bis，出厂链两枚 × gate 0.25/1.0） | 每支 ≈46 秒、每枚档 ≈14.7 MB；**会更新权重** | ⇒ **回 owner**（这是扩样，不在既有批文内） |

## 5. 出口（三条互斥，先点名）

* **J-N3c-自述 与 J-N3c-兼容锚 都成立** ⇒ DEBT-G53 修法② 结清，τ 才取得"面内自述"的发表资格；
  下一格＝按 PLAN-N3-07 §4 在两链各冻一份 τ（先解决格宽那条粒度缺陷，**动常量须 owner 签字**）。
* **自述成立但兼容锚红** ⇒ 本件**不出版**，先修读侧版本兼容（升版把旧面读坏＝仪器事故，不是结果）。
* **自述不成立**（还缺任何一道阈或任一支 EMA）⇒ 如实登记"面仍不能解释自己"，并**收回**"重放已足够"这句：
  那意味着未来每一张面都要重放一遍才能定位，代价与风险都更高。

## 6. 本件不做什么

不改常量、不放行生长、不动 `reset_dynamics`、不放大剂量、不重跑 N3 甲两臂、不把"面能自述"当成"τ 已定好"、
不借本件重开 ㊵-499 的在场性判级、不把 DEBT-G54（措辞门覆盖面）混进本件（那是"先修仓库"那一档，另立）。

## 7. 不变项

M6 收官、M8 挂起；N3 甲甲臂起跑、`git push origin main`（本机现测未推送 38 条）、N2 第三件改权重实验立项、
N3 乙 τ 走甲/乙/丙——四格仍等 owner；`output/n3_04/`（233.7 MB）只读、已挂账在 09 §3.2 第 4 项；
产品内核与 `taiji/**`、`seed/**` 未动；台账行序以行首标号为准（㊵-419⑥）。
