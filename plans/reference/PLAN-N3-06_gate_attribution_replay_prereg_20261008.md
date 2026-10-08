# PLAN-N3-06 · "六道分项闸里是哪一道在拦"——零算力重放仪预注册（2026-10-08 起草，**判据先冻、跑前不解释**）

> **来源与效力**：09 §2 N3 与 [PLAN-N3-04 判读](PLAN-N3-04_ADJUDICATION_20261008.md) ㊵-499④ 留下的那句
> "**本件不指认是哪道分项闸在拦**（面件只自述六道闸中的三项，要定位得先把分项闸通过态写进件）"。
> 起草时按那句去设计"改训练器记录器＋重跑 5 支面"，**读码后发现不必**：产品自己的决策对象已经带着五支 EMA
> 与 `reasons`（`taiji/adaptive_residual_growth.py:468-488`），而面上已记录的原始观测足以把同一支
> `AdaptiveResidualGrowthTrigger` **逐 tick 重放**出来 ⇒ 本件＝**零算力、零改产品码、零新面**的在库件复算。
> 成熟度＝归因/定位类，不声称任何能力结论。
> **2026-10-08 同日跑完并判读**：判 **`single_gate_blocking`**（四张面一致）——拦阻者是合成量那道闸
> （默认阈 0.70 对 EMA 上界 0.67813–0.685877），其余五道没有一道恒假；
> 读数件 `reports/taiji_n3_06_gate_attribution_20261008.json`（rc=0），判读在
> [PLAN-N3-06_ADJUDICATION_20261008](PLAN-N3-06_ADJUDICATION_20261008.md)。§2/§3 的判据与验收式此后不得再改；
> 实现期新增的两条（A-5、反事实块）已在 §2quater/§3bis **按"跑前未冻"就地自报**，不追溯改写。

## 0. 读码读出来的一条**口径更正**（先记这条，因为它改变既有读数的名字）

门比的**不是**面上记的那个 `pressure`：

* `AdaptiveResidualGrowthTrigger.observe` 的合取闸（`taiji/adaptive_residual_growth.py:437-444`）是**六支 EMA 各自的**
  `*_ema >= policy.minimum_*`，六项一 `and`；
* `pressure_ema`（:388-395）＝ `0.30·residual_error_ema + 0.25·fast_slow_conflict_ema + 0.20·activity_saturation_ema + 0.25·utility_gap_ema`
  ——**五个 EMA 的加权和**，而面件每行记的 `pressure` 是**五个原始信号的加权和**（`AdaptiveResidualGrowthPressure.pressure`）；
* EMA 更新发生在比较**之前**（:422-435），初值 `residual/fast_slow/activity/utility = 0.0`、`resource_state_ema = 1.0`（:359-363），`rate = policy.ema_rate = 0.25`（:188）。

⇒ **㊵-498/499 里那句"`pressure` max 0.7092 已越过默认阈 0.70"说的是原始口径，而门看的是 EMA 口径**；
"够得着但不触发"因此在**不引入任何新事实**的情况下就有了一条候选解释（EMA 从 0 起步、且六个阈必须**同时**满足）。
这条更正**不撤销**既有任何判级（那些判级读的本来就是原始分布），但**冻结在其上的 τ 定义名字要改**：
入册 **DEBT-G53**（"τ 的口径与闸的口径不同名"），修法在 §5。

## 1. 重放的合法性（这是本件的全部风险所在，先把它变成机检）

`trigger.observe(pressure, structural_budget=…)` 对预算的**唯一**用法是两处比较
（`budget >= policy.growth_resource_cost` 进 `should_propose` :454、`budget < cost` 进 `structural_budget_insufficient` :464），
而 `structural_budget` 来自 `int(self.config.development_structural_budget)`（`taiji/model.py:1084`）——**不是状态相关量**；
面上每行的 `resource_state` 正是同一个比较的结果（:1085）。
⇒ 用 `structural_budget = policy.growth_resource_cost` 重放，与原始跑**在闸的语义上等价**，
而这条等价**不由我论证、由 §2 的 A-1 逐行核对**（重放出的 `should_propose` 必须与面里记的 `decision_should_propose` 逐行相同）。

重放**不重抄任何算式**：观测由 `AdaptiveResidualGrowthPressure.create(...)` 构造（加权和与 `pressure_digest` 都由产品自己算），
决策由同一支 `AdaptiveResidualGrowthTrigger` 产出；仪器只做"读面→构造→observe→摊平读数"。

## 2. 验收式（先冻；任一不符 ⇒ rc=2 响亮拒绝，不出版读数）

* **A-1 重放忠实性（锚点）**：逐行 `replay_should_propose[i] == face_should_propose[i]`，**4,000 行全等**才出版；
  任一行不等 ⇒ `refused_replay_not_faithful`，并点名首个失配行号与两侧读数。这条是"外部重算"与"产品自述"的接口。
* **A-2 观测自洽**：重放构造出的 `pressure` 值必须与面里记的原始 `pressure` **逐位同**（同一支加权和、同一份输入）；
  `resource_state` 必须逐行 ∈ `{0.0, 1.0}` 且与 `cost` 的比较方向一致。
* **A-3 面完整**：缺 `kind=face` 头、缺任何一行、行数 < `MIN_OBSERVATIONS`（既有下限 500）、
  `evidence_id` 重复（trigger 自己会响亮拒绝）、`parent_checkpoint_digest` 中途变化 ⇒ 各自拒绝。
* **A-4 判据用词纪律**：输出里不许出现 `audit_taiji_prereg_exit_wording.py` 那张词表里的四个模糊量词
  （词表由该仪器自述，本件不抄写它们——抄进来会被这道门判成本件自己的"模糊词未钉数值"命中，㊵-505 实跑撞过一次）。

## 2quater. 实现期新增的一条锚点（**跑前未冻**，自报；它只会多拒判、不会促判）

* **A-5 合取对账**：六道闸各自判"过没过"之后取 AND，其为真步数必须与产品 `reasons` 里
  `pressure_below_threshold` 的**反面**计数相等（:437-444 那条 `and` 链与 :460-461 那条 `if` 是同一件事）；
  不等 ⇒ `raise` ⇒ rc=2。这条比 A-1 更严（A-1 只比 `should_propose` 这一个布尔），
  它把"我用六条读数另算一次"重新钉回"产品自己算的那一次"。
  ⇒ 判读件里以 `g_n3c_meets_pressure_crosscheck_derived_steps` 与 `..._equal` 发布，实测 `0 == 0`（四张面）。
* **为何算违规而不算修订**：新增的是**更严的拒判条件**，不改 §3 的判级线；发表时按
  "跑前未冻"就地标注（㊵-419⑥ 同一族纪律），不追溯改写成"一直都冻着"。

## 3. 判据 J-N3c-定位（三档互斥，先冻；每档都有名字，不留含糊出口）

对每张面先算六道闸各自的**为真步数**（`*_ema >= 对应 minimum_*`，用产品自己维护的 EMA）与 `meets_pressure` 为真步数：

* **J-N3c-单闸**：恰有**一道**闸为真步数 == 0 ⇒ 判它为阻塞者（并报告它的 EMA 末态、p90、距阈差 `threshold − p90`）；
* **J-N3c-多闸**：**≥2 道**为真步数 == 0 ⇒ 判"合取式不能唯一指认"，全部点名并各报同样四项；
* **J-N3c-非闸**：六道**都有**为真的步、而 `should_propose` 仍全 0 ⇒ 阻塞归**持续性/预算**两闸，
  此时按 `reasons` 直方图指认（`pressure_persistence_below_threshold` 与 `structural_budget_insufficient` 各自计数，
  并报 `consecutive_pressure_steps` 的最大值与 `required_pressure_steps`）。

三档互斥且覆盖全空间（0 道恒假 ∧ 触发为零 ⇒ 必落第三档）。**同时报** `meets_pressure` 为真的最长连续段，
因为它与 `required_pressure_steps` 的差就是第三档的定量内容。

## 3bis. 反事实块（同样**跑前未冻**，明写"不参与判级"）

实现时加了 `counterfactual_not_judged`：(i) 摘掉合成量那道闸后其余五道**同时**为真的步数与最长段，
(ii) 那些步里合成量的上界＝**τ 的可触发上界**，(iii) 把 §1 那条既有冻结规则（p90 向上取整到 0.05 格）
**原样**套到 EMA 口径得到的候选 τ，以及该候选下六道合取的真步数／最长段／`would_have_triggered`。
三条例外必须一起读：**(a)** 它们不参与 J-N3c 判级；**(b)** `would_have_triggered` 只在**首次触发点之前**有效
——一旦真放行生长，产品会清连续段并挂载候选，其后轨迹与原始跑必然分岔；**(c)** 它不是"跑了会赢"的实验结论，
只是算术定价。加它的理由：只指认闸而不给价，等于把可执行的那一半留给猜测。

## 4. 面与预算（零新跑）

| 项 | 值 |
|---|---|
| 输入 | 已在盘的 4 张在线面 `output/n3_04/{beta,circuit}_{gate025,gate100}/pressure.jsonl`（每张 4,001 行＝1 头＋4,000 观测）＋ canary 那枚 199 观测面**不用于判据**（低于样本下限，只作 A-3 的反例支） |
| 算力 | 纯读取＋重放，秒级；**不训练、不改权重、不动产品码** |
| 磁盘 | 一枚判读件 `reports/taiji_n3_06_gate_attribution_20261008.json` |
| 注 | 这 4 张面是 owner 第四次弹窗批过的同一批产物，本件只换"读法"不换"跑法" ⇒ 不构成扩样 |

## 5. 出口（读数的去向，按预注册的分支走，不临场挑）

* 判"单闸" ⇒ 该闸就是"N3 乙放行不成立"的成因，**下一步的候选形状写死为两条**并回 owner 选：
  (甲) 该信号在本装配上定义域受限（若是 `activity_saturation`：`gate` 只影响 bridge 参与读出的深度），
  (乙) τ 连同其余五道 `minimum_*` **一起**在这条链上重冻（＝DEBT-G53 的修法：把阈定义到 EMA 侧，
  并把 `ema_rate`/初值写进定义件——动 `AdaptiveResidualGrowthPolicy` 常量需签字）。
* 判"多闸" ⇒ 本件**不给出可发表的 τ**，并如实写"六项合取在这条链上至少 K 道恒假"，K 由读数给。
* 判"非闸" ⇒ 持续性/预算侧成为下一格；`required_pressure_steps` 与 `growth_resource_cost` 是候选旋钮（**改常量需 owner 签字**）。
* 任一档都**不改变** ㊵-499 的"在场性成立、阈值不成立"判级；本件只回答"哪一道在拦"。
* **另有第二件待立**（不在本件做）：把五支 EMA＋`reasons` 写进未来面行的记录器改动（`train_seed_corpus.py:556-567` 那支 `_record_pressure`
  里 decision 就在手边，加字段零成本），并给 face 头补上现在缺的四道 `minimum_*` 与 `ema_rate`（:535-539 只自述三项）。
  本件的重放读法**不能替代**那次改动——它证的是"这批在库面上能定位"，那条证的是"未来的面自带定位"。

## 6. 本件不做什么

不改 `AdaptiveResidualGrowthPolicy` 任何常量、不动产品码、不起任何新面、不重跑训练、
不声称"改了这道闸就能触发"（那是另一件需要单变量臂的实验）、不把 EMA 口径的读数与原始口径的读数互换。

## 7. 不变项

M6 收官、M8 挂起；N3 甲甲臂起跑、`git push origin main`、DEBT-G52 两修法选一、N2 第三件改权重实验立项——四格仍等 owner；
`output/n3_04/`（233.7 MB）已挂账在 09 §3.2 第 4 项，本件**只读**它；台账行序以行首标号为准（㊵-419⑥）。
