# PLAN-N5-23 同配置重复臂（十裁⑴ 乙）的噪声地板与幅度判据——判据先冻，起跑在本文入库之后

冻结时刻：2026-10-13（㊵-776）。批准出处＝owner 十裁⑴ 选**乙**（加同配置重复臂，约 2×10.2 h 训练＋两趟 CAP 评测）。
执行体＝批次 C 那台在册跑器 `scripts/training/train_taiji_r2_readout_retrain.py`，**不新增臂、不改 `ARMS`、不动 `taiji/`**。
本件冻的是"什么读数才算买到一条幅度句"，在任何重复臂读数取到**之前**。

## §0 为什么这一档必须存在（全是实测，可复算）

* 批次 C 的配对判读三条句子各自缺前置：能力面 `direction=not_resolved`（入场维 {C,D,E}，只有 D 动一题，`delta=-1`，阈值 3）；
  机制句被名集不齐挡住（DEBT-G133／G131）；**幅度句被 `J-N5t-4`／`J-N5u-4` 明写"需同配置重复臂"挡着**。
* 现在唯一的"重复"证据是两臂自取速率各一枚样本（241.952872 对 322.26993）且跑在不同窗口 ⇒ 那 25% 的差**不许**当散布用。

## §1 重复臂的规格（与批次 C 只差"再实现一次"这一件事）

同一基座 `checkpoints/seed_a31self_with_circuit.pt`（sha `d6169a358eaee6d194d4795e3167a7bcbb42dfde57b92aa1199bcebbed89699b`）、
同一臂 A、同一预算 `--symbols 8900000`、同一偏移 `--skip-symbols 11199800`、同 `--fresh`；
治疗档 `--generated-feed-every 50`、对照档 `--generated-feed-every 0`。落点改名到 `output/n5_20_batchC2_treated`／`output/n5_20_batchC2_control`，
**不覆盖批次 C 的任何在盘产物**。两臂之间不许插入任何对 `taiji/`／`seed/`／`scripts/training/`（跑器 import 链）的提交。

## §2 判据（冻五条，每条写明"什么情况下它为假"）

* **J-N5v-1 重复资格**：两枚新臂件必须与批次 C 同 `runner_commit`（并排仪现读的 `_runner_commit_at` 形）、同 base sha、同 `symbols`、同 `skip_symbols`、同 `--fresh` 真。
  为假的情形＝任一项不同 ⇒ 这不是重复而是第四臂，本档全部读数只许写成"另一次跑"，**不许**拿来推地板。
* **J-N5v-2 地板逐指标出版**：对每个进判据的指标 `m`（首四个＝`cumulative_mean_surprise`、`cumulative_online_accuracy`、CAP 逐题翻转数、`generated_write_count`），
  `floor_m = max(|m_重复治疗 − m_批次C治疗|, |m_重复对照 − m_批次C对照|)`。地板必须**逐指标点名**（键名＋单位＋分母），不许出一个全局数。
  为假的情形＝地板来自引用或来自"我觉得计时噪声大概多少"，而不是这四枚两两差。
* **J-N5v-3 幅度句的唯一出口**：某指标上要写"治疗臂比对照臂高/低 X"，必须 `|treated − control| > floor_m`；
  否则只许写"该差落在同配置重复的散布内（floor=…）"。分支互斥两支：`floor_m=0`（两次逐位相同）⇒ 走 `ruler_no_spread`，
  **不发表**该指标的地板（"没有散布"不等于"散布很小"，它意味着仪器可能没被走到）；`floor_m` 存在且 ≥ 观察差 ⇒ `magnitude_blocked`。
* **J-N5v-4 CAP 面同法**：重复两臂各再取一趟 CAP（同 `git_head`、同 `eval_set_sha256`、默认链路、不传 `--relax-legacy-guard` 等旗标），
  地板＝逐维翻转计数的重复散布；`J-N5u-3` 的 `not_resolved` 只有在新读数把 D 维带离 `|delta| ≤ floor_D` 时才许改判，**阈值 3 一字不动**。
* **J-N5v-5 负载自证**：每臂自述必须与同一次窗口的负载读数一起入库（`measured_symbols_per_second`、`wall_seconds`、逻辑处理器数、当时活跃 python 进程数）。
  为假的情形＝报价旁边没有负载出处。窗口内**不跑整族回归、不跑 CAP 评测**（除 J-N5v-4 指定的那一趟，且它排在两臂都退出之后）。

## §3 守卫（两条）

* **G-N5v-1 不许在飞的臂被中途代码变更污染**：两臂起跑时刻必须早于任何对跑器 import 链的提交时刻；判定＝并排仪现读的两臂 `runner_commit` 相同且等于批次 C 那枚 `9299352d4…`。
* **G-N5v-2 写盘前置**：判读器在三批汇总行齐备前什么都不写（`progress_exit.json` 的 `reached_budget=true` ＋ `run_report.status=completed` 才算齐备）；
  它的拒绝分支必须实测走一次并留下 `NOT-READY` 原文。

## §4 落笔即钉的三行

* **预期绿**：`J-N5v-1`（资格）、`J-N5v-5`（负载自证）；`J-N5v-2` 出版的是地板值不是结论。
* **仍红时收回哪句**：`J-N5v-1` 红 ⇒ 当场收回"这是重复臂"，本档降级成"第四次跑"，不许用它推地板；
  `J-N5v-5` 红 ⇒ 收回一切墙钟与速率报价，只留产物计数。
* **发表资格前置**：`J-N5v-2` 的地板未逐指标成形之前，"批次 C 的幅度"这一族句子**一枚都不许写**；
  地板成形之后，能写的也只有"哪些指标的观察差离开/没离开各自地板"。

## §5 十四裁⑴ 落地（2026-10-14 ㊵-825）：本链不再等「第二个档」

* **史实**：㊵-824 证明 `config.seed` 是结构身份（换种子 ⇒ `ValueError: receptor channel map does not match architecture`，`taiji/organs.py:99-104`／`:162-165`），owner 十四裁 ⑴ 选「丙＝承认本管线无可用混杂档」。
* **对 `J-N5v-3` 的影响（不改写原文，只标可走分支）**：本链（同 base、同 skip=11199800、同跑器版）上地板只有 `floor_m = 0` 那一支 ⇒ 只有 `ruler_no_spread` 可出版；`magnitude_blocked` 那一支**在本链永不成立**，因为它要求先有一个已出版的非零 `floor_m`。两种说法不许混写：批次 C 的幅度句是「无地板可依」，不是「差被地板挡住」。
* **发表资格前置（新增一句）**：任何把批次 C 的 `not_resolved` 往「已判／已挡」任一侧改写的句子，都必须先点名它引用的是哪一支；引用不到分支名的，按不可发表处理。

## §6 十五裁⑷甲 落地（2026-10-14 ㊵-833）：`J-N5v-2` 的「首四个」里第三支为何不出地板，以及本档至今没出版过地板

* **口径差（不改写 §2 原文，只补）**：§2 把 `generated_write_count` 列进"首四个"进判据的指标，而仪器
  `scripts/training/audit_taiji_n5_23_repeat_arm_noise_floor.py:39` 的 `METRICS` 只含
  `cumulative_mean_surprise`／`cumulative_online_accuracy` 两支训练面读数，CAP 那一支走
  `--cap-pairing-*` 入参（逐维地板见 `:185`）。⇒ 只按 §2 字面核对会得出「地板成形 2/4」的错判。
* **为什么 `generated_write_count` 不该有地板**：它在这条链上是**预算算术决定量**
  （8,900,000 ÷ `every_symbols=50` ＝ 178,000；对照臂 `every_symbols=0` ⇒ 0），不是观察量；
  给它出地板等于给一条除法出散布。它的证据归属在并排仪
  `audit_taiji_batch_c_arm_parity.py` 的 `treated_feeds_match_budget_arithmetic` 与
  `j_n5r_2_treated_write_count_positive` 那一档（＝可归因性），不在"噪声地板"这一档。
* **本档的出版状态（重要，防误读）**：2026-10-14 那对重复臂按 `J-N5v-1` **判红**——并排仪现读两臂
  `runner_commit` 同为 `fe39dfadef6e…`，而批次 C 那枚在库件现读为 `9299352d439b…`（差一枚 ㊵-756 的跑器写靶补行尾）。
  §2 原文对这一支的处置是「只许写成另一次跑、不许拿来推地板」，故本档**至今未出版过任何一枚地板**。
* **仪器读数的引用规矩**：地板仪的资格面**不含** `runner_commit`（`:12` 那条只比基座 sha／预算／`skip_symbols`／
  `fresh`／`every_symbols`），所以它会在跑器版差照样报 `verdict="holds"` 与 `same_configuration_within_role=true`。
  ⇒ 引本档的地板读数必须**同时**引并排仪的 `runner_commit_per_arm` 逐臂点名；只引地板仪那一句的，按不可发表处理。
