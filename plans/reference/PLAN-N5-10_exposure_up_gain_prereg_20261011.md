# PLAN-N5-10 换暴露量档（收益侧分辨率）——判据先冻、本件不开跑（2026-10-11）

## §0 谁批的、批的是什么

owner 弹窗 #24 第②裁＝**「开：换更高暴露量档」**（原话选项描述为"例如把 60,000 tick 提到 150,000，
先另立一件把判据冻住"）。这条裁定改变的是 **N5 出口的第二格**：㊵-665／666 那次按"证伪"这一路收口时，
收益侧（`J-N5b-4`）从来不是"判为不成立"，而是 **`ruler_unusable`＝这把尺答不了**（09 §3.2 与
`PLAN-N5-09` §6 都这么写着）。本件要买回的是**分辨率**，不是新机制。

**本件不动产品码、不动任何已冻判据的数值、不新增阈值。** 会更新权重的是 §5 那三臂，它们只能在本件
落盘并过措辞门之后起跑。

## §1 为什么 60,000 那一档答不了（全部是现读，不是转述）

`reports/taiji_n5_07_gain_pair_metric_20261010.json` 现读：

* `arm_A.acc_noise_band_adjacent_max = 0.022779`、`arm_B.acc_noise_band_adjacent_max = 0.020879`；
* 过线界 `line = 0.02`（PLAN-N5-02／03 冻值，住在仪器默认里＝**DEBT-G90**）；
* ⇒ 判读器 `adjudicate_taiji_n5_shadow_gate.py:295` 的 `ruler_usable`（要求两枚 band 都 > 0 **且**
  `line ≥ band_max`）为 **false** ⇒ `j_n5b_4 = ruler_unusable`；
* 同件里 `arm_A / arm_B` 的五段段均值末三段是 `0.2294／0.2443／0.2439` 与 `0.2290／0.2438／0.2442`
  ⇒ 两臂末两段均值之差只有 **−0.0004**，落在被这把尺自己判定为"比线还宽"的带里。

**结论形状**：60k 这一档能说的是"影子通电进了训练流并且写了权重"（㊵-662／663 的 20 槽＝10 枚权重），
**不能**说"涨点"也不能说"不涨"。

## §2 剂量算式（为什么取 150,000，而不是随手翻倍）

噪声带取自相邻段均值之差，段数由出数仪器固定为 5 ⇒ 每段样本量 ∝ 符号数 ⇒ **band ∝ 1/√(符号数)**。
按 §1 那两枚现读带外推：

* 要让 `band_max < 0.02` 只需 `150000/60000 > (0.022779/0.02)² = 1.297` ⇒ **≥ 77,900 符号**；
* 取 150,000（×2.5）⇒ 预期 `band_max ≈ 0.022779/√2.5 = 0.01442`，对 `line = 0.02` 留 **≈ 28% 余量**。

**这条外推是假设不是读数**，所以 §3 的 `J-N5j-1` 把它当成**待验的一格**而不是前提：真跑出来若
`band_max ≥ 0.02`，本档按预注册记 `ruler_unusable`，**不许**再悄悄把剂量往上抬第二次——那时该走的是
DEBT-G90 的判据升版（带 owner 裁定），不是"再跑一档更大的"。

## §3 判据（三条，全部沿用已冻的公式与数值；本件只新增"取哪一档"这一件事）

* **J-N5j-1（尺可用，前置）**：按 `adjudicate_taiji_n5_shadow_gate.py` 的现公式，
  `bands.treated > 0 ∧ bands.control > 0 ∧ 0.02 ≥ max(bands)` ⇒ `ruler_usable = true`。
  为 false ⇒ 本档只出版分辨率陈述，`J-N5j-2` 不判级（**不**记成"不成立"）。
* **J-N5j-2（收益侧，母量与线都是冻的）**：`delta = last_two_segment_mean(treated) − last_two_segment_mean(control)`，
  `delta > 0.02` ⇒ `holds`；`delta ≤ 0.02` ⇒ `not_holds`；`J-N5j-1` 未过时本条记 `unverified_ruler_unusable`。
* **J-N5j-3（保持侧在新档不许借用旧档的噪声地板）**：`j_n5b_5` 只有在本档自己的 `noise_floor`
  （§5 的第三臂）到位之后才许出值；缺它 ⇒ 新档保持侧一律 `unverified_noise_floor_missing`
  （PLAN-N5-08 §2 的取数式原文，本件不放宽）。

## §4 守卫（五条，各自能为假）

* **G-N5j-1 单变量**：两臂 argv 逐字相同，只差 `--n5-shadow-gate 1.0` 与三个落点路径；由现成仪器
  `check_taiji_n5_pairing_preflight.py` 判（**不在这里重抄它的五维核对**）。
* **G-N5j-2 塌臂排除**：`compare_taiji_n5_arm_tensors.py` 必须报 `differing_slots > 0`；
  为 0 ⇒ 本档塌成一枚权重，`J-N5j-2` 无对象，按 PLAN-N5-04 §3 的 G-N5d-4 出口点名，**不新造判级词**。
* **G-N5j-3 同底同源**：三臂 `--resume` 都是 `checkpoints/seed_beta.pt`，其 sha256 前缀
  `ad2a06465e0ef78c`（4,143,542 B，2026-10-11 现读）写进本件并逐臂核对。
* **G-N5j-4 完成记账**：三臂各自 `progress_exit.json` 的 `reached_budget == true`；
  任一为 false ⇒ 记"半档"并如实发表，判据一字不改。
* **G-N5j-5 负载自述（本件新买到的那格）**：三臂 exit 件都必须带 `machine_load.status == "ok"`
  （DEBT-G94② 已于 2026-10-11 落地）；缺该键或为 `unavailable` ⇒ 引用这三臂的秒数时必须写"负载档未出版"。
  反向守卫：三臂的 `machine_load` 若互不相同到足以解释秒数差（用 ㊵-671 那枚比值口径），
  **不得**把秒数差说成"改动让它变慢"。

## §5 三臂形状（命令逐字由 60k 两臂的信封 `metadata.command_surface.argv` 现读导出，只改两处）

改动只有 `--max-symbols 60000 → 150000` 与落点 `output/n5_07_arms/* → output/n5_10_arms/*`：

```
# 治疗臂
scripts/training/train_seed_corpus.py --corpus data/simple_zh/dialogue_extended_clean.jsonl
  --resume checkpoints/seed_beta.pt --checkpoint output/n5_10_arms/treated/checkpoint.pt
  --progress output/n5_10_arms/treated/progress.json
  --pressure-record output/n5_10_arms/treated/pressure.jsonl
  --readout predictive --n5-shadow --developmental-fast-slow --developmental-bridge-gate 1
  --growth-min-pressure 0.64 --max-symbols 150000 --n5-shadow-gate 1.0
# 控制臂：与上面逐字相同，**去掉** `--n5-shadow-gate 1.0`
# 第三臂（本档噪声地板）：与控制臂逐字相同，只把三个落点换成 output/n5_10_arms/noise_floor/*
```

τ 仍取 **0.64**（β 链在 v2 标定面上现冻的值，PLAN-N3-09 规则）；**DEBT-G86** 那条限制照在——
"用更长的面重算 τ"是反的，本件把长跑面只当行为记录、不当分布样本。基座是 `seed_beta.pt`（β 链，
不含 circuit），与本件对照的 60k 两臂同链；跨链搬 τ 被 PLAN-N3-03 §4 的五元组禁令挡住，本件不碰。

## §6 预算（一律取实测，不外推自"线性假设"）

* 60k 同配置三档实测 elapsed：**314.346／370.638／418.550 秒**（`output/n5_07_arms/{control,treated,determinism}`
  的 `progress_exit.json` 现读）——**同配置两档差 33%，而权重逐位相同**（㊵-662／663）。
  ⇒ 秒数只能按区间报价：150k（×2.5）≈ **每臂 786–1,046 秒**，三臂串行 ≈ **40–53 分钟**。
* 磁盘：每臂目录实测 64,788 KB（60k 档，含 `checkpoint.pt` 21,926,613 B 与 `.pt.history`）
  ⇒ 三臂 ≈ **190 MB**，落点 `output/n5_10_arms/`（`.pt`／`.jsonl` 均被 `.gitignore:62/:157` 忽略，
  只有 `.json` 会在 `git status` 显形；`git check-ignore -v` 每次引用前现测）。
* 判读侧零新增算力：全部走既有仪器（配对件＋`adjudicate_taiji_n5_shadow_gate.py`）。

## §7 判据载体在哪（措辞门要求的一节）

J-N5j-1／2 的**执行体**是既有仪器 `scripts/training/adjudicate_taiji_n5_shadow_gate.py`
（`judge_metric_lane()`，:235-296；`ruler_usable` 的式子在 :295），本件不重算任何一步；
J-N5j-3 的执行体是同器的 `judge_retention_lane`（PLAN-N5-08 §2 已接）。
G-N5j-1／2／3 的执行体分别是 `check_taiji_n5_pairing_preflight.py`、`compare_taiji_n5_arm_tensors.py`、
exit 件里的 `checkpoint_sha256`／`corpus_fingerprint`。G-N5j-4／5 的执行体是训练器自身的
`progress_exit.json`。**本文件里没有任何一条判据是只手算的**——每条都点名了出版它的仪器。

## §8 出口（三条互斥）

* `J-N5j-1` 过 ∧ `J-N5j-2 = holds` ⇒ 收益侧第一次可以发表"在 150,000 tick、这套装配、这条链上，
  影子通电的母量优势过 +0.02 线"；仍**不许**外推到别的暴露量／别的链。
* `J-N5j-1` 过 ∧ `J-N5j-2 = not_holds` ⇒ 收益侧判**不成立**（这才是"证伪"意义上的第二格），
  N5 的 `j_n5b_6` 合取按 PLAN-N5-09 归类，`J-N5b-4` 从 `ruler_unusable` 升级为有值成员。
* `J-N5j-1` 不过 ⇒ 收回"这一档把分辨率买回来了"这句，登记"在 +0.02 这条线下，
  150k 仍不足以让这把尺可用"，**转 DEBT-G90 的判据升版裁定**（不再自行加剂量、不搬别的尺代答）。

## §9 落笔即钉的三行

* **预期为绿的成员**：`G-N5j-1`（argv 由信封现读逐字导出，只差冻过的那两处）、`G-N5j-3`（同底同件）、
  `G-N5j-5`（DEBT-G94② 已在训练器里落地并有测）。
* **仍红时收回的句子并点名**：若 `G-N5j-2` 报 `differing_slots = 0` ⇒ 收回"这一档是两臂"整句并点名
  ㊵-654 那次同形状的塌臂（当时按 G-N5d-4 收回，这次同样不新造判级词）；若 `J-N5j-1` 不过 ⇒ 收回
  "换暴露量能买回分辨率"这句，并点名本件 §2 的外推假设。
* **发表资格前置**：任何 `holds` 的对外表述必须同时给出 ①`ruler_usable=true` 与两枚 band 的数值，
  ②`noise_floor` 来自**本档**第三臂（不是 60k 那枚 0），③三臂 `machine_load` 的并排值——三条缺一即视为未判。

## §10 不变项

M6 收官、M8 挂起；N1／N2／N3 已收口，N4 交完设计预注册，N6 随实施项清；
**DEBT-G78（世界学习器接进语料训练链）按 owner #24 第①裁继续挂起**，本件不碰 `taiji/`；
05 债表现至 **DEBT-G94**；台账行序以行首标号为准（㊵-419⑥）。
