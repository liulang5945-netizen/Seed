# PLAN-N5-04：保持侧换底预注册（续训双臂，2026-10-09）

> **状态**：起草即冻结——本件 §2/§3 的判据与守卫在**任何跑之前**落盘，此后只许带日期就地更正或再次升版，不许事后调线。
> **为什么现在写它**：㊵-613 否证了 `J-N5b-5` 在现行 G/H 两臂上的可取数性，并给出换底方案。本件不推翻 09 的 N5 目标，
> 只把「影子通电有没有代价」这一问改成**算术上能回答**的形状。

## §0 已实测的前置事实（全部来自工具输出，逐条带出处）

- **F-1 现行两臂没有 before 面**：`output/n5_shadow_gate_on/progress_exit.json` 与 `output/n5_shadow_gate_off/progress_exit.json`
  自述 `base_ticks=0`、`ticks=60000`、`reached_budget=true`、`unique_documents=76`，日志首行 `{"ticks": 60000.0, "parameters": 946931.0}`，
  墙钟 `396.80334510002285 s`／`334.63762799999677 s` ⇒ 两臂是从头训练、训练器只在退出时写件。
- **F-2 从头臂上的保持侧是恒真式**：未训底子的七列读数为 0（[[p3b-two-arm-training-design]] 已记「全 0」是未训底子的假读数），
  而保持判据是 `after < before` ⇒ `before=0` 时**任何非负读数都判成 `retention_holds`**。一条不能为 false 的判据不配叫判据
  （[[guard-must-be-able-to-fail]]）。
- **F-3 before 面已经在库里**：`reports/taiji_n2_cap0_before_20261008.json`（format `taiji-cap0-baseline-v1`，
  `checkpoint=checkpoints\seed_a31self_with_circuit.pt`）与 `reports/taiji_n2_replay24_before_20261008.json`
  （format `taiji-a30-repetition-penalty-v1`，`checkpoint=checkpoints/seed_a31self_with_circuit.pt`，`started_utc=2026-10-08T03:09:27Z`）。
  ⇒ 换底到「以该件为共同基件的续训双臂」，`J-N5b-5` 需要的「前」不需要重跑，只需按字节核对同源。
- **F-4 出件方与命令面**：`scripts/training/eval_taiji_cap0_baseline.py`（`--checkpoint/--report/--dimensions`，
  评价集 `plans/manifests/cap0_eval_set_v2.json`，`eval_set_sha256` 出版在 `identity` 层）；
  `scripts/training/measure_taiji_a30_repetition_penalty.py`（`--checkpoint/--circuit/--chain/--manifest/--limit/--offset/--penalties/--out-report`，
  该件 `--help` 在 GBK 控制台 `UnicodeEncodeError` ⇒ 调用须带 `PYTHONUTF8=1`）。
  回路件 `output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt` 实测在库（186659 字节）。
- **F-5 通电侧已判完、不在本件重判**：treated `gate=1.0/req=1.0/candidate_gate=0.2582141160964966/utility=+0.0005751861608587205/counterfactual=-0.0010580363106349466`，
  control `gate=0.0/req=None/candidate_gate=0.5/utility=0.0/counterfactual=0.0`；学习侧母量 `delta=-0.0004` 落在噪声带内、未过已冻 +0.02 线。

## §1 本件的换底主张（一句话）

把 N5 的通电对照从「从头 60k 两臂」改为「**同一基件续训、只动 `--n5-shadow-gate` 一个变量的两臂**」，
before 面直接复用 F-3 的已入库件；这样 `J-N5b-5` 的分子分母同源，且 `J-N5b-6` 的合取第一次成为可判对象。

## §2 判据（先冻，跑后不得改）

- **J-N5d-1（保持侧，主判据）**：两臂各出一对面件（cap0 面＋replay 面），交给**未改动的** `adjudicate_taiji_n2_04_retention_pair.py` 判读。
  七列＝`count_taiji_n2_03_attribution.py:30-38` 的 `DROP_COLUMNS`（`cap0:E`、`replay_strict_hits:{1.0,2.0}`、
  `replay_well_formed:{0.0,0.5,1.0,2.0}`），**本件不新列、不许事后加列**。
  两臂的保持侧 verdict 必须**各自抄录**（`retention_holds`／`cost_persists`），两臂同结论才算这条判据有读数；
  一臂 `holds` 一臂 `persists` ⇒ 记 `arm_dependent`，本件不许挑对自己有利的那一支发表。
- **J-N5d-2（学习侧，沿用已冻线）**：收益判据仍是 PLAN-N5-02/03 的那把尺（+0.02 线、`ruler_usable` 公式
  `band_a>0 ∧ band_b>0 ∧ |Δ| ≥ max(bands)`），本件**不重述也不放宽**。若换底后分辨仍不如噪声带 ⇒ 结论写作分辨率陈述，不写「影子无效应」。
- **J-N5d-3（合取）**：`J-N5b-6` 只有在 J-N5d-1 与 J-N5d-2 都有可判读数时才允许填值；任一为 `unverified` ⇒ 合取保持 `not_adjudicable`。

## §3 守卫（fail-closed，逐条点名拒判形状）

- **G-N5d-1 同源核对（硬前置）**：after 面的 `eval_set_sha256` 与 `items_sha256`/`first_item`/`limit` 必须与该臂所用 before 面逐字相同；
  任一不一致 ⇒ 判读器按既有代码路径 rc=2（`faces_not_same_source`），**不许**以「题集看起来一样」代答。
- **G-N5d-2 单变量**：两臂除 `--n5-shadow-gate` 请求值外，`--resume` 基件 sha、语料指纹、`readout`、`--max-symbols`、seed 必须逐字相同；
  四元组由 `adjudicate_taiji_n5_shadow_gate.py` 从面头读取并出版，本件不另写一份。
- **G-N5d-3 默认位不动**：`taiji/config.py` 与产品默认位一律不改；通电只进命令（owner 第十二次弹窗的裁定沿用）。
- **G-N5d-4 过程侧计数器**：本件起，`n5_shadow` 块必须带 `shadow_branch_hits`／`shadow_materialized` 两枚在场计数器
  （㊵-593 承诺、㊵-613 查实读回 `None`＝从未进块）。取数纪律＝`'k' in block` 判在场、`.get()` 判值；
  计数器没回来之前，「影子确曾被喂过」这句**不许**发表（[[probe-output-must-be-verified-present]]）。
- **G-N5d-5 零产品码改动面**：本件的跑只允许加训练器旗标，不允许改 `taiji/` 观测语义；若要改，另立实施预注册。

## §4 失败出口（预先指定谁判据谁定位）

- before/after 不同源、或七列取不到数 ⇒ 整件 `not_judged`（rc=2），收回「保持侧已判」这句，并点名是 G-N5d-1 拦的。
- 两臂都 `cost_persists` ⇒ 负结果照常出版：通电买到的是链路，代价在权重更新那一侧（与 N2 乙档 `cost_persists` 同型，不泛化到 N5）。
- 两臂都 `retention_holds` 且收益未过线 ⇒ `J-N5b-6 = not_holds`，N5 的「自进化唤醒」按 09 §4 分流转「换装配」或「换预算」，二者都要另立预注册。
- 一臂 `holds` 一臂 `persists` ⇒ `arm_dependent`，本件不发表方向性结论，先查两臂差异是否只有那枚旗标。

## §5 预算与前置（不预先声称跑得起）

- 两臂续训：单臂 60k 符号的墙钟实测过 `396.803 s`／`334.638 s`（从头档，同一语料 108327171 字节）⇒ 续训档按同量级报价，
  起跑前做一次提交内存预检。
- 四张 after 面件（2 臂 × cap0＋replay）：零训练、CPU；`--limit 24` 档沿用 N2 那批面的形状。
- **训练跑属改权重** ⇒ 回 owner 批（本件只冻判据与命令面，不起跑）。owner 批准前，本件的落点就是「判据已冻、可取数性已修复」，
  不是「保持侧已判」。

## §6 发表资格前置（末尾三行）

- 预期为绿的成员：G-N5d-1/G-N5d-2（同源与单变量核对）＋ J-N5b-1（在场性五枚键，现行仪器已能出 `present`）。
- 仍红时收回的句子：「N5 保持侧有读数」「`J-N5b-6` 已判」；届时点名是哪一支守卫（G-N5d-1 或 G-N5d-4）拦的，并登记缺陷入 05。
- 发表资格前置：四张 after 面件与两份判读件同批入库，且引用本件时必须带「续训双臂、before 复用 20261008 件」这句范围限定。
