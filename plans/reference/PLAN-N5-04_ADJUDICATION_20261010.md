# PLAN-N5-04 判读（2026-10-10 ㊵-654）：双臂续训跑完——`J-N5d-1` 按冻结出口收回，代价归因排除一枚变量

> 判读对象＝`--n5-shadow-gate` 单变量双臂续训（[PLAN-N5-04](PLAN-N5-04_retention_rebase_prereg_20261009.md) §2/§3 冻结的形状）。
> 本件不推翻任何已冻判据；它按 §3 G-N5d-4 与 §4 预设的出口**收回句子并点名拦阻者**。
> 台账：[08 ㊵-654](../../active/roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md)；债：[DEBT-G84](../../active/roadmap/05_TECH_DEBT_REGISTER.md)。

## §0 一句话结论

**两臂跑完了，但这两臂不是两臂**：它们的产品自述里 `n5_shadow` 整块缺席、权重 238/238 逐位相同，
所以 `J-N5d-1`（保持侧主判据）**没有可发表的读数**——按 §4 预先指定的出口，收回"N5 保持侧有读数"
这句，点名拦阻者是 **G-N5d-4**（不是 G-N5d-1，两条都查过，见 §3）。

同时这一跑**并非白跑**：它把"保持代价来自影子喂料"这个候选解释**排除**了——七列的下跌发生在两臂
共有的变量上（同基座＋同语料＋273 tick 续训）。这句是本件唯一发表的能力侧归因。

## §1 跑的形状（复算入口）

| 项 | 治疗臂 | 控制臂 |
| --- | --- | --- |
| 基座（`--resume`） | `checkpoints/seed_a31self_with_circuit.pt` | 同 |
| 落点 | `output/n5d_rebase_treated/checkpoint.pt` | `output/n5d_rebase_control/checkpoint.pt` |
| `.pt` 摘要（前 16） | `be3e16e433b308ab` | `7d4fb08090746950` |
| 唯一变量 | `--n5-shadow-gate 1.0` | 该旗标缺席 |
| 窗口 | `--max-symbols 60000`／273 tick | 同 |
| 完成记账墙钟 | `450.29400739981793 s` | `453.1858198000118 s` |
| 压强面 | 9,729 行（9,727 条带决策读数） | 同 |

单变量核对由跑道器的 `lineage_refusals` 执行并出版 `lineage_ok`（摘掉逐臂落点旗标与那枚门旗标后
argv 逐位相同；两臂落点不许指向同一档）。

## §2 七列读数（两台判读器各自抄录，未改动判读器）

| 列 | before | 治疗臂 after | 控制臂 after |
| --- | --- | --- | --- |
| `cap0:E` | 1 | **0** | **0** |
| `replay_strict_hits:1.0` | 6 | 6 | 6 |
| `replay_strict_hits:2.0` | 6 | 6 | 6 |
| `replay_well_formed:0.0` | 24 | **21** | **21** |
| `replay_well_formed:0.5` | 24 | **20** | **20** |
| `replay_well_formed:1.0` | 24 | **21** | **21** |
| `replay_well_formed:2.0` | 25 | **23** | **23** |

两臂 verdict 都是 `cost_persists`（7 列比对、5 列下跌），同源守卫 `G_N2c_3` 两臂都 `ok`。
件＝`reports/taiji_n5d_rebase_treated_pair_20261010.json`／`..._control_pair_20261010.json`。

**但这张表两列同值这件事不是"结论稳健"，是"只有一枚权重"**：见 §3。

## §3 拦阻者点名（三条独立读数，不靠单一把尺）

1. **逐张量比对**：两档 `torch.load` 后按张量名比内容 ⇒ `DIFFERING = 0 of 238`。`.pt` 摘要之差
   只住在非张量信封（计数器／路径／时间戳）。按 [[checkpoint-entry-names-not-portable-across-save-shapes]]
   这条尺只在"同一路径产出的两档"间有效——本件两臂正是同一路径同一形状，故成立。
2. **提议从未发生**：两臂压强面 9,727 条带 `decision_should_propose` 的行里真值 **0**；
   `decision_pressure` 上界 **0.5571776111896318**（两臂同一数），既够不到本次显式给的
   `--growth-min-pressure 0.65`，也够不到默认 `minimum_pressure=0.70`。
   `adaptive_shadow` 只在 `should_propose` 为真的观测里 materialize（`train_seed_corpus.py:851-861`）
   ⇒ 影子支整场不存在 ⇒ `--n5-shadow-gate 1.0` **没有作用对象**。
3. **产品自述在场性**（现成尺 `adjudicate_taiji_n5_shadow_gate.py`，本件一字未改）：两臂
   `j_n5b_1 = ran_not_measured`、`missing_self_report_keys` 五枚齐；`pairing.status =
   pairing_invalid`、`reason = identical_requested_gate`（`treated_requested` 与
   `control_requested` 同为 `null`）；`j_n5b_6 = not_adjudicable_until_2_3_4_5_are_all_measured`；
   rc=2。件＝`reports/taiji_n5d_rebase_shadow_gate_20261010.json`。

⇒ **G-N5d-4**（"计数器没回来之前『影子确曾被喂过』这句不许发表"）为拦阻者；
⇒ **G-N5d-1 未拦**（`preflight.json` 三枚键全 `verified`，`G_N2c_3` 两臂 `ok`）；
⇒ **G-N5d-2 未拦**（argv 确实只差那一枚旗标）——这暴露的正是本件的债：**单变量核对通过
不代表该变量有作用面**（DEBT-G84）。

## §4 合取与后续判据状态

- `J-N5d-1`＝**无读数**（收回"保持侧已判"）。
- `J-N5d-2`（学习侧收益，沿用 +0.02 线／`ruler_usable` 公式）＝**未取数**——本件跑的是保持侧四张面，
  增益面没有作为 `--face-treated`／`--face-control` 线进去，那台仪器照实报
  `j_n5b_2 = unverified_missing_face`（不是"收益为零"）。
- `J-N5d-3`（合取）＝`not_adjudicable`，由 §3-3 那台仪器的 `j_n5b_6` 直接给出，本件不另算。
- **配对有效性不是判据而是守卫**：G-N5d-2（单变量）**通过**、G-N5d-4（自述计数器）**拦下**——
  这一对"守卫过、判据无对象"的组合就是 DEBT-G84 的内容。
- **可发表的正面读数只有两条**：
  ①273 tick 续训在 `with_circuit` 基座上造成 5/7 列下跌，且**不来自那枚旗标**（旗标无对象）；
  ②保持集与主训练语料分离：`data/simple_zh/dialogue_extended_clean.jsonl`（108,327,171 B）上
  `windows_found_in_corpus 0/116`、`hit_rate 0.0`、`skipped_short_rows 35`，两臂同值
  （件＝`reports/taiji_n5d_rebase_*_pair_maincorpus_20261010.json`）⇒ DEBT-G49 那族污染在这一档
  不存在，下跌不可用"题集本就在训练语料里"解释。

## §5 本轮把什么变成了件里可查的东西

| 原先只存在于记忆里 | 现在的机械执行 |
| --- | --- |
| "分离机检的材料＝本次通电自己的夜间件"（㊵-565③） | `consolidation_provenance` 按夜间件名时间戳 vs 该臂 `progress_exit.json` 运行窗口判 `arm_local`／`substituted`／`unknown`；`substituted` 需 `--allow-substituted-consolidation-material` 才放行，并出版 `<out-dir>/consolidation_material.json` |
| "影子支有没有被走到" | 跑道器新增 `gate:` 步＝把 §3-3 那台**现成**仪器排到四张面之前；实测 rc=2 时一条判读步都没跑（不付 2×13 分钟） |

测＝`tests/taiji_native/test_n5_16_retention_lane_runner_contract.py` **23 passed**（原 18），
含"未点名材料不许被读成对"与"三档互斥"两支。

## §6 下一格（两条，性质不同，不许合并）

- **我这侧、零长跑、零产品码**：拆开 §3-3 那条同形——训练器自述里**无条件**写
  `shadow_gate_requested`，且**同一批**给判读器补"在场 ≠ 被走到"那一支
  （只补前者会让 `j_n5b_1` 变绿而臂仍然是塌的＝一个更危险的假通过）。PLAN-N5-04 §3 的 G-N5d-4
  届时带日期补一句限定，不动原句。
- **等 owner**：要让 `J-N5d-1` 真有读数，前置是使提议真发生——这与 N3 乙线的 τ 口径是同一件事的
  另一头（见 09 §3.2 第 3 条），或另立一件把影子支从 `should_propose` 解耦。两者都要花机时或动
  训练器／产品语义。**在本格之前，"再跑一次双臂"是重复支付**，不做。
