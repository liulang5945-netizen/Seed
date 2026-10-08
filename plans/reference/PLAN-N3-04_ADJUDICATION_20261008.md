# PLAN-N3-04 判读 · 乙步骤二（出厂系装配的在线压强面）——2026-10-08 跑完并入库

> **判读对象**＝[PLAN-N3-04](PLAN-N3-04_developmental_assembly_prereg_20261008.md) §2 的合取（J-N3b-在场 ∧ J-N3b-阈）＋§4 的四条守卫；
> 链与命令形状按 §5quater（owner 2026-10-08 第四次弹窗裁"换出厂链跑"）。
> **一句话结论**：**在场性成立、阈值不成立**——四维里原本恒零的两维在装配上整场都在产出，`pressure` 的上限被打破
> （max 0.709197，越过默认 `minimum_pressure=0.70`），但**触发一次都没有**（`should_propose` 全 0、最长连续段 0）
> ⇒ 按 §2 已冻的"无动态范围则 τ 不得发表"，**本件不产出 τ**，那一格继续空着。
> 判读仪＝`count_taiji_n3_pressure_thresholds.py --prereg n3-04`（四支 rc=0），读数件＝
> `reports/taiji_n3_04_beta_gate025_20261008.json`、`..._beta_gate100_...`、`..._circuit_gate025_...`、`..._circuit_gate100_...`。

## 1. 两条主判据

| 判据 | §2 冻结口径 | 实测（四张面，各 3,999 条观测、覆盖率 1.0） | 判定 |
|---|---|---|---|
| **J-N3b-在场** | 打开的那一维 `zero_count/records < 0.05` 且 p90 > 0；丙＋乙臂额外要求 `activity_saturation` 同判 | `fast_slow_conflict` zero_share **0.00025**（3,999 里 1 个零）、p90 0.500128／0.500126／0.500315／0.500317；`activity_saturation` zero_share **0.0**、p50 恒 0.333333、p90 0.395833（circuit_gate100 为 0.385417）、max 0.5 | **成立（四张全过）** |
| **J-N3b-阈**（含动态范围证明） | `τ = ceil(p90(pressure)→0.05 档)`，且 `should_propose` 为真的**连续段 ≥ required_pressure_steps(=3)** 至少一次 | `p90(pressure)`＝0.668414／0.668136／0.675289／0.67194 ⇒ **τ 按规则算出 0.70**（恰等于默认 `minimum_pressure`）；但 `should_propose_total`＝**0**、最长连续段＝**0** ⇒ 动态范围证明**不过** | **不成立** ⇒ τ 不得发表（§2 原文那一支） |

`pressure` 的分布本身确实越过了旧上限：max ＝ **0.701569／0.705043／0.709197／0.709126**，四张面都摸过 0.70 之上。
仪器自己的两格也同步翻了：`two_signals_zero_throughout` ＝ **False**（步骤一那两张面是 True）、
`derived_pressure_ceiling_with_both_zero` ＝ **None**（旧档给 0.425）。
⇒ ㊵-484③ 那条"`pressure = 0.425·r ≤ 0.425`，所以默认阈算术不可达"的结论**在这档装配上不再成立**；
取而代之的新事实是：**上限够得着，但仍一次都不触发**（`default_policy_should_propose_share` ＝ 0.0，四张都是）。

## 2. 口径否证支（§2 第三条：这不是失败，是限制 τ 的来源）

| 面 | `control` 侧 | `abs_delta = |residual_error p90 − (1 − online_accuracy)|` | 判定 |
|---|---|---|---|
| beta_gate025 | 同链不开旗标臂（实测） | **0.211236** | 触发 ⇒ 与 accuracy 口径不同源 |
| beta_gate100 | 同上 | **0.211202** | 触发 ⇒ 同上 |
| circuit_gate025 | **无同链控制臂** | — | `missing_control_face` ⇒ `not_frozen_caliber_unchecked` |
| circuit_gate100 | **无同链控制臂** | — | 同上 |

⇒ 这条第三次触发（前两次＝㊵-484②），**"accuracy 0.59 ⇒ 残差阈 ≥0.4"那个口径在此链上仍不可用**。
后两格不是漏跑：**§4bis 只冻了一支默认关守卫臂**（落在 beta 链上），所以 circuit 那两支的口径对照今天做不了——
这是面数形状的一条缺口，登记在 §5。

## 3. 装配自述（§3 的五列都在场，机检为真）

| 列 | 读数 |
|---|---|
| `developmental_bundle_mounted` | 四张都为真；`bundle_fast_is_zero=true`（**挂载瞬间 fast 为零**，与 PLAN-N3-03 §3 那条一致——写入靠后面的 `fast_slow`） |
| `learning_mode` | 挂载后与收尾两次都读到 `fast_slow`（`mode_reapplied_after_load` 里**没有** `read_only` ⇒ 机检通过） |
| `bridge_gate` 请求值／读回值 | 0.25/0.25 与 1.0/1.0（四张都对得上；收尾 `bridge_gate_at_close` 同值） |
| `pressure_identity` | 判读器现算的最大偏差 ≤ 1e-16 量级（四张都过身份校验） |
| `coverage` | 全部 **1.0** |

## 4. 两档 gate 的差（机制定位用，不参与判级）

同一条 beta 链上 `0.25` 与 `1.0` 的差**极小**：`online_accuracy` 0.1690422605651413 对 0.16879219804951237、
`pressure` max 0.701569 对 0.705043、p90 0.668414 对 0.668136。
⇒ 在这条链这一预算上，**放行幅度不是压强的主控变量**；真正把分布顶起来的是"两维从恒零变成在产出"这件事本身。
链之间的差远大于 gate 之间的差：circuit 链 `residual_error` p50＝0.999107（zero_share 0.0235）对 beta 链 0.968711，
`pressure` p50 0.651833 对 0.639824。

## 5. 守卫与两条必须说出来的自我矛盾（**不在这里替自己改判**）

- **G-N3b-1（默认关＝八键逐位不变）按 §4bis 的字面口径：不过。** 实测同链开/不开旗标两支的差＝
  **3 个键有差、5 个键逐位同**：`online_accuracy` 0.21280320080020004 → 0.1690422605651413、
  `mean_surprise` 3.3646340823358925 → 3.729552795078888、`holdout_surprise` 2.9334548671532925 → 3.079834115409947；
  `ticks`/`window_ticks`/`base_ticks`/`ticks_at_exit`/`reached_budget` 五键逐位同。
- **而这条守卫与 G-N3b-2 在同一臂上互相矛盾**：G-N3b-2（㊵-483 收紧后的形状）明令"放行臂**不假设中性**，
  把差量出来并披露"；G-N3b-1 却要求同一臂"八键逐位相同"。⇒ 本件里**两条不可能同时满足**。
  我没有替自己挑轻的那条读法：字面读 G-N3b-1 ⇒ **整件不判**；按 ㊵-483③ 的精神读 ⇒ 上面的差就是该报的数。
  **本判读件按后一种读法把数报全，同时把这条矛盾登记为 DEBT-G51**，判据（§2 两条）不受它影响、
  且无论按哪种读法**τ 都不发表**（动态范围那一条是独立不过的）。
  修法（写在册里，不在本件里追改）＝守卫要**按旗标分档**：只读挂载旗标要求逐位同，改行为旗标要求"差必须被量出来并披露"。
- **面数形状缺口**：§4bis 的"4 张在线面＋1 支默认关守卫臂"没有为第二条链配控制臂 ⇒ 两支面的口径对照做不了（§2 表里那两格）。
  这不是可以靠"拿另一条链的数凑"补的（跨链搬＝N3-03 §4 的搬运禁令）。
- 其余两条守卫生效：G-N3b-3（全程不调 `propose_adaptive_residual_growth_candidate()`、不改产品 policy 常量）、
  G-N3b-4（五支面一律写 `output/n3_04/` 新档，`--progress` 显式给；`DEFAULT_CHECKPOINT` 与厂档未被覆写）。

## 6. 下一步（按 §2 的分流，不在本件里换次要指标）

1. **要回答"为什么够得着却不触发"**，缺的是**分项闸读数在场**：面件目前只自述 `minimum_pressure`／
   `required_pressure_steps`／`growth_resource_cost` 三项，`AdaptiveResidualGrowthPolicy` 的另外三道 `minimum_*`
   分项闸没有进件 ⇒ 本件**不指认是哪一道在拦**（不指认是纪律，不是偷懒）。
   ⇒ 另立一件：把六分项闸的逐条通过态写进面内自述，再跑同形状的对照。**那需要改的是仪器记录，不是判据**。
2. **τ 那一格仍空着**（§2 已冻："无动态范围 ⇒ τ 不得发表"），09 §2 N5 的"残差阈"仍不得从 accuracy 推。
3. N3 乙对 N5 的意义要重新表述：现在有的结论是**"装配能把四维打开、上限能破，但在 4,000 符号预算上触发数为零"**——
   这既不是"没接线"，也不是"接线就该生长"，而是一条**预算／触发尺度的不匹配**线索。N5 的唤醒调度因此仍不能开工。

## 7. 资产与门

- **面件已入库**（`output/` 那棵树曾被一次资产收束清掉 24 GB，所以原始面不留在里面）：
  `reports/taiji_n3_04_face_{beta_gate025,beta_gate100,circuit_gate025,circuit_gate100,beta_guard_off}_20261008.json`
  ＋同源退出记账 `reports/taiji_n3_04_exit_<臂名>_20261008.json`。字节数与 sha256 前 16 位（复算时按字节对）：
  `beta_gate025` 2,656,020 B / `fdc02a91cfc7fcad`；`beta_gate100` 2,656,098 B / `9d7d121ec0b10c51`；
  `circuit_gate025` 2,649,712 B / `73b6b1ee1f6cd9ca`；`circuit_gate100` 2,650,182 B / `2c89bf07b62f3d15`；
  `beta_guard_off` 2,555,745 B / `63fc864af45cfd1d`。
- 读数件 4 枚：`reports/taiji_n3_04_{beta,circuit}_{gate025,gate100}_20261008.json`（判读 rc 均 0）。
- 复算入口（一句话可跑）：`python scripts/training/count_taiji_n3_pressure_thresholds.py --pressure reports/taiji_n3_04_face_<臂名>_20261008.json --prereg n3-04`
  ⇒ 除口径对照那一段之外**全部可复算**（分布、在场性、动态范围、身份校验、覆盖率都不依赖外部件）。
  口径对照需要控制臂的进度 JSONL，它躺在 `output/n3_04/beta_guard_off/progress.jsonl`（`*.jsonl` 被 gitignore ⇒ **未入库**）；
  但它喂给判读器的那一个数已经进了读数件：`caliber_check.control_online_accuracy_last_line = 0.212803`
  （`one_minus_accuracy = 0.787197`、`residual_error_p90 = 0.998433`、`abs_delta = 0.211236`）。
  **若那枚 output 以后被清掉，重跑会得到 `missing_control_face` 而不是另一个数**——这条 fail-closed 形状是有意留的，
  所以口径对照的可复算性以"数在件里"为准，不以"文件在不在"为准。circuit 两支本来就没有控制臂，保持 `missing_control_face`。
- 判读器与旗标的契约测＝㊵-496 那 15 件（本轮未改 `.py`）。
- 门：仓根 `ruff check .`／`black --check .`（本轮未动代码，沿用 ㊵-496 的 0 error／1491 unchanged）、
  doc-sync 43/0/0＋hygiene 18/0/0 提交前实跑、判据措辞门对本件与新预注册 `status:ok`
  （08/09 各剩一处**既存**未钉"显著"，行号 1545 与 53，`git diff` 内零命中）；**不声称 M7-CI 绿**（推送被策略层拦下，命令已交回 owner）。

## 8. 不变项

M6 收官、M8 挂起；`DEFAULT_CHECKPOINT` 与厂档未动；`organs`/`learn` 产品默认位保持 False；
待 owner＝**DEBT-G49 的重建实施**（PLAN-N3-05 已冻验收式，动 `train_seed_corpus.py` 前须批）／
**分项闸读数的另一件预注册**（§6 第 1 条）／N1 H-S1b 训练机时／N2 第三件（保持集）；
台账行序以行首标号为准（㊵-419⑥）。
