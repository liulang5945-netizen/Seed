# PLAN-N5-03 判据升版：给 `ruler_usable` 补机械公式（取代 PLAN-N5-02 的 J-N5b-4 半句）

- 性质：**判据升版**（不是追改）。`PLAN-N5-02` §2 J-N5b-4 写了"必须先出版 `ruler_usable` 为真"，
  但**没给算法**——判读器因此只能把它标成 `undefined_in_frozen_prereg_needs_version_bump`
  （㊵-599）。缺公式的判据等于没判据：谁都能事后把它读成想要的意思。
- 本篇只补这一条公式与它的裁定映射，不动 J-N5b-1/2/3/5/6，也不动过线界 `+0.02`。
- 旧件不删：`PLAN-N5-02` 保留原样，其 J-N5b-4 的"先出版 `ruler_usable`"这半句由本篇接管；
  索引里两条并存，读 J-N5b-4 时以本篇的公式为准。

## 1. 公式（只用噪声带与过线界，**不看 delta 的符号或大小**）

- `band_treated` ＝ 治疗臂自己的相邻段最大跳幅（由 `adjudicate_taiji_n3a_scaling_probe.py`
  的 `acc_noise_band_adjacent_max` 原样给出，本器不重算）。
- `band_control` ＝ 对照臂同量。
- **定义**：`ruler_usable := (band_treated > 0) and (band_control > 0) and (line >= max(band_treated, band_control))`。
- 为什么是这个形状：一把尺能不能回答"是否超过 `line`"，取决于**门槛是否压得住尺自身的抖动**；
  这与被比的两臂差多少无关。所以公式刻意**只吃带与线**，不吃 `delta` ⇒ 不可能被用来
  把已经看到的 `delta=-0.0004000000000000002` 救成"成立"或压成"不成立"。
  形状沿用本仓既有的"斜率 vs 自取噪声带"判据（N3 甲那一支），不是新造。

## 2. 裁定映射（三条都写死，不许滑）

- `ruler_usable == False` ⇒ `j_n5b_4 := ruler_unusable`，**这次比较作废**：
  既不写"成立"也不写"不成立"，只写"这把尺答不了这个问题"，并点名是哪一臂的带把线吞了。
- `ruler_usable == True and delta > line` ⇒ `holds`。
- `ruler_usable == True and delta <= line` ⇒ `not_holds`；
  若同时 `abs(delta) < max(band_treated, band_control)`，必须**追加一句**
  "该差值落在噪声带内"，说明这次跑**分辨不了这个量级的效应**——
  这是仪器分辨率的陈述，**不是**"影子无效应"的陈述（两者不能互换）。

## 3. 落地要求

- 判读器 `adjudicate_taiji_n5_shadow_gate.py` 实现该公式，`metric_lane` 里把
  `ruler_usable` 从字符串标记换成布尔值，并新增 `bands`／`within_noise_band`／`verdict`；
- 契约测必须覆盖三支（unusable／holds／not_holds）＋"带吞线"这一支能为假，
  数值用手推的小整数夹具，不抄真跑输出；
- 真跑数据只被读进来算，不得反过来决定公式参数。

## 4. 现有真跑数据代入（一次算完，写死在这里）

`line=0.02`、`band_treated=0.00225`、`band_control=0.00205`、`delta=-0.0004000000000000002`
（前三枚取自 `reports/taiji_n5_pair_metric_source_20261009.json`，`delta` 取自配对判读件）：
⇒ `ruler_usable = True`（线压得住带），`delta <= line` ⇒ **`not_holds`**，
且 `abs(delta)=0.0004 < max(band)=0.00225` ⇒ 必须附带"差值落在噪声带内、这次跑分辨不了该量级"这句。
**不得**写成"影子无贡献"；合取项 J-N5b-5（保持侧）仍未跑，`j_n5b_6` 照旧是"不可判"。
