# PLAN-N4-06 设计预注册：产品档寻址尺的扰动强度与"尺可用"判据（判据先冻，不开跑）

- 承接：PLAN-N4-05 J-N4e-4/5；㊵-606② 的实测——同一批产品档材料（120 条、`cue_dim=32`、
  由 `settle_action` 写入）在现寻址面上给出 `wrong_top1_rate=0.0`、`block_means=[0,0,0,0,0]`、
  `noise_band_adjacent_block_max=0.0`、`ruler_usable=false`、`top1_score_mean=0.947025629132986`。
- 诊断（不含结论）：现行查询扰动是"把 cue **末位**换成相邻条的末位"——为 32 **字节**档设计的形状。
  在 32 维连续认知向量上，改 1 个分量几乎不改余弦，所以"错名册第一位"这件事**结构上无从发生**。
- 性质：判据冻结。本篇不产读数、不改码；事后只许就地日期更正或升版。

## 1. 扰动定义（唯一、可复算、自报幅度）

- 对每条查询：取另一条记忆的 cue 方向 `d`（**donor 与本条不同名册条目**），把查询 cue 朝 `d`
  旋转固定角度 `θ`（Gram–Schmidt 于本条正交补上），得到 `cue'`；
- 面件对每条查询出版**实际达到的角度** `achieved_angle_rad` 与 `cos_before`／`cos_after`，
  不许只报名义 `θ`；
- 两档同时跑：`θ_small`、`θ_large`（见 §2 的取值与理由），**同一材料、同一库、同一 limit=3**。

## 2. `θ` 的取值与"为什么不是事后拟合"

- `θ_small = 0.35 rad`（`cos≈0.9394`）、`θ_large = 0.70 rad`（`cos≈0.7648`）。
- 取值理由**只依赖已发表的一条几何事实**：`㊵-606②` 的 `top1_score_mean=0.947`。
  两档跨过它，意味着"如果尺有效，错名册必须有可能发生"。
  这是**对扰动能不能够到相似度的判断**，不是对结果方向的判断——两档结果无论谁高谁低都可发表。
- 若有人此后按结果回头改 `θ` ⇒ 必须升版并在索引里标本篇作废（本仓既有先例：`J-N3a` → `J-N3a′`）。

## 3. 判据（五条，先冻）

- **J-N4f-1（在场性）**：面件必须出版 `faces_provenance`（㊵-607 那枚）、
  `perturbation_angle_rad`、`achieved_angle_rad` 列表、`cos_before`／`cos_after` 分布、
  两档各自的 `wrong_top1_rate`／`block_means`／`noise_band_adjacent_block_max`；
  缺任一 ⇒ `rc=2` 记 `ran_not_measured`，不发表任何寻址质量陈述。
- **J-N4f-2（尺可用的定义，沿用 PLAN-N5-03 的形状）**：
  `ruler_usable := band_small > 0 且 band_large > 0 且 |wrong_top1(θ_small) - wrong_top1(θ_large)| ≥ max(band_small, band_large)`；
  不满足 ⇒ 判 `ruler_unusable`，**两档都不许发表优劣**，只许写"这把尺答不了这个问题"。
- **J-N4f-3（方向性预注册）**：若尺可用，**且** `wrong_top1(θ_large) > wrong_top1(θ_small)`
  ⇒ 判 `angle_sensitive`（尺随扰动变差，说明它在测"能不能找回来"）；
  反向或无差 ⇒ 判 `not_angle_sensitive`，此时**即使数值好看也不得**宣称产品档寻址有效。
- **J-N4f-4（跨档禁令不变）**：产品档与 harness 档（`0.06`）cue 形状不同 ⇒
  **禁止**比两档数值或据此说"更好/更差"（J-N4e-4 重申）。
- **J-N4f-5（写入溯源）**：本次面件的 `episodic_writes` 必须 > 0 且
  生成件里"自己写库"的调用点数按 **AST** 计数为 0（grep 会被文档提法假命中，㊵-606⑥）。

## 4. 失败出口

- 尺不可用 ⇒ 结论写成"当前产品 cue 形状下，余弦寻址尺在 120 条规模上分辨不了扰动"，
  并回 05 立债；**不许**改用第二种相似度量直到出数（那是换指标救结论）。
- 若 `achieved_angle_rad` 与名义 `θ` 偏差 > 0.02 rad ⇒ 扰动实现有缺陷，先修实现，不出读数。
- 本篇通过后，`J-N4d-3` 的"三条件同真"才第一次可能被满足（材料档=product、写入>0、尺可用）。
