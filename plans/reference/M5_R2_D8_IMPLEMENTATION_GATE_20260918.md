# M5 R2-D8 实现门报告（§2 fixture v3 + §3 门）—— 已通过

日期：2026-09-18。状态：**实现门通过**；probe 正在运行（train-only）。
合同：[M5_R2_D8_SCALE_CONTRACT_FROZEN_20260918](M5_R2_D8_SCALE_CONTRACT_FROZEN_20260918.md)。
启动依据：用户指令「按照计划结合实际开发阶段持续推进项目」（合同 §6 要求"启动条件：用户指令"）。

## §1 实现方式（最小侵入，可审计）

**未另建 builder**，而是在冻结的 `scripts/training/build_taiji_r2_d1_measurement_fixture.py`
中**新增 `--variant v3` 分支**，复用同一套 `_records_for_split` / `_deterministic_order` /
`_verify` / `reference_answer`：

- `TRAIN_OBJECTS_V3`（32 个对象）、`TRAIN_COLORS_V3`（6 单字 + 6 六字节双字）；
- `_assert_v3_pool_clean()`：新字符**不在冻结词表**内 + **train 值字 ∩ dev∪final 值字 = ∅**
  + 无重复 + train 对象不复用 dev/final 词；
- `_records_for_split(..., objects_override=None)`、`_verify(..., objects_by_split=None)`
  —— **新增参数默认 None ⇒ v1/v2 行为逐位不变**。

## §2 **机制零改动证明（两项，均已通过）**

1. **v1/v2 逐字节重放**：把 v1、v2 重新生成到临时路径，与**封存 fixture** 比对
   ⇒ `byte-identical-to-sealed=True`（两个 variant 均是）——**证明本次改动未影响既有行为**；
2. **D7 T1 重放**（合同 §3.3）：`probe_taiji_r2_d7_char_token.py --arm T1` 在 **v2 train**、
   seed 20260917 上重放，输出到临时目录（不覆盖 D7 冻结报告）：

| 量 | 重放值 | D7 记录值 |
|---|---|---|
| `copy_supported_M1` | **1.0** | 1.0 |
| multibyte / singlebyte | **72/72 / 72/72** | 同 |
| copy 概率（intact / misbound） | **1.0 / 0.0696** | 同量级 |
| **`bias_end`** | **0.17751** | **+0.178** |
| 全门 | **true** | passed |

⇒ 读取路径与 char-v1 机制**未因本次参数化改动而偏移**。

## §3 fixture v3（合同 §2 + §3.1/§3.2）

| 项 | 值 |
|---|---|
| fixture | `tests/fixtures/r2_d1_measurement_v3.jsonl`（sha256 `bdbaba2f…`）|
| 数据报告 | `reports/r2_d1_data_contract_v3_20260918.json` |
| `corpus_digest` | `a73703de2759491c923066e24beac56f…` |
| **train** | **1278 episodes**（unique_pairs 1278）|
| train 形状 | fact 384 / negation 384 / same_opening_fact 384 / unknown 32 / same_opening_unknown 32 / combo_same 31 / combo_diff 31 |
| train 颜色 | 黄 青 紫 金 棕 褐 + 琥珀 珊瑚 翡翠 玛瑙 琉璃 玳瑁（**6 单字 + 6 双字**）|
| **dev / final** | 98 / 94 —— **与 v1 逐字节相同**（`identical=True`，两次独立比对）|
| 门 | **11/11 通过**（含 `entities_disjoint_across_splits`、`pair_structure`、`reference_self_check`）|

## §4 ⚠️ 与合同的一处数值差异（**显式登记，未擅自改动规则**）

合同 §2 写「行数 174→**1246**（值行 32×12×3=1152 + unknown 2×32=64 + combo 相邻对 31×2=62）」，
但其自列构成之和为 **1278**，与所写的 1246 **相差 32**。

⇒ **处置**：优先遵守合同更根本的不变量——「形状/模板/train 句式/翻转对结构/确定性顺序规则
全部不变」；因此**按规则生成，实际 train = 1278**，**未为凑 1246 而删行**。
本差异为**合同内部算术笔误**，不影响"唯一变量 = train 组合放大"这一设计
（放大倍数由 174→1278，即 **7.34×**，合同正文的 7.2× 亦据 1246 推算）。

**若用户认为应以 1246 为准**，需裁定"删哪 32 行"——那会引入一个**规则外的自由度**，
本报告不建议；此处仅登记事实。

## §5 下一步（本合同内自动推进）

**§4 probe**（v3 train + T1，seed 20260917、30 epochs、lr 0.01、microbatch 8）正在运行；
通过则进 **§5 matched dev（3 seeds）**，按 §5 预承诺路由裁决。
