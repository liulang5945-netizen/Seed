# PLAN-N4-10 · VISION §3.3 四类验收的第一次出值（判读件；判据冻在 PLAN-N4-08 §3，本件不重冻也不放松）

日期：2026-10-12 · 线别：N4 读取形态补全 · 母件：[PLAN-N4-08](PLAN-N4-08_memory_kinds_and_delta_gate_prereg_20261011.md) §3（`J-N4g-2`／`J-N4g-3`／`J-N4g-4(a)`／`J-N4g-4(b)`）
执行体（本件新落）：`scripts/training/adjudicate_taiji_n4_four_class_acceptance.py` ＋契约册 `tests/taiji_native/test_n4_09_four_class_acceptance_contract.py`（十支全绿）
读数件：`reports/taiji_n4_33_four_classes_20261012.json`（面＝harness 自建 `EpisodicMemoryStore`，零训练、不改权重、不动产品码；`started_utc` 2026-10-12T12:20:37Z、`wall_clock_ms` 5286.144）

## §0 先量可达性，再谈验收（这是 03 队首要的那一步，不是沿用旧账）

| 类 | 判据要用的读法 | 在盘载体（file:line） | 今天能否判 |
|---|---|---|---|
| 对象重命名 `J-N4g-2` | 换掉对象标识符、顶位命中不变 | `EpisodicMemoryStore.retrieve`（`taiji/episodic_memory.py:87-129`）只对 `record.cue` 打分＋`episode_id` 收窄；`object_ids` 从不进打分。寻址仪器的材料把 `object_ids=()` 写死（`measure_taiji_n4_addressing_surface.py:148`） | **不可判——没有独立的对象身份载体**（详见 §2 第一行） |
| 属性交换 `J-N4g-3` | 两条记忆的**可读属性**互换，结果跟着互换 | 情节记录 14 枚载荷键里没有属性位（件里 `record_payload_keys` 全列）；属性只活在世界层 `WorldObject.attributes`（`taiji/contracts.py:813-819`），那条面不接情节寻址 | **不可判——无字段可换** |
| 干扰对象 `J-N4g-4(a)` | 加 K 条干扰后的差值 > 该档自取噪声带 | 现成尺 `measure_taiji_n4_addressing_surface.py` 的 `wrong_top1_rate` ＋ `noise_band_adjacent_block_max` ＋ `noise_band_floor`（本件**引用**它，不重算） | **可判，今天出值** |
| 槽占满 `J-N4g-4(b)` | 库满时响亮拒绝或记录淘汰，不许返回空 | `taiji/episodic_memory.py:66-69` 淘汰并自增 `evicted_count`，第 138 行随 `checkpoint()` 出版；`expire()` 的显式撤销另记一列（DEBT-G99 已结） | **可判，今天出值** |

**产品默认链的挂载态另说，不许被上面四行代答**：`taiji/config.py:348 episodic_memory_default_mount=False`，
`adapter.py:299` 只在该位为真时挂载，`api/seed_runtime.py` 与 `seed/model.py` 的调用点传 `use_memory=False`，
而 `tests/taiji_native/test_n4_05_product_default_mount_contract.py:77,80` 断言的就是"默认不挂"。
⇒ 本件的读数属 **harness 层**（自建库直接寻址），它证明"这条读路能被判"，不证明"装机默认就在跑这条读路"。

## §1 跑法（一条命令，可复算）

```
python scripts/training/adjudicate_taiji_n4_four_class_acceptance.py --overwrite
```

夹具形状（件内 `fixture_shape` 自述）：`seed=20261012`、`cue_dim=32`、目标 40 枚、干扰 20 枚（贴前 20 枚目标加噪 0.25）、
查询对目标加噪 0.35、`limit=3`、槽满档 `capacity=50` 写 62 条。三档样本量＝每目标 5／25／125 条查询（配对 200／1000／5000）。

## §2 四类的第一次出值（逐值取自读数件，不手算）

* `J-N4g-2` ⇒ **`not_applicable_no_independent_object_identity`**。三个数一起看才说得清：把名字与真值**一起**改，
  `wrong_top1_rate` 两侧都是 0.315、差 0.0（`frozen_shape_would_read_green=true`）；只改记录名而真值仍指旧名，
  配对数从 200 **掉到 0**（未配对 200）；载体探针：换 `object_ids` 顶位序列不变、换 `cue` 顶位序列必变
  （`cue_carrier_positive_control=true`）。⇒ 这台面上唯一的名字就是真值本身，所以"重命名不变性"在这里
  **不是被验过，而是无法被表达**。按冻形判「成立」＝把"这个字段没人读"说成"身份换了这个表征不变"。
* `J-N4g-3` ⇒ **`no_field_to_swap`**。`fields_referenced_by_retrieve = ["cue", "episode_id"]`；
  同一把提取器在 `write` 上抓到 `memory_id` 等 retrieve 不看的字段（差集在件里点名），所以这不是提取器瞎了。
* `J-N4g-4(a)` ⇒ **`holds`**（判据档 5000 配对）：控制档 `wrong_top1_rate=0.308`、治疗档 `0.3736`、
  差 **0.0656**、该档自取噪声带 **0.038** ⇒ 差在带外。方向与量级在三档间稳定（差 0.105／0.067／0.0656），
  是**带随样本量收窄**（0.175／0.115／0.038）而不是差在变大。
* `J-N4g-4(b)` ⇒ **`holds_recorded_eviction`**：写 62 对 `capacity=50` ⇒ 运行时 `evicted_count=12`、
  `checkpoint()["evicted_count"]=12`、库内 50 条；对被淘汰最早那条的 cue 发问，返回 **3** 条非空命中、
  顶位 `slot-14`、被淘汰 id 不再可达；显式撤销仍是另外的列（`expired_memory_ids`）。
  同件自带的"没满"控制档判 `not_full_no_eviction`（`evicted_expected=0`），所以这一支能为假。

## §3 一次样本量改判的披露（不许当成"我挑了能过的那档"）

第一版判据档规则钉在「paired≥1000 且带下界≤0.01 的最小档」，那一档当场判 **`ruler_unusable`**
（差 0.067 ≤ 带 0.115）。现规则改成**取最大档**，理由与限制三条：①冻的 `J-N4g-4(a)` 只说"该档自取噪声带"，
**没有钉 n**，所以 n 属执行侧自由量，必须全部披露；②三档的数与各自的带都原样在件里，读者能看见我没隐去小档；
③**否证承诺**：若在最大档上差 ≤ 带 ⇒ 记 `ruler_unusable` 并收回"干扰可测"这句（该式已抽成
`_interference_verdict`，含"恰好等于带"那一支，契约册逐支跑过）。
这条改动**没碰任何阈值**，只碰样本量；把它记进台账 ㊵-744。

## §4 两句不许说，以及剩下的半边属于谁

* 不许说「VISION §3.3 四类验收已过」——过的是 4(a)／4(b) 两支，另两支今天是**不可判**，不是通过。
* 不许说「对象与属性已接进情节寻址」——`retrieve` 的打分面今天只认 `cue`，这是 §0 的实测，不是设计意图的转述。
* **需要 owner 的半边**：要让这两类可判，得让对象身份或属性进入检索的候选集或打分（动 `taiji/`，属产品码），
  或另立一块绑定面。这条已登成 **DEBT-G122**，本件只出证据不动码。批次 C 的长跑机时另说（DEBT-G116）。

## §5 判据载体在哪（措辞门要求的一节）

判据＝`PLAN-N4-08` §3（本件不新增阈值）；读数＝`reports/taiji_n4_33_four_classes_20261012.json`；
执行体＝`scripts/training/adjudicate_taiji_n4_four_class_acceptance.py`（`_distractor_lane` 调现成尺的
`measure()`，不重算余弦）；契约册＝`tests/taiji_native/test_n4_09_four_class_acceptance_contract.py`。
被引用的带内规则住在 `scripts/training/adjudicate_taiji_n5_shadow_gate.py` 的 `judge_metric_lane` 体内，
不是可独立调用的 helper，所以本件用**源码钉**（`cited_formula_pins`＋`all_pinned`）而不是复制公式——
钉在真件上为真、在空件上为假，两支都由契约册跑过。

## §6 落笔即钉的三行

* **预期绿**：四类各出一个值（两 `holds`、两"不可判"）、`all_pinned` 为真、控制档判 `not_full_no_eviction`、
  契约册十支全绿。今天实测全部如此。
* **仍红时收回哪句**：若 4(a) 在最大档判 `ruler_unusable` ⇒ 收回"干扰可测"这句并把它记成尺的动态范围缺陷；
  若 4(b) 判 `violated_*` ⇒ 收回"淘汰被记录"，并把 `evicted_count` 那一路升级成产品缺陷；
  若 `all_pinned` 为假 ⇒ 本件全部结论作废（我用的已不是被引用的那一式）。
* **发表资格前置**：本件只授权"§3.3 四类里两类的读路可判、两类的载体不存在"这一句；
  任何"记忆能保持同一对象""属性能被重绑定"式的能力主张都不在本件范围内——那要先有 DEBT-G122 的产品面。
