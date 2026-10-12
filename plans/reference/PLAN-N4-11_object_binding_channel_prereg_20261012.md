# PLAN-N4-11 · 对象↔情节绑定通道（预注册：判据先冻、本件不开跑、动码排在冻件入库之后）

日期：2026-10-12 · 线别：N4 读取形态补全 · 批准：**七裁⑴ 选乙＝新建对象↔情节绑定通道（上限最高）**（台账 ㊵-745）
母件：[PLAN-N4-08](PLAN-N4-08_memory_kinds_and_delta_gate_prereg_20261011.md) §3（`J-N4g-2`／`J-N4g-3`）＋
[PLAN-N4-10](PLAN-N4-10_four_class_acceptance_adjudication_20261012.md) §0／§2（那两类的**不可判**实测）
缺陷本体：**DEBT-G122**——`EpisodicMemoryStore.retrieve`（`taiji/episodic_memory.py:87-129`）只对 `record.cue` 打分，
`object_ids`（`taiji/contracts.py:1956`）虽在合同里却从不进检索，于是"换名字还认得"这件事**无法被表达**。

## §0 本件要修的是"载体"，不是"读数"

㊵-744 的读数已经证明：把名字与真值一起改 ⇒ 两侧 `wrong_top1_rate` 同为 0.315、差 0.0；只改记录名 ⇒ 配对从 200 掉到 0。
所以今天那条"差为 0"**不是不变性被验过，而是那个字段没人读**。本件的目标不是再跑一次寻址，
而是让"对象"成为检索能看见的第二种键——这样 `J-N4g-2` 才第一次有能为假的测法。

**范围收在一句话里**：只做**对象身份**的绑定通道（用合同里已有的 `object_ids`，**不加任何新载荷键**）。
属性位（`J-N4g-3` 要的那个"可读属性"）本件**不兑现**，理由与后续形状写在 §4。

## §1 实施形状（三条硬约束）

1. **索引是派生物，不是持久物**：`EpisodicMemoryStore` 维护 `object_index: dict[str, tuple[str, ...]]`
   （对象标识 → 该对象的 `memory_id` 列表），在 `write()` 里从 `record.object_ids` 现建、在
   `from_checkpoint()` 里对每条记录重放同一逻辑。**不许进 `to_payload()`／`checkpoint()`**——
   加持久键会扩迁移面（㊵-729 批次 B 的教训：230 枚在册 `.pt` 的载入半径），也会撞上
   "披露键被别线做精确相等比较"那条坑。
2. **默认位逐位不动**：`retrieve(...)` 新增的关键字参数 `object_id=None`，为 `None` 时**走今天这条码路**；
   产品默认链的挂载位 `episodic_memory_default_mount` 仍为 `False`（`taiji/config.py:348`），
   `tests/taiji_native/test_n4_05_product_default_mount_contract.py:77,80` 那两枚断言**零改动仍须为绿**。
3. **未绑定不许伪装成空库**：`object_id` 给了一个没有绑定的名字 ⇒ **响亮抛错**（`ValueError`，消息点名是哪个名字），
   绝不返回 `()`。这条是 DEBT-G58 同族的硬规矩——"零命中"与"这个键不存在"在读取面上必须是两件事。

## §2 判据（冻五条，每条写明"什么情况下它为假"）

* **J-N4h-1 默认逐位中性**：同一份库、同一批 cue，改动前后不带 `object_id` 的 `retrieve` 必须返回**逐位相同**的
  `(memory_id, score)` 序列，且 `sorted(checkpoint())` 键集与改动前**完全一致**。
  为假的情形＝任一位不同，或键集变大（那说明我把索引写进了持久面）。
* **J-N4h-2 绑定被走到**：给定 `object_id` 时，命中集合必须是该对象绑定集合的**子集**，且当该对象有多条记录、
  其中一条 cue 最接近时顶位必须是它——**且同一条 query 在不带 `object_id` 时能拿到别的对象**，
  否则这支测不到"过滤真的发生了"。为假的情形＝过滤被忽略（返回集合等于全库）。
* **J-N4h-3 重命名可判**：`rename_object(old, new)` 之后——按新名检索必须得到与按旧名检索**同一批 `memory_id`**；
  按旧名检索必须走 §1.3 的响亮出口；且**记录本身逐位不变**（只动索引，不动 `memory_id`、不动 `cue`）。
  为假的情形＝改名要靠重写记录（那就不叫"同一对象换名字"）。
* **J-N4h-4 老判据换载体后必须能出值**：`J-N4g-2` 在绑定通道上用 §1 的形状重跑一次，必须从
  `not_applicable_no_independent_object_identity` 变成**可判**（`holds` 或 `not_holds`，方向由读数决定，本件不预设）。
  为假的情形＝仍判 `not_applicable`（那说明通道没被 `J-N4g-2` 用到，我造了个没人消费的索引）。
* **J-N4h-5 属性位仍然不在**：本件必须机检"情节记录依旧没有属性字段"并且结论里**不许**出现
  "属性交换已可判"。为假的情形＝有人顺手加了属性位却没走 §4 的第二批（那会同时踩迁移面与"两类一起过"的口径）。

## §3 守卫（两条，各带负对照）

* **G-N4h-1 索引不落持久面**：`checkpoint()` 与 `to_payload()` 的键集必须与改动前逐位相同；
  负对照＝把 `object_index` 塞进 `to_payload()` 后这支必须红（同族形状见 `test_n5_30` 的 payload 那支）。
* **G-N4h-2 响亮出口不许被吞**：对未绑定对象的检索必须抛 `ValueError`；负对照＝把该分支改成
  `return ()` 之后，同一支测必须红（不许让"空"成为合法答案）。

## §4 明确不做的（避免"看起来完成了"式收缩）

* **属性交换不在本件**。属性要成为可检索的东西，得先有一个**值位**：或加 `EpisodicMemoryRecord` 的可选字段
  （要按 DEBT-G99 的可选键回填形状，且必须配"旧 payload 缺键不炸"的正反两支），或另立 `attribute_index`
  把属性绑到对象而不是绑到记录。两者都比本件大一格，**排 #81，另立冻件**。在那之前 `J-N4g-3` 保持
  `no_field_to_swap`，`09 §2` 的 N4 出口也**不许**被写成"§3.3 四类已过"。
* 不碰产品默认挂载位；不碰 `Taiji`／`EpisodicField` 那条内部情节场（那是 N5 的生成面，不是寻址面）；
  不为让 `J-N4g-2` 变绿而改它的冻结文本——要改的是**载体**，判据原文照旧。

## §5 判据载体在哪（措辞门要求的一节）

执行体＝`taiji/episodic_memory.py`（新增 `object_index`／`rename_object`／`retrieve(object_id=...)`）；
判据载体＝`tests/taiji_native/test_n4_10_object_binding_channel_contract.py`（号现取：`test_n4_*` 用到 09）；
`J-N4h-1` 的逐位比较必须在**同一份库、同一批 cue** 上做改动前后两跑，摘要取法复用
`scripts/training/measure_taiji_n4_addressing_surface.py` 的 `measure()`，不许另写一份余弦；
`J-N4h-4` 复用 `scripts/training/adjudicate_taiji_n4_four_class_acceptance.py` 的 `J-N4g-2` 那一支，不新建第二把尺。

## §6 落笔即钉的三行

* **预期绿**：`J-N4h-1`（默认逐位中性＋键集不增）、`J-N4h-2`（过滤能被区分出来）、`J-N4h-3`（改名不动记录）、
  `J-N4h-5`（属性位确实没有）、`G-N4h-1`／`G-N4h-2` 两支各带负对照。`J-N4h-4` **不预设方向**，只要求出值。
* **仍红时收回哪句**：若 `J-N4h-1` 红 ⇒ 收回"默认不动"，本件回退（新增参数必须默认中性，这是硬前置）；
  若 `J-N4h-2`／`J-N4h-3` 红 ⇒ 收回"对象成为第二种可寻址键"，DEBT-G122 不许结清；
  若 `J-N4h-4` 仍判 `not_applicable` ⇒ 本件全部结论降级为"造了个没人消费的索引"，并把 `J-N4g-2` 的执行体改回来。
* **发表资格前置**：本件全绿只授权"重命名这件事第一次可测"这一句；
  在 `J-N4h-4` 出值之前，「§3.3 对象重命名验收已过」「记忆能保持同一对象」两句都不许说；
  即便出值，也只覆盖 `J-N4g-2`，**不覆盖属性交换**（§4），更不构成 N4 整行收口（S3 轨迹态仍未实现）。


## §7 落地记录（2026-10-12 ㊵-746；§1–§6 原文一字不动）

* **判据逐支的现值**（读数件 `reports/taiji_n4_33_four_classes_20261012_after_binding_channel.json`，
  `wall_clock_ms` 5558.833；上一件 `..._20261012.json` **原样封存未改**，因为它被 §2／§3 逐值引用）：
  `J-N4h-4 = holds`——绑定通道的四个数在件里并排出版：改名前后命中序列同为 `['m-0']`、
  重绑记录数 `1`、`old_name_still_resolves=false`（旧名走响亮出口），
  而**冻面（ruler）的结论没有被悄悄改写**，`classes.rename.verdict` 仍是
  `not_applicable_no_independent_object_identity`——一个键说通道可判、一个键说旧面仍无载体，
  不许合并成一句。
* **`J-N4h-1` 的"改动前后"落在已封存件上**：契约册拿**改动前那份读数**当参照重跑判读器，
  三档样本量的 `base/treated_wrong_top1_rate` 逐值相等、`distractor`／`slot_full`／`attribute_swap`
  三支 verdict 相等 ⇒ 默认码路（不带 `object_id`）没被这次产品改动碰动。
* **`G-N4h-1` 钉住了持久面**：`checkpoint()` 的键集被钉成 7 枚实名
  （`capacity`／`cue_dim`／`expired_memory_ids`／`evicted_count`／`format`／
  `implicit_provenance_count`／`records`），`object_index` 不在其中；记录 payload 的 14 枚键同样钉死，
  `attributes` 不在里面（`J-N4h-5` 的机检形）。
* **一处形状更正（不是放松）**：`J-N4g-3` 那支原来的正向对照写成「write 比 retrieve 多抓到字段」，
  绑定通道落地后 retrieve 也开始引用 `memory_id`，那一支按字面变成空差集。
  现改成两枚各自能为假的实测：提取器必须在 retrieve 里认出 `cue`、在 write 里认出 `memory_id`，
  另加一枚 `object_ids_referenced_by_retrieve=false`（打分仍不看对象字段）。
  同一处更正要求 `PLAN-N4-10` §2 那句 `fields_referenced_by_retrieve = ["cue", "episode_id"]`
  就地打过期注——已按「原文不删＋带日期追加「的规矩做，见该件 §2 末注。
* **测到的面**：`tests/taiji_native/test_n4_10_object_binding_channel_contract.py` 九支＋
  `tests/taiji_native/test_n4_09_four_class_acceptance_contract.py` 十支＝19 passed（24.43s）。
  九支里含三条响亮出口的负对照：未绑定对象名必须抛、改名不许合并两个对象、
  淘汰与撤销都要解绑（否则绑定会指向库里已不存在的记录）。
* **本件没做到的那一半（照 §4 继续欠着）**：`J-N4g-3` 属性交换仍是 `no_field_to_swap`，
  因为记录里仍没有值位。那一半排 **#81**，形状要么加可选字段并按 DEBT-G99 的旧 payload 回填，
  要么另立 `attribute_index` 把属性绑到对象；两条都动 `taiji/`，本格不越。
