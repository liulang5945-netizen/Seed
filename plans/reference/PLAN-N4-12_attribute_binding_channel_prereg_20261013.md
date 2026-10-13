# PLAN-N4-12 属性位与第二可寻址维（实施预注册，判据先冻）

冻结时刻：2026-10-13（㊵-757）。批文＝owner 八裁⑴ 甲（"做成第二可寻址维"，动 `taiji/` 已批）。
执行前提＝㊵-756 ①那条锁的到期说明（对照臂已作为独立进程起跑，改源文件不再影响两臂）。
本件**只加一种可寻址维**，不主张任何表征层结论。

## §1 形状（写死，实现不许临场改）

1. 记录侧新增 **`attribute_ids: tuple[str, ...] = ()`**——`taiji/contracts.py:1942-1957` 那一族
   第五个 id 字段（`event_ids`／`assembly_ids`／`object_ids`／`relation_ids` 的同族），
   一律经 `_normalize_ids` 归一，条目写作 `"color=red"` 这种"名=值"字符串。
   **禁止**引入 `Mapping[str, str]`：它会破坏 frozen dataclass 的不可变语义，且需要一套新的归一器。
2. 载荷键必须**可选**：`to_payload()` 无条件写，`from_payload()` 用 `payload.get(key, default)` 读
   （照 `provenance`／`tick` 的先例，`contracts.py:1473-1486`）。历史 `.pt`／JSON 件不带这枚键时
   按空补，并且出版**披露计数**（照 `EpisodicMemoryStore.implicit_provenance_count` 的形状）：
   语义＝"本次载入时有几条记录没声明属性位"，只数现读的，**不许回填**成件里那个历史值。
3. 存储侧建**第二派生索引** `_attributes`／`_attribute_of`，与已落地的对象绑定通道同构
   （`taiji/episodic_memory.py`）：派生索引**不进 `checkpoint()`**，容量淘汰与显式 `expire()` 都要同步解绑。
4. 检索侧新增 `retrieve(..., attribute=None)`：给定属性名时先响亮拒绝未绑定的属性
   （DEBT-G58：零命中不等于"这枚键不存在"），再把候选集收窄到绑定过该属性的记录；
   与 `episode_id`／`object_id` 是**合取**关系，不许互相覆盖。
5. `rename_attribute(old, new)` 只动索引、不动记录，且**拒绝合并**（两枚属性同时在场时报错，
   与 `rename_object` 同形）。

## §2 判据（冻三条，每条写明"什么情况下它为假"）

* **J-N4g-3a 属性位在场且可读回**：构造一条带 `attribute_ids` 的记录，经 `to_payload()` → `from_payload()`
  之后属性集合必须逐项相等；旧载荷（不带该键）载入必须成功且 `implicit_attribute_count` 等于这类记录的条数。
  为假的情形＝载入抛错（说明键被写成了必需项，历史件当场拒载），或 `implicit_attribute_count` 是回填值而不是现数。
* **J-N4g-3b 默认码路逐位中性**：`attribute=None`（缺省）时，检索结果与改动前那份**封存读数**必须逐位一致
  （参照件＝`reports/taiji_n4_33_four_classes_20261012_after_binding_channel.json` 的读取面）。
  为假的情形＝任何一命中顺序或分值变化 ⇒ 说明默认路径被新索引影响，本件退回重冻而不是"把差异解释成无害"。
* **J-N4g-3c 属性交换可判**：两条记录的对象、cue、`episode_id` 全部不动，只互换它们的 `attribute_ids`，
  按属性检索的 top-1 必须随之互换；并且必须同时出版两枚探针：换 cue 后 top-1 必变（正向对照，证明读数不是坏的）、
  换 `attribute_ids` 后按属性检索必变。为假的情形**有两种且必须分开记**：
  ①互换后不变 ⇒ 判 `no_consumer`，并当场收回"§3.3 四维可判"那句；
  ②两枚探针都为"变"但按 cue 检索也变了 ⇒ 说明属性被塞进了打分，判 `contaminated_scoring`，本件作废。

## §3 守卫（两条，各带负对照）

* **G-N4s-1 属性不进打分**：`EpisodicMemoryStore` 的打分表达式里不许出现 `attribute`，
  判定＝AST 数 `retrieve` 函数体内以 `attribute` 为操作数的算术/相似度调用，必须为 0；
  负对照＝往同一把尺喂一段含 `score + len(record.attribute_ids)` 的合成源码，计数必须为 1。
* **G-N4s-2 持久面不被悄悄加宽**：`checkpoint()` 的键集必须与既有实名集合逐枚相等（多一枚就红）；
  负对照＝把 `_attributes` 塞进 `checkpoint()` 的字典，`test_n4_11` 那条必须红。

## §4 半径与谁能做

动 `taiji/contracts.py`（字段＋`to_payload`／`from_payload`＋披露计数）、`taiji/episodic_memory.py`
（第二索引、`retrieve` 参数、`rename_attribute`、解绑三处）、`tests/taiji_native/test_n4_11_attribute_binding_channel_contract.py`（新册）、
`tests/taiji_native/test_n4_10_object_binding_channel_contract.py`（payload 键集钉子与那句 `assert "attributes" not in RECORD_PAYLOAD_KEYS` **就地重钉**，留"旧清单自本注起作废"的日期注，不许删断言变绿）、
`scripts/training/adjudicate_taiji_n4_four_class_acceptance.py`（`J-N4g-3` 从 `no_field_to_swap` 换成真出值）。
判据先冻再动码；批文已给（八裁⑴ 甲）。**不做**：不改 `EpisodicField` 的张量宽度，不碰 `PROVENANCE_KINDS`，
不把属性写进 cue 生成链。

## §5 判据载体在哪

`tests/taiji_native/test_n4_11_attribute_binding_channel_contract.py`（J-N4g-3a／3b、G-N4s-1／2 的机检形）＋
`reports/taiji_n4_33_four_classes_20261013_after_attribute_channel.json`（J-N4g-3c 的第一次出值，由在册判读器
`scripts/training/adjudicate_taiji_n4_four_class_acceptance.py` 产出，判读件引用它而不手抄数）。

## §6 落笔即钉的三行

* **预期绿**：J-N4g-3a、J-N4g-3b、G-N4s-1、G-N4s-2 四支（默认中性与持久面不加宽都是"不改码路"的应然结果）。
* **仍红时收回哪句**：J-N4g-3c 若落 `no_consumer` ⇒ 收回"属性交换可判"并保留 09 §2 N4 出口的"三类可判＋一类记为缺口"；
  若落 `contaminated_scoring` ⇒ 本件全部作废重冻，并点名是哪一处把属性接进了打分；
  J-N4g-3b 若不为逐位中性 ⇒ 收回"默认不动"，索引改动回退。
* **发表资格前置**：本件全绿之前，"§3.3 四类验收已全部可判"这句不许说；
  全绿之后也只授权"**可判性**"这一层——属性交换的**能力主张**（换了属性还能找对那件事）不属本件，
  它要另一枚带真实分布的载体，届时要点名是哪台仪器、哪张面、什么样本量。
