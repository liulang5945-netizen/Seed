"""DEBT-G68／㊵-593：`_persist` 的**填充顺序**必须是"先填信封，再落盘"。

这条红是这么来的：D/E 两次跑满 60k、面里确有提议（下标 263），但读侧拿到的 `n5_shadow` 是
`None`。我起初把这句写成"键存在但为 null"——**那是读法错了**：`env.get("n5_shadow")` 分不清
"键不存在"与"值为 null"（上一号已就地更正）。真因在源码顺序上：
`envelope["n5_shadow"] = {...}` 与 `envelope["episodic_memory"] = ...` 两段住在
`atomic_save(envelope, checkpoint_path)` **之后** ⇒ 只改内存对象、从不进磁盘。

守卫形状＝按源码索引比顺序，三支都能为假：

* `n5_shadow` 填充的索引 **必须早于** `atomic_save` 调用（旧写法为晚 ⇒ 必红）；
* `episodic_memory` 填充同理（它和影子是同一类"落盘后才补"的受害者）；
* 两段都必须真实存在（缺任一段不许悄悄绿——把守卫删了也算红）。

HEAD 面上这一条为假（旧顺序），因此它不是恒真式；改回旧顺序会当场红。
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN = (REPO / "scripts" / "training" / "train_seed_corpus.py").read_text(encoding="utf-8")

SAVE_CALL = "atomic_save(envelope, checkpoint_path)"
SHADOW_FILL = 'envelope["n5_shadow"] = {'
MEMORY_FILL = 'envelope["episodic_memory"] = episodic_store.checkpoint()'


def test_shadow_fill_precedes_the_persist_call() -> None:
    assert SHADOW_FILL in TRAIN and SAVE_CALL in TRAIN
    assert TRAIN.index(SHADOW_FILL) < TRAIN.index(SAVE_CALL)


def test_episodic_fill_precedes_the_persist_call() -> None:
    assert MEMORY_FILL in TRAIN and SAVE_CALL in TRAIN
    assert TRAIN.index(MEMORY_FILL) < TRAIN.index(SAVE_CALL)


def test_only_one_persist_call_in_this_block() -> None:
    #: 若有人再插一次 `atomic_save(envelope, ...)`（把两段之一挪到它之后），计数就会变，
    #: 上面两条的"第一个 save 位置"参照也随之失真——这里先响。
    assert TRAIN.count(SAVE_CALL) == 1
