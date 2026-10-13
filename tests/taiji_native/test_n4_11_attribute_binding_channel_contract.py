"""`PLAN-N4-12` 的属性位契约册（J-N4g-3a／3b／3c＋G-N4s-1／2）。

本册钉的是**第二可寻址维**这一件事：`attribute_ids` 成为检索能看见的第三种键，
而默认那条码路、打分表达式、持久面三处都逐位不动。批文＝owner 八裁⑴ 甲。

四条各自能为假的形状：
①往返（J-N4g-3a）——旧载荷缺键必须载入成功并出版**现数**的披露计数，不是读件里那个历史值；
②默认中性（J-N4g-3b）——同一份库用"不带属性"的老载荷载进来，检索结果与分数必须逐位相同；
③互换可判（J-N4g-3c）——只换 `attribute_ids`（对象、cue、episode 都不动）顶位必须跟着换，
  且**不带属性那张面必须不动**；两枚探针分开出版，缺任何一枚都不算过；
④守卫（G-N4s-1／2）——属性不进打分、派生索引不进持久面，两条都配负对照。

本册全绿只授权「属性交换可判」这一层**可判性**（冻件 §6 第三条），不授权任何能力主张。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from taiji.contracts import EpisodicMemoryRecord
from taiji.episodic_memory import EpisodicMemoryStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = PROJECT_ROOT / "scripts" / "training" / "adjudicate_taiji_n4_four_class_acceptance.py"
STORE_SOURCE = (PROJECT_ROOT / "taiji" / "episodic_memory.py").read_text(encoding="utf-8")

#: ㊵-757 升版：属性位的披露计数进 checkpoint()（可选键，格式串仍 v1），派生索引两枚不进。
#: 旧清单（7 枚、无 implicit_attribute_count）自本注起作废——这是"同轮重钉"，不是删断言变绿。
CHECKPOINT_KEYS = {
    "capacity",
    "cue_dim",
    "expired_memory_ids",
    "evicted_count",
    "format",
    "implicit_attribute_count",
    "implicit_provenance_count",
    "records",
}
#: 载荷里**不许**出现的两枚派生索引名（进了就是悄悄加宽持久面）。
DERIVED_INDEX_NAMES = ("_attributes", "_attribute_of", "_objects", "_object_of")
SHARED = "task=same"


def _instrument():
    spec = importlib.util.spec_from_file_location("n4_12_instrument", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(
    memory_id: str,
    cue: tuple[float, ...],
    *,
    object_ids: tuple[str, ...] = (),
    attribute_ids: tuple[str, ...] = (),
) -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        memory_id=memory_id,
        episode_id="episode-shared",
        tick=1,
        cue=torch.tensor([float(v) for v in cue], dtype=torch.float32),
        action_intent=None,
        outcome=None,
        world_transition=None,
        prediction_error=0.0,
        provenance="n4-12-contract",
        event_ids=(),
        assembly_ids=(),
        object_ids=object_ids,
        relation_ids=(),
        attribute_ids=attribute_ids,
    )


def _pair(*, swapped: bool = False) -> EpisodicMemoryStore:
    """两条正交 cue 的记录：共享属性 `task=same` 给正向对照用，`color=*` 各绑一条给互换取。"""

    red, blue = "color=red", "color=blue"
    first = (blue, SHARED) if swapped else (red, SHARED)
    second = (red, SHARED) if swapped else (blue, SHARED)
    store = EpisodicMemoryStore(capacity=8)
    store.write(_record("m-a", (1.0, 0.0, 0.0, 0.0), object_ids=("OBJ-SAME",), attribute_ids=first))
    store.write(
        _record("m-b", (0.0, 1.0, 0.0, 0.0), object_ids=("OBJ-SAME",), attribute_ids=second)
    )
    return store


CUE_ONE = (1.0, 0.0, 0.0, 0.0)
CUE_TWO = (0.0, 1.0, 0.0, 0.0)


def _face(store: EpisodicMemoryStore, cue: tuple[float, ...]) -> list[tuple[str, float]]:
    return [
        (hit.record.memory_id, hit.score)
        for hit in store.retrieve(torch.tensor(list(cue), dtype=torch.float32), limit=2)
    ]


def _digest(store: EpisodicMemoryStore) -> str:
    return hashlib.sha256(
        json.dumps(store.checkpoint(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def test_attribute_ids_survive_the_payload_round_trip() -> None:
    """J-N4g-3a 第一支：构造→载荷→构造，属性集合逐项相等（顺序与去重由 `_normalize_ids` 定形）。"""

    record = _record("m-1", CUE_ONE, attribute_ids=("color=red", "size=big"))
    restored = EpisodicMemoryRecord.from_payload(record.to_payload())
    assert restored.attribute_ids == ("color=red", "size=big")
    assert _record("m-2", CUE_ONE).attribute_ids == ()


def test_legacy_payload_without_the_key_loads_and_is_counted_now() -> None:
    """J-N4g-3a 第二支：历史件不带这枚键必须**载入成功**，且披露计数是本次现数。

    负对照就在断言里：把件里的 `implicit_attribute_count` 写成 0，载入后出版的仍必须是 2——
    回填会把刚数出来的数覆盖成 0（㊵-681 的 `implicit_provenance_count` 同一族）。
    """

    payloads = []
    for name in ("m-1", "m-2"):
        item = _record(name, CUE_ONE).to_payload()
        item.pop("attribute_ids")
        payloads.append(item)
    legacy = {
        "format": "taiji-episodic-memory-v1",
        "capacity": 8,
        "cue_dim": 4,
        "records": payloads,
        "implicit_attribute_count": 0,
    }
    store = EpisodicMemoryStore.from_checkpoint(legacy)
    assert store.count == 2
    assert store.implicit_attribute_count == 2
    assert store.checkpoint()["implicit_attribute_count"] == 2
    assert all(record.attribute_ids == () for record in store.records)


def test_swapping_attributes_moves_the_attribute_face_only() -> None:
    """J-N4g-3c：对象、cue、episode 全不动，只换 `attribute_ids`——按属性检索顶位必须跟着换。"""

    labeled, swapped = _pair(), _pair(swapped=True)
    query = torch.tensor(list(CUE_ONE), dtype=torch.float32)
    top_labeled = labeled.retrieve(query, limit=1, attribute="color=red")[0].record.memory_id
    top_swapped = swapped.retrieve(query, limit=1, attribute="color=red")[0].record.memory_id
    assert (top_labeled, top_swapped) == ("m-a", "m-b"), (top_labeled, top_swapped)
    #: 同一枚交换**不许**动不带属性的那张面——动了就说明属性进了打分（contaminated_scoring）。
    assert _face(labeled, CUE_ONE) == _face(swapped, CUE_ONE)
    #: 正向对照：共享属性下候选集仍是两条，顶位由 cue 说话；换 cue 顶位就得换。
    second = labeled.retrieve(
        torch.tensor(list(CUE_TWO), dtype=torch.float32), limit=1, attribute=SHARED
    )
    assert second[0].record.memory_id == "m-b", second


def test_attribute_filter_narrows_and_is_conjunctive_with_the_object_filter() -> None:
    """§1.4：属性与对象是合取，谁也不许覆盖谁；未绑定的属性名走响亮出口。"""

    store = _pair()
    query = torch.tensor(list(CUE_ONE), dtype=torch.float32)
    assert [
        hit.record.memory_id for hit in store.retrieve(query, limit=2, attribute="color=red")
    ] == ["m-a"]
    #: 两枚同时点名且指向不同记录 ⇒ 交集为空，但两枚键都必须在场（不是"后一个顶掉前一个"）。
    assert store.retrieve(query, limit=2, object_id="OBJ-SAME", attribute="color=red")
    assert [
        hit.record.memory_id
        for hit in store.retrieve(query, limit=2, object_id="OBJ-SAME", attribute=SHARED)
    ] == ["m-a", "m-b"]
    with pytest.raises(ValueError, match="no episodic records are bound to attribute"):
        store.retrieve(query, limit=1, attribute="color=green")
    #: 既有语义不被新参数挤掉：limit<=0 仍返回空元组，且在属性名未绑定时也不抛。
    assert store.retrieve(query, limit=0, attribute="color=green") == ()


def test_rename_attribute_moves_bindings_and_never_touches_records() -> None:
    """§1.5：改名只动索引；改名前后记录逐位相同，且旧名必须走响亮出口（与对象维同形）。"""

    store = _pair()
    before = _digest(store)
    moved = store.rename_attribute("color=red", "color=crimson")
    assert moved == 1
    assert _digest(store) == before
    query = torch.tensor(list(CUE_ONE), dtype=torch.float32)
    assert [
        hit.record.memory_id for hit in store.retrieve(query, limit=1, attribute="color=crimson")
    ] == ["m-a"]
    with pytest.raises(ValueError, match="no episodic records are bound to attribute"):
        store.retrieve(query, limit=1, attribute="color=red")
    #: 记录本身没被改写：属性仍住在索引里，读回来还是换名后的那一枚。
    assert store.records[0].attribute_ids[0] == "color=red"
    #: 拒绝合并那支的取法：旧名必须**还在**（用刚改出来的 crimson），否则先撞上"未绑定"那出口。
    with pytest.raises(ValueError, match="already bound"):
        store.rename_attribute("color=crimson", "color=blue")
    with pytest.raises(ValueError, match="requires both identifiers"):
        store.rename_attribute("color=crimson", "")


def test_eviction_and_revocation_both_drop_attribute_bindings() -> None:
    """容量淘汰与显式撤销都要解绑，否则属性检索会指向库里已不存在的记录。"""

    store = EpisodicMemoryStore(capacity=2)
    store.write(_record("m-0", CUE_ONE, attribute_ids=("color=red",)))
    store.write(_record("m-1", CUE_TWO, attribute_ids=("color=blue",)))
    query = torch.tensor(list(CUE_ONE), dtype=torch.float32)
    store.write(_record("m-8", (0.0, 0.0, 1.0, 0.0), attribute_ids=("color=green",)))
    assert store.evicted_count == 1
    assert "color=red" not in store.attribute_identifiers()
    with pytest.raises(ValueError, match="no episodic records are bound to attribute"):
        store.retrieve(query, limit=1, attribute="color=red")
    store.expire("m-1")
    assert "color=blue" not in store.attribute_identifiers()
    assert store.attribute_identifiers() == ("color=green",)


def test_attribute_never_enters_the_scoring_expression() -> None:
    """G-N4s-1：真件上违规 0，但同一把尺必须能数出越界样本——两向都要在进程内测过。"""

    instrument = _instrument()
    face = instrument._attribute_scoring_face(STORE_SOURCE)
    assert face == {"attribute_operands": 0, "similarity_calls": 3}, face
    #: `similarity_calls` 为 0 时"违规 0"是假绿，所以这台尺必须自证扫到了打分面。
    assert face["similarity_calls"] > 0

    fabricated = (
        "def retrieve(self, cue):\n"
        "    score = 1.0\n"
        "    return score + len(record.attribute_ids)\n"
    )
    assert instrument._attribute_scoring_face(fabricated)["attribute_operands"] == 1
    #: 负对照的第二向：提取器不能靠"什么都没找到"来通过。
    with pytest.raises(ValueError, match="retrieve"):
        instrument._attribute_scoring_face("def nothing(): return None\n")


def test_checkpoint_face_keeps_the_derived_indexes_out() -> None:
    """G-N4s-2：持久面键集与实名集合逐枚相等；派生索引塞进来必须红（`==` 不是 `⊆`）。"""

    store = _pair()
    checkpoint = store.checkpoint()
    assert set(checkpoint) == CHECKPOINT_KEYS, sorted(checkpoint)
    assert not set(checkpoint) & set(DERIVED_INDEX_NAMES)

    #: 同一把尺的负对照：多塞一枚派生索引，这条判据必须立刻为假。
    widened = dict(checkpoint)
    widened["_attributes"] = {"color=red": ["m-a"]}
    assert set(widened) != CHECKPOINT_KEYS


def test_reported_verdict_is_in_the_frozen_vocabulary(tmp_path) -> None:
    """判级词不许自造：仪器出的属性维 verdict 必须落在冻件 §2 点名的出口里。"""

    instrument = _instrument()
    out = tmp_path / "four.json"
    assert instrument.main(["--out-report", str(out)]) == 0
    block: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))["classes"]["attribute_swap"]
    assert block["verdict"] in {
        "holds",
        "no_consumer",
        "contaminated_scoring",
        "ruler_unusable",
        "no_field_to_swap",
    }, block["verdict"]
    probe = block["probe"]
    assert probe["cue_probe_moves_top1"] is True
    assert probe["cue_only_face_unchanged_after_swap"] is True
    assert probe["unbound_attribute_raises"] is True
    assert block["attribute_ids_referenced_by_retrieve"] is False


def test_swap_verdict_can_fire_every_branch() -> None:
    """判级式四支都要能为假（含"顶位换了但正向对照不换"＝尺失效那一支）。"""

    instrument = _instrument()
    assert instrument._swap_verdict(False, False, False, False) == "no_field_to_swap"
    assert instrument._swap_verdict(True, True, True, True) == "contaminated_scoring"
    assert instrument._swap_verdict(True, False, True, False) == "no_consumer"
    assert instrument._swap_verdict(True, True, False, False) == "ruler_unusable"
    assert instrument._swap_verdict(True, True, True, False) == "holds"
