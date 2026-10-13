"""`PLAN-N4-11` 的绑定通道契约册（J-N4h-1／2／3／5＋G-N4h-1／2）。

本册钉的是**载体**：对象标识成为检索能看见的第二种键，而默认那条码路逐位不动。
两支"必须不动"的钉法各自能为假：
①默认中性＝重跑四类判读器，拿它和**已入库的上一份读数件**逐值比（那件套的是改动前的产品码）；
②持久面＝把 `checkpoint()` 的键集钉成实名集合，多一个键就红（㊵-729 批次 B 的迁移面教训）。
`J-N4h-4` 要的方向不在这里判——它由判读器出值，本册只保证"通道存在且被 §1 的形状约束着"。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from taiji.contracts import EpisodicMemoryRecord
from taiji.episodic_memory import EpisodicMemoryStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEALED_READING = PROJECT_ROOT / "reports/taiji_n4_33_four_classes_20261012.json"
INSTRUMENT = PROJECT_ROOT / "scripts" / "training" / "adjudicate_taiji_n4_four_class_acceptance.py"

#: ㊵-757（PLAN-N4-12）升版：属性位的披露计数进 `checkpoint()`（可选键，格式串仍 v1）。
#: 旧清单（7 枚、无 `implicit_attribute_count`）自本注起作废；派生索引仍不许进这一面，
#: 那条判据住在 `test_n4_11::test_checkpoint_face_keeps_the_derived_indexes_out`。
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
#: ㊵-757 同轮重钉：记录载荷多一枚 `attribute_ids`（第二可寻址维），旧 14 枚清单自本注起作废。
RECORD_PAYLOAD_KEYS = {
    "action_intent",
    "assembly_ids",
    "attribute_ids",
    "cue",
    "episode_id",
    "event_ids",
    "memory_id",
    "object_ids",
    "outcome",
    "prediction_error",
    "provenance",
    "relation_ids",
    "tick",
    "version",
    "world_transition",
}


def _record(memory_id: str, cue, object_ids: tuple[str, ...]) -> EpisodicMemoryRecord:

    return EpisodicMemoryRecord(
        memory_id=memory_id,
        episode_id=f"episode-{memory_id}",
        tick=1,
        cue=torch.tensor([float(v) for v in cue], dtype=torch.float32),
        action_intent=None,
        outcome=None,
        world_transition=None,
        prediction_error=0.0,
        provenance="n4-10-contract",
        event_ids=(),
        assembly_ids=(),
        object_ids=object_ids,
        relation_ids=(),
    )


def _store(records: int = 3, *, capacity: int = 64) -> EpisodicMemoryStore:
    """三条互不相似的单位向量：0 号与 1 号正交，用来让"该取谁"有唯一答案。"""

    store = EpisodicMemoryStore(capacity=capacity)
    for index in range(records):
        cue = torch.zeros(4)
        cue[index % 4] = 1.0
        store.write(_record(f"m-{index}", tuple(cue.tolist()), (f"OBJ-{index}",)))
    return store


def _digest(store: EpisodicMemoryStore) -> str:
    return hashlib.sha256(
        json.dumps(store.checkpoint(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def test_default_retrieve_reproduces_the_sealed_reading(tmp_path) -> None:
    """J-N4h-1：改动后的默认码路必须复刻**改动前**那份已入库读数（逐值，不是"看着差不多"）。

    ㊵-757 就地收窄这支的口径（原文不删）：`attribute_swap` 那一块**本来就是要变的**——属性位落地
    之前它是 `no_field_to_swap`，之后它第一次出真值，所以"逐位中性"这句对那一格不成立，也不该成立
    （PLAN-N4-12 §2 的 J-N4g-3b 参照件指的是读取面，不是那一格的 verdict）。
    其余三类与 distractor 的每一条 lane 仍按逐值比，一条都不许松。
    """

    sealed = json.loads(SEALED_READING.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("n4_10_instrument", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    out = tmp_path / "rerun.json"
    assert module.main(["--out-report", str(out)]) == 0
    fresh = json.loads(out.read_text(encoding="utf-8"))
    for name in ("rename", "distractor", "slot_full"):
        assert fresh["classes"][name]["verdict"] == sealed["classes"][name]["verdict"], name
    #: 变了的那一格也必须**点名它变成了什么**，不许悄悄消失在册外。
    assert fresh["classes"]["attribute_swap"]["verdict"] != "no_field_to_swap"
    assert fresh["classes"]["attribute_swap"]["record_has_attribute_field"] is True
    sealed_lanes = sealed["classes"]["distractor"]["lanes"]
    fresh_lanes = fresh["classes"]["distractor"]["lanes"]
    for index, lane in enumerate(fresh_lanes):
        assert lane["base_wrong_top1_rate"] == sealed_lanes[index]["base_wrong_top1_rate"], index
        assert lane["treated_wrong_top1_rate"] == sealed_lanes[index]["treated_wrong_top1_rate"]


def test_checkpoint_face_is_pinned_and_carries_no_index() -> None:
    """G-N4h-1：绑定索引是派生物，多一个持久键就必须红（迁移面＝在册 .pt 的载入半径）。"""

    store = _store()
    assert set(store.checkpoint()) == CHECKPOINT_KEYS, sorted(store.checkpoint())
    assert "object_index" not in store.checkpoint()
    assert set(_record("m-1", [1, 0, 0, 0], ("OBJ-1",)).to_payload()) == RECORD_PAYLOAD_KEYS


def test_object_filter_narrows_and_is_not_ignored() -> None:
    """J-N4h-2：过滤必须被走到——同一条 query 不带 object_id 时顶位是另一个对象。"""

    store = _store(records=3)
    query = torch.tensor([0.0, 1.0, 0.0, 0.0])
    open_face = [hit.record.memory_id for hit in store.retrieve(query, limit=3)]
    assert open_face[0] == "m-1", open_face
    narrowed = [hit.record.memory_id for hit in store.retrieve(query, limit=3, object_id="OBJ-0")]
    assert narrowed == ["m-0"], narrowed
    assert set(narrowed) <= set(store.bound_memory_ids("OBJ-0"))


def test_unbound_object_raises_instead_of_returning_empty() -> None:
    """§1.3／G-N4h-2：没有绑定的对象名是响亮错，不许伪装成"零命中＝空库"（DEBT-G58 同族）。"""

    store = _store(records=1)
    with pytest.raises(ValueError, match="no episodic records are bound"):
        store.retrieve(torch.tensor([1.0, 0.0, 0.0, 0.0]), limit=1, object_id="OBJ-nope")
    #: 同一把尺的另一种答案：limit<=0 那条既有语义仍返回空元组，不能被上面那支吞掉。
    assert store.retrieve(torch.tensor([1.0, 0.0, 0.0, 0.0]), limit=0) == ()


def test_rename_moves_the_binding_and_never_touches_records() -> None:
    """J-N4h-3：改名只动索引；改名前后记录逐位相同，且旧名必须走响亮出口。"""

    store = _store(records=3)
    before = _digest(store)
    moved = store.rename_object("OBJ-1", "OBJ-renamed")
    assert moved == 1
    assert _digest(store) == before
    query = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert [
        hit.record.memory_id for hit in store.retrieve(query, limit=2, object_id="OBJ-renamed")
    ] == ["m-1"]
    with pytest.raises(ValueError, match="no episodic records are bound"):
        store.retrieve(query, limit=1, object_id="OBJ-1")


def test_rename_refuses_to_merge_two_objects() -> None:
    store = _store(records=3)
    with pytest.raises(ValueError, match="already bound"):
        store.rename_object("OBJ-0", "OBJ-1")


def test_eviction_and_revocation_both_drop_bindings() -> None:
    """淘汰／撤销都要解绑，否则绑定会指向库里已不存在的记录。"""

    store = _store(records=2, capacity=2)

    cue = torch.tensor([0.0, 0.0, 0.0, 1.0])
    store.write(_record("m-8", tuple(cue.tolist()), ("OBJ-8",)))
    assert store.evicted_count == 1
    assert "OBJ-0" not in store.object_identifiers()
    with pytest.raises(ValueError, match="no episodic records are bound"):
        store.bound_memory_ids("OBJ-0")
    store.expire("m-1")
    assert "OBJ-1" not in store.object_identifiers()


def test_attribute_slot_is_now_present_and_pinned_by_its_real_name() -> None:
    """J-N4h-5 就地重钉（㊵-757）：这一支原来钉的是「属性位不存在」，现在属性位来了。

    原来那两行读的是 `hasattr(record, "attributes")` 与 `"attributes" not in RECORD_PAYLOAD_KEYS`——
    真加的字段叫 **`attribute_ids`**，所以那两行在属性位落地之后**仍然全绿**（它们量的名字从来不是
    那一枚）。这就是"钉子按错名＝零判别力"的现场（[[instrument-name-must-resolve-on-the-holder-class]]
    同族）：旧断言自本注起作废，改成按真名钉"在场"，判据载体搬到
    `test_n4_11_attribute_binding_channel_contract.py`。
    """

    record = _record("m-1", [1, 0, 0, 0], ("OBJ-1",))
    assert record.attribute_ids == ()
    assert "attribute_ids" in RECORD_PAYLOAD_KEYS
    assert "attribute_ids" in record.to_payload()
    assert hasattr(record, "attributes") is False


def test_j_n4h_4_gives_the_rename_acceptance_a_carrier(tmp_path) -> None:
    """J-N4h-4：绑定通道必须让 `J-N4g-2` 从「不可判」变成出一个值——通道不能是没人消费的索引。"""

    spec = importlib.util.spec_from_file_location("n4_10_adj", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    out = tmp_path / "channel.json"
    assert module.main(["--out-report", str(out)]) == 0
    rename = json.loads(out.read_text(encoding="utf-8"))["classes"]["rename"]
    assert rename["j_n4h_4_verdict"] in {"holds", "not_holds"}, rename["j_n4h_4_verdict"]
    channel = rename["binding_channel"]
    assert channel["hits_before_rename"] == channel["hits_after_rename"]
    assert channel["records_rebound"] == 1
    #: 旧名必须走响亮出口。若哪天它变成"返回空"，这一支就红——那是 DEBT-G58 的混淆复活。
    assert channel["old_name_still_resolves"] is False
    #: 冻面的结论不许被通道悄悄改写：ruler 上仍然是没有独立身份载体。
    assert rename["verdict"] == "not_applicable_no_independent_object_identity"
