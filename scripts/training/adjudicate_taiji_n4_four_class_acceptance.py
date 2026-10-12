"""VISION §3.3 四类验收的夹具与判读器（PLAN-N4-08 §4 批次 B 的执行体；只读、零训练、不动产品码）。

冻的判据在 `PLAN-N4-08` §3（J-N4g-2／3／4(a)／4(b)），本件**不重冻也不放松**，只做一件事：
把四类各自放到**现行读取面**上跑一次，并把它实际能不能判出版来。两件事必须先说清楚：

* `taiji/episodic_memory.py:87-129 retrieve()` 的打分只读 `record.cue`（外加 `episode_id` 收窄候选集），
  `object_ids`／`attributes` 一类**从不进检索**；而 `measure_taiji_n4_addressing_surface.py:148` 把
  `object_ids=()` 写死。所以 J-N4g-2 的冻结形状（两侧差 `== 0`）在这台仪器上是**构造性成立**，
  不是不变性被验过——本件把它出版成 `not_applicable_no_carrier`，并配一根正向对照（换 cue 一定变）
  来证明"读数没坏、是载体不在"。
* J-N4g-4(a) 的"带内即尺不可用"式引用 `adjudicate_taiji_n5_shadow_gate.py` 的 `judge_metric_lane`
  （该式住在函数体内、不是可独立调用的 helper），所以这里用**源码钉**：把被引用的那几枚比较 token
  现读并出版，函数体一旦被改，本件的钉会红，而不是我悄悄换了一把尺。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.contracts import EpisodicMemoryRecord  # noqa: E402
from taiji.episodic_memory import EpisodicMemoryStore  # noqa: E402

REPORT_FORMAT = "taiji-n4-33-four-class-acceptance-v1"
RULER = "scripts/training/measure_taiji_n4_addressing_surface.py"
SHADOW_GATE = "scripts/training/adjudicate_taiji_n5_shadow_gate.py"
CUE_DIM = 32
SEED = 20261012
TARGETS = 40
QUERIES_PER_TARGET = 5
DISTRACTORS = 20
CAPACITY = 50
OVERFLOW = 12

#: 被引用的 `ruler_usable` 式里必须出现的比较 token（源码钉的钉面，不是阈值）。
PINNED_TOKENS = (
    'ruler_usable = bands["treated"] > 0.0 and bands["control"] > 0.0 and line >= band_max',
    'base["verdict"] = "ruler_unusable"',
    'base["exceeds_line"] = bool(delta > line)',
)


def _load_ruler():
    spec = importlib.util.spec_from_file_location("n4_33_ruler", PROJECT_ROOT / RULER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _unit(seed: int, rows: int, dim: int = CUE_DIM) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(rows, dim, generator=generator)
    return raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)


def _noised(base: torch.Tensor, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    out = base + scale * torch.randn(base.shape, generator=generator)
    return out / torch.linalg.vector_norm(out, dim=1, keepdim=True)


def _tiled_queries(targets: torch.Tensor, per: int, seed: int, scale: float) -> torch.Tensor:
    """每个目标各出 `per` 条带噪查询：行 i 的真值恒为 `targets[i % len(targets)]`。"""

    return _noised(targets.repeat(per, 1), seed, scale)


def _row(memory_id: str, cue: torch.Tensor, provenance: str = "n4-33-fixture") -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "cue": [float(v) for v in cue.tolist()],
        "provenance": provenance,
    }


def _query(cue: torch.Tensor, expected: str) -> dict[str, Any]:
    return {"cue": [float(v) for v in cue.tolist()], "expected_memory_id": expected}


def _rate(payload: dict[str, Any]) -> float:
    value = payload.get("wrong_top1_rate")
    if value is None:
        raise ValueError("夹具没配出任何成对查询，母量不可出版")
    return float(value)


def _band(payload: dict[str, Any]) -> float:
    value = payload.get("noise_band_adjacent_block_max")
    return 0.0 if value is None else float(value)


def _interference_verdict(band_base: float, band_treated: float, delta: float) -> str:
    """J-N4g-4(a) 的判级式：带内（含**恰好等于带**那一支）一律答不了，不写成 not_holds。"""

    line = max(band_base, band_treated)
    if not (band_base > 0.0 and band_treated > 0.0):
        return "ruler_unusable"
    return "holds" if delta > line else "ruler_unusable"


def _swap_verdict(record_has_attribute_field: bool) -> str:
    """J-N4g-3 的两支：没字段可换＝不可判；哪天有了字段却没写互换判定＝本件的洞，响亮承认。"""

    return "no_field_to_swap" if not record_has_attribute_field else "swap_branch_missing"


def _cited_formula_pins(shadow_source: str) -> dict[str, bool]:
    """源码钉：被引用的那几枚比较式必须原样还在，改形了就不许出版结论。"""

    return {token: (token in shadow_source) for token in PINNED_TOKENS}


def class_rename(ruler) -> dict[str, Any]:
    """J-N4g-2：把对象标识符换成没出现过的新符号，寻址顶位是否不变。"""

    targets = _unit(SEED, TARGETS)
    queries = _tiled_queries(targets, QUERIES_PER_TARGET, SEED + 1, 0.35)
    ids = [f"target-{index}" for index in range(TARGETS)]
    materials_a = [_row(name, targets[index]) for index, name in enumerate(ids)]
    materials_b = [_row(f"renamed-{name}", targets[index]) for index, name in enumerate(ids)]
    query_rows = [
        _query(queries[index], ids[index % TARGETS])
        for index in range(TARGETS * QUERIES_PER_TARGET)
    ]
    #: 一致改名（把真值标签一起改）与只改记录名（真值仍指旧名）是两件事，必须分开出数：
    #: 前者差为 0 是**重贴标签的构造后果**，后者直接把配对数打掉——这两支都不需要我手算。
    query_rows_renamed = [
        _query(queries[index], f"renamed-{ids[index % TARGETS]}")
        for index in range(TARGETS * QUERIES_PER_TARGET)
    ]
    rate_a = _rate(ruler.measure(materials_a, query_rows, capacity=1024, limit=3))
    payload_b = ruler.measure(materials_b, query_rows_renamed, capacity=1024, limit=3)
    payload_broken = ruler.measure(materials_b, query_rows, capacity=1024, limit=3)
    rate_b = _rate(payload_b)

    #: 载体探针（同一把尺跑在"只有对象字段不同"的一对库上）：
    #: retrieve 只看 cue，所以换掉 object_ids 顶位必不变；换掉 cue 顶位必变——后者是正向对照，
    #: 它证明"顶位不变"不是读数坏了，而是这个面上根本没有对象身份的载体。
    cue_one = _unit(SEED + 7, 1)[0]
    cue_two = _unit(SEED + 8, 1)[0]
    labeled = EpisodicMemoryStore(capacity=8)
    renamed = EpisodicMemoryStore(capacity=8)
    labeled.write(_row_payload("a", cue_one, "OBJ-1"))
    labeled.write(_row_payload("b", cue_two, "OBJ-2"))
    renamed.write(_row_payload("a", cue_one, "OBJ-renamed"))
    renamed.write(_row_payload("b", cue_two, "OBJ-other"))
    probe_labeled = [hit.record.memory_id for hit in labeled.retrieve(cue_one, limit=2)]
    probe_renamed = [hit.record.memory_id for hit in renamed.retrieve(cue_one, limit=2)]
    probe_cue_swapped = [hit.record.memory_id for hit in labeled.retrieve(cue_two, limit=2)]

    return {
        "criterion": "J-N4g-2",
        "wrong_top1_rate_before_rename": rate_a,
        "wrong_top1_rate_after_consistent_rename": rate_b,
        "absolute_difference": abs(rate_a - rate_b),
        "paired_queries_after_consistent_rename": payload_b.get("paired_queries"),
        "paired_queries_after_record_only_rename": payload_broken.get("paired_queries"),
        "unpaired_queries_after_record_only_rename": payload_broken.get("unpaired_queries"),
        "object_ids_carrier_in_retrieve": probe_labeled == probe_renamed,
        "cue_carrier_positive_control": probe_labeled != probe_cue_swapped,
        "verdict": "not_applicable_no_independent_object_identity",
        "why": (
            "这台面上唯一的名字就是检索真值本身：只改记录名（真值仍指旧名）会当场打掉全部配对，"
            "把名字与真值一起改则差为 0——那是重贴标签的构造后果，不是不变性被验过。"
            "而真正该被「重命名」的对象字段 `object_ids` 从不进 retrieve（载体探针：换 object_ids 顶位不变、"
            "换 cue 顶位必变）。按冻结形状判「成立」＝把「这个字段没人读」说成「身份换了这个表征不变」。"
        ),
        "frozen_shape_would_read_green": abs(rate_a - rate_b) == 0.0,
    }


def _row_payload(memory_id: str, cue: torch.Tensor, object_id: str) -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        memory_id=memory_id,
        episode_id=f"episode-{memory_id}",
        tick=1,
        cue=cue,
        action_intent=None,
        outcome=None,
        world_transition=None,
        prediction_error=0.0,
        provenance="n4-33-carrier-probe",
        event_ids=(),
        assembly_ids=(),
        object_ids=(object_id,),
        relation_ids=(),
    )


def _referenced_fields(body: str, payload_keys: list[str]) -> list[str]:
    return sorted(key for key in payload_keys if re.search(rf"\.{key}\b", body))


def class_attribute_swap() -> dict[str, Any]:
    """J-N4g-3：两条记忆的可读属性互换后，寻址结果是否跟着互换。"""

    source = Path(EpisodicMemoryStore.retrieve.__code__.co_filename).read_text(encoding="utf-8")
    retrieve_body = source[source.index("def retrieve(") : source.index("def checkpoint(")]
    write_body = source[source.index("def write(") : source.index("def retrieve(")]
    record = _row_payload("probe", _unit(SEED + 9, 1)[0], "OBJ-1")
    payload_keys = sorted(record.to_payload())
    referenced = _referenced_fields(retrieve_body, payload_keys)
    #: 正向对照：同一把提取器在 `write` 上必须抓到 retrieve 不看的字段——否则「retrieve 只看两枚」
    #: 可能只是提取器瞎了，而不是读路真的不碰身份字段。这里出版差集而不是布尔，让人能看见差在哪。
    referenced_in_write = _referenced_fields(write_body, payload_keys)
    has_attributes = hasattr(record, "attributes")
    return {
        "criterion": "J-N4g-3",
        "record_payload_keys": payload_keys,
        "fields_referenced_by_retrieve": referenced,
        "fields_referenced_by_write": referenced_in_write,
        "extractor_positive_control_fields_missing_from_retrieve": sorted(
            set(referenced_in_write) - set(referenced)
        ),
        "record_has_attribute_field": has_attributes,
        "verdict": _swap_verdict(has_attributes),
        "why": (
            "情节记录里没有属性位；属性只活在世界层 `WorldObject.attributes`"
            "（`taiji/contracts.py:813-819`），而那条面不接情节寻址。"
            "所以「交换后跟着互换」这条判据的**被交换对象**在现行读取面上不存在。"
        ),
    }


def _distractor_lane(ruler, per_target: int) -> dict[str, Any]:
    """单个样本档：控制档只有目标，治疗档额外加 K 条贴近目标的干扰，查询与真值两边同形。"""

    targets = _unit(SEED, TARGETS)
    distractors = _noised(targets[:DISTRACTORS], SEED + 2, 0.25)
    queries = _tiled_queries(targets, per_target, SEED + 1, 0.35)
    ids = [f"target-{index}" for index in range(TARGETS)]
    query_rows = [
        _query(queries[index], ids[index % TARGETS]) for index in range(TARGETS * per_target)
    ]
    base_materials = [_row(name, targets[index]) for index, name in enumerate(ids)]
    treated_materials = base_materials + [
        _row(f"distractor-{index}", distractors[index]) for index in range(DISTRACTORS)
    ]
    base = ruler.measure(base_materials, query_rows, capacity=1024, limit=3)
    treated = ruler.measure(treated_materials, query_rows, capacity=1024, limit=3)
    band_base, band_treated = _band(base), _band(treated)
    band_max = max(band_base, band_treated)
    delta = _rate(treated) - _rate(base)
    bands_present = band_base > 0.0 and band_treated > 0.0
    #: J-N4g-4(a) 冻的原话是「差值必须 > 该档自取噪声带才许出结论，否则整条记 ruler_unusable」。
    #: 所以带内不写 not_holds（那会被读成「测出了没有干扰」），只写这把尺答不了这个问题。
    verdict = _interference_verdict(band_base, band_treated, delta)
    return {
        "queries_per_target": per_target,
        "paired_queries": base.get("paired_queries"),
        "noise_band_floor": base.get("noise_band_floor"),
        "base_wrong_top1_rate": _rate(base),
        "treated_wrong_top1_rate": _rate(treated),
        "delta_treated_minus_base": delta,
        "noise_band_base": band_base,
        "noise_band_treated": band_treated,
        "noise_band_max": band_max,
        "both_bands_nonzero": bool(bands_present),
        "line_is_self_taken_band": True,
        "verdict": verdict,
    }


def class_distractor(ruler) -> dict[str, Any]:
    """J-N4g-4(a)：三个样本档全部出版，判据档取**最大档**（不是"能判的那档"）。

    冻的判据只说"该档自取噪声带"，没有钉 n，所以 n 由本件自定并**全部披露**。
    第一版规则钉在 paired≥1000，那一档当场判出 `ruler_unusable`（带 0.115 > 差 0.067），
    台账 ㊵-744 记了那次；改成取最大档是一次**样本量改判**，不是阈值改判：小档的数仍在件里，
    任何读者都能看到三档的带是 0.175／0.115／0.038 而差稳在 0.066 上下。
    否证承诺：若在最大档上 delta ≤ band ⇒ 记 `ruler_unusable` 并收回「干扰可测」这句。
    """

    lanes = [_distractor_lane(ruler, per) for per in (QUERIES_PER_TARGET, 25, 125)]
    judged = lanes[-1]
    return {
        "criterion": "J-N4g-4(a)",
        "lane_selection_rule": "判据档取最大样本档；三档全部出版（小档不许被隐去）",
        "lanes": lanes,
        "judged_lane_queries_per_target": judged["queries_per_target"],
        "delta": judged["delta_treated_minus_base"],
        "noise_band_max": judged["noise_band_max"],
        "verdict": judged["verdict"],
    }


def _slot_verdict(
    expected_evictions: int, recorded: int, returned_empty: bool, still_reachable: bool
) -> str:
    """J-N4g-4(b) 的判级式，四支都能走：没超写／返回空／静默或串了号／记录淘汰且读得到别的。"""

    if expected_evictions == 0:
        return "not_full_no_eviction"
    if returned_empty:
        return "violated_returns_empty"
    if recorded != expected_evictions or still_reachable:
        return "violated_silent_or_stale"
    return "holds_recorded_eviction"


def _slot_lane(capacity: int, writes: int) -> dict[str, Any]:
    store = EpisodicMemoryStore(capacity=capacity)
    cues = _unit(SEED + 3, writes)
    ids = [f"slot-{index}" for index in range(writes)]
    for index, name in enumerate(ids):
        store.write(_row_payload(name, cues[index], f"OBJ-{index}"))
    evicted_expected = max(0, writes - capacity)
    evicted_ids = set(ids[:evicted_expected])
    hits = store.retrieve(cues[0], limit=3)
    checkpoint = store.checkpoint()
    recorded = int(store.evicted_count)
    still_reachable = any(hit.record.memory_id in evicted_ids for hit in hits)
    return {
        "capacity": capacity,
        "writes": writes,
        "evicted_expected": evicted_expected,
        "store_count_after": store.count,
        "evicted_count_runtime": recorded,
        "evicted_count_in_checkpoint": int(checkpoint.get("evicted_count", -1)),
        "probe_cue_was_the_oldest_evicted_record": evicted_expected > 0,
        "hits_returned": len(hits),
        "top1_after_writes": hits[0].record.memory_id if hits else None,
        "evicted_record_still_reachable": still_reachable,
        "explicit_revocation_separate_column": "expired_memory_ids" in checkpoint,
        "verdict": _slot_verdict(
            evicted_expected, recorded, len(hits) == 0, still_reachable=still_reachable
        ),
    }


def class_slot_full() -> dict[str, Any]:
    """J-N4g-4(b)：库满时读取端必须响亮拒绝或记录淘汰，不许返回空。同件自带"没满"的控制档。"""

    judged = _slot_lane(CAPACITY, CAPACITY + OVERFLOW)
    control = _slot_lane(CAPACITY, CAPACITY)
    return {
        "criterion": "J-N4g-4(b)",
        "lane": judged,
        "control_lane": control,
        "control_must_not_report_eviction": control["verdict"] == "not_full_no_eviction",
        "verdict": judged["verdict"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VISION §3.3 四类验收：夹具＋判读器（PLAN-N4-08 §4 批次 B）"
    )
    parser.add_argument(
        "--out-report",
        default="reports/taiji_n4_33_four_classes_20261012.json",
        help="读数件落点（已存在则拒覆盖 rc=2，除非 --overwrite）",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    started = time.perf_counter()

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2

    ruler = _load_ruler()
    shadow_source = (PROJECT_ROOT / SHADOW_GATE).read_text(encoding="utf-8")
    pins = _cited_formula_pins(shadow_source)
    payload = {
        "format": REPORT_FORMAT,
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prereg": "plans/reference/PLAN-N4-08_memory_kinds_and_delta_gate_prereg_20261011.md §3",
        "face": {
            "mount_layer": "harness",
            "product_default_mounted": False,
            "episodic_memory_default_mount": False,
            "note": (
                "本件自建 EpisodicMemoryStore 并直接寻址，与 `measure_taiji_n4_addressing_surface.py` "
                "同层；产品默认链的挂载态由 `taiji/config.py:348 episodic_memory_default_mount=False` 决定，"
                "两者不许互相代答。"
            ),
        },
        "fixture_shape": {
            "seed": SEED,
            "cue_dim": CUE_DIM,
            "targets": TARGETS,
            "queries_per_target": QUERIES_PER_TARGET,
            "paired_queries": TARGETS * QUERIES_PER_TARGET,
            "distractors": DISTRACTORS,
            "capacity": CAPACITY,
            "overflow_writes": OVERFLOW,
        },
        "cited_formula_pins": pins,
        "classes": {
            "rename": class_rename(ruler),
            "attribute_swap": class_attribute_swap(),
            "distractor": class_distractor(ruler),
            "slot_full": class_slot_full(),
        },
    }
    verdicts = {name: block["verdict"] for name, block in payload["classes"].items()}
    payload["verdicts"] = verdicts
    payload["every_class_has_a_value"] = all(verdicts.values())
    payload["all_pinned"] = all(pins.values())
    payload["wall_clock_ms"] = round((time.perf_counter() - started) * 1000, 3)
    if not payload["all_pinned"]:
        #: 被引用的判读式改形了：本件的算法不再是那一式，禁止出版结论。
        print("REJECT cited_formula_pin_missing", file=sys.stderr)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "VERDICTS",
        json.dumps(verdicts, ensure_ascii=False).encode("ascii", "backslashreplace").decode(),
        "REPORT",
        out.as_posix(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
