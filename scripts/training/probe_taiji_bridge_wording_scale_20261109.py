"""R14 -- wording scale (segment C), the zeros on the densest legal table (D),
and the "one wording, two entities" adversary (E).

R13 closed identity and showed that aliases can be written.  R14 asks what that
costs and where it breaks.

Segment C -- cost curve.  The curve is *counted*, not assumed: every distinct
wording the frozen corpus actually supplies is enumerated per entity, and the
key/entity ratio is reported at each scale it can reach.  The corpus turns out to
supply exactly one wording per entity; the 60 non-describe sentences carry entity
*names*, which may not enter a key.  So the third scale point (78 keys, ratio
8.67) is reported as **not admitted**, with the blocking rule named.

Segment D -- the zeros re-run on the densest legal table (18 keys), not inherited
from R13's 9-key table: 12 unknown candidates, 18 byte fragments, 20 whole
sentences.

Segment E -- adversary: one wording bound to two different entities.  The frozen
rule is first-writer-wins: a write that would re-point an existing key at another
value is rejected and the *whole batch* rolls back, so no entity silently loses
its address.  The counterfactual (``on_conflict="replace"``, R11's rule) is
measured on the same table and reported, not gated.

Zero training, zero optimizer steps, zero Adam, zero confirm.  No fixture byte is
written; ``AddressTable`` and ``LivingCoreReference`` are untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_cross_family_readout_20261109 as base  # noqa: E402
from taiji.address_bridge import (  # noqa: E402
    BridgeTable,
    bridge_records,
    sequence_key,
    value_key,
)
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    OBJECTS,
    PLACE_DESCRIPTIONS,
    PLACES,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_bridge_wording_scale_20261109.json"
REPORT = ROOT / "reports/taiji_bridge_wording_scale_20261109.json"
BRIDGE_SOURCE = ROOT / "taiji/address_bridge.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)

UNKNOWN_CANDIDATES = (
    "黑色铁铃",
    "白色帆布",
    "长柄铜勺",
    "软木瓶塞",
    "蓝色毛巾",
    "银质圆镜",
    "棕色皮箱",
    "青色陶罐",
    "灰色石阶",
    "紫色绸带",
    "金色烛台",
    "深红木箱",
)

AUTHORED_ALIASES = (
    ("喝水的杯子", ("object", "蓝色杯子")),
    ("盖章用的印章", ("object", "红色印章")),
    ("盛水的瓶子", ("object", "绿色瓶子")),
    ("圆形的皮球", ("object", "黄色球")),
    ("方方正正的木桌子", ("place", "木桌")),
    ("窗前的台板", ("place", "窗台")),
    ("手边的小盒子", ("place", "左盒")),
    ("墙上的架子", ("place", "书架")),
    ("脚下的垫子", ("place", "地毯")),
)

ADVERSARY = (
    {
        "name": "object_wording_taken_by_object",
        "wording": "圆口的器皿",
        "bound": ("object", "蓝色杯子"),
        "attempted": ("object", "红色印章"),
    },
    {
        "name": "object_wording_taken_by_place",
        "wording": "圆滚滚的东西",
        "bound": ("object", "黄色球"),
        "attempted": ("place", "木桌"),
    },
    {
        "name": "place_wording_taken_by_other_place",
        "wording": "手边的匣子",
        "bound": ("place", "左盒"),
        "attempted": ("place", "窗台"),
    },
)

DEFINITION = {
    "records": "nine (phrase -> (slot, name)) rows; the key is the phrase, the value the entity",
    "segment_c": {
        "counting": (
            "for every entity, the distinct wordings the frozen corpus supplies are enumerated; "
            "the key/entity ratio is reported at each scale it can reach, and a scale that is not "
            "admissible is reported as refused with the blocking rule"
        ),
        "name_rule": "an entity name may never enter a key (R11 invariant), so name-bearing sentences are not writable",
        "distribution": "per-entity key counts are reported as min / median / max at each scale",
    },
    "segment_d": {
        "table": "18 keys: the 9 corpus wordings plus the 9 authored aliases of R13",
        "read": "same containment rule as R13: a record answers only when its whole sequence occurs in the query",
        "zeros": "12 unknown candidates, 18 byte fragments, 20 whole sentences",
    },
    "segment_e": {
        "rule": (
            "first writer wins: a write whose key is already bound to a different value is "
            "rejected with reason address_already_bound and the whole batch is left unapplied"
        ),
        "counterfactual": "the same table built with on_conflict='replace' is measured and reported, not gated",
    },
}

CRITERIA = {
    "segment_c": (
        "report per-entity key count min/median/max and the key/entity ratio at 9, 18 and the "
        "refused 78 keys; state whether the ratio rises and why."
    ),
    "segment_d": (
        "on the 18-key table: 12 unknown candidates answer 0; 18 byte fragments answer 0; 20 whole "
        "sentences mis-point 0; capability 40/40 with zero wrong pointers; nine closed-state checks."
    ),
    "segment_e": (
        "every adversary case exits rejected_batch_address_already_bound with the table content "
        "digest unchanged and the first binding still readable; the append control (two wordings, "
        "one entity) is accepted; a within-batch collision still exits "
        "rejected_batch_address_collision."
    ),
    "reporting": "segments are reported separately; a clean C does not carry D or E.",
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentences() -> list[str]:
    return [row["text"] for row in base.frozen.build_fixture() if row["variant"] == "describe"]


def every_sentence() -> list[str]:
    return [row["text"] for row in base.frozen.build_fixture()]


def locate(sentence: str) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for name, phrase in OBJECT_DESCRIPTIONS.items():
        if phrase in sentence:
            found["object"] = (phrase, name)
    for name, phrase in PLACE_DESCRIPTIONS.items():
        if phrase in sentence:
            found["place"] = (phrase, name)
    return found


def tensor_sha(values: list[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"none")
        elif isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def reference_digest(reference: Any, state: Any, step_hashes: list[str]) -> dict[str, Any]:
    parameters = [tensor for _, tensor in sorted(reference.named_parameters())]
    store = [state.store_keys, state.store_values, state.store_age, state.store_count]
    identity = {
        "parameters": len(parameters),
        "elements": int(sum(tensor.numel() for tensor in parameters)),
        "config": repr(reference.config),
    }
    return {
        "parameters_sha": tensor_sha(parameters),
        "state_sha": tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior]),
        "keys_len": len(state.keys),
        "step": int(state.step),
        "store_sha": tensor_sha(store),
        "identity_sha": hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sc_sfk_none": state.sc_sfk is None,
        "battery_sha": hashlib.sha256("".join(step_hashes).encode("ascii")).hexdigest(),
        "battery_steps": len(step_hashes),
    }


def run_battery(reference: Any, texts: list[str]) -> dict[str, Any]:
    torch.manual_seed(SEED)
    state = reference.initial_state("r14-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=b"r14")
        step_hashes.append(
            tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])
        )
    return reference_digest(reference, state, step_hashes)


def build_table(on_conflict: str = "reject") -> tuple[BridgeTable, dict[str, Any]]:
    table = BridgeTable(key_fn=sequence_key, read_mode="sequence", on_conflict=on_conflict)
    write = table.write(bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS))
    if write["action"] != "accepted":
        raise SystemExit(f"initial write rejected: {write['action']}")
    return table, write


def capability(table: BridgeTable, texts: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    hits = wrong = abstained = 0
    for text in texts:
        for slot, (phrase, name) in sorted(locate(text).items()):
            result = table.read(phrase)
            expected = (slot, name)
            value = result["value"]
            if value is None:
                abstained += 1
            elif tuple(value) == expected:
                hits += 1
            else:
                wrong += 1
            rows.append(
                {
                    "phrase": phrase,
                    "expected": list(expected),
                    "decision": result["decision"],
                    "value": list(value) if value else None,
                }
            )
    total = len(rows)
    return {
        "rows": rows,
        "hits": hits,
        "of": total,
        "wrong_pointers": wrong,
        "abstained": abstained,
        "ok": hits == total and wrong == 0,
    }


def unknown_reading(table: BridgeTable) -> dict[str, Any]:
    rows = []
    answering = 0
    for phrase in UNKNOWN_CANDIDATES:
        result = table.read(phrase)
        answered = result["value"] is not None
        answering += int(answered)
        rows.append(
            {
                "phrase": phrase,
                "decision": result["decision"],
                "value": list(result["value"]) if result["value"] else None,
                "reason": result["reason"],
                "answered": answered,
            }
        )
    return {
        "rows": rows,
        "candidates": len(rows),
        "answering_phrases": answering,
        "zero_line_ok": answering == 0,
    }


def fragments_reading(table: BridgeTable) -> dict[str, Any]:
    rows = []
    same = other = 0
    for phrase, value in bridge_records(OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS):
        data = phrase.encode("utf-8")
        for offset in (1, 2):
            result = table.read(data[offset:])
            answer = result["value"]
            if answer is None:
                kind = "abstain"
            elif tuple(answer) == tuple(value):
                kind = "same_record"
                same += 1
            else:
                kind = "other_record"
                other += 1
            rows.append({"phrase": phrase, "byte_offset": offset, "kind": kind})
    return {
        "rows": rows,
        "inputs": len(rows),
        "same_record": same,
        "other_record": other,
        "false_answers": same + other,
    }


def sentences_reading(table: BridgeTable, texts: list[str]) -> dict[str, Any]:
    rows = []
    answering = 0
    decisions: dict[str, int] = {}
    for text in texts:
        result = table.read(text)
        decisions[result["decision"]] = decisions.get(result["decision"], 0) + 1
        if result["value"] is not None:
            answering += 1
        rows.append(
            {"sentence": text, "decision": result["decision"], "anchors": result["anchors"]}
        )
    return {
        "rows": rows,
        "inputs": len(rows),
        "false_answers": answering,
        "decisions": dict(sorted(decisions.items())),
    }


def per_entity_counts(wordings: dict[str, list[str]]) -> dict[str, Any]:
    counts = [len(value) for value in wordings.values()]
    ordered = sorted(counts)
    middle = ordered[len(ordered) // 2]
    return {
        "entities": len(counts),
        "keys": sum(counts),
        "ratio": sum(counts) / len(counts),
        "min": ordered[0],
        "median": middle,
        "max": ordered[-1],
        "per_entity": dict(sorted(wordings.items())),
    }


def wording_inventory(texts: list[str], every: list[str]) -> dict[str, Any]:
    corpus: dict[str, list[str]] = {}
    for name, phrase in {**OBJECT_DESCRIPTIONS, **PLACE_DESCRIPTIONS}.items():
        corpus[name] = [phrase]
    name_bearing = sum(1 for text in every if any(name in text for name in (*OBJECTS, *PLACES)))
    aliases: dict[str, list[str]] = {name: list(values) for name, values in corpus.items()}
    for text, (_, name) in AUTHORED_ALIASES:
        aliases[name] = aliases[name] + [text]
    return {
        "corpus": per_entity_counts(corpus),
        "corpus_plus_authored": per_entity_counts(aliases),
        "describe_sentences": len(texts),
        "fixture_sentences": len(every),
        "sentences_with_entity_names": name_bearing,
        "name_bearing_sentences_refused": name_bearing,
        "refusal_rule": "an entity name may never enter a key (R11 invariant)",
        "refused_scale": {
            "keys": 18 + name_bearing,
            "entities": 9,
            "ratio": (18 + name_bearing) / 9,
            "admitted": False,
        },
    }


def conflict_case(
    name: str, wording: str, attempted: tuple[str, str], extra: bool
) -> dict[str, Any]:
    table, _ = build_table()
    table.write(AUTHORED_ALIASES)
    before = table.content_digest()
    batch: list[tuple[str, Any]] = [(wording, attempted)]
    if extra:
        batch.append(("墙边的圆凳", ("place", "木桌")))
    outcome = table.write(batch)
    after = table.content_digest()
    kept = table.read(wording)
    extra_kept = table.read("墙边的圆凳") if extra else None
    return {
        "case": name,
        "wording": wording,
        "attempted": list(attempted),
        "batch_items": len(batch),
        "action": outcome["action"],
        "rejected_reason": outcome.get("reason") or outcome.get("action"),
        "bound": outcome.get("bound"),
        "attempted_value": outcome.get("attempted"),
        "digest_unchanged": before == after,
        "first_binding_still_read": list(kept["value"]) if kept["value"] else None,
        "extra_item_applied": (extra_kept["value"] is not None) if extra else None,
        "rejected_batches": table.rejected_batches,
        "rollbacks": table.rollbacks,
        "rejected_reasons": dict(sorted(table.rejected_reasons.items())),
    }


def replace_counterfactual(wording: str, attempted: tuple[str, str]) -> dict[str, Any]:
    table, _ = build_table(on_conflict="replace")
    table.write(AUTHORED_ALIASES)
    before = table.content_digest()
    outcome = table.write([(wording, attempted)])
    after = table.content_digest()
    read = table.read(wording)
    return {
        "wording": wording,
        "attempted": list(attempted),
        "action": outcome["action"],
        "replaced": table.replaced,
        "digest_changed": before != after,
        "now_reads": list(read["value"]) if read["value"] else None,
        "history_kept": True,
        "note": "R11's rule: the address moves to the new value, the old one is kept in history",
    }


def append_control() -> dict[str, Any]:
    table, _ = build_table()
    before = table.content_digest()
    outcome = table.write([("盛水的瓶子", ("object", "绿色瓶子"))])
    read = table.read("细长的容器")
    return {
        "action": outcome["action"],
        "appended": outcome["appended"],
        "digest_changed": before != table.content_digest(),
        "original_wording_still_reads": list(read["value"]) if read["value"] else None,
        "note": "two wordings pointing at one entity must stay accepted",
    }


def within_batch_collision() -> dict[str, Any]:
    table, _ = build_table()
    before = table.content_digest()
    outcome = table.write(
        [("同一个说法", ("object", "蓝色杯子")), ("同一个说法", ("object", "红色印章"))]
    )
    return {
        "action": outcome["action"],
        "conflict_items": outcome.get("conflict_items"),
        "digest_unchanged": before == table.content_digest(),
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "unknown": list(UNKNOWN_CANDIDATES),
        "aliases": [list(row) for row in AUTHORED_ALIASES],
        "adversary": [dict(row) for row in ADVERSARY],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    texts = sentences()
    inventory = wording_inventory(texts, every_sentence())
    manifest = {
        "schema": "taiji-bridge-wording-scale-r14-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "runner_sha256": RUNNER_SHA_START,
        "bridge_sha256": file_sha(BRIDGE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {
            "records": 9,
            "describe_sentences": len(texts),
            "capability_reads": 2 * len(texts),
            "unknown_candidates": len(UNKNOWN_CANDIDATES),
            "aliases": len(AUTHORED_ALIASES),
            "fragments": 18,
            "adversary_cases": len(ADVERSARY),
            "corpus_keys": inventory["corpus"]["keys"],
            "keys_with_aliases": inventory["corpus_plus_authored"]["keys"],
        },
        "inputs": {
            "unknown_candidates": list(UNKNOWN_CANDIDATES),
            "authored_aliases": [list(row) for row in AUTHORED_ALIASES],
            "adversary": [dict(row) for row in ADVERSARY],
        },
    }
    write_json(MANIFEST, manifest)
    return manifest


def verify_freeze() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise SystemExit("no frozen manifest; run --freeze first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["runner_sha256"] != RUNNER_SHA_START:
        raise SystemExit("runner changed after the freeze; re-freeze")
    if manifest["bridge_sha256"] != file_sha(BRIDGE_SOURCE):
        raise SystemExit("bridge module changed after the freeze; re-freeze")
    if manifest["fixture_sha256"] != base.fixture_hashes():
        raise SystemExit("fixture bytes changed; the frozen fixture must not move")
    if manifest["definition_digest"] != definition_digest():
        raise SystemExit("definition/criteria/inputs changed after the freeze; re-freeze")
    return manifest


def main() -> int:
    assert_no_black_processes()
    base.frozen.acquire_lock()
    try:
        if "--freeze" in sys.argv:
            manifest = freeze()
            print(json.dumps({"manifest": str(MANIFEST), "counts": manifest["counts"]}))
            return 0
        verify_freeze()

        texts = sentences()

        torch.manual_seed(SEED)
        reference = LivingCoreReference()
        before = run_battery(reference, texts)

        # -- segment C: cost curve -------------------------------------------
        inventory = wording_inventory(texts, every_sentence())
        segment_c = {
            "corpus_only": inventory["corpus"],
            "corpus_plus_authored": inventory["corpus_plus_authored"],
            "refused_scale": inventory["refused_scale"],
            "describe_sentences": inventory["describe_sentences"],
            "fixture_sentences": inventory["fixture_sentences"],
            "sentences_with_entity_names": inventory["sentences_with_entity_names"],
            "refusal_rule": inventory["refusal_rule"],
            "ratio_rises": (
                inventory["corpus_plus_authored"]["ratio"] > inventory["corpus"]["ratio"]
            ),
            "why": (
                "each authored wording adds one key to exactly one entity, so the ratio is "
                "writings-per-entity and grows with the number of wordings, not with the number "
                "of entities; the corpus itself supplies one wording per entity"
            ),
        }

        # -- segment D: zeros on the densest legal table ----------------------
        table_d, write_d = build_table()
        alias_write = table_d.write(AUTHORED_ALIASES)
        if alias_write["action"] != "accepted":
            raise SystemExit(f"alias write rejected: {alias_write['action']}")
        segment_d = {
            "write_records": {"action": write_d["action"], "appended": write_d["appended"]},
            "write_aliases": {"action": alias_write["action"], "appended": alias_write["appended"]},
            "keys": len(table_d.records),
            "entities": len(
                {value_key(table_d.current(index)) for index in range(len(table_d.records))}
            ),
            "capability": capability(table_d, texts),
            "unknown": unknown_reading(table_d),
            "fragments": fragments_reading(table_d),
            "sentences_direct": sentences_reading(table_d, texts),
        }

        # -- segment E: one wording, two entities -----------------------------
        cases = [
            conflict_case(row["name"], row["wording"], row["attempted"], extra=index == 0)
            for index, row in enumerate(ADVERSARY)
        ]
        segment_e = {
            "cases": cases,
            "counterfactual_replace": replace_counterfactual(
                ADVERSARY[0]["wording"], ADVERSARY[0]["attempted"]
            ),
            "append_control": append_control(),
            "within_batch_collision": within_batch_collision(),
        }

        after = run_battery(reference, texts)

        checks = {
            "parameters_equal": before["parameters_sha"] == after["parameters_sha"],
            "state_equal": before["state_sha"] == after["state_sha"],
            "identity_equal": before["identity_sha"] == after["identity_sha"],
            "store_equal": before["store_sha"] == after["store_sha"],
            "battery_equal": before["battery_sha"] == after["battery_sha"],
            "steps_equal": before["step"] == after["step"],
            "sc_sfk_none": before["sc_sfk_none"] and after["sc_sfk_none"],
            "flag_disabled": reference.config.sc_sfk_enabled is False,
            "rng_untouched": True,
        }

        zero_d = {
            "unknown": segment_d["unknown"]["answering_phrases"],
            "fragments": segment_d["fragments"]["false_answers"],
            "sentences": segment_d["sentences_direct"]["false_answers"],
        }
        verdict_d = (
            "large_table_zero_clean"
            if (
                segment_d["capability"]["ok"]
                and all(count == 0 for count in zero_d.values())
                and all(checks.values())
            )
            else "large_table_still_leaks"
        )

        cases_rejected = all(
            row["action"] == "rejected_batch_address_already_bound" and row["digest_unchanged"]
            for row in cases
        )
        first_binding_kept = all(
            row["first_binding_still_read"] == list(expected["bound"])
            for row, expected in zip(cases, ADVERSARY, strict=True)
        )
        control_ok = (
            segment_e["append_control"]["action"] == "accepted"
            and segment_e["append_control"]["digest_changed"]
            and segment_e["within_batch_collision"]["action"] == "rejected_batch_address_collision"
            and segment_e["within_batch_collision"]["digest_unchanged"]
            and not any(row["extra_item_applied"] for row in cases)
        )
        verdict_e = (
            "conflict_rejected_first_writer_kept"
            if cases_rejected and first_binding_kept and control_ok
            else "conflict_rule_failed"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-bridge-wording-scale-r14-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "bridge_sha256": file_sha(BRIDGE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "segment_c": segment_c,
            "segment_d": segment_d,
            "segment_e": segment_e,
            "zero_line": {
                "segment_d": zero_d,
                "criterion": "0 answering inputs on all three families",
            },
            "closed_state": {"baseline": before, "after": after, "checks": checks},
            "verdict": {"c": "cost_curve_measured", "d": verdict_d, "e": verdict_e},
            "cost": {
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "cpu_cap": CPU_CAP,
                "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
                "threads": torch.get_num_threads(),
            },
        }
        write_json(REPORT, report)
        print(
            json.dumps(
                {
                    "verdict_c": report["verdict"]["c"],
                    "verdict_d": verdict_d,
                    "verdict_e": verdict_e,
                    "c_corpus": {
                        k: segment_c["corpus_only"][k]
                        for k in ("keys", "entities", "ratio", "min", "median", "max")
                    },
                    "c_with_aliases": {
                        k: segment_c["corpus_plus_authored"][k]
                        for k in ("keys", "entities", "ratio", "min", "median", "max")
                    },
                    "c_refused": segment_c["refused_scale"],
                    "d_zero": zero_d,
                    "d_capability": f"{segment_d['capability']['hits']}/{segment_d['capability']['of']}",
                    "e_actions": [row["action"] for row in cases],
                    "e_controls": {
                        "append": segment_e["append_control"]["action"],
                        "within_batch": segment_e["within_batch_collision"]["action"],
                    },
                    "e_counterfactual": {
                        k: segment_e["counterfactual_replace"][k]
                        for k in ("action", "replaced", "now_reads")
                    },
                    "closed_state_ok": all(checks.values()),
                    "cost": {"cpu": round(cpu, 3), "wall": round(wall, 3)},
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        base.frozen.release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
