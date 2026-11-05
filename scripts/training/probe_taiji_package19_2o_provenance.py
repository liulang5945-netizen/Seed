"""19.2o: zero-update provenance and fabrication recompute on the four step300 bundles.

Owner ruling 2026-11-05 (D16-20, ledger section 16.13) replaced the single
accuracy gate with three acceptance tiers:

* L1 mechanical invariants (100%, zero tolerance): bitwise restore, UTF-8
  legality, confirm read count, parent / candidate identity;
* L2 content fidelity (100% auditable, not 100% correct): ``provenance_rate``
  and ``fabricated_bytes``, with ``misbound_bytes`` as a third, separately
  counted class;
* L3 selection accuracy reported as selection / random baseline / reachable
  ceiling, with thresholds to be ruled on later.

This package recomputes L2 on the four 19.2j step300 bundles (seeds 28031/28032
x arms O/N) with **0 Adam updates**, **0 saved bundles**, **0 confirm reads**,
and no change to any judging rule.  The slice size is chosen once from a single
8-example price reading so the whole package stays inside 150 CPU (D16-17
magnitude gate: stop only if the estimate exceeds 2 x the cap).

Content definition (fixed before the run): the two names inside the action,
object bytes 3-14 (12 bytes) and place bytes 21-26 (6 bytes), so 18 bytes per
example.  A span counts as provable only when its bytes occur **contiguously**
in the record text, which avoids single-byte coincidences in UTF-8 Chinese text.

Every span falls into exactly one of:

* ``provenance`` - contiguous in the bound record (the route the model chose);
* ``misbound``  - not in the bound record, but contiguous in another candidate;
* ``fabricated`` - contiguous in no candidate record.

and, independently, one of four selection/copy cells:

* ``aligned`` (right route, right content);
* ``right_route_wrong_content`` (selected right, copied wrong);
* ``wrong_route_right_content`` (selected wrong, copied right - the key cell
  that separates "selection error" from "fabrication");
* ``wrong_route_wrong_content`` (selected wrong, copied wrong), with a subtype
  for bound-record copy / other-record copy / fabrication / truncation.

Run: ``python scripts/training/probe_taiji_package19_2o_provenance.py``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

torch.set_num_threads(1)

import attribute_taiji_package19_2f_new_description_wall as wall  # noqa: E402
import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402
import train_taiji_package19_2j_from_scratch_selection_ab as j  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

ROUTING_ARM = "B_stop_lock"
CPU_CAP = 150.0
BUDGET_GATE_MULTIPLE = 2.0
PRICING_EXAMPLES = 8
PRICING_OVERHEAD_CPU = 10.0
SLICE_OPTIONS = (96, 72, 60, 48, 36, 24)
OBJECT_START = 3
OBJECT_BYTES = 12
PLACE_START = 21
PLACE_BYTES = 6
CONTENT_BYTES_PER_EXAMPLE = OBJECT_BYTES + PLACE_BYTES
RANDOM_BASELINE = 1.0 / 9.0  # 8 records + one null row
NON_NULL_BASELINE = 1.0 / 8.0
REACHABLE_CEILING = 1.0  # 18.1 c_gamma0 = 1.000 byte exact
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 20261105

BUNDLES = {
    "28031-O": ROOT
    / "output/taiji_package19_2j_from_scratch_selection_ab/28031-O_baseline-step300.pt",
    "28031-N": ROOT
    / "output/taiji_package19_2j_from_scratch_selection_ab/28031-N_alignment-step300.pt",
    "28032-O": ROOT
    / "output/taiji_package19_2j_from_scratch_selection_ab/28032-O_baseline-step300.pt",
    "28032-N": ROOT
    / "output/taiji_package19_2j_from_scratch_selection_ab/28032-N_alignment-step300.pt",
}
TRAIN_MANIFEST = ROOT / "plans/manifests/taiji_package19_2j_from_scratch_selection_ab_20261104.json"
PARENT = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"

REPORT = ROOT / "reports/taiji_package19_2o_provenance_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2o_provenance_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2o_provenance_progress_20261105.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_progress(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def world_slice(count: int) -> tuple[int, ...]:
    """Even stride over the 16 dev indices of every world (8-15 are new descriptions)."""
    per_world = count // 12
    if per_world < 1 or count % 12:
        raise ValueError(f"slice must be a multiple of 12, got {count}")
    if per_world == 1:
        return (0,)
    indices = {round(k * 15 / (per_world - 1)) for k in range(per_world)}
    if len(indices) != per_world:
        raise RuntimeError(f"slice indices collided for per_world={per_world}")
    return tuple(sorted(indices))


def choose_slice(price: float) -> int:
    for count in SLICE_OPTIONS:
        if 4 * count * price + PRICING_OVERHEAD_CPU <= CPU_CAP:
            return count
    raise RuntimeError(f"even the smallest slice exceeds the cap at price {price:.4f}")


def analyze_span(fixture: dict, decoded: dict, start: int, length: int, kind: str) -> dict:
    target = wall.payload_bytes(fixture)
    mapping = probe_mod.content_positions(fixture)
    raw = decoded["raw"]
    emitted = bytes(raw[start : start + length])
    gold = bytes(target[start : start + length])
    gold_index, _gold_offset = mapping[start]
    route = decoded["routes"][start] if start < len(decoded["routes"]) else -1
    records = fixture["raw_example"]["records"]
    bound_text = records[route]["raw"].encode("utf-8") if 0 <= route < len(records) else b""
    other_texts = [
        record["raw"].encode("utf-8") for index, record in enumerate(records) if index != route
    ]
    truncated = len(emitted) != length
    in_bound = (not truncated) and emitted in bound_text
    in_other = (not truncated) and any(emitted in text for text in other_texts)
    route_ok = route == gold_index
    content_ok = (not truncated) and emitted == gold
    if truncated:
        cell = "truncated"
        subtype = "truncated"
    elif route_ok and content_ok:
        cell = "aligned"
        subtype = None
    elif route_ok:
        cell = "right_route_wrong_content"
        subtype = (
            "copied_bound_record"
            if in_bound
            else ("copied_other_record" if in_other else "fabricated")
        )
    elif content_ok:
        cell = "wrong_route_right_content"
        subtype = None
    else:
        cell = "wrong_route_wrong_content"
        subtype = (
            "copied_bound_record"
            if in_bound
            else ("copied_other_record" if in_other else "fabricated")
        )
    if in_bound:
        provenance, misbound, fabricated = length, 0, 0
    elif in_other:
        provenance, misbound, fabricated = 0, length, 0
    else:
        provenance, misbound, fabricated = 0, 0, length
    return {
        "world": fixture["world"],
        "index": fixture["index"],
        "start": start,
        "kind": kind,
        "new_description": bool(fixture["flags"]["new_description"]),
        "route": route,
        "gold_route": gold_index,
        "route_ok": bool(route_ok),
        "content_ok": bool(content_ok),
        "truncated": bool(truncated),
        "cell": cell,
        "subtype": subtype,
        "bytes": length,
        "provenance": provenance,
        "misbound": misbound,
        "fabricated": fabricated,
        "emitted": emitted.decode("utf-8", errors="replace"),
        "gold": gold.decode("utf-8", errors="replace"),
    }


def bucket_add(target: dict, key: str, row: dict, length: int) -> None:
    cell = target.setdefault(key, {"spans": 0, "bytes": 0})
    cell["spans"] += 1
    cell["bytes"] += length
    for field in ("provenance", "misbound", "fabricated"):
        cell[field] = cell.get(field, 0) + row[field]


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def bootstrap_interval(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "low": None, "high": None}
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    means = []
    for _round in range(BOOTSTRAP_ROUNDS):
        picks = torch.randint(0, len(values), (len(values),), generator=generator)
        means.append(statistics.fmean(values[int(index)] for index in picks))
    means.sort()
    return {
        "mean": statistics.fmean(values),
        "low": means[int(0.025 * BOOTSTRAP_ROUNDS)],
        "high": means[int(0.975 * BOOTSTRAP_ROUNDS) - 1],
    }


def price_one_bundle(dev: tuple[dict, ...]) -> float:
    candidate, _optimizer, _state, _index, _cursor = load_routing_bundle(BUNDLES["28031-O"])
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = ROUTING_ARM
    priced = tuple(fixture for fixture in dev if fixture["index"] in (0, 15))[:PRICING_EXAMPLES]
    start_cpu = time.process_time()
    for fixture in priced:
        wall.decode_fixture(probe, fixture)
    return (time.process_time() - start_cpu) / len(priced)


def evaluate_bundle(
    name: str,
    path: Path,
    dev: tuple[dict, ...],
    indices: tuple[int, ...],
    handle,
) -> dict:
    candidate, _optimizer, _state, _index, _cursor = load_routing_bundle(path)
    identity = j.candidate_identity(candidate)
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = ROUTING_ARM
    fixtures = tuple(fixture for fixture in dev if fixture["index"] in indices)
    rows = []
    utf8_illegal = 0
    completion = 0
    for fixture in fixtures:
        decoded = wall.decode_fixture(probe, fixture)
        try:
            bytes(decoded["raw"]).decode("utf-8")
        except UnicodeDecodeError:
            utf8_illegal += 1
        completion += int(wall.example_metrics(fixture, decoded)["completed"])
        for start, length, kind in (
            (OBJECT_START, OBJECT_BYTES, "object"),
            (PLACE_START, PLACE_BYTES, "place"),
        ):
            span = analyze_span(fixture, decoded, start, length, kind)
            rows.append(span)
            append_progress(
                handle,
                {
                    "event": "span",
                    "bundle": name,
                    "world": span["world"],
                    "index": span["index"],
                    "kind": kind,
                    "cell": span["cell"],
                    "subtype": span["subtype"],
                    "route_ok": span["route_ok"],
                    "content_ok": span["content_ok"],
                },
            )

    total_bytes = sum(row["bytes"] for row in rows)
    provenance_bytes = sum(row["provenance"] for row in rows)
    misbound_bytes = sum(row["misbound"] for row in rows)
    fabricated_bytes = sum(row["fabricated"] for row in rows)
    categories: dict[str, dict] = {}
    subtypes: dict[str, dict] = {}
    by_kind: dict[str, dict] = {}
    by_age: dict[str, dict] = {}
    for row in rows:
        bucket_add(categories, row["cell"], row, row["bytes"])
        if row["subtype"]:
            bucket_add(subtypes, f"{row['cell']}/{row['subtype']}", row, row["bytes"])
        bucket_add(by_kind, row["kind"], row, row["bytes"])
        bucket_add(
            by_age,
            "new_description" if row["new_description"] else "old_description",
            row,
            row["bytes"],
        )

    object_rows = [row for row in rows if row["kind"] == "object"]
    place_rows = [row for row in rows if row["kind"] == "place"]
    selection = rate(sum(int(row["route_ok"]) for row in rows), len(rows))
    selection_flags = [1.0 if row["route_ok"] else 0.0 for row in rows]
    return {
        "bundle": name,
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256_file(path),
        "candidate_sha_v2": identity["candidate_sha_v2"],
        "candidate_identity": identity,
        "examples": len(fixtures),
        "spans": len(rows),
        "utf8_illegal_examples": utf8_illegal,
        "completion_rate": rate(completion, len(fixtures)),
        "total_content_bytes": total_bytes,
        "provenance_bytes": provenance_bytes,
        "misbound_bytes": misbound_bytes,
        "fabricated_bytes": fabricated_bytes,
        "truncated_bytes": sum(row["bytes"] for row in rows if row["truncated"]),
        "provenance_rate": rate(provenance_bytes, total_bytes),
        "misbound_rate": rate(misbound_bytes, total_bytes),
        "fabricated_rate": rate(fabricated_bytes, total_bytes),
        "accounting_closed": provenance_bytes + misbound_bytes + fabricated_bytes == total_bytes,
        "selection": selection,
        "selection_object": rate(
            sum(int(row["route_ok"]) for row in object_rows), len(object_rows)
        ),
        "selection_place": rate(sum(int(row["route_ok"]) for row in place_rows), len(place_rows)),
        "selection_bootstrap": bootstrap_interval(selection_flags),
        "random_baseline": RANDOM_BASELINE,
        "random_baseline_non_null": NON_NULL_BASELINE,
        "reachable_ceiling": REACHABLE_CEILING,
        "relative_to_ceiling": rate(
            selection - RANDOM_BASELINE, REACHABLE_CEILING - RANDOM_BASELINE
        ),
        "categories": categories,
        "subtypes": subtypes,
        "by_kind": by_kind,
        "by_age": by_age,
    }


def write_report_and_manifest(report: dict, manifest: dict) -> None:
    tmp = REPORT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, REPORT)
    manifest["report"] = {
        "path": str(REPORT.relative_to(ROOT)),
        "sha256": sha256_file(REPORT),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = MANIFEST.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, MANIFEST)


def l1_identity_checks(results: dict, expected: dict) -> dict[str, dict]:
    checks = {}
    for name, result in results.items():
        expected_cell = expected[name]
        checks[name] = {
            "file_sha256_match": result["file_sha256"] == expected_cell["file_sha256"],
            "candidate_sha_v2_match": result["candidate_sha_v2"]
            == expected_cell["candidate_sha_v2"],
        }
    return checks


def identity_only() -> int:
    """Recheck L1 bundle identity from the finished run; no decode, no reload of weights.

    The first run stored the whole identity block under ``candidate_sha_v2``, so the
    string comparison against the 19.2j manifest failed on type alone.  The digest
    itself was produced correctly by that run; this pass re-derives the comparison
    and re-hashes the files, and moves the full block to ``candidate_identity``.
    """
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_sha = sha256_file(REPORT)
    expected = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))["bundles"]

    results = {}
    for name, path in BUNDLES.items():
        cell = report["bundles"][name]
        block = cell.get("candidate_identity")
        if not isinstance(block, dict):
            block = cell["candidate_sha_v2"]
        results[name] = {
            "file_sha256": sha256_file(path),
            "candidate_sha_v2": block["candidate_sha_v2"],
        }
        cell["candidate_sha_v2"] = block["candidate_sha_v2"]
        cell["candidate_identity"] = block

    checks = l1_identity_checks(results, expected)
    l1 = report["acceptance_tiers"]["L1"]
    l1["identity_checks"] = checks
    l1["identity_recompute"] = {
        "mode": "identity_only",
        "source_report_sha256": source_sha,
        "note": "first run compared a dict against a string; digests were recomputed, "
        "files re-hashed, no decode and no weight reload",
    }
    manifest["bundles"] = {
        name: {
            "file_sha256": results[name]["file_sha256"],
            "candidate_sha_v2": results[name]["candidate_sha_v2"],
        }
        for name in results
    }

    cost_cpu = time.process_time() - start_cpu
    cost_wall = time.perf_counter() - start_wall
    report["cost"]["cpu_seconds"] = report["cost"]["cpu_seconds"] + cost_cpu
    report["cost"]["wall_seconds"] = report["cost"]["wall_seconds"] + cost_wall
    report["cost"]["over_budget"] = report["cost"]["cpu_seconds"] > CPU_CAP
    manifest["cost_cpu"] = report["cost"]["cpu_seconds"]
    l1["identity_recompute"]["cost_cpu"] = cost_cpu
    write_report_and_manifest(report, manifest)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        append_progress(
            handle,
            {
                "event": "identity_recompute",
                "checks": checks,
                "cost_cpu": cost_cpu,
                "all_match": all(all(row.values()) for row in checks.values()),
            },
        )
    print(
        json.dumps({"identity_checks": checks, "cost_cpu": cost_cpu}, ensure_ascii=False, indent=2)
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-only", action="store_true")
    args = parser.parse_args(argv)
    if args.identity_only:
        return identity_only()

    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    PROGRESS.unlink(missing_ok=True)

    payload = premise.build_payload()
    dev = premise.fixtures(payload, "dev")
    expected = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))["bundles"]

    with PROGRESS.open("a", encoding="utf-8") as handle:
        append_progress(
            handle,
            {
                "event": "setup",
                "dev_examples": len(dev),
                "bundles": list(BUNDLES),
                "cpu_cap": CPU_CAP,
                "adam_updates": 0,
                "confirm_reads": 0,
            },
        )
        price = price_one_bundle(dev)
        slice_count = choose_slice(price)
        indices = world_slice(slice_count)
        estimate = 4 * slice_count * price + PRICING_OVERHEAD_CPU
        append_progress(
            handle,
            {
                "event": "pricing",
                "price_cpu_per_example": price,
                "slice_per_bundle": slice_count,
                "indices": list(indices),
                "estimate_cpu": estimate,
                "over_budget": estimate > CPU_CAP * BUDGET_GATE_MULTIPLE,
            },
        )

        results = {}
        for name, path in BUNDLES.items():
            results[name] = evaluate_bundle(name, path, dev, indices, handle)

        pooled = {"spans": 0, "bytes": 0}
        for result in results.values():
            pooled["spans"] += result["spans"]
            pooled["bytes"] += result["total_content_bytes"]
        pooled["provenance_bytes"] = sum(r["provenance_bytes"] for r in results.values())
        pooled["misbound_bytes"] = sum(r["misbound_bytes"] for r in results.values())
        pooled["fabricated_bytes"] = sum(r["fabricated_bytes"] for r in results.values())
        pooled["provenance_rate"] = rate(pooled["provenance_bytes"], pooled["bytes"])
        pooled["misbound_rate"] = rate(pooled["misbound_bytes"], pooled["bytes"])
        pooled["fabricated_rate"] = rate(pooled["fabricated_bytes"], pooled["bytes"])
        pooled["accounting_closed"] = (
            pooled["provenance_bytes"] + pooled["misbound_bytes"] + pooled["fabricated_bytes"]
            == pooled["bytes"]
        )

        identity_checks = l1_identity_checks(results, expected)

        report = {
            "schema": "taiji-package19-2o-provenance-v1",
            "step": "19.2o",
            "date": "2026-11-05",
            "purpose": "D16-20 L2 recompute: provenance_rate / fabricated_bytes and the "
            "selection-vs-copy split",
            "acceptance_tiers": {
                "L1": {
                    "definition": "bitwise restore, UTF-8 legality, confirm reads == 0, "
                    "parent/candidate identity",
                    "restore": "not rerun here (no bundle written); last evidence 19.2l-B "
                    "roundtrip max_abs_diff = 0.0",
                    "utf8_illegal_examples": {
                        name: result["utf8_illegal_examples"] for name, result in results.items()
                    },
                    "confirm_reads": 0,
                    "identity_checks": identity_checks,
                    "parent": {
                        "path": str(PARENT.relative_to(ROOT)),
                        "sha256": sha256_file(PARENT),
                    },
                },
                "L2": {
                    "definition": "content bytes 18/example; provenance = contiguous in bound "
                    "record; fabricated = contiguous in no candidate; misbound = contiguous in "
                    "another candidate",
                    "content_positions": {
                        "object": [OBJECT_START, OBJECT_START + OBJECT_BYTES],
                        "place": [PLACE_START, PLACE_START + PLACE_BYTES],
                        "bytes_per_example": CONTENT_BYTES_PER_EXAMPLE,
                    },
                    "pooled": pooled,
                },
                "L3": {
                    "definition": "selection vs random baseline 1/9 (non-null 1/8) and reachable "
                    "ceiling 1.0; thresholds ruled later by owner",
                    "pooled_selection": rate(
                        sum(result["selection"] for result in results.values()), len(results)
                    ),
                    "random_baseline": RANDOM_BASELINE,
                    "random_baseline_non_null": NON_NULL_BASELINE,
                    "reachable_ceiling": REACHABLE_CEILING,
                },
            },
            "slice": {
                "per_bundle_examples": slice_count,
                "indices_per_world": list(indices),
                "price_cpu_per_example": price,
                "estimate_cpu": estimate,
                "cpu_cap": CPU_CAP,
                "budget_gate_multiple": BUDGET_GATE_MULTIPLE,
                "pricing_reads": 1,
            },
            "bundles": results,
            "fabrication_split": {
                "aligned": "select right, copy right",
                "right_route_wrong_content": "select right, copy wrong",
                "wrong_route_right_content": "select wrong, copy right",
                "wrong_route_wrong_content": "select wrong, copy wrong",
                "truncated_bytes_are_subset_of_fabricated": True,
            },
            "discipline": {
                "adam_updates": 0,
                "bundles_saved": 0,
                "confirm_reads": 0,
                "judging_rule_changed": False,
                "thresholds_changed": False,
                "eval_slice_changed": False,
                "fsync_per_line": True,
                "format_gate": "closing commit only",
            },
            "cost": {
                "cpu_seconds": None,
                "wall_seconds": None,
                "cap_cpu": CPU_CAP,
                "over_budget": False,
            },
            "inputs": {
                "train_manifest": {
                    "path": str(TRAIN_MANIFEST.relative_to(ROOT)),
                    "sha256": sha256_file(TRAIN_MANIFEST),
                },
                "script": {
                    "path": str(Path(__file__).resolve().relative_to(ROOT)),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            },
        }
        report["cost"]["cpu_seconds"] = time.process_time() - start_cpu
        report["cost"]["wall_seconds"] = time.perf_counter() - start_wall
        report["cost"]["over_budget"] = report["cost"]["cpu_seconds"] > CPU_CAP
        append_progress(handle, {"event": "report", "cost": report["cost"]})

    manifest = {
        "schema": "taiji-package19-2o-manifest-v1",
        "step": "19.2o",
        "date": "2026-11-05",
        "progress": {
            "path": str(PROGRESS.relative_to(ROOT)),
            "sha256": sha256_file(PROGRESS),
        },
        "bundles": {
            name: {
                "file_sha256": result["file_sha256"],
                "candidate_sha_v2": result["candidate_sha_v2"],
            }
            for name, result in results.items()
        },
        "provenance_rate": pooled["provenance_rate"],
        "fabricated_bytes": pooled["fabricated_bytes"],
        "cost_cpu": report["cost"]["cpu_seconds"],
        "adam_updates": 0,
        "bundles_saved": 0,
        "confirm_reads": 0,
    }
    write_report_and_manifest(report, manifest)

    print(
        json.dumps(
            {
                "slice": report["slice"],
                "pooled": pooled,
                "identity_checks": identity_checks,
                "cost": report["cost"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
