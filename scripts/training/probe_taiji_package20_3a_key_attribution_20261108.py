"""20.3a read-key replay: attribute the collapse without training.

Replays the exact 20.3a primary world (``fixture.build_world(SEED, 0)``) through
the same store-enabled CF1 graph, reconstructs every store write (slot -> the
query that generated the key) and every read (query -> the keys present at that
moment) from the replay log, and then measures how the read margin changes under
four key transforms fitted on the final 32 keys:

  raw                   keys as written
  mean_removed          keys - centroid (removes one shared constant direction)
  top_direction_removed keys - projection on the raw top singular direction
  pca_whitened          centred keys projected on the centred rank-r basis and
                        scaled to unit variance per direction

Pre-registered thresholds and decision tree (written before the replay):

  MARGIN_MIN        = 0.02  (same number as growth_design.THRESHOLDS["margin"])
  SEPARABLE_TOP_SHARE_MAX = 0.50  (a separated key set has no dominant direction)
  LOW_RANK_FRACTION = 0.50  (effective_rank_centered / min(n-1, width))
  CEILING_DELTA     = 0.05  (margin gain below this counts as no improvement)
  RECOVERY_DELTA    = 0.30  (reserved; recorded but not used by this replay)

  1. capacity_shortage : raw margin >= MARGIN_MIN (the keys already separate the
                         reads) AND the 32 slots are saturated with evictions
                         -> what is missing is slot capacity (growth lever).
  2. shared_offset     : raw margin < MARGIN_MIN, centring or removing the top
                         raw direction lifts the median margin to >= MARGIN_MIN,
                         AND the centred top_share <= SEPARABLE_TOP_SHARE_MAX
                         -> one shared constant direction; subtraction fixes it.
  3. intrinsic_low_rank: centring alone does not recover but whitening does
                         (margin >= MARGIN_MIN), AND the centred effective rank
                         is <= LOW_RANK_FRACTION * min(n-1, width)
                         -> keys live in a low-dimensional subspace; a new
                         projection structure is needed, not scaling.
  4. structure_ceiling : neither offset removal nor rank lifting reaches
                         MARGIN_MIN and the best margin gain is < CEILING_DELTA
                         -> the read-key structure is capped; growth is not
                         applicable to this problem.
  5. undecidable       : anything else; the report names what is missing.

No training, no optimizer step, no Adam, no confirm read, no new parameter.
None of the frozen 25 files is modified; the 20.3a probe is imported read-only.
Readings are appended to a jsonl with fsync; the report is written then renamed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import growth_design as gd  # noqa: E402
from taiji import rule_learning_20_1 as fixture  # noqa: E402

REPORT = ROOT / "reports/taiji_package20_3a_key_attribution_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_3a_key_attribution_20261108.json"
PROGRESS = ROOT / "reports/taiji_package20_3a_key_attribution_20261108_progress.jsonl"
SOURCE = ROOT / "scripts/training/probe_taiji_package20_3a_addressed_store.py"
SOURCE_REPORT = ROOT / "reports/taiji_package20_3a_addressed_store_20261105.json"

CPU_CAP = 300.0
CPU_GUARD = 285.0
MARGIN_MIN = 0.02
SEPARABLE_TOP_SHARE_MAX = 0.50
LOW_RANK_FRACTION = 0.50
CEILING_DELTA = 0.05
RECOVERY_DELTA = 0.30
PROBE_TURN = 34

STARTED = time.time()
STARTED_CPU = time.process_time()


def cost() -> tuple[float, float]:
    return time.process_time() - STARTED_CPU, time.time() - STARTED


def guard(stage: str) -> None:
    cpu = cost()[0]
    if cpu > CPU_GUARD:
        raise SystemExit(f"cpu budget exhausted at {stage}: {cpu:.2f}s")


def append_progress(record: dict) -> None:
    record = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_probe():
    spec = importlib.util.spec_from_file_location("probe_20_3a", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_20_3a"] = module
    spec.loader.exec_module(module)
    return module


def decode(raw):
    return raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8")


def store_sequence(probe, record: dict, world: dict) -> tuple[list[dict], dict[int, bytes], dict]:
    """Reconstruct the read events and the final slot -> query mapping.

    A read happens before the write of the same event, so a query never sees the
    key it is about to write.  The store is cleared at the first turn of a new
    session, exactly as ``RuleLearningGraph._with_store`` does.
    """
    turns = {int(turn["turn"]): turn for turn in world["response"]["turns"]}
    release_items = {
        (int(release["after_turn"]), int(release["version"])): release["items"]
        for release in world["response"]["releases"]
    }
    events_by_turn: dict[int, list[dict]] = {}
    for event in record["release_events"]:
        events_by_turn.setdefault(int(event["after_turn"]), []).append(event)

    store: dict[int, bytes] = {}
    reads: list[dict] = []
    counters = {"clears": 0, "turn_writes": 0, "release_writes": 0, "evictions": 0}
    for row in record["rows"]:
        turn = int(row["turn"])
        scope = row["store_scope"] or {}
        if scope.get("cleared"):
            store.clear()
            counters["clears"] += 1
        query = decode(turns[turn]["query"])
        reads.append(
            {
                "turn": turn,
                "source": "turn",
                "query": query,
                "present": list(store.values()),
            }
        )
        write = row["store_write"]
        if write and write.get("slot") is not None:
            store[int(write["slot"])] = query
            counters["turn_writes"] += 1
            counters["evictions"] += int(bool(write.get("evicted")))
        for event in events_by_turn.get(turn, []):
            items = release_items[(int(event["after_turn"]), int(event["version"]))]
            for index, item in enumerate(event["store_events"]):
                item_query = decode(items[index]["query"])
                reads.append(
                    {
                        "turn": turn,
                        "source": "release",
                        "query": item_query,
                        "present": list(store.values()),
                    }
                )
                item_write = item["write"]
                if item_write and item_write.get("slot") is not None:
                    store[int(item_write["slot"])] = item_query
                    counters["release_writes"] += 1
                    counters["evictions"] += int(bool(item_write.get("evicted")))
    return reads, store, counters


def occupied_high_water(reads: list[dict]) -> int:
    return max((len(event["present"]) for event in reads), default=0)


def fit_transforms(keys: np.ndarray) -> dict:
    center = keys.mean(axis=0, keepdims=True)
    _, singular_raw, basis_raw = np.linalg.svd(keys, full_matrices=False)
    top = basis_raw[0]
    centered = keys - center
    _, singular_centered, basis_centered = np.linalg.svd(centered, full_matrices=False)
    if singular_centered.size and float(singular_centered[0]) > 0:
        rank = int((singular_centered > 1e-6 * singular_centered[0]).sum())
    else:
        rank = 0
    rank = max(rank, 1)
    return {
        "center": center,
        "top": top,
        "basis": basis_centered[:rank].T,
        "scale": np.clip(singular_centered[:rank], 1e-3 * float(singular_centered[0]), None),
        "rank": rank,
        "singular_raw": singular_raw,
        "singular_centered": singular_centered,
    }


def apply_transform(values: np.ndarray, name: str, fit: dict) -> np.ndarray:
    if name == "raw":
        return values
    if name == "mean_removed":
        return values - fit["center"]
    if name == "top_direction_removed":
        return values - np.outer(values @ fit["top"], fit["top"])
    if name == "pca_whitened":
        return ((values - fit["center"]) @ fit["basis"]) / fit["scale"]
    raise ValueError(f"unknown transform {name!r}")


def read_margins(events: list[dict], key_of, fit: dict, name: str) -> dict:
    tops: list[float] = []
    margins: list[float] = []
    for event in events:
        if not event["present"]:
            continue
        query = apply_transform(key_of(event["query"])[None, :], name, fit)[0]
        keys = apply_transform(np.stack([key_of(raw) for raw in event["present"]]), name, fit)
        query_norm = float(np.linalg.norm(query))
        key_norms = np.linalg.norm(keys, axis=1)
        if query_norm <= 0 or not np.all(key_norms > 0):
            continue
        scores = np.sort((keys @ query) / np.clip(key_norms * query_norm, 1e-12, None))[::-1]
        tops.append(float(scores[0]))
        if scores.size >= 2:
            margins.append(float(scores[0] - scores[1]))
    return {
        "reads": len(tops),
        "median_top_cosine": float(np.median(tops)) if tops else None,
        "p95_top_cosine": float(np.percentile(tops, 95)) if tops else None,
        "median_margin": float(np.median(margins)) if margins else None,
        "p95_margin": float(np.percentile(margins, 95)) if margins else None,
    }


def final_geometry(keys: np.ndarray, fit: dict, name: str) -> dict:
    transformed = apply_transform(keys, name, fit)
    spectrum = gd.key_spectrum(transformed)
    margins = gd.read_margin_stats(transformed)
    return {
        "top_share": spectrum["top_share"],
        "top_share_centered": spectrum["top_share_centered"],
        "effective_rank": spectrum["effective_rank"],
        "effective_rank_centered": spectrum["effective_rank_centered"],
        "mean_abs_cos": spectrum["mean_abs_cos"],
        "p95_top_cos": margins["p95_top_cos"],
        "median_margin": margins["median_margin"],
    }


def probe_turn_read(record: dict, turn_number: int) -> dict | None:
    for row in record["rows"]:
        if int(row["turn"]) == turn_number:
            return row["store_read"]
    return None


def recorded_probe_read() -> dict | None:
    if not SOURCE_REPORT.exists():
        return None
    payload = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    read = payload.get("gates", {}).get("leverage", {}).get("read_at_probe")
    if not read:
        return None
    return {
        "occupied": read.get("occupied"),
        "top_cosine": read.get("top_cosine"),
        "second_cosine": read.get("second_cosine"),
        "margin": read.get("margin"),
    }


def main() -> int:
    probe = load_probe()
    probe.P2M = probe.load_module(probe.P2, "probe_20_2_credit")
    guard("import")

    append_progress(
        {
            "stage": "thresholds",
            "written_before_reading": True,
            "margin_min": MARGIN_MIN,
            "separable_top_share_max": SEPARABLE_TOP_SHARE_MAX,
            "low_rank_fraction": LOW_RANK_FRACTION,
            "ceiling_delta": CEILING_DELTA,
            "recovery_delta_unused": RECOVERY_DELTA,
            "replay_runner_sha256": file_sha(Path(__file__)),
            "source_runner_sha256": file_sha(SOURCE),
            "cpu": round(cost()[0], 3),
        }
    )

    bundle = torch.load(probe.BUNDLE, map_location="cpu", weights_only=False)
    candidate = probe.load_routing_bundle(probe.BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    parent = candidate.subject
    graph = probe.build_graph(parent, gate, "CF1", True)
    world = fixture.build_world(fixture.SEED, 0)
    guard("setup")

    record = probe.run_world(graph, world)
    guard("replay")

    final = record["final_state"].living
    if final.store_keys is None or final.store_age is None:
        raise AssertionError("store is not attached on this graph")
    occupied_mask = (final.store_age >= 0).detach().cpu().numpy()
    all_keys = final.store_keys.detach().cpu().double().numpy()
    occupied_keys = all_keys[occupied_mask]
    occupied_ids = [int(index) for index in np.flatnonzero(occupied_mask)]

    reads, final_map, counters = store_sequence(probe, record, world)
    if set(final_map) != set(occupied_ids):
        raise AssertionError(
            f"reconstructed slots {sorted(final_map)} != occupied slots {sorted(occupied_ids)}"
        )

    cache: dict[bytes, np.ndarray] = {}

    def key_of(raw: bytes) -> np.ndarray:
        if raw not in cache:
            cache[raw] = graph.subject.store_query_key(final, raw).detach().cpu().double().numpy()
        return cache[raw]

    fit = fit_transforms(occupied_keys)
    names = ("raw", "mean_removed", "top_direction_removed", "pca_whitened")
    turn_reads = [event for event in reads if event["source"] == "turn"]
    margins = {}
    geometry = {}
    for name in names:
        margins[name] = {
            "turn": read_margins(turn_reads, key_of, fit, name),
            "all": read_margins(reads, key_of, fit, name),
        }
        geometry[name] = final_geometry(occupied_keys, fit, name)
        guard(f"transform {name}")

    probe_row = probe_turn_read(record, PROBE_TURN)
    replayed_probe_read = None
    for event in turn_reads:
        if event["turn"] != PROBE_TURN or not event["present"]:
            continue
        query = key_of(event["query"])
        keys = np.stack([key_of(raw) for raw in event["present"]])
        scores = np.sort(
            (keys @ query)
            / np.clip(np.linalg.norm(keys, axis=1) * float(np.linalg.norm(query)), 1e-12, None)
        )[::-1]
        replayed_probe_read = {
            "occupied": int(len(event["present"])),
            "top_cosine": float(scores[0]),
            "second_cosine": float(scores[1]) if scores.size > 1 else None,
            "margin": float(scores[0] - scores[1]) if scores.size > 1 else None,
        }
        break

    identities = {}
    for slot, raw in final_map.items():
        identities.setdefault(raw, []).append(slot)
    duplicate_identities = sum(1 for slots in identities.values() if len(slots) > 1)

    raw_margin = margins["raw"]["turn"]["median_margin"] or 0.0
    centered_margin = max(
        margins["mean_removed"]["turn"]["median_margin"] or 0.0,
        margins["top_direction_removed"]["turn"]["median_margin"] or 0.0,
    )
    whitened_margin = margins["pca_whitened"]["turn"]["median_margin"] or 0.0
    best_margin = max(centered_margin, whitened_margin)
    gain = best_margin - raw_margin
    centered_top_share = geometry["mean_removed"]["top_share_centered"]
    centered_rank = geometry["mean_removed"]["effective_rank_centered"]
    dims = max(min(len(occupied_keys) - 1, occupied_keys.shape[1]), 1)
    centered_rank_fraction = centered_rank / dims
    saturated = len(occupied_ids) >= all_keys.shape[0]

    capacity_pass = bool(raw_margin >= MARGIN_MIN and saturated and counters["evictions"] > 0)
    high_water = occupied_high_water(reads)
    shared_pass = bool(
        raw_margin < MARGIN_MIN
        and centered_margin >= MARGIN_MIN
        and centered_top_share <= SEPARABLE_TOP_SHARE_MAX
    )
    low_rank_pass = bool(
        not shared_pass
        and whitened_margin >= MARGIN_MIN
        and centered_rank_fraction <= LOW_RANK_FRACTION
    )
    ceiling_pass = bool(best_margin < MARGIN_MIN and gain < CEILING_DELTA)
    if capacity_pass:
        attribution = "capacity_shortage"
    elif shared_pass:
        attribution = "shared_offset"
    elif low_rank_pass:
        attribution = "intrinsic_low_rank"
    elif ceiling_pass:
        attribution = "structure_ceiling"
    else:
        attribution = "undecidable"

    cases = {
        "capacity_shortage": {
            "pass": capacity_pass,
            "decidable": bool(saturated or counters["evictions"] > 0),
            "criterion": ("raw margin >= MARGIN_MIN and occupied == slots and evictions > 0"),
            "evidence": {
                "raw_margin": raw_margin,
                "occupied_final": len(occupied_ids),
                "slots": int(all_keys.shape[0]),
                "occupied_high_water": high_water,
                "evictions": counters["evictions"],
            },
            "missing_if_undecidable": (
                "slot capacity never binds in this world (high water "
                f"{high_water}/{int(all_keys.shape[0])}, {counters['evictions']} evictions), "
                "so slot shortage and key-dimension insufficiency cannot be exercised; "
                "needs a load-saturated replay or a controlled key-width ablation"
            ),
        },
        "shared_offset": {
            "pass": shared_pass,
            "criterion": (
                "raw margin < MARGIN_MIN and centred margin >= MARGIN_MIN and "
                "centred top_share <= SEPARABLE_TOP_SHARE_MAX"
            ),
            "evidence": {
                "raw_margin": raw_margin,
                "centered_margin": centered_margin,
                "centered_top_share": centered_top_share,
            },
            "missing_if_undecidable": None,
        },
        "intrinsic_low_rank": {
            "pass": low_rank_pass,
            "criterion": (
                "centred margin < MARGIN_MIN and whitened margin >= MARGIN_MIN and "
                "centred effective_rank_fraction <= LOW_RANK_FRACTION"
            ),
            "evidence": {
                "centered_margin": centered_margin,
                "whitened_margin": whitened_margin,
                "centered_rank": centered_rank,
                "centered_rank_fraction": centered_rank_fraction,
                "rank_dims": dims,
            },
            "missing_if_undecidable": None,
        },
        "structure_ceiling": {
            "pass": ceiling_pass,
            "criterion": ("best margin < MARGIN_MIN and best margin gain < CEILING_DELTA"),
            "evidence": {"best_margin": best_margin, "gain": gain},
            "missing_if_undecidable": None,
        },
    }
    if attribution == "undecidable":
        cases["undecidable"] = {
            "pass": True,
            "reason": (
                "margin improves by more than CEILING_DELTA but stays below "
                "MARGIN_MIN under every fitted transform, or no read event was "
                "available; a labelled retrieval target or a supervised "
                "attribute probe is required to separate residual ambiguity "
                "from structural noise"
            ),
            "evidence": {
                "raw_margin": raw_margin,
                "best_margin": best_margin,
                "gain": gain,
                "reads": margins["raw"]["turn"]["reads"],
            },
        }
    else:
        cases["undecidable"] = {"pass": False, "reason": None, "evidence": None}

    recorded_actual = recorded_probe_read()
    match = None
    if recorded_actual and replayed_probe_read and recorded_actual["top_cosine"] is not None:
        match = {
            "delta_top_cosine": abs(
                float(recorded_actual["top_cosine"]) - replayed_probe_read["top_cosine"]
            ),
            "delta_margin": (
                None
                if recorded_actual["margin"] is None or replayed_probe_read["margin"] is None
                else abs(float(recorded_actual["margin"]) - replayed_probe_read["margin"])
            ),
            "tolerance": 1e-4,
        }
        match["reproduced"] = bool(
            match["delta_top_cosine"] <= match["tolerance"]
            and (match["delta_margin"] is None or match["delta_margin"] <= match["tolerance"])
        )

    payload = {
        "format": "taiji-20-3a-key-attribution-v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "replay_runner_sha256": file_sha(Path(__file__)),
        "source_runner_sha256": file_sha(SOURCE),
        "parent_sha256": probe.P2M.weights_digest(parent),
        "world": {
            "seed": fixture.SEED,
            "index": 0,
            "turns": len(world["response"]["turns"]),
            "reads_turn": len(turn_reads),
            "reads_all": len(reads),
        },
        "thresholds": {
            "margin_min": MARGIN_MIN,
            "separable_top_share_max": SEPARABLE_TOP_SHARE_MAX,
            "low_rank_fraction": LOW_RANK_FRACTION,
            "ceiling_delta": CEILING_DELTA,
            "recovery_delta_unused": RECOVERY_DELTA,
        },
        "store": {
            "occupied": len(occupied_ids),
            "slots": int(all_keys.shape[0]),
            "unique_query_identities": len(identities),
            "duplicate_identities": duplicate_identities,
            **counters,
        },
        "reproduction": {
            "recorded_20_3a_probe_read": recorded_actual,
            "replayed_probe_read": replayed_probe_read,
            "match": match,
            "replay_row_read_at_probe": probe_row,
        },
        "transforms": {
            "rank": fit["rank"],
            "singular_raw": [float(value) for value in fit["singular_raw"]],
            "singular_centered": [float(value) for value in fit["singular_centered"]],
            "whitening_floor_ratio": 1e-3,
            "fitted_on": "final occupied store keys",
        },
        "read_margins": margins,
        "final_geometry": geometry,
        "decision": {
            "attribution": attribution,
            "raw_margin": raw_margin,
            "centered_margin": centered_margin,
            "whitened_margin": whitened_margin,
            "best_margin": best_margin,
            "gain": gain,
            "centered_top_share": centered_top_share,
            "centered_effective_rank": centered_rank,
            "centered_rank_fraction": centered_rank_fraction,
            "capacity_saturated": saturated,
            "occupied_high_water": high_water,
            "limitation": (
                "margin is geometric discriminability (top-1 vs top-2 cosine), not task "
                "utility; a larger margin does not by itself prove the 20.3a store would "
                "improve the rule task"
            ),
        },
        "cases": cases,
        "cpu_seconds": round(cost()[0], 3),
        "wall_seconds": round(cost()[1], 3),
    }
    guard("write")
    write_json(REPORT, payload)
    manifest = {
        "report": REPORT.relative_to(ROOT).as_posix(),
        "report_sha256": file_sha(REPORT),
        "replay_runner_sha256": payload["replay_runner_sha256"],
        "source_runner_sha256": payload["source_runner_sha256"],
        "attribution": attribution,
        "cpu_seconds": payload["cpu_seconds"],
        "wall_seconds": payload["wall_seconds"],
    }
    write_json(MANIFEST, manifest)
    append_progress(
        {
            "stage": "written",
            "attribution": attribution,
            "decision": payload["decision"],
            "cpu": round(cost()[0], 3),
        }
    )
    print(json.dumps(manifest, ensure_ascii=False))
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
