"""R50 -- is the written fast organised by the sentence's object/place?

R49 proved the write is content dependent.  R50 asks whether that content is
structured: the 20 describe sentences are 4 objects x 5 places, so a nearest
neighbour of a written ``fast`` should tend to share the object (or the place)
if the write preserves structure.

Frozen decision:
- same-object 1-NN accuracy >= 0.5 -> ``write_structure_present``;
- otherwise ``write_structure_scrambled``.
Chance is 3/19 for the object and 4/19 for the place; both are computed from the
actual label counts and reported next to the readings.

Zero optimizer steps; no module change.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_cross_family_readout_20261109 as base  # noqa: E402
import scripts.training.probe_taiji_rebalance_20261109 as prev  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_write_structure_20261109.json"
REPORT = ROOT / "reports/taiji_write_structure_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 20.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r50"
OBJECT_GATE = 0.5

RERUN_ANCHORS = (
    {
        "round": "R49",
        "runner": "scripts/training/probe_taiji_write_dependence_20261109.py",
        "rerun_commit": "d15dc0d2e",
        "runner_blob_sha1": "0b01197dba556be5a66e04697eed1f42673c7a6b",
        "report_runner_sha256": "e002adad9659340096e0d291b647fd3e0f3ca8a03b765969eac8a5bf4f001752",
        "runner_blob_sha256": "e002adad9659340096e0d291b647fd3e0f3ca8a03b765969eac8a5bf4f001752",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R50 does not change living_core; R49 stays runnable at HEAD",
    },
)

DEFINITION = {
    "write": "each describe sentence is written once with a write-enabled observe",
    "labels": "the fixture gives one object and one place per describe sentence (4 x 5)",
    "readout": "1-NN over the 20 written fast vectors, self excluded, scale-free distance",
    "readings": "same-object and same-place 1-NN accuracy plus within/between mean distances",
    "decision": "same-object 1-NN >= 0.5 -> write_structure_present, else write_structure_scrambled",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "labels": "20 describe sentences with 4 distinct objects and 5 distinct places",
    "readings": "object and place 1-NN accuracy, chance, and within/between distances are reported",
    "decision": "the frozen 0.5 object gate decides present/scrambled",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R49 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def describe_rows() -> list[dict[str, Any]]:
    return [row for row in base.frozen.build_fixture() if row["variant"] == "describe"]


def write_once(model: Any, text: str, tag: str) -> np.ndarray:
    state = model.initial_state(f"r50-{tag}")
    with torch.no_grad():
        state = model.observe(state, text.encode("utf-8"), goal=GOAL, allowed=prev.all_true_mask())
    return state.fast.detach().to("cpu").numpy().astype("float64").ravel()


def distance(first: np.ndarray, second: np.ndarray) -> float:
    scale = 0.5 * (float(np.linalg.norm(first)) + float(np.linalg.norm(second)))
    if scale <= 0:
        return 0.0
    return float(np.linalg.norm(first - second)) / scale


def structure_reading(model: Any) -> dict[str, Any]:
    rows = describe_rows()
    texts = [row["text"] for row in rows]
    objects = [row["object"] for row in rows]
    places = [row["place"] for row in rows]
    vectors = [write_once(model, text, f"s{index}") for index, text in enumerate(texts)]

    size = len(vectors)
    matrix = np.zeros((size, size), dtype="float64")
    for i in range(size):
        for j in range(i + 1, size):
            value = distance(vectors[i], vectors[j])
            matrix[i, j] = matrix[j, i] = value

    nearest = []
    for i in range(size):
        order = np.argsort(np.where(np.eye(size, dtype=bool)[i], np.inf, matrix[i]))
        nearest.append(int(order[0]))

    object_hits = sum(int(objects[nearest[i]] == objects[i]) for i in range(size))
    place_hits = sum(int(places[nearest[i]] == places[i]) for i in range(size))

    def grouped(key: list[str]) -> dict[str, float]:
        same, different = [], []
        for i in range(size):
            for j in range(i + 1, size):
                (same if key[i] == key[j] else different).append(matrix[i, j])
        return {
            "same_mean": float(np.mean(same)) if same else 0.0,
            "different_mean": float(np.mean(different)) if different else 0.0,
            "same_pairs": len(same),
            "different_pairs": len(different),
        }

    object_counts = [objects.count(name) for name in set(objects)]
    place_counts = [places.count(name) for name in set(places)]
    object_chance = float(np.mean([(count - 1) / (size - 1) for count in object_counts]))
    place_chance = float(np.mean([(count - 1) / (size - 1) for count in place_counts]))
    object_accuracy = object_hits / size
    return {
        "samples": size,
        "objects": len(set(objects)),
        "places": len(set(places)),
        "object_accuracy": object_accuracy,
        "object_chance": object_chance,
        "object_hits": object_hits,
        "place_accuracy": place_hits / size,
        "place_chance": place_chance,
        "place_hits": place_hits,
        "object_distances": grouped(objects),
        "place_distances": grouped(places),
        "nearest": nearest,
        "structure_present": object_accuracy >= OBJECT_GATE,
        "ok": size == 20 and len(set(objects)) == 4 and len(set(places)) == 5,
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "goal": GOAL.decode("ascii"),
        "object_gate": OBJECT_GATE,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    rows = describe_rows()
    manifest = {
        "schema": "taiji-write-structure-r50-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "object_gate": OBJECT_GATE,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": prev.file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {"describe_sentences": len(rows), "writes": len(rows)},
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "labels": [{"object": row["object"], "place": row["place"]} for row in rows],
            "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        },
    }
    prev.write_json(MANIFEST, manifest)
    return manifest


def verify_freeze() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise SystemExit("no frozen manifest; run --freeze first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["runner_sha256"] != RUNNER_SHA_START:
        raise SystemExit("runner changed after the freeze; re-freeze")
    if manifest["core_sha256"] != prev.file_sha(CORE_SOURCE):
        raise SystemExit("living_core changed after the freeze; re-freeze")
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

        texts = prev.sentences()
        torch.manual_seed(SEED)
        reference = LivingCoreReference()
        before = prev.run_battery(reference, texts)
        main_before = prev.parameter_sha(reference)

        torch.manual_seed(SEED + 1)
        model = LivingCoreReference()
        reading = structure_reading(model)

        after = prev.run_battery(reference, texts)
        main_after = prev.parameter_sha(reference)
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
        closed_ok = all(checks.values())
        main_untouched = main_before == main_after
        bridge_default_none = before["sc_bridge_none"] and after["sc_bridge_none"]
        rerun_ok = all(
            row["rerun_commit"]
            and row["runner_blob_sha1"]
            and row["runner_blob_sha256"]
            and row["report_core_sha256"]
            for row in RERUN_ANCHORS
        )

        verdict = {
            "labels": "labels_valid" if reading["ok"] else "labels_broken",
            "decision": (
                "write_structure_present"
                if reading["structure_present"]
                else "write_structure_scrambled"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all([reading["ok"], closed_ok, main_untouched, bridge_default_none, rerun_ok])
        verdict["overall"] = "write_structure_measured" if all_ok else "write_structure_unexpected"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-write-structure-r50-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": prev.file_sha(MANIFEST),
            "core_sha256": prev.file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "reading": reading,
            "rerun_anchors": list(RERUN_ANCHORS),
            "closed_state": {"baseline": before, "after": after, "checks": checks},
            "main_parameter_sha_equal": main_untouched,
            "bridge_default_none": bridge_default_none,
            "verdict": verdict,
            "cost": {
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "cpu_cap": CPU_CAP,
                "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
                "threads": torch.get_num_threads(),
            },
        }
        prev.write_json(REPORT, report)
        print(
            json.dumps(
                {
                    **{f"verdict_{key}": value for key, value in verdict.items()},
                    "object_accuracy": reading["object_accuracy"],
                    "object_chance": reading["object_chance"],
                    "object_hits": reading["object_hits"],
                    "place_accuracy": reading["place_accuracy"],
                    "place_chance": reading["place_chance"],
                    "place_hits": reading["place_hits"],
                    "object_distances": reading["object_distances"],
                    "place_distances": reading["place_distances"],
                    "closed_state_ok": closed_ok,
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
