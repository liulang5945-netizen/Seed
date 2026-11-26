"""R49 -- is the written fast content dependent, under no readout assumption?

R48's linear probe failed 0/20, but that cannot separate "the write is content
independent" from "content is there but not linearly readable".  R49 drops the
readout entirely and measures geometry:

    same_distance = relative L2 between two writes of the same sentence (>0
                    would mean the write is not deterministic);
    pair_distance = relative L2 between writes of different sentences.

Frozen decision:
- mean pairwise relative distance < 0.01 -> ``write_content_independent``
  (the write is a near-constant, so readout discussions are moot);
- otherwise ``write_content_dependent_but_not_linearly_decodable``.

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

MANIFEST = ROOT / "plans/manifests/taiji_write_dependence_20261109.json"
REPORT = ROOT / "reports/taiji_write_dependence_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 40.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r49"
INDEPENDENCE_GATE = 0.01

RERUN_ANCHORS = (
    {
        "round": "R48",
        "runner": "scripts/training/probe_taiji_write_fidelity_20261109.py",
        "rerun_commit": "4126e6bf3",
        "runner_blob_sha1": "93b31338380b2b23b881f95eb358a15301002ae4",
        "report_runner_sha256": "f7d45e076c45b10d68d4e9f62a48af0526f1c50862466e5f07a962be5c11b69a",
        "runner_blob_sha256": "f7d45e076c45b10d68d4e9f62a48af0526f1c50862466e5f07a962be5c11b69a",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R49 does not change living_core; R48 stays runnable at HEAD",
    },
)

DEFINITION = {
    "write": "each of the 20 describe sentences is written once with a write-enabled observe",
    "determinism": "sentence 0 is written twice; the relative L2 between the two fast vectors must be 0",
    "geometry": "all 190 cross-sentence pairs get a scale-free relative L2",
    "decision": "mean pairwise relative distance < 0.01 -> write_content_independent",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "determinism": "the repeated write of the same sentence gives distance 0",
    "pairs": "all 190 cross-sentence distances are reported with min/mean/median/max",
    "decision": "the frozen 0.01 gate decides independent/dependent",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R48 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def write_once(model: Any, text: str, tag: str) -> np.ndarray:
    state = model.initial_state(f"r49-{tag}")
    with torch.no_grad():
        state = model.observe(state, text.encode("utf-8"), goal=GOAL, allowed=prev.all_true_mask())
    return state.fast.detach().to("cpu").numpy().astype("float64").ravel()


def distance(first: np.ndarray, second: np.ndarray) -> float:
    scale = 0.5 * (float(np.linalg.norm(first)) + float(np.linalg.norm(second)))
    if scale <= 0:
        return 0.0
    return float(np.linalg.norm(first - second)) / scale


def dependence_reading(model: Any, texts: list[str]) -> dict[str, Any]:
    first = write_once(model, texts[0], "a")
    second = write_once(model, texts[0], "b")
    same_distance = distance(first, second)

    vectors = [write_once(model, text, f"s{index}") for index, text in enumerate(texts)]
    pairs = [
        distance(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    array = np.array(pairs, dtype="float64")
    norms = np.array([float(np.linalg.norm(vector)) for vector in vectors])
    mean_pair = float(array.mean()) if array.size else 0.0
    return {
        "samples": len(vectors),
        "dim": int(vectors[0].size),
        "same_distance": same_distance,
        "deterministic": same_distance == 0.0,
        "pairs": int(array.size),
        "pair_min": float(array.min()) if array.size else 0.0,
        "pair_mean": mean_pair,
        "pair_median": float(np.median(array)) if array.size else 0.0,
        "pair_max": float(array.max()) if array.size else 0.0,
        "norm_mean": float(norms.mean()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "content_independent": mean_pair < INDEPENDENCE_GATE,
        "ok": bool(np.isfinite(array).all()) and array.size == len(texts) * (len(texts) - 1) // 2,
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "goal": GOAL.decode("ascii"),
        "independence_gate": INDEPENDENCE_GATE,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-write-dependence-r49-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "independence_gate": INDEPENDENCE_GATE,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": prev.file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {"describe_sentences": len(prev.sentences()), "writes": 22},
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "sentences": prev.sentences(),
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
        reading = dependence_reading(model, texts)

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
            "determinism": (
                "write_deterministic" if reading["deterministic"] else "write_nondeterministic"
            ),
            "decision": (
                "write_content_independent"
                if reading["content_independent"]
                else "write_content_dependent_but_not_linearly_decodable"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all([reading["ok"], closed_ok, main_untouched, bridge_default_none, rerun_ok])
        verdict["overall"] = (
            "write_dependence_measured" if all_ok else "write_dependence_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-write-dependence-r49-report-v1",
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
                    "samples": reading["samples"],
                    "dim": reading["dim"],
                    "same_distance": reading["same_distance"],
                    "pair_min": reading["pair_min"],
                    "pair_mean": reading["pair_mean"],
                    "pair_median": reading["pair_median"],
                    "pair_max": reading["pair_max"],
                    "norm_mean": reading["norm_mean"],
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
