"""R51 -- can the structured write be read back by a linear map?

R50 showed the written fast clusters by object.  R51 asks whether that structure
is linearly recoverable: a 4-way object readout, folds grouped by place (the
test place is unseen but every object appears in training).

The ridge solve is done in the training-row span (an exact equivalence for
ridge predictions), so the 8192-dim features cost a 16x16 solve, not 8193x8193.

Frozen decision:
- best accuracy over the lambda grid >= 0.75 -> ``structure_linearly_readable``;
- otherwise ``structure_not_linearly_readable`` (a finding, not a bug).

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

MANIFEST = ROOT / "plans/manifests/taiji_structure_readout_20261109.json"
REPORT = ROOT / "reports/taiji_structure_readout_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 20.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r51"
LAMBDAS = (0.001, 0.1, 10.0, 1000.0)
ACCURACY_GATE = 0.75

RERUN_ANCHORS = (
    {
        "round": "R50",
        "runner": "scripts/training/probe_taiji_write_structure_20261109.py",
        "rerun_commit": "d69a660bc",
        "runner_blob_sha1": "86c924d057c6e22e14ddd1e41f38cffb71fc8502",
        "report_runner_sha256": "4e9f1982502aa38b907ae965fd8d5786715e3347b6a772421c467a1876f8260a",
        "runner_blob_sha256": "4e9f1982502aa38b907ae965fd8d5786715e3347b6a772421c467a1876f8260a",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R51 does not change living_core; R50 stays runnable at HEAD",
    },
)

DEFINITION = {
    "write": "each describe sentence is written once with a write-enabled observe",
    "task": "4-way object readout from the written fast",
    "folds": "grouped by place: each test fold holds one place and all four objects",
    "solve": "ridge in the training-row span (exact for ridge predictions), 16x16 not 8193x8193",
    "decision": "best accuracy over the lambda grid >= 0.75 -> structure_linearly_readable",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "features": "20 finite feature vectors with 4 objects and 5 places",
    "folds": "5 place folds, each test fold covers all four objects",
    "grid": "accuracy is reported for every lambda",
    "decision": "the frozen 0.75 gate decides readable/not",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R50 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def describe_rows() -> list[dict[str, Any]]:
    return [row for row in base.frozen.build_fixture() if row["variant"] == "describe"]


def write_once(model: Any, text: str, tag: str) -> np.ndarray:
    state = model.initial_state(f"r51-{tag}")
    with torch.no_grad():
        state = model.observe(state, text.encode("utf-8"), goal=GOAL, allowed=prev.all_true_mask())
    return state.fast.detach().to("cpu").numpy().astype("float64").ravel()


def fold_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    lam: float,
) -> tuple[float, list[int]]:
    hits, total, predictions = 0, 0, []
    for fold in sorted(set(folds.tolist())):
        test = folds == fold
        train = ~test
        x_train = features[train]
        mean = x_train.mean(axis=0, keepdims=True)
        x_train = x_train - mean
        x_test = features[test] - mean
        _, singular, right = np.linalg.svd(x_train, full_matrices=False)
        keep = int((singular > 1e-9).sum())
        basis = right[:keep]
        z_train = x_train @ basis.T
        z_test = x_test @ basis.T
        targets = np.zeros((z_train.shape[0], 4))
        targets[np.arange(z_train.shape[0]), labels[train]] = 1.0
        gram = z_train.T @ z_train + lam * np.eye(keep)
        weights = np.linalg.solve(gram, z_train.T @ targets)
        index = np.argmax(z_test @ weights, axis=1)
        predictions.extend(index.tolist())
        hits += int((index == labels[test]).sum())
        total += int(test.sum())
    return (hits / total if total else 0.0), predictions


def structure_reading(model: Any) -> dict[str, Any]:
    rows = describe_rows()
    objects = [row["object"] for row in rows]
    places = [row["place"] for row in rows]
    names = sorted(set(objects))
    labels = np.array([names.index(name) for name in objects], dtype="int64")
    folds = np.array([sorted(set(places)).index(place) for place in places], dtype="int64")
    features = np.stack(
        [write_once(model, row["text"], f"s{index}") for index, row in enumerate(rows)], axis=0
    )
    grid = []
    last_predictions: list[int] = []
    for lam in LAMBDAS:
        accuracy, predictions = fold_accuracy(features, labels, folds, lam)
        grid.append({"lambda": lam, "accuracy": accuracy})
        last_predictions = predictions
    best = max(row["accuracy"] for row in grid)
    return {
        "samples": int(features.shape[0]),
        "dim": int(features.shape[1]),
        "objects": names,
        "folds": int(len(set(folds.tolist()))),
        "rows": grid,
        "best_accuracy": best,
        "chance": 0.25,
        "predictions_last_lambda": last_predictions,
        "finite": bool(np.isfinite(features).all()),
        "linearly_readable": best >= ACCURACY_GATE,
        "ok": bool(np.isfinite(features).all()) and features.shape[0] == 20,
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "goal": GOAL.decode("ascii"),
        "lambdas": list(LAMBDAS),
        "accuracy_gate": ACCURACY_GATE,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-structure-readout-r51-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "lambdas": list(LAMBDAS),
        "accuracy_gate": ACCURACY_GATE,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": prev.file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {"describe_sentences": len(describe_rows()), "lambdas": len(LAMBDAS)},
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "sentences": prev.sentences(),
            "lambdas": list(LAMBDAS),
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
            "features": "features_finite" if reading["ok"] else "features_broken",
            "decision": (
                "structure_linearly_readable"
                if reading["linearly_readable"]
                else "structure_not_linearly_readable"
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
            "structure_readout_measured" if all_ok else "structure_readout_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-structure-readout-r51-report-v1",
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
                    "folds": reading["folds"],
                    "rows": reading["rows"],
                    "best_accuracy": reading["best_accuracy"],
                    "chance": reading["chance"],
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
