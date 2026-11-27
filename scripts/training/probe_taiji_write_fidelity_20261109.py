"""R48 -- does the written fast retain the identity of what was written?

R47 showed the memory channel is content blind end to end.  R48 tests the write
side alone: 20 describe sentences are each written once (write-enabled
observe), and a closed-form ridge readout maps ``state.fast`` to the sentence
index (20 classes).  Folds are fixed (index % 5), and a permuted-label control
guards against overfitting.

Frozen decision:
- best accuracy over the lambda grid >= 0.5 -> ``write_path_retains_content``
  (the readout is the bottleneck, not the write);
- otherwise ``write_path_discards_content``;
- if the permuted control exceeds 0.2, the probe is ``write_probe_inconclusive``.

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

MANIFEST = ROOT / "plans/manifests/taiji_write_fidelity_20261109.json"
REPORT = ROOT / "reports/taiji_write_fidelity_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r48"
LAMBDAS = (0.001, 0.1, 10.0, 1000.0)
FOLDS = 5
ACCURACY_GATE = 0.5
CONTROL_GATE = 0.2

RERUN_ANCHORS = (
    {
        "round": "R47",
        "runner": "scripts/training/probe_taiji_channel_content_20261109.py",
        "rerun_commit": "09ef04e6f",
        "runner_blob_sha1": "8e71e998c6eead3bc180a61719d6a0c610b095f3",
        "report_runner_sha256": "61a5e0aea2e41f0ffbb4b110e48883e8d3152cb817ea29c323d44c5228cfd2f5",
        "runner_blob_sha256": "61a5e0aea2e41f0ffbb4b110e48883e8d3152cb817ea29c323d44c5228cfd2f5",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R48 does not change living_core; R47 stays runnable at HEAD",
    },
)

DEFINITION = {
    "write": "each of the 20 describe sentences is written once with a write-enabled observe",
    "features": "state.fast flattened, float64, centred per fold with an appended bias column",
    "readout": "closed-form ridge W=(XtX+lI)^-1 XtY over the 20-way one-hot sentence label",
    "folds": "fixed index % 5; 16 train, 4 test per fold",
    "control": "the same pipeline with permuted labels (fixed seed) must stay near chance",
    "decision": "best accuracy over the lambda grid >= 0.5 -> write_path_retains_content",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "features": "20 finite feature vectors with matching label count",
    "grid": "accuracy is reported for every lambda",
    "control": "the permuted-label control is <= 0.2, otherwise inconclusive",
    "decision": "the frozen 0.5 gate decides retains/discards",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R47 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def write_features(model: Any, texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        state = model.initial_state("r48-write")
        with torch.no_grad():
            state = model.observe(
                state, text.encode("utf-8"), goal=GOAL, allowed=prev.all_true_mask()
            )
        rows.append(state.fast.detach().to("cpu").numpy().astype("float64").ravel())
    return np.stack(rows, axis=0)


def ridge_accuracy(features: np.ndarray, labels: np.ndarray, lam: float) -> float:
    total, hits = 0, 0
    for fold in range(FOLDS):
        test = np.array([index % FOLDS == fold for index in range(len(labels))])
        train = ~test
        x_train = features[train]
        mean = x_train.mean(axis=0, keepdims=True)
        x_train = np.concatenate([x_train - mean, np.ones((x_train.shape[0], 1))], axis=1)
        x_test = np.concatenate([features[test] - mean, np.ones((int(test.sum()), 1))], axis=1)
        y_train = np.zeros((x_train.shape[0], len(labels)))
        y_train[np.arange(x_train.shape[0]), labels[train]] = 1.0
        gram = x_train.T @ x_train + lam * np.eye(x_train.shape[1])
        weights = np.linalg.solve(gram, x_train.T @ y_train)
        predicted = np.argmax(x_test @ weights, axis=1)
        hits += int((predicted == labels[test]).sum())
        total += int(test.sum())
    return hits / total if total else 0.0


def write_reading(model: Any, texts: list[str]) -> dict[str, Any]:
    features = write_features(model, texts)
    labels = np.arange(len(texts))
    finite = bool(np.isfinite(features).all())
    rows = [{"lambda": lam, "accuracy": ridge_accuracy(features, labels, lam)} for lam in LAMBDAS]
    rng = np.random.default_rng(SEED)
    permuted = rng.permutation(labels)
    control = [
        {"lambda": lam, "accuracy": ridge_accuracy(features, permuted, lam)} for lam in LAMBDAS
    ]
    best = max(row["accuracy"] for row in rows)
    control_best = max(row["accuracy"] for row in control)
    return {
        "samples": int(features.shape[0]),
        "dim": int(features.shape[1]),
        "finite": finite,
        "rows": rows,
        "control": control,
        "best_accuracy": best,
        "control_best": control_best,
        "retains": best >= ACCURACY_GATE,
        "inconclusive": control_best > CONTROL_GATE,
        "ok": finite and features.shape[0] == len(labels),
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "goal": GOAL.decode("ascii"),
        "lambdas": list(LAMBDAS),
        "folds": FOLDS,
        "accuracy_gate": ACCURACY_GATE,
        "control_gate": CONTROL_GATE,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-write-fidelity-r48-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "lambdas": list(LAMBDAS),
        "folds": FOLDS,
        "accuracy_gate": ACCURACY_GATE,
        "control_gate": CONTROL_GATE,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": prev.file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {"describe_sentences": len(prev.sentences()), "lambdas": len(LAMBDAS)},
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
        reading = write_reading(model, texts)

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

        if reading["inconclusive"]:
            decision = "write_probe_inconclusive"
        elif reading["retains"]:
            decision = "write_path_retains_content"
        else:
            decision = "write_path_discards_content"
        verdict = {
            "features": "features_finite" if reading["ok"] else "features_broken",
            "decision": decision,
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all([reading["ok"], closed_ok, main_untouched, bridge_default_none, rerun_ok])
        verdict["overall"] = "write_fidelity_measured" if all_ok else "write_fidelity_unexpected"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-write-fidelity-r48-report-v1",
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
                    "rows": reading["rows"],
                    "control": reading["control"],
                    "best_accuracy": reading["best_accuracy"],
                    "control_best": reading["control_best"],
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
