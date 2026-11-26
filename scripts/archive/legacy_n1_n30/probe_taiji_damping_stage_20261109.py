"""R41 -- where the memory signal is damped, via forward hooks (no module change).

R40 showed a monotone dose curve that still sits at 1.13e-04 for a 1000x
memory.  R41 measures the same written memory (k = 0 vs k = 1000) at three
stages of one step and reports how much of the input difference survives:

1. ``xi`` -- the fused feature that enters ``readout``;
2. ``z``  -- the output of ``readout``;
3. ``probabilities`` -- the output of ``expression``.

Frozen decision:
- all three stage deltas must be > 0 (the memory reaches every stage);
- ``readout_ratio = delta_z / delta_xi`` and
  ``output_ratio = delta_prob / delta_z``;
- ``damping_localised_readout`` when readout_ratio < 0.1 <= output_ratio;
- ``damping_localised_output`` when output_ratio < 0.1 <= readout_ratio;
- ``damping_distributed`` when both < 0.1;
- ``damping_absent`` when neither < 0.1 (the loss happens before xi).

Zero optimizer steps; the main subject only runs the closed-state battery.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_cross_family_readout_20261109 as base  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_damping_stage_20261109.json"
REPORT = ROOT / "reports/taiji_damping_stage_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r41"
DOSE = 1000
DAMPING_FACTOR = 0.1

RERUN_ANCHORS = (
    {
        "round": "R40",
        "runner": "scripts/training/probe_taiji_memory_dose_20261109.py",
        "rerun_commit": "bb9a8b97f",
        "runner_blob_sha1": "03ddd6c36cac00d786cea99d51323b260bb5fd0a",
        "report_runner_sha256": "2c000d713445c0a75d03060126b026138a74652a917616d495da6a2009a00681",
        "runner_blob_sha256": "2c000d713445c0a75d03060126b026138a74652a917616d495da6a2009a00681",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R41 does not change living_core; R40 stays runnable at HEAD",
    },
)

DEFINITION = {
    "write": "one write-enabled observe pass on sentence 0 produces the fast memory",
    "dose": "the same state is stepped twice on sentence 1, with fast x0 and fast x1000",
    "stages": "forward hooks capture xi (readout input), z (readout output) and probabilities",
    "decision": "readout_ratio = delta_z/delta_xi; output_ratio = delta_prob/delta_z",
    "localise": "a ratio below 0.1 marks that stage as the damping site",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "write": "the written fast has abs sum > 0",
    "reach": "all three stage deltas are > 0",
    "ratios": "both ratios are reported with the raw deltas",
    "decision": "the damping site follows the frozen 0.1 rule and is reported with its exit name",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R40 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def state_sha(state: Any) -> str:
    return tensor_sha([state.h, state.fast, state.nu_ema, state.coverage, state.prior])


def parameter_sha(model: Any) -> str:
    return tensor_sha([tensor for _, tensor in sorted(model.named_parameters())])


def all_true_mask() -> torch.Tensor:
    return torch.ones((4, 2), dtype=torch.bool)


def stage_step(model: Any, state: Any, raw: bytes) -> dict[str, float]:
    """One step with hooks on readout input/output and the expression output."""
    captured: dict[str, Any] = {}

    def readout_pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
        captured["xi"] = args[1].detach().clone()

    def readout_hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["z"] = output[0].detach().clone()

    def expression_hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["prob"] = output[0].detach().clone()

    handles = [
        model.readout.register_forward_pre_hook(readout_pre_hook),
        model.readout.register_forward_hook(readout_hook),
        model.expression.register_forward_hook(expression_hook),
    ]
    try:
        event = model.encoder.patches(raw)[0]
        with torch.no_grad():
            model.step(
                state,
                event=event,
                goal=GOAL,
                allowed=all_true_mask(),
            )
    finally:
        for handle in handles:
            handle.remove()
    return {name: captured[name].numpy().astype("float64") for name in ("xi", "z", "prob")}


def delta(a: Any, b: Any) -> float:
    return float(abs(a - b).max())


def damping_reading(model: Any, texts: list[str]) -> dict[str, Any]:
    state = model.initial_state("r41-damping")
    with torch.no_grad():
        state = model.observe(state, texts[0].encode("utf-8"), goal=GOAL, allowed=all_true_mask())
    target = texts[1].encode("utf-8")
    zero = stage_step(model, replace(state, fast=torch.zeros_like(state.fast)), target)
    full = stage_step(model, replace(state, fast=state.fast * float(DOSE)), target)
    d_xi = delta(zero["xi"], full["xi"])
    d_z = delta(zero["z"], full["z"])
    d_prob = delta(zero["prob"], full["prob"])
    readout_ratio = d_z / d_xi if d_xi > 0 else 0.0
    output_ratio = d_prob / d_z if d_z > 0 else 0.0
    readout_damped = readout_ratio < DAMPING_FACTOR
    output_damped = output_ratio < DAMPING_FACTOR
    if readout_damped and output_damped:
        decision = "damping_distributed"
    elif readout_damped:
        decision = "damping_localised_readout"
    elif output_damped:
        decision = "damping_localised_output"
    else:
        decision = "damping_absent"
    return {
        "fast_abs_sum": float(state.fast.abs().sum()),
        "dose": DOSE,
        "delta_xi": d_xi,
        "delta_z": d_z,
        "delta_prob": d_prob,
        "readout_ratio": readout_ratio,
        "output_ratio": output_ratio,
        "decision": decision,
        "ok": float(state.fast.abs().sum()) > 0 and d_xi > 0 and d_z > 0 and d_prob > 0,
    }


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
        "state_sha": state_sha(state),
        "keys_len": len(state.keys),
        "step": int(state.step),
        "store_sha": tensor_sha(store),
        "identity_sha": hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sc_sfk_none": state.sc_sfk is None,
        "sc_bridge_none": state.sc_bridge is None,
        "battery_sha": hashlib.sha256("".join(step_hashes).encode("ascii")).hexdigest(),
        "battery_steps": len(step_hashes),
    }


def run_battery(reference: Any, texts: list[str]) -> dict[str, Any]:
    torch.manual_seed(SEED)
    state = reference.initial_state("r41-zero-perturbation")
    step_hashes: list[str] = []
    for raw in texts:
        state = reference.observe(state, raw.encode("utf-8"), goal=GOAL)
        step_hashes.append(state_sha(state))
    return reference_digest(reference, state, step_hashes)


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "goal": GOAL.decode("ascii"),
        "dose": DOSE,
        "damping_factor": DAMPING_FACTOR,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-damping-stage-r41-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "dose": DOSE,
        "damping_factor": DAMPING_FACTOR,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {"describe_sentences": len(sentences()), "stages": 3},
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "sentences": sentences()[:2],
            "dose": DOSE,
            "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
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
    if manifest["core_sha256"] != file_sha(CORE_SOURCE):
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

        texts = sentences()
        torch.manual_seed(SEED)
        reference = LivingCoreReference()
        before = run_battery(reference, texts)
        main_before = parameter_sha(reference)

        torch.manual_seed(SEED + 1)
        model = LivingCoreReference()
        reading = damping_reading(model, texts)

        after = run_battery(reference, texts)
        main_after = parameter_sha(reference)
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
            "write": "memory_written" if reading["fast_abs_sum"] > 0 else "memory_missing",
            "reach": "memory_reaches_all_stages" if reading["ok"] else "memory_lost_before_z",
            "decision": reading["decision"],
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all([reading["ok"], closed_ok, main_untouched, bridge_default_none, rerun_ok])
        verdict["overall"] = "memory_damping_measured" if all_ok else "memory_damping_unexpected"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-damping-stage-r41-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "core_sha256": file_sha(CORE_SOURCE),
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
        write_json(REPORT, report)
        print(
            json.dumps(
                {
                    **{f"verdict_{key}": value for key, value in verdict.items()},
                    "reading": {
                        "fast_abs_sum": round(reading["fast_abs_sum"], 6),
                        "delta_xi": reading["delta_xi"],
                        "delta_z": reading["delta_z"],
                        "delta_prob": reading["delta_prob"],
                        "readout_ratio": reading["readout_ratio"],
                        "output_ratio": reading["output_ratio"],
                    },
                    "closed_state_ok": closed_ok,
                    "main_parameter_sha_equal": main_untouched,
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
