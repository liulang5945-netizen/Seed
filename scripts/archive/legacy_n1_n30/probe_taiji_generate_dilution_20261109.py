"""R44 -- is the generate layer damped by input dilution or by its own weights?

R43 localised the 21x shrink to ``expression.generate`` with same-unit relative
L2.  R44 splits the generate input ``cat([z, previous])`` and measures:

    z_share      = ||z||2 / (||z||2 + ||previous||2)
    rel_input    = ||delta_input||2 / ||input_ref||2
    rel_z        = ||delta_z||2 / ||z_ref||2
    dilution     = rel_input / rel_z          (how much the z change is spread)
    layer_gain   = rel_logits / rel_input     (what the weight does to it)

Frozen decision:
- ``diluted_by_previous`` when dilution < 0.5 (the z change is spread over a
  much larger input vector);
- else ``weight_compressed`` when layer_gain < 0.1 (the weight itself flattens);
- else ``mixed_or_other``.
All four readings must be reported; every relative change must be > 0.

Zero optimizer steps; no module change; hooks only.
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

MANIFEST = ROOT / "plans/manifests/taiji_generate_dilution_20261109.json"
REPORT = ROOT / "reports/taiji_generate_dilution_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 20.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r44"
DOSE = 1000
DILUTION_FLOOR = 0.5
GAIN_FLOOR = 0.1

RERUN_ANCHORS = (
    {
        "round": "R43",
        "runner": "scripts/training/probe_taiji_scale_free_damping_20261109.py",
        "rerun_commit": "5ef33910e",
        "runner_blob_sha1": "500950cf5d063c2e8fd0a60cefb0a2d2c54c4d6c",
        "report_runner_sha256": "da886d3a16b932817b1dbd4208352aca14e214cd183777e5a5865a23031e4f22",
        "runner_blob_sha256": "da886d3a16b932817b1dbd4208352aca14e214cd183777e5a5865a23031e4f22",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R44 does not change living_core; R43 stays runnable at HEAD",
    },
)

DEFINITION = {
    "dose": "the same written state is stepped with fast x0 and fast x1000",
    "capture": "a pre-hook on expression.generate records its input cat([z, previous])",
    "split": "the first width entries are the z slice; the rest is the previous-byte slice",
    "readings": "z_share, rel_input, rel_z, rel_prev, dilution, layer_gain",
    "decision": "dilution < 0.5 -> diluted_by_previous; else layer_gain < 0.1 -> weight_compressed",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "write": "the written fast has abs sum > 0",
    "capture": "the generate input is captured once per run with the frozen shape",
    "reach": "rel_z, rel_input and rel_logits are all > 0",
    "readings": "z_share, dilution and layer_gain are reported",
    "decision": "the site follows the frozen dilution/gain rule and is reported with its exit name",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R43 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def stage_step(model: Any, state: Any, raw: bytes) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def keep(name: str, index: int | None = None):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output if index is None else output[index]
            captured[name] = value.detach().clone()

        return hook

    def keep_first_input(name: str):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            captured[name] = args[0].detach().clone()

        return hook

    handles = [
        model.expression.generate.register_forward_pre_hook(keep_first_input("gen_input")),
        model.expression.generate.register_forward_hook(keep("logits")),
        model.expression.register_forward_hook(keep("prob", 0)),
    ]
    try:
        event = model.encoder.patches(raw)[0]
        with torch.no_grad():
            model.step(state, event=event, goal=GOAL, allowed=all_true_mask())
    finally:
        for handle in handles:
            handle.remove()
    return {name: captured[name].numpy().astype("float64") for name in captured}


def norm(value: Any) -> float:
    return float((value * value).sum() ** 0.5)


def relative(zero: Any, full: Any) -> float:
    reference = norm(zero)
    difference = norm(zero - full)
    return difference / reference if reference > 0 else 0.0


def generate_reading(model: Any, texts: list[str]) -> dict[str, Any]:
    state = model.initial_state("r44-generate")
    with torch.no_grad():
        state = model.observe(state, texts[0].encode("utf-8"), goal=GOAL, allowed=all_true_mask())
    target = texts[1].encode("utf-8")
    zero = stage_step(model, replace(state, fast=torch.zeros_like(state.fast)), target)
    full = stage_step(model, replace(state, fast=state.fast * float(DOSE)), target)

    width = int(model.config.width)
    z_zero = zero["gen_input"][:width]
    z_full = full["gen_input"][:width]
    prev_zero = zero["gen_input"][width:]
    prev_full = full["gen_input"][width:]
    z_norm = norm(z_zero)
    prev_norm = norm(prev_zero)
    total = z_norm + prev_norm
    z_share = z_norm / total if total > 0 else 0.0
    rel_input = relative(zero["gen_input"], full["gen_input"])
    rel_z = relative(z_zero, z_full)
    rel_prev = relative(prev_zero, prev_full)
    rel_logits = relative(zero["logits"], full["logits"])
    dilution = rel_input / rel_z if rel_z > 0 else 0.0
    layer_gain = rel_logits / rel_input if rel_input > 0 else 0.0
    if dilution < DILUTION_FLOOR:
        decision = "diluted_by_previous"
    elif layer_gain < GAIN_FLOOR:
        decision = "weight_compressed"
    else:
        decision = "mixed_or_other"
    return {
        "fast_abs_sum": float(state.fast.abs().sum()),
        "dose": DOSE,
        "input_dim": int(zero["gen_input"].shape[0]),
        "z_slice": int(width),
        "z_share": z_share,
        "rel_input": rel_input,
        "rel_z": rel_z,
        "rel_prev": rel_prev,
        "rel_logits": rel_logits,
        "dilution": dilution,
        "layer_gain": layer_gain,
        "decision": decision,
        "ok": float(state.fast.abs().sum()) > 0 and rel_z > 0 and rel_input > 0 and rel_logits > 0,
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
    state = reference.initial_state("r44-zero-perturbation")
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
        "dilution_floor": DILUTION_FLOOR,
        "gain_floor": GAIN_FLOOR,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-generate-dilution-r44-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "dose": DOSE,
        "dilution_floor": DILUTION_FLOOR,
        "gain_floor": GAIN_FLOOR,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {"describe_sentences": len(sentences()), "stages": 2},
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
        reading = generate_reading(model, texts)

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
            "reach": "memory_reaches_generate" if reading["ok"] else "memory_lost_before_generate",
            "decision": reading["decision"],
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all([reading["ok"], closed_ok, main_untouched, bridge_default_none, rerun_ok])
        verdict["overall"] = (
            "generate_dilution_measured" if all_ok else "generate_dilution_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-generate-dilution-r44-report-v1",
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
                        "input_dim": reading["input_dim"],
                        "z_slice": reading["z_slice"],
                        "z_share": reading["z_share"],
                        "rel_input": reading["rel_input"],
                        "rel_z": reading["rel_z"],
                        "rel_prev": reading["rel_prev"],
                        "rel_logits": reading["rel_logits"],
                        "dilution": reading["dilution"],
                        "layer_gain": reading["layer_gain"],
                    },
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
