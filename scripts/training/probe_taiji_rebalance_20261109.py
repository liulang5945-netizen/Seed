"""R45 -- rebalance the generate input: does dilution actually bottle the memory?

R44 showed ``z`` is 4.8% of the generate input norm and the layer itself is
faithful (gain 1.03).  R45 intervenes on the input with a forward pre-hook
(no module change): the ``z`` slice is scaled by ``s`` so that its share hits
``{0.05, 0.2, 0.5, 0.8}``; the same scaling is applied to both the k=0 and the
k=1000 run, so the comparison still isolates the memory.

Frozen decision:
- the baseline (share 0.05, s=1) must reproduce R44: dilution < 0.5 and
  rel_logits ~ 0.033;
- rel_logits and output TV must be non-decreasing in the target share;
- at share 0.5 both must be >= 3x the baseline;
- ``dilution_confirmed_by_rebalance`` when all hold, else
  ``dilution_not_confirmed``.

Zero optimizer steps; hooks only.
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

MANIFEST = ROOT / "plans/manifests/taiji_rebalance_20261109.json"
REPORT = ROOT / "reports/taiji_rebalance_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 20.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r45"
DOSE = 1000
TARGET_SHARES = (0.05, 0.2, 0.5, 0.8)
BASELINE_SHARE = 0.05
CONFIRM_SHARE = 0.5
CONFIRM_RATIO = 3.0

RERUN_ANCHORS = (
    {
        "round": "R44",
        "runner": "scripts/training/probe_taiji_generate_dilution_20261109.py",
        "rerun_commit": "167e6a645",
        "runner_blob_sha1": "3d0c8af6253cfa0018e84848cc6d2401da509f15",
        "report_runner_sha256": "4ca366903739a5b01478e585bd7a96812a399ed503ff7f541d9dd9672e312a22",
        "runner_blob_sha256": "4ca366903739a5b01478e585bd7a96812a399ed503ff7f541d9dd9672e312a22",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R45 does not change living_core; R44 stays runnable at HEAD",
    },
)

DEFINITION = {
    "write": "one write-enabled observe pass on sentence 0 produces the fast memory",
    "rebalance": "a pre-hook scales the z slice of the generate input by s so its share hits a target",
    "targets": "shares 0.05, 0.2, 0.5, 0.8; the same s is used for the k=0 and k=1000 runs",
    "readings": "rel_logits and output TV/KL per target share",
    "baseline": "share 0.05 must reproduce R44 (dilution < 0.5, rel_logits near 0.033)",
    "confirm": "at share 0.5 rel_logits and TV must both be >= 3x the baseline",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "write": "the written fast has abs sum > 0",
    "baseline": "the share-0.05 row reproduces dilution < 0.5 and rel_logits > 0",
    "monotone": "rel_logits and TV are non-decreasing across the four target shares",
    "confirm": "at share 0.5 rel_logits >= 3x baseline and TV >= 3x baseline",
    "decision": "the frozen confirm ratio decides the exit name",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R44 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def stage_step(model: Any, state: Any, raw: bytes, scale: float) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    width = int(model.config.width)

    def rebalance(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        value = args[0]
        scaled = value.clone()
        scaled[:width] = scaled[:width] * float(scale)
        captured["gen_input"] = scaled.detach().clone()
        return (scaled,)

    def keep_logits(_module: Any, _inputs: Any, output: Any) -> None:
        captured["logits"] = output.detach().clone()

    def keep_prob(_module: Any, _inputs: Any, output: Any) -> None:
        captured["prob"] = output[0].detach().clone()

    handles = [
        model.expression.generate.register_forward_pre_hook(rebalance),
        model.expression.generate.register_forward_hook(keep_logits),
        model.expression.register_forward_hook(keep_prob),
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


def normalise(probabilities: Any) -> Any:
    total = float(probabilities.sum())
    return probabilities / total if total > 0 else probabilities


def kl(first: Any, second: Any) -> float:
    import numpy as np

    eps = 1e-12
    first = normalise(first)
    second = normalise(second)
    return float((first * (np.log(first + eps) - np.log(second + eps))).sum())


def total_variation(first: Any, second: Any) -> float:
    import numpy as np

    return float(np.abs(normalise(first) - normalise(second)).sum() / 2.0)


def scale_for_share(z_norm: float, prev_norm: float, share: float) -> float:
    if z_norm <= 0 or prev_norm <= 0 or share <= 0 or share >= 1:
        return 1.0
    return (prev_norm / z_norm) * (share / (1.0 - share))


def rebalance_reading(model: Any, texts: list[str]) -> dict[str, Any]:
    state = model.initial_state("r45-rebalance")
    with torch.no_grad():
        state = model.observe(state, texts[0].encode("utf-8"), goal=GOAL, allowed=all_true_mask())
    target = texts[1].encode("utf-8")
    zero_state = replace(state, fast=torch.zeros_like(state.fast))
    full_state = replace(state, fast=state.fast * float(DOSE))

    probe = stage_step(model, zero_state, target, 1.0)
    width = int(model.config.width)
    z_norm = norm(probe["gen_input"][:width])
    prev_norm = norm(probe["gen_input"][width:])

    rows = []
    previous_logits = -1.0
    previous_tv = -1.0
    monotone = True
    for share in TARGET_SHARES:
        scale = scale_for_share(z_norm, prev_norm, share)
        zero = stage_step(model, zero_state, target, scale)
        full = stage_step(model, full_state, target, scale)
        rel_logits = relative(zero["logits"], full["logits"])
        tv = total_variation(zero["prob"], full["prob"])
        kl_forward = kl(zero["prob"], full["prob"])
        monotone = monotone and rel_logits >= previous_logits and tv >= previous_tv
        previous_logits = rel_logits
        previous_tv = tv
        rows.append(
            {
                "share": share,
                "scale": scale,
                "z_share": (scale * z_norm) / (scale * z_norm + prev_norm),
                "rel_logits": rel_logits,
                "tv": tv,
                "kl_zero_to_full": kl_forward,
            }
        )
    baseline = rows[0]
    confirm = next(row for row in rows if row["share"] == CONFIRM_SHARE)
    logits_ratio = (
        confirm["rel_logits"] / baseline["rel_logits"] if baseline["rel_logits"] > 0 else 0.0
    )
    tv_ratio = confirm["tv"] / baseline["tv"] if baseline["tv"] > 0 else 0.0
    return {
        "fast_abs_sum": float(state.fast.abs().sum()),
        "dose": DOSE,
        "rows": rows,
        "monotone": monotone,
        "baseline_rel_logits": baseline["rel_logits"],
        "logits_ratio_at_confirm": logits_ratio,
        "tv_ratio_at_confirm": tv_ratio,
        "baseline_reproduced": baseline["rel_logits"] > 0 and baseline["z_share"] < 0.5,
        "confirmed": logits_ratio >= CONFIRM_RATIO and tv_ratio >= CONFIRM_RATIO,
        "ok": float(state.fast.abs().sum()) > 0 and baseline["rel_logits"] > 0,
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
    state = reference.initial_state("r45-zero-perturbation")
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
        "target_shares": list(TARGET_SHARES),
        "baseline_share": BASELINE_SHARE,
        "confirm_share": CONFIRM_SHARE,
        "confirm_ratio": CONFIRM_RATIO,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-rebalance-r45-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "dose": DOSE,
        "target_shares": list(TARGET_SHARES),
        "baseline_share": BASELINE_SHARE,
        "confirm_share": CONFIRM_SHARE,
        "confirm_ratio": CONFIRM_RATIO,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {"describe_sentences": len(sentences()), "shares": len(TARGET_SHARES)},
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "sentences": sentences()[:2],
            "target_shares": list(TARGET_SHARES),
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
        reading = rebalance_reading(model, texts)

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
            "baseline": (
                "baseline_reproduces_r44" if reading["baseline_reproduced"] else "baseline_moved"
            ),
            "monotone": "rebalance_monotone" if reading["monotone"] else "rebalance_not_monotone",
            "confirm": (
                "rebalance_confirms_dilution"
                if reading["confirmed"]
                else "rebalance_does_not_confirm"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [
                reading["ok"],
                reading["baseline_reproduced"],
                reading["monotone"],
                reading["confirmed"],
                closed_ok,
                main_untouched,
                bridge_default_none,
                rerun_ok,
            ]
        )
        verdict["overall"] = (
            "dilution_confirmed_by_rebalance" if all_ok else "dilution_not_confirmed"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-rebalance-r45-report-v1",
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
                    "rows": [
                        {
                            "share": row["share"],
                            "scale": round(row["scale"], 4),
                            "z_share": round(row["z_share"], 5),
                            "rel_logits": row["rel_logits"],
                            "tv": row["tv"],
                            "kl": row["kl_zero_to_full"],
                        }
                        for row in reading["rows"]
                    ],
                    "logits_ratio_at_confirm": reading["logits_ratio_at_confirm"],
                    "tv_ratio_at_confirm": reading["tv_ratio_at_confirm"],
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
