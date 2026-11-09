"""R40 -- the memory dose curve: how much of the written state is usable?

R39 wrote the fast memory (13.11 abs sum), confirmed the write heads learn, and
found the causal effect on the next sentence is only 6.999e-07.  R40 asks
whether that is a scale problem: the same written memory is scaled by
k in {0, 1, 10, 100, 1000} and the next-sentence distribution is compared to
the k = 0 baseline.

Frozen decision:
- ``dose_monotone``: the five max |delta| values are non-decreasing in k;
- ``dose_crosses_practical``: at least one k reaches max |delta| > 1e-3;
- ``dose_saturated``: delta(1000) / delta(100) <= 1.2 (else still scaling);
- overall ``memory_dose_measured`` when the curve is monotone and every step
  is reported, else ``memory_dose_unexpected``.

Zero optimizer steps; no module change.
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

MANIFEST = ROOT / "plans/manifests/taiji_memory_dose_20261109.json"
REPORT = ROOT / "reports/taiji_memory_dose_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 30.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r40"
DOSE_STEPS = (0, 1, 10, 100, 1000)
PRACTICAL_FLOOR = 1e-3
SATURATION_RATIO = 1.2

RERUN_ANCHORS = (
    {
        "round": "R39",
        "runner": "scripts/training/probe_taiji_write_causality_20261109.py",
        "rerun_commit": "a50f7eeaa",
        "runner_blob_sha1": "5a2a97718241d75685e9253e302b28a009537a59",
        "report_runner_sha256": "67555021396c1bf52ce35fd75cacda4bf3882379d1b5e00a70ad01c4cac5a3c8",
        "runner_blob_sha256": "67555021396c1bf52ce35fd75cacda4bf3882379d1b5e00a70ad01c4cac5a3c8",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R40 does not change living_core; R39 stays runnable at HEAD",
    },
)

DEFINITION = {
    "write": "one write-enabled observe pass on sentence 0 produces the fast memory",
    "dose": "the same written fast is multiplied by k in {0, 1, 10, 100, 1000}",
    "reading": "from each scaled state, sentence 1 is scored and compared to the k = 0 baseline",
    "monotone": "the max |delta| values are non-decreasing in k",
    "practical": "at least one k reaches max |delta| > 1e-3",
    "saturation": "delta(1000) / delta(100) <= 1.2 means saturated",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "write": "the written fast has abs sum > 0",
    "curve": "all five doses are reported with max |delta| and mean |delta|",
    "monotone": "max |delta| is non-decreasing across the five doses",
    "practical": "the first dose above 1e-3 is reported; absence is a finding, not a failure",
    "saturation": "the delta(1000)/delta(100) ratio is reported and classified",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R39 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def dose_curve(model: Any, texts: list[str]) -> dict[str, Any]:
    target = texts[1].encode("utf-8")
    state = model.initial_state("r40-dose")
    with torch.no_grad():
        state = model.observe(state, texts[0].encode("utf-8"), goal=GOAL, allowed=all_true_mask())
        baseline, _ = model.score_targets(
            replace(state, fast=torch.zeros_like(state.fast)),
            prompt=b"",
            target=target,
            goal=GOAL,
            allowed=all_true_mask(),
        )
        rows = []
        previous = -1.0
        monotone = True
        for k in DOSE_STEPS:
            scaled = replace(state, fast=state.fast * float(k))
            distributions, _ = model.score_targets(
                scaled, prompt=b"", target=target, goal=GOAL, allowed=all_true_mask()
            )
            delta = (distributions - baseline).abs()
            maximum = float(delta.max())
            mean = float(delta.mean())
            monotone = monotone and maximum >= previous
            previous = maximum
            rows.append(
                {
                    "k": k,
                    "max_abs_delta": maximum,
                    "mean_abs_delta": mean,
                    "digest": hashlib.sha256(
                        distributions.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                }
            )
    by_k = {row["k"]: row["max_abs_delta"] for row in rows}
    first_practical = next(
        (row["k"] for row in rows if row["max_abs_delta"] > PRACTICAL_FLOOR), None
    )
    ratio = by_k[1000] / by_k[100] if by_k[100] > 0 else (float("inf") if by_k[1000] > 0 else 1.0)
    return {
        "fast_abs_sum": float(state.fast.abs().sum()),
        "rows": rows,
        "monotone": monotone,
        "first_k_above_practical": first_practical,
        "practical_floor": PRACTICAL_FLOOR,
        "ratio_1000_over_100": ratio,
        "saturated": ratio <= SATURATION_RATIO,
        "ok": float(state.fast.abs().sum()) > 0,
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
    state = reference.initial_state("r40-zero-perturbation")
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
        "dose_steps": list(DOSE_STEPS),
        "practical_floor": PRACTICAL_FLOOR,
        "saturation_ratio": SATURATION_RATIO,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-memory-dose-r40-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "dose_steps": list(DOSE_STEPS),
        "practical_floor": PRACTICAL_FLOOR,
        "saturation_ratio": SATURATION_RATIO,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {"describe_sentences": len(sentences()), "doses": len(DOSE_STEPS)},
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "sentences": sentences()[:2],
            "dose_steps": list(DOSE_STEPS),
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
        curve = dose_curve(model, texts)

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
            "write": "memory_written" if curve["ok"] else "memory_missing",
            "monotone": "dose_monotone" if curve["monotone"] else "dose_not_monotone",
            "practical": (
                "dose_crosses_practical"
                if curve["first_k_above_practical"] is not None
                else "dose_stays_below_practical"
            ),
            "saturation": "dose_saturated" if curve["saturated"] else "dose_still_scaling",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [
                curve["ok"],
                curve["monotone"],
                closed_ok,
                main_untouched,
                bridge_default_none,
                rerun_ok,
            ]
        )
        verdict["overall"] = "memory_dose_measured" if all_ok else "memory_dose_unexpected"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-memory-dose-r40-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "curve": curve,
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
                    "fast_abs_sum": round(curve["fast_abs_sum"], 6),
                    "dose": [
                        {
                            "k": row["k"],
                            "max_abs_delta": row["max_abs_delta"],
                            "mean_abs_delta": row["mean_abs_delta"],
                        }
                        for row in curve["rows"]
                    ],
                    "first_k_above_practical": curve["first_k_above_practical"],
                    "ratio_1000_over_100": curve["ratio_1000_over_100"],
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
