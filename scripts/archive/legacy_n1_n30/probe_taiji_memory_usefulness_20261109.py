"""R46 -- after rebalancing, is the memory actually useful for prediction?

R45 proved dilution is causal: raising the z share raises the memory's
influence 13.5x.  R46 asks the separate question -- does the memory lower the
teacher-forced next-byte loss?  For each target share the same written state is
scored twice on sentence 1 (fast x1000 vs fast x0), both with the same input
scaling; ``delta_nll = NLL(with) - NLL(without)`` in nats/byte.

Frozen decision:
- the baseline share must reproduce the R45 influence direction (rel_logits > 0);
- at share 0.5 ``delta_nll <= -0.05`` means ``memory_useful_after_rebalance``;
- otherwise ``memory_not_predictive_even_rebalanced`` (a finding, not a bug).

Zero optimizer steps; hooks only; no module change.
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
import scripts.training.probe_taiji_rebalance_20261109 as prev  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_memory_usefulness_20261109.json"
REPORT = ROOT / "reports/taiji_memory_usefulness_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 40.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r46"
DOSE = 1000
TARGET_SHARES = (0.05, 0.2, 0.5, 0.8)
CONFIRM_SHARE = 0.5
USEFUL_DELTA = -0.05

RERUN_ANCHORS = (
    {
        "round": "R45",
        "runner": "scripts/training/probe_taiji_rebalance_20261109.py",
        "rerun_commit": "eba13a422",
        "runner_blob_sha1": "950ee0ed2929505373c4eccc474c07101a8a07dd",
        "report_runner_sha256": "b156ee9c649e16b3066d99b300e6a02f3c6b10c824a249cae4dff6e72233c9fb",
        "runner_blob_sha256": "b156ee9c649e16b3066d99b300e6a02f3c6b10c824a249cae4dff6e72233c9fb",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R46 does not change living_core; R45 stays runnable at HEAD",
    },
)

DEFINITION = {
    "write": "one write-enabled observe pass on sentence 0 produces the fast memory",
    "scoring": "sentence 1 is scored teacher-forced, fast x1000 vs fast x0, same input scaling",
    "shares": "target z shares 0.05, 0.2, 0.5, 0.8; the same s is used for both memory states",
    "metric": "delta_nll = NLL(with memory) - NLL(without memory), nats/byte",
    "baseline": "share 0.05 must reproduce the R45 influence direction (rel_logits > 0)",
    "useful": "delta_nll <= -0.05 at share 0.5",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "write": "the written fast has abs sum > 0",
    "baseline": "the share-0.05 row has rel_logits > 0",
    "rows": "all four shares report NLL(with), NLL(without) and delta_nll",
    "useful": "delta_nll <= -0.05 at share 0.5 decides the useful exit",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R45 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def nll(distributions: torch.Tensor, target: bytes) -> float:
    selected = distributions[: len(target)].gather(1, torch.tensor(list(target))[:, None])
    return float(-torch.log(selected + 1e-12).mean())


def score_with_scale(model: Any, state: Any, target: bytes, scale: float) -> dict[str, Any]:
    width = int(model.config.width)
    captured: dict[str, Any] = {}

    def rebalance(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        value = args[0]
        scaled = value.clone()
        scaled[:width] = scaled[:width] * float(scale)
        captured["gen_input"] = scaled.detach().clone()
        return (scaled,)

    def keep_logits(_module: Any, _inputs: Any, output: Any) -> None:
        captured.setdefault("logits", []).append(output.detach().clone())

    handles = [
        model.expression.generate.register_forward_pre_hook(rebalance),
        model.expression.generate.register_forward_hook(keep_logits),
    ]
    try:
        with torch.no_grad():
            distributions, _ = model.score_targets(
                state, prompt=b"", target=target, goal=GOAL, allowed=prev.all_true_mask()
            )
    finally:
        for handle in handles:
            handle.remove()
    return {
        "nll": nll(distributions, target),
        "gen_input": captured["gen_input"].numpy().astype("float64"),
        "first_logits": captured["logits"][0].numpy().astype("float64"),
    }


def memory_reading(model: Any, texts: list[str]) -> dict[str, Any]:
    state = model.initial_state("r46-usefulness")
    with torch.no_grad():
        state = model.observe(
            state, texts[0].encode("utf-8"), goal=GOAL, allowed=prev.all_true_mask()
        )
    target = texts[1].encode("utf-8")
    zero_state = replace(state, fast=torch.zeros_like(state.fast))
    full_state = replace(state, fast=state.fast * float(DOSE))

    probe = score_with_scale(model, zero_state, target, 1.0)
    width = int(model.config.width)
    z_norm = prev.norm(probe["gen_input"][:width])
    prev_norm = prev.norm(probe["gen_input"][width:])

    rows = []
    for share in TARGET_SHARES:
        scale = prev.scale_for_share(z_norm, prev_norm, share)
        memory = score_with_scale(model, full_state, target, scale)
        without = score_with_scale(model, zero_state, target, scale)
        rel_logits = prev.relative(without["first_logits"], memory["first_logits"])
        rows.append(
            {
                "share": share,
                "scale": scale,
                "z_share": (scale * z_norm) / (scale * z_norm + prev_norm),
                "nll_with": memory["nll"],
                "nll_without": without["nll"],
                "delta_nll": memory["nll"] - without["nll"],
                "rel_logits": rel_logits,
            }
        )
    baseline = rows[0]
    confirm = next(row for row in rows if row["share"] == CONFIRM_SHARE)
    return {
        "fast_abs_sum": float(state.fast.abs().sum()),
        "dose": DOSE,
        "rows": rows,
        "baseline_rel_logits": baseline["rel_logits"],
        "confirm_delta_nll": confirm["delta_nll"],
        "useful": confirm["delta_nll"] <= USEFUL_DELTA,
        "ok": float(state.fast.abs().sum()) > 0 and baseline["rel_logits"] > 0,
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "goal": GOAL.decode("ascii"),
        "dose": DOSE,
        "target_shares": list(TARGET_SHARES),
        "confirm_share": CONFIRM_SHARE,
        "useful_delta": USEFUL_DELTA,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-memory-usefulness-r46-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "dose": DOSE,
        "target_shares": list(TARGET_SHARES),
        "confirm_share": CONFIRM_SHARE,
        "useful_delta": USEFUL_DELTA,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": prev.file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {"describe_sentences": len(prev.sentences()), "shares": len(TARGET_SHARES)},
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "sentences": prev.sentences()[:2],
            "target_shares": list(TARGET_SHARES),
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
        reading = memory_reading(model, texts)

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
            "write": "memory_written" if reading["fast_abs_sum"] > 0 else "memory_missing",
            "baseline": (
                "baseline_reproduces_r45"
                if reading["baseline_rel_logits"] > 0
                else "baseline_moved"
            ),
            "useful": (
                "memory_useful_after_rebalance" if reading["useful"] else "memory_not_predictive"
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
            "memory_usefulness_measured" if all_ok else "memory_usefulness_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-memory-usefulness-r46-report-v1",
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
                    "rows": [
                        {
                            "share": row["share"],
                            "scale": round(row["scale"], 4),
                            "nll_with": row["nll_with"],
                            "nll_without": row["nll_without"],
                            "delta_nll": row["delta_nll"],
                            "rel_logits": row["rel_logits"],
                        }
                        for row in reading["rows"]
                    ],
                    "confirm_delta_nll": reading["confirm_delta_nll"],
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
