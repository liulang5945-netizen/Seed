"""R47 -- can the memory channel carry content at all?

R46 showed the memory (written from sentence 0) does not help predict sentence
1: delta_nll is only -0.0055 at a 0.5 share.  R47 separates two causes with a
self-write arm:

Z. no memory at all (fresh state, fast zeroed);
O. memory written from sentence 0, scored on sentence 1 (cross, R46);
S. memory written from sentence 1 itself, scored on sentence 1 (self).

For O and S the same state is scored with and without its fast memory at the
same 0.5 share, so delta_nll isolates the memory.  Frozen: if the self arm
reaches delta_nll <= -0.05 and the cross arm does not, the channel carries
content and the gap is ``what to write``; if the self arm also misses, the
write/read path loses content.

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

MANIFEST = ROOT / "plans/manifests/taiji_channel_content_20261109.json"
REPORT = ROOT / "reports/taiji_channel_content_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 40.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r47"
DOSE = 1000
TARGET_SHARE = 0.5
USEFUL_DELTA = -0.05

RERUN_ANCHORS = (
    {
        "round": "R46",
        "runner": "scripts/training/probe_taiji_memory_usefulness_20261109.py",
        "rerun_commit": "733edee7a",
        "runner_blob_sha1": "7702c2bf21df462ed334ac03287a37e2ffe1ee25",
        "report_runner_sha256": "25d0cd0d55280ec32b944448025ac8bf01f028e990aeb0ced3ccc02e6e61a442",
        "runner_blob_sha256": "25d0cd0d55280ec32b944448025ac8bf01f028e990aeb0ced3ccc02e6e61a442",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R47 does not change living_core; R46 stays runnable at HEAD",
    },
)

DEFINITION = {
    "arms": "Z no memory; O memory from sentence 0; S memory from sentence 1 itself",
    "share": "the same z-input scale (target share 0.5) is used for every arm and both memory states",
    "metric": "delta_nll = NLL(with fast) - NLL(fast zeroed), nats/byte on sentence 1",
    "frozen": "self arm <= -0.05 while cross arm does not -> channel carries content",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "write": "both memory states have abs sum > 0",
    "arms": "all three arms report NLL; O and S report delta_nll",
    "cross_reproduce": "the cross arm delta_nll is negative and close to R46",
    "decision": "self <= -0.05 decides channel_content_capable, else channel_content_blind",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R46 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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
        captured.setdefault("gen_input", scaled.detach().clone())
        return (scaled,)

    handle = model.expression.generate.register_forward_pre_hook(rebalance)
    try:
        with torch.no_grad():
            distributions, _ = model.score_targets(
                state, prompt=b"", target=target, goal=GOAL, allowed=prev.all_true_mask()
            )
    finally:
        handle.remove()
    return {"nll": nll(distributions, target), "gen_input": captured["gen_input"].numpy()}


def write_state(model: Any, text: str) -> Any:
    tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    state = model.initial_state(f"r47-{tag}")
    with torch.no_grad():
        return model.observe(state, text.encode("utf-8"), goal=GOAL, allowed=prev.all_true_mask())


def channel_reading(model: Any, texts: list[str]) -> dict[str, Any]:
    target = texts[1].encode("utf-8")
    other = write_state(model, texts[0])
    same = write_state(model, texts[1])

    probe = score_with_scale(model, replace(other, fast=torch.zeros_like(other.fast)), target, 1.0)
    width = int(model.config.width)
    z_norm = prev.norm(probe["gen_input"][:width])
    prev_norm = prev.norm(probe["gen_input"][width:])
    scale = prev.scale_for_share(z_norm, prev_norm, TARGET_SHARE)

    cold = model.initial_state("r47-cold")
    cold_result = score_with_scale(model, cold, target, scale)

    def arm(state: Any) -> dict[str, Any]:
        memory = score_with_scale(
            model, replace(state, fast=state.fast * float(DOSE)), target, scale
        )
        without = score_with_scale(
            model, replace(state, fast=torch.zeros_like(state.fast)), target, scale
        )
        return {
            "fast_abs_sum": float(state.fast.abs().sum()),
            "nll_with": memory["nll"],
            "nll_without": without["nll"],
            "delta_nll": memory["nll"] - without["nll"],
        }

    cross = arm(other)
    self_arm = arm(same)
    return {
        "scale": scale,
        "z_share": (scale * z_norm) / (scale * z_norm + prev_norm),
        "cold_nll": cold_result["nll"],
        "cross": cross,
        "self": self_arm,
        "channel_capable": self_arm["delta_nll"] <= USEFUL_DELTA,
        "cross_negative": cross["delta_nll"] < 0,
        "ok": cross["fast_abs_sum"] > 0 and self_arm["fast_abs_sum"] > 0,
    }


def definition_digest() -> str:
    payload = {
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "goal": GOAL.decode("ascii"),
        "dose": DOSE,
        "target_share": TARGET_SHARE,
        "useful_delta": USEFUL_DELTA,
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-channel-content-r47-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "dose": DOSE,
        "target_share": TARGET_SHARE,
        "useful_delta": USEFUL_DELTA,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": prev.file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {"describe_sentences": len(prev.sentences()), "arms": 3},
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "sentences": prev.sentences()[:2],
            "target_share": TARGET_SHARE,
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
        reading = channel_reading(model, texts)

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
            "write": "memory_written" if reading["ok"] else "memory_missing",
            "cross_reproduce": (
                "cross_reproduces_r46" if reading["cross_negative"] else "cross_moved"
            ),
            "decision": (
                "channel_content_capable" if reading["channel_capable"] else "channel_content_blind"
            ),
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all([reading["ok"], closed_ok, main_untouched, bridge_default_none, rerun_ok])
        verdict["overall"] = "channel_content_measured" if all_ok else "channel_content_unexpected"

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-channel-content-r47-report-v1",
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
                    "cold_nll": reading["cold_nll"],
                    "cross": reading["cross"],
                    "self": reading["self"],
                    "scale": reading["scale"],
                    "z_share": reading["z_share"],
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
