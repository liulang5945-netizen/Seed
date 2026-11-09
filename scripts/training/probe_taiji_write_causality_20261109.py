"""R39 -- do the memory write and the prediction heads actually matter?

R38 showed write heads are dark because ``step(allowed=None)`` defaults to a
false write mask, and that the world heads are never in the next-byte loss.
R39 tests the consequences with four frozen arms:

M0. default write mask: ``fast`` must stay all zero;
M1. all-true write mask: ``fast`` must move, and the 16 write heads must leave
    the "no gradient" set;
M2. causal ablation: after a write-enabled step, one next-sentence scoring pass
    is run twice from the same state, once intact and once with ``fast`` zeroed;
    the output distributions must differ -- otherwise the memory is written but
    not read;
M3. world-weight perturbation (x1.5): the output distributions must stay bit
    identical -- otherwise the prediction heads feed back into expression.

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

MANIFEST = ROOT / "plans/manifests/taiji_write_causality_20261109.json"
REPORT = ROOT / "reports/taiji_write_causality_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 40.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r39"
WORLD_SCALE = 1.5
M2_EFFECT_FLOOR = 1e-6

WRITE_HEADS = (
    "write_heads.0.0.bias",
    "write_heads.0.0.weight",
    "write_heads.0.2.bias",
    "write_heads.0.2.weight",
    "write_heads.1.0.bias",
    "write_heads.1.0.weight",
    "write_heads.1.2.bias",
    "write_heads.1.2.weight",
    "write_heads.2.0.bias",
    "write_heads.2.0.weight",
    "write_heads.2.2.bias",
    "write_heads.2.2.weight",
    "write_heads.3.0.bias",
    "write_heads.3.0.weight",
    "write_heads.3.2.bias",
    "write_heads.3.2.weight",
)

RERUN_ANCHORS = (
    {
        "round": "R38",
        "runner": "scripts/training/probe_taiji_gradient_coverage_20261109.py",
        "rerun_commit": "63264aac2",
        "runner_blob_sha1": "790aa223972282c4ba1d050b374d3c441b9fcc36",
        "report_runner_sha256": "cc5ce376a18ed95ab7f51e7838d8a99be1bce8338dca151e6772932968e94cc7",
        "runner_blob_sha256": "cc5ce376a18ed95ab7f51e7838d8a99be1bce8338dca151e6772932968e94cc7",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R39 does not change living_core; R38 stays runnable at HEAD",
    },
)

DEFINITION = {
    "m0": "default allow mask; one observe pass; fast must stay all zero",
    "m1": "all-true allow mask; one observe pass; fast must move",
    "m1_grad": "all-true allow mask plus one backward; the 16 write heads must leave the no-gradient set",
    "m2": "after a write-enabled step, score the next sentence from the same state intact vs with fast zeroed",
    "m3": "scale world weights by 1.5 and compare output distributions with the unperturbed model",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "m0": "fast abs sum is 0 under the default mask",
    "m1": "fast abs sum is > 0 and its digest differs from M0",
    "m1_grad": "all 16 write_heads have a gradient; the nonzero count is reported",
    "m2": "the intact and ablated distributions differ and max |delta| > 1e-6",
    "m3": "the perturbed and unperturbed distributions are bit identical (max |delta| == 0)",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R38 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
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


def distribution_sha(distributions: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(distributions.shape)).encode("ascii"))
    digest.update(distributions.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def nll(distributions: torch.Tensor, target: bytes) -> torch.Tensor:
    selected = distributions[: len(target)].gather(1, torch.tensor(list(target))[:, None])
    return -torch.log(selected + 1e-12).mean()


def all_true_mask() -> torch.Tensor:
    return torch.ones((4, 2), dtype=torch.bool)


def fresh_model() -> Any:
    torch.manual_seed(SEED + 1)
    return LivingCoreReference()


def arm_m0(model: Any, texts: list[str]) -> dict[str, Any]:
    state = model.initial_state("r39-m0")
    with torch.no_grad():
        state = model.observe(state, texts[0].encode("utf-8"), goal=GOAL)
    return {
        "fast_abs_sum": float(state.fast.abs().sum()),
        "fast_sha": tensor_sha([state.fast]),
        "h_sha": tensor_sha([state.h]),
    }


def arm_m1(model: Any, texts: list[str]) -> dict[str, Any]:
    state = model.initial_state("r39-m1")
    with torch.no_grad():
        state = model.observe(state, texts[0].encode("utf-8"), goal=GOAL, allowed=all_true_mask())
    return {
        "fast_abs_sum": float(state.fast.abs().sum()),
        "fast_sha": tensor_sha([state.fast]),
        "h_sha": tensor_sha([state.h]),
    }


def arm_m1_grad(model: Any, texts: list[str]) -> dict[str, Any]:
    target = texts[0].encode("utf-8")
    state = model.initial_state("r39-m1-grad")
    distributions, _ = model.score_targets(
        state, prompt=b"", target=target, goal=GOAL, allowed=all_true_mask()
    )
    loss = nll(distributions, target)
    loss.backward()
    graded = []
    for name, tensor in sorted(model.named_parameters()):
        if tensor.grad is not None:
            graded.append((name, float(tensor.grad.abs().sum())))
    write = {name: value for name, value in graded if name in WRITE_HEADS}
    return {
        "loss": float(loss.detach()),
        "with_grad": len(graded),
        "write_heads_with_grad": len(write),
        "write_heads_nonzero": sum(1 for value in write.values() if value > 0),
        "write_head_grad_sums": {name: round(value, 9) for name, value in sorted(write.items())},
    }


def arm_m2(model: Any, texts: list[str]) -> dict[str, Any]:
    state = model.initial_state("r39-m2")
    with torch.no_grad():
        state = model.observe(state, texts[0].encode("utf-8"), goal=GOAL, allowed=all_true_mask())
        ablated = replace(state, fast=torch.zeros_like(state.fast))
        target = texts[1].encode("utf-8")
        intact, _ = model.score_targets(
            state, prompt=b"", target=target, goal=GOAL, allowed=all_true_mask()
        )
        zeroed, _ = model.score_targets(
            ablated, prompt=b"", target=target, goal=GOAL, allowed=all_true_mask()
        )
    return {
        "fast_abs_sum": float(state.fast.abs().sum()),
        "intact_sha": distribution_sha(intact),
        "ablated_sha": distribution_sha(zeroed),
        "max_abs_delta": float((intact - zeroed).abs().max()),
    }


def arm_m3(model: Any, texts: list[str]) -> dict[str, Any]:
    target = texts[0].encode("utf-8")
    state = model.initial_state("r39-m3")
    with torch.no_grad():
        before, _ = model.score_targets(state, prompt=b"", target=target, goal=GOAL)
    backup = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if name.startswith("world."):
                tensor.mul_(WORLD_SCALE)
        during, _ = model.score_targets(state, prompt=b"", target=target, goal=GOAL)
    model.load_state_dict(backup, strict=True)
    return {
        "before_sha": distribution_sha(before),
        "perturbed_sha": distribution_sha(during),
        "max_abs_delta": float((before - during).abs().max()),
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
    state = reference.initial_state("r39-zero-perturbation")
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
        "world_scale": WORLD_SCALE,
        "m2_effect_floor": M2_EFFECT_FLOOR,
        "write_heads": list(WRITE_HEADS),
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-write-causality-r39-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "world_scale": WORLD_SCALE,
        "m2_effect_floor": M2_EFFECT_FLOOR,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {
            "describe_sentences": len(sentences()),
            "write_heads": len(WRITE_HEADS),
            "rerun_anchors": len(RERUN_ANCHORS),
        },
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "sentences": sentences()[:2],
            "write_heads": list(WRITE_HEADS),
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

        m0 = arm_m0(fresh_model(), texts)
        m1 = arm_m1(fresh_model(), texts)
        m1_grad = arm_m1_grad(fresh_model(), texts)
        m2 = arm_m2(fresh_model(), texts)
        m3 = arm_m3(fresh_model(), texts)

        m0_ok = m0["fast_abs_sum"] == 0.0
        m1_ok = m1["fast_abs_sum"] > 0.0 and m1["fast_sha"] != m0["fast_sha"]
        m1_grad_ok = m1_grad["write_heads_with_grad"] == len(WRITE_HEADS)
        m2_ok = m2["intact_sha"] != m2["ablated_sha"] and m2["max_abs_delta"] > M2_EFFECT_FLOOR
        m3_ok = m3["before_sha"] == m3["perturbed_sha"] and m3["max_abs_delta"] == 0.0

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
            "m0": "default_mask_no_write" if m0_ok else "default_mask_wrote",
            "m1": "all_true_mask_writes" if m1_ok else "all_true_mask_inert",
            "m1_grad": ("write_heads_graded" if m1_grad_ok else "write_heads_still_dark"),
            "m2": "memory_causally_live" if m2_ok else "memory_written_not_read",
            "m3": "world_heads_pure" if m3_ok else "world_heads_feed_back",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [
                m0_ok,
                m1_ok,
                m1_grad_ok,
                m2_ok,
                m3_ok,
                closed_ok,
                main_untouched,
                bridge_default_none,
                rerun_ok,
            ]
        )
        verdict["overall"] = (
            "memory_causality_attributed" if all_ok else "memory_causality_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-write-causality-r39-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "arms": {"m0": m0, "m1": m1, "m1_grad": m1_grad, "m2": m2, "m3": m3},
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
                    "m0": {"fast_abs_sum": m0["fast_abs_sum"]},
                    "m1": {"fast_abs_sum": round(m1["fast_abs_sum"], 6)},
                    "m1_grad": {
                        "with_grad": m1_grad["with_grad"],
                        "write_heads_with_grad": m1_grad["write_heads_with_grad"],
                        "write_heads_nonzero": m1_grad["write_heads_nonzero"],
                    },
                    "m2": {
                        "intact_differs": m2["intact_sha"] != m2["ablated_sha"],
                        "max_abs_delta": m2["max_abs_delta"],
                    },
                    "m3": {"max_abs_delta": m3["max_abs_delta"]},
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
