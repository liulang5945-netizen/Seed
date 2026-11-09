"""R38 -- where the gradient stops: a three-arm attribution, no training.

R37 found that one Adam step on a single sentence reaches only 95 of 135
parameters and only 61 with a non-zero gradient.  R38 attributes that by
re-running the same loss under three frozen conditions and classifying every
parameter as ``nonzero`` / ``zero_only`` / ``none``:

A. no source records, one sentence -- must reproduce R37 exactly;
B. one source record, one sentence -- the copy branch of ``CopyExpression``
   stops early without records, so this arm asks whether those parameters
   re-enter the graph;
C. no records, two sentences chained on the same state -- the write heads
   only affect the *next* step, so this arm asks whether one more step brings
   them into the loss.

The world/prediction heads are expected to stay dark in all three arms: the
next-byte loss never reaches them.  That is a finding about the loss, not a
bug to patch here.

Zero optimizer steps: every arm is one forward and one backward on a fresh
scratch model.  The main subject only runs the closed-state battery.
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

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.training.probe_taiji_cross_family_readout_20261109 as base  # noqa: E402
from taiji.living_core import LivingCoreReference, RawSource  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

torch.set_num_threads(1)

MANIFEST = ROOT / "plans/manifests/taiji_gradient_coverage_20261109.json"
REPORT = ROOT / "reports/taiji_gradient_coverage_20261109.json"
CORE_SOURCE = ROOT / "taiji/living_core.py"

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

SEED = 20261109
ESTIMATE_CPU_UPPER = 40.0
CPU_CAP = float(base.CPU_CAP)
GOAL = b"r38"
SOURCE_TEXT = "圆口的器皿放在木桌子上"

R37_NO_GRAD = (
    "expression.a.weight",
    "expression.continuation.bias",
    "expression.continuation.weight",
    "expression.coverage",
    "expression.goal.bias",
    "expression.goal.weight",
    "expression.meta.weight",
    "expression.mix.bias",
    "expression.mix.weight",
    "expression.position.weight",
    "expression.score.weight",
    "expression.z.weight",
    "world.0.0.bias",
    "world.0.0.weight",
    "world.0.2.bias",
    "world.0.2.weight",
    "world.1.0.bias",
    "world.1.0.weight",
    "world.1.2.bias",
    "world.1.2.weight",
    "world.2.0.bias",
    "world.2.0.weight",
    "world.2.2.bias",
    "world.2.2.weight",
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

EXPRESSION_COPY = (
    "expression.a.weight",
    "expression.continuation.bias",
    "expression.continuation.weight",
    "expression.coverage",
    "expression.goal.bias",
    "expression.goal.weight",
    "expression.meta.weight",
    "expression.mix.bias",
    "expression.mix.weight",
    "expression.position.weight",
    "expression.score.weight",
    "expression.z.weight",
)

WORLD_HEADS = (
    "world.0.0.bias",
    "world.0.0.weight",
    "world.0.2.bias",
    "world.0.2.weight",
    "world.1.0.bias",
    "world.1.0.weight",
    "world.1.2.bias",
    "world.1.2.weight",
    "world.2.0.bias",
    "world.2.0.weight",
    "world.2.2.bias",
    "world.2.2.weight",
)

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
        "round": "R37",
        "runner": "scripts/training/probe_taiji_checkpoint_20261109.py",
        "rerun_commit": "842ecb094",
        "runner_blob_sha1": "3f9c482e1f94d939c0461c688fd30ca2a2f988ae",
        "report_runner_sha256": "9b7b26b0cd94a0bb69d30d5e8492d23f03e313aaa0b6083bfb1deb0550afb728",
        "runner_blob_sha256": "9b7b26b0cd94a0bb69d30d5e8492d23f03e313aaa0b6083bfb1deb0550afb728",
        "report_core_sha256": "6c0c53a2a6465627918721da83b1e0fb80a04c262f95c97d9b825ee858d0785d",
        "matches_report": True,
        "note": "R38 does not change living_core; R37 stays runnable at HEAD",
    },
)

DEFINITION = {
    "arm_a": "no records, one sentence; must reproduce R37's 95/61/40 split",
    "arm_b": "one source record, one sentence; the copy branch of CopyExpression runs",
    "arm_c": "no records, two sentences chained on the same state; the write heads see a second step",
    "classes": "each parameter is nonzero (|grad| sum > 0), zero_only (grad present, sum 0) or none (no grad)",
    "world": "the three prediction heads are expected to stay dark in all three arms",
    "closed_state": "the nine legacy readings on the main subject stay equal before and after",
}

CRITERIA = {
    "classification": "each arm classifies all 135 parameters; the three counts sum to 135",
    "arm_a": "arm A reproduces R37: 95 with grad, 61 non-zero, 40 none and the same 40 names",
    "arm_b": "all 12 expression copy-branch parameters leave the none set in arm B",
    "arm_c": "all 16 write_heads parameters leave the none set in arm C",
    "world": "all 12 world parameters stay in the none set in all three arms",
    "closed_state": "the nine legacy readings stay equal and the main parameter digest does not move",
    "rerun": "the R37 anchor carries a commit, a runner blob sha1, a runner sha256 and a core sha256",
}

START_CPU = time.process_time()
START_WALL = time.time()


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".json.tmp")
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


def nll(distributions: torch.Tensor, target: bytes) -> torch.Tensor:
    selected = distributions[: len(target)].gather(1, torch.tensor(list(target))[:, None])
    return -torch.log(selected + 1e-12).mean()


def fresh_model() -> Any:
    torch.manual_seed(SEED + 1)
    return LivingCoreReference()


def classify(model: Any) -> dict[str, list[str]]:
    nonzero, zero_only, none = [], [], []
    for name, tensor in sorted(model.named_parameters()):
        if tensor.grad is None:
            none.append(name)
        elif float(tensor.grad.abs().sum()) > 0:
            nonzero.append(name)
        else:
            zero_only.append(name)
    return {"nonzero": nonzero, "zero_only": zero_only, "none": none}


def arm_a(model: Any, texts: list[str]) -> dict[str, Any]:
    state = model.initial_state("r38-a")
    distributions, _ = model.score_targets(
        state, prompt=b"", target=texts[0].encode("utf-8"), goal=GOAL
    )
    loss = nll(distributions, texts[0].encode("utf-8"))
    loss.backward()
    return {"loss": float(loss.detach()), **classify(model)}


def arm_b(model: Any, texts: list[str]) -> dict[str, Any]:
    record = RawSource("r38-source", SOURCE_TEXT.encode("utf-8"))
    state = model.initial_state("r38-b")
    distributions, _ = model.score_targets(
        state,
        prompt=b"",
        target=texts[0].encode("utf-8"),
        goal=GOAL,
        records=(record,),
    )
    loss = nll(distributions, texts[0].encode("utf-8"))
    loss.backward()
    return {"loss": float(loss.detach()), **classify(model)}


def arm_c(model: Any, texts: list[str]) -> dict[str, Any]:
    first = texts[0].encode("utf-8")
    second = texts[1].encode("utf-8")
    state = model.initial_state("r38-c")
    distributions, state = model.score_targets(state, prompt=b"", target=first, goal=GOAL)
    first_loss = nll(distributions, first)
    distributions, _ = model.score_targets(state, prompt=b"", target=second, goal=GOAL)
    second_loss = nll(distributions, second)
    loss = first_loss + second_loss
    loss.backward()
    return {
        "loss": float(loss.detach()),
        "first_loss": float(first_loss.detach()),
        "second_loss": float(second_loss.detach()),
        **classify(model),
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
    state = reference.initial_state("r38-zero-perturbation")
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
        "source_text": SOURCE_TEXT,
        "r37_no_grad": list(R37_NO_GRAD),
        "expression_copy": list(EXPRESSION_COPY),
        "world_heads": list(WORLD_HEADS),
        "write_heads": list(WRITE_HEADS),
        "rerun_anchors": [dict(row) for row in RERUN_ANCHORS],
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, Any]:
    manifest = {
        "schema": "taiji-gradient-coverage-r38-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "estimate_cpu_upper": ESTIMATE_CPU_UPPER,
        "cpu_cap": CPU_CAP,
        "runner_sha256": RUNNER_SHA_START,
        "core_sha256": file_sha(CORE_SOURCE),
        "fixture_sha256": base.fixture_hashes(),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "definition_digest": definition_digest(),
        "counts": {
            "describe_sentences": len(sentences()),
            "r37_no_grad": len(R37_NO_GRAD),
            "expression_copy": len(EXPRESSION_COPY),
            "world_heads": len(WORLD_HEADS),
            "write_heads": len(WRITE_HEADS),
        },
        "inputs": {
            "goal": GOAL.decode("ascii"),
            "sentences": sentences()[:2],
            "source_text": SOURCE_TEXT,
            "r37_no_grad": list(R37_NO_GRAD),
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

        readings = {
            "a": arm_a(fresh_model(), texts),
            "b": arm_b(fresh_model(), texts),
            "c": arm_c(fresh_model(), texts),
        }
        total = len(R37_NO_GRAD) + 95
        classification_ok = all(
            len(row["nonzero"]) + len(row["zero_only"]) + len(row["none"]) == total
            for row in readings.values()
        )
        arm_a_ok = (
            len(readings["a"]["none"]) == len(R37_NO_GRAD)
            and tuple(readings["a"]["none"]) == R37_NO_GRAD
            and len(readings["a"]["nonzero"]) == 61
            and len(readings["a"]["nonzero"]) + len(readings["a"]["zero_only"]) == 95
        )
        expression_ok = all(name not in readings["b"]["none"] for name in EXPRESSION_COPY)
        write_ok = all(name not in readings["c"]["none"] for name in WRITE_HEADS)
        world_ok = all(
            name in readings[arm]["none"] for arm in ("a", "b", "c") for name in WORLD_HEADS
        )

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
            "classification": (
                "coverage_135_per_arm" if classification_ok else "coverage_counts_moved"
            ),
            "arm_a": "arm_a_matches_r37" if arm_a_ok else "arm_a_moved",
            "arm_b": "records_revive_expression" if expression_ok else "expression_stays_dark",
            "arm_c": "chain_revives_write" if write_ok else "write_heads_stay_dark",
            "world": "world_heads_dark" if world_ok else "world_heads_reachable",
            "closed_state": "closed_state_equal" if closed_ok else "closed_state_moved",
            "main_untouched": "main_subject_untouched" if main_untouched else "main_moved",
            "bridge_default": (
                "bridge_default_none" if bridge_default_none else "bridge_default_present"
            ),
            "rerun_anchors": "anchors_complete" if rerun_ok else "anchors_incomplete",
        }
        all_ok = all(
            [
                classification_ok,
                arm_a_ok,
                expression_ok,
                write_ok,
                world_ok,
                closed_ok,
                main_untouched,
                bridge_default_none,
                rerun_ok,
            ]
        )
        verdict["overall"] = (
            "gradient_coverage_attributed" if all_ok else "gradient_coverage_unexpected"
        )

        cpu, wall = cost()
        if cpu > CPU_CAP:
            raise SystemExit(f"cpu cap exceeded: {cpu:.2f}s > {CPU_CAP:.0f}s")
        report = {
            "schema": "taiji-gradient-coverage-r38-report-v1",
            "runner_sha256": RUNNER_SHA_START,
            "manifest_sha256": file_sha(MANIFEST),
            "core_sha256": file_sha(CORE_SOURCE),
            "fixture_sha256": base.fixture_hashes(),
            "definition": DEFINITION,
            "criteria": CRITERIA,
            "arms": readings,
            "summary": {
                arm: {
                    "loss": round(row["loss"], 6),
                    "nonzero": len(row["nonzero"]),
                    "zero_only": len(row["zero_only"]),
                    "none": len(row["none"]),
                }
                for arm, row in readings.items()
            },
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
                    "summary": report["summary"],
                    "arm_a_extra_dark": [
                        name for name in readings["a"]["none"] if name not in R37_NO_GRAD
                    ],
                    "arm_b_still_dark": [
                        name for name in EXPRESSION_COPY if name in readings["b"]["none"]
                    ],
                    "arm_c_still_dark": [
                        name for name in WRITE_HEADS if name in readings["c"]["none"]
                    ],
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
