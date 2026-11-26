"""R021 forward and equation coverage receipt; never runs a backward or optimizer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji import living_core as core  # noqa: E402

TEST_FILE = ROOT / "tests/taiji_native/test_living_core_reference.py"
CONTRACT = ROOT / "plans/reference/SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021_20261031.md"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checks():
    spec = importlib.util.spec_from_file_location("r021_forward_checks", TEST_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def location(symbol) -> dict:
    lines, start = inspect.getsourcelines(symbol)
    return {
        "file": str(Path(inspect.getfile(symbol)).relative_to(ROOT).as_posix()),
        "symbol": symbol.__qualname__,
        "line": start,
        "last_line": start + len(lines) - 1,
    }


def verify() -> dict:
    start = time.perf_counter()
    cpu = time.process_time()
    checks = load_checks()
    results = []
    for name in sorted(name for name in vars(checks) if name.startswith("test_r021_")):
        function = getattr(checks, name)
        try:
            function()
            result = {"id": name, "passed": True}
        except Exception as exc:  # Evidence records an actual forward failure, not a fake PASS.
            result = {"id": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        result["test"] = location(function)
        results.append(result)
    model, state, _, _ = checks.model_case()
    groups = [
        (
            "I1",
            [
                core.BytePatchEncoder.context,
                core.BytePatchEncoder.patches,
                core.BytePatchEncoder.goal_vector,
                core.LivingCoreReference.observe,
            ],
            ("encoder",),
            ["event patches", "goal", "raw-byte positions"],
            ["i1_patch_pooling_and_causality", "target_is_consumed_after_prediction"],
        ),
        (
            "I2",
            [core.LivingCoreReference.update_nu],
            ("nu_projection",),
            ["nu_ema", "known/missing"],
            ["i2_nu_missing_and_projection"],
        ),
        (
            "I3",
            [core.LivingCoreReference.source_view, core.EpisodicRead.forward],
            ("episodic",),
            ["legal source positions", "null", "r_i"],
            ["i3_legal_sources_null_and_two_tanh", "current_encoding_and_world_heads"],
        ),
        (
            "I4",
            [core.LivingCoreReference.messages],
            ("message_q", "message_k", "message_v", "message_goal"),
            ["old H", "messages"],
            ["i4_formal_group_qkv"],
        ),
        (
            "F_READ",
            [core.LivingCoreReference.read_fast],
            ("fast_queries", "fast_outputs"),
            ["old F", "F transpose query", "f_i"],
            ["fast_read_and_dynamics"],
        ),
        (
            "I5",
            [core.QueryRead.forward, core.LivingCoreReference.step],
            ("xi", "readout"),
            ["updated H", "xi", "z", "ordered head concatenation"],
            ["i5_query_read_and_head_order"],
        ),
        (
            "I6",
            [
                core.CopyExpression.forward,
                core.LivingCoreReference.consume_output,
                core.LivingCoreReference.align_positions,
                core.LivingCoreReference.score_targets,
            ],
            ("expression",),
            ["copy/generation distribution", "coverage", "prior", "position keys"],
            [
                "i6_mixture_and_no_source",
                "i6_coverage_stays_inside_source",
                "target_is_consumed_after_prediction",
            ],
        ),
        (
            "DYNAMICS",
            [core.state_normalize, core.unit_ball, core.LivingCoreReference.step],
            ("W", "U", "V", "R", "P", "tau_logit"),
            ["H", "seven-block x", "preactivation"],
            ["fast_read_and_dynamics", "budgets_and_malformed_interfaces_fail"],
        ),
        (
            "F_WRITE",
            [core.LivingCoreReference.edit_fast, core.LivingCoreReference.write_fast],
            ("write_heads",),
            ["F", "continuous chi", "per-head hard permission"],
            ["continuous_fast_update_and_permissions"],
        ),
        (
            "WORLD_FORWARD",
            [core.LivingCoreReference.step],
            ("world",),
            ["3 prediction/reward heads; no targets or update"],
            ["current_encoding_and_world_heads"],
        ),
    ]
    inventory = []
    unowned = []
    for name, parameter in model.named_parameters():
        owners = [
            identity
            for identity, _, prefixes, _, _ in groups
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        ]
        if len(owners) != 1:
            unowned.append({"parameter": name, "owners": owners})
        inventory.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "requires_grad": parameter.requires_grad,
                "forward_owner": owners,
                "update_owner": "Psi" if name in {"U", "V"} else "Theta",
                "gradient_validation": "not_run_stage_09.2.2",
            }
        )
    known_checks = {row["id"]: row for row in results}
    matrix = []
    for identity, symbols, prefixes, states, names in groups:
        ids = ["test_r021_" + name for name in names]
        matrix.append(
            {
                "equation": identity,
                "spec": CONTRACT.relative_to(ROOT).as_posix(),
                "sources": [location(symbol) for symbol in symbols],
                "parameters": [
                    row["name"] for row in inventory if row["forward_owner"] == [identity]
                ],
                "registered_parameter_prefixes": list(prefixes),
                "state_or_intermediate": states,
                "check_ids": ids,
                "status": (
                    "forward_checked"
                    if all(known_checks.get(name, {}).get("passed") for name in ids)
                    else "failed"
                ),
            }
        )
    deferred = [
        {"equation": "learning_ownership_and_actual_updates", "stage": "09.2.2"},
        {"equation": "coverage_birth_and_noise_calibration", "stage": "09.2.3"},
        {"equation": "encoder_state_migration", "stage": "09.2.3"},
        {"equation": "full_subject_restore_and_next_update", "stage": "09.2.4"},
        {"equation": "learned_endogenous_activity_and_world_targets", "stage": "09.3/09.4"},
        {"equation": "meta_learning_consolidation_and_retention", "stage": "09.4"},
    ]
    passed = all(row["passed"] for row in results) and not unowned
    passed = passed and all(row["requires_grad"] for row in inventory)
    passed = passed and all(row["status"] == "forward_checked" for row in matrix)
    parameter_hash = hashlib.sha256()
    for name, parameter in model.named_parameters():
        parameter_hash.update(name.encode("utf-8"))
        parameter_hash.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return {
        "schema": "taiji-r021-formal-forward-coverage-v1",
        "date": "2026-10-31",
        "baseline_commit": "da4415e58",
        "status": "passed_r021_forward_only" if passed else "failed",
        "config": asdict(model.config),
        "dtype": str(model.W.dtype),
        "torch_version": core.torch.__version__,
        "device": str(model.W.device),
        "fixture_parameter_sha256": parameter_hash.hexdigest(),
        "forward_tolerances": {"atol": 1e-12, "rtol": 1e-12},
        "subject_id": state.subject_id,
        "checks": results,
        "coverage_matrix": matrix,
        "parameter_inventory": inventory,
        "unowned_parameters": unowned,
        "explicitly_not_implemented": deferred,
        "source_hashes": {
            p.relative_to(ROOT).as_posix(): digest(p)
            for p in (CONTRACT, Path(core.__file__), TEST_FILE, Path(__file__))
        },
        "optimizer_steps": 0,
        "backward_calls": 0,
        "data_splits_read": [],
        "full_subject_restore": False,
        "default_model_replaced": False,
        "cpu_seconds_checks_only": time.process_time() - cpu,
        "wall_seconds_checks_only": time.perf_counter() - start,
        "limits": "Forward and explicit coverage only. Trainable registration is not gradient "
        "ownership validation. No full subject, learning, growth, retention, recovery, pricing "
        "or capability approval is implied.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("R021 receipt is write-once")
    result = verify()
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(result["checks"]),
                "failed": [row["id"] for row in result["checks"] if not row["passed"]],
                "unowned_parameters": result["unowned_parameters"],
            }
        )
    )
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
