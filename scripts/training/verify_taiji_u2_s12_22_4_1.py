"""Independent-process recomputation of one u1 22.4.1 S12 receipt."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.training.u2_attempt_ledger_22_3_4_3 import AttemptLedger
from scripts.training.u2_update_executor_22_4_1 import load_json, write_once
from scripts.training.u2_update_support_22_3_4_3 import distributions, file_sha, readback
from scripts.training.u2_update_support_22_4_1 import (
    activity_receipt,
    baseline_fingerprint,
    baseline_outputs,
    baseline_run_conditions,
    check_adam_equations,
    gradients_for,
    normalized_sha256,
    output_displacement,
    readback_baseline,
    shadow_terms,
)
from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_learning_transaction import rebuild_state
from taiji.u2_recovery import run_conditions

ROOT = Path(__file__).resolve().parents[2]


def close_enough(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12 + 1e-9 * abs(right)


def receipt_equal(left: dict, right: dict) -> bool:
    if [row["parameter"] for row in left["parameters"]] != [
        row["parameter"] for row in right["parameters"]
    ]:
        return False
    for old_row, new_row in zip(left["parameters"], right["parameters"], strict=True):
        for key in ("gradient_norm", "change", "margin"):
            if not close_enough(new_row[key], old_row[key]):
                return False
        if old_row["barely_live"] != new_row["barely_live"]:
            return False
    return close_enough(right["min_margin"], left["min_margin"]) and (
        left["barely_live_parameters"] == right["barely_live_parameters"]
    )


def ledger_run(run: dict) -> dict:
    return {
        "run_id": run["run_id"],
        "authorized": run["authorized"],
        "owner_approval_ref": run["owner_approval_ref"],
        "manifest_sha256": run["references"]["manifest_sha256"],
    }


def precontract(run: dict) -> dict:
    identity = run["update_id"]
    return {
        "run": {
            "attempt_order": [identity],
            "attempt_parents": {identity: "parent.bin"},
            "attempt_posts": {identity: "post.bin"},
            "budget_max": run["attempts_authorized"],
        }
    }


def verify(s12_path: Path) -> dict:
    torch.set_num_threads(1)
    started, wall = time.process_time(), time.monotonic()
    steps = {"calls": 0}
    original_step = torch.optim.Adam.step

    def tripwire(self, *args, **kwargs):
        steps["calls"] += 1
        raise AssertionError("the S12 worker must not call optimizer.step")

    torch.optim.Adam.step = tripwire
    try:
        s12 = load_json(s12_path)
        root = ROOT / s12["run_root"]
        run = load_json(root / "run.json")
        execution = load_json(ROOT / s12["execution_manifest"])
        assert normalized_sha256(ROOT / s12["execution_manifest"]) == s12["manifest_sha256"]
        for name, pin in execution["authority"]["source_pins"].items():
            assert normalized_sha256(ROOT / name) == pin, name
        assert (
            normalized_sha256(ROOT / s12["admission_manifest"]) == s12["admission_manifest_sha256"]
        )

        ledger = AttemptLedger(
            ROOT / run["paths"]["ledger"],
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        state = ledger.read()
        identity = s12["attempt_id"]
        assert identity in state["opened"] and identity in state["closed"], "ledger mapping"
        assert state["opened"][identity]["seq"] == s12["ledger_seq_open"]
        close_row = state["closed"][identity]
        assert close_row["outcome"] == s12["status"]
        if s12.get("ledger_seq_close") is None:
            # 顺序契约：回执先写、close 后写，close 行携带回执文件的 sha256。
            assert close_row.get("s12_sha256") == file_sha(s12_path), "close row s12 attestation"
            lifecycle = [
                row for row in state["rows"] if row.get("event") in {"open", "close", "reconcile"}
            ]
            assert lifecycle[-1].get("seq") == close_row["seq"], "close is not the last event"
        else:
            assert close_row["seq"] == s12["ledger_seq_close"]

        baseline = s12.get("execution_graph") is not None
        arm = s12.get("arm") or s12["attempt_id"]
        parent_expected = {
            "state_sha256": s12["pre_state_sha256"],
            "parameter_sha256": s12["pre_parameter_sha256"],
            "moment_sha256": s12["pre_moment_sha256"],
            "history_reference": s12["pre_history_reference"],
            "revision": s12["pre_revision"],
        }
        if baseline:
            parent, parent_proof = readback_baseline(
                root / "parent.bin", parent_expected, execution, arm=arm
            )
        else:
            parent, parent_proof = readback(root / "parent.bin", parent_expected)
        checkpoints = {
            "parent": parent_proof["sha256"],
            "parent_readback": True,
        }
        if s12["status"] != "accepted":
            return {
                "schema": "taiji-u2-22.4.1-s12-worker-v1",
                "verdict": "pass",
                "s12": str(s12_path),
                "status": "rejected_verified",
                "checkpoints": checkpoints,
                "optimizer_step_calls": steps["calls"],
                "cpu_seconds": time.process_time() - started,
                "wall_seconds": time.monotonic() - wall,
            }

        assert s12["optimizer_step_calls"] == 1
        post_expected = {
            "state_sha256": s12["post_state_sha256"],
            "parameter_sha256": s12["post_parameter_sha256"],
            "moment_sha256": s12["post_moment_sha256"],
            "history_reference": s12["history_reference"],
            "revision": s12["revision"],
        }
        if baseline:
            return verify_baseline(
                s12=s12,
                s12_path=s12_path,
                execution=execution,
                arm=arm,
                root=root,
                parent=parent,
                post_expected=post_expected,
                checkpoints=checkpoints,
                steps=steps,
                started=started,
                wall=wall,
            )
        post, post_proof = readback(root / "post.bin", post_expected)
        checkpoints["post"] = post_proof["sha256"]
        checkpoints["post_readback"] = True
        assert post.model.parameter_revision == parent.model.parameter_revision + 1

        gradients, losses = gradients_for(parent, execution)
        assert value_digest(gradients) == s12["gradient_sha256"], "gradient digest"
        for left, right in zip(
            s12["parameter_rows"],
            [
                {
                    "parameter": name,
                    "gradient_norm": None if gradient is None else float(gradient.norm()),
                    "max_parameter_change": float(
                        (parent.model.state_dict()[name] - post.model.state_dict()[name])
                        .abs()
                        .max()
                    ),
                }
                for name, gradient in gradients.items()
            ],
            strict=True,
        ):
            assert left["parameter"] == right["parameter"]
            for key in ("gradient_norm", "max_parameter_change"):
                if left[key] is None or right[key] is None:
                    assert left[key] == right[key]
                else:
                    assert close_enough(right[key], left[key]), (left["parameter"], key)

        adam = check_adam_equations(parent, post, gradients, execution)
        before = {name: value.detach().clone() for name, value in parent.model.state_dict().items()}
        after = post.model.state_dict()
        recomputed = activity_receipt(parent.model, before, after, gradients, execution)
        # shadow_terms 内部经 choose_operation 调 torch.autograd.grad，必须允许梯度；
        # 这里只取数值，取完立刻 detach。
        with torch.enable_grad():
            post_loss, _ = shadow_terms(post.model, execution)
        post_loss = post_loss.detach()
        recomputed["output_displacement_descriptive"] = {
            **output_displacement(distributions(parent), distributions(post)),
            "same_fixture_loss_delta": float(post_loss) - float(s12["losses"]["total"]),
        }
        assert receipt_equal(s12["activity_receipt"], recomputed), "activity receipt"
        for key, value in recomputed["output_displacement_descriptive"].items():
            assert close_enough(
                value, s12["activity_receipt"]["output_displacement_descriptive"][key]
            ), key

        rebuilt, rebuild_metrics = rebuild_state(post.model, post.history, post.state)
        assert_values_equal(value_digest(rebuilt), value_digest(post.state), "worker.state_rebuild")
        assert run_conditions(post.model) == s12["run_conditions"]
        assert steps["calls"] == 0, "the worker must not step"
        report = {
            "schema": "taiji-u2-22.4.1-s12-worker-v1",
            "verdict": "pass",
            "s12": str(s12_path),
            "status": "accepted_verified",
            "checkpoints": checkpoints,
            "gradient_sha256": value_digest(gradients),
            "parameter_rows": len(gradients),
            "adam_equations": len(adam),
            "losses_recomputed": losses,
            "activity_receipt_rederived": True,
            "optimizer_step_calls": steps["calls"],
            "run_conditions": run_conditions(post.model),
            "cpu_seconds": time.process_time() - started,
            "wall_seconds": time.monotonic() - wall,
        }
        return report
    finally:
        torch.optim.Adam.step = original_step


def verify_baseline(
    *,
    s12: dict,
    s12_path: Path,
    execution: dict,
    arm: str,
    root: Path,
    parent,
    post_expected: dict,
    checkpoints: dict,
    steps: dict,
    started: float,
    wall: float,
) -> dict:
    """Independent recomputation for the baseline arm.

    与 u1 的差别只有三处：读回走基线 checkpoint、梯度用基线指纹、没有主体 state 可重建。
    其余（梯度摘要、逐参数行、Adam 方程、活性回执、同一夹具损失差）逐项相同。
    """
    post, post_proof = readback_baseline(root / "post.bin", post_expected, execution, arm=arm)
    checkpoints["post"] = post_proof["sha256"]
    checkpoints["post_readback"] = True
    assert post.revision == parent.revision + 1, "baseline revision"

    gradients, losses = gradients_for(
        parent, execution, arm=arm, fingerprint_fn=baseline_fingerprint
    )
    assert value_digest(gradients) == s12["gradient_sha256"], "gradient digest"
    for left, right in zip(
        s12["parameter_rows"],
        [
            {
                "parameter": name,
                "gradient_norm": None if gradient is None else float(gradient.norm()),
                "max_parameter_change": float(
                    (parent.model.state_dict()[name] - post.model.state_dict()[name]).abs().max()
                ),
            }
            for name, gradient in gradients.items()
        ],
        strict=True,
    ):
        assert left["parameter"] == right["parameter"]
        for key in ("gradient_norm", "max_parameter_change"):
            if left[key] is None or right[key] is None:
                assert left[key] == right[key]
            else:
                assert close_enough(right[key], left[key]), (left["parameter"], key)

    adam = check_adam_equations(parent, post, gradients, execution)
    before = {name: value.detach().clone() for name, value in parent.model.state_dict().items()}
    after = post.model.state_dict()
    recomputed = activity_receipt(parent.model, before, after, gradients, execution, arm=arm)
    with torch.enable_grad():
        post_loss, _ = shadow_terms(post.model, execution, arm=arm)
    post_loss = post_loss.detach()
    parent_output = baseline_outputs(parent, execution)
    post_output = baseline_outputs(post, execution)
    recomputed["output_displacement_descriptive"] = {
        **output_displacement(parent_output, post_output),
        "same_fixture_loss_delta": float(post_loss) - float(s12["losses"]["total"]),
        "policy_note": post_output["policy_note"],
    }
    assert receipt_equal(s12["activity_receipt"], recomputed), "activity receipt"
    for key, value in recomputed["output_displacement_descriptive"].items():
        if key == "policy_note":
            assert value == s12["activity_receipt"]["output_displacement_descriptive"][key], key
        else:
            assert close_enough(
                value, s12["activity_receipt"]["output_displacement_descriptive"][key]
            ), key

    assert baseline_run_conditions(post.model) == s12["run_conditions"]
    assert steps["calls"] == 0, "the worker must not step"
    return {
        "schema": "taiji-u2-22.4.1-s12-worker-v1",
        "verdict": "pass",
        "s12": str(s12_path),
        "status": "accepted_verified_baseline",
        "execution_graph": s12["execution_graph"],
        "checkpoints": checkpoints,
        "gradient_sha256": value_digest(gradients),
        "parameter_rows": len(gradients),
        "adam_equations": len(adam),
        "losses_recomputed": losses,
        "activity_receipt_rederived": True,
        "state_rebuild": "not_applicable_baseline_has_no_subject_state",
        "policy": "structural_zero_no_policy_head",
        "optimizer_step_calls": steps["calls"],
        "run_conditions": baseline_run_conditions(post.model),
        "cpu_seconds": time.process_time() - started,
        "wall_seconds": time.monotonic() - wall,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(json.dumps({"verdict": "fail", "error": "usage: verify <s12.json>"}))
        return 2
    s12_path = Path(argv[1]).resolve()
    try:
        report = verify(s12_path)
    except Exception as error:  # noqa: BLE001 - the worker must fail closed
        report = {
            "schema": "taiji-u2-22.4.1-s12-worker-v1",
            "verdict": "fail",
            "s12": str(s12_path),
            "error": f"{type(error).__name__}: {error}",
        }
    destination = s12_path.with_name(s12_path.stem + "_worker.json")
    if not destination.exists():
        write_once(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
