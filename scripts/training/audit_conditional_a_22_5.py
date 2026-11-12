"""Close the frozen intervention and trace native input sensitivity without updates."""

# ruff: noqa: E402 -- include startup in the measured interval
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.training.check_report_run_conditions import write_report
from scripts.training.conditional_a_support_22_5 import paired_example
from scripts.training.first_a_support_22_5 import load_learner, output, prepare
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_subject import bounded

REPORT = ROOT / "reports/taiji_u2_conditional_A_closeout_20261112.json"


def guard():
    if time.process_time() > 120 or time.monotonic() - STARTED > 240:
        raise RuntimeError("readonly_A_audit_budget_stop")


def trace(model, example):
    guard()
    public = example["public"]
    tokens = model.encode(public["observations"], public["coordinates"], public["outcome"])
    state = prepare(model, public, example["id"])
    goal = model.language_goal(state)
    h = model.read_h(state, goal)
    z = torch.cat(
        [
            model.object_z(state),
            bounded(model.background)[None, None].expand(model.c.branches, 1, -1),
        ],
        1,
    )
    score = (model.o_key(z) * model.o_query(h)[:, None]).sum(-1) / math.sqrt(model.c.d)
    score = score + torch.cat(
        [F.logsigmoid(state["l"]).squeeze(-1), state["l"].new_zeros(model.c.branches, 1)], -1
    )
    attention = score.softmax(-1)
    o = model.read_o(state, h)
    reconstructed = (attention[..., None] * model.o_value(z)).sum(-2)
    torch.testing.assert_close(o, reconstructed, atol=1e-7, rtol=1e-6)
    probability, _, _ = output(model, state)
    return {
        "rgb_tokens": tokens["rgb"],
        "object_z": model.object_z(state),
        "object_position": state["p"],
        "H": state["H"],
        "F": state["F"],
        "request_embedding": goal,
        "read_h": h,
        "object_attention": attention,
        "read_o": o,
        "branch_context": model.branch_context(state, goal, state["prediction_action"]),
        "first_byte_probability": probability.exp(),
    }


def comparison(original, changed):
    return {
        name: {
            "delta_rms": float((value - changed[name]).square().mean().sqrt()),
            "reference_rms": float(value.square().mean().sqrt()),
            "max_abs": float((value - changed[name]).abs().max()),
        }
        for name, value in original.items()
    }


def audit():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    torch.set_num_threads(1)
    manifest_path = ROOT / "plans/manifests/taiji_u2_conditional_A_execution_20261112_v2.json"
    result_path = ROOT / "reports/taiji_u2_conditional_A_20261112.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(result_path.read_text(encoding="utf-8"))
    assert results["verdict"] == "completed_conditional_A_intervention"
    for path, pin in manifest["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / path).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == pin
        )
    data = atomic_load(Path(manifest["run_root"]) / "data.bin")
    assert value_digest(data) == manifest["data_digest"]
    audits, diagnostics = {}, {}
    with StepGuard() as blocked, torch.no_grad():
        for arm, recipe in manifest["arms"].items():
            guard()
            result = results["arms"][arm]
            directory = Path(recipe["initial"]).parent
            claims = sorted(p.name.removeprefix("claim-") for p in directory.glob("claim-*.bin"))
            checkpoints = sorted(
                p.name.removeprefix("update-") for p in directory.glob("update-*.bin")
            )
            expected = [f"{i:06}.bin" for i in range(1, recipe["updates"] + 1)]
            assert claims == checkpoints == expected
            model, optimizer, payload = load_learner(result["checkpoint"])
            initial = atomic_load(recipe["initial"])
            assert value_digest(initial["parameters"]) == recipe["parent_parameter_digest"]
            assert payload["update"] == recipe["updates"] == result["updates"]
            assert payload["cursor"] == 2 * recipe["updates"]
            assert payload["data_digest"] == manifest["data_digest"]
            assert optimizer.state
            assert max(int(s["step"]) for s in optimizer.state.values()) == recipe["updates"]
            audits[arm] = {
                "updates": recipe["updates"],
                "matching_claim_and_checkpoint_files": len(checkpoints),
                "final_cursor": payload["cursor"],
                "adam_parameter_states": len(optimizer.state),
                "initial_parameter_digest": value_digest(initial["parameters"]),
                "final_parameter_digest": value_digest(model.state_dict()),
                "checkpoint_sha256": hashlib.sha256(
                    Path(result["checkpoint"]).read_bytes()
                ).hexdigest(),
                "initial_fit": result["initial"]["accuracy"],
                "final_fit": result["final"]["accuracy"],
                "pair_margin_before": result["pair_margin_before"],
                "pair_margin_after": result["pair_margin_after"],
                "training_cpu_seconds": result["training_cpu"],
            }
            if recipe["architecture"] != "native":
                continue
            before = value_digest(model.state_dict())
            rows = []
            for split in ("train", "fit"):
                for example in data[split][:8]:
                    original = trace(model, example)
                    repeated = trace(model, example)
                    assert all(torch.equal(original[k], repeated[k]) for k in original)
                    cut = copy.deepcopy(example)
                    cut["public"]["observations"]["rgb"].zero_()
                    rows.append(
                        {
                            "id": example["id"],
                            "split": split,
                            "rgb_zero": comparison(original, trace(model, cut)),
                            "different_entity_query": comparison(
                                original, trace(model, paired_example(example))
                            ),
                        }
                    )
            summaries = {}
            for intervention in ("rgb_zero", "different_entity_query"):
                summaries[intervention] = {
                    stage: {
                        metric: sum(row[intervention][stage][metric] for row in rows) / len(rows)
                        for metric in ("delta_rms", "reference_rms", "max_abs")
                    }
                    for stage in rows[0][intervention]
                }
            assert before == value_digest(model.state_dict())
            diagnostics[arm] = {"samples": len(rows), "mean_sensitivity": summaries, "rows": rows}
    assert sum(a["updates"] for a in audits.values()) == results["optimizer_steps"] == 282
    for architecture in ("native", "transformer"):
        assert (
            audits[f"{architecture}-token"]["initial_parameter_digest"]
            == audits[f"{architecture}-conditional"]["initial_parameter_digest"]
        )
    assert results["accounted_cpu_seconds"] <= manifest["cpu_cap"]
    assert results["wall_seconds"] <= manifest["wall_cap"]
    report = {
        "schema": "taiji-conditional-A-closeout-v1",
        "verdict": "intervention_not_supported_no_A_parent",
        "checkpoint_audit": audits,
        "native_sensitivity": diagnostics,
        "scope": "Post-hoc forward sensitivity on train8 and already-read development-fit8. Sensitivity magnitude does not establish semantic information loss or task success. Claim filenames and final checkpoint contents checked; original runner did per-step readback.",
        "evidence_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (manifest_path, result_path, Path(__file__))
        },
        "training_optimizer_steps": results["optimizer_steps"],
        "audit_optimizer_steps": blocked.calls,
        "intervention_accounted_cpu_seconds": results["accounted_cpu_seconds"],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
            "torch": torch.__version__,
        },
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
    }
    guard()
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cpu_seconds": report["cpu_seconds"],
                "sensitivity": {
                    arm: value["mean_sensitivity"] for arm, value in diagnostics.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    audit()
