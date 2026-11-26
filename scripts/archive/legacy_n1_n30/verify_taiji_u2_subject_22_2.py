"""Bounded U2 22.2 reference graph, ownership and engineering-update audit.

Preflight uses synthetic forward/backward only. The explicitly requested
engineering mode has one durable, single-writer cursor and at most two Adam
updates across attempts. No project dataset or existing model is loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

WALL_STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from taiji import u2_operators
from taiji.u2_operators import fp1
from taiji.u2_subject import U2Config, U2Subject, run_reference_sequence, synthetic_fixture

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_2_contract_20261110.json"
WORK = ROOT / ".seed_test_tmp/u2_22_2_engineering_20261110"


def digest(path):
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    return copy.deepcopy(value)


def trees_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if is_dataclass(left):
        return type(left) is type(right) and trees_equal(asdict(left), asdict(right))
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and set(left) == set(right)
            and all(trees_equal(item, right[key]) for key, item in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(trees_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def atomic_save(payload, path):
    temporary = path.with_suffix(".writing")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def gradient_snapshot(model, losses):
    named = list(model.named_parameters())
    parameters = [parameter for _, parameter in named]
    owners = model.parameter_owners()
    assert set(owners) == {name for name, _ in named}, "Incomplete parameter owner registry"
    assert losses["total"].ndim == 0 and bool(torch.isfinite(losses["total"]))
    total = torch.autograd.grad(losses["total"], parameters, retain_graph=True, allow_unused=True)
    components = {}
    for loss_name, loss in losses.items():
        if loss_name == "total":
            continue
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError(f"Invalid loss component {loss_name}")
        if loss.requires_grad:
            components[loss_name] = torch.autograd.grad(
                loss, parameters, retain_graph=True, allow_unused=True
            )
    rows = {}
    for index, ((name, parameter), gradient) in enumerate(zip(named, total, strict=True)):
        component_norms = {
            key: None if values[index] is None else float(values[index].detach().norm())
            for key, values in components.items()
        }
        rows[name] = {
            "shape": list(parameter.shape),
            "elements": parameter.numel(),
            "declared_loss_owners": owners[name],
            "gradient_present": gradient is not None,
            "finite": gradient is not None and bool(torch.isfinite(gradient).all()),
            "norm": None if gradient is None else float(gradient.detach().norm()),
            "max_component": None if gradient is None else float(gradient.detach().abs().max()),
            "component_norms": component_norms,
        }
    return rows, {name: gradient for (name, _), gradient in zip(named, total, strict=True)}


def parameter_values(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def parameter_changes(before, model):
    return {
        name: {
            "norm": float((parameter.detach() - before[name]).norm()),
            "max_component": float((parameter.detach() - before[name]).abs().max()),
            "changed_elements": int((parameter.detach() != before[name]).sum()),
        }
        for name, parameter in model.named_parameters()
    }


def read_slice_credit(model, gradients):
    d = model.c.d
    result = {}
    for name in ("native_w", "native_r"):
        gradient = gradients[name]
        result[name] = (
            None if gradient is None else float(gradient[..., 5 * d : 6 * d].detach().norm())
        )
    return result


def newborn_slice_credit(gradients):
    return {
        "incoming": (
            None
            if gradients["native_r"] is None
            else float(gradients["native_r"][:, -1:].detach().norm())
        ),
        "outgoing": (
            None
            if gradients["native_p"] is None
            else float(gradients["native_p"][..., -1:].detach().norm())
        ),
    }


def source_location(symbol):
    lines, start = inspect.getsourcelines(symbol)
    return {
        "path": Path(inspect.getfile(symbol)).relative_to(ROOT).as_posix(),
        "symbol": symbol.__qualname__,
        "line": start,
        "last_line": start + len(lines) - 1,
    }


def prepare_small_subject():
    model = U2Subject(U2Config.small()).to(dtype=torch.float64)
    fixture = synthetic_fixture(model.c, torch.float64)
    losses, state, trace, credit = run_reference_sequence(model, fixture)
    task_credit = model.collect_birth_credit(losses["task"], trace, branch=0)["credit"]
    parent_loss = float(losses["total"].detach())
    parent_parameters = parameter_values(model)
    model.birth_units(task_credit.detach(), count=1, branch=0, birth_id="engineering-generation-1")
    born_losses, born_state, born_trace, born_credit = run_reference_sequence(model, fixture)
    tolerance = 1e-10
    assert abs(float(born_losses["total"].detach()) - parent_loss) <= tolerance
    for name, value in parent_parameters.items():
        if name == "native_r":
            assert torch.equal(value, model.native_r[:, : value.shape[1]])
        elif name == "native_p":
            assert torch.equal(value, model.native_p[..., : value.shape[-1]])
        else:
            assert torch.equal(value, dict(model.named_parameters())[name])
    assert model.c.units == model.native_r.shape[1] == model.native_p.shape[-1]
    return (
        model,
        fixture,
        {
            "generation": model.generation,
            "neutral_loss_absolute_difference": abs(
                float(born_losses["total"].detach()) - parent_loss
            ),
            "units": model.c.units,
            "parent_task_credit_norm": float(task_credit.detach().norm()),
        },
    )


def verify_bundle_round_trip(model, state, optimizer):
    bundle = model.export_bundle(state, optimizer.state_dict())
    path = WORK / "before_updates.pt"
    atomic_save(bundle, path)
    restored_bundle = torch.load(path, map_location="cpu", weights_only=False)
    restored_model, restored_state = U2Subject.from_bundle(restored_bundle)
    assert trees_equal(model.state_dict(), restored_model.state_dict())
    assert trees_equal(bundle["state"], restored_state)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3, weight_decay=0)
    restored_optimizer.load_state_dict(restored_bundle["optimizer"])
    assert trees_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    return {
        "parameters_equal": True,
        "state_equal": True,
        "optimizer_equal": True,
        "file_bytes": path.stat().st_size,
    }


def second_generation_check(model, fixture, optimizer):
    bundle = model.export_bundle(model.initial("growth-copy"), optimizer.state_dict())
    child, child_state = U2Subject.from_bundle(bundle)
    child_optimizer = torch.optim.Adam(child.parameters(), lr=1e-3, weight_decay=0)
    child_optimizer.load_state_dict(optimizer.state_dict())
    old_structure = clone_tree(child.structure())
    old_r = child.native_r.detach().clone()
    old_p = child.native_p.detach().clone()
    old_r_moments = clone_tree(child_optimizer.state[child.native_r])
    old_p_moments = clone_tree(child_optimizer.state[child.native_p])
    losses, _, trace, _ = run_reference_sequence(child, fixture)
    before = float(losses["total"].detach())
    task_credit = child.collect_birth_credit(losses["task"], trace, branch=1)["credit"]
    child.birth_units(
        task_credit.detach(),
        count=1,
        branch=1,
        optimizer=child_optimizer,
        birth_id="engineering-generation-2",
    )
    after_losses, _, _, _ = run_reference_sequence(child, fixture)
    assert abs(float(after_losses["total"].detach()) - before) <= 1e-10
    assert torch.equal(old_r, child.native_r[:, : old_r.shape[1]])
    assert torch.equal(old_p, child.native_p[..., : old_p.shape[-1]])
    structure = child.structure()
    for index in range(child.c.groups):
        assert structure["unit_ids"][index][:-1] == old_structure["unit_ids"][index]
        assert structure["unit_branches"][index][:-1] == old_structure["unit_branches"][index]
    for current, previous, axis in (
        (child.native_r, old_r_moments, 1),
        (child.native_p, old_p_moments, 2),
    ):
        for key in ("exp_avg", "exp_avg_sq"):
            actual = child_optimizer.state[current][key]
            prefix = [slice(None)] * actual.ndim
            prefix[axis] = slice(0, previous[key].shape[axis])
            suffix = [slice(None)] * actual.ndim
            suffix[axis] = slice(previous[key].shape[axis], None)
            assert torch.equal(actual[tuple(prefix)], previous[key])
            assert not bool(actual[tuple(suffix)].any())
    assert child.c.units == child.native_r.shape[1] == child.native_p.shape[-1]
    return {
        "generations": child.generation,
        "units": child.c.units,
        "old_lineage_preserved": True,
        "old_moments_preserved": True,
        "new_moments_zero": True,
        "neutral_loss_absolute_difference": abs(float(after_losses["total"].detach()) - before),
    }


def generated_utterance_check(model, state):
    utterance = model.begin_utterance(state)
    goal = model.goal(state["goal_raw"])
    action = state["prediction_action"]
    before = {key: clone_tree(state[key]) for key in ("F", "p", "v", "s", "logw", "evidence")}
    output_bytes = []
    with torch.no_grad():
        for _ in range(model.c.max_output_bytes + 1):
            distribution, _, output = model.output(utterance, goal, action)
            symbol = int(distribution.argmax())
            utterance = model.accept_byte(utterance, symbol, output)
            if symbol == 256:
                break
            output_bytes.append(symbol)
        else:
            raise AssertionError("Expression exceeded its declared byte budget")
    raw = bytes(output_bytes)
    raw.decode("utf-8")
    assert not utterance["utterance_active"]
    assert all(trees_equal(utterance[key], value) for key, value in before.items())
    return {
        "bytes": len(raw),
        "byte_limit": model.c.max_output_bytes,
        "valid_utf8": True,
        "EOS_reached": True,
        "world_and_evidence_unchanged": True,
        "quality_evaluated": False,
    }


def engineering_run(contract, *, execute):
    if not WORK.resolve().is_relative_to(ROOT.resolve() / ".seed_test_tmp"):
        raise ValueError("Engineering work directory resolved outside its workspace")
    WORK.mkdir(parents=True, exist_ok=True)
    if WORK.is_symlink():
        raise ValueError("Engineering work directory must not be a link")
    if execute and (WORK / "completed.json").exists():
        raise ValueError("The two-update engineering window has already been closed")
    lock = WORK / "writer.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        model, fixture, first_birth = prepare_small_subject()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=0)
        cursor_path = WORK / "engineering_cursor.pt"
        cursor = None
        if cursor_path.exists():
            cursor = torch.load(cursor_path, map_location="cpu", weights_only=False)
            if cursor["contract_sha256"] != digest(CONTRACT):
                raise ValueError("Engineering cursor belongs to another frozen contract")
            if cursor["attempted"] != cursor["completed"]:
                raise ValueError("Uncertain update attempt cannot be automatically replayed")
            model, _ = U2Subject.from_bundle(cursor["bundle"])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8, weight_decay=0)
            optimizer.load_state_dict(cursor["bundle"]["optimizer"])
        losses, state, trace, credit = run_reference_sequence(model, fixture)
        assert state["pending"] is None
        bundle_check = verify_bundle_round_trip(model, state, optimizer)
        history = [] if cursor is None else cursor["history"]
        completed = 0 if cursor is None else cursor["completed"]
        snapshot, gradients = gradient_snapshot(model, losses)
        isolated_value = torch.autograd.grad(
            trace["read_R_injection_loss"],
            model.record_value.weight,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if not history:
            history.append(
                {
                    "stage": 0,
                    "parameters": snapshot,
                    "read_block_gradients": read_slice_credit(model, gradients),
                    "newborn_gradient": newborn_slice_credit(gradients),
                    "isolated_record_value_credit": (
                        None if isolated_value is None else float(isolated_value.detach().norm())
                    ),
                    "losses": {key: float(value.detach()) for key, value in losses.items()},
                }
            )
        if execute:
            while completed < 2:
                if state["pending"] is not None:
                    raise ValueError("Parameter update while a result is pending")
                before = parameter_values(model)
                payload = {
                    "contract_sha256": digest(CONTRACT),
                    "attempted": completed + 1,
                    "completed": completed,
                    "bundle": model.export_bundle(state, optimizer.state_dict()),
                    "history": history,
                }
                atomic_save(payload, cursor_path)
                optimizer.zero_grad(set_to_none=True)
                for name, parameter in model.named_parameters():
                    gradient = gradients[name]
                    parameter.grad = None if gradient is None else gradient.detach().clone()
                optimizer.step()
                model.mark_parameter_update()
                completed += 1
                losses, state, trace, credit = run_reference_sequence(model, fixture)
                state["resources"]["updates"] = completed
                snapshot, gradients = gradient_snapshot(model, losses)
                isolated_value = torch.autograd.grad(
                    trace["read_R_injection_loss"],
                    model.record_value.weight,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                history.append(
                    {
                        "stage": completed,
                        "parameters": snapshot,
                        "actual_changes": parameter_changes(before, model),
                        "read_block_gradients": read_slice_credit(model, gradients),
                        "newborn_gradient": newborn_slice_credit(gradients),
                        "newborn_update": {
                            "incoming": float(
                                (
                                    model.native_r[:, -1:].detach() - before["native_r"][:, -1:]
                                ).norm()
                            ),
                            "outgoing": float(
                                (
                                    model.native_p[..., -1:].detach() - before["native_p"][..., -1:]
                                ).norm()
                            ),
                        },
                        "isolated_record_value_credit": (
                            None
                            if isolated_value is None
                            else float(isolated_value.detach().norm())
                        ),
                        "losses": {key: float(value.detach()) for key, value in losses.items()},
                    }
                )
                atomic_save(
                    {
                        "contract_sha256": digest(CONTRACT),
                        "attempted": completed,
                        "completed": completed,
                        "bundle": model.export_bundle(state, optimizer.state_dict()),
                        "history": history,
                    },
                    cursor_path,
                )
        names = list(dict(model.named_parameters()))
        ledger = {}
        for name in names:
            values = [row["parameters"][name] for row in history]
            changes = [row["actual_changes"][name] for row in history if "actual_changes" in row]
            active = any(row["norm"] is not None and row["norm"] > 0 for row in values)
            changed = any(row["changed_elements"] > 0 for row in changes)
            ledger[name] = {
                "stages": values,
                "updates": changes,
                "active": active,
                "actual_update": changed,
                "small_signal": active and max(row["max_component"] or 0 for row in values) < 1e-7,
            }
        if execute:
            assert completed == 2
            failures = [
                name
                for name, row in ledger.items()
                if not row["active"]
                or not row["actual_update"]
                or not all(stage["finite"] for stage in row["stages"])
            ]
            assert not failures, {"unresolved_parameters": failures}
            assert history[0]["read_block_gradients"] == {"native_w": 0.0, "native_r": 0.0}
            assert all(
                value is not None and value > 0
                for value in history[1]["read_block_gradients"].values()
            )
            assert history[0]["isolated_record_value_credit"] in (None, 0.0)
            assert history[1]["isolated_record_value_credit"] > 0
            assert history[0]["newborn_gradient"]["incoming"] > 0
            assert history[0]["newborn_gradient"]["outgoing"] == 0
            assert history[1]["newborn_update"]["incoming"] > 0
            assert history[1]["newborn_update"]["outgoing"] == 0
            assert history[2]["newborn_update"]["outgoing"] > 0
            second_birth = second_generation_check(model, fixture, optimizer)
            critical_names = (
                "f_key.weight",
                "f_query.weight",
                "f_value.first.weight",
                "f_chi.first.weight",
                "record_query.weight",
                "record_key.weight",
                "record_position_key.weight",
                "byte_head.weight",
            )
            task_credit = {
                name: history[-1]["parameters"][name]["component_norms"]["task"]
                for name in critical_names
            }
            assert all(
                value is not None and value > 0 for value in task_credit.values()
            ), task_credit
            lesions = {}
            baseline_task = float(losses["task"].detach())
            for cut in ("fast", "record_injection"):
                cut_losses, _, _, _ = run_reference_sequence(model, fixture, cuts=(cut,))
                lesions[cut] = {
                    "task_nll_difference": float(cut_losses["task"].detach()) - baseline_task,
                    "scope": "same-weights full-trajectory intervention; no task benefit claim",
                }
            utterance = generated_utterance_check(model, state)
        else:
            second_birth = {"status": "deferred_until_two_engineering_updates"}
            task_credit, lesions, utterance = {}, {}, {}
        return {
            "optimizer_steps": completed,
            "first_birth": first_birth,
            "second_birth": second_birth,
            "task_credit_to_responsible_parameters": task_credit,
            "interventions": lesions,
            "free_utterance": utterance,
            "bundle_before_update": bundle_check,
            "parameters": ledger,
            "history_summary": [
                {
                    "stage": row["stage"],
                    "read_block_gradients": row["read_block_gradients"],
                    "newborn_gradient": row["newborn_gradient"],
                    "newborn_update": row.get("newborn_update"),
                    "isolated_record_value_credit": row["isolated_record_value_credit"],
                    "losses": row["losses"],
                }
                for row in history
            ],
            "state_fields": sorted(state),
            "trace": clone_tree(trace),
            "config": asdict(model.c),
        }
    finally:
        lock.unlink()


def shape_and_gradient_check(config, dtype):
    model = U2Subject(config).to(dtype=dtype)
    fixture = synthetic_fixture(config, dtype)
    losses, state, trace, credit = run_reference_sequence(model, fixture)
    parameters, gradients = gradient_snapshot(model, losses)
    invalid = [
        name for name, row in parameters.items() if not row["gradient_present"] or not row["finite"]
    ]
    assert not invalid, invalid
    assert list(state["H"].shape) == [config.branches, config.groups, config.d]
    assert list(state["F"].shape) == [config.branches, config.key, config.key]
    return {
        "config": asdict(config),
        "dtype": str(dtype),
        "H": list(state["H"].shape),
        "F": list(state["F"].shape),
        "parameters": parameters,
        "losses": {key: float(value.detach()) for key, value in losses.items()},
    }


def json_safe(value):
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            scalar = float(detached)
            return scalar if torch.isfinite(detached) else str(scalar)
        finite = detached[torch.isfinite(detached)]
        return {
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "finite_elements": finite.numel(),
            "elements": detached.numel(),
            "finite_norm": float(finite.to(torch.float64).norm()),
        }
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if is_dataclass(value):
        return {field.name: json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, bytes):
        return {"raw_bytes_hex": value.hex()}
    return value


def operator_oracle_check():
    path = ROOT / "scripts/training/verify_taiji_u2_math_22_1.py"
    spec = importlib.util.spec_from_file_location("u2_22_2_independent_math_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    for name in (
        "fp1",
        "record_inputs",
        "initial_cursor",
        "emission",
        "accept",
        "revoke_records",
        "utf8_mask",
    ):
        setattr(oracle, name, getattr(u2_operators, name))
    groups = oracle.run_checks()
    assert len(groups) == 13 and all(row["passed"] for row in groups)
    return {"groups": len(groups), "scope": "promoted operators under frozen independent fixtures"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-updates", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    target = args.report.resolve() if args.report else None
    if target is not None and (
        target.parent != ROOT / "reports" or target.exists() or not args.execute_updates
    ):
        parser.error("A formal report needs engineering mode and a new file inside reports/.")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for path, pin in contract["source_pins"].items():
        if digest(ROOT / path) != pin:
            raise RuntimeError(f"Frozen source changed: {path}")
    torch.set_num_threads(1)
    operators = operator_oracle_check()
    small = engineering_run(contract, execute=args.execute_updates)
    small32 = shape_and_gradient_check(U2Config.small(), torch.float32)
    nominal = shape_and_gradient_check(U2Config.nominal(), torch.float32)
    assert sorted(small["state_fields"]) == sorted(contract["state_fields"])
    assert sorted(small["parameters"]) == sorted(contract["parameter_names"])
    cpu, wall = time.process_time(), time.monotonic() - WALL_STARTED
    assert cpu <= contract["budget"]["cpu_seconds"] and wall <= contract["budget"]["wall_seconds"]
    report = {
        "schema": "taiji-u2-22.2-reference-graph-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "contract_sha256": digest(CONTRACT),
        "verdict": (
            "passed_reference_graph_only" if args.execute_updates else "preflight_no_updates"
        ),
        "small_float64": small,
        "operators": operators,
        "small_float32": small32,
        "nominal_float32": nominal,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "optimizer_steps": small["optimizer_steps"],
        "training_steps": 0,
        "project_dataset_reads": 0,
        "existing_checkpoints_loaded": 0,
        "behavior_claim": False,
        "independent_process_restore_verified": False,
        "coverage": contract["coverage"],
        "code_locations": {
            "subject": source_location(U2Subject),
            "FP1": source_location(fp1),
            "sequence": source_location(run_reference_sequence),
        },
    }
    if target is not None:
        serialized = json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        closed = {
            "contract_sha256": digest(CONTRACT),
            "optimizer_steps": small["optimizer_steps"],
            "report": target.relative_to(ROOT).as_posix(),
            "report_sha256": digest(target),
            "engineering_weights_discarded": True,
        }
        with (WORK / "completed.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(closed, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        for name in ("before_updates.pt", "engineering_cursor.pt"):
            (WORK / name).unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "parameter_tensors": len(small["parameters"]),
                "optimizer_steps": small["optimizer_steps"],
                "cpu_seconds": cpu,
                "wall_seconds": wall,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
