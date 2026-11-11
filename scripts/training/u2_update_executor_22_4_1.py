"""The twelve frozen checks and one charged step of the 22.4.1 u1 update.

Every check maps to a row of the authorized run.json ``run_contract`` and fails
with that row's exit code. No check may be skipped, merged or renumbered.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from scripts.training.u2_attempt_ledger_22_3_4_3 import (
    LedgerError,
)
from scripts.training.u2_update_support_22_3_4_3 import (
    ReadbackError,
    distributions,
    readback,
)
from scripts.training.u2_update_support_22_4_1 import (
    ADAM_FLOAT32_TOLERANCE,
    DEFAULT_ARM,
    BaselineRuntime,
    U1Runtime,
    active_parameters,
    activity_receipt,
    baseline_fingerprint,
    baseline_outputs,
    baseline_run_conditions,
    fingerprint,
    fork_candidate,
    gradients_for,
    normalized_sha256,
    output_displacement,
    readback_baseline,
    save_baseline,
    shadow_terms,
)
from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_recovery import run_conditions

ROOT = Path(__file__).resolve().parents[2]
RUN_SCHEMA = "taiji-u2-22.4.1-run-v1"
PARENT_MANIFEST = ROOT / "plans/manifests/taiji_u2_22_3_accepted_parent_20261111.json"
# 冻结的损失口径。非默认臂不能只跟自己那份 manifest 比对（那是同义反复），
# 而要按它声明的 profile 名对照这张表：改权重但留着 profile 名就会被 R08 拦下。
FROZEN_WEIGHT_PROFILES = {
    "F1-style": {"predictive": 1.0, "actor": 0.1, "critic": 0.1, "verbatim": 0.0},
    "F2-style": {"predictive": 0.1, "actor": 0.1, "critic": 0.1, "verbatim": 1.0},
}


class ContractError(LedgerError):
    """Identity, authorization, manifest or data-scope violation (exit 5)."""

    exit_code = 5


class BudgetError(LedgerError):
    """Attempt, quota or ledger-accounting violation (exit 3)."""

    exit_code = 3


class ActivityGateError(LedgerError):
    """A declared active parameter failed the frozen activity floor (exit 6)."""

    exit_code = 6


class CheckpointMismatch(ReadbackError, LedgerError):
    """A parent or post checkpoint did not read back bitwise (exit 4)."""

    exit_code = 4


def readback_or_exit(path, expected):
    try:
        return readback(path, expected)
    except ReadbackError as error:
        raise CheckpointMismatch(str(error)) from error


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path: Path, payload: dict[str, Any]) -> None:
    from scripts.training.check_report_run_conditions import write_report

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    staging = path.with_name(path.name + ".writing")
    write_report(staging, payload)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(staging.read_text(encoding="utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    staging.unlink()


def execution_manifest(run: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    path = ROOT / run["references"]["execution_manifest"]
    return load_json(path), path


def admission_manifest(run: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    path = ROOT / run["references"]["admission_manifest"]
    return load_json(path), path


def check_r01(
    run: dict[str, Any], root_name: str, execution: dict[str, Any] | None = None
) -> dict[str, Any]:
    required = (
        "schema",
        "run_id",
        "update_id",
        "attempt_id",
        "attempts_authorized",
        "optimizer_steps_authorized",
        "precision",
        "cpu_cap",
        "wall_cap",
        "owner_approval_ref",
        "references",
        "data_scope",
        "paths",
        "run_contract",
        "exit_codes",
    )
    missing = [key for key in required if key not in run]
    if missing:
        raise ContractError(f"R01 run.json is missing fields: {missing}")
    if run["schema"] != RUN_SCHEMA:
        raise ContractError(f"R01 unexpected run schema: {run['schema']}")
    identity = run["update_id"]
    # 身份不再对常量：update_id 必须与 attempt_id 自洽，且是该 run 自己引用的
    # 执行 manifest 里真实存在的臂。u1 仍只认 u1-candidate-float32（v1 只有这一个臂）。
    if not isinstance(identity, str) or not identity or run["attempt_id"] != identity:
        raise ContractError(f"R01 attempt_id does not bind update_id {identity!r}")
    if execution is None:
        execution = load_json(ROOT / run["references"]["execution_manifest"])
    if identity not in execution.get("arms", {}):
        raise ContractError(f"R01 update_id {identity} is not an arm of the execution manifest")
    if run["run_id"] != root_name.removeprefix("taiji_u2_22_4_1_"):
        raise ContractError("R01 run_id does not match the run directory")
    contract = run["run_contract"]
    expected = [f"R{index:02d}" for index in range(1, 13)]
    if [row.get("id") for row in contract] != expected:
        raise ContractError("R01 run_contract is not the frozen R01..R12 list")
    codes = {row["id"]: row["fail_exit"] for row in contract}
    frozen = {
        "R01": 5,
        "R02": 5,
        "R03": 5,
        "R04": 5,
        "R05": 5,
        "R06": 3,
        "R07": 5,
        "R08": 5,
        "R09": 4,
        "R10": 3,
        "R11": 4,
        "R12": 6,
    }
    if codes != frozen:
        raise ContractError(f"R01 exit-code table changed: {codes}")
    return {"id": "R01", "run_id": run["run_id"], "checks": len(contract)}


def check_r02(run: dict[str, Any]) -> dict[str, Any]:
    reference = run.get("owner_approval_ref")
    if run.get("authorized") is not True:
        raise ContractError("R02 authorized is not true")
    if not isinstance(reference, str) or not reference.strip():
        raise ContractError("R02 owner_approval_ref is empty")
    identity = run["update_id"]
    if identity not in reference:
        raise ContractError(f"R02 owner_approval_ref does not bind the update id {identity}")
    return {"id": "R02", "owner_approval_ref": reference}


def check_r03(run: dict[str, Any], manifest_path: Path | None = None) -> dict[str, Any]:
    path = manifest_path or ROOT / run["references"]["execution_manifest"]
    actual = normalized_sha256(path)
    expected = run["references"]["manifest_sha256"]
    if actual != expected:
        raise ContractError(f"R03 execution manifest hash mismatch: {actual} != {expected}")
    return {"id": "R03", "path": str(Path(path).relative_to(ROOT).as_posix()), "sha256": actual}


def check_r04(run: dict[str, Any], manifest_path: Path | None = None) -> dict[str, Any]:
    path = manifest_path or ROOT / run["references"]["admission_manifest"]
    actual = normalized_sha256(path)
    expected = run["references"]["admission_manifest_sha256"]
    if actual != expected:
        raise ContractError(f"R04 admission manifest hash mismatch: {actual} != {expected}")
    return {"id": "R04", "path": str(Path(path).relative_to(ROOT).as_posix()), "sha256": actual}


def check_r05(
    run: dict[str, Any],
    execution: dict[str, Any],
    base_parameter_digest: str,
    fork_evidence: dict[str, Any],
) -> dict[str, Any]:
    parent = load_json(PARENT_MANIFEST)
    accepted = parent.get("accepted_parent", {})
    if parent.get("status") != "accepted_engineering_initialization":
        raise ContractError("R05 accepted parent is not the 22.3 engineering initialization")
    if accepted.get("profile") != "U2Config.small" or accepted.get("seed") != 20261110:
        raise ContractError("R05 accepted parent profile/seed changed")
    if "float32" not in accepted.get("precisions", []) or accepted.get("device") != "cpu":
        raise ContractError("R05 accepted parent does not cover the float32 CPU arm")
    if accepted.get("parameter_tensors") != 191:
        raise ContractError("R05 accepted parent parameter count changed")
    frozen_parent = accepted.get("initial_parameter_sha256", {}).get("float32")
    if frozen_parent != base_parameter_digest:
        raise ContractError(
            f"R05 rebuilt base parent differs from the accepted identity: "
            f"{base_parameter_digest} != {frozen_parent}"
        )
    receipt_path = ROOT / execution["authority"]["interface_receipt"]
    if normalized_sha256(receipt_path) != execution["authority"]["interface_receipt_sha256"]:
        raise ContractError("R05 interface receipt hash changed")
    receipt = load_json(receipt_path)
    if receipt.get("verdict") != "passed_zero_update_metadata_groups_interface":
        raise ContractError("R05 interface receipt is not a pass")
    if receipt.get("contract_sha256") != execution["authority"]["interface_contract_sha256"]:
        raise ContractError("R05 interface receipt binds another contract")
    recorded = {row["id"]: row for row in receipt.get("checks", []) if row.get("passed") is True}
    if "metadata_fork_inherits_parent" not in recorded:
        raise ContractError("R05 interface receipt lacks the metadata fork check")
    frozen_fork = recorded["metadata_fork_inherits_parent"]["detail"]["precisions"]["float32"]
    if fork_evidence["summary"] != frozen_fork:
        raise ContractError(
            "R05 rebuilt metadata fork differs from the interface receipt: "
            f"{fork_evidence['summary']} != {frozen_fork}"
        )
    # R05 只要求两件事：磁盘源码等于执行 manifest 冻结的 pin；接口回执通过。
    # 回执覆盖的 pin 集合可以小于 manifest（本轮回执只覆盖 5 个核心文件与复算脚本，
    # 未覆盖 taiji/u2_task_contract.py）；凡是回执确实记了的项，必须与 manifest 一致，
    # 覆盖范围原样记录在返回值里，供 S12 审计，不做静默省略。
    pins = execution["authority"]["source_pins"]
    recorded_pins = receipt.get("source_pins", {})
    for name, expected in pins.items():
        actual = normalized_sha256(ROOT / name)
        if actual != expected:
            raise ContractError(f"R05 frozen source changed: {name}")
        if name in recorded_pins and recorded_pins[name] != expected:
            raise ContractError(f"R05 interface receipt pin differs: {name}")
    return {
        "id": "R05",
        "accepted_parent": str(PARENT_MANIFEST.relative_to(ROOT).as_posix()),
        "interface_receipt": str(receipt_path.relative_to(ROOT).as_posix()),
        "fork": fork_evidence,
        "source_pins": pins,
        "receipt_pin_coverage": {
            "recorded": sorted(recorded_pins),
            "manifest_only": sorted(set(pins) - set(recorded_pins)),
            "receipt_only": sorted(set(recorded_pins) - set(pins)),
        },
    }


def check_r06(run: dict[str, Any], admission: dict[str, Any], consumed: int) -> dict[str, Any]:
    identity = run["update_id"]
    updates = {row["id"]: row for row in admission["updates"]}
    if identity not in updates:
        raise ContractError(f"R06 the admission manifest does not list {identity}")
    entry = updates[identity]
    if admission["quota"]["attempts_consumed"] != 0:
        raise BudgetError("R06 frozen admission already records consumed attempts")
    if consumed != 0:
        raise BudgetError("R06 the durable ledger already holds an attempt")
    if run["attempts_authorized"] != 1:
        raise BudgetError("R06 this run authorizes more than one attempt")
    if run["optimizer_steps_authorized"] != 1 or entry["optimizer_steps"] != 1:
        raise BudgetError("R06 this run is not a single optimizer step")
    if run["cpu_cap"] > entry["cpu_cap"] or run["cpu_cap"] > 120:
        raise BudgetError("R06 cpu cap exceeds the frozen price")
    if run["wall_cap"] > entry["wall_cap"] or run["wall_cap"] > 240:
        raise BudgetError("R06 wall cap exceeds the frozen price")
    if run["precision"] != entry["precision"]:
        raise BudgetError("R06 precision differs from the admission entry")
    return {
        "id": "R06",
        "attempts_authorized": 1,
        "optimizer_steps": 1,
        "cpu_cap": run["cpu_cap"],
        "wall_cap": run["wall_cap"],
    }


def check_r07(run: dict[str, Any], recipe_sha: str, slice_sha: str) -> dict[str, Any]:
    scope = run["data_scope"]
    for key in (
        "reads_project_corpus",
        "reads_confirm_partition",
        "reads_development_check",
        "reads_final",
        "touches_sealed_set",
        "contains_task1_batch",
    ):
        if scope.get(key) is not False:
            raise ContractError(f"R07 data scope flag is not false: {key}")
    if scope.get("allowed_partitions") != ["engineering_slice"]:
        raise ContractError("R07 the only allowed partition is engineering_slice")
    if not scope.get("data_recipe_sha256") or not scope.get("slice_sha256"):
        raise ContractError("R07 data digests were not frozen before the ledger opened")
    if scope["data_recipe_sha256"] != recipe_sha:
        raise ContractError("R07 data_recipe_sha256 does not match the generated slice")
    if scope["slice_sha256"] != slice_sha:
        raise ContractError("R07 slice_sha256 does not match the generated slice")
    return {
        "id": "R07",
        "data_recipe_sha256": recipe_sha,
        "slice_sha256": slice_sha,
        "partitions": ["engineering_slice"],
    }


def check_r08(
    manifest: dict[str, Any], model, optimizer, *, arm: str = DEFAULT_ARM
) -> dict[str, Any]:
    row = manifest["arms"][arm]
    optimizer_fields = ("lr", "betas", "eps", "weight_decay", "amsgrad")
    frozen_optimizer = {key: manifest["optimizer"][key] for key in optimizer_fields}
    actual = {
        "lr": optimizer.defaults["lr"],
        "betas": tuple(optimizer.defaults["betas"]),
        "eps": optimizer.defaults["eps"],
        "weight_decay": optimizer.defaults["weight_decay"],
        "amsgrad": optimizer.defaults["amsgrad"],
    }
    expected = {
        **frozen_optimizer,
        "betas": tuple(frozen_optimizer["betas"]),
    }
    if actual != expected:
        raise ContractError(f"R08 optimizer differs from the freeze: {actual} != {expected}")
    weights = dict(row["weights"])
    if set(weights) != {"predictive", "actor", "critic", "verbatim"}:
        raise ContractError("R08 the loss term list changed")
    if arm == DEFAULT_ARM:
        if weights != {"predictive": 1.0, "actor": 0.1, "critic": 0.1, "verbatim": 0.0}:
            raise ContractError(f"R08 u1 weights are not the frozen F1 profile: {weights}")
    else:
        frozen_profile = FROZEN_WEIGHT_PROFILES.get(row.get("weight_profile"))
        if frozen_profile is None:
            raise ContractError(
                f"R08 {arm} declares an unknown weight profile {row.get('weight_profile')!r}"
            )
        if weights != frozen_profile:
            raise ContractError(
                f"R08 {arm} weights drift from the frozen {row['weight_profile']} profile: {weights}"
            )
    if manifest["warmup"]["enabled"] or manifest["gradient_clip"]["enabled"]:
        raise ContractError("R08 warmup or clipping was enabled outside the freeze")
    names = active_parameters(model, manifest, arm=arm)
    if (
        names != list(row["active_parameters"])
        and getattr(getattr(model, "c", None), "group_units", None) is None
    ):
        raise ContractError(f"R08 active parameter list changed: {names}")
    missing = [name for name in names if name not in dict(model.named_parameters())]
    if missing:
        raise ContractError(f"R08 declared active parameters are absent: {missing}")
    return {
        "id": "R08",
        "optimizer": {key: value for key, value in actual.items()},
        "weights": weights,
        "active_parameters": names,
        "group_extensions": getattr(getattr(model, "c", None), "group_units", None) is not None,
    }


def check_r09(parent_path: Path, runtime: U1Runtime) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = fingerprint(runtime)
    runtime.save(parent_path)
    restored, proof = readback_or_exit(parent_path, expected)
    five = {
        "state_sha256": "state_sha256",
        "parameter_sha256": "parameter_sha256",
        "moment_sha256": "moment_sha256",
        "history_reference": "history_reference",
        "revision": "revision",
    }
    actual = fingerprint(restored)
    mismatched = [key for key in five if actual[key] != expected[key]]
    if mismatched:
        raise CheckpointMismatch(f"R09 parent checkpoint read-back differs: {mismatched}")
    return proof, {"id": "R09", "five_fields": sorted(five), "sha256": proof["sha256"]}


def check_r10(ledger) -> dict[str, Any]:
    state = ledger.read()
    first = state["rows"][0]
    if first.get("event") != "run_start":
        raise BudgetError("R10 the ledger does not start with run_start")
    if first.get("owner_approval_ref") != ledger.run["owner_approval_ref"]:
        raise BudgetError("R10 run_start does not carry the recorded owner reference")
    if first.get("manifest_sha256") != ledger.run["manifest_sha256"]:
        raise BudgetError("R10 run_start does not bind the execution manifest")
    if state["consumed"] != 0 or state["opened"]:
        raise BudgetError("R10 the ledger already consumed an attempt")
    if len(state["rows"]) != 1:
        raise BudgetError("R10 the ledger is not a single durable run_start line")
    return {
        "id": "R10",
        "rows": len(state["rows"]),
        "head": state["head"],
        "chain": "valid",
        "fsync": "run_start appended with os.fsync before any optimization",
    }


def check_r11(post_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    restored, proof = readback_or_exit(post_path, expected)
    actual = fingerprint(restored)
    mismatched = [
        key
        for key in (
            "state_sha256",
            "parameter_sha256",
            "moment_sha256",
            "history_reference",
            "revision",
        )
        if actual[key] != expected[key]
    ]
    if mismatched:
        raise CheckpointMismatch(f"R11 post checkpoint read-back differs: {mismatched}")
    return {"id": "R11", "five_fields": True, "sha256": proof["sha256"]}


def check_r09_baseline(
    parent_path: Path,
    runtime: BaselineRuntime,
    manifest: dict[str, Any],
    *,
    arm: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """R09 for the baseline arm: same five fields, same exit code, same create-once rule."""
    expected = baseline_fingerprint(runtime)
    save_baseline(runtime, parent_path, manifest)
    try:
        restored, proof = readback_baseline(parent_path, expected, manifest, arm=arm)
    except ReadbackError as error:
        raise CheckpointMismatch(str(error)) from error
    five = ("state_sha256", "parameter_sha256", "moment_sha256", "history_reference", "revision")
    actual = baseline_fingerprint(restored)
    mismatched = [key for key in five if actual[key] != expected[key]]
    if mismatched:
        raise CheckpointMismatch(f"R09 parent checkpoint read-back differs: {mismatched}")
    return proof, {"id": "R09", "five_fields": sorted(five), "sha256": proof["sha256"]}


def check_r11_baseline(
    post_path: Path, expected: dict[str, Any], manifest: dict[str, Any], *, arm: str
) -> dict[str, Any]:
    try:
        restored, proof = readback_baseline(post_path, expected, manifest, arm=arm)
    except ReadbackError as error:
        raise CheckpointMismatch(str(error)) from error
    actual = baseline_fingerprint(restored)
    mismatched = [
        key
        for key in (
            "state_sha256",
            "parameter_sha256",
            "moment_sha256",
            "history_reference",
            "revision",
        )
        if actual[key] != expected[key]
    ]
    if mismatched:
        raise CheckpointMismatch(f"R11 post checkpoint read-back differs: {mismatched}")
    return {"id": "R11", "five_fields": True, "sha256": proof["sha256"]}


def perform_baseline(
    runtime: BaselineRuntime,
    ledger,
    counter,
    manifest: dict[str, Any],
    post_path: Path,
    *,
    arm: str,
):
    """One charged step on the frozen baseline: gradients, Adam, receipt, checkpoint.

    该图没有主体 state，也不做参数迁移：单步就是在这张图上原地更新那 35 项活性参数。
    R09 已先把更新前的 parent.bin 落盘，因此拒绝路径仍有一份未被改动的父状态可读。
    """
    started, wall = time.process_time(), time.monotonic()
    before = baseline_fingerprint(runtime)
    before_output = baseline_outputs(runtime, manifest)
    before_state = {
        name: value.detach().clone() for name, value in runtime.model.state_dict().items()
    }
    gradients, losses = gradients_for(
        runtime, manifest, arm=arm, fingerprint_fn=baseline_fingerprint
    )
    named = dict(runtime.model.named_parameters())
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.grad = gradients[name].clone()
    runtime.optimizer.step()
    step_calls = int(counter.current in counter.called)
    if step_calls != 1:
        raise BudgetError("the frozen attempt must call optimizer.step exactly once")
    runtime.revision += 1
    with torch.no_grad():
        for parameter in named.values():
            parameter.grad = None
    after_state = runtime.model.state_dict()
    receipt = activity_receipt(
        runtime.model, before_state, after_state, gradients, manifest, arm=arm
    )
    after_output = baseline_outputs(runtime, manifest)
    with torch.enable_grad():
        post_loss, _ = shadow_terms(runtime.model, manifest, arm=arm)
    post_loss = post_loss.detach()
    receipt["output_displacement_descriptive"] = {
        **output_displacement(before_output, after_output),
        "same_fixture_loss_delta": float(post_loss) - losses["total"],
        "policy_note": after_output["policy_note"],
    }
    expected = baseline_fingerprint(runtime)
    try:
        save_baseline(runtime, post_path, manifest)
    except ReadbackError as error:
        raise CheckpointMismatch(str(error)) from error
    return {
        "status": "accepted",
        "arm": arm,
        "execution_graph": dict(runtime.graph),
        "pre": before,
        "post": expected,
        "gradient_sha256": value_digest(gradients),
        "gradients": gradients,
        "parameter_rows": [
            {
                "parameter": name,
                "gradient_norm": None if gradient is None else float(gradient.norm()),
                "max_parameter_change": float((before_state[name] - after_state[name]).abs().max()),
            }
            for name, gradient in gradients.items()
        ],
        "losses": losses,
        "losses_after": {"total": float(post_loss)},
        "migration": {
            "status": "committed",
            "mechanism": "in-place Adam step on the frozen baseline graph",
            "arm": arm,
            "revision": runtime.revision,
            "state_migration": "not_applicable_baseline_has_no_subject_state",
        },
        "activity_receipt": receipt,
        "next_output": after_output["next_output"],
        "policy": after_output["policy"],
        "before_output": before_output,
        "run_conditions": baseline_run_conditions(runtime.model),
        "optimizer_step_calls": step_calls,
        "cpu_seconds": time.process_time() - started,
        "wall_seconds": time.monotonic() - wall,
        "readbacks": runtime.readbacks,
    }


def rejected_result_baseline(
    runtime: BaselineRuntime,
    error: Exception,
    step_calls: int,
    receipt: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    current = baseline_fingerprint(runtime)
    output = baseline_outputs(runtime, manifest)
    return {
        "status": "rejected",
        "arm": runtime.arm,
        "execution_graph": dict(runtime.graph),
        "pre": current,
        "post": current,
        "gradient_sha256": value_digest(None),
        "gradients": None,
        "parameter_rows": [],
        "losses": None,
        "losses_after": None,
        "migration": {"status": "rejected", "error": f"{type(error).__name__}: {error}"},
        "activity_receipt": receipt,
        "next_output": output["next_output"],
        "policy": output["policy"],
        "before_output": output,
        "run_conditions": baseline_run_conditions(runtime.model),
        "optimizer_step_calls": step_calls,
        "readbacks": list(runtime.readbacks),
        "error": str(error),
    }


def check_r12(receipt: dict[str, Any]) -> dict[str, Any]:
    if not receipt["parameters"]:
        raise ActivityGateError("R12 the activity receipt has no parameter rows")
    return {"id": "R12", **receipt}


def perform(
    runtime: U1Runtime,
    ledger,
    counter,
    manifest: dict[str, Any],
    post_path: Path,
    *,
    arm: str = DEFAULT_ARM,
):
    """One charged step: gradients, frozen Adam step, commit, activity gate, save."""
    from scripts.training.u2_update_executor_22_3_4_3 import admit

    started, wall = time.process_time(), time.monotonic()
    before = fingerprint(runtime)
    before_output = distributions(runtime)
    before_state = {
        name: value.detach().clone() for name, value in runtime.model.state_dict().items()
    }
    active_model = runtime.model
    gradients, losses = gradients_for(runtime, manifest, arm=arm)
    candidate, optimizer = fork_candidate(runtime, gradients, manifest)
    step_calls = int(counter.current in counter.called)
    if step_calls != 1:
        raise BudgetError("the frozen attempt must call optimizer.step exactly once")
    migration = runtime.migrate_parameters(candidate, optimizer=optimizer, acceptance=admit)
    if migration["status"] != "committed":
        raise BudgetError(f"the frozen attempt did not commit: {migration}")
    after_state = runtime.model.state_dict()
    receipt = activity_receipt(
        active_model, before_state, after_state, gradients, manifest, arm=arm
    )
    after_output = distributions(runtime)
    # shadow_terms 内部的资格分由 model.choose_operation 用 torch.autograd.grad 计算，
    # 关掉梯度会让这条路径直接失败（"does not require grad and does not have a grad_fn"）。
    # 因此后置损失必须在允许梯度的上下文里算，只取数值时再 detach。
    with torch.enable_grad():
        post_loss, _ = shadow_terms(runtime.model, manifest, arm=arm)
    post_loss = post_loss.detach()
    receipt["output_displacement_descriptive"] = {
        **output_displacement(before_output, after_output),
        "same_fixture_loss_delta": float(post_loss) - losses["total"],
    }
    expected = fingerprint(runtime)
    try:
        runtime.save(post_path)
    except ReadbackError as error:
        raise CheckpointMismatch(str(error)) from error
    return {
        "status": "accepted",
        "pre": before,
        "post": expected,
        "gradient_sha256": value_digest(gradients),
        "gradients": gradients,
        "parameter_rows": [
            {
                "parameter": name,
                "gradient_norm": None if gradient is None else float(gradient.norm()),
                "max_parameter_change": float((before_state[name] - after_state[name]).abs().max()),
            }
            for name, gradient in gradients.items()
        ],
        "losses": losses,
        "losses_after": {"total": float(post_loss)},
        "migration": migration,
        "activity_receipt": receipt,
        "next_output": after_output["next_output"],
        "policy": after_output["policy"],
        "before_output": before_output,
        "run_conditions": run_conditions(runtime.model),
        "optimizer_step_calls": step_calls,
        "cpu_seconds": time.process_time() - started,
        "wall_seconds": time.monotonic() - wall,
        "readbacks": runtime.readbacks,
    }


def rejected_result(
    runtime: U1Runtime, error: Exception, step_calls: int, receipt: dict[str, Any] | None
) -> dict[str, Any]:
    current = fingerprint(runtime)
    output = distributions(runtime)
    return {
        "status": "rejected",
        "pre": current,
        "post": current,
        "gradient_sha256": value_digest(None),
        "gradients": None,
        "parameter_rows": [],
        "losses": None,
        "losses_after": None,
        "migration": {"status": "rejected", "error": f"{type(error).__name__}: {error}"},
        "activity_receipt": receipt,
        "next_output": output["next_output"],
        "policy": output["policy"],
        "before_output": output,
        "run_conditions": run_conditions(runtime.model),
        "optimizer_step_calls": step_calls,
        "readbacks": list(runtime.readbacks),
        "error": str(error),
    }


def five_field_gate(expected: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    keys = ("state_sha256", "parameter_sha256", "moment_sha256", "history_reference", "revision")
    assert_values_equal(
        {key: expected[key] for key in keys},
        {key: actual[key] for key in keys},
        label,
    )


__all__ = [
    "ADAM_FLOAT32_TOLERANCE",
    "ActivityGateError",
    "BudgetError",
    "CheckpointMismatch",
    "ContractError",
    "check_r01",
    "check_r02",
    "check_r03",
    "check_r04",
    "check_r05",
    "check_r06",
    "check_r07",
    "check_r08",
    "check_r09",
    "check_r09_baseline",
    "check_r10",
    "check_r11",
    "check_r11_baseline",
    "check_r12",
    "execution_manifest",
    "admission_manifest",
    "five_field_gate",
    "load_json",
    "perform",
    "perform_baseline",
    "readback_or_exit",
    "rejected_result",
    "rejected_result_baseline",
    "write_once",
]
