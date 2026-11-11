"""Run or self-check the single owner-authorized u1 22.4.1 engineering attempt."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.training.u2_attempt_ledger_22_3_4_3 import AttemptLedger, LedgerError
from scripts.training.u2_update_executor_22_4_1 import (
    ActivityGateError,
    BudgetError,
    admission_manifest,
    check_r01,
    check_r02,
    check_r03,
    check_r04,
    check_r05,
    check_r06,
    check_r07,
    check_r08,
    check_r09,
    check_r10,
    check_r11,
    check_r12,
    execution_manifest,
    load_json,
    perform,
    readback_or_exit,
    rejected_result,
    write_once,
)
from scripts.training.u2_update_support_22_3_4_3 import (
    StepCounter,
    file_sha,
)
from scripts.training.u2_update_support_22_4_1 import (
    ARM,
    activity_receipt,
    fingerprint,
    prepare_parent,
    recipe_digest,
    slice_digest,
)
from taiji.u2_history import value_digest
from taiji.u2_subject import U2Config, U2Subject

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "plans/manifests/taiji_u2_22_4_1_u1_run_draft_20261111.json"
SELFCHECK = ROOT / "output/taiji_u2_22_4_1_selfcheck_20261111T140217Z/selfcheck.json"


def fork_evidence(dtype: torch.dtype) -> dict[str, Any]:
    base = U2Subject(U2Config.small()).to(dtype=dtype)
    candidate = base.metadata_candidate()
    state = base.state_dict()
    inherited = all(
        torch.equal(value, candidate.state_dict()[name]) for name, value in state.items()
    )
    summary = (
        f"parent={len(list(base.parameters()))},"
        f"candidate={len(list(candidate.parameters()))},"
        f"revision={base.parameter_revision}->{candidate.parameter_revision},"
        f"inherited={inherited}"
    )
    return {"summary": summary, "base_digest": value_digest(state), "inherited": inherited}


def ledger_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "authorized": run["authorized"],
        "owner_approval_ref": run["owner_approval_ref"],
        "manifest_sha256": run["references"]["manifest_sha256"],
    }


def precontract(run: dict[str, Any]) -> dict[str, Any]:
    identity = run["update_id"]
    return {
        "run": {
            "attempt_order": [identity],
            "attempt_parents": {identity: "parent.bin"},
            "attempt_posts": {identity: "post.bin"},
            "budget_max": run["attempts_authorized"],
        }
    }


def check_selfcheck_gate() -> None:
    if not SELFCHECK.exists() or load_json(SELFCHECK).get("verdict") != "pass":
        raise BudgetError(
            f"the negative self-check has not passed yet: {SELFCHECK.relative_to(ROOT).as_posix()}"
        )


def s12_payload(
    run: dict[str, Any],
    execution: dict[str, Any],
    result: dict[str, Any],
    ledger_summary: dict[str, Any],
    identity: str,
    cpu_seconds: float,
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": "taiji-u2-22.4.1-s12-v1",
        "run_id": run["run_id"],
        "attempt_id": identity,
        "update_id": run["update_id"],
        "arm": ARM,
        "precision": run["precision"],
        "status": result["status"],
        "weight_profile": execution["arms"][ARM]["weight_profile"],
        "weights": execution["arms"][ARM]["weights"],
        "ledger_seq_open": ledger_summary["open_seq"],
        "ledger_seq_close": ledger_summary["close_seq"],
        "parent_checkpoint": "parent.bin",
        "parent_sha256": None,
        "post_checkpoint": "post.bin" if result["status"] == "accepted" else None,
        "post_sha256": result.get("post_sha256"),
        "pre_state_sha256": result["pre"]["state_sha256"],
        "pre_parameter_sha256": result["pre"]["parameter_sha256"],
        "pre_moment_sha256": result["pre"]["moment_sha256"],
        "pre_history_reference": result["pre"]["history_reference"],
        "pre_revision": result["pre"]["revision"],
        "post_state_sha256": result["post"]["state_sha256"],
        "post_parameter_sha256": result["post"]["parameter_sha256"],
        "post_moment_sha256": result["post"]["moment_sha256"],
        "history_reference": result["post"]["history_reference"],
        "revision": result["post"]["revision"],
        "gradient_sha256": result["gradient_sha256"],
        "next_output": result["next_output"],
        "policy": result["policy"],
        "run_conditions": result["run_conditions"],
        "parameter_rows": result["parameter_rows"],
        "losses": result["losses"],
        "losses_after": result["losses_after"],
        "optimizer_step_calls": result["optimizer_step_calls"],
        "migration": result["migration"],
        "activity_receipt": result["activity_receipt"],
        "conditions": {
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "device": "cpu",
            "dtype": run["precision"],
        },
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "execution_manifest": run["references"]["execution_manifest"],
        "manifest_sha256": run["references"]["manifest_sha256"],
        "admission_manifest": run["references"]["admission_manifest"],
        "admission_manifest_sha256": run["references"]["admission_manifest_sha256"],
        "owner_approval_ref": run["owner_approval_ref"],
        "run_root": run["paths"]["run_root"],
        "parent_readback": next(
            (proof for proof in result["readbacks"] if Path(proof["path"]).name == "parent.bin"),
            None,
        ),
        "post_readback": next(
            (proof for proof in result["readbacks"] if Path(proof["path"]).name == "post.bin"),
            None,
        ),
        "failed_checkpoint": "post.bin" if result["status"] != "accepted" else None,
        "error": result.get("error"),
    }


def reject_attempt(
    *,
    run: dict[str, Any],
    execution: dict[str, Any],
    runtime: Any,
    ledger: AttemptLedger,
    identity: str,
    post_path: Path,
    parent_path: Path,
    step_calls: int,
    started: float,
    wall: float,
    error: Exception,
    checks: list[dict[str, Any]],
    s12_path: Path | None = None,
) -> dict[str, Any]:
    """open 之后的唯一收尾路径：任何异常都必须落 close(rejected) 与 rejected S12。

    不区分异常种类——账本类、读回类、纯 RuntimeError 一律走这里，避免出现悬空 open。
    """
    receipt = getattr(error, "receipt", None)
    if post_path.exists():
        post_path.unlink()
    result = rejected_result(runtime, error, step_calls, receipt)
    closed = ledger.close(
        identity,
        outcome="rejected",
        error=f"{type(error).__name__}: {error}",
        post_checkpoint=None,
        optimizer_step_calls=step_calls,
    )
    s12 = s12_payload(
        run,
        execution,
        result,
        {"open_seq": 1, "close_seq": closed["seq"]},
        identity,
        time.process_time() - started,
        time.monotonic() - wall,
    )
    s12["parent_sha256"] = normalized_file_sha(parent_path) if parent_path.exists() else None
    destination = s12_path or (ROOT / run["paths"]["s12"])
    if destination.exists():
        destination.unlink()
    write_once(destination, s12)
    code = getattr(error, "exit_code", 3)
    failed_id = {4: "R11", 6: "R12"}.get(code, "R10")
    checks.append({"id": failed_id, "passed": False, "error": f"{type(error).__name__}: {error}"})
    return {
        "status": "rejected",
        "exit_code": code,
        "checks": checks,
        "error": f"{type(error).__name__}: {error}",
        "s12": str(destination),
        "ledger": run["paths"]["ledger"],
    }


def execute(root: Path) -> dict[str, Any]:
    check_selfcheck_gate()
    torch.set_num_threads(1)
    started, wall = time.process_time(), time.monotonic()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_path = root / "run.json"
    if not run_path.exists():
        write_once(run_path, load_json(DRAFT))
    run = load_json(run_path)
    checks: list[dict[str, Any]] = []
    checks.append(check_r01(run, root.name))
    checks.append(check_r02(run))
    checks.append(check_r03(run))
    checks.append(check_r04(run))
    execution, _ = execution_manifest(run)
    admission, _ = admission_manifest(run)
    evidence = fork_evidence(torch.float32)
    checks.append(check_r05(run, execution, evidence["base_digest"], evidence))
    ledger_path = ROOT / run["paths"]["ledger"]
    consumed = 1 if ledger_path.exists() else 0
    checks.append(check_r06(run, admission, consumed))

    runtime = prepare_parent(execution, root)
    checks.append(check_r08(execution, runtime.model, runtime.optimizer))

    recipe_sha = recipe_digest(execution)
    slice_sha = slice_digest(execution, runtime.model)
    scope = run["data_scope"]
    if scope.get("data_recipe_sha256") is None and scope.get("slice_sha256") is None:
        scope["data_recipe_sha256"] = recipe_sha
        scope["slice_sha256"] = slice_sha
        write_once(run_path.with_name("run.json.next"), run)
        os.replace(run_path.with_name("run.json.next"), run_path)
    checks.append(check_r07(run, recipe_sha, slice_sha))

    ledger = AttemptLedger(
        ledger_path,
        ledger_run(run),
        execution["authority"]["source_pins"],
        precontract(run),
    )
    ledger.start()
    checks.append(check_r10(ledger))

    parent_path = root / "parent.bin"
    proof, r09 = check_r09(parent_path, runtime)
    r09["checkpoint"] = proof["sha256"]
    checks.append(r09)

    counter = StepCounter(ledger, root / "claims")
    original_step = torch.optim.Adam.step
    identity = run["update_id"]
    post_path = root / "post.bin"
    ledger.open(identity, parent_checkpoint="parent.bin")
    counter.current = identity
    torch.optim.Adam.step = counter.wrapper(original_step)
    try:
        try:
            result = perform(runtime, ledger, counter, execution, post_path)
        except Exception as error:  # noqa: BLE001 - open 之后任何异常都必须落账，不允许悬空
            return reject_attempt(
                run=run,
                execution=execution,
                runtime=runtime,
                ledger=ledger,
                identity=identity,
                post_path=post_path,
                parent_path=parent_path,
                step_calls=int(identity in counter.called),
                started=started,
                wall=wall,
                error=error,
                checks=checks,
            )
        checks.append(check_r11(post_path, result["post"]))
        checks.append(check_r12(result["activity_receipt"]))
        result["post_sha256"] = normalized_file_sha(post_path)
        closed = ledger.close(
            identity,
            outcome="accepted",
            post_checkpoint="post.bin",
            optimizer_step_calls=1,
        )
    finally:
        torch.optim.Adam.step = original_step

    state = ledger.read()
    s12 = s12_payload(
        run,
        execution,
        result,
        {"open_seq": 1, "close_seq": closed["seq"]},
        identity,
        time.process_time() - started,
        time.monotonic() - wall,
    )
    s12["parent_sha256"] = normalized_file_sha(parent_path)
    write_once(ROOT / run["paths"]["s12"], s12)

    worker = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.training.verify_taiji_u2_s12_22_4_1",
            str(ROOT / run["paths"]["s12"]),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    worker_path = ROOT / run["paths"]["s12_worker"]
    worker_report = load_json(worker_path) if worker_path.exists() else None
    if worker.returncode != 0 or not worker_report or worker_report.get("verdict") != "pass":
        raise BudgetError(
            f"the independent S12 worker did not pass: {worker.returncode} {worker.stderr[-400:]}"
        )

    total_cpu = time.process_time() - started + worker_report.get("cpu_seconds", 0.0)
    total_wall = time.monotonic() - wall
    summary = {
        "schema": "taiji-u2-22.4.1-run-summary-v1",
        "status": "accepted",
        "run_id": run["run_id"],
        "update_id": run["update_id"],
        "checks": checks,
        "s12": run["paths"]["s12"],
        "worker": run["paths"]["s12_worker"],
        "worker_verdict": worker_report["verdict"],
        "ledger": {
            "path": run["paths"]["ledger"],
            "rows": len(state["rows"]),
            "consumed": state["consumed"],
            "head": state["head"],
            "closed_outcome": closed["outcome"],
        },
        "cpu_seconds": total_cpu,
        "wall_seconds": total_wall,
        "cpu_cap": run["cpu_cap"],
        "wall_cap": run["wall_cap"],
        "within_cap": total_cpu <= run["cpu_cap"] and total_wall <= run["wall_cap"],
    }
    write_once(root / "run_summary.json", summary)
    return summary


def normalized_file_sha(path: Path) -> str:
    return file_sha(path)


def self_check(directory: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    started, wall = time.process_time(), time.monotonic()
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    run = load_json(DRAFT)
    execution, _ = execution_manifest(run)
    admission, _ = admission_manifest(run)
    cases: list[dict[str, Any]] = []
    calls = {"step": 0}
    original_step = torch.optim.Adam.step

    def tripwire(self, *args, **kwargs):
        calls["step"] += 1
        raise AssertionError("optimizer.step must not run during the self-check")

    def expect(name: str, code: int, action) -> None:
        try:
            action()
        except LedgerError as error:
            actual = getattr(error, "exit_code", None)
            cases.append(
                {
                    "case": name,
                    "expected_exit": code,
                    "actual_exit": actual,
                    "pass": actual == code,
                    "detail": f"{type(error).__name__}: {error}",
                }
            )
        else:
            cases.append({"case": name, "expected_exit": code, "actual_exit": 0, "pass": False})

    torch.optim.Adam.step = tripwire
    try:
        expect(
            "authorization_missing",
            5,
            lambda: check_r02({**run, "authorized": False}),
        )
        expect(
            "update_id_mismatch",
            5,
            lambda: check_r01(
                {**run, "update_id": "u2-strong-baseline"},
                "taiji_u2_22_4_1_" + run["run_id"],
            ),
        )
        tampered = directory / "execution_manifest_tampered.json"
        tampered.write_text(
            json.dumps({**execution, "tampered": True}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        expect("manifest_hash_tampered", 5, lambda: check_r03(run, tampered))

        consumed_ledger = AttemptLedger(
            directory / "attempts_consumed.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        consumed_ledger.start()
        consumed_ledger.open(run["update_id"], parent_checkpoint="parent.bin")

        def double_open():
            consumed_ledger.open(run["update_id"], parent_checkpoint="parent.bin")

        expect("attempt_already_consumed", 3, double_open)
        expect(
            "attempt_already_consumed_r06",
            3,
            lambda: check_r06(run, admission, consumed=1),
        )

        runtime = prepare_parent(execution, directory)
        parent_path = directory / "parent.bin"
        runtime.save(parent_path)
        wrong = {**fingerprint(runtime), "revision": fingerprint(runtime)["revision"] + 7}

        def wrong_readback():
            readback_or_exit(parent_path, wrong)

        expect("parent_readback_mismatch", 4, wrong_readback)

        before = {
            name: value.detach().clone() for name, value in runtime.model.state_dict().items()
        }
        stub = {name: torch.ones_like(value) for name, value in runtime.model.named_parameters()}
        gate_ledger = AttemptLedger(
            directory / "attempts_gate.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        gate_ledger.start()
        gate_ledger.open(run["update_id"], parent_checkpoint="parent.bin")
        gate_capture: dict[str, Any] = {}

        def dead_change():
            try:
                activity_receipt(runtime.model, before, before, stub, execution)
            except ActivityGateError as error:
                gate_capture["receipt"] = error.receipt
                gate_ledger.close(
                    run["update_id"],
                    outcome="rejected",
                    error=f"{type(error).__name__}: {error}",
                    post_checkpoint=None,
                    optimizer_step_calls=0,
                )
                raise

        expect("activity_gate_not_live", 6, dead_change)
        gate_state = gate_ledger.read()
        cases.append(
            {
                "case": "activity_gate_billed_and_rejected",
                "expected": "consumed=1,outcome=rejected",
                "actual": f"consumed={gate_state['consumed']},"
                f"outcome={gate_state['closed'][run['update_id']]['outcome']},"
                f"receipt_rows={len((gate_capture.get('receipt') or {}).get('parameters', []))}",
                "pass": gate_state["consumed"] == 1
                and gate_state["closed"][run["update_id"]]["outcome"] == "rejected"
                and bool((gate_capture.get("receipt") or {}).get("parameters")),
            }
        )

        # 非账本类异常也必须收尾：注入 RuntimeError 走 reject_attempt，
        # 断言账本出现 close(rejected)、没有悬空 open、计费只加 1 次、rejected S12 落盘。
        generic_ledger = AttemptLedger(
            directory / "attempts_generic.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        generic_ledger.start()
        generic_ledger.open(run["update_id"], parent_checkpoint="parent.bin")
        generic_s12 = directory / "rejected_s12_generic.json"
        generic_outcome = reject_attempt(
            run=run,
            execution=execution,
            runtime=runtime,
            ledger=generic_ledger,
            identity=run["update_id"],
            post_path=directory / "post_generic.bin",
            parent_path=parent_path,
            step_calls=0,
            started=time.process_time(),
            wall=time.monotonic(),
            error=RuntimeError("injected non-ledger failure"),
            checks=[],
            s12_path=generic_s12,
        )
        generic_state = generic_ledger.read()
        dangling = [
            identity
            for identity in generic_state["opened"]
            if identity not in generic_state["closed"]
        ]
        generic_close = generic_state["closed"].get(run["update_id"], {})
        cases.append(
            {
                "case": "generic_exception_after_open",
                "expected": "close=rejected,dangling=0,consumed=1,exit=3,s12_present",
                "actual": f"close={generic_close.get('outcome')},dangling={len(dangling)},"
                f"consumed={generic_state['consumed']},exit={generic_outcome['exit_code']},"
                f"s12={generic_s12.exists()}",
                "pass": generic_close.get("outcome") == "rejected"
                and not dangling
                and generic_state["consumed"] == 1
                and generic_outcome["exit_code"] == 3
                and generic_s12.exists(),
            }
        )
    finally:
        torch.optim.Adam.step = original_step

    cases.append(
        {
            "case": "no_optimizer_step",
            "expected": 0,
            "actual": calls["step"],
            "pass": calls["step"] == 0,
        }
    )
    report = {
        "schema": "taiji-u2-22.4.1-selfcheck-v1",
        "verdict": "pass" if all(case["pass"] for case in cases) else "fail",
        "optimizer_step_calls": calls["step"],
        "cases": cases,
        "cpu_seconds": time.process_time() - started,
        "wall_seconds": time.monotonic() - wall,
    }
    write_once(directory / "selfcheck.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_check:
            report = self_check(SELFCHECK.parent)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["verdict"] == "pass" else 1
        if args.run:
            if not args.run_root:
                raise BudgetError("--run requires --run-root")
            summary = execute(Path(args.run_root))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
    except LedgerError as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "exit_code": getattr(error, "exit_code", 3),
                    "error": f"{type(error).__name__}: {error}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return getattr(error, "exit_code", 3)
    except Exception as error:  # noqa: BLE001 - the run must fail closed with evidence
        print(
            json.dumps(
                {
                    "status": "failed",
                    "exit_code": 3,
                    "error": f"{type(error).__name__}: {error}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3
    parser.error("choose --run or --self-check")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
