"""Run or self-check one frozen 22.4.1 engineering attempt."""

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
    CheckpointMismatch,
    ContractError,
    admission_manifest,
    check_r01,
    check_r02,
    check_r03,
    check_r04,
    check_r05,
    check_r05_baseline,
    check_r06,
    check_r07,
    check_r08,
    check_r09,
    check_r09_baseline,
    check_r10,
    check_r11,
    check_r11_baseline,
    check_r12,
    execution_manifest,
    load_json,
    perform,
    perform_baseline,
    readback_or_exit,
    rejected_result,
    rejected_result_baseline,
    write_once,
)
from scripts.training.u2_update_support_22_3_4_3 import (
    StepCounter,
    file_sha,
)
from scripts.training.u2_update_support_22_4_1 import (
    DEFAULT_ARM,
    BaselineRuntime,
    activity_receipt,
    baseline_fingerprint,
    baseline_outputs,
    fingerprint,
    normalized_sha256,
    prepare_parent,
    readback_baseline,
    recipe_digest,
    save_baseline,
    shadow_terms,
    slice_digest,
)
from taiji.u2_history import value_digest
from taiji.u2_subject import U2Config, U2Subject

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "plans/manifests/taiji_u2_22_4_1_u1_run_draft_20261111.json"
U2_DRAFT = ROOT / "plans/manifests/taiji_u2_22_4_1_u2_run_draft_20261111.json"
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
    return _s12_with_graph(
        {
            "schema": "taiji-u2-22.4.1-s12-v1",
            "run_id": run["run_id"],
            "attempt_id": identity,
            "update_id": run["update_id"],
            "arm": run["update_id"],
            "precision": run["precision"],
            "status": result["status"],
            "weight_profile": execution["arms"][run["update_id"]]["weight_profile"],
            "weights": execution["arms"][run["update_id"]]["weights"],
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
                (
                    proof
                    for proof in result["readbacks"]
                    if Path(proof["path"]).name == "parent.bin"
                ),
                None,
            ),
            "post_readback": next(
                (proof for proof in result["readbacks"] if Path(proof["path"]).name == "post.bin"),
                None,
            ),
            "failed_checkpoint": "post.bin" if result["status"] != "accepted" else None,
            "error": result.get("error"),
        },
        result,
    )


def _s12_with_graph(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """只有非默认臂（基线执行图）写 execution_graph；u1 的回执字节因此完全不变。"""
    if "execution_graph" in result:
        payload["execution_graph"] = result["execution_graph"]
    return payload


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
    顺序与成功路径一致：先写回执、再读回、最后 close，close 行携带回执 sha256；
    只有回执本身写不出来时才降级 close（行内记 s12_error），以保证账本仍然结算。
    """
    receipt = getattr(error, "receipt", None)
    if post_path.exists():
        post_path.unlink()
    result = (
        rejected_result_baseline(runtime, error, step_calls, receipt, execution)
        if isinstance(runtime, BaselineRuntime)
        else rejected_result(runtime, error, step_calls, receipt)
    )
    destination = s12_path or (ROOT / run["paths"]["s12"])
    s12 = s12_payload(
        run,
        execution,
        result,
        {"open_seq": 1, "close_seq": None},
        identity,
        time.process_time() - started,
        time.monotonic() - wall,
    )
    s12["parent_sha256"] = normalized_file_sha(parent_path) if parent_path.exists() else None
    fields: dict[str, Any] = {
        "outcome": "rejected",
        "error": f"{type(error).__name__}: {error}",
        "post_checkpoint": None,
        "optimizer_step_calls": step_calls,
    }
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        write_once(destination, s12)
        if load_json(destination) != json.loads(json.dumps(s12, ensure_ascii=False)):
            raise LedgerError(
                f"the rejected S12 receipt did not read back unchanged: {destination}"
            )
        fields["s12_path"] = (
            destination.relative_to(ROOT).as_posix()
            if destination.is_relative_to(ROOT)
            else str(destination)
        )
        fields["s12_sha256"] = normalized_file_sha(destination)
    except Exception as receipt_error:  # noqa: BLE001 - 回执写不出来也要结算，不能悬空
        fields["s12_error"] = f"{type(receipt_error).__name__}: {receipt_error}"
    ledger.close(identity, **fields)
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


def accepted_tail(
    *,
    ledger: AttemptLedger,
    identity: str,
    s12: dict[str, Any],
    destination: Path,
    post_checkpoint: str = "post.bin",
) -> dict[str, Any]:
    """成功路径的顺序契约：先写回执，再读回回执，最后 close。

    close 必须是账本最后一条事件，且只出现一次；close 行携带回执文件的 sha256，
    因此"回执先于 close"这件事可以只凭账本复核，不需要相信任何口供。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_once(destination, s12)
    read_back = load_json(destination)
    if read_back != json.loads(json.dumps(s12, ensure_ascii=False)):
        raise LedgerError(f"the S12 receipt did not read back unchanged: {destination}")
    receipt_sha = normalized_file_sha(destination)
    closed = ledger.close(
        identity,
        outcome="accepted",
        post_checkpoint=post_checkpoint,
        optimizer_step_calls=1,
        s12_path=(
            destination.relative_to(ROOT).as_posix()
            if destination.is_relative_to(ROOT)
            else str(destination)
        ),
        s12_sha256=receipt_sha,
    )
    close_receipt_guard(ledger, identity, destination)
    state = ledger.read()
    if state["rows"][-1].get("seq") != closed["seq"]:
        raise LedgerError("close is not the last ledger event")
    return closed


def close_receipt_guard(ledger: AttemptLedger, identity: str, s12_path: Path) -> None:
    """accepted close 必须自证回执：行内要有 s12_sha256，回执里要有实测 cpu/wall。

    缺任一项即退 3。这条守卫把"先 close 后回执"和"回执没有实测成本"都变成硬失败——
    账本里承诺的回执必须真的存在、真的对得上、真的记录了代价。
    """
    state = ledger.read()
    rows = [
        row
        for row in state["rows"]
        if row.get("event") == "close" and row.get("attempt_id") == identity
    ]
    if len(rows) != 1:
        raise LedgerError("the accepted close is not the only close row for this attempt")
    row = rows[0]
    if not row.get("s12_sha256"):
        raise LedgerError("the close row is missing s12_sha256")
    if not s12_path.exists():
        raise LedgerError(f"the attested S12 receipt is missing: {s12_path}")
    if row["s12_sha256"] != normalized_file_sha(s12_path):
        raise LedgerError("the close row s12_sha256 does not match the receipt on disk")
    receipt = load_json(s12_path)
    for field in ("cpu_seconds", "wall_seconds"):
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LedgerError(f"the S12 receipt has no measured {field}")


def bind_run(run_path: Path, *, contract_path: Path | None = None) -> dict[str, Any]:
    """同一 run_root 只允许绑定一份 run.json；第二个 run 复用必须被拒绝。"""
    draft = load_json(contract_path or DRAFT)
    if not run_path.exists():
        write_once(run_path, draft)
    run = load_json(run_path)
    if run["run_id"] != draft["run_id"]:
        raise BudgetError(
            f"the run root already holds run_id={run['run_id']}, "
            f"not the contracted {draft['run_id']}"
        )
    return run


def worker_gate(returncode: int, report: dict[str, Any] | None, stderr: str) -> None:
    """独立 worker 未通过时退 3；此时 attempt 已结算，绝不允许再写一次 close。"""
    if returncode != 0 or not report or report.get("verdict") != "pass":
        raise BudgetError(f"the independent S12 worker did not pass: {returncode} {stderr[-400:]}")


def execute(root: Path, *, contract_path: Path | None = None) -> dict[str, Any]:
    check_selfcheck_gate()
    torch.set_num_threads(1)
    started, wall = time.process_time(), time.monotonic()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_path = root / "run.json"
    run = bind_run(run_path, contract_path=contract_path)
    checks: list[dict[str, Any]] = []
    checks.append(check_r01(run, root.name))
    checks.append(check_r02(run))
    checks.append(check_r03(run))
    checks.append(check_r04(run))
    execution, _ = execution_manifest(run)
    admission, _ = admission_manifest(run)
    if run["update_id"] == DEFAULT_ARM:
        evidence = fork_evidence(torch.float32)
        checks.append(check_r05(run, execution, evidence["base_digest"], evidence))
    ledger_path = ROOT / run["paths"]["ledger"]
    consumed = 1 if ledger_path.exists() else 0
    checks.append(check_r06(run, admission, consumed))

    runtime = prepare_parent(execution, root, arm=run["update_id"])
    baseline = isinstance(runtime, BaselineRuntime)
    if baseline:
        # u2 的父是冻结执行图身份，R05 需要重建后的 runtime；判据编号与退码不变。
        checks.insert(4, check_r05_baseline(run, execution, runtime, arm=run["update_id"]))
    checks.append(check_r08(execution, runtime.model, runtime.optimizer, arm=run["update_id"]))

    recipe_sha = recipe_digest(execution)
    slice_sha = slice_digest(execution, runtime.model, arm=run["update_id"])
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
    if baseline:
        proof, r09 = check_r09_baseline(parent_path, runtime, execution, arm=run["update_id"])
    else:
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
            if baseline:
                result = perform_baseline(
                    runtime, ledger, counter, execution, post_path, arm=identity
                )
            else:
                result = perform(runtime, ledger, counter, execution, post_path, arm=identity)
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
        if baseline:
            checks.append(check_r11_baseline(post_path, result["post"], execution, arm=identity))
        else:
            checks.append(check_r11(post_path, result["post"]))
        checks.append(check_r12(result["activity_receipt"]))
        result["post_sha256"] = normalized_file_sha(post_path)
        s12 = s12_payload(
            run,
            execution,
            result,
            {"open_seq": 1, "close_seq": None},
            identity,
            time.process_time() - started,
            time.monotonic() - wall,
        )
        s12["parent_sha256"] = normalized_file_sha(parent_path)
        closed = accepted_tail(
            ledger=ledger,
            identity=identity,
            s12=s12,
            destination=ROOT / run["paths"]["s12"],
        )
    finally:
        torch.optim.Adam.step = original_step

    state = ledger.read()

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
    worker_gate(worker.returncode, worker_report, worker.stderr)

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
    if baseline:
        summary["execution_graph"] = dict(runtime.graph)
        summary["known_gaps"] = [
            "u2 的父是冻结初始化（training_runs=0、optimizer_steps=0），R05 按"
            "预注册 PREREG-TAIJI-U2_22_4_1_R05_PARENT_IDENTITY 检查执行图身份；"
            "它不签“从已训练状态继续”，不得引用为续训证据。",
            "该图没有策略头与主体 state，policy 与 state 迁移是结构空，已如实声明。",
            "摘要依赖 threads=1 才可复现，跨线程数不同；只登记，须单独预注册才能修。",
        ]
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

        # R06 四条资源/精度/步数闸门：只改内存副本，不碰冻结草稿。
        # 每条都必须退 3，且见证账本一个字节都不能多（拒绝发生在 open 之前）。
        r06_path = directory / "attempts_r06.ndjson"
        r06_ledger = AttemptLedger(
            r06_path,
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        r06_ledger.start()

        def expect_no_ledger_write(name: str, action) -> None:
            before = r06_path.read_bytes()
            try:
                action()
            except LedgerError as error:
                actual, detail = (
                    getattr(error, "exit_code", None),
                    f"{type(error).__name__}: {error}",
                )
            else:
                actual, detail = 0, "no exception"
            after = r06_path.read_bytes()
            cases.append(
                {
                    "case": name,
                    "expected": f"exit=3,ledger_unchanged（{detail}）",
                    "actual": f"exit={actual},ledger_unchanged={after == before},"
                    f"rows={len(r06_ledger.read()['rows'])},"
                    f"consumed={r06_ledger.read()['consumed']}",
                    "pass": actual == 3
                    and after == before
                    and len(r06_ledger.read()["rows"]) == 1
                    and r06_ledger.read()["consumed"] == 0,
                }
            )

        expect_no_ledger_write(
            "r06_cpu_cap_over_price",
            lambda: check_r06({**run, "cpu_cap": 121}, admission, consumed=0),
        )
        expect_no_ledger_write(
            "r06_wall_cap_over_price",
            lambda: check_r06({**run, "wall_cap": 241}, admission, consumed=0),
        )
        expect_no_ledger_write(
            "r06_precision_mismatch",
            lambda: check_r06({**run, "precision": "float64"}, admission, consumed=0),
        )
        expect_no_ledger_write(
            "r06_optimizer_steps_not_one",
            lambda: check_r06({**run, "optimizer_steps_authorized": 2}, admission, consumed=0),
        )

        # 额度闸门：假 precontract 把 budget_max 压到 0（不碰冻结草稿），
        # open 必须退 3，且账本一个字节都不能多——闸门拒绝不能留下计费行。
        zero_contract = precontract(run)
        zero_contract["run"]["budget_max"] = 0
        zero_path = directory / "attempts_budget_zero.ndjson"
        zero_ledger = AttemptLedger(
            zero_path,
            ledger_run(run),
            execution["authority"]["source_pins"],
            zero_contract,
        )
        zero_ledger.start()
        zero_start_bytes = zero_path.read_bytes()
        zero_exit: int | None = None
        try:
            zero_ledger.open(run["update_id"], parent_checkpoint="parent.bin")
        except LedgerError as error:
            zero_exit = getattr(error, "exit_code", None)
        zero_after_bytes = zero_path.read_bytes()
        zero_rows = len(zero_ledger.read()["rows"])
        cases.append(
            {
                "case": "budget_max_zero_gate",
                "expected": "exit=3,bytes_unchanged,rows=1,consumed=0",
                "actual": f"exit={zero_exit},bytes_unchanged={zero_after_bytes == zero_start_bytes},"
                f"rows={zero_rows},consumed={zero_ledger.read()['consumed']}",
                "pass": zero_exit == 3
                and zero_after_bytes == zero_start_bytes
                and zero_rows == 1
                and zero_ledger.read()["consumed"] == 0,
            }
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

        # 成功路径：临时 run 根、假优化器（不调用 optimizer.step、不消耗真实额度）。
        # 走完 回执 → 读回回执 → close 的顺序，断言 close 是账本最后一条事件、
        # close 只出现一次、close 行携带回执 sha256、计费只加 1 次。
        success_ledger = AttemptLedger(
            directory / "attempts_success.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        success_ledger.start()
        success_ledger.open(run["update_id"], parent_checkpoint="parent.bin")
        fake_before = {
            name: value.detach().clone() for name, value in runtime.model.state_dict().items()
        }
        fake_after = dict(fake_before)
        for name in execution["arms"][DEFAULT_ARM]["active_parameters"]:
            fake_after[name] = fake_before[name] + 1e-3
        stub = {name: torch.ones_like(value) for name, value in runtime.model.named_parameters()}
        fake_receipt = activity_receipt(runtime.model, fake_before, fake_after, stub, execution)
        # 假优化器只改"读数"，不改模型：这份 post.bin 与父状态相同，仅用来做一次耐久读回。
        success_post = directory / "post.bin"
        runtime.save(success_post)
        success_s12_path = directory / "success_s12.json"
        fake_s12 = {
            "schema": "taiji-u2-22.4.1-s12-v1",
            "status": "accepted",
            "run_id": run["run_id"],
            "attempt_id": run["update_id"],
            "ledger_seq_open": 1,
            "ledger_seq_close": None,
            "activity_receipt": fake_receipt,
            "post_sha256": file_sha(success_post),
            "conditions": {"threads": torch.get_num_threads()},
            "cpu_seconds": 0.01,
            "wall_seconds": 0.01,
        }
        accepted_tail(
            ledger=success_ledger,
            identity=run["update_id"],
            s12=fake_s12,
            destination=success_s12_path,
        )
        success_state = success_ledger.read()
        success_rows = success_state["rows"]
        success_closes = [row for row in success_rows if row.get("event") == "close"]
        cases.append(
            {
                "case": "accepted_path_order",
                "expected": "last_event=close,closes=1,s12_sha256_matches,consumed=1",
                "actual": f"last={success_rows[-1].get('event')},closes={len(success_closes)},"
                f"s12_sha256_matches="
                f"{bool(success_closes) and success_closes[0].get('s12_sha256') == file_sha(success_s12_path)},"
                f"consumed={success_state['consumed']}",
                "pass": success_rows[-1].get("event") == "close"
                and len(success_closes) == 1
                and success_closes[0].get("s12_sha256") == file_sha(success_s12_path)
                and success_state["consumed"] == 1,
            }
        )
        expect(
            "duplicate_close_rejected",
            3,
            lambda: success_ledger.close(
                run["update_id"],
                outcome="accepted",
                post_checkpoint="post.bin",
                optimizer_step_calls=1,
            ),
        )

        # ---- 收尾分支：R11 读回失败必须落 close(rejected) + rejected S12 ----
        r11_ledger = AttemptLedger(
            directory / "attempts_r11.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        r11_ledger.start()
        r11_ledger.open(run["update_id"], parent_checkpoint="parent.bin")
        r11_s12 = directory / "r11_rejected_s12.json"
        r11_result = reject_attempt(
            run=run,
            execution=execution,
            runtime=runtime,
            ledger=r11_ledger,
            identity=run["update_id"],
            post_path=directory / "post_r11.bin",
            parent_path=parent_path,
            step_calls=1,
            started=started,
            wall=wall,
            error=CheckpointMismatch("R11 post-save readback mismatch"),
            checks=[],
            s12_path=r11_s12,
        )
        r11_state = r11_ledger.read()
        r11_closes = [row for row in r11_state["rows"] if row.get("event") == "close"]
        cases.append(
            {
                "case": "r11_readback_failure_closeout",
                "expected": "exit=4,status=rejected,last=close,closes=1,s12=present,s12_sha256_matches,consumed=1",
                "actual": f"exit={r11_result['exit_code']},status={r11_result['status']},"
                f"last={r11_state['rows'][-1].get('event')},closes={len(r11_closes)},"
                f"s12={r11_s12.exists()},"
                f"s12_sha256_matches="
                f"{bool(r11_closes) and r11_closes[0].get('s12_sha256') == file_sha(r11_s12)},"
                f"consumed={r11_state['consumed']}",
                "pass": r11_result["exit_code"] == 4
                and r11_result["status"] == "rejected"
                and r11_state["rows"][-1].get("event") == "close"
                and len(r11_closes) == 1
                and r11_s12.exists()
                and r11_closes[0].get("s12_sha256") == file_sha(r11_s12)
                and r11_state["consumed"] == 1,
            }
        )

        # ---- 收尾分支：worker 未通过退 3，且不得第二次 close ----
        accepted_rows = len(success_ledger.read()["rows"])
        expect("worker_failed_exit3", 3, lambda: worker_gate(1, None, "worker crashed"))
        cases.append(
            {
                "case": "worker_failure_does_not_reclose",
                "expected": "rows unchanged,closes=1",
                "actual": f"rows={len(success_ledger.read()['rows'])},"
                f"closes={len([r for r in success_ledger.read()['rows'] if r.get('event') == 'close'])}",
                "pass": len(success_ledger.read()["rows"]) == accepted_rows
                and len([r for r in success_ledger.read()["rows"] if r.get("event") == "close"])
                == 1,
            }
        )

        # ---- 回执自证：rejected 也必须"回执先落、close 在后"，且 close 带回执 sha256 ----
        order_ledger = AttemptLedger(
            directory / "attempts_rejected_order.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        order_ledger.start()
        order_ledger.open(run["update_id"], parent_checkpoint="parent.bin")
        order_s12 = directory / "rejected_s12_order.json"
        order_outcome = reject_attempt(
            run=run,
            execution=execution,
            runtime=runtime,
            ledger=order_ledger,
            identity=run["update_id"],
            post_path=directory / "post_order.bin",
            parent_path=parent_path,
            step_calls=1,
            started=time.process_time(),
            wall=time.monotonic(),
            error=CheckpointMismatch("R11 post-save readback mismatch"),
            checks=[],
            s12_path=order_s12,
        )
        order_state = order_ledger.read()
        order_close = order_state["closed"][run["update_id"]]
        order_receipt = load_json(order_s12)
        cases.append(
            {
                "case": "rejected_s12_precedes_close",
                "expected": "exit=4,last=close,s12_sha256_matches,seq_close_is_null,cost_present",
                "actual": f"exit={order_outcome['exit_code']},"
                f"last={order_state['rows'][-1].get('event')},"
                f"s12_sha256_matches={order_close.get('s12_sha256') == file_sha(order_s12)},"
                f"seq_close={order_receipt.get('ledger_seq_close')},"
                f"cost={order_receipt.get('cpu_seconds') is not None and order_receipt.get('wall_seconds') is not None}",
                "pass": order_outcome["exit_code"] == 4
                and order_state["rows"][-1].get("event") == "close"
                and order_close.get("s12_sha256") == file_sha(order_s12)
                and order_receipt.get("ledger_seq_close") is None
                and isinstance(order_receipt.get("cpu_seconds"), (int, float))
                and isinstance(order_receipt.get("wall_seconds"), (int, float)),
            }
        )

        # close 不带 s12_sha256：守卫必须退 3（"先 close 后回执"在这里变成硬失败）
        bare_ledger = AttemptLedger(
            directory / "attempts_close_without_sha.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        bare_ledger.start()
        bare_ledger.open(run["update_id"], parent_checkpoint="parent.bin")
        bare_receipt = directory / "receipt_with_cost.json"
        bare_receipt.write_text(
            json.dumps({"cpu_seconds": 1.5, "wall_seconds": 2.5}, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        bare_ledger.close(
            run["update_id"],
            outcome="accepted",
            post_checkpoint="post.bin",
            optimizer_step_calls=1,
        )
        expect(
            "close_missing_s12_sha256",
            3,
            lambda: close_receipt_guard(bare_ledger, run["update_id"], bare_receipt),
        )

        # 回执存在、sha 也对，但没有实测 cpu／wall：同样退 3
        cost_ledger = AttemptLedger(
            directory / "attempts_receipt_without_cost.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        cost_ledger.start()
        cost_ledger.open(run["update_id"], parent_checkpoint="parent.bin")
        cost_receipt = directory / "receipt_without_cost.json"
        cost_receipt.write_text(
            json.dumps({"status": "accepted"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        cost_ledger.close(
            run["update_id"],
            outcome="accepted",
            post_checkpoint="post.bin",
            optimizer_step_calls=1,
            s12_path=str(cost_receipt),
            s12_sha256=file_sha(cost_receipt),
        )
        expect(
            "s12_without_cpu_wall",
            3,
            lambda: close_receipt_guard(cost_ledger, run["update_id"], cost_receipt),
        )

        # ---- 合同类拒绝路径（低成本可复现的部分）----
        tampered_admission = directory / "admission_manifest_tampered.json"
        tampered_admission.write_text(
            json.dumps({**admission, "tampered": True}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        expect("admission_hash_tampered", 5, lambda: check_r04(run, tampered_admission))
        expect(
            "data_scope_flag_mutated",
            5,
            lambda: check_r07(
                {**run, "data_scope": {**run["data_scope"], "reads_final": True}},
                "1" * 64,
                "2" * 64,
            ),
        )
        expect(
            "data_digest_mismatch",
            5,
            lambda: check_r07(
                {
                    **run,
                    "data_scope": {
                        **run["data_scope"],
                        "data_recipe_sha256": "0" * 64,
                        "slice_sha256": "0" * 64,
                    },
                },
                "1" * 64,
                "2" * 64,
            ),
        )
        expect(
            "optimizer_hyperparameter_changed",
            5,
            lambda: check_r08(
                {**execution, "optimizer": {**execution["optimizer"], "lr": 1e-2}},
                runtime.model,
                runtime.optimizer,
            ),
        )
        expect(
            "declared_active_parameter_absent",
            5,
            lambda: check_r08(
                {
                    **execution,
                    "arms": {
                        **execution["arms"],
                        DEFAULT_ARM: {
                            **execution["arms"][DEFAULT_ARM],
                            "active_parameters": [
                                *execution["arms"][DEFAULT_ARM]["active_parameters"],
                                "not_a_parameter",
                            ],
                        },
                    },
                },
                runtime.model,
                runtime.optimizer,
            ),
        )

        # ---- 账本拒绝路径与两个修复入口 ----
        duplicate_ledger = AttemptLedger(
            directory / "attempts_dup_start.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        duplicate_ledger.start()
        # 注入第二条 run_start：模拟手工篡改或旧缺陷，用来验证 R10 的"唯一首行"判据。
        duplicate_ledger._append("run_start", None, 0, 0, {})
        expect("ledger_run_start_not_unique", 3, duplicate_ledger.read)

        expect(
            "ledger_run_identity_mismatch",
            3,
            lambda: AttemptLedger(
                directory / "attempts_dup_start.ndjson",
                {**ledger_run(run), "run_id": "20261111T000000Z"},
                execution["authority"]["source_pins"],
                precontract(run),
            ).read(),
        )

        tail_ledger = AttemptLedger(
            directory / "attempts_tail.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        tail_ledger.start()
        complete = (directory / "attempts_tail.ndjson").read_bytes()
        (directory / "attempts_tail_truncated.ndjson").write_bytes(
            complete + b'{"seq": 1, "event": "clo'
        )
        truncated = AttemptLedger(
            directory / "attempts_tail_truncated.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        expect("tail_truncated_detected", 3, truncated.read)
        repair_proof = truncated.repair_tail(directory / "attempts_tail_repair_proof.json")
        repaired = truncated.read()
        cases.append(
            {
                "case": "repair_tail_restores_chain",
                "expected": "removed_bytes>0,rows=1,consumed=0",
                "actual": f"removed={repair_proof['removed_bytes']},"
                f"rows={len(repaired['rows'])},consumed={repaired['consumed']}",
                "pass": repair_proof["removed_bytes"] > 0
                and len(repaired["rows"]) == 1
                and repaired["consumed"] == 0,
            }
        )

        reconcile_ledger = AttemptLedger(
            directory / "attempts_reconcile.ndjson",
            ledger_run(run),
            execution["authority"]["source_pins"],
            precontract(run),
        )
        reconcile_ledger.start()
        reconcile_ledger.open(run["update_id"], parent_checkpoint="parent.bin")
        reconciled_rows = reconcile_ledger.reconcile()
        again = reconcile_ledger.reconcile()
        reconciled_state = reconcile_ledger.read()
        cases.append(
            {
                "case": "reconcile_open_without_close",
                "expected": "rows=1,event=reconcile,consumed=1,idempotent",
                "actual": f"rows={len(reconciled_rows)},"
                f"event={reconciled_rows[0].get('event') if reconciled_rows else None},"
                f"consumed={reconciled_state['consumed']},again={len(again)}",
                "pass": len(reconciled_rows) == 1
                and reconciled_rows[0].get("event") == "reconcile"
                and reconciled_state["consumed"] == 1
                and again == []
                and run["update_id"] in reconciled_state["reconciled"],
            }
        )

        # ---- 入口分支：同一 run_root 复用、CLI 参数冲突、缺 --run-root ----
        reuse_root = directory / "reuse"
        reuse_root.mkdir()
        write_once(reuse_root / "run.json", {**run, "run_id": "20261111T000000Z"})
        expect("run_root_reused_by_second_run", 3, lambda: bind_run(reuse_root / "run.json"))

        try:
            main(["--run", "--self-check", "--run-root", str(directory / "cli")])
        except SystemExit as error:
            cli_conflict = error.code
        else:
            cli_conflict = 0
        cases.append(
            {
                "case": "cli_conflict_run_and_selfcheck",
                "expected_exit": 2,
                "actual_exit": cli_conflict,
                "pass": cli_conflict == 2,
            }
        )
        cli_missing_root = main(["--run"])
        cases.append(
            {
                "case": "cli_run_without_root",
                "expected_exit": 3,
                "actual_exit": cli_missing_root,
                "pass": cli_missing_root == 3,
            }
        )

        # ---- u2 草稿：臂参数化后的 0 步验证（不写 run.json、不写账本、0 step） ----
        u2_draft = load_json(U2_DRAFT)
        u2_root = "taiji_u2_22_4_1_20261111T000000Z"
        expect("u2_draft_r01_rejects_null_run_id", 5, lambda: check_r01(u2_draft, u2_root))
        expect("u2_draft_r02_rejects_missing_authorization", 5, lambda: check_r02(u2_draft))
        # 仅内存绑定（不落盘）：把 owner 放行后才能确定的三个字段补齐，验证 R01–R05 是否只差这三项。
        bound = {
            **u2_draft,
            "run_id": "20261111T000000Z",
            "authorized": True,
            "owner_approval_ref": "selfcheck:u2-strong-baseline",
            "attempts_authorized": 1,
            "optimizer_steps_authorized": 1,
        }
        u2_execution, _ = execution_manifest(bound)
        u2_admission, _ = admission_manifest(bound)
        u2_dir = directory / "u2-parent"
        u2_dir.mkdir(parents=True, exist_ok=True)
        u2_runtime = prepare_parent(u2_execution, u2_dir, arm="u2-strong-baseline")
        for name, action in (
            ("u2_bound_r01_identity_binds_arm", lambda: check_r01(bound, u2_root)),
            ("u2_bound_r02_authorization_binds_arm", lambda: check_r02(bound)),
            ("u2_bound_r03_execution_manifest_sha", lambda: check_r03(bound)),
            ("u2_bound_r04_admission_and_identity_sha", lambda: check_r04(bound)),
            (
                "u2_bound_r05_frozen_identity",
                lambda: check_r05_baseline(
                    bound, u2_execution, u2_runtime, arm="u2-strong-baseline"
                ),
            ),
            # F-20261112-E2E-01（裁定 A）的正向见证：准入 u2 行的 precision 改实后，
            # R06 必须真的放行；判据本身未动，此条只证明闸门不再常年关闭。
            ("u2_bound_r06_real_precision_accepted", lambda: check_r06(bound, u2_admission, 0)),
        ):
            try:
                detail = action()
            except (ContractError, BudgetError, CheckpointMismatch, ActivityGateError) as error:
                cases.append(
                    {
                        "case": name,
                        "expected": "pass",
                        "actual": f"exit {error.exit_code}",
                        "pass": False,
                    }
                )
            else:
                cases.append(
                    {
                        "case": name,
                        "expected": "pass",
                        "actual": "pass",
                        "pass": True,
                        "detail": json.dumps(detail, ensure_ascii=False)[:160],
                    }
                )

        # ---- u2 R05：父身份漂移/有训练祖先/未选中/模块 pin 漂移都退 5；threads 只记录 ----
        def write_variant(payload: dict[str, Any], filename: str) -> Path:
            path = directory / filename
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return path

        def run_with(**overrides: Any) -> dict[str, Any]:
            return {**bound, "references": {**bound["references"], **overrides}}

        def execution_with(identity_path: Path) -> dict[str, Any]:
            variant = json.loads(json.dumps(u2_execution, ensure_ascii=False))
            row = variant["arms"]["u2-strong-baseline"]
            row["baseline_identity_ref"] = str(identity_path)
            row["baseline_identity_sha256"] = normalized_sha256(identity_path)
            return variant

        identity_payload = load_json(
            ROOT / u2_execution["arms"]["u2-strong-baseline"]["baseline_identity_ref"]
        )
        expect(
            "u2_r05_identity_hash_drift_rejected",
            5,
            lambda: check_r05_baseline(
                run_with(baseline_identity_sha256="0" * 64),
                u2_execution,
                u2_runtime,
                arm="u2-strong-baseline",
            ),
        )
        ancestor = json.loads(json.dumps(identity_payload, ensure_ascii=False))
        ancestor["conditions"]["training_runs"] = 1
        ancestor_path = write_variant(ancestor, "r05-identity-trained.json")
        expect(
            "u2_r05_training_ancestor_rejected",
            5,
            lambda: check_r05_baseline(
                run_with(baseline_identity_sha256=normalized_sha256(ancestor_path)),
                execution_with(ancestor_path),
                u2_runtime,
                arm="u2-strong-baseline",
            ),
        )
        module_drift = json.loads(json.dumps(identity_payload, ensure_ascii=False))
        module_drift["architecture"]["module_sha256"] = "0" * 64
        module_drift_path = write_variant(module_drift, "r05-identity-module-drift.json")
        expect(
            "u2_r05_module_pin_drift_rejected",
            5,
            lambda: check_r05_baseline(
                run_with(baseline_identity_sha256=normalized_sha256(module_drift_path)),
                execution_with(module_drift_path),
                u2_runtime,
                arm="u2-strong-baseline",
            ),
        )
        report_payload = load_json(ROOT / bound["references"]["measurement_report"])
        report_payload["selection"]["selected"] = "gru_sequence"
        report_path = write_variant(report_payload, "r05-report-not-selected.json")
        expect(
            "u2_r05_measurement_not_selected_rejected",
            5,
            lambda: check_r05_baseline(
                run_with(
                    measurement_report=str(report_path),
                    measurement_report_sha256=normalized_sha256(report_path),
                ),
                u2_execution,
                u2_runtime,
                arm="u2-strong-baseline",
            ),
        )
        threads_variant = json.loads(json.dumps(identity_payload, ensure_ascii=False))
        threads_variant["conditions"]["threads"] = 4
        threads_path = write_variant(threads_variant, "r05-identity-threads.json")
        try:
            threads_detail = check_r05_baseline(
                run_with(baseline_identity_sha256=normalized_sha256(threads_path)),
                execution_with(threads_path),
                u2_runtime,
                arm="u2-strong-baseline",
            )
        except ContractError as error:
            cases.append(
                {
                    "case": "u2_r05_threads_recorded_not_judged",
                    "expected": "pass",
                    "actual": f"exit {error.exit_code}",
                    "pass": False,
                }
            )
        else:
            cases.append(
                {
                    "case": "u2_r05_threads_recorded_not_judged",
                    "expected": "pass",
                    "actual": "pass",
                    "pass": threads_detail["threads_policy"] == "threads 只记录、不判假",
                    "detail": f"threads={threads_detail['conditions']['threads']}",
                }
            )
        # ---- u2 臂可执行性：R08 按 F2 profile 通过，权重漂移与未知 profile 会被拦 ----
        try:
            u2_r08 = check_r08(
                u2_execution, u2_runtime.model, u2_runtime.optimizer, arm="u2-strong-baseline"
            )
        except (ContractError, BudgetError) as error:
            cases.append(
                {
                    "case": "u2_bound_r08_weights_and_active_set",
                    "expected": "pass",
                    "actual": f"exit {error.exit_code}",
                    "pass": False,
                }
            )
        else:
            cases.append(
                {
                    "case": "u2_bound_r08_weights_and_active_set",
                    "expected": "pass",
                    "actual": "pass",
                    "pass": True,
                    "detail": (
                        f"active={len(u2_r08['active_parameters'])}, "
                        f"weights={u2_r08['weights']}"
                    ),
                }
            )
        drifted = json.loads(json.dumps(u2_execution, ensure_ascii=False))
        drifted["arms"]["u2-strong-baseline"]["weights"]["verbatim"] = 0.5
        expect(
            "u2_r08_weight_drift_rejected",
            5,
            lambda: check_r08(
                drifted, u2_runtime.model, u2_runtime.optimizer, arm="u2-strong-baseline"
            ),
        )
        unknown_profile = json.loads(json.dumps(u2_execution, ensure_ascii=False))
        unknown_profile["arms"]["u2-strong-baseline"]["weight_profile"] = "F9-style"
        expect(
            "u2_r08_unknown_profile_rejected",
            5,
            lambda: check_r08(
                unknown_profile, u2_runtime.model, u2_runtime.optimizer, arm="u2-strong-baseline"
            ),
        )
        # u2 的执行图必须就是身份清单冻结的基线本身：参数量、参数清单、三槽与 F2 损失都要对上。
        u2_row = u2_execution["arms"]["u2-strong-baseline"]
        u2_graph = getattr(u2_runtime, "graph", {}) or {}
        u2_model_names = sorted(dict(u2_runtime.model.named_parameters()))
        u2_roles = u2_row.get("active_parameter_roles") or {}
        role_order = ("predictive_head", "recurrent_core", "output_head")
        u2_role_names = {
            role: list(u2_roles.get(role, {}).get("param_names") or []) for role in role_order
        }
        u2_role_union = sorted({name for names in u2_role_names.values() for name in names})
        u2_declared = list(u2_row.get("active_parameters") or [])
        u2_draft_slots = {
            role: list(u2_draft["identity_slots"][role]["param_names"]) for role in role_order
        }
        u2_loss, u2_pieces = shadow_terms(u2_runtime.model, u2_execution, arm="u2-strong-baseline")
        u2_gradients = torch.autograd.grad(
            u2_loss, list(u2_runtime.model.parameters()), allow_unused=True
        )
        u2_dead = [
            name
            for name, gradient in zip(u2_model_names, u2_gradients, strict=True)
            if gradient is None or float(gradient.norm()) == 0.0
        ]
        u2_graph_ok = (
            u2_graph.get("architecture_id"),
            u2_graph.get("param_total"),
            u2_graph.get("param_tensors"),
        ) == ("transformer_sequence", 21977, 35)
        cases.append(
            {
                "case": "u2_execution_graph_is_frozen_baseline",
                "expected": "transformer_sequence 21977/35",
                "actual": (
                    f"{u2_graph.get('architecture_id')} "
                    f"{u2_graph.get('param_total')}/{u2_graph.get('param_tensors')}"
                ),
                "pass": u2_graph_ok,
                "detail": f"inventory={u2_graph.get('param_inventory_sha256')}",
            }
        )
        sentinel_u2_names = [
            "native_w",
            "initial_h",
            "decoder_query.rgb.weight",
            "record_metadata.weight",
            "f_key.weight",
        ]
        lingering = sorted(set(sentinel_u2_names) & set(u2_declared))
        cases.append(
            {
                "case": "u2_no_u2_parameter_names_on_arm",
                "expected": [],
                "actual": lingering,
                "pass": not lingering,
            }
        )
        cases.append(
            {
                "case": "u2_role_slots_backfilled_from_graph",
                "expected": "output_head=byte_head, 另两槽不存在",
                "actual": {role: len(u2_role_names[role]) for role in role_order},
                "pass": (
                    u2_role_names["output_head"] == ["byte_head.weight", "byte_head.bias"]
                    and not u2_role_names["predictive_head"]
                    and not u2_role_names["recurrent_core"]
                    and set(u2_role_union) <= set(u2_declared)
                ),
            }
        )
        pieces = {name: float(value) for name, value in u2_pieces.items()}
        cases.append(
            {
                "case": "u2_verbatim_term_positive",
                "expected": "> 0",
                "actual": round(pieces["verbatim"], 6),
                "pass": pieces["verbatim"] > 0.0,
                "detail": json.dumps(pieces, ensure_ascii=False),
            }
        )
        cases.append(
            {
                "case": "u2_f2_gradients_alive",
                "expected": [],
                "actual": u2_dead,
                "pass": not u2_dead,
            }
        )
        # ---- u2 一步执行路径（0 步自检只能验证不落参的部分：写盘往返、账本闸门、公式量级） ----
        roundtrip_dir = directory / "u2-roundtrip"
        roundtrip_dir.mkdir(parents=True, exist_ok=True)
        u2_parent_expected = baseline_fingerprint(u2_runtime)
        save_baseline(u2_runtime, roundtrip_dir / "parent.bin", u2_execution)
        try:
            u2_restored, u2_proof = readback_baseline(
                roundtrip_dir / "parent.bin",
                u2_parent_expected,
                u2_execution,
                arm="u2-strong-baseline",
            )
        except Exception as error:  # noqa: BLE001 - 0 步往返失败必须显式记红
            cases.append(
                {
                    "case": "u2_baseline_parent_checkpoint_round_trip",
                    "expected": "five fields equal",
                    "actual": f"{type(error).__name__}: {error}",
                    "pass": False,
                }
            )
        else:
            five = (
                "state_sha256",
                "parameter_sha256",
                "moment_sha256",
                "history_reference",
                "revision",
            )
            actual_five = baseline_fingerprint(u2_restored)
            cases.append(
                {
                    "case": "u2_baseline_parent_checkpoint_round_trip",
                    "expected": "five fields equal",
                    "actual": sorted(five),
                    "pass": all(actual_five[key] == u2_parent_expected[key] for key in five),
                    "detail": (
                        f"bytes={u2_proof['bytes']} revision={actual_five['revision']} "
                        f"parameters={actual_five['parameter_sha256'][:16]}"
                    ),
                }
            )
        gate_dir = directory / "u2-gate"
        gate_dir.mkdir(parents=True, exist_ok=True)
        gate_ledger = AttemptLedger(
            gate_dir / "attempts.ndjson",
            ledger_run(bound),
            u2_execution["authority"]["source_pins"],
            precontract(bound),
        )
        gate_ledger.start()
        gate_counter = StepCounter(gate_ledger, gate_dir / "claims")
        gate_counter.current = "u2-strong-baseline"
        torch.optim.Adam.step = gate_counter.wrapper(original_step)
        try:
            expect(
                "u2_baseline_step_requires_open_attempt",
                3,
                lambda: u2_runtime.optimizer.step(),
            )
        finally:
            torch.optim.Adam.step = tripwire
        # 首次 Adam 步（零动量）每元素位移 = lr·g/(|g|+eps)：上界是 lr，|g|≫eps 时贴住 lr。
        group = u2_runtime.optimizer.param_groups[0]
        lr = float(group["lr"])
        beta1, beta2 = (float(value) for value in group["betas"])
        eps = float(group["eps"])
        biggest = 0.0
        shortfall = 0.0
        below_eps = 0
        elements = 0
        for gradient in u2_gradients:
            step = lr * gradient / (gradient.abs() + eps)
            biggest = max(biggest, float(step.abs().max()))
            strong = gradient.abs() >= eps * 1e3
            elements += gradient.numel()
            below_eps += int((~strong).sum())
            if bool(strong.any()):
                shortfall = max(shortfall, float((step.abs()[strong] - lr).abs().max()))
        cases.append(
            {
                "case": "u2_baseline_first_step_displacement_is_lr",
                "expected": "max|Δ|<=lr and |Δ|=lr where |g|>=1e3·eps",
                "actual": f"max|Δ|={biggest:.6e} shortfall={shortfall:.3e}",
                "pass": biggest <= lr * (1.0 + 1e-6) and shortfall <= lr * 1e-3,
                "detail": (
                    f"lr={lr} betas=({beta1},{beta2}) eps={eps}; "
                    f"elements={elements} below_eps={below_eps}"
                ),
            }
        )
        u2_outputs = baseline_outputs(u2_runtime, u2_execution)
        out_features = int(u2_runtime.model.byte_head.out_features)
        cases.append(
            {
                "case": "u2_baseline_policy_is_structural_zero",
                "expected": f"policy=[] and next_output={out_features}",
                "actual": {
                    "policy": u2_outputs["policy"],
                    "next_output": len(u2_outputs["next_output"]),
                },
                "pass": u2_outputs["policy"] == []
                and len(u2_outputs["next_output"]) == out_features,
            }
        )
        u2_rejected = rejected_result_baseline(
            u2_runtime, RuntimeError("self-check"), 0, None, u2_execution
        )
        cases.append(
            {
                "case": "u2_baseline_rejected_result_shape",
                "expected": "rejected + execution_graph",
                "actual": {
                    "status": u2_rejected["status"],
                    "param_total": (u2_rejected["execution_graph"] or {}).get("param_total"),
                },
                "pass": u2_rejected["status"] == "rejected"
                and (u2_rejected["execution_graph"] or {}).get("param_total") == 21977
                and u2_rejected["policy"] == [],
            }
        )
        u2_preflight = {
            "arm": "u2-strong-baseline",
            "graph_architecture_id": u2_graph.get("architecture_id"),
            "graph_param_total": u2_graph.get("param_total"),
            "graph_param_tensors": u2_graph.get("param_tensors"),
            "graph_param_inventory_sha256": u2_graph.get("param_inventory_sha256"),
            "graph_identity_sha256": u2_graph.get("identity_sha256"),
            "dtype_field": u2_row.get("dtype"),
            "dtype_is_torch_dtype": hasattr(torch, str(u2_row.get("dtype"))),
            "active_parameters_present": bool(u2_declared),
            "active_parameters_count": len(u2_declared),
            "active_set_equals_graph_inventory": u2_declared == u2_model_names,
            "role_slots": {
                role: {
                    "exists": bool(u2_roles.get(role, {}).get("exists")),
                    "count": len(u2_role_names[role]),
                    "param_names": u2_role_names[role],
                }
                for role in role_order
            },
            "role_union_subset_of_active": set(u2_role_union) <= set(u2_declared),
            "roles_match_draft": u2_role_names == u2_draft_slots,
            "missing_in_model": sorted(set(u2_declared) - set(u2_model_names)),
            "verbatim": {
                "target": u2_execution["recipe"]["u2_shadow"]["target"],
                "target_bytes": len(u2_execution["recipe"]["u2_shadow"]["target"].encode("utf-8")),
                **{name: round(value, 6) for name, value in pieces.items()},
            },
            "dead_gradient_parameters": u2_dead,
            "conditional_active_parameters_present": bool(
                u2_row.get("conditional_active_parameters")
            ),
            "step_path_implemented": True,
            "step_path": {
                "mechanism": "in-place Adam step on the frozen baseline graph",
                "state_migration": "not_applicable_baseline_has_no_subject_state",
                "checkpoint": "parent.bin/post.bin, five fields, create-once + fresh read-back",
                "self_check_coverage": [
                    "u2_baseline_parent_checkpoint_round_trip",
                    "u2_baseline_step_requires_open_attempt",
                    "u2_baseline_first_step_displacement_is_lr",
                    "u2_baseline_policy_is_structural_zero",
                    "u2_baseline_rejected_result_shape",
                ],
                "real_step_note": (
                    "真实落参的一步被自检 tripwire 禁止（0 步契约），改由离线探针验证；"
                    "数值记在 DECISION-LEDGER D-20261111-21。"
                ),
            },
            "semantic_gaps": [
                "actor/critic 在该基线上没有对应头，两项是结构零（已声明，不冒充 U2 参数）。",
                "R05 已按预注册改查本臂执行图身份（10 项，含无训练祖先）：u2 的父是冻结初始化，"
                "只签“更新能落地、状态能往返、独立进程能复现”，不签“从已训练状态继续”，"
                "不得引用为续训证据。",
                "该图没有策略头与主体 state：policy 与 state 迁移是结构空，一步是原地更新，不做 U2 式参数迁移。",
            ],
            "blocking_gaps": [],
        }
        if not u2_graph_ok:
            u2_preflight["blocking_gaps"].append(
                "u2 执行图不是冻结的 transformer_sequence 21977/35"
            )
        if not u2_preflight["dtype_is_torch_dtype"]:
            u2_preflight["blocking_gaps"].append("dtype 字段是说明文字而非 torch dtype")
        if not u2_preflight["active_parameters_present"]:
            u2_preflight["blocking_gaps"].append("缺 active_parameters 扁平表")
        if not u2_preflight["active_set_equals_graph_inventory"]:
            u2_preflight["blocking_gaps"].append("臂行活性清单不等于执行图的参数清单")
        if u2_preflight["missing_in_model"]:
            u2_preflight["blocking_gaps"].append("臂行声明的活性参数不在执行图里")
        if lingering:
            u2_preflight["blocking_gaps"].append("臂行仍挂着 U2 参数名")
        if not u2_preflight["role_union_subset_of_active"]:
            u2_preflight["blocking_gaps"].append("角色槽参数不在活性清单里")
        if not u2_preflight["roles_match_draft"]:
            u2_preflight["blocking_gaps"].append("臂行角色槽与草稿 identity_slots 不一致")
        if u2_dead:
            u2_preflight["blocking_gaps"].append("F2 损失下有参数拿不到梯度")
        if pieces["verbatim"] <= 0.0:
            u2_preflight["blocking_gaps"].append("verbatim 项不是正数")
        if not u2_preflight["conditional_active_parameters_present"]:
            u2_preflight["blocking_gaps"].append("缺 conditional_active_parameters 文本")
        u2_preflight["executable"] = not u2_preflight["blocking_gaps"]
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
        "u2_binding_preflight": u2_preflight,
        "cpu_seconds": time.process_time() - started,
        "wall_seconds": time.monotonic() - wall,
    }
    write_once(directory / "selfcheck.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root")
    parser.add_argument("--run-contract", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.run and args.self_check:
        parser.error("choose --run or --self-check, not both")
    if args.run_contract and not args.run:
        parser.error("--run-contract requires --run")
    try:
        if args.self_check:
            report = self_check(SELFCHECK.parent)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["verdict"] == "pass" else 1
        if args.run:
            if not args.run_root:
                raise BudgetError("--run requires --run-root")
            if args.run_contract:
                summary = execute(Path(args.run_root), contract_path=args.run_contract)
            else:
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
