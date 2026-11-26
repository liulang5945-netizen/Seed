"""One-shot end-to-end evidence for the u2 baseline arm (evidence only).

Purpose: prove that the *code path* of a u2 attempt runs end to end on this
machine -- full ledger (run_start -> open -> close, every append fsynced), one
Adam step on the frozen 21,977/35 execution graph, checkpoint write + immediate
five-field read-back, S12 receipt, and the independent worker recomputation.

This is NOT a u2 training run and NOT a release:

  * it does not touch the 22.4.1 admission quota (the frozen admission manifest
    keeps ``attempts_consumed = 0``) -- it uses its own ledger;
  * everything is marked ``evidence_only``: run_id, run.json, the open row, the
    S12 receipt, and a post-close annotation row with CPU and ledger path;
  * the readings only certify "the code path runs". They may not be cited as
    u2 mechanism evidence or performance evidence, and they are not evidence
    that u2 is authorized.

Run: python -m scripts.training.evidence_taiji_u2_end_to_end_22_4_1
Create-once: refuses to overwrite an existing evidence run or ledger.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.training.run_taiji_u2_22_4_1 import (
    accepted_tail,
    check_selfcheck_gate,
    ledger_run,
    normalized_file_sha,
    precontract,
    s12_payload,
    worker_gate,
)
from scripts.training.u2_attempt_ledger_22_3_4_3 import AttemptLedger
from scripts.training.u2_update_executor_22_4_1 import (
    BudgetError,
    admission_manifest,
    check_r01,
    check_r02,
    check_r03,
    check_r04,
    check_r05_baseline,
    check_r07,
    check_r08,
    check_r09_baseline,
    check_r10,
    check_r11_baseline,
    check_r12,
    execution_manifest,
    load_json,
    perform_baseline,
    write_once,
)
from scripts.training.u2_update_support_22_3_4_3 import StepCounter
from scripts.training.u2_update_support_22_4_1 import (
    prepare_parent,
    recipe_digest,
    slice_digest,
)

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "plans/manifests/taiji_u2_22_4_1_u2_run_draft_20261111.json"

# 证据专用身份：run_id 含 EVIDE，账本/S12 文件名含 EVIDENCE_ONLY，两者都不出现在真实 run 里。
RUN_ID = "20261112TEVID01Z"
RUN_ROOT = ROOT / f"reports/taiji_u2_22_4_1_{RUN_ID}"
LEDGER = "reports/taiji_u2_22_4_1_EVIDENCE_ONLY_20261112_attempts.ndjson"
S12 = "reports/taiji_u2_22_4_1_EVIDENCE_ONLY_20261112_s12_u2-strong-baseline.json"
S12_WORKER = "reports/taiji_u2_22_4_1_EVIDENCE_ONLY_20261112_s12_u2-strong-baseline_worker.json"
REPORT = ROOT / "reports/taiji_u2_22_4_1_end_to_end_evidence_20261112.json"
OWNER_REF = "evidence-only:2026-11-12#u2-strong-baseline"

LIMITATIONS = [
    "本读数只签“代码路径可跑通”：账本三步齐全、一步能落地、checkpoint 能写能读回、S12 可被独立 worker 复算。",
    "不得当作 u2 的机制证据：本 run 的父是冻结初始化（training_runs=0、optimizer_steps=0），"
    "手机制不能用它证明“从已训练状态继续”或“学到了什么”。",
    "不得当作性能证据：单步 CPU/墙钟只说明它没超预算，不是吞吐或收敛速度。",
    "不得当作 u2 放行依据：授权是证据专用（evidence_only），真实运行需要 owner 另给 run_id 与 owner_approval_ref。",
]


def sandbox_run() -> dict[str, Any]:
    """把未授权草稿改造成证据专用 run：只改身份与授权字段，判据与额度原样不动。"""
    run = load_json(DRAFT)
    run.update(
        {
            "draft": False,
            "status": "evidence_only_sandbox",
            "authorized": True,
            "authorization_note": (
                "证据专用沙箱授权：仅用于证明 u2 的代码路径可跑通，不构成 u2 训练放行，"
                "不得被复制为真实 run.json。"
            ),
            "run_id": RUN_ID,
            "attempts_authorized": 1,
            "optimizer_steps_authorized": 1,
            "owner_approval_ref": OWNER_REF,
            "evidence_only": True,
            "evidence_kind": "end_to_end_sandbox_20261112",
            "evidence_scope": "code_path_only; not mechanism evidence; not performance evidence",
        }
    )
    run["paths"] = {
        "run_json": RUN_ROOT.relative_to(ROOT).as_posix() + "/run.json",
        "run_root": RUN_ROOT.relative_to(ROOT).as_posix() + "/",
        "ledger": LEDGER,
        "s12": S12,
        "s12_worker": S12_WORKER,
    }
    return run


def refuse_if_present() -> None:
    existing = [
        path
        for path in (RUN_ROOT, ROOT / LEDGER, ROOT / S12, ROOT / S12_WORKER, REPORT)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "one-shot evidence already exists; delete it manually before re-running: "
            + ", ".join(path.relative_to(ROOT).as_posix() for path in existing)
        )


def main() -> int:
    started, wall = time.process_time(), time.monotonic()
    check_selfcheck_gate()
    torch.set_num_threads(1)
    refuse_if_present()

    run = sandbox_run()
    identity = run["update_id"]
    RUN_ROOT.mkdir(parents=True)
    run_path = RUN_ROOT / "run.json"
    write_once(run_path, run)

    checks: list[dict[str, Any]] = [
        check_r01(run, RUN_ROOT.name),
        check_r02(run),
        check_r03(run),
        check_r04(run),
    ]
    execution, _ = execution_manifest(run)
    admission, _ = admission_manifest(run)
    # R06 是 22.4.1 的价目/额度闸门；本证据 run 明确不计费（独立账本、admission 不动），
    # 因此不套用 R06，而是把"该闸门对 u2 目前判不过"这件事作为发现如实记录（见 findings）。
    u2_entry = next(row for row in admission["updates"] if row["id"] == identity)
    r06_scope = {
        "id": "R06",
        "applied": False,
        "reason": (
            "evidence-only ledger: this run is outside the 22.4.1 attempt quota, so the "
            "price/quota gate does not apply; caps are still measured and reported below"
        ),
        "frozen_admission": {
            "cpu_cap": None if u2_entry is None else u2_entry["cpu_cap"],
            "wall_cap": None if u2_entry is None else u2_entry["wall_cap"],
            "optimizer_steps": None if u2_entry is None else u2_entry["optimizer_steps"],
            "precision": None if u2_entry is None else u2_entry["precision"],
            "quota_attempts_consumed": admission["quota"]["attempts_consumed"],
        },
    }
    checks.append(r06_scope)

    runtime = prepare_parent(execution, RUN_ROOT, arm=identity)
    checks.insert(4, check_r05_baseline(run, execution, runtime, arm=identity))
    checks.append(check_r08(execution, runtime.model, runtime.optimizer, arm=identity))

    recipe_sha = recipe_digest(execution)
    slice_sha = slice_digest(execution, runtime.model, arm=identity)
    scope = run["data_scope"]
    if scope.get("data_recipe_sha256") is None and scope.get("slice_sha256") is None:
        scope["data_recipe_sha256"] = recipe_sha
        scope["slice_sha256"] = slice_sha
        write_once(run_path.with_name("run.json.next"), run)
        os.replace(run_path.with_name("run.json.next"), run_path)
    checks.append(check_r07(run, recipe_sha, slice_sha))

    ledger = AttemptLedger(
        ROOT / LEDGER,
        {**ledger_run(run), "evidence_only": True},
        execution["authority"]["source_pins"],
        precontract(run),
    )
    ledger.start()
    checks.append(check_r10(ledger))

    parent_path = RUN_ROOT / "parent.bin"
    proof, r09 = check_r09_baseline(parent_path, runtime, execution, arm=identity)
    r09["checkpoint"] = proof["sha256"]
    checks.append(r09)

    counter = StepCounter(ledger, RUN_ROOT / "claims")
    post_path = RUN_ROOT / "post.bin"
    original_step = torch.optim.Adam.step
    # open 行的字段在第一次 optimizer.step 之前落盘并 fsync；evidence 标记就在这一行。
    ledger.open(
        identity,
        parent_checkpoint="parent.bin",
        evidence_only=True,
        evidence_kind=run["evidence_kind"],
        evidence_run_id=RUN_ID,
        evidence_ledger=LEDGER,
    )
    counter.current = identity
    torch.optim.Adam.step = counter.wrapper(original_step)
    try:
        result = perform_baseline(runtime, ledger, counter, execution, post_path, arm=identity)
        checks.append(check_r11_baseline(post_path, result["post"], execution, arm=identity))
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
        s12["evidence_only"] = True
        s12["evidence_kind"] = run["evidence_kind"]
        s12["limitations"] = list(LIMITATIONS)
        accepted_tail(
            ledger=ledger,
            identity=identity,
            s12=s12,
            destination=ROOT / S12,
        )
    finally:
        torch.optim.Adam.step = original_step

    worker = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.training.verify_taiji_u2_s12_22_4_1",
            str(ROOT / S12),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    worker_report = load_json(ROOT / S12_WORKER) if (ROOT / S12_WORKER).exists() else None
    worker_gate(worker.returncode, worker_report, worker.stderr)

    total_cpu = time.process_time() - started + worker_report.get("cpu_seconds", 0.0)
    total_wall = time.monotonic() - wall
    # close 之后的非结算行：记清“哪一次、花多少 CPU、哪本账本”，不改已用额度。
    ledger.annotate(
        identity,
        evidence_only=True,
        evidence_kind=run["evidence_kind"],
        evidence_run_id=RUN_ID,
        ledger_path=LEDGER,
        cpu_seconds=total_cpu,
        wall_seconds=total_wall,
        note="one-shot end-to-end evidence; not a u2 attempt, not a release",
    )
    state = ledger.read()

    activity = result["activity_receipt"]
    report = {
        "schema": "taiji-u2-22.4.1-end-to-end-evidence-v1",
        "evidence_only": True,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": (
            "证明 u2 的代码路径在本机可端到端跑通：账本 run_start→open→close 齐全、"
            "一步落参、checkpoint 写读五项、S12 与独立 worker 复算一致。"
        ),
        "run_id": RUN_ID,
        "update_id": identity,
        "arm": identity,
        "command": "python -m scripts.training.evidence_taiji_u2_end_to_end_22_4_1",
        "owner_approval_ref": OWNER_REF,
        "run_root": run["paths"]["run_root"],
        "run_json_sha256": normalized_file_sha(run_path),
        "ledger": {
            "path": LEDGER,
            "sha256": normalized_file_sha(ROOT / LEDGER),
            "rows": len(state["rows"]),
            "events": [row["event"] for row in state["rows"]],
            "consumed": state["consumed"],
            "head": state["head"],
            "durability": "every row appended with os.fsync before the next event",
        },
        "s12": {
            "path": S12,
            "sha256": normalized_file_sha(ROOT / S12),
            "parent_sha256": s12["parent_sha256"],
            "post_sha256": s12["post_sha256"],
            "status": s12["status"],
        },
        "worker": {
            "path": S12_WORKER,
            "sha256": normalized_file_sha(ROOT / S12_WORKER),
            "verdict": worker_report["verdict"],
            "status": worker_report["status"],
            "adam_equations": worker_report.get("adam_equations"),
            "parameter_rows": worker_report.get("parameter_rows"),
            "checkpoints": worker_report.get("checkpoints"),
            "optimizer_step_calls": worker_report.get("optimizer_step_calls"),
        },
        "checkpoints": {
            "parent": {
                "path": "parent.bin",
                "sha256": r09["checkpoint"],
                "five_fields": result["pre"],
            },
            "post": {
                "path": "post.bin",
                "sha256": result["post_sha256"],
                "five_fields": result["post"],
            },
            "readback": "fresh rebuild from the frozen identity, then load state/moments",
        },
        "activity": {
            "count": len(activity["parameters"]),
            "floors": activity["floors"],
            "min_margin": activity["min_margin"],
            "barely_live_parameters": activity["barely_live_parameters"],
            "parameters": activity["parameters"],
        },
        "losses": result["losses"],
        "losses_after": result["losses_after"],
        "gradient_sha256": result["gradient_sha256"],
        "execution_graph": result["execution_graph"],
        "run_conditions": result["run_conditions"],
        "checks": checks,
        "cpu_seconds": total_cpu,
        "wall_seconds": total_wall,
        "limits": {
            "cpu_cap": run["cpu_cap"],
            "wall_cap": run["wall_cap"],
            "within_cap": total_cpu <= run["cpu_cap"] and total_wall <= run["wall_cap"],
            "attempts_authorized": run["attempts_authorized"],
            "optimizer_steps_authorized": run["optimizer_steps_authorized"],
            "admission_attempts_consumed": admission["quota"]["attempts_consumed"],
        },
        "limitations": LIMITATIONS,
        "findings": [
            {
                "id": "F-20261112-E2E-01",
                "topic": "R06 对 u2 的 precision 断言目前无法成立",
                "evidence": {
                    "run_precision": run["precision"],
                    "admission_precision": u2_entry["precision"],
                    "rule": "check_r06: run.precision == admission entry precision（逐字相等）",
                },
                "impact": (
                    "真实 u2 run.json 写 float32 会在 R06 退 3；照抄文案又会让 dtype 相关断言不成立。"
                    "本轮证据因不计费而不套用 R06，但真实运行必须先把这条修好。"
                ),
                "status": "open_owner_decision",
                "options": [
                    "A 把 admission 的 u2 行 precision 改成实测 float32（改冻结文件与其 pin，需 owner 签字）",
                    "B 为 u2 另立 R06 口径的预注册（换被检查对象，退码与编号不动）",
                    "C 在契约里写明 u2 不适用 R06（不推荐：价目闸门会对 u2 失效）",
                ],
                "recommended": "A：该行是身份落定前的占位文案，身份已冻结为 float32，改回实测值不改任何判据也不弱化闸门。",
            }
        ],
        "audit_distinction": {
            "evidence_ledger": LEDGER,
            "real_run_ledger_pattern": "reports/taiji_u2_22_4_1_attempts_<run_id>.ndjson",
            "markers": [
                f"run_id={RUN_ID} 含 TEVID；真实 run_id 为 2026MMDDTHHMMSSZ",
                "run.json 带 evidence_only=true 与 status=evidence_only_sandbox",
                "账本 run_start 的 run_id 同上，open 行带 evidence_only=true 与 evidence_ledger",
                "close 之后有一行 annotation（真实 run 不会出现该事件），记 CPU 与账本路径",
                "账本与 S12 文件名含 EVIDENCE_ONLY；真实运行不带该词",
                "准入 manifest 的 attempts_consumed 仍为 0，output/ 下没有新增真实 run 根",
            ],
            "not_usable_as": [
                "u2 的机制证据（父是冻结初始化，无训练祖先）",
                "u2 的性能证据（只测了单步可跑通）",
                "u2 的放行依据（授权仅限证据）",
            ],
        },
    }
    if not report["limits"]["within_cap"]:
        raise BudgetError(
            f"evidence run exceeded its frozen price: cpu {total_cpu:.3f} / wall {total_wall:.3f}"
        )
    write_once(REPORT, report)

    print(
        json.dumps(
            {
                "status": "evidence_accepted",
                "report": REPORT.relative_to(ROOT).as_posix(),
                "run_root": report["run_root"],
                "ledger": LEDGER,
                "s12": S12,
                "worker": S12_WORKER,
                "worker_verdict": worker_report["verdict"],
                "adam_equations": worker_report.get("adam_equations"),
                "activity_rows": len(activity["parameters"]),
                "barely_live_parameters": activity["barely_live_parameters"],
                "min_margin": activity["min_margin"],
                "cpu_seconds": total_cpu,
                "wall_seconds": total_wall,
                "evidence_only": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - fail closed, keep whatever landed
        print(
            json.dumps(
                {
                    "status": "failed",
                    "exit_code": getattr(error, "exit_code", 3),
                    "error": f"{type(error).__name__}: {error}",
                    "note": "partial artifacts are kept on purpose; delete manually to retry",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(getattr(error, "exit_code", 3)) from error
