"""PLAN68: fixed seed13 calibration of the frozen actual-Adam displacement policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import audit_taiji_n1_group_relative_adam as engineering  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

bounded = engineering.bounded
runner = bounded.runner
paired = runner.paired
REPORT = ROOT / "reports/taiji_n1_group_relative_adam_train_20261015.json"
PREFLIGHT_SHA = "88313caa6a27dc9a7dfcb2340b15f1e009526fd0afa82db1e60a9bf6362d07ff"
SOURCE_CHECKPOINT_SHA = "0f1cbc2d606d5a26a155216dec17061972ce18ad87d5c80768264a857ef78293"
PROTOCOL = {
    **{
        k: v
        for k, v in runner.core.PROTOCOL.items()
        if k not in ("seed", "arm", "cpu_limit_seconds")
    },
    "schema": "n1-group-relative-adam-calibration-v1",
    "seed": 13,
    "arm": "position",
    "updates": 6144,
    "qa_sequences": 9216,
    "signature_clone_updates": 4,
    "signature_clone_qa_sequences": 8,
    "update_policy": bounded.POLICY,
    "cpu_limit_seconds": 300,
    "preflight_sha256": PREFLIGHT_SHA,
    "source_report_sha256": engineering.replay.SOURCE_SHA,
    "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA,
    "ancestor52_checkpoint_sha256": runner.core.PROTOCOL["source_checkpoint_sha256"],
    "initialization": "frozen63-s13-position-zero-full-bundle",
    "source_initialization": "frozen63-s13-position-zero-full-bundle",
    "baseline_sha256": PREFLIGHT_SHA,
    "confirmation_queries": 0,
    "dev_queries": 0,
    "holdout_queries": 0,
}


def identity():
    return dict(
        protocol=PROTOCOL, engineering=engineering.identity(), runner=paired.sha(Path(__file__))
    )


def readiness():
    if paired.sha(engineering.REPORT) != PREFLIGHT_SHA:
        raise ValueError("frozen67 engineering report required")
    evidence = json.loads(engineering.REPORT.read_text(encoding="utf-8"))
    if (
        not evidence["complete"]
        or evidence["code_identity"] != engineering.identity()
        or evidence["mechanical_trajectory_updates"] != 2
        or evidence["signature_clone_updates"] != 4
        or len(evidence["restore_checks"]) != 2
    ):
        raise ValueError("complete unchanged bounded-update engineering required")
    record, guards, paths = engineering.readiness()
    if paired.sha(paths["zero"]) != SOURCE_CHECKPOINT_SHA:
        raise ValueError("exact original63 seed13 zero bytes required")
    guards[str(engineering.REPORT)] = PREFLIGHT_SHA
    for check in evidence["restore_checks"]:
        if not check["equal"] or paired.sha(Path(check["path"])) != check["sha256"]:
            raise ValueError("67 checkpoint bytes or independent recovery changed")
        guards[check["path"]] = check["sha256"]
    return record, guards, paths


def load(payload):
    if (
        payload.get("schema") != "n1-group-relative-adam-checkpoint-v1"
        or payload.get("protocol") != PROTOCOL
        or payload.get("update_policy") != bounded.POLICY
        or payload.get("completed_updates") not in (0, 6144)
    ):
        raise ValueError("fixed calibration protocol, final phase and update marker required")
    model = runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(deepcopy(payload["candidate"]["optimizer"]))
    completed = payload["completed_updates"]
    if len(optimizer.state) != (35 if completed == 0 else 41):
        raise ValueError("all trained Adam moments required")
    for name, parameter in model.named_parameters():
        if parameter in optimizer.state and int(
            optimizer.state[parameter]["step"]
        ) != runner.expected_step(name, "position", completed):
            raise ValueError("source plus fixed updates required for each Adam state")
    bounded.grouped_parameters(model)
    return model, optimizer


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = load(payload)
    return dict(
        signature=engineering.signature(model, optimizer),
        cpu_seconds=time.process_time(),
        signature_clone_updates=1,
    )


def independent(path, expected, budget):
    budget.guard(reserve=10)
    try:
        output = subprocess.run(
            [sys.executable, str(Path(__file__)), "--restore-worker", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=min(30, budget.limit - budget.used() - 2),
            env=dict(
                os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
            ),
        )
        response = json.loads(output.stdout)
    except Exception:
        budget.accounting_complete = False
        raise
    budget.children += response["cpu_seconds"]
    budget.guard()
    if response["signature"] != expected or response["signature_clone_updates"] != 1:
        raise ValueError("independent full calibration recovery/next bounded step differs")
    return dict(path=str(path), sha256=paired.sha(path), equal=True, **response)


def audit():
    folder = Path(get_external_path("data/checkpoints")) / "n1_group_relative_adam_train_20261015"
    if REPORT.exists() or folder.exists():
        raise FileExistsError("exclusive fixed calibration, no restart or overwrite")
    budget = paired.CpuBudget(300)
    record, guards, paths = readiness()
    source = torch.load(paths["zero"], map_location="cpu", weights_only=True)
    if (
        source["protocol"] != runner.PROTOCOL
        or source["seed"] != 13
        or source["arm"] != "position"
        or source["completed_updates"] != 0
    ):
        raise ValueError("unmodified original63 seed13 zero required, no synthetic sessions")
    train = runner.core.checked_train()
    curriculum = runner.core.checked_curriculum(train)
    probes = [
        runner.core.renaming.rename_item(e, p["names"])
        for e, p in zip(train, curriculum["probes"], strict=True)
    ]
    views = dict(
        original=train,
        pair=runner.core.checked_pairs(train),
        rename=probes,
        rename_pair=runner.core.build_pairs(probes),
    )
    expected_zero = [
        r
        for r in record["rows"]
        if r["seed"] == 13 and r["arm"] == "position" and r["completed_updates"] == 0
    ]
    if len(expected_zero) != 576:
        raise ValueError("full576 native zero source rows required")
    folder.mkdir(exist_ok=False)
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        source_checkpoint=str(paths["zero"]),
        completed_updates=0,
        signature_clone_updates=0,
        losses=[],
        measurements=[],
        rows=[],
        milestones=[],
        restore_checks=[],
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        can_promote=False,
        status="running",
    )
    model = optimizer = None
    try:
        model = runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(source["candidate"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(deepcopy(source["candidate"]["optimizer"]))
        if paired.fingerprint(model.bundle(optimizer)) != paired.fingerprint(source["candidate"]):
            raise ValueError("original zero full bundle differs")
        saved = engineering.save(model, optimizer, folder / "zero.pt", 0, protocol=PROTOCOL)
        result["signature_clone_updates"] += 1
        result["restore_checks"].append(
            independent(folder / "zero.pt", saved["expected_signature"], budget)
        )
        result["signature_clone_updates"] += 1
        for step in range(6145):
            if step in (0, 3072, 6144):
                rows, summary = runner.evaluate(model, 13, "position", step, views, budget)
                if step == 0 and rows != expected_zero:
                    raise ValueError("zero full576 scores failed exact reproduction")
                result["rows"].extend(rows)
                result["milestones"].append(dict(summary, completed_updates=step))
                print(
                    f"bounded68 step={step} original={summary['free_exact']} pair={summary['pair_exact']} rename={summary['rename_free_exact']} rename_pair={summary['rename_pair_exact']} cpu={budget.used():.2f}",
                    flush=True,
                )
            if step == 6144:
                break
            budget.guard(reserve=10)
            item, second = runner.core.renaming.train_item(step, train, curriculum)
            loss, measurement = bounded.train_step(
                model, optimizer, item, f"bounded68-train-{step}", second
            )
            result["completed_updates"] += 1
            result["losses"].append(loss)
            result["measurements"].append(dict(step=step, groups=measurement))
        for name, parameter in model.named_parameters():
            if int(optimizer.state[parameter]["step"]) != runner.expected_step(
                name, "position", 6144
            ):
                raise ValueError("final Adam steps differ from fixed recipe")
        saved = engineering.save(model, optimizer, folder / "final.pt", 6144, protocol=PROTOCOL)
        result["signature_clone_updates"] += 1
        original_sessions = source["candidate"]["sessions"]
        final_sessions = saved["candidate"]["sessions"]
        result["source_sessions_unchanged"] = all(
            final_sessions.get(k) == v for k, v in original_sessions.items()
        )
        result["new_session_count"] = len(final_sessions) - len(original_sessions)
        if not result["source_sessions_unchanged"] or result["new_session_count"] != 6144:
            raise ValueError("original sessions changed or fixed history count differs")
        result["restore_checks"].append(
            independent(folder / "final.pt", saved["expected_signature"], budget)
        )
        result["signature_clone_updates"] += 1
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="operator interruption; no automatic restart")
    finally:
        if result["status"] != "complete" and model is not None and optimizer is not None:
            try:
                engineering.save(
                    model,
                    optimizer,
                    folder / "partial.pt",
                    result["completed_updates"],
                    protocol=PROTOCOL,
                )
                result["signature_clone_updates"] += 1
                result["partial_checkpoint"] = str(folder / "partial.pt")
            except Exception as error:
                result["partial_error"] = str(error)
        result.update(
            cpu_seconds=budget.used(),
            cpu_accounting_complete=budget.accounting_complete,
            sources_unchanged=all(paired.sha(Path(p)) == digest for p, digest in guards.items()),
            code_unchanged=identity() == result["code_identity"],
        )
        result["complete"] = (
            result["status"] == "complete"
            and result["completed_updates"] == 6144
            and sum(r["qa_sequences"] for r in result["losses"]) == 9216
            and len(result["measurements"]) == 6144
            and len(result["rows"]) == 1728
            and result["signature_clone_updates"] == 4
            and len(result["restore_checks"]) == 2
            and result["cpu_seconds"] <= 300
            and all(
                result[k]
                for k in (
                    "sources_unchanged",
                    "code_unchanged",
                    "cpu_accounting_complete",
                    "source_sessions_unchanged",
                )
            )
        )
        result["training_sufficient"] = result["complete"] and runner.sufficient(
            result["milestones"][-1]
        )
        paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-worker", type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker:
        print(json.dumps(restore(args.restore_worker)))
    elif args.execute_approved:
        output = audit()
        print(
            json.dumps(
                {
                    k: output.get(k)
                    for k in (
                        "complete",
                        "training_sufficient",
                        "completed_updates",
                        "cpu_seconds",
                        "error",
                    )
                }
            )
        )
        raise SystemExit(0 if output["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
