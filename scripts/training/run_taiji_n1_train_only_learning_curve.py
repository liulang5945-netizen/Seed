"""SPEC-N1-17 single-model train-only diagnostic, never a promotion trial."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import run_taiji_n1_native_evidence_paired as paired  # noqa: E402
from scripts.training.audit_taiji_n1_native_evidence_train_fit import (  # noqa: E402
    BASELINE,
    BASELINE_SHA,
    measure,
    summarize,
    training_items,
)
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.native_evidence_candidate import NativeEvidenceCandidate  # noqa: E402

MILESTONES = (0, 384, 768, 1536, 3072)
PROTOCOL = {
    "schema": "n1-train-only-learning-curve-v1",
    "seed": 11,
    "arm": "position",
    "parent_updates": 384,
    "additional_updates": 3072,
    "milestones": list(MILESTONES),
    "cpu_limit_seconds": 300,
    "holdout_queries": 0,
    "dev_queries": 0,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "manifest_sha256": paired.MANIFEST_SHA,
}


def signature(model, optimizer):
    digest = paired.fingerprint(model.bundle(optimizer))
    with torch.no_grad():
        logits, _, _ = model.step(
            model.evidence("zero-probe"), "zero-probe", model.question_state(b"material?"), 256
        )
    return {"candidate": digest, "first_logits": paired.fingerprint(logits)}


def save(model, optimizer, path, completed, parent_sha):
    payload = {
        "protocol": PROTOCOL,
        "candidate": model.bundle(optimizer),
        "additional_updates": completed,
        "parent_sha256": parent_sha,
        "expected_signature": signature(model, optimizer),
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL:
        raise ValueError("curve checkpoint protocol mismatch")
    model = NativeEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    return {"signature": signature(model, optimizer), "cpu_seconds": time.process_time()}


def independent(path, expected, budget):
    budget.guard(reserve=10)
    command = [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        response = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=min(30, budget.limit - budget.used() - 2),
            env=env,
        )
        result = json.loads(response.stdout)
    except Exception:
        budget.accounting_complete = False
        raise
    budget.children += result["cpu_seconds"]
    budget.guard()
    if result["signature"] != expected:
        raise ValueError("curve independent restoration differs from live parent")
    return {
        "path": str(path),
        "sha256": paired.sha(path),
        "cpu_seconds": result["cpu_seconds"],
        "equal": True,
    }


def audit():
    report = ROOT / "reports/taiji_n1_train_only_learning_curve_20261015.json"
    if report.exists():
        raise FileExistsError("curve report exists; automatic repeat prohibited")
    if paired.sha(BASELINE) != BASELINE_SHA:
        raise ValueError("parent formal report changed")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    parents = [
        entry
        for entry in baseline["restore_checks"]
        if entry["checkpoint"].endswith("s11_position_trained.pt")
    ]
    if len(parents) != 1:
        raise ValueError("unique seed11 position parent required")
    parent, parent_sha = Path(parents[0]["checkpoint"]), parents[0]["sha256"]
    if paired.sha(parent) != parent_sha:
        raise ValueError("frozen parent checkpoint changed")
    train = training_items(paired.checked_manifest())
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    if (payload["seed"], payload["arm"], payload["completed_updates"]) != (11, "position", 384):
        raise ValueError("unexpected parent training identity")
    model = NativeEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    folder = Path(get_external_path("data/checkpoints")) / "n1_train_only_learning_curve_20261015"
    folder.mkdir(exist_ok=False)
    budget = paired.CpuBudget()
    result = {
        "protocol": PROTOCOL,
        "parent_checkpoint": str(parent),
        "parent_sha256": parent_sha,
        "rows": [],
        "losses": [],
        "restore_checks": [],
        "milestones": [],
        "code_identity": paired.identity(),
        "audit_code_sha256": paired.sha(Path(__file__)),
        "completed_additional_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "can_promote": False,
        "status": "running",
    }
    try:
        path = folder / "zero.pt"
        saved = save(model, optimizer, path, 0, parent_sha)
        result["restore_checks"].append(independent(path, saved["expected_signature"], budget))
        for step in range(3073):
            if step in MILESTONES:
                before = paired.fingerprint(model.state_dict())
                rows = []
                for item in train:
                    budget.guard()
                    row = measure(model, item, "position", f"curve-eval-{step}-{item['id']}")
                    rows.append(dict(row, seed=11, arm="position", additional_updates=step))
                if paired.fingerprint(model.state_dict()) != before:
                    raise ValueError("learning-curve evaluation changed parameters")
                result["rows"].extend(rows)
                summary = summarize(rows)["11"]["position"]
                result["milestones"].append(dict(summary, additional_updates=step))
                print(
                    f"curve step={step} train_exact={summary['free_exact']}/96 cpu={budget.used():.2f}",
                    flush=True,
                )
            if step == 3072:
                break
            budget.guard()
            result["losses"].append(
                paired.train_step(
                    model, optimizer, train[step % 96], f"curve-train-{step}", "position"
                )
            )
            result["completed_additional_updates"] += 1
        path = folder / "final.pt"
        saved = save(model, optimizer, path, 3072, parent_sha)
        result["restore_checks"].append(independent(path, saved["expected_signature"], budget))
        result["status"] = "complete"
    except paired.BudgetStop as error:
        result.update(status="budget-stop", error=str(error))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="operator interruption; no automatic retry")
    finally:
        if result["status"] != "complete":
            path = folder / "partial.pt"
            try:
                save(model, optimizer, path, result["completed_additional_updates"], parent_sha)
                result["partial_checkpoint"] = {"path": str(path), "sha256": paired.sha(path)}
            except Exception as error:
                result["partial_checkpoint_error"] = str(error)
        result.update(
            cpu_seconds=budget.used(),
            cpu_accounting_complete=budget.accounting_complete,
            parent_unchanged=paired.sha(parent) == parent_sha,
        )
        result["code_unchanged"] = (
            paired.identity() == result["code_identity"]
            and paired.sha(Path(__file__)) == result["audit_code_sha256"]
        )
        result["complete"] = (
            result["status"] == "complete"
            and len(result["rows"]) == 480
            and result["completed_additional_updates"] == 3072
            and len(result["restore_checks"]) == 2
            and result["cpu_seconds"] <= 300
            and result["parent_unchanged"]
            and budget.accounting_complete
            and result["code_unchanged"]
        )
        paired.write_new(report, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker:
        print(json.dumps(restore(args.restore_worker)))
    else:
        output = audit()
        print(
            json.dumps(
                {key: output[key] for key in ("status", "complete", "cpu_seconds", "milestones")}
            )
        )
        raise SystemExit(0 if output["complete"] else 1)
