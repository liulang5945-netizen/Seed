"""SPEC-N1-19 fixed fresh-model, train-only learning sufficiency calibration."""

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
    measure,
    summarize,
)
from scripts.training.build_taiji_n1_content_curriculum_v3 import validate  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.native_evidence_candidate import NativeEvidenceCandidate  # noqa: E402

MANIFEST = ROOT / "plans/manifests/n1_native_evidence_curriculum_v3_20261015.json"
MANIFEST_SHA = "ac56fb0aba2ef602779f58749e6f8c35324c40c433fd16ce55b89689235b0604"
MILESTONES = (0, 768, 1536, 3072, 6144)
PROTOCOL = {
    "schema": "n1-v3-train-sufficiency-v1",
    "manifest_sha256": MANIFEST_SHA,
    "seed": 11,
    "arm": "position",
    "initialization": "fresh",
    "updates": 6144,
    "milestones": list(MILESTONES),
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "dimensions": {"embedding": 16, "hidden": 32, "key": 16},
    "max_length": 64,
    "cpu_limit_seconds": 300,
    "dev_queries": 0,
    "holdout_queries": 0,
    "train_gate": {"exact": 173, "each_kind": 40},
}


def training_items(manifest):
    validate(manifest)
    return [item for item in manifest["episodes"] if item["split"] == "train"]


def code_identity():
    paths = (
        Path(__file__),
        ROOT / "scripts/training/build_taiji_n1_content_curriculum_v3.py",
        ROOT / "scripts/training/audit_taiji_n1_native_evidence_train_fit.py",
    )
    return {
        "protocol": PROTOCOL,
        "code_sha256": {
            **paired.identity()["code_sha256"],
            **{str(path.relative_to(ROOT)): paired.sha(path) for path in paths},
        },
        "python": sys.version,
        "torch": str(torch.__version__),
    }


def signature(model, optimizer):
    candidate = paired.fingerprint(model.bundle(optimizer))
    with torch.no_grad():
        logits, _, _ = model.step(
            model.evidence("zero-probe"), "zero-probe", model.question_state(b"material?"), 256
        )
    return {"candidate": candidate, "first_logits": paired.fingerprint(logits)}


def save(model, optimizer, path, completed):
    payload = {
        "protocol": PROTOCOL,
        "candidate": model.bundle(optimizer),
        "completed_updates": completed,
        "expected_signature": signature(model, optimizer),
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL:
        raise ValueError("v3 calibration checkpoint protocol mismatch")
    model = NativeEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    return {"signature": signature(model, optimizer), "cpu_seconds": time.process_time()}


def independent(path, expected, budget):
    budget.guard(reserve=10)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        response = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)],
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
        raise ValueError("restored v3 candidate differs from live parent")
    return {"path": str(path), "sha256": paired.sha(path), **result, "equal": True}


def sufficient(summary):
    return summary["free_exact"] >= 173 and all(
        count >= 40 for count in summary["free_by_kind"].values()
    )


def audit():
    report = ROOT / "reports/taiji_n1_v3_train_calibration_20261015.json"
    if report.exists():
        raise FileExistsError("v3 calibration exists; no overwrite or automatic retry")
    if paired.sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("frozen v3 manifest changed")
    train = training_items(json.loads(MANIFEST.read_text(encoding="utf-8")))
    torch.manual_seed(11)
    model = NativeEvidenceCandidate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    paired.observe(model, paired.project(train[0]), "zero-probe")
    folder = Path(get_external_path("data/checkpoints")) / "n1_v3_train_calibration_20261015"
    folder.mkdir(exist_ok=False)
    budget = paired.CpuBudget(limit=300)
    result = {
        "protocol": PROTOCOL,
        "code_identity": code_identity(),
        "rows": [],
        "losses": [],
        "milestones": [],
        "restore_checks": [],
        "completed_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "can_promote": False,
        "status": "running",
    }
    try:
        path = folder / "zero.pt"
        saved = save(model, optimizer, path, 0)
        result["restore_checks"].append(independent(path, saved["expected_signature"], budget))
        for step in range(PROTOCOL["updates"] + 1):
            if step in MILESTONES:
                before = paired.fingerprint(model.state_dict())
                rows = []
                for item in train:
                    budget.guard()
                    row = measure(model, item, "position", f"v3-eval-{step}-{item['id']}")
                    rows.append(dict(row, seed=11, arm="position", completed_updates=step))
                if paired.fingerprint(model.state_dict()) != before:
                    raise ValueError("train-only evaluation changed parameters")
                result["rows"].extend(rows)
                summary = summarize(rows)["11"]["position"]
                result["milestones"].append(dict(summary, completed_updates=step))
                print(
                    f"v3 step={step} train_exact={summary['free_exact']}/192 cpu={budget.used():.2f}",
                    flush=True,
                )
            if step == PROTOCOL["updates"]:
                break
            budget.guard()
            result["losses"].append(
                paired.train_step(
                    model, optimizer, train[step % len(train)], f"v3-train-{step}", "position"
                )
            )
            result["completed_updates"] += 1
        path = folder / "final.pt"
        saved = save(model, optimizer, path, PROTOCOL["updates"])
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
            try:
                path = folder / "partial.pt"
                save(model, optimizer, path, result["completed_updates"])
                result["partial_checkpoint"] = {"path": str(path), "sha256": paired.sha(path)}
            except Exception as error:
                result["partial_checkpoint_error"] = str(error)
        result.update(
            cpu_seconds=budget.used(),
            cpu_accounting_complete=budget.accounting_complete,
            code_unchanged=code_identity() == result["code_identity"],
            manifest_unchanged=paired.sha(MANIFEST) == MANIFEST_SHA,
        )
        result["complete"] = (
            result["status"] == "complete"
            and len(result["rows"]) == 960
            and result["completed_updates"] == 6144
            and len(result["restore_checks"]) == 2
            and result["cpu_seconds"] <= 300
            and result["cpu_accounting_complete"]
            and result["code_unchanged"]
            and result["manifest_unchanged"]
        )
        result["training_sufficient"] = result["complete"] and sufficient(result["milestones"][-1])
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
        print(json.dumps({key: output[key] for key in ("complete", "cpu_seconds", "milestones")}))
        raise SystemExit(0 if output["complete"] else 1)
