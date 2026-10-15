"""Approved PLAN28/SPEC29: learned relative address preflight and train-only pilot."""

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
from scripts.training import run_taiji_n1_v3_train_calibration as calibration  # noqa: E402
from scripts.training.audit_taiji_n1_native_evidence_train_fit import (
    measure,
    summarize,
)  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.native_evidence_fidelity import (  # noqa: E402
    AUXILIARY_WEIGHT,
    LOSS_CONTRACT,
    learning_loss,
    observation_logits,
)
from taiji.native_relative_evidence_candidate import RelativeEvidenceCandidate  # noqa: E402

BASELINE = ROOT / "reports/taiji_n1_iterative_read_train_20261015.json"
BASELINE_SHA = "874d06d0452b7e09305232eed3c7d7d95a62f9de82b1971ee35bc1f49f1dff4b"
PREFLIGHT = ROOT / "reports/taiji_n1_relative_read_preflight_20261015.json"
REPORT = ROOT / "reports/taiji_n1_relative_read_train_20261015.json"
PROTOCOL = {
    **calibration.PROTOCOL,
    "schema": "n1-relative-evidence-train-v1",
    "read_hops": 2,
    "candidate_format": RelativeEvidenceCandidate.FORMAT,
    "address_mechanism": RelativeEvidenceCandidate.ADDRESS_MECHANISM,
    "parameter_count": 40469,
    "loss_contract": LOSS_CONTRACT,
    "observation_weight": AUXILIARY_WEIGHT,
    "baseline_sha256": BASELINE_SHA,
}


def code_identity():
    return {
        **calibration.code_identity(),
        "protocol": PROTOCOL,
        "code_sha256": {
            **calibration.code_identity()["code_sha256"],
            "taiji/native_relative_evidence_candidate.py": paired.sha(
                ROOT / "taiji/native_relative_evidence_candidate.py"
            ),
            "taiji/native_iterative_evidence_candidate.py": paired.sha(
                ROOT / "taiji/native_iterative_evidence_candidate.py"
            ),
            "taiji/native_evidence_fidelity.py": paired.sha(
                ROOT / "taiji/native_evidence_fidelity.py"
            ),
            str(Path(__file__).relative_to(ROOT)): paired.sha(Path(__file__)),
        },
    }


def checked_train():
    if paired.sha(calibration.MANIFEST) != calibration.MANIFEST_SHA:
        raise ValueError("frozen v3 manifest changed")
    return calibration.training_items(json.loads(calibration.MANIFEST.read_text(encoding="utf-8")))


def fresh(train):
    torch.manual_seed(11)
    model = RelativeEvidenceCandidate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    paired.observe(model, paired.project(train[0]), "zero-probe")
    return model, optimizer


def baseline_initial_match(model):
    if paired.sha(BASELINE) != BASELINE_SHA:
        raise ValueError("frozen one-hop baseline report changed")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    zero = [check for check in baseline["restore_checks"] if check["path"].endswith("zero.pt")]
    if len(zero) != 1 or not baseline["complete"]:
        raise ValueError("complete baseline and unique initial checkpoint required")
    path = Path(zero[0]["path"])
    if paired.sha(path) != zero[0]["sha256"]:
        raise ValueError("one-hop initial checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    shared = {
        key: value for key, value in model.state_dict().items() if not key.startswith("location.")
    }
    if paired.fingerprint(payload["candidate"]["state"]) != paired.fingerprint(shared):
        raise ValueError("relative model old layers must match two-hop initial parameters")
    return {
        "path": str(path),
        "sha256": zero[0]["sha256"],
        "equal": True,
        "shared_parameter_count": sum(v.numel() for v in shared.values()),
    }


@torch.no_grad()
def signature(model, optimizer):
    original = calibration.signature(model, optimizer)
    tape = model.evidence("zero-probe")
    logits, targets = observation_logits(model, tape, "zero-probe")
    _, state, _ = model.step(tape, "zero-probe", model.question_state(b"material?"), 256)
    second_logits, state, _ = model.step(tape, "zero-probe", state, ord("a"))
    return {
        **original,
        "observation_logits": paired.fingerprint(logits),
        "targets": paired.fingerprint(targets),
        "second_logits": paired.fingerprint(second_logits),
        "address_state": paired.fingerprint(state.address),
    }


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
        raise ValueError("fidelity checkpoint training protocol mismatch")
    model = RelativeEvidenceCandidate.restore_bundle(payload["candidate"])
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
        raise ValueError("fidelity restoration differs from live parent")
    return {"path": str(path), "sha256": paired.sha(path), **result, "equal": True}


def train_step(model, optimizer, item, session):
    if item["split"] != "train":
        raise ValueError("fidelity updates require train episodes")
    data = paired.project(item)
    paired.observe(model, data, session)
    optimizer.zero_grad(set_to_none=True)
    losses = learning_loss(model, session, data.question, item["answer"].encode())
    if not torch.isfinite(losses.total):
        raise ValueError("nonfinite fidelity loss")
    losses.total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("nonfinite fidelity gradient")
    optimizer.step()
    return {name: float(getattr(losses, name).detach()) for name in ("total", "qa", "observation")}


def preflight():
    if PREFLIGHT.exists():
        raise FileExistsError("fidelity preflight exists; no overwrite or repeat")
    train = checked_train()
    budget = paired.CpuBudget(limit=60)
    result = {
        "schema": "n1-relative-evidence-preflight-v1",
        "code_identity": code_identity(),
        "manifest_sha256": calibration.MANIFEST_SHA,
        "rows": [],
        "optimizer_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "status": "running",
    }
    try:
        for kind in paired.KINDS:
            budget.guard()
            item = next(item for item in train if item["kind"] == kind)
            model, _ = fresh(train)
            session = f"preflight-{kind}"
            paired.observe(model, paired.project(item), session)
            before = paired.fingerprint(model.state_dict())
            losses = {}
            for arm in paired.ARMS:
                model.zero_grad(set_to_none=True)
                loss = learning_loss(
                    model, session, item["question"].encode(), item["answer"].encode(), arm=arm
                )
                loss.total.backward()
                early = loss.tape.states[0].grad
                early_norm = None if early is None else float(early.norm())
                if arm == "cut":
                    if early is not None or any(
                        p.grad is not None for p in model.writer.parameters()
                    ):
                        raise ValueError("auxiliary objective bypassed the writer credit cut")
                elif early_norm is None or early_norm <= 0:
                    raise ValueError("early writer credit missing")
                losses[arm] = loss
                result["rows"].append(
                    {
                        "kind": kind,
                        "arm": arm,
                        "early_grad_norm": early_norm,
                        "total_loss": float(loss.total.detach()),
                        "observation_loss": float(loss.observation.detach()),
                    }
                )
            if not torch.equal(losses["position"].total, losses["cut"].total):
                raise ValueError("credit cut changed the forward objective")
            if not torch.equal(losses["position"].observation, losses["terminal"].observation):
                raise ValueError("terminal control was denied the same observation objective")
            if paired.fingerprint(model.state_dict()) != before:
                raise ValueError("preflight changed parameters")
        model, optimizer = fresh(train)
        result["baseline_initial_match"] = baseline_initial_match(model)
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        folder = Path(get_external_path("data/checkpoints")) / "n1_relative_read_preflight_20261015"
        folder.mkdir(exist_ok=False)
        saved = save(model, optimizer, folder / "zero.pt", 0)
        result["restore"] = independent(folder / "zero.pt", saved["expected_signature"], budget)
        result["status"] = "complete"
        result["parameters_unchanged"] = True
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        result.update(
            cpu_seconds=budget.used(),
            cpu_accounting_complete=budget.accounting_complete,
            code_unchanged=code_identity() == result["code_identity"],
        )
        result["passed"] = (
            result["status"] == "complete"
            and len(result["rows"]) == 12
            and result["cpu_seconds"] <= 60
            and result["cpu_accounting_complete"]
            and result["code_unchanged"]
        )
        paired.write_new(PREFLIGHT, result)
    return result


@torch.no_grad()
def observe_metrics(model, session):
    logits, labels = observation_logits(model, model.evidence(session), session)
    return {
        "observation_tokens": len(labels),
        "observation_correct": int((logits.argmax(dim=-1) == labels).sum()),
        "observation_ce": float(torch.nn.functional.cross_entropy(logits, labels)),
    }


def audit():
    if REPORT.exists():
        raise FileExistsError("fidelity training report exists; no overwrite or repeat")
    ready = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if not ready["passed"] or ready["code_identity"] != code_identity():
        raise ValueError("matching successful fidelity preflight required before training")
    train = checked_train()
    model, optimizer = fresh(train)
    initial = baseline_initial_match(model)
    folder = Path(get_external_path("data/checkpoints")) / "n1_relative_read_train_20261015"
    folder.mkdir(exist_ok=False)
    budget = paired.CpuBudget(limit=300)
    result = {
        "protocol": PROTOCOL,
        "code_identity": code_identity(),
        "preflight_sha256": paired.sha(PREFLIGHT),
        "baseline_initial_match": initial,
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
        saved = save(model, optimizer, folder / "zero.pt", 0)
        result["restore_checks"].append(
            independent(folder / "zero.pt", saved["expected_signature"], budget)
        )
        for step in range(PROTOCOL["updates"] + 1):
            if step in calibration.MILESTONES:
                before = paired.fingerprint(model.state_dict())
                rows = []
                for item in train:
                    budget.guard()
                    session = f"relative-eval-{step}-{item['id']}"
                    row = measure(model, item, "position", session)
                    rows.append(
                        dict(
                            row,
                            **observe_metrics(model, session),
                            seed=11,
                            arm="position",
                            completed_updates=step,
                        )
                    )
                if paired.fingerprint(model.state_dict()) != before:
                    raise ValueError("fidelity evaluation changed parameters")
                result["rows"].extend(rows)
                summary = summarize(rows)["11"]["position"]
                summary["observation_accuracy"] = sum(r["observation_correct"] for r in rows) / sum(
                    r["observation_tokens"] for r in rows
                )
                summary["mean_observation_ce"] = sum(r["observation_ce"] for r in rows) / len(rows)
                result["milestones"].append(dict(summary, completed_updates=step))
                print(
                    f"relative step={step} train_exact={summary['free_exact']}/192 observation={summary['observation_accuracy']:.4f} cpu={budget.used():.2f}",
                    flush=True,
                )
            if step == PROTOCOL["updates"]:
                break
            budget.guard()
            result["losses"].append(
                train_step(model, optimizer, train[step % len(train)], f"relative-train-{step}")
            )
            result["completed_updates"] += 1
        saved = save(model, optimizer, folder / "final.pt", PROTOCOL["updates"])
        result["restore_checks"].append(
            independent(folder / "final.pt", saved["expected_signature"], budget)
        )
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
            manifest_unchanged=paired.sha(calibration.MANIFEST) == calibration.MANIFEST_SHA,
            baseline_unchanged=paired.sha(BASELINE) == BASELINE_SHA,
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
            and result["baseline_unchanged"]
        )
        result["training_sufficient"] = result["complete"] and calibration.sufficient(
            result["milestones"][-1]
        )
        paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-worker", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker:
        print(json.dumps(restore(args.restore_worker)))
    elif args.preflight:
        output = preflight()
        print(json.dumps({key: output[key] for key in ("passed", "cpu_seconds", "rows")}))
        raise SystemExit(0 if output["passed"] else 1)
    elif args.execute_approved:
        output = audit()
        print(json.dumps({key: output[key] for key in ("complete", "cpu_seconds", "milestones")}))
        raise SystemExit(0 if output["complete"] else 1)
    else:
        print(json.dumps({"read_only": True, "protocol": PROTOCOL}))
