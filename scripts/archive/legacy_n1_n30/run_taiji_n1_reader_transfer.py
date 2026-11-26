"""Approved PLAN41: fixed 9216-update question alignment train-only calibration."""

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
from scripts.training import run_taiji_n1_relative_continuation as pretrained  # noqa: E402
from scripts.training import run_taiji_n1_v3_train_calibration as calibration  # noqa: E402
from scripts.training.audit_taiji_n1_native_evidence_train_fit import (
    measure,
    summarize,
)  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.native_evidence_fidelity import (  # noqa: E402
    AUXILIARY_WEIGHT,
    LOSS_CONTRACT,
    observation_logits,
)
from taiji.native_paired_question_learning import (  # noqa: E402
    LOSS_CONTRACT as PAIRED_LOSS_CONTRACT,
)
from taiji.native_paired_question_learning import (
    paired_question_loss,
)
from taiji.native_residual_question_evidence_candidate import (
    ResidualQuestionEvidenceCandidate,
)  # noqa: E402

BASELINE = ROOT / "reports/taiji_n1_residual_question_preflight_20261015.json"
BASELINE_SHA = "50155a01e1159aad3d78ed49dc9d6b8c177e4a48ab3c1c33e09219e3a4f0e1bd"
PREFLIGHT = BASELINE
REPORT = ROOT / "reports/taiji_n1_reader_transfer_train_20261015.json"
PAIR_MANIFEST = ROOT / "plans/manifests/n1_paired_questions_20261015.json"
PAIR_MANIFEST_SHA = "1c3857b1ae66c227503d443ea13051a4dbaa0b65707f4fb259a3fe64c222fddb"
PROTOCOL = {
    **calibration.PROTOCOL,
    "schema": "n1-reader-transfer-train-v1",
    "updates": 6144,
    "milestones": [0, 3072, 6144],
    "question_mechanism": ResidualQuestionEvidenceCandidate.QUESTION_MECHANISM,
    "initialization": "transfer-fixed-s11-relative-9216-fresh-adam-new-rng",
    "source_updates": 9216,
    "source_report_sha256": "871f054fcff6a72f713849de2a8b3b1ae6680e4581bccad317fef65a3c133045",
    "source_checkpoint_sha256": "e274c81914ccc86dacaf9cf3b7690bc8126f671b24b104e8e153f793eb422747",
    "read_hops": 2,
    "candidate_format": ResidualQuestionEvidenceCandidate.FORMAT,
    "address_mechanism": ResidualQuestionEvidenceCandidate.ADDRESS_MECHANISM,
    "parameter_count": 44085,
    "loss_contract": PAIRED_LOSS_CONTRACT,
    "prior_loss_contract": LOSS_CONTRACT,
    "paired_manifest_sha256": PAIR_MANIFEST_SHA,
    "qa_sequences": 9216,
    "pair_gate": 87,
    "observation_weight": AUXILIARY_WEIGHT,
    "baseline_sha256": BASELINE_SHA,
}


def code_identity():
    return {
        **calibration.code_identity(),
        "protocol": PROTOCOL,
        "pretrained_source_code": pretrained.identity(),
        "code_sha256": {
            **calibration.code_identity()["code_sha256"],
            "taiji/native_paired_question_learning.py": paired.sha(
                ROOT / "taiji/native_paired_question_learning.py"
            ),
            "taiji/native_residual_question_evidence_candidate.py": paired.sha(
                ROOT / "taiji/native_residual_question_evidence_candidate.py"
            ),
            "taiji/native_question_aligned_evidence_candidate.py": paired.sha(
                ROOT / "taiji/native_question_aligned_evidence_candidate.py"
            ),
            "taiji/native_relative_evidence_candidate.py": paired.sha(
                ROOT / "taiji/native_relative_evidence_candidate.py"
            ),
            "taiji/native_iterative_evidence_candidate.py": paired.sha(
                ROOT / "taiji/native_iterative_evidence_candidate.py"
            ),
            "scripts/training/audit_taiji_n1_native_evidence_train_fit.py": paired.sha(
                ROOT / "scripts/training/audit_taiji_n1_native_evidence_train_fit.py"
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


def build_pairs(train):
    from copy import deepcopy

    pairs = []
    for item in train:
        if item["split"] != "train":
            raise ValueError("pair generation only accepts train")
        if item["kind"] not in ("lookup", "unknown"):
            continue
        old = item["question"].removeprefix("material of ").removesuffix("?")
        names = [o.split()[0] for o in item["observations"]]
        if item["kind"] == "lookup":
            indices = [i for i, name in enumerate(names) if name != old]
            if old not in names or len(indices) != 1:
                raise ValueError("lookup must have exactly one alternate observed entity")
            index = indices[0]
        else:
            if old in names:
                raise ValueError("unknown question must refer to absent entity")
            index = 0
        changed = deepcopy(item)
        changed.update(
            id="pair-" + item["id"],
            parent_id=item["id"],
            original_kind=item["kind"],
            kind="lookup",
            question=f"material of {names[index]}?",
            answer=item["observations"][index].removesuffix(".").split()[-1],
        )
        pairs.append(changed)
    if len(pairs) != 96 or len({p["parent_id"] for p in pairs}) != 96:
        raise ValueError("96 unique train question pairs required")
    return pairs


def checked_pairs(train):
    if paired.sha(PAIR_MANIFEST) != PAIR_MANIFEST_SHA:
        raise ValueError("frozen pair manifest changed")
    payload = json.loads(PAIR_MANIFEST.read_text(encoding="utf-8"))
    if payload["source_manifest_sha256"] != calibration.MANIFEST_SHA or payload[
        "pairs"
    ] != build_pairs(train):
        raise ValueError("pairs must derive exclusively from frozen train")
    return payload["pairs"]


def transfer_source():
    if paired.sha(pretrained.REPORT) != PROTOCOL["source_report_sha256"]:
        raise ValueError("frozen pretrained report changed")
    record = json.loads(pretrained.REPORT.read_text(encoding="utf-8"))
    if not record["complete"] or not record["training_sufficient"]:
        raise ValueError("complete pretrained source required")
    entries = [e for e in record["frozen_models"] if e["seed"] == 11 and e["arm"] == "position"]
    if len(entries) != 1 or entries[0]["sha256"] != PROTOCOL["source_checkpoint_sha256"]:
        raise ValueError("fixed pretrained seed11 position required")
    path = Path(entries[0]["path"])
    if paired.sha(path) != entries[0]["sha256"]:
        raise ValueError("pretrained checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["protocol"] != pretrained.PROTOCOL
        or payload["completed_updates"] != 9216
        or payload["seed"] != 11
        or payload["arm"] != "position"
    ):
        raise ValueError("pretrained protocol mismatch")
    if paired.fingerprint(payload["candidate"]["state"]) != entries[0]["parameters"]:
        raise ValueError("pretrained parameter identity mismatch")
    return record, payload, path


def fresh(train):
    _, payload, _ = transfer_source()
    torch.manual_seed(11)
    model = ResidualQuestionEvidenceCandidate()
    missing, unexpected = model.load_state_dict(payload["candidate"]["state"], strict=False)
    expected = {
        "question_key.weight",
        "question_value.weight",
        "question_fusion.weight",
        "question_fusion.bias",
    }
    if (
        set(missing) != expected
        or unexpected
        or sum(v.numel() for v in payload["candidate"]["state"].values()) != 40469
    ):
        raise ValueError("exact full old reader and only new question heads required")
    if not all(
        torch.equal(model.state_dict()[k], v) for k, v in payload["candidate"]["state"].items()
    ):
        raise ValueError("pretrained weights did not copy exactly")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    paired.observe(model, paired.project(train[0]), "zero-probe")
    return model, optimizer


def transfer_equal(rows, source_record):
    old = {
        r["id"]: r
        for r in source_record["train_rows"]
        if r["seed"] == 11 and r["arm"] == "position"
    }
    if len(rows) != 192 or {r["id"] for r in rows} != set(old):
        return False
    return sum(r["free_exact"] for r in rows) == 191 and all(
        all(r[k] == old[r["id"]][k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))
        and abs(r["teacher_ce"] - old[r["id"]]["teacher_ce"]) <= 1e-6
        for r in rows
    )


def baseline_initial_match(model):
    if paired.sha(BASELINE) != BASELINE_SHA:
        raise ValueError("37 engineering evidence changed")
    ready = json.loads(BASELINE.read_text(encoding="utf-8"))
    if not ready["complete"] or ready["code_identity"]["model"] != paired.sha(
        ROOT / "taiji/native_residual_question_evidence_candidate.py"
    ):
        raise ValueError("matching complete engineering preflight required")
    if sum(p.numel() for p in model.parameters()) != 44085:
        raise ValueError("frozen parameter count changed")
    return {"engineering_sha256": BASELINE_SHA, "passed": True}


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
        "question_states": paired.fingerprint(state.question_states),
        "question_alignment": paired.fingerprint(state.alignment),
        "conditioned_values": paired.fingerprint(state.conditioned_tape.values),
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
    model = ResidualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
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


def train_step(model, optimizer, item, session, second=None):
    if item["split"] != "train":
        raise ValueError("fidelity updates require train episodes")
    if item["kind"] in ("lookup", "unknown") and second is None:
        raise ValueError("paired question required for lookup and unknown")
    if second is not None and (
        second["split"] != "train"
        or second["parent_id"] != item["id"]
        or second["observations"] != item["observations"]
    ):
        raise ValueError("second question must share the same train observations")
    data = paired.project(item)
    paired.observe(model, data, session)
    optimizer.zero_grad(set_to_none=True)
    losses = paired_question_loss(
        model,
        session,
        data.question,
        item["answer"].encode(),
        second_question=None if second is None else second["question"].encode(),
        second_target=None if second is None else second["answer"].encode(),
    )
    if not torch.isfinite(losses.total):
        raise ValueError("nonfinite fidelity loss")
    losses.total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("nonfinite fidelity gradient")
    optimizer.step()
    return {
        **{name: float(getattr(losses, name).detach()) for name in ("total", "qa", "observation")},
        "qa_sequences": 1 if second is None else 2,
    }


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
    if not ready["complete"] or paired.sha(PREFLIGHT) != BASELINE_SHA:
        raise ValueError("matching successful fidelity preflight required before training")
    train = checked_train()
    pairs = checked_pairs(train)
    pair_by_parent = {p["parent_id"]: p for p in pairs}
    model, optimizer = fresh(train)
    source_record, _, source_path = transfer_source()
    initial = baseline_initial_match(model)
    folder = Path(get_external_path("data/checkpoints")) / "n1_reader_transfer_train_20261015"
    folder.mkdir(exist_ok=False)
    budget = paired.CpuBudget(limit=300)
    result = {
        "protocol": PROTOCOL,
        "code_identity": code_identity(),
        "preflight_sha256": paired.sha(PREFLIGHT),
        "paired_manifest_sha256": PAIR_MANIFEST_SHA,
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
            if step in PROTOCOL["milestones"]:
                before = paired.fingerprint(model.state_dict())
                rows = []
                for item in train:
                    budget.guard()
                    session = f"question-alignment-eval-{step}-{item['id']}"
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
                if step == 0:
                    result["transfer_behavior_equal"] = transfer_equal(rows, source_record)
                    if not result["transfer_behavior_equal"]:
                        raise ValueError("pretrained all-192 behavior changed before training")
                for row in rows:
                    row["evaluation_set"] = "original"
                result["rows"].extend(rows)
                paired_rows = []
                for item in pairs:
                    budget.guard()
                    row = measure(
                        model, item, "position", f"paired-question-eval-{step}-{item['id']}"
                    )
                    paired_rows.append(
                        dict(
                            row,
                            seed=11,
                            arm="position",
                            completed_updates=step,
                            evaluation_set="pair",
                        )
                    )
                result["rows"].extend(paired_rows)
                summary = summarize(rows)["11"]["position"]
                summary["pair_exact"] = sum(r["free_exact"] for r in paired_rows)
                summary["pair_items"] = len(paired_rows)
                summary["observation_accuracy"] = sum(r["observation_correct"] for r in rows) / sum(
                    r["observation_tokens"] for r in rows
                )
                summary["mean_observation_ce"] = sum(r["observation_ce"] for r in rows) / len(rows)
                result["milestones"].append(dict(summary, completed_updates=step))
                print(
                    f"residual-question step={step} train_exact={summary['free_exact']}/192 observation={summary['observation_accuracy']:.4f} cpu={budget.used():.2f}",
                    flush=True,
                )
            if step == PROTOCOL["updates"]:
                break
            budget.guard()
            result["losses"].append(
                train_step(
                    model,
                    optimizer,
                    train[step % len(train)],
                    f"paired-question-train-{step}",
                    pair_by_parent.get(train[step % len(train)]["id"]),
                )
            )
            result["completed_updates"] += 1
        result["final_optimizer_state_entries"] = len(optimizer.state)
        if len(optimizer.state) != len(list(model.parameters())):
            raise ValueError("trained optimizer must contain every parameter moment")
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
            pretrained_source_unchanged=paired.sha(pretrained.REPORT)
            == PROTOCOL["source_report_sha256"]
            and paired.sha(source_path) == PROTOCOL["source_checkpoint_sha256"],
            pair_manifest_unchanged=paired.sha(PAIR_MANIFEST) == PAIR_MANIFEST_SHA,
        )
        result["complete"] = (
            result["status"] == "complete"
            and len(result["rows"]) == 864
            and result["completed_updates"] == 6144
            and sum(loss["qa_sequences"] for loss in result["losses"]) == 9216
            and len(result["restore_checks"]) == 2
            and result["cpu_seconds"] <= 300
            and result["cpu_accounting_complete"]
            and result["code_unchanged"]
            and result["manifest_unchanged"]
            and result["baseline_unchanged"]
            and result["pair_manifest_unchanged"]
            and result["pretrained_source_unchanged"]
            and result.get("transfer_behavior_equal", False)
        )
        result["original_training_sufficient"] = result["complete"] and calibration.sufficient(
            result["milestones"][-1]
        )
        result["pair_training_sufficient"] = (
            result["complete"] and result["milestones"][-1]["pair_exact"] >= 87
        )
        result["training_sufficient"] = (
            result["original_training_sufficient"] and result["pair_training_sufficient"]
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
        print(json.dumps({key: output[key] for key in ("complete", "cpu_seconds", "milestones")}))
        raise SystemExit(0 if output["complete"] else 1)
    else:
        print(json.dumps({"read_only": True, "protocol": PROTOCOL}))
