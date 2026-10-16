"""PLAN81 synthetic-only full fresh quartet save/recovery preflight."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.training import _taiji_n1_transition_factorial_learning as learning  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

train = learning.train
engineering = train.engineering
paired = train.paired
torch = train.torch
factorial = learning.factorial
REPORT = ROOT / "reports/taiji_n1_transition_factorial_preflight_20261016.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
FACTORIAL_SHA = "842d34edee083ffaa82f7d4e83349d97a688804f4acd679d425636ab7fe13290"
PROTOCOL = dict(
    schema="n1-transition-factorial-preflight-v1",
    seed=13,
    arm="position",
    parameters=62056,
    initialization=engineering.INITIALIZATION,
    learning=learning.POLICY,
    main_trajectory_updates=0,
    signature_clone_updates=4,
    synthetic_qa_sequences=16,
    cpu_limit_seconds=120,
    curriculum_queries=0,
    dev_queries=0,
    holdout_queries=0,
    confirmation_queries=0,
)
SYNTHETIC = dict(
    id="synthetic81-transition",
    kind="transition",
    split="train",
    observations=["synthetic faces north.", "synthetic turns right."],
    question="direction of synthetic?",
    answer="east",
)


def identity():
    return dict(protocol=PROTOCOL, learning=learning.identity(), runner=paired.sha(Path(__file__)))


def signature(model, optimizer):
    before = paired.fingerprint(model.bundle(optimizer))
    if optimizer.state or model._observations or before != engineering.initial_fingerprint(13):
        raise ValueError("exact fresh seed13 empty Adam/history required")
    with torch.random.fork_rng():
        bundle = deepcopy(model.bundle(optimizer))
        clone = engineering.LocalRelationEvidenceCandidate.restore_bundle(bundle)
        opt = torch.optim.Adam(clone.parameters(), lr=0.001)
        opt.load_state_dict(bundle["optimizer"])
        trajectory = []
        for step in (1, 2):
            loss, groups = learning.train_step(
                clone, opt, SYNTHETIC, f"signature81-{step}", "position", audit=True
            )
            if len(opt.state) != 41 or any(
                int(opt.state[p]["step"]) != step for p in clone.parameters()
            ):
                raise ValueError("all41 parameters must receive exactly one Adam proposal")
            if len(clone._observations) != 4 * step or loss["qa_sequences"] != 4:
                raise ValueError("four distinct observations and QA per update required")
            gru = math.sqrt(
                sum(
                    float(torch.linalg.vector_norm(p.grad)) ** 2
                    for n, p in clone.named_parameters()
                    if n.startswith("relation_sequence.") and p.grad is not None
                )
            )
            if (
                not math.isfinite(gru)
                or (step == 2 and gru <= 0)
                or groups["writer"]["gradient_l2"] <= 0
            ):
                raise ValueError("finite writer/second-step relation GRU credit required")
            trajectory.append(dict(loss=loss, groups=groups, relation_gru_gradient_l2=gru))
        result = dict(
            bundle=before, next_bundle=paired.fingerprint(clone.bundle(opt)), trajectory=trajectory
        )
    if paired.fingerprint(model.bundle(optimizer)) != before:
        raise ValueError("signature altered fresh parent or RNG")
    return result


def load(payload):
    if (
        payload.get("schema") != "n1-transition-factorial-zero-v1"
        or payload.get("protocol") != PROTOCOL
        or payload.get("completed_updates") != 0
    ):
        raise ValueError("strict fresh quartet zero marker required")
    bundle = payload["candidate"]
    model = engineering.LocalRelationEvidenceCandidate.restore_bundle(bundle)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(deepcopy(bundle["optimizer"]))
    if paired.fingerprint(model.bundle(optimizer)) != payload["expected_signature"]["bundle"]:
        raise ValueError("full zero restore differs")
    return model, optimizer


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = load(payload)
    return dict(
        signature=signature(model, optimizer),
        cpu_seconds=time.process_time(),
        signature_clone_updates=2,
    )


def audit():
    folder = (
        Path(get_external_path("data/checkpoints")) / "n1_transition_factorial_preflight_20261016"
    )
    if REPORT.exists() or ATTEMPT.exists() or folder.exists():
        raise FileExistsError("one exclusive preflight; no retry")
    budget = paired.CpuBudget(120)
    if paired.sha(factorial.REPORT) != FACTORIAL_SHA:
        raise ValueError("exact complete80 required")
    source = json.loads(factorial.REPORT.read_text(encoding="utf-8"))
    if not source["complete"] or source["code_identity"] != factorial.identity():
        raise ValueError("unchanged complete80 required")
    guards = dict(source["source_guards"])
    guards[str(factorial.REPORT)] = FACTORIAL_SHA
    if any(paired.sha(Path(p)) != h for p, h in guards.items()):
        raise ValueError("source chain changed")
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        status="running",
        signature_clone_updates=0,
        synthetic_qa_sequences=0,
        main_trajectory_updates=0,
        curriculum_queries=0,
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        restore_checks=[],
        can_promote=False,
    )
    factorial.write_new(
        ATTEMPT,
        dict(protocol=PROTOCOL, code_identity=result["code_identity"], source_guards=guards),
    )
    folder.mkdir(exist_ok=False)
    try:
        model, optimizer = engineering.fresh(13)
        expected = signature(model, optimizer)
        result["signature_clone_updates"] += 2
        result["synthetic_qa_sequences"] += 8
        path = folder / "s13_position_zero.pt"
        with path.open("xb") as stream:
            torch.save(
                dict(
                    schema="n1-transition-factorial-zero-v1",
                    protocol=PROTOCOL,
                    completed_updates=0,
                    candidate=model.bundle(optimizer),
                    expected_signature=expected,
                ),
                stream,
            )
        budget.guard(reserve=15)
        try:
            output = subprocess.run(
                [sys.executable, str(Path(__file__)), "--restore-worker", str(path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=min(45, budget.limit - budget.used() - 2),
                env=dict(
                    os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
                ),
            )
            recovered = json.loads(output.stdout)
        except Exception:
            budget.accounting_complete = False
            raise
        budget.children += recovered["cpu_seconds"]
        result["signature_clone_updates"] += recovered["signature_clone_updates"]
        result["synthetic_qa_sequences"] += 8
        if recovered["signature"] != expected or recovered["signature_clone_updates"] != 2:
            raise ValueError("independent two quartet steps differ")
        result["restore_checks"].append(
            dict(path=str(path), sha256=paired.sha(path), equal=True, **recovered)
        )
        result["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    result.update(
        cpu_seconds=budget.used(),
        cpu_accounting_complete=budget.accounting_complete,
        sources_unchanged=all(paired.sha(Path(p)) == h for p, h in guards.items()),
        code_unchanged=identity() == result["code_identity"],
    )
    result["complete"] = (
        result["status"] == "complete"
        and result["cpu_seconds"] <= 120
        and result["cpu_accounting_complete"]
        and result["sources_unchanged"]
        and result["code_unchanged"]
        and result["signature_clone_updates"] == 4
        and result["synthetic_qa_sequences"] == 16
        and len(result["restore_checks"]) == 1
    )
    factorial.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker:
        print(json.dumps(restore(args.restore_worker)))
    elif args.execute_approved:
        record = audit()
        print(json.dumps({k: record.get(k) for k in ("complete", "cpu_seconds", "error")}))
        raise SystemExit(0 if record["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
