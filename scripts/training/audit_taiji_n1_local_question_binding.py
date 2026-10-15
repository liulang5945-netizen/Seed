"""PLAN48 engineering: no updates or curriculum queries, full independent restore."""

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

from scripts.training import _taiji_n1_local_question_transfer as transfer  # noqa: E402
from scripts.training import run_taiji_n1_renaming as parent  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.native_evidence_candidate import Observation  # noqa: E402

paired = parent.paired
REPORT = ROOT / "reports/taiji_n1_local_question_binding_preflight_20261015.json"
PROTOCOL = dict(
    schema="n1-local-question-preflight-v1",
    parameters=47480,
    added_parameters=3395,
    cpu_limit_seconds=60,
    parameter_updates=0,
    curriculum_queries=0,
    dev_queries=0,
    holdout_queries=0,
)
SESSION = "local48-engineering"
QUESTION = b"material of alpha?"
TARGET = b"quartz"


def prepare(model):
    if SESSION in model._observations:
        return
    model.observe(SESSION, Observation("text", "alpha uses quartz.", 1, "one"))
    model.observe(SESSION, Observation("text", "bravo uses copper.", 2, "two"))


@torch.no_grad()
def signature(model, optimizer):
    tape = model.evidence(SESSION)
    logits, attention = model.teacher_logits(tape, SESSION, QUESTION, TARGET)
    initial = model.question_state(QUESTION)
    bound, alignment, salience = model.bind_question(tape, initial)
    return dict(
        bundle=paired.fingerprint(model.bundle(optimizer)),
        logits=paired.fingerprint(logits),
        attention=paired.fingerprint(attention),
        bound_hidden=paired.fingerprint(bound.hidden),
        question_states=paired.fingerprint(bound.question_states),
        local_alignment=paired.fingerprint(alignment),
        salience=paired.fingerprint(salience),
        raw=model.generate_trace(tape, SESSION, QUESTION, 32).raw.hex(),
    )


def save(model, optimizer, path):
    payload = dict(
        protocol=PROTOCOL,
        candidate=model.bundle(optimizer),
        expected_signature=signature(model, optimizer),
    )
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL:
        raise ValueError("local engineering protocol mismatch")
    model = transfer.LocalQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    return dict(signature=signature(model, optimizer), cpu_seconds=time.process_time())


def code_identity():
    paths = [
        Path(__file__),
        Path(transfer.__file__),
        ROOT / "taiji/native_local_question_evidence_candidate.py",
    ]
    return dict(
        source=parent.code_identity(),
        code_sha256={str(p.relative_to(ROOT)): paired.sha(p) for p in paths},
    )


def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive preflight; no automatic repeat")
    _, payload, source_path = parent.transfer_source()
    old = parent.ResidualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    model, optimizer = transfer.migrate(payload)
    prepare(old)
    prepare(model)
    folder = Path(get_external_path("data/checkpoints")) / "n1_local_question_preflight_20261015"
    folder.mkdir(exist_ok=False)
    budget = paired.CpuBudget(60)
    guards = {str(p): paired.sha(p) for p in (parent.pretrained.REPORT, source_path)}
    result = dict(
        **PROTOCOL,
        code_identity=code_identity(),
        source_guards=guards,
        checks={},
        status="running",
        complete=False,
        can_promote=False,
    )
    try:
        checks = result["checks"]
        checks["parameter_count"] = sum(p.numel() for p in model.parameters()) == 47480
        checks["shared_adam"] = len(optimizer.state) == len(list(old.parameters())) and all(
            int(v["step"]) == 6144 for v in optimizer.state.values()
        )
        old_tape, tape = old.evidence(SESSION), model.evidence(SESSION)
        with torch.no_grad():
            a, aw = old.teacher_logits(old_tape, SESSION, QUESTION, TARGET)
            b, bw = model.teacher_logits(tape, SESSION, QUESTION, TARGET)
            checks["zero_logits_exact"] = torch.equal(a, b) and torch.equal(aw, bw)
            checks["zero_free_exact"] = old.generate_trace(
                old_tape, SESSION, QUESTION, 32
            ) == model.generate_trace(tape, SESSION, QUESTION, 32)
        before = paired.fingerprint(model.bundle(optimizer))
        logits, _ = model.teacher_logits(model.evidence(SESSION), SESSION, QUESTION, TARGET)
        logits.square().mean().backward()
        checks["zero_fusion_credit"] = bool(model.local_fusion.weight.grad.abs().sum() > 0)
        checks["zero_projection_expected"] = bool(
            model.local_projection.weight.grad.abs().sum() == 0
        )
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            saved_weight = model.local_fusion.weight.clone()
            model.local_fusion.weight.copy_(
                torch.linspace(-0.02, 0.02, model.local_fusion.weight.numel()).reshape_as(
                    model.local_fusion.weight
                )
            )
        logits, _ = model.teacher_logits(model.evidence(SESSION), SESSION, QUESTION, TARGET)
        logits.square().mean().backward()
        checks["nonzero_local_credit"] = all(
            p.grad is not None
            and bool(torch.isfinite(p.grad).all())
            and bool(p.grad.abs().sum() > 0)
            for p in (
                model.local_projection.weight,
                model.local_salience.weight,
                model.local_log_temperature,
                model.writer.weight_ih,
                model.question.weight_ih,
            )
        )
        model.zero_grad(set_to_none=True)
        cut = model.evidence(SESSION, cut_credit=True)
        logits, _ = model.teacher_logits(cut, SESSION, QUESTION, TARGET)
        logits.square().mean().backward()
        checks["cut_writer_credit"] = model.writer.weight_ih.grad is None or bool(
            model.writer.weight_ih.grad.abs().sum() == 0
        )
        with torch.no_grad():
            model.local_fusion.weight.copy_(saved_weight)
        model.zero_grad(set_to_none=True)
        checks["synthetic_probe_unchanged"] = before == paired.fingerprint(model.bundle(optimizer))
        if not all(checks.values()):
            raise ValueError("engineering credit/neutrality checks failed")
        budget.guard(reserve=10)
        saved = save(model, optimizer, folder / "zero.pt")
        budget.accounting_complete = False
        response = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--restore-worker",
                str(folder / "zero.pt"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=min(30, budget.limit - budget.used() - 2),
            env=dict(
                os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
            ),
        )
        worker = json.loads(response.stdout)
        budget.children += worker["cpu_seconds"]
        budget.accounting_complete = True
        checks["independent_restore"] = worker["signature"] == saved["expected_signature"]
        result["restore_check"] = dict(
            path=str(folder / "zero.pt"), sha256=paired.sha(folder / "zero.pt"), **worker
        )
        result["status"] = "complete"
    except Exception as error:
        result.update(status="rejected", error=f"{type(error).__name__}: {error}")
    finally:
        result["source_unchanged"] = all(paired.sha(Path(p)) == h for p, h in guards.items())
        result["code_unchanged"] = result["code_identity"] == code_identity()
        result["cpu_seconds"] = budget.used()
        result["cpu_accounting_complete"] = budget.accounting_complete
        result["complete"] = (
            result["status"] == "complete"
            and all(result["checks"].values())
            and result["source_unchanged"]
            and result["code_unchanged"]
            and result["cpu_seconds"] <= 60
            and budget.accounting_complete
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
            json.dumps({k: output.get(k) for k in ("complete", "cpu_seconds", "checks", "error")})
        )
        raise SystemExit(0 if output["complete"] else 1)
    else:
        print(json.dumps(PROTOCOL))
