"""PLAN60: synthetic zero-update credit and independent checkpoint recovery."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import _taiji_n1_local_relation_transfer as transfer  # noqa: E402
from scripts.training import audit_taiji_n1_bound_question_interchange as previous  # noqa: E402
from scripts.training import run_taiji_n1_contextual_question as source  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.native_evidence_candidate import Observation  # noqa: E402

paired = source.paired
REPORT = ROOT / "reports/taiji_n1_local_relation_preflight_20261015.json"
SESSIONS = ("relation60-reference", "relation60-learner")
QUESTIONS = (b"material of alpha?", b"material of delta?")


def prepare(model):
    for session, name in zip(SESSIONS, ("alpha", "delta"), strict=True):
        if session in model._observations:
            raise ValueError("engineering sessions must be fresh")
        model.observe(session, Observation("text", f"{name} uses quartz.", 1, "one"))
        model.observe(session, Observation("text", "bravo uses copper.", 2, "two"))


@torch.no_grad()
def signature(model, optimizer):
    return dict(
        bundle=paired.fingerprint(model.bundle(optimizer)),
        logits=[
            paired.fingerprint(model.teacher_logits(model.evidence(s), s, q, b"quartz"))
            for s, q in zip(SESSIONS, QUESTIONS, strict=True)
        ],
    )


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol"] != "local-relation60-engineering-v1":
        raise ValueError("wrong engineering checkpoint")
    model = transfer.LocalRelationEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    if len(optimizer.state) != len(list(model.parameters())) - len(transfer.NEW_PARAMETERS):
        raise ValueError("full Adam required")
    return dict(signature=signature(model, optimizer), cpu_seconds=time.process_time())


def identity():
    return dict(
        source=source.code_identity(),
        code={
            str(p.relative_to(ROOT)): paired.sha(p)
            for p in (
                Path(__file__),
                Path(transfer.__file__),
                ROOT / "taiji/native_local_relation_evidence_candidate.py",
            )
        },
    )


def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive engineering report")
    if paired.sha(source.REPORT) != previous.SOURCE_SHA:
        raise ValueError("52 report digest mismatch")
    record = json.loads(source.REPORT.read_text(encoding="utf-8"))
    if not record["complete"] or record["code_identity"] != source.code_identity():
        raise ValueError("complete unchanged52 required")
    (entry,) = [e for e in record["restore_checks"] if Path(e["path"]).name == "final.pt"]
    path = Path(entry["path"])
    if paired.sha(path) != entry["sha256"]:
        raise ValueError("source checkpoint mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol"] != source.PROTOCOL or payload["completed_updates"] != 6144:
        raise ValueError("fixed52 final required")
    old = source.ContextualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    model, optimizer = transfer.migrate(payload)
    original = paired.fingerprint(model.bundle(optimizer))
    prepare(model)
    before = signature(model, optimizer)
    result = dict(
        schema="local-relation60-preflight-v1",
        parameter_updates=0,
        curriculum_queries=0,
        free_queries=0,
        dev_queries=0,
        holdout_queries=0,
        code_identity=identity(),
        source_guards={
            str(p): paired.sha(p)
            for p in (
                source.REPORT,
                path,
                source.calibration.MANIFEST,
                source.PAIR_MANIFEST,
                source.CURRICULUM,
            )
        },
        checks={},
        complete=False,
        status="running",
        can_promote=False,
    )
    child_cpu = 0
    try:
        checks = result["checks"]
        checks["parameters"] = sum(p.numel() for p in model.parameters()) == 62056
        checks["old_adam"] = len(optimizer.state) == len(list(old.parameters()))
        prepare(old)
        checks["zero_logits_addresses_exact"] = all(
            paired.fingerprint(model.teacher_logits(model.evidence(s), s, q, b"quartz"))
            == paired.fingerprint(old.teacher_logits(old.evidence(s), s, q, b"quartz"))
            for s, q in zip(SESSIONS, QUESTIONS, strict=True)
        )
        model.zero_grad(set_to_none=True)
        logits, _ = model.teacher_logits(
            model.evidence(SESSIONS[0]), SESSIONS[0], QUESTIONS[0], b"quartz"
        )
        logits.square().mean().backward()
        checks["zero_head_credit"] = bool(model.relation_projection.weight.grad.abs().sum() > 0)
        checks["zero_gru_expected"] = bool(
            model.relation_sequence.weight_ih_l0.grad.abs().sum() == 0
        )
        probe = transfer.LocalRelationEvidenceCandidate.restore_bundle(model.bundle(optimizer))
        with torch.no_grad():
            probe.relation_projection.weight.normal_(0, 0.03)
        for cut in (False, True):
            probe.zero_grad(set_to_none=True)
            logits, _ = probe.teacher_logits(
                probe.evidence(SESSIONS[0], cut_credit=cut), SESSIONS[0], QUESTIONS[0], b"quartz"
            )
            logits.square().mean().backward()
            checks[f"relation_gru_{cut}"] = bool(
                probe.relation_sequence.weight_ih_l0.grad.abs().sum() > 0
            )
            checks[f"local_credit_{cut}"] = bool(probe.local_projection.weight.grad.abs().sum() > 0)
            writer = sum(
                float(p.grad.abs().sum())
                for n, p in probe.named_parameters()
                if n.startswith("writer.") and p.grad is not None
            )
            checks[f"writer_{cut}"] = writer == 0 if cut else writer > 0
        checks["neutral"] = signature(model, optimizer) == before
        folder = (
            Path(get_external_path("data/checkpoints")) / "n1_local_relation_preflight_20261015"
        )
        folder.mkdir(exist_ok=False)
        checkpoint = folder / "zero.pt"
        with checkpoint.open("xb") as f:
            torch.save(
                dict(protocol="local-relation60-engineering-v1", candidate=model.bundle(optimizer)),
                f,
            )
        child = subprocess.run(
            [sys.executable, str(Path(__file__)), "--restore", str(checkpoint)],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        restored = json.loads(child.stdout)
        child_cpu = restored["cpu_seconds"]
        checks["independent_restore"] = restored["signature"] == before
        result["restore_check"] = dict(path=str(checkpoint), sha256=paired.sha(checkpoint))
        for session in SESSIONS:
            model._observations.pop(session)
        checks["source_bundle_unchanged"] = paired.fingerprint(model.bundle(optimizer)) == original
        if not all(checks.values()):
            raise ValueError("engineering check failed")
        result.update(complete=True, status="complete")
    except Exception as error:
        result.update(status="rejected", error=f"{type(error).__name__}: {error}")
    finally:
        result["cpu_seconds"] = time.process_time() + child_cpu
        result["sources_unchanged"] = all(
            paired.sha(Path(p)) == h for p, h in result["source_guards"].items()
        )
        result["code_unchanged"] = result["code_identity"] == identity()
        result["complete"] &= (
            result["sources_unchanged"] and result["code_unchanged"] and result["cpu_seconds"] <= 60
        )
        paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore:
        print(json.dumps(restore(args.restore)))
    elif args.execute_approved:
        output = audit()
        print(json.dumps(output))
        raise SystemExit(0 if output["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, cpu_limit_seconds=60)))
