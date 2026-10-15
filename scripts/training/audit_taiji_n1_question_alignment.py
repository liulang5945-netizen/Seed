"""PLAN35 zero-update engineering preflight; no curriculum model queries."""

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
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.contracts import Observation  # noqa: E402
from taiji.native_evidence_fidelity import learning_loss  # noqa: E402
from taiji.native_question_aligned_evidence_candidate import (  # noqa: E402
    QuestionAlignedEvidenceCandidate,
)

REPORT = ROOT / "reports/taiji_n1_question_alignment_preflight_20261015.json"
PROTOCOL = {
    "schema": "n1-question-alignment-preflight-v1",
    "parameter_updates": 0,
    "cpu_limit_seconds": 60,
    "curriculum_queries": 0,
}


def identity():
    from taiji import native_question_aligned_evidence_candidate as model_code

    return {"runner": paired.sha(Path(__file__)), "model": paired.sha(Path(model_code.__file__))}


@torch.no_grad()
def signature(model, optimizer):
    tape = model.evidence("probe")
    logits, state, weights = model.step(tape, "probe", model.question_state(b"which?"), 256)
    second, state, _ = model.step(tape, "probe", state, ord("a"))
    return {
        "bundle": paired.fingerprint(model.bundle(optimizer)),
        "first_logits": paired.fingerprint(logits),
        "second_logits": paired.fingerprint(second),
        "alignment": paired.fingerprint(state.alignment),
        "address": paired.fingerprint(state.address),
        "weights": paired.fingerprint(weights),
        "teacher": paired.fingerprint(model.teacher_logits(tape, "probe", b"which?", b"abc")[0]),
        "raw": model.generate_trace(tape, "probe", b"which?", 8).raw.hex(),
    }


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol"] != PROTOCOL or payload["completed_updates"] != 0:
        raise ValueError("zero-update protocol required")
    model = QuestionAlignedEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    return {"signature": signature(model, optimizer), "cpu_seconds": time.process_time()}


def audit():
    directory = (
        Path(get_external_path("data/checkpoints")) / "n1_question_alignment_preflight_20261015"
    )
    if REPORT.exists() or directory.exists():
        raise FileExistsError("exclusive preflight already exists")
    frozen = identity()
    torch.manual_seed(11)
    model = QuestionAlignedEvidenceCandidate()
    model.observe("probe", Observation("text", "abc", 1, "source-1"))
    model.observe("probe", Observation("text", "de", 2, "source-2"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = paired.fingerprint(model.bundle(optimizer))
    checks = {}
    for arm in paired.ARMS:
        model.zero_grad(set_to_none=True)
        loss = learning_loss(model, "probe", b"which?", b"abc", arm=arm)
        loss.total.backward()
        checks[f"{arm}_finite"] = bool(torch.isfinite(loss.total)) and all(
            p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()
        )
        checks[f"{arm}_writer_credit"] = (
            all(p.grad is None for p in model.writer.parameters())
            if arm == "cut"
            else any(
                p.grad is not None and p.grad.abs().sum().item() > 0
                for p in model.writer.parameters()
            )
        )
    model.zero_grad(set_to_none=True)
    checks["no_updates"] = before == paired.fingerprint(model.bundle(optimizer))
    checks["parameter_count"] = sum(p.numel() for p in model.parameters()) == 44085
    if not all(checks.values()):
        raise ValueError("engineering credit or parameter guard failed")
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "zero.pt"
    expected = signature(model, optimizer)
    payload = {
        "protocol": PROTOCOL,
        "completed_updates": 0,
        "candidate": model.bundle(optimizer),
        "expected_signature": expected,
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
    checkpoint_sha = paired.sha(path)
    if time.process_time() >= 40:
        raise paired.BudgetStop("restore reserve reached")
    response = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=25,
        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
    )
    restored = json.loads(response.stdout)
    checks["independent_restore"] = restored["signature"] == expected
    checks["checkpoint_unchanged"] = paired.sha(path) == checkpoint_sha
    checks["code_unchanged"] = identity() == frozen
    cpu = time.process_time() + restored["cpu_seconds"] + 5.0
    report = {
        "protocol": PROTOCOL,
        "code_identity": frozen,
        "checks": checks,
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha,
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_scope": "fresh zero-update Adam; no populated moment continuation claimed",
        "cpu_seconds": cpu,
        "prior_failed_attempt_cpu_charge_seconds": 5.0,
        "prior_attempt_note": "Path API returns str; failed before directory/model/query creation. Prior tool wall 2.86s with one-thread environment, conservatively charged 5s; not a measured CPU reading.",
        "parameter_count": 44085,
        "added_parameters": 3616,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "curriculum_queries": 0,
        "can_promote": False,
        "complete": all(checks.values()) and cpu <= 60,
    }
    paired.write_new(REPORT, report)
    return report


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    result = restore(args.restore_worker) if args.restore_worker else audit()
    print(json.dumps(result))
    raise SystemExit(0 if result.get("complete", True) else 1)
