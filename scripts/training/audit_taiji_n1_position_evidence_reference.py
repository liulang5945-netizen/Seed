"""SPEC-N1-11 engineering audit, never a capability training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.position_evidence_reference import (  # noqa: E402
    ObservedSource,
    PositionEvidenceReference,
)

QUESTION = b"material?"
TARGET = b"copper"
LIMIT = 60.0


def digest(tensors):
    result = hashlib.sha256()
    for tensor in tensors:
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def signature(model, sources):
    tape = model.write(sources, sources[0].session)
    logits, _ = model.teacher_logits(tape, tape.session, QUESTION, TARGET)
    return {
        "parameters": digest(model.state_dict().values()),
        "logits": digest([logits]),
        "rng": digest([torch.get_rng_state()]),
        "sources": [[s.session, s.source, s.provenance, s.content.hex()] for s in sources],
    }


def update_fixture(model, sources, optimizer):
    optimizer.zero_grad(set_to_none=True)
    tape = model.write(sources, sources[0].session)
    logits, _ = model.teacher_logits(tape, tape.session, QUESTION, TARGET)
    labels = torch.tensor([*TARGET, 256])
    torch.nn.functional.cross_entropy(logits, labels).backward()
    optimizer.step()


def worker(path, resume):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, sources = PositionEvidenceReference.restore(payload)
    before = signature(model, sources)
    if resume:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(payload["optimizer"])
        update_fixture(model, sources, optimizer)
    return {"before": before, "after": signature(model, sources), "cpu": time.process_time()}


def independent(path, resume=False):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(path)]
    if resume:
        command.append("--resume")
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=45
    )
    return json.loads(result.stdout)


def save_exclusive(payload, path):
    with path.open("xb") as stream:
        torch.save(payload, stream)


def audit():
    folder = ROOT / ".seed_test_tmp/n1_position_reference_20261015"
    report = ROOT / "reports/taiji_n1_position_evidence_reference_20261015.json"
    if report.exists() or folder.exists():
        raise FileExistsError("audit artifacts already exist; no overwrite or automatic retry")
    folder.mkdir(parents=True)
    result = {"schema": "n1-position-reference-audit-v1", "capability_claim": False}
    try:
        torch.manual_seed(11)
        model = PositionEvidenceReference()
        sources = (ObservedSource("engineering", "observation", b"agent uses copper"),)
        zero = folder / "zero.pt"
        save_exclusive(model.checkpoint(sources), zero)
        initial = signature(model, sources)
        zero_worker = independent(zero)
        if zero_worker["before"] != initial:
            raise RuntimeError("zero-update independent restoration failed; updates prohibited")
        full = model.write(sources, "engineering")
        cut = model.write(sources, "engineering", cut_credit=True)
        logits, attention = model.teacher_logits(full, "engineering", QUESTION, TARGET)
        cut_logits, _ = model.teacher_logits(cut, "engineering", QUESTION, TARGET)
        torch.nn.functional.cross_entropy(logits[-1:], torch.tensor([256])).backward()
        norms = [float(state.grad.norm()) for state in (full.states[0], full.states[-1])]
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(cut_logits[-1:], torch.tensor([256])).backward()
        controls = {
            "forward_bitwise_equal": torch.equal(logits, cut_logits),
            "early_and_last_credit_finite_nonzero": all(0 < n < float("inf") for n in norms),
            "cut_write_credit_unreachable": all(s.grad is None for s in cut.states),
            "answer_steps_reread": not torch.equal(attention[0], attention[-1]),
        }
        if not all(controls.values()):
            raise RuntimeError(f"credit controls failed: {controls}")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        update_fixture(model, sources, optimizer)
        updated = folder / "one_step.pt"
        save_exclusive(model.checkpoint(sources, optimizer), updated)
        before_resume = signature(model, sources)
        resumed = independent(updated, resume=True)
        # A second step is only a continuation-equivalence fixture, not fitting.
        update_fixture(model, sources, optimizer)
        restored = resumed["before"] == before_resume and resumed["after"] == signature(
            model, sources
        )
        cpu = time.process_time() + zero_worker["cpu"] + resumed["cpu"]
        result.update(
            controls=controls,
            first_last_gradient_norms=norms,
            zero_restore_equal=True,
            optimizer_resume_equal=restored,
            independent_workers=2,
            fixture_steps_parent=2,
            fixture_steps_worker=1,
            cpu_seconds=cpu,
            cpu_limit_seconds=LIMIT,
            checkpoint_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (zero, updated)
            },
            source_sha256=hashlib.sha256(
                Path(ROOT / "taiji/position_evidence_reference.py").read_bytes()
            ).hexdigest(),
            initial=initial,
            updated=before_resume,
            native_credit_validated=False,
            passed=restored and cpu <= LIMIT,
        )
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = worker(args.worker, args.resume) if args.worker else audit()
    print(json.dumps(output, ensure_ascii=False))
    if output.get("passed") is False:
        sys.exit(1)
