"""SPEC-N1-13: self-contained candidate recovery, no capability training."""

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

from scripts.training.audit_taiji_n1_position_evidence_reference import (  # noqa: E402
    digest,
    save_exclusive,
    update_fixture,
)
from taiji.contracts import Observation  # noqa: E402
from taiji.native_evidence_candidate import NativeEvidenceCandidate  # noqa: E402


def signature(model):
    return {
        "parameters": digest(model.state_dict().values()),
        "rng": digest([torch.get_rng_state()]),
        "sessions": model.bundle()["sessions"],
        "logits": {
            session: digest(
                [model.teacher_logits(model.evidence(session), session, b"material?", b"metal")[0]]
            )
            for session in ("engineering", "separate")
        },
    }


def worker(path, resume):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = NativeEvidenceCandidate.restore_bundle(payload)
    before = signature(model)
    if resume:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(payload["optimizer"])
        update_fixture(model, model.sources("engineering"), optimizer)
    return {"before": before, "after": signature(model), "cpu": time.process_time()}


def independent(path, resume=False):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(path)]
    if resume:
        command.append("--resume")
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=45, check=True
    )
    return json.loads(result.stdout)


def audit():
    folder = ROOT / ".seed_test_tmp/n1_native_evidence_restore_20261015"
    report = ROOT / "reports/taiji_n1_native_evidence_restore_20261015.json"
    if folder.exists() or report.exists():
        raise FileExistsError("candidate audit refuses overwrite or automatic retry")
    folder.mkdir(parents=True)
    result = {
        "schema": "n1-native-evidence-recovery-v1",
        "capability_claim": False,
        "cpu_limit_seconds": 60,
    }
    try:
        torch.manual_seed(13)
        model = NativeEvidenceCandidate()
        model.observe("engineering", Observation("text", "agent uses copper", 1, "source"))
        model.observe("engineering", Observation("text", "other uses zinc", 2, "second"))
        model.observe("separate", Observation("text", "agent uses silver", 1, "source"))
        zero = folder / "zero.pt"
        save_exclusive(model.bundle(), zero)
        initial = signature(model)
        checked = independent(zero)
        if checked["before"] != initial:
            raise RuntimeError("candidate zero-update recovery failed; no updates allowed")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        update_fixture(model, model.sources("engineering"), optimizer)
        updated = folder / "one_step.pt"
        save_exclusive(model.bundle(optimizer), updated)
        before_resume = signature(model)
        resumed = independent(updated, resume=True)
        update_fixture(model, model.sources("engineering"), optimizer)
        equal = resumed["before"] == before_resume and resumed["after"] == signature(model)
        cpu = time.process_time() + checked["cpu"] + resumed["cpu"]
        result.update(
            zero_restore_equal=True,
            optimizer_resume_equal=equal,
            cpu_seconds=cpu,
            passed=equal and cpu <= 60,
            fixture_steps_parent=2,
            fixture_steps_worker=1,
            independent_workers=2,
            initial=initial,
            updated=before_resume,
            checkpoint_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (zero, updated)
            },
            code_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    ROOT / "taiji/native_evidence_candidate.py",
                    ROOT / "taiji/position_evidence_reference.py",
                    Path(__file__),
                )
            },
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
