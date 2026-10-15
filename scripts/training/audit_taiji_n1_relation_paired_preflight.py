"""PLAN62: nine paired zero checkpoints, no confirmation inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import _taiji_n1_relation_paired_transfer as transfer  # noqa: E402
from scripts.training import build_taiji_n1_confirmation_v4 as face  # noqa: E402
from scripts.training import run_taiji_n1_local_relation as calibration  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.native_evidence_candidate import Observation  # noqa: E402

paired = transfer.source.paired
REPORT = ROOT / "reports/taiji_n1_relation_paired_preflight_20261015.json"
SOURCE61_SHA = "277b1c8e6930b90174ab7f3102f9ce75b43e16f38def6dab721b4d2e859826f2"
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")


def signature(model, optimizer, seed, arm):
    session = f"relation62-synthetic-{seed}-{arm}"
    tape = model.evidence(session, cut_credit=arm == "cut", terminal_per_source=arm == "terminal")
    with torch.no_grad():
        logits, attention = model.teacher_logits(tape, session, b"material of alpha?", b"quartz")
    return dict(
        bundle=paired.fingerprint(model.bundle(optimizer)),
        logits=paired.fingerprint(logits),
        attention=paired.fingerprint(attention),
    )


def restore(path):
    import time

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["schema"] != "relation62-paired-zero-v1"
        or payload["seed"] not in SEEDS
        or payload["arm"] not in ARMS
    ):
        raise ValueError("frozen zero checkpoint identity required")
    model = transfer.LocalRelationEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    if len(optimizer.state) != 35:
        raise ValueError("all35 old Adam moments required, new6 absent")
    return dict(
        signature=signature(model, optimizer, payload["seed"], payload["arm"]),
        cpu_seconds=time.process_time(),
    )


def identity():
    return dict(
        source=calibration.code_identity(),
        code={
            str(p.relative_to(ROOT)): paired.sha(p)
            for p in (Path(__file__), Path(transfer.__file__), Path(face.__file__))
        },
    )


def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive paired preflight")
    if paired.sha(calibration.REPORT) != SOURCE61_SHA:
        raise ValueError("61 report changed")
    passed = json.loads(calibration.REPORT.read_text(encoding="utf-8"))
    if (
        not passed["complete"]
        or not passed["training_sufficient"]
        or passed["code_identity"] != calibration.code_identity()
    ):
        raise ValueError("61 four-gate completion required")
    receipt = json.loads(face.REPORT.read_text(encoding="utf-8"))
    if (
        not receipt["complete"]
        or receipt["manifest_sha256"] != face.sha(face.MANIFEST)
        or receipt["code_sha256"] != face.sha(Path(face.__file__))
        or receipt["summary"]["model_queries"] != 0
    ):
        raise ValueError("unqueried frozen v4 face required")
    _, payload, source_path = transfer.source_payload()
    budget = paired.CpuBudget(180)
    budget.children += receipt["cpu_seconds"]
    folder = Path(get_external_path("data/checkpoints")) / "n1_relation_paired_preflight_20261015"
    folder.mkdir(exist_ok=False)
    guards = {
        str(p): paired.sha(p)
        for p in (
            calibration.REPORT,
            transfer.source.REPORT,
            source_path,
            face.MANIFEST,
            face.REPORT,
            *face.SOURCES,
        )
    }
    result = dict(
        schema="relation62-paired-preflight-v1",
        parameter_updates=0,
        curriculum_queries=0,
        free_queries=0,
        dev_queries=0,
        holdout_queries=0,
        code_identity=identity(),
        source_guards=guards,
        initial_fingerprints={},
        restore_checks=[],
        checks={},
        complete=False,
        status="running",
        can_promote=False,
    )
    try:
        for seed in SEEDS:
            initial = []
            for arm in ARMS:
                budget.guard(reserve=5)
                model, optimizer = transfer.migrate(payload, seed)
                initial.append(paired.fingerprint(model.bundle(optimizer)))
                session = f"relation62-synthetic-{seed}-{arm}"
                model.observe(session, Observation("text", "alpha uses quartz.", 1, "one"))
                model.observe(session, Observation("text", "bravo uses copper.", 2, "two"))
                expected = signature(model, optimizer, seed, arm)
                path = folder / f"s{seed}_{arm}_zero.pt"
                with path.open("xb") as f:
                    torch.save(
                        dict(
                            schema="relation62-paired-zero-v1",
                            seed=seed,
                            arm=arm,
                            candidate=model.bundle(optimizer),
                        ),
                        f,
                    )
                try:
                    child = subprocess.run(
                        [sys.executable, str(Path(__file__)), "--restore-worker", str(path)],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=min(30, budget.limit - budget.used() - 2),
                    )
                    restored = json.loads(child.stdout)
                except Exception:
                    budget.accounting_complete = False
                    raise
                budget.children += restored["cpu_seconds"]
                if restored["signature"] != expected:
                    raise ValueError("independent restore differs")
                result["restore_checks"].append(
                    dict(seed=seed, arm=arm, path=str(path), sha256=paired.sha(path), equal=True)
                )
                if signature(model, optimizer, seed, arm) != expected:
                    raise ValueError("parent changed")
            result["initial_fingerprints"][str(seed)] = initial
            result["checks"][f"paired_{seed}"] = len(set(initial)) == 1
        if len(result["restore_checks"]) != 9 or not all(result["checks"].values()):
            raise ValueError("all9 paired zeros and independent recoveries required")
        result.update(complete=True, status="complete")
    except Exception as error:
        result.update(status="rejected", error=f"{type(error).__name__}: {error}")
    finally:
        result["cpu_seconds"] = budget.used()
        result["cpu_accounting_complete"] = budget.accounting_complete
        result["sources_unchanged"] = all(paired.sha(Path(p)) == h for p, h in guards.items())
        result["code_unchanged"] = result["code_identity"] == identity()
        result["complete"] &= (
            result["sources_unchanged"]
            and result["code_unchanged"]
            and budget.accounting_complete
            and result["cpu_seconds"] <= 180
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
        print(json.dumps({k: output.get(k) for k in ("complete", "cpu_seconds", "error")}))
        raise SystemExit(0 if output["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, models=9, updates=0)))
