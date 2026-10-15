"""PLAN67: synthetic engineering of the fixed five-group actual-step policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import _taiji_n1_group_relative_adam as bounded  # noqa: E402
from scripts.training import audit_taiji_n1_relation_update_replay as replay  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

runner = bounded.runner
paired = runner.paired
REPORT = ROOT / "reports/taiji_n1_group_relative_adam_preflight_20261015.json"
REPLAY_SHA = "a8674e0bd8a068a72b28045404d568213258d6bf4141686cd8f49767f9fa2232"
PROTOCOL = dict(
    schema="n1-group-relative-adam-preflight-v1",
    update_policy=bounded.POLICY,
    source_sha256=replay.SOURCE_SHA,
    replay_sha256=REPLAY_SHA,
    mechanical_trajectory_updates=2,
    signature_clone_updates=4,
    cpu_limit_seconds=60,
    curriculum_queries=0,
    dev_queries=0,
    holdout_queries=0,
    confirmation_queries=0,
)
SYNTHETIC = dict(
    id="synthetic67",
    kind="lookup",
    split="train",
    observations=["alpha uses quartz.", "bravo uses copper."],
    question="material of alpha?",
    answer="quartz",
)


def identity():
    return dict(
        protocol=PROTOCOL,
        source=runner.identity(),
        replay=replay.identity(),
        helper=paired.sha(Path(bounded.__file__)),
        runner=paired.sha(Path(__file__)),
    )


def signature(model, optimizer):
    before = paired.fingerprint(model.bundle(optimizer))
    with torch.random.fork_rng():
        candidate = deepcopy(model.bundle(optimizer))
        probe = runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(candidate)
        opt = torch.optim.Adam(probe.parameters(), lr=0.001)
        opt.load_state_dict(candidate["optimizer"])
        loss, measurement = bounded.train_step(
            probe,
            opt,
            SYNTHETIC,
            "bounded67-signature",
            runner.core.renaming.second_question(SYNTHETIC),
        )
        next_bundle = paired.fingerprint(probe.bundle(opt))
    if before != paired.fingerprint(model.bundle(optimizer)):
        raise ValueError("next-step clone signature changed original state or RNG")
    return dict(
        bundle=before, next_bundle=next_bundle, next_loss=loss, next_measurement=measurement
    )


def save(model, optimizer, path, completed, *, protocol=PROTOCOL):
    payload = dict(
        schema="n1-group-relative-adam-checkpoint-v1",
        protocol=protocol,
        update_policy=deepcopy(bounded.POLICY),
        completed_updates=completed,
        candidate=model.bundle(optimizer),
        expected_signature=signature(model, optimizer),
    )
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def load(payload, *, protocol=PROTOCOL):
    if (
        payload.get("schema") != "n1-group-relative-adam-checkpoint-v1"
        or payload.get("protocol") != protocol
        or payload.get("update_policy") != bounded.POLICY
        or payload.get("completed_updates") not in (0, 2)
    ):
        raise ValueError("explicit frozen bounded-update marker and protocol required")
    model = runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(deepcopy(payload["candidate"]["optimizer"]))
    if len(optimizer.state) != (35 if payload["completed_updates"] == 0 else 41):
        raise ValueError("full source and new Adam state required")
    for name, parameter in model.named_parameters():
        if parameter in optimizer.state and int(
            optimizer.state[parameter]["step"]
        ) != runner.expected_step(name, "position", payload["completed_updates"]):
            raise ValueError("per-parameter old/new Adam steps differ")
    return model, optimizer


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = load(payload)
    return dict(
        signature=signature(model, optimizer),
        cpu_seconds=time.process_time(),
        signature_clone_updates=1,
    )


def independent(path, expected, budget):
    budget.guard(reserve=10)
    try:
        output = subprocess.run(
            [sys.executable, str(Path(__file__)), "--restore-worker", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=min(30, budget.limit - budget.used() - 2),
            env=dict(
                os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
            ),
        )
        response = json.loads(output.stdout)
    except Exception:
        budget.accounting_complete = False
        raise
    budget.children += response["cpu_seconds"]
    budget.guard()
    if response["signature"] != expected:
        raise ValueError("independent full recovery or next bounded update differs")
    return dict(path=str(path), sha256=paired.sha(path), equal=True, **response)


def readiness():
    record, guards, paths = replay.readiness()
    if paired.sha(replay.REPORT) != REPLAY_SHA:
        raise ValueError("frozen66 report required")
    evidence = json.loads(replay.REPORT.read_text(encoding="utf-8"))
    if not evidence["complete"] or evidence["code_identity"] != replay.identity():
        raise ValueError("unchanged complete66 required")
    guards[str(replay.REPORT)] = REPLAY_SHA
    return record, guards, paths


def audit():
    folder = (
        Path(get_external_path("data/checkpoints")) / "n1_group_relative_adam_preflight_20261015"
    )
    if REPORT.exists() or folder.exists():
        raise FileExistsError("exclusive engineering batch; no restart")
    budget = paired.CpuBudget(60)
    _, guards, paths = readiness()
    source = torch.load(paths["zero"], map_location="cpu", weights_only=True)
    if source["protocol"] != runner.PROTOCOL or source["completed_updates"] != 0:
        raise ValueError("frozen63 zero source required")
    folder.mkdir(exist_ok=False)
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        mechanical_trajectory_updates=0,
        signature_clone_updates=0,
        losses=[],
        measurements=[],
        restore_checks=[],
        curriculum_queries=0,
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        status="running",
        can_promote=False,
    )
    try:
        model = runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(source["candidate"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(deepcopy(source["candidate"]["optimizer"]))
        groups = bounded.grouped_parameters(model)
        result["group_names"] = {k: [n for n, _ in values] for k, values in groups.items()}
        if paired.fingerprint(model.bundle(optimizer)) != paired.fingerprint(source["candidate"]):
            raise ValueError("source zero full bundle differs")
        before_head = model.relation_projection.weight.detach().clone()
        zero = save(model, optimizer, folder / "zero.pt", 0)
        result["signature_clone_updates"] += 1
        result["restore_checks"].append(
            independent(folder / "zero.pt", zero["expected_signature"], budget)
        )
        result["signature_clone_updates"] += 1
        for step in range(2):
            budget.guard(reserve=10)
            loss, measurements = bounded.train_step(
                model,
                optimizer,
                SYNTHETIC,
                f"bounded67-main-{step}",
                runner.core.renaming.second_question(SYNTHETIC),
                audit=True,
            )
            result["mechanical_trajectory_updates"] += 1
            result["losses"].append(loss)
            result["measurements"].append(measurements)
        if not any(s["clipped"] for m in result["measurements"] for s in m.values()):
            raise ValueError("synthetic engineering must exercise an actual capped proposal")
        if torch.equal(before_head, model.relation_projection.weight) or not all(
            m["writer"]["gradient_l2"] > 0 for m in result["measurements"]
        ):
            raise ValueError("zero relation head must learn and writer credit must remain")
        final = save(model, optimizer, folder / "final.pt", 2)
        result["signature_clone_updates"] += 1
        result["restore_checks"].append(
            independent(folder / "final.pt", final["expected_signature"], budget)
        )
        result["signature_clone_updates"] += 1
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="operator interruption; no restart")
    finally:
        result.update(
            cpu_seconds=budget.used(),
            cpu_accounting_complete=budget.accounting_complete,
            sources_unchanged=all(paired.sha(Path(p)) == digest for p, digest in guards.items()),
            code_unchanged=identity() == result["code_identity"],
        )
        result["complete"] = (
            result["status"] == "complete"
            and result["mechanical_trajectory_updates"] == 2
            and result["signature_clone_updates"] == 4
            and len(result["restore_checks"]) == 2
            and result["cpu_seconds"] <= 60
            and all(
                result[k]
                for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
            )
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
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
