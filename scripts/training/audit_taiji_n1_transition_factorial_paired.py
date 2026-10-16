"""PLAN83: nine fresh quartet zeros and two-step same-arm recovery engineering."""

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
import torch  # noqa: E402

from scripts.training import (
    run_taiji_n1_transition_factorial_calibration as calibration,
)  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.native_local_relation_evidence_candidate import (
    LocalRelationEvidenceCandidate,
)  # noqa: E402

base = calibration.engineering.engineering
train = calibration.engineering.train
paired = train.paired
update = calibration.update
bounded = train.update.bounded
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
REPORT = ROOT / "reports/taiji_n1_transition_factorial_paired_preflight_20261016.json"
CALIBRATION_SHA = "a0dd6e74c669275fbfd7e629e17a8ff7a6122e232bb6797574f7bfaf0a73160e"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
INITIALIZATION = "full-fresh-no-inheritance-v1"
PROTOCOL = dict(
    schema="n1-transition-factorial-paired-engineering-v1",
    initialization=INITIALIZATION,
    seeds=list(SEEDS),
    arms=list(ARMS),
    parameters=62056,
    read_hops=2,
    update_policy=bounded.POLICY,
    learning_policy=update.POLICY,
    lr=0.001,
    main_trajectory_updates=0,
    signature_clone_updates=36,
    synthetic_qa_sequences=144,
    cpu_limit_seconds=120,
    curriculum_queries=0,
    dev_queries=0,
    holdout_queries=0,
    confirmation_queries=0,
)
SYNTHETIC = calibration.engineering.SYNTHETIC


def identity():
    return dict(
        protocol=PROTOCOL, calibration=calibration.identity(), runner=paired.sha(Path(__file__))
    )


def fresh(seed):
    if seed not in SEEDS:
        raise ValueError("frozen fresh seed required")
    torch.manual_seed(seed)
    model = LocalRelationEvidenceCandidate()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bounded.grouped_parameters(model)
    if model._observations or optimizer.state or model.READ_HOPS != 2:
        raise ValueError("entire fresh model/Adam/history required")
    return model, optimizer


def initial_fingerprint(seed):
    with torch.random.fork_rng():
        model, optimizer = fresh(seed)
        return paired.fingerprint(model.bundle(optimizer))


def signature(model, optimizer, seed, arm):
    if arm not in ARMS:
        raise ValueError("frozen arm required")
    before = paired.fingerprint(model.bundle(optimizer))
    if before != initial_fingerprint(seed):
        raise ValueError("signature parent must be exact untouched full-fresh seed")
    with torch.random.fork_rng():
        candidate = deepcopy(model.bundle(optimizer))
        probe = LocalRelationEvidenceCandidate.restore_bundle(candidate)
        opt = torch.optim.Adam(probe.parameters(), lr=0.001)
        opt.load_state_dict(candidate["optimizer"])
        writer = {
            n: p.detach().clone() for n, p in probe.named_parameters() if n.startswith("writer.")
        }
        head = probe.relation_projection.weight.detach().clone()
        trajectory = []
        for step in (1, 2):
            loss, measurements = update.train_step(
                probe,
                opt,
                SYNTHETIC,
                f"factorial83-signature-{seed}-{arm}-{step}",
                arm,
                train.runner.core.renaming.second_question(SYNTHETIC),
                audit=True,
            )
            if len(opt.state) != (37 if arm == "cut" else 41):
                raise ValueError("fresh cut must have37 states; native arms41")
            for name, parameter in probe.named_parameters():
                if arm == "cut" and name in writer:
                    if parameter in opt.state or not torch.equal(parameter, writer[name]):
                        raise ValueError(
                            "from-start cut writer must remain unchanged with no moments"
                        )
                elif int(opt.state[parameter]["step"]) != step:
                    raise ValueError("fresh non-cut Adam count differs")
            writer_gradient = measurements["writer"]["gradient_l2"]
            if (arm == "cut" and writer_gradient != 0) or (arm != "cut" and writer_gradient <= 0):
                raise ValueError("fresh writer credit differs from same-arm contract")
            if step == 1 and torch.equal(head, probe.relation_projection.weight):
                raise ValueError("zero relation head must learn on first step")
            relation_gradient = math.sqrt(
                sum(
                    float(torch.linalg.vector_norm(p.grad)) ** 2
                    for n, p in probe.named_parameters()
                    if n.startswith("relation_sequence.") and p.grad is not None
                )
            )
            if not math.isfinite(relation_gradient) or (step == 2 and relation_gradient <= 0):
                raise ValueError("relation GRU credit must become positive on second step")
            if len(probe._observations) != 4 * step or loss["qa_sequences"] != 4:
                raise ValueError("four complete histories and QA per quartet step required")
            trajectory.append(
                dict(
                    step=step,
                    loss=loss,
                    measurements=measurements,
                    relation_gru_gradient_l2=relation_gradient,
                    adam_states=len(opt.state),
                )
            )
        result = dict(
            bundle=before, next_bundle=paired.fingerprint(probe.bundle(opt)), trajectory=trajectory
        )
    if paired.fingerprint(model.bundle(optimizer)) != before:
        raise ValueError("two-step clone changed original full fresh state or RNG")
    return result


def save(model, optimizer, path, seed, arm):
    payload = dict(
        schema="n1-transition-factorial-paired-zero-v1",
        protocol=PROTOCOL,
        initialization=INITIALIZATION,
        update_policy=deepcopy(update.POLICY),
        seed=seed,
        arm=arm,
        completed_updates=0,
        candidate=model.bundle(optimizer),
        expected_signature=signature(model, optimizer, seed, arm),
    )
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def load(payload):
    if (
        payload.get("schema") != "n1-transition-factorial-paired-zero-v1"
        or payload.get("protocol") != PROTOCOL
        or payload.get("initialization") != INITIALIZATION
        or payload.get("update_policy") != update.POLICY
        or payload.get("seed") not in SEEDS
        or payload.get("arm") not in ARMS
        or payload.get("completed_updates") != 0
    ):
        raise ValueError("strict fresh initialization marker, phase, rule and identity required")
    model = LocalRelationEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(deepcopy(payload["candidate"]["optimizer"]))
    if (
        optimizer.state
        or model._observations
        or paired.fingerprint(model.bundle(optimizer)) != initial_fingerprint(payload["seed"])
    ):
        raise ValueError("exact fresh parameters/RNG, empty Adam and history required")
    return model, optimizer


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = load(payload)
    return dict(
        signature=signature(model, optimizer, payload["seed"], payload["arm"]),
        cpu_seconds=time.process_time(),
        signature_clone_updates=2,
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
        result = json.loads(output.stdout)
    except Exception:
        budget.accounting_complete = False
        raise
    budget.children += result["cpu_seconds"]
    budget.guard()
    if result["signature"] != expected or result["signature_clone_updates"] != 2:
        raise ValueError("independent fresh full two-step same-arm recovery differs")
    return dict(path=str(path), sha256=paired.sha(path), equal=True, **result)


def readiness():
    if paired.sha(calibration.REPORT) != CALIBRATION_SHA:
        raise ValueError("exact complete82 calibration required")
    record = json.loads(calibration.REPORT.read_text(encoding="utf-8"))
    validate_calibration(record)
    guards = dict(record["source_guards"])
    guards[str(calibration.REPORT)] = CALIBRATION_SHA
    for check in record["restore_checks"]:
        if not check["equal"]:
            raise ValueError("all82 restorations must match")
        guards[check["path"]] = check["sha256"]
    if any(paired.sha(Path(p)) != h for p, h in guards.items()):
        raise ValueError("82 source/restoration chain changed")
    if str(base.REPORT) not in guards:
        raise ValueError("original74 nine zero source missing")
    old = json.loads(base.REPORT.read_text(encoding="utf-8"))
    if (
        not old["complete"]
        or old["code_identity"] != base.identity()
        or len(old["restore_checks"]) != 9
    ):
        raise ValueError("original74 complete nine fresh zeros required")
    paths = {}
    for check in old["restore_checks"]:
        path = Path(check["path"])
        if not check["equal"] or paired.sha(path) != check["sha256"]:
            raise ValueError("original74 zero bytes changed")
        guards[str(path)] = check["sha256"]
        for seed in SEEDS:
            for arm in ARMS:
                if path.name == f"s{seed}_{arm}_zero.pt":
                    if (seed, arm) in paths:
                        raise ValueError("duplicate fresh zero")
                    paths[seed, arm] = path
    if len(paths) != 9:
        raise ValueError("all nine source zeros required")
    return guards, paths


def validate_calibration(record):
    if (
        not record["complete"]
        or not record["training_sufficient"]
        or record["code_identity"] != calibration.identity()
        or record["protocol"] != calibration.PROTOCOL
        or record["completed_updates"] != 24576
        or len(record["losses"]) != 24576
        or len(record["measurements"]) != 24576
        or len(record["rows"]) != 4800
        or record["score_queries"] != 4800
        or sum(x["qa_sequences"] for x in record["losses"]) != 55296
        or record["new_session_count"] != 43008
        or len(record["final_steps"]) != 41
        or set(record["final_steps"].values()) != {24576}
        or len(record["restore_checks"]) != 2
        or record["signature_clone_updates"] != 8
        or record["synthetic_qa_sequences"] != 32
        or record["cpu_seconds"] > 1800
        or not all(
            record[k] for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
        )
        or not calibration.sufficient(record["milestones"][-1])
        or any(record[k] != 0 for k in ("dev_queries", "holdout_queries", "confirmation_queries"))
    ):
        raise ValueError("complete82 fixed trajectory, full recovery and six gates required")


def audit():
    folder = (
        Path(get_external_path("data/checkpoints"))
        / "n1_transition_factorial_paired_preflight_20261016"
    )
    if REPORT.exists() or ATTEMPT.exists() or folder.exists():
        raise FileExistsError("exclusive fresh engineering; no restart")
    budget = paired.CpuBudget(120)
    guards, paths = readiness()
    calibration.engineering.factorial.write_new(
        ATTEMPT, dict(protocol=PROTOCOL, code_identity=identity(), source_guards=guards)
    )
    folder.mkdir(exist_ok=False)
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        status="running",
        initial_fingerprints={str(seed): [] for seed in SEEDS},
        restore_checks=[],
        signature_clone_updates=0,
        main_trajectory_updates=0,
        curriculum_queries=0,
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        can_promote=False,
    )
    try:
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=10)
                model, optimizer = base.load(
                    torch.load(paths[seed, arm], map_location="cpu", weights_only=True)
                )
                result["initial_fingerprints"][str(seed)].append(
                    paired.fingerprint(model.bundle(optimizer))
                )
                path = folder / f"s{seed}_{arm}_zero.pt"
                saved = save(model, optimizer, path, seed, arm)
                result["signature_clone_updates"] += 2
                result["restore_checks"].append(
                    independent(path, saved["expected_signature"], budget)
                )
                result["signature_clone_updates"] += 2
                print(
                    f"factorial83 seed={seed} arm={arm} restored cpu={budget.used():.2f}",
                    flush=True,
                )
        if (
            any(
                len(values) != 3 or len(set(values)) != 1
                for values in result["initial_fingerprints"].values()
            )
            or len({v[0] for v in result["initial_fingerprints"].values()}) != 3
        ):
            raise ValueError("three same-seed arms and three distinct full fresh seeds required")
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="no automatic restart")
    finally:
        result.update(
            sources_unchanged=all(paired.sha(Path(p)) == h for p, h in guards.items()),
            code_unchanged=identity() == result["code_identity"],
            cpu_accounting_complete=budget.accounting_complete,
        )
        result["cpu_seconds"] = budget.used()
        result["synthetic_qa_sequences"] = 4 * result["signature_clone_updates"]
        result["complete"] = (
            result["status"] == "complete"
            and len(result["restore_checks"]) == 9
            and result["signature_clone_updates"] == 36
            and result["synthetic_qa_sequences"] == 144
            and result["cpu_seconds"] <= 120
            and all(
                result[k]
                for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
            )
        )
        calibration.engineering.factorial.write_new(REPORT, result)
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
        result = audit()
        print(json.dumps({k: result.get(k) for k in ("complete", "cpu_seconds", "error")}))
        raise SystemExit(0 if result["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
