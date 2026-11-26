"""PLAN77: fixed nine-model full-fresh train-only training; no new confirmation queries."""

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

from scripts.training import _taiji_n1_evidence_shard as shard  # noqa: E402
from scripts.training import audit_taiji_n1_fresh_full_history as engineering  # noqa: E402
from scripts.training import build_taiji_n1_fresh_paired_admission as admission  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

runner = engineering.train.runner
paired = runner.paired
update = engineering.update
REPORT = ROOT / "reports/taiji_n1_fresh_paired_train_20261016.json"
ADMISSION_SHA = "8c3e8856c89b2087f0f4db946bf45e547f19e6cef75523494aa6e96e2087b938"
UPDATES = 24576
MILESTONES = (0, 6144, 12288, 18432, 24576)
PROTOCOL = admission.FORMAL


def identity():
    return dict(
        protocol=PROTOCOL, admission=admission.identity(), runner=paired.sha(Path(__file__))
    )


def readiness():
    if paired.sha(admission.REPORT) != ADMISSION_SHA:
        raise ValueError("exact frozen76 admission required")
    receipt = json.loads(admission.REPORT.read_text(encoding="utf-8"))
    if not receipt["complete"] or receipt["code_identity"] != admission.identity():
        raise ValueError("complete unchanged admission required")
    _, guards = admission.readiness()
    guards[str(admission.REPORT)] = ADMISSION_SHA
    guards[receipt["manifest"]["path"]] = receipt["manifest"]["sha256"]
    if any(paired.sha(Path(p)) != h for p, h in guards.items()):
        raise ValueError("sealed sources differ")
    preflight = json.loads(engineering.REPORT.read_text(encoding="utf-8"))
    paths = {}
    for check in preflight["restore_checks"]:
        path = Path(check["path"])
        for seed in engineering.SEEDS:
            for arm in engineering.ARMS:
                if path.name == f"s{seed}_{arm}_zero.pt":
                    paths[seed, arm] = path
    if len(paths) != 9:
        raise ValueError("nine original74 zeros required")
    return guards, paths


def validate_state(model, optimizer, completed, seed, arm):
    if seed not in engineering.SEEDS or arm not in engineering.ARMS:
        raise ValueError("fixed seed and arm required")
    if type(completed) is not int or not 0 <= completed <= UPDATES + 2:
        raise ValueError("fresh nonnegative fixed-range update count required")
    if len(model._observations) != completed:
        raise ValueError("one unique session per completed update required")
    if completed == 0:
        if optimizer.state or paired.fingerprint(
            model.bundle(optimizer)
        ) != engineering.initial_fingerprint(seed):
            raise ValueError("exact fresh zero with empty Adam required")
    else:
        expected = 37 if arm == "cut" else 41
        if len(optimizer.state) != expected:
            raise ValueError("exact same-arm Adam states required")
        with torch.random.fork_rng():
            fresh, _ = engineering.fresh(seed)
        original = dict(fresh.named_parameters())
        for name, parameter in model.named_parameters():
            if arm == "cut" and name.startswith("writer."):
                if parameter in optimizer.state or not torch.equal(parameter, original[name]):
                    raise ValueError("fresh cut writer changed or acquired Adam")
            elif int(optimizer.state[parameter]["step"]) != completed:
                raise ValueError("fresh Adam update count differs")
    update.bounded.grouped_parameters(model)


def signature(model, optimizer, completed, seed, arm):
    validate_state(model, optimizer, completed, seed, arm)
    before = paired.fingerprint(model.bundle(optimizer))
    with torch.random.fork_rng():
        candidate = deepcopy(model.bundle(optimizer))
        clone = engineering.LocalRelationEvidenceCandidate.restore_bundle(candidate)
        opt = torch.optim.Adam(clone.parameters(), lr=0.001)
        opt.load_state_dict(candidate["optimizer"])
        trajectory = []
        for offset in (1, 2):
            item = engineering.SYNTHETIC
            loss, measurements = update.train_step(
                clone,
                opt,
                item,
                f"fresh77-signature-{seed}-{arm}-{completed}-{offset}",
                arm,
                runner.core.renaming.second_question(item),
                audit=True,
            )
            validate_state(clone, opt, completed + offset, seed, arm)
            trajectory.append(dict(loss=loss, measurements=measurements))
        result = dict(
            bundle=before, next_bundle=paired.fingerprint(clone.bundle(opt)), trajectory=trajectory
        )
    if paired.fingerprint(model.bundle(optimizer)) != before:
        raise ValueError("next two updates changed parent parameters/Adam/RNG/history")
    return result


def save(model, optimizer, path, completed, seed, arm):
    if completed not in (0, UPDATES):
        raise ValueError("zero/final saves only; partial has separate rejected phase")
    payload = dict(
        schema="n1-fresh-paired-training-checkpoint-v1",
        protocol=PROTOCOL,
        initialization=engineering.INITIALIZATION,
        update_policy=deepcopy(update.POLICY),
        seed=seed,
        arm=arm,
        completed_updates=completed,
        phase="zero" if completed == 0 else "final",
        candidate=model.bundle(optimizer),
        expected_signature=signature(model, optimizer, completed, seed, arm),
    )
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def load(payload):
    completed = payload.get("completed_updates")
    seed, arm = payload.get("seed"), payload.get("arm")
    if (
        payload.get("schema") != "n1-fresh-paired-training-checkpoint-v1"
        or payload.get("protocol") != PROTOCOL
        or payload.get("initialization") != engineering.INITIALIZATION
        or payload.get("update_policy") != update.POLICY
        or seed not in engineering.SEEDS
        or arm not in engineering.ARMS
        or completed not in (0, UPDATES)
        or payload.get("phase") != ("zero" if completed == 0 else "final")
    ):
        raise ValueError("fixed fresh training marker, identity, phase and rule required")
    model = engineering.LocalRelationEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(deepcopy(payload["candidate"]["optimizer"]))
    validate_state(model, optimizer, completed, seed, arm)
    return model, optimizer


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = load(payload)
    return dict(
        signature=signature(
            model, optimizer, payload["completed_updates"], payload["seed"], payload["arm"]
        ),
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
            timeout=min(45, budget.limit - budget.used() - 2),
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
        raise ValueError("independent complete two-step fresh training recovery differs")
    return dict(path=str(path), sha256=paired.sha(path), equal=True, **result)


def scheduled(step, items, curriculum):
    if type(step) is not int or not 0 <= step < UPDATES:
        raise ValueError("fixed four-cycle schedule required")
    return runner.core.renaming.train_item(step % 6144, items, curriculum)


def seal_record(record):
    phase = "complete" if record["complete"] else "partial"
    record["phase"] = phase
    expected = {key: len(record[key]) for key in shard.FIELDS}
    if record["complete"] and expected != dict(losses=UPDATES, measurements=UPDATES, rows=2880):
        raise ValueError("complete shard counts differ")
    target = REPORT.with_name(f"{REPORT.stem}_s{record['seed']}_{record['arm']}.json.gz")
    key = {name: record[name] for name in ("seed", "arm", "protocol", "phase")}
    seal = shard.write_new(target, record, key, expected)
    shard.read_checked(target, seal, identity=key)
    return dict(
        seed=record["seed"],
        arm=record["arm"],
        complete=record["complete"],
        training_sufficient=record["training_sufficient"],
        shard=seal,
        completed_updates=record["completed_updates"],
        milestones=record["milestones"],
        restore_checks=record["restore_checks"],
        signature_clone_updates=record["signature_clone_updates"],
        final_steps=record.get("final_steps"),
        new_session_count=record.get("new_session_count"),
        error=record.get("error"),
    )


def train_one(seed, arm, source_path, folder, items, curriculum, views, budget):
    record = dict(
        seed=seed,
        arm=arm,
        protocol=PROTOCOL,
        completed_updates=0,
        signature_clone_updates=0,
        losses=[],
        measurements=[],
        rows=[],
        milestones=[],
        restore_checks=[],
        status="running",
        complete=False,
        training_sufficient=False,
    )
    model = optimizer = None
    try:
        source = torch.load(source_path, map_location="cpu", weights_only=True)
        model, optimizer = engineering.load(source)
        validate_state(model, optimizer, 0, seed, arm)
        zero = folder / f"s{seed}_{arm}_zero.pt"
        saved = save(model, optimizer, zero, 0, seed, arm)
        record["signature_clone_updates"] += 2
        record["restore_checks"].append(independent(zero, saved["expected_signature"], budget))
        record["signature_clone_updates"] += 2
        for step in range(UPDATES + 1):
            if step in MILESTONES:
                rows, summary = runner.evaluate(model, seed, arm, step, views, budget)
                record["rows"].extend(rows)
                record["milestones"].append(dict(summary, completed_updates=step))
                print(
                    f"fresh77 seed={seed} arm={arm} step={step} original={summary['free_exact']} pair={summary['pair_exact']} rename={summary['rename_free_exact']} rename_pair={summary['rename_pair_exact']} cpu={budget.used():.2f}",
                    flush=True,
                )
            if step == UPDATES:
                break
            budget.guard(reserve=15)
            item, second = scheduled(step, items, curriculum)
            loss, measurement = update.train_step(
                model, optimizer, item, f"fresh77-{seed}-{arm}-{step}", arm, second
            )
            record["completed_updates"] += 1
            record["losses"].append(loss)
            record["measurements"].append(dict(step=step, cycle=step // 6144, groups=measurement))
        validate_state(model, optimizer, UPDATES, seed, arm)
        final = folder / f"s{seed}_{arm}_final.pt"
        saved = save(model, optimizer, final, UPDATES, seed, arm)
        record["signature_clone_updates"] += 2
        record["restore_checks"].append(independent(final, saved["expected_signature"], budget))
        record["signature_clone_updates"] += 2
        record["final_steps"] = {
            n: int(optimizer.state[p]["step"]) if p in optimizer.state else None
            for n, p in model.named_parameters()
        }
        record["new_session_count"] = len(model._observations)
        record["complete"] = (
            record["completed_updates"] == UPDATES
            and len(record["rows"]) == 2880
            and len(record["measurements"]) == UPDATES
            and len(record["losses"]) == UPDATES
            and sum(x["qa_sequences"] for x in record["losses"]) == 36864
            and len(record["restore_checks"]) == 2
            and record["signature_clone_updates"] == 8
        )
        if not record["complete"]:
            raise ValueError("full fixed trajectory evidence incomplete")
        record["training_sufficient"] = runner.sufficient(record["milestones"][-1])
        record["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="stopped", complete=False, error=f"{type(error).__name__}: {error}")
        if model is not None and optimizer is not None:
            path = folder / f"s{seed}_{arm}_partial.pt"
            with path.open("xb") as stream:
                torch.save(
                    dict(
                        schema="n1-fresh-paired-training-checkpoint-v1",
                        protocol=PROTOCOL,
                        seed=seed,
                        arm=arm,
                        phase="partial",
                        completed_updates=record["completed_updates"],
                        candidate=model.bundle(optimizer),
                    ),
                    stream,
                )
            record["partial_checkpoint"] = dict(
                path=str(path), sha256=paired.sha(path), automatic_resume_allowed=False
            )
    return seal_record(record)


def audit():
    folder = Path(get_external_path("data/checkpoints")) / "n1_fresh_paired_train_20261016"
    if REPORT.exists() or folder.exists() or list(REPORT.parent.glob(f"{REPORT.stem}_s*.json.gz")):
        raise FileExistsError("exclusive fixed nine-model batch; no restart")
    budget = paired.CpuBudget(9000)
    guards, sources = readiness()
    items = runner.core.checked_train()
    curriculum = runner.core.checked_curriculum(items)
    probes = [
        runner.core.renaming.rename_item(e, p["names"])
        for e, p in zip(items, curriculum["probes"], strict=True)
    ]
    views = dict(
        original=items,
        pair=runner.core.checked_pairs(items),
        rename=probes,
        rename_pair=runner.core.build_pairs(probes),
    )
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        models=[],
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        can_promote=False,
        status="running",
    )
    folder.mkdir(exist_ok=False)
    try:
        for seed in engineering.SEEDS:
            for arm in engineering.ARMS:
                record = train_one(
                    seed, arm, sources[seed, arm], folder, items, curriculum, views, budget
                )
                result["models"].append(record)
                if not record["complete"]:
                    raise ValueError("model stopped; entire fixed batch stops")
                budget.guard(reserve=10)
        result["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    result.update(
        sources_unchanged=all(paired.sha(Path(p)) == h for p, h in guards.items()),
        code_unchanged=identity() == result["code_identity"],
        cpu_accounting_complete=budget.accounting_complete,
    )
    result["completed_updates"] = sum(m["completed_updates"] for m in result["models"])
    result["signature_clone_updates"] = sum(m["signature_clone_updates"] for m in result["models"])
    result["synthetic_qa_sequences"] = 2 * result["signature_clone_updates"]
    result["cpu_seconds"] = budget.used()
    result["complete"] = (
        result["status"] == "complete"
        and len(result["models"]) == 9
        and all(m["complete"] for m in result["models"])
        and result["completed_updates"] == 221184
        and result["signature_clone_updates"] == 72
        and result["cpu_seconds"] <= 9000
        and all(
            result[k] for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
        )
    )
    result["training_sufficient"] = result["complete"] and all(
        m["training_sufficient"] for m in result["models"] if m["arm"] == "position"
    )
    engineering.train.write_new(REPORT, result)
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
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in (
                        "complete",
                        "training_sufficient",
                        "completed_updates",
                        "cpu_seconds",
                        "error",
                    )
                }
            )
        )
        raise SystemExit(0 if result["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
