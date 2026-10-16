"""PLAN84: fixed nine-model fresh factorial learning; no confirmation queries."""

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
from scripts.training import audit_taiji_n1_transition_factorial_paired as engineering  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

calibration = engineering.calibration
runner = calibration.runner
paired = runner.paired
update = engineering.update
REPORT = ROOT / "reports/taiji_n1_transition_factorial_paired_train_20261016.json"
PREFLIGHT_SHA = "4b157a1eaf2f0ec6cae20eef705a441a408f24a5cc7816dd1c84fa59acb9dd06"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
UPDATES = 24576
MILESTONES = (0, 6144, 12288, 18432, 24576)
PROTOCOL = dict(
    schema="n1-transition-factorial-fresh-paired-training-v1",
    seeds=list(engineering.SEEDS),
    arms=list(engineering.ARMS),
    initialization=engineering.INITIALIZATION,
    updates_per_model=24576,
    total_updates=221184,
    qa_sequences_per_model=55296,
    total_qa_sequences=497664,
    training_sessions_per_model=43008,
    total_training_sessions=387072,
    milestones=list(MILESTONES),
    known_score_rows=43200,
    independent_restores=18,
    signature_clone_updates=72,
    synthetic_qa_sequences=288,
    cpu_limit_seconds=14400,
    parameters=62056,
    learning=update.POLICY,
    update_policy=engineering.bounded.POLICY,
    cycle_length=6144,
    cycles=4,
    factorial_joint_gate=44,
    dev_queries=0,
    holdout_queries=0,
    confirmation_queries=0,
)


def identity():
    return dict(
        protocol=PROTOCOL,
        engineering=engineering.identity(),
        shard=paired.sha(Path(shard.__file__)),
        runner=paired.sha(Path(__file__)),
    )


def readiness():
    if paired.sha(engineering.REPORT) != PREFLIGHT_SHA:
        raise ValueError("exact complete83 preflight required")
    receipt = json.loads(engineering.REPORT.read_text(encoding="utf-8"))
    if (
        not receipt["complete"]
        or receipt["code_identity"] != engineering.identity()
        or len(receipt["restore_checks"]) != 9
        or receipt["signature_clone_updates"] != 36
        or receipt["synthetic_qa_sequences"] != 144
        or receipt["cpu_seconds"] > 120
        or not all(
            receipt[k] for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
        )
    ):
        raise ValueError("complete unchanged83 nine quartet recoveries required")
    guards = dict(receipt["source_guards"])
    guards[str(engineering.REPORT)] = PREFLIGHT_SHA
    paths = {}
    for check in receipt["restore_checks"]:
        path = Path(check["path"])
        if not check["equal"] or paired.sha(path) != check["sha256"]:
            raise ValueError("83 full fresh zero bytes changed")
        guards[str(path)] = check["sha256"]
        for seed in engineering.SEEDS:
            for arm in engineering.ARMS:
                if path.name == f"s{seed}_{arm}_zero.pt":
                    if (seed, arm) in paths:
                        raise ValueError("duplicate fresh source")
                    paths[seed, arm] = path
    if set(paths) != {(s, a) for s in engineering.SEEDS for a in engineering.ARMS}:
        raise ValueError("all nine83 zeros required")
    if any(paired.sha(Path(p)) != h for p, h in guards.items()):
        raise ValueError("complete83 source chain changed")
    return guards, paths


def validate_state(model, optimizer, completed, seed, arm, *, sessions=None):
    if seed not in engineering.SEEDS or arm not in engineering.ARMS:
        raise ValueError("fixed seed and arm required")
    if type(completed) is not int or not 0 <= completed <= UPDATES + 2:
        raise ValueError("fresh nonnegative fixed-range update count required")
    if len(model._observations) != (
        calibration.expected_sessions(completed) if sessions is None else sessions
    ):
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
    engineering.bounded.grouped_parameters(model)


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
                f"factorial84-signature-{seed}-{arm}-{completed}-{offset}",
                arm,
                runner.core.renaming.second_question(item),
                audit=True,
            )
            validate_state(
                clone,
                opt,
                completed + offset,
                seed,
                arm,
                sessions=calibration.expected_sessions(completed) + 4 * offset,
            )
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
        schema="n1-transition-factorial-paired-training-checkpoint-v1",
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
        payload.get("schema") != "n1-transition-factorial-paired-training-checkpoint-v1"
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
    record["qa_sequences"] = sum(x["qa_sequences"] for x in record["losses"])
    phase = "complete" if record["complete"] else "partial"
    record["phase"] = phase
    expected = {key: len(record[key]) for key in shard.FIELDS}
    if record["complete"] and expected != dict(losses=UPDATES, measurements=UPDATES, rows=4800):
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
        score_queries=record["score_queries"],
        qa_sequences=record["qa_sequences"],
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
        score_queries=0,
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
                rows, summary = evaluate(model, optimizer, seed, arm, step, views, budget, record)
                record["milestones"].append(dict(summary, completed_updates=step))
                print(
                    f"factorial84 seed={seed} arm={arm} step={step} original={summary['free_exact']} pair={summary['pair_exact']} rename={summary['rename_free_exact']} rename_pair={summary['rename_pair_exact']} cpu={budget.used():.2f}",
                    flush=True,
                )
            if step == UPDATES:
                break
            budget.guard(reserve=15)
            item, second = scheduled(step, items, curriculum)
            loss, measurement = update.train_step(
                model, optimizer, item, f"factorial84-{seed}-{arm}-{step}", arm, second
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
            and len(record["rows"]) == 4800
            and record["score_queries"] == 4800
            and record["new_session_count"] == 43008
            and len(record["measurements"]) == UPDATES
            and len(record["losses"]) == UPDATES
            and sum(x["qa_sequences"] for x in record["losses"]) == 55296
            and len(record["restore_checks"]) == 2
            and record["signature_clone_updates"] == 8
        )
        if not record["complete"]:
            raise ValueError("full fixed trajectory evidence incomplete")
        record["training_sufficient"] = calibration.sufficient(record["milestones"][-1])
        record["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="stopped", complete=False, error=f"{type(error).__name__}: {error}")
        if model is not None and optimizer is not None:
            path = folder / f"s{seed}_{arm}_partial.pt"
            with path.open("xb") as stream:
                torch.save(
                    dict(
                        schema="n1-transition-factorial-paired-training-checkpoint-v1",
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


def evaluate(model, optimizer, seed, arm, step, views, budget, evidence):
    before = paired.fingerprint(model.bundle(optimizer))
    rows = []
    try:
        for view, items in views.items():
            for item in items:
                budget.guard(reserve=15)
                session = f"score84-{seed}-{arm}-{step}-{view}-{item['id']}"
                if session in model._observations:
                    raise ValueError("score session collision")
                evidence["score_queries"] += 1
                try:
                    row = runner.core.measure(model, item, arm, session)
                    rows.append(
                        dict(
                            row,
                            seed=seed,
                            arm=arm,
                            completed_updates=step,
                            evaluation_set=view,
                        )
                    )
                finally:
                    model._observations.pop(session, None)
        selected = {v: [r for r in rows if r["evaluation_set"] == v] for v in views}
        summary = runner.core.summarize(selected["original"])[str(seed)][arm]
        renamed = runner.core.summarize(selected["rename"])[str(seed)][arm]
        summary.update(
            pair_exact=sum(r["free_exact"] for r in selected["pair"]),
            rename_free_exact=renamed["free_exact"],
            rename_free_by_kind=renamed["free_by_kind"],
            rename_pair_exact=sum(r["free_exact"] for r in selected["rename_pair"]),
            factorial={},
        )
        for view in ("original", "rename"):
            items = [item for item in views[view] if item["kind"] == "transition"]
            baseline = {row["id"]: row for row in selected[view] if row["kind"] == "transition"}
            entries = [
                entry
                for item in items
                for entry in calibration.engineering.factorial.variants(item)
            ]
            result = dict(rows=[], model_queries=0, native_reproductions=0)
            try:
                calibration.engineering.factorial.evaluate(
                    model, optimizer, seed, arm, entries, baseline, budget, result
                )
            finally:
                evidence["score_queries"] += result["model_queries"]
                rows.extend(
                    dict(row, completed_updates=step, evaluation_set=f"factorial_{view}")
                    for row in result["rows"]
                )
            if result["model_queries"] != 192 or result["native_reproductions"] != 48:
                raise ValueError("all native rows must reproduce same-checkpoint scores")
            summary["factorial"][view] = calibration.quartet_summary(result["rows"])
        return rows, summary
    finally:
        evidence["rows"].extend(rows)
        if paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("full scoring altered model/Adam/RNG/history")


def known_views():
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
    return items, curriculum, views


def audit():
    folder = (
        Path(get_external_path("data/checkpoints"))
        / "n1_transition_factorial_paired_train_20261016"
    )
    if (
        REPORT.exists()
        or ATTEMPT.exists()
        or folder.exists()
        or list(REPORT.parent.glob(f"{REPORT.stem}_s*.json.gz"))
    ):
        raise FileExistsError("exclusive fixed nine-model batch; no restart")
    budget = paired.CpuBudget(14400)
    guards, sources = readiness()
    items, curriculum, views = known_views()
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
    calibration.engineering.factorial.write_new(
        ATTEMPT, dict(protocol=PROTOCOL, code_identity=identity(), source_guards=guards)
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
    result["synthetic_qa_sequences"] = 4 * result["signature_clone_updates"]
    result["qa_sequences"] = sum(m["qa_sequences"] for m in result["models"])
    result["score_queries"] = sum(m["score_queries"] for m in result["models"])
    result["new_session_count"] = sum(m.get("new_session_count") or 0 for m in result["models"])
    result["cpu_seconds"] = budget.used()
    result["complete"] = (
        result["status"] == "complete"
        and len(result["models"]) == 9
        and all(m["complete"] for m in result["models"])
        and result["completed_updates"] == 221184
        and result["signature_clone_updates"] == 72
        and result["synthetic_qa_sequences"] == 288
        and result["qa_sequences"] == 497664
        and result["score_queries"] == 43200
        and result["new_session_count"] == 387072
        and result["cpu_seconds"] <= 14400
        and all(
            result[k] for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
        )
    )
    result["training_sufficient"] = result["complete"] and all(
        m["training_sufficient"] for m in result["models"] if m["arm"] == "position"
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
