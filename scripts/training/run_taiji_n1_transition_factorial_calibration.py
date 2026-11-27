"""PLAN82: fixed fresh seed13 factorial learning; no confirmation queries."""

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

from scripts.training import (
    audit_taiji_n1_transition_factorial_learning as engineering,
)  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

runner = engineering.train.runner
paired = runner.paired
update = engineering.learning
REPORT = ROOT / "reports/taiji_n1_transition_factorial_calibration_20261016.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
PREFLIGHT_SHA = "ac9e5cb95eb8beda998cc12b37d9036963c275083e7008d7f8018571056f3c21"
ZERO_SHA = "c8579e7e8ea360d21cd10ecc4f5b393a19e53306a8567a507edaa199f5e4485a"
UPDATES = 24576
MILESTONES = (0, 6144, 12288, 18432, 24576)
PROTOCOL = dict(
    schema="n1-transition-factorial-calibration-v1",
    initialization=engineering.engineering.INITIALIZATION,
    seed=13,
    arm="position",
    parameters=62056,
    updates=UPDATES,
    qa_sequences=55296,
    milestones=list(MILESTONES),
    known_score_rows=4800,
    training_sessions=43008,
    learning=update.POLICY,
    factorial_joint_gate=44,
    cycle_length=6144,
    cycles=4,
    update_policy=update.POLICY,
    signature_clone_updates=8,
    synthetic_qa_sequences=32,
    cpu_limit_seconds=1800,
    preflight_sha256=PREFLIGHT_SHA,
    source_zero_sha256=ZERO_SHA,
    dev_queries=0,
    holdout_queries=0,
    confirmation_queries=0,
)


def identity():
    return dict(
        protocol=PROTOCOL, engineering=engineering.identity(), runner=paired.sha(Path(__file__))
    )


def readiness():
    if paired.sha(engineering.REPORT) != PREFLIGHT_SHA:
        raise ValueError("exact complete81 preflight required")
    record = json.loads(engineering.REPORT.read_text(encoding="utf-8"))
    if (
        not record["complete"]
        or record["code_identity"] != engineering.identity()
        or len(record["restore_checks"]) != 1
        or record["signature_clone_updates"] != 4
    ):
        raise ValueError("unchanged complete81 full quartet recovery required")
    guards = dict(record["source_guards"])
    guards[str(engineering.REPORT)] = PREFLIGHT_SHA
    check = record["restore_checks"][0]
    selected = Path(check["path"])
    if not check["equal"] or check["sha256"] != ZERO_SHA:
        raise ValueError("exact fresh seed13 zero required")
    guards[str(selected)] = ZERO_SHA
    if any(paired.sha(Path(p)) != h for p, h in guards.items()):
        raise ValueError("complete81 source/zero chain changed")
    return guards, selected


def expected_sessions(completed):
    if type(completed) is not int or not 0 <= completed <= UPDATES:
        raise ValueError("fixed main update range required")
    transitions = (completed // 8) * 2 + max(0, min(completed % 8 - 4, 2))
    return completed + 3 * transitions


def validate_state(model, optimizer, completed, *, sessions=None):
    if type(completed) is not int or not 0 <= completed <= UPDATES + 2:
        raise ValueError("fresh nonnegative fixed-range update count required")
    if len(model._observations) != (expected_sessions(completed) if sessions is None else sessions):
        raise ValueError("all unique main or clone quartet sessions required")
    if completed == 0:
        if optimizer.state or paired.fingerprint(
            model.bundle(optimizer)
        ) != engineering.engineering.initial_fingerprint(13):
            raise ValueError("exact fresh zero with empty Adam required")
    else:
        if len(optimizer.state) != 41 or any(
            int(optimizer.state[p]["step"]) != completed for p in model.parameters()
        ):
            raise ValueError("all41 Adam states must have full fresh update count")
    engineering.train.update.bounded.grouped_parameters(model)


def signature(model, optimizer, completed):
    validate_state(model, optimizer, completed)
    before = paired.fingerprint(model.bundle(optimizer))
    with torch.random.fork_rng():
        candidate = deepcopy(model.bundle(optimizer))
        clone = engineering.engineering.LocalRelationEvidenceCandidate.restore_bundle(candidate)
        opt = torch.optim.Adam(clone.parameters(), lr=0.001)
        opt.load_state_dict(candidate["optimizer"])
        trajectory = []
        for offset in (1, 2):
            item = engineering.SYNTHETIC
            loss, measurements = update.train_step(
                clone,
                opt,
                item,
                f"factorial82-signature-{completed}-{offset}",
                "position",
                None,
                audit=True,
            )
            validate_state(
                clone, opt, completed + offset, sessions=expected_sessions(completed) + 4 * offset
            )
            trajectory.append(dict(loss=loss, measurements=measurements))
        result = dict(
            bundle=before, next_bundle=paired.fingerprint(clone.bundle(opt)), trajectory=trajectory
        )
    if paired.fingerprint(model.bundle(optimizer)) != before:
        raise ValueError("next two updates changed parent parameters/Adam/RNG/history")
    return result


def save(model, optimizer, path, completed):
    if completed not in (0, UPDATES):
        raise ValueError("zero/final saves only; partial has separate rejected phase")
    payload = dict(
        schema="n1-transition-factorial-training-checkpoint-v1",
        protocol=PROTOCOL,
        initialization=engineering.engineering.INITIALIZATION,
        update_policy=deepcopy(update.POLICY),
        seed=13,
        arm="position",
        completed_updates=completed,
        phase="zero" if completed == 0 else "final",
        candidate=model.bundle(optimizer),
        expected_signature=signature(model, optimizer, completed),
    )
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def load(payload):
    completed = payload.get("completed_updates")
    if (
        payload.get("schema") != "n1-transition-factorial-training-checkpoint-v1"
        or payload.get("protocol") != PROTOCOL
        or payload.get("initialization") != engineering.engineering.INITIALIZATION
        or payload.get("update_policy") != update.POLICY
        or payload.get("seed") != 13
        or payload.get("arm") != "position"
        or completed not in (0, UPDATES)
        or payload.get("phase") != ("zero" if completed == 0 else "final")
    ):
        raise ValueError("fixed fresh training marker, identity, phase and rule required")
    model = engineering.engineering.LocalRelationEvidenceCandidate.restore_bundle(
        payload["candidate"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(deepcopy(payload["candidate"]["optimizer"]))
    validate_state(model, optimizer, completed)
    return model, optimizer


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = load(payload)
    return dict(
        signature=signature(model, optimizer, payload["completed_updates"]),
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


def quartet_summary(rows):
    conditions = engineering.factorial.CONDITIONS
    if len(rows) != 192 or len({(r["parent_id"], r["condition"]) for r in rows}) != 192:
        raise ValueError("all192 unique quartet rows required")
    parents = {r["parent_id"] for r in rows}
    grouped = {p: {r["condition"]: r for r in rows if r["parent_id"] == p} for p in parents}
    if len(parents) != 48 or any(set(g) != set(conditions) for g in grouped.values()):
        raise ValueError("all48 complete four-condition parents required")
    return dict(
        items=192,
        parents=48,
        by_condition={
            c: sum(r["free_exact"] for r in rows if r["condition"] == c) for c in conditions
        },
        joint_four_exact=sum(all(r["free_exact"] for r in g.values()) for g in grouped.values()),
    )


def sufficient(summary):
    return runner.sufficient(summary) and all(
        summary["factorial"][view]["joint_four_exact"] >= 44 for view in ("original", "rename")
    )


def evaluate(model, optimizer, step, views, budget, evidence):
    before = paired.fingerprint(model.bundle(optimizer))
    rows = []
    try:
        for view, items in views.items():
            for item in items:
                budget.guard(reserve=15)
                session = f"score82-{step}-{view}-{item['id']}"
                if session in model._observations:
                    raise ValueError("score session collision")
                evidence["score_queries"] += 1
                try:
                    row = runner.core.measure(model, item, "position", session)
                    rows.append(
                        dict(
                            row,
                            seed=13,
                            arm="position",
                            completed_updates=step,
                            evaluation_set=view,
                        )
                    )
                finally:
                    model._observations.pop(session, None)
        selected = {v: [r for r in rows if r["evaluation_set"] == v] for v in views}
        summary = runner.core.summarize(selected["original"])["13"]["position"]
        renamed = runner.core.summarize(selected["rename"])["13"]["position"]
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
            entries = [entry for item in items for entry in engineering.factorial.variants(item)]
            result = dict(rows=[], model_queries=0, native_reproductions=0)
            try:
                engineering.factorial.evaluate(
                    model, optimizer, 13, "position", entries, baseline, budget, result
                )
            finally:
                evidence["score_queries"] += result["model_queries"]
                rows.extend(
                    dict(row, completed_updates=step, evaluation_set=f"factorial_{view}")
                    for row in result["rows"]
                )
            if result["model_queries"] != 192 or result["native_reproductions"] != 48:
                raise ValueError("all native rows must reproduce same-checkpoint scores")
            summary["factorial"][view] = quartet_summary(result["rows"])
        return rows, summary
    finally:
        evidence["rows"].extend(rows)
        if paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("full scoring altered model/Adam/RNG/history")


def audit():
    folder = (
        Path(get_external_path("data/checkpoints")) / "n1_transition_factorial_calibration_20261016"
    )
    if REPORT.exists() or ATTEMPT.exists() or folder.exists():
        raise FileExistsError("exclusive fixed calibration; no restart or overwrite")
    budget = paired.CpuBudget(1800)
    guards, source_path = readiness()
    source = torch.load(source_path, map_location="cpu", weights_only=True)
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
    engineering.factorial.write_new(
        ATTEMPT, dict(protocol=PROTOCOL, code_identity=identity(), source_guards=guards)
    )
    folder.mkdir(exist_ok=False)
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        source_checkpoint=str(source_path),
        completed_updates=0,
        signature_clone_updates=0,
        losses=[],
        measurements=[],
        rows=[],
        score_queries=0,
        milestones=[],
        restore_checks=[],
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        can_promote=False,
        status="running",
    )
    model = optimizer = None
    try:
        model, optimizer = engineering.load(source)
        validate_state(model, optimizer, 0)
        saved = save(model, optimizer, folder / "zero.pt", 0)
        result["signature_clone_updates"] += 2
        result["restore_checks"].append(
            independent(folder / "zero.pt", saved["expected_signature"], budget)
        )
        result["signature_clone_updates"] += 2
        for step in range(UPDATES + 1):
            if step in MILESTONES:
                rows, summary = evaluate(model, optimizer, step, views, budget, result)
                result["milestones"].append(dict(summary, completed_updates=step))
                print(
                    f"factorial82 step={step} original={summary['free_exact']} pair={summary['pair_exact']} rename={summary['rename_free_exact']} rename_pair={summary['rename_pair_exact']} cpu={budget.used():.2f}",
                    flush=True,
                )
            if step == UPDATES:
                break
            budget.guard(reserve=10)
            item, second = scheduled(step, items, curriculum)
            loss, measurement = update.train_step(
                model, optimizer, item, f"factorial82-train-{step}", "position", second
            )
            result["completed_updates"] += 1
            result["losses"].append(loss)
            result["measurements"].append(dict(step=step, cycle=step // 6144, groups=measurement))
        validate_state(model, optimizer, UPDATES)
        saved = save(model, optimizer, folder / "final.pt", UPDATES)
        result["signature_clone_updates"] += 2
        result["restore_checks"].append(
            independent(folder / "final.pt", saved["expected_signature"], budget)
        )
        result["signature_clone_updates"] += 2
        result["final_steps"] = {
            n: int(optimizer.state[p]["step"]) for n, p in model.named_parameters()
        }
        result["new_session_count"] = len(model._observations)
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="no automatic restart")
    finally:
        if result["status"] != "complete" and model is not None and optimizer is not None:
            try:
                path = folder / "partial.pt"
                with path.open("xb") as stream:
                    torch.save(
                        dict(
                            schema="n1-transition-factorial-training-checkpoint-v1",
                            protocol=PROTOCOL,
                            phase="partial",
                            completed_updates=result["completed_updates"],
                            candidate=model.bundle(optimizer),
                        ),
                        stream,
                    )
                result["partial_checkpoint"] = dict(
                    path=str(path), sha256=paired.sha(path), automatic_resume_allowed=False
                )
            except Exception as error:
                result["partial_error"] = str(error)
        result.update(
            sources_unchanged=all(paired.sha(Path(p)) == h for p, h in guards.items()),
            code_unchanged=identity() == result["code_identity"],
            cpu_accounting_complete=budget.accounting_complete,
        )
        result["cpu_seconds"] = budget.used()
        result["synthetic_qa_sequences"] = 4 * result["signature_clone_updates"]
        result["complete"] = (
            result["status"] == "complete"
            and result["completed_updates"] == UPDATES
            and sum(x["qa_sequences"] for x in result["losses"]) == 55296
            and len(result["losses"]) == UPDATES
            and len(result["measurements"]) == UPDATES
            and len(result["rows"]) == 4800
            and result["score_queries"] == 4800
            and result.get("new_session_count") == 43008
            and result["signature_clone_updates"] == 8
            and len(result["restore_checks"]) == 2
            and result["cpu_seconds"] <= 1800
            and all(
                result[k]
                for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
            )
        )
        result["training_sufficient"] = result["complete"] and sufficient(result["milestones"][-1])
        engineering.factorial.write_new(REPORT, result)
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
