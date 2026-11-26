"""PLAN71: one sealed confirmation attempt, frozen models, no optimizer steps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import build_taiji_n1_confirmation_v4 as face  # noqa: E402
from scripts.training import run_taiji_n1_group_relative_paired as train  # noqa: E402

runner = train.runner
paired = train.paired
REPORT = ROOT / "reports/taiji_n1_frozen_v4_confirmation_20261016.json"
ATTEMPT = ROOT / "reports/taiji_n1_frozen_v4_confirmation_attempt_20261016.json"
TRAIN_SHA = "f0d4e6d66dc31759a8ce9aef3888ce7f8bd385a5e5c6381b698b275d0a6be5cf"
PROTOCOL = dict(
    schema="n1-frozen-v4-confirmation-v1",
    seeds=list(train.SEEDS),
    arms=list(train.ARMS),
    source_report_sha256=TRAIN_SHA,
    confirmation_items_per_model=32,
    confirmation_queries=288,
    parameter_updates=0,
    optimizer_steps=0,
    signature_clone_updates=0,
    cpu_limit_seconds=180,
    max_generated_bytes=64,
    gates=dict(each_position_seed=16, each_kind=2, aggregate_advantage_each_control=12),
    v3_dev_queries=0,
    v3_holdout_queries=0,
    retry_allowed=False,
)


def identity():
    return dict(
        protocol=PROTOCOL,
        training=train.identity(),
        face_builder=paired.sha(Path(face.__file__)),
        runner=paired.sha(Path(__file__)),
    )


def readiness():
    if paired.sha(train.REPORT) != TRAIN_SHA:
        raise ValueError("exact frozen complete70 report required")
    record = json.loads(train.REPORT.read_text(encoding="utf-8"))
    expected_models = {(seed, arm) for seed in train.SEEDS for arm in train.ARMS}
    if (
        not record["complete"]
        or not record["confirmation_admitted"]
        or record["code_identity"] != train.identity()
        or {(m["seed"], m["arm"]) for m in record["models"]} != expected_models
        or len(record["models"]) != 9
        or len(record["restore_checks"]) != 18
        or record["completed_updates"] != 55296
        or record["confirmation_queries"] != 0
        or not all(
            runner.sufficient(m["milestones"][-1])
            for m in record["models"]
            if m["arm"] == "position"
        )
    ):
        raise ValueError("complete70 nine models/18 recoveries and all original gates required")
    _, guards, seals = train.readiness()
    guards[str(train.REPORT)] = TRAIN_SHA
    selected = {}
    for check in record["restore_checks"]:
        path = Path(check["path"])
        if not check["equal"] or paired.sha(path) != check["sha256"]:
            raise ValueError("70 checkpoint bytes changed")
        guards[str(path)] = check["sha256"]
        for seed, arm in expected_models:
            if path.name == f"s{seed}_{arm}_final.pt":
                if (seed, arm) in selected:
                    raise ValueError("duplicate frozen endpoint")
                selected[seed, arm] = path
    if set(selected) != expected_models:
        raise ValueError("all nine final endpoints required")
    if not all(paired.sha(Path(p)) == h for p, h in guards.items()):
        raise ValueError("frozen source chain changed")
    runner.seal.check_all()
    manifest = json.loads(face.MANIFEST.read_text(encoding="utf-8"))
    face.validate(manifest)
    return guards, seals, selected, manifest["episodes"]


@torch.no_grad()
def measure(model, item, arm, session):
    """Explicit holdout path; never relabel a held-out item to bypass train guards."""
    if item["split"] != "holdout" or arm not in train.ARMS:
        raise ValueError("confirmation holdout split and frozen arm required")
    data = paired.project(item)
    paired.observe(model, data, session)
    tape = paired.evidence(model, session, arm)
    target = item["answer"].encode()
    labels = torch.tensor([*target, 256])
    logits, _ = model.teacher_logits(tape, session, data.question, target)
    predicted = logits.argmax(dim=-1)
    trace = model.generate_trace(tape, session, data.question, 64)
    return dict(
        id=item["id"],
        kind=item["kind"],
        split=item["split"],
        target_symbols=labels.tolist(),
        teacher_symbols=predicted.tolist(),
        teacher_tokens_correct=int((predicted == labels).sum()),
        teacher_exact=torch.equal(predicted, labels),
        teacher_ce=float(torch.nn.functional.cross_entropy(logits, labels)),
        raw_hex=trace.raw.hex(),
        output_hex=trace.output.hex(),
        stop_reason=trace.stop_reason,
        generated_steps=trace.generated_steps,
        free_exact=trace.raw == target and trace.stop_reason == "boundary",
    )


def evaluate(model, optimizer, seed, arm, items, budget, *, output=None, accounting=None):
    before = paired.fingerprint(model.bundle(optimizer))
    rows = [] if output is None else output
    start = len(rows)
    try:
        for item in items:
            budget.guard(reserve=5)
            session = f"confirmation71-{seed}-{arm}-{item['id']}"
            if session in model._observations:
                raise ValueError("confirmation session collides with frozen history")
            if accounting is not None:
                accounting["confirmation_queries"] += 1
            try:
                row = measure(model, item, arm, session)
                rows.append(dict(row, seed=seed, arm=arm))
            finally:
                model._observations.pop(session, None)
    finally:
        if paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("confirmation changed parameters, optimizer, RNG or history")
    return rows[start:]


def adjudicate(rows):
    keys = {(r["seed"], r["arm"], r["id"]) for r in rows}
    ids = {r["id"] for r in rows}
    if len(rows) != 288 or len(keys) != 288 or len(ids) != 32:
        raise ValueError("all288 unique paired confirmation rows required")
    for seed in train.SEEDS:
        for arm in train.ARMS:
            selected = [r for r in rows if r["seed"] == seed and r["arm"] == arm]
            if {r["id"] for r in selected} != ids or any(
                sum(r["kind"] == k for r in selected) != 8 for k in paired.KINDS
            ):
                raise ValueError("same32 items/four classes per frozen model required")
    summary = runner.core.summarize(rows)
    positions = [summary[str(seed)]["position"] for seed in train.SEEDS]
    totals = {
        arm: sum(summary[str(seed)][arm]["free_exact"] for seed in train.SEEDS)
        for arm in train.ARMS
    }
    gates = dict(
        each_position_seed=all(s["free_exact"] >= 16 for s in positions),
        each_position_kind=all(
            all(s["free_by_kind"][k] >= 2 for k in paired.KINDS) for s in positions
        ),
        advantage_terminal=totals["position"] >= totals["terminal"] + 12,
        advantage_cut=totals["position"] >= totals["cut"] + 12,
    )
    return dict(
        summary=summary, aggregate_exact=totals, gates=gates, behavioral_pass=all(gates.values())
    )


def audit():
    if REPORT.exists() or ATTEMPT.exists():
        raise FileExistsError(
            "one confirmation attempt only; face remains consumed after interruption"
        )
    budget = paired.CpuBudget(180)
    guards, seals, paths, items = readiness()
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        confirmation_seals=seals,
        status="running",
        rows=[],
        restore_checks=[],
        parameter_updates=0,
        optimizer_steps=0,
        signature_clone_updates=0,
        dev_queries=0,
        v3_holdout_queries=0,
        confirmation_queries=0,
        confirmation_face_consumed=True,
        can_promote=False,
    )
    train.write_new(
        ATTEMPT,
        dict(
            protocol=PROTOCOL,
            code_identity=result["code_identity"],
            source_guards=guards,
            confirmation_seals=seals,
            face_consumed=True,
        ),
    )
    try:
        for seed in train.SEEDS:
            for arm in train.ARMS:
                budget.guard(reserve=5)
                path = paths[seed, arm]
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if (
                    payload["seed"] != seed
                    or payload["arm"] != arm
                    or payload["completed_updates"] != 6144
                ):
                    raise ValueError("exact frozen final seed and arm required")
                model, optimizer = train.load(payload)
                fingerprint = paired.fingerprint(model.bundle(optimizer))
                if (
                    fingerprint != paired.fingerprint(payload["candidate"])
                    or fingerprint != payload["expected_signature"]["bundle"]
                ):
                    raise ValueError("full final model/Adam/RNG/history restore differs from70")
                result["restore_checks"].append(
                    dict(seed=seed, arm=arm, path=str(path), full_bundle_equal=True)
                )
                rows = evaluate(
                    model,
                    optimizer,
                    seed,
                    arm,
                    items,
                    budget,
                    output=result["rows"],
                    accounting=result,
                )
                print(
                    f"confirmation71 seed={seed} arm={arm} exact={sum(r['free_exact'] for r in rows)}/32 cpu={budget.used():.2f}",
                    flush=True,
                )
        result.update(adjudicate(result["rows"]))
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="face consumed; no automatic retry")
    finally:
        result.update(
            sources_unchanged=all(paired.sha(Path(p)) == h for p, h in guards.items()),
            code_unchanged=identity() == result["code_identity"],
        )
        result["cpu_seconds"] = budget.used()
        result["complete"] = (
            result["status"] == "complete"
            and len(result["rows"]) == 288
            and len(result["restore_checks"]) == 9
            and result["cpu_seconds"] <= 180
            and result["sources_unchanged"]
            and result["code_unchanged"]
        )
        result["passed"] = result["complete"] and result.get("behavioral_pass", False)
        train.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.execute_approved:
        result = audit()
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in (
                        "complete",
                        "passed",
                        "aggregate_exact",
                        "gates",
                        "cpu_seconds",
                        "error",
                    )
                }
            )
        )
        raise SystemExit(0 if result["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
