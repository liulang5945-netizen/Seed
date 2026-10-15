"""PLAN72: frozen writer-only exchanges on previously known training/probe items."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import eval_taiji_n1_frozen_v4_confirmation as confirmation  # noqa: E402

train = confirmation.train
runner = train.runner
paired = train.paired
REPORT = ROOT / "reports/taiji_n1_frozen_writer_exchange_20261016.json"
CONFIRMATION_SHA = "eaee44f6eb980d9b0a3632a39a32c3898d6eecae72e869cc5fff33c45160250e"
CONDITIONS = (
    ("position", "native"),
    ("position", "mother_writer"),
    ("cut", "native"),
    ("cut", "position_writer"),
)
PROTOCOL = dict(
    schema="n1-frozen-writer-exchange-v1",
    confirmation_sha256=CONFIRMATION_SHA,
    seeds=list(train.SEEDS),
    conditions=[list(c) for c in CONDITIONS],
    known_rows=6912,
    parameter_updates=0,
    optimizer_steps=0,
    signature_clone_updates=0,
    dev_queries=0,
    holdout_queries=0,
    confirmation_queries=0,
    cpu_limit_seconds=180,
    swapped_parameters="exactly-four-writer.*-parameters-only",
)


def identity():
    return dict(
        protocol=PROTOCOL, confirmation=confirmation.identity(), runner=paired.sha(Path(__file__))
    )


def readiness():
    if paired.sha(confirmation.REPORT) != CONFIRMATION_SHA:
        raise ValueError("frozen complete71 required")
    record = json.loads(confirmation.REPORT.read_text(encoding="utf-8"))
    if (
        not record["complete"]
        or record["passed"]
        or record["code_identity"] != confirmation.identity()
        or record["confirmation_queries"] != 288
    ):
        raise ValueError("unchanged71 with original failed credit gate required")
    guards = dict(record["source_guards"])
    guards[str(confirmation.REPORT)] = CONFIRMATION_SHA
    guards[str(confirmation.ATTEMPT)] = (
        "7cfd1a0479ab349169dbb68eacb8a4d7d3fc8b722c2a7e55e14187e5591ede98"
    )
    if not all(paired.sha(Path(p)) == h for p, h in guards.items()):
        raise ValueError("source chain or sealed checkpoint changed")
    baseline = json.loads(train.REPORT.read_text(encoding="utf-8"))
    if not baseline["complete"] or baseline["code_identity"] != train.identity():
        raise ValueError("unchanged complete70 required")
    _, mother, _ = runner.transfer.source_payload()
    paths = {}
    for check in baseline["restore_checks"]:
        for seed in train.SEEDS:
            for arm in ("position", "cut"):
                if Path(check["path"]).name == f"s{seed}_{arm}_final.pt":
                    paths[seed, arm] = Path(check["path"])
    if len(paths) != 6:
        raise ValueError("six frozen position/cut endpoints required")
    return baseline, mother, paths, guards


def writer_state(state):
    values = {n: v.detach().clone() for n, v in state.items() if n.startswith("writer.")}
    if len(values) != 4:
        raise ValueError("exactly four writer parameters required")
    return values


@contextmanager
def exchange(model, optimizer, donor):
    before = paired.fingerprint(model.bundle(optimizer))
    parameters = dict(model.named_parameters())
    original = writer_state(parameters)
    if set(donor) != set(original) or any(
        donor[n].shape != original[n].shape or donor[n].dtype != original[n].dtype for n in original
    ):
        raise ValueError("exact writer keys, shapes and dtype required")
    try:
        with torch.no_grad():
            for name, value in donor.items():
                parameters[name].copy_(value)
        yield
    finally:
        with torch.no_grad():
            for name, value in original.items():
                parameters[name].copy_(value)
        if paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("writer exchange changed parent parameters/Adam/RNG/history")


def measure_condition(model, optimizer, seed, arm, condition, views, budget, output):
    before = paired.fingerprint(model.bundle(optimizer))
    rows = []
    try:
        for view, items in views.items():
            for item in items:
                budget.guard(reserve=5)
                session = f"writer72-{seed}-{arm}-{condition}-{view}-{item['id']}"
                if session in model._observations:
                    raise ValueError("diagnostic session collides with frozen history")
                try:
                    measured = runner.core.measure(model, item, arm, session)
                    row = dict(
                        measured, seed=seed, arm=arm, completed_updates=6144, evaluation_set=view
                    )
                    rows.append(row)
                    output.append(dict(row, condition=condition))
                finally:
                    model._observations.pop(session, None)
    finally:
        if paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("known-item evaluation changed complete intervened state")
    selected = {v: [r for r in rows if r["evaluation_set"] == v] for v in views}
    summary = runner.core.summarize(selected["original"])[str(seed)][arm]
    renamed = runner.core.summarize(selected["rename"])[str(seed)][arm]
    summary.update(
        pair_exact=sum(r["free_exact"] for r in selected["pair"]),
        rename_free_exact=renamed["free_exact"],
        rename_free_by_kind=renamed["free_by_kind"],
        rename_pair_exact=sum(r["free_exact"] for r in selected["rename_pair"]),
    )
    return rows, summary


def drift(first, second):
    distance = math.sqrt(
        sum(float(torch.linalg.vector_norm(first[n] - second[n])) ** 2 for n in first)
    )
    base = math.sqrt(sum(float(torch.linalg.vector_norm(first[n])) ** 2 for n in first))
    return dict(
        l2=distance,
        relative_l2=distance / max(base, 1e-30),
        first=paired.fingerprint(first),
        second=paired.fingerprint(second),
    )


def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive fixed diagnostic, no restart")
    budget = paired.CpuBudget(180)
    baseline, mother, paths, guards = readiness()
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
        status="running",
        rows=[],
        conditions=[],
        writer_drift=[],
        parameter_updates=0,
        optimizer_steps=0,
        signature_clone_updates=0,
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        can_promote=False,
    )
    try:
        for seed in train.SEEDS:
            models = {}
            for arm in ("position", "cut"):
                payload = torch.load(paths[seed, arm], map_location="cpu", weights_only=True)
                if (
                    payload["completed_updates"] != 6144
                    or payload["seed"] != seed
                    or payload["arm"] != arm
                ):
                    raise ValueError("frozen final identity required")
                model, optimizer = train.load(payload)
                if (
                    paired.fingerprint(model.bundle(optimizer))
                    != payload["expected_signature"]["bundle"]
                ):
                    raise ValueError("full parent recovery differs from70")
                models[arm] = (model, optimizer)
            native = {arm: writer_state(m.state_dict()) for arm, (m, _) in models.items()}
            parent = writer_state(mother["candidate"]["state"])
            if paired.fingerprint(native["cut"]) != paired.fingerprint(parent):
                raise ValueError("cut writer must exactly retain mother parameters")
            result["writer_drift"].append(dict(seed=seed, **drift(parent, native["position"])))
            for arm, condition in CONDITIONS:
                model, optimizer = models[arm]
                donor = (
                    native[arm]
                    if condition == "native"
                    else parent if condition == "mother_writer" else native["position"]
                )
                with exchange(model, optimizer, donor):
                    rows, summary = measure_condition(
                        model, optimizer, seed, arm, condition, views, budget, result["rows"]
                    )
                    if condition == "native":
                        expected = [
                            r
                            for r in baseline["rows"]
                            if r["seed"] == seed
                            and r["arm"] == arm
                            and r["completed_updates"] == 6144
                        ]
                        if len(expected) != 576 or rows != expected:
                            raise ValueError("native576 rows must reproduce70 exactly")
                result["conditions"].append(
                    dict(
                        seed=seed,
                        arm=arm,
                        condition=condition,
                        summary=summary,
                        native_reproduced=condition == "native",
                        parent_unchanged=True,
                    )
                )
                print(
                    f"writer72 seed={seed} arm={arm} condition={condition} original={summary['free_exact']} pair={summary['pair_exact']} rename={summary['rename_free_exact']} rename_pair={summary['rename_pair_exact']} cpu={budget.used():.2f}",
                    flush=True,
                )
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="no automatic restart")
    finally:
        result.update(
            sources_unchanged=all(paired.sha(Path(p)) == h for p, h in guards.items()),
            code_unchanged=identity() == result["code_identity"],
        )
        result["cpu_seconds"] = budget.used()
        result["complete"] = (
            result["status"] == "complete"
            and len(result["rows"]) == 6912
            and len(result["conditions"]) == 12
            and result["cpu_seconds"] <= 180
            and result["sources_unchanged"]
            and result["code_unchanged"]
        )
        train.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.execute_approved:
        result = audit()
        print(json.dumps({k: result.get(k) for k in ("complete", "cpu_seconds", "error")}))
        raise SystemExit(0 if result["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
