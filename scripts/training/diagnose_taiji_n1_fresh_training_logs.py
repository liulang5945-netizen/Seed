"""PLAN79: immutable training logs only; no checkpoint loads or model queries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.training import _taiji_n1_fresh_training_receipt as receipt  # noqa: E402

train = receipt.train
REPORT = ROOT / "reports/taiji_n1_fresh_training_log_diagnostic_20261016.json"
TRAIN_SHA = "dbfecdfbebe46cdd1401f8946020e2d7415eda9ffa0be523a7659d0cbf6fe2c3"  # Frozen complete77 terminal bytes.
PROTOCOL = dict(
    schema="n1-fresh-training-log-diagnostic-v1",
    source_report_sha256=TRAIN_SHA,
    model_loads=0,
    model_queries=0,
    parameter_updates=0,
    confirmation_queries=0,
    cpu_limit_seconds=30,
    milestones=list(train.MILESTONES),
    window_updates=6144,
    final_window_updates=1024,
)
VIEWS = dict(original=192, pair=96, rename=192, rename_pair=96)
GROUPS = ("reader", "writer", "local", "context", "relation")


def identity():
    return dict(
        protocol=PROTOCOL,
        training=train.identity(),
        receipt_validator=train.paired.sha(Path(receipt.__file__)),
        runner=train.paired.sha(Path(__file__)),
    )


def validate_complete(record):
    """Completeness admits failed training evidence, never confirmation queries."""
    expected = {(s, a) for s in train.engineering.SEEDS for a in train.engineering.ARMS}
    if (
        not record["complete"]
        or record["status"] != "complete"
        or record["protocol"] != train.PROTOCOL
        or record["code_identity"] != train.identity()
        or len(record["models"]) != 9
        or {(m["seed"], m["arm"]) for m in record["models"]} != expected
        or record["completed_updates"] != 221184
        or record["signature_clone_updates"] != 72
        or record["synthetic_qa_sequences"] != 144
        or not 0 <= record["cpu_seconds"] <= 9000
        or not all(
            record[k] for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
        )
        or any(record[k] != 0 for k in ("dev_queries", "holdout_queries", "confirmation_queries"))
        or record["training_sufficient"]
        != all(m["training_sufficient"] for m in record["models"] if m["arm"] == "position")
    ):
        raise ValueError("complete unchanged nine-model training evidence required")


def stats(values):
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("nonempty finite full-window values required")
    return dict(count=len(values), mean=fmean(values), min=min(values), max=max(values))


def profiles(record):
    if (
        len(record["losses"]) != 24576
        or len(record["measurements"]) != 24576
        or any(
            m["step"] != i or m["cycle"] != i // 6144 for i, m in enumerate(record["measurements"])
        )
    ):
        raise ValueError("complete ordered updates and fixed cycles required")
    result = []
    for start, end in (*((i, i + 6144) for i in range(0, 24576, 6144)), (23552, 24576)):
        losses = record["losses"][start:end]
        measurements = record["measurements"][start:end]
        if len(losses) != end - start or len(measurements) != end - start:
            raise ValueError("full fixed loss and update windows required")
        groups = {}
        for group in GROUPS:
            selected = [m["groups"][group] for m in measurements]
            groups[group] = dict(
                clipped_count=sum(m["clipped"] for m in selected),
                clipped_fraction=fmean(m["clipped"] for m in selected),
                **{
                    field: stats([m[field] for m in selected])
                    for field in (
                        "actual_l2",
                        "cap",
                        "gradient_l2",
                        "parameter_l2",
                        "proposed_l2",
                        "scale",
                    )
                },
            )
        result.append(
            dict(
                start=start,
                end=end,
                qa_sequences=sum(x["qa_sequences"] for x in losses),
                losses={k: stats([x[k] for x in losses]) for k in ("total", "qa", "observation")},
                groups=groups,
            )
        )
    return result


def transition_cell(item):
    if item["kind"] != "transition" or item["split"] != "train":
        raise ValueError("known train transition only")
    direction = item["observations"][0].removesuffix(".").split()[-1]
    action = item["observations"][1].removesuffix(".").split()[-1]
    if direction not in ("north", "south", "east", "west") or action not in ("left", "right"):
        raise ValueError("frozen direction and turn semantics required")
    return f"{direction}/{action}"


def analyze_rows(rows, views, seed, arm):
    expected = {
        (step, view, item["id"])
        for step in train.MILESTONES
        for view, items in views.items()
        for item in items
    }
    keys = {(r["completed_updates"], r["evaluation_set"], r["id"]) for r in rows}
    if len(rows) != 2880 or len(keys) != 2880 or keys != expected:
        raise ValueError("all exact known rows, five phases and four views required")
    if any(r["seed"] != seed or r["arm"] != arm for r in rows):
        raise ValueError("row model identity differs")
    items_by_view = {v: {item["id"]: item for item in items} for v, items in views.items()}
    if any(r["kind"] != items_by_view[r["evaluation_set"]][r["id"]]["kind"] for r in rows):
        raise ValueError("row kind differs from known curriculum")
    results, previous = [], {}
    for step in train.MILESTONES:
        for view, count in VIEWS.items():
            selected = [
                r for r in rows if r["completed_updates"] == step and r["evaluation_set"] == view
            ]
            if len(selected) != count:
                raise ValueError("full per-view denominator required")
            by_kind = {}
            current = {r["id"]: r["free_exact"] for r in selected}
            for kind in train.paired.KINDS:
                subset = [r for r in selected if r["kind"] == kind]
                by_kind[kind] = dict(
                    items=len(subset),
                    free_exact=sum(r["free_exact"] for r in subset),
                    teacher_exact=sum(r["teacher_exact"] for r in subset),
                    teacher_free_disagree=sum(
                        r["teacher_exact"] != r["free_exact"] for r in subset
                    ),
                    gained=(
                        None
                        if view not in previous
                        else sum(r["free_exact"] and not previous[view][r["id"]] for r in subset)
                    ),
                    lost=(
                        None
                        if view not in previous
                        else sum(not r["free_exact"] and previous[view][r["id"]] for r in subset)
                    ),
                )
            cells = {}
            if view in ("original", "rename"):
                for direction in ("north", "south", "east", "west"):
                    for action in ("left", "right"):
                        key = f"{direction}/{action}"
                        subset = [
                            r
                            for r in selected
                            if r["kind"] == "transition"
                            and transition_cell(items_by_view[view][r["id"]]) == key
                        ]
                        confusion = Counter()
                        for row in subset:
                            item = items_by_view[view][row["id"]]
                            raw = bytes.fromhex(row["raw_hex"])
                            label = (
                                raw.decode("ascii")
                                if raw in (b"north", b"south", b"east", b"west")
                                else "hex:" + raw.hex()
                            )
                            confusion[f"{item['answer']}->{label}/{row['stop_reason']}"] += 1
                        cells[key] = dict(
                            items=len(subset),
                            free_exact=sum(r["free_exact"] for r in subset),
                            teacher_exact=sum(r["teacher_exact"] for r in subset),
                            confusion=dict(confusion),
                        )
            results.append(
                dict(
                    completed_updates=step,
                    view=view,
                    items=count,
                    by_kind=by_kind,
                    transition_cells=cells,
                )
            )
            previous[view] = current
    return results


def validate_phase_summary(phases, milestones):
    for summary in milestones:
        selected = {
            p["view"]: p for p in phases if p["completed_updates"] == summary["completed_updates"]
        }
        for view, total, kinds in (
            ("original", "free_exact", "free_by_kind"),
            ("rename", "rename_free_exact", "rename_free_by_kind"),
        ):
            actual = {k: v["free_exact"] for k, v in selected[view]["by_kind"].items()}
            if sum(actual.values()) != summary[total] or actual != summary[kinds]:
                raise ValueError("raw phase rows disagree with sealed milestone scores")
        for view, field in (("pair", "pair_exact"), ("rename_pair", "rename_pair_exact")):
            if sum(v["free_exact"] for v in selected[view]["by_kind"].values()) != summary[field]:
                raise ValueError("raw pair rows disagree with sealed milestone scores")


def execute():
    if REPORT.exists():
        raise FileExistsError("exclusive immutable diagnostic")
    if not isinstance(TRAIN_SHA, str) or len(TRAIN_SHA) != 64:
        raise ValueError("terminal77 report SHA not yet bound")
    budget = train.paired.CpuBudget(30)
    if train.paired.sha(train.REPORT) != TRAIN_SHA:
        raise ValueError("exact terminal77 report required")
    source = json.loads(train.REPORT.read_text(encoding="utf-8"))
    validate_complete(source)
    guards, _ = train.readiness()
    guards[str(train.REPORT)] = TRAIN_SHA
    items = train.runner.core.checked_train()
    curriculum = train.runner.core.checked_curriculum(items)
    probes = [
        train.runner.core.renaming.rename_item(e, p["names"])
        for e, p in zip(items, curriculum["probes"], strict=True)
    ]
    views = dict(
        original=items,
        pair=train.runner.core.checked_pairs(items),
        rename=probes,
        rename_pair=train.runner.core.build_pairs(probes),
    )
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        models=[],
        model_loads=0,
        model_queries=0,
        parameter_updates=0,
        confirmation_queries=0,
        can_promote=False,
        status="running",
    )
    try:
        for model in source["models"]:
            budget.guard(reserve=3)
            record = receipt.read_model(model)
            guards[model["shard"]["path"]] = model["shard"]["compressed_sha256"]
            for check in model["restore_checks"]:
                guards[check["path"]] = check["sha256"]
            phases = analyze_rows(record["rows"], views, model["seed"], model["arm"])
            validate_phase_summary(phases, model["milestones"])
            result["models"].append(
                dict(
                    seed=model["seed"],
                    arm=model["arm"],
                    training_sufficient=model["training_sufficient"],
                    milestones=model["milestones"],
                    phases=phases,
                    update_profiles=profiles(record),
                )
            )
        result["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    result.update(
        sources_unchanged=all(train.paired.sha(Path(p)) == h for p, h in guards.items()),
        code_unchanged=identity() == result["code_identity"],
    )
    result["cpu_seconds"] = budget.used()
    result["complete"] = (
        result["status"] == "complete"
        and len(result["models"]) == 9
        and result["sources_unchanged"]
        and result["code_unchanged"]
        and result["cpu_seconds"] <= 30
    )
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    if args.execute_approved:
        result = execute()
        print(json.dumps({k: result.get(k) for k in ("complete", "cpu_seconds", "error")}))
        raise SystemExit(0 if result["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
