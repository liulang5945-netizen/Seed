"""PLAN80: frozen train-only factorial interventions, no optimizer steps."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.training import diagnose_taiji_n1_fresh_training_logs as logs  # noqa: E402

train = logs.train
torch = train.torch
MANIFEST = ROOT / "plans/manifests/n1_fresh_transition_factorial_20261016.json"
REPORT = ROOT / "reports/taiji_n1_fresh_transition_factorial_20261016.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
TRAIN_SHA = "dbfecdfbebe46cdd1401f8946020e2d7415eda9ffa0be523a7659d0cbf6fe2c3"
LOG_SHA = "ab01c31d1ed6852dbe2a0c48ca3fefd826c2ff2b72e7503eacfe53af2b8ed336"
MODELS = ((11, "position"), (12, "position"), (13, "position"), (13, "cut"))
CONDITIONS = ("original", "turn_flip", "direction_opposite", "both")
DIRECTIONS = ("north", "east", "south", "west")
BASE_FIELDS = (
    "id",
    "kind",
    "target_symbols",
    "teacher_symbols",
    "teacher_tokens_correct",
    "teacher_exact",
    "teacher_ce",
    "raw_hex",
    "output_hex",
    "stop_reason",
    "generated_steps",
    "free_exact",
)
PROTOCOL = dict(
    schema="n1-fresh-transition-factorial-diagnostic-v1",
    training_sha256=TRAIN_SHA,
    log_sha256=LOG_SHA,
    models=[list(x) for x in MODELS],
    conditions=list(CONDITIONS),
    parents=48,
    manifest_items=192,
    model_queries=768,
    model_loads=4,
    parameter_updates=0,
    optimizer_steps=0,
    signature_clone_updates=0,
    dev_queries=0,
    v3_holdout_queries=0,
    v4_queries=0,
    confirmation_queries=0,
    max_generated_bytes=64,
    cpu_limit_seconds=60,
    retry_allowed=False,
)


def identity():
    return dict(
        protocol=PROTOCOL,
        training=train.identity(),
        logs=logs.identity(),
        runner=train.paired.sha(Path(__file__)),
    )


def write_new(path, record):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def variants(item):
    if item["split"] != "train" or item["kind"] != "transition" or len(item["observations"]) != 2:
        raise ValueError("known two-observation train transition required")
    face_prefix, direction = item["observations"][0].removesuffix(".").rsplit(" ", 1)
    turn_prefix, turn = item["observations"][1].removesuffix(".").rsplit(" ", 1)
    if (
        not face_prefix.endswith(" faces")
        or not turn_prefix.endswith(" turns")
        or face_prefix.removesuffix(" faces") != turn_prefix.removesuffix(" turns")
        or direction not in DIRECTIONS
        or turn not in ("left", "right")
    ):
        raise ValueError("frozen face/turn grammar required")
    result = []
    for condition in CONDITIONS:
        d = DIRECTIONS[
            (
                DIRECTIONS.index(direction)
                + (2 if condition in ("direction_opposite", "both") else 0)
            )
            % 4
        ]
        t = ("right" if turn == "left" else "left") if condition in ("turn_flip", "both") else turn
        candidate = deepcopy(item)
        candidate["observations"] = [f"{face_prefix} {d}.", f"{turn_prefix} {t}."]
        candidate["answer"] = DIRECTIONS[(DIRECTIONS.index(d) + (1 if t == "right" else -1)) % 4]
        if condition == "original" and candidate != item:
            raise ValueError("original annotation and bytes must be exactly reproduced")
        result.append(
            dict(
                case_id=f"{item['id']}:{condition}",
                parent_id=item["id"],
                condition=condition,
                initial_direction=d,
                turn=t,
                item=candidate,
            )
        )
    return result


def build(items):
    selected = [x for x in items if x["kind"] == "transition"]
    if len(selected) != 48 or len({x["id"] for x in selected}) != 48:
        raise ValueError("all48 unique training parents required")
    return dict(
        schema="n1-fresh-transition-factorial-manifest-v1",
        protocol=PROTOCOL,
        code_identity=identity(),
        entries=[v for item in selected for v in variants(item)],
    )


def readiness():
    if train.paired.sha(train.REPORT) != TRAIN_SHA or train.paired.sha(logs.REPORT) != LOG_SHA:
        raise ValueError("exact complete77 and79 sources required")
    source = json.loads(train.REPORT.read_text(encoding="utf-8"))
    diagnostic = json.loads(logs.REPORT.read_text(encoding="utf-8"))
    logs.validate_complete(source)
    if (
        not diagnostic["complete"]
        or diagnostic["code_identity"] != logs.identity()
        or len(diagnostic["models"]) != 9
    ):
        raise ValueError("complete unchanged79 nine-shard validation required")
    guards = dict(diagnostic["source_guards"])
    guards[str(logs.REPORT)] = LOG_SHA
    if any(train.paired.sha(Path(p)) != h for p, h in guards.items()):
        raise ValueError("frozen training/shard/restore source chain changed")
    selected, baselines = {}, {}
    for model in source["models"]:
        key = model["seed"], model["arm"]
        if key not in MODELS:
            continue
        record = logs.receipt.read_model(model)
        baselines[key] = {
            r["id"]: r
            for r in record["rows"]
            if r["completed_updates"] == 24576
            and r["evaluation_set"] == "original"
            and r["kind"] == "transition"
        }
        if len(baselines[key]) != 48:
            raise ValueError("all48 sealed native transition rows required")
        paths = [
            Path(c["path"])
            for c in model["restore_checks"]
            if Path(c["path"]).name == f"s{key[0]}_{key[1]}_final.pt"
        ]
        if len(paths) != 1:
            raise ValueError("one frozen final endpoint per selected model required")
        selected[key] = paths[0]
    if set(selected) != set(MODELS):
        raise ValueError("all four preregistered endpoints required")
    return guards, selected, baselines, train.runner.core.checked_train()


def evaluate(model, optimizer, seed, arm, entries, baseline, budget, result):
    before = train.paired.fingerprint(model.bundle(optimizer))
    try:
        for entry in entries:
            budget.guard(reserve=4)
            session = f"factorial80-{seed}-{arm}-{entry['case_id']}"
            if session in model._observations:
                raise ValueError("diagnostic session collides with frozen history")
            result["model_queries"] += 1
            try:
                row = train.runner.core.measure(model, entry["item"], arm, session)
                result["rows"].append(
                    dict(
                        row,
                        seed=seed,
                        arm=arm,
                        **{
                            k: entry[k]
                            for k in (
                                "case_id",
                                "parent_id",
                                "condition",
                                "initial_direction",
                                "turn",
                            )
                        },
                    )
                )
                if entry["condition"] == "original":
                    if any(row[k] != baseline[entry["parent_id"]][k] for k in BASE_FIELDS):
                        raise ValueError("native row does not exactly reproduce77")
                    result["native_reproductions"] += 1
            finally:
                model._observations.pop(session, None)
    finally:
        if train.paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("factorial queries changed parameters/Adam/RNG/history")


def summarize(rows):
    keys = {(r["seed"], r["arm"], r["case_id"]) for r in rows}
    if len(rows) != 768 or len(keys) != 768 or {(r["seed"], r["arm"]) for r in rows} != set(MODELS):
        raise ValueError("all768 unique factorial rows required")
    result = []
    shared_cases = None
    for seed, arm in MODELS:
        selected = [r for r in rows if r["seed"] == seed and r["arm"] == arm]
        cases = {r["case_id"] for r in selected}
        if shared_cases is not None and cases != shared_cases:
            raise ValueError("all models must share the same192 cases")
        shared_cases = cases
        parents = {r["parent_id"] for r in selected}
        if len(parents) != 48 or Counter(r["condition"] for r in selected) != {
            c: 48 for c in CONDITIONS
        }:
            raise ValueError("48 parents and all four complete conditions required")
        grouped = {p: {r["condition"]: r for r in selected if r["parent_id"] == p} for p in parents}
        if any(
            set(g) != set(CONDITIONS) or sum(r["parent_id"] == p for r in selected) != 4
            for p, g in grouped.items()
        ):
            raise ValueError("each parent requires exactly the four conditions")

        def response(row):
            return row["raw_hex"], row["stop_reason"]

        conditions = {}
        for condition in CONDITIONS:
            subset = [r for r in selected if r["condition"] == condition]
            cells = {}
            for direction in DIRECTIONS:
                for turn in ("left", "right"):
                    cell = [
                        r
                        for r in subset
                        if r["initial_direction"] == direction and r["turn"] == turn
                    ]
                    cells[f"{direction}/{turn}"] = dict(
                        items=len(cell),
                        free_exact=sum(r["free_exact"] for r in cell),
                        teacher_exact=sum(r["teacher_exact"] for r in cell),
                    )
            conditions[condition] = dict(
                items=48,
                free_exact=sum(r["free_exact"] for r in subset),
                teacher_exact=sum(r["teacher_exact"] for r in subset),
                cells=cells,
            )
        result.append(
            dict(
                seed=seed,
                arm=arm,
                conditions=conditions,
                joint_four_exact=sum(
                    all(r["free_exact"] for r in g.values()) for g in grouped.values()
                ),
                single_flip_changed={
                    c: sum(response(g[c]) != response(g["original"]) for g in grouped.values())
                    for c in ("turn_flip", "direction_opposite")
                },
                double_flip_unchanged=sum(
                    response(g["both"]) == response(g["original"]) for g in grouped.values()
                ),
            )
        )
    return result


def execute():
    if REPORT.exists() or ATTEMPT.exists():
        raise FileExistsError("one fixed diagnostic attempt; no automatic retry")
    budget = train.paired.CpuBudget(60)
    guards, paths, baselines, items = readiness()
    manifest = build(items)
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text(encoding="utf-8")) != manifest:
            raise ValueError("existing deterministic manifest differs; no overwrite")
    else:
        write_new(MANIFEST, manifest)
    guards[str(MANIFEST)] = train.paired.sha(MANIFEST)
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        rows=[],
        restore_checks=[],
        model_loads=0,
        model_queries=0,
        native_reproductions=0,
        parameter_updates=0,
        optimizer_steps=0,
        signature_clone_updates=0,
        dev_queries=0,
        v3_holdout_queries=0,
        v4_queries=0,
        confirmation_queries=0,
        can_promote=False,
        status="running",
    )
    write_new(
        ATTEMPT,
        dict(
            protocol=PROTOCOL,
            code_identity=result["code_identity"],
            source_guards=guards,
            confirmation_face_consumed=False,
        ),
    )
    try:
        for seed, arm in MODELS:
            budget.guard(reserve=5)
            payload = torch.load(paths[seed, arm], map_location="cpu", weights_only=True)
            result["model_loads"] += 1
            if (
                payload["seed"] != seed
                or payload["arm"] != arm
                or payload["completed_updates"] != 24576
            ):
                raise ValueError("fixed full final endpoint required")
            model, optimizer = train.load(payload)
            fingerprint = train.paired.fingerprint(model.bundle(optimizer))
            if (
                fingerprint != train.paired.fingerprint(payload["candidate"])
                or fingerprint != payload["expected_signature"]["bundle"]
            ):
                raise ValueError("full frozen77 model/Adam/RNG/history restoration differs")
            evaluate(
                model,
                optimizer,
                seed,
                arm,
                manifest["entries"],
                baselines[seed, arm],
                budget,
                result,
            )
            result["restore_checks"].append(
                dict(
                    seed=seed,
                    arm=arm,
                    path=str(paths[seed, arm]),
                    full_bundle_equal=True,
                    queries_state_neutral=True,
                )
            )
        result["summary"] = summarize(result["rows"])
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
        and len(result["rows"]) == 768
        and result["model_queries"] == 768
        and result["model_loads"] == 4
        and result["native_reproductions"] == 192
        and len(result["restore_checks"]) == 4
        and result["sources_unchanged"]
        and result["code_unchanged"]
        and result["cpu_seconds"] <= 60
    )
    write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.execute_approved:
        result = execute()
        print(
            json.dumps(
                {k: result.get(k) for k in ("complete", "cpu_seconds", "model_queries", "error")}
            )
        )
        raise SystemExit(0 if result["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
