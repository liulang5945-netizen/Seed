"""PLAN65: offline analysis of frozen PLAN63 logs; no model imports or queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/taiji_n1_relation_paired_train_20261015.json"
SOURCE_SHA = "d982909f6722d37040fb21f7d869aed6817f3a42a3c4b08300b6746ef14e2922"
REPORT = ROOT / "reports/taiji_n1_relation_seed_drift_20261015.json"
MANIFEST = ROOT / "plans/manifests/n1_native_evidence_curriculum_v3_20261015.json"
MANIFEST_SHA = "ac56fb0aba2ef602779f58749e6f8c35324c40c433fd16ce55b89689235b0604"
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
VIEWS = {"original": 192, "pair": 96, "rename": 192, "rename_pair": 96}
STEPS = (0, 3072, 6144)
KINDS = ("lookup", "revision", "transition", "unknown")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory(rows, seed, arm, view, expected_count):
    selected = [r for r in rows if (r["seed"], r["arm"], r["evaluation_set"]) == (seed, arm, view)]
    by_step = {}
    summaries = []
    for step in STEPS:
        subset = [r for r in selected if r["completed_updates"] == step]
        indexed = {r["id"]: r for r in subset}
        if len(subset) != expected_count or len(indexed) != expected_count:
            raise ValueError("complete unique fixed milestone rows required")
        by_step[step] = indexed
        summaries.append(
            dict(
                step=step,
                free_exact=sum(r["free_exact"] for r in subset),
                teacher_exact=sum(r["teacher_exact"] for r in subset),
                mean_teacher_ce=statistics.mean(r["teacher_ce"] for r in subset),
                by_kind={
                    k: dict(
                        items=sum(r["kind"] == k for r in subset),
                        free_exact=sum(r["free_exact"] for r in subset if r["kind"] == k),
                        teacher_exact=sum(r["teacher_exact"] for r in subset if r["kind"] == k),
                    )
                    for k in KINDS
                },
                stop_reasons=dict(Counter(r["stop_reason"] for r in subset)),
                mean_generated_steps=statistics.mean(r["generated_steps"] for r in subset),
            )
        )
    if any(set(by_step[s]) != set(by_step[0]) for s in STEPS):
        raise ValueError("milestone item identities changed")
    transitions = []
    for before, after in zip(STEPS[:-1], STEPS[1:], strict=True):
        flips = []
        for key, old in by_step[before].items():
            new = by_step[after][key]
            if old["kind"] != new["kind"] or old["target_symbols"] != new["target_symbols"]:
                raise ValueError("kind or target changed across milestones")
            if bool(old["free_exact"]) != bool(new["free_exact"]):
                flips.append(
                    dict(
                        id=key,
                        kind=old["kind"],
                        direction="lost" if old["free_exact"] else "gained",
                        teacher_before=old["teacher_exact"],
                        teacher_after=new["teacher_exact"],
                        ce_before=old["teacher_ce"],
                        ce_after=new["teacher_ce"],
                        raw_before=old["raw_hex"],
                        raw_after=new["raw_hex"],
                        stop_before=old["stop_reason"],
                        stop_after=new["stop_reason"],
                    )
                )
        transitions.append(
            dict(
                before=before,
                after=after,
                counts={d: sum(f["direction"] == d for f in flips) for d in ("lost", "gained")},
                flips=flips,
            )
        )
    return dict(seed=seed, arm=arm, view=view, milestones=summaries, transitions=transitions)


def loss_windows(losses, train):
    if len(losses) != 6144 or len(train) != 192:
        raise ValueError("full fixed loss stream and original training order required")
    output = []
    for start in range(0, 6144, 1024):
        subset = losses[start : start + 1024]
        groups = {}
        for offset, loss in enumerate(subset, start):
            kind = train[(offset // 2) % 192]["kind"]
            view = "renamed" if offset % 2 else "original"
            expected_qa = 2 if kind in ("lookup", "unknown") else 1
            if loss["qa_sequences"] != expected_qa:
                raise ValueError("loss schedule does not match the frozen parent stream")
            groups.setdefault(f"{view}/{kind}", []).append(loss)
        output.append(
            dict(
                start=start,
                end=start + 1024,
                **{
                    field: dict(
                        mean=statistics.mean(r[field] for r in subset),
                        maximum=max(r[field] for r in subset),
                        p95=sorted(r[field] for r in subset)[int(0.95 * (len(subset) - 1))],
                    )
                    for field in ("total", "qa", "observation")
                },
                groups={
                    key: dict(items=len(value), mean_qa=statistics.mean(r["qa"] for r in value))
                    for key, value in groups.items()
                },
            )
        )
    return output


def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive offline diagnostic; no overwrite")
    if len(SOURCE_SHA) != 64 or sha(SOURCE) != SOURCE_SHA or sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("terminal source report must be frozen before this diagnostic")
    record = json.loads(SOURCE.read_text(encoding="utf-8"))
    if (
        not record["complete"]
        or record["status"] != "complete"
        or record["completed_updates"] != 55296
        or len(record["rows"]) != 15552
        or len(record["restore_checks"]) != 18
        or len(record["losses"]) != 55296
        or sum(r["qa_sequences"] for r in record["losses"]) != 82944
        or not all(
            record[k] for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
        )
        or any(record[k] != 0 for k in ("dev_queries", "holdout_queries", "confirmation_queries"))
    ):
        raise ValueError("complete fixed PLAN63 and zero unseen queries required")
    train = [
        e
        for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["episodes"]
        if e["split"] == "train"
    ]
    trajectories = [
        trajectory(record["rows"], seed, arm, view, count)
        for seed in SEEDS
        for arm in ARMS
        for view, count in VIEWS.items()
    ]
    losses = [
        dict(
            seed=seed,
            arm=arm,
            windows=loss_windows(
                [r for r in record["losses"] if (r["seed"], r["arm"]) == (seed, arm)], train
            ),
        )
        for seed in SEEDS
        for arm in ARMS
    ]
    result = dict(
        schema="n1-relation-seed-drift-offline-v1",
        source_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        code_sha256=sha(Path(__file__)),
        trajectories=trajectories,
        loss_windows=losses,
        parameter_updates=0,
        model_queries=0,
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        can_promote=False,
        cpu_limit_seconds=30,
        cpu_seconds=time.process_time(),
        sources_unchanged=sha(SOURCE) == SOURCE_SHA and sha(MANIFEST) == MANIFEST_SHA,
    )
    result["complete"] = result["sources_unchanged"] and result["cpu_seconds"] <= 30
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    if parser.parse_args().execute_approved:
        result = audit()
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "model_queries")}))
    else:
        print(
            json.dumps(dict(read_only=True, source_frozen=len(SOURCE_SHA) == 64, model_queries=0))
        )
