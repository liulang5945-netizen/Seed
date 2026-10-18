"""PLAN-N1-114: attribute seed13/position update metrics by frozen task schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASELINE_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_training_20261017.json"
CANDIDATE_REPORT = ROOT / "reports/taiji_n1_length_balanced_training_20261018.json"
BASELINE_SCHEDULE = ROOT / "plans/manifests/n1_hard_neighbor_training_schedule_v1_20261017.json"
CANDIDATE_SCHEDULE = (
    ROOT / "plans/manifests/n1_hard_neighbor_length_balanced_schedule_v1_20261018.json"
)
REPORT = ROOT / "reports/taiji_n1_seed13_parameter_update_trajectory_attribution_20261018.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")

EXPECTED = {
    BASELINE_REPORT: "dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909",
    CANDIDATE_REPORT: "e848433de597d378e9e1dc52bf488f296f90c5ce67f3855e19bcfe49268aaa0c",
    BASELINE_SCHEDULE: "dc3a96ad17d43bd898e403ecd119c12c781317a71ab06ba39cdab6db24c40d50",
    CANDIDATE_SCHEDULE: "b2d637fcc1e096d40bba35bbc91bc66b0c27257cd74a4bcac5f5f488f55b6d4c",
}
CPU_LIMIT_SECONDS = 120.0
SEED = 13
ARM = "position"
UPDATES = 24_576
MILESTONES = (0, 6_144, 12_288, 18_432, 24_576)
GROUPS = ("reader", "writer", "local", "context", "relation")
KINDS = ("lookup", "revision", "transition", "unknown")
RUNS = ("baseline", "candidate")
CODE_PATHS = (
    ROOT / "scripts/training/build_taiji_n1_hard_neighbor_training_schedule.py",
    ROOT / "scripts/training/build_taiji_n1_length_balanced_training_schedule.py",
    ROOT / "scripts/training/run_taiji_n1_hard_neighbor_training.py",
    ROOT / "scripts/training/run_taiji_n1_length_balanced_training.py",
    ROOT / "scripts/training/_taiji_n1_evidence_shard.py",
)

PROTOCOL = {
    "schema": "n1-seed13-update-group-trajectory-attribution-v1",
    "seed": SEED,
    "arm": ARM,
    "updates_per_endpoint": UPDATES,
    "milestones": list(MILESTONES),
    "schedule_views": ["raw", "hard"],
    "task_kinds": list(KINDS),
    "update_groups": list(GROUPS),
    "source_shards": 2,
    "model_loads": 0,
    "checkpoint_payload_reads": 0,
    "model_constructions": 0,
    "model_queries": 0,
    "parameter_updates": 0,
    "optimizer_steps": 0,
    "checkpoint_saves": 0,
    "dev_queries": 0,
    "holdout_queries": 0,
    "confirmation_queries": 0,
    "cpu_limit_seconds": CPU_LIMIT_SECONDS,
    "retry_allowed": False,
    "can_promote": False,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


class CpuBudget:
    def __init__(self):
        self.started = time.process_time()

    def used(self) -> float:
        return time.process_time() - self.started

    def guard(self, reserve: float = 0.0) -> None:
        if self.used() + reserve >= CPU_LIMIT_SECONDS:
            raise TimeoutError("PLAN-N1-114 CPU limit reached")


def _endpoint(report: dict, *, training_sufficient: bool) -> dict:
    if (
        report.get("complete") is not True
        or report.get("status") != "complete"
        or report.get("can_promote") is not False
        or report.get("training_sufficient") is not training_sufficient
        or report.get("dev_queries") != 0
        or report.get("holdout_queries") != 0
        or report.get("confirmation_queries") != 0
    ):
        raise ValueError("complete train-only endpoint report required")
    found = [
        model
        for model in report.get("models", [])
        if (model.get("seed"), model.get("arm")) == (SEED, ARM)
    ]
    if len(found) != 1 or found[0].get("completed_updates") != UPDATES:
        raise ValueError("one complete seed13/position endpoint required")
    endpoint = found[0]
    shard_identity = endpoint.get("shard", {}).get("identity", {})
    if (
        endpoint.get("complete") is not True
        or endpoint.get("training_sufficient") is not training_sufficient
        or len(endpoint.get("restore_checks", [])) != 2
        or any(check.get("equal") is not True for check in endpoint["restore_checks"])
        or (shard_identity.get("seed"), shard_identity.get("arm"), shard_identity.get("phase"))
        != (SEED, ARM, "complete")
    ):
        raise ValueError("both frozen endpoint recovery checks must have passed")
    return endpoint


def _material_length(item: dict) -> int | None:
    if item.get("kind") == "transition":
        return None
    lengths = {
        len(line.removesuffix(".").split()[-1])
        for line in item.get("observations", [])
        if " uses " in line
    }
    if not lengths or len(lengths) != 1:
        raise ValueError("each material training item must have one uniform uses-value length")
    return next(iter(lengths))


def _label(path: str, item: dict, length: int | None, condition: str, cycle: int) -> dict:
    if item.get("kind") not in KINDS or length not in (None, 6, 7):
        raise ValueError("unsupported frozen task kind or material length")
    if (item["kind"] == "transition") != (length is None):
        raise ValueError("transition must be length-neutral; material tasks need length6/7")
    return {
        "path": path,
        "kind": item["kind"],
        "length": length,
        "condition": condition,
        "cycle": cycle,
    }


def _transition_input(item: dict) -> tuple:
    return (
        item.get("kind"),
        tuple(item.get("observations", [])),
        item.get("question"),
        item.get("answer"),
    )


def build_step_labels(
    items: list[dict],
    baseline_schedule: dict,
    candidate_schedule: dict,
    *,
    second_question_fn=None,
) -> dict:
    if second_question_fn is None:
        from scripts.training import build_taiji_n1_hard_neighbor_training_schedule as base_builder

        second_question_fn = base_builder.core.renaming.second_question

    raw_candidate = {
        (row["parent_id"], row["raw_slot"]): row for row in candidate_schedule["raw_views"]
    }
    hard_by_run = {
        "baseline": {row["schedule_index"]: row for row in baseline_schedule["items"]},
        "candidate": {row["schedule_index"]: row for row in candidate_schedule["items"]},
    }
    if (
        len(items) != 192
        or len(baseline_schedule["items"]) != 3_072
        or len(candidate_schedule["items"]) != 3_072
        or len(candidate_schedule["raw_views"]) != 3_072
        or len(raw_candidate) != 3_072
        or any(len(hard_by_run[run]) != 3_072 for run in RUNS)
    ):
        raise ValueError("frozen 192-parent and 3072-per-view schedule coverage required")
    if baseline_schedule.get("source_manifest_sha256") != candidate_schedule.get(
        "source_manifest_sha256"
    ):
        raise ValueError("baseline and candidate schedules use different train manifests")

    labels = {run: [] for run in RUNS}
    transition_pairs = 0
    for step in range(UPDATES):
        local = step % 6_144
        cycle = step // 6_144
        parent_index = (local // 2) % 192
        parent = items[parent_index]
        if local % 2 == 0:
            raw_slot = local // 384
            candidate_row = raw_candidate[(parent["id"], raw_slot)]
            if candidate_row["parent_kind"] != parent["kind"]:
                raise ValueError("candidate raw-view task kind differs from frozen parent")
            selected = {
                "baseline": (parent, _material_length(parent), "raw"),
                "candidate": (
                    candidate_row["item"],
                    candidate_row["material_length_chars"],
                    "raw",
                ),
            }
            condition = "raw"
            if parent["kind"] == "transition":
                if _transition_input(parent) != _transition_input(candidate_row["item"]):
                    raise ValueError("raw transition training input changed under length balancing")
                expected_second = second_question_fn(parent)
                if candidate_row.get("second_question") != expected_second:
                    raise ValueError(
                        "raw transition second question changed under length balancing"
                    )
                transition_pairs += 1
        else:
            schedule_index = local // 2
            baseline_row = hard_by_run["baseline"][schedule_index]
            candidate_row = hard_by_run["candidate"][schedule_index]
            compare_fields = (
                "schedule_index",
                "parent_id",
                "parent_kind",
                "condition",
                "observation_count",
                "hamming_distance",
                "target_positions",
            )
            if any(baseline_row.get(field) != candidate_row.get(field) for field in compare_fields):
                raise ValueError("hard-view task identity/condition differs across schedules")
            if (
                baseline_row.get("schedule_index") != schedule_index
                or baseline_row.get("parent_id") != parent["id"]
            ):
                raise ValueError("hard-view schedule index/parent does not match update step")
            selected = {
                "baseline": (
                    baseline_row["item"],
                    _material_length(baseline_row["item"]),
                    "hard",
                ),
                "candidate": (
                    candidate_row["item"],
                    candidate_row["material_length_chars"],
                    "hard",
                ),
            }
            condition = str(baseline_row["condition"])
            if parent["kind"] == "transition":
                if _transition_input(baseline_row["item"]) != _transition_input(
                    candidate_row["item"]
                ):
                    raise ValueError(
                        "hard transition training input changed under length balancing"
                    )
                if baseline_row.get("second_question") != candidate_row.get("second_question"):
                    raise ValueError(
                        "hard transition second question changed under length balancing"
                    )
                transition_pairs += 1
        for run, (item, length, path) in selected.items():
            if item.get("kind") != parent["kind"]:
                raise ValueError("scheduled task kind differs from frozen parent")
            labels[run].append(_label(path, item, length, condition, cycle))
    if transition_pairs == 0 or any(len(labels[run]) != UPDATES for run in RUNS):
        raise ValueError("complete transition input parity and update schedule required")
    return {"labels": labels, "transition_updates_identical": transition_pairs}


def _bucket_key(label: dict) -> tuple:
    return (label["path"], label["kind"], label["length"], label["condition"])


def summarize_trajectory(record: dict, labels: list[dict], budget: CpuBudget) -> dict:
    if len(record.get("losses", [])) != UPDATES or len(record.get("measurements", [])) != UPDATES:
        raise ValueError("all 24,576 loss/measurement updates required")
    aggregate = defaultdict(
        lambda: {"losses": [], "measurements": defaultdict(list), "cycles": defaultdict(list)}
    )
    by_cycle = defaultdict(lambda: {"losses": [], "measurements": defaultdict(list)})
    for step, (label, loss, measurement) in enumerate(
        zip(labels, record["losses"], record["measurements"], strict=True)
    ):
        if measurement.get("step") != step or measurement.get("cycle") != step // 6_144:
            raise ValueError("training measurements are not in frozen update order")
        if not all(math.isfinite(float(loss[name])) for name in ("qa", "observation", "total")):
            raise ValueError("training loss log contains a nonfinite value")
        key = _bucket_key(label)
        summary = aggregate[key]
        summary["losses"].append(loss)
        summary["cycles"][label["cycle"]].append(loss)
        cycle_summary = by_cycle[(label["cycle"], *key)]
        cycle_summary["losses"].append(loss)
        if set(measurement.get("groups", {})) != set(GROUPS):
            raise ValueError("all five updater groups required per measurement")
        for group in GROUPS:
            values = measurement["groups"][group]
            for metric in ("actual_l2", "gradient_l2", "proposed_l2", "cap", "scale"):
                value = float(values[metric])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("update-group log contains invalid magnitude")
                summary["measurements"][group].append({**values, "metric": metric, "value": value})
                cycle_summary["measurements"][group].append(
                    {**values, "metric": metric, "value": value}
                )
        if step % 2_048 == 0:
            budget.guard(reserve=1)

    def reduce(buckets: dict) -> dict:
        result = {}
        for key, bucket in buckets.items():
            name = "/".join("none" if part is None else str(part) for part in key)
            loss_rows = bucket["losses"]
            group_stats = {}
            for group in GROUPS:
                rows = bucket["measurements"][group]
                by_metric = {
                    metric: []
                    for metric in ("actual_l2", "gradient_l2", "proposed_l2", "cap", "scale")
                }
                clipped = 0
                for row in rows:
                    by_metric[row["metric"]].append(row["value"])
                    clipped = max(clipped, int(row["clipped"]))
                group_stats[group] = {
                    f"mean_{metric}": statistics.fmean(values) if values else None
                    for metric, values in by_metric.items()
                }
                group_stats[group]["clipped_updates"] = sum(
                    int(row["clipped"]) for row in rows if row["metric"] == "actual_l2"
                )
            result[name] = {
                "updates": len(loss_rows),
                "qa_sequences": sum(int(row["qa_sequences"]) for row in loss_rows),
                "mean_loss": {
                    field: statistics.fmean(float(row[field]) for row in loss_rows)
                    for field in ("qa", "observation", "total")
                },
                "group_updates": group_stats,
            }
        return result

    return {"all_cycles": reduce(aggregate), "by_cycle": reduce(by_cycle)}


def matched_bucket_differences(baseline: dict, candidate: dict) -> dict:
    differences = {}
    for bucket in sorted(set(baseline["all_cycles"]) & set(candidate["all_cycles"])):
        base, cand = baseline["all_cycles"][bucket], candidate["all_cycles"][bucket]
        differences[bucket] = {
            "baseline_updates": base["updates"],
            "candidate_updates": cand["updates"],
            "mean_loss_delta_candidate_minus_baseline": {
                field: cand["mean_loss"][field] - base["mean_loss"][field]
                for field in ("qa", "observation", "total")
            },
            "group_mean_actual_l2_delta_candidate_minus_baseline": {
                group: cand["group_updates"][group]["mean_actual_l2"]
                - base["group_updates"][group]["mean_actual_l2"]
                for group in GROUPS
            },
            "group_clipped_update_delta_candidate_minus_baseline": {
                group: cand["group_updates"][group]["clipped_updates"]
                - base["group_updates"][group]["clipped_updates"]
                for group in GROUPS
            },
        }
    return differences


def analyze() -> dict:
    if REPORT.exists() or ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-114 is write-once; report or attempt already exists")
    for path, expected in EXPECTED.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen PLAN-N1-114 input hash mismatch: {path}")
    budget = CpuBudget()
    guards = {
        str(path.resolve()): sha256_file(path)
        for path in (*EXPECTED, *CODE_PATHS, Path(__file__).resolve())
    }
    attempt = {
        "schema": "n1-seed13-update-group-trajectory-attempt-v1",
        "status": "started",
        "immutable_start_record": True,
        "protocol": PROTOCOL,
        "source_guards": guards,
        "model_queries": 0,
        "parameter_updates": 0,
    }
    result = {
        "schema": PROTOCOL["schema"],
        "protocol": PROTOCOL,
        "status": "running",
        "source_guards": guards,
        "rows_read": 0,
        "model_loads": 0,
        "checkpoint_payload_reads": 0,
        "model_constructions": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "checkpoint_saves": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
    }
    try:
        from scripts.training import _taiji_n1_evidence_shard as shard
        from scripts.training import build_taiji_n1_hard_neighbor_training_schedule as base_builder

        items, manifest_sha = base_builder.train_items_from_frozen_manifest()
        if manifest_sha != base_builder.MANIFEST_SHA256:
            raise ValueError("frozen train manifest digest differs")
        baseline_schedule = json.loads(BASELINE_SCHEDULE.read_text(encoding="utf-8"))
        candidate_schedule = json.loads(CANDIDATE_SCHEDULE.read_text(encoding="utf-8"))
        if any(
            schedule.get("source_manifest_sha256") != manifest_sha
            for schedule in (baseline_schedule, candidate_schedule)
        ):
            raise ValueError("frozen schedules do not bind the same train manifest")
        if (
            baseline_schedule.get("source_manifest_sha256") != manifest_sha
            or candidate_schedule.get("source_manifest_sha256") != manifest_sha
        ):
            raise ValueError("baseline/candidate schedules must bind the frozen train manifest")
        if (
            len(baseline_schedule.get("items", [])) != 3_072
            or len(candidate_schedule.get("items", [])) != 3_072
            or len(candidate_schedule.get("raw_views", [])) != 3_072
        ):
            raise ValueError("candidate raw/hard schedule must each have3072 updates per cycle")
        labels = build_step_labels(items, baseline_schedule, candidate_schedule)
        reports = {
            "baseline": json.loads(BASELINE_REPORT.read_text(encoding="utf-8")),
            "candidate": json.loads(CANDIDATE_REPORT.read_text(encoding="utf-8")),
        }
        if any(
            reports[run].get("schedule_sha256") != EXPECTED[schedule_path]
            for run, schedule_path in (
                ("baseline", BASELINE_SCHEDULE),
                ("candidate", CANDIDATE_SCHEDULE),
            )
        ):
            raise ValueError("training reports and schedule artifact hashes differ")
        if reports["candidate"].get("length_balance_valid") is not True:
            raise ValueError("candidate length-balance ledger must be complete and valid")
        endpoint_models = {
            "baseline": _endpoint(reports["baseline"], training_sufficient=True),
            "candidate": _endpoint(reports["candidate"], training_sufficient=False),
        }
        shard_paths = {}
        for run in RUNS:
            seal = endpoint_models[run]["shard"]
            shard_path = Path(seal["path"]).resolve()
            if not shard_path.is_relative_to(ROOT.resolve()):
                raise ValueError("training shard must resolve inside the repository")
            guards[str(shard_path)] = sha256_file(shard_path)
            shard_paths[run] = shard_path
        result["source_guards"] = guards
        attempt["source_guards"] = guards
        write_new(ATTEMPT, attempt)
        records = {}
        for run in RUNS:
            model = endpoint_models[run]
            seal = model["shard"]
            identity = seal["identity"]
            record = shard.read_checked(shard_paths[run], seal, identity=identity)
            records[run] = record
            result["rows_read"] += len(record["losses"]) + len(record["measurements"])
        summaries = {
            run: summarize_trajectory(records[run], labels["labels"][run], budget) for run in RUNS
        }
        result.update(
            status="complete",
            step_labels={run: len(labels["labels"][run]) for run in RUNS},
            transition_updates_identical=labels["transition_updates_identical"],
            summaries=summaries,
            matched_bucket_differences=matched_bucket_differences(
                summaries["baseline"], summaries["candidate"]
            ),
            schedule_hashes={
                "baseline": EXPECTED[BASELINE_SCHEDULE],
                "candidate": EXPECTED[CANDIDATE_SCHEDULE],
            },
            training_report_hashes={
                "baseline": EXPECTED[BASELINE_REPORT],
                "candidate": EXPECTED[CANDIDATE_REPORT],
            },
        )
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    result["cpu_seconds"] = budget.used()
    result["sources_unchanged"] = all(
        Path(path).is_file() and sha256_file(Path(path)) == digest
        for path, digest in guards.items()
    )
    result["complete"] = (
        result["status"] == "complete"
        and result.get("step_labels") == {"baseline": UPDATES, "candidate": UPDATES}
        and result["transition_updates_identical"] > 0
        and result["rows_read"] == 2 * UPDATES * 2
        and result["model_loads"] == 0
        and result["checkpoint_payload_reads"] == 0
        and result["model_constructions"] == 0
        and result["model_queries"] == 0
        and result["parameter_updates"] == 0
        and result["optimizer_steps"] == 0
        and result["checkpoint_saves"] == 0
        and all(
            result[key] == 0 for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
        and result["sources_unchanged"]
        and result["cpu_seconds"] <= CPU_LIMIT_SECONDS
    )
    write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    if not parser.parse_args().execute:
        raise SystemExit("use --execute for the authorized one-shot offline attribution")
    result = analyze()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "cpu_seconds",
                    "rows_read",
                    "transition_updates_identical",
                    "error",
                )
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["complete"] else 1)
