"""PLAN-N1-113: read-only parameter-group delta audit for two frozen endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASELINE_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_training_20261017.json"
CANDIDATE_REPORT = ROOT / "reports/taiji_n1_length_balanced_training_20261018.json"
BASELINE_CHECKPOINT = (
    ROOT / "data/checkpoints/n1_hard_neighbor_training_v1_20261017/s13_position_final.pt"
)
CANDIDATE_CHECKPOINT = (
    ROOT
    / "data/checkpoints/n1_hard_neighbor_length_balanced_training_20261018/s13_position_final.pt"
)
PLAN112_REPORT = ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_r1_20261018.json"
REPORT = ROOT / "reports/taiji_n1_position_weight_checkpoint_parameter_delta_s13_20261018.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")

EXPECTED = {
    BASELINE_REPORT: "dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909",
    CANDIDATE_REPORT: "e848433de597d378e9e1dc52bf488f296f90c5ce67f3855e19bcfe49268aaa0c",
    BASELINE_CHECKPOINT: "09df1d47eb490ef9afc5d563d08b40eca6b51951096875d016a67f133f7ed653",
    CANDIDATE_CHECKPOINT: "d215a51d571976f216af43d0b7372f3cb702b6258237a7144252df4609f3b881",
    PLAN112_REPORT: "e38b08a857ffe69acb8fbf57e040ed672df538b88988dc333e1182e58ffb16bf",
}
CPU_LIMIT_SECONDS = 60.0
SEED = 13
ARM = "position"
UPDATES = 24_576
PARAMETERS = 62_056
TENSOR_COUNT = 41
GROUPS = ("reader", "writer", "local", "context", "relation")
CODE_PATHS = (
    ROOT / "scripts/training/_taiji_n1_group_relative_adam.py",
    ROOT / "scripts/training/run_taiji_n1_relation_paired.py",
    ROOT / "scripts/training/run_taiji_n1_local_relation.py",
    ROOT / "scripts/training/run_taiji_n1_contextual_question.py",
    ROOT / "scripts/training/run_taiji_n1_local_question.py",
)

PROTOCOL = {
    "schema": "n1-position-checkpoint-parameter-delta-attribution-v1",
    "seed": SEED,
    "arm": ARM,
    "updates": UPDATES,
    "parameters": PARAMETERS,
    "parameter_tensors": TENSOR_COUNT,
    "groups": list(GROUPS),
    "checkpoint_payload_reads": 2,
    "model_loads": 0,
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
            raise TimeoutError("PLAN-N1-113 CPU limit reached")


def verify_reports() -> tuple[dict, dict, dict]:
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE_REPORT.read_text(encoding="utf-8"))
    crossover = json.loads(PLAN112_REPORT.read_text(encoding="utf-8"))
    for report in (baseline, candidate):
        if (
            report.get("complete") is not True
            or report.get("status") != "complete"
            or report.get("dev_queries") != 0
            or report.get("holdout_queries") != 0
            or report.get("confirmation_queries") != 0
        ):
            raise ValueError("complete train-only endpoint reports required")
    if (
        crossover.get("complete") is not True
        or crossover.get("interpretation") != "evidence_toward_candidate_network_weights"
        or crossover.get("model_queries") != 1_536
        or crossover.get("parameter_updates") != 0
    ):
        raise ValueError("complete matching PLAN-N1-112 crossover report required")
    for report, expected_sufficient, expected_joint, expected_checkpoint_hash in (
        (baseline, True, (48, 48), EXPECTED[BASELINE_CHECKPOINT]),
        (candidate, False, (5, 4), EXPECTED[CANDIDATE_CHECKPOINT]),
    ):
        records = [
            item for item in report["models"] if (item.get("seed"), item.get("arm")) == (SEED, ARM)
        ]
        if len(records) != 1:
            raise ValueError("one sealed seed13/position report endpoint required")
        model = records[0]
        if (
            model.get("completed_updates") != UPDATES
            or model.get("complete") is not True
            or model.get("training_sufficient") is not expected_sufficient
            or len(model.get("restore_checks", [])) != 2
            or any(check.get("equal") is not True for check in model["restore_checks"])
        ):
            raise ValueError("complete independently restored matching endpoint required")
        final_checks = [
            check
            for check in model["restore_checks"]
            if Path(check["path"]).name == "s13_position_final.pt"
        ]
        if len(final_checks) != 1 or final_checks[0].get("sha256") != expected_checkpoint_hash:
            raise ValueError("final endpoint hash does not match the frozen parameter source")
        actual_joint = tuple(
            model["milestones"][-1]["factorial"][view]["joint_four_exact"]
            for view in ("original", "rename")
        )
        if actual_joint != expected_joint:
            raise ValueError("endpoint training gates differ from the PLAN-N1-112 comparison")
    return baseline, candidate, crossover


def parameter_groups(names: set[str]) -> dict[str, list[str]]:
    from scripts.training import _taiji_n1_group_relative_adam as bounded

    groups = {name: [] for name in GROUPS}
    for name in sorted(names):
        group = (
            "writer" if name.startswith("writer.") else bounded.runner.core.parameter_group(name)
        )
        if group not in groups:
            raise ValueError(f"parameter maps to unknown updater group: {name} -> {group}")
        groups[group].append(name)
    if (
        sum(map(len, groups.values())) != TENSOR_COUNT
        or not all(groups.values())
        or sum(len(names) for names in groups.values()) != len(names)
    ):
        raise ValueError("frozen five-group updater map must cover all41 tensors")
    return groups


def compute_group_deltas(
    baseline_state: dict,
    candidate_state: dict,
    groups: dict,
    torch,
    *,
    expected_parameters: int = PARAMETERS,
    expected_tensor_count: int = TENSOR_COUNT,
) -> dict:
    if set(baseline_state) != set(candidate_state):
        raise ValueError("endpoint state keys differ")
    names = set(baseline_state)
    if {name for values in groups.values() for name in values} != names:
        raise ValueError("group map must cover every endpoint tensor exactly once")
    stats = {}
    tensor_rows = []
    all_delta_sq = 0.0
    all_baseline_sq = 0.0
    all_candidate_sq = 0.0
    all_dot = 0.0
    all_changed = 0
    for group, group_names in groups.items():
        baseline_sq = candidate_sq = delta_sq = dot = 0.0
        changed = changed_tensors = 0
        for name in group_names:
            base = baseline_state[name]
            candidate = candidate_state[name]
            if base.shape != candidate.shape or base.dtype != candidate.dtype:
                raise ValueError(f"endpoint tensor shape/dtype differs: {name}")
            b = base.detach().cpu().to(dtype=torch.float64)
            c = candidate.detach().cpu().to(dtype=torch.float64)
            delta = c - b
            base_norm_sq = float(torch.sum(b * b))
            candidate_norm_sq = float(torch.sum(c * c))
            delta_norm_sq = float(torch.sum(delta * delta))
            dot_value = float(torch.sum(b * c))
            changed_values = int(torch.count_nonzero(delta))
            baseline_sq += base_norm_sq
            candidate_sq += candidate_norm_sq
            delta_sq += delta_norm_sq
            dot += dot_value
            changed += changed_values
            changed_tensors += int(changed_values > 0)
            tensor_rows.append(
                {
                    "group": group,
                    "name": name,
                    "shape": list(base.shape),
                    "l2_delta": math.sqrt(delta_norm_sq),
                    "relative_l2_delta": math.sqrt(delta_norm_sq)
                    / max(math.sqrt(base_norm_sq), 1e-12),
                    "max_abs_delta": float(torch.max(torch.abs(delta))) if delta.numel() else 0.0,
                    "changed_scalars": changed_values,
                }
            )
        base_norm = math.sqrt(baseline_sq)
        candidate_norm = math.sqrt(candidate_sq)
        delta_norm = math.sqrt(delta_sq)
        stats[group] = {
            "tensors": len(group_names),
            "scalars": sum(baseline_state[name].numel() for name in group_names),
            "baseline_l2": base_norm,
            "candidate_l2": candidate_norm,
            "delta_l2": delta_norm,
            "relative_delta_to_baseline_l2": delta_norm / max(base_norm, 1e-12),
            "cosine_similarity": (
                dot / (base_norm * candidate_norm) if base_norm and candidate_norm else None
            ),
            "changed_tensors": changed_tensors,
            "changed_scalars": changed,
        }
        all_delta_sq += delta_sq
        all_baseline_sq += baseline_sq
        all_candidate_sq += candidate_sq
        all_dot += dot
        all_changed += changed
    tensor_rows.sort(key=lambda row: (row["l2_delta"], row["name"]), reverse=True)
    aggregate = {
        "tensors": sum(value["tensors"] for value in stats.values()),
        "scalars": sum(value["scalars"] for value in stats.values()),
        "baseline_l2": math.sqrt(all_baseline_sq),
        "candidate_l2": math.sqrt(all_candidate_sq),
        "delta_l2": math.sqrt(all_delta_sq),
        "cosine_similarity": (
            all_dot / math.sqrt(all_baseline_sq * all_candidate_sq)
            if all_baseline_sq and all_candidate_sq
            else None
        ),
        "changed_scalars": all_changed,
    }
    if aggregate["scalars"] != expected_parameters or aggregate["tensors"] != expected_tensor_count:
        raise ValueError("aggregate parameter counts differ from frozen model contract")
    if not math.isclose(
        aggregate["delta_l2"] ** 2,
        sum(value["delta_l2"] ** 2 for value in stats.values()),
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError("five-group squared delta norms do not reconcile")
    return {"groups": stats, "aggregate": aggregate, "top_tensors": tensor_rows[:15]}


def analyze() -> dict:
    if REPORT.exists() or ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-113 is write-once; report or attempt already exists")
    for path, expected in EXPECTED.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    baseline_report, candidate_report, crossover_report = verify_reports()
    torch = __import__("torch")
    torch.set_num_threads(1)
    budget = CpuBudget()
    guarded_paths = (*EXPECTED, *CODE_PATHS, Path(__file__).resolve())
    guards = {str(path.resolve()): sha256_file(path) for path in guarded_paths}
    attempt = {
        "schema": "n1-position-checkpoint-parameter-delta-attempt-v1",
        "status": "running",
        "protocol": PROTOCOL,
        "source_guards": guards,
        "model_queries": 0,
        "parameter_updates": 0,
    }
    write_new(ATTEMPT, attempt)
    result = {
        "schema": PROTOCOL["schema"],
        "protocol": PROTOCOL,
        "status": "running",
        "source_guards": guards,
        "checkpoint_payload_reads": 0,
        "model_loads": 0,
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
        endpoints = {}
        for name, path in (("baseline", BASELINE_CHECKPOINT), ("candidate", CANDIDATE_CHECKPOINT)):
            budget.guard(reserve=10)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            result["checkpoint_payload_reads"] += 1
            report = baseline_report if name == "baseline" else candidate_report
            model_record = next(
                item
                for item in report["models"]
                if (item.get("seed"), item.get("arm")) == (SEED, ARM)
            )
            final_check = next(
                check
                for check in model_record["restore_checks"]
                if Path(check["path"]).name == "s13_position_final.pt"
            )
            if (
                payload.get("seed") != SEED
                or payload.get("arm") != ARM
                or payload.get("completed_updates") != UPDATES
                or payload.get("phase") != "final"
                or payload.get("protocol") != model_record["shard"]["identity"]["protocol"]
                or not final_check.get("equal")
                or sha256_file(path) != final_check.get("sha256")
            ):
                raise ValueError(f"checkpoint does not match its sealed {name} endpoint")
            endpoints[name] = payload["candidate"]["state"]

        baseline_state, candidate_state = endpoints["baseline"], endpoints["candidate"]
        if (
            len(baseline_state) != TENSOR_COUNT
            or sum(t.numel() for t in baseline_state.values()) != PARAMETERS
        ):
            raise ValueError("checkpoint parameter topology differs from frozen model contract")
        updater = __import__("scripts.training._taiji_n1_group_relative_adam", fromlist=["POLICY"])
        groups = {key: [] for key in GROUPS}
        for name in baseline_state:
            group = (
                "writer"
                if name.startswith("writer.")
                else updater.runner.core.parameter_group(name)
            )
            if group not in groups:
                raise ValueError(f"parameter group is outside the frozen five-group policy: {name}")
            groups[group].append(name)
        if (
            set(groups) != set(updater.POLICY["groups"])
            or not all(groups.values())
            or sum(len(names) for names in groups.values()) != TENSOR_COUNT
        ):
            raise ValueError("parameter grouping differs from the frozen updater policy")
        result["parameter_deltas"] = compute_group_deltas(
            baseline_state, candidate_state, groups, torch
        )
        result["endpoint_hashes_before"] = {
            "baseline": sha256_file(BASELINE_CHECKPOINT),
            "candidate": sha256_file(CANDIDATE_CHECKPOINT),
        }
        result["endpoint_hashes_after"] = {
            "baseline": sha256_file(BASELINE_CHECKPOINT),
            "candidate": sha256_file(CANDIDATE_CHECKPOINT),
        }
        result["training_reports"] = {
            "baseline": {"path": str(BASELINE_REPORT), "sha256": EXPECTED[BASELINE_REPORT]},
            "candidate": {"path": str(CANDIDATE_REPORT), "sha256": EXPECTED[CANDIDATE_REPORT]},
        }
        result["plan112_report"] = {
            "path": str(PLAN112_REPORT),
            "sha256": EXPECTED[PLAN112_REPORT],
            "interpretation": crossover_report["interpretation"],
        }
        result["group_mapping_code_hashes"] = {
            str(path.relative_to(ROOT)): sha256_file(path) for path in CODE_PATHS
        }
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    result["cpu_seconds"] = budget.used()
    result["sources_unchanged"] = all(
        Path(path).is_file() and sha256_file(Path(path)) == digest
        for path, digest in guards.items()
    )
    result["complete"] = (
        result["status"] == "complete"
        and result["checkpoint_payload_reads"] == 2
        and result["model_loads"] == 0
        and result["model_constructions"] == 0
        and result["model_queries"] == 0
        and result["parameter_updates"] == 0
        and result["optimizer_steps"] == 0
        and result["checkpoint_saves"] == 0
        and result["endpoint_hashes_before"] == result["endpoint_hashes_after"]
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
                    "checkpoint_payload_reads",
                    "model_queries",
                    "error",
                )
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["complete"] else 1)
