"""One-shot PLAN-N1-129 audit of native question-to-observation alignment mass."""

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

import torch  # noqa: E402

from scripts.training import eval_taiji_n1_v11_after_n120 as e11  # noqa: E402
from scripts.training import (
    eval_taiji_n1_v13_v14_full_context_question_binding_mask as n128,
)  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-129_native_question_binding_mass_audit_draft_20261019.md"
PLAN_SHA256 = "7e35f01322ea6ecf88f1f53f8124d1b21378d3c8ae7f085c0c5ca52e2cce12ae"
N128_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-128_execution_closeout_20261019.md"
N128_READINESS = (
    ROOT / "reports/taiji_n1_v13_v14_full_context_question_binding_mask_readiness_20261019.json"
)
N128_ATTEMPT = (
    ROOT / "reports/taiji_n1_v13_v14_full_context_question_binding_mask_attempt_20261019.json"
)
N128_RESULT = (
    ROOT / "reports/taiji_n1_v13_v14_full_context_question_binding_mask_result_20261019.json"
)
N128_CLOSEOUT_SHA256 = "b8c42f2795a1e7e5a7cdf9393138dfdfcd0fd1709b6d907e7dd9f20fe2967df4"
N128_READINESS_SHA256 = "0141c12b7a45a2e48b54b6965b86a97dcbdc5b246a4a7f129ebfd590b5fac90c"
N128_ATTEMPT_SHA256 = "761e1a6a9b583d795aec8b2548671100d69a89475db42d23aa2a743a1cdfa8d4"
N128_RESULT_SHA256 = "a9278debaf3c2c57f4a6678a692c8b4afa6e9a5e867745d3d7c5ba5526d6b28e"
READINESS = ROOT / "reports/taiji_n1_v13_v14_native_question_binding_mass_readiness_20261019.json"
ATTEMPT = ROOT / "reports/taiji_n1_v13_v14_native_question_binding_mass_attempt_20261019.json"
RESULT = ROOT / "reports/taiji_n1_v13_v14_native_question_binding_mass_result_20261019.json"
CPU_LIMIT_SECONDS = 600
SEEDS = n128.SEEDS
ARMS = n128.ARMS
KINDS = n128.KINDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _mass_metrics(alignment: torch.Tensor, salience: torch.Tensor, allowed: torch.Tensor) -> dict:
    if (
        alignment.ndim != 2
        or alignment.shape[-1] != allowed.numel()
        or salience.ndim != 1
        or salience.numel() != alignment.shape[0]
        or not bool(allowed.any())
    ):
        raise ValueError("native local alignment and target-byte mask shapes differ")
    if not torch.isfinite(alignment).all() or not torch.isfinite(salience).all():
        raise ValueError("native question-binding distribution contains non-finite values")
    row_sums = alignment.sum(-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-6, atol=1e-6):
        raise ValueError("native question alignment rows are not normalized")
    if not torch.allclose(salience.sum(), salience.new_tensor(1.0), rtol=1e-6, atol=1e-6):
        raise ValueError("native question salience is not normalized")

    target_mass = alignment[:, allowed].sum(-1)
    target_share = float(allowed.float().mean().item())
    argmax_target = allowed[alignment.argmax(-1)].float()
    weighted_mass = (target_mass * salience).sum()
    return {
        "question_byte_count": int(alignment.shape[0]),
        "tape_byte_count": int(alignment.shape[1]),
        "target_byte_count": int(allowed.sum().item()),
        "target_byte_share": target_share,
        "target_alignment_mass_mean": float(target_mass.mean().item()),
        "target_alignment_mass_min": float(target_mass.min().item()),
        "target_alignment_mass_max": float(target_mass.max().item()),
        "target_alignment_mass_by_question_byte": [float(value) for value in target_mass.tolist()],
        "target_alignment_mass_enrichment": float(target_mass.mean().item()) / target_share,
        "salience_weighted_target_alignment_mass": float(weighted_mass.item()),
        "question_byte_argmax_on_target_fraction": float(argmax_target.mean().item()),
        "native_alignment_row_sum_min": float(row_sums.min().item()),
        "native_alignment_row_sum_max": float(row_sums.max().item()),
        "question_salience_sum": float(salience.sum().item()),
    }


def _synthetic_preflight() -> dict:
    alignment = torch.tensor(
        [[0.1, 0.2, 0.5, 0.1, 0.1], [0.3, 0.1, 0.1, 0.4, 0.1]],
        dtype=torch.float64,
    )
    salience = torch.tensor([0.5, 0.5], dtype=torch.float64)
    allowed = torch.tensor([True, True, False, False, False])
    metrics = _mass_metrics(alignment, salience, allowed)
    if not math.isclose(metrics["target_alignment_mass_mean"], 0.35, abs_tol=1e-12):
        raise AssertionError("synthetic target alignment mass is miscomputed")
    if not math.isclose(metrics["salience_weighted_target_alignment_mass"], 0.35, abs_tol=1e-12):
        raise AssertionError("synthetic salience-weighted target mass is miscomputed")
    return {
        "complete": True,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_binding_evaluations": 0,
        "generated_answer_queries": 0,
        "parameter_updates": 0,
        "synthetic_target_mass_mean": metrics["target_alignment_mass_mean"],
        "synthetic_salience_weighted_mass": metrics["salience_weighted_target_alignment_mass"],
        "normalization_assertions": True,
    }


def _source_set() -> dict:
    for path, expected in (
        (PLAN, PLAN_SHA256),
        (N128_CLOSEOUT, N128_CLOSEOUT_SHA256),
        (N128_READINESS, N128_READINESS_SHA256),
        (N128_ATTEMPT, N128_ATTEMPT_SHA256),
        (N128_RESULT, N128_RESULT_SHA256),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen N1-128/N1-129 source changed: {path}")
    n128_result = read_json(N128_RESULT)
    if (
        n128_result.get("complete") is not True
        or n128_result.get("model_queries") != 180
        or n128_result.get("parameter_updates") != 0
        or n128_result.get("source_hashes_unchanged") is not True
        or n128_result.get("can_promote") is not False
        or len(n128_result.get("rows", [])) != 180
    ):
        raise ValueError("complete N1-128 diagnostic receipts are required")
    bound = n128._source_set()
    prior_rows = {}
    for row in n128_result["rows"]:
        key = (row["face"], row["seed"], row["arm"], row["id"])
        if key in prior_rows:
            raise ValueError(f"duplicate N1-128 saved result row: {key}")
        prior_rows[key] = row
    for face_label, (_, _, items) in bound["face_data"].items():
        expected = {
            (face_label, seed, arm, item["id"]) for seed in SEEDS for arm in ARMS for item in items
        }
        if not expected.issubset(prior_rows):
            raise ValueError(f"N1-128 question-mask rows are incomplete for {face_label}")

    sources = dict(bound["source_hashes"])
    for path in (
        PLAN,
        N128_CLOSEOUT,
        N128_READINESS,
        N128_ATTEMPT,
        N128_RESULT,
        Path(__file__).resolve(),
    ):
        sources[str(Path(path).resolve())] = sha256(Path(path))
    return {**bound, "n128_result": n128_result, "prior_rows": prior_rows, "source_hashes": sources}


def source_readiness() -> dict:
    if READINESS.exists() or ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("PLAN-N1-129 outputs are write-once")
    started = time.process_time()
    preflight = _synthetic_preflight()
    bound = _source_set()
    for path, expected in {**bound["source_hashes"], **bound["checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed during N1-129 readiness: {path}")
    cpu = time.process_time() - started
    complete = cpu <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan129-native-question-binding-mass-readiness-v1",
        "status": "ready_for_execution" if complete else "stopped_cpu_limit",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "source_hashes": bound["source_hashes"],
        "checkpoint_hashes": bound["checkpoint_hashes"],
        "checkpoint_hash_count": len(bound["checkpoint_hashes"]),
        "face_sha256": {
            label: sha256(n128.n127.FACE_SOURCES[label]["face"]) for label in bound["face_data"]
        },
        "n128_result_sha256": sha256(N128_RESULT),
        "planned_binding_evaluations": 180,
        "planned_generated_answer_queries": 0,
        "instrument_preflight": preflight,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_binding_evaluations": 0,
        "generated_answer_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "unknown_queries": 0,
        "can_promote": False,
    }
    write_new(READINESS, report)
    return report


def _verify_readiness() -> dict:
    if not READINESS.is_file() or ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("one valid readiness and no prior query attempt are required")
    report = read_json(READINESS)
    if (
        report.get("complete") is not True
        or report.get("status") != "ready_for_execution"
        or report.get("plan_sha256") != PLAN_SHA256
        or report.get("model_loads") != 0
        or report.get("checkpoint_loads") != 0
        or report.get("model_binding_evaluations") != 0
        or report.get("generated_answer_queries") != 0
        or report.get("parameter_updates") != 0
        or report.get("checkpoint_hash_count") != 6
        or report.get("planned_binding_evaluations") != 180
        or report.get("planned_generated_answer_queries") != 0
        or report.get("instrument_preflight", {}).get("complete") is not True
        or float(report.get("cpu_seconds", math.inf)) > CPU_LIMIT_SECONDS
    ):
        raise ValueError("N1-129 readiness is stale or incomplete")
    for path, expected in {**report["source_hashes"], **report["checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed after N1-129 readiness: {path}")
    report["resolved_readiness_path"] = str(READINESS.resolve())
    report["resolved_readiness_sha256"] = sha256(READINESS)
    return report


def _measure_binding(model, item: dict, arm: str, session: str) -> dict:
    paired = e11.scorer.paired
    data = paired.project(item)
    paired.observe(model, data, session)
    tape = paired.evidence(model, session, arm)
    allowed, target_details = n128.n127._tape_mask(tape, item)
    question_state = model.question_state(data.question)
    bound_state, alignment, salience = model.bind_question(tape, question_state)
    metrics = _mass_metrics(alignment, salience, allowed)
    if bound_state.input_tape is not None and bound_state.input_tape is not tape:
        raise ValueError("native question binding changed its full observation tape")
    if len(tape.sources) != 16:
        raise ValueError("native binding mass audit requires the original sixteen-row context")
    return {
        "id": item["id"],
        "kind": item["kind"],
        "split": item["split"],
        **target_details,
        **metrics,
    }


def _score_endpoint(
    bound: dict,
    face_label: str,
    seed: int,
    arm: str,
    items: list[dict],
    budget: e11.CpuDeadline,
) -> tuple[list[dict], int, bool, str | None]:
    rows = []
    evaluations = 0
    loaded = False
    error = None
    model = optimizer = paired = None
    before = None
    path, digest = bound["selected_paths"][seed, arm]
    try:
        model, optimizer, before, paired = e11._load_model(
            bound["candidate_globals"], path, digest, seed, arm, bound["protocol"]
        )
        loaded = True
        for item in items:
            budget.guard(reserve=5)
            session = f"n1-plan129-{face_label}-{seed}-{arm}-{item['id']}"
            if session in model._observations:
                raise ValueError("N1-129 audit session collides with frozen history")
            evaluations += 1
            try:
                measurement = _measure_binding(model, item, arm, session)
                key = (face_label, seed, arm, item["id"])
                native = bound["native_rows"][face_label][seed, arm, item["id"]]
                decoder = bound["question_rows"][key]
                binding = bound["prior_rows"][key]
                pruned = bound["pruned_rows"][face_label][seed, arm, item["id"]]
                rows.append(
                    {
                        **measurement,
                        "face": face_label,
                        "seed": seed,
                        "arm": arm,
                        "native_full_context_free_exact": native["free_exact"],
                        "decoder_mask_free_exact": decoder["free_exact"],
                        "question_binding_mask_free_exact": binding["free_exact"],
                        "source_pruned_free_exact": pruned["free_exact"],
                    }
                )
            finally:
                model._observations.pop(session, None)
        if paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("native binding audit changed model, optimizer, RNG or history")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if loaded and model is not None:
            try:
                if paired.fingerprint(model.bundle(optimizer)) != before:
                    error += "; frozen bundle changed"
            except BaseException as fingerprint_error:
                error += (
                    f"; fingerprint check {type(fingerprint_error).__name__}: {fingerprint_error}"
                )
    finally:
        if loaded:
            del model, optimizer
    return rows, evaluations, loaded, error


def _aggregate(selected: list[dict]) -> dict:
    if not selected:
        return {"rows": 0}
    fields = (
        "target_alignment_mass_mean",
        "target_byte_share",
        "target_alignment_mass_enrichment",
        "salience_weighted_target_alignment_mass",
        "question_byte_argmax_on_target_fraction",
    )
    return {
        "rows": len(selected),
        **{
            f"mean_{field}": sum(row[field] for row in selected) / len(selected) for field in fields
        },
    }


def _summaries(rows: list[dict]) -> dict:
    report = {}
    for face_label in n128.n127.FACE_SOURCES:
        report[face_label] = {}
        for arm in ARMS:
            selected = [row for row in rows if row["face"] == face_label and row["arm"] == arm]
            report[face_label][arm] = {
                "all": _aggregate(selected),
                "native_correct": _aggregate(
                    [row for row in selected if row["native_full_context_free_exact"]]
                ),
                "native_incorrect": _aggregate(
                    [row for row in selected if not row["native_full_context_free_exact"]]
                ),
                "question_binding_mask_correct": _aggregate(
                    [row for row in selected if row["question_binding_mask_free_exact"]]
                ),
                "question_binding_mask_incorrect": _aggregate(
                    [row for row in selected if not row["question_binding_mask_free_exact"]]
                ),
                "by_kind": {
                    kind: _aggregate([row for row in selected if row["kind"] == kind])
                    for kind in KINDS
                },
            }
    return report


def run() -> dict:
    if ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("PLAN-N1-129 activation audit is one-shot")
    stage_started = time.process_time()
    readiness = _verify_readiness()
    bound = _source_set()
    elapsed = time.process_time() - stage_started
    remaining = CPU_LIMIT_SECONDS - float(readiness["cpu_seconds"]) - elapsed
    if remaining <= 0:
        raise TimeoutError("N1-129 cumulative CPU budget exhausted before binding audit")
    budget = e11.CpuDeadline(remaining)
    torch.set_num_threads(1)
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan129-native-question-binding-mass-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": readiness["resolved_readiness_sha256"],
            "checkpoint_hashes": readiness["checkpoint_hashes"],
            "planned_binding_evaluations": 180,
            "generated_answer_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    rows = []
    binding_evaluations = 0
    model_loads = 0
    errors = []
    try:
        for face_label, (_, _, items) in bound["face_data"].items():
            for seed in SEEDS:
                for arm in ARMS:
                    budget.guard(reserve=5)
                    endpoint_rows, evaluations, loaded, error = _score_endpoint(
                        bound, face_label, seed, arm, items, budget
                    )
                    rows.extend(endpoint_rows)
                    binding_evaluations += evaluations
                    model_loads += int(loaded)
                    if error:
                        errors.append(f"{face_label}/{seed}/{arm}: {error}")
                        raise RuntimeError(
                            "endpoint stopped; preserve activation rows and do not retry"
                        )
    except BaseException as exc:
        if not errors:
            errors.append(f"{type(exc).__name__}: {exc}")

    source_hashes = {
        **readiness["source_hashes"],
        **readiness["checkpoint_hashes"],
        readiness["resolved_readiness_path"]: readiness["resolved_readiness_sha256"],
        str(ATTEMPT.resolve()): sha256(ATTEMPT),
    }
    unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == expected
        for path, expected in source_hashes.items()
    )
    stage_cpu = time.process_time() - stage_started
    cumulative = float(readiness["cpu_seconds"]) + stage_cpu
    complete = (
        not errors
        and len(rows) == 180
        and binding_evaluations == 180
        and model_loads == 12
        and unchanged
        and cumulative <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan129-native-question-binding-mass-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": readiness["resolved_readiness_sha256"],
        "attempt_sha256": sha256(ATTEMPT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "checkpoint_hashes": readiness["checkpoint_hashes"],
        "rows": rows,
        "summary": _summaries(rows),
        "model_loads": model_loads,
        "model_binding_evaluations": binding_evaluations,
        "generated_answer_queries": 0,
        "teacher_forced_queries": 0,
        "output_tokens_generated": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "unknown_queries": 0,
        "cpu_seconds": round(stage_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "activation_audit_is_capability": False,
        "can_promote": False,
        "errors": errors,
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--readiness", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    result = source_readiness() if args.readiness else run()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "schema",
                    "status",
                    "complete",
                    "cpu_seconds",
                    "cumulative_cpu_seconds",
                    "model_binding_evaluations",
                    "generated_answer_queries",
                    "can_promote",
                    "errors",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
