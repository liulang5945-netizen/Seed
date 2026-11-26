"""One-shot PLAN-N1-130 per-relation native question-binding mass audit."""

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
    eval_taiji_n1_v13_v14_native_question_binding_mass as n129,
)  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-130_native_relation_row_binding_mass_audit_draft_20261019.md"
PLAN_SHA256 = "9cc65dde0e99667e6039be9e11a7844a7d4ca15ec038141c1d59aae753aeff03"
N129_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-129_execution_closeout_20261019.md"
N129_READINESS = (
    ROOT / "reports/taiji_n1_v13_v14_native_question_binding_mass_readiness_20261019.json"
)
N129_ATTEMPT = ROOT / "reports/taiji_n1_v13_v14_native_question_binding_mass_attempt_20261019.json"
N129_RESULT = ROOT / "reports/taiji_n1_v13_v14_native_question_binding_mass_result_20261019.json"
N129_CLOSEOUT_SHA256 = "b09a555febb3ba62058edff2a8e6613736fd36a5e159003cf78bfc58ac0f6fd5"
N129_READINESS_SHA256 = "efed73791e26d97a6b3d91e65db1a758def484b686ce1e254ca4c956f63b743e"
N129_ATTEMPT_SHA256 = "070c02572e0efd0ad568ae7153a9c9cb9402b1381cbbe676caede481aa7f64cd"
N129_RESULT_SHA256 = "a533803ccf39d2f2f023a2e977374a9da4fc4737202fa74fe5b26605aa4bfd9d"
READINESS = (
    ROOT / "reports/taiji_n1_v13_v14_native_relation_row_binding_mass_readiness_20261019.json"
)
ATTEMPT = ROOT / "reports/taiji_n1_v13_v14_native_relation_row_binding_mass_attempt_20261019.json"
RESULT = ROOT / "reports/taiji_n1_v13_v14_native_relation_row_binding_mass_result_20261019.json"
CPU_LIMIT_SECONDS = 600
SEEDS = n129.SEEDS
ARMS = n129.ARMS
KINDS = n129.KINDS


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


def _target_fact_rows(item: dict) -> list[tuple[int, str, str]]:
    subject = item["entities"][0]
    facts = []
    for index, text in enumerate(item["observations"]):
        if item["kind"] == "lookup" and text.startswith(subject + " uses "):
            facts.append((index, "uses", text))
        elif item["kind"] == "revision":
            if text.startswith(subject + " uses "):
                facts.append((index, "uses", text))
            elif text.startswith(subject + " now uses "):
                facts.append((index, "now uses", text))
        elif item["kind"] == "transition":
            if text.startswith(subject + " faces "):
                facts.append((index, "faces", text))
            elif text.startswith(subject + " turns "):
                facts.append((index, "turns", text))
        elif item["kind"] not in KINDS:
            raise ValueError(f"unsupported relation-row audit kind: {item['kind']}")
    expected = {"lookup": 1, "revision": 2, "transition": 2}[item["kind"]]
    if len(facts) != expected:
        raise ValueError(f"expected {expected} task facts for {item['id']}, got {len(facts)}")
    if [fact[0] for fact in facts] != sorted(fact[0] for fact in facts):
        raise ValueError("target fact rows lost their original source order")
    return facts


def _synthetic_preflight() -> dict:
    fixtures = (
        {
            "id": "lookup",
            "kind": "lookup",
            "entities": ["alpha"],
            "observations": ["alpha uses red.", "beta uses blue."],
        },
        {
            "id": "revision",
            "kind": "revision",
            "entities": ["alpha"],
            "observations": ["alpha uses red.", "beta uses blue.", "alpha now uses green."],
        },
        {
            "id": "transition",
            "kind": "transition",
            "entities": ["alpha"],
            "observations": ["alpha faces west.", "beta faces north.", "alpha turns right."],
        },
    )
    fact_counts = {item["kind"]: len(_target_fact_rows(item)) for item in fixtures}
    if fact_counts != {"lookup": 1, "revision": 2, "transition": 2}:
        raise AssertionError("synthetic relation-row selection failed")
    partition_checks = {}
    for item in fixtures:
        positions = [
            (f"source-{row_index + 1}", byte_index)
            for row_index, text in enumerate(item["observations"])
            for byte_index in range(len(text.encode("utf-8")))
        ]
        fact_rows = _target_fact_rows(item)
        masks = [
            torch.tensor([source == f"source-{index + 1}" for source, _ in positions])
            for index, _, _ in fact_rows
        ]
        union = torch.stack(masks).any(0)
        total = torch.stack(masks).int().sum(0)
        if bool((total > 1).any()) or int(union.sum().item()) != sum(
            int(mask.sum().item()) for mask in masks
        ):
            raise AssertionError("synthetic fact-row masks must be disjoint and complete")
        partition_checks[item["kind"]] = True
    return {
        "complete": True,
        "model_loads": 0,
        "model_binding_evaluations": 0,
        "generated_answer_queries": 0,
        "parameter_updates": 0,
        "fact_row_counts": fact_counts,
        "source_order_preserved": True,
        "mask_partition_contract": partition_checks,
    }


def _source_set() -> dict:
    for path, expected in (
        (PLAN, PLAN_SHA256),
        (N129_CLOSEOUT, N129_CLOSEOUT_SHA256),
        (N129_READINESS, N129_READINESS_SHA256),
        (N129_ATTEMPT, N129_ATTEMPT_SHA256),
        (N129_RESULT, N129_RESULT_SHA256),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen N1-129/N1-130 source changed: {path}")
    n129_result = read_json(N129_RESULT)
    if (
        n129_result.get("complete") is not True
        or n129_result.get("model_binding_evaluations") != 180
        or n129_result.get("generated_answer_queries") != 0
        or n129_result.get("parameter_updates") != 0
        or n129_result.get("source_hashes_unchanged") is not True
        or len(n129_result.get("rows", [])) != 180
    ):
        raise ValueError("complete PLAN-N1-129 activation audit is required")
    bound = n129._source_set()
    sources = dict(bound["source_hashes"])
    for path in (
        PLAN,
        N129_CLOSEOUT,
        N129_READINESS,
        N129_ATTEMPT,
        N129_RESULT,
        Path(__file__).resolve(),
    ):
        sources[str(Path(path).resolve())] = sha256(Path(path))
    return {**bound, "n129_result": n129_result, "source_hashes": sources}


def source_readiness() -> dict:
    if READINESS.exists() or ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("PLAN-N1-130 outputs are write-once")
    started = time.process_time()
    preflight = _synthetic_preflight()
    bound = _source_set()
    for path, expected in {**bound["source_hashes"], **bound["checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed during N1-130 readiness: {path}")
    cpu = time.process_time() - started
    complete = cpu <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan130-native-relation-row-binding-mass-readiness-v1",
        "status": "ready_for_execution" if complete else "stopped_cpu_limit",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "source_hashes": bound["source_hashes"],
        "checkpoint_hashes": bound["checkpoint_hashes"],
        "checkpoint_hash_count": len(bound["checkpoint_hashes"]),
        "face_sha256": {
            label: sha256(n129.n128.n127.FACE_SOURCES[label]["face"])
            for label in bound["face_data"]
        },
        "n129_result_sha256": sha256(N129_RESULT),
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
        raise ValueError("N1-130 readiness is stale or incomplete")
    for path, expected in {**report["source_hashes"], **report["checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed after N1-130 readiness: {path}")
    report["resolved_readiness_path"] = str(READINESS.resolve())
    report["resolved_readiness_sha256"] = sha256(READINESS)
    return report


def _measure(model, item: dict, arm: str, session: str) -> dict:
    paired = e11.scorer.paired
    data = paired.project(item)
    paired.observe(model, data, session)
    tape = paired.evidence(model, session, arm)
    target_mask, _ = n129.n128.n127._tape_mask(tape, item)
    facts = _target_fact_rows(item)
    if len(tape.sources) != 16 or len(tape.positions) != target_mask.numel():
        raise ValueError("native row audit requires the complete original sixteen-row tape")

    row_masks = []
    for observation_index, relation, text in facts:
        source_id = f"source-{observation_index + 1}"
        row_mask = torch.tensor(
            [source == source_id for source, _ in tape.positions],
            dtype=torch.bool,
            device=tape.values.device,
        )
        if not bool(row_mask.any()):
            raise ValueError(f"target fact row is absent from the tape: {source_id}")
        row_masks.append((observation_index, relation, text, source_id, row_mask))
    combined = torch.stack([row_mask for *_, row_mask in row_masks]).any(0)
    if not torch.equal(combined, target_mask):
        raise ValueError("individual target relation masks do not partition the target rows")

    question_state = model.question_state(data.question)
    bound_state, alignment, salience = model.bind_question(tape, question_state)
    if bound_state.input_tape is not None and bound_state.input_tape is not tape:
        raise ValueError("native binding changed the full evidence tape")
    if not torch.isfinite(alignment).all() or not torch.isfinite(salience).all():
        raise ValueError("native relation-row alignment contains non-finite values")
    if not torch.allclose(
        alignment.sum(-1),
        torch.ones(alignment.shape[0], device=alignment.device),
        rtol=1e-6,
        atol=1e-6,
    ) or not torch.allclose(salience.sum(), salience.new_tensor(1.0), rtol=1e-6, atol=1e-6):
        raise ValueError("native alignment or question salience is not normalized")

    argmax_positions = alignment.argmax(-1)
    fact_rows = []
    for observation_index, _text, source_id, row_mask in row_masks:
        row_mass = alignment[:, row_mask].sum(-1)
        row_share = float(row_mask.float().mean().item())
        row_argmax = row_mask[argmax_positions].float()
        fact_rows.append(
            {
                "observation_index": observation_index,
                "relation": relation,
                "source_id": source_id,
                "fact_byte_count": int(row_mask.sum().item()),
                "tape_byte_share": row_share,
                "alignment_mass_mean": float(row_mass.mean().item()),
                "alignment_mass_min": float(row_mass.min().item()),
                "alignment_mass_max": float(row_mass.max().item()),
                "alignment_mass_by_question_byte": [float(value) for value in row_mass.tolist()],
                "alignment_mass_enrichment": float(row_mass.mean().item()) / row_share,
                "salience_weighted_alignment_mass": float((row_mass * salience).sum().item()),
                "question_byte_argmax_on_row_fraction": float(row_argmax.mean().item()),
            }
        )
    distractor_mask = ~target_mask
    distractor_mass = alignment[:, distractor_mask].sum(-1)
    fact_mass_sum = torch.stack([alignment[:, row_mask].sum(-1) for *_, row_mask in row_masks]).sum(
        0
    )
    target_mass = alignment[:, target_mask].sum(-1)
    partition_error = float((fact_mass_sum - target_mass).abs().max().item())
    if partition_error > 1e-6:
        raise ValueError("individual fact-row masses do not sum to target-row mass")
    return {
        "id": item["id"],
        "kind": item["kind"],
        "split": item["split"],
        "question_byte_count": int(alignment.shape[0]),
        "full_context_row_count": len(tape.sources),
        "target_fact_rows": fact_rows,
        "distractor_alignment_mass_mean": float(distractor_mass.mean().item()),
        "distractor_alignment_mass_by_question_byte": [
            float(value) for value in distractor_mass.tolist()
        ],
        "target_fact_mass_partition_error_max": partition_error,
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
            session = f"n1-plan130-{face_label}-{seed}-{arm}-{item['id']}"
            if session in model._observations:
                raise ValueError("N1-130 audit session collides with frozen history")
            evaluations += 1
            try:
                measurement = _measure(model, item, arm, session)
                key = (face_label, seed, arm, item["id"])
                native = bound["native_rows"][face_label][seed, arm, item["id"]]
                decoder = bound["question_rows"][key]
                binding = bound["prior_rows"][key]
                pruned = bound["pruned_rows"][face_label][seed, arm, item["id"]]
                measurement.update(
                    face=face_label,
                    seed=seed,
                    arm=arm,
                    native_full_context_free_exact=native["free_exact"],
                    decoder_mask_free_exact=decoder["free_exact"],
                    question_binding_mask_free_exact=binding["free_exact"],
                    source_pruned_free_exact=pruned["free_exact"],
                )
                rows.append(measurement)
            finally:
                model._observations.pop(session, None)
        if paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("relation-row audit changed model, optimizer, RNG or history")
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


def _aggregate_fact_rows(fact_rows: list[dict]) -> dict:
    if not fact_rows:
        return {"rows": 0}
    return {
        "rows": len(fact_rows),
        "mean_tape_byte_share": sum(row["tape_byte_share"] for row in fact_rows) / len(fact_rows),
        "mean_alignment_mass": sum(row["alignment_mass_mean"] for row in fact_rows)
        / len(fact_rows),
        "mean_enrichment": sum(row["alignment_mass_enrichment"] for row in fact_rows)
        / len(fact_rows),
        "mean_salience_weighted_mass": sum(
            row["salience_weighted_alignment_mass"] for row in fact_rows
        )
        / len(fact_rows),
        "mean_question_byte_argmax_fraction": sum(
            row["question_byte_argmax_on_row_fraction"] for row in fact_rows
        )
        / len(fact_rows),
    }


def _summaries(rows: list[dict]) -> dict:
    summaries = {}
    for face_label in n129.n128.n127.FACE_SOURCES:
        summaries[face_label] = {}
        for arm in ARMS:
            selected = [row for row in rows if row["face"] == face_label and row["arm"] == arm]
            relation_types = sorted(
                {fact["relation"] for row in selected for fact in row["target_fact_rows"]}
            )
            by_relation = {}
            for relation in relation_types:
                relation_facts = [
                    fact
                    for row in selected
                    for fact in row["target_fact_rows"]
                    if fact["relation"] == relation
                ]
                by_relation[relation] = {
                    "all": _aggregate_fact_rows(relation_facts),
                    "native_correct": _aggregate_fact_rows(
                        [
                            fact
                            for row in selected
                            if row["native_full_context_free_exact"]
                            for fact in row["target_fact_rows"]
                            if fact["relation"] == relation
                        ]
                    ),
                    "native_incorrect": _aggregate_fact_rows(
                        [
                            fact
                            for row in selected
                            if not row["native_full_context_free_exact"]
                            for fact in row["target_fact_rows"]
                            if fact["relation"] == relation
                        ]
                    ),
                    "question_binding_mask_correct": _aggregate_fact_rows(
                        [
                            fact
                            for row in selected
                            if row["question_binding_mask_free_exact"]
                            for fact in row["target_fact_rows"]
                            if fact["relation"] == relation
                        ]
                    ),
                    "question_binding_mask_incorrect": _aggregate_fact_rows(
                        [
                            fact
                            for row in selected
                            if not row["question_binding_mask_free_exact"]
                            for fact in row["target_fact_rows"]
                            if fact["relation"] == relation
                        ]
                    ),
                }
            summaries[face_label][arm] = {
                "rows": len(selected),
                "relation_types": by_relation,
            }
    return summaries


def run() -> dict:
    if ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("PLAN-N1-130 activation audit is one-shot")
    stage_started = time.process_time()
    readiness = _verify_readiness()
    bound = _source_set()
    elapsed = time.process_time() - stage_started
    remaining = CPU_LIMIT_SECONDS - float(readiness["cpu_seconds"]) - elapsed
    if remaining <= 0:
        raise TimeoutError("N1-130 cumulative CPU budget exhausted before binding audit")
    budget = e11.CpuDeadline(remaining)
    torch.set_num_threads(1)
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan130-native-relation-row-binding-mass-attempt-v1",
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
        "schema": "n1-plan130-native-relation-row-binding-mass-result-v1",
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
