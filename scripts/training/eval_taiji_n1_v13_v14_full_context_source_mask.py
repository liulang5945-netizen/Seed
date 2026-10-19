"""One-shot PLAN-N1-127 full-context oracle decoder-source-mask diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import build_taiji_n1_v13_after_n122 as b13  # noqa: E402
from scripts.training import build_taiji_n1_v14_after_n123 as b14  # noqa: E402
from scripts.training import eval_taiji_n1_v11_after_n120 as e11  # noqa: E402
from scripts.training import eval_taiji_n1_v13_after_n122 as e13  # noqa: E402
from scripts.training import (
    eval_taiji_n1_v13_v14_source_pruned as n126,
)  # noqa: E402
from scripts.training import eval_taiji_n1_v14_after_n123 as e14  # noqa: E402
from scripts.training import (
    run_taiji_n1_expanded_normalized_rwl_length_balanced as n120,
)  # noqa: E402
from taiji.native_relative_evidence_candidate import RelativeEvidenceCandidate  # noqa: E402

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-127_full_context_source_attention_mask_diagnostic_draft_20261019.md"
)
PLAN_SHA256 = "a3f75cc4a8c524eed31643aeb12e47d797075f0569df8886257f49e475149ac8"
PLAN126 = (
    ROOT / "plans/reference/PLAN-N1-126_source_pruned_saved_error_diagnostic_draft_20261019.md"
)
PLAN126_SHA256 = "ed843b5af3cf23b5034905c84296aca77f6b24cc7050c08760a8432322b14eef"
PLAN126_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-126_execution_closeout_20261019.md"
PLAN126_RUNNER = ROOT / "scripts/training/eval_taiji_n1_v13_v14_source_pruned.py"
PLAN126_READINESS = ROOT / "reports/taiji_n1_v13_v14_source_pruned_readiness_20261019.json"
PLAN126_ATTEMPT = ROOT / "reports/taiji_n1_v13_v14_source_pruned_attempt_20261019.json"
PLAN126_RESULT = ROOT / "reports/taiji_n1_v13_v14_source_pruned_result_20261019.json"
PLAN126_READINESS_SHA256 = "d7b4c267c052aad09120891ec4983183ff01fdd6d8208dfee6347e66e50387ba"
PLAN126_ATTEMPT_SHA256 = "0f31b3eb76a921af00c32186dfedff3c37016ff566df423ada73368099578270"
PLAN126_RESULT_SHA256 = "12c4c582c682b71a6a52185dfc855bdc7b165d12fcb5082a193f304f8a980102"

READINESS = ROOT / "reports/taiji_n1_v13_v14_full_context_source_mask_readiness_20261019.json"
ATTEMPT = ROOT / "reports/taiji_n1_v13_v14_full_context_source_mask_attempt_20261019.json"
RESULT = ROOT / "reports/taiji_n1_v13_v14_full_context_source_mask_result_20261019.json"
CPU_LIMIT_SECONDS = 600
SEEDS = e11.SEEDS
ARMS = ("position", "cut")
KINDS = ("lookup", "revision", "transition")

FACE_SOURCES = {
    "v13": {
        "face": b13.FACE,
        "checks": b13.FACE_CHECKS,
        "readiness": b13.READINESS,
        "confirmation": b13.CONFIRM_REPORT,
        "range": b13.RANGE_REPORT,
        "builder": b13,
        "evaluator": e13,
        "seed": 113,
        "schema": "n1-hard-neighbor-v13-confirmation-v1",
    },
    "v14": {
        "face": b14.FACE,
        "checks": b14.FACE_CHECKS,
        "readiness": b14.READINESS,
        "confirmation": b14.CONFIRM_REPORT,
        "range": b14.RANGE_REPORT,
        "builder": b14,
        "evaluator": e14,
        "seed": 114,
        "schema": "n1-hard-neighbor-v14-confirmation-v1",
    },
}


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


def _masked_softmax(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 1 or allowed.shape != scores.shape or not bool(allowed.any()):
        raise ValueError("source attention requires a nonempty one-dimensional tape mask")
    return torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=0)


def _masked_address(address: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    if address.ndim != 1 or allowed.shape != address.shape or not bool(allowed.any()):
        raise ValueError("relative address requires a nonempty one-dimensional tape mask")
    retained = address.masked_fill(~allowed, 0)
    total = retained.sum()
    if not bool(torch.isfinite(total)) or float(total.item()) <= 0:
        raise ValueError("oracle source mask removed all carried address mass")
    return retained / total


def _assert_probability_contract(weights: torch.Tensor, allowed: torch.Tensor) -> None:
    if weights.ndim != 1 or weights.shape != allowed.shape:
        raise ValueError("decoder attention shape differs from full evidence tape")
    if bool(torch.count_nonzero(weights.masked_select(~allowed)).item()):
        raise ValueError("decoder attention assigned probability outside target rows")
    if not torch.isfinite(weights).all() or not torch.allclose(
        weights.sum(), weights.new_tensor(1.0), rtol=1e-6, atol=1e-6
    ):
        raise ValueError("masked decoder attention is not a normalized distribution")


def instrument_preflight() -> dict:
    scores = torch.tensor([1.25, -0.5, 0.75, 2.0, -1.0, 0.1], dtype=torch.float64)
    all_allowed = torch.ones_like(scores, dtype=torch.bool)
    native = torch.softmax(scores, dim=0)
    unmasked = _masked_softmax(scores, all_allowed)
    if not torch.equal(native, unmasked):
        raise AssertionError("all-allowed source mask must reproduce native softmax bit-for-bit")

    allowed = torch.tensor([True, True, False, True, False, False])
    masked = _masked_softmax(scores, allowed)
    _assert_probability_contract(masked, allowed)

    # Two three-byte sources with the same source-local clamped shifts as the model tape.
    shifts = torch.tensor(
        [[0, 0, 1, 3, 3, 4], [0, 1, 2, 3, 4, 5], [1, 2, 2, 4, 5, 5]],
        dtype=torch.long,
    )
    tape = SimpleNamespace(shift_indices=shifts)
    source_mask = torch.tensor([True, True, True, False, False, False])
    address = _masked_address(torch.tensor([0.2, 0.5, 0.3, 0.0, 0.0, 0.0]), source_mask)
    moved = RelativeEvidenceCandidate.transport(tape, address, torch.tensor([0.2, 0.6, 0.2]))
    _assert_probability_contract(moved, source_mask)
    return {
        "complete": True,
        "model_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "all_allowed_softmax_bitwise_equal": True,
        "masked_probability_outside_target": 0,
        "masked_probability_sum": float(masked.sum().item()),
        "source_local_transport_outside_target": 0,
        "transport_probability_sum": float(moved.sum().item()),
    }


def _target_row_indices(item: dict) -> list[int]:
    subject = item["entities"][0]
    selected = []
    for index, row in enumerate(item["observations"]):
        if item["kind"] == "lookup":
            keep = row.startswith(subject + " uses ")
        elif item["kind"] == "revision":
            keep = row.startswith(subject + " uses ") or row.startswith(subject + " now uses ")
        elif item["kind"] == "transition":
            keep = row.startswith(subject + " faces ") or row.startswith(subject + " turns ")
        else:
            raise ValueError(f"unsupported source-mask task: {item['kind']}")
        if keep:
            selected.append(index)
    expected = {"lookup": 1, "revision": 2, "transition": 2}[item["kind"]]
    if len(selected) != expected:
        raise ValueError(f"wrong target-row count for {item['id']}: {selected}")
    return selected


def _tape_mask(tape, item: dict) -> tuple[torch.Tensor, dict]:
    if len(tape.sources) != len(item["observations"]) or len(tape.positions) == 0:
        raise ValueError("full original observation tape is required")
    target_rows = set(_target_row_indices(item))
    expected_sources = {f"source-{index + 1}": index for index in range(len(tape.sources))}
    for source_index, source in enumerate(tape.sources):
        if source.source != f"source-{source_index + 1}" or source.content != item["observations"][
            source_index
        ].encode("utf-8"):
            raise ValueError("observed sources differ from the original full-context item")
    allowed = torch.tensor(
        [expected_sources[source] in target_rows for source, _ in tape.positions],
        dtype=torch.bool,
        device=tape.values.device,
    )
    full_count = sum(len(source.content) for source in tape.sources)
    if len(allowed) != full_count or not bool(allowed.any()):
        raise ValueError("attention mask does not span the full evidence tape")
    allowed_rows = [index for index in range(len(tape.sources)) if index in target_rows]
    return allowed, {
        "target_observation_indices": allowed_rows,
        "target_source_ids": [f"source-{index + 1}" for index in allowed_rows],
        "allowed_byte_positions": int(allowed.sum().item()),
        "full_byte_positions": full_count,
        "observed_rows": len(tape.sources),
    }


def _masked_free_trace(model, tape, session: str, question: bytes, item: dict):
    allowed, details = _tape_mask(tape, item)
    native_softmax = torch.softmax
    native_step = model.step
    native_transport = model.transport
    native_generate = model.generate_trace
    audit = {
        **details,
        "masked_content_softmax_calls": 0,
        "checked_decoder_steps": 0,
        "masked_address_transport_calls": 0,
    }

    def source_softmax(scores, *args, **kwargs):
        dim = kwargs.get("dim", args[0] if args else None)
        if (
            isinstance(scores, torch.Tensor)
            and scores.ndim == 1
            and dim == 0
            and scores.numel() == allowed.numel()
        ):
            audit["masked_content_softmax_calls"] += 1
            result = native_softmax(scores.masked_fill(~allowed, -torch.inf), *args, **kwargs)
            _assert_probability_contract(result, allowed)
            return result
        return native_softmax(scores, *args, **kwargs)

    def checked_step(step_tape, step_session, state, previous):
        logits, next_state, weights = native_step(step_tape, step_session, state, previous)
        _assert_probability_contract(weights, allowed)
        audit["checked_decoder_steps"] += 1
        return logits, next_state, weights

    def masked_transport(transport_tape, address, mixture):
        selected_address = _masked_address(address, allowed)
        _assert_probability_contract(selected_address, allowed)
        moved = native_transport(transport_tape, selected_address, mixture)
        moved = _masked_address(moved, allowed)
        _assert_probability_contract(moved, allowed)
        audit["masked_address_transport_calls"] += 1
        return moved

    def generate_with_mask(generate_tape, generate_session, generate_question, max_length=256):
        if (
            generate_tape is not tape
            or generate_session != session
            or generate_question != question
        ):
            raise ValueError("masked generation received a different tape or question")
        model.step = checked_step
        model.transport = masked_transport
        torch.softmax = source_softmax
        try:
            trace = native_generate(generate_tape, generate_session, generate_question, max_length)
        finally:
            torch.softmax = native_softmax
            if "step" in model.__dict__:
                del model.step
            if "transport" in model.__dict__:
                del model.transport
        if audit["checked_decoder_steps"] != trace.generated_steps:
            raise ValueError("final decoder attention was not checked on every generated step")
        if audit["masked_content_softmax_calls"] != 2 * trace.generated_steps:
            raise ValueError("both native and relative content reads must be masked per step")
        if audit["masked_address_transport_calls"] != max(trace.generated_steps - 1, 0):
            raise ValueError("carried relative address was not checked at every later step")
        audit["generated_steps"] = trace.generated_steps
        audit["boundary_native"] = trace.stop_reason == "boundary"
        return trace

    model.generate_trace = generate_with_mask
    try:
        trace = model.generate_trace(tape, session, question, 64)
    finally:
        if "generate_trace" in model.__dict__:
            del model.generate_trace
        if "step" in model.__dict__:
            del model.step
        if "transport" in model.__dict__:
            del model.transport
        torch.softmax = native_softmax
    return trace, audit


def _measure(model, item: dict, arm: str, session: str) -> tuple[dict, dict]:
    scorer = e11.scorer
    paired = scorer.paired
    if item.get("split") != "holdout" or arm not in ARMS:
        raise ValueError("frozen holdout split and position/cut arm required")
    data = paired.project(item)
    paired.observe(model, data, session)
    tape = paired.evidence(model, session, arm)
    target = item["answer"].encode()
    labels = torch.tensor([*target, 256])
    logits, _ = model.teacher_logits(tape, session, data.question, target)
    predicted = logits.argmax(dim=-1)
    trace, attention_audit = _masked_free_trace(model, tape, session, data.question, item)
    row = dict(
        id=item["id"],
        kind=item["kind"],
        split=item["split"],
        target_symbols=labels.tolist(),
        teacher_symbols=predicted.tolist(),
        teacher_tokens_correct=int((predicted == labels).sum().item()),
        teacher_exact=bool(torch.equal(predicted, labels)),
        teacher_ce=float(torch.nn.functional.cross_entropy(logits, labels).item()),
        raw_hex=trace.raw.hex(),
        output_hex=trace.output.hex(),
        stop_reason=trace.stop_reason,
        generated_steps=trace.generated_steps,
        free_exact=trace.raw == target and trace.stop_reason == "boundary",
    )
    return row, attention_audit


def _source_set() -> dict:
    for path, expected in (
        (PLAN, PLAN_SHA256),
        (PLAN126, PLAN126_SHA256),
        (PLAN126_READINESS, PLAN126_READINESS_SHA256),
        (PLAN126_ATTEMPT, PLAN126_ATTEMPT_SHA256),
        (PLAN126_RESULT, PLAN126_RESULT_SHA256),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen N1-126/127 contract source changed: {path}")
    n126_result = read_json(PLAN126_RESULT)
    n126_readiness = read_json(PLAN126_READINESS)
    if (
        n126_result.get("complete") is not True
        or n126_result.get("model_queries") != 180
        or n126_result.get("parameter_updates") != 0
        or n126_result.get("dev_queries") != 0
        or n126_result.get("can_promote") is not False
        or n126_readiness.get("model_queries") != 0
        or n126_readiness.get("parameter_updates") != 0
    ):
        raise ValueError("completed PLAN-N1-126 zero-update diagnostic receipts are required")

    face_data: dict[str, tuple[dict, dict, list[dict]]] = {}
    native_rows: dict[str, dict] = {}
    pruned_rows: dict[str, dict] = {}
    source_hashes: dict[str, str] = {}
    checkpoint_readiness: dict[str, dict] = {}

    for label, spec in FACE_SOURCES.items():
        ready = spec["builder"]._verify_readiness()
        face, checks = spec["evaluator"]._face_and_checks()
        range_report = read_json(spec["range"])
        confirmation = read_json(spec["confirmation"])
        if (
            face.get("schema") != spec["schema"]
            or face.get("seed") != spec["seed"]
            or checks.get("accepted") is not True
            or range_report.get("complete") is not True
            or range_report.get("range_gate", {}).get("range_check_passed") is not True
            or range_report.get("model_queries") != 180
            or confirmation.get("complete") is not True
            or confirmation.get("confirmation_queries") != 360
            or confirmation.get("total_model_queries") != 540
            or confirmation.get("source_hashes_unchanged") is not True
            or confirmation.get("can_promote") is not False
        ):
            raise ValueError(f"frozen {label} face and source reports are incomplete")
        items = [
            item
            for item in face["episodes"]
            if item["id"] in set(face["subsets"]["confirm"]) and item["kind"] in KINDS
        ]
        if len(items) != 15 or any(
            sum(item["kind"] == kind for item in items) != 5 for kind in KINDS
        ):
            raise ValueError(f"{label} source-mask denominator must be 5 per decodable kind")
        item_ids = {item["id"] for item in items}
        selected_native: dict[tuple[int, str, str], dict] = {}
        for row in confirmation["candidate_rows"]:
            key = (row["seed"], row["arm"], row["id"])
            if row["seed"] in SEEDS and row["arm"] in ARMS and row["id"] in item_ids:
                if key in selected_native:
                    raise ValueError(f"duplicate saved full-context row: {label}/{key}")
                selected_native[key] = row
        selected_pruned: dict[tuple[int, str, str], dict] = {}
        for row in n126_result["rows"]:
            if row.get("face") != label:
                continue
            key = (row["seed"], row["arm"], row["id"])
            if row["arm"] in ARMS and row["id"] in item_ids:
                if key in selected_pruned:
                    raise ValueError(f"duplicate saved source-pruned row: {label}/{key}")
                selected_pruned[key] = row
        if len(selected_native) != 90 or len(selected_pruned) != 90:
            raise ValueError(f"{label} full/native/pruned rows must each cover 15 × 6 cases")
        for item in items:
            n126.relevant_rows(item)
            _target_row_indices(item)
        face_data[label] = (face, checks, items)
        native_rows[label] = selected_native
        pruned_rows[label] = selected_pruned
        checkpoint_readiness[label] = ready
        for path in (
            spec["face"],
            spec["checks"],
            spec["readiness"],
            spec["range"],
            spec["confirmation"],
            spec["builder"].SOURCE_EXCLUSIONS,
            spec["builder"].FACE_ATTEMPT,
            spec["builder"].RANGE_ATTEMPT,
            spec["builder"].CONFIRM_ATTEMPT,
        ):
            if Path(path).is_file():
                source_hashes[str(Path(path).resolve())] = sha256(Path(path))
        source_hashes.update(ready["source_hashes"])

    readiness13 = checkpoint_readiness["v13"]
    readiness14 = checkpoint_readiness["v14"]
    if readiness13["checkpoint_hashes"] != readiness14["checkpoint_hashes"]:
        raise ValueError("v13/v14 checkpoint hash maps differ")
    baseline, candidate, _, _, restore, _ = e14._bound_inputs()
    _, candidate_paths = e14._checkpoint_paths(baseline, candidate)
    if (
        len(candidate_paths) != 9
        or restore.get("independent_full_bundle_restores") != 18
        or restore.get("all_full_bundles_equal") is not True
    ):
        raise ValueError("N1-120 full-bundle restore evidence is required")
    selected_paths = {(seed, arm): candidate_paths[seed, arm] for seed in SEEDS for arm in ARMS}
    checkpoint_hashes = {str(path.resolve()): digest for path, digest in selected_paths.values()}
    if len(checkpoint_hashes) != 6 or any(
        sha256(Path(path)) != digest for path, digest in checkpoint_hashes.items()
    ):
        raise ValueError("exact six candidate position/cut checkpoint hashes are required")
    if n126_result.get("endpoint_checkpoint_hashes") != checkpoint_hashes:
        raise ValueError("PLAN-N1-126 and N1-127 must reuse the exact frozen checkpoints")

    scorer_readiness = e11._verify_scorer_readiness()
    if scorer_readiness.get("model_queries") != 0 or scorer_readiness.get("parameter_updates") != 0:
        raise ValueError("frozen scorer readiness must remain query/update free")
    n120.configure()
    _, candidate_globals, _ = n120._BASE.make_runner(load_inputs=False)
    if not callable(candidate_globals.get("load")) or not callable(e11.scorer.measure):
        raise TypeError("frozen candidate loader and native measurement path are required")

    source_hashes.update(scorer_readiness["source_hashes"])
    source_hashes.update(n126_readiness["source_hashes"])
    explicit_paths = (
        PLAN,
        PLAN126,
        PLAN126_CLOSEOUT,
        PLAN126_RUNNER,
        PLAN126_READINESS,
        PLAN126_ATTEMPT,
        PLAN126_RESULT,
        Path(__file__).resolve(),
        Path(e13.__file__).resolve(),
        Path(e14.__file__).resolve(),
        Path(n126.__file__).resolve(),
        Path(e11.__file__).resolve(),
        Path(e11.scorer.__file__).resolve(),
    )
    for path in explicit_paths:
        source_hashes[str(Path(path).resolve())] = sha256(Path(path))
    for path, expected in checkpoint_hashes.items():
        if not Path(path).is_file() or sha256(Path(path)) != expected:
            raise ValueError(f"candidate checkpoint changed: {path}")
    return {
        "face_data": face_data,
        "native_rows": native_rows,
        "pruned_rows": pruned_rows,
        "source_hashes": source_hashes,
        "checkpoint_hashes": checkpoint_hashes,
        "selected_paths": selected_paths,
        "protocol": candidate["protocol"],
        "candidate_globals": candidate_globals,
        "scorer_readiness": scorer_readiness,
        "n126_result": n126_result,
    }


def source_readiness() -> dict:
    if READINESS.exists() or ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("PLAN-N1-127 outputs are write-once")
    started = time.process_time()
    preflight = instrument_preflight()
    bound = _source_set()
    for path, expected in {**bound["source_hashes"], **bound["checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed during N1-127 readiness: {path}")
    cpu = time.process_time() - started
    ready = cpu <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan127-full-context-source-mask-readiness-v1",
        "status": "ready_for_execution" if ready else "stopped_cpu_limit",
        "complete": ready,
        "plan_sha256": PLAN_SHA256,
        "source_hashes": bound["source_hashes"],
        "checkpoint_hashes": bound["checkpoint_hashes"],
        "checkpoint_hash_count": len(bound["checkpoint_hashes"]),
        "face_sha256": {label: sha256(spec["face"]) for label, spec in FACE_SOURCES.items()},
        "confirmation_sha256": {
            label: sha256(spec["confirmation"]) for label, spec in FACE_SOURCES.items()
        },
        "plan126_result_sha256": sha256(PLAN126_RESULT),
        "planned_model_queries": 180,
        "instrument_preflight": preflight,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
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
        or report.get("model_queries") != 0
        or report.get("parameter_updates") != 0
        or report.get("checkpoint_hash_count") != 6
        or report.get("planned_model_queries") != 180
        or report.get("instrument_preflight", {}).get("complete") is not True
        or float(report.get("cpu_seconds", math.inf)) > CPU_LIMIT_SECONDS
    ):
        raise ValueError("N1-127 readiness is stale or incomplete")
    for path, expected in {**report["source_hashes"], **report["checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed after N1-127 readiness: {path}")
    report["resolved_readiness_path"] = str(READINESS.resolve())
    report["resolved_readiness_sha256"] = sha256(READINESS)
    return report


def _score_endpoint(
    bound: dict,
    face_label: str,
    seed: int,
    arm: str,
    items: list[dict],
    budget: e11.CpuDeadline,
) -> tuple[list[dict], int, bool, str | None]:
    rows: list[dict] = []
    audits: list[dict] = []
    queries = 0
    loaded = False
    error = None
    model = optimizer = paired = None
    before = None
    path, digest = bound["selected_paths"][seed, arm]
    try:
        model, optimizer, before, paired = e11._load_model(
            bound["candidate_globals"],
            path,
            digest,
            seed,
            arm,
            bound["protocol"],
        )
        loaded = True
        for item in items:
            budget.guard(reserve=5)
            session = f"n1-plan127-{face_label}-{seed}-{arm}-{item['id']}"
            if session in model._observations:
                raise ValueError("N1-127 scoring session collides with frozen history")
            queries += 1
            try:
                row, audit = _measure(model, item, arm, session)
                native = bound["native_rows"][face_label][seed, arm, item["id"]]
                pruned = bound["pruned_rows"][face_label][seed, arm, item["id"]]
                rows.append(
                    {
                        **row,
                        "face": face_label,
                        "seed": seed,
                        "arm": arm,
                        "intervention": "oracle_decoder_source_mask_full_context",
                        "native_full_context_free_exact": native["free_exact"],
                        "native_full_context_output_hex": native["output_hex"],
                        "source_pruned_free_exact": pruned["free_exact"],
                        "source_pruned_output_hex": pruned["output_hex"],
                        "native_to_mask_gain": row["free_exact"] and not native["free_exact"],
                        "native_to_mask_loss": native["free_exact"] and not row["free_exact"],
                        "pruned_to_mask_gain": row["free_exact"] and not pruned["free_exact"],
                        "pruned_to_mask_loss": pruned["free_exact"] and not row["free_exact"],
                        "attention_audit": audit,
                    }
                )
                audits.append(audit)
            finally:
                model._observations.pop(session, None)
        if paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("frozen scoring changed model, optimizer, RNG or stored history")
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
    return rows, audits, queries, loaded, error


def _summaries(rows: list[dict]) -> dict:
    summaries = {}
    for face_label in FACE_SOURCES:
        summaries[face_label] = {}
        for arm in ARMS:
            selected = [row for row in rows if row["face"] == face_label and row["arm"] == arm]
            summaries[face_label][arm] = {
                "rows": len(selected),
                "native_full_context_exact": sum(
                    row["native_full_context_free_exact"] for row in selected
                ),
                "source_pruned_exact": sum(row["source_pruned_free_exact"] for row in selected),
                "source_mask_full_context_exact": sum(row["free_exact"] for row in selected),
                "native_to_mask_gains": sum(row["native_to_mask_gain"] for row in selected),
                "native_to_mask_losses": sum(row["native_to_mask_loss"] for row in selected),
                "pruned_to_mask_gains": sum(row["pruned_to_mask_gain"] for row in selected),
                "pruned_to_mask_losses": sum(row["pruned_to_mask_loss"] for row in selected),
                "exact_by_kind": {
                    kind: sum(row["kind"] == kind and row["free_exact"] for row in selected)
                    for kind in KINDS
                },
                "masked_content_softmax_calls": sum(
                    row["attention_audit"]["masked_content_softmax_calls"] for row in selected
                ),
                "checked_decoder_steps": sum(
                    row["attention_audit"]["checked_decoder_steps"] for row in selected
                ),
                "masked_address_transport_calls": sum(
                    row["attention_audit"]["masked_address_transport_calls"] for row in selected
                ),
            }
    return summaries


def run() -> dict:
    if ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("PLAN-N1-127 query attempt is one-shot")
    stage_started = time.process_time()
    readiness = _verify_readiness()
    bound = _source_set()
    stage_elapsed = time.process_time() - stage_started
    remaining = CPU_LIMIT_SECONDS - float(readiness["cpu_seconds"]) - stage_elapsed
    if remaining <= 0:
        raise TimeoutError("N1-127 cumulative CPU budget exhausted before queries")
    budget = e11.CpuDeadline(remaining)
    torch.set_num_threads(1)
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan127-full-context-source-mask-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": readiness["resolved_readiness_sha256"],
            "checkpoint_hashes": readiness["checkpoint_hashes"],
            "planned_queries": 180,
            "intervention": "oracle_source_mask_on_both_content_reads_and_carried_address",
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    rows: list[dict] = []
    model_loads = 0
    model_queries = 0
    errors = []
    try:
        for face_label, (_, _, items) in bound["face_data"].items():
            for seed in SEEDS:
                for arm in ARMS:
                    budget.guard(reserve=5)
                    endpoint_rows, _, queries, loaded, error = _score_endpoint(
                        bound, face_label, seed, arm, items, budget
                    )
                    model_loads += int(loaded)
                    model_queries += queries
                    rows.extend(endpoint_rows)
                    if error:
                        errors.append(f"{face_label}/{seed}/{arm}: {error}")
                        raise RuntimeError(
                            "endpoint stopped; preserve partial evidence and do not retry"
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
    cumulative_cpu = float(readiness["cpu_seconds"]) + stage_cpu
    audit_totals = {
        "masked_content_softmax_calls": sum(
            row["attention_audit"]["masked_content_softmax_calls"] for row in rows
        ),
        "checked_decoder_steps": sum(
            row["attention_audit"]["checked_decoder_steps"] for row in rows
        ),
        "masked_address_transport_calls": sum(
            row["attention_audit"]["masked_address_transport_calls"] for row in rows
        ),
        "nonzero_attention_outside_target": 0,
        "nonzero_address_outside_target": 0,
    }
    complete = (
        not errors
        and len(rows) == 180
        and model_queries == 180
        and model_loads == 12
        and unchanged
        and cumulative_cpu <= CPU_LIMIT_SECONDS
        and audit_totals["masked_content_softmax_calls"]
        == 2 * audit_totals["checked_decoder_steps"]
    )
    report = {
        "schema": "n1-plan127-full-context-source-mask-result-v1",
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
        "attention_instrument_totals": audit_totals,
        "model_loads": model_loads,
        "model_queries": model_queries,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "unknown_queries": 0,
        "cpu_seconds": round(stage_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "oracle_intervention_is_capability": False,
        "can_promote": False,
        "errors": errors,
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--readiness", action="store_true")
    mode.add_argument("--run", action="store_true")
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
                    "model_queries",
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
