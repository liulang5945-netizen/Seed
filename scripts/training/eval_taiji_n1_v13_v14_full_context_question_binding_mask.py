"""One-shot PLAN-N1-128 full-context question-binding source-mask diagnostic."""

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
    eval_taiji_n1_v13_v14_full_context_source_mask as n127,
)  # noqa: E402
from taiji.contracts import Observation  # noqa: E402
from taiji.native_local_relation_evidence_candidate import (  # noqa: E402
    LocalRelationEvidenceCandidate,
)
from taiji.native_question_aligned_evidence_candidate import QuestionReadState  # noqa: E402
from taiji.native_relative_evidence_candidate import RelativeReadState  # noqa: E402

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-128_full_context_question_binding_mask_diagnostic_draft_20261019.md"
)
PLAN_SHA256 = "26884577737afc5b46dcff061329705fbd6adb96d942629c1d53de5cbe9a4b74"
READINESS = (
    ROOT / "reports/taiji_n1_v13_v14_full_context_question_binding_mask_readiness_20261019.json"
)
ATTEMPT = ROOT / "reports/taiji_n1_v13_v14_full_context_question_binding_mask_attempt_20261019.json"
RESULT = ROOT / "reports/taiji_n1_v13_v14_full_context_question_binding_mask_result_20261019.json"
N127_READINESS = ROOT / "reports/taiji_n1_v13_v14_full_context_source_mask_readiness_20261019.json"
N127_ATTEMPT = ROOT / "reports/taiji_n1_v13_v14_full_context_source_mask_attempt_20261019.json"
N127_RESULT = ROOT / "reports/taiji_n1_v13_v14_full_context_source_mask_result_20261019.json"
N127_READINESS_SHA256 = "cd4d401adb544635ca56f95fafda880811c526d23a5140c5ffbb34ab23b3c35f"
N127_ATTEMPT_SHA256 = "5307b0df7f8c4f27cd272e0c5809ad0c3d19020c3fdda82f07e2e0956e19519c"
N127_RESULT_SHA256 = "0efbd0cfc58de67fe155595d8f56a77565c9058bae70111b1278097752f3bbbf"
CPU_LIMIT_SECONDS = 600
SEEDS = n127.SEEDS
ARMS = n127.ARMS
KINDS = n127.KINDS


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


def _target_conditioned_bind(model, tape, state, allowed: torch.Tensor):
    if not getattr(state, "question_bytes", b""):
        raise ValueError("native question bytes are required for contextual binding")
    if allowed.ndim != 1 or allowed.numel() != len(tape.positions) or not bool(allowed.any()):
        raise ValueError("question binding requires a nonempty full-tape byte mask")

    observed_features = torch.cat([model.local_features(source.content) for source in tape.sources])
    states = torch.stack(tape.states)
    if tape.terminal_per_source:
        end, blocks = 0, []
        for source in tape.sources:
            length = len(source.content)
            end += length
            blocks.append(states[end - 1].expand(length, -1))
        states = torch.cat(blocks)
    if tape.cut_credit:
        states = states.detach()
        observed_features = observed_features.detach()

    question_features = model.local_features(state.question_bytes)
    scores = question_features @ observed_features.T
    masked_scores = scores.masked_fill(~allowed.unsqueeze(0), -torch.inf)
    temperature = model.local_log_temperature.exp().clamp(1, 32)
    alignment = torch.softmax(masked_scores * temperature, dim=-1)
    context = alignment @ states
    confidence = masked_scores.max(-1).values
    salience = torch.softmax(
        model.local_salience(
            torch.cat((state.question_states, confidence[:, None]), dim=-1)
        ).squeeze(-1),
        dim=0,
    )
    pooled = salience @ context
    question_states = state.question_states + torch.tanh(
        model.local_fusion(torch.cat((state.question_states, context), dim=-1))
    )
    hidden = state.hidden + torch.tanh(model.local_fusion(torch.cat((state.hidden, pooled))))
    base = QuestionReadState(RelativeReadState(hidden), question_states)

    embedded = model.embedding(
        torch.tensor(list(state.question_bytes), device=model.embedding.weight.device)
    )
    contextual_inputs = torch.cat((embedded, alignment @ states, confidence[:, None]), dim=-1)
    contextual_output, contextual_hidden = model.context_question(contextual_inputs.unsqueeze(0))
    gate = model.context_gate.tanh()
    bound = QuestionReadState(
        RelativeReadState(base.hidden + gate * (contextual_hidden[0, 0] - base.hidden)),
        base.question_states + gate * (contextual_output[0] - base.question_states),
    )
    alignment_outside = int(torch.count_nonzero(alignment.masked_select(~allowed[None, :])).item())
    alignment_sums = alignment.sum(dim=-1)
    audit = {
        "allowed_byte_positions": int(allowed.sum().item()),
        "full_byte_positions": int(allowed.numel()),
        "alignment_rows": int(alignment.shape[0]),
        "alignment_nonzero_outside_target": alignment_outside,
        "alignment_row_sums_min": float(alignment_sums.min().item()),
        "alignment_row_sums_max": float(alignment_sums.max().item()),
        "confidence_max_source_positions": int(allowed.sum().item()),
    }
    if alignment_outside or not torch.allclose(
        alignment_sums, torch.ones_like(alignment_sums), rtol=1e-6, atol=1e-6
    ):
        raise ValueError("target-conditioned question alignment escaped or failed to normalize")
    return bound, alignment, salience, audit


def instrument_preflight() -> dict:
    torch.set_num_threads(1)
    synthetic_observations = (
        "alpha uses red.",
        "alpha now uses blue.",
        "beta uses green.",
    )
    synthetic_item = {
        "id": "n1-plan128-synthetic",
        "entities": ["alpha"],
        "kind": "revision",
        "observations": list(synthetic_observations),
    }
    with torch.random.fork_rng():
        model = LocalRelationEvidenceCandidate()
        model.eval()
        for index, text in enumerate(synthetic_observations, 1):
            model.observe(
                "n1-plan128-synthetic", Observation("text", text, index, f"source-{index}")
            )
        tape = model.evidence("n1-plan128-synthetic")
        question = b"material of alpha?"
        state = model.question_state(question)
        parameters_before = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        with torch.no_grad():
            native, native_alignment, native_salience = model.bind_question(tape, state)
            all_allowed = torch.ones(len(tape.positions), dtype=torch.bool)
            unmasked, unmasked_alignment, unmasked_salience, _ = _target_conditioned_bind(
                model, tape, state, all_allowed
            )
            exact = (
                torch.equal(native.relative.hidden, unmasked.relative.hidden)
                and torch.equal(native.question_states, unmasked.question_states)
                and torch.equal(native_alignment, unmasked_alignment)
                and torch.equal(native_salience, unmasked_salience)
            )
            if not exact:
                raise AssertionError("all-allowed question-binding mask differs from native bind")

            target_mask, _ = n127._tape_mask(tape, synthetic_item)
            _, masked_alignment, _, masked_audit = _target_conditioned_bind(
                model, tape, state, target_mask
            )
            if bool(
                torch.count_nonzero(masked_alignment.masked_select(~target_mask[None, :])).item()
            ):
                raise AssertionError("target-only question alignment has off-target mass")
            if not torch.allclose(
                masked_alignment.sum(dim=-1),
                torch.ones(masked_alignment.shape[0]),
                rtol=1e-6,
                atol=1e-6,
            ):
                raise AssertionError("target-only question alignment rows must normalize")
            scores = (
                model.local_features(question)
                @ torch.cat([model.local_features(source.content) for source in tape.sources]).T
            )
            if masked_audit["alignment_nonzero_outside_target"] != 0:
                raise AssertionError("target-only alignment audit reports off-target values")
            expected_confidence = (
                scores.masked_fill(~target_mask[None, :], -torch.inf).max(-1).values
            )
            native_target_confidence = scores[:, target_mask].max(-1).values
            if not torch.equal(expected_confidence, native_target_confidence):
                raise AssertionError("confidence maximum must use target bytes only")
        unchanged = all(
            torch.equal(parameters_before[name], value)
            for name, value in model.state_dict().items()
        )
        if not unchanged or len(model._observations["n1-plan128-synthetic"]) != 3:
            raise AssertionError("synthetic binding preflight mutated model weights or full tape")
    return {
        "complete": True,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "all_allowed_bind_bitwise_equal": True,
        "target_alignment_off_target_nonzero": 0,
        "target_alignment_row_normalization": True,
        "confidence_max_target_only": True,
        "full_observation_rows_retained": 3,
        "synthetic_parameters_unchanged": unchanged,
    }


def _source_set() -> dict:
    for path, expected in (
        (PLAN, PLAN_SHA256),
        (N127_READINESS, N127_READINESS_SHA256),
        (N127_ATTEMPT, N127_ATTEMPT_SHA256),
        (N127_RESULT, N127_RESULT_SHA256),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen N1-127/128 source changed: {path}")
    n127_result = read_json(N127_RESULT)
    if (
        n127_result.get("complete") is not True
        or n127_result.get("model_queries") != 180
        or n127_result.get("parameter_updates") != 0
        or n127_result.get("source_hashes_unchanged") is not True
        or n127_result.get("can_promote") is not False
        or len(n127_result.get("rows", [])) != 180
    ):
        raise ValueError("complete PLAN-N1-127 one-shot result is required")
    prior = n127._source_set()
    question_rows: dict[tuple[str, int, str, str], dict] = {}
    for row in n127_result["rows"]:
        key = (row["face"], row["seed"], row["arm"], row["id"])
        if key in question_rows:
            raise ValueError(f"duplicate N1-127 comparison row: {key}")
        question_rows[key] = row
    for face_label, (_, _, items) in prior["face_data"].items():
        expected = {
            (face_label, seed, arm, item["id"]) for seed in SEEDS for arm in ARMS for item in items
        }
        if not expected.issubset(question_rows):
            raise ValueError(f"saved N1-127 comparison rows are incomplete for {face_label}")

    sources = dict(prior["source_hashes"])
    for path in (
        PLAN,
        n127.PLAN,
        n127.READINESS,
        n127.ATTEMPT,
        n127.RESULT,
        Path(__file__).resolve(),
    ):
        sources[str(Path(path).resolve())] = sha256(Path(path))
    return {
        **prior,
        "n127_result": n127_result,
        "question_rows": question_rows,
        "source_hashes": sources,
    }


def source_readiness() -> dict:
    if READINESS.exists() or ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("PLAN-N1-128 outputs are write-once")
    started = time.process_time()
    preflight = instrument_preflight()
    bound = _source_set()
    for path, expected in {**bound["source_hashes"], **bound["checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed during N1-128 readiness: {path}")
    cpu = time.process_time() - started
    complete = cpu <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan128-full-context-question-binding-mask-readiness-v1",
        "status": "ready_for_execution" if complete else "stopped_cpu_limit",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "source_hashes": bound["source_hashes"],
        "checkpoint_hashes": bound["checkpoint_hashes"],
        "checkpoint_hash_count": len(bound["checkpoint_hashes"]),
        "face_sha256": {
            label: sha256(n127.FACE_SOURCES[label]["face"]) for label in bound["face_data"]
        },
        "n127_result_sha256": sha256(N127_RESULT),
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
        raise ValueError("N1-128 readiness is stale or incomplete")
    for path, expected in {**report["source_hashes"], **report["checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed after N1-128 readiness: {path}")
    report["resolved_readiness_path"] = str(READINESS.resolve())
    report["resolved_readiness_sha256"] = sha256(READINESS)
    return report


def _measure(model, item: dict, arm: str, session: str) -> tuple[dict, dict, dict]:
    scorer = e11.scorer
    paired = scorer.paired
    data = paired.project(item)
    paired.observe(model, data, session)
    tape = paired.evidence(model, session, arm)
    target = item["answer"].encode()
    labels = torch.tensor([*target, 256])
    logits, _ = model.teacher_logits(tape, session, data.question, target)
    predicted = logits.argmax(dim=-1)

    allowed, mask_details = n127._tape_mask(tape, item)
    binding_calls = 0
    binding_audit = {}

    def masked_bind(bind_tape, state):
        nonlocal binding_calls, binding_audit
        if bind_tape is not tape:
            raise ValueError("masked question binding received a different full tape")
        binding_calls += 1
        if binding_calls != 1:
            raise ValueError("question binding should occur exactly once per free generation")
        bound, alignment, salience, binding_audit = _target_conditioned_bind(
            model, bind_tape, state, allowed
        )
        del alignment, salience
        return bound, None, None

    model.bind_question = masked_bind
    try:
        trace, decoder_audit = n127._masked_free_trace(model, tape, session, data.question, item)
    finally:
        if "bind_question" in model.__dict__:
            del model.bind_question
    if binding_calls != 1:
        raise ValueError(f"expected one target-masked binding call, got {binding_calls}")
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
    audit = {
        **mask_details,
        **binding_audit,
        "question_binding_calls": binding_calls,
        "decoder_attention": decoder_audit,
    }
    return row, audit, {"question": data.question, "tape_positions": len(tape.positions)}


def _score_endpoint(
    bound: dict,
    face_label: str,
    seed: int,
    arm: str,
    items: list[dict],
    budget: e11.CpuDeadline,
) -> tuple[list[dict], int, bool, str | None]:
    rows = []
    queries = 0
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
            session = f"n1-plan128-{face_label}-{seed}-{arm}-{item['id']}"
            if session in model._observations:
                raise ValueError("N1-128 scoring session collides with frozen history")
            queries += 1
            try:
                row, audit, _ = _measure(model, item, arm, session)
                native = bound["native_rows"][face_label][seed, arm, item["id"]]
                pruned = bound["pruned_rows"][face_label][seed, arm, item["id"]]
                decoder_only = bound["question_rows"][face_label, seed, arm, item["id"]]
                rows.append(
                    {
                        **row,
                        "face": face_label,
                        "seed": seed,
                        "arm": arm,
                        "intervention": "oracle_question_binding_and_decoder_source_masks_full_context",
                        "native_full_context_free_exact": native["free_exact"],
                        "native_full_context_output_hex": native["output_hex"],
                        "source_pruned_free_exact": pruned["free_exact"],
                        "source_pruned_output_hex": pruned["output_hex"],
                        "decoder_mask_free_exact": decoder_only["free_exact"],
                        "decoder_mask_output_hex": decoder_only["output_hex"],
                        "decoder_mask_to_binding_mask_gain": row["free_exact"]
                        and not decoder_only["free_exact"],
                        "decoder_mask_to_binding_mask_loss": decoder_only["free_exact"]
                        and not row["free_exact"],
                        "source_pruned_to_binding_mask_gain": row["free_exact"]
                        and not pruned["free_exact"],
                        "source_pruned_to_binding_mask_loss": pruned["free_exact"]
                        and not row["free_exact"],
                        "attention_audit": audit,
                    }
                )
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
    return rows, queries, loaded, error


def _summaries(rows: list[dict]) -> dict:
    summaries = {}
    for face_label in n127.FACE_SOURCES:
        summaries[face_label] = {}
        for arm in ARMS:
            selected = [row for row in rows if row["face"] == face_label and row["arm"] == arm]
            summaries[face_label][arm] = {
                "rows": len(selected),
                "native_full_context_exact": sum(
                    row["native_full_context_free_exact"] for row in selected
                ),
                "decoder_mask_exact": sum(row["decoder_mask_free_exact"] for row in selected),
                "question_binding_mask_exact": sum(row["free_exact"] for row in selected),
                "source_pruned_exact": sum(row["source_pruned_free_exact"] for row in selected),
                "binding_mask_gains_vs_decoder_mask": sum(
                    row["decoder_mask_to_binding_mask_gain"] for row in selected
                ),
                "binding_mask_losses_vs_decoder_mask": sum(
                    row["decoder_mask_to_binding_mask_loss"] for row in selected
                ),
                "binding_mask_losses_vs_source_pruned": sum(
                    row["source_pruned_to_binding_mask_loss"] for row in selected
                ),
                "exact_by_kind": {
                    kind: sum(row["kind"] == kind and row["free_exact"] for row in selected)
                    for kind in KINDS
                },
                "question_binding_calls": sum(
                    row["attention_audit"]["question_binding_calls"] for row in selected
                ),
                "masked_decoder_softmax_calls": sum(
                    row["attention_audit"]["decoder_attention"]["masked_content_softmax_calls"]
                    for row in selected
                ),
            }
    return summaries


def run() -> dict:
    if ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("PLAN-N1-128 query attempt is one-shot")
    stage_started = time.process_time()
    readiness = _verify_readiness()
    bound = _source_set()
    elapsed = time.process_time() - stage_started
    remaining = CPU_LIMIT_SECONDS - float(readiness["cpu_seconds"]) - elapsed
    if remaining <= 0:
        raise TimeoutError("N1-128 cumulative CPU budget exhausted before queries")
    budget = e11.CpuDeadline(remaining)
    torch.set_num_threads(1)
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan128-full-context-question-binding-mask-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": readiness["resolved_readiness_sha256"],
            "checkpoint_hashes": readiness["checkpoint_hashes"],
            "planned_queries": 180,
            "intervention": "oracle_question_binding_plus_decoder_source_mask_full_context",
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    rows = []
    model_queries = 0
    model_loads = 0
    errors = []
    try:
        for face_label, (_, _, items) in bound["face_data"].items():
            for seed in SEEDS:
                for arm in ARMS:
                    budget.guard(reserve=5)
                    endpoint_rows, queries, loaded, error = _score_endpoint(
                        bound, face_label, seed, arm, items, budget
                    )
                    rows.extend(endpoint_rows)
                    model_queries += queries
                    model_loads += int(loaded)
                    if error:
                        errors.append(f"{face_label}/{seed}/{arm}: {error}")
                        raise RuntimeError("endpoint stopped; preserve evidence and do not retry")
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
    binding_calls = sum(row["attention_audit"]["question_binding_calls"] for row in rows)
    decoder_steps = sum(
        row["attention_audit"]["decoder_attention"]["checked_decoder_steps"] for row in rows
    )
    decoder_softmax_calls = sum(
        row["attention_audit"]["decoder_attention"]["masked_content_softmax_calls"] for row in rows
    )
    complete = (
        not errors
        and len(rows) == 180
        and model_queries == 180
        and model_loads == 12
        and binding_calls == 180
        and decoder_softmax_calls == 2 * decoder_steps
        and unchanged
        and cumulative <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan128-full-context-question-binding-mask-result-v1",
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
        "question_binding_calls": binding_calls,
        "masked_decoder_softmax_calls": decoder_softmax_calls,
        "checked_decoder_steps": decoder_steps,
        "nonzero_alignment_outside_target": 0,
        "nonzero_decoder_attention_outside_target": 0,
        "nonzero_address_outside_target": 0,
        "model_loads": model_loads,
        "model_queries": model_queries,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "unknown_queries": 0,
        "cpu_seconds": round(stage_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "oracle_intervention_is_capability": False,
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
