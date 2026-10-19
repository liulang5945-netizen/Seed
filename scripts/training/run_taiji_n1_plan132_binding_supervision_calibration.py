"""PLAN-N1-132 train-only local-alignment supervision calibration.

The runner separates zero-query readiness, zero-copy checkpoint recovery,
and the one-shot training stage. It never evaluates dev or holdout data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import eval_taiji_n1_v11_after_n120 as e11  # noqa: E402
from scripts.training import (
    run_taiji_n1_expanded_normalized_rwl_length_balanced as n120,
)  # noqa: E402
from taiji.native_local_relation_evidence_candidate import (  # noqa: E402
    LocalRelationEvidenceCandidate,
)

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-132_trainable_question_binding_supervision_calibration_draft_20261019.md"
)
PLAN_SHA256 = "4177e1fa9da00ef1911a3c23f8598a67efd2e5fd5e54b2d4bd96b02fd60dd10f"
N131_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-131_execution_closeout_20261019.md"
N131_RESULT = ROOT / "reports/taiji_n1_transition_operator_error_attribution_20261019.json"
N131_RESULT_SHA256 = "90bf723459b085753a58fb78ade0914d78f4be10e422f6b934a53cdbf98edffc"

READINESS = (
    ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_readiness_20261019.json"
)
SOURCE_BINDING = (
    ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_source_binding_20261019.json"
)
READINESS_REVALIDATION = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_readiness_revalidation_20261019.json"
)
SOURCE_BINDING_REVALIDATION = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_source_binding_revalidation_20261019.json"
)
READINESS_REVALIDATION_V2 = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_readiness_revalidation_v2_20261019.json"
)
SOURCE_BINDING_REVALIDATION_V2 = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_source_binding_revalidation_v2_20261019.json"
)
PREFLIGHT = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_20261019.json"
)
PREFLIGHT_REVALIDATION = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_revalidation_20261019.json"
)
PREFLIGHT_ATTEMPT = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_attempt_20261019.json"
)
PREFLIGHT_REPAIR_ATTEMPT = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_repair_attempt_20261019.json"
)
PREFLIGHT_REPAIR2_ATTEMPT = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_repair2_attempt_20261019.json"
)
PREFLIGHT_BINDING = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_binding_20261019.json"
)
PREFLIGHT_REVALIDATION_BINDING = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_revalidation_binding_20261019.json"
)
ATTEMPT = ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_attempt_20261019.json"
PROGRESS = ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_progress_20261019.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_result_20261019.json"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/n1_plan132_binding_supervision_calibration_20261019"

SEEDS = (11, 12, 13)
ARM = "position"
PARENT_UPDATES = 24_576
ADDITIONAL_UPDATES = 9_216
TOTAL_UPDATES = PARENT_UPDATES + ADDITIONAL_UPDATES
BINDING_WEIGHT = 0.25
BINDING_EPSILON = 1e-8
FINAL_LR = 0.00025
CPU_LIMIT_SECONDS = 2_400.0
PREFLIGHT_CPU_LIMIT_SECONDS = 300.0
GROUPS = ("reader", "writer", "local", "context", "relation")
TRAIN_KINDS = ("lookup", "revision", "transition", "unknown")


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


class CpuBudget:
    def __init__(self, limit: float, *, child_cpu: float = 0.0):
        self.start = time.process_time()
        self.limit = limit
        self.children = child_cpu
        self.accounting_complete = True

    def used(self) -> float:
        return time.process_time() - self.start + self.children

    def guard(self, reserve: float = 0.0) -> None:
        if self.used() + reserve >= self.limit:
            raise TimeoutError("PLAN-N1-132 cumulative CPU budget reached")


def _query_subject(question: str) -> str:
    if not question.endswith("?") or " of " not in question:
        raise ValueError(f"unsupported v3 question format: {question!r}")
    subject = question.rsplit(" of ", 1)[1][:-1]
    if not subject:
        raise ValueError("question must identify its target entity")
    return subject


def _target_observation_indices(query: dict) -> list[int] | None:
    kind = query["kind"]
    if kind == "unknown":
        return None
    subject = _query_subject(query["question"])
    selected = []
    for index, row in enumerate(query["observations"]):
        if kind == "lookup" and row.startswith(subject + " uses "):
            selected.append(index)
        elif kind == "revision" and (
            row.startswith(subject + " uses ") or row.startswith(subject + " now uses ")
        ):
            selected.append(index)
        elif kind == "transition" and (
            row.startswith(subject + " faces ") or row.startswith(subject + " turns ")
        ):
            selected.append(index)
        elif kind not in TRAIN_KINDS:
            raise ValueError(f"unsupported v3 training kind: {kind}")
    expected = {"lookup": 1, "revision": 2, "transition": 2}[kind]
    if len(selected) != expected:
        raise ValueError(f"expected {expected} target facts for {query.get('id')}: {selected}")
    return selected


def _target_byte_mask(tape, query: dict) -> torch.Tensor | None:
    selected = _target_observation_indices(query)
    if selected is None:
        return None
    if len(tape.sources) != len(query["observations"]):
        raise ValueError("target-row supervision requires the complete observation tape")
    for index, source in enumerate(tape.sources):
        if source.source != f"source-{index + 1}" or source.content != query["observations"][
            index
        ].encode("utf-8"):
            raise ValueError("target-row supervision saw reordered or modified observations")
    selected_set = set(selected)
    mask = torch.tensor(
        [int(source.removeprefix("source-")) - 1 in selected_set for source, _ in tape.positions],
        dtype=torch.bool,
        device=tape.values.device,
    )
    if not bool(mask.any()):
        raise ValueError("known training query must select at least one fact byte")
    return mask


def _query_specs(item: dict, second: dict | None) -> list[dict]:
    queries = [item] + ([] if second is None else [second])
    seen = set()
    specs = []
    for query in queries:
        if query.get("split") != "train" or query.get("observations") != item["observations"]:
            raise ValueError("paired alignment supervision must remain on the same train tape")
        question = query["question"].encode("utf-8")
        if question in seen:
            raise ValueError("duplicate paired question would double-count the auxiliary loss")
        seen.add(question)
        specs.append({"question": question, "query": query})
    if item["kind"] == "unknown" and second is None:
        raise ValueError("unknown parent requires its known paired lookup for binding supervision")
    return specs


def _training_step(
    model, optimizer, item: dict, second: dict | None, session: str, g: dict
) -> dict:
    if item.get("split") != "train" or item.get("kind") not in TRAIN_KINDS:
        raise ValueError("frozen v3 train item required")
    if second is not None and (
        second.get("split") != "train" or second.get("observations") != item["observations"]
    ):
        raise ValueError("paired query must preserve the same training observations")

    specs = _query_specs(item, second)
    specs_by_question = {spec["question"]: spec["query"] for spec in specs}
    paired = g["paired"]
    core = g["update"].train.runner.core
    paired.observe(model, paired.project(item), session)
    optimizer.zero_grad(set_to_none=True)
    native_bind = model.bind_question
    captured = []

    def capture_native_alignment(tape, state):
        bound_state, alignment, salience = native_bind(tape, state)
        question = state.question_bytes
        query = specs_by_question.get(question)
        if query is None:
            raise ValueError("teacher-forced binding query differs from scheduled train questions")
        if alignment.ndim != 2 or not torch.isfinite(alignment).all():
            raise ValueError("native binding alignment must be finite and two-dimensional")
        if not torch.allclose(
            alignment.sum(-1),
            torch.ones(alignment.shape[0], device=alignment.device),
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("native binding alignment rows must remain normalized")
        target_mask = _target_byte_mask(tape, query)
        captured.append((query["kind"], alignment, target_mask))
        return bound_state, alignment, salience

    model.bind_question = capture_native_alignment
    try:
        losses = core.paired_question_loss(
            model,
            session,
            item["question"].encode("utf-8"),
            item["answer"].encode("utf-8"),
            second_question=None if second is None else second["question"].encode("utf-8"),
            second_target=None if second is None else second["answer"].encode("utf-8"),
            arm=ARM,
        )
    finally:
        if "bind_question" in model.__dict__:
            del model.bind_question
    if len(captured) != len(specs):
        raise ValueError(f"expected {len(specs)} native binding captures, got {len(captured)}")

    binding_terms = []
    target_masses = []
    for kind, alignment, target_mask in captured:
        if kind == "unknown":
            if target_mask is not None:
                raise ValueError("unknown original question cannot have a positive target row")
            continue
        if target_mask is None:
            raise ValueError("answerable query must have target fact rows")
        mass = alignment[:, target_mask].sum(-1)
        if not torch.isfinite(mass).all() or bool((mass <= 0).any()):
            raise ValueError("target alignment mass must be positive and finite")
        binding_terms.append(-torch.log(mass + BINDING_EPSILON).mean())
        target_masses.append(float(mass.detach().mean().item()))
    if not binding_terms:
        raise ValueError("each scheduled update must contain an answerable binding query")
    binding_loss = torch.stack(binding_terms).mean()
    total_loss = losses.total + BINDING_WEIGHT * binding_loss
    if not torch.isfinite(total_loss):
        raise ValueError("nonfinite combined objective")
    total_loss.backward()
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise ValueError("nonfinite N1-132 gradient")

    for group in optimizer.param_groups:
        group["lr"] = FINAL_LR
    _, groups = n120.expanded.bounded_step(model, optimizer, optimizer.step, audit=False)
    return {
        "total": float(total_loss.detach().item()),
        "base_total": float(losses.total.detach().item()),
        "qa": float(losses.qa.detach().item()),
        "observation": float(losses.observation.detach().item()),
        "binding": float(binding_loss.detach().item()),
        "binding_queries": len(binding_terms),
        "captured_queries": len(captured),
        "mean_target_alignment_mass": sum(target_masses) / len(target_masses),
        "qa_sequences": len(specs),
        "groups": groups,
    }


def _synthetic_objective_preflight(g: dict) -> dict:
    torch.set_num_threads(1)
    unknown = {
        "id": "n1-plan132-synthetic-unknown",
        "split": "train",
        "kind": "unknown",
        "entities": ["ghost"],
        "observations": ["alpha uses copper.", "beta uses silver."],
        "question": "material of ghost?",
        "answer": "unknown",
    }
    paired_lookup = {
        "id": "pair-n1-plan132-synthetic-unknown",
        "parent_id": unknown["id"],
        "split": "train",
        "kind": "lookup",
        "entities": ["beta", "ghost", "alpha"],
        "observations": list(unknown["observations"]),
        "question": "material of beta?",
        "answer": "silver",
    }
    revision = {
        "id": "n1-plan132-synthetic-revision",
        "split": "train",
        "kind": "revision",
        "entities": ["alpha", "beta"],
        "observations": ["alpha uses copper.", "alpha now uses silver."],
        "question": "material of alpha?",
        "answer": "silver",
    }
    transition = {
        "id": "n1-plan132-synthetic-transition",
        "split": "train",
        "kind": "transition",
        "entities": ["alpha", "beta"],
        "observations": ["alpha faces north.", "alpha turns right."],
        "question": "direction of alpha?",
        "answer": "east",
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        gradient_model = LocalRelationEvidenceCandidate()
        gradient_model.train()
        g["paired"].observe(
            gradient_model,
            g["paired"].project(transition),
            "n1-plan132-binding-gradient-probe",
        )
        tape = g["paired"].evidence(gradient_model, "n1-plan132-binding-gradient-probe", ARM)
        state = gradient_model.question_state(transition["question"].encode("utf-8"))
        _, alignment, _ = gradient_model.bind_question(tape, state)
        target_mask = _target_byte_mask(tape, transition)
        binding_probe = -torch.log(alignment[:, target_mask].sum(-1) + BINDING_EPSILON).mean()
        binding_probe.backward()
        binding_gradient_norm = float(
            torch.linalg.vector_norm(gradient_model.local_projection.weight.grad).item()
        )
        if not math.isfinite(binding_gradient_norm) or binding_gradient_norm <= 0:
            raise AssertionError("binding supervision must reach local question projection")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(132)
        model = LocalRelationEvidenceCandidate()
        optimizer = torch.optim.Adam(model.parameters(), lr=FINAL_LR)
        start_bundle = e11.scorer.paired.fingerprint(model.bundle(optimizer))
        unknown_metrics = _training_step(
            model, optimizer, unknown, paired_lookup, "n1-plan132-synthetic-unknown", g
        )
        revision_metrics = _training_step(
            model, optimizer, revision, None, "n1-plan132-synthetic-revision", g
        )
        transition_metrics = _training_step(
            model, optimizer, transition, None, "n1-plan132-synthetic-transition", g
        )
        if (
            unknown_metrics["binding_queries"] != 1
            or unknown_metrics["captured_queries"] != 2
            or revision_metrics["binding_queries"] != 1
            or transition_metrics["binding_queries"] != 1
            or len(optimizer.state) != 41
            or any(int(optimizer.state[p]["step"]) != 3 for p in model.parameters())
        ):
            raise AssertionError("synthetic objective/query-mask contract failed")
        bundle = model.bundle(optimizer)
        clone = LocalRelationEvidenceCandidate.restore_bundle(copy.deepcopy(bundle))
        clone_optimizer = torch.optim.Adam(clone.parameters(), lr=FINAL_LR)
        clone_optimizer.load_state_dict(copy.deepcopy(bundle["optimizer"]))
        restored = e11.scorer.paired.fingerprint(clone.bundle(clone_optimizer))
        if restored != e11.scorer.paired.fingerprint(bundle):
            raise AssertionError("synthetic full model/Adam/RNG/history restore differs")
        end_bundle = e11.scorer.paired.fingerprint(model.bundle(optimizer))
        history_count = len(model._observations)
    if start_bundle == end_bundle:
        raise AssertionError("synthetic optimizer did not exercise model updates")
    return {
        "complete": True,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "formal_training_updates": 0,
        "synthetic_optimizer_updates": 3,
        "unknown_original_binding_queries": 0,
        "unknown_paired_lookup_binding_queries": 1,
        "revision_binding_queries": 1,
        "transition_binding_queries": 1,
        "synthetic_full_bundle_restore": True,
        "synthetic_binding_gradient_norm": binding_gradient_norm,
        "synthetic_history_count": history_count,
        "synthetic_optimizer_updates_applied": True,
    }


def _build_context(*, load_inputs: bool = True) -> dict:
    n120.configure()
    _, g, inputs = n120._BASE.make_runner(load_inputs=load_inputs)
    baseline, candidate, revalidation, prequery, restore, face_readiness = e11._read_bound_inputs()
    candidate_paths = e11._final_paths(candidate, cohort="candidate")
    sufficient = {
        int(model["seed"]): model
        for model in revalidation.get("models", [])
        if model.get("arm") == ARM
    }
    if set(sufficient) != set(SEEDS) or not all(
        model.get("training_sufficient") is True for model in sufficient.values()
    ):
        raise ValueError("all three N1-120 position parents must pass the frozen gate")
    if (
        restore.get("independent_full_bundle_restores") != 18
        or restore.get("all_full_bundles_equal") is not True
        or len(candidate_paths) != 9
    ):
        raise ValueError("complete N1-120 bundle restore evidence is required")
    parents = {}
    for seed in SEEDS:
        path, digest = candidate_paths[(seed, ARM)]
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"N1-120 parent checkpoint changed: {path}")
        parents[seed] = (path, digest)
    if load_inputs and (inputs is None or len(inputs.get("items", [])) != 192):
        raise ValueError("exactly 192 train parents are required")
    if load_inputs and any(item.get("split") != "train" for item in inputs["items"]):
        raise ValueError("N1-132 inputs must be training episodes only")
    source_hashes = {}
    for path in n120._BASE.source_paths():
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"N1-120 source path missing: {source}")
        source_hashes[str(source)] = sha256(source)
    for path in (PLAN, N131_CLOSEOUT, N131_RESULT, Path(__file__).resolve()):
        source_hashes[str(Path(path).resolve())] = sha256(Path(path))
    parent_hashes = {str(path.resolve()): digest for path, digest in parents.values()}
    return {
        "g": g,
        "inputs": inputs,
        "candidate": candidate,
        "revalidation": revalidation,
        "prequery": prequery,
        "restore": restore,
        "face_readiness": face_readiness,
        "parents": parents,
        "parent_hashes": parent_hashes,
        "source_hashes": source_hashes,
    }


def _simulate_schedule(g: dict, inputs: dict) -> dict:
    if "length_ledger" not in g:
        raise ValueError("N1-120 candidate scheduler must expose its length ledger")
    g["length_ledger"].clear()
    counts = Counter()
    parent_counts = Counter()
    binding_queries = 0

    def scheduled_parent_id(item: dict) -> str:
        if item.get("parent_id"):
            return item["parent_id"]
        match = re.match(r"(?:raw|hard)-(train-\d+)", item.get("id", ""))
        return match.group(1) if match else item["id"]

    for step in range(ADDITIONAL_UPDATES):
        item, second = g["scheduled"](step, inputs["items"], inputs["curriculum"])
        if item.get("split") != "train":
            raise ValueError("N1-132 schedule emitted a non-train item")
        if item["kind"] == "unknown" and second is None:
            raise ValueError("unknown schedule item lacks its paired lookup view")
        if second is not None and (
            second.get("split") != "train" or second.get("observations") != item["observations"]
        ):
            raise ValueError("paired question must preserve the train observation tape")
        parent_counts[scheduled_parent_id(item)] += 1
        counts[item["kind"]] += 1
        for spec in _query_specs(item, second):
            if _target_observation_indices(spec["query"]) is not None:
                binding_queries += 1
    expected_kinds = Counter({kind: 2304 for kind in TRAIN_KINDS})
    if len(parent_counts) != 192 or set(parent_counts.values()) != {48} or counts != expected_kinds:
        raise ValueError("fixed 9,216-step schedule must cover every parent/kind equally")
    if binding_queries != 11_520:
        raise ValueError(f"fixed schedule binding-query denominator changed: {binding_queries}")
    ledger = g["length_ledger"]
    path_kind_counts = Counter()
    for (path, kind, _length), count in ledger.items():
        path_kind_counts[(path, kind)] += int(count)
    expected_path_kind = {(path, kind): 1152 for path in ("raw", "hard") for kind in TRAIN_KINDS}
    if path_kind_counts != expected_path_kind or sum(ledger.values()) != ADDITIONAL_UPDATES:
        raise ValueError("9,216-step raw/hard per-kind schedule counts differ")
    return {
        "steps": ADDITIONAL_UPDATES,
        "parents": len(parent_counts),
        "updates_per_parent": 48,
        "updates_by_kind": dict(sorted(counts.items())),
        "answerable_binding_queries": binding_queries,
        "raw_hard_length_ledger": {
            "/".join(str(x) for x in key): int(value) for key, value in sorted(ledger.items())
        },
    }


def source_readiness() -> dict:
    if any(
        path.exists()
        for path in (
            READINESS,
            SOURCE_BINDING,
            PREFLIGHT,
            PREFLIGHT_ATTEMPT,
            PREFLIGHT_BINDING,
            ATTEMPT,
            PROGRESS,
            RESULT,
            CHECKPOINT_ROOT,
        )
    ):
        raise FileExistsError("N1-132 outputs/checkpoint root are write-once")
    started = time.process_time()
    bound = _build_context(load_inputs=True)
    schedule = _simulate_schedule(bound["g"], bound["inputs"])
    synthetic = _synthetic_objective_preflight(bound["g"])
    unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in {**bound["source_hashes"], **bound["parent_hashes"]}.items()
    )
    cpu = time.process_time() - started
    complete = unchanged and cpu <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan132-binding-supervision-readiness-v1",
        "status": "ready_for_checkpoint_preflight" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "source_hashes": bound["source_hashes"],
        "parent_checkpoint_hashes": bound["parent_hashes"],
        "parent_checkpoint_count": len(bound["parent_hashes"]),
        "schedule": schedule,
        "synthetic_objective_preflight": synthetic,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "source_hashes_unchanged": unchanged,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "formal_training_updates": 0,
        "dev_queries": 0,
        "v13_queries": 0,
        "v14_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
    }
    write_new(READINESS, report)
    binding = {
        "schema": "n1-plan132-binding-supervision-source-binding-v1",
        "status": "complete" if complete else "stopped",
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_hashes": bound["source_hashes"],
        "parent_checkpoint_hashes": bound["parent_hashes"],
    }
    write_new(SOURCE_BINDING, binding)
    return report


def source_revalidation() -> dict:
    """Rebind sources after zero-load preflight code-path failures; never retry training."""
    if READINESS_REVALIDATION_V2.exists() or SOURCE_BINDING_REVALIDATION_V2.exists():
        raise FileExistsError("N1-132 source revalidation v2 is write-once")
    if not READINESS.is_file() or not SOURCE_BINDING.is_file() or not PREFLIGHT_ATTEMPT.is_file():
        raise FileNotFoundError("initial readiness, binding, and failed preflight receipt required")
    if not CHECKPOINT_ROOT.is_dir():
        raise FileExistsError("source revalidation requires the failed preflight and empty root")
    failed_preflight = None
    if PREFLIGHT.exists() or PREFLIGHT_BINDING.exists():
        if not PREFLIGHT.is_file() or not PREFLIGHT_BINDING.is_file():
            raise FileExistsError("failed preflight report and binding must both be preserved")
        failed_preflight = read_json(PREFLIGHT)
        failed_binding = read_json(PREFLIGHT_BINDING)
        if (
            failed_preflight.get("complete") is not False
            or failed_preflight.get("status") != "stopped"
            or failed_preflight.get("model_loads") != 0
            or failed_preflight.get("checkpoint_loads") != 0
            or failed_preflight.get("parameter_updates") != 0
            or failed_preflight.get("zero_checkpoints") != []
            or failed_preflight.get("source_hashes_unchanged") is not True
            or failed_binding.get("preflight_sha256") != sha256(PREFLIGHT)
            or failed_binding.get("status") != "stopped"
        ):
            raise ValueError("failed preflight must be bound and stopped before checkpoint loads")

    attempt_paths = [
        path for path in (PREFLIGHT_ATTEMPT, PREFLIGHT_REPAIR_ATTEMPT) if path.is_file()
    ]
    for attempt_path in attempt_paths:
        attempt = read_json(attempt_path)
        if (
            attempt.get("status") != "started"
            or attempt.get("model_loads") != 0
            or attempt.get("parameter_updates") != 0
        ):
            raise ValueError(
                f"preflight must have stopped before loading or updating: {attempt_path}"
            )
    if any(CHECKPOINT_ROOT.iterdir()):
        raise ValueError("all failed preflight attempts must have left the checkpoint root empty")

    parent_readiness_path = (
        READINESS_REVALIDATION if READINESS_REVALIDATION.is_file() else READINESS
    )
    parent_binding_path = (
        SOURCE_BINDING_REVALIDATION if SOURCE_BINDING_REVALIDATION.is_file() else SOURCE_BINDING
    )
    started = time.process_time()
    bound = _build_context(load_inputs=False)
    previous = read_json(parent_readiness_path)
    previous_binding = read_json(parent_binding_path)
    expected_binding_key = (
        "readiness_sha256"
        if parent_readiness_path == READINESS
        else "readiness_revalidation_sha256"
    )
    if (
        previous.get("plan_sha256") != PLAN_SHA256
        or previous_binding.get(expected_binding_key) != sha256(parent_readiness_path)
        or previous_binding.get("source_hashes") != previous.get("source_hashes")
        or previous_binding.get("parent_checkpoint_hashes")
        != previous.get("parent_checkpoint_hashes")
    ):
        raise ValueError("previous N1-132 zero-query receipts are inconsistent")

    runner_key = str(Path(__file__).resolve())
    previous_sources = previous["source_hashes"]
    current_sources = dict(bound["source_hashes"])
    current_sources.update(bound["parent_hashes"])
    current_sources[str(READINESS.resolve())] = sha256(READINESS)
    current_sources[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    current_sources[str(parent_readiness_path.resolve())] = sha256(parent_readiness_path)
    current_sources[str(parent_binding_path.resolve())] = sha256(parent_binding_path)
    for attempt_path in attempt_paths:
        current_sources[str(attempt_path.resolve())] = sha256(attempt_path)
    if failed_preflight is not None:
        current_sources[str(PREFLIGHT.resolve())] = sha256(PREFLIGHT)
        current_sources[str(PREFLIGHT_BINDING.resolve())] = sha256(PREFLIGHT_BINDING)
        for receipt in failed_binding.get("attempt_receipts", []):
            receipt_path = Path(receipt["path"])
            if not receipt_path.is_file() or sha256(receipt_path) != receipt["sha256"]:
                raise ValueError(f"failed preflight attempt receipt changed: {receipt_path}")
            current_sources[str(receipt_path.resolve())] = receipt["sha256"]
    stale = [
        path
        for path, digest in previous_sources.items()
        if path != runner_key and current_sources.get(path) != digest
    ]
    missing = [
        path for path in previous_sources if path != runner_key and path not in current_sources
    ]
    if stale or missing or previous.get("parent_checkpoint_hashes") != bound["parent_hashes"]:
        raise ValueError(f"non-runner source or parent checkpoint changed: {(stale + missing)[:4]}")
    if failed_preflight is not None:
        failed_sources = failed_preflight.get("source_hashes", {})
        stale_failed = [
            path
            for path, digest in failed_sources.items()
            if path != runner_key and current_sources.get(path) != digest
        ]
        if stale_failed:
            raise ValueError(
                f"non-runner source changed after failed preflight: {stale_failed[:4]}"
            )

    unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in {**current_sources, **bound["parent_hashes"]}.items()
    )
    cpu = time.process_time() - started
    recorded_failed_preflight_cpu = (
        float(failed_preflight.get("cpu_seconds", 0.0)) if failed_preflight is not None else 0.0
    )
    failed_attempts_without_final_receipt = max(
        0,
        len(attempt_paths) - len((failed_preflight or {}).get("attempt_receipts", [])),
    )
    unmetered_attempt_reserve = 10.0 * failed_attempts_without_final_receipt
    cumulative = (
        float(previous.get("cumulative_readiness_cpu_seconds", previous["cpu_seconds"]))
        + recorded_failed_preflight_cpu
        + unmetered_attempt_reserve
        + cpu
    )
    complete = unchanged and cumulative <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan132-binding-supervision-readiness-revalidation-v2",
        "status": "ready_for_checkpoint_preflight" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "parent_readiness_path": str(parent_readiness_path.resolve()),
        "parent_readiness_sha256": sha256(parent_readiness_path),
        "parent_source_binding_path": str(parent_binding_path.resolve()),
        "parent_source_binding_sha256": sha256(parent_binding_path),
        "superseded_preflight_attempts": [
            {"path": str(path.resolve()), "sha256": sha256(path), "model_loads": 0}
            for path in attempt_paths
        ],
        "source_hashes": current_sources,
        "parent_checkpoint_hashes": bound["parent_hashes"],
        "checkpoint_hash_count": len(bound["parent_hashes"]),
        "schedule": previous["schedule"],
        "synthetic_objective_preflight": previous["synthetic_objective_preflight"],
        "previous_runner_sha256": previous_sources[runner_key],
        "current_runner_sha256": sha256(Path(__file__).resolve()),
        "only_runner_source_changed": True,
        "failed_preflight_checkpoint_files": 0,
        "recorded_failed_preflight_cpu_seconds": recorded_failed_preflight_cpu,
        "unmetered_attempt_cpu_reserve_seconds": unmetered_attempt_reserve,
        "failed_preflight_report_sha256": (None if failed_preflight is None else sha256(PREFLIGHT)),
        "cpu_seconds": round(cpu, 6),
        "cumulative_readiness_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "formal_training_updates": 0,
        "dev_queries": 0,
        "v13_queries": 0,
        "v14_queries": 0,
        "confirmation_queries": 0,
        "source_hashes_unchanged": unchanged,
        "can_promote": False,
    }
    write_new(READINESS_REVALIDATION_V2, report)
    write_new(
        SOURCE_BINDING_REVALIDATION_V2,
        {
            "schema": "n1-plan132-binding-supervision-source-binding-revalidation-v2",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "readiness_revalidation_sha256": sha256(READINESS_REVALIDATION_V2),
            "source_hashes": current_sources,
            "parent_checkpoint_hashes": bound["parent_hashes"],
        },
    )
    return report


def _verify_readiness() -> tuple[dict, dict]:
    if READINESS_REVALIDATION_V2.is_file():
        readiness_path = READINESS_REVALIDATION_V2
        binding_path = SOURCE_BINDING_REVALIDATION_V2
    elif READINESS_REVALIDATION.is_file():
        readiness_path = READINESS_REVALIDATION
        binding_path = SOURCE_BINDING_REVALIDATION
    else:
        readiness_path = READINESS
        binding_path = SOURCE_BINDING
    revalidated = readiness_path != READINESS
    if not readiness_path.is_file() or not binding_path.is_file():
        raise FileNotFoundError("N1-132 zero-query readiness and source binding required")
    readiness = read_json(readiness_path)
    binding = read_json(binding_path)
    expected_status = "ready_for_checkpoint_preflight"
    expected_cpu = readiness.get(
        "cumulative_readiness_cpu_seconds",
        readiness.get("cpu_seconds", math.inf),
    )
    if (
        readiness.get("complete") is not True
        or readiness.get("status") != expected_status
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("model_loads") != 0
        or readiness.get("checkpoint_loads") != 0
        or readiness.get("formal_training_updates") != 0
        or readiness.get("schedule", {}).get("steps") != ADDITIONAL_UPDATES
        or readiness.get("schedule", {}).get("answerable_binding_queries") != 11_520
        or readiness.get("synthetic_objective_preflight", {}).get("complete") is not True
        or binding.get("status") != "complete"
        or binding.get("plan_sha256") != PLAN_SHA256
        or binding.get("readiness_sha256" if not revalidated else "readiness_revalidation_sha256")
        != sha256(readiness_path)
        or binding.get("source_hashes") != readiness.get("source_hashes")
        or binding.get("parent_checkpoint_hashes") != readiness.get("parent_checkpoint_hashes")
        or float(expected_cpu) > CPU_LIMIT_SECONDS
    ):
        raise ValueError("N1-132 readiness or source binding is stale/incomplete")
    for path, digest in {**binding["source_hashes"], **binding["parent_checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != digest:
            raise ValueError(f"N1-132 readiness source changed: {path}")
    readiness["resolved_readiness_path"] = str(readiness_path.resolve())
    readiness["resolved_readiness_sha256"] = sha256(readiness_path)
    readiness["resolved_source_binding_path"] = str(binding_path.resolve())
    readiness["resolved_source_binding_sha256"] = sha256(binding_path)
    readiness["cumulative_preflight_cpu_seconds"] = float(expected_cpu)
    return readiness, binding


def _restore_bundle_payload(
    payload: dict,
) -> tuple[LocalRelationEvidenceCandidate, torch.optim.Optimizer]:
    if payload.get("schema") != "n1-plan132-binding-supervision-checkpoint-v1":
        raise ValueError("N1-132 checkpoint schema mismatch")
    if payload.get("plan_sha256") != PLAN_SHA256 or payload.get("arm") != ARM:
        raise ValueError("N1-132 checkpoint plan/arm mismatch")
    parent_path = Path(payload["parent_checkpoint_path"])
    if not parent_path.is_file() or sha256(parent_path) != payload["parent_checkpoint_sha256"]:
        raise ValueError("N1-120 parent checkpoint changed")
    model = LocalRelationEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=FINAL_LR)
    optimizer.load_state_dict(copy.deepcopy(payload["candidate"]["optimizer"]))
    if e11.scorer.paired.fingerprint(model.bundle(optimizer)) != payload["bundle_fingerprint"]:
        raise ValueError("complete N1-132 model/Adam/RNG/history restore differs")
    expected = PARENT_UPDATES + int(payload["additional_updates"])
    if len(optimizer.state) != 41 or any(
        int(optimizer.state[parameter]["step"]) != expected for parameter in model.parameters()
    ):
        raise ValueError("N1-132 all-41-parameter Adam step count differs")
    if len(model._observations) != payload["history_count"]:
        raise ValueError("N1-132 observation history count differs")
    return model, optimizer


def _restore_worker(path: Path) -> dict:
    started = time.process_time()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = _restore_bundle_payload(payload)
    signature = e11.scorer.paired.fingerprint(model.bundle(optimizer))
    return {
        "complete": signature == payload["bundle_fingerprint"],
        "signature": signature,
        "bundle_fingerprint": payload["bundle_fingerprint"],
        "additional_updates": payload["additional_updates"],
        "history_count": len(model._observations),
        "optimizer_parameters": len(optimizer.state),
        "cpu_seconds": round(time.process_time() - started, 6),
    }


def _save_checkpoint(
    model,
    optimizer,
    path: Path,
    *,
    seed: int,
    parent_path: Path,
    parent_hash: str,
    additional_updates: int,
) -> dict:
    if path.exists():
        raise FileExistsError(f"N1-132 checkpoint already exists: {path}")
    candidate = model.bundle(optimizer)
    payload = {
        "schema": "n1-plan132-binding-supervision-checkpoint-v1",
        "plan_sha256": PLAN_SHA256,
        "seed": seed,
        "arm": ARM,
        "parent_checkpoint_path": str(parent_path.resolve()),
        "parent_checkpoint_sha256": parent_hash,
        "parent_updates": PARENT_UPDATES,
        "additional_updates": additional_updates,
        "total_updates": PARENT_UPDATES + additional_updates,
        "phase": (
            "zero"
            if additional_updates == 0
            else "final"
            if additional_updates == ADDITIONAL_UPDATES
            else "partial"
        ),
        "history_count": len(model._observations),
        "objective": {
            "base": "frozen_n1_120_qa_plus_observation_fidelity",
            "binding_loss": "mean_negative_log_target_fact_row_alignment_mass",
            "binding_weight": BINDING_WEIGHT,
            "epsilon": BINDING_EPSILON,
            "learning_rate": FINAL_LR,
            "trust_region": "n1_120_expanded_normalized_coupled_rwl_adam",
        },
        "candidate": candidate,
    }
    payload["bundle_fingerprint"] = e11.scorer.paired.fingerprint(candidate)
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def _independent_restore(path: Path, budget: CpuBudget) -> dict:
    budget.guard(reserve=5)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    response = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=min(60, max(1.0, budget.limit - budget.used() - 1.0)),
        env=env,
    )
    result = json.loads(response.stdout)
    budget.children += float(result["cpu_seconds"])
    budget.guard()
    if result.get("complete") is not True or result.get("signature") != result.get(
        "bundle_fingerprint"
    ):
        raise ValueError(f"independent N1-132 restore differs: {path.name}")
    return {"path": str(path.resolve()), "sha256": sha256(path), **result, "equal": True}


def checkpoint_preflight() -> dict:
    if (
        PREFLIGHT_REVALIDATION.exists()
        or PREFLIGHT_REVALIDATION_BINDING.exists()
        or PREFLIGHT_REPAIR2_ATTEMPT.exists()
    ):
        raise FileExistsError("N1-132 checkpoint preflight is write-once")
    readiness, binding = _verify_readiness()
    prior_preflight = None
    if PREFLIGHT.exists() or PREFLIGHT_BINDING.exists():
        if not PREFLIGHT.is_file() or not PREFLIGHT_BINDING.is_file():
            raise FileExistsError(
                "preserved failed checkpoint preflight report/binding pair is incomplete"
            )
        prior_preflight = read_json(PREFLIGHT)
        prior_binding = read_json(PREFLIGHT_BINDING)
        if (
            prior_preflight.get("complete") is not False
            or prior_preflight.get("model_loads") != 0
            or prior_preflight.get("checkpoint_loads") != 0
            or prior_preflight.get("parameter_updates") != 0
            or prior_preflight.get("zero_checkpoints") != []
            or prior_binding.get("preflight_sha256") != sha256(PREFLIGHT)
        ):
            raise FileExistsError("previous checkpoint preflight is not eligible for repair")
        preflight_path = PREFLIGHT_REVALIDATION
        preflight_binding_path = PREFLIGHT_REVALIDATION_BINDING
    else:
        preflight_path = PREFLIGHT
        preflight_binding_path = PREFLIGHT_BINDING
    superseded_attempts = [
        path for path in (PREFLIGHT_ATTEMPT, PREFLIGHT_REPAIR_ATTEMPT) if path.is_file()
    ]
    if superseded_attempts:
        if not READINESS_REVALIDATION_V2.is_file():
            raise FileExistsError("preserved failed attempts require source revalidation v2")
        for old_path in superseded_attempts:
            old_attempt = read_json(old_path)
            if (
                old_attempt.get("status") != "started"
                or old_attempt.get("model_loads") != 0
                or old_attempt.get("parameter_updates") != 0
            ):
                raise FileExistsError(
                    f"failed preflight is not eligible for continuation: {old_path}"
                )
        if any(CHECKPOINT_ROOT.iterdir()):
            raise FileExistsError("failed preflights must leave checkpoint root empty")
        attempt_path = PREFLIGHT_REPAIR2_ATTEMPT
    else:
        if CHECKPOINT_ROOT.exists():
            raise FileExistsError("checkpoint root exists without a recorded failed attempt")
        attempt_path = PREFLIGHT_ATTEMPT
    started = time.process_time()
    preflight_sources = {**binding["source_hashes"], **binding["parent_checkpoint_hashes"]}
    if prior_preflight is not None:
        preflight_sources[str(PREFLIGHT.resolve())] = sha256(PREFLIGHT)
        preflight_sources[str(PREFLIGHT_BINDING.resolve())] = sha256(PREFLIGHT_BINDING)
    for receipt_path in (
        READINESS_REVALIDATION,
        SOURCE_BINDING_REVALIDATION,
        READINESS_REVALIDATION_V2,
        SOURCE_BINDING_REVALIDATION_V2,
    ):
        if receipt_path.is_file():
            preflight_sources[str(receipt_path.resolve())] = sha256(receipt_path)
    write_new(
        attempt_path,
        {
            "schema": "n1-plan132-checkpoint-preflight-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": readiness["resolved_readiness_sha256"],
            "readiness_path": readiness["resolved_readiness_path"],
            "source_binding_sha256": readiness["resolved_source_binding_sha256"],
            "supersedes_failed_attempts": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in superseded_attempts
            ],
            "parent_checkpoint_hashes": binding["parent_checkpoint_hashes"],
            "model_loads": 0,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    if not CHECKPOINT_ROOT.exists():
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    bound = _build_context(load_inputs=False)
    g = bound["g"]
    budget = CpuBudget(
        min(
            PREFLIGHT_CPU_LIMIT_SECONDS,
            CPU_LIMIT_SECONDS - readiness["cumulative_preflight_cpu_seconds"],
        )
    )
    zeros = []
    errors = []
    try:
        baseline, candidate, _, _, restore, _ = e11._read_bound_inputs()
        if restore.get("all_full_bundles_equal") is not True:
            raise ValueError("N1-120 full-bundle restore receipt must remain valid")
        all_paths = e11._final_paths(candidate, cohort="candidate")
        for seed in SEEDS:
            budget.guard(reserve=5)
            parent_path, parent_hash = all_paths[seed, ARM]
            model, optimizer, before, paired = e11._load_model(
                g, parent_path, parent_hash, seed, ARM, candidate["protocol"]
            )
            if len(optimizer.state) != 41 or len(model._observations) == 0:
                raise ValueError("complete N1-120 position Adam/history bundle required")
            if any(int(optimizer.state[p]["step"]) != PARENT_UPDATES for p in model.parameters()):
                raise ValueError("N1-120 position Adam must be exactly at update24,576")
            zero_path = CHECKPOINT_ROOT / f"s{seed}_{ARM}_zero.pt"
            saved = _save_checkpoint(
                model,
                optimizer,
                zero_path,
                seed=seed,
                parent_path=parent_path,
                parent_hash=parent_hash,
                additional_updates=0,
            )
            if paired.fingerprint(model.bundle(optimizer)) != before:
                raise ValueError("zero-copy save changed the N1-120 parent bundle")
            restored = _independent_restore(zero_path, budget)
            zeros.append(
                {
                    "seed": seed,
                    "arm": ARM,
                    "parent_checkpoint_path": str(parent_path.resolve()),
                    "parent_checkpoint_sha256": parent_hash,
                    "path": str(zero_path.resolve()),
                    "sha256": sha256(zero_path),
                    "bundle_fingerprint": saved["bundle_fingerprint"],
                    "history_count": len(model._observations),
                    "independent_restore": restored,
                    "equal": True,
                }
            )
            del model, optimizer
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in preflight_sources.items()
    )
    cpu = time.process_time() - started + budget.children
    attempt_receipts = []
    for old_path in superseded_attempts:
        attempt_receipts.append(
            {
                "path": str(old_path.resolve()),
                "sha256": sha256(old_path),
                "superseded": True,
            }
        )
    attempt_receipts.append(
        {"path": str(attempt_path.resolve()), "sha256": sha256(attempt_path), "superseded": False}
    )
    complete = (
        not errors
        and len(zeros) == 3
        and all(row["equal"] for row in zeros)
        and unchanged
        and cpu <= PREFLIGHT_CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan132-binding-supervision-checkpoint-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": readiness["resolved_readiness_sha256"],
        "source_binding_sha256": readiness["resolved_source_binding_sha256"],
        "source_hashes": preflight_sources,
        "source_hashes_unchanged": unchanged,
        "attempt_receipts": attempt_receipts,
        "parent_checkpoint_hashes": binding["parent_checkpoint_hashes"],
        "zero_checkpoints": zeros,
        "model_loads": len(zeros),
        "checkpoint_loads": len(zeros) * 2,
        "independent_restores": len(zeros),
        "formal_training_updates": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "v13_queries": 0,
        "v14_queries": 0,
        "confirmation_queries": 0,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": PREFLIGHT_CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
    }
    write_new(preflight_path, report)
    write_new(
        preflight_binding_path,
        {
            "schema": "n1-plan132-checkpoint-preflight-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "preflight_sha256": sha256(preflight_path),
            "attempt_receipts": attempt_receipts,
            "parent_checkpoint_hashes": binding["parent_checkpoint_hashes"],
            "zero_checkpoint_hashes": {row["path"]: row["sha256"] for row in zeros},
        },
    )
    return report


def _verify_preflight() -> tuple[dict, dict, dict]:
    readiness, source_binding = _verify_readiness()
    if PREFLIGHT_REVALIDATION.is_file():
        preflight_path = PREFLIGHT_REVALIDATION
        preflight_binding_path = PREFLIGHT_REVALIDATION_BINDING
    else:
        preflight_path = PREFLIGHT
        preflight_binding_path = PREFLIGHT_BINDING
    if not preflight_path.is_file() or not preflight_binding_path.is_file():
        raise FileNotFoundError("complete N1-132 zero-copy checkpoint preflight required")
    preflight = read_json(preflight_path)
    binding = read_json(preflight_binding_path)
    if (
        preflight.get("complete") is not True
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("source_hashes_unchanged") is not True
        or preflight.get("parameter_updates") != 0
        or preflight.get("independent_restores") != 3
        or binding.get("preflight_sha256") != sha256(preflight_path)
        or binding.get("status") != "complete"
    ):
        raise ValueError("N1-132 zero-copy preflight is stale or incomplete")
    checked = {
        **preflight["source_hashes"],
        **preflight["parent_checkpoint_hashes"],
        **binding["zero_checkpoint_hashes"],
    }
    for receipt in binding.get("attempt_receipts", []):
        checked[receipt["path"]] = receipt["sha256"]
    for path, digest in checked.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != digest:
            raise ValueError(f"N1-132 preflight source/checkpoint changed: {path}")
    return readiness, source_binding, preflight


def _load_n132_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = LocalRelationEvidenceCandidate.restore_bundle(copy.deepcopy(payload["candidate"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=FINAL_LR)
    optimizer.load_state_dict(copy.deepcopy(payload["candidate"]["optimizer"]))
    if e11.scorer.paired.fingerprint(model.bundle(optimizer)) != payload["bundle_fingerprint"]:
        raise ValueError("N1-132 complete-bundle load differs")
    expected = PARENT_UPDATES + int(payload["additional_updates"])
    if len(optimizer.state) != 41 or any(
        int(optimizer.state[parameter]["step"]) != expected for parameter in model.parameters()
    ):
        raise ValueError("N1-132 Adam step count differs")
    if len(model._observations) != payload["history_count"]:
        raise ValueError("N1-132 observation history count differs")
    return model, optimizer, payload


def _evaluate_train_only(g: dict, model, optimizer, seed: int, inputs: dict, budget: CpuBudget):
    before = e11.scorer.paired.fingerprint(model.bundle(optimizer))
    evidence = {"rows": [], "score_queries": 0}
    rows, summary = g["evaluate"](
        model,
        optimizer,
        seed,
        ARM,
        TOTAL_UPDATES,
        inputs["views"],
        budget,
        evidence,
    )
    if e11.scorer.paired.fingerprint(model.bundle(optimizer)) != before:
        raise ValueError("train-only sufficiency scoring changed the model/Adam/RNG/history")
    sufficient = bool(g["calibration"].sufficient(summary))
    return rows, summary, sufficient, evidence["score_queries"]


def run_training() -> dict:
    readiness, _, preflight = _verify_preflight()
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("N1-132 calibration is one-shot; no restart")
    stage_started = time.process_time()
    bound = _build_context(load_inputs=True)
    g, inputs = bound["g"], bound["inputs"]
    active_preflight_path = (
        PREFLIGHT_REVALIDATION if PREFLIGHT_REVALIDATION.is_file() else PREFLIGHT
    )
    active_preflight_binding_path = (
        PREFLIGHT_REVALIDATION_BINDING
        if PREFLIGHT_REVALIDATION_BINDING.is_file()
        else PREFLIGHT_BINDING
    )
    source_hashes = {
        **readiness["source_hashes"],
        **readiness["parent_checkpoint_hashes"],
        str(READINESS.resolve()): sha256(READINESS),
        str(SOURCE_BINDING.resolve()): sha256(SOURCE_BINDING),
        str(active_preflight_path.resolve()): sha256(active_preflight_path),
        str(active_preflight_binding_path.resolve()): sha256(active_preflight_binding_path),
    }
    for receipt_path in (
        READINESS_REVALIDATION,
        SOURCE_BINDING_REVALIDATION,
        READINESS_REVALIDATION_V2,
        SOURCE_BINDING_REVALIDATION_V2,
    ):
        if receipt_path.is_file():
            source_hashes[str(receipt_path.resolve())] = sha256(receipt_path)
    for receipt in preflight.get("attempt_receipts", []):
        source_hashes[receipt["path"]] = receipt["sha256"]
    for path, digest in source_hashes.items():
        if not Path(path).is_file() or sha256(Path(path)) != digest:
            raise ValueError(f"N1-132 bound source changed before training: {path}")
    setup_cpu = time.process_time() - stage_started
    prior_cpu = (
        float(readiness["cumulative_preflight_cpu_seconds"])
        + float(preflight["cpu_seconds"])
        + setup_cpu
    )
    remaining = CPU_LIMIT_SECONDS - prior_cpu
    if remaining <= 0:
        raise TimeoutError("N1-132 cumulative CPU budget exhausted before training")
    budget = CpuBudget(remaining)
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan132-binding-supervision-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "checkpoint_preflight_sha256": sha256(PREFLIGHT),
            "source_hashes": source_hashes,
            "parent_checkpoint_hashes": readiness["parent_checkpoint_hashes"],
            "seed_order": list(SEEDS),
            "arm": ARM,
            "additional_updates_per_seed": ADDITIONAL_UPDATES,
            "parameter_updates": 0,
            "dev_queries": 0,
            "v13_queries": 0,
            "v14_queries": 0,
            "confirmation_queries": 0,
            "can_promote": False,
        },
    )

    records = []
    completed_updates = 0
    train_score_queries = 0
    errors = []
    active = None
    model = optimizer = None
    try:
        with PROGRESS.open("x", encoding="utf-8", newline="\n") as progress:
            for seed in SEEDS:
                budget.guard(reserve=60)
                zero_path = CHECKPOINT_ROOT / f"s{seed}_{ARM}_zero.pt"
                model, optimizer, zero_payload = _load_n132_checkpoint(zero_path)
                active = {
                    "seed": seed,
                    "completed_additional_updates": 0,
                    "parent_checkpoint_path": zero_payload["parent_checkpoint_path"],
                    "parent_checkpoint_sha256": zero_payload["parent_checkpoint_sha256"],
                }
                start_history = len(model._observations)
                if any(
                    int(optimizer.state[p]["step"]) != PARENT_UPDATES for p in model.parameters()
                ):
                    raise ValueError("N1-120 parent Adam state is not at 24,576 updates")
                g["length_ledger"].clear()
                step_losses = []
                for step in range(ADDITIONAL_UPDATES):
                    budget.guard(reserve=30)
                    item, second = g["scheduled"](step, inputs["items"], inputs["curriculum"])
                    metrics = _training_step(
                        model,
                        optimizer,
                        item,
                        second,
                        f"n1-plan132-{seed}-{step}",
                        g,
                    )
                    step_losses.append(metrics)
                    active["completed_additional_updates"] = step + 1
                    completed_updates += 1
                    progress.write(
                        json.dumps(
                            {"seed": seed, "arm": ARM, "step": step, **metrics},
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    if (step + 1) % 64 == 0:
                        progress.flush()
                    if (step + 1) % 512 == 0:
                        progress.flush()
                        print(
                            f"plan132 seed={seed} step={step + 1}/{ADDITIONAL_UPDATES} "
                            f"qa={metrics['qa']:.4f} obs={metrics['observation']:.4f} "
                            f"bind={metrics['binding']:.4f} cpu={budget.used():.1f}",
                            flush=True,
                        )

                total_steps = {int(optimizer.state[p]["step"]) for p in model.parameters()}
                if (
                    total_steps != {TOTAL_UPDATES}
                    or len(model._observations) != start_history + ADDITIONAL_UPDATES
                ):
                    raise ValueError("fixed continuation final Adam/history counts differ")
                ledger = {
                    "/".join(str(part) for part in key): int(value)
                    for key, value in sorted(g["length_ledger"].items())
                }
                final_path = CHECKPOINT_ROOT / f"s{seed}_{ARM}_final.pt"
                final_payload = _save_checkpoint(
                    model,
                    optimizer,
                    final_path,
                    seed=seed,
                    parent_path=Path(zero_payload["parent_checkpoint_path"]),
                    parent_hash=zero_payload["parent_checkpoint_sha256"],
                    additional_updates=ADDITIONAL_UPDATES,
                )
                final_restore = _independent_restore(final_path, budget)
                score_rows, score_summary, sufficient, queries = _evaluate_train_only(
                    g, model, optimizer, seed, inputs, budget
                )
                train_score_queries += queries
                means = {
                    name: sum(row[name] for row in step_losses) / len(step_losses)
                    for name in ("total", "base_total", "qa", "observation", "binding")
                }
                record = {
                    "seed": seed,
                    "arm": ARM,
                    "parent_checkpoint_path": zero_payload["parent_checkpoint_path"],
                    "parent_checkpoint_sha256": zero_payload["parent_checkpoint_sha256"],
                    "zero_checkpoint_path": str(zero_path.resolve()),
                    "zero_checkpoint_sha256": sha256(zero_path),
                    "additional_updates": ADDITIONAL_UPDATES,
                    "total_updates": TOTAL_UPDATES,
                    "initial_history_count": start_history,
                    "final_history_count": len(model._observations),
                    "optimizer_step_values": sorted(total_steps),
                    "length_balance_ledger": ledger,
                    "loss_means": means,
                    "loss_step_count": len(step_losses),
                    "binding_supervised_query_terms": sum(
                        row["binding_queries"] for row in step_losses
                    ),
                    "train_sufficiency": score_summary,
                    "training_sufficient": sufficient,
                    "train_score_queries": queries,
                    "train_score_rows": score_rows,
                    "final_checkpoint_path": str(final_path.resolve()),
                    "final_checkpoint_sha256": sha256(final_path),
                    "final_bundle_fingerprint": final_payload["bundle_fingerprint"],
                    "final_independent_restore": final_restore,
                }
                records.append(record)
                del model, optimizer
                model = optimizer = None
                active = None
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    partial_checkpoint = None
    if active is not None and model is not None and optimizer is not None:
        try:
            partial_path = CHECKPOINT_ROOT / f"s{active['seed']}_{ARM}_partial.pt"
            partial_payload = _save_checkpoint(
                model,
                optimizer,
                partial_path,
                seed=active["seed"],
                parent_path=Path(active["parent_checkpoint_path"]),
                parent_hash=active["parent_checkpoint_sha256"],
                additional_updates=active["completed_additional_updates"],
            )
            partial_checkpoint = {
                "path": str(partial_path.resolve()),
                "sha256": sha256(partial_path),
                "additional_updates": active["completed_additional_updates"],
                "bundle_fingerprint": partial_payload["bundle_fingerprint"],
                "automatic_resume_allowed": False,
            }
        except BaseException as exc:
            errors.append(f"partial checkpoint save {type(exc).__name__}: {exc}")
    source_hashes_unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in source_hashes.items()
    )
    new_checkpoints = {
        str(path.resolve()): sha256(path) for path in CHECKPOINT_ROOT.glob("*.pt") if path.is_file()
    }
    stage_cpu = time.process_time() - stage_started + budget.children
    cumulative_cpu = float(readiness["cpu_seconds"]) + float(preflight["cpu_seconds"]) + stage_cpu
    all_sufficient = len(records) == 3 and all(record["training_sufficient"] for record in records)
    complete = (
        not errors
        and completed_updates == len(SEEDS) * ADDITIONAL_UPDATES
        and train_score_queries == len(SEEDS) * 960
        and len(records) == 3
        and len(new_checkpoints) == 6
        and source_hashes_unchanged
        and cumulative_cpu <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan132-binding-supervision-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_binding_sha256": sha256(SOURCE_BINDING),
        "checkpoint_preflight_sha256": sha256(PREFLIGHT),
        "attempt_sha256": sha256(ATTEMPT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": source_hashes_unchanged,
        "parent_checkpoint_hashes": readiness["parent_checkpoint_hashes"],
        "new_checkpoint_hashes": new_checkpoints,
        "models": records,
        "partial_checkpoint": partial_checkpoint,
        "training_updates": completed_updates,
        "train_score_queries": train_score_queries,
        "dev_queries": 0,
        "v13_queries": 0,
        "v14_queries": 0,
        "confirmation_queries": 0,
        "parameter_updates": completed_updates,
        "optimizer_steps": completed_updates,
        "cpu_seconds": round(stage_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "training_sufficient_all_seeds": all_sufficient,
        "can_promote": False,
        "errors": errors,
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--revalidate-sources", action="store_true")
    modes.add_argument("--checkpoint-preflight", action="store_true")
    modes.add_argument("--execute-approved", action="store_true")
    modes.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker is not None:
        result = _restore_worker(args.restore_worker)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("complete") else 1
    if args.validate_only:
        result = source_readiness()
    elif args.revalidate_sources:
        result = source_revalidation()
    elif args.checkpoint_preflight:
        result = checkpoint_preflight()
    else:
        result = run_training()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "status",
                    "cpu_seconds",
                    "cumulative_cpu_seconds",
                    "training_updates",
                    "train_score_queries",
                    "independent_restores",
                    "training_sufficient_all_seeds",
                    "dev_queries",
                    "v13_queries",
                    "v14_queries",
                    "confirmation_queries",
                    "can_promote",
                    "errors",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
