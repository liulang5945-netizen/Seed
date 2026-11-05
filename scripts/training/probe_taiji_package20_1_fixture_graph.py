"""20.1: rule fixture, two reference graphs, freeze tables, leak probes, dry forward.

Zero training, zero optimizer steps.  The fixture is generated on the fly; the
truth block never reaches a graph.  Both assemblies are reference-only
(``initialized_untrained``): they carry the parent-bundle pointer, the identity
and the freeze tables, not a second copy of the weights.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.rule_learning_20_1_identity import (  # noqa: E402
    graph_identity,
    sha256_file,
)
from taiji.rule_learning_20_1_net import (  # noqa: E402
    ANSWER_BYTES,
    DECISION_POSITIONS,
    TEMPLATE_BYTES,
    RuleLearningGraph,
    copy_subject,
    state_digest,
)
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

REPORT = ROOT / "reports/taiji_package20_1_rule_learning_fixture_graph_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_1_rule_learning_fixture_graph_20261105.json"
PROGRESS = ROOT / "reports/taiji_package20_1_rule_learning_fixture_graph_20261105_progress.jsonl"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"
OUT_CF1 = ROOT / "output/taiji_package20_1_cf1_initialized_untrained.pt"
OUT_BEH = ROOT / "output/taiji_package20_1_b_eh_initialized_untrained.pt"

CPU_CAP = 500.0
CPU_GUARD = 480.0
WALL_CAP = 4500.0
FIXTURE_WORLDS = 4
FORWARD_WORLDS = 2
FORBIDDEN_IDENTIFIERS = {
    "truth",
    "truth_table",
    "solver",
    "solve",
    "xor",
    "labels",
    "label_of",
    "rule_labels",
}
FORBIDDEN_IMPORTS = {"taiji.rule_learning_20_1"}

START_WALL = time.time()
START_CPU = time.process_time()


class BudgetStop(RuntimeError):
    pass


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def guard() -> None:
    cpu, wall = cost()
    if cpu > CPU_GUARD or wall > WALL_CAP:
        raise BudgetStop(f"budget stop at cpu={cpu:.2f} wall={wall:.2f}")


def append_progress(record: dict) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def weights_digest(subject) -> str:
    rows = []
    for name, tensor in sorted(subject.state_dict().items()):
        data = tensor.detach().cpu().numpy().tobytes()
        rows.append((name, str(tensor.dtype), tuple(tensor.shape), sha256_bytes(data)))
    return sha256_bytes(repr(rows).encode("utf-8"))


def static_leak_scan() -> dict:
    """(a) answer chain must not name or import the truth/solver machinery."""
    path = ROOT / "taiji/rule_learning_20_1_net.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.append(node.value)
    forbidden_names = sorted({name for name in names if name in FORBIDDEN_IDENTIFIERS})
    forbidden_imports = sorted({name for name in imports if name in FORBIDDEN_IMPORTS})
    signature = list(inspect.signature(RuleLearningGraph.run_turn).parameters)
    forbidden_signature = sorted(name for name in signature if name in FORBIDDEN_IDENTIFIERS)
    return {
        "module": "taiji/rule_learning_20_1_net.py",
        "imports": sorted(set(imports)),
        "forbidden_names": forbidden_names,
        "forbidden_imports": forbidden_imports,
        "forward_signature": signature,
        "forbidden_signature": forbidden_signature,
        "pass": not forbidden_names and not forbidden_imports and not forbidden_signature,
    }


def legal_vocab(released: list[dict]) -> tuple[int, ...]:
    """Constrained-decode alphabet: template bytes plus released feedback bytes.

    The two decision bytes are not added unless a released line actually carries
    them, so every constrained output byte is traceable to a record or the
    template.
    """
    ids = set(TEMPLATE_BYTES.values())
    for item in released:
        ids.update(item["feedback"].encode("utf-8"))
    return tuple(sorted(ids))


def run_world(graph: RuleLearningGraph, world: dict, *, write_enabled: bool) -> dict:
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    turns: list[dict] = []
    released: list[dict] = []
    restores = 0
    restore_exact = 0
    releases: list[dict] = []
    for turn in response["turns"]:
        guard()
        if turn["turn"] in (11, 35):
            payload = graph.save_state(state)
            restored = graph.restore_state(payload)
            restores += 1
            if state_digest(restored) == state_digest(state):
                restore_exact += 1
            state = restored
        result = graph.run_turn(
            state,
            turn,
            released,
            write_enabled=write_enabled,
            vocab=legal_vocab(released),
            now=int(turn["turn"]),
        )
        state = result["state"]
        turns.append(
            {
                "turn": result["turn"],
                "session": int(turn["session"]),
                "kind": turn["kind"],
                "released_lines": len(released),
                "bytes": list(result["bytes"]),
                "constrained_bytes": list(result["constrained_bytes"]),
                "logits": result["logits"],
                "provenance": result["provenance"],
                "constrained_provenance": result["constrained_provenance"],
            }
        )
        for release in response["releases"]:
            if release["after_turn"] == turn["turn"]:
                state, info = graph.release(
                    state, release, released, write_enabled=write_enabled, now=int(turn["turn"])
                )
                releases.append(info)
                released.extend(
                    {
                        "source_id": item["source_id"],
                        "feedback": item["feedback"],
                        "version": item["version"],
                        "after_turn": turn["turn"],
                    }
                    for item in release["items"]
                )
    return {
        "turns": turns,
        "releases": releases,
        "restores": restores,
        "restore_exact": restore_exact,
        "final_state_sha": state_digest(state),
        "journal_entries": len(state.journal),
        "applied_receipts": sum(1 for entry in state.journal if entry["applied"]),
        "released_lines": len(released),
    }


def compare_runs(left: dict, right: dict) -> dict:
    max_delta = 0.0
    identical_bytes = 0
    compared_bytes = 0
    length_mismatch = 0
    for first, second in zip(left["turns"], right["turns"], strict=True):
        if first["bytes"] != second["bytes"]:
            length_mismatch += 1
        for a, b in zip(first["logits"], second["logits"], strict=True):
            max_delta = max(max_delta, float((a - b).abs().max()))
        for a, b in zip(first["bytes"], second["bytes"], strict=True):
            compared_bytes += 1
            if a == b:
                identical_bytes += 1
    return {
        "turns_compared": len(left["turns"]),
        "bytes_compared": compared_bytes,
        "byte_identical": identical_bytes,
        "max_abs_logit_delta": max_delta,
        "byte_mismatch_turns": length_mismatch,
    }


def provenance_summary(run: dict) -> dict:
    counts: dict[str, int] = {"feedback": 0, "template": 0, "generate": 0, "truth": 0}
    decision: dict[str, int] = {"feedback": 0, "template": 0, "generate": 0, "truth": 0}
    covered = [turn for turn in run["turns"] if turn["released_lines"] > 0]
    for turn in covered:
        for position, item in enumerate(turn["constrained_provenance"]):
            counts[item["source"]] = counts.get(item["source"], 0) + 1
            if position in DECISION_POSITIONS:
                decision[item["source"]] = decision.get(item["source"], 0) + 1
    total = sum(counts.values())
    return {
        "turns_covered": len(covered),
        "bytes": total,
        "sources": counts,
        "decision_sources": decision,
        "truth_attribution": counts.get("truth", 0),
        "every_byte_attributed": total == ANSWER_BYTES * len(covered),
    }


def assembly_payload(graph: RuleLearningGraph, identity: dict, label: str) -> dict:
    return {
        "schema": "taiji-package20-1-initialized-state-v1",
        "label": label,
        "arm": graph.arm,
        "fast_enabled": bool(graph.fast_enabled),
        "candidate_sha_v2": identity["candidate_sha_v2"],
        "parent_bundle": str(BUNDLE.relative_to(ROOT)),
        "parent_bundle_sha256": sha256_file(BUNDLE),
        "parent_parameter_sha256": weights_digest(graph.subject),
        "parameters_held_here": 0,
        "initialized_untrained": True,
        "state_schema": identity["state_schema"],
        "parameter_table": graph.parameter_table(),
        "state_table": graph.state_table(),
        "group_counts": graph.group_counts(),
    }


def main() -> int:
    worlds = [fixture.build_world(fixture.SEED, index) for index in range(FIXTURE_WORLDS)]
    turns = [turn for world in worlds for turn in world["response"]["turns"]]
    sessions = [session for world in worlds for session in world["response"]["sessions"]]
    correction = [turn for turn in turns if turn["correction_surface"]]
    counterexamples = [
        item for world in worlds for item in world["scoring"]["feedback_counterexamples"]
    ]
    changed = [turn for world in worlds for turn in world["scoring"]["changed_query"]]
    scale = {
        "worlds": len(worlds),
        "turns_per_world": fixture.TOTAL_TURNS,
        "turns_total": len(turns),
        "correction_surface_per_world": fixture.CORRECTION_TURNS,
        "correction_surface_total": len(correction),
        "first_answer_total": sum(1 for turn in turns if turn["first_answer"]),
        "sessions_per_world": len(fixture.SESSION_SIZES),
        "session_sizes": list(fixture.SESSION_SIZES),
        "sessions_total": len(sessions),
        "feedback_batches_per_world": 2,
        "feedback_lines_total": fixture.example_stats(worlds)["total"]["lines"],
        "feedback_counterexamples_total": len(counterexamples),
        "changed_query_total": len(changed),
        "changed_query_per_world": len(changed) / len(worlds),
        "rule_versions_per_world": 2,
        "rules_available": len(fixture.RULES),
    }
    examples = fixture.example_stats(worlds)
    turn_bytes = fixture.turn_byte_stats(worlds)
    records = {
        "worlds": [fixture.record_bytes(world) for world in worlds],
        "max_world_bytes": max(fixture.record_bytes(world) for world in worlds),
        "e_budget_bytes": 4096,
    }
    append_progress({"stage": "fixture", "scale": scale, "cpu": cost()[0], "wall": cost()[1]})

    scan = static_leak_scan()
    append_progress({"stage": "static_scan", "pass": scan["pass"], "cpu": cost()[0]})

    candidate = load_routing_bundle(BUNDLE)[0]
    bundle_gates = torch.load(BUNDLE, map_location="cpu", weights_only=False).get("growth_gates")
    gate = dict(bundle_gates[0]) if bundle_gates else None
    subject = copy_subject(candidate.subject)
    parent_parameters = sum(parameter.numel() for parameter in subject.parameters())
    cf1_subject = copy_subject(subject)
    beh_subject = copy_subject(subject)
    cf1 = RuleLearningGraph(cf1_subject, arm="CF1", growth_gates=gate)
    beh = RuleLearningGraph(beh_subject, arm="B_EH", growth_gates=gate)
    cf1_identity = graph_identity(cf1)
    beh_identity = graph_identity(beh)
    rows_cf1 = {(row["name"], tuple(row["shape"]), row["dtype"]) for row in cf1.parameter_table()}
    rows_beh = {(row["name"], tuple(row["shape"]), row["dtype"]) for row in beh.parameter_table()}
    identity_section = {
        "cf1": cf1_identity,
        "b_eh": beh_identity,
        "isomorphic": rows_cf1 == rows_beh,
        "parameter_delta": cf1_identity["parameter_count"] - beh_identity["parameter_count"],
        "parent_parameter_count": parent_parameters,
        "new_parameters": cf1_identity["parameter_count"] - parent_parameters,
        "shared_init_sha256": weights_digest(subject),
    }
    append_progress(
        {
            "stage": "identity",
            "cf1": cf1_identity["candidate_sha_v2"],
            "b_eh": beh_identity["candidate_sha_v2"],
            "cpu": cost()[0],
        }
    )

    torch.save(assembly_payload(cf1, cf1_identity, "initialized_untrained"), OUT_CF1)
    torch.save(assembly_payload(beh, beh_identity, "initialized_untrained"), OUT_BEH)
    assemblies = {
        "cf1": {
            "path": str(OUT_CF1.relative_to(ROOT)),
            "sha256": sha256_file(OUT_CF1),
            "bytes": OUT_CF1.stat().st_size,
            "label": "initialized_untrained",
        },
        "b_eh": {
            "path": str(OUT_BEH.relative_to(ROOT)),
            "sha256": sha256_file(OUT_BEH),
            "bytes": OUT_BEH.stat().st_size,
            "label": "initialized_untrained",
        },
    }
    append_progress({"stage": "assemblies", "cpu": cost()[0]})

    forward_worlds = worlds[:FORWARD_WORLDS]
    cf1_runs = [run_world(cf1, world, write_enabled=True) for world in forward_worlds]
    transparent = run_world(cf1, forward_worlds[0], write_enabled=False)
    beh_runs = [run_world(beh, world, write_enabled=False) for world in forward_worlds]
    shuffled_world = fixture.shuffled_truth(forward_worlds[0], random.Random(fixture.SEED + 17))
    shuffled_run = run_world(cf1, shuffled_world, write_enabled=True)
    append_progress({"stage": "dry_forward", "cpu": cost()[0], "wall": cost()[1]})

    shapes = sorted({tuple(turn["logits"].shape) for run in cf1_runs for turn in run["turns"]})
    finite = all(
        bool(torch.isfinite(turn["logits"]).all()) for run in cf1_runs for turn in run["turns"]
    )
    parity = compare_runs(transparent, beh_runs[0])
    shuffled = compare_runs(cf1_runs[0], shuffled_run)
    restore = {
        "attempts": sum(run["restores"] for run in cf1_runs),
        "exact": sum(run["restore_exact"] for run in cf1_runs),
    }
    dry = {
        "worlds_forwarded": len(forward_worlds),
        "turns_forwarded": sum(len(run["turns"]) for run in cf1_runs),
        "logit_shape": list(shapes[0]) if len(shapes) == 1 else [list(shape) for shape in shapes],
        "answer_bytes": ANSWER_BYTES,
        "finite_logits": finite,
        "cf1_fast_writes": sum(item["written"] for run in cf1_runs for item in run["releases"]),
        "b_eh_fast_writes": sum(item["written"] for run in beh_runs for item in run["releases"]),
        "cf1_journal": sum(run["journal_entries"] for run in cf1_runs),
        "cf1_applied_receipts": sum(run["applied_receipts"] for run in cf1_runs),
        "released_lines": sum(run["released_lines"] for run in cf1_runs),
        "session_restore": restore,
    }
    probes = {
        "static": scan,
        "shuffle": shuffled,
        "backtrace": provenance_summary(cf1_runs[0]),
        "parity_b_eh_vs_cf1_fast_off": parity,
        "backtrace_second_world": provenance_summary(cf1_runs[1]),
    }
    append_progress({"stage": "probes", "cpu": cost()[0], "wall": cost()[1]})

    gates = {
        "fixture_51_turns": scale["turns_per_world"] == 51,
        "fixture_22_correction": scale["correction_surface_per_world"] == 22,
        "fixture_three_sessions": scale["session_sizes"] == [10, 24, 17],
        "twenty_rules": scale["rules_available"] == 20,
        "counterexamples_at_least_one": counterexamples != [],
        "changed_query_min_four": len(changed) / len(worlds) >= 4,
        "graphs_isomorphic": identity_section["isomorphic"],
        "static_no_truth_or_solver": scan["pass"],
        "shuffle_byte_identical": shuffled["byte_identical"] == shuffled["bytes_compared"],
        "shuffle_logits_identical": shuffled["max_abs_logit_delta"] == 0.0,
        "backtrace_no_truth": probes["backtrace"]["truth_attribution"] == 0
        and probes["backtrace_second_world"]["truth_attribution"] == 0,
        "backtrace_all_legal": probes["backtrace"]["sources"]["generate"] == 0
        and probes["backtrace_second_world"]["sources"]["generate"] == 0,
        "backtrace_every_byte": probes["backtrace"]["every_byte_attributed"]
        and probes["backtrace_second_world"]["every_byte_attributed"],
        "dry_forward_finite": finite,
        "dry_forward_shape_15x257": shapes == [(ANSWER_BYTES, 257)],
        "parity_fast_off_identical": parity["max_abs_logit_delta"] == 0.0
        and parity["byte_identical"] == parity["bytes_compared"],
        "session_restore_exact": restore["exact"] == restore["attempts"]
        and restore["attempts"] > 0,
        "zero_new_parameters": identity_section["new_parameters"] == 0,
        "same_parameter_count": identity_section["parameter_delta"] == 0,
    }
    verdict = "ready" if all(gates.values()) else "blocked"
    cpu, wall = cost()
    over = cpu > CPU_CAP or wall > WALL_CAP
    report = {
        "package": "20_1",
        "date": "2026-11-05",
        "verdict": verdict,
        "fixture": {
            "scale": scale,
            "examples": examples,
            "turn_bytes": turn_bytes,
            "records": records,
            "digest": fixture.fixture_digest(worlds),
        },
        "identity": identity_section,
        "assemblies": assemblies,
        "dry_forward": dry,
        "probes": probes,
        "gates": gates,
        "cost": {
            "cpu_seconds": round(cpu, 2),
            "wall_seconds": round(wall, 2),
            "cpu_cap": CPU_CAP,
            "wall_cap": WALL_CAP,
            "over_budget": over,
            "training_steps": 0,
            "optimizer_steps": 0,
            "confirm_reads": 0,
            "new_parameters": 0,
        },
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "package": "20_1",
            "verdict": verdict,
            "candidate_sha_v2": {
                "cf1": cf1_identity["candidate_sha_v2"],
                "b_eh": beh_identity["candidate_sha_v2"],
            },
            "assemblies": assemblies,
            "gates": gates,
            "cost": report["cost"],
        },
    )
    append_progress({"stage": "report", "verdict": verdict, "cpu": cpu, "wall": wall})
    print(
        json.dumps(
            {"verdict": verdict, "cpu": round(cpu, 2), "wall": round(wall, 2), "gates": gates},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
