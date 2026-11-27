"""FEC-1/W1(a) retest: content-conditioned write keys, zero training.

Executes the W1(a) retest pinned by
plans/manifests/taiji_fec1_w1a_retest_feature_card_20261108.json:
the NEW graph (content keys enabled, at initialization, both arms) under the
UNCHANGED parent criteria.  The disabled graph must first reproduce
graph.run_turn bit-identically (parity gate) proving the additive change is
inert when off.

Zero training, zero optimizer steps, zero confirm reads.  Analysis machinery
is imported verbatim from the committed FEC-1 runner (sha-pinned there).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_f_expressive_capacity_20261108 as fec1  # noqa: E402
import probe_taiji_package20_4c_r2_formal_comparison as r2  # noqa: E402

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.rule_learning_20_1_net import ANSWER_BYTES, RuleState, enable_content_keys  # noqa: E402

CARD = ROOT / "plans/manifests/taiji_fec1_w1a_retest_feature_card_20261108.json"
CARD_SHA256 = "72f5db4befdfccad894bd44b76e7195ad6db81a26f8c640fa44fae435c2fe94f"
REPORT = ROOT / "reports/taiji_fec1_w1a_retest_20261108.json"
EXEC_MANIFEST = ROOT / "plans/manifests/taiji_fec1_w1a_retest_execution_20261108.json"
PROGRESS = ROOT / "reports/taiji_fec1_w1a_retest_20261108_progress.jsonl"
CONTENT_SEED = 20261031
ARMS = ("CF1", "B_EH")

START_CPU = time.process_time()
START_WALL = time.time()
BUDGET = {"cpu_stop": 2000.0, "wall_cap": 3000.0}


class BudgetStop(RuntimeError):
    pass


class EngineeringStop(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return fec1.sha256_file(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_progress(record: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def check_budget() -> None:
    cpu, wall = cost()
    if cpu >= BUDGET["cpu_stop"] or wall >= BUDGET["wall_cap"]:
        raise BudgetStop(f"budget stop at cpu={cpu:.2f} wall={wall:.2f}")


class RetestCapture(fec1.WriteCapture):
    """FEC-1 capture with the POST-injection cross-check.

    With content keys enabled the trace's write_keys are post-injection; the
    expected value is recomputed from the hook raw, the step's goal bytes and
    the registered projection, and must match bitwise."""

    def __init__(self, subject) -> None:
        super().__init__(subject)
        self.step_goal: bytes | None = None

    def expected_keys(self, raws: dict[int, torch.Tensor]) -> torch.Tensor:
        c = self.config
        if self.step_goal is None:
            raise EngineeringStop("retest capture requires the step goal bytes")
        feature = self.subject.content_feature(self.step_goal)
        boost = fec1.unit_ball(self.subject.key_content_proj @ feature)
        keys = []
        for group in range(c.groups):
            raw = raws[group].reshape(c.fast_heads, -1)
            pre = fec1.unit_ball(raw[:, : c.key_width])
            keys.append(fec1.unit_ball(pre + c.content_gain * boost))
        return torch.stack(keys)

    def take_step(self, trace) -> dict[int, torch.Tensor]:
        if len(self.buffer) != self.config.groups:
            raise EngineeringStop(
                f"hook buffer holds {len(self.buffer)} entries, expected {self.config.groups}"
            )
        raws = dict(self.buffer)
        self.buffer.clear()
        if not torch.equal(self.expected_keys(raws), trace["write_keys"]):
            raise EngineeringStop("post-injection key disagrees with recomputation")
        return raws


def observe_capture(capture, living, raw: bytes, goal, records, allowed, now, step_kwargs):
    final = None
    final_raws = None
    for event in capture.subject.encoder.patches(raw):
        capture.step_goal = goal
        out = capture.subject.step(
            living, event=event, goal=goal, records=records, allowed=allowed, now=now, **step_kwargs
        )
        final_raws = capture.take_step(out.trace)
        living = out.state
        final = out
    return living, final, final_raws


def step_capture(capture, living, **kwargs):
    capture.step_goal = kwargs.get("goal")
    out = capture.subject.step(living, **kwargs)
    capture.take_step(out.trace)
    return out


def replay(graph, world: dict, turn_allowed: torch.Tensor, release_allowed: torch.Tensor) -> dict:
    """FEC-1 driver adapted: keys come from the trace (post-injection), values
    and decays from the hooks; the answer loop passes the goal for the boost."""
    subject = graph.subject
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    capture = RetestCapture(subject)
    released: list[dict] = []
    rows: list[dict] = []
    face_events: dict[int, dict] = {}
    release_events: list[dict] = []
    release_writes: list[dict] = []
    face_written: set[int] = set()
    try:
        with torch.no_grad():
            for turn in response["turns"]:
                check_budget()
                number = int(turn["turn"])
                session = int(turn["session"])
                version = int(turn["version"])
                query = turn["query"].encode("utf-8")
                records = graph._records(released, number)
                living = state.living
                fast_before = living.fast.clone()
                living, obs_final, obs_raws = observe_capture(
                    capture, living, query, query, records, turn_allowed, number, graph.step_kwargs
                )
                if bool(turn["correction_surface"]):
                    _, values, decays = capture.parts(obs_raws)
                    face_events[number] = {
                        "keys": obs_final.trace["write_keys"].detach().double().cpu(),
                        "values": values,
                        "decays": decays,
                        "queries": obs_final.trace["fast_queries"].detach().double().cpu(),
                    }
                previous = EOS
                emitted: list[int] = []
                for _position in range(ANSWER_BYTES):
                    event = graph._patch(subject, previous)
                    out = step_capture(
                        capture,
                        living,
                        event=event,
                        goal=query,
                        records=records,
                        previous_byte=previous,
                        allowed=turn_allowed,
                        now=number,
                        **graph.step_kwargs,
                    )
                    choice = int(torch.argmax(out.probabilities[:256]))
                    emitted.append(choice)
                    living = subject.consume_output(out, choice)
                    previous = choice
                entry = {
                    "prediction_id": f"{number}-{state.revision}",
                    "turn": number,
                    "session": session,
                    "version": version,
                    "query_sha256": hashlib.sha256(query).hexdigest(),
                    "model_version": graph.arm,
                    "output": bytes(emitted).hex(),
                    "released": False,
                    "applied": False,
                }
                journal = state.journal + (entry,)
                if not torch.equal(fast_before, living.fast):
                    face_written.add(number)
                for release in response["releases"]:
                    if int(release["after_turn"]) != number:
                        continue
                    recs = graph._records(released, number)
                    written = 0
                    for item in release["items"]:
                        feedback = item["feedback"].encode("utf-8")
                        item_query = item["query"].encode("utf-8")
                        before = living.fast.clone()
                        living, _rel, _raws = observe_capture(
                            capture,
                            living,
                            feedback,
                            item_query,
                            recs,
                            release_allowed,
                            number,
                            graph.step_kwargs,
                        )
                        if not torch.equal(before, living.fast):
                            written += 1
                    applied_turns = {int(item["turn"]) for item in release["items"]}
                    journal = tuple(
                        (
                            dict(
                                old,
                                released=True,
                                applied=True,
                                release_version=int(release["version"]),
                            )
                            if int(old["turn"]) in applied_turns and not old["applied"]
                            else old
                        )
                        for old in journal
                    )
                    release_writes.append(
                        {
                            "after_turn": number,
                            "version": int(release["version"]),
                            "written": written,
                        }
                    )
                    released.extend(
                        {
                            "source_id": item["source_id"],
                            "feedback": item["feedback"],
                            "version": item["version"],
                            "after_turn": number,
                        }
                        for item in release["items"]
                    )
                state = RuleState(living, journal, state.revision)
                gold = world["scoring"]["turns"][str(number)]["expected_action"].encode("utf-8")
                rows.append(
                    {
                        "turn": number,
                        "session": session,
                        "query": turn["query"],
                        "correction_surface": bool(turn["correction_surface"]),
                        "released_lines": len(released),
                        "gold": gold,
                        "fast": state.living.fast.detach().cpu().clone(),
                        "h": state.living.h.detach().cpu().clone(),
                        "nu_ema": state.living.nu_ema.detach().cpu().clone(),
                        "coverage": state.living.coverage.detach().cpu().clone(),
                        "prior": state.living.prior.detach().cpu().clone(),
                        "keys_count": len(state.living.keys),
                        "step": int(state.living.step),
                        "revision": int(state.revision),
                        "journal_len": len(state.journal),
                        "kind": turn["kind"],
                        "version": version,
                    }
                )
    finally:
        capture.remove()
    return {
        "rows": rows,
        "face_events": face_events,
        "release_events": release_events,
        "release_writes": release_writes,
        "face_written_turns": sorted(face_written),
        "final_fast": rows[-1]["fast"].clone(),
        "written_lines": sum(item["written"] for item in release_writes),
    }


def evaluate_instance(graph, world: dict, criteria: dict, card: dict) -> dict:
    groups = graph.subject.config.groups
    heads = graph.subject.config.fast_heads
    off = torch.zeros((groups, heads), dtype=torch.bool)
    on = torch.ones((groups, heads), dtype=torch.bool)
    release_actual = torch.full((groups, heads), bool(graph.fast_enabled), dtype=torch.bool)

    actual = replay(graph, world, off, release_actual)
    actual_rows = [
        row for row in actual["rows"] if row["correction_surface"] and row["released_lines"] > 0
    ]
    if len(actual_rows) != card["probe_face"]["expected_surfaces"]:
        raise EngineeringStop(f"expected 22 actual faces, got {len(actual_rows)}")
    face_turns = {row["turn"] for row in actual_rows}
    timing_coverage = len(set(actual["face_written_turns"]) & face_turns) / len(actual_rows)

    forced = replay(graph, world, on, on)
    forced_rows = [
        row for row in forced["rows"] if row["correction_surface"] and row["released_lines"] > 0
    ]
    if len(forced_rows) != card["probe_face"]["expected_surfaces"]:
        raise EngineeringStop(f"expected 22 forced faces, got {len(forced_rows)}")
    gold = [row["gold"] for row in forced_rows]

    stacks = fec1.stack_fields(forced["face_events"], groups, heads)
    key_result = fec1.key_gates({name: f["keys"] for name, f in stacks.items()}, criteria)
    align = fec1.query_alignment(stacks)
    value_ranks = {name: fec1.effective_rank(f["values"])[0] for name, f in stacks.items()}
    query_ranks = {name: fec1.effective_rank(f["queries"])[0] for name, f in stacks.items()}
    cap = fec1.capacity_of(stacks)

    f_matrix = np.stack([row["fast"].double().reshape(-1).numpy() for row in forced_rows])
    dof = int(np.unique(f_matrix, axis=0).shape[0])
    m_by_stack = forced["final_fast"].double().numpy().reshape(groups, heads, 32, 32)
    stack_readouts = {
        name: fec1.closed_form_stack(
            m_by_stack[int(name[1]), int(name[3])], f["keys"], f["values"], gold
        )
        for name, f in stacks.items()
    }
    best_stack = max(stack_readouts, key=lambda n: stack_readouts[n]["decision_bit_accuracy"])
    closed_form_accuracy = float(stack_readouts[best_stack]["decision_bit_accuracy"])
    best_bits = fec1.per_face_bits(stack_readouts[best_stack], gold)
    controls = fec1.control_probes(forced_rows, f_matrix, criteria)
    margin = closed_form_accuracy - controls["strongest"]

    actual_stacks = fec1.release_stack_fields(actual["release_events"], groups, heads)
    actual_key_result = (
        fec1.key_gates({name: f["keys"] for name, f in actual_stacks.items()}, criteria)
        if actual_stacks
        else {"pass": False, "binding": {}, "per_stack": {}}
    )

    extras = {
        "query_alignment": align["median_all"],
        "query_effective_rank": min(query_ranks.values()),
        "value_effective_rank": min(value_ranks.values()),
        "capacity": cap["capacity"],
        "closed_form_accuracy": closed_form_accuracy,
        "margin": margin,
        "effective_dof": dof,
    }
    boundaries = fec1.boundary_flags(key_result["binding"], extras, criteria["gates"])
    verdict = fec1.checkpoint_verdict(
        key_result["pass"], extras, actual_key_result["pass"], boundaries, dof, len(forced_rows)
    )
    return {
        "verdict": verdict,
        "timing_coverage_actual": timing_coverage,
        "actual_written_lines": actual["written_lines"],
        "forced_turn_write_faces": len(forced["face_written_turns"]),
        "key_gates_forced": key_result,
        "key_gates_actual_pass": actual_key_result.get("pass", False),
        "query_alignment": align,
        "value_effective_ranks": value_ranks,
        "query_effective_ranks": query_ranks,
        "capacity": cap,
        "closed_form": {
            "binding_stack": best_stack,
            "binding_decision_bit_accuracy": closed_form_accuracy,
            "binding_event_id_accuracy": stack_readouts[best_stack]["event_id_accuracy"],
            "per_stack": {
                name: {
                    "decision_bit_accuracy": item["decision_bit_accuracy"],
                    "event_id_accuracy": item["event_id_accuracy"],
                }
                for name, item in stack_readouts.items()
            },
            "bootstrap": fec1.paired_bootstrap(best_bits, fec1.BOOTSTRAP_RESAMPLES),
        },
        "controls": controls,
        "margin": margin,
        "effective_dof": {"distinct_f_vectors": dof},
        "boundary_flags": boundaries,
    }


def main() -> int:
    runner_sha = sha256_file(Path(__file__).resolve())
    try:
        card = json.loads(CARD.read_text(encoding="utf-8"))
        if sha256_file(CARD) != CARD_SHA256:
            raise EngineeringStop("retest feature card sha mismatch")
        if card["parent_criteria_sha256"] != fec1.CRITERIA_SHA256:
            raise EngineeringStop("retest card points at different criteria")
        prereg = json.loads(fec1.PREREG.read_text(encoding="utf-8"))
        if fec1.canonical_sha(prereg["criteria"]) != fec1.CRITERIA_SHA256:
            raise EngineeringStop("parent criteria sha mismatch")
        criteria = prereg["criteria"]
        if card["probe_face"]["seed"] != fixture.SEED or card["probe_face"]["world_index"] != 0:
            raise EngineeringStop("probe face drifted from retest card")
    except EngineeringStop as error:
        append_progress({"stage": "preflight", "ok": False, "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "FEC-1/W1(a)-retest",
                "status": "engineering_stop",
                "verdict": "engineering_stop",
                "reason": str(error),
                "runner_sha256": runner_sha,
                "readings_produced": 0,
            },
        )
        print(json.dumps({"verdict": "engineering_stop", "reason": str(error)}))
        return 0

    append_progress(
        {
            "stage": "preflight",
            "ok": True,
            "card_sha256": CARD_SHA256,
            "runner_sha256": runner_sha,
            "cpu": round(cost()[0], 3),
        }
    )
    world = fixture.build_world(card["probe_face"]["seed"], card["probe_face"]["world_index"])
    parent_payload = torch.load(r2.BUNDLE, map_location="cpu", weights_only=True)
    gates = parent_payload.get("growth_gates")
    gate = dict(gates[0]) if gates else None
    candidate = r2.load_routing_bundle(r2.BUNDLE)[0]

    results: list[dict] = []
    identities: dict[str, str] = {}
    try:
        # ---- parity gate FIRST: disabled graph must be bit-identical ----
        parity_graph = r2.build_graph(candidate, gate, "CF1")
        parity_graph.subject.eval()
        parity = fec1.driver_equivalence(parity_graph, world)
        append_progress({"stage": "parity", "ok": parity["ok"], "cpu": round(cost()[0], 3)})
        if not parity["ok"]:
            raise EngineeringStop(f"disabled-graph parity failed: {parity}")

        for arm in ARMS:
            graph = r2.build_graph(candidate, gate, arm)
            graph.subject.eval()
            enable_content_keys(graph, seed=CONTENT_SEED)
            graph.subject.eval()
            identities[arm] = r2.weights_digest(graph.subject)
            result = evaluate_instance(graph, world, criteria, card)
            result["arm"] = arm
            result["weights_digest"] = identities[arm]
            results.append(result)
            append_progress(
                {
                    "stage": "instance",
                    "arm": arm,
                    "verdict": result["verdict"],
                    "margin": result["margin"],
                    "key_rank": result["key_gates_forced"]["binding"]["key_effective_rank"],
                    "pairwise": result["key_gates_forced"]["binding"][
                        "pairwise_separable_fraction"
                    ],
                    "cpu": round(cost()[0], 3),
                }
            )
    except BudgetStop as error:
        append_progress({"stage": "budget_stop", "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "FEC-1/W1(a)-retest",
                "status": "budget_stop",
                "verdict": "inconclusive",
                "reason": str(error),
                "results": results,
            },
        )
        print(json.dumps({"verdict": "inconclusive", "status": "budget_stop"}))
        return 0
    except EngineeringStop as error:
        append_progress({"stage": "engineering_stop", "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "FEC-1/W1(a)-retest",
                "status": "engineering_stop",
                "verdict": "engineering_stop",
                "reason": str(error),
                "runner_sha256": runner_sha,
                "readings_produced": len(results),
            },
        )
        print(json.dumps({"verdict": "engineering_stop", "reason": str(error)}))
        return 0

    verdicts = [entry["verdict"] for entry in results]
    disagreement = len(set(verdicts)) > 1
    verdict = "inconclusive" if disagreement else verdicts[0]
    margins = [entry["margin"] for entry in results]
    cpu, wall = cost()
    write_json(
        REPORT,
        {
            "schema": "taiji-fec1-w1a-retest-v1",
            "package": "FEC-1/W1(a)-retest",
            "date": "2026-11-08",
            "status": "completed",
            "verdict": verdict,
            "verdict_basis": "identical decision rule as the parent prereg; 2/2 instance agreement required",
            "per_instance_verdicts": verdicts,
            "disagreement": disagreement,
            "binding_margin": float(min(margins)),
            "discipline": {
                "training": 0,
                "optimizer_steps": 0,
                "confirm_reads": 0,
                "checkpoint_writes": 0,
                "criteria_changed": False,
            },
            "provenance": {
                "retest_card_sha256": CARD_SHA256,
                "parent_criteria_sha256": fec1.CRITERIA_SHA256,
                "runner_sha256": runner_sha,
                "content_seed": CONTENT_SEED,
                "weights_digests": identities,
                "parity": {"ok": True, "detail": "disabled graph bit-identical to graph.run_turn"},
            },
            "results": results,
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
            "scope": (
                "Function-class retest on the NEW graph at initialization only; "
                "expressive_capacity_present is an architecture-requirement input, "
                "not training admission, not a capability claim."
            ),
        },
    )
    write_json(
        EXEC_MANIFEST,
        {
            "schema": "taiji-fec1-w1a-retest-execution-manifest-v1",
            "package": "FEC-1/W1(a)-retest",
            "date": "2026-11-08",
            "retest_card": "plans/manifests/taiji_fec1_w1a_retest_feature_card_20261108.json",
            "retest_card_sha256": CARD_SHA256,
            "parent_criteria_sha256": fec1.CRITERIA_SHA256,
            "report": "reports/taiji_fec1_w1a_retest_20261108.json",
            "report_sha256": sha256_file(REPORT),
            "runner": "scripts/training/probe_taiji_fec1_w1a_retest_20261108.py",
            "runner_sha256": runner_sha,
            "submitted_runner_sha256": runner_sha,
            "verdict": verdict,
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
        },
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "per_instance": verdicts,
                "margins": [round(m, 4) for m in margins],
                "cpu": round(cpu, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
