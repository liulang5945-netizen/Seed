"""FEC-1: zero-training expressive-capacity probe of the fast weight F.

Executes the frozen pre-registration manifest against the four 20.4c_r2 final
trained checkpoints on the 20.2s/21.3 probe face (fixture world seed 20261105,
index 0, 22 correction surfaces).

Three write conditions, one driver:
  actual      - graph semantics as-is (turn writes off, release writes per arm)
  forced      - allowed=torch.ones((groups, fast_heads)) on EVERY subject call
  random_init - the 20.1 initialized graph (no checkpoint load), both masks

The driver replicates graph.run_turn / graph.release at the subject level and is
validated BIT-IDENTICAL to them under the actual mask before any reading is
produced; the ONLY delta under forced is the permission mask.  Write keys and
values are captured with read-only forward hooks on subject.write_heads and
cross-checked bitwise against the step trace.

Zero training, zero optimizer steps, zero new parameters, zero checkpoint
writes, zero confirm reads.  Any integrity mismatch is engineering_stop.
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

import probe_taiji_package20_2_credit_recovery as p2  # noqa: E402
import probe_taiji_package20_2s_f_information as p2s  # noqa: E402
import probe_taiji_package20_4c_r2_formal_comparison as r2  # noqa: E402

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.rule_learning_20_1_net import ANSWER_BYTES, RuleState  # noqa: E402

PREREG = ROOT / "plans/manifests/taiji_f_expressive_capacity_prereg_20261108.json"
FEATURE_CARD = ROOT / "plans/manifests/taiji_f_expressive_capacity_feature_card_20261108.json"
REPORT = ROOT / "reports/taiji_f_expressive_capacity_20261108.json"
EXEC_MANIFEST = ROOT / "plans/manifests/taiji_f_expressive_capacity_execution_20261108.json"
PROGRESS = ROOT / "reports/taiji_f_expressive_capacity_20261108_progress.jsonl"

CRITERIA_SHA256 = "1e0dde222d557b4ea97eee528b9f73a2aec3068a2c462e927465e415fa3b6142"
FEATURE_CARD_SHA256 = "e91252dd6a0d6e9cafdd6c9fc9a8fe25f34ba5a370c8b8ace061c2b8a85ef7c3"
R2_SCRIPT_SHA256 = "112c76ae117255bb5da7d60a46312bf82382430fc3a99ab8d7ce6a1f173d4095"
SHARED_INIT_SHA256 = "c5ebe35b9bf9442383ba0f568f8b7825466ec97f31ee35625182270aee1934c6"

CHECKPOINT_ARM_SEEDS = (
    ("CF1", 20261105),
    ("CF1", 20261106),
    ("B_EH", 20261105),
    ("B_EH", 20261106),
)
QUERY_KEY_DIM = 128
RANDOM_SEED = 20261105
BOOTSTRAP_SEED = 20261107
BOOTSTRAP_RESAMPLES = 2000

START_CPU = time.process_time()
START_WALL = time.time()
BUDGET = {"cpu_soft_cap": 1200.0, "cpu_stop_threshold": 1500.0, "wall_cap": 2250.0}


class BudgetStop(RuntimeError):
    pass


class EngineeringStop(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def checkpoint_sha_registry() -> dict:
    """The prereg manifest points at the 21.1 report cells for the four
    checkpoint shas; build {filename: sha} from that frozen report."""
    report = json.loads(
        (ROOT / "reports/taiji_package21_1_readout_decomposition_20261107.json").read_text(
            encoding="utf-8"
        )
    )
    registry: dict[str, str] = {}
    for cell in report.get("cells", []):
        path = Path(cell["checkpoint"])
        registry[path.name] = cell["checkpoint_sha256"]
    return registry


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
    if cpu >= BUDGET["cpu_stop_threshold"] or wall >= BUDGET["wall_cap"]:
        raise BudgetStop(f"budget stop at cpu={cpu:.2f} wall={wall:.2f}")


def query_key(text: str) -> np.ndarray:
    key = np.zeros(QUERY_KEY_DIM, dtype=np.float64)
    for value in text.encode("utf-8"):
        key[value % QUERY_KEY_DIM] += 1.0
    return key / max(float(np.linalg.norm(key)), 1e-12)


def unit_ball(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + x.square().sum(dim=-1, keepdim=True)).sqrt()


def load_frozen() -> tuple[dict, dict, dict]:
    # Checkpoint integrity is verified here BEFORE r2.load_checkpoint (a
    # committed, sha-pinned helper) deserializes any payload: the trust anchor
    # is the pinned sha256, not the pickle itself.
    if sha256_file(Path(r2.__file__).resolve()) != R2_SCRIPT_SHA256:
        raise EngineeringStop("build_graph provider script sha mismatch")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    card = json.loads(FEATURE_CARD.read_text(encoding="utf-8"))
    if canonical_sha(prereg["criteria"]) != CRITERIA_SHA256:
        raise EngineeringStop("prereg criteria sha mismatch")
    if sha256_file(FEATURE_CARD) != FEATURE_CARD_SHA256:
        raise EngineeringStop("feature card sha mismatch")
    if card["criteria_sha256"] != CRITERIA_SHA256:
        raise EngineeringStop("feature card points at different criteria")
    if r2.ADDRESSED_STORE is not False:
        raise EngineeringStop("addressed store must be disabled")
    if card["probe_face"]["world_index"] != 0 or card["probe_face"]["seed"] != fixture.SEED:
        raise EngineeringStop("probe face drifted from feature card")
    expected = checkpoint_sha_registry()
    for arm, seed in CHECKPOINT_ARM_SEEDS:
        path = r2.CHECKPOINTS / f"{arm}_{seed}_200.pt"
        if not path.exists():
            raise EngineeringStop(f"missing checkpoint {path}")
        if sha256_file(path) != expected[f"{arm}_{seed}_200.pt"]:
            raise EngineeringStop(f"checkpoint sha mismatch {path}")
    return prereg, card, prereg["criteria"]


class WriteCapture:
    """Read-only forward hooks on subject.write_heads; drained once per step."""

    def __init__(self, subject) -> None:
        self.subject = subject
        self.config = subject.config
        self.buffer: list[tuple[int, torch.Tensor]] = []
        self.handles = [
            network.register_forward_hook(self._make(group))
            for group, network in enumerate(subject.write_heads)
        ]

    def _make(self, group: int):
        def hook(_module, _inputs, output):
            self.buffer.append((group, output.detach().clone()))

        return hook

    def take_step(self, trace) -> dict[int, torch.Tensor]:
        """Drain the buffer for one step and cross-check against the trace."""
        if len(self.buffer) != self.config.groups:
            raise EngineeringStop(
                f"hook buffer holds {len(self.buffer)} entries, expected {self.config.groups}"
            )
        raws = dict(self.buffer)
        self.buffer.clear()
        for group in range(self.config.groups):
            raw = raws[group].reshape(self.config.fast_heads, -1)
            key = unit_ball(raw[:, : self.config.key_width])
            if not torch.equal(key, trace["write_keys"][group]):
                raise EngineeringStop("hook key disagrees with trace write_keys")
        return raws

    def parts(
        self, raws: dict[int, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.config
        keys, values, decays = [], [], []
        for group in range(c.groups):
            raw = raws[group].reshape(c.fast_heads, -1)
            key = unit_ball(raw[:, : c.key_width])
            value = torch.sigmoid(
                raw[:, 2 * c.key_width + c.value_width : 2 * c.key_width + 2 * c.value_width]
            ) * unit_ball(torch.tanh(raw[:, c.key_width : c.key_width + c.value_width]))
            decay = torch.sigmoid(
                raw[:, c.key_width + c.value_width : 2 * c.key_width + c.value_width]
            )
            keys.append(key)
            values.append(value)
            decays.append(decay)
        return (
            torch.stack(keys).double().cpu(),
            torch.stack(values).double().cpu(),
            torch.stack(decays).double().cpu(),
        )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def observe_capture(
    capture: WriteCapture, living, raw: bytes, goal, records, allowed, now, step_kwargs
):
    """subject.observe replicated patch-by-patch.

    Returns (state, final step, final step's hook raws); every patch step's
    hooks are drained and cross-checked, so the caller must NOT take again."""
    final = None
    final_raws = None
    for event in capture.subject.encoder.patches(raw):
        out = capture.subject.step(
            living, event=event, goal=goal, records=records, allowed=allowed, now=now, **step_kwargs
        )
        final_raws = capture.take_step(out.trace)
        living = out.state
        final = out
    return living, final, final_raws


def step_capture(capture: WriteCapture, living, **kwargs):
    out = capture.subject.step(living, **kwargs)
    capture.take_step(out.trace)
    return out


def replay(graph, world: dict, turn_allowed: torch.Tensor, release_allowed: torch.Tensor) -> dict:
    with torch.no_grad():
        return _replay_impl(graph, world, turn_allowed, release_allowed)


def _replay_impl(
    graph, world: dict, turn_allowed: torch.Tensor, release_allowed: torch.Tensor
) -> dict:
    """One 51-turn replay with explicit permission masks.

    Replicates graph.run_turn / graph.release (store disabled) at subject level;
    validated bit-identical to them under the actual masks before any reading."""
    subject = graph.subject
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    capture = WriteCapture(subject)
    released: list[dict] = []
    rows: list[dict] = []
    face_events: dict[int, dict] = {}
    release_events: list[dict] = []
    release_writes: list[dict] = []
    face_written: set[int] = set()
    try:
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
                keys, values, decays = capture.parts(obs_raws)
                face_events[number] = {
                    "keys": keys,
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
            # ---- trailing releases (as graph.release, store disabled) ----
            for release in response["releases"]:
                if int(release["after_turn"]) != number:
                    continue
                recs = graph._records(released, number)
                written = 0
                for item in release["items"]:
                    feedback = item["feedback"].encode("utf-8")
                    item_query = item["query"].encode("utf-8")
                    before = living.fast.clone()
                    living, rel_final, rel_raws = observe_capture(
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
                    keys, values, decays = capture.parts(rel_raws)
                    release_events.append(
                        {
                            "after_turn": number,
                            "turn": int(item["turn"]),
                            "keys": keys,
                            "values": values,
                            "decays": decays,
                        }
                    )
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
                    {"after_turn": number, "version": int(release["version"]), "written": written}
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
                    "kind": turn["kind"],
                    "version": version,
                    "query": turn["query"],
                    "correction_surface": bool(turn["correction_surface"]),
                    "released_lines": len(released),
                    "bytes": tuple(emitted),
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


def driver_equivalence(graph, world: dict) -> dict:
    """driver(actual masks) must reproduce graph.run_turn / graph.release
    bit-identically: per-turn emitted bytes, final F, release written counts."""
    reference_rows = []
    state = graph.initial_state(world["response"]["world_id"])
    released: list[dict] = []
    release_counts = []
    for turn in world["response"]["turns"]:
        number = int(turn["turn"])
        result = graph.run_turn(
            state, turn, released, write_enabled=True, vocab=p2.legal_vocab(released), now=number
        )
        state = result["state"]
        reference_rows.append(tuple(int(value) for value in result["bytes"]))
        for release in world["response"]["releases"]:
            if int(release["after_turn"]) != number:
                continue
            state, info = graph.release(state, release, released, write_enabled=True, now=number)
            release_counts.append(int(info["written"]))
            released.extend(
                {
                    "source_id": item["source_id"],
                    "feedback": item["feedback"],
                    "version": item["version"],
                    "after_turn": number,
                }
                for item in release["items"]
            )
    reference_final = state.living.fast.detach().cpu().clone()
    groups = graph.subject.config.groups
    heads = graph.subject.config.fast_heads
    turn_mask = torch.zeros((groups, heads), dtype=torch.bool)
    release_mask = torch.full((groups, heads), bool(graph.fast_enabled), dtype=torch.bool)
    mine = replay(graph, world, turn_mask, release_mask)
    mismatches = [
        index
        for index, (left, right) in enumerate(
            zip(reference_rows, (r["bytes"] for r in mine["rows"]), strict=False)
        )
        if left != right
    ]
    final_delta = float((reference_final - mine["final_fast"]).abs().max())
    counts_match = release_counts == [item["written"] for item in mine["release_writes"]]
    return {
        "ok": bool(not mismatches and final_delta == 0.0 and counts_match),
        "byte_mismatch_turns": mismatches,
        "final_fast_max_abs_delta": final_delta,
        "release_written_counts_match": bool(counts_match),
        "reference_written_total": int(sum(release_counts)),
        "driver_written_total": int(mine["written_lines"]),
    }


def effective_rank(matrix: np.ndarray) -> tuple[int, np.ndarray]:
    if matrix.shape[0] < 2:
        return int(matrix.shape[0]), np.zeros(1)
    singular = np.linalg.svd(matrix, compute_uv=False)
    top = float(singular[0]) if singular[0] > 0 else 1.0
    return int((singular >= 1e-2 * top).sum()), singular


def pairwise_stats(keys: np.ndarray, threshold: float) -> dict:
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    unit = keys / np.maximum(norms, 1e-12)
    gram = unit @ unit.T
    upper = gram[np.triu_indices(len(keys), k=1)]
    return {
        "pairwise_separable_fraction": float((np.abs(upper) <= threshold).mean()),
        "pairwise_cosine_median": float(np.median(upper)),
        "pairwise_cosine_max_abs": float(np.abs(upper).max()),
    }


def key_gates(stack_map: dict, criteria: dict) -> dict:
    gates = criteria["gates"]
    per_stack = {}
    for name, keys in stack_map.items():
        rank, singular = effective_rank(keys)
        finite = singular[singular >= 1e-2 * singular[0]] if len(singular) else singular
        condition = float(finite[0] / finite[-1]) if len(finite) > 1 else float("inf")
        per_stack[name] = {
            "effective_rank": rank,
            "condition_number": condition,
            "key_norm_median": float(np.median(np.linalg.norm(keys, axis=1))),
            **pairwise_stats(keys, gates["key_cosine_threshold"]),
        }
    binding = {
        "key_effective_rank": min(item["effective_rank"] for item in per_stack.values()),
        "pairwise_separable_fraction": min(
            item["pairwise_separable_fraction"] for item in per_stack.values()
        ),
        "key_norm_median": min(item["key_norm_median"] for item in per_stack.values()),
        "condition_number": min(item["condition_number"] for item in per_stack.values()),
    }
    return {
        "per_stack": per_stack,
        "binding": binding,
        "pass": bool(
            binding["key_effective_rank"] >= gates["key_effective_rank_min"]
            and binding["pairwise_separable_fraction"]
            >= gates["key_pairwise_separable_fraction_min"]
            and binding["key_norm_median"] >= gates["key_norm_median_min"]
            and binding["condition_number"] <= gates["condition_number_max"]
        ),
    }


def closed_form_stack(
    m: np.ndarray, keys: np.ndarray, values: np.ndarray, gold: list[bytes]
) -> dict:
    """V_hat = M K (G + eps I)^-1; NN decode to event id; decision-bit scoring."""
    count = keys.shape[0]
    k_mat = keys.T  # (32, count)
    gram = k_mat.T @ k_mat + 1e-6 * np.eye(count)
    v_hat = m @ k_mat @ np.linalg.inv(gram)  # (32, count)
    predicted_ids = []
    for column in range(count):
        distances = np.linalg.norm(v_hat[:, column][:, None] - values.T, axis=0)
        predicted_ids.append(int(np.argmin(distances)))
    event_accuracy = float(
        np.mean([int(pred == index) for index, pred in enumerate(predicted_ids)])
    )
    predicted_answers = [gold[pred] for pred in predicted_ids]
    scored = p2s.score_bytes(predicted_answers, gold)
    return {
        "event_id_accuracy": event_accuracy,
        "decision_bit_accuracy": float(scored["decision_bit_accuracy"]),
        "decisions_correct": int(scored["decisions_correct"]),
        "predicted_ids": predicted_ids,
    }


def stack_fields(events: dict, groups: int, heads: int) -> dict:
    turns = sorted(events)
    out: dict[str, dict[str, np.ndarray]] = {}
    for group in range(groups):
        for head in range(heads):
            name = f"g{group}h{head}"
            out[name] = {
                "keys": torch.stack([events[t]["keys"][group, head] for t in turns]).numpy(),
                "values": torch.stack([events[t]["values"][group, head] for t in turns]).numpy(),
                "decays": torch.stack([events[t]["decays"][group, head] for t in turns]).numpy(),
                "queries": torch.stack([events[t]["queries"][group, head] for t in turns]).numpy(),
            }
    return out


def release_stack_fields(events: list[dict], groups: int, heads: int) -> dict:
    out: dict[str, dict[str, np.ndarray]] = {}
    if not events:
        return out
    for group in range(groups):
        for head in range(heads):
            out[f"g{group}h{head}"] = {
                "keys": torch.stack([event["keys"][group, head] for event in events]).numpy(),
                "values": torch.stack([event["values"][group, head] for event in events]).numpy(),
                "decays": torch.stack([event["decays"][group, head] for event in events]).numpy(),
            }
    return out


def capacity_of(stacks: dict) -> dict:
    per_head = {}
    for name, fields in stacks.items():
        decay_scalar = float(np.median(fields["decays"]))
        tau = 1.0 / (1.0 - decay_scalar) if decay_scalar < 1.0 else float("inf")
        per_head[name] = {"decay": decay_scalar, "tau_decay": tau, "head_capacity": min(32.0, tau)}
    total = float(sum(item["head_capacity"] for item in per_head.values()))
    return {"per_head": per_head, "capacity": total}


def query_alignment(stacks: dict) -> dict:
    all_cosines = []
    per_stack = {}
    for name, fields in stacks.items():
        q = fields["queries"]
        k = fields["keys"]
        q_unit = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        k_unit = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-12)
        cosines = np.sum(q_unit * k_unit, axis=1)
        per_stack[name] = float(np.median(cosines))
        all_cosines.extend(cosines.tolist())
    return {"median_all": float(np.median(all_cosines)), "per_stack_median": per_stack}


def control_probes(rows: list[dict], f_matrix: np.ndarray, criteria: dict) -> dict:
    gold = [row["gold"] for row in rows]
    queries = np.stack([query_key(row["query"]) for row in rows])
    x = np.concatenate([f_matrix, queries], axis=1)
    window = np.zeros((len(rows), 2), dtype=np.float64)
    for index, row in enumerate(rows):
        window[index, 0 if row["session"] == 1 else 1] = 1.0
    window_probe = p2s.probe(window, gold, p2s.decode_decision)
    permutation = []
    for seed in criteria["label_permutation_seeds"]:
        order = np.random.default_rng(seed).permutation(len(gold))
        permuted = [gold[int(index)] for index in order]
        permutation.append(
            {
                "seed": int(seed),
                "probe": p2s.probe(x, permuted, p2s.decode_decision)["metrics"][
                    "decision_bit_accuracy"
                ],
            }
        )
    perm_max = max(entry["probe"] for entry in permutation)
    rng = np.random.default_rng(RANDOM_SEED)
    random_x = rng.normal(size=x.shape)
    for index in range(len(random_x)):
        scale = float(np.linalg.norm(x[index])) / max(float(np.linalg.norm(random_x[index])), 1e-12)
        random_x[index] *= scale
    random_probe = p2s.probe(random_x, gold, p2s.decode_decision)
    other_x = np.stack([p2s.other_features(row) for row in rows])
    other_probe = p2s.probe(other_x, gold, p2s.decode_decision)
    return {
        "window_identity": float(window_probe["metrics"]["decision_bit_accuracy"]),
        "label_permutation_5_seeds": {
            "per_seed": {str(entry["seed"]): entry["probe"] for entry in permutation},
            "max": float(perm_max),
        },
        "random_projection_same_dim_same_norm": float(
            random_probe["metrics"]["decision_bit_accuracy"]
        ),
        "no_f_other_persistent_state": float(other_probe["metrics"]["decision_bit_accuracy"]),
        "strongest": float(
            max(
                perm_max,
                random_probe["metrics"]["decision_bit_accuracy"],
                other_probe["metrics"]["decision_bit_accuracy"],
                window_probe["metrics"]["decision_bit_accuracy"],
            )
        ),
    }


def criteria_gates() -> dict:
    return json.loads(PREREG.read_text(encoding="utf-8"))["criteria"]["gates"]


def boundary_flags(binding: dict, extras: dict, gates: dict) -> dict:
    checks = {
        "key_effective_rank": (binding["key_effective_rank"], gates["key_effective_rank_min"]),
        "pairwise_separable_fraction": (
            binding["pairwise_separable_fraction"],
            gates["key_pairwise_separable_fraction_min"],
        ),
        "key_norm_median": (binding["key_norm_median"], gates["key_norm_median_min"]),
        "condition_number": (binding["condition_number"], gates["condition_number_max"]),
        "query_alignment": (extras["query_alignment"], gates["query_alignment_cosine_median_min"]),
        "query_effective_rank": (extras["query_effective_rank"], gates["query_effective_rank_min"]),
        "value_effective_rank": (extras["value_effective_rank"], gates["value_effective_rank_min"]),
        "capacity": (extras["capacity"], gates["capacity_min"]),
        "closed_form_accuracy": (extras["closed_form_accuracy"], gates["probe_accuracy_min"]),
        "margin": (extras["margin"], gates["probe_margin_min"]),
        "effective_dof": (extras["effective_dof"], gates["effective_dof_min"]),
    }
    return {
        name: bool(abs(float(value) - float(gate)) < 1e-12)
        for name, (value, gate) in checks.items()
    }


def checkpoint_verdict(
    g_pass: bool, extras: dict, actual_key_pass: bool, boundaries: dict, dof: int, assessable: int
) -> str:
    gates = criteria_gates()
    if extras["capacity"] < gates["capacity_min"]:
        return "capacity_below_event_count"
    if not g_pass:
        return "key_space_collapsed_structurally"
    if (
        extras["query_alignment"] < gates["query_alignment_cosine_median_min"]
        or extras["query_effective_rank"] < gates["query_effective_rank_min"]
    ):
        return "query_key_misaligned"
    if extras["value_effective_rank"] < gates["value_effective_rank_min"]:
        return "value_space_collapsed"
    if not actual_key_pass:
        return "key_space_collapsed_under_actual_timing"
    if (
        extras["closed_form_accuracy"] >= gates["probe_accuracy_min"]
        and extras["margin"] >= gates["probe_margin_min"]
        and dof >= gates["effective_dof_min"]
        and assessable == 22
        and not any(boundaries.values())
    ):
        return "expressive_capacity_present"
    return "inconclusive"


def per_face_bits(readout: dict, gold: list[bytes]) -> list[int]:
    answers = [gold[pred] for pred in readout["predicted_ids"]]
    bits = []
    for answer, truth in zip(answers, gold, strict=True):
        bits.append(
            sum(1 for position in p2s.DECISION_POSITIONS if answer[position] == truth[position])
        )
    return bits


def paired_bootstrap(per_face: list[int], resamples: int) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bits = np.asarray(per_face, dtype=np.float64)
    accuracies = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(bits), size=len(bits))
        accuracies.append(float(bits[chosen].mean()))
    return {
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "faces",
        "mean": float(np.mean(accuracies)),
        "percentile_2_5": float(np.percentile(accuracies, 2.5)),
        "percentile_97_5": float(np.percentile(accuracies, 97.5)),
    }


def evaluate_checkpoint(graph, world: dict, criteria: dict, card: dict) -> dict:
    groups = graph.subject.config.groups
    heads = graph.subject.config.fast_heads
    turn_off = torch.zeros((groups, heads), dtype=torch.bool)
    turn_on = torch.ones((groups, heads), dtype=torch.bool)
    release_actual = torch.full((groups, heads), bool(graph.fast_enabled), dtype=torch.bool)

    # ---- actual condition ----
    actual = replay(graph, world, turn_off, release_actual)
    actual_rows = [
        row for row in actual["rows"] if row["correction_surface"] and row["released_lines"] > 0
    ]
    if len(actual_rows) != card["probe_face"]["expected_surfaces"]:
        raise EngineeringStop(f"expected 22 actual faces, got {len(actual_rows)}")
    face_turns = {row["turn"] for row in actual_rows}
    covered = len(set(actual["face_written_turns"]) & face_turns)
    timing_coverage = covered / len(actual_rows)

    # ---- forced condition (fresh replay, same weights) ----
    forced = replay(graph, world, turn_on, turn_on)
    forced_rows = [
        row for row in forced["rows"] if row["correction_surface"] and row["released_lines"] > 0
    ]
    if len(forced_rows) != card["probe_face"]["expected_surfaces"]:
        raise EngineeringStop(f"expected 22 forced faces, got {len(forced_rows)}")
    gold = [row["gold"] for row in forced_rows]

    stacks = stack_fields(forced["face_events"], groups, heads)
    key_result = key_gates({name: f["keys"] for name, f in stacks.items()}, criteria)
    align = query_alignment(stacks)
    value_ranks = {name: effective_rank(f["values"])[0] for name, f in stacks.items()}
    query_ranks = {name: effective_rank(f["queries"])[0] for name, f in stacks.items()}
    cap = capacity_of(stacks)

    f_matrix = np.stack([row["fast"].double().reshape(-1).numpy() for row in forced_rows])
    dof = int(np.unique(f_matrix, axis=0).shape[0])
    session_counts = {
        str(session): int(
            np.unique(
                f_matrix[[i for i, r in enumerate(forced_rows) if r["session"] == session]], axis=0
            ).shape[0]
        )
        for session in (1, 2)
    }

    m_by_stack = forced["final_fast"].double().numpy().reshape(groups, heads, 32, 32)
    stack_readouts = {
        name: closed_form_stack(
            m_by_stack[int(name[1]), int(name[3])], f["keys"], f["values"], gold
        )
        for name, f in stacks.items()
    }
    best_stack = max(stack_readouts, key=lambda n: stack_readouts[n]["decision_bit_accuracy"])
    closed_form_accuracy = float(stack_readouts[best_stack]["decision_bit_accuracy"])
    best_bits = per_face_bits(stack_readouts[best_stack], gold)
    controls = control_probes(forced_rows, f_matrix, criteria)
    margin = closed_form_accuracy - controls["strongest"]

    actual_stacks = release_stack_fields(actual["release_events"], groups, heads)
    actual_key_result = (
        key_gates({name: f["keys"] for name, f in actual_stacks.items()}, criteria)
        if actual_stacks
        else {"pass": False, "binding": {}, "per_stack": {}}
    )
    actual_final = actual["final_fast"].double().numpy().reshape(groups, heads, 32, 32)
    actual_gold = [
        world["scoring"]["turns"][str(event["turn"])]["expected_action"].encode("utf-8")
        for event in actual["release_events"]
    ]
    actual_readouts = {}
    for name, fields in actual_stacks.items():
        if fields["keys"].shape[0] >= 2:
            actual_readouts[name] = closed_form_stack(
                actual_final[int(name[1]), int(name[3])],
                fields["keys"],
                fields["values"],
                actual_gold,
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
    boundaries = boundary_flags(key_result["binding"], extras, criteria["gates"])
    verdict = checkpoint_verdict(
        key_result["pass"], extras, actual_key_result["pass"], boundaries, dof, len(forced_rows)
    )
    return {
        "verdict": verdict,
        "checkpoint": None,
        "arm": None,
        "seed": None,
        "timing_coverage_actual": timing_coverage,
        "timing_insufficient": bool(timing_coverage < criteria["gates"]["timing_coverage_min"]),
        "actual_written_lines": actual["written_lines"],
        "forced_turn_write_faces": len(forced["face_written_turns"]),
        "key_gates_forced": key_result,
        "key_gates_actual": {
            "pass": actual_key_result.get("pass", False),
            "binding": actual_key_result.get("binding", {}),
            "event_count": int(sum(f["keys"].shape[0] for f in actual_stacks.values())),
        },
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
            "actual_timing_readouts": {
                name: {
                    "decision_bit_accuracy": item["decision_bit_accuracy"],
                    "event_id_accuracy": item["event_id_accuracy"],
                }
                for name, item in actual_readouts.items()
            },
            "bootstrap": paired_bootstrap(best_bits, BOOTSTRAP_RESAMPLES),
        },
        "controls": controls,
        "margin": margin,
        "effective_dof": {"distinct_f_vectors": dof, "session_counts": session_counts},
        "boundary_flags": boundaries,
    }


def aggregate(results: list[dict]) -> dict:
    verdicts = [entry["verdict"] for entry in results]
    disagreement = len(set(verdicts)) > 1
    margins = [entry["margin"] for entry in results]
    strictest = int(np.argmin(margins))
    return {
        "per_checkpoint_verdicts": verdicts,
        "disagreement": bool(disagreement),
        "binding_checkpoint": results[strictest]["checkpoint"],
        "binding_margin": float(margins[strictest]),
        "verdict": "inconclusive" if disagreement else verdicts[0],
        "verdict_basis": (
            "strictest (minimum margin) of four checkpoints; disagreement forces inconclusive"
        ),
    }


def main() -> int:
    runner_sha = sha256_file(Path(__file__).resolve())
    try:
        prereg, card, criteria = load_frozen()
    except EngineeringStop as error:
        append_progress({"stage": "preflight", "ok": False, "reason": str(error)})
        write_json(
            REPORT,
            {
                "schema": "taiji-fec-1-expressive-capacity-v1",
                "package": "FEC-1",
                "status": "engineering_stop",
                "verdict": "engineering_stop",
                "reason": str(error),
                "criteria_sha256": CRITERIA_SHA256,
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
            "criteria_sha256": CRITERIA_SHA256,
            "feature_card_sha256": FEATURE_CARD_SHA256,
            "runner_sha256": runner_sha,
            "cpu": round(cost()[0], 3),
        }
    )
    face = card["probe_face"]
    world = fixture.build_world(face["seed"], face["world_index"])
    parent_payload = torch.load(r2.BUNDLE, map_location="cpu", weights_only=True)
    gates = parent_payload.get("growth_gates")
    gate = dict(gates[0]) if gates else None
    candidate = r2.load_routing_bundle(r2.BUNDLE)[0]

    # ---- random_init graph: no checkpoint load (20.1 initialized semantics) ----
    random_graph = r2.build_graph(candidate, gate, "CF1")
    random_graph.subject.eval()
    random_identity = r2.weights_digest(random_graph.subject)
    if random_identity != SHARED_INIT_SHA256:
        raise EngineeringStop(
            f"random_init graph identity {random_identity} != 20.1 shared_init {SHARED_INIT_SHA256}"
        )
    append_progress({"stage": "graph_identity", "ok": True, "cpu": round(cost()[0], 3)})

    results: list[dict] = []
    try:
        # ---- instrument validation first (bit-identity with graph.run_turn) ----
        for arm, seed in CHECKPOINT_ARM_SEEDS[:2]:
            payload = r2.load_checkpoint(r2.CHECKPOINTS / f"{arm}_{seed}_200.pt")
            graph = r2.build_graph(candidate, gate, arm)
            graph.subject.load_state_dict(payload["subject"])
            graph.subject.eval()
            equivalence = driver_equivalence(graph, world)
            append_progress(
                {
                    "stage": "equivalence",
                    "checkpoint": f"{arm}_{seed}",
                    "ok": equivalence["ok"],
                    "cpu": round(cost()[0], 3),
                }
            )
            if not equivalence["ok"]:
                raise EngineeringStop(f"driver equivalence failed for {arm}_{seed}: {equivalence}")

        # ---- random_init condition (attribution only, not a gate) ----
        groups = random_graph.subject.config.groups
        heads = random_graph.subject.config.fast_heads
        off = torch.zeros((groups, heads), dtype=torch.bool)
        on = torch.ones((groups, heads), dtype=torch.bool)
        random_actual = replay(random_graph, world, off, off)
        random_forced = replay(random_graph, world, on, on)
        random_stacks = stack_fields(random_forced["face_events"], groups, heads)
        random_keys = key_gates({n: f["keys"] for n, f in random_stacks.items()}, criteria)
        attribution = {
            "random_init_forced_key_gates": random_keys["binding"],
            "random_init_forced_pass": random_keys["pass"],
            "random_init_actual_written_lines": random_actual["written_lines"],
            "reading": (
                "architecture_inherent"
                if not random_keys["pass"]
                else "keys_separable_without_training"
            ),
            "note": "attribution only, not a gate (spec section 3.6); forced-vs-forced is the like-for-like basis",
        }
        append_progress(
            {"stage": "random_init", "pass": random_keys["pass"], "cpu": round(cost()[0], 3)}
        )

        # ---- four checkpoints ----
        for arm, seed in CHECKPOINT_ARM_SEEDS:
            name = f"{arm}_{seed}_200.pt"
            payload = r2.load_checkpoint(r2.CHECKPOINTS / name)
            if payload["arm"] != arm or int(payload["seed"]) != seed:
                raise EngineeringStop("checkpoint identity mismatch")
            graph = r2.build_graph(candidate, gate, arm)
            graph.subject.load_state_dict(payload["subject"])
            graph.subject.eval()
            result = evaluate_checkpoint(graph, world, criteria, card)
            result["checkpoint"] = name
            result["arm"] = arm
            result["seed"] = seed
            results.append(result)
            append_progress(
                {
                    "stage": "checkpoint",
                    "checkpoint": name,
                    "verdict": result["verdict"],
                    "margin": result["margin"],
                    "closed_form": result["closed_form"]["binding_decision_bit_accuracy"],
                    "dof": result["effective_dof"]["distinct_f_vectors"],
                    "timing_coverage": result["timing_coverage_actual"],
                    "cpu": round(cost()[0], 3),
                }
            )
    except BudgetStop as error:
        append_progress({"stage": "budget_stop", "reason": str(error)})
        write_json(
            REPORT,
            {
                "schema": "taiji-fec-1-expressive-capacity-v1",
                "package": "FEC-1",
                "status": "budget_stop",
                "verdict": "inconclusive",
                "reason": str(error),
                "cost": {"cpu_seconds": round(cost()[0], 2), "wall_seconds": round(cost()[1], 2)},
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
                "schema": "taiji-fec-1-expressive-capacity-v1",
                "package": "FEC-1",
                "status": "engineering_stop",
                "verdict": "engineering_stop",
                "reason": str(error),
                "criteria_sha256": CRITERIA_SHA256,
                "runner_sha256": runner_sha,
                "readings_produced": len(results),
            },
        )
        print(json.dumps({"verdict": "engineering_stop", "reason": str(error)}))
        return 0

    aggregation = aggregate(results)
    cpu, wall = cost()
    report = {
        "schema": "taiji-fec-1-expressive-capacity-v1",
        "package": "FEC-1",
        "date": "2026-11-08",
        "status": "completed",
        "verdict": aggregation["verdict"],
        "verdict_basis": aggregation["verdict_basis"],
        "aggregation": aggregation,
        "discipline": {
            "training": 0,
            "optimizer_steps": 0,
            "adam": 0,
            "new_parameters": 0,
            "checkpoint_writes": 0,
            "confirm_reads": 0,
            "graph_edits": 0,
            "only_intervention": "allowed permission mask",
        },
        "provenance": {
            "criteria_sha256": CRITERIA_SHA256,
            "feature_card_sha256": FEATURE_CARD_SHA256,
            "runner_sha256": runner_sha,
            "build_graph_provider_sha256": R2_SCRIPT_SHA256,
            "checkpoints": {
                f"{arm}_{seed}_200.pt": sha256_file(r2.CHECKPOINTS / f"{arm}_{seed}_200.pt")
                for arm, seed in CHECKPOINT_ARM_SEEDS
            },
            "random_init_shared_init_sha256": SHARED_INIT_SHA256,
            "random_init_graph_weights_sha256": random_identity,
            "probe_face": card["probe_face"],
        },
        "attribution": attribution,
        "results": results,
        "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
        "scope": (
            "Function-class verdict on the current graph and checkpoint lineage only; "
            "expressive_capacity_present would be an architecture-requirement input, "
            "not training admission, not pricing, not a capability claim."
        ),
    }
    write_json(REPORT, report)
    write_json(
        EXEC_MANIFEST,
        {
            "schema": "taiji-fec-1-execution-manifest-v1",
            "package": "FEC-1",
            "date": "2026-11-08",
            "prereg_manifest": "plans/manifests/taiji_f_expressive_capacity_prereg_20261108.json",
            "criteria_sha256": CRITERIA_SHA256,
            "feature_card": "plans/manifests/taiji_f_expressive_capacity_feature_card_20261108.json",
            "feature_card_sha256": FEATURE_CARD_SHA256,
            "report": "reports/taiji_f_expressive_capacity_20261108.json",
            "report_sha256": sha256_file(REPORT),
            "runner": "scripts/training/probe_taiji_f_expressive_capacity_20261108.py",
            "runner_sha256": runner_sha,
            "submitted_runner_sha256": runner_sha,
            "verdict": aggregation["verdict"],
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
        },
    )
    print(
        json.dumps(
            {
                "verdict": aggregation["verdict"],
                "per_checkpoint": aggregation["per_checkpoint_verdicts"],
                "binding_margin": aggregation["binding_margin"],
                "cpu": round(cpu, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
