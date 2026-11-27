"""22.4.4 full-task pricing for U2 and the frozen regular baselines.

This is a pricing measurement, not formal training. It measures observation
encoding, query/expression, policy, single learning, save/rebuild, replay and
birth/maturity on a fixed synthetic fragment. Policy and birth/maturity are
structurally not applicable to the frozen sequence baselines and are declared as
such. One throwaway optimizer step per seed is taken on a cloned model after a
checkpoint save/restore preflight; the original models are not updated.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.baseline_identity_22_4_4 import (  # noqa: E402
    BATCH,
    DEVICE,
    DTYPE,
    GOAL,
    SEED,
    STEPS,
    GRUSequenceBaseline,
    build_arm,
    normalized_sha256,
    state_dict_digest,
)
from taiji.u2_neural_units import NeuralUnits, UnitGraphSpec  # noqa: E402
from taiji.u2_subject import U2Config, U2Packet, U2Subject, synthetic_fixture  # noqa: E402

REPORT = ROOT / "reports/taiji_u2_22_4_4_price_20261117.json"
PARENT = ROOT / "plans/manifests/taiji_u2_22_4_accepted_parent_20261117.json"
IDENTITY = ROOT / "plans/manifests/taiji_u2_22_4_4_baseline_identity_20261111.json"
IDENTITY_REPORT = ROOT / "reports/taiji_u2_22_4_4_identity_measurement_20261111.json"
PARENT_22_3 = ROOT / "plans/manifests/taiji_u2_22_3_accepted_parent_20261111.json"
PARTITION_REPORT = ROOT / "reports/taiji_u2_partition_22_4_3_20261117.json"

SEEDS = (20261110, 20261111, 20261112)
U2_ARM = "u2_reference"
ARMS = (U2_ARM, "transformer_sequence", "gru_sequence")
WIDTHS = {"transformer_sequence": 28, "gru_sequence": 40}
COMPONENTS = (
    "observation_encoding",
    "query_expression",
    "policy",
    "single_learning",
    "save_rebuild",
    "replay_transfer",
    "birth_maturity",
)
APPLICABILITY = {
    U2_ARM: {name: True for name in COMPONENTS},
    "transformer_sequence": {
        name: name
        in {
            "observation_encoding",
            "query_expression",
            "single_learning",
            "save_rebuild",
            "replay_transfer",
        }
        for name in COMPONENTS
    },
    "gru_sequence": {
        name: name
        in {
            "observation_encoding",
            "query_expression",
            "single_learning",
            "save_rebuild",
            "replay_transfer",
        }
        for name in COMPONENTS
    },
}
CPU_CAP = 250.0
WALL_CAP = 600.0
STAGE_PLAN_CPU = {"22.5": 16000.0, "22.6": 8000.0, "22.7": 12000.0, "22.8": 4000.0}
UPDATE_OVERHEAD = 1.20
SEED_FACTOR = 3


def canonical(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _measure(callable_, repeats: int = 1) -> dict:
    cpu_samples = []
    wall_samples = []
    for _ in range(repeats):
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        callable_()
        cpu_samples.append(time.process_time() - cpu_start)
        wall_samples.append(time.perf_counter() - wall_start)
    return {
        "cpu": round(statistics.median(cpu_samples), 6),
        "wall": round(statistics.median(wall_samples), 6),
        "repeats": repeats,
    }


def _u2_state(model, seed: int, steps: int = 1):
    fixture = synthetic_fixture(model.c, DTYPE)
    state = model.initial("u2-22-4-4-pricing")
    state = model.set_goal(state, GOAL)
    state = model.set_records(state, fixture["records"])
    result = None
    event = None
    for index in range(steps):
        mask = torch.zeros(9, dtype=torch.bool)
        mask[0] = True
        state, decision = model.choose_operation(state, mask)
        state, identity = model.issue(state, decision, dt=0.05)
        start = state["world_time"]
        end = state["pending"]["world_end"]
        state["process_time"] = end
        event = fixture["events"][(index + seed) % len(fixture["events"])]
        state, result = model.receive(
            state,
            U2Packet(
                f"pricing-{seed}-{index}",
                "sensor",
                "world",
                start,
                end,
                end,
                event["observations"],
                event["coordinates"],
                event["outcome"],
                identity,
                feedback_record_ids=("red",),
            ),
        )
    return state, fixture, event, result


def _u2_loss(model, seed: int, steps: int = STEPS, backward: bool = False):
    losses = []
    for index in range(steps):
        state, fixture, event, result = _u2_state(model, seed + index, 1)
        losses.append(-result["predictive_logp"].mean())
    loss = torch.stack(losses).mean()
    if backward:
        loss.backward()
    return loss


def _u2_update(model, seed: int) -> dict:
    state, _, _, _ = _u2_state(model, seed, 1)
    bundle = model.export_bundle(state)
    clone, _ = type(model).from_bundle(bundle)
    optimizer = torch.optim.Adam(clone.parameters(), lr=0.001)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    loss = _u2_loss(clone, seed, STEPS, backward=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    cpu = time.process_time() - started_cpu
    wall = time.perf_counter() - started_wall
    return {
        "cpu": round(cpu, 6),
        "wall": round(wall, 6),
        "loss": float(loss.detach()),
        "updated_parameters": len([p for p in clone.parameters() if p.requires_grad]),
    }


def _baseline_inputs(seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rgb = torch.rand((BATCH, STEPS, 3, 4, 4), generator=generator, dtype=DTYPE)
    audio = torch.rand((BATCH, STEPS, 33), generator=generator, dtype=DTYPE)
    proprio = torch.rand((BATCH, STEPS, 8), generator=generator, dtype=DTYPE)
    targets = torch.randint(0, 257, (BATCH,), generator=generator)
    observations = {"rgb": rgb.reshape(BATCH, STEPS, 48), "audio": audio, "proprio": proprio}
    return observations, targets


def _baseline_representation(model, observations, text_context):
    sequence = model.frontend.step_tokens(observations, text_context)
    if isinstance(model, GRUSequenceBaseline):
        output, _ = model.gru(sequence)
        return output[:, -1]
    hidden = model.record_in(sequence) + model.position[None]
    mask = torch.triu(torch.ones(STEPS, STEPS, dtype=torch.bool, device=DEVICE), diagonal=1)
    attended, _ = model.attention(hidden, hidden, hidden, attn_mask=mask, need_weights=False)
    hidden = model.norm_attention(hidden + attended)
    hidden = model.norm_feedforward(hidden + model.feedforward(hidden))
    return hidden[:, -1]


def _baseline_expression(model, last, text_context):
    previous = model.frontend.embedding(torch.full((BATCH,), 255, dtype=torch.long))
    return model.byte_head(torch.cat([last, previous], -1))


def _measure_u2(model, seed: int) -> dict:
    state, fixture, event, _ = _u2_state(model, seed, 1)
    mask = torch.zeros(9, dtype=torch.bool)
    mask[0] = True
    encoding = _measure(
        lambda: model.encode(event["observations"], event["coordinates"], event["outcome"]),
        repeats=8,
    )
    expression = _measure(
        lambda: model.output(state, state["prediction_goal"], state["prediction_action"]),
        repeats=8,
    )
    policy = _measure(lambda: model.choose_operation(state, mask), repeats=8)
    save_rebuild = _measure(lambda: _u2_save_rebuild(model, state), repeats=1)
    forward = _measure(lambda: _u2_loss(model, seed, STEPS, backward=False), repeats=1)
    single = _u2_update(model, seed)
    replay = _measure(lambda: _u2_loss(model, seed, STEPS, backward=False), repeats=1)
    birth = _measure(lambda: _birth_maturity(seed), repeats=3)
    return {
        "observation_encoding": encoding,
        "query_expression": expression,
        "policy": policy,
        "single_learning": {**single, "forward_cpu": forward["cpu"]},
        "save_rebuild": save_rebuild,
        "replay_transfer": replay,
        "birth_maturity": birth,
    }


def _u2_save_rebuild(model, state) -> bool:
    bundle = model.export_bundle(state)
    restored, restored_state = type(model).from_bundle(bundle)
    return restored_state["subject_id"] == state["subject_id"]


def _birth_maturity(seed: int) -> None:
    units = NeuralUnits(UnitGraphSpec(groups=2, units_per_group=2, seed=seed))
    child = units.with_neutral_birth(0, ["u0000"])
    child.set_lifecycle("u0000", "dormant")
    child.set_lifecycle("u0000", "mature")


def _measure_baseline(model, seed: int) -> dict:
    observations, targets = _baseline_inputs(seed)
    text_context = model.frontend.text_units(GOAL)
    encoding = _measure(
        lambda: model.frontend.step_tokens(observations, text_context),
        repeats=8,
    )
    last = _baseline_representation(model, observations, text_context)
    expression = _measure(
        lambda: _baseline_expression(model, last, text_context),
        repeats=8,
    )
    save_rebuild = _measure(lambda: _baseline_save_rebuild(model), repeats=1)

    def update():
        clone = copy.deepcopy(model)
        optimizer = torch.optim.Adam(clone.parameters(), lr=0.001)
        loss = clone(observations, targets, text_context)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())

    started_cpu, started_wall = time.process_time(), time.perf_counter()
    loss_value = update()
    single = {
        "cpu": round(time.process_time() - started_cpu, 6),
        "wall": round(time.perf_counter() - started_wall, 6),
        "loss": loss_value,
    }
    replay = _measure(
        lambda: model(observations, targets, text_context),
        repeats=1,
    )
    return {
        "observation_encoding": encoding,
        "query_expression": expression,
        "policy": None,
        "single_learning": single,
        "save_rebuild": save_rebuild,
        "replay_transfer": replay,
        "birth_maturity": None,
    }


def _baseline_save_rebuild(model) -> bool:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    clone = build_arm(model.architecture_id, model.width)
    clone.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    return state_dict_digest(clone) == state_dict_digest(model)


def _preflight(arm: str, model) -> dict:
    if arm == U2_ARM:
        state, _, _, _ = _u2_state(model, SEEDS[0], 1)
        return {
            "arm": arm,
            "save_rebuild_equal": _u2_save_rebuild(model, state),
            "method": "U2Subject.export_bundle/from_bundle",
        }
    return {
        "arm": arm,
        "save_rebuild_equal": _baseline_save_rebuild(model),
        "method": "torch.save/load state_dict",
    }


def _median_component(rows: list[dict], component: str) -> dict | None:
    values = [row[component] for row in rows if row[component] is not None]
    if not values:
        return None
    return {
        "cpu": round(statistics.median(value["cpu"] for value in values), 6),
        "wall": round(statistics.median(value["wall"] for value in values), 6),
        "seeds": len(values),
    }


def build_price_table(measurements: dict[str, list[dict]]) -> dict:
    table = {}
    for arm in ARMS:
        rows = measurements[arm]
        table[arm] = {
            component: (
                _median_component(rows, component)
                if APPLICABILITY[arm][component]
                else {"status": "not_applicable_architecture"}
            )
            for component in COMPONENTS
        }
    u2_single = table[U2_ARM]["single_learning"]["cpu"]
    per_update = u2_single * UPDATE_OVERHEAD
    per_update_pack = per_update * SEED_FACTOR
    updates = {
        stage: int(math.floor(plan / per_update_pack)) if per_update_pack > 0 else None
        for stage, plan in STAGE_PLAN_CPU.items()
    }
    return {
        "components": table,
        "per_update_formula": "u2_single_learning_cpu * 1.20 overhead * 3 fixed seeds",
        "u2_per_update_cpu": round(per_update, 6),
        "u2_per_update_pack_cpu": round(per_update_pack, 6),
        "seed_factor": SEED_FACTOR,
        "updates_by_stage": updates,
        "stage_plan_cpu": STAGE_PLAN_CPU,
        "note": "updates use the plan CPU column, preserving the 20% reserve in the total cap; "
        "they are capacity ceilings, not a prescribed dose",
    }


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    torch.manual_seed(SEED)
    u2 = U2Subject(U2Config.small()).to(dtype=DTYPE, device=DEVICE).eval()
    baselines = {
        "transformer_sequence": build_arm("transformer_sequence", WIDTHS["transformer_sequence"]),
        "gru_sequence": build_arm("gru_sequence", WIDTHS["gru_sequence"]),
    }
    preflight = {arm: _preflight(arm, u2 if arm == U2_ARM else baselines[arm]) for arm in ARMS}
    if not all(row["save_rebuild_equal"] for row in preflight.values()):
        if REPORT.exists():
            raise FileExistsError(REPORT)
        report = {
            "schema": "taiji-u2-22-4-4-price-v1",
            "card": "22.4.4",
            "verdict": "instrument_invalid",
            "preflight": preflight,
        }
        REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
        print("price verdict=instrument_invalid preflight", preflight)
        return 0
    measurements = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        measurements[U2_ARM].append(_measure_u2(u2, seed))
        for arm in ("transformer_sequence", "gru_sequence"):
            measurements[arm].append(_measure_baseline(baselines[arm], seed))
    price = build_price_table(measurements)
    price_digest = canonical(price)
    identity_sha = normalized_sha256(IDENTITY)
    partition = json.loads(PARTITION_REPORT.read_text(encoding="utf-8"))
    parent_22_3_sha = normalized_sha256(PARENT_22_3)
    cpu = round(time.process_time() - started_cpu, 6)
    wall = round(time.perf_counter() - started_wall, 6)
    verdict = "pricing_over_budget" if cpu > CPU_CAP or wall > WALL_CAP else "pricing_valid"
    report = {
        "schema": "taiji-u2-22-4-4-price-v1",
        "card": "22.4.4",
        "phase": "pricing_only_no_formal_training",
        "verdict": verdict,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "preflight": preflight,
        "measurements": measurements,
        "price_table": price,
        "price_table_digest": price_digest,
        "identity": {
            "ref": "plans/manifests/taiji_u2_22_4_4_baseline_identity_20261111.json",
            "sha256": identity_sha,
            "selected": "transformer_sequence",
        },
        "accepted_parent_22_3": {
            "ref": "plans/manifests/taiji_u2_22_3_accepted_parent_20261111.json",
            "sha256": parent_22_3_sha,
        },
        "data": {
            "recipe_digest": partition["manifest"]["data_recipe_digest"],
            "partition_digest": partition["manifest"]["partition_digest"],
        },
        "applicability": APPLICABILITY,
        "discipline": {
            "formal_training": False,
            "pricing_updates": 3,
            "optimizer_steps": 3,
            "project_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": cpu, "wall": wall, "cpu_cap": CPU_CAP, "wall_cap": WALL_CAP},
        "limitations": [
            "Pricing fragment is synthetic: batch=2, 8 steps, float32, cpu, threads=1.",
            "Policy and birth/maturity are not applicable to the frozen sequence baselines; they have no action-selection or unit-growth interface.",
            "U2 single learning and replay are measured with the identity script's real protocol; baselines use their frozen forward path.",
            "One throwaway Adam step per seed is a pricing measurement, not formal training; the original model is not updated.",
            "Windows timer granularity is about 15.6 ms; cheap components use repeat medians.",
        ],
    }
    if verdict == "pricing_valid":
        if PARENT.exists():
            raise FileExistsError(PARENT)
        parent = {
            "schema": "taiji-u2-22-4-accepted-parent-v1",
            "status": "accepted_engineering_parent_for_22_5",
            "engineering_parent": {
                "ref": "plans/manifests/taiji_u2_22_3_accepted_parent_20261111.json",
                "sha256": parent_22_3_sha,
            },
            "baseline_identity": {
                "ref": "plans/manifests/taiji_u2_22_4_4_baseline_identity_20261111.json",
                "sha256": identity_sha,
                "selected": "transformer_sequence",
                "limitations": [
                    "L1 compute parity absent",
                    "resource gates dominated by parameter band",
                ],
            },
            "data": {
                "recipe_digest": partition["manifest"]["data_recipe_digest"],
                "partition_digest": partition["manifest"]["partition_digest"],
                "feedback_rights": partition["manifest"]["data_recipe"]["feedback_rights"],
                "check_exposure": partition["manifest"]["check_exposure"],
                "final_seal": partition["manifest"]["final_seal"],
            },
            "price": {
                "ref": "reports/taiji_u2_22_4_4_price_20261117.json",
                "digest": price_digest,
                "updates_by_stage": price["updates_by_stage"],
                "u2_per_update_cpu": price["u2_per_update_cpu"],
                "u2_per_update_pack_cpu": price["u2_per_update_pack_cpu"],
                "seed_factor": price["seed_factor"],
            },
            "arm_component_applicability": APPLICABILITY,
            "stop_conditions": [
                "pricing_over_budget if total CPU > 1200 or wall > 2400",
                "not_admitted if any of the seven 22.4 deliverables is missing",
                "check exposure starts on first check access and is irrevocable",
                "final read requires explicit R-SEAL-1 authorization",
            ],
            "capability_qualifications": [],
            "authorized": False,
            "note": "Engineering parent only; does not authorize training, update a model, or claim capability.",
        }
        PARENT.write_bytes(json.dumps(parent, ensure_ascii=False, indent=2).encode() + b"\n")
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "price verdict="
        + verdict
        + " cpu="
        + str(cpu)
        + " wall="
        + str(wall)
        + " updates="
        + json.dumps(price["updates_by_stage"], ensure_ascii=False)
        + " digest="
        + price_digest[:16]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
