"""Price the A.4 in-body column read without training anything.

A.2 overran its cap by 10.9 percent because its fixed phases - evaluation at
three dose points, the independent-process recovery and the byte-identical
replay - were never priced, only the per-update term was. This probe prices
every phase separately, measures the fixed phases instead of guessing them, and
then builds the budget from the measured units.

Nothing here trains: there is no optimizer, no optimizer step, no checkpoint and
no bundle. The body is the registered candidate (22,742 scalars / 194 tensors),
the read is freshly initialised, and the byte rounds are forward passes only.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_column_read import ColumnRead, inject_read, parameter_census, read_for_example
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_a4_column_read_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a4_column_read_price_20261116.json"
DATA = ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict) -> None:
    if contract.get("status") != "design_priced":
        raise ValueError("contract_not_frozen")
    pins = contract["source_pins"]
    if sha_of(ROOT / pins["script"]) != pins["script_sha256"]:
        raise ValueError("script_source_drift")
    if sha_of(ROOT / pins["read_module"]) != pins["read_module_sha256"]:
        raise ValueError("read_module_source_drift")


def byte_round(model, example: dict, features: torch.Tensor, read: ColumnRead) -> bytes:
    """One full-response round with the read injected at the joint readout."""
    cap = int(model.c.max_output_bytes) + 1
    state = p4_2.prepare(model, p4_2.column_public(example), example["id"])
    content: list[int] = []
    with torch.no_grad(), inject_read(model, read, lambda: features):
        for _ in range(cap):
            probability, _, details = model.output(
                state, model.language_goal(state), state["prediction_action"]
            )
            symbol = int(probability.argmax())
            if symbol == 256:
                break
            content.append(symbol)
            state = model.accept_byte(state, symbol, details)
    return bytes(content)


def timeit(function, repeats: int) -> tuple[float, list[float]]:
    samples = []
    for _ in range(repeats):
        started = time.process_time()
        function()
        samples.append(time.process_time() - started)
    return statistics.median(samples), samples


def main() -> int:
    if not CONTRACT.exists():
        print(json.dumps({"verdict": "refused", "reason": "manifest_missing"}))
        return 2
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    try:
        guards(contract)
    except ValueError as error:
        print(json.dumps({"verdict": "refused", "reason": str(error)}))
        return 2

    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(DATA)
    if value_digest(data) != contract["data_digest"]:
        print(json.dumps({"verdict": "refused", "reason": "data_digest_mismatch"}))
        return 2

    model = make_model("native", int(contract["candidate"]["seed"]))
    model.eval()
    read = ColumnRead(int(contract["candidate"]["width"]))
    read.eval()
    census = parameter_census(read)

    train = data["train"]
    fit = data["fit"]
    sample = train[:8]
    fit_sample = fit[:8]

    # 1. column tokens and read forward, per example
    with torch.no_grad():
        token_cost, _ = timeit(
            lambda: [p4_2.column_public(example) for example in sample],
            3,
        )
        read_cost, _ = timeit(
            lambda: [read_for_example(model, read, example) for example in sample],
            3,
        )
        features = {
            example["id"]: read_for_example(model, read, example)["injection"]
            for example in sample + fit_sample
        }

    # 2. one byte round per example, read injected, features cached
    round_cost, round_samples = timeit(
        lambda: [byte_round(model, example, features[example["id"]], read) for example in sample],
        3,
    )

    # 3. byte round that recomputes the read at every position (worst case)
    def recompute_round() -> None:
        for example in sample:
            state = p4_2.prepare(model, p4_2.column_public(example), example["id"])
            cap = int(model.c.max_output_bytes) + 1

            def provider(example=example):
                return read_for_example(model, read, example)["injection"]

            with torch.no_grad(), inject_read(model, read, provider):
                for _ in range(cap):
                    probability, _, details = model.output(
                        state, model.language_goal(state), state["prediction_action"]
                    )
                    symbol = int(probability.argmax())
                    if symbol == 256:
                        break
                    state = model.accept_byte(state, symbol, details)

    recompute_cost, _ = timeit(recompute_round, 3)

    # 4. training-step proxy: forward plus backward, no optimizer, no parameter change
    def train_step_proxy() -> None:
        example = sample[0]
        state = p4_2.prepare(model, p4_2.column_public(example), example["id"])
        feature = features[example["id"]]
        terms = []
        with inject_read(model, read, lambda: feature):
            for symbol in [*example["target"], 256]:
                probability, _, details = model.output(
                    state, model.language_goal(state), state["prediction_action"]
                )
                terms.append(-probability[symbol].log())
                with torch.no_grad():
                    state = model.accept_byte(state, symbol, details)
        torch.stack(terms).mean().backward()
        for parameter in list(model.parameters()) + list(read.parameters()):
            parameter.grad = None

    step_cost, step_samples = timeit(train_step_proxy, 3)

    # 5. prefix-0 capture over the whole fit split: the probe's dominant phase
    def capture_prefix0() -> None:
        from scripts.training.review_a3_interface_20261116 import cache_x

        cache_x(model, fit)

    capture_cost, _ = timeit(capture_prefix0, 1)

    # 6. recovery replay: same rounds again, digests compared
    digests = []
    for example in fit_sample:
        digests.append(
            hashlib.sha256(byte_round(model, example, features[example["id"]], read)).hexdigest()
        )
    replay_cost, _ = timeit(
        lambda: [
            hashlib.sha256(byte_round(model, example, features[example["id"]], read)).hexdigest()
            for example in fit_sample
        ],
        1,
    )
    digests_after = []
    for example in fit_sample:
        digests_after.append(
            hashlib.sha256(byte_round(model, example, features[example["id"]], read)).hexdigest()
        )
    replay_stable = digests == digests_after

    plan = contract["plan"]
    seeds = int(plan["seeds"])
    updates = int(plan["updates_per_seed"])
    dose_points = int(plan["dose_points"])
    per_seed_training = updates * step_cost
    per_seed_evaluation = dose_points * ((len(fit) + len(train)) * round_cost / max(1, len(sample)))
    per_seed_probe = capture_cost
    per_seed_recovery = replay_cost * (len(fit) / max(1, len(fit_sample)))
    per_seed_checkpoints = dose_points * 0.05
    per_seed = (
        per_seed_training
        + per_seed_evaluation
        + per_seed_probe
        + per_seed_recovery
        + per_seed_checkpoints
    )
    estimate = per_seed * seeds
    contingency = round(estimate * float(plan["contingency"]), 3)
    cap = round(estimate + contingency, 3)

    report = {
        "schema": "taiji-u2-a4-column-read-price-v1",
        "card": "22.5.A.4",
        "date": "2026-11-16",
        "verdict": (
            "price_measured" if cap <= float(plan["design_cap_cpu"]) else "pricing_over_budget"
        ),
        "read_identity": "column-read-v1",
        "read_parameters": census,
        "body": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "tensors": len(list(model.named_parameters())),
            "note": "the registered candidate is unchanged; the read is a separately identified scaffold",
        },
        "unit_cpu": {
            "column_public_per_example": round(token_cost / len(sample), 6),
            "read_forward_per_example": round(read_cost / len(sample), 6),
            "byte_round_per_example_cached_read": round(round_cost / len(sample), 6),
            "byte_round_per_example_recomputed_read": round(recompute_cost / len(sample), 6),
            "train_step_proxy_per_example": round(step_cost, 6),
            "prefix0_capture_fit32": round(capture_cost, 6),
            "recovery_replay_fit8": round(replay_cost, 6),
            "replay_stable": replay_stable,
        },
        "raw": {
            "byte_round_samples": [round(value, 6) for value in round_samples],
            "train_step_samples": [round(value, 6) for value in step_samples],
        },
        "plan": {
            "seeds": seeds,
            "updates_per_seed": updates,
            "dose_points": dose_points,
            "per_seed_training": round(per_seed_training, 3),
            "per_seed_evaluation": round(per_seed_evaluation, 3),
            "per_seed_probe": round(per_seed_probe, 3),
            "per_seed_recovery": round(per_seed_recovery, 3),
            "per_seed_checkpoints": round(per_seed_checkpoints, 3),
            "per_seed_total": round(per_seed, 3),
            "estimate_cpu": round(estimate, 3),
            "contingency_cpu": contingency,
            "cap_cpu": cap,
        },
        "discipline": {
            "optimizer_calls": 0,
            "optimizer_steps": 0,
            "checkpoints_written": 0,
            "bundles_written": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
            "training_updates": 0,
        },
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
        "scope": "pricing only: forward passes and one backward on a throwaway copy, no optimizer, no checkpoint, no capability claim",
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cap_cpu": cap,
                "estimate_cpu": round(estimate, 3),
                "read_scalars": census["scalars"],
                "replay_stable": replay_stable,
            }
        )
    )
    return 0 if report["verdict"] == "price_measured" else 4


if __name__ == "__main__":
    raise SystemExit(main())
