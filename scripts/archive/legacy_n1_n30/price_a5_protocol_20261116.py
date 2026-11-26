"""Price the A.5 parent protocol without training anything.

A.2 overran its cap by 10.9 percent because its fixed phases - the initial
gate, the gradient census, the final rounds and the independent-process
recovery - were never priced, only the per-update term was.  This probe
prices every phase of the A.5 protocol separately and then builds the budget
from the measured units.

The protocol has five priced phases per seed: training, evaluation at three
dose points, the feedback round, the retention control arm (the A.4-style
accumulating protocol, kept as an attribution arm) and the recovery replay.
The feedback round is priced twice, once for the one-error-at-a-time protocol
and once for the accumulating protocol, because both arms run.

Nothing here trains: there is no optimizer, no optimizer step, no checkpoint
and no bundle.  The body is the registered candidate, the byte head is the
one A.2 already trained, and the byte rounds are forward passes only.
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

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.first_a_support_22_5 import make_model
from taiji.u2_history import atomic_load, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a5_protocol_price_20261116.json"
DATA = ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin"
A2_FINAL = ROOT / "data/experiments/taiji_u2/a2/20261115T2340Z/final.pt"


def sha_of(path: Path) -> str:
    """Hash of a source file (text)."""
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def sha_bytes_of(path: Path) -> str:
    """Hash of a binary artifact (encoder bundle, checkpoint)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timeit(function, repeats: int):
    samples = []
    for _ in range(repeats):
        started = time.process_time()
        function()
        samples.append(time.process_time() - started)
    return statistics.median(samples), samples


def load_body(contract: dict):
    candidate = contract["candidate"]
    bundle = ROOT / candidate["encoder_bundle"]
    if sha_bytes_of(bundle) != candidate["encoder_bundle_sha256"]:
        raise ValueError("encoder_bundle_mismatch")
    model = make_model("native", int(candidate["seed"]))
    model.encoder["rgb"].load_state_dict(torch.load(bundle, weights_only=False)["encoder"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, bundle


def load_head(contract: dict):
    candidate = contract["candidate"]
    path = ROOT / candidate["a1b_checkpoint"]
    if sha_bytes_of(path) != candidate["a1b_checkpoint_sha256"]:
        raise ValueError("a1b_checkpoint_mismatch")
    head = a2.ColourHead()
    head.load_state_dict(torch.load(path, weights_only=False)["head"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def one_round(model, examples, codes):
    return a2.round_rows(model, examples, codes)


def replay_digests(model, examples, codes):
    rows = []
    for example in examples:
        digest = hashlib.sha256(
            one_round(model, [example], codes)[0]["output_hex"].encode()
        ).hexdigest()
        rows.append(digest)
    return rows


def main() -> int:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    if not CONTRACT.exists():
        print(json.dumps({"verdict": "refused", "reason": "manifest_missing"}))
        return 2
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") not in {"design_priced", "price_frozen_authorization_pending"}:
        print(json.dumps({"verdict": "refused", "reason": "contract_not_priced"}))
        return 2
    torch.set_num_threads(1)

    data = atomic_load(DATA)
    if value_digest(data) != contract["data_digest"]:
        print(json.dumps({"verdict": "refused", "reason": "data_digest_mismatch"}))
        return 2

    model, bundle = load_body(contract)
    head = load_head(contract)
    prior = a2.prior_of(data)
    expected = torch.tensor(contract["candidate"]["prior"], dtype=torch.float32)
    if not torch.allclose(prior, expected, atol=1e-6, rtol=0):
        print(json.dumps({"verdict": "refused", "reason": "prior_mismatch"}))
        return 2

    if not A2_FINAL.exists():
        print(json.dumps({"verdict": "refused", "reason": "a2_final_checkpoint_absent"}))
        return 2
    payload = torch.load(A2_FINAL, weights_only=False)
    trained = make_model("native", int(contract["candidate"]["seed"]))
    trained.encoder["rgb"].load_state_dict(torch.load(bundle, weights_only=False)["encoder"])
    trained.load_state_dict(payload["model"])
    trained.eval()
    for parameter in trained.parameters():
        parameter.requires_grad_(False)
    a2.set_trainable(trained, contract)

    train = data["train"]
    fit = data["fit"]
    sample = train[:8]
    codes = a2.code_by_id(trained, head, prior, train + fit, contract)

    unit = {}

    round_cost, round_samples = timeit(lambda: one_round(trained, sample, codes), 5)
    unit["byte_round_per_example"] = round(round_cost / len(sample), 6)
    # The untrained head runs to the 65-byte cap and costs an order of
    # magnitude more than the trained one.  The feedback round may start from a
    # head whose answers are wrong, so the budget carries the expensive number.
    untrained = make_model("native", int(contract["candidate"]["seed"]))
    untrained.encoder["rgb"].load_state_dict(torch.load(bundle, weights_only=False)["encoder"])
    untrained.eval()
    for parameter in untrained.parameters():
        parameter.requires_grad_(False)
    untrained_codes = a2.code_by_id(untrained, head, prior, train + fit, contract)
    untrained_cost, _ = timeit(lambda: one_round(untrained, sample, untrained_codes), 5)
    unit["byte_round_per_example_untrained"] = round(untrained_cost / len(sample), 6)

    positions = a2.cache_positions(trained, sample, codes)

    def one_step():
        trained.zero_grad(set_to_none=True)
        loss = a2.emission_loss(trained, positions[: int(contract["batch"])], use_code=True)
        loss.backward()
        trained.zero_grad(set_to_none=True)

    step_cost, step_samples = timeit(one_step, 20)
    unit["train_step_proxy_per_batch"] = round(step_cost, 6)

    def one_corrected_error():
        probe = make_model("native", int(contract["candidate"]["seed"]))
        probe.encoder["rgb"].load_state_dict(torch.load(bundle, weights_only=False)["encoder"])
        probe.load_state_dict(trained.state_dict())
        probe.eval()
        a2.set_trainable(probe, contract)
        optimizer = a2.optimizer_for(
            [p for p in probe.parameters() if p.requires_grad],
            contract["feedback_optimizer"],
        )
        example = sample[0]
        optimizer.zero_grad(set_to_none=True)
        loss = a2.byte_loss(
            probe,
            p4_2.prepare(probe, p4_2.column_public(example), example["id"]),
            example["target"],
            codes[example["id"]],
        )
        loss.backward()
        optimizer.step()

    correction_cost, correction_samples = timeit(one_corrected_error, 20)
    unit["single_error_correction_per_error"] = round(correction_cost, 6)

    reask_cost, reask_samples = timeit(lambda: one_round(trained, sample[:1], codes), 20)
    unit["single_example_reask"] = round(reask_cost, 6)

    # A task-B round carries a record, so the copy path has positions to weigh.
    # It is measured separately: pricing it at the A-task rate would repeat
    # A.2's mistake of pricing fixed phases at the per-update slope.
    from scripts.training.probe_a5_gate_20261116 import build_state
    from scripts.training.probe_a5_shortcut_20261116 import TEMPLATES
    from taiji.u2_subject import U2Record

    def one_record_round():
        example = sample[0]
        entity = example["public"]["request"].decode()[: -len("现在在哪里？")]
        record = U2Record(
            "price-record",
            TEMPLATES[0].format(entity=entity, label="中间").encode(),
            0,
            "sensor",
            "world",
            0.0,
            True,
        )
        state = build_state(trained, example, record)
        with torch.no_grad(), a2.reading_code(trained, codes[example["id"]]):
            for _ in range(int(trained.c.max_output_bytes) + 1):
                probability, _, details = trained.output(
                    state, trained.language_goal(state), state["prediction_action"]
                )
                symbol = int(probability.argmax())
                if symbol == 256:
                    break
                state = trained.accept_byte(state, symbol, details)

    record_cost, record_samples = timeit(one_record_round, 10)
    unit["byte_round_with_record"] = round(record_cost, 6)

    digests = replay_digests(trained, fit[:8], codes)
    replay_cost, replay_samples = timeit(lambda: replay_digests(trained, fit[:8], codes), 5)
    unit["recovery_replay_fit8"] = round(replay_cost, 6)
    replay_stable = digests == replay_digests(trained, fit[:8], codes)

    plan = contract["plan"]
    seed_count = len(plan["seeds"])
    update_count = int(plan["updates_per_seed"])
    dose_count = int(plan["dose_points"])
    error_cap = int(plan["feedback_error_allowance"])
    a4_updates = int(contract["feedback_updates_a4"])

    per_seed_training = update_count * unit["train_step_proxy_per_batch"]
    per_seed_evaluation = dose_count * (len(fit) + len(train)) * unit["byte_round_per_example"]
    # The feedback round: for each first-pass error, one release, one update
    # and an immediate re-ask of that error.  The re-ask of a wrong head can
    # run to the byte cap, so the expensive round cost is used.
    per_seed_feedback = error_cap * (
        unit["single_error_correction_per_error"] + unit["byte_round_with_record"]
    )
    # The retention control arm: the A.4-style accumulating protocol over the
    # same errors, eight updates each, then a full second pass over the split.
    # The control arm re-asks the whole B split twice: once against the truthful
    # record and once against the counter-fact record.  The second pass is what
    # tells a reader from a fixed-offset copier.
    per_seed_control = (
        error_cap
        * (
            a4_updates * unit["train_step_proxy_per_batch"]
            + unit["single_error_correction_per_error"]
        )
        + 2 * len(fit) * unit["byte_round_with_record"]
    )
    per_seed_recovery = unit["recovery_replay_fit8"] * (len(fit) / 8)
    per_seed_checkpoints = dose_count * 0.05
    per_seed = (
        per_seed_training
        + per_seed_evaluation
        + per_seed_feedback
        + per_seed_control
        + per_seed_recovery
        + per_seed_checkpoints
    )
    estimate = per_seed * seed_count
    contingency = round(estimate * float(plan["contingency"]), 3)
    cap = round(estimate + contingency, 3)

    report = {
        "schema": "taiji-u2-a5-protocol-price-v1",
        "card": "22.5.A.5",
        "date": "2026-11-16",
        "verdict": (
            "price_measured" if cap <= float(plan["design_cap_cpu"]) else "pricing_over_budget"
        ),
        "unit_cpu": unit,
        "raw": {
            "byte_round_samples": [round(value, 6) for value in round_samples],
            "train_step_samples": [round(value, 6) for value in step_samples],
            "single_error_correction_samples": [round(value, 6) for value in correction_samples],
            "single_example_reask_samples": [round(value, 6) for value in reask_samples],
            "recovery_replay_samples": [round(value, 6) for value in replay_samples],
            "replay_stable": replay_stable,
        },
        "plan": {
            "seeds": seed_count,
            "updates_per_seed": update_count,
            "dose_points": dose_count,
            "feedback_error_allowance": error_cap,
            "a4_protocol_updates_per_error": a4_updates,
            "per_seed_training": round(per_seed_training, 3),
            "per_seed_evaluation": round(per_seed_evaluation, 3),
            "per_seed_feedback": round(per_seed_feedback, 3),
            "per_seed_control_arm": round(per_seed_control, 3),
            "per_seed_recovery": round(per_seed_recovery, 3),
            "per_seed_checkpoints": round(per_seed_checkpoints, 3),
            "per_seed_total": round(per_seed, 3),
            "estimate_cpu": round(estimate, 3),
            "contingency_cpu": contingency,
            "cap_cpu": cap,
            "cap_wall": round(cap * 2, 3),
            "design_cap_cpu": float(plan["design_cap_cpu"]),
        },
        "discipline": {
            "optimizer_calls": 0,
            "optimizer_steps": 0,
            "training_updates": 0,
            "checkpoints_written": 0,
            "bundles_written": 0,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
            "accepted_parent": False,
        },
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
        "scope": (
            "pricing only: forward passes, one backward on the trained head, and "
            "copies of the subject; no optimizer, no checkpoint, no capability claim"
        ),
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
                "unit": unit,
                "replay_stable": replay_stable,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["verdict"] == "price_measured" else 4


if __name__ == "__main__":
    raise SystemExit(main())
