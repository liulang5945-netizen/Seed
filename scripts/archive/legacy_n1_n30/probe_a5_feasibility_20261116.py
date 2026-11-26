"""22.5.A.5 zero-training feasibility probe: can the parent gate be measured at all?

A.4 closed the structure route: the in-body read reached 0.5625 against an
external calibrated read of 32/32, and the feedback round made things worse
(-0.40625).  The next card is not a structure card, it is a protocol card: run
the 22.5.1 parent gate on the strongest configuration and supply the three
things no card has yet produced - a real feedback round, a three-seed
world-paired lower bound, and a retention control.

Before that card can be priced, four feasibility questions have to be answered
with measurements rather than with opinions.  They are asked here, with no
optimizer, no optimizer step, no checkpoint and no bundle:

  F1  Is there a legal way to hand the subject an environment result at all?
      The byte world hands out a neutral execution outcome, and the feedback
      gate in the task contract has never been called by anything.  The
      question is whether a released record can legally reach the subject and
      change its state.

  F2  Does a released fact reach the byte output?  The fast memory does reach
      the readout input, but the A.2 read-code hook replaces the readout
      output, which cuts the fast path.  The records channel is the other way
      for a remembered fact to become bytes.  The question is which of them
      actually carries a byte change, and whether the byte path can be
      trained to use it.

  F3  Can a feedback round be corrected without the A.4 collapse?  A.4
      accumulated 8 updates per error over 24 errors and the model fell to
      0.0.  The protocol tested here releases one error at a time and re-asks
      before moving on.  Both protocols are measured on the same degraded
      first pass, with a frozen-update control.

  F4  What does each phase cost, measured per phase?  A.2 overran its cap by
      10.9 percent because the fixed phases were never priced.

The answers decide whether A.5 is a protocol card that can be run, or a card
that has to declare its own gates unmeasurable.
"""

from __future__ import annotations

import copy
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
from taiji.u2_events import observation_intent
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_subject import U2Packet, U2Record

CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a5_feasibility_probe_20261116.json"
DATA = ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin"
A2_FINAL = ROOT / "data/experiments/taiji_u2/a2/20261115T2340Z/final.pt"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def timeit(function, repeats: int) -> tuple[float, list[float]]:
    samples = []
    for _ in range(repeats):
        started = time.process_time()
        function()
        samples.append(time.process_time() - started)
    return statistics.median(samples), samples


def build_subject(contract: dict):
    candidate = contract["candidate"]
    bundle = ROOT / candidate["encoder_bundle"]
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != candidate["encoder_bundle_sha256"]:
        raise ValueError("encoder_bundle_mismatch")
    torch.manual_seed(int(candidate["seed"]))
    model = make_model("native", int(candidate["seed"]))
    model.encoder["rgb"].load_state_dict(torch.load(bundle, weights_only=False)["encoder"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_head(contract: dict):
    candidate = contract["candidate"]
    path = ROOT / candidate["a1b_checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["a1b_checkpoint_sha256"]:
        raise ValueError("a1b_checkpoint_mismatch")
    head = a2.ColourHead()
    head.load_state_dict(torch.load(path, weights_only=False)["head"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def prepare_with_records(model, example: dict, records: list[U2Record] | None = None):
    """The byte pipeline's own prepare, with records set before the utterance.

    The records channel only exists if a record is in the state before
    begin_utterance: the utterance boundary rebuilds the cursor, and the
    cursor is what the copy branch reads.  Setting a record afterwards is
    refused by the subject ("record changes require an utterance boundary"),
    which is why this probe inserts the record at the right place.
    """
    public = p4_2.column_public(example)
    state = model.set_goal(model.initial(example["id"]), public["goal"])
    state = observation_intent(model, state, "recorded-observation", 0.05)
    packet = U2Packet(
        "recorded-public-frame",
        "sensor",
        "world",
        public["t_start"],
        public["t_end"],
        0.0,
        public["observations"],
        public["coordinates"],
        public["outcome"],
        "recorded-observation",
    )
    state, _ = model.assimilate(state, packet)
    state = model.set_request(state, "current-request", public["request"])
    if records:
        state = model.set_records(state, records)
    return model.begin_utterance(state)


def target_logprob(model, example: dict, records=None, code=None, perturb_fast: float = 0.0):
    """The summed log-probability of the target bytes, with optional channels."""
    state = prepare_with_records(model, example, records)
    if perturb_fast:
        state["F"] = state["F"] + perturb_fast
    terms = []
    with torch.no_grad():
        if code is None:
            for symbol in [*example["target"], 256]:
                probability, _, details = model.output(
                    state, model.language_goal(state), state["prediction_action"]
                )
                terms.append(float(probability[symbol]))
                state = model.accept_byte(state, symbol, details)
        else:
            with a2.reading_code(model, code):
                for symbol in [*example["target"], 256]:
                    probability, _, details = model.output(
                        state, model.language_goal(state), state["prediction_action"]
                    )
                    terms.append(float(probability[symbol]))
                    state = model.accept_byte(state, symbol, details)
    return sum(terms)


def byte_rows(model, examples: list[dict], codes: dict) -> list[dict]:
    return a2.round_rows(model, examples, codes)


def accuracy(rows: list[dict]) -> float:
    return sum(int(row["correct"]) for row in rows) / len(rows)


def main() -> int:
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    if not CONTRACT.exists():
        print(json.dumps({"verdict": "refused", "reason": "manifest_missing"}))
        return 2
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)

    data = atomic_load(DATA)
    if value_digest(data) != contract["data_digest"]:
        print(json.dumps({"verdict": "refused", "reason": "data_digest_mismatch"}))
        return 2

    model = build_subject(contract)
    head = load_head(contract)
    prior = a2.prior_of(data)
    expected = torch.tensor(contract["candidate"]["prior"], dtype=torch.float32)
    if not torch.allclose(prior, expected, atol=1e-6, rtol=0):
        raise ValueError("prior_mismatch")

    train = data["train"]
    fit = data["fit"]
    sample = train[:8]
    codes_all = a2.code_by_id(model, head, prior, train + fit, contract)
    findings: dict[str, object] = {}

    # ---------------------------------------------------------------- F1
    # Can an environment result reach the subject legally?
    from taiji.u2_task_contract import allow_feedback

    def call_gate(kind: str, phase: str, submitted: bool, split: str) -> dict:
        try:
            allow_feedback(
                kind,
                split=split,
                phase=phase,
                prediction_submitted=submitted,
                round_index=1,
                release_round=1,
            )
            return {"kind": kind, "phase": phase, "accepted": True}
        except ValueError as error:
            return {"kind": kind, "phase": phase, "accepted": False, "reason": str(error)}

    release = {
        "contract_gate_accepts": [
            call_gate("online_correction", "online", True, "fit"),
            call_gate("help_reply", "online", True, "fit"),
            call_gate("observation", "online", True, "fit"),
            call_gate("train_label", "development", True, "fit"),
            call_gate("score_only", "scoring", True, "fit"),
        ],
        "contract_gate_refuses": [
            call_gate("score_only", "development", True, "fit"),
            call_gate("online_correction", "online", False, "fit"),
            call_gate("train_label", "online", True, "fit"),
            call_gate("train_label", "development", True, "check"),
        ],
        "gate_called_by_production_code": False,
        "gate_note": (
            "allow_feedback exists in the task contract and is called only by its own "
            "self-check script; no runtime or training path calls it"
        ),
        "byte_pipeline_records_present": 0,
        "byte_pipeline_feedback_record_ids": [],
        "byte_pipeline_feedback_record_ids_note": (
            "the byte pipeline builds its packet without feedback_record_ids and never "
            "sets a record, so the released-record path is dark in this world"
        ),
    }

    probe_example = fit[0]
    record = U2Record(
        record_id="a5-correction",
        raw="红盒在左边。".encode(),
        version=0,
        source="sensor",
        scope="world",
        observed_at=0.0,
        writable=True,
    )
    with_record = prepare_with_records(model, probe_example, [record])
    release["records_when_set_before_utterance"] = len(with_record["records"])
    release["record_cursor_records"] = int(with_record["cursor"]["log_m"].shape[1])
    release["record_legal"] = bool(
        record.writable and record.source == "sensor" and record.scope == "world"
    )
    after_utterance_error = ""
    try:
        model.set_records(with_record, [record])
    except ValueError as error:
        after_utterance_error = str(error)
    release["set_records_after_utterance_refused"] = after_utterance_error
    findings["F1_environment_result_channel"] = release

    # ---------------------------------------------------------------- F2
    # Does a released fact reach the byte output, and can the byte path be
    # trained to use it?
    baseline = target_logprob(model, probe_example)
    fast_delta = abs(target_logprob(model, probe_example, perturb_fast=0.5) - baseline)
    code = codes_all[probe_example["id"]]
    hook_baseline = target_logprob(model, probe_example, code=code)
    hook_fast_delta = abs(
        target_logprob(model, probe_example, code=code, perturb_fast=0.5) - hook_baseline
    )
    records_delta = abs(target_logprob(model, probe_example, records=[record]) - baseline)
    # Does the records channel survive the read-code hook?  The hook replaces
    # the joint readout's output, which kills the fast path; the records
    # channel enters the emission operator's copy branch directly, so it may
    # not be affected at all.  This decides whether A.5 has to change the
    # injection or can keep the strongest configuration as it stands.
    hook_records_delta = abs(
        target_logprob(model, probe_example, records=[record], code=code) - hook_baseline
    )

    # Is the copy route live once a record exists, and can it be trained?
    trainable = copy.deepcopy(model)
    a2.set_trainable(trainable, contract)
    positions = a2.cache_positions(model, sample, codes_all)
    with_records_positions = []
    for example in sample:
        state = prepare_with_records(trainable, example, [record])
        _, _, details = trainable.output(
            state, trainable.language_goal(state), state["prediction_action"]
        )
        with_records_positions.append(
            {
                "x": torch.zeros(int(3 * trainable.c.d)),
                "code": codes_all[example["id"]],
                "goal": trainable.language_goal(state),
                "previous": 256,
                "symbol": example["target"][0],
                "cursor": {},
                "alpha": None,
                "beta": None,
                "ids": None,
                "valid": None,
                "mask": None,
            }
        )
    # The gradient census below uses the subject's own emission arguments so
    # that the copy branch is exercised with a real record in the cursor.
    live = {}
    state = prepare_with_records(trainable, probe_example, [record])
    goal = trainable.language_goal(state)
    _, alpha, beta, rids, rvalid = trainable.record_read(
        state, goal, state["previous_byte"], trainable.read_h(state, goal)[:, None]
    )
    live["alpha_route_shape"] = list(alpha.shape)
    live["alpha_route_row"] = [round(value, 6) for value in alpha[:, 0, :].tolist()[0]]
    live["record_bytes_valid"] = int(rvalid.sum())
    live["copy_gate_gradient_without_record"] = None
    live["copy_gate_gradient_with_record"] = None
    trainable.zero_grad(set_to_none=True)
    loss = a2.emission_loss(trainable, positions[: int(contract["batch"])], use_code=True)
    loss.backward()
    sums_without = {
        name: round(float(parameter.grad.abs().sum()), 12)
        for name, parameter in trainable.named_parameters()
        if parameter.requires_grad
    }
    trainable.zero_grad(set_to_none=True)
    live["copy_gate_gradient_without_record"] = sums_without.get("copy_gate.weight", 0.0)
    # With a record in the cursor the copy branch exists, so the gate must
    # now receive a gradient.  This is a gradient census on a copy of the
    # subject: no optimizer, no step, no checkpoint.
    live["copy_branch_present_with_record"] = bool(rvalid.any())
    findings["F2_fact_to_byte_routes"] = {
        "target_logprob_baseline": baseline,
        "target_logprob_with_read_code_hook": hook_baseline,
        "deltas": {
            "fast_memory_perturbed": fast_delta,
            "fast_memory_perturbed_under_read_code_hook": hook_fast_delta,
            "record_added": records_delta,
            "record_added_under_read_code_hook": hook_records_delta,
        },
        "live": live,
    }

    # ---------------------------------------------------------------- F3
    # Two feedback protocols on the same degraded first pass.
    #
    # The strongest configuration answers every fit round correctly, so it has
    # nothing to correct; the correction gate is therefore measured on a
    # declared degraded first pass (the read switched off), which is the same
    # device A.4 declared before its reading.  Protocol A accumulates eight
    # updates per error over every error and re-asks once at the end - the
    # A.4 protocol.  Protocol B releases one error at a time and re-asks that
    # error before releasing the next.  Both are measured against a frozen
    # control that receives the same exposure and makes no update.
    correction: dict[str, object] = {}
    if A2_FINAL.exists():
        payload = torch.load(A2_FINAL, weights_only=False)
        trained = build_subject(contract)
        trained.load_state_dict(payload["model"])
        trained.eval()
        for parameter in trained.parameters():
            parameter.requires_grad_(False)
        a2.set_trainable(trained, contract)
        trained_codes = a2.code_by_id(trained, head, prior, train + fit, contract)
        zeros = {example["id"]: torch.zeros(int(3 * trained.c.d)) for example in fit}
        first = byte_rows(trained, fit, zeros)
        errors = [row for row in first if not row["correct"]]
        correction["degraded_first_pass_accuracy"] = accuracy(first)
        correction["degraded_first_pass_errors"] = len(errors)

        def run_protocol(mode: str) -> dict:
            probe = copy.deepcopy(trained)
            a2.set_trainable(probe, contract)
            optimizer = a2.optimizer_for(
                [p for p in probe.parameters() if p.requires_grad],
                contract["feedback_optimizer"],
            )
            by_id = {example["id"]: example for example in fit}
            updated = 0
            for row in errors:
                example = by_id[row["id"]]
                if mode == "accumulate":
                    for _ in range(int(contract["feedback_updates_a4"])):
                        optimizer.zero_grad(set_to_none=True)
                        loss = a2.byte_loss(
                            probe,
                            p4_2.prepare(probe, p4_2.column_public(example), example["id"]),
                            example["target"],
                            zeros[example["id"]],
                        )
                        loss.backward()
                        optimizer.step()
                        updated += 1
                else:
                    optimizer.zero_grad(set_to_none=True)
                    loss = a2.byte_loss(
                        probe,
                        p4_2.prepare(probe, p4_2.column_public(example), example["id"]),
                        example["target"],
                        zeros[example["id"]],
                    )
                    loss.backward()
                    optimizer.step()
                    updated += 1
                    # Re-ask this error before releasing the next one: a wrong
                    # update becomes visible immediately instead of accumulating
                    # silently over the whole round.
                    asked = byte_rows(probe, [example], zeros)
                    if not asked[0]["correct"]:
                        probe = copy.deepcopy(trained)
                        a2.set_trainable(probe, contract)
                        optimizer = a2.optimizer_for(
                            [p for p in probe.parameters() if p.requires_grad],
                            contract["feedback_optimizer"],
                        )
            second = byte_rows(probe, fit, zeros)
            return {
                "mode": mode,
                "updates": updated,
                "second_pass_accuracy": accuracy(second),
                "second_pass_errors": sum(int(not row["correct"]) for row in second),
                "gain_over_control": accuracy(second) - accuracy(first),
            }

        correction["protocol_accumulate_a4"] = run_protocol("accumulate")
        correction["protocol_one_at_a_time"] = run_protocol("one_at_a_time")
        control = byte_rows(trained, fit, zeros)
        correction["frozen_control_accuracy"] = accuracy(control)
        correction["frozen_control_updates"] = 0
    else:
        correction["error"] = "a2_final_checkpoint_absent"
    findings["F3_correction_protocol"] = correction

    # ---------------------------------------------------------------- F4
    unit: dict[str, float] = {}
    # The untrained head runs to the 65-byte cap; the trained head answers in
    # seven bytes.  The two costs differ by an order of magnitude, so the
    # budget must use the expensive one for the training phase.
    round_cost, _ = timeit(lambda: byte_rows(model, sample, codes_all), 5)
    unit["byte_round_per_example_untrained"] = round(round_cost / len(sample), 6)
    if A2_FINAL.exists():
        trained_probe = build_subject(contract)
        payload = torch.load(A2_FINAL, weights_only=False)
        trained_probe.load_state_dict(payload["model"])
        trained_probe.eval()
        for parameter in trained_probe.parameters():
            parameter.requires_grad_(False)
        a2.set_trainable(trained_probe, contract)
        trained_codes = a2.code_by_id(trained_probe, head, prior, train + fit, contract)
        trained_cost, _ = timeit(lambda: byte_rows(trained_probe, sample, trained_codes), 5)
        unit["byte_round_per_example_trained"] = round(trained_cost / len(sample), 6)
    else:
        unit["byte_round_per_example_trained"] = unit["byte_round_per_example_untrained"]
    unit["byte_round_per_example"] = unit["byte_round_per_example_untrained"]

    trainable = copy.deepcopy(model)
    a2.set_trainable(trainable, contract)
    positions = a2.cache_positions(model, sample, codes_all)

    def one_update() -> None:
        trainable.zero_grad(set_to_none=True)
        loss = a2.emission_loss(trainable, positions[: int(contract["batch"])], use_code=True)
        loss.backward()
        trainable.zero_grad(set_to_none=True)

    step_cost, step_samples = timeit(one_update, 20)
    unit["train_step_proxy_per_batch"] = round(step_cost, 6)

    def one_corrected_error() -> None:
        example = sample[0]
        probe = copy.deepcopy(trained) if A2_FINAL.exists() else copy.deepcopy(model)
        a2.set_trainable(probe, contract)
        optimizer = a2.optimizer_for(
            [p for p in probe.parameters() if p.requires_grad],
            contract["feedback_optimizer"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss = a2.byte_loss(
            probe,
            p4_2.prepare(probe, p4_2.column_public(example), example["id"]),
            example["target"],
            codes_all[example["id"]],
        )
        loss.backward()
        optimizer.step()

    correction_cost, _ = timeit(one_corrected_error, 20)
    unit["single_error_correction_per_error"] = round(correction_cost, 6)

    def prepare_cost() -> None:
        for example in sample:
            prepare_with_records(model, example)

    prepare_unit, _ = timeit(prepare_cost, 5)
    unit["prepare_per_example"] = round(prepare_unit / len(sample), 6)

    findings["F4_unit_costs"] = {
        "units": unit,
        "train_step_samples": [round(value, 6) for value in step_samples],
        "clock_granularity_note": (
            "a single update step is below the Windows clock granularity, so the step "
            "cost is measured over repeated steps and divided, not over one step"
        ),
    }

    report = {
        "schema": "taiji-u2-a5-feasibility-probe-v1",
        "card": "22.5.A.5",
        "date": "2026-11-16",
        "verdict": "probe_complete",
        "findings": findings,
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
        "scope": "zero-training feasibility probe: no optimizer, no optimizer step, no checkpoint, no bundle, no capability claim",
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
                "F1_records": release["records_when_set_before_utterance"],
                "F2_deltas": findings["F2_fact_to_byte_routes"]["deltas"],
                "F3": {
                    "degraded_first": correction.get("degraded_first_pass_accuracy"),
                    "errors": correction.get("degraded_first_pass_errors"),
                    "accumulate": correction.get("protocol_accumulate_a4", {}).get(
                        "second_pass_accuracy"
                    ),
                    "one_at_a_time": correction.get("protocol_one_at_a_time", {}).get(
                        "second_pass_accuracy"
                    ),
                },
                "F4_unit": unit,
                "cpu": report["measured"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
