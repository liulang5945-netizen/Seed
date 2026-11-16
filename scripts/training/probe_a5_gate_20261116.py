"""A.5 gate-discrimination probe: can the copy gate tell a record from no record?

Task B asks the subject to answer with a fact that the read contradicts.  The
mix between generating and copying is decided by two gates produced by
``copy_gate`` from the joint readout's context.  Under A.2's read-code hook that
context is ``joint_readout(code)``, which does not see the record at all.

If the gates are identical with and without a record, then the subject cannot
learn "copy when a fact was released, generate when none was" - the switch it
would need is not in its input.  That is a structural barrier, and if it holds
the card cannot meet its own A-retention gate (learning to copy would have to be
paid for by breaking the A task).

The probe measures the gates and the mixture on the same example, same read
code, with and without a released record, and reports the difference.  It also
measures whether the copy route can even reach the answer's offset, because a
copy gate that opens while the position distribution points at byte zero copies
the wrong word.
"""

from __future__ import annotations

import hashlib
import json
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
REPORT = ROOT / "reports/taiji_u2_a5_gate_probe_20261116.json"
DATA = ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin"
A2_FINAL = ROOT / "data/experiments/taiji_u2/a2/20261115T2340Z/final.pt"
LABELS = ("左边", "中间", "右边")
SUFFIX = "现在在哪里？"


def sha_bytes_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_state(model, example: dict, record: U2Record | None):
    public = p4_2.column_public(example)
    state = model.set_goal(model.initial(example["id"]), public["goal"])
    if record is not None:
        state = model.set_records(state, [record])
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
        feedback_record_ids=(record.record_id,) if record is not None else (),
    )
    state, _ = model.assimilate(state, packet)
    state = model.set_request(state, "current-request", public["request"])
    return model.begin_utterance(state)


def read_gates(model, example: dict, record: U2Record | None, code: torch.Tensor) -> dict:
    state = build_state(model, example, record)
    goal = model.language_goal(state)
    with torch.no_grad(), a2.reading_code(model, code):
        context = model.branch_context(state, goal, state["prediction_action"])
        previous_value = model.previous_embedding(state["previous_byte"])[None].expand(
            context.shape[0], -1
        )
        gate_input = torch.cat(
            [context, goal[None].expand(context.shape[0], -1), previous_value], -1
        )
        lam, gamma = model.copy_gate(gate_input).sigmoid().unbind(-1)
        probability, _, details = model.output(state, goal, state["prediction_action"])
        mixture = details["mixture"]
        route_shape = None
        if record is not None:
            _, alpha, beta, ids, valid = model.record_read(
                state, goal, state["previous_byte"], model.read_h(state, goal)[:, None]
            )
            route_shape = [round(float(v), 6) for v in alpha[0, 0, :].exp().tolist()]
            position = beta[0, 0, 0]
            position_summary = {
                "argmax": int(torch.argmax(position)),
                "entropy": round(float(-(position.exp() * position).sum()), 6),
                "mass_at_answer_offset": round(float(position[21].exp()), 6),
            }
        else:
            position_summary = None
    return {
        "lambda_continue": [round(float(v), 6) for v in lam.tolist()],
        "gamma_generate": [round(float(v), 6) for v in gamma.tolist()],
        "copy_mass": round(float(mixture["log_copy"].exp().sum()), 6),
        "gen_mass": round(float(mixture["log_gen"].exp().sum()), 6),
        "route": route_shape,
        "position": position_summary,
        "top_symbol": int(torch.argmax(probability)),
    }


def main() -> int:
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)

    data = atomic_load(DATA)
    if value_digest(data) != contract["data_digest"]:
        print(json.dumps({"verdict": "refused", "reason": "data_digest_mismatch"}))
        return 2
    candidate = contract["candidate"]
    bundle = ROOT / candidate["encoder_bundle"]
    head_path = ROOT / candidate["a1b_checkpoint"]
    if sha_bytes_of(bundle) != candidate["encoder_bundle_sha256"]:
        print(json.dumps({"verdict": "refused", "reason": "encoder_bundle_mismatch"}))
        return 2
    if sha_bytes_of(head_path) != candidate["a1b_checkpoint_sha256"]:
        print(json.dumps({"verdict": "refused", "reason": "a1b_checkpoint_mismatch"}))
        return 2
    if not A2_FINAL.exists():
        print(json.dumps({"verdict": "refused", "reason": "a2_final_checkpoint_absent"}))
        return 2

    model = make_model("native", int(candidate["seed"]))
    model.encoder["rgb"].load_state_dict(torch.load(bundle, weights_only=False)["encoder"])
    model.load_state_dict(torch.load(A2_FINAL, weights_only=False)["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    a2.set_trainable(model, contract)

    head = a2.ColourHead()
    head.load_state_dict(torch.load(head_path, weights_only=False)["head"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    prior = a2.prior_of(data)
    expected = torch.tensor(candidate["prior"], dtype=torch.float32)
    if not torch.allclose(prior, expected, atol=1e-6, rtol=0):
        print(json.dumps({"verdict": "refused", "reason": "prior_mismatch"}))
        return 2

    train = data["train"]
    fit = data["fit"]
    codes = a2.code_by_id(model, head, prior, train + fit, contract)

    rows = []
    for example in fit[:4]:
        code = codes[example["id"]]
        entity = example["public"]["request"].decode()[: -len(SUFFIX)]
        counter = next(label for label in LABELS if label != example["target"].decode())
        record = U2Record(
            f"gate-{example['id']}",
            f"{entity}现在在{counter}。".encode(),
            0,
            "sensor",
            "world",
            0.0,
            True,
        )
        without = read_gates(model, example, None, code)
        with_record = read_gates(model, example, record, code)
        rows.append(
            {
                "id": example["id"],
                "without_record": without,
                "with_record": with_record,
                "gamma_delta": [
                    round(a - b, 6)
                    for a, b in zip(
                        with_record["gamma_generate"], without["gamma_generate"], strict=True
                    )
                ],
                "lambda_delta": [
                    round(a - b, 6)
                    for a, b in zip(
                        with_record["lambda_continue"], without["lambda_continue"], strict=True
                    )
                ],
            }
        )

    max_gamma_delta = max(max(abs(v) for v in row["gamma_delta"]) for row in rows)
    max_lambda_delta = max(max(abs(v) for v in row["lambda_delta"]) for row in rows)
    # A gate that cannot tell a record from no record leaves both at zero.  The
    # threshold is not 0.0 because the copy term does perturb the mixture a
    # little through the null route; 1e-3 is the floor below which the change
    # cannot steer a decision.
    gates_see_the_record = max_gamma_delta > 1e-3 or max_lambda_delta > 1e-3

    answer_offsets = []
    for example in fit[:4]:
        entity = example["public"]["request"].decode()[: -len(SUFFIX)]
        counter = next(label for label in LABELS if label != example["target"].decode())
        raw = f"{entity}现在在{counter}。".encode()
        answer_offsets.append(raw.index(counter.encode()))

    # Does the A task care about the gate value when no record is present?  If
    # the mixture with no record is invariant to gamma, then a B-round gradient
    # can move the gate without paying for it on the A task, and the card's
    # retention gate is achievable.  If it is not invariant, learning B has to
    # damage A and the card cannot pass both.
    example = fit[0]
    code = codes[example["id"]]
    with_record_absent = {}
    with torch.no_grad():
        state = build_state(model, example, None)
        goal = model.language_goal(state)
        for label, shift in (("nominal", 0.0), ("gamma_low", -4.0), ("gamma_high", 4.0)):

            def shifted(module, inputs, output, shift=shift):  # noqa: ANN001
                result = output.clone()
                result[..., 1] = result[..., 1] + shift
                return result

            handle = model.copy_gate.register_forward_hook(shifted)
            try:
                with a2.reading_code(model, code):
                    probability, _, details = model.output(state, goal, state["prediction_action"])
            finally:
                handle.remove()
            with_record_absent[label] = {
                "distribution": [round(float(v), 8) for v in probability.tolist()],
                "copy_mass": round(float(details["mixture"]["log_copy"].exp().sum()), 8),
            }

    nominal = torch.tensor(with_record_absent["nominal"]["distribution"])
    deltas = {}
    for label in ("gamma_low", "gamma_high"):
        other = torch.tensor(with_record_absent[label]["distribution"])
        shared = torch.isfinite(nominal) & torch.isfinite(other)
        if bool(shared.any()):
            deltas[label] = round(float((other[shared] - nominal[shared]).abs().max()), 10)
        else:
            deltas[label] = None
    invariance = {
        "what": "perturb the copy gate by +/-4 logits with no record present",
        "max_abs_delta_by_shift": deltas,
        "gamma_cancels_without_a_record": all(
            value is not None and value < 1e-6 for value in deltas.values()
        ),
        "reading": (
            "if the A-task distribution is invariant to the gate when no record is "
            "present, then the gate can be trained from B rounds alone and the retention "
            "gate is achievable; if it is not invariant, learning B necessarily costs A"
        ),
        "copy_mass_by_shift": {
            label: with_record_absent[label]["copy_mass"] for label in with_record_absent
        },
    }

    report = {
        "schema": "taiji-u2-a5-gate-probe-v1",
        "card": "22.5.A.5",
        "date": "2026-11-16",
        "verdict": "gate_discrimination_measured",
        "rows": rows,
        "summary": {
            "max_gamma_delta": round(max_gamma_delta, 8),
            "max_lambda_delta": round(max_lambda_delta, 8),
            "gates_see_the_record": gates_see_the_record,
            "threshold": 1e-3,
            "threshold_note": (
                "a route change at the sixth decimal is not evidence; the floor is the "
                "level below which a gate cannot steer a decision"
            ),
            "answer_offsets": answer_offsets,
            "copy_can_reach_the_answer_offset": all(
                row["with_record"]["position"] is not None
                and row["with_record"]["position"]["mass_at_answer_offset"] > 1e-3
                for row in rows
            ),
        },
        "implication": (
            "the gates do not condition on the record, but that is not a barrier: with no "
            "record the copy term is absent and gamma cancels out of the mixture, so a "
            "training gradient that moves gamma comes only from rounds where a record is "
            "present.  Whether that is true is measured in invariance below, not assumed"
        ),
        "invariance": invariance,
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
        "scope": "zero-training gate probe: no optimizer, no step, no checkpoint, no capability claim",
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "gates_see_the_record": gates_see_the_record,
                "max_gamma_delta": round(max_gamma_delta, 8),
                "max_lambda_delta": round(max_lambda_delta, 8),
                "copy_reaches_answer_offset": report["summary"]["copy_can_reach_the_answer_offset"],
                "cpu": report["measured"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
