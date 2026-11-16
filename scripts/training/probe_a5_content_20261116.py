"""A.5 content-sensitivity probe: does the record channel carry the fact at all?

The protocol-fit probe found the record routing is numerically query-blind: a
paraphrase and a different entity both leave the route mass unchanged beyond
the sixth decimal.  That makes the "unseen query" gate decorative on the frozen
subject, and it raises a sharper question that has to be answered before any
budget is spent:

    does the record channel react to *what the record says*, or only to its
    shape?

If it is shape-only, then whatever the subject learns is "copy bytes 15..21 of
whatever record is present" - which reproduces the answer for one template and
says nothing about carrying a fact.  That would make task B a fraud: the
"fact" would never be read, only its position.

The probe holds everything fixed - same example, same read code, same record
length, same template - and varies only the answer word inside the record.  It
then compares:

  C1  the byte distribution at the first position
  C2  the copy position distribution (where in the record the subject looks)
  C3  the route mass

A channel that carries a fact must move C1 and C2 when the word changes.  A
channel that is shape-only leaves both where they were.
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
REPORT = ROOT / "reports/taiji_u2_a5_content_probe_20261116.json"
DATA = ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin"
A2_FINAL = ROOT / "data/experiments/taiji_u2/a2/20261115T2340Z/final.pt"
LABELS = ("左边", "中间", "右边", "未知")
SUFFIX = "现在在哪里？"


def sha_bytes_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_state(model, example: dict, record: U2Record):
    public = p4_2.column_public(example)
    state = model.set_goal(model.initial(example["id"]), public["goal"])
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
        feedback_record_ids=(record.record_id,),
    )
    state, _ = model.assimilate(state, packet)
    state = model.set_request(state, "current-request", public["request"])
    return model.begin_utterance(state)


def probe_record(model, example: dict, record: U2Record, code: torch.Tensor) -> dict:
    """Read out the record channel for one record, at the first position."""
    state = build_state(model, example, record)
    goal = model.language_goal(state)
    with torch.no_grad():
        _, alpha, beta, ids, valid = model.record_read(
            state, goal, state["previous_byte"], model.read_h(state, goal)[:, None]
        )
        probability, _, details = model.output(state, goal, state["prediction_action"])
    mixture = details["mixture"]
    # The position distribution the copy branch would use for this record.
    position = beta[0, 0, 0]
    return {
        "record": record.raw.decode(),
        "top_symbols": [int(v) for v in torch.topk(probability, 3).indices.tolist()],
        "top_logprob": [round(float(v), 6) for v in torch.topk(probability, 3).values.tolist()],
        "answer_byte_logprob": round(float(probability[int(record.raw[21])]), 6),
        "route_mass": [round(float(v), 6) for v in alpha[0, 0, :].exp().tolist()],
        "position_entropy": round(float(-(position.exp() * position).sum()), 6),
        "position_argmax": int(torch.argmax(position)),
        "copy_mass": round(float(mixture["log_copy"].exp().sum()), 6),
        "gen_mass": round(float(mixture["log_gen"].exp().sum()), 6),
        "distribution": [round(float(v), 6) for v in probability.tolist()],
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
    example = fit[0]
    code = codes[example["id"]]
    entity = example["public"]["request"].decode()[: -len(SUFFIX)]

    # Same length, same template, only the answer word differs.
    variants = [
        U2Record(
            f"fact-{label}", f"{entity}现在在{label}。".encode(), 0, "sensor", "world", 0.0, True
        )
        for label in LABELS[:3]
    ]
    rows = [probe_record(model, example, record, code) for record in variants]

    # C1: does the first-position distribution move when the word changes?
    # The distribution carries -inf on masked symbols, so the comparison is
    # taken over the joint finite support: subtracting raw values would turn
    # every comparison into NaN and the check would silently pass as "no move".
    def finite_vector(row: dict) -> torch.Tensor:
        values = torch.tensor(row["distribution"])
        return values.masked_fill(~torch.isfinite(values), float("-inf"))

    base = finite_vector(rows[0])
    c1 = []
    for row in rows[1:]:
        other = finite_vector(row)
        shared = torch.isfinite(base) & torch.isfinite(other)
        if not bool(shared.any()):
            c1.append(
                {
                    "record": row["record"],
                    "max_abs_delta": None,
                    "l1_delta": None,
                    "support_overlap": 0,
                }
            )
            continue
        delta = (other[shared] - base[shared]).abs()
        c1.append(
            {
                "record": row["record"],
                "max_abs_delta": round(float(delta.max()), 10),
                "l1_delta": round(float(delta.sum()), 10),
                "support_overlap": int(shared.sum()),
            }
        )
    c1_moves = any(
        item["max_abs_delta"] is not None and item["max_abs_delta"] > 1e-4 for item in c1
    )

    # C2: does the copy position distribution move?
    c2 = [
        {
            "record": row["record"],
            "position_entropy": row["position_entropy"],
            "position_argmax": row["position_argmax"],
        }
        for row in rows
    ]
    c2_moves = (
        len({item["position_argmax"] for item in c2}) > 1
        or len({item["position_entropy"] for item in c2}) > 1
    )

    # C3: does the route mass move?
    c3_moves = len({tuple(row["route_mass"]) for row in rows}) > 1

    # Is the answer-carrying byte even addressable?  The record ends with the
    # answer word plus a full stop, so its first byte sits at a fixed offset
    # from the end for every variant of the same length.
    offsets = [len(record.raw) for record in variants]

    report = {
        "schema": "taiji-u2-a5-content-probe-v1",
        "card": "22.5.A.5",
        "date": "2026-11-16",
        "verdict": "content_sensitivity_measured",
        "design": {
            "question": "does the record channel react to what the record says, or only to its shape?",
            "fixed": ["example", "read code", "record template", "record length"],
            "varied": "the answer word inside the record",
        },
        "rows": rows,
        "C1_distribution_moves": c1_moves,
        "C1_deltas": c1,
        "C2_position_moves": c2_moves,
        "C2_positions": c2,
        "C3_route_moves": c3_moves,
        "record_byte_offsets": offsets,
        "threshold_note": (
            "1e-4 is used as the floor for a real move: the protocol-fit probe showed "
            "route changes at the sixth decimal that are numerically nonzero but "
            "operationally meaningless, so a bare inequality is not evidence"
        ),
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
        "scope": "zero-training content probe: no optimizer, no step, no checkpoint, no capability claim",
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "C1_distribution_moves": c1_moves,
                "C1_max_delta": max(item["max_abs_delta"] for item in c1),
                "C2_position_moves": c2_moves,
                "C2_argmax": [item["position_argmax"] for item in c2],
                "C3_route_moves": c3_moves,
                "note": (
                    "a shape-only channel leaves C1 and C2 where they were, which would "
                    "make task B a copying exercise rather than a fact test"
                ),
                "cpu": report["measured"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
