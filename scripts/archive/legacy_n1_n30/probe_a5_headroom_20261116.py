"""A.5 headroom probe: is a feedback gain even possible on this protocol?

The card's feedback-gain gate asks whether a subject that receives a legally
released environment fact answers better than a control that receives the same
release and makes no update.  That gate can be zero by construction in two
different ways, and both have bitten this line before:

* if the released fact does not reach the byte output at all, no update can
  help and the gain is structurally zero (A.2's copy gate was inert this way);
* if the released fact reaches the output so easily that the frozen subject
  already answers correctly, both arms improve equally and the gain is zero
  again - the same non-identifiability A.2 recorded for its correction gate.

The card's task B is a fact revision: the environment releases a record
stating a new position for the queried entity, the answer target is that new
position, and the first pass (which only sees the read) is wrong by
construction.  This probe measures the two ends of that protocol with no
training at all:

  first pass    the frozen subject, read code only, scored against the new fact
  with record   the same subject with the record legally released, same scoring

If the second number is high, there is no headroom and the card has to change
its task.  If it is low while the record is demonstrably in the state, there is
headroom and an update has something to buy.
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
REPORT = ROOT / "reports/taiji_u2_a5_headroom_probe_20261116.json"
DATA = ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin"
A2_FINAL = ROOT / "data/experiments/taiji_u2/a2/20261115T2340Z/final.pt"
LABELS = ("左边", "中间", "右边", "未知")
SUFFIX = "现在在哪里？"


def sha_bytes_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revised_label(true_label: str) -> str:
    """The new fact for task B: the entity has moved to the next position."""
    if true_label not in LABELS[:3]:
        raise ValueError("revised_fact_requires_a_position")
    return LABELS[(LABELS.index(true_label) + 1) % 3]


def build_state(model, example: dict, record: U2Record | None, release: bool):
    """Prepare the byte pipeline, optionally with a legally released record.

    The record has to be in the state before begin_utterance: the utterance
    boundary rebuilds the cursor, and the cursor is what the copy branch reads.
    The release then travels through the packet's feedback_record_ids, which is
    the only legal way for a record to reach the subject's fast memory.
    """
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
        feedback_record_ids=(record.record_id,) if (record is not None and release) else (),
    )
    state, _ = model.assimilate(state, packet)
    state = model.set_request(state, "current-request", public["request"])
    return model.begin_utterance(state)


def byte_round(model, example: dict, code: torch.Tensor, record: U2Record | None, release: bool):
    """One greedy full-response round, read code hooked, optional record."""
    state = build_state(model, example, record, release)
    cap = int(model.c.max_output_bytes) + 1
    content: list[int] = []
    eos = False
    with torch.no_grad(), a2.reading_code(model, code):
        for _ in range(cap):
            probability, _, details = model.output(
                state, model.language_goal(state), state["prediction_action"]
            )
            symbol = int(probability.argmax())
            if symbol == 256:
                eos = True
                break
            content.append(symbol)
            state = model.accept_byte(state, symbol, details)
    raw = bytes(content)
    try:
        raw.decode("utf-8", errors="strict")
        valid = True
    except UnicodeDecodeError:
        valid = False
    return {"raw": raw, "eos": eos, "valid": valid}


def copy_diagnostics(model, example: dict, code: torch.Tensor, record: U2Record):
    """How much mass the copy branch puts on the record, at the first position."""
    state = build_state(model, example, record, release=True)
    goal = model.language_goal(state)
    with torch.no_grad():
        _, alpha, beta, ids, valid = model.record_read(
            state, goal, state["previous_byte"], model.read_h(state, goal)[:, None]
        )
        probability, _, details = model.output(state, goal, state["prediction_action"])
        mixture = details["mixture"]
    route_to_record = float(alpha[0, 0, 0])
    route_to_null = float(alpha[0, 0, -1])
    copy_mass = float(mixture["log_copy"].exp().sum())
    gen_mass = float(mixture["log_gen"].exp().sum())
    return {
        "route_to_record": round(route_to_record, 6),
        "route_to_null": round(route_to_null, 6),
        "copy_mass": round(copy_mass, 6),
        "gen_mass": round(gen_mass, 6),
        "record_bytes_valid": int(valid.sum()),
        "record_ids": [int(value) for value in ids[0].tolist()],
        "top_symbols": [int(value) for value in torch.topk(probability, 3).indices.tolist()],
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
    if sha_bytes_of(bundle) != candidate["encoder_bundle_sha256"]:
        print(json.dumps({"verdict": "refused", "reason": "encoder_bundle_mismatch"}))
        return 2
    head_path = ROOT / candidate["a1b_checkpoint"]
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
    for example in fit:
        true_label = example["target"].decode()
        if true_label not in LABELS[:3]:
            continue  # an absent entity has no position to revise
        new_label = revised_label(true_label)
        entity = example["public"]["request"].decode()[: -len(SUFFIX)]
        record = U2Record(
            record_id=f"fact-{example['id']}",
            raw=f"{entity}现在在{new_label}。".encode(),
            version=0,
            source="sensor",
            scope="world",
            observed_at=0.0,
            writable=True,
        )
        code = codes[example["id"]]
        first = byte_round(model, example, code, None, False)
        with_record = byte_round(model, example, code, record, True)
        diagnostics = copy_diagnostics(model, example, code, record)
        rows.append(
            {
                "id": example["id"],
                "true_label": true_label,
                "new_label": new_label,
                "record_raw": record.raw.decode(),
                "record_bytes": len(record.raw),
                "answer_bytes": len(new_label.encode()),
                "first_pass_output": first["raw"].decode(errors="replace"),
                "first_pass_correct_on_new_fact": bool(first["raw"] == new_label.encode()),
                "with_record_output": with_record["raw"].decode(errors="replace"),
                "with_record_correct_on_new_fact": bool(with_record["raw"] == new_label.encode()),
                "with_record_contains_record": bool(record.raw in with_record["raw"]),
                "diagnostics": diagnostics,
            }
        )

    denominator = len(rows)
    first_hits = sum(int(row["first_pass_correct_on_new_fact"]) for row in rows)
    record_hits = sum(int(row["with_record_correct_on_new_fact"]) for row in rows)
    contains = sum(int(row["with_record_contains_record"]) for row in rows)
    copy_masses = [row["diagnostics"]["copy_mass"] for row in rows]
    gen_masses = [row["diagnostics"]["gen_mass"] for row in rows]

    report = {
        "schema": "taiji-u2-a5-headroom-probe-v1",
        "card": "22.5.A.5",
        "date": "2026-11-16",
        "verdict": "headroom_measured",
        "task_b": {
            "definition": (
                "fact revision: the environment releases a record stating the queried "
                "entity's new position; the answer target is that new position; the "
                "first pass only sees the read and is wrong by construction"
            ),
            "denominator": denominator,
            "denominator_note": "fit examples whose true label is a position; absent-entity examples excluded and counted",
            "excluded_absent": len(fit) - denominator,
        },
        "headroom": {
            "first_pass_correct": first_hits,
            "first_pass_accuracy": round(first_hits / denominator, 6),
            "with_record_correct": record_hits,
            "with_record_accuracy": round(record_hits / denominator, 6),
            "with_record_contains_record": contains,
            "headroom_points": round((record_hits - first_hits) / denominator, 6),
        },
        "copy_branch": {
            "median_copy_mass": round(float(torch.tensor(copy_masses).median()), 6),
            "median_gen_mass": round(float(torch.tensor(gen_masses).median()), 6),
            "note": (
                "copy mass is the total probability the emission mixture assigns to "
                "copying from the record; gen mass is the generator's share.  A copy "
                "mass near zero means the frozen subject does not use the record at all"
            ),
        },
        "rows": rows,
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
        "scope": "zero-training headroom probe: no optimizer, no step, no checkpoint, no capability claim",
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
                "denominator": denominator,
                "first_pass": f"{first_hits}/{denominator}",
                "with_record": f"{record_hits}/{denominator}",
                "contains_record": f"{contains}/{denominator}",
                "median_copy_mass": report["copy_branch"]["median_copy_mass"],
                "cpu": report["measured"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
