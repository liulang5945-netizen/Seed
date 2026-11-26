"""A.5 shortcut probe: can task B be passed without reading the fact?

The content probe showed the record channel does react to the answer word, so
the channel is not shape-only.  It also showed something worse for the
protocol: every entity name in this fixture is twelve bytes, and the record
template is fixed, so the answer word always sits at byte offset 21 of a
thirty-byte record.

That makes a degenerate policy available: "copy bytes 21..27 of whatever record
is present".  It passes the gate, it never reads the fact, and it survives a
record whose answer is wrong.  A gate that can be passed that way is decorative.

This probe measures the shortcut and tests the fix before the card is priced:

  S1  offset spread under the current single template        (expect zero)
  S2  offset spread under a small set of templates           (expect spread)
  S3  does a counter-fact record move the output?  A subject that reads the
      record must follow it; a subject that copies a fixed offset cannot.

S3 is measured on the frozen subject, so the expected answer is "no" - the
point is that the *protocol* can tell the two policies apart, which is what the
counter-fact arm in the card is for.
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
from taiji.u2_task_episode import NAMES

CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a5_shortcut_probe_20261116.json"
DATA = ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin"
A2_FINAL = ROOT / "data/experiments/taiji_u2/a2/20261115T2340Z/final.pt"
LABELS = ("左边", "中间", "右边")
SUFFIX = "现在在哪里？"

# Four templates whose answer word lands at four different byte offsets, so a
# fixed-offset copy cannot pass.  Every template states the same fact.
TEMPLATES = (
    "{entity}现在在{label}。",
    "现在在{label}的是{entity}。",
    "{label}就是{entity}此刻的位置。",
    "据最新观测，{entity}的位置是{label}。",
)


def sha_bytes_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_state(model, example: dict, record: U2Record, request: bytes):
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
    state = model.set_request(state, "current-request", request)
    return model.begin_utterance(state)


def byte_round(model, example: dict, record: U2Record, code: torch.Tensor, request: bytes) -> bytes:
    state = build_state(model, example, record, request)
    cap = int(model.c.max_output_bytes) + 1
    content: list[int] = []
    with torch.no_grad(), a2.reading_code(model, code):
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

    # S1: under the single template, is the answer offset constant?
    single = []
    for entity in NAMES:
        for label in LABELS:
            raw = TEMPLATES[0].format(entity=entity, label=label).encode()
            single.append(
                {
                    "entity": entity,
                    "label": label,
                    "offset": raw.index(label.encode()),
                    "length": len(raw),
                }
            )
    single_offsets = sorted({item["offset"] for item in single})

    # S2: under the four templates, does the offset spread?
    spread = []
    for template in TEMPLATES:
        for entity in (NAMES[0],):
            for label in (LABELS[0],):
                raw = template.format(entity=entity, label=label).encode()
                spread.append(
                    {
                        "template": template,
                        "record": raw.decode(),
                        "offset": raw.index(label.encode()),
                        "length": len(raw),
                    }
                )
    spread_offsets = sorted({item["offset"] for item in spread})

    # S3: does a counter-fact record move the frozen subject's output?
    example = fit[0]
    entity = example["public"]["request"].decode()[: -len(SUFFIX)]
    code = codes[example["id"]]
    request = example["public"]["request"]
    truth = example["target"].decode()
    counter_label = next(label for label in LABELS if label != truth)
    truthful = U2Record(
        "fact-true",
        TEMPLATES[0].format(entity=entity, label=truth).encode(),
        0,
        "sensor",
        "world",
        0.0,
        True,
    )
    counter = U2Record(
        "fact-counter",
        TEMPLATES[0].format(entity=entity, label=counter_label).encode(),
        0,
        "sensor",
        "world",
        0.0,
        True,
    )
    out_truthful = byte_round(model, example, truthful, code, request)
    out_counter = byte_round(model, example, counter, code, request)
    s3 = {
        "truth": truth,
        "counter_label": counter_label,
        "output_with_truthful_record": out_truthful.decode(errors="replace"),
        "output_with_counter_record": out_counter.decode(errors="replace"),
        "output_follows_record": out_truthful != out_counter,
        "note": (
            "a subject that reads the record must answer differently when the record "
            "says something different; a fixed-offset copier answers the same either way. "
            "The frozen subject is expected NOT to follow the record - that is the "
            "headroom - so this number reports the protocol's ability to tell the two "
            "policies apart, not a capability"
        ),
    }

    report = {
        "schema": "taiji-u2-a5-shortcut-probe-v1",
        "card": "22.5.A.5",
        "date": "2026-11-16",
        "verdict": "shortcut_measured",
        "S1_fixed_template": {
            "offset_values": single_offsets,
            "offset_is_constant": len(single_offsets) == 1,
            "records": single[:3],
            "risk": (
                "with every entity name at twelve bytes and one template, the answer "
                "always sits at byte 21 of a thirty-byte record, so 'copy bytes 21..27' "
                "passes the gate without reading the fact"
            ),
        },
        "S2_template_spread": {
            "templates": list(TEMPLATES),
            "offset_values": spread_offsets,
            "offset_spread": len(spread_offsets),
            "records": spread,
            "fix": (
                "four templates put the answer at four different offsets, so a "
                "fixed-offset copy cannot pass; the subject must locate the answer in "
                "the record it was given"
            ),
        },
        "S3_counter_fact_arm": s3,
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
        "scope": "zero-training shortcut probe: no optimizer, no step, no checkpoint, no capability claim",
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "S1_offsets": single_offsets,
                "S1_constant": len(single_offsets) == 1,
                "S2_offsets": spread_offsets,
                "S2_spread": len(spread_offsets),
                "S3_follows_record": s3["output_follows_record"],
                "S3_truthful": s3["output_with_truthful_record"],
                "S3_counter": s3["output_with_counter_record"],
                "cpu": report["measured"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
