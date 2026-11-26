"""A.5 protocol-fit probe: unseen query and cross-session persistence.

22.5.1 asks for three things the A.5 card has to measure, not assume:

  * ``B采用新任务/事实版本，反馈后在未见query评分`` - the score is taken on a query
    the subject has not answered before, not on the query it just failed;
  * ``跨session保存F/E`` - the released fact survives a session boundary and is
    still usable afterwards;
  * the fact is *discriminated*: with more than one fact released, the route
    has to pick the one the query is about.

The headroom probe answered "is there anything to learn at all".  This probe
answers the three protocol questions above, still with no training:

  U1  with a released record, does an unseen paraphrase of the request reach
      the same answer the original wording would?  (routing by wording)
  U2  does the released record survive a save/reload boundary and still route?
  U3  with two records released, does the route mass differ by query entity?

If U1 fails, the "unseen query" gate cannot be met by this pipeline and the
card has to say so; if U2 fails, "cross-session F/E" cannot be met; if U3
fails, a multi-fact B task is not constructible and the card is single-fact.
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
REPORT = ROOT / "reports/taiji_u2_a5_protocol_fit_probe_20261116.json"
DATA = ROOT / "data/experiments/taiji_u2/first_a/20261112T074836Z/data.bin"
A2_FINAL = ROOT / "data/experiments/taiji_u2/a2/20261115T2340Z/final.pt"
LABELS = ("左边", "中间", "右边", "未知")
SUFFIX = "现在在哪里？"


def sha_bytes_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_state(model, example: dict, records, release: bool, request: bytes | None = None):
    public = p4_2.column_public(example)
    state = model.set_goal(model.initial(example["id"]), public["goal"])
    if records:
        state = model.set_records(state, records)
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
        feedback_record_ids=tuple(r.record_id for r in records) if (records and release) else (),
    )
    state, _ = model.assimilate(state, packet)
    state = model.set_request(state, "current-request", request or public["request"])
    return model.begin_utterance(state)


def route_masses(model, example: dict, records, request: bytes | None = None):
    """Route mass over records plus the null route, and the copy share."""
    state = build_state(model, example, records, release=True, request=request)
    goal = model.language_goal(state)
    with torch.no_grad():
        _, alpha, beta, ids, valid = model.record_read(
            state, goal, state["previous_byte"], model.read_h(state, goal)[:, None]
        )
        probability, _, details = model.output(state, goal, state["prediction_action"])
    mixture = details["mixture"]
    return {
        "route_probabilities": [round(float(value), 6) for value in alpha[0, 0, :].exp().tolist()],
        "copy_mass": round(float(mixture["log_copy"].exp().sum()), 6),
        "gen_mass": round(float(mixture["log_gen"].exp().sum()), 6),
        "record_ids": [int(value) for value in ids[0].tolist()],
        "first_symbol": int(torch.argmax(probability)),
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

    fit = data["fit"]
    example = fit[0]
    entity = example["public"]["request"].decode()[: -len(SUFFIX)]
    record = U2Record(
        record_id="fact-a",
        raw=f"{entity}现在在中间。".encode(),
        version=0,
        source="sensor",
        scope="world",
        observed_at=0.0,
        writable=True,
    )
    second = U2Record(
        record_id="fact-b",
        raw="另一个物体现在在左边。".encode(),
        version=0,
        source="sensor",
        scope="world",
        observed_at=0.0,
        writable=True,
    )

    # U1: does the request wording change the routing of the same record?
    original = example["public"]["request"]
    paraphrase = f"{entity}现在位于哪里？".encode()
    third_person = f"请问{entity}现在在哪里？".encode()
    u1 = {
        "original_request": original.decode(),
        "original_route": route_masses(model, example, [record], original),
        "paraphrase_request": paraphrase.decode(),
        "paraphrase_route": route_masses(model, example, [record], paraphrase),
        "prefixed_request": third_person.decode(),
        "prefixed_route": route_masses(model, example, [record], third_person),
    }
    u1["route_changes_with_wording"] = (
        u1["original_route"]["route_probabilities"][0]
        != u1["paraphrase_route"]["route_probabilities"][0]
    )

    # U2: does the released record survive a save/reload boundary?
    state = build_state(model, example, [record], release=True)
    saved = {
        "records": state["records"],
        "released_feedback_records": state["released_feedback_records"],
        "F": state["F"].detach().clone(),
        "subject_id": state["subject_id"],
    }
    buffer = __import__("io").BytesIO()
    torch.save(saved, buffer)
    reloaded = torch.load(__import__("io").BytesIO(buffer.getvalue()), weights_only=False)
    u2 = {
        "records_survive": len(reloaded["records"]) == 1,
        "record_raw_equal": reloaded["records"][0].raw == record.raw,
        "released_ids_survive": tuple(reloaded["released_feedback_records"]) == ("fact-a",),
        "fast_memory_equal": bool(torch.equal(reloaded["F"], state["F"])),
    }
    # Rebuild a fresh subject and re-set the reloaded record: does routing work
    # after the boundary, or does it need the original state?
    after = build_state(model, example, reloaded["records"], release=False)
    goal_after = model.language_goal(after)
    with torch.no_grad():
        _, alpha_after, _, _, valid_after = model.record_read(
            after, goal_after, after["previous_byte"], model.read_h(after, goal_after)[:, None]
        )
    u2["routing_after_reload"] = [round(float(v), 6) for v in alpha_after[0, 0, :].exp().tolist()]
    u2["record_bytes_after_reload"] = int(valid_after.sum())

    # U3: with two records, does the route differ by which entity is asked?
    both = [record, second]
    entity_a_query = route_masses(model, example, both, example["public"]["request"])
    other_public = dict(example["public"])
    other_public["request"] = "另一个物体现在在哪里？".encode()
    other_example = {**example, "public": other_public}
    entity_b_query = route_masses(model, other_example, both, other_public["request"])
    u3 = {
        "two_records": len(both),
        "route_for_first_entity": entity_a_query["route_probabilities"],
        "route_for_second_entity": entity_b_query["route_probabilities"],
        "route_differs_by_entity": (
            entity_a_query["route_probabilities"] != entity_b_query["route_probabilities"]
        ),
        "note": (
            "three entries are route-to-record-a, route-to-record-b and the null route; "
            "a route that does not differ by entity means multi-fact B is not constructible"
        ),
    }

    report = {
        "schema": "taiji-u2-a5-protocol-fit-probe-v1",
        "card": "22.5.A.5",
        "date": "2026-11-16",
        "verdict": "protocol_fit_measured",
        "U1_unseen_query": u1,
        "U2_cross_session": u2,
        "U3_multi_fact_discrimination": u3,
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
        "scope": "zero-training protocol-fit probe: no optimizer, no step, no checkpoint, no capability claim",
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "U1_wording_changes_route": u1["route_changes_with_wording"],
                "U1_original": u1["original_route"]["route_probabilities"],
                "U1_paraphrase": u1["paraphrase_route"]["route_probabilities"],
                "U2": u2,
                "U3_differs_by_entity": u3["route_differs_by_entity"],
                "U3_a": u3["route_for_first_entity"],
                "U3_b": u3["route_for_second_entity"],
                "cpu": report["measured"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
