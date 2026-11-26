"""Zero-training grounding probe for the C (retrieval) line design.

B.1/B.2 located the joint-gate failure in the record representation/readout:
the value readout is untrained at start and byte CE never forces content use.
Before designing a retrieval mechanism, this probe measures what the EXISTING
record channel already does on the A.5 pre-feedback bundle, so the design rests
on measurement rather than assumption (A.5/B.1 showed multi-record routing is
query-blind; a query-match gate must not be assumed selective):

1. routing query selectivity: route_score and null logit for a matching-entity
   record vs a different-entity distractor, paired per example;
2. position attention: does the copy-branch attention mass point at the answer
   word, and does it differ matching vs distractor (entropy + answer mass);
3. value-pathway label structure: record_value(tokens) at the answer position -
   is there label separability even at birth (record_output is zero at start,
   so the readout itself carries nothing yet)?

Everything is no_grad on the frozen start bundle. No optimizer step.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_a5_22_5 as a5
from scripts.training import run_b1_separation_20261116 as b1
from taiji.u2_history import atomic_load
from taiji.u2_subject import record_inputs

REPORT = ROOT / "reports/taiji_u2_c1_feasibility_20261116.json"


def channel_vectors(model, state, code):
    """Replicate record_read's single-query internals to expose route/position/value."""
    from scripts.training import run_a2_byte_20261115 as a2

    goal = model.language_goal(state)
    c = model.c
    with torch.no_grad(), a2.reading_code(model, code):
        queries = model.read_h(state, goal)[:, None]
        tokens, ids, valid = model.record_units(state)
        goal_units = model.text_units(state["goal_raw"])
        if goal_units.numel():
            scores = model.record_goal_query(queries) @ goal_units.T / math.sqrt(c.d)
            goal_context = scores.softmax(-1) @ goal_units
        else:
            goal_context = goal[None, None].expand(c.branches, 1, -1)
        previous_value = model.previous_embedding(state["previous_byte"])[None, None].expand(
            c.branches, 1, -1
        )
        query = model.record_query(torch.cat([queries, goal_context, previous_value], -1))
        pools = tokens.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        route_score = query @ model.record_key(pools).T / math.sqrt(c.d)
        null = model.record_null(query)
        position_query = model.record_position_query(
            torch.cat(
                [
                    query[:, 0, None].expand(-1, len(pools), -1),
                    pools[None].expand(c.branches, -1, -1),
                ],
                -1,
            )
        )
        position_score = torch.einsum(
            "brd,rpd->brp", position_query, model.record_position_key(tokens)
        ) / math.sqrt(c.d)
        position_score = (
            position_score
            - torch.nn.functional.softplus(model.coverage_logit) * state["cursor"]["coverage"]
        )
        log_alpha, log_beta = record_inputs(
            torch.cat([route_score[:, 0], null[:, 0]], -1), position_score, valid
        )
        weights = (log_alpha[:, :-1, None] + log_beta).exp()
        values = model.record_value(tokens)
        context = (
            torch.einsum("brp,rpd->bd", weights, values)
            + log_alpha[:, -1:].exp() * model.record_null_value
        )
        read = model.record_output(context)
    return {
        "route_score": route_score[:, 0],
        "null": null[:, 0],
        "weights": weights[0, 0],
        "valid": valid[0],
        "values": values[0],
        "read_norm": float(read.norm()),
        "ids": ids[0],
    }


def answer_span(raw: bytes, label: str) -> tuple[int, int]:
    start = raw.index(label.encode())
    return start, start + len(label.encode())


def mass_entropy(vec, valid, span):
    w = vec[valid]
    total = w.sum().clamp_min(1e-12)
    p = w / total
    entropy = float(-(p * p.clamp_min(1e-12).log()).sum())
    lo, hi = span
    mask = torch.zeros_like(vec)
    mask[lo:hi] = 1.0
    mass = float((vec * mask).sum() / total)
    return mass, entropy


def main() -> int:
    torch.set_num_threads(1)
    cpu_start = time.process_time()
    contract = json.loads(b1.CONTRACT.read_text(encoding="utf-8"))
    data = atomic_load(ROOT / contract["data_root"])
    items, _ = a5.task_b_examples(data, contract)
    from scripts.training import run_b2_scoping_20261116 as b2

    distractors = b2.distractor_items(data, contract)
    distractor_by_id = {d["example"]["id"]: d for d in distractors}
    template_of = b1.template_of_factory(contract)
    seed = int(contract["plan"]["seeds"][0])
    model = b1.build_v4_reference(contract, seed)
    codes = b1.codes_for(contract, model, data)
    rows = []
    value_by_label = {}
    for index, item in enumerate(items):
        example = item["example"]
        public = a5.p4_2.column_public(example)
        code = codes[example["id"]]
        template = template_of(index)
        match_raw = item["records"][template]
        match_state = a5.episode(
            model, public, example["id"], [a5.record_of(match_raw, "a5-fact")], True
        )
        mv = channel_vectors(model, match_state, code)
        d = distractor_by_id[example["id"]]
        dist_state = a5.episode(
            model, public, example["id"], [a5.record_of(d["raw"], "b2-distractor")], True
        )
        dv = channel_vectors(model, dist_state, code)
        m_span = answer_span(match_raw, item["revised"])
        d_label = None
        for lab in a5.LABELS[:3]:
            if lab.encode() in d["raw"]:
                d_label = lab
                break
        d_span = answer_span(d["raw"], d_label)
        m_mass, m_ent = mass_entropy(mv["weights"], mv["valid"], m_span)
        d_mass, d_ent = mass_entropy(dv["weights"], dv["valid"], d_span)
        rows.append(
            {
                "id": example["id"],
                "truth": item["truth"],
                "revised": item["revised"],
                "route_match": float(mv["route_score"][0]),
                "route_distractor": float(dv["route_score"][0]),
                "route_delta": float(mv["route_score"][0] - dv["route_score"][0]),
                "null_match": float(mv["null"][0]),
                "null_distractor": float(dv["null"][0]),
                "answer_mass_match": round(m_mass, 6),
                "answer_mass_distractor": round(d_mass, 6),
                "position_entropy_match": round(m_ent, 6),
                "position_entropy_distractor": round(d_ent, 6),
                "read_norm_match": round(mv["read_norm"], 6),
            }
        )
        # value vector at the answer position, by stated label (matching records)
        lo, hi = m_span
        val = mv["values"][lo:hi].mean(0)
        value_by_label.setdefault(item["revised"], []).append(val)
    # label separability of value vectors at birth
    labels = sorted(value_by_label)
    centroids = {lab: torch.stack(value_by_label[lab]).mean(0) for lab in labels}
    between, within = [], []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            between.append(
                float(torch.nn.functional.cosine_similarity(centroids[a], centroids[b], 0))
            )
        for v in value_by_label[a]:
            within.append(float(torch.nn.functional.cosine_similarity(v, centroids[a], 0)))
    route_deltas = [r["route_delta"] for r in rows]
    payload = {
        "schema": "taiji-u2-c1-feasibility-v1",
        "card": "22.5.C.1",
        "date": "2026-11-16",
        "seed": seed,
        "optimizer_steps": 0,
        "n_pairs": len(rows),
        "route_delta_mean": round(sum(route_deltas) / len(route_deltas), 6),
        "route_delta_min": round(min(route_deltas), 6),
        "route_delta_max": round(max(route_deltas), 6),
        "route_match_gt_distractor": sum(int(d > 0) for d in route_deltas),
        "answer_mass_match_mean": round(sum(r["answer_mass_match"] for r in rows) / len(rows), 6),
        "answer_mass_distractor_mean": round(
            sum(r["answer_mass_distractor"] for r in rows) / len(rows), 6
        ),
        "position_entropy_match_mean": round(
            sum(r["position_entropy_match"] for r in rows) / len(rows), 6
        ),
        "read_norm_match_mean": round(sum(r["read_norm_match"] for r in rows) / len(rows), 6),
        "value_label_between_cosine_mean": (
            round(sum(between) / len(between), 6) if between else None
        ),
        "value_label_within_cosine_mean": round(sum(within) / len(within), 6) if within else None,
        "value_label_counts": {lab: len(value_by_label[lab]) for lab in labels},
        "rows": rows,
        "cost": {"cpu": round(time.process_time() - cpu_start, 6)},
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    summary = {k: v for k, v in payload.items() if k not in ("rows", "schema", "card", "date")}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
