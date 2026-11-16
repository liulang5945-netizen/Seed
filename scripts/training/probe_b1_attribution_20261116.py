"""Zero-training attribution for the B.1 negative result.

B.1 measured retention_broken_under_separation: purity 176/176 (the separation
was physically executed), knockout restores the baseline (the damage is
adapter-mediated), yet two of three seeds lost 0.5 of A while correction stayed
at 0/0/2 of 24. This probe asks where the damage comes from, without a single
optimizer step:

* decompose the adapter's delta on record-free A episodes into the part driven
  by the record readout and the part driven by the context alone (zero the
  record half of the input and compare);
* measure whether the record readout separates presence at all (A episode vs
  the same example with a legally released record);
* measure whether the adapter's output differs between A and B contexts of the
  same example (conditioning);
* break the damaged A rounds down by answer class;
* report adapter weight norms per seed (did a seed learn ~nothing?).

Everything runs under no_grad on the saved post-feedback bundles.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_a5_22_5 as a5
from scripts.training import run_b1_separation_20261116 as b1
from taiji.u2_history import atomic_load

REPORT = ROOT / "reports/taiji_u2_b1_attribution_20261116.json"


def episode_vectors(model, public, identity, records, release, code):
    state = a5.episode(model, public, identity, records, release)
    goal = model.language_goal(state)
    with torch.no_grad(), a2.reading_code(model, code):
        context = model.branch_context(state, goal, state["prediction_action"])
        read, *_ = model.record_read(
            state, goal, state["previous_byte"], model.read_h(state, goal)[:, None]
        )
        read = read[:, 0]
        delta_full = model.correction_adapter(torch.cat([read, context], -1))
        delta_rec0 = model.correction_adapter(torch.cat([torch.zeros_like(read), context], -1))
    return read, delta_full, delta_rec0


def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float(numerator.norm() / denominator.norm().clamp_min(1e-12))


def main() -> int:
    torch.set_num_threads(1)
    cpu_start = time.process_time()
    contract = json.loads(b1.CONTRACT.read_text(encoding="utf-8"))
    data = atomic_load(ROOT / contract["data_root"])
    items, _ = a5.task_b_examples(data, contract)
    item_by_id = {item["example"]["id"]: item for item in items}
    template_of = b1.template_of_factory(contract)
    root = ROOT / contract["run_root"]
    seeds_out = []
    for seed in [int(value) for value in contract["plan"]["seeds"]]:
        arm = json.loads((root / f"seed-{seed}-learned.json").read_text(encoding="utf-8"))
        model, _ = b1.build_b1(contract, seed)
        model.load_state_dict(
            torch.load(root / f"seed-{seed}" / "bundle-post-feedback.pt", weights_only=False)[
                "model"
            ]
        )
        model.eval()
        codes = b1.codes_for(contract, model, data)
        a_vectors, b_vectors = {}, {}
        for example in data["fit"]:
            public = a5.p4_2.column_public(example)
            code = codes[example["id"]]
            a_vectors[example["id"]] = episode_vectors(
                model, public, example["id"], None, False, code
            )
            if example["id"] in item_by_id:
                item = item_by_id[example["id"]]
                index = items.index(item)
                record = a5.record_of(item["records"][template_of(index)], "a5-fact")
                b_vectors[example["id"]] = episode_vectors(
                    model, public, example["id"], [record], True, code
                )
        record_contribution = [ratio(full - rec0, full) for _, full, rec0 in a_vectors.values()]
        presence = [
            ratio(b_vectors[eid][0] - a_vectors[eid][0], a_vectors[eid][0]) for eid in b_vectors
        ]
        conditioning = [
            ratio(b_vectors[eid][1] - a_vectors[eid][1], a_vectors[eid][1]) for eid in b_vectors
        ]
        broken = {}
        for base, final in zip(arm["rows"]["a_baseline"], arm["rows"]["a_final"], strict=True):
            if base["correct"] and not final["correct"]:
                broken[base["target"]] = broken.get(base["target"], 0) + 1
        adapter = model.correction_adapter
        seeds_out.append(
            {
                "seed": seed,
                "damage": arm["damage"],
                "correction_rate": arm["second_pass_accuracy"],
                "a_record_input_contribution_ratio_mean": round(
                    sum(record_contribution) / len(record_contribution), 6
                ),
                "presence_separability_ratio_mean": round(sum(presence) / len(presence), 6),
                "ab_conditioning_ratio_mean": round(sum(conditioning) / len(conditioning), 6),
                "record_readout_norm_a_mean": round(
                    sum(float(v[0].norm()) for v in a_vectors.values()) / len(a_vectors), 6
                ),
                "record_readout_norm_b_mean": round(
                    sum(float(v[0].norm()) for v in b_vectors.values()) / len(b_vectors), 6
                ),
                "adapter_delta_norm_a_mean": round(
                    sum(float(v[1].norm()) for v in a_vectors.values()) / len(a_vectors), 6
                ),
                "adapter_delta_norm_b_mean": round(
                    sum(float(v[1].norm()) for v in b_vectors.values()) / len(b_vectors), 6
                ),
                "broken_a_by_class": broken,
                "adapter_weight_norms": {
                    "first_weight": round(float(adapter.first.weight.norm()), 6),
                    "last_weight": round(float(adapter.last.weight.norm()), 6),
                    "last_bias": round(float(adapter.last.bias.norm()), 6),
                },
            }
        )
    payload = {
        "schema": "taiji-u2-b1-attribution-v1",
        "card": "22.5.B.1",
        "date": "2026-11-16",
        "optimizer_steps": 0,
        "seeds": seeds_out,
        "cost": {"cpu": round(time.process_time() - cpu_start, 6)},
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
