"""Zero-training unseen-template evaluation of the 22.5.1 joint-gate children.

The joint gate (fit face) was measured on the two seen templates.  This script
re-evaluates the three saved children and the accepted parent on the same fit 24
items using ONLY the unseen (third, inverted) template for B/D/multi records, and
reports both D references.  Zero optimizer steps, zero check/final reads, no
capability signing: bounded generalisation evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples as generate_v2  # noqa: E402
from scripts.training.s0_parent_loader_20261119 import (  # noqa: E402
    CHECKPOINTS,
    load_historical,
)
from taiji.u2_22_5_pipeline import (  # noqa: E402
    DEFAULT_TEMPLATES,
    build_b_items,
    decode_episode,
    make_record,
)
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_1_unseen_template_20261120.json"
ARTIFACTS = ROOT / "output/s2_joint_gate_20261120"
PARENT_SEED = 20261112
SEEDS = (20261110, 20261111, 20261112)
UNSEEN = DEFAULT_TEMPLATES[2]
SEEN = DEFAULT_TEMPLATES[:2]


def upper_bound(values: list[float], resamples: int = 10000, seed: int = 20261120) -> float:
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    draws = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.95))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    fit = generate_v2("fit", 32)
    items = build_b_items(fit, templates=SEEN)
    parent, _payload, _audit = load_historical(CHECKPOINTS[PARENT_SEED])
    parent_rows = []
    with torch.no_grad():
        for index, item in enumerate(items):
            ex = item["example"]
            other = items[(index + 1) % len(items)]
            other2 = items[(index + 2) % len(items)]
            rec_d = make_record(
                UNSEEN.format(entity=other["entity"], label=other["revised"]).encode(),
                f"unseen-d-{index}",
                version=1,
            )
            a_out, _ = decode_episode(parent, ex["public"], ex["id"], records=None)
            d_out, _ = decode_episode(parent, ex["public"], ex["id"], records=[rec_d])
            b_out, _ = decode_episode(parent, ex["public"], ex["id"], records=None)
            parent_rows.append(
                {
                    "a_output": a_out.hex(),
                    "a_truth": a_out == ex["target"],
                    "d_same_condition": d_out == a_out,
                    "d_truth": d_out == ex["target"],
                    "b_truth": b_out == ex["target"],
                }
            )
            del other2
    rows = []
    for seed in SEEDS:
        payload = torch.load(ARTIFACTS / f"child-{seed}.pt", map_location="cpu", weights_only=False)
        child = U2Subject(U2Config(**payload["config"]))
        child.load_state_dict(payload["state_dict"])
        child_rows = []
        with torch.no_grad():
            for index, item in enumerate(items):
                ex = item["example"]
                other = items[(index + 1) % len(items)]
                other2 = items[(index + 2) % len(items)]
                rec_b = make_record(
                    UNSEEN.format(entity=item["entity"], label=item["revised"]).encode(),
                    f"unseen-b-{index}",
                    version=1,
                )
                rec_d = make_record(
                    UNSEEN.format(entity=other["entity"], label=other["revised"]).encode(),
                    f"unseen-d-{index}",
                    version=1,
                )
                rec_d2 = make_record(
                    UNSEEN.format(entity=other2["entity"], label=other2["revised"]).encode(),
                    f"unseen-d2-{index}",
                    version=1,
                )
                b_out, b_eos = decode_episode(child, ex["public"], ex["id"], records=[rec_b])
                d_out, d_eos = decode_episode(child, ex["public"], ex["id"], records=[rec_d])
                m_out, m_eos = decode_episode(
                    child, ex["public"], ex["id"], records=[rec_d, rec_b, rec_d2]
                )
                child_rows.append(
                    {
                        "id": ex["id"],
                        "b": bool(b_eos and b_out == item["target"]),
                        "d_same_condition": bool(
                            d_eos and d_out == bytes.fromhex(parent_rows[index]["a_output"])
                        ),
                        "d_truth": bool(d_eos and d_out == ex["target"]),
                        "multi": bool(m_eos and m_out == item["target"]),
                    }
                )
        same_condition_drop = [
            float(parent_rows[index]["d_same_condition"])
            - float(child_rows[index]["d_same_condition"])
            for index in range(len(items))
        ]
        truth_drop = [
            float(parent_rows[index]["a_truth"]) - float(child_rows[index]["d_truth"])
            for index in range(len(items))
        ]
        rows.append(
            {
                "seed": seed,
                "b_correct": sum(1 for row in child_rows if row["b"]),
                "d_child_same_condition": sum(1 for row in child_rows if row["d_same_condition"]),
                "d_parent_same_condition": sum(1 for row in parent_rows if row["d_same_condition"]),
                "d_child_truth": sum(1 for row in child_rows if row["d_truth"]),
                "a_parent_truth": sum(1 for row in parent_rows if row["a_truth"]),
                "multi_correct": sum(1 for row in child_rows if row["multi"]),
                "same_condition_drop_upper_bound": round(upper_bound(same_condition_drop), 4),
                "truth_reference_drop_upper_bound": round(upper_bound(truth_drop), 4),
                "n": len(items),
            }
        )
    payload = {
        "schema": "taiji-u2-22-5-1-unseen-template-v1",
        "card": "22.5.1 joint-gate children, unseen-template probe (fit 24 items)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "face": {
            "templates_seen": list(SEEN),
            "template_unseen": UNSEEN,
            "items": len(items),
        },
        "parent": {
            "seed": PARENT_SEED,
            "checkpoint": str(CHECKPOINTS[PARENT_SEED].relative_to(ROOT)),
            "a_truth": sum(1 for row in parent_rows if row["a_truth"]),
        },
        "rows": rows,
        "scope": (
            "unseen TEMPLATE generalisation on the fit items; not a new task and not a "
            "check adjudication; zero training, zero check/final reads, no signing"
        ),
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(json.dumps({"rows": rows, "cpu": payload["cost"]["cpu"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
