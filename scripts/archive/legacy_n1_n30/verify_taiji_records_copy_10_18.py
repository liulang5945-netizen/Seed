"""10.18: can the subject copy an entity name out of a glossary record?

10.17 showed a state-feature readout does not compose descriptor -> entity. Before any
new mechanism is proposed, check the cheapest model-native question: given a prompt that
ends at the descriptor and a glossary line in ``records``, does the subject find the
entity name more likely?

* arm A: prompt ``"<descriptor>是"``, target ``"<entity>"``, records = full glossary;
* arm B: same, records = ();
* control: same prompt, target = a **different** entity -- records must not help there.

No parameter moves. If A does not beat B, the subject cannot even copy from records and
the two-stage route is dead on arrival.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.constrained_learning import career_nll  # noqa: E402
from taiji.native_goal_glossary import glossary_records, journal_with_records  # noqa: E402

BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
GAIN_FLOOR = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.18 card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)

    records = glossary_records()
    pairs = []
    for index, source in enumerate(records):
        line = source.raw.decode()
        descriptor, entity = line.rstrip("。").split("是", 1)
        wrong = records[(index + 1) % len(records)].raw.decode().rstrip("。").split("是", 1)[1]
        pairs.append({"descriptor": descriptor, "entity": entity, "wrong": wrong})

    rows = []
    gains = []
    control_gains = []
    with torch.no_grad():
        for index, pair in enumerate(pairs):
            prompt = f"{pair['descriptor']}是"
            with_records = float(
                career_nll(
                    subject,
                    journal_with_records(prompt, pair["entity"], f"a{index}", records),
                    provider,
                ).detach()
            )
            without_records = float(
                career_nll(
                    subject,
                    journal_with_records(prompt, pair["entity"], f"b{index}", ()),
                    provider,
                ).detach()
            )
            wrong_with = float(
                career_nll(
                    subject,
                    journal_with_records(prompt, pair["wrong"], f"c{index}", records),
                    provider,
                ).detach()
            )
            gains.append(without_records - with_records)
            control_gains.append(with_records - wrong_with)
            rows.append(
                {
                    "descriptor": pair["descriptor"],
                    "entity": pair["entity"],
                    "wrong": pair["wrong"],
                    "nll_with_records": with_records,
                    "nll_without_records": without_records,
                    "nll_wrong_target": wrong_with,
                    "copy_gain": without_records - with_records,
                }
            )

    mean_gain = sum(gains) / len(gains)
    improved = sum(1 for gain in gains if gain > 0)
    wrong_harder = sum(1 for gain in control_gains if gain > 0)
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif improved == len(gains) and mean_gain >= GAIN_FLOOR and wrong_harder == len(gains):
        status = "records_copy_supported"
    elif improved <= len(gains) // 2:
        status = "records_copy_absent"
    else:
        status = "records_copy_inconclusive"

    report = {
        "schema": "taiji-records-copy-10-18-v1",
        "step": "10.18",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "pairs": len(pairs),
            "prompt": "<descriptor>是",
            "target": "<entity>",
            "gain_floor": GAIN_FLOOR,
            "control": "same prompt, a different entity as target",
            "training": "none",
        },
        "parent_model_digest": digest,
        "mean_copy_gain": mean_gain,
        "improved": improved,
        "wrong_target_harder": wrong_harder,
        "rows": rows,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Copy check only: it asks whether records make the entity name more likely, "
            "not whether the subject selects an action. A pass means a two-stage design has "
            "a model-native first hop; it is not P2 and authorises nothing."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_glossary.py",
                "scripts/training/verify_taiji_records_copy_10_18.py",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "mean_copy_gain": round(mean_gain, 4),
                "improved": improved,
                "pairs": len(pairs),
                "wrong_target_harder": wrong_harder,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
