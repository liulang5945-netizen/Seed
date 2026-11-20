"""23.1 data contract: generate composition partitions, check leakage and freeze digests (0 check).

Builds the `taiji-23-composition-v1` partitions (train 512 / fit 128 / check 128 /
final 128 worlds; 3 items per world = 2,688 items), verifies that no
(template_group, direction, pair_order) triple leaks from train into the evaluation
partitions, re-checks the symbolic oracle, writes the four partition digests and emits a
frozen contract manifest.  Zero training, zero check/final reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from taiji.native_composition import (  # noqa: E402
    FIXTURE_ID,
    _relation,
    build_partition,
    partition_digest,
    world_scene,
)

OUT = ROOT / "reports/taiji_u2_23_1_contract_20261120.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_23_1_composition_contract_20261120.json"
PARTITIONS = ("train", "fit", "check", "final")
ALLOWED_ANSWERS = {"左边", "中间", "右边", "未知"}


def axes_triples(rows: list[dict[str, Any]]) -> set:
    return {
        (
            row["kind"],
            row["axes"]["template_group"],
            row["axes"]["direction"],
            str(row["axes"]["pair_order"]),
        )
        for row in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    rows = {partition: build_partition(partition) for partition in PARTITIONS}
    digests = {partition: partition_digest(partition) for partition in PARTITIONS}
    digests_again = {partition: partition_digest(partition) for partition in PARTITIONS}
    reproducible = digests == digests_again
    triples = {partition: axes_triples(rows[partition]) for partition in PARTITIONS}
    leakage = any(triples["train"] & triples[partition] for partition in ("fit", "check", "final"))
    # Symbolic oracle re-check: recompute Q2/Q3 answers directly from the scene.
    oracle_total = 0
    oracle_correct = 0
    for partition in PARTITIONS:
        for row in rows[partition]:
            if row["kind"] not in {"q2", "q3"}:
                continue
            scene = world_scene(row["seed"])
            bodies = scene["bodies"]
            entity = bodies[row["seed"] % len(bodies)]["name"]
            direction = row["axes"]["direction"]
            if row["kind"] == "q2":
                expected = _relation(entity, direction, bodies)
                expected = expected if expected is not None else "未知"
            else:
                d1, d2 = direction, row["axes"]["pair_order"][1]
                first = _relation(entity, d1, bodies)
                second = _relation(first, d2, bodies) if first is not None else None
                expected = second if second is not None else "未知"
            oracle_total += 1
            oracle_correct += int(expected == row["answer"])
    oracle_accuracy = oracle_correct / max(1, oracle_total)
    text_ok = all(
        isinstance(row["text"], str) and row["text"].encode("utf-8").decode("utf-8") == row["text"]
        for partition in PARTITIONS
        for row in rows[partition]
    )
    answers_ok = all(
        row["answer"] in ALLOWED_ANSWERS or row["answer"].endswith("圆片")
        for partition in PARTITIONS
        for row in rows[partition]
    )
    counts = {partition: len(rows[partition]) for partition in PARTITIONS}
    gates = {
        "c1_sizes": counts == {"train": 1536, "fit": 384, "check": 384, "final": 384},
        "c2_no_leakage": not leakage,
        "c3_oracle": oracle_accuracy == 1.0,
        "c4_reproducible": reproducible,
        "c5_text": text_ok and answers_ok,
    }
    verdict = "data_contract_frozen" if all(gates.values()) else "data_contract_invalid"
    criteria = {
        "fixture": FIXTURE_ID,
        "partitions": PARTITIONS,
        "counts": counts,
        "digests": digests,
        "oracle_accuracy": oracle_accuracy,
        "leakage": leakage,
    }
    contract = {
        "schema": "taiji-u2-23-1-composition-contract-v1",
        "card": "23.1 composition data contract",
        "status": "frozen" if verdict == "data_contract_frozen" else "invalid",
        "criteria": criteria,
        "criteria_sha256": hashlib.sha256(
            json.dumps(criteria, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "check_reads": 0,
        "final_reads": 0,
    }
    CONTRACT.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    payload = {
        "schema": "taiji-u2-23-1-contract-v1",
        "card": "23.1 composition data contract generation and acceptance",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "fixture": FIXTURE_ID,
            "worlds": 896,
            "items_per_world": 3,
            "partitions": PARTITIONS,
        },
        "counts": counts,
        "digests": digests,
        "reproducible": reproducible,
        "leakage": leakage,
        "oracle_accuracy": oracle_accuracy,
        "gates": gates,
        "verdict": verdict,
        "contract": str(CONTRACT.relative_to(ROOT)),
        "discipline": {"check_reads": 0, "final_reads": 0, "training_steps": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "counts": counts,
                "oracle": oracle_accuracy,
                "cpu": payload["cost"]["cpu"],
                "digests": digests,
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "data_contract_frozen" else 1


if __name__ == "__main__":
    raise SystemExit(main())
