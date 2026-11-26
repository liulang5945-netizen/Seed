"""P3.18 split-matched data: 96 three-object scenes split 64/32.

Reuses the frozen fixture generator from the first-A support module, but draws
both partitions from the three-object regime so train and fit share one
distribution. Writes a new data file; the original data.bin is untouched.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import generate_examples
from taiji.u2_history import atomic_save, value_digest

OUT_DATA = ROOT / "data/experiments/taiji_u2/p3_18/three_object_split_20261114.bin"
OUT_REPORT = ROOT / "reports/taiji_u2_p3_18_split_20261114.json"


def main() -> int:
    pool = generate_examples("fit", 96)
    train, fit = pool[:64], pool[64:]
    for index, example in enumerate(train):
        example["id"] = f"train-{index}"
    for index, example in enumerate(fit):
        example["id"] = f"fit-{index}"
    data = {"train": train, "fit": fit}
    digest = value_digest(data)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(data, OUT_DATA)
    report = {
        "schema": "taiji-u2-p3-18-split-v1",
        "date": "2026-11-14",
        "card": "22.5.P3.18",
        "source": "generate_examples('fit', 96) from scripts/training/first_a_support_22_5.py",
        "objects_per_scene": 3,
        "split": {"train": len(train), "fit": len(fit)},
        "data_path": str(OUT_DATA.relative_to(ROOT)).replace("\\", "/"),
        "data_digest": digest,
        "label_histogram_train": dict(
            collections.Counter(example["target"].decode() for example in train)
        ),
        "label_histogram_fit": dict(
            collections.Counter(example["target"].decode() for example in fit)
        ),
        "discipline": {"training_runs": 0, "optimizer_calls": 0, "model_forward_passes": 0},
        "scope": "data preparation only; no model, no training, no capability claim",
    }
    write_report(OUT_REPORT, report)
    print(
        json.dumps(
            {
                "digest": digest,
                "train": len(train),
                "fit": len(fit),
                "labels_fit": report["label_histogram_fit"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
