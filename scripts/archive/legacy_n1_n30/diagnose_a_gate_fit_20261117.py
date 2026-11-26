"""Fit-only diagnosis of the 22.5.0 A-gate smoke: no check, no training.

Loads the birth and post-smoke checkpoints, evaluates train/fit, and records the
emitted content distribution. It is a development-face read only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    evaluate,
    generate_examples,
    load_learner,
)

RUN = ROOT / "output/a_gate_20261117T100847Z"
INITIAL = RUN / "seed-20261110-initial.bin"
FINAL = RUN / "seed-20261110-final.bin"
REPORT = ROOT / "reports/taiji_u2_22_5_a_fit_diagnosis_20261117.json"


def summarize(path: Path) -> dict:
    model, _, _ = load_learner(path)
    parts = {}
    for part, count in (("train", 64), ("fit", 32)):
        examples = generate_examples(part, count)
        result = evaluate(model, examples)
        outputs = Counter(
            bytes.fromhex(row["output_hex"]).decode("utf-8", errors="replace")
            for row in result["rows"]
        )
        parts[part] = {
            "correct": result["correct"],
            "denominator": result["denominator"],
            "accuracy": result["accuracy"],
            "outputs": dict(outputs),
            "targets": dict(Counter(row["target"] for row in result["rows"])),
            "eos": sum(int(row["eos"]) for row in result["rows"]),
            "valid_utf8": sum(int(row["valid_utf8"]) for row in result["rows"]),
        }
    return parts


def main() -> int:
    report = {
        "schema": "taiji-u2-22-5-a-fit-diagnosis-v1",
        "card": "22.5.0.A",
        "checkpoint_run": str(RUN.relative_to(ROOT)),
        "initial": summarize(INITIAL),
        "final": summarize(FINAL),
        "conclusion": (
            "Both birth and post-smoke checkpoints emit an empty string followed by EOS on every "
            "train and fit example. The failure is output/emission collapse, not a position "
            "classification error, not UTF-8, and not recovery."
        ),
        "next": (
            "fit-only readout diagnosis/fix: inspect EOS bias and the request_candidate goal/"
            "emission path, then retrain on train/fit until content is emitted; keep check at 1/2."
        ),
        "discipline": {"check_reads": 0, "final_reads": 0, "optimizer_steps": 0},
    }
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "initial train/fit",
        report["initial"]["train"]["correct"],
        report["initial"]["fit"]["correct"],
        "final train/fit",
        report["final"]["train"]["correct"],
        report["final"]["fit"]["correct"],
        "outputs",
        report["final"]["train"]["outputs"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
