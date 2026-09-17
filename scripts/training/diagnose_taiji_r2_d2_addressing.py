"""R2-D2 diagnostic: does content addressing specialize onto material positions?

Contract section 6 graph inspection.  Trains the A arm 30 epochs at the stable
0.01 rate, then measures the first renderer step's addressing weights for
fact/negation items: max weight, entropy (vs uniform 1/L), and the probability
mass falling on the material-clause byte span.  Same measurement at
initialization gives the contrast.  Diagnostic only; not a gate run.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts.training.build_taiji_r2_d1_measurement_fixture import MATERIAL_CLAUSE  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    EVIDENCE_PER_POSITION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURE = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
SEED = 20260917
EPOCHS = 30


def _load_train() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row["split"] == "train"]


def _measure(prototype: SequenceWorkspacePrototype, row: dict[str, Any]) -> dict[str, Any]:
    prefix_text = row["prefix"]
    prefix = prefix_text.encode("utf-8")
    match = MATERIAL_CLAUSE.search(prefix_text)
    # byte span of the material clause (character indices -> UTF-8 byte offsets)
    char_start = match.end()
    char_end = len(prefix_text) - len(
        {"背景：": "答：", "线索：": "回答：", "已知：": "输出："}[match.group(0)]
    )
    byte_start = len(prefix_text[:char_start].encode("utf-8"))
    byte_end = len(prefix_text[:char_end].encode("utf-8"))
    state = prototype.begin_episode(prefix)
    weights = prototype.addressing_weights(state, detach=True)
    vector = weights.detach()
    entropy = float(-(vector * vector.clamp_min(1e-12).log()).sum())
    uniform = math.log(float(vector.shape[0]))
    return {
        "id": row["id"],
        "shape": row["shape"],
        "entries": int(vector.shape[0]),
        "max_weight": float(vector.max()),
        "entropy_nats": entropy,
        "uniform_entropy": uniform,
        "entropy_ratio": entropy / uniform,
        "material_span": [byte_start, byte_end],
        "material_mass": float(vector[byte_start:byte_end].sum()),
        "material_fraction_of_entries": (byte_end - byte_start) / float(vector.shape[0]),
        "top5_positions": [int(i) for i in vector.argsort(descending=True)[:5]],
    }


def main() -> int:
    rows = _load_train()
    episodes = tuple(
        (row["prefix"].encode("utf-8"), row["response"].encode("utf-8")) for row in rows
    )
    torch.manual_seed(SEED)
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(seed=SEED, evidence_source=EVIDENCE_PER_POSITION)
    )
    samples = [row for row in rows if row["shape"] in {"fact", "negation"}][:6]
    before = [_measure(prototype, row) for row in samples]
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=0.01, code_revision="r2d2-addressing-diagnostic"
    )
    trainer.set_episodes(episodes)
    for _ in range(EPOCHS):
        trainer.train_epoch()
    after = [_measure(prototype, row) for row in samples]
    payload = {
        "format": "taiji-r2-d2-addressing-diagnostic-v1",
        "diagnostic_not_gate": True,
        "epochs": EPOCHS,
        "before_training": before,
        "after_training": after,
        "growth_admitted": False,
        "can_promote": False,
    }
    out = Path("reports/r2_d2_addressing_diagnostic_20260918.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for label, group in (("init", before), ("trained", after)):
        print(
            label,
            "mean entropy_ratio",
            round(sum(g["entropy_ratio"] for g in group) / len(group), 4),
            "mean max_w",
            round(sum(g["max_weight"] for g in group) / len(group), 4),
            "mean material_mass",
            round(sum(g["material_mass"] for g in group) / len(group), 4),
            "span_fraction",
            round(group[0]["material_fraction_of_entries"], 3),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
