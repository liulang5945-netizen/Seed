"""10.7g: are object and place separable in the subject's action encodings?

10.7f showed a 1,536-parameter head can memorise six samples but not transfer. One
explanation is the head; the other is the features. With a balanced 2x3 design of
(object, place) sentences a two-way decomposition separates them:

* main effects of object and of place,
* the residual interaction.

Under pure noise a balanced 2x3 design puts ``(1+2)/(6-1) = 0.6`` of the energy into
the main effects, so 0.6 is the null, not zero. Additive features clear it; entangled
ones fall below it.

It reads the encoder only: no parameters change.
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
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402

BUDGET_CPU_SECONDS = 120
BUDGET_WALL_SECONDS = 240
ADDITIVE_SHARE = 0.8
ENTANGLED_SHARE = 0.45


def sentence(obj: str, place: str) -> str:
    return f"把{obj}放到{place}"


def decompose(table: dict[tuple[str, str], torch.Tensor]) -> dict:
    grand = torch.stack(list(table.values())).mean(dim=0)
    object_effect = {}
    for obj in OBJECTS:
        rows = [table[(obj, place)] for place in PLACES]
        object_effect[obj] = torch.stack(rows).mean(dim=0) - grand
    place_effect = {}
    for place in PLACES:
        rows = [table[(obj, place)] for obj in OBJECTS]
        place_effect[place] = torch.stack(rows).mean(dim=0) - grand
    total = 0.0
    main = 0.0
    resid = 0.0
    for obj in OBJECTS:
        for place in PLACES:
            value = table[(obj, place)]
            residual = value - grand - object_effect[obj] - place_effect[place]
            total += float((value - grand).norm() ** 2)
            resid += float(residual.norm() ** 2)
    main = total - resid
    return {
        "ss_total": total,
        "ss_main": main,
        "ss_residual": resid,
        "additive_share": main / total if total > 0 else None,
        "object_effect_norms": {k: float(v.norm()) for k, v in object_effect.items()},
        "place_effect_norms": {k: float(v.norm()) for k, v in place_effect.items()},
    }


def classify(share) -> str:
    if share is None:
        return "features_undefined"
    if share >= ADDITIVE_SHARE:
        return "features_additive"
    if share <= ENTANGLED_SHARE:
        return "features_entangled"
    return "features_mixed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7g probe is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    encoder = subject.encoder

    with torch.no_grad():
        table = {
            (obj, place): encoder.context(sentence(obj, place).encode()).mean(dim=0).detach()
            for obj in OBJECTS
            for place in PLACES
        }
    stats = decompose(table)
    verdict = classify(stats["additive_share"])

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    else:
        status = verdict

    report = {
        "schema": "taiji-feature-additivity-10-7g-v1",
        "step": "10.7g",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "design": "2 objects x 3 places",
            "null_share": (len(OBJECTS) - 1 + len(PLACES) - 1) / (len(OBJECTS) * len(PLACES) - 1),
            "additive_share": ADDITIVE_SHARE,
            "entangled_share": ENTANGLED_SHARE,
            "training": "none",
        },
        "parent_model_digest": digest,
        "stats": stats,
        "verdict": verdict,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Feature geometry only: it decomposes the encoder's action vectors and says "
            "nothing about any task reading. Additive features would make a factorised "
            "readout plausible; entangled ones would not. No parameter changes."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in ("scripts/training/probe_feature_additivity_10_7g.py",)
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
                "additive_share": stats["additive_share"],
                "null_share": report["declared"]["null_share"],
                "object_effects": {k: round(v, 4) for k, v in stats["object_effect_norms"].items()},
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
