"""PLAN-N1-90 pre-flight: dry-run the confirmation code path on a synthetic face.

The v5 face is one-shot, so two things must be known before it is entered rather than
discovered inside it: that the reused 85 evaluator can load a *family* endpoint through
the protocol-swapping wrapper and score 32 items without changing model, optimizer, RNG
or history, and how much process CPU that path actually costs. Items here are fabricated
(fixed RNG over a small alphabet, shapes taken from the known train items' median
lengths) and split-marked holdout only to exercise the holdout branch, so this instrument
produces **no capability reading** and queries no v5, v3 holdout, v4 or dev item.
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation as base85,
)
from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation_run89 as run90,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_transition_factorial_lr_decay_run88 as lr88,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_transition_factorial_lr_decay_run89 as lr89,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_transition_factorial_paired as train,
)

OUTPUT = ROOT / "reports/taiji_n1_confirmation_face_cost_run89_dryrun.json"
ITEMS_PER_MODEL = 32
MODELS_ON_THE_FACE = 9
CPU_LIMIT_SECONDS = 180
ALPHABET = "abcdefghij"


def shapes() -> dict:
    items, _, _ = train.known_views()
    return {
        "question_chars": int(statistics.median(len(item["question"]) for item in items)),
        "answer_chars": int(statistics.median(len(item["answer"]) for item in items)),
        "observations": int(statistics.median(len(item["observations"]) for item in items)),
        "observation_chars": int(
            statistics.median(len(obs) for item in items for obs in item["observations"])
        ),
    }


def synthetic(shape: dict, rng: random.Random) -> list[dict]:
    kinds = ("lookup", "revision", "transition", "unknown")
    words = []
    for index in range(ITEMS_PER_MODEL):
        question = "".join(rng.choice(ALPHABET) for _ in range(shape["question_chars"]))
        observations = [
            "".join(rng.choice(ALPHABET) for _ in range(shape["observation_chars"]))
            for _ in range(shape["observations"])
        ]
        answer = "".join(rng.choice(ALPHABET) for _ in range(shape["answer_chars"]))
        words.append(
            {
                "id": f"timing-{index:03d}",
                "split": "holdout",
                "kind": kinds[index % 4],
                "entities": [f"e{index}"],
                "observations": observations,
                "question": question,
                "answer": answer,
            }
        )
    return words


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("the dry-run report is a single attempt file")
    train.torch.set_num_threads(1)
    started = time.process_time()
    lr89.configure("terminal")
    report = json.loads(lr89.REPORT.read_text(encoding="utf-8"))
    seed, path, expected_sha = None, None, None
    for model in report["models"]:
        for check in model["restore_checks"]:
            if check["path"].endswith("_final.pt"):
                seed, path, expected_sha = model["seed"], Path(check["path"]), check["sha256"]
    assert path is not None and path.is_file(), path
    assert lr88.sha256(path) == expected_sha, "endpoint bytes differ from the sealed receipt"

    shape = shapes()
    items = synthetic(shape, random.Random(20261017))
    payload = train.torch.load(path, map_location="cpu", weights_only=True)
    before_protocol = train.PROTOCOL
    model, optimizer = run90.load_with_family_protocol(payload)
    protocol_restored = train.PROTOCOL is before_protocol
    fingerprint_before = train.paired.fingerprint(model.bundle(optimizer))
    # A probe budget far above the frozen limit: the point is to measure the real cost,
    # then compare it against the 180 seconds the face is allowed, not to abort mid-way.
    budget = train.paired.CpuBudget(900)
    rows = base85.evaluate(model, optimizer, seed, payload["arm"], items, budget)
    fingerprint_after = train.paired.fingerprint(model.bundle(optimizer))
    cpu_seconds = round(time.process_time() - started + budget.children, 6)

    exact = sum(bool(row["free_exact"]) for row in rows)
    result = {
        "schema": "n1-confirmation-face-cost-dryrun-v1",
        "plan": "PLAN-N1-90",
        "complete": len(rows) == ITEMS_PER_MODEL and protocol_restored,
        "purpose": (
            "timing and neutrality only; items are fabricated, so no score here is "
            "a capability or confirmation reading"
        ),
        "not_a_capability_reading": True,
        "endpoint": {
            "seed": seed,
            "arm": payload["arm"],
            "path": str(path),
            "sha256": expected_sha,
        },
        "family_protocol_schema": payload["protocol"]["schema"],
        "loader_protocol_restored": protocol_restored,
        "shape_used": shape,
        "items": len(rows),
        "free_exact_of_synthetic": exact,
        "state_neutral": fingerprint_before == fingerprint_after,
        "bundle_fingerprint": str(fingerprint_after)[:64],
        "cpu_seconds_for_one_model": cpu_seconds,
        "projected_cpu_seconds_nine_models": round(cpu_seconds * MODELS_ON_THE_FACE, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "fits_within_limit": cpu_seconds * MODELS_ON_THE_FACE <= CPU_LIMIT_SECONDS,
        "v5_manifest_queried": False,
        "v3_holdout_queries": 0,
        "v4_queries": 0,
        "dev_queries": 0,
        "confirmation_face_consumed": False,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "can_promote": False,
    }
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["complete"] and result["fits_within_limit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
