"""A-prime instrument: paired mean-surprise of the two P3b arms on one frozen, unseen row set.

Why a second instrument exists: at P3a level the CAP-0 behavioural surface scores 3/20 (E), 1/16 (D)
and 0/14 (C), and bucketing every machine-scored item by the length of its expected answer shows 19
of 20 E items already expect a 1-2 character token.  So the zeros are a capability floor, not an
unexpressible question -- and with 20 items one flip is 0.05, coarser than the effect the criteria
ask about.  ``mean_surprise`` over a fixed, arm-independent corpus is continuous and needs no
generation, so it costs seconds instead of minutes.

Read-only by construction: it calls ``Seed.score_bytes`` (documented "Evaluate raw-byte prediction
without mutating persistent state"), trains nothing, and never writes under ``checkpoints/``.  Only
the manifest is committed; the bytes are re-derived from the source and hash-checked at run time, so
there is no second copy to drift.

**Shape of the measurement** (disclosed because it differs from the training stream): each held-out
row is scored as its own document the way the trainer feeds one row
(``sensor.symbols(data, include_boundary=True)`` adds the boundary symbol), but substrate state does
**not** carry from one row to the next.  This is a validation-set NLL, not a continuing stream.

Pre-declared reading rules: plans/reference/M5_P3B_SECOND_CAMPAIGN_DECISION_BRIEF_20260916.md §2c.
They were written before any number existed; this module encodes them, it does not tune them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts" / "training")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch  # noqa: E402
from build_p3b_arm_corpus import _documents  # noqa: E402
from train_p3b_aligned import _install_legacy_guard, build_config_from_checkpoint  # noqa: E402

from seed import Seed  # noqa: E402

SOURCE = PROJECT_ROOT / "data" / "simple_zh" / "simple_zh_texts.jsonl"
MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "p3b_heldout_eval_manifest.json"
ARM_MANIFESTS = (
    PROJECT_ROOT / "plans" / "manifests" / "p3b_dialogue_fresh_manifest.json",
    PROJECT_ROOT / "plans" / "manifests" / "p3b_all_fresh_manifest.json",
)
#: §2c rule 1: the holdout starts after the highest row index either arm emitted, so it is disjoint
#: from both training streams *and* from the ancestor's seen prefix (rows 0-2,892).
MIN_START_ROW = max(
    int(json.loads(path.read_text(encoding="utf-8"))["last_emitted_row"]) for path in ARM_MANIFESTS
)
EVAL_BYTES = 65_536
#: All three states already exist on disk, so answering A-prime costs no training time.
STATES = {
    "start": PROJECT_ROOT / "checkpoints" / "seed_beta.pt",
    "treatment": PROJECT_ROOT
    / "checkpoints"
    / "p3b"
    / "snapshots"
    / "seed_aligned_tick_17000000.pt",
    "control": PROJECT_ROOT
    / "checkpoints"
    / "p3b"
    / "snapshots"
    / "seed_aligned_control_tick_17000000.pt",
}
REQUIRED_SCORE_KEYS = ("observations", "accuracy", "mean_surprise")


def derive_eval_rows(
    start_row: int, max_bytes: int = EVAL_BYTES
) -> tuple[list[bytes], dict[str, Any]]:
    """UTF-8 bytes of every row at or after ``start_row``, up to ``max_bytes`` in total."""

    if start_row <= MIN_START_ROW:
        raise SystemExit(
            f"start_row {start_row} overlaps the arms (last emitted row {MIN_START_ROW}); "
            "a held-out set must be disjoint from every training stream"
        )
    rows: list[bytes] = []
    indices: list[int] = []
    total = 0
    for index, text in _documents(SOURCE):
        if index < start_row:
            continue
        encoded = text.encode("utf-8")
        if rows and total + len(encoded) > max_bytes:
            break
        rows.append(encoded)
        indices.append(index)
        total += len(encoded)
    if not rows:
        raise SystemExit(f"no usable rows at or after {start_row} in {SOURCE}")
    meta = {
        "source": str(SOURCE.relative_to(PROJECT_ROOT)),
        "row_index_convention": "0-based file line index, as yielded by _documents",
        "start_row": start_row,
        "rows_used": len(rows),
        "first_row_index": indices[0],
        "last_row_index": indices[-1],
        "total_bytes": total,
        "sha256": hashlib.sha256(b"".join(rows)).hexdigest(),
        "disjoint_from_arms_because_start_row_above": MIN_START_ROW,
        "scoring_shape": "one score_bytes call per row, boundary added by the sensor, "
        "no state carried between rows",
    }
    return rows, meta


def write_manifest(meta: dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "taiji-p3b-heldout-eval-v1", **meta}
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def load_state(checkpoint: Path) -> tuple[Seed, dict[str, Any]]:
    """Rebuild config from the envelope and restore under the chain the arms trained on."""

    config, start = build_config_from_checkpoint(checkpoint)
    guard = _install_legacy_guard()
    model = Seed(config, device=torch.device("cpu"), episode_id="p3b-heldout-eval")
    model.restore(torch.load(checkpoint, weights_only=False))
    return model, {"chain": guard, "start": start, "tick": int(model.tick)}


def score_rows(model: Seed, rows: list[bytes]) -> dict[str, float]:
    """Observation-weighted mean of per-row surprise, failing closed on a missing field."""

    observations = 0.0
    surprise_sum = 0.0
    correct = 0.0
    for row in rows:
        out = model.score_bytes(row)
        missing = [key for key in REQUIRED_SCORE_KEYS if key not in out]
        if missing:
            raise SystemExit(f"score_bytes returned no {missing}; refusing to aggregate a guess")
        count = float(out["observations"])
        observations += count
        surprise_sum += float(out["mean_surprise"]) * count
        correct += float(out["accuracy"]) * count
    if observations <= 0:
        raise SystemExit("no scored observations in the held-out set")
    return {
        "observations": observations,
        "mean_surprise": surprise_sum / observations,
        "accuracy": correct / observations,
    }


def paired_verdict(per_state: dict[str, dict[str, float]]) -> dict[str, Any]:
    """§2c rule 4: resolved only if the split-half noise is beaten **and** both halves agree."""

    def noise_of(state: str) -> float:
        halves = per_state[state]
        return abs(halves["slice_a_mean_surprise"] - halves["slice_b_mean_surprise"])

    noise = max(noise_of(state) for state in per_state)
    treated = per_state["treatment"]
    controlled = per_state["control"]
    whole = treated["whole_mean_surprise"] - controlled["whole_mean_surprise"]
    first = treated["slice_a_mean_surprise"] - controlled["slice_a_mean_surprise"]
    second = treated["slice_b_mean_surprise"] - controlled["slice_b_mean_surprise"]
    same_sign = (first > 0) == (second > 0)
    return {
        "split_half_noise": round(noise, 6),
        "treatment_minus_control_whole": round(whole, 6),
        "treatment_minus_control_slice_a": round(first, 6),
        "treatment_minus_control_slice_b": round(second, 6),
        "both_halves_same_sign": same_sign,
        "verdict_label": "resolved" if same_sign and abs(whole) > noise else "not_resolved",
        "meaning_of_not_resolved": "this instrument cannot separate the arms at this budget; "
        "it is NOT evidence that the data distribution does not matter",
    }


def run(start_row: int, report_path: Path) -> dict[str, Any]:
    rows, meta = derive_eval_rows(start_row)
    if MANIFEST.exists():
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if recorded.get("sha256") != meta["sha256"]:
            raise SystemExit(
                f"held-out bytes changed (manifest {str(recorded.get('sha256'))[:12]} vs recomputed "
                f"{meta['sha256'][:12]}) -- refusing to score against a moved evaluation set"
            )
    else:
        write_manifest(meta)
    split = len(rows) // 2
    slices = {"whole": rows, "slice_a": rows[:split], "slice_b": rows[split:]}
    per_state: dict[str, dict[str, float]] = {}
    ticks: dict[str, int] = {}
    for state, checkpoint in STATES.items():
        if not checkpoint.exists():
            raise SystemExit(f"state {state}: missing checkpoint {checkpoint}")
        model, info = load_state(checkpoint)
        ticks[state] = info["tick"]
        scored = {name: score_rows(model, part) for name, part in slices.items()}
        del model
        per_state[state] = {
            f"{name}_{key}": value
            for name, values in scored.items()
            for key, value in values.items()
        }
        print(
            json.dumps(
                {
                    "event": "p3b_heldout_state",
                    "state": state,
                    "tick": info["tick"],
                    "whole": scored["whole"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    report = {
        "format": "taiji-p3b-heldout-surprise-v1",
        "prereading_rules": "plans/reference/M5_P3B_SECOND_CAMPAIGN_DECISION_BRIEF_20260916.md §2c",
        "eval_set": meta,
        "split_rows_after": len(slices["slice_a"]),
        "state_ticks": ticks,
        "scores": per_state,
        "paired": paired_verdict(per_state),
        "per_arm_vs_start": {
            f"{arm}_minus_start": round(
                per_state[arm]["whole_mean_surprise"] - per_state["start"]["whole_mean_surprise"], 6
            )
            for arm in ("treatment", "control")
        },
        "per_arm_vs_start_meaning": "process record only -- a one-sided delta is not the data effect "
        "(§2c rule 3)",
        "does_not_do": [
            "no training and no checkpoint writes (score_bytes is documented non-mutating)",
            "no CAP-0 behavioural claim: this measures statistical fit only",
            "does not satisfy the frozen §4 'two consecutive matched checkpoints' rule",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P3b A-prime: paired mean-surprise on a frozen holdout"
    )
    parser.add_argument("--start-row", type=int, default=MIN_START_ROW + 1)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p3b_heldout_surprise_20260916.json",
    )
    parser.add_argument("--freeze-only", action="store_true", help="write/verify the manifest only")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.freeze_only:
        _, meta = derive_eval_rows(args.start_row)
        write_manifest(meta)
        print(json.dumps({"event": "p3b_heldout_frozen", **meta}, ensure_ascii=True), flush=True)
        return 0
    result = run(args.start_row, args.report)
    print(
        json.dumps(
            {
                "event": "p3b_heldout_done",
                "paired": result["paired"],
                "per_arm_vs_start": result["per_arm_vs_start"],
                "seconds": round(time.perf_counter() - started, 1),
            },
            ensure_ascii=True,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
