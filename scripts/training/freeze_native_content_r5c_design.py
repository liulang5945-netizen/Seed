"""R5c design-freeze support: pointer-CE target rule and gradient-path check.

Train-only, read-only decision support for the pending R5c ruling.  Freezes
the last free parameters of the R5c draft:

* value-substring occurrence statistics over train fact turns, which select
  the declared pointer-CE target rule (supervise the occurrence positions
  when the value appears exactly once; skip turns with no occurrence);
* a gradient-path probe on the frozen q1 seed0 endpoint: the copy pointer
  distribution over the selected record's span receives an auxiliary CE
  toward the value-byte positions, and the backward pass must reach
  copy_key/copy_query with finite gradients (no optimizer step);
* per-category scored-turn counts for the declared exposure-weighting
  contingency.

No optimizer update is performed; query/holdout/sealed stay unread.
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

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.native_content_eval import FACT_ROLES, TIME_FOR_CATEGORY  # noqa: E402
from taiji.native_content_training_v2 import R5BRunTrainer  # noqa: E402

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
TRAINING_REPORT = ROOT / "reports/taiji_native_core_r5b_training_20261030.json"
REPORT = ROOT / "reports/taiji_native_core_r5c_design_freeze_20261031.json"

RUN_NAME = "q1_new_core_v2_seed0"
POINTER_CE_WEIGHT = 0.3


def _occurrence_stats(payload: dict[str, Any]) -> dict[str, Any]:
    stats = {"fact_turns": 0, "single": 0, "multiple": 0, "absent": 0}
    for episode in payload["episodes"]:
        for turn in episode["turns"]:
            if turn["role"] not in FACT_ROLES:
                continue
            stats["fact_turns"] += 1
            value = turn.get("value")
            if not value:
                stats["absent"] += 1
                continue
            count = str(turn["text"]).count(str(value))
            if count == 1:
                stats["single"] += 1
            elif count > 1:
                stats["multiple"] += 1
            else:
                stats["absent"] += 1
    return stats


def _category_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in payload["episodes"]:
        for turn in episode["turns"]:
            if turn.get("scored"):
                counts[str(turn["category"])] = counts.get(str(turn["category"]), 0) + 1
    return counts


def _gradient_probe(payload: dict[str, Any]) -> dict[str, Any]:
    training = json.loads(TRAINING_REPORT.read_text(encoding="utf-8"))
    trainer = R5BRunTrainer.load(
        ROOT / training["runs"][RUN_NAME]["endpoint_restore"]["checkpoint"]
    )

    def _expected_value(episode: dict[str, Any], turn: dict[str, Any]) -> str | None:
        history = [
            candidate
            for candidate in episode["turns"]
            if int(candidate["index"]) < int(turn["index"])
            and candidate["role"] in FACT_ROLES
            and candidate.get("entity") == turn.get("entity")
            and candidate.get("attribute") == turn.get("attribute")
        ]
        history.sort(key=lambda item: int(item["index"]))
        return str(history[-1].get("value")) if history else None

    episode = next(
        (
            candidate
            for candidate in payload["episodes"]
            if any(
                turn.get("scored")
                and turn.get("category") == "current_query"
                and _expected_value(candidate, turn)
                for turn in candidate["turns"]
            )
        ),
        None,
    )
    if episode is None:
        raise ValueError("train manifest has no episode with a value-bearing current_query")
    session = str(episode["episode_id"]).replace(":", "_")
    bundle = trainer.adapter.fresh_bundle(trainer.model)
    scored = None
    for turn in episode["turns"]:
        if turn["role"] in FACT_ROLES:
            raw = str(turn["text"]).encode("utf-8")
            bundle.ingest(
                raw_span=raw,
                source_input_id=f"{episode['episode_id']}:{turn['index']}",
                source_sha256=hashlib.sha256(raw).hexdigest(),
                owner="diag",
                scope="diag",
                session=session,
                turn_index=int(turn["index"]),
                revision=0 if turn["role"] in {"fact_v0", "distractor"} else 1,
                entity=turn.get("entity"),
                attribute=turn.get("attribute"),
                value=turn.get("value"),
                status="resolved",
            )
        if turn.get("scored") and turn.get("category") == "current_query" and scored is None:
            scored = turn
    if scored is None:
        raise ValueError("probe episode has no current_query scored turn")
    value = _expected_value(episode, scored)
    if not value:
        raise ValueError("selected probe turn has no derivable value")
    condition = bundle.condition(
        query=str(scored["text"]).encode("utf-8"),
        time_intent=TIME_FOR_CATEGORY[str(scored["category"])],
        owner="diag",
        scope="diag",
        session=session,
    )
    state = bundle.model.conditioned_state(
        condition, native_state_ref="diag:freeze", scope_id="diag", session_id=session
    )
    cache = bundle.model.content_read_cache(state, condition, store=bundle.store)
    hidden = state.hidden if state.decoder_hidden is None else state.decoder_hidden
    read, scores = bundle.model._read(cache, hidden)
    span = cache.content_spans[int(torch.argmax(scores))]
    tokens = torch.tensor(list(span), dtype=torch.long)
    keys = bundle.model.copy_key(bundle.model.core.embedding(tokens))
    positions = keys @ bundle.model.copy_query(hidden) / (32**0.5)
    weights = torch.softmax(positions, dim=0)
    value_bytes = value.encode("utf-8")
    hits = [
        index
        for index in range(len(span) - len(value_bytes) + 1)
        if bytes(span[index : index + len(value_bytes)]) == value_bytes
    ]
    target = torch.zeros(len(span), dtype=torch.float32)
    for start in hits:
        target[start : start + len(value_bytes)] = 1.0 / len(hits)
    auxiliary = -(target * torch.log(weights.clamp_min(1e-12))).sum()
    auxiliary.backward()
    grads = {
        name: None if parameter.grad is None else bool(torch.isfinite(parameter.grad).all())
        for name, parameter in bundle.model.named_parameters()
        if name.startswith("copy_")
    }
    return {
        "selected_span": span.decode("utf-8", errors="replace"),
        "value": value,
        "value_occurrences_in_span": len(hits),
        "pointer_ce_weight_declared": POINTER_CE_WEIGHT,
        "auxiliary_loss_finite": bool(torch.isfinite(auxiliary)),
        "copy_gradients_finite": grads,
        "gradient_path_ok": bool(grads.get("copy_key.weight") and grads.get("copy_query.weight")),
    }


def _main(approve: bool) -> int:
    if not approve:
        raise SystemExit("freeze support requires explicit approval (--approve-r5b)")
    if REPORT.exists():
        raise SystemExit(f"freeze receipt is write-once: {REPORT}")
    payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    if payload.get("split") != "train" or not payload.get("training_read"):
        raise SystemExit("freeze support requires the declared readable train split")
    started_at = datetime.now(UTC)
    started_cpu = time.process_time()
    occurrence = _occurrence_stats(payload)
    categories = _category_counts(payload)
    probe = _gradient_probe(payload)
    receipt = {
        "format": "taiji-native-core-r5c-design-freeze-v1",
        "version": 1,
        "date": "2026-10-31",
        "draft_contract": "plans/reference/SPEC-NATIVE-CONTENT-CORE_R5C_COPYEXPR_CANDIDATE_20261031.md",
        "run": RUN_NAME,
        "manifest": {
            "path": TRAIN.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
            "episodes": len(payload["episodes"]),
        },
        "pointer_ce_target_rule": {
            "occurrence_stats": occurrence,
            "declared_rule": "supervise the copy pointer on the value-substring byte positions when the value occurs exactly once in the selected span; turns without an occurrence are excluded from the auxiliary term",
            "coverage": occurrence["single"] / max(occurrence["fact_turns"], 1),
        },
        "gradient_probe": probe,
        "exposure_weighting_contingency": {
            "scored_turns_by_category": categories,
            "current_fraction": categories.get("current_query", 0)
            / max(sum(categories.values()), 1),
        },
        "read_boundaries": {
            "query_split_read": False,
            "holdout_split_read": False,
            "sealed_read": False,
            "optimizer_updates": 0,
        },
        "cost": {"cpu_seconds": time.process_time() - started_cpu},
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    REPORT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": "freeze_support_complete", "report": str(REPORT)}, ensure_ascii=False)
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve-r5b",
        action="store_true",
        help="required: covered by the user-approved R5b execution (read-only diagnosis)",
    )
    args = parser.parse_args()
    return _main(approve=args.approve_r5b)


if __name__ == "__main__":
    raise SystemExit(main())
