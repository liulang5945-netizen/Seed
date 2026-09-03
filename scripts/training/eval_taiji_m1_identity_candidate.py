"""Review a manually gated candidate checkpoint for the native identity organ.

This is a promotion review, not an activation path.  It materializes the
candidate checkpoint in memory, verifies its content digest and fresh-process
restore, and writes a small content-addressed manifest.  The default Taiji
configuration remains disabled and no binary checkpoint is committed by this
review.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_b5_memory import build_corpus as build_b5_corpus  # noqa: E402
from scripts.training.eval_taiji_foundation_baseline import (  # noqa: E402
    build_delayed_memory_smoke_corpus,
)
from scripts.training.eval_taiji_m1_identity_admission import (  # noqa: E402
    _accuracy,
    _actions,
    _review_config,
)
from scripts.training.eval_taiji_m1_identity_organ_canary import (  # noqa: E402
    _config,
    _core_digest,
    _fresh_process_probe,
    _organ_digest,
)
from scripts.training.eval_taiji_m1_identity_shadow import (  # noqa: E402
    B1_HOLDOUT,
    B1_TRAIN,
)
from taiji import DelayedMemoryTask, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-identity-candidate-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_identity_candidate_20260902.json"
REVIEW_GAIN = 32.0


def _train_b2(model: Taiji, corpus: Any) -> float:
    for episode in corpus.train:
        DelayedMemoryTask._write_episode(model, episode)
    return _accuracy(model, corpus.holdout, _actions(*corpus.train))


def _train_b5(model: Taiji, corpus: Any) -> dict[str, Any]:
    actions = _actions(*corpus.phase_a_train, *corpus.phase_b_train)
    for episode in corpus.phase_a_train:
        DelayedMemoryTask._write_episode(model, episode)
    parent_checkpoint = deepcopy(model.checkpoint())
    parent_old = _accuracy(model, corpus.phase_a_holdout, actions)
    parent_retention = _accuracy(model, corpus.phase_a_retention, actions)
    for episode in corpus.phase_b_train:
        DelayedMemoryTask._write_episode(model, episode)
    child_checkpoint = deepcopy(model.checkpoint())
    return {
        "parent_old_holdout": parent_old,
        "parent_retention": parent_retention,
        "child_old_holdout": _accuracy(model, corpus.phase_a_holdout, actions),
        "child_retention": _accuracy(model, corpus.phase_a_retention, actions),
        "child_new_holdout": _accuracy(model, corpus.phase_b_holdout, actions),
        "parent_checkpoint": parent_checkpoint,
        "child_checkpoint": child_checkpoint,
    }


def _b1_score(model: Taiji) -> dict[str, float]:
    model.learn_bytes(B1_TRAIN, epochs=1)
    return model.score_bytes(B1_HOLDOUT)


def _variant_record(
    *,
    default: Taiji,
    candidate: Taiji,
    b2: Any,
    b5: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    default_b2 = _train_b2(default, b2)
    candidate_b2 = _train_b2(candidate, b2)
    lesion_after_b2 = Taiji.from_checkpoint(deepcopy(candidate.checkpoint()))
    lesion_after_b2.identity_organ.lesion()
    lesion_b2 = _accuracy(lesion_after_b2, b2.holdout, _actions(*b2.train))

    default_b5 = _train_b5(default, b5)
    candidate_b5 = _train_b5(candidate, b5)
    lesion_after_b5 = Taiji.from_checkpoint(deepcopy(candidate_b5["child_checkpoint"]))
    lesion_after_b5.identity_organ.lesion()
    lesion_b5 = {
        "child_old_holdout": _accuracy(
            lesion_after_b5,
            b5.phase_a_holdout,
            _actions(*b5.phase_a_train, *b5.phase_b_train),
        ),
        "child_new_holdout": _accuracy(
            lesion_after_b5,
            b5.phase_b_holdout,
            _actions(*b5.phase_a_train, *b5.phase_b_train),
        ),
    }

    default_b1 = _b1_score(default)
    candidate_b1 = _b1_score(candidate)
    lesion_b1 = _b1_score(lesion_after_b5)
    metrics = {
        "b1": {
            "default": default_b1,
            "candidate": candidate_b1,
            "lesion": lesion_b1,
            "candidate_mean_surprise_delta": (
                candidate_b1["mean_surprise"] - default_b1["mean_surprise"]
            ),
            "lesion_mean_surprise_delta": (
                lesion_b1["mean_surprise"] - default_b1["mean_surprise"]
            ),
        },
        "b2": {
            "default_holdout": default_b2,
            "candidate_holdout": candidate_b2,
            "lesion_holdout": lesion_b2,
        },
        "b5": {
            "default": {
                key: value
                for key, value in default_b5.items()
                if not key.endswith("checkpoint")
            },
            "candidate": {
                key: value
                for key, value in candidate_b5.items()
                if not key.endswith("checkpoint")
            },
            "lesion": lesion_b5,
        },
    }
    return (
        metrics,
        candidate_b5["child_checkpoint"],
        lesion_after_b5.checkpoint(),
    )


def _budget_record(default: Taiji, candidate: Taiji, lesion: Taiji) -> dict[str, Any]:
    def one(model: Taiji) -> dict[str, Any]:
        organ = model.identity_organ
        return {
            "identity_organ_enabled": organ is not None,
            "active_parameter_count": model.parameter_count(),
            "planned_active_parameter_count": model.config.planned_active_parameter_count,
            "parameter_count_matches_plan": (
                model.parameter_count() == model.config.planned_active_parameter_count
            ),
            "identity_parameter_count": None if organ is None else organ.parameter_count,
        }

    return {"default": one(default), "candidate": one(candidate), "lesion": one(lesion)}


def _schema_record(
    *, default_checkpoint: dict[str, Any], candidate_checkpoint: dict[str, Any], lesion_checkpoint: dict[str, Any]
) -> dict[str, Any]:
    candidate_payload = candidate_checkpoint["identity_organ"]
    lesion_payload = lesion_checkpoint["identity_organ"]
    return {
        "default_has_identity_payload": "identity_organ" in default_checkpoint,
        "candidate_has_identity_payload": "identity_organ" in candidate_checkpoint,
        "lesion_has_identity_payload": "identity_organ" in lesion_checkpoint,
        "candidate_format": candidate_payload["format"],
        "candidate_version": candidate_payload["version"],
        "lesion_format": lesion_payload["format"],
        "candidate_lineage_core_matches": (
            _core_digest(candidate_checkpoint)
            == candidate_payload["lineage"]["parent_checkpoint_digest"]
        ),
        "lesion_lineage_core_matches": (
            _core_digest(lesion_checkpoint)
            == lesion_payload["lineage"]["parent_checkpoint_digest"]
        ),
        "candidate_restore_exact": (
            content_digest(Taiji.from_checkpoint(deepcopy(candidate_checkpoint)).checkpoint())
            == content_digest(candidate_checkpoint)
        ),
        "lesion_restore_exact": (
            content_digest(Taiji.from_checkpoint(deepcopy(lesion_checkpoint)).checkpoint())
            == content_digest(lesion_checkpoint)
        ),
    }


def _record_passes(record: dict[str, Any]) -> bool:
    schema = record["schema"]
    budget = record["budget"]
    metrics = record["metrics"]
    checkpoint = record["checkpoint"]
    return bool(
        schema["default_has_identity_payload"] is False
        and schema["candidate_has_identity_payload"]
        and schema["lesion_has_identity_payload"]
        and schema["candidate_format"] == "taiji-native-identity-organ-v2"
        and schema["candidate_version"] == 3
        and schema["candidate_lineage_core_matches"]
        and schema["lesion_lineage_core_matches"]
        and schema["candidate_restore_exact"]
        and schema["lesion_restore_exact"]
        and all(item["parameter_count_matches_plan"] for item in budget.values())
        and abs(metrics["b1"]["candidate_mean_surprise_delta"]) <= 1e-12
        and abs(metrics["b1"]["lesion_mean_surprise_delta"]) <= 1e-12
        and metrics["b2"]["candidate_holdout"] >= metrics["b2"]["default_holdout"]
        and metrics["b5"]["candidate"]["child_old_holdout"]
        >= metrics["b5"]["candidate"]["parent_old_holdout"]
        and metrics["b5"]["candidate"]["child_new_holdout"]
        > metrics["b5"]["default"]["child_new_holdout"]
        and checkpoint["candidate_fresh_process_source"] == "identity-route"
        and checkpoint["candidate_fresh_process_persistent_digest_unchanged"]
        and checkpoint["candidate_fresh_process_digest_matches"]
        and checkpoint["default_rollback_exact"]
        and checkpoint["lesion_digest_differs_from_candidate"]
    )


def _record(seed: int) -> dict[str, Any]:
    b2 = build_delayed_memory_smoke_corpus(count=16)
    b5 = build_b5_corpus(train_count=16, holdout_count=8, retention_count=8)
    default = Taiji(_config(seed, enabled=False), episode_id=f"m1-31-default-{seed}")
    candidate = Taiji(
        _review_config(seed, enabled=True), episode_id=f"m1-31-candidate-{seed}"
    )
    baseline_default_checkpoint = deepcopy(default.checkpoint())
    candidate_start_checkpoint = deepcopy(candidate.checkpoint())
    # Rebuild metric variants from clean same-seed models so the candidate,
    # default and lesion measurements do not depend on evaluation order.
    default_metrics_model = Taiji(
        _config(seed, enabled=False), episode_id=f"m1-31-default-metrics-{seed}"
    )
    candidate_metrics_model = Taiji(
        _review_config(seed, enabled=True), episode_id=f"m1-31-candidate-metrics-{seed}"
    )
    metrics, candidate_checkpoint, lesion_checkpoint = _variant_record(
        default=default_metrics_model,
        candidate=candidate_metrics_model,
        b2=b2,
        b5=b5,
    )
    candidate = candidate_metrics_model
    lesion = Taiji.from_checkpoint(deepcopy(candidate_checkpoint))
    lesion.identity_organ.lesion()

    fresh = _fresh_process_probe(candidate_checkpoint, b5.phase_a_holdout[0].cue)
    default_rollback = Taiji.from_checkpoint(deepcopy(baseline_default_checkpoint))
    candidate_checkpoint_model = Taiji.from_checkpoint(deepcopy(candidate_checkpoint))
    lesion_checkpoint_model = Taiji.from_checkpoint(deepcopy(lesion_checkpoint))
    candidate_organ_digest = _organ_digest(candidate_checkpoint_model)
    lesion_organ_digest = _organ_digest(lesion_checkpoint_model)
    record = {
        "seed": seed,
        "review_gain": REVIEW_GAIN,
        "default_config_enabled": default.config.identity_organ_enabled,
        "candidate_config_enabled": candidate.config.identity_organ_enabled,
        "candidate_start_checkpoint_digest": content_digest(candidate_start_checkpoint),
        "candidate_checkpoint_digest": content_digest(candidate_checkpoint),
        "lesion_checkpoint_digest": content_digest(lesion_checkpoint),
        "schema": _schema_record(
            default_checkpoint=baseline_default_checkpoint,
            candidate_checkpoint=candidate_checkpoint,
            lesion_checkpoint=lesion_checkpoint,
        ),
        "budget": _budget_record(default, candidate, lesion),
        "metrics": metrics,
        "checkpoint": {
            "candidate_fresh_process_source": fresh["source"],
            "candidate_fresh_process_persistent_digest_unchanged": fresh[
                "persistent_digest_unchanged"
            ],
            "candidate_fresh_process_digest_matches": (
                fresh["loaded_checkpoint_digest"] == content_digest(candidate_checkpoint)
            ),
            "default_rollback_exact": (
                content_digest(default_rollback.checkpoint())
                == content_digest(baseline_default_checkpoint)
            ),
            "candidate_organ_digest": candidate_organ_digest,
            "lesion_organ_digest": lesion_organ_digest,
            "lesion_digest_differs_from_candidate": (
                candidate_organ_digest != lesion_organ_digest
            ),
        },
    }
    record["passed"] = _record_passes(record)
    return record


def run_review(*, seeds: tuple[int, ...]) -> dict[str, Any]:
    records = [_record(seed) for seed in seeds]
    manifest = {
        "format": FORMAT,
        "version": 1,
        "status": "candidate-review",
        "candidate_kind": "episodic-identity-organ",
        "activation": "manual-admission-required",
        "default_identity_organ_enabled": False,
        "shared_decoder_default_fallback": True,
        "review_gain": REVIEW_GAIN,
        "records": [
            {
                "seed": record["seed"],
                "candidate_checkpoint_digest": record["candidate_checkpoint_digest"],
                "lesion_checkpoint_digest": record["lesion_checkpoint_digest"],
                "passed": record["passed"],
            }
            for record in records
        ],
    }
    return {
        "format": FORMAT,
        "version": 1,
        "status": "candidate-review",
        "candidate": {
            "candidate_digest": content_digest(manifest),
            "artifact_kind": "content-addressed-review-manifest",
            "checkpoint_payload": "generated-and-verified-in-memory",
            "candidate_kind": "episodic-identity-organ",
            "activation": "manual-admission-required",
            "default_replacement": False,
        },
        "manifest": manifest,
        "diagnostics": {
            "records": records,
            "all_records_pass": all(record["passed"] for record in records),
            "default_candidate_ready": False,
        },
        "gate": {
            "passed": all(record["passed"] for record in records),
            "criterion": (
                "candidate identity checkpoint must round-trip with lineage,"
                " preserve the B1 decoder boundary, outperform shared B2/B5"
                " controls, and remain manual/non-default"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_review(seeds=tuple(int(seed) for seed in args.seeds))
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
