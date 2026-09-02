"""Run mixed B1/B2/B5 shadow admission for the optional identity organ.

The shadow verifies that episodic identity evidence remains scoped to action and
memory queries. Byte prediction and generation use use_identity=False and must
remain equivalent to the shared-only control.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    REVIEW_GAIN,
    _accuracy,
    _actions,
    _config,
    _review_config,
)
from scripts.training.eval_taiji_m1_identity_organ_canary import (  # noqa: E402
    _fresh_process_probe,
    _organ_digest,
    _query,
)
from taiji import DelayedMemoryTask, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-identity-shadow-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_identity_shadow_20260902.json"
B1_TRAIN = (
    b"Taiji native prediction observes byte streams and learns local transitions. "
    b"Memory is queried through explicit action context. "
) * 12
B1_HOLDOUT = (
    b"Native byte prediction remains separate from episodic identity evidence. "
) * 4
B1_RETENTION = (
    b"Read-only byte scoring must preserve the learned native checkpoint. "
) * 4


def _b1_score(model: Taiji, *, train: bool) -> dict[str, float]:
    if train:
        model.learn_bytes(B1_TRAIN, epochs=1)
    return model.score_bytes(B1_HOLDOUT)


def _b1_retention(model: Taiji) -> dict[str, float]:
    return model.score_bytes(B1_RETENTION)


def _prepare_b2(model: Taiji, episodes: tuple[Any, ...]) -> None:
    for episode in episodes:
        DelayedMemoryTask._write_episode(model, episode)


def _shadow_record(seed: int) -> dict[str, Any]:
    b2 = build_delayed_memory_smoke_corpus(count=16)
    b5 = build_b5_corpus(train_count=16, holdout_count=8, retention_count=8)
    b2_actions = _actions(*b2.train)
    b5_actions = _actions(*b5.phase_a_train, *b5.phase_b_train)

    identity = Taiji(
        _review_config(seed, enabled=True), episode_id=f"m1-30-identity-{seed}"
    )
    control = Taiji(_config(seed, enabled=False), episode_id=f"m1-30-control-{seed}")
    _prepare_b2(identity, b2.train)
    _prepare_b2(control, b2.train)

    identity_b2 = _accuracy(identity, b2.holdout, b2_actions)
    control_b2 = _accuracy(control, b2.holdout, b2_actions)

    identity_b1_train = _b1_score(identity, train=True)
    identity_b1_retention = _b1_retention(identity)
    control_b1_train = _b1_score(control, train=True)
    control_b1_retention = _b1_retention(control)
    identity_b1_after_b2_checkpoint = content_digest(identity.checkpoint())

    for episode in b5.phase_a_train:
        DelayedMemoryTask._write_episode(identity, episode)
        DelayedMemoryTask._write_episode(control, episode)
    parent_identity_old = _accuracy(identity, b5.phase_a_holdout, b5_actions)
    parent_control_old = _accuracy(control, b5.phase_a_holdout, b5_actions)

    for episode in b5.phase_b_train:
        DelayedMemoryTask._write_episode(identity, episode)
        DelayedMemoryTask._write_episode(control, episode)
    child_identity_old = _accuracy(identity, b5.phase_a_holdout, b5_actions)
    child_identity_new = _accuracy(identity, b5.phase_b_holdout, b5_actions)
    child_control_old = _accuracy(control, b5.phase_a_holdout, b5_actions)
    child_control_new = _accuracy(control, b5.phase_b_holdout, b5_actions)

    identity_b1_after_b5 = _b1_score(identity, train=False)
    identity_b1_retention_after_b5 = _b1_retention(identity)
    control_b1_after_b5 = _b1_score(control, train=False)
    control_b1_retention_after_b5 = _b1_retention(control)

    no_change_before = _organ_digest(identity)
    bound_step = _query(identity, b5.phase_a_holdout[0].cue)
    unknown_step = _query(identity, 250)
    no_change_after = _organ_digest(identity)

    checkpoint = identity.checkpoint()
    fresh = _fresh_process_probe(checkpoint, b5.phase_a_holdout[0].cue)
    return {
        "seed": seed,
        "review_gain": REVIEW_GAIN,
        "b1": {
            "after_b2_identity": identity_b1_train,
            "after_b2_shared_control": control_b1_train,
            "after_b2_mean_surprise_delta": (
                identity_b1_train["mean_surprise"] - control_b1_train["mean_surprise"]
            ),
            "after_b2_retention_identity": identity_b1_retention,
            "after_b2_retention_shared_control": control_b1_retention,
            "after_b2_retention_mean_surprise_delta": (
                identity_b1_retention["mean_surprise"]
                - control_b1_retention["mean_surprise"]
            ),
            "after_b5_identity": identity_b1_after_b5,
            "after_b5_shared_control": control_b1_after_b5,
            "after_b5_mean_surprise_delta": (
                identity_b1_after_b5["mean_surprise"]
                - control_b1_after_b5["mean_surprise"]
            ),
            "after_b5_retention_identity": identity_b1_retention_after_b5,
            "after_b5_retention_shared_control": control_b1_retention_after_b5,
            "after_b5_retention_mean_surprise_delta": (
                identity_b1_retention_after_b5["mean_surprise"]
                - control_b1_retention_after_b5["mean_surprise"]
            ),
            "identity_checkpoint_digest_after_b2": identity_b1_after_b2_checkpoint,
        },
        "b2": {
            "identity_holdout": identity_b2,
            "shared_control_holdout": control_b2,
        },
        "b5": {
            "parent_identity_old_holdout": parent_identity_old,
            "parent_shared_control_old_holdout": parent_control_old,
            "child_identity_old_holdout": child_identity_old,
            "child_identity_new_holdout": child_identity_new,
            "child_shared_control_old_holdout": child_control_old,
            "child_shared_control_new_holdout": child_control_new,
        },
        "provenance": {
            "bound_source": bound_step.identity_recall.source,
            "bound_provenance": bound_step.identity_recall.provenance,
            "unbound_source": unknown_step.identity_recall.source,
            "unbound_provenance": unknown_step.identity_recall.provenance,
            "final_action_owner": "ByteMotor",
            "action_intent_generated": False,
        },
        "no_change": {
            "bound_used": bound_step.identity_recall.used,
            "unbound_fallback": not unknown_step.identity_recall.used,
            "organ_digest_unchanged": no_change_before == no_change_after,
        },
        "checkpoint": {
            "fresh_process_source": fresh["source"],
            "fresh_process_persistent_digest_unchanged": fresh[
                "persistent_digest_unchanged"
            ],
            "fresh_process_checkpoint_digest_matches": (
                fresh["loaded_checkpoint_digest"] == content_digest(checkpoint)
            ),
        },
    }


def _record_passes(record: dict[str, Any]) -> bool:
    b1 = record["b1"]
    b2 = record["b2"]
    b5 = record["b5"]
    provenance = record["provenance"]
    no_change = record["no_change"]
    checkpoint = record["checkpoint"]
    return bool(
        abs(b1["after_b2_mean_surprise_delta"]) <= 1e-12
        and abs(b1["after_b2_retention_mean_surprise_delta"]) <= 1e-12
        and abs(b1["after_b5_mean_surprise_delta"]) <= 1e-12
        and abs(b1["after_b5_retention_mean_surprise_delta"]) <= 1e-12
        and b2["identity_holdout"] >= b2["shared_control_holdout"]
        and b5["child_identity_old_holdout"] >= b5["parent_identity_old_holdout"]
        and b5["child_identity_new_holdout"] >= b5["child_shared_control_new_holdout"]
        and provenance["bound_source"] == "identity-route"
        and provenance["unbound_source"] == "shared-fallback"
        and provenance["final_action_owner"] == "ByteMotor"
        and provenance["action_intent_generated"] is False
        and no_change["bound_used"]
        and no_change["unbound_fallback"]
        and no_change["organ_digest_unchanged"]
        and checkpoint["fresh_process_source"] == "identity-route"
        and checkpoint["fresh_process_persistent_digest_unchanged"]
        and checkpoint["fresh_process_checkpoint_digest_matches"]
    )


def run_shadow(*, seeds: tuple[int, ...]) -> dict[str, Any]:
    records = [_shadow_record(seed) for seed in seeds]
    return {
        "seeds": list(seeds),
        "default_identity_organ_enabled": False,
        "identity_evidence_scope": "explicit episodic/action query; byte prediction and generation disabled",
        "records": records,
        "all_records_pass": all(_record_passes(record) for record in records),
        "default_candidate_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    diagnostics = run_shadow(seeds=tuple(int(seed) for seed in args.seeds))
    result = {
        "format": FORMAT,
        "version": 1,
        "status": "shadow-admission",
        "identity_route_default": "disabled",
        "shared_decoder_default_fallback": True,
        "action_intent_execution": False,
        "diagnostics": diagnostics,
        "canary_passed": bool(diagnostics["all_records_pass"]),
        "default_candidate_ready": False,
        "report_path": str(args.report),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
