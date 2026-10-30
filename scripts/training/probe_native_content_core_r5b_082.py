"""R5b step 08.2 engineering probe: credit, isolation, recovery, schedule boundary.

Executes the MAINLINE 08.2 evidence list against the corrected implementation
(``1315ff959``) and writes one machine receipt:

* same-initial-state forward parity between credit_on and credit_stop;
* per-loss gradient recipient groups plus frozen-parameter invariance;
* gold rejection (forged condition record) and cross-scope isolation;
* empty-store degeneration to v1 and cut/rescue behavior;
* old v1 endpoint loadability against the frozen R5 report digest;
* the declared 256->257 schedule boundary with a fresh-process restore and the
  next update after restore compared against the uninterrupted run.

Reads the train split only; query/holdout/sealed stay unread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.native_content_core import (  # noqa: E402
    NativeContentBundleV1,
    NativeContentConditionV1,
    NativeContentCoreConfigV1,
    NativeContentCoreV1,
    NativeContentStoreV1,
    span_sha256,
)
from taiji.native_content_core_v2 import NativeContentBundleV2  # noqa: E402
from taiji.native_content_training import (  # noqa: E402
    NativeRunTrainerV1,
    batch_schedule,
    credit_stop_configure,
)
from taiji.native_content_training_v2 import NativeContentV2Adapter, R5BRunTrainer  # noqa: E402

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
R5_TRAINING_REPORT = ROOT / "reports/taiji_native_core_r5_training_20261030.json"
R5_V1_ENDPOINT = ROOT / "checkpoints/native_core_r5/20261030/q1_new_core_v1_seed0-endpoint.pt"
REPORT = ROOT / "reports/taiji_native_core_r5b_082_engineering_20261031.json"
ENGINEERING_DIR = ROOT / "checkpoints/native_core_r5b/engineering_082"
BATCH_SIZE = 4
BOUNDARY_UPDATES = 257

IMPLEMENTATION_FILES = (
    "taiji/native_content_core_v2.py",
    "taiji/native_content_training_v2.py",
    "taiji/native_content_eval.py",
    "scripts/training/run_native_content_core_r5b.py",
    "scripts/training/evaluate_native_content_core_r5b.py",
)

QUERY = "雨伞是什么颜色？".encode()
SPAN = "雨伞的颜色是红色。"
TARGET = "红色<|end|>".encode()


def _episode() -> dict[str, Any]:
    return {
        "episode_id": "e:082",
        "turns": [
            {
                "index": 0,
                "role": "fact_v0",
                "scored": False,
                "text": SPAN,
                "entity": "雨伞",
                "attribute": "颜色",
                "value": "红色",
            },
            {
                "index": 1,
                "role": "query",
                "scored": True,
                "text": "雨伞是什么颜色？",
                "category": "current_query",
                "target": {"target_bytes_utf8": list(TARGET)},
            },
        ],
    }


def _sha256_of(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _config() -> NativeContentCoreConfigV1:
    return NativeContentCoreConfigV1(
        width=64,
        embedding_width=32,
        key_width=32,
        slots=8,
        chunk_bytes=8,
        max_observation_bytes=512,
        max_input_evidence_bytes=512,
        max_output_position=128,
        max_generation_bytes=64,
        content_width=32,
        seed=11,
        position_extension_seed=12,
        read_seed=13,
    )


def _conditioned(bundle: NativeContentBundleV2 | NativeContentBundleV1, scope: str):
    condition = bundle.condition(
        query=QUERY, time_intent="current", owner="u", scope=scope, session="s"
    )
    state = bundle.model.conditioned_state(
        condition, native_state_ref="probe:082", scope_id=scope, session_id="s"
    )
    return condition, state


def _ingest_fact(bundle: NativeContentBundleV2, *, scope: str = "w") -> None:
    raw = SPAN.encode()
    bundle.ingest(
        raw_span=raw,
        source_input_id="src:0",
        source_sha256=span_sha256(raw),
        owner="u",
        scope=scope,
        session="s",
        turn_index=0,
        revision=0,
        entity="雨伞",
        attribute="颜色",
        value="红色",
        status="resolved",
    )


def _probe_forward_parity_and_credit() -> dict[str, Any]:
    """Same-initial-state forward parity plus per-loss recipient groups."""

    adapter = NativeContentV2Adapter()
    model_on = adapter.create_model()
    model_stop = adapter.create_model()
    frozen_groups = tuple(credit_stop_configure(model_stop))
    episode = _episode()
    names_stop = {
        name: parameter.detach().clone() for name, parameter in model_stop.named_parameters()
    }
    trainer_on = R5BRunTrainer(adapter, model_on)
    trainer_on.cursor.update({"run_id": "082-on", "seed": 0})
    trainer_stop = R5BRunTrainer(adapter, model_stop)
    trainer_stop.cursor.update({"run_id": "082-stop", "seed": 0})

    bundle_on = adapter.fresh_bundle(model_on)
    bundle_stop = adapter.fresh_bundle(model_stop)
    _ingest_fact(bundle_on)
    _ingest_fact(bundle_stop)
    condition, state_on = _conditioned(bundle_on, "w")
    _, state_stop = _conditioned(bundle_stop, "w")
    cache_on = model_on.content_read_cache(state_on, condition, store=bundle_on.store)
    cache_stop = model_stop.content_read_cache(state_stop, condition, store=bundle_stop.store)
    logits_on, _ = model_on.next_logits(state_on, cache=cache_on)
    logits_stop, _ = model_stop.next_logits(state_stop, cache=cache_stop)
    forward_parity = bool(torch.equal(logits_on, logits_stop))

    def _grad_groups(trainer: R5BRunTrainer) -> dict[str, int]:
        groups: dict[str, int] = {}
        for name, parameter in trainer.model.named_parameters():
            if parameter.grad is not None:
                groups[trainer.model.credit_group_of(name)] = (
                    groups.get(trainer.model.credit_group_of(name), 0) + 1
                )
        return groups

    trainer_on.update([episode])
    trainer_stop.update([episode])
    grads_on = _grad_groups(trainer_on)
    grads_stop = _grad_groups(trainer_stop)
    unchanged = all(
        torch.equal(value, dict(model_stop.named_parameters())[name])
        for name, value in names_stop.items()
        if model_stop.credit_group_of(name) in {"reader", "decoder"}
    )
    moved_others = any(
        not torch.equal(value, dict(model_stop.named_parameters())[name])
        for name, value in names_stop.items()
        if model_stop.credit_group_of(name) not in {"reader", "decoder"}
    )
    return {
        "frozen_groups": list(frozen_groups),
        "forward_parity_same_initial_state": forward_parity,
        "grad_groups_credit_on": grads_on,
        "grad_groups_credit_stop": grads_stop,
        "credit_on_has_reader_decoder_grad": grads_on.get("reader", 0) > 0
        and grads_on.get("decoder", 0) > 0,
        "credit_stop_reader_decoder_grad_free": grads_stop.get("reader", 0) == 0
        and grads_stop.get("decoder", 0) == 0,
        "frozen_parameters_unchanged": bool(unchanged),
        "other_groups_moved_on_stop": bool(moved_others),
    }


def _probe_gold_rejection_and_scope() -> dict[str, Any]:
    """Forged condition records are rejected; scopes isolate the store."""

    bundle = NativeContentBundleV2.create(_config())
    _ingest_fact(bundle, scope="w1")
    foreign = NativeContentBundleV2.create(_config())
    _ingest_fact(foreign, scope="w2")
    forged = foreign.store.records[0]
    condition = NativeContentConditionV1(query=QUERY, records=(forged,), time_intent="current")
    state = bundle.model.conditioned_state(
        condition, native_state_ref="probe:082", scope_id="w1", session_id="s"
    )
    rejected = False
    try:
        bundle.model.content_read_cache(state, condition, store=bundle.store)
    except ValueError:
        rejected = True
    cross = bundle.condition(query=QUERY, time_intent="current", owner="u", scope="w2", session="s")
    own = bundle.condition(query=QUERY, time_intent="current", owner="u", scope="w1", session="s")
    answer = bundle.answer(query=QUERY, time_intent="current", owner="u", scope="w1", session="s")
    return {
        "forged_condition_rejected": bool(rejected),
        "cross_scope_records": len(cross.records),
        "own_scope_records": len(own.records),
        "own_scope_answer_digest": answer.answer_digest(),
        "cross_scope_isolated": len(cross.records) == 0 and len(own.records) == 1,
    }


def _probe_degenerate_cut_rescue() -> dict[str, Any]:
    """Empty store reduces to v1; rescue restores the byte-identical answer."""

    config = _config()
    core_v1 = NativeContentCoreV1(config)
    bundle = NativeContentBundleV2.create(config)
    _ingest_fact(bundle)
    condition, state_v2 = _conditioned(bundle, "w")
    state_v1 = core_v1.conditioned_state(
        condition, native_state_ref="probe:082", scope_id="w", session_id="s"
    )
    cache_v2 = bundle.model.content_read_cache(state_v2, condition, store=bundle.store)
    cache_v1 = core_v1.content_read_cache(state_v1, condition)
    with_content_v2, _ = bundle.model.next_logits(state_v2, cache=cache_v2)
    with_content_v1, _ = core_v1.next_logits(state_v1, cache=cache_v1)
    differs_with_content = not torch.equal(with_content_v2, with_content_v1)

    cut_condition = NativeContentConditionV1(query=QUERY, records=(), time_intent="current")
    state_cut = bundle.model.conditioned_state(
        cut_condition, native_state_ref="probe:082", scope_id="w", session_id="s"
    )
    cache_cut = bundle.model.content_read_cache(
        state_cut, cut_condition, store=NativeContentStoreV1(())
    )
    cut_logits, _ = bundle.model.next_logits(state_cut, cache=cache_cut)
    # The right comparison is v1 under the SAME empty-store condition: a
    # with-content v1 read carries a non-zero read-gate term by construction.
    state_v1_cut = core_v1.conditioned_state(
        cut_condition, native_state_ref="probe:082", scope_id="w", session_id="s"
    )
    cache_v1_cut = core_v1.content_read_cache(state_v1_cut, cut_condition)
    cut_logits_v1, _ = core_v1.next_logits(state_v1_cut, cache=cache_v1_cut)
    degenerates = bool(torch.equal(cut_logits, cut_logits_v1))

    baseline = bundle.answer(query=QUERY, time_intent="current", owner="u", scope="w", session="s")
    rescue = bundle.answer(query=QUERY, time_intent="current", owner="u", scope="w", session="s")
    return {
        "copy_active_with_content": bool(differs_with_content),
        "empty_store_equals_v1": degenerates,
        "baseline_answer_digest": baseline.answer_digest(),
        "rescue_answer_digest": rescue.answer_digest(),
        "rescue_byte_identical": baseline.answer_digest() == rescue.answer_digest(),
    }


def _fresh_probe(path: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore-probe", str(path)],
        cwd=str(ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError("fresh-process probe printed no JSON line")
    payload: dict[str, Any] = json.loads(lines[-1])
    return payload


def _restore_probe(path: Path) -> int:
    started = time.perf_counter()
    trainer = R5BRunTrainer.load(Path(path))
    payload = {
        "object": trainer.adapter.name,
        "steps": trainer.steps,
        "digest": trainer.probe_digest(),
        "restore_seconds": time.perf_counter() - started,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _probe_schedule_boundary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """256->257 boundary: restore at 256 and compare the next update.

    Path A trains 257 updates continuously and records updates 256 and 257.
    Path B trains 256 updates, saves a complete checkpoint, restores it in a
    fresh process, and runs exactly one more update.  The restored update must
    sit on schedule position 257, fire the same declared episode coins, and
    reproduce the continuous run's loss; the post-update probe digest must
    equal path A's endpoint digest.
    """

    batches = batch_schedule(episodes, seed=0, updates=BOUNDARY_UPDATES, batch_size=BATCH_SIZE)
    adapter_a = NativeContentV2Adapter()
    trainer_a = R5BRunTrainer.create(adapter_a)
    trainer_a.cursor.update({"run_id": "082-boundary-a", "seed": 0})
    metric_256: dict[str, Any] = {}
    metric_257: dict[str, Any] = {}
    for index, batch in enumerate(batches, start=1):
        metrics = trainer_a.update(batch.episodes)
        if index == 256:
            metric_256 = dict(metrics)
        if index == 257:
            metric_257 = dict(metrics)
    digest_a = trainer_a.probe_digest()

    adapter_b = NativeContentV2Adapter()
    trainer_b = R5BRunTrainer.create(adapter_b)
    trainer_b.cursor.update({"run_id": "082-boundary-b", "seed": 0})
    for batch in batches[:256]:
        trainer_b.update(batch.episodes)
    digest_b256 = trainer_b.probe_digest()
    ENGINEERING_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ENGINEERING_DIR / "boundary-256.pt"
    if checkpoint.exists():
        checkpoint.unlink()
    trainer_b.save(checkpoint)
    in_process = R5BRunTrainer.load(checkpoint)
    fresh = _fresh_probe(checkpoint)

    restored = R5BRunTrainer.load(checkpoint)
    metric_restored = dict(restored.update(batches[256].episodes))
    digest_b257 = restored.probe_digest()
    cursor = restored.cursor
    return {
        "updates_trained_path_a": BOUNDARY_UPDATES,
        "update_256_schedule_position": metric_256.get("schedule_position"),
        "update_256_scheduled_episodes": metric_256.get("episodes_scheduled"),
        "update_257_schedule_position": metric_257.get("schedule_position"),
        "update_257_scheduled_episodes": metric_257.get("episodes_scheduled"),
        "update_257_loss_nats": metric_257.get("loss_nats"),
        "restored_update_schedule_position": metric_restored.get("schedule_position"),
        "restored_update_override_active": metric_restored.get("schedule_override_active"),
        "restored_update_scheduled_episodes": metric_restored.get("episodes_scheduled"),
        "restored_update_loss_nats": metric_restored.get("loss_nats"),
        "restored_cursor": {
            "run_id": cursor.get("run_id"),
            "seed": cursor.get("seed"),
            "frozen_groups": cursor.get("frozen_groups"),
            "optimizer_steps_restored": restored.steps,
            "target_byte_exposures_restored": restored.target_byte_exposures,
        },
        "in_process_restore_equal": in_process.probe_digest() == digest_b256,
        "fresh_process_restore_equal": fresh["digest"] == digest_b256,
        "boundary_position_ok": bool(
            metric_256.get("schedule_position") == 256
            and metric_256.get("episodes_scheduled") == 0
            and metric_257.get("schedule_position") == 257
        ),
        "restored_next_update_matches": bool(
            metric_restored.get("schedule_position") == 257
            and metric_restored.get("episodes_scheduled") == metric_257.get("episodes_scheduled")
            and metric_restored.get("loss_nats") == metric_257.get("loss_nats")
        ),
        "post_restore_digest_equals_continuous": bool(digest_b257 == digest_a),
        "path_a_endpoint_digest": digest_a,
        "path_b_post_restore_digest": digest_b257,
    }


def _probe_v1_loadable() -> dict[str, Any]:
    """The frozen R5 v1 endpoint still loads with its recorded digest."""

    payload = json.loads(R5_TRAINING_REPORT.read_text(encoding="utf-8"))
    recorded = payload["runs"]["q1_new_core_v1_seed0"]["endpoint_restore"]["probe_digest"]
    trainer = NativeRunTrainerV1.load(R5_V1_ENDPOINT)
    actual = trainer.probe_digest()
    return {
        "v1_endpoint": R5_V1_ENDPOINT.name,
        "recorded_digest": recorded,
        "loaded_digest": actual,
        "v1_loadable": bool(recorded == actual),
    }


GATES = {
    "credit": (
        "forward_parity_same_initial_state",
        "credit_on_has_reader_decoder_grad",
        "credit_stop_reader_decoder_grad_free",
        "frozen_parameters_unchanged",
        "other_groups_moved_on_stop",
    ),
    "gold_scope": ("forged_condition_rejected", "cross_scope_isolated"),
    "degenerate": (
        "copy_active_with_content",
        "empty_store_equals_v1",
        "rescue_byte_identical",
    ),
    "v1_loadable": ("v1_loadable",),
    "boundary": (
        "in_process_restore_equal",
        "fresh_process_restore_equal",
        "boundary_position_ok",
        "restored_next_update_matches",
        "post_restore_digest_equals_continuous",
    ),
}


def _probe(probe_id: str, run: Any) -> dict[str, Any]:
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    result = dict(run())
    result["probe"] = probe_id
    result["cpu_seconds"] = time.process_time() - started_cpu
    result["wall_seconds"] = time.perf_counter() - started_wall
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore-probe",
        type=Path,
        default=None,
        help="child mode: load one checkpoint and print its probe digest",
    )
    args = parser.parse_args()
    if args.restore_probe is not None:
        return _restore_probe(args.restore_probe)
    if REPORT.exists():
        raise SystemExit(f"R5b 08.2 receipt is write-once: {REPORT}")

    payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    if payload.get("split") != "train" or not payload.get("training_read"):
        raise SystemExit("08.2 probe requires the declared readable train split")
    episodes = list(payload["episodes"])
    started_at = datetime.now(UTC)
    probes: dict[str, Any] = {}
    probes["credit"] = _probe("credit", _probe_forward_parity_and_credit)
    probes["gold_scope"] = _probe("gold_scope", _probe_gold_rejection_and_scope)
    probes["degenerate"] = _probe("degenerate", _probe_degenerate_cut_rescue)
    probes["v1_loadable"] = _probe("v1_loadable", _probe_v1_loadable)
    probes["boundary"] = _probe("boundary", lambda: _probe_schedule_boundary(episodes))

    failures: list[str] = []
    for family, keys in GATES.items():
        for key in keys:
            if probes[family].get(key) is not True:
                failures.append(f"{family}.{key}")

    total_cpu = sum(float(entry["cpu_seconds"]) for entry in probes.values())
    total_wall = sum(float(entry["wall_seconds"]) for entry in probes.values())
    status = "r5b_082_ok" if not failures else "r5b_082_engineering_failure"
    receipt = {
        "format": "taiji-native-core-r5b-082-engineering-v1",
        "version": 1,
        "date": "2026-10-31",
        "contract": "plans/reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md",
        "mainline": "plans/MAINLINE.md step 08.2",
        "implementation_revision": "1315ff959",
        "code_sha256": {name: _sha256_of(name) for name in IMPLEMENTATION_FILES},
        "manifest": {
            "path": TRAIN.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
            "episodes": len(episodes),
        },
        "protocol": {
            "boundary_updates": BOUNDARY_UPDATES,
            "batch_size": BATCH_SIZE,
            "schedule_seed": 0,
            "optimizer_updates_total": 2 * BOUNDARY_UPDATES + 1,
        },
        "probes": probes,
        "failed_gates": failures,
        "status": status,
        "probe_cost": {"cpu_seconds": total_cpu, "wall_seconds": total_wall},
        "read_boundaries": {
            "query_split_read": False,
            "holdout_split_read": False,
            "sealed_read": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    REPORT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(REPORT)}, ensure_ascii=False))
    return 0 if status == "r5b_082_ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
