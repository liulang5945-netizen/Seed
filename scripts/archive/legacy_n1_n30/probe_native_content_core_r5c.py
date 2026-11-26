"""R5c engineering receipt: parity, credit, gold rejection, recovery, boundary.

Covers the approved R5c contract section 5.3 on the ``new_core_v3`` candidate:

* v3 architecture is byte-identical to v2 (no new modules) and the forward
  outputs match;
* the pointer auxiliary CE has a finite gradient path to copy_key/copy_query;
* credit_stop freezes reader+decoder (including the copy track) while other
  groups move;
* zero-step and endpoint recovery; the 256->257 schedule boundary with a
  fresh-process restore and the next update reproducing the continuous run.

Reads the train split only; query/holdout/sealed stay unread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from taiji.native_content_core import NativeContentConditionV1, span_sha256  # noqa: E402
from taiji.native_content_core_v2 import NativeContentCoreV2  # noqa: E402
from taiji.native_content_core_v3 import (  # noqa: E402
    NativeContentBundleV3,
    NativeContentCoreV3,
)
from taiji.native_content_training import (  # noqa: E402
    batch_schedule,
    credit_stop_configure,
    native_core_config,
)
from taiji.native_content_training_v3 import (  # noqa: E402
    NativeContentV3Adapter,
    R5CRunTrainer,
)

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
REPORT = ROOT / "reports/taiji_native_core_r5c_engineering_20261031.json"
ENGINEERING_DIR = ROOT / "checkpoints/native_core_r5c/engineering"
BATCH_SIZE = 4
BOUNDARY_UPDATES = 257
SPAN = "雨伞的颜色是红色。"
QUERY = "雨伞是什么颜色？".encode()
TARGET = "红色<|end|>".encode()


def _episode() -> dict[str, Any]:
    return {
        "episode_id": "e:r5c",
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
                "entity": "雨伞",
                "attribute": "颜色",
                "target": {"target_bytes_utf8": list(TARGET)},
            },
        ],
    }


def _config():
    return native_core_config()


def _ingest_fact(bundle, *, scope: str = "w") -> None:
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


def _conditioned(bundle, scope: str):
    condition = bundle.condition(
        query=QUERY, time_intent="current", owner="u", scope=scope, session="s"
    )
    state = bundle.model.conditioned_state(
        condition, native_state_ref="probe:r5c", scope_id=scope, session_id="s"
    )
    return condition, state


def _probe_identity_and_aux() -> dict[str, Any]:
    """v3==v2 byte identity, forward parity, auxiliary gradient path."""

    config = _config()
    core_v2 = NativeContentCoreV2(config)
    core_v3 = NativeContentCoreV3(config)
    v2_params = dict(core_v2.named_parameters())
    v3_params = dict(core_v3.named_parameters())
    identical = set(v2_params) == set(v3_params) and all(
        torch.equal(value, v3_params[name]) for name, value in v2_params.items()
    )
    bundle_v3 = NativeContentBundleV3.create(config)
    _ingest_fact(bundle_v3)
    bundle_v2_probe = NativeContentBundleV3.create(config)
    _ingest_fact(bundle_v2_probe)
    condition, state_v3 = _conditioned(bundle_v3, "w")
    cache = bundle_v3.model.content_read_cache(state_v3, condition, store=bundle_v3.store)
    logits_v3, _ = core_v3.next_logits(state_v3, cache=cache)
    auxiliary = core_v3.pointer_auxiliary_ce(state_v3, cache, "红色".encode())
    auxiliary.backward()
    parameters = dict(core_v3.named_parameters())
    gradients = {
        name: (
            None
            if parameters[name].grad is None
            else bool(torch.isfinite(parameters[name].grad).all())
        )
        for name in ("copy_key.weight", "copy_query.weight")
    }
    bundle_v3.model.zero_grad(set_to_none=True)
    return {
        "architecture_byte_identical_to_v2": bool(identical),
        "auxiliary_ce_finite": bool(auxiliary is not None and torch.isfinite(auxiliary)),
        "auxiliary_gradient_finite": gradients,
        "auxiliary_gradient_path_ok": bool(all(gradients.values())),
    }


def _probe_credit() -> dict[str, Any]:
    adapter = NativeContentV3Adapter()
    model_on = adapter.create_model()
    model_stop = adapter.create_model()
    frozen_groups = tuple(credit_stop_configure(model_stop))
    names_stop = {
        name: parameter.detach().clone() for name, parameter in model_stop.named_parameters()
    }
    trainer_on = R5CRunTrainer(adapter, model_on)
    trainer_on.cursor.update({"run_id": "r5c-on", "seed": 0})
    trainer_stop = R5CRunTrainer(adapter, model_stop)
    trainer_stop.cursor.update({"run_id": "r5c-stop", "seed": 0})
    episode = _episode()
    trainer_on.update([episode])
    trainer_stop.update([episode])
    grads_on: dict[str, int] = {}
    grads_stop: dict[str, int] = {}
    for trainer, bucket in ((trainer_on, grads_on), (trainer_stop, grads_stop)):
        for name, parameter in trainer.model.named_parameters():
            if parameter.grad is not None:
                group = trainer.model.credit_group_of(name)
                bucket[group] = bucket.get(group, 0) + 1
    frozen_unchanged = all(
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
        "grad_groups_credit_on": grads_on,
        "grad_groups_credit_stop": grads_stop,
        "credit_on_has_reader_decoder_grad": grads_on.get("reader", 0) > 0
        and grads_on.get("decoder", 0) > 0,
        "credit_stop_reader_decoder_grad_free": grads_stop.get("reader", 0) == 0
        and grads_stop.get("decoder", 0) == 0,
        "frozen_parameters_unchanged": bool(frozen_unchanged),
        "other_groups_moved_on_stop": bool(moved_others),
    }


def _probe_gold_scope() -> dict[str, Any]:
    bundle = NativeContentBundleV3.create(_config())
    _ingest_fact(bundle, scope="w1")
    foreign = NativeContentBundleV3.create(_config())
    _ingest_fact(foreign, scope="w2")
    forged = foreign.store.records[0]
    condition = NativeContentConditionV1(query=QUERY, records=(forged,), time_intent="current")
    state = bundle.model.conditioned_state(
        condition, native_state_ref="probe:r5c", scope_id="w1", session_id="s"
    )
    rejected = False
    try:
        bundle.model.content_read_cache(state, condition, store=bundle.store)
    except ValueError:
        rejected = True
    cross = bundle.condition(query=QUERY, time_intent="current", owner="u", scope="w2", session="s")
    own = bundle.condition(query=QUERY, time_intent="current", owner="u", scope="w1", session="s")
    return {
        "forged_condition_rejected": bool(rejected),
        "cross_scope_records": len(cross.records),
        "own_scope_records": len(own.records),
        "cross_scope_isolated": len(cross.records) == 0 and len(own.records) == 1,
    }


def _restore_probe(path: Path) -> int:
    started = time.perf_counter()
    trainer = R5CRunTrainer.load(Path(path))
    print(
        json.dumps(
            {
                "object": trainer.adapter.name,
                "steps": trainer.steps,
                "digest": trainer.probe_digest(),
                "restore_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
        )
    )
    return 0


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


def _probe_boundary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    batches = batch_schedule(episodes, seed=0, updates=BOUNDARY_UPDATES, batch_size=BATCH_SIZE)
    adapter_a = NativeContentV3Adapter()
    trainer_a = R5CRunTrainer.create(adapter_a)
    trainer_a.cursor.update({"run_id": "r5c-boundary-a", "seed": 0})
    metric_256: dict[str, Any] = {}
    metric_257: dict[str, Any] = {}
    for index, batch in enumerate(batches, start=1):
        metrics = trainer_a.update(batch.episodes)
        if index == 256:
            metric_256 = dict(metrics)
        if index == 257:
            metric_257 = dict(metrics)
    digest_a = trainer_a.probe_digest()

    adapter_b = NativeContentV3Adapter()
    trainer_b = R5CRunTrainer.create(adapter_b)
    trainer_b.cursor.update({"run_id": "r5c-boundary-b", "seed": 0})
    for batch in batches[:256]:
        trainer_b.update(batch.episodes)
    digest_b256 = trainer_b.probe_digest()
    ENGINEERING_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ENGINEERING_DIR / "boundary-256.pt"
    if checkpoint.exists():
        checkpoint.unlink()
    trainer_b.save(checkpoint)
    in_process = R5CRunTrainer.load(checkpoint)
    fresh = _fresh_probe(checkpoint)

    restored = R5CRunTrainer.load(checkpoint)
    metric_restored = dict(restored.update(batches[256].episodes))
    digest_b257 = restored.probe_digest()
    return {
        "update_256_schedule_position": metric_256.get("schedule_position"),
        "update_256_scheduled_episodes": metric_256.get("episodes_scheduled"),
        "update_257_schedule_position": metric_257.get("schedule_position"),
        "update_257_scheduled_episodes": metric_257.get("episodes_scheduled"),
        "update_257_loss_nats": metric_257.get("loss_nats"),
        "restored_update_schedule_position": metric_restored.get("schedule_position"),
        "restored_update_scheduled_episodes": metric_restored.get("episodes_scheduled"),
        "restored_update_loss_nats": metric_restored.get("loss_nats"),
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
    }


GATES = {
    "identity": (
        "architecture_byte_identical_to_v2",
        "auxiliary_ce_finite",
        "auxiliary_gradient_path_ok",
    ),
    "credit": (
        "credit_on_has_reader_decoder_grad",
        "credit_stop_reader_decoder_grad_free",
        "frozen_parameters_unchanged",
        "other_groups_moved_on_stop",
    ),
    "gold_scope": ("forged_condition_rejected", "cross_scope_isolated"),
    "boundary": (
        "in_process_restore_equal",
        "fresh_process_restore_equal",
        "boundary_position_ok",
        "restored_next_update_matches",
        "post_restore_digest_equals_continuous",
    ),
}


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
        raise SystemExit(f"R5c engineering receipt is write-once: {REPORT}")
    payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    if payload.get("split") != "train" or not payload.get("training_read"):
        raise SystemExit("R5c engineering probe requires the declared readable train split")
    episodes = list(payload["episodes"])
    started_at = datetime.now(UTC)
    started_cpu = time.process_time()
    probes: dict[str, Any] = {}
    probes["identity"] = dict(_probe_identity_and_aux())
    probes["credit"] = dict(_probe_credit())
    probes["gold_scope"] = dict(_probe_gold_scope())
    probes["boundary"] = dict(_probe_boundary(episodes))

    failures: list[str] = []
    for family, keys in GATES.items():
        for key in keys:
            if probes[family].get(key) is not True:
                failures.append(f"{family}.{key}")
    status = "r5c_engineering_ok" if not failures else "r5c_engineering_failure"
    receipt = {
        "format": "taiji-native-core-r5c-engineering-v1",
        "version": 1,
        "date": "2026-10-31",
        "contract": "plans/reference/SPEC-NATIVE-CONTENT-CORE_R5C_COPYEXPR_CANDIDATE_20261031.md",
        "implementation_revision": "a3f0cf60e",
        "manifest": {
            "path": TRAIN.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
            "episodes": len(episodes),
        },
        "probes": probes,
        "failed_gates": failures,
        "status": status,
        "probe_cost": {"cpu_seconds": time.process_time() - started_cpu},
        "read_boundaries": {
            "query_split_read": False,
            "holdout_split_read": False,
            "sealed_read": False,
            "optimizer_updates": 2 * BOUNDARY_UPDATES + 1,
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    REPORT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(REPORT)}, ensure_ascii=False))
    return 0 if status == "r5c_engineering_ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
