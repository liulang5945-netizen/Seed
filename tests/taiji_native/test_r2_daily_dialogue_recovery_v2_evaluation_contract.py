"""Synthetic contracts for one-shot R2 recovery-v2 paired evaluation."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as evaluation


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm_fixture(tmp_path: Path, arm: str) -> tuple[dict, dict, Path]:
    arm_root = tmp_path / "checkpoints" / arm
    arm_root.mkdir(parents=True)
    source_path = tmp_path / f"{arm}-source.txt"
    source_path.write_text(f"source:{arm}", encoding="utf-8")
    bindings = {
        "run": evaluation.RUN_ID,
        "arm": arm,
        "zero_sha256": hashlib.sha256(f"zero:{arm}".encode()).hexdigest(),
        "source_hashes": {str(source_path.resolve()): _sha(source_path)},
    }
    probe_hashes = {"train_probe": hashlib.sha256(f"probe:{arm}".encode()).hexdigest()}
    readiness = {
        "schema": "taiji-r2-dialogue-recovery-readiness-v1",
        "status": "ready",
        "arm": arm,
        "bindings": bindings,
        "per_episode_cpu_reserve": evaluation.ARM_EPISODE_RESERVES[arm],
        "worst_episode_probe_hashes": probe_hashes,
        "formal_updates": 0,
        "dev_queries": 0,
        "can_promote": False,
    }
    stages = {}
    for stage, (episodes, updates, filename) in evaluation.EXPECTED_STAGE_COUNTS.items():
        checkpoint = arm_root / filename
        checkpoint.write_bytes(f"{arm}:{stage}:checkpoint".encode())
        checkpoint_sha = _sha(checkpoint)
        digest = hashlib.sha256(f"digest:{arm}:{stage}".encode()).hexdigest()
        restore = {
            "complete": True,
            "equal": True,
            "stage": stage,
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "checkpoint_digest": digest,
            "episode_count": episodes,
            "global_step": updates,
            "cpu_seconds": 0.25,
        }
        stages[stage] = {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "episodes": episodes,
            "target_byte_updates": updates,
            "restore": restore,
        }
    result = {
        "schema": "taiji-r2-dialogue-recovery-result-v1",
        "status": "complete",
        "complete": True,
        "arm": arm,
        "restart_from_zero": True,
        "bindings": bindings,
        "stages": stages,
        "per_episode_cpu_reserve": evaluation.ARM_EPISODE_RESERVES[arm],
        "worst_episode_probe_hashes": probe_hashes,
        "cpu_seconds": 100.0,
        "dev_queries": 0,
        "generation_queries": 0,
        "can_promote": False,
    }
    return readiness, result, arm_root


def _valid_records(tmp_path: Path) -> tuple[dict[str, dict], dict, dict]:
    records = {}
    source_maps = []
    for arm in evaluation.ARMS:
        readiness, result, arm_root = _arm_fixture(tmp_path, arm)
        record = evaluation.validate_arm_result(arm, readiness, result, arm_root=arm_root)
        records[arm] = record
        source_maps.append(record["source_hashes"])
    baseline_path = tmp_path / "parent.pt"
    baseline_path.write_bytes(b"parent checkpoint")
    baseline = evaluation.validate_baseline_checkpoint(
        baseline_path,
        expected_sha256=_sha(baseline_path),
    )
    source_hashes = evaluation._merge_hash_maps(*source_maps)
    preflight = {
        "validated_arms": records,
        "report_hashes": {
            arm: {
                "result_sha256": hashlib.sha256(f"result:{arm}".encode()).hexdigest(),
                "readiness_sha256": hashlib.sha256(f"readiness:{arm}".encode()).hexdigest(),
            }
            for arm in evaluation.ARMS
        },
        "baseline": baseline,
        "endpoints": evaluation.build_endpoint_map(baseline, records),
        "source_hashes": source_hashes,
    }
    return records, baseline, preflight


def _readiness_path_payload(preflight: dict) -> dict:
    return evaluation._readiness_payload(preflight)


def _output_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        key: tmp_path / f"eval_{key}.{ 'jsonl' if key in {'ledger', 'progress'} else 'json' }"
        for key in (
            "readiness",
            "attempt",
            "ledger",
            "progress",
            "face",
            "generations",
            "result",
            "cost_receipt",
            "review",
            "failure",
        )
    }


def _write_readiness(paths: dict[str, Path], preflight: dict) -> None:
    paths["readiness"].write_text(
        json.dumps(_readiness_path_payload(preflight), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _prompts() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            episode_id=f"episode-{index:02d}",
            family_id=f"family-{index:02d}",
            system="system",
            context="context",
            history=(("先前问题", "先前回答"),),
            user_input=f"本轮问题{index}",
            unknown_policy="unknown-safe",
        )
        for index in range(evaluation.FACE_SIZE)
    )


def _trainer_factory(endpoints, restores):
    class FakeModel:
        def __init__(self, endpoint_id: str) -> None:
            self.endpoint_id = endpoint_id

        def checkpoint(self):
            return {"endpoint_id": self.endpoint_id}

        def parameter_count(self):
            return 100

    class FakeTrainer:
        def __init__(self, endpoint_id: str) -> None:
            self.model = FakeModel(endpoint_id)
            self.global_step = 0 if endpoint_id == "parent_baseline" else 113_209
            self.episode_count = 0 if endpoint_id == "parent_baseline" else 525

    trainers = {
        endpoint["endpoint_id"]: FakeTrainer(endpoint["endpoint_id"]) for endpoint in endpoints
    }
    states = {
        endpoint_id: {
            "model_digest": restores[endpoint_id]["model_digest"],
            "checkpoint_sha256": restores[endpoint_id]["sha256"],
            "global_step": trainer.global_step,
            "episode_count": trainer.episode_count,
            "parameter_count": 100,
        }
        for endpoint_id, trainer in trainers.items()
    }
    return trainers, object(), states


def _restore_receipts(endpoints):
    receipts = {}
    for endpoint in endpoints:
        endpoint_id = endpoint["endpoint_id"]
        parent = endpoint_id == "parent_baseline"
        receipts[endpoint_id] = {
            "complete": True,
            "equal": True,
            "endpoint_id": endpoint_id,
            "stage": "parent_baseline" if parent else "assistant_sft",
            "path": str(endpoint["path"]),
            "sha256": endpoint["sha256"],
            "checkpoint_digest": endpoint["checkpoint_digest"],
            "model_digest": f"model:{endpoint_id}",
            "episode_count": 0 if parent else 525,
            "global_step": 0 if parent else 113_209,
            "cpu_seconds": 0.01,
            "sampled_child_cpu_seconds": 0.01,
            "generation_queries": 0,
            "parameter_updates": 0,
            "optimizer_updates": 0,
            "can_promote": False,
        }
    return receipts


def _generation_output() -> dict:
    raw = "好".encode()
    return {
        "generated_text": "好",
        "raw_generated_sha256": hashlib.sha256(raw).hexdigest(),
        "generated_bytes_hex": raw.hex(),
        "generated_bytes_length": len(raw),
        "utf8_valid": True,
        "no_replacement": True,
        "response_boundary_present": True,
        "stop_reason": "end_marker",
        "pii_pattern_redacted": False,
        "repeated_4gram_rate": 0.0,
    }


def test_fixed_four_endpoint_mapping_and_156_query_denominator(tmp_path):
    records, baseline, _ = _valid_records(tmp_path)
    endpoints = evaluation.build_endpoint_map(baseline, records)
    assert [endpoint["endpoint_id"] for endpoint in endpoints] == list(evaluation.ENDPOINT_ORDER)
    assert endpoints[0]["source"] == "fixed_parent_checkpoint"
    assert all(endpoint["source"] == "assistant_sft_final" for endpoint in endpoints[1:])
    assert all(endpoint["path"].name == "assistant_sft_final.pt" for endpoint in endpoints[1:])
    assert evaluation.expected_generation_count() == 156
    with pytest.raises(evaluation.Refuse, match="exactly 39"):
        evaluation.expected_generation_count(38)


def test_recovery_restart_from_zero_must_remain_true(tmp_path):
    """The extension branch may reject restart_from_zero; the frozen branch must keep requiring it."""
    readiness, result, root = _arm_fixture(tmp_path, "warm_start")
    result["restart_from_zero"] = False
    with pytest.raises(evaluation.Refuse, match="incomplete or contains an unsafe"):
        evaluation.validate_arm_result("warm_start", readiness, result, arm_root=root)


def test_arm_result_rejects_source_drift_and_endpoint_escape(tmp_path):
    readiness, result, root = _arm_fixture(tmp_path, "warm_start")
    source_path = Path(next(iter(result["bindings"]["source_hashes"])))
    source_path.write_text("mutated", encoding="utf-8")
    with pytest.raises(evaluation.Refuse, match="source is missing or changed"):
        evaluation.validate_arm_result("warm_start", readiness, result, arm_root=root)

    readiness, result, root = _arm_fixture(tmp_path / "escape", "warm_start")
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    result["stages"]["assistant_sft"]["path"] = str(outside.resolve())
    with pytest.raises(evaluation.Refuse, match="escapes its frozen root"):
        evaluation.validate_arm_result("warm_start", readiness, result, arm_root=root)


def test_arm_result_rejects_capability_or_restore_mismatch(tmp_path):
    readiness, result, root = _arm_fixture(tmp_path, "full_turn_1m")
    result["can_promote"] = True
    with pytest.raises(evaluation.Refuse, match="unsafe query/capability"):
        evaluation.validate_arm_result("full_turn_1m", readiness, result, arm_root=root)

    readiness, result, root = _arm_fixture(tmp_path / "restore", "full_turn_1m")
    result["stages"]["pretraining"]["restore"]["global_step"] -= 1
    with pytest.raises(evaluation.Refuse, match="restore receipt denominator"):
        evaluation.validate_arm_result("full_turn_1m", readiness, result, arm_root=root)


def test_prompt_face_is_stable_unique_and_prompt_only():
    prompts = _prompts()
    first = evaluation.freeze_prompt_face(prompts)
    second = evaluation.freeze_prompt_face(prompts)
    assert first == second
    assert len(first["rows"]) == evaluation.FACE_SIZE
    assert first["rows"][0]["split"] == "dev"
    assert "assistant_target" not in first["rows"][0]
    assert "reference_answer" not in first["rows"][0]
    assert len(first["prompt_hashes"]) == 39
    with pytest.raises(evaluation.Refuse, match="repeated family or episode"):
        evaluation.freeze_prompt_face((*prompts[:-1], prompts[0]))
    changed = list(prompts)
    changed[0] = SimpleNamespace(**{**vars(prompts[0]), "user_input": "changed prompt"})
    assert evaluation.freeze_prompt_face(changed)["face_sha256"] != first["face_sha256"]
    invalid_output = _generation_output()
    invalid_output["raw_generated_sha256"] = "0" * 64
    with pytest.raises(evaluation.Refuse, match="raw SHA"):
        evaluation._validate_generation_output(invalid_output)


def test_resource_and_cpu_wall_budget_guards_fail_closed():
    snapshot = evaluation.validate_resource_snapshot([0.0, 85.0], 8.0)
    assert snapshot["available_memory_gib"] == 8.0
    with pytest.raises(evaluation.Refuse, match="85%"):
        evaluation.validate_resource_snapshot([85.1, 1.0], 16.0)
    with pytest.raises(evaluation.Refuse, match="GiB"):
        evaluation.validate_resource_snapshot([1.0, 2.0], 7.99)
    with pytest.raises(evaluation.Refuse, match="two fresh"):
        evaluation.validate_resource_snapshot([1.0], 16.0)

    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

    cpu_clock = Clock()
    wall_clock = Clock()
    budget = evaluation.EvaluationBudget(
        cpu_limit_seconds=10.0,
        wall_limit_seconds=10.0,
        cpu_clock=cpu_clock,
        wall_clock=wall_clock,
    )
    cpu_clock.value = 2.0
    budget.charge_child_cpu(3.0)
    budget.guard(reserve_cpu_seconds=5.0, reserve_wall_seconds=8.0)
    wall_clock.value = 11.0
    with pytest.raises(TimeoutError, match="wall=11.000"):
        budget.guard()


def test_metadata_preflight_is_model_and_prompt_free(tmp_path, monkeypatch):
    source_path = tmp_path / "frozen-source.txt"
    source_path.write_text("frozen", encoding="utf-8")
    baseline_path = tmp_path / "baseline.pt"
    baseline_path.write_bytes(b"baseline")
    report_paths = {}
    for arm in evaluation.ARMS:
        readiness, result, root = _arm_fixture(tmp_path, arm)
        report_root = tmp_path / "input-reports" / arm
        report_root.mkdir(parents=True)
        readiness_path = report_root / "readiness.json"
        result_path = report_root / "result.json"
        readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
        result_path.write_text(json.dumps(result), encoding="utf-8")
        report_paths[arm] = {"root": root, "readiness": readiness_path, "result": result_path}

    monkeypatch.setattr(evaluation.recovery, "paths", lambda arm: report_paths[arm])
    monkeypatch.setattr(evaluation.base, "BASELINE_CHECKPOINT", baseline_path)
    monkeypatch.setattr(evaluation.base, "BASELINE_CHECKPOINT_SHA256", _sha(baseline_path))
    monkeypatch.setattr(
        evaluation.base,
        "_select_dev_prompts",
        lambda: (_ for _ in ()).throw(AssertionError("selector must not run")),
    )
    monkeypatch.setattr(
        evaluation.torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("torch.load must not run")),
    )
    freeze_path = tmp_path / "freeze.txt"
    freeze_path.write_text("freeze", encoding="utf-8")
    freeze_hashes = {str(freeze_path.resolve()): _sha(freeze_path)}
    preflight = evaluation.metadata_preflight(eval_hashes=lambda: freeze_hashes)
    assert [endpoint["endpoint_id"] for endpoint in preflight["endpoints"]] == list(
        evaluation.ENDPOINT_ORDER
    )
    assert preflight["report_hashes"]["full_turn_1m"]["result_sha256"] == _sha(
        report_paths["full_turn_1m"]["result"]
    )


def test_validate_only_writes_idempotent_readiness_without_attempt(tmp_path):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / "reports")
    paths["readiness"].parent.mkdir(parents=True)
    first = evaluation.validate_only(paths=paths, preflight_fn=lambda: preflight)
    second = evaluation.validate_only(paths=paths, preflight_fn=lambda: preflight)
    assert first == second
    assert paths["readiness"].is_file()
    assert not paths["attempt"].exists()


def test_execute_fsyncs_restore_and_query_intents_before_calls_and_preserves_checkpoints(
    tmp_path, monkeypatch
):
    records, baseline, preflight = _valid_records(tmp_path)
    endpoints = preflight["endpoints"]
    paths = _output_paths(tmp_path / "reports")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    receipts = _restore_receipts(endpoints)
    restore_calls = []
    selector_calls = []
    generation_calls = []

    def restore(endpoint_id):
        restore_calls.append(endpoint_id)
        assert paths["attempt"].is_file()
        assert paths["ledger"].is_file()
        event = json.loads(paths["ledger"].read_text(encoding="utf-8").splitlines()[-1])
        assert event["event"] == "restore_started"
        assert event["endpoint_id"] == endpoint_id
        return receipts[endpoint_id]

    def select():
        selector_calls.append(True)
        events = [
            json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()
        ]
        completed = [event for event in events if event["event"] == "restore_completed"]
        assert [event["endpoint_id"] for event in completed] == list(evaluation.ENDPOINT_ORDER)
        assert any(event["event"] == "prompt_selection_started" for event in events)
        assert not any(event["event"] == "query_started" for event in events)
        return _prompts()

    def generate(trainer, prompt):
        generation_calls.append((trainer.model.endpoint_id, prompt.family_id))
        events = [
            json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()
        ]
        assert any(event["event"] == "query_started" for event in events)
        return _generation_output()

    monkeypatch.setattr(
        evaluation.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=16 * 1024**3),
    )
    worker = _FakeSupervisedWorker(receipts, endpoints, startup_cpu_seconds=0.5)
    result = evaluation.execute(
        **_execute_kwargs(
            paths,
            preflight,
            lambda **_kwargs: worker,
            _DurabilityApi(),
            prompt_selector=select,
            host_cpu_fn=lambda: 1_000.0,
            boot_time_fn=lambda: 1.0,
        )
    )
    assert [
        item["endpoint_id"] for item in worker.requests if item["operation"] == "restore"
    ] == list(evaluation.ENDPOINT_ORDER)
    assert selector_calls == [True]
    assert len([item for item in worker.requests if item["operation"] == "generate"]) == 156
    ledger_events = [
        json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()
    ]
    supervision_ready = next(
        event for event in ledger_events if event["event"] == "worker_supervision_ready"
    )
    first_restore = next(event for event in ledger_events if event["event"] == "restore_started")
    assert supervision_ready["worker_startup_cpu_seconds_charged_total"] == 0.5
    assert first_restore["child_cpu_seconds_before"] == 0.5
    assert result["query_attempts"] == 156
    assert result["persisted_generation_outputs"] == 156
    assert result["generation_queries"] == 156
    assert result["can_promote"] is False
    assert result["reference_answer_scoring_queries"] == 0
    assert result["exact_match_scoring_queries"] == 0
    assert result["endpoint_checkpoints_before"] == result["endpoint_checkpoints_after"]
    assert paths["result"].is_file()
    assert paths["cost_receipt"].is_file()
    receipt = json.loads(paths["cost_receipt"].read_text(encoding="utf-8"))
    assert result["cost_receipt_sha256"] == _sha(paths["cost_receipt"])
    assert receipt["cpu_seconds_total_before_receipt_publish"] >= 0
    assert receipt["wall_seconds_before_receipt_publish"] >= 0
    assert receipt["result_publication_excluded"] is True
    assert paths["review"].is_file()
    assert all(Path(item["path"]).is_file() for item in result["fresh_process_restores"].values())


def test_failed_restore_consumes_attempt_before_selector_and_cannot_retry(tmp_path):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / "reports")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    selected = []
    worker = _FakeSupervisedWorker(
        _restore_receipts(preflight["endpoints"]),
        preflight["endpoints"],
        fail_operation="restore",
    )
    kwargs = _execute_kwargs(
        paths,
        preflight,
        lambda **_kwargs: worker,
        _DurabilityApi(),
        prompt_selector=lambda: selected.append(True) or _prompts(),
        host_cpu_fn=lambda: 1_000.0,
        boot_time_fn=lambda: 1.0,
    )
    with pytest.raises(evaluation.Refuse, match="attempt is consumed"):
        evaluation.execute(**kwargs)
    with pytest.raises(evaluation.Refuse, match="attempt already exists"):
        evaluation.execute(**kwargs)
    assert len(worker.requests) == 1
    assert worker.requests[0]["endpoint_id"] == "parent_baseline"
    assert selected == []
    assert paths["attempt"].is_file()
    assert paths["failure"].is_file()
    assert not paths["face"].exists()


def test_interrupted_restore_tail_is_audited_without_resume_or_prompt(tmp_path):
    paths = _output_paths(tmp_path / "interrupted")
    paths["readiness"].parent.mkdir(parents=True)
    paths["readiness"].write_text("{}", encoding="utf-8")
    paths["attempt"].write_text("{}", encoding="utf-8")
    paths["ledger"].write_text(
        "\n".join(
            (
                json.dumps({"event": "attempt_started"}),
                json.dumps(
                    {
                        "event": "restore_started",
                        "endpoint_id": "warm_start",
                        "host_cpu_seconds_before": 1_000.0,
                        "boot_identity": 10.0,
                        "logical_cpu_upper": 24,
                        "timestamp_utc": "2026-10-21T00:00:00+00:00",
                        "unknown_tail": True,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(evaluation.Refuse, match="attempt already exists"):
        evaluation.execute(
            paths=paths,
            preflight_fn=lambda: (_ for _ in ()).throw(
                AssertionError("preflight/retry must not run")
            ),
            host_cpu_fn=lambda: 1_123.0,
            boot_time_fn=lambda: 10.0,
        )
    events = [json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()]
    tail = [event for event in events if event["event"] == "restore_tail_audited"]
    assert len(tail) == 1
    assert tail[0]["charged_child_cpu_upper"] == 123.0
    assert tail[0]["unknown_tail"] is True
    assert tail[0]["no_resume_or_retry"] is True
    assert not paths["face"].exists()


def test_interrupted_query_and_selector_intents_are_audited_unknown_and_not_retried(tmp_path):
    paths = _output_paths(tmp_path / "interrupted-query")
    paths["readiness"].parent.mkdir(parents=True)
    paths["readiness"].write_text("{}", encoding="utf-8")
    paths["attempt"].write_text("{}", encoding="utf-8")
    paths["ledger"].write_text(
        "\n".join(
            (
                json.dumps({"event": "attempt_started"}),
                json.dumps(
                    {
                        "event": "query_started",
                        "task_id": "query-crashed",
                        "operation": "generate",
                        "endpoint_id": "warm_start",
                        "host_cpu_seconds_before": 100.0,
                        "boot_identity": 10.0,
                        "logical_cpu_upper": 16,
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "unknown_tail": True,
                    }
                ),
                json.dumps(
                    {
                        "event": "prompt_selection_started",
                        "attempt_id": "synthetic-attempt",
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "unknown_tail": True,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    preflight_calls = []
    with pytest.raises(evaluation.Refuse, match="attempt already exists"):
        evaluation.execute(
            paths=paths,
            preflight_fn=lambda: preflight_calls.append(True),
            host_cpu_fn=lambda: 130.0,
            boot_time_fn=lambda: 10.0,
            durability_api=_DurabilityApi(),
        )
    assert preflight_calls == []
    events = [json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()]
    query_tail = next(event for event in events if event["event"] == "query_tail_audited")
    selector_tail = next(
        event for event in events if event["event"] == "prompt_selection_tail_audited"
    )
    assert query_tail["charged_child_cpu_upper"] == 30.0
    assert query_tail["unknown_tail"] is True
    assert selector_tail["prompt_face_persisted"] is False
    assert selector_tail["no_resume_or_retry"] is True


def test_missing_result_and_resource_failure_refuse_before_attempt_or_prompt(tmp_path):
    paths = _output_paths(tmp_path / "missing")
    paths["readiness"].parent.mkdir(parents=True)
    paths["readiness"].write_text("{}", encoding="utf-8")
    missing_readiness = tmp_path / "missing-readiness.json"
    missing_readiness.write_text("{}", encoding="utf-8")
    selector_calls = []
    with pytest.raises(evaluation.Refuse, match="cannot read evaluation metadata"):
        evaluation.metadata_preflight(
            arm_paths=lambda _arm: {
                "root": tmp_path / "missing-root",
                "readiness": missing_readiness,
                "result": tmp_path / "missing-result.json",
            },
            eval_hashes=lambda: {},
        )

    _, _, preflight = _valid_records(tmp_path / "resource")
    resource_paths = _output_paths(tmp_path / "resource-reports")
    resource_paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(resource_paths, preflight)
    with pytest.raises(evaluation.Refuse, match="resource busy"):
        evaluation.execute(
            paths=resource_paths,
            preflight_fn=lambda: preflight,
            resources_fn=lambda: (_ for _ in ()).throw(evaluation.Refuse("resource busy")),
            disk_fn=lambda _path: {"free_bytes": 2 * evaluation.MIN_OUTPUT_DISK_BYTES},
            prompt_selector=lambda: selector_calls.append(True) or _prompts(),
        )
    assert selector_calls == []
    assert not resource_paths["attempt"].exists()


def test_failed_generation_is_persisted_as_one_attempt_and_never_retried(tmp_path, monkeypatch):
    _, _, preflight = _valid_records(tmp_path)
    endpoints = preflight["endpoints"]
    paths = _output_paths(tmp_path / "reports")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    worker = _FakeSupervisedWorker(
        _restore_receipts(endpoints),
        endpoints,
        fail_operation="generate",
    )

    monkeypatch.setattr(
        evaluation.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=16 * 1024**3),
    )
    kwargs = _execute_kwargs(
        paths,
        preflight,
        lambda **_kwargs: worker,
        _DurabilityApi(),
        prompt_selector=_prompts,
        host_cpu_fn=lambda: 1_000.0,
        boot_time_fn=lambda: 1.0,
    )
    with pytest.raises(evaluation.Refuse, match="one-shot evaluation attempt is consumed"):
        evaluation.execute(**kwargs)
    with pytest.raises(evaluation.Refuse, match="attempt already exists"):
        evaluation.execute(**kwargs)
    assert len(worker.requests) == 5
    ledger = [json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()]
    started = [event for event in ledger if event["event"] == "query_started"]
    failed = [event for event in ledger if event["event"] == "query_timeout"]
    assert len(started) == 1
    assert len(failed) == 1
    assert failed[0]["endpoint_id"] == "parent_baseline"


def test_whole_run_reserve_is_10200_cpu_and_wall_seconds():
    assert evaluation._whole_run_reserve() == (10_200.0, 10_200.0)


def test_generation_deadline_shares_remaining_global_budget_and_refuses_zero():
    budget = evaluation.EvaluationBudget(
        charged_parent_cpu_seconds=13_600.0,
        charged_wall_seconds=13_600.0,
    )
    cpu_share, wall_share = evaluation._task_limits(
        budget=budget,
        operation="generate",
        remaining_tasks=156,
        future_cpu_reserve=300.0,
        future_wall_reserve=300.0,
    )
    assert 0 < cpu_share < evaluation.MAX_SINGLE_GENERATION_CPU_SECONDS
    assert cpu_share == pytest.approx(wall_share)
    exhausted = evaluation.EvaluationBudget(
        charged_parent_cpu_seconds=14_100.0,
        charged_wall_seconds=14_100.0,
    )
    with pytest.raises(TimeoutError, match="no finite global budget"):
        evaluation._task_limits(
            budget=exhausted,
            operation="generate",
            remaining_tasks=156,
            future_cpu_reserve=300.0,
            future_wall_reserve=300.0,
        )


def test_zero_global_budget_refuses_before_worker_or_selector(tmp_path):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / "zero-budget")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    workers, selectors = [], []
    with pytest.raises(TimeoutError, match="budget exceeded"):
        evaluation.execute(
            **_execute_kwargs(
                paths,
                preflight,
                lambda **_kwargs: workers.append(True),
                _DurabilityApi(),
                prompt_selector=lambda: selectors.append(True) or _prompts(),
                budget_factory=lambda **_kwargs: evaluation.EvaluationBudget(
                    charged_parent_cpu_seconds=14_300.0,
                    charged_wall_seconds=14_300.0,
                ),
            )
        )
    assert workers == []
    assert selectors == []
    assert not paths["attempt"].exists()


class _DurabilityApi:
    def __init__(
        self, *, filesystem="NTFS", drive_type=3, is_unc=False, fail_flush=False, fail_move_at=None
    ):
        self.filesystem = filesystem
        self.drive_type = drive_type
        self.is_unc = is_unc
        self.fail_flush = fail_flush
        self.fail_move_at = fail_move_at
        self.move_calls = []

    def inspect(self, _path):
        return {
            "is_unc": self.is_unc,
            "drive_type": self.drive_type,
            "filesystem": self.filesystem,
            "volume_root": "C:\\",
        }

    def flush_file(self, fd):
        if self.fail_flush:
            raise OSError("synthetic FlushFileBuffers failure")
        os.fsync(fd)

    def move_file_write_through(self, source, destination, flags):
        self.move_calls.append((source, destination, flags))
        assert flags == evaluation.MOVEFILE_WRITE_THROUGH
        if self.fail_move_at == len(self.move_calls):
            raise OSError("synthetic MoveFileExW failure")
        os.rename(source, destination)


class _FakeSupervisedWorker:
    def __init__(
        self,
        receipts,
        endpoints,
        *,
        fail_operation=None,
        startup_cpu_seconds=0.0,
        ipc_write_thread_stopped=True,
        writer_stops_after_terminate_calls=None,
    ):
        self.receipts = receipts
        self.endpoints = {item["endpoint_id"]: item for item in endpoints}
        self.fail_operation = fail_operation
        self.initial_child_cpu_seconds = startup_cpu_seconds
        self.ipc_write_thread_stopped = ipc_write_thread_stopped
        self.writer_stops_after_terminate_calls = writer_stops_after_terminate_calls
        self.terminate_calls = 0
        self.requests = []
        self.closed = False

    def request(self, payload, **_kwargs):
        self.requests.append(dict(payload))
        operation = payload["operation"]
        if operation == self.fail_operation:
            self.closed = True
            raise evaluation.WorkerTaskFailure(
                f"synthetic {operation} timeout",
                observed_child_cpu_seconds=0.2,
                timed_out=True,
                worker_tree_stopped=True,
                ipc_write_thread_stopped=self.ipc_write_thread_stopped,
            )
        if operation == "restore":
            return (
                {
                    "status": "ok",
                    "operation": "restore",
                    "task_id": payload["task_id"],
                    "endpoint_id": payload["endpoint_id"],
                    "receipt": dict(self.receipts[payload["endpoint_id"]]),
                },
                0.2,
                0.01,
            )
        if operation == "generate":
            endpoint = self.endpoints[payload["endpoint_id"]]
            return (
                {
                    "status": "ok",
                    "operation": "generate",
                    "task_id": payload["task_id"],
                    "endpoint_id": payload["endpoint_id"],
                    "output": _generation_output(),
                    "model_digest_before": f"model:{payload['endpoint_id']}",
                    "model_digest_after": f"model:{payload['endpoint_id']}",
                    "checkpoint_sha256_before": endpoint["sha256"],
                    "checkpoint_sha256_after": endpoint["sha256"],
                },
                0.01,
                0.01,
            )
        if operation == "shutdown":
            self.closed = True
            return {"status": "ok", "operation": "shutdown"}, 0.01, 0.01
        raise AssertionError(operation)

    def shutdown(self, **_kwargs):
        self.closed = True
        return 0.01, 0.01

    def terminate_tree(self):
        self.closed = True
        self.terminate_calls += 1
        if (
            self.writer_stops_after_terminate_calls is not None
            and self.terminate_calls >= self.writer_stops_after_terminate_calls
        ):
            self.ipc_write_thread_stopped = True
        return True


def _execute_kwargs(paths, preflight, worker_factory, durability_api, **extra):
    kwargs = {
        "paths": paths,
        "preflight_fn": lambda: preflight,
        "resources_fn": lambda: {
            "cpu_percent_samples": [1.0, 2.0],
            "available_memory_gib": 16.0,
        },
        "disk_fn": lambda _path: {"free_bytes": 2 * evaluation.MIN_OUTPUT_DISK_BYTES},
        "worker_factory": worker_factory,
        "durability_api": durability_api,
        "durability_probe_fn": lambda _directory, **_kwargs: {
            "directory": str(paths["attempt"].parent.resolve()),
            "volume_root": "C:\\",
            "filesystem": "NTFS",
            "drive_type": 3,
        },
    }
    kwargs.update(extra)
    return kwargs


def test_worker_rejects_direct_and_replayed_authorization_before_corpus_read(tmp_path, monkeypatch):
    attempt_path = tmp_path / "attempt.json"
    ledger_path = tmp_path / "ledger.jsonl"
    token = "a" * 48
    attempt_path.write_text(
        json.dumps(
            {
                "schema": "taiji-r2-dialogue-recovery-v2-paired-evaluation-attempt-v1",
                "status": "started",
                "run": evaluation.RUN_ID,
                "attempt_id": "synthetic-attempt",
            }
        ),
        encoding="utf-8",
    )
    ledger_path.write_text(
        json.dumps(
            {
                "event": "restore_started",
                "task_id": "restore-parent",
                "attempt_id": "synthetic-attempt",
                "operation": "restore",
                "endpoint_id": "parent_baseline",
                "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_calls = []
    monkeypatch.setattr(
        evaluation.base, "_load_assistant_train_corpus", lambda: corpus_calls.append(True)
    )
    with pytest.raises(evaluation.Refuse, match="token does not match"):
        evaluation._worker_handle_task(
            {
                "operation": "restore",
                "endpoint_id": "parent_baseline",
                "task_id": "restore-parent",
                "attempt_id": "synthetic-attempt",
                "token": "unauthorized-token".ljust(48, "x"),
            },
            attempt_path=attempt_path,
            ledger_path=ledger_path,
            durability_api=_DurabilityApi(),
            expected_attempt_path=attempt_path,
            expected_ledger_path=ledger_path,
            job_membership_check=lambda: True,
            trainers={},
            endpoint_specs={},
            corpus_holder={"assistant": None},
        )
    qualification = {
        "directory": str(tmp_path.resolve()),
        "volume_root": "C:\\",
        "filesystem": "NTFS",
        "drive_type": 3,
    }
    authorization_args = {
        "operation": "restore",
        "endpoint_id": "parent_baseline",
        "task_id": "restore-parent",
        "token": token,
        "attempt_path": attempt_path,
        "ledger_path": ledger_path,
        "api": _DurabilityApi(),
        "qualification": qualification,
        "expected_attempt_path": attempt_path,
        "expected_ledger_path": ledger_path,
    }
    evaluation._authorize_worker_task(**authorization_args)
    with pytest.raises(evaluation.Refuse, match="claim already exists"):
        evaluation._authorize_worker_task(**authorization_args)
    assert corpus_calls == []
    assert list(tmp_path.glob("*.claim.json"))


def test_direct_private_worker_without_parent_job_is_refused_before_attempt_read(
    tmp_path, monkeypatch
):
    corpus_calls = []
    monkeypatch.setattr(
        evaluation.base, "_load_assistant_train_corpus", lambda: corpus_calls.append(True)
    )
    with pytest.raises(evaluation.Refuse, match="outside the parent-supervised Job Object"):
        evaluation._worker_handle_task(
            {
                "operation": "restore",
                "endpoint_id": "parent_baseline",
                "task_id": "direct",
                "token": "x" * 48,
            },
            attempt_path=tmp_path / "missing-attempt.json",
            ledger_path=tmp_path / "missing-ledger.jsonl",
            durability_api=_DurabilityApi(),
            trainers={},
            endpoint_specs={},
            corpus_holder={"assistant": None},
            job_membership_check=lambda: False,
        )
    assert corpus_calls == []


def test_worker_command_uses_repo_root_module_entrypoint(tmp_path):
    command = evaluation._worker_command(
        tmp_path / "attempt.json",
        tmp_path / "ledger.jsonl",
        "Local\\TaijiR2Eval_synthetic",
    )
    assert command[1:3] == ["-m", "scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2"]
    assert "--worker" in command
    assert "--job-name" in command


def test_private_worker_module_entrypoint_refuses_missing_parent_markers_before_data_access(
    tmp_path,
):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.training.evaluate_taiji_r2_daily_dialogue_recovery_v2",
            "--worker",
            "--attempt-path",
            str(tmp_path / "missing-attempt.json"),
            "--ledger-path",
            str(tmp_path / "missing-ledger.jsonl"),
            "--job-name",
            "Local\\TaijiR2Eval_missing",
        ],
        cwd=evaluation.ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "worker requires an existing parent attempt and ledger" in result.stderr


def _fake_supervised_worker_for_blocking_ipc(
    tmp_path,
    writer_release,
    write_started,
    write_calls,
    *,
    release_writer_on_terminate=True,
    cancel_synchronous_io_fn=None,
):
    class FakeProcess:
        pid = 234_567
        _handle = 88

        def __init__(self):
            self.returncode = None
            self.stdin = SimpleNamespace(write=self._write, flush=lambda: None)
            self.stdout = io.StringIO("")

        def _write(self, payload):
            write_calls.append(payload)
            write_started.set()
            writer_release.wait(timeout=2.0)
            return len(payload)

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

    class FakeProcessInfo:
        def cpu_times(self):
            return SimpleNamespace(user=0.0, system=0.0)

    class FakeProcessApi:
        @staticmethod
        def Process(_pid):
            return FakeProcessInfo()

        @staticmethod
        def pid_exists(_pid):
            return False

    process = FakeProcess()

    class FakeJob:
        active = 1

        def accounting(self):
            return SimpleNamespace(ActiveProcesses=self.active, TotalProcesses=1)

        def cpu_seconds(self):
            return 0.0

        def terminate_and_confirm(self, *, timeout_seconds):
            if release_writer_on_terminate:
                writer_release.set()
            process.kill()
            self.active = 0
            return True

        def close(self):
            return None

    job = FakeJob()
    worker = evaluation.SupervisedWorker(
        tmp_path / "attempt.json",
        tmp_path / "ledger.jsonl",
        command=[sys.executable, "-c", "pass"],
        cwd=tmp_path,
        popen_factory=lambda *_args, **_kwargs: process,
        process_api=FakeProcessApi(),
        job_factory=lambda *_args, **_kwargs: job,
        resume_fn=lambda _pid: None,
        cancel_synchronous_io_fn=cancel_synchronous_io_fn
        or evaluation._cancel_synchronous_thread_io,
        poll_interval_seconds=0.005,
    )
    return worker, process, job


@pytest.mark.skipif(os.name != "nt", reason="Windows supervised IPC deadline contract")
def test_ipc_pipe_write_is_supervised_and_cancellable_by_job_deadline(tmp_path):
    release, write_started = threading.Event(), threading.Event()
    write_calls = []
    worker, process, job = _fake_supervised_worker_for_blocking_ipc(
        tmp_path, release, write_started, write_calls
    )
    release_timer = threading.Timer(0.5, release.set)
    release_timer.start()
    started = time.monotonic()
    with pytest.raises(evaluation.WorkerTaskFailure) as caught:
        worker.request(
            {"operation": "probe", "task_id": "blocked-write", "endpoint_id": "worker"},
            budget=evaluation.EvaluationBudget(),
            task_cpu_limit_seconds=5.0,
            task_wall_limit_seconds=0.05,
            future_cpu_reserve_seconds=0.0,
            future_wall_reserve_seconds=0.0,
        )
    elapsed = time.monotonic() - started
    release_timer.cancel()
    assert write_started.is_set()
    assert caught.value.timed_out is True
    assert caught.value.worker_tree_stopped is True
    assert job.active == 0
    assert process.poll() is not None
    assert elapsed < 0.25


@pytest.mark.skipif(os.name != "nt", reason="Windows synchronous IPC cancellation contract")
def test_ipc_deadline_cancels_and_joins_writer_after_job_termination(tmp_path):
    release, write_started = threading.Event(), threading.Event()
    write_calls, cancel_calls = [], []

    def cancel_writer(thread_id):
        cancel_calls.append(thread_id)
        release.set()
        return True

    worker, process, job = _fake_supervised_worker_for_blocking_ipc(
        tmp_path,
        release,
        write_started,
        write_calls,
        release_writer_on_terminate=False,
        cancel_synchronous_io_fn=cancel_writer,
    )
    started = time.monotonic()
    with pytest.raises(evaluation.WorkerTaskFailure) as caught:
        worker.request(
            {"operation": "probe", "task_id": "cancel-writer", "endpoint_id": "worker"},
            budget=evaluation.EvaluationBudget(),
            task_cpu_limit_seconds=5.0,
            task_wall_limit_seconds=0.05,
            future_cpu_reserve_seconds=0.0,
            future_wall_reserve_seconds=0.0,
        )
    elapsed = time.monotonic() - started

    assert write_started.is_set()
    assert cancel_calls and cancel_calls[0] > 0
    assert caught.value.timed_out is True
    assert caught.value.worker_tree_stopped is True
    assert caught.value.ipc_write_thread_stopped is True
    assert worker.ipc_write_thread_stopped is True
    assert job.active == 0
    assert process.poll() is not None
    assert elapsed < 0.25


@pytest.mark.skipif(os.name != "nt", reason="Windows synchronous pipe cancellation integration")
def test_cancel_synchronous_thread_io_unblocks_a_native_pipe_read():
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreatePipe.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        ctypes.POINTER(wintypes.HANDLE),
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.CreatePipe.restype = wintypes.BOOL
    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    read_handle, write_handle = wintypes.HANDLE(), wintypes.HANDLE()
    assert kernel32.CreatePipe(ctypes.byref(read_handle), ctypes.byref(write_handle), None, 0)
    started = threading.Event()
    result = []

    def blocked_read():
        buffer = ctypes.create_string_buffer(1)
        bytes_read = wintypes.DWORD()
        started.set()
        succeeded = kernel32.ReadFile(
            read_handle,
            buffer,
            1,
            ctypes.byref(bytes_read),
            None,
        )
        result.append((bool(succeeded), ctypes.get_last_error()))

    reader = threading.Thread(target=blocked_read, name="r2-sync-read-test", daemon=True)
    try:
        reader.start()
        assert started.wait(timeout=1.0)
        deadline = time.monotonic() + 1.0
        while reader.is_alive() and time.monotonic() < deadline:
            try:
                evaluation._cancel_synchronous_thread_io(reader.native_id or 0)
            except OSError as exc:
                if getattr(exc, "winerror", None) not in {87, 1168}:
                    raise
            reader.join(timeout=0.02)
        assert not reader.is_alive()
        assert result and result[0][0] is False
        assert result[0][1] == 995  # ERROR_OPERATION_ABORTED
    finally:
        if reader.is_alive():
            kernel32.CloseHandle(write_handle)
            write_handle = wintypes.HANDLE()
            reader.join(timeout=1.0)
        if read_handle:
            kernel32.CloseHandle(read_handle)
        if write_handle:
            kernel32.CloseHandle(write_handle)


@pytest.mark.skipif(os.name != "nt", reason="Windows bounded IPC frame contract")
def test_oversized_ipc_frame_is_rejected_before_write_and_consumes_worker(tmp_path):
    release, write_started = threading.Event(), threading.Event()
    write_calls = []
    worker, process, job = _fake_supervised_worker_for_blocking_ipc(
        tmp_path, release, write_started, write_calls
    )
    with pytest.raises(evaluation.WorkerTaskFailure, match="IPC frame") as caught:
        worker.request(
            {
                "operation": "probe",
                "task_id": "oversized",
                "endpoint_id": "worker",
                "extra": "x" * 65536,
            },
            budget=evaluation.EvaluationBudget(),
            task_cpu_limit_seconds=5.0,
            task_wall_limit_seconds=1.0,
            future_cpu_reserve_seconds=0.0,
            future_wall_reserve_seconds=0.0,
        )
    assert caught.value.timed_out is True
    assert caught.value.worker_tree_stopped is True
    assert write_calls == []
    assert not write_started.is_set()
    assert job.active == 0
    assert process.poll() is not None


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object process-tree contract")
def test_windows_job_deadline_kills_hanging_worker_and_child_process(tmp_path):
    pid_file = (tmp_path / "child.pid").resolve()
    source = (
        "import ctypes,os,pathlib,subprocess,sys,time; "
        "k=ctypes.WinDLL('kernel32',use_last_error=True); k.GetCurrentProcess.restype=ctypes.c_void_p; "
        "k.OpenJobObjectW.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_wchar_p]; k.OpenJobObjectW.restype=ctypes.c_void_p; "
        "k.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]; k.IsProcessInJob.restype=ctypes.c_int; "
        "sys.stdin.readline(); h=k.OpenJobObjectW(4,0,os.environ['TAIJI_R2_EVALUATION_JOB_NAME']); b=ctypes.c_int(); "
        "assert h and k.IsProcessInJob(k.GetCurrentProcess(),h,ctypes.byref(b)) and b.value; "
        "k.CloseHandle(h); "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid),encoding='utf-8'); "
        "time.sleep(30)"
    )
    worker = evaluation.SupervisedWorker(
        tmp_path / "attempt.json",
        tmp_path / "ledger.jsonl",
        command=[sys.executable, "-c", source],
        cwd=evaluation.ROOT,
    )
    budget = evaluation.EvaluationBudget()
    with pytest.raises(evaluation.WorkerTaskFailure, match="deadline") as caught:
        worker.request(
            {"operation": "probe", "task_id": "hang", "endpoint_id": "worker"},
            budget=budget,
            task_cpu_limit_seconds=10.0,
            task_wall_limit_seconds=0.15,
            future_cpu_reserve_seconds=0.0,
            future_wall_reserve_seconds=0.0,
        )
    assert caught.value.timed_out is True
    assert caught.value.worker_tree_stopped is True
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    assert not evaluation.psutil.pid_exists(child_pid)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object CPU deadline contract")
def test_windows_job_cpu_deadline_kills_busy_worker_and_child_process(tmp_path):
    pid_file = (tmp_path / "busy-child.pid").resolve()
    source = (
        "import ctypes,os,pathlib,subprocess,sys,time; "
        "k=ctypes.WinDLL('kernel32',use_last_error=True); k.GetCurrentProcess.restype=ctypes.c_void_p; "
        "k.OpenJobObjectW.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_wchar_p]; k.OpenJobObjectW.restype=ctypes.c_void_p; "
        "k.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]; k.IsProcessInJob.restype=ctypes.c_int; "
        "sys.stdin.readline(); h=k.OpenJobObjectW(4,0,os.environ['TAIJI_R2_EVALUATION_JOB_NAME']); b=ctypes.c_int(); "
        "assert h and k.IsProcessInJob(k.GetCurrentProcess(),h,ctypes.byref(b)) and b.value; k.CloseHandle(h); "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid),encoding='utf-8'); "
        "start=time.process_time()\nwhile time.process_time()-start < 5: pass"
    )
    worker = evaluation.SupervisedWorker(
        tmp_path / "attempt.json",
        tmp_path / "ledger.jsonl",
        command=[sys.executable, "-c", source],
        cwd=evaluation.ROOT,
    )
    with pytest.raises(evaluation.WorkerTaskFailure, match="CPU/wall deadline") as caught:
        worker.request(
            {"operation": "probe", "task_id": "busy", "endpoint_id": "worker"},
            budget=evaluation.EvaluationBudget(),
            task_cpu_limit_seconds=0.12,
            task_wall_limit_seconds=5.0,
            future_cpu_reserve_seconds=0.0,
            future_wall_reserve_seconds=0.0,
        )
    assert caught.value.timed_out is True
    assert caught.value.worker_tree_stopped is True
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    assert not evaluation.psutil.pid_exists(child_pid)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object assignment failure contract")
def test_job_assignment_failure_kills_handshake_process_before_any_ipc(tmp_path):
    class FakeProcess:
        pid = 123_456
        _handle = 77

        def __init__(self):
            self.kill_called = False
            self.write_calls = 0
            self.stdin = SimpleNamespace(write=self._write, flush=lambda: None)
            self.stdout = None
            self.returncode = None

        def _write(self, _value):
            self.write_calls += 1

        def kill(self):
            self.kill_called = True
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

        def poll(self):
            return self.returncode

    class FakeProcessInfo:
        def cpu_times(self):
            return SimpleNamespace(user=0.01, system=0.01)

    class FakeProcessApi:
        @staticmethod
        def Process(_pid):
            return FakeProcessInfo()

    process = FakeProcess()
    creation_flags = []
    resume_calls = []

    def popen_factory(*_args, **kwargs):
        creation_flags.append(kwargs.get("creationflags", 0))
        return process

    with pytest.raises(evaluation.Refuse, match="cannot receive authorization"):
        evaluation.SupervisedWorker(
            tmp_path / "attempt.json",
            tmp_path / "ledger.jsonl",
            command=[sys.executable, "-c", "pass"],
            cwd=tmp_path,
            popen_factory=popen_factory,
            process_api=FakeProcessApi(),
            job_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("synthetic AssignProcessToJobObject failure")
            ),
            resume_fn=lambda _pid: resume_calls.append(True),
        )
    assert process.kill_called is True
    assert process.write_calls == 0
    assert creation_flags == [evaluation.CREATE_SUSPENDED]
    assert resume_calls == []


@pytest.mark.skipif(os.name != "nt", reason="Windows suspended Job Object bootstrap contract")
def test_worker_resumes_only_after_job_assignment_and_verification(tmp_path):
    actions = []

    class FakeProcess:
        pid = 123_458
        _handle = 79
        returncode = None
        stdin = SimpleNamespace(write=lambda _value: None, flush=lambda: None)
        stdout = io.StringIO("")

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    process = FakeProcess()

    class FakeProcessInfo:
        def cpu_times(self):
            return SimpleNamespace(user=0.0, system=0.0)

    class FakeProcessApi:
        @staticmethod
        def Process(_pid):
            return FakeProcessInfo()

        @staticmethod
        def pid_exists(_pid):
            return False

    class FakeJob:
        active = 1

        def accounting(self):
            return SimpleNamespace(ActiveProcesses=self.active, TotalProcesses=1)

        def cpu_seconds(self):
            return 0.0

        def terminate_and_confirm(self, *, timeout_seconds):
            self.active = 0
            process.kill()
            return True

        def close(self):
            return None

    job = FakeJob()

    def popen_factory(_command, **kwargs):
        assert kwargs.get("creationflags") == evaluation.CREATE_SUSPENDED
        actions.append("created_suspended")
        return process

    def job_factory(_handle, *, name):
        assert name.startswith("Local\\TaijiR2Eval_")
        actions.append("job_assigned")
        return job

    def resume_fn(pid):
        assert pid == process.pid
        actions.append("resumed")

    worker = evaluation.SupervisedWorker(
        tmp_path / "attempt.json",
        tmp_path / "ledger.jsonl",
        command=[sys.executable, "-c", "pass"],
        cwd=tmp_path,
        popen_factory=popen_factory,
        process_api=FakeProcessApi(),
        job_factory=job_factory,
        resume_fn=resume_fn,
    )
    assert actions == ["created_suspended", "job_assigned", "resumed"]
    assert worker.initial_child_cpu_seconds == 0.0


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object terminate-failure contract")
def test_unconfirmed_job_termination_fails_closed_without_psutil_fallback(tmp_path):
    class FakeProcess:
        pid = 123_457
        _handle = 78
        stdout = None
        stdin = SimpleNamespace(write=lambda _value: None, flush=lambda: None)
        kill_called = False

        def poll(self):
            return None

        def wait(self, timeout=None):
            return None

        def kill(self):
            self.kill_called = True

    class FakeProcessInfo:
        def cpu_times(self):
            return SimpleNamespace(user=0.0, system=0.0)

    class FakeProcessApi:
        @staticmethod
        def Process(_pid):
            return FakeProcessInfo()

    class UnconfirmedJob:
        closed = False

        def accounting(self):
            return SimpleNamespace(ActiveProcesses=1, TotalProcesses=1)

        def cpu_seconds(self):
            return 0.0

        def terminate_and_confirm(self, *, timeout_seconds):
            raise OSError("synthetic TerminateJobObject failure")

        def close(self):
            self.closed = True

    process = FakeProcess()
    job = UnconfirmedJob()
    worker = evaluation.SupervisedWorker(
        tmp_path / "attempt.json",
        tmp_path / "ledger.jsonl",
        command=[sys.executable, "-c", "pass"],
        cwd=tmp_path,
        popen_factory=lambda *_args, **_kwargs: process,
        process_api=FakeProcessApi(),
        job_factory=lambda *_args, **_kwargs: job,
        resume_fn=lambda _pid: None,
    )
    assert worker.terminate_tree() is False
    assert job.closed is True
    assert process.kill_called is False
    assert process.poll() is None


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object kill-on-close contract")
def test_windows_job_close_kills_all_attached_worker_descendants(tmp_path):
    pid_file = (tmp_path / "child.pid").resolve()
    source = (
        "import ctypes,json,os,pathlib,subprocess,sys,time; "
        "k=ctypes.WinDLL('kernel32',use_last_error=True); k.GetCurrentProcess.restype=ctypes.c_void_p; "
        "k.OpenJobObjectW.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_wchar_p]; k.OpenJobObjectW.restype=ctypes.c_void_p; "
        "k.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]; k.IsProcessInJob.restype=ctypes.c_int; "
        "sys.stdin.readline(); h=k.OpenJobObjectW(4,0,os.environ['TAIJI_R2_EVALUATION_JOB_NAME']); b=ctypes.c_int(); "
        "assert h and k.IsProcessInJob(k.GetCurrentProcess(),h,ctypes.byref(b)) and b.value; "
        "k.CloseHandle(h); "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid),encoding='utf-8'); "
        "print(json.dumps({'status':'ok','task_id':'close','endpoint_id':'worker'}),flush=True); "
        "time.sleep(30)"
    )
    worker = evaluation.SupervisedWorker(
        tmp_path / "attempt.json",
        tmp_path / "ledger.jsonl",
        command=[sys.executable, "-c", source],
        cwd=evaluation.ROOT,
    )
    worker.request(
        {"operation": "probe", "task_id": "close", "endpoint_id": "worker"},
        budget=evaluation.EvaluationBudget(),
        task_cpu_limit_seconds=5.0,
        task_wall_limit_seconds=5.0,
        future_cpu_reserve_seconds=0.0,
        future_wall_reserve_seconds=0.0,
    )
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    worker.job.close()
    worker.process.wait(timeout=5.0)
    assert worker.process.poll() is not None
    assert not evaluation.psutil.pid_exists(child_pid)


@pytest.mark.parametrize(
    "filesystem,drive_type,is_unc",
    [
        ("ReFS", 3, False),
        ("NTFS", 2, False),
        ("FAT32", 3, False),
        ("NTFS", 4, False),
        ("NTFS", 3, True),
    ],
)
def test_ntfs_qualification_rejects_unsupported_path_before_marker(
    tmp_path, filesystem, drive_type, is_unc
):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(evaluation.Refuse, match="DURABILITY_UNSUPPORTED"):
        evaluation.probe_ntfs_durability(
            tmp_path,
            api=_DurabilityApi(filesystem=filesystem, drive_type=drive_type, is_unc=is_unc),
            platform_name="nt",
        )


def test_durable_publish_uses_write_through_and_refuses_flush_move_failures(tmp_path):
    api = _DurabilityApi()
    qualification = evaluation.probe_ntfs_durability(tmp_path, api=api, platform_name="nt")
    target = tmp_path / "marker.json"
    digest = evaluation._durable_write_json(
        target, {"id": "once"}, qualification=qualification, api=api
    )
    assert evaluation.sha256_file(target) == digest
    assert api.move_calls[0][2] == evaluation.MOVEFILE_WRITE_THROUGH
    with pytest.raises(FileExistsError):
        evaluation._durable_write_json(
            target, {"id": "overwrite"}, qualification=qualification, api=api
        )
    failing_flush = _DurabilityApi(fail_flush=True)
    with pytest.raises(OSError, match="FlushFileBuffers"):
        evaluation._durable_write_json(
            tmp_path / "flush-fail.json",
            {"id": "flush"},
            qualification=qualification,
            api=failing_flush,
        )
    failing_move = _DurabilityApi(fail_move_at=1)
    with pytest.raises(OSError, match="MoveFileExW"):
        evaluation._durable_write_json(
            tmp_path / "move-fail.json",
            {"id": "move"},
            qualification=qualification,
            api=failing_move,
        )
    assert not (tmp_path / "flush-fail.json").exists()
    assert not (tmp_path / "move-fail.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows local NTFS write-through contract")
def test_real_windows_ntfs_write_through_publish_is_write_once(tmp_path):
    api = evaluation.WindowsDurabilityApi()
    qualification = evaluation.probe_ntfs_durability(
        tmp_path,
        api=api,
        platform_name="nt",
    )
    target = tmp_path / "real-ntfs-marker.json"
    expected_sha = evaluation._durable_write_json(
        target,
        {"synthetic": True, "attempt": "one-shot"},
        qualification=qualification,
        api=api,
    )
    assert evaluation.sha256_file(target) == expected_sha
    with pytest.raises(FileExistsError):
        evaluation._durable_write_json(
            target,
            {"synthetic": True, "attempt": "overwrite"},
            qualification=qualification,
            api=api,
        )


def test_second_initial_marker_failure_consumes_attempt_without_worker(tmp_path):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / "reports")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    api = _DurabilityApi(fail_move_at=2)
    workers = []
    with pytest.raises(evaluation.Refuse, match="initial ledger publication"):
        evaluation.execute(
            **_execute_kwargs(
                paths,
                preflight,
                lambda **_kwargs: workers.append(True),
                api,
            )
        )
    assert paths["attempt"].is_file()
    assert not paths["ledger"].exists()
    assert workers == []


def test_attempt_flush_failure_refuses_before_marker_or_worker(tmp_path):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / "reports-flush")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    workers = []
    with pytest.raises(OSError, match="FlushFileBuffers"):
        evaluation.execute(
            **_execute_kwargs(
                paths,
                preflight,
                lambda **_kwargs: workers.append(True),
                _DurabilityApi(fail_flush=True),
            )
        )
    assert not paths["attempt"].exists()
    assert not paths["ledger"].exists()
    assert workers == []


def test_source_drift_after_four_restores_is_rejected_before_selector(tmp_path):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / "reports")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    receipts = _restore_receipts(preflight["endpoints"])
    worker = _FakeSupervisedWorker(receipts, preflight["endpoints"])
    calls = []
    preflight_calls = 0

    def changing_preflight():
        nonlocal preflight_calls
        preflight_calls += 1
        if preflight_calls <= 1:
            return preflight
        changed = dict(preflight)
        changed["source_hashes"] = {"synthetic-source": "0" * 64}
        return changed

    with pytest.raises(evaluation.Refuse, match="changed before prompt selection"):
        evaluation.execute(
            **_execute_kwargs(
                paths,
                preflight,
                lambda **_kwargs: worker,
                _DurabilityApi(),
                preflight_fn=changing_preflight,
                prompt_selector=lambda: calls.append(True) or _prompts(),
            )
        )
    assert [task["operation"] for task in worker.requests] == ["restore"] * 4
    assert calls == []
    events = [json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()]
    assert not any(event["event"] == "prompt_selection_started" for event in events)


@pytest.mark.parametrize(
    "fail_operation,expected_event", [("restore", "restore_timeout"), ("generate", "query_timeout")]
)
def test_supervised_worker_timeout_is_one_shot_and_stops_tree(
    tmp_path, fail_operation, expected_event
):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / f"reports-{fail_operation}")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    receipts = _restore_receipts(preflight["endpoints"])
    worker = _FakeSupervisedWorker(
        receipts,
        preflight["endpoints"],
        fail_operation=fail_operation,
    )
    selected = []
    with pytest.raises(evaluation.Refuse, match="one-shot evaluation attempt is consumed"):
        evaluation.execute(
            **_execute_kwargs(
                paths,
                preflight,
                lambda **_kwargs: worker,
                _DurabilityApi(),
                prompt_selector=lambda: selected.append(True) or _prompts(),
            )
        )
    assert worker.closed is True
    assert selected == ([] if fail_operation == "restore" else [True])
    events = [json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()]
    assert sum(event["event"] == expected_event for event in events) == 1
    assert sum(event["event"] == "restore_started" for event in events) == (
        1 if fail_operation == "restore" else 4
    )


def test_unstopped_ipc_writer_state_is_preserved_in_failure_tail(tmp_path):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / "reports-unjoined-ipc-writer")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    receipts = _restore_receipts(preflight["endpoints"])
    worker = _FakeSupervisedWorker(
        receipts,
        preflight["endpoints"],
        fail_operation="generate",
        ipc_write_thread_stopped=False,
    )

    with pytest.raises(evaluation.Refuse, match="one-shot evaluation attempt is consumed"):
        evaluation.execute(
            **_execute_kwargs(
                paths,
                preflight,
                lambda **_kwargs: worker,
                _DurabilityApi(),
                prompt_selector=_prompts,
            )
        )

    events = [json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()]
    stopped = next(event for event in events if event["event"] == "evaluation_stopped")
    assert stopped["worker_tree_exit_unverified"] is False
    assert stopped["ipc_writer_exit_unverified"] is True
    assert stopped["unknown_tail"] is True
    assert stopped["task_id"] is not None
    assert stopped["operation"] == "generate"
    assert not any(event["event"] == "query_timeout" for event in events)


def test_writer_stopping_on_outer_cleanup_keeps_unwritten_query_terminal_unknown(tmp_path):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / "reports-late-writer-stop")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    receipts = _restore_receipts(preflight["endpoints"])
    worker = _FakeSupervisedWorker(
        receipts,
        preflight["endpoints"],
        fail_operation="generate",
        ipc_write_thread_stopped=False,
        writer_stops_after_terminate_calls=2,
    )

    with pytest.raises(evaluation.Refuse, match="one-shot evaluation attempt is consumed"):
        evaluation.execute(
            **_execute_kwargs(
                paths,
                preflight,
                lambda **_kwargs: worker,
                _DurabilityApi(),
                prompt_selector=_prompts,
            )
        )

    events = [json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()]
    stopped = next(event for event in events if event["event"] == "evaluation_stopped")
    assert stopped["task_id"] is not None
    assert stopped["operation"] == "generate"
    assert stopped["ipc_writer_exit_unverified"] is False
    assert stopped["task_terminal_missing"] is True
    assert stopped["unknown_tail"] is True
    assert not any(event["event"] == "query_timeout" for event in events)


def test_durable_restore_terminal_is_not_duplicated_when_progress_append_fails(
    tmp_path, monkeypatch
):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / "reports-terminal-before-progress-failure")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    receipts = _restore_receipts(preflight["endpoints"])
    worker = _FakeSupervisedWorker(receipts, preflight["endpoints"])
    selector_calls = []
    original_append = evaluation._append_jsonl

    def fail_after_restore_terminal(path, payload, *, flush_file=None):
        if (
            Path(path).resolve() == paths["progress"].resolve()
            and payload.get("event") == "restore_completed"
        ):
            raise OSError("synthetic progress append failure after durable restore terminal")
        return original_append(path, payload, flush_file=flush_file)

    monkeypatch.setattr(evaluation, "_append_jsonl", fail_after_restore_terminal)
    with pytest.raises(evaluation.Refuse, match="one-shot evaluation attempt is consumed"):
        evaluation.execute(
            **_execute_kwargs(
                paths,
                preflight,
                lambda **_kwargs: worker,
                _DurabilityApi(),
                prompt_selector=lambda: selector_calls.append(True) or _prompts(),
            )
        )

    events = [json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()]
    task_id = next(event["task_id"] for event in events if event["event"] == "restore_started")
    restore_terminals = [
        event
        for event in events
        if event.get("operation") == "restore"
        and event.get("task_id") == task_id
        and event.get("event") in {"restore_completed", "restore_timeout", "restore_failed"}
    ]
    assert len(restore_terminals) == 1
    assert restore_terminals[0]["event"] == "restore_completed"
    assert selector_calls == []
    stopped = next(event for event in events if event["event"] == "evaluation_stopped")
    assert stopped["task_terminal_missing"] is False
    assert stopped["unknown_tail"] is False
    assert stopped["operation"] == "restore"


@pytest.mark.parametrize(
    "operation,intent_event,expected_query_attempts",
    [("restore", "restore_started", 0), ("generate", "query_started", 1)],
)
def test_durable_intent_progress_failure_preserves_open_task_unknown(
    tmp_path, monkeypatch, operation, intent_event, expected_query_attempts
):
    _, _, preflight = _valid_records(tmp_path)
    paths = _output_paths(tmp_path / f"reports-{operation}-intent-progress-failure")
    paths["readiness"].parent.mkdir(parents=True)
    _write_readiness(paths, preflight)
    receipts = _restore_receipts(preflight["endpoints"])
    worker = _FakeSupervisedWorker(receipts, preflight["endpoints"])
    selector_calls = []
    original_append = evaluation._append_jsonl

    def fail_after_intent_ledger(path, payload, *, flush_file=None):
        if (
            Path(path).resolve() == paths["progress"].resolve()
            and payload.get("event") == intent_event
        ):
            raise OSError("synthetic progress append failure after durable task intent")
        return original_append(path, payload, flush_file=flush_file)

    monkeypatch.setattr(evaluation, "_append_jsonl", fail_after_intent_ledger)
    with pytest.raises(evaluation.Refuse, match="one-shot evaluation attempt is consumed"):
        evaluation.execute(
            **_execute_kwargs(
                paths,
                preflight,
                lambda **_kwargs: worker,
                _DurabilityApi(),
                prompt_selector=lambda: selector_calls.append(True) or _prompts(),
            )
        )

    events = [json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8").splitlines()]
    intent = next(event for event in events if event["event"] == intent_event)
    stopped = next(event for event in events if event["event"] == "evaluation_stopped")
    assert stopped["task_id"] == intent["task_id"]
    assert stopped["operation"] == operation
    assert stopped["task_terminal_missing"] is True
    assert stopped["task_intent_unverified"] is False
    assert stopped["unknown_tail"] is True
    assert stopped["query_attempts"] == expected_query_attempts
    assert not any(
        event.get("task_id") == intent["task_id"]
        and event["event"]
        in {"restore_completed", "restore_timeout", "restore_failed", "query_output_persisted"}
        for event in events
    )
    assert selector_calls == ([] if operation == "restore" else [True])
    assert not any(request["operation"] == "generate" for request in worker.requests)
