"""Synthetic four-restore/156-query integration using the real token and face guards."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.training import (
    evaluate_taiji_r2_daily_dialogue_continuation_extension_worker_v1 as worker,
)
from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as frozen


class Durability:
    def inspect(self, _path):
        return {"is_unc": False, "drive_type": 3, "filesystem": "NTFS", "volume_root": "C:\\"}

    def flush_file(self, fd):
        os.fsync(fd)

    def move_file_write_through(self, source, destination, flags):
        assert flags == frozen.MOVEFILE_WRITE_THROUGH
        os.rename(source, destination)


def test_all_four_restores_and_156_queries_use_the_declared_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        name: tmp_path / f"recovery_{name}.json"
        for name in ("attempt", "ledger", "readiness", "face")
    }
    rows = []
    for endpoint in frozen.ENDPOINT_ORDER:
        checkpoint = tmp_path / f"{endpoint}.pt"
        checkpoint.write_bytes(b"synthetic checkpoint; never deserialized")
        rows.append(
            {
                "endpoint_id": endpoint,
                "path": str(checkpoint),
                "sha256": frozen.sha256_file(checkpoint),
                "checkpoint_digest": "d" * 64,
                "source": "synthetic",
                "endpoint_kind": (
                    "parent_baseline" if endpoint == "parent_baseline" else "continuation_extension"
                ),
                "lineage_snapshot_sha256": "a" * 64,
            }
        )
    preflight = {
        "endpoints": rows,
        "report_hashes": {},
        "source_hashes": {},
        "baseline": {"sha256": "b" * 64},
    }
    readiness = frozen._readiness_payload(preflight)
    attempt = {
        "schema": "taiji-r2-dialogue-recovery-v2-paired-evaluation-attempt-v1",
        "run": frozen.RUN_ID,
        "status": "started",
        "attempt_id": "synthetic-namespace-attempt",
        "readiness": readiness,
        "source_hashes": {},
        "input_report_hashes": {},
    }
    paths["attempt"].write_text(json.dumps(attempt), encoding="utf-8")
    paths["readiness"].write_text(json.dumps(readiness), encoding="utf-8")
    paths["ledger"].write_text("", encoding="utf-8")

    def append(event):
        with paths["ledger"].open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")

    def load(endpoint, _corpus):
        parent = endpoint["endpoint_id"] == "parent_baseline"
        model = SimpleNamespace(
            checkpoint=lambda: {"id": endpoint["endpoint_id"]}, parameter_count=lambda: 1
        )
        return SimpleNamespace(
            model=model, episode_count=0 if parent else 525, global_step=0 if parent else 113209
        ), frozen.content_digest(model.checkpoint())

    monkeypatch.setattr(frozen, "_load_endpoint_trainer", load)
    monkeypatch.setattr(
        frozen.base, "_load_assistant_train_corpus", lambda: SimpleNamespace(digest="c" * 64)
    )

    def forbid_checkpoint(*_args, **_kwargs):
        pytest.fail("namespace integration must not deserialize any model")

    monkeypatch.setattr(worker.torch, "load", forbid_checkpoint)
    output = {
        "generated_text": "好",
        "raw_generated_sha256": hashlib.sha256("好".encode()).hexdigest(),
        "generated_bytes_hex": "好".encode().hex(),
        "generated_bytes_length": 3,
        "utf8_valid": True,
        "no_replacement": True,
        "response_boundary_present": True,
        "stop_reason": "end_marker",
        "pii_pattern_redacted": False,
        "repeated_4gram_rate": 0.0,
    }
    monkeypatch.setattr(frozen.base, "_generate_readonly", lambda *_: dict(output))
    trainers, endpoint_specs, corpus = {}, {}, {"assistant": None}

    def invoke(operation, endpoint, task_id, extra):
        token = hashlib.sha256(task_id.encode()).hexdigest()
        intent = {
            "event": "restore_started" if operation == "restore" else "query_started",
            "attempt_id": attempt["attempt_id"],
            "task_id": task_id,
            "operation": operation,
            "endpoint_id": endpoint,
            "worker_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            **extra,
        }
        append(intent)
        request = {
            "operation": operation,
            "endpoint_id": endpoint,
            "task_id": task_id,
            "attempt_id": attempt["attempt_id"],
            "token": token,
            **extra,
        }
        result = worker._handle_task(
            request,
            attempt_path=paths["attempt"],
            ledger_path=paths["ledger"],
            durability_api=Durability(),
            trainers=trainers,
            endpoint_specs=endpoint_specs,
            corpus_holder=corpus,
            job_membership_check=lambda: True,
            preflight_fn=lambda: preflight,
            evaluation_paths_fn=lambda: paths,
        )
        assert result["status"] == "ok"
        with pytest.raises(frozen.Refuse, match="claim already exists"):
            worker._handle_task(
                request,
                attempt_path=paths["attempt"],
                ledger_path=paths["ledger"],
                durability_api=Durability(),
                trainers=trainers,
                endpoint_specs=endpoint_specs,
                corpus_holder=corpus,
                job_membership_check=lambda: True,
                preflight_fn=lambda: preflight,
                evaluation_paths_fn=lambda: paths,
            )
        append(
            {
                "event": (
                    "restore_completed" if operation == "restore" else "query_output_persisted"
                ),
                "task_id": task_id,
            }
        )
        return result

    for index, row in enumerate(rows):
        assert (
            invoke(
                "restore",
                row["endpoint_id"],
                f"restore-{index}",
                {"checkpoint_sha256": row["sha256"]},
            )["receipt"]["complete"]
            is True
        )
    assert tuple(trainers) == frozen.ENDPOINT_ORDER
    face = frozen.freeze_prompt_face(
        [
            SimpleNamespace(
                episode_id=f"e-{i}",
                family_id=f"f-{i}",
                system="合成",
                context="",
                history=(),
                user_input=f"合成输入{i}",
                unknown_policy="synthetic",
            )
            for i in range(39)
        ]
    )
    paths["face"].write_text(json.dumps(face, ensure_ascii=False), encoding="utf-8")
    append(
        {
            "event": "prompt_face_persisted",
            "attempt_id": attempt["attempt_id"],
            "face_sha256": face["face_sha256"],
        }
    )
    generated = 0
    for index in range(1, 40):
        for endpoint_index, endpoint in enumerate(frozen.ENDPOINT_ORDER):
            result = invoke(
                "generate",
                endpoint,
                f"query-{index}-{endpoint_index}",
                {
                    "index": index,
                    "face_path": str(paths["face"]),
                    "face_sha256": face["face_sha256"],
                    "prompt_sha256": face["prompt_hashes"][index - 1],
                },
            )
            assert result["output"]["generated_text"] == "好"
            assert result["checkpoint_sha256_before"] == result["checkpoint_sha256_after"]
            generated += 1
    assert generated == 156
    assert len(list(tmp_path.glob("*.claim.json"))) == 160
    with pytest.raises(frozen.Refuse, match="fixed persisted prompt face"):
        invoke(
            "generate",
            "parent_baseline",
            "query-foreign-face",
            {
                "index": 1,
                "face_path": str(tmp_path / "foreign.json"),
                "face_sha256": face["face_sha256"],
                "prompt_sha256": face["prompt_hashes"][0],
            },
        )
    with pytest.raises(frozen.Refuse, match="outside the fixed evaluation contract"):
        worker._handle_task(
            {
                "operation": "restore",
                "endpoint_id": "parent_baseline",
                "task_id": "foreign-attempt",
                "attempt_id": attempt["attempt_id"],
                "token": "a" * 48,
            },
            attempt_path=tmp_path / "foreign-attempt.json",
            ledger_path=paths["ledger"],
            durability_api=Durability(),
            trainers={},
            endpoint_specs={},
            corpus_holder={},
            job_membership_check=lambda: True,
            preflight_fn=lambda: preflight,
            evaluation_paths_fn=lambda: paths,
        )
