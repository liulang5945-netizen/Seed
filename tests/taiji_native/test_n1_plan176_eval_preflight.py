"""The one-use pilot evaluation validates all dependencies before consumption."""

import copy
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.training import evaluate_taiji_n1_plan176_foundation_pilot as evaluation
from scripts.training.build_taiji_n1_plan176_foundation_corpus import (
    as_language_episode,
    build_rows,
)


@pytest.fixture
def synthetic_preflight(tmp_path, monkeypatch):
    """No actual checkpoint, sealed file or N175 source is opened by these tests."""
    root = tmp_path
    paths = {
        "ROOT": root,
        "CORPUS": root / "reports/corpus.json",
        "SELECTION": root / "reports/selection.json",
        "N175_SELECTION": root / "reports/n175-selection.json",
        "SPEC": root / "plans/reference/spec.md",
        "CHECKPOINTS": root / "data/checkpoints/pilot",
        "PARENT": root / "data/checkpoints/parent-v3.pt",
        "V4_PARENT": root / "data/checkpoints/parent-v4.pt",
        "PARENT_REPORT": root / "reports/parent-v3.json",
        "BASE": root / "reports/evaluation",
        "ATTEMPT": root / "reports/evaluation.attempt.json",
        "COSTS": root / "reports/evaluation.costs.jsonl",
    }
    for name, path in paths.items():
        monkeypatch.setattr(evaluation, name, path)
    monkeypatch.setattr(evaluation, "report_path", lambda arm: root / f"reports/{arm}.json")

    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def source(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(path.relative_to(root)), encoding="utf-8")

    source_names = (
        "scripts/training/run_taiji_n1_plan176_foundation_pilot.py",
        "scripts/training/calibrate_taiji_n1_plan176_foundation.py",
        "scripts/training/audit_taiji_n1_plan176_learning_paths.py",
        "scripts/training/build_taiji_n1_plan176_foundation_corpus.py",
        "scripts/training/calibrate_taiji_n1_plan175_v3_train_only.py",
        "scripts/data_prep/audit_taiji_r2_daily_dialogue_material_inventory.py",
        "taiji/conditional_language_v2.py",
        "taiji/conditional_language_v3.py",
        "taiji/conditional_language_v4.py",
    )
    for name in source_names:
        source(root / name)
    source(paths["SPEC"])
    rows = sorted(
        (
            row
            for row in build_rows()
            if row["split"] == "train" and row["curriculum_level"] in {"F0", "F1"}
        ),
        key=lambda row: row["episode_id"],
    )
    sealed_rows = [
        row
        for row in build_rows()
        if row["split"] == "sealed" and row["curriculum_level"] in {"F0", "F1"}
    ]
    train = root / "data/conversation/synthetic/train.jsonl"
    train.parent.mkdir(parents=True)
    train.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    train_sha = evaluation.digest(train)
    old_ids = [f"old-{index}" for index in range(16)]
    selection = {
        "registered_before_training": True,
        "source_train_sha256": train_sha,
        "source_sealed_sha256": "sealed-hash-only",
        "train_episode_ids": [row["episode_id"] for row in rows],
        "sealed_F0_F1_episode_ids": [row["episode_id"] for row in sealed_rows],
        "F0_retention_episode_ids": [
            row["episode_id"] for row in rows if row["curriculum_level"] == "F0"
        ][:16],
        "N175_train_retention_episode_ids": old_ids,
    }
    write(paths["SELECTION"], selection)
    write(paths["N175_SELECTION"], {"episode_ids": old_ids})
    write(
        paths["CORPUS"],
        {
            "output_directory": "data/conversation/synthetic",
            "split_sha256": {"train": train_sha, "sealed": "sealed-hash-only"},
        },
    )
    payloads = {}
    parents = {}

    def sign(payload):
        payload["checkpoint_digest"] = evaluation.content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        return payload

    for version, name in ((3, "PARENT"), (4, "V4_PARENT")):
        source(paths[name])
        payload = sign(
            {
                "version": version,
                "steps": 16520,
                "target_byte_exposures": 1000,
                "weights": [version, 1, 2],
                "cursor": {"parent": True},
            }
        )
        payloads[paths[name]] = payload
        parents[version] = payload
    write(paths["PARENT_REPORT"], {"final_checkpoint_sha256": evaluation.digest(paths["PARENT"])})
    write(
        root / "reports/taiji_n1_plan176_v4_parent_migration_20261026.json",
        {"checkpoint_sha256": evaluation.digest(paths["V4_PARENT"])},
    )
    reports = {}
    for arm in evaluation.ARMS:
        version = 4 if arm == "curriculum_v4" else 3
        batches = evaluation.schedule(rows, arm)
        order_sha = hashlib.sha256(
            json.dumps(
                [[rows[index]["episode_id"] for index in batch] for batch in batches]
            ).encode()
        ).hexdigest()
        durable = []
        for count in (0, 640, 2560):
            path = (
                paths["CHECKPOINTS"] / arm / ("zero.pt" if count == 0 else f"batch_{count:05d}.pt")
            )
            source(path)
            payload = copy.deepcopy(parents[version])
            payload["steps"] += count
            payload["target_byte_exposures"] += sum(
                len(as_language_episode(rows[index]).target_bytes)
                for batch in batches[:count]
                for index in batch
            )
            payload["cursor"] = {"next_batch": count, "order_sha256": order_sha}
            if count:
                payload["cursor"].update({"arm": arm, "train_corpus_sha256": train_sha})
                durable.append(
                    {
                        "next_batch": count,
                        "path": str(path.relative_to(root)),
                        "sha256": evaluation.digest(path),
                    }
                )
            payloads[path] = sign(payload)
        source_paths = {
            "spec": paths["SPEC"],
            "selection": paths["SELECTION"],
            "runner": root / source_names[0],
            "candidate": root / f"taiji/conditional_language_v{version}.py",
            "restore_helper": root / source_names[1],
        }
        parent = paths["V4_PARENT" if version == 4 else "PARENT"]
        report = {
            "status": "complete",
            "arm": arm,
            "version": version,
            "planned_updates": 2560,
            "completed_updates": 2560,
            "planned_target_byte_exposures": 190080,
            "target_byte_exposures": 190080,
            "sealed_text_read": 0,
            "n175_key_opened": False,
            "train_corpus_sha256": train_sha,
            "order_sha256": order_sha,
            "parent_checkpoint": str(parent.relative_to(root)),
            "parent_checkpoint_sha256": evaluation.digest(parent),
            "initial_steps": 16520,
            "initial_target_byte_exposures": 1000,
            "zero_fresh_restore": {"status": "passed", "steps": 16520},
            "final_fresh_restore": {"status": "passed", "steps": 19080},
            "sources_sha256": {
                name: evaluation.digest(path) for name, path in source_paths.items()
            },
            "durable_checkpoints": durable,
            "final_checkpoint": durable[-1]["path"],
            "final_checkpoint_sha256": durable[-1]["sha256"],
        }
        reports[arm] = report
        write(evaluation.report_path(arm), report)

    class FakeTrainer:
        @classmethod
        def from_checkpoint(cls, payload):
            assert payload["checkpoint_digest"] == sign(copy.deepcopy(payload))["checkpoint_digest"]
            return (
                SimpleNamespace(
                    steps=payload["steps"],
                    target_byte_exposures=payload["target_byte_exposures"],
                    model=object(),
                ),
                None,
                payload["cursor"],
            )

    monkeypatch.setattr(evaluation, "ConditionalLanguageTrainerV3", FakeTrainer)
    monkeypatch.setattr(evaluation, "ConditionalLanguageTrainerV4", FakeTrainer)
    monkeypatch.setattr(
        evaluation.torch, "load", lambda path, **kwargs: copy.deepcopy(payloads[Path(path)])
    )
    old_pairs = [
        (
            {},
            SimpleNamespace(
                episode_id=identity, target_bytes=b"old target", prompt_bytes=b"old prompt"
            ),
        )
        for identity in old_ids
    ]
    monkeypatch.setattr(
        evaluation, "load_train_episodes", lambda: (old_pairs, {"heldout_text_nodes_parsed": 0})
    )
    return SimpleNamespace(
        root=root,
        rows=rows,
        sealed_rows=sealed_rows,
        reports=reports,
        payloads=payloads,
        old_pairs=old_pairs,
        write=write,
        sign=sign,
    )


def budget():
    return evaluation.EvaluationBudget(time.process_time(), time.monotonic())


def test_preflight_loads_all_dependencies_without_a_sealed_file(synthetic_preflight):
    prepared = evaluation.preflight(budget())
    assert set(prepared["earlier"]) == set(evaluation.ARMS)
    assert set(prepared["trainers"]) == set(evaluation.ARMS)
    assert len(prepared["old_by_id"]) == 16
    assert all(
        set(arm) == {"0", "640", "2560"}
        for arm in prepared["provenance"]["checkpoint_sha256"].values()
    )
    assert not (synthetic_preflight.root / "data/conversation/synthetic/sealed.jsonl").exists()
    assert not evaluation.BASE.exists() and not evaluation.ATTEMPT.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("arm", "curriculum_v3"),
        ("version", 4),
        ("order_sha256", "wrong-order"),
        ("train_corpus_sha256", "wrong-train"),
        ("initial_steps", 0),
    ],
)
def test_preflight_rejects_wrong_run_identity_source_or_order(synthetic_preflight, field, value):
    report = synthetic_preflight.reports["flat_v3"]
    report[field] = value
    synthetic_preflight.write(evaluation.report_path("flat_v3"), report)
    with pytest.raises(ValueError, match="source gate failed"):
        evaluation.preflight(budget())
    assert not evaluation.ATTEMPT.exists()


@pytest.mark.parametrize("source", ["runner", "restore_helper", "spec", "selection", "candidate"])
def test_preflight_rejects_changed_frozen_source(synthetic_preflight, source):
    report = synthetic_preflight.reports["flat_v3"]
    report["sources_sha256"][source] = "changed"
    synthetic_preflight.write(evaluation.report_path("flat_v3"), report)
    with pytest.raises(ValueError, match="source gate failed"):
        evaluation.preflight(budget())


@pytest.mark.parametrize("fault", ["sha", "cursor", "steps", "exposures", "unreadable"])
def test_preflight_checks_the_640_dependency_before_consumption(synthetic_preflight, fault):
    path = evaluation.CHECKPOINTS / "flat_v3/batch_00640.pt"
    payload = synthetic_preflight.payloads[path]
    if fault == "sha":
        path.write_bytes(b"changed checkpoint")
    elif fault == "cursor":
        payload["cursor"]["next_batch"] = 639
    elif fault == "steps":
        payload["steps"] += 1
    elif fault == "exposures":
        payload["target_byte_exposures"] += 1
    else:
        synthetic_preflight.payloads.pop(path)
    synthetic_preflight.sign(payload)
    with pytest.raises((ValueError, KeyError)):
        evaluation.preflight(budget())
    assert not evaluation.BASE.exists() and not evaluation.ATTEMPT.exists()


def test_preflight_rejects_zero_that_does_not_equal_the_parent(synthetic_preflight):
    payload = synthetic_preflight.payloads[evaluation.CHECKPOINTS / "flat_v3/zero.pt"]
    payload["weights"][0] += 1
    synthetic_preflight.sign(payload)
    with pytest.raises(ValueError, match="zero checkpoint differs"):
        evaluation.preflight(budget())


def test_preflight_rejects_missing_old_retention_probe(synthetic_preflight):
    synthetic_preflight.old_pairs.pop()
    with pytest.raises(ValueError, match="N175 retention identities"):
        evaluation.preflight(budget())
    assert not evaluation.ATTEMPT.exists()


@dataclass(frozen=True)
class FakePromptState:
    workspace: tuple = (1,)
    evidence_refs: tuple = ((0, 0, 1),)


class MemoryLog:
    def __init__(self):
        self.lines = []

    def write(self, value):
        self.lines.append(value)

    def flush(self):
        pass

    def fileno(self):
        return 0


def fake_generation_setup(prepared, monkeypatch, on_generate=lambda size: None):
    calls = []

    class FakeModel:
        def generate(self, state, *, max_new_bytes):
            calls.append(max_new_bytes)
            on_generate(max_new_bytes)
            return SimpleNamespace(generated="蓝色。\n<|end|>".encode())

    prepared["trainers"] = {arm: SimpleNamespace(model=FakeModel()) for arm in evaluation.ARMS}
    prepared["earlier"] = {arm: SimpleNamespace(model=FakeModel()) for arm in evaluation.ARMS}
    monkeypatch.setattr(evaluation, "prompt_state", lambda model, prompt: FakePromptState())
    monkeypatch.setattr(evaluation.os, "fsync", lambda descriptor: None)
    return calls


def test_fixed_generation_preserves_all_1360_outputs_with_no_model(
    synthetic_preflight, monkeypatch
):
    prepared = evaluation.preflight(budget())
    calls = fake_generation_setup(prepared, monkeypatch)
    records, log = [], MemoryLog()
    evaluation.generate_fixed_outputs(
        prepared, synthetic_preflight.sealed_rows, budget(), log, records
    )
    assert len(records) == len(log.lines) == 1360
    assert Counter(row["condition"] for row in records) == {
        "matched": 768,
        "related_fact_flip": 192,
        "irrelevant_fact_flip": 192,
        "workspace_cut": 64,
        "F0_train_at640": 48,
        "F0_train_final": 48,
        "N175_train_retention": 48,
    }
    assert Counter(calls) == {96: 1312, 128: 48}


def test_cpu_budget_still_stops_during_old_retention_generation(synthetic_preflight, monkeypatch):
    prepared = evaluation.preflight(budget())
    clock = [0.0]
    monkeypatch.setattr(evaluation.time, "process_time", lambda: clock[0])

    def consume_on_old_generation(size):
        if size == 128:
            clock[0] += 991

    calls = fake_generation_setup(prepared, monkeypatch, consume_on_old_generation)
    records, log = [], MemoryLog()
    with pytest.raises(TimeoutError, match="1000 CPU-second"):
        evaluation.generate_fixed_outputs(
            prepared,
            synthetic_preflight.sealed_rows,
            evaluation.EvaluationBudget(0.0, 0.0),
            log,
            records,
        )
    assert calls.count(128) == 1
    assert len(records) == len(log.lines) == 1313
    assert records[-1]["condition"] == "N175_train_retention"


def test_cpu_budget_includes_the_old_train_loader(synthetic_preflight, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(evaluation.time, "process_time", lambda: clock[0])

    def costly_loader():
        clock[0] += 991
        return synthetic_preflight.old_pairs, {}

    monkeypatch.setattr(evaluation, "load_train_episodes", costly_loader)
    with pytest.raises(TimeoutError, match="1000 CPU-second"):
        evaluation.preflight(evaluation.EvaluationBudget(0.0, 0.0))
    assert not evaluation.ATTEMPT.exists()


def test_failed_preflight_costs_remain_durable_and_count_toward_the_limit(
    synthetic_preflight, monkeypatch
):
    evaluation.record_cost(
        {"status": "preflight_failed_no_sealed_consumption", "process_cpu_seconds": 2.5}
    )
    evaluation.record_cost(
        {"status": "preflight_failed_no_sealed_consumption", "process_cpu_seconds": 3.5}
    )
    assert evaluation.prior_evaluation_cpu() == 6
    monkeypatch.setattr(evaluation.time, "process_time", lambda: 985.0)
    with pytest.raises(TimeoutError, match="1000 CPU-second"):
        evaluation.EvaluationBudget(0.0, 0.0, evaluation.prior_evaluation_cpu()).check()
    assert not evaluation.BASE.exists() and not evaluation.ATTEMPT.exists()


def test_f2_body_is_skipped_without_json_decoding():
    f0 = {"curriculum_level": "F0", "episode_id": "synthetic-f0", "response": "copy"}
    # Invalid escapes in the unwanted body deliberately fail a full JSON parse.
    f2 = '{"curriculum_level":"F2","user_input":"\\uZZZZ"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(f2)
    assert evaluation.selected_f0_f1_rows([json.dumps(f0), f2], budget()) == [f0]


def test_color_interventions_keep_the_world_oracle_consistent():
    row = next(
        row
        for row in build_rows()
        if row["split"] == "sealed" and row["metadata"]["query_kind"] == "primary_color"
    )
    related, irrelevant = evaluation.color_variants(row)
    assert related["metadata"]["primary_pair"][1] == related["metadata"]["answer_from_world"]
    assert related["response"] == related["metadata"]["answer_from_world"] + "。"
    assert related["metadata"]["primary_pair"] != row["metadata"]["primary_pair"]
    assert irrelevant["metadata"]["distractor_pair"] != row["metadata"]["distractor_pair"]
    assert irrelevant["metadata"]["answer_from_world"] == row["metadata"]["answer_from_world"]
    assert related["user_input"] != row["user_input"]
    assert row["metadata"]["primary_pair"][1] + "。" == row["response"]
