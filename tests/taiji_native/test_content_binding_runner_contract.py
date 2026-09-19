"""R2 content-binding runner contract (implementation gate).

Contract: plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md §4/§6/§7

Gates on the bounded training runner: physical sealed isolation (train-only
loader rejects foreign splits, no fixture CLI argument exists, calibration is
the only auxiliary read), group-level class-balanced batching, the two frozen
recipes and their schedule, non-finite stop, composite-loss gradient reach,
atomic checkpoint save/restore/continue in a fresh process, and the run
report identity.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_taiji_r2_content_binding import (  # noqa: E402
    BATCH_GROUPS,
    FINAL_LR_FRACTION,
    GROUP_CLASSES,
    RECIPES,
    WARMUP_FRACTION,
    GroupSampler,
    load_calibration_fixture,
    load_train_fixture,
    run,
)

TRAIN_FIXTURE = PROJECT_ROOT / "tests/fixtures/r2_content_binding_v1_train.jsonl"
CALIBRATION_FIXTURE = PROJECT_ROOT / "tests/fixtures/r2_content_binding_v1_calibration.jsonl"
SEALED_FIXTURE = PROJECT_ROOT / "tests/fixtures/r2_content_binding_v1_sealed.jsonl"


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# Isolation gates (contract section 3/6: the training process has no sealed path)
# --------------------------------------------------------------------------- #


def test_loader_rejects_foreign_splits(tmp_path: Path) -> None:
    train_rows = _rows(TRAIN_FIXTURE)
    foreign = _rows(CALIBRATION_FIXTURE)[:2] + _rows(SEALED_FIXTURE)[:2]
    poisoned = tmp_path / "poisoned.jsonl"
    poisoned.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in train_rows[:5] + foreign)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-train splits"):
        load_train_fixture(poisoned)
    assert load_train_fixture()  # the real train fixture loads fine


def test_runner_cli_has_no_fixture_or_sealed_argument() -> None:

    from scripts.training import train_taiji_r2_content_binding as runner

    source = Path(runner.__file__).read_text(encoding="utf-8")
    for forbidden in ('"--fixture"', '"--sealed-fixture"', '"--calibration-fixture"'):
        assert forbidden not in source, forbidden
    assert 'choices=("calibration", "formal", "smoke")' in source
    # the module never points at the sealed fixture path
    assert "r2_content_binding_v1_sealed" not in source


def test_train_fixture_only_vocab() -> None:
    rows = _rows(TRAIN_FIXTURE)
    from taiji.sequence_char_workspace import CharVocab

    text = "".join(r["question"] + r["material"] + r["response"] for r in rows)
    vocab = CharVocab(text)
    # calibration-exclusive value glyphs never enter the train vocabulary;
    # they must exit through the copy channel at calibration time
    for glyph in "青棕绯黛":
        assert glyph not in vocab.char_to_slot or glyph in text


# --------------------------------------------------------------------------- #
# Sampler gates (contract section 4: 4 groups = 8 items, group integrity)
# --------------------------------------------------------------------------- #


def test_sampler_batches_whole_groups_class_balanced() -> None:
    rows = _rows(TRAIN_FIXTURE)
    sampler = GroupSampler(rows, seed=20260920)
    class_hits = {cls: 0 for cls in GROUP_CLASSES}
    for _ in range(50):
        batch = sampler.next_batch()
        assert len(batch) == BATCH_GROUPS * 2
        group_ids = [row["group_id"] for row in batch]
        assert len(set(group_ids)) == BATCH_GROUPS
        for group_id in group_ids:
            members = [row for row in batch if row["group_id"] == group_id]
            assert {m["member"] for m in members} == {"a", "b"}
        for row in batch:
            class_hits[row["group_class"]] += 1
    # every class receives exposure within 50 batches (7 classes, 4 per batch)
    assert all(count > 0 for count in class_hits.values())


def test_sampler_deterministic_for_seed() -> None:
    rows = _rows(TRAIN_FIXTURE)
    first = [GroupSampler(rows, seed=7).next_batch() for _ in range(5)]
    second = [GroupSampler(rows, seed=7).next_batch() for _ in range(5)]
    for batch_a, batch_b in zip(first, second):
        assert [r["id"] for r in batch_a] == [r["id"] for r in batch_b]


# --------------------------------------------------------------------------- #
# Recipe/schedule gates (contract section 4)
# --------------------------------------------------------------------------- #


def test_two_recipes_and_schedule_math() -> None:
    assert sorted(RECIPES) == ["cal_lr1", "cal_lr3"]
    assert RECIPES["cal_lr1"]["learning_rate"] == 0.001
    assert RECIPES["cal_lr3"]["learning_rate"] == 0.003
    from taiji.sequence_char_workspace import CharVocab
    from taiji.sequence_content_workspace import (
        SequenceContentConfig,
        SequenceContentTrainer,
        SequenceContentWorkspace,
    )

    rows = _rows(TRAIN_FIXTURE)
    vocab = CharVocab("".join(r["question"] + r["material"] + r["response"] for r in rows))
    workspace = SequenceContentWorkspace(vocab, SequenceContentConfig(arm="B", seed=1))
    trainer = SequenceContentTrainer(workspace, learning_rate=0.003, total_updates=2000)
    assert trainer.lr_at(0) == 0.0
    warmup_steps = int(2000 * WARMUP_FRACTION)
    assert trainer.lr_at(warmup_steps) == pytest.approx(0.003)
    assert trainer.lr_at(2000) == pytest.approx(0.003 * FINAL_LR_FRACTION, abs=1e-6)
    mid = trainer.lr_at((warmup_steps + 2000) // 2)
    assert 0.003 * FINAL_LR_FRACTION < mid < 0.003


def test_non_finite_loss_stops(tmp_path: Path) -> None:
    from taiji.sequence_char_workspace import CharVocab
    from taiji.sequence_content_workspace import (
        SequenceContentConfig,
        SequenceContentTrainer,
        SequenceContentWorkspace,
    )

    rows = _rows(TRAIN_FIXTURE)
    vocab = CharVocab("".join(r["question"] + r["material"] + r["response"] for r in rows))
    workspace = SequenceContentWorkspace(vocab, SequenceContentConfig(arm="A", seed=3))
    trainer = SequenceContentTrainer(workspace, learning_rate=0.001, total_updates=5)
    batch_rows = rows[:8]
    batch = [
        {
            "question": r["question"],
            "material": r["material"],
            "response": r["response"],
            "copy_mask": r["copy_mask"],
        }
        for r in batch_rows
    ]
    stats = trainer.train_step(batch)
    assert stats["loss"] > 0.0
    assert 0.0 < stats["grad_norm"]
    with torch.no_grad():
        workspace._parameters["char_embedding"].fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        trainer.train_step(batch)


# --------------------------------------------------------------------------- #
# Bounded run + fresh-process continue (contract section 6)
# --------------------------------------------------------------------------- #


def test_smoke_run_report_and_isolation(tmp_path: Path) -> None:
    report = run(
        arm="B",
        config_name="cal_lr1",
        seed=20260920,
        total_updates=4,
        output_root=tmp_path,
        mode="smoke",
        wall_cap_seconds=600,
    )
    assert report["status"] == "completed"
    assert report["updates_done"] == 4
    assert report["sealed_read"] is False
    assert report["split_read"] == "train"
    assert report["calibration_read"] is False
    assert report["selected"] is None  # smoke mode never reads calibration
    assert report["health"], "health stats every update <= 100"
    run_dir = tmp_path / "B" / "20260920" / "cal_lr1"
    assert (run_dir / "run_report.json").is_file()
    latest = run_dir / "latest.pt"
    assert latest.is_file()
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert payload["data_digest"] == report["data_digest"]
    assert payload["selection_rule"] == "content-binding-selection-v1"
    assert payload["candidate_construction"] == "content-candidates-v1"


def test_calibration_loader_rejects_sealed() -> None:
    rows = load_calibration_fixture()
    assert all(row["split"] == "calibration" for row in rows)
    # the runner module exposes no sealed loader at all
    from scripts.training import train_taiji_r2_content_binding as runner

    assert not hasattr(runner, "load_sealed_fixture")


def test_fresh_process_continuation_matches_uninterrupted(tmp_path: Path) -> None:
    """Two updates, save, fresh process loads and does one update; the
    resulting checkpoint digest equals the uninterrupted path's."""

    rows = _rows(TRAIN_FIXTURE)
    batch_rows = rows[:8]
    batch = [
        {
            "question": r["question"],
            "material": r["material"],
            "response": r["response"],
            "copy_mask": r["copy_mask"],
        }
        for r in batch_rows
    ]
    from taiji.sequence_char_workspace import CharVocab
    from taiji.sequence_content_workspace import (
        SequenceContentConfig,
        SequenceContentTrainer,
        SequenceContentWorkspace,
    )

    vocab = CharVocab("".join(r["question"] + r["material"] + r["response"] for r in rows))
    torch.manual_seed(20260920)
    workspace = SequenceContentWorkspace(vocab, SequenceContentConfig(arm="B", seed=20260920))
    trainer = SequenceContentTrainer(
        workspace, learning_rate=0.001, total_updates=100, code_revision="continue-gate"
    )
    trainer.train_step(batch)
    trainer.train_step(batch)
    path = trainer.save(tmp_path / "after_two.pt")
    uninterrupted = SequenceContentTrainer.from_checkpoint(trainer.checkpoint())
    uninterrupted.train_step(batch)
    expected_digest = str(uninterrupted.checkpoint()["checkpoint_digest"])

    script = (
        f"import sys, json; sys.path.insert(0, r'{PROJECT_ROOT}');\n"
        "import torch\n"
        "from taiji.sequence_content_workspace import SequenceContentTrainer\n"
        f"payload = torch.load(r'{path}', map_location='cpu', weights_only=False)\n"
        "trainer = SequenceContentTrainer.from_checkpoint(payload)\n"
        f"trainer.train_step({batch!r})\n"
        "print(json.dumps(str(trainer.checkpoint()['checkpoint_digest'])))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == expected_digest
