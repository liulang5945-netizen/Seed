"""R2-D2 train-only learnability probe (contract section 5.2, one A-arm run).

Reads only the R2-D1 train split (corpus digest checked, final sealed).  Before
any training it performs the real checkpoint preflight demanded by the
contract: a zero-step atomic save is reloaded in a fresh subprocess, logits
must reproduce bit-for-bit, and one optimizer step must land identically.
Then the per-position evidence arm (graph v3) trains for 30 frozen epochs.
Pass line (frozen before seeing dev): mean train sequence loss <= 0.50 AND
train M1 (whole free-output exact on train) >= 0.90.  Train M3/M4 are
reported descriptively.  Dev and final are never read; no capability claim.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts.training.eval_taiji_r2_d1_surface_policies import metrics  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    EVIDENCE_PER_POSITION,
    SEQUENCE_WORKSPACE_VERSION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURE = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
CORPUS_DIGEST = "53ac9f88695f135d0bb04b4d25d98ab686c82175c45ca8d4d53e64639efbeecb"
CHECKPOINT_DIR = Path("reports/r2_d2_checkpoints/probe")
DEFAULT_REPORT = Path("reports/r2_d2_learnability_probe_20260918.json")
FORMAT = "taiji-r2-d2-learnability-probe-v1"
EPOCHS = 30
SEED = 20260917
FROZEN_LEARNING_RATE = 0.05
LOSS_GATE = 0.50
M1_GATE = 0.90
WALL_CAP_SECONDS = 20 * 60


def _load_split(split: str) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [record for record in records if record["split"] == split]
    if not rows:
        raise RuntimeError(f"{split} split is empty")
    return rows


def _episodes(rows: list[dict[str, Any]]) -> tuple[tuple[bytes, bytes], ...]:
    return tuple(
        (record["prefix"].encode("utf-8"), record["response"].encode("utf-8")) for record in rows
    )


def _train_metrics(
    prototype: SequenceWorkspacePrototype, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Free greedy generation on train, scored with the frozen M1-M5 family."""

    scored: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.no_grad():
        for record in rows:
            prefix = record["prefix"].encode("utf-8")
            gold = record["response"]
            generated = prototype.generate(prefix).bytes_out.decode("utf-8", errors="replace")
            loss, _ = prototype.sequence_loss(prefix, gold.encode("utf-8"))
            losses.append(float(loss))
            scored.append(
                {
                    "id": record["id"],
                    "shape": record["shape"],
                    "source_group": record["source_group"],
                    "content_dependent": record["content_dependent"],
                    "pair_id": record["pair_id"],
                    "pair_type": record["pair_type"],
                    "pair_role": record["pair_role"],
                    "gold": gold,
                    "prediction": generated,
                    "correct": generated == gold,
                }
            )
    measured: dict[str, Any] = metrics(scored)
    measured["mean_sequence_loss"] = sum(losses) / len(losses)
    return measured


def _checkpoint_preflight(
    trainer: SequenceWorkspaceTrainer,
    episodes: tuple[tuple[bytes, bytes], ...],
    target: Path,
) -> dict[str, Any]:
    """Zero-step save -> fresh-process restore -> identical one-step continue."""

    prefix, response = episodes[0]
    expected_logits = content_digest(
        trainer.prototype.teacher_forced_logits(prefix, response).detach()
    )
    path = trainer.save(target)

    uninterrupted = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    uninterrupted.train_step(episodes[:8])
    expected_continued = str(uninterrupted.checkpoint()["checkpoint_digest"])

    script = (
        f"import sys, json; sys.path.insert(0, r'{str(PROJECT_ROOT)}');"
        "import torch;"
        "from taiji.internalization import content_digest;"
        "from taiji.sequence_workspace import SequenceWorkspaceTrainer;"
        f"payload = torch.load(r'{str(path)}', map_location='cpu', weights_only=False);"
        "trainer = SequenceWorkspaceTrainer.from_checkpoint(payload);"
        f"logits = trainer.prototype.teacher_forced_logits({prefix!r}, {response!r});"
        f"trainer.train_step({list(episodes[:8])!r});"
        "print(json.dumps({'logits': content_digest(logits.detach()),"
        " 'continued': str(trainer.checkpoint()['checkpoint_digest']),"
        " 'version': int(trainer.checkpoint()['version']),"
        " 'evidence_source': trainer.prototype.config.evidence_source}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"preflight subprocess failed: {result.stderr}")
    observed: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    evidence: dict[str, Any] = {
        "zero_step_path": str(path),
        "expected_logits_digest": expected_logits,
        "observed_logits_digest": observed["logits"],
        "logits_match": observed["logits"] == expected_logits,
        "expected_continuation_digest": expected_continued,
        "observed_continuation_digest": observed["continued"],
        "continuation_matches": observed["continued"] == expected_continued,
        "restored_version": observed["version"],
        "restored_evidence_source": observed["evidence_source"],
        "passed": (
            observed["logits"] == expected_logits
            and observed["continued"] == expected_continued
            and observed["version"] == SEQUENCE_WORKSPACE_VERSION
            and observed["evidence_source"] == EVIDENCE_PER_POSITION
        ),
    }
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    fixture_path = PROJECT_ROOT / FIXTURE
    raw_records = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    digest = content_digest(raw_records)
    if digest != CORPUS_DIGEST:
        raise RuntimeError(f"R2-D1 corpus digest mismatch: {digest}")

    rows = _load_split("train")
    episodes = _episodes(rows)
    started = time.monotonic()

    torch.manual_seed(SEED)
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(seed=SEED, evidence_source=EVIDENCE_PER_POSITION)
    )
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=FROZEN_LEARNING_RATE, code_revision="r2d2-learnability"
    )
    trainer.set_episodes(episodes)

    checkpoint_dir = PROJECT_ROOT / CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preflight = _checkpoint_preflight(
        trainer, episodes, checkpoint_dir / "a_seed20260917_zero_step.pt"
    )
    if not preflight["passed"]:
        raise RuntimeError(f"checkpoint preflight failed: {preflight}")

    initial = _train_metrics(prototype, rows)
    trajectory: list[dict[str, Any]] = [{"epoch": 0, **_compact(initial)}]
    episodes_trained = 0
    timed_out = False
    for epoch in range(EPOCHS):
        record = trainer.train_epoch()
        episodes_trained += int(record["episodes"])
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            evaluation = _train_metrics(prototype, rows)
            trajectory.append({"epoch": epoch + 1, **_compact(evaluation)})
        if time.monotonic() - started > WALL_CAP_SECONDS:
            timed_out = True
            break
    final = _train_metrics(prototype, rows)
    elapsed = time.monotonic() - started

    trainer.save(checkpoint_dir / "a_seed20260917_epoch30.pt")

    final_m1 = float(final["M1_exact"]["value"])
    gate = {
        "loss_le_0_50": bool(final["mean_sequence_loss"] <= LOSS_GATE),
        "train_m1_ge_0_90": bool(final_m1 >= M1_GATE),
        "within_wall_cap": bool(elapsed <= WALL_CAP_SECONDS and not timed_out),
        "preflight_passed": bool(preflight["passed"]),
    }
    passed = all(gate.values())
    payload = {
        "format": FORMAT,
        "version": 1,
        "contract": "plans/reference/M5_R2_D2_PER_POSITION_EVIDENCE_PREREGISTRATION_FROZEN_20260918.md",
        "graph_version": SEQUENCE_WORKSPACE_VERSION,
        "arm": "A_per_position",
        "corpus_digest": digest,
        "split_read": "train",
        "dev_read": False,
        "final_read": False,
        "seed": SEED,
        "epochs_run": len([item for item in trajectory if item["epoch"] > 0]),
        "epochs_frozen": EPOCHS,
        "frozen_learning_rate": FROZEN_LEARNING_RATE,
        "train_episodes": len(rows),
        "optimizer_steps": episodes_trained,
        "elapsed_seconds": elapsed,
        "wall_cap_seconds": WALL_CAP_SECONDS,
        "parameter_count": prototype.parameter_count(),
        "preflight": preflight,
        "initial": initial,
        "final": final,
        "trajectory": trajectory,
        "gate": gate,
        "outcome": "passed" if passed else "failed",
        "reading": (
            "train-only learnability probe per contract 5.2; train M3/M4 are "
            "descriptive, dev and final unread, not a capability claim"
        ),
        "growth_admitted": False,
        "can_promote": False,
    }
    report_path = PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "outcome": payload["outcome"],
                "elapsed_seconds": round(elapsed, 1),
                "initial_loss": round(initial["mean_sequence_loss"], 4),
                "final_loss": round(final["mean_sequence_loss"], 4),
                "final_train_M1": round(final_m1, 4),
                "descriptive_train_M3": round(float(final["M3_content_exact"]["value"]), 4),
                "descriptive_train_M4": round(float(final["M4_flip_pair"]["value"]), 4),
                "gate": gate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


def _compact(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: round(float(value), 6) if isinstance(value, float) else value
        for key, value in evaluation.items()
        if key
        in {
            "mean_sequence_loss",
            "M1_exact",
            "M2_shape_macro",
            "M3_content_exact",
            "M4_flip_pair",
            "M5_inv_pair",
        }
    }


if __name__ == "__main__":
    raise SystemExit(main())
