"""R2-D2 H-A2 copy-mixture learnability probe (amendment section 5.2).

One A2 run (per-position evidence + copy mixture, seed 20260917, 30 epochs,
lr 0.01).  Preflight first (v4 checkpoint).  Frozen pass line: exact M1 on the
copy-supported shapes (fact / negation / same_opening_fact = 144 of 174 train
items) >= 0.90 with finite declining loss.  Unknown and two-fact combination
shapes are structurally uncopyable (their answers never appear in the
material); they are reported descriptively and carry the B/C routing signal.
Train only; dev and final unread.
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
CHECKPOINT_DIR = Path("reports/r2_d2_checkpoints/copy_probe")
DEFAULT_REPORT = Path("reports/r2_d2_copy_learnability_probe_20260918.json")
FORMAT = "taiji-r2-d2-copy-learnability-probe-v1"
EPOCHS = 30
SEED = 20260917
FROZEN_LEARNING_RATE = 0.01
COPY_SUPPORTED_SHAPES = ("fact", "negation", "same_opening_fact")
COPY_M1_GATE = 0.90
WALL_CAP_SECONDS = 20 * 60


def _load_train() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row["split"] == "train"]


def _preflight(
    trainer: SequenceWorkspaceTrainer,
    episodes: tuple[tuple[bytes, bytes], ...],
    target: Path,
) -> dict[str, Any]:
    prefix, response = episodes[0]
    expected = content_digest(trainer.prototype.teacher_forced_logits(prefix, response).detach())
    path = trainer.save(target)
    uninterrupted = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    uninterrupted.train_step(episodes[:8])
    continued = str(uninterrupted.checkpoint()["checkpoint_digest"])
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
        " 'copy': bool(trainer.prototype.config.copy_mixture)}))"
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
    return {
        "expected_logits_digest": expected,
        "observed_logits_digest": observed["logits"],
        "logits_match": observed["logits"] == expected,
        "continuation_matches": observed["continued"] == continued,
        "restored_version": observed["version"],
        "restored_copy": observed["copy"],
        "passed": (
            observed["logits"] == expected
            and observed["continued"] == continued
            and observed["version"] == SEQUENCE_WORKSPACE_VERSION
            and observed["copy"] is True
        ),
    }


def _evaluate(prototype: SequenceWorkspacePrototype, rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    supported = [row for row in scored if row["shape"] in COPY_SUPPORTED_SHAPES]
    measured["copy_supported_M1"] = {
        "numerator": sum(1 for row in supported if row["correct"]),
        "denominator": len(supported),
        "value": sum(1 for row in supported if row["correct"]) / len(supported),
        "shapes": list(COPY_SUPPORTED_SHAPES),
    }
    per_shape: dict[str, dict[str, Any]] = {}
    for shape in sorted({row["shape"] for row in scored}):
        group = [row for row in scored if row["shape"] == shape]
        per_shape[shape] = {
            "denominator": len(group),
            "value": sum(1 for row in group if row["correct"]) / len(group),
        }
    measured["per_shape_exact"] = per_shape
    return measured


def _compact(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_sequence_loss": round(float(evaluation["mean_sequence_loss"]), 6),
        "M1_exact": evaluation["M1_exact"]["value"],
        "copy_supported_M1": evaluation["copy_supported_M1"]["value"],
        "M3_content_exact": evaluation["M3_content_exact"]["value"],
        "M4_flip_pair": evaluation["M4_flip_pair"]["value"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    raw = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    digest = content_digest(raw)
    if digest != CORPUS_DIGEST:
        raise RuntimeError(f"R2-D1 corpus digest mismatch: {digest}")

    rows = _load_train()
    episodes = tuple(
        (row["prefix"].encode("utf-8"), row["response"].encode("utf-8")) for row in rows
    )
    started = time.monotonic()
    torch.manual_seed(SEED)
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(seed=SEED, evidence_source=EVIDENCE_PER_POSITION, copy_mixture=True)
    )
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=FROZEN_LEARNING_RATE, code_revision="r2d2-copy-probe"
    )
    trainer.set_episodes(episodes)
    checkpoint_dir = PROJECT_ROOT / CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(trainer, episodes, checkpoint_dir / "a2_seed20260917_zero_step.pt")
    if not preflight["passed"]:
        raise RuntimeError(f"checkpoint preflight failed: {preflight}")

    initial = _evaluate(prototype, rows)
    trajectory = [{"epoch": 0, **_compact(initial)}]
    for epoch in range(EPOCHS):
        trainer.train_epoch()
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            trajectory.append({"epoch": epoch + 1, **_compact(_evaluate(prototype, rows))})
    final = _evaluate(prototype, rows)
    elapsed = time.monotonic() - started
    trainer.save(checkpoint_dir / "a2_seed20260917_epoch30.pt")

    supported_m1 = float(final["copy_supported_M1"]["value"])
    gate = {
        "copy_supported_m1_ge_0_90": supported_m1 >= COPY_M1_GATE,
        "finite_declining_loss": bool(final["mean_sequence_loss"] < initial["mean_sequence_loss"]),
        "within_wall_cap": elapsed <= WALL_CAP_SECONDS,
        "preflight_passed": preflight["passed"],
    }
    passed = all(gate.values())
    payload = {
        "format": FORMAT,
        "version": 1,
        "contract": "plans/reference/M5_R2_D2_COPY_MIXTURE_AMENDMENT_FROZEN_20260918.md",
        "graph_version": SEQUENCE_WORKSPACE_VERSION,
        "arm": "A2_per_position_copy",
        "corpus_digest": digest,
        "split_read": "train",
        "dev_read": False,
        "final_read": False,
        "seed": SEED,
        "epochs_frozen": EPOCHS,
        "frozen_learning_rate": FROZEN_LEARNING_RATE,
        "train_episodes": len(rows),
        "copy_supported_shapes": list(COPY_SUPPORTED_SHAPES),
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
            "copy-mixture train-only learnability per amendment 5.2; "
            "uncopyable shapes and M3/M4 are descriptive; not a capability claim"
        ),
        "growth_admitted": False,
        "can_promote": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "outcome": payload["outcome"],
                "elapsed_seconds": round(elapsed, 1),
                "loss": [
                    round(initial["mean_sequence_loss"], 4),
                    round(final["mean_sequence_loss"], 4),
                ],
                "copy_supported_M1": round(supported_m1, 4),
                "overall_M1": round(float(final["M1_exact"]["value"]), 4),
                "per_shape": {
                    shape: round(values["value"], 3)
                    for shape, values in final["per_shape_exact"].items()
                },
                "descriptive_M4": round(float(final["M4_flip_pair"]["value"]), 4),
                "gate": gate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
