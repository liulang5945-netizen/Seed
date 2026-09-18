"""R2-D4 H-T copy-value supervision probe (contract section 4).

One run on the v6 graph (H=4 heads + positional keys, 96,034 parameters,
unchanged) with the single frozen objective variable: auxiliary NLL of the
true answer value bytes under the **copy component** (lambda = 1.0).  Value
positions are derived per shape (fact/sof: whole response; negation: minus the
leading `不是`; unknown/combo answers carry no color value).  Frozen gates:
copy-supported M1 >= 0.90 (P1), value-position copy probability >= 0.90 (P2),
that readout collapsing to <= 0.50 under the entry-rotation misbind lesion
(P3), and the Q1 stability criteria (P4).  Train only; dev and final unread.
"""

from __future__ import annotations

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
    SEQUENCE_WORKSPACE_VERSION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURE = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
CORPUS_DIGEST = "53ac9f88695f135d0bb04b4d25d98ab686c82175c45ca8d4d53e64639efbeecb"
CHECKPOINT_DIR = Path("reports/r2_d4_checkpoints/copy_supervision_probe")
OUT_REPORT = Path("reports/r2_d4_copy_supervision_probe_20260918.json")
CONTRACT = "plans/reference/M5_R2_D4_COPY_SUPERVISION_CONTRACT_FROZEN_20260918.md"
EPOCHS = 30
SEED = 20260917
LEARNING_RATE = 0.01
MICROBATCH = 8
LAMBDA_COPY = 1.0
COPY_SUPPORTED_SHAPES = ("fact", "negation", "same_opening_fact")
NO_VALUE_SHAPES = ("unknown", "same_opening_unknown", "combination_same", "combination_different")
NEGATION_LEAD = "不是".encode("utf-8")
WALL_CAP_SECONDS = 20 * 60


def value_mask_for(shape: str, response: str) -> tuple[bool, ...]:
    """R2-D4 contract section 2.2: response-byte mask of answer value positions.

    fact / same_opening_fact answers are the bare color value; negation answers
    are `不是` + value; the remaining shapes never contain a color value, so
    their masks are all-False and contribute nothing to the auxiliary term.
    """

    encoded = response.encode("utf-8")
    if shape in ("fact", "same_opening_fact"):
        return (True,) * len(encoded)
    if shape == "negation":
        if encoded[: len(NEGATION_LEAD)] != NEGATION_LEAD:
            raise ValueError(f"negation response must start with 不是: {response!r}")
        return (False,) * len(NEGATION_LEAD) + (True,) * (len(encoded) - len(NEGATION_LEAD))
    if shape not in NO_VALUE_SHAPES:
        raise ValueError(f"unknown shape for value mask: {shape!r}")
    return (False,) * len(encoded)


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
    masks: tuple[tuple[bool, ...], ...],
    target: Path,
) -> dict[str, Any]:
    prefix, response = episodes[0]
    expected = content_digest(trainer.prototype.teacher_forced_logits(prefix, response).detach())
    path = trainer.save(target)
    uninterrupted = SequenceWorkspaceTrainer.from_checkpoint(trainer.checkpoint())
    uninterrupted.enable_copy_value_supervision(LAMBDA_COPY)
    uninterrupted.train_step(episodes[:8], value_masks=[tuple(masks[i]) for i in range(8)])
    continued = str(uninterrupted.checkpoint()["checkpoint_digest"])
    script = (
        f"import sys, json; sys.path.insert(0, r'{str(PROJECT_ROOT)}');"
        "import torch;"
        "from taiji.internalization import content_digest;"
        "from taiji.sequence_workspace import SequenceWorkspaceTrainer;"
        f"payload = torch.load(r'{str(path)}', map_location='cpu', weights_only=False);"
        "trainer = SequenceWorkspaceTrainer.from_checkpoint(payload);"
        f"trainer.enable_copy_value_supervision({LAMBDA_COPY!r});"
        f"logits = trainer.prototype.teacher_forced_logits({prefix!r}, {response!r});"
        f"trainer.train_step({list(episodes[:8])!r},"
        f" value_masks={[tuple(masks[i]) for i in range(8)]!r});"
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


def _copy_value_readout(
    prototype: SequenceWorkspacePrototype,
    rows: list[dict[str, Any]],
    *,
    entry_rotation: int | None = 0,
) -> dict[str, Any]:
    """Mean copy-component probability of the true value bytes (P2/P3 readout).

    ``entry_rotation=None`` applies the matched-dev misbind convention per row
    (``len(prefix) // 2``); an int applies the same literal rotation to all rows.
    """

    boundary = int(prototype.config.boundary_symbol)
    total = 0.0
    count = 0
    with torch.no_grad():
        for row in rows:
            if row["shape"] not in COPY_SUPPORTED_SHAPES:
                continue
            mask = value_mask_for(row["shape"], row["response"])
            if not any(mask):
                continue
            prefix = row["prefix"].encode("utf-8")
            response = row["response"].encode("utf-8")
            rotation = len(prefix) // 2 if entry_rotation is None else entry_rotation
            _, copies = prototype.teacher_forced_mixture_and_copy(
                prefix, response, entry_rotation=rotation
            )
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            targets = torch.tensor(list(response) + [boundary], dtype=torch.long)[:-1]
            hits = copies[:-1][mask_tensor].gather(1, targets[mask_tensor].unsqueeze(1))
            total += float(hits.sum())
            count += int(hits.numel())
    value = total / count if count else 0.0
    return {"mean_copy_value_prob": value, "positions": count, "entry_rotation": entry_rotation}


def _compact(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_sequence_loss": round(float(evaluation["mean_sequence_loss"]), 6),
        "M1_exact": evaluation["M1_exact"]["value"],
        "copy_supported_M1": evaluation["copy_supported_M1"]["value"],
        "M3_content_exact": evaluation["M3_content_exact"]["value"],
        "M4_flip_pair": evaluation["M4_flip_pair"]["value"],
    }


def main() -> int:
    started = time.monotonic()
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
    masks = tuple(value_mask_for(row["shape"], row["response"]) for row in rows)
    supervised = sum(1 for mask in masks if any(mask))
    torch.manual_seed(SEED)
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=SEED,
            evidence_source="per_position",
            copy_mixture=True,
            question_conditioned_start=True,
            readout_heads=4,
            positional_keys=True,
        )
    )
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=LEARNING_RATE, code_revision="r2-d4-copy-supervision-probe"
    )
    trainer.enable_copy_value_supervision(LAMBDA_COPY)
    trainer.set_episodes(episodes)
    checkpoint_dir = PROJECT_ROOT / CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(trainer, episodes, masks, checkpoint_dir / "zero_step.pt")
    if not preflight["passed"]:
        raise RuntimeError(f"checkpoint preflight failed: {preflight}")

    initial = _evaluate(prototype, rows)
    trajectory = [{"epoch": 0, **_compact(initial)}]
    steps_taken = 0
    for epoch in range(EPOCHS):
        for start in range(0, len(episodes), MICROBATCH):
            batch = list(episodes[start : start + MICROBATCH])
            batch_masks = [masks[index] for index in range(start, start + len(batch))]
            trainer.train_step(batch, value_masks=batch_masks)
            steps_taken += 1
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            evaluation = _evaluate(prototype, rows)
            trajectory.append({"epoch": epoch + 1, **_compact(evaluation)})
    final = _evaluate(prototype, rows)
    elapsed = time.monotonic() - started
    trainer.save(checkpoint_dir / "epoch30.pt")

    supported_m1 = float(final["copy_supported_M1"]["value"])
    per_shape_final = {shape: values["value"] for shape, values in final["per_shape_exact"].items()}
    copy_intact = _copy_value_readout(prototype, rows)
    copy_misbound = _copy_value_readout(prototype, rows, entry_rotation=None)
    trajectory_by_epoch = {item["epoch"]: item for item in trajectory}
    m1_e25 = float(trajectory_by_epoch[25]["copy_supported_M1"])
    m1_e30 = float(trajectory_by_epoch[30]["copy_supported_M1"])
    # Q1 amendment four section 3 stability gate, exact wording: pairwise checks
    # over the observed trajectory **excluding the epoch-0 pre-training reading**.
    loss_points = [point["mean_sequence_loss"] for point in trajectory[1:]]
    bounded_increases = all(
        later <= earlier * 1.15
        for earlier, later in zip(loss_points[:-1], loss_points[1:], strict=True)
    )
    gate = {
        "P1_copy_supported_m1_ge_0_90": supported_m1 >= 0.90,
        "P2_copy_value_prob_ge_0_90": float(copy_intact["mean_copy_value_prob"]) >= 0.90,
        "P3_misbind_copy_prob_le_0_50": float(copy_misbound["mean_copy_value_prob"]) <= 0.50,
        "P4_loss_increases_bounded_15pct": bounded_increases,
        "P4_no_late_collapse_gt_0_10": m1_e30 >= m1_e25 - 0.10,
        "preflight_passed": bool(preflight["passed"]),
        "within_wall_cap": elapsed <= WALL_CAP_SECONDS,
    }
    passed = all(gate.values())
    report = {
        "format": "taiji-r2-d4-copy-supervision-probe-v1",
        "version": 1,
        "contract": CONTRACT,
        "graph_version": SEQUENCE_WORKSPACE_VERSION,
        "arm": f"ht_h4_pe_copyvalue_lambda{LAMBDA_COPY}_microbatch{MICROBATCH}",
        "corpus_digest": digest,
        "split_read": "train",
        "dev_read": False,
        "final_read": False,
        "seed": SEED,
        "epochs_frozen": EPOCHS,
        "frozen_learning_rate": LEARNING_RATE,
        "microbatch_size": MICROBATCH,
        "lambda_copy_value": LAMBDA_COPY,
        "supervised_episodes": supervised,
        "train_episodes": len(episodes),
        "copy_supported_shapes": list(COPY_SUPPORTED_SHAPES),
        "no_value_shapes": list(NO_VALUE_SHAPES),
        "optimizer_steps": steps_taken,
        "elapsed_seconds": elapsed,
        "wall_cap_seconds": WALL_CAP_SECONDS,
        "parameter_count": prototype.parameter_count(),
        "preflight": preflight,
        "initial": {key: value for key, value in initial.items() if key != "per_shape_exact"},
        "final": {key: value for key, value in final.items() if key != "per_shape_exact"},
        "per_shape_exact": per_shape_final,
        "trajectory": trajectory,
        "copy_value_readout_intact": copy_intact,
        "copy_value_readout_misbound": copy_misbound,
        "gate": gate,
        "outcome": "passed" if passed else "failed",
        "reading": (
            "copy-component value supervision on the unchanged v6 graph; "
            "train-only learnability and copy-addressing readouts; not a capability claim"
        ),
        "growth_admitted": False,
        "can_promote": False,
    }
    (PROJECT_ROOT / OUT_REPORT).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "outcome": report["outcome"],
                "copy_supported_M1": supported_m1,
                "per_shape": per_shape_final,
                "copy_value_prob_intact": round(float(copy_intact["mean_copy_value_prob"]), 4),
                "copy_value_prob_misbound": round(float(copy_misbound["mean_copy_value_prob"]), 4),
                "loss_trajectory": [item["mean_sequence_loss"] for item in trajectory],
                "gate": gate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
