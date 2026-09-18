"""R2-D5 factorial learnability probes (contract section 4).

One run per surviving factor arm, seed 20260917, micro-batch 8, 30 epochs,
lr 0.01, lambda=1.0 copy-value supervision throughout (the D4 objective stays
frozen as the shared base):

- ``A10``: original train + graph v7 persistence (mechanism only).  Gates are
  the D4 regression gates - persistence must not break single-byte binding;
  the trained bias value is reported descriptively.
- ``A01``: fixture v2 train (three 6-byte colors) + graph v6 (coverage only).
  Core new gate: M1 over the 72 multi-byte value episodes >= 0.90, single-byte
  regression >= 0.90, copy-value probability / misbind / stability as before.
- ``A11``: v2 train + v7 (both factors), same gates as A01 plus bias.

Train only; dev and final unread.  Usage:
``python scripts/training/probe_taiji_r2_d5_factorial.py --arm A01``
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
from scripts.training.probe_taiji_r2_d4_copy_supervision import (  # noqa: E402
    COPY_SUPPORTED_SHAPES,
    value_mask_for,
)
from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    EVIDENCE_PER_POSITION,
    SEQUENCE_WORKSPACE_VERSION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURES = {
    "v1": Path("tests/fixtures/r2_d1_measurement_v1.jsonl"),
    "v2": Path("tests/fixtures/r2_d1_measurement_v2.jsonl"),
}
V1_CORPUS_DIGEST = "53ac9f88695f135d0bb04b4d25d98ab686c82175c45ca8d4d53e64639efbeecb"
V2_CORPUS_DIGEST = "182593f8dead"  # recomputed and asserted at load time
NEW_COLORS = ("琥珀", "珊瑚", "翡翠")
CHECKPOINT_ROOT = Path("reports/r2_d5_checkpoints")
OUT_DIR = Path("reports")
CONTRACT = "plans/reference/M5_R2_D5_MULTIBYTE_FACTORIAL_CONTRACT_FROZEN_20260918.md"
EPOCHS = 30
SEED = 20260917
LEARNING_RATE = 0.01
MICROBATCH = 8
LAMBDA_COPY = 1.0
WALL_CAP_SECONDS = 20 * 60

ARMS = {
    # arm -> (fixture, copy_persistence, label)
    "A10": ("v1", True, "hm_persistence_only"),
    "A01": ("v2", False, "he_coverage_only"),
    "A11": ("v2", True, "hehm_coverage_and_persistence"),
}


def _load_train(version: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURES[version]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    digest = content_digest(rows)
    if version == "v1" and digest != V1_CORPUS_DIGEST:
        raise RuntimeError(f"v1 corpus digest mismatch: {digest}")
    return [row for row in rows if row["split"] == "train"]


def _is_multibyte(record: dict[str, Any]) -> bool:
    response = record["response"]
    if record["shape"] == "negation":
        response = response[2:]
    return len(response.encode("utf-8")) > 3


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
    uninterrupted.train_step(episodes[:8], value_masks=[masks[i] for i in range(8)])
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
    per_shape: dict[str, dict[str, Any]] = {}
    for shape in sorted({row["shape"] for row in scored}):
        group = [row for row in scored if row["shape"] == shape]
        per_shape[shape] = {
            "denominator": len(group),
            "value": sum(1 for row in group if row["correct"]) / len(group),
        }
    measured["per_shape_exact"] = per_shape
    supported = [row for row in scored if row["shape"] in COPY_SUPPORTED_SHAPES]
    measured["copy_supported_M1"] = {
        "numerator": sum(1 for row in supported if row["correct"]),
        "denominator": len(supported),
        "value": sum(1 for row in supported if row["correct"]) / len(supported),
        "shapes": list(COPY_SUPPORTED_SHAPES),
    }
    return measured


def _group_exact(prototype: SequenceWorkspacePrototype, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Split copy-supported episodes by answer value length (v2 arms)."""

    groups: dict[str, list[dict[str, Any]]] = {"multibyte": [], "singlebyte": []}
    with torch.no_grad():
        for row in rows:
            if row["shape"] not in COPY_SUPPORTED_SHAPES:
                continue
            key = "multibyte" if _is_multibyte(row) else "singlebyte"
            generated = (
                prototype.generate(row["prefix"].encode("utf-8")).bytes_out.decode(
                    "utf-8", errors="replace"
                )
                == row["response"]
            )
            groups[key].append(generated)
    return {
        name: {
            "numerator": sum(1 for ok in members if ok),
            "denominator": len(members),
            "value": sum(1 for ok in members if ok) / len(members) if members else 0.0,
        }
        for name, members in groups.items()
    }


def _copy_value_readout(
    prototype: SequenceWorkspacePrototype,
    rows: list[dict[str, Any]],
    *,
    misbind: bool = False,
) -> dict[str, Any]:
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
            rotation = len(prefix) // 2 if misbind else 0
            _, copies = prototype.teacher_forced_mixture_and_copy(
                prefix, response, entry_rotation=rotation
            )
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            targets = torch.tensor(list(response) + [boundary], dtype=torch.long)[:-1]
            hits = copies[:-1][mask_tensor].gather(1, targets[mask_tensor].unsqueeze(1))
            total += float(hits.sum())
            count += int(hits.numel())
    return {"mean_copy_value_prob": total / count if count else 0.0, "positions": count}


def _compact(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_sequence_loss": round(float(evaluation["mean_sequence_loss"]), 6),
        "M1_exact": evaluation["M1_exact"]["value"],
        "copy_supported_M1": evaluation["copy_supported_M1"]["value"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    args = parser.parse_args()
    fixture_version, persistence, label = ARMS[args.arm]

    started = time.monotonic()
    rows = _load_train(fixture_version)
    episodes = tuple(
        (row["prefix"].encode("utf-8"), row["response"].encode("utf-8")) for row in rows
    )
    masks = tuple(value_mask_for(row["shape"], row["response"]) for row in rows)
    torch.manual_seed(SEED)
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=SEED,
            evidence_source=EVIDENCE_PER_POSITION,
            copy_mixture=True,
            question_conditioned_start=True,
            readout_heads=4,
            positional_keys=True,
            copy_persistence=persistence,
        )
    )
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=LEARNING_RATE, code_revision=f"r2-d5-{args.arm}-probe"
    )
    trainer.enable_copy_value_supervision(LAMBDA_COPY)
    trainer.set_episodes(episodes)
    checkpoint_dir = PROJECT_ROOT / CHECKPOINT_ROOT / args.arm
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(trainer, episodes, masks, checkpoint_dir / "zero_step.pt")
    if not preflight["passed"]:
        raise RuntimeError(f"checkpoint preflight failed: {preflight}")

    initial = _evaluate(prototype, rows)
    trajectory = [{"epoch": 0, **_compact(initial)}]
    for epoch in range(EPOCHS):
        for start in range(0, len(episodes), MICROBATCH):
            batch = list(episodes[start : start + MICROBATCH])
            batch_masks = [masks[index] for index in range(start, start + len(batch))]
            trainer.train_step(batch, value_masks=batch_masks)
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            trajectory.append({"epoch": epoch + 1, **_compact(_evaluate(prototype, rows))})
    final = _evaluate(prototype, rows)
    elapsed = time.monotonic() - started
    trainer.save(checkpoint_dir / "epoch30.pt")

    supported_m1 = float(final["copy_supported_M1"]["value"])
    per_shape_final = {shape: values["value"] for shape, values in final["per_shape_exact"].items()}
    copy_intact = _copy_value_readout(prototype, rows)
    copy_misbound = _copy_value_readout(prototype, rows, misbind=True)
    bias_end = (
        float(prototype._parameters["copy_persist_bias"].detach().item()) if persistence else None
    )
    loss_points = [point["mean_sequence_loss"] for point in trajectory[1:]]
    bounded_increases = all(
        later <= earlier * 1.15
        for earlier, later in zip(loss_points[:-1], loss_points[1:], strict=True)
    )
    traj_by_epoch = {item["epoch"]: item for item in trajectory}
    m1_e25 = float(traj_by_epoch[25]["copy_supported_M1"])
    m1_e30 = float(traj_by_epoch[30]["copy_supported_M1"])
    gate: dict[str, Any] = {
        "copy_supported_m1_ge_0_90": supported_m1 >= 0.90,
        "copy_value_prob_ge_0_90": float(copy_intact["mean_copy_value_prob"]) >= 0.90,
        "misbind_copy_prob_le_0_50": float(copy_misbound["mean_copy_value_prob"]) <= 0.50,
        "loss_increases_bounded_15pct": bounded_increases,
        "no_late_collapse_gt_0_10": m1_e30 >= m1_e25 - 0.10,
        "preflight_passed": bool(preflight["passed"]),
        "within_wall_cap": elapsed <= WALL_CAP_SECONDS,
    }
    groups: dict[str, Any] | None = None
    if fixture_version == "v2":
        groups = _group_exact(prototype, rows)
        gate["multibyte_m1_ge_0_90"] = float(groups["multibyte"]["value"]) >= 0.90
        gate["singlebyte_m1_ge_0_90"] = float(groups["singlebyte"]["value"]) >= 0.90
    passed = all(gate.values())
    report = {
        "format": "taiji-r2-d5-factorial-probe-v1",
        "version": 1,
        "contract": CONTRACT,
        "graph_version": SEQUENCE_WORKSPACE_VERSION,
        "arm": args.arm,
        "arm_label": label,
        "fixture": fixture_version,
        "copy_persistence": persistence,
        "lambda_copy_value": LAMBDA_COPY,
        "microbatch_size": MICROBATCH,
        "corpus_digest": content_digest(
            [
                json.loads(line)
                for line in (PROJECT_ROOT / FIXTURES[fixture_version])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        ),
        "split_read": "train",
        "dev_read": False,
        "final_read": False,
        "seed": SEED,
        "epochs_frozen": EPOCHS,
        "frozen_learning_rate": LEARNING_RATE,
        "train_episodes": len(episodes),
        "elapsed_seconds": elapsed,
        "wall_cap_seconds": WALL_CAP_SECONDS,
        "parameter_count": prototype.parameter_count(),
        "copy_persist_bias_end": bias_end,
        "preflight": preflight,
        "final": {key: value for key, value in final.items() if key != "per_shape_exact"},
        "per_shape_exact": per_shape_final,
        "value_length_groups": groups,
        "trajectory": trajectory,
        "copy_value_readout_intact": copy_intact,
        "copy_value_readout_misbound": copy_misbound,
        "gate": gate,
        "outcome": "passed" if passed else "failed",
        "reading": (
            "R2-D5 factorial learnability probe (train-only); persistence/coverage "
            "factor readouts; not a capability claim"
        ),
        "growth_admitted": False,
        "can_promote": False,
    }
    out_path = PROJECT_ROOT / OUT_DIR / f"r2_d5_probe_{args.arm}_{label}_20260918.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "outcome": report["outcome"],
                "copy_supported_M1": round(supported_m1, 4),
                "per_shape": {k: round(v, 3) for k, v in per_shape_final.items()},
                "value_length_groups": groups,
                "copy_value_prob_intact": round(float(copy_intact["mean_copy_value_prob"]), 4),
                "copy_value_prob_misbound": round(
                    float(copy_misbound["mean_copy_value_prob"]), 4
                ),
                "bias_end": bias_end,
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
