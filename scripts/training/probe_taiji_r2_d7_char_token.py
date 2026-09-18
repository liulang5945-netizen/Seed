"""R2-D7 H-Tok character-token learnability probes (contract section 4).

Two arms on the v2 train split (multibyte values), seed 20260917, micro-batch
8, 30 epochs, lr 0.01:

- ``T0``: char graph WITHOUT induction (copy_value supervision only) —
  isolates the granularity effect from the induction prior.
- ``T1``: char graph WITH induction (full D4+D6 mechanism stack at the
  character unit) — the primary arm.

Gates (train-only): copy-supported M1 >= 0.90, multibyte value rows >= 0.90,
value-position copy probability >= 0.90, entry-row misbind collapse <= 0.50,
Q1 stability.  The induction bias final value is descriptive.  dev/final
never read.  Usage: ``python scripts/training/probe_taiji_r2_d7_char_token.py --arm T1``
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

from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_char_workspace import (  # noqa: E402
    CHAR_BOUNDARY_SLOT,
    SEQUENCE_CHAR_WORKSPACE_VERSION,
    CharVocab,
    SequenceCharConfig,
    SequenceCharTrainer,
    SequenceCharWorkspace,
)

FIXTURE = Path("tests/fixtures/r2_d1_measurement_v2.jsonl")
CHECKPOINT_ROOT = Path("reports/r2_d7_checkpoints")
OUT_DIR = Path("reports")
CONTRACT = "plans/reference/M5_R2_D7_CHAR_TOKEN_CONTRACT_FROZEN_20260918.md"
EPOCHS = 30
SEED = 20260917
LEARNING_RATE = 0.01
MICROBATCH = 8
LAMBDA_COPY = 1.0
COPY_SUPPORTED_SHAPES = ("fact", "negation", "same_opening_fact")
WALL_CAP_SECONDS = 20 * 60
ARMS = {"T0": False, "T1": True}  # arm -> induction flag


def value_mask_for_char(shape: str, response: str) -> tuple[bool, ...]:
    """Character-unit value mask (contract section 2.4)."""

    if shape in ("fact", "same_opening_fact"):
        return (True,) * len(response)
    if shape == "negation":
        if response[:2] != "不是":
            raise ValueError(f"negation response must start with 不是: {response!r}")
        return (False, False) + (True,) * (len(response) - 2)
    if shape in ("unknown", "same_opening_unknown", "combination_same", "combination_different"):
        return (False,) * len(response)
    raise ValueError(f"unknown shape for value mask: {shape!r}")


def _load_train() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row["split"] == "train"]


def _train_text(rows: list[dict[str, Any]]) -> str:
    return "".join(row["prefix"] + row["response"] for row in rows)


def _is_multibyte(row: dict[str, Any]) -> bool:
    value = row["response"][2:] if row["shape"] == "negation" else row["response"]
    return len(value) > 1


def _preflight(
    trainer: SequenceCharTrainer,
    episodes: tuple[tuple[str, str], ...],
    masks: tuple[tuple[bool, ...], ...],
    target: Path,
) -> dict[str, Any]:
    prefix, response = episodes[0]
    mixed, _ = trainer.workspace.teacher_forced_distributions(prefix, response)
    expected = content_digest(mixed.detach())
    path = trainer.save(target)
    uninterrupted = SequenceCharTrainer.from_checkpoint(trainer.checkpoint())
    uninterrupted.enable_copy_value_supervision(LAMBDA_COPY)
    uninterrupted.train_step(episodes[:8], value_masks=[masks[i] for i in range(8)])
    continued = str(uninterrupted.checkpoint()["checkpoint_digest"])
    script = (
        f"import sys, json; sys.path.insert(0, r'{str(PROJECT_ROOT)}');"
        "import torch;"
        "from taiji.internalization import content_digest;"
        "from taiji.sequence_char_workspace import SequenceCharTrainer;"
        f"payload = torch.load(r'{str(path)}', map_location='cpu', weights_only=False);"
        "trainer = SequenceCharTrainer.from_checkpoint(payload);"
        f"trainer.enable_copy_value_supervision({LAMBDA_COPY!r});"
        f"mixed, _ = trainer.workspace.teacher_forced_distributions({prefix!r}, {response!r});"
        f"trainer.train_step({list(episodes[:8])!r},"
        f" value_masks={[tuple(masks[i]) for i in range(8)]!r});"
        "print(json.dumps({'mixed': content_digest(mixed.detach()),"
        " 'continued': str(trainer.checkpoint()['checkpoint_digest']),"
        " 'version': int(trainer.checkpoint()['version'])}))"
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
        "expected_mixture_digest": expected,
        "observed_mixture_digest": observed["mixed"],
        "mixture_match": observed["mixed"] == expected,
        "continuation_matches": observed["continued"] == continued,
        "restored_version": observed["version"],
        "passed": (
            observed["mixed"] == expected
            and observed["continued"] == continued
            and observed["version"] == SEQUENCE_CHAR_WORKSPACE_VERSION
        ),
    }


def _evaluate(workspace: SequenceCharWorkspace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.no_grad():
        for row in rows:
            generated = workspace.generate(row["prefix"]).text
            loss, _ = workspace.sequence_loss(row["prefix"], row["response"])
            losses.append(float(loss))
            scored.append(
                {
                    "id": row["id"],
                    "shape": row["shape"],
                    "gold": row["response"],
                    "prediction": generated,
                    "correct": generated == row["response"],
                }
            )
    supported = [row for row in scored if row["shape"] in COPY_SUPPORTED_SHAPES]

    def rate(members: list[dict[str, Any]]) -> dict[str, Any]:
        hits = sum(1 for row in members if row["correct"])
        return {
            "numerator": hits,
            "denominator": len(members),
            "value": hits / len(members) if members else 0.0,
        }

    per_shape = {
        shape: rate([row for row in scored if row["shape"] == shape])
        for shape in sorted({row["shape"] for row in scored})
    }
    return {
        "M1_exact": rate(scored),
        "copy_supported_M1": rate(supported),
        "per_shape_exact": per_shape,
        "mean_sequence_loss": sum(losses) / len(losses),
    }


def _value_groups(workspace: SequenceCharWorkspace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[bool]] = {"multibyte": [], "singlebyte": []}
    with torch.no_grad():
        for row in rows:
            if row["shape"] not in COPY_SUPPORTED_SHAPES:
                continue
            key = "multibyte" if _is_multibyte(row) else "singlebyte"
            groups[key].append(workspace.generate(row["prefix"]).text == row["response"])
    return {
        name: {
            "numerator": sum(1 for ok in members if ok),
            "denominator": len(members),
            "value": sum(1 for ok in members if ok) / len(members) if members else 0.0,
        }
        for name, members in groups.items()
    }


def _copy_value_readout(
    workspace: SequenceCharWorkspace, rows: list[dict[str, Any]], *, misbind: bool = False
) -> dict[str, Any]:
    total = 0.0
    count = 0
    with torch.no_grad():
        for row in rows:
            if row["shape"] not in COPY_SUPPORTED_SHAPES:
                continue
            mask = value_mask_for_char(row["shape"], row["response"])
            if not any(mask):
                continue
            rotation = len(row["prefix"]) // 2 if misbind else 0
            _, copies = workspace.teacher_forced_distributions(
                row["prefix"], row["response"], entry_rotation=rotation
            )
            # Target slots come from the INTACT episode: the lesion may not
            # change the answer's identity space, only which row supplies it
            # (byte-graph twin: p_copy's index space stays the true bytes).
            state = workspace.begin_episode(row["prefix"])
            missing = [c for c in row["response"] if c not in state.char2slot]
            assert not missing, f"target glyph outside material: {missing}"
            slots = torch.tensor([state.char2slot[c] for c in row["response"]], dtype=torch.long)
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            hits = copies[:-1][mask_tensor].gather(1, slots[mask_tensor].unsqueeze(1))
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
    induction = ARMS[args.arm]

    started = time.monotonic()
    rows = _load_train()
    vocab = CharVocab(_train_text(rows))
    episodes = tuple((row["prefix"], row["response"]) for row in rows)
    masks = tuple(value_mask_for_char(row["shape"], row["response"]) for row in rows)
    torch.manual_seed(SEED)
    workspace = SequenceCharWorkspace(
        vocab, SequenceCharConfig(seed=SEED, copy_induction=induction)
    )
    trainer = SequenceCharTrainer(
        workspace, learning_rate=LEARNING_RATE, code_revision=f"r2-d7-{args.arm}-probe"
    )
    trainer.enable_copy_value_supervision(LAMBDA_COPY)
    trainer.set_episodes(episodes)
    checkpoint_dir = PROJECT_ROOT / CHECKPOINT_ROOT / args.arm
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(trainer, episodes, masks, checkpoint_dir / "zero_step.pt")
    if not preflight["passed"]:
        raise RuntimeError(f"char preflight failed: {preflight}")
    # Rebuild identical params after the subprocess touched nothing (fresh RNG).
    torch.manual_seed(SEED)
    workspace = SequenceCharWorkspace(
        vocab, SequenceCharConfig(seed=SEED, copy_induction=induction)
    )
    trainer = SequenceCharTrainer(
        workspace, learning_rate=LEARNING_RATE, code_revision=f"r2-d7-{args.arm}-probe"
    )
    trainer.enable_copy_value_supervision(LAMBDA_COPY)
    trainer.set_episodes(episodes)

    initial = _evaluate(workspace, rows)
    trajectory = [{"epoch": 0, **_compact(initial)}]
    for epoch in range(EPOCHS):
        for start in range(0, len(episodes), MICROBATCH):
            batch = list(episodes[start : start + MICROBATCH])
            batch_masks = [masks[index] for index in range(start, start + len(batch))]
            trainer.train_step(batch, value_masks=batch_masks)
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            trajectory.append({"epoch": epoch + 1, **_compact(_evaluate(workspace, rows))})
    final = _evaluate(workspace, rows)
    elapsed = time.monotonic() - started
    trainer.save(checkpoint_dir / "epoch30.pt")

    supported_m1 = float(final["copy_supported_M1"]["value"])
    per_shape_final = {shape: values["value"] for shape, values in final["per_shape_exact"].items()}
    groups = _value_groups(workspace, rows)
    copy_intact = _copy_value_readout(workspace, rows)
    copy_misbound = _copy_value_readout(workspace, rows, misbind=True)
    bias_end = (
        float(workspace._parameters["copy_induce_bias"].detach().item()) if induction else None
    )
    loss_points = [point["mean_sequence_loss"] for point in trajectory[1:]]
    bounded_increases = all(
        later <= earlier * 1.15
        for earlier, later in zip(loss_points[:-1], loss_points[1:], strict=True)
    )
    traj_by_epoch = {item["epoch"]: item for item in trajectory}
    gate: dict[str, Any] = {
        "copy_supported_m1_ge_0_90": supported_m1 >= 0.90,
        "multibyte_m1_ge_0_90": float(groups["multibyte"]["value"]) >= 0.90,
        "singlebyte_m1_ge_0_90": float(groups["singlebyte"]["value"]) >= 0.90,
        "copy_value_prob_ge_0_90": float(copy_intact["mean_copy_value_prob"]) >= 0.90,
        "misbind_copy_prob_le_0_50": float(copy_misbound["mean_copy_value_prob"]) <= 0.50,
        "loss_increases_bounded_15pct": bounded_increases,
        "no_late_collapse_gt_0_10": traj_by_epoch[30]["copy_supported_M1"]
        >= traj_by_epoch[25]["copy_supported_M1"] - 0.10,
        "preflight_passed": bool(preflight["passed"]),
        "within_wall_cap": elapsed <= WALL_CAP_SECONDS,
    }
    passed = all(gate.values())
    report = {
        "format": "taiji-r2-d7-char-token-probe-v1",
        "version": 1,
        "contract": CONTRACT,
        "graph": "char-v1",
        "arm": args.arm,
        "copy_induction": induction,
        "lambda_copy_value": LAMBDA_COPY,
        "microbatch_size": MICROBATCH,
        "fixture": "v2-train",
        "corpus_digest": content_digest(
            [
                json.loads(line)
                for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        ),
        "vocab_size": int(vocab.size),
        "split_read": "train",
        "dev_read": False,
        "final_read": False,
        "seed": SEED,
        "epochs_frozen": EPOCHS,
        "frozen_learning_rate": LEARNING_RATE,
        "train_episodes": len(episodes),
        "elapsed_seconds": elapsed,
        "wall_cap_seconds": WALL_CAP_SECONDS,
        "parameter_count": workspace.parameter_count(),
        "copy_induce_bias_end": bias_end,
        "preflight": preflight,
        "final": {key: value for key, value in final.items() if key != "per_shape_exact"},
        "per_shape_exact": final["per_shape_exact"],
        "value_length_groups": groups,
        "trajectory": trajectory,
        "copy_value_readout_intact": copy_intact,
        "copy_value_readout_misbound": copy_misbound,
        "gate": gate,
        "outcome": "passed" if passed else "failed",
        "reading": (
            "R2-D7 H-Tok character-unit learnability probe (train-only); "
            "not a capability claim"
        ),
        "growth_admitted": False,
        "can_promote": False,
    }
    out_path = PROJECT_ROOT / OUT_DIR / f"r2_d7_probe_{args.arm}_20260918.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "outcome": report["outcome"],
                "copy_supported_M1": round(supported_m1, 4),
                "per_shape": {k: round(v["value"], 3) for k, v in final["per_shape_exact"].items()},
                "groups": groups,
                "copy_value_prob_intact": round(float(copy_intact["mean_copy_value_prob"]), 4),
                "copy_value_prob_misbound": round(float(copy_misbound["mean_copy_value_prob"]), 4),
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
