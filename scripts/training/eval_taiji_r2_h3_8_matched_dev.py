"""R2-H3.8 matched dev execution under the frozen preregistration.

Preregistration: plans/reference/M5_R2_H3_8_MATCHED_DEV_PREREGISTRATION_FROZEN_20260917.md
(frozen).  Three seeds by two arms plus the workspace-lesion column, the frozen
numeric budget, and the frozen gates.  The final split is never read and no
promotion is implied by any outcome.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURE = Path("tests/fixtures/r2_h3_8_joint_sequence_v1.jsonl")
CORPUS_DIGEST = "5518ff500bbcb3551bc60cc13783442cd9fce870e8d17ee9dd4a7e165a17f6a8"
PREREGISTRATION = "plans/reference/M5_R2_H3_8_MATCHED_DEV_PREREGISTRATION_FROZEN_20260917.md"
DEFAULT_REPORT = Path("reports/r2_h3_8_matched_dev_20260917.json")
REPORT_FORMAT = "taiji-r2-h3-8-matched-dev-v1"
SEEDS = (20260917, 20260918, 20260919)
EPOCHS = 30
LEARNING_RATE = 0.01
CHECKPOINT_EPOCHS = (0, 15, 30)
WALL_CAP_SECONDS = 20 * 60.0
MIN_FREE_BYTES = 512 * 1024**2
MAX_CHECKPOINT_BYTES = 8 * 1024**2
GENERATION_LIMIT = 64
FIT_THRESHOLD = 0.70
WIN_MARGIN = 0.03
BOUNDARY_TOLERANCE = 0.05


def _load_split(split: str) -> tuple[tuple[bytes, bytes], ...]:
    episodes: list[tuple[bytes, bytes]] = []
    for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["split"] != split:
            continue
        episodes.append((record["prefix"].encode("utf-8"), record["response"].encode("utf-8")))
    if not episodes:
        raise RuntimeError(f"H3.8 {split} split is empty")
    return tuple(episodes)


def _teacher_forced(
    prototype: SequenceWorkspacePrototype, episodes: tuple[tuple[bytes, bytes], ...]
) -> dict[str, float]:
    positions = 0
    correct = 0
    losses: list[float] = []
    with torch.no_grad():
        for prefix, response in episodes:
            loss, metrics = prototype.sequence_loss(prefix, response)
            losses.append(float(loss))
            positions += int(metrics["positions"])
            correct += int(metrics["correct"])
    return {
        "accuracy": correct / max(1, positions),
        "positions": positions,
        "mean_loss": sum(losses) / len(losses),
    }


def _generation_metrics(
    prototype: SequenceWorkspacePrototype, episodes: tuple[tuple[bytes, bytes], ...]
) -> dict[str, float]:
    exact = 0
    boundary = 0
    for prefix, response in episodes:
        result = prototype.generate(prefix, max_bytes=GENERATION_LIMIT)
        boundary += int(result.stopped_on_boundary)
        exact += int(result.stopped_on_boundary and result.bytes_out == response)
    total = max(1, len(episodes))
    return {
        "exact_response_rate": exact / total,
        "boundary_stop_rate": boundary / total,
        "episodes": len(episodes),
    }


def _fresh_process_check(output_dir: Path, checkpoint: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.training.eval_taiji_r2_h3_8_matched_dev",
            "--verify",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fresh-process verification failed: {result.stderr[-400:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _verify_mode(checkpoint: Path, output_dir: Path) -> int:
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = SequenceWorkspaceTrainer.from_checkpoint(payload)
    digest = str(trainer.checkpoint()["checkpoint_digest"])
    probe = SequenceWorkspaceTrainer.from_checkpoint(payload)
    probe.train_step([_load_split("train")[0]])
    print(json.dumps({"digest": digest, "continued": str(probe.checkpoint()["checkpoint_digest"])}))
    return 0


def _run_arm_seed(
    arm: str,
    seed: int,
    train: tuple[tuple[bytes, bytes], ...],
    dev: tuple[tuple[bytes, bytes], ...],
    work_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    workspace_enabled = arm == "workspace"
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(seed=seed, workspace_enabled=workspace_enabled)
    )
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=LEARNING_RATE, code_revision="h38-matched-dev"
    )
    trainer.set_episodes(train)
    checkpoints: list[dict[str, Any]] = []
    for epoch in range(EPOCHS + 1):
        if epoch in CHECKPOINT_EPOCHS:
            path = trainer.save(work_dir / f"{arm}-{seed}-e{epoch}.pt")
            size = path.stat().st_size
            if size > MAX_CHECKPOINT_BYTES:
                raise RuntimeError(f"checkpoint exceeds the frozen storage cap: {size}")
            fresh = _fresh_process_check(work_dir, path)
            probe = SequenceWorkspaceTrainer.from_checkpoint(
                torch.load(path, map_location="cpu", weights_only=False)
            )
            probe.train_step([train[0]])
            if fresh["digest"] != str(trainer.checkpoint()["checkpoint_digest"]):
                raise RuntimeError("fresh-process digest mismatch")
            if fresh["continued"] != str(probe.checkpoint()["checkpoint_digest"]):
                raise RuntimeError("fresh-process continuation mismatch")
            checkpoints.append({"epoch": epoch, "bytes": size, "verified": True})
        if epoch == EPOCHS:
            break
        trainer.train_epoch()
        if time.perf_counter() - started > WALL_CAP_SECONDS:
            raise TimeoutError("arm exceeded the frozen wall cap")
    elapsed = time.perf_counter() - started
    train_fit = _teacher_forced(prototype, train)
    dev_forced = _teacher_forced(prototype, dev)
    dev_generation = _generation_metrics(prototype, dev)
    lesion = None
    if workspace_enabled:
        lesioned = SequenceWorkspacePrototype.from_parameter_payload(
            prototype.config, prototype.parameter_payload()
        )
        with torch.no_grad():
            lesioned.named_parameter("workspace_key").zero_()
            lesioned.named_parameter("workspace_value").zero_()
        lesion = {
            "dev_accuracy_unlesioned": dev_forced["accuracy"],
            "dev_accuracy_lesioned": _teacher_forced(lesioned, dev)["accuracy"],
        }
        lesion["drop"] = lesion["dev_accuracy_unlesioned"] - lesion["dev_accuracy_lesioned"]
    return {
        "arm": arm,
        "seed": seed,
        "parameter_count": prototype.parameter_count(),
        "elapsed_seconds": round(elapsed, 3),
        "checkpoints": checkpoints,
        "train_fit": train_fit,
        "dev": {
            **{f"forced_{key}": value for key, value in dev_forced.items()},
            **dev_generation,
        },
        "lesion": lesion,
        "final_checkpoint_digest": str(trainer.checkpoint()["checkpoint_digest"]),
    }


def _gates(results: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    fitted = {
        seed: all(
            results[seed][arm]["train_fit"]["accuracy"] >= FIT_THRESHOLD
            for arm in ("workspace", "baseline")
        )
        for seed in SEEDS
    }
    fit_seeds = [seed for seed in SEEDS if fitted[seed]]
    win = {
        seed: results[seed]["workspace"]["dev"]["forced_accuracy"]
        > results[seed]["baseline"]["dev"]["forced_accuracy"]
        for seed in SEEDS
    }
    boundary = {
        seed: results[seed]["workspace"]["dev"]["boundary_stop_rate"]
        >= results[seed]["baseline"]["dev"]["boundary_stop_rate"] - BOUNDARY_TOLERANCE
        for seed in SEEDS
    }
    lesion = {seed: float(results[seed]["workspace"]["lesion"]["drop"]) > 0.0 for seed in SEEDS}
    mean_workspace = statistics.mean(
        results[seed]["workspace"]["dev"]["forced_accuracy"] for seed in SEEDS
    )
    mean_baseline = statistics.mean(
        results[seed]["baseline"]["dev"]["forced_accuracy"] for seed in SEEDS
    )
    gates = {
        "g_fit_all_seeds": all(fitted.values()),
        "g_win_per_seed": all(win.values()),
        "g_win_mean_margin": (mean_workspace - mean_baseline) >= WIN_MARGIN,
        "g_boundary_non_regression": all(boundary.values()),
        "g_lesion_positive": all(lesion.values()),
    }
    return {
        "fit_per_seed": fitted,
        "fit_seeds": fit_seeds,
        "win_per_seed": win,
        "boundary_per_seed": boundary,
        "lesion_per_seed": lesion,
        "mean_dev_accuracy": {
            "workspace": mean_workspace,
            "baseline": mean_baseline,
            "margin": mean_workspace - mean_baseline,
        },
        "gates": gates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    if args.verify is not None:
        return _verify_mode(args.verify, args.output_dir or Path("."))

    torch.set_num_threads(1)
    digest = content_digest(
        [
            json.loads(line)
            for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines()
        ]
    )
    if digest != CORPUS_DIGEST:
        raise RuntimeError(f"H3.8 corpus digest mismatch: {digest}")
    if shutil.disk_usage(PROJECT_ROOT).free < MIN_FREE_BYTES:
        raise RuntimeError("insufficient free disk for the frozen storage cap")

    train = _load_split("train")
    dev = _load_split("dev")
    work_root = Path(tempfile.mkdtemp(prefix="h38-dev-"))
    results: dict[int, dict[str, dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for seed in SEEDS:
            results[seed] = {}
            for arm in ("workspace", "baseline"):
                work_dir = work_root / f"{arm}-{seed}"
                work_dir.mkdir(parents=True, exist_ok=True)
                try:
                    results[seed][arm] = _run_arm_seed(arm, seed, train, dev, work_dir)
                except Exception as error:  # noqa: BLE001
                    failures.append(
                        {"seed": seed, "arm": arm, "error": f"{type(error).__name__}: {error}"}
                    )
        gates = _gates(results) if all(len(item) == 2 for item in results.values()) else None
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    if gates is None:
        outcome = "failed"
    elif not gates["gates"]["g_fit_all_seeds"] and len(gates["fit_seeds"]) < 2:
        outcome = "computation_graph_return"
    elif not gates["gates"]["g_lesion_positive"]:
        outcome = "workspace_unused"
    elif gates["gates"]["g_win_per_seed"] and gates["gates"]["g_win_mean_margin"]:
        outcome = "workspace_benefit_supported"
    else:
        outcome = "no_workspace_benefit"

    payload = {
        "format": REPORT_FORMAT,
        "version": 1,
        "preregistration": PREREGISTRATION,
        "corpus_digest": digest,
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "final_read": False,
        "results": {str(seed): results[seed] for seed in results},
        "gates": gates,
        "failures": failures,
        "outcome": outcome,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "growth_admitted": False,
        "can_promote": False,
    }
    report_path = PROJECT_ROOT / args.report
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "outcome": outcome,
                "gates": gates["gates"] if gates else None,
                "mean": gates["mean_dev_accuracy"] if gates else None,
                "failures": failures,
            },
            indent=2,
        )
    )
    return 0 if outcome == "workspace_benefit_supported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
