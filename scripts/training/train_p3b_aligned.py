"""P3b trainer: continue ``seed_beta.pt`` on one of two arms (P3b 预注册 §2 / §2.2).

The arms differ **only** in which rows of the same unseen source window they read
(``build_p3b_arm_corpus.py --rule dialogue`` vs ``--rule all``); see
``M5_P3B_NOVELTY_MATCHED_ARMS_AMENDMENT_20260915.md`` for why both are sliced past the
11,199,800 symbols this start checkpoint had already consumed.

Why this file exists instead of calling ``train_seed_corpus.py --resume``: no ``--scale``
reproduces the stored v8 profile of ``seed_beta.pt`` (``predictive_context_fan_in 12 vs 24``,
``memory_time_dim 8 vs 16`` ...), so ``Taiji.restore``'s architecture-equality guard rejects a
CLI-resumed run.  The config is therefore rebuilt **from the checkpoint envelope itself**, which
is also what keeps the before/after comparison of P3b the same model.

Discipline inherited from the preregistration (§2/§5, §4 stop conditions):
* the objective function is untouched -- byte-level next-symbol ``observe(..., learn=True)``;
* no architecture, loader or binder change -- the legacy guard is patched **in process only**;
* the product entry checkpoint and the start checkpoint are protected from being overwritten;
* the budget tier comes from the frozen throughput calibration, not from a new number;
* ``--arm`` is **required, with no default**: an omitted arm used to resolve to ``treatment``,
  whose checkpoint, progress log and run report are the live campaign's own files.

Usage::

    python -X utf8 -u scripts/training/train_p3b_aligned.py --arm treatment --budget-tier 48h
    python -X utf8 -u scripts/training/train_p3b_aligned.py --arm control --budget-tier 48h
    python -X utf8 -u scripts/training/train_p3b_aligned.py --arm treatment \\
        --budget-tier custom --max-symbols 12000 --checkpoint-every 6000   # plumbing smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_seed_corpus import run_training  # noqa: E402
from seed import SeedConfig  # noqa: E402

START_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_beta.pt"
#: Both arm corpora are sliced from the **same unseen source window**: the start checkpoint had
#: already consumed the first 11,199,800 symbols of ``simple_zh_texts.jsonl``, so a stream that
#: begins at row 0 spends ~23% of its budget replaying data the state has seen (treatment 22.27%,
#: control 23.69%).  Slicing both arms to the same window leaves row selection as the only
#: difference.  See M5_P3B_NOVELTY_MATCHED_ARMS_AMENDMENT_20260915.md.
SUBSET_PATH = PROJECT_ROOT / "data" / "p3b_dialogue_fresh.jsonl"
CONTROL_CORPUS = PROJECT_ROOT / "data" / "p3b_all_fresh.jsonl"
SUBSET_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "p3b_dialogue_fresh_manifest.json"
CONTROL_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "p3b_all_fresh_manifest.json"
#: Both arms are manifest-bound: a corpus that drifted from its manifest means the arm read
#: something other than the registered stream, and treatment minus control stops being an effect.
ARM_MANIFESTS: dict[Path, Path] = {SUBSET_PATH: SUBSET_MANIFEST, CONTROL_CORPUS: CONTROL_MANIFEST}
CALIBRATION_REPORT = PROJECT_ROOT / "reports" / "taiji_p3b_throughput_calibration_20260915.json"
PREREG_PATH = "plans/reference/M5_P3B_ALIGNED_LANGUAGE_TRAINING_PREREGISTRATION_20260915.md"
WORKING_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "p3b" / "seed_aligned.pt"
PROGRESS_PATH = PROJECT_ROOT / "reports" / "p3b_aligned_progress.jsonl"
RUN_REPORT = PROJECT_ROOT / "reports" / "taiji_p3b_training_run_20260915.json"


def arm_paths(arm: str) -> tuple[Path, Path, Path]:
    """Checkpoint / progress / report locations for one arm (arms never share files)."""

    stem = "seed_aligned" if arm == "treatment" else f"seed_aligned_{arm}"
    return (
        PROJECT_ROOT / "checkpoints" / "p3b" / f"{stem}.pt",
        PROJECT_ROOT / "reports" / f"p3b_aligned_progress_{arm}.jsonl",
        PROJECT_ROOT / "reports" / f"taiji_p3b_training_run_{arm}_20260915.json",
    )


#: Checkpoints the run must never write to: the product default entry and the start state.
PROTECTED_OUTPUTS = (
    PROJECT_ROOT / "checkpoints" / "seed_corpus.pt",
    PROJECT_ROOT / "checkpoints" / "seed_beta.pt",
    PROJECT_ROOT / "checkpoints" / "resumed_seed_corpus.pt",
    PROJECT_ROOT / "checkpoints" / "seed_corpus_prev_20260823.pt",
)

#: §2.1 frozen throughput: 273.7 steps/s.  Tiers are symbol counts, derived once and frozen.
BUDGET_TIERS = {"16h": 15_760_000, "48h": 47_280_000}
CALIBRATED_STEPS_PER_SECOND = 273.7
DEFAULT_CHECKPOINT_EVERY = 1_000_000  # ~1.01 h per stage at the calibrated throughput
DEFAULT_PROGRESS_EVERY = 50_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard_data_provenance(corpus: Path) -> dict[str, object]:
    """Fail closed on either arm's corpus; only an ad-hoc file is accepted unbound."""

    if not corpus.exists():
        raise SystemExit(f"P3b corpus missing: {corpus}")
    actual = _sha256(corpus)
    resolved = corpus.resolve()
    binding = "none (ad-hoc corpus: neither arm's registered stream)"
    for arm_path, manifest_path in ARM_MANIFESTS.items():
        if resolved != arm_path.resolve():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = str(manifest.get("output_sha256", ""))
        if expected and actual != expected:
            raise SystemExit(
                f"P3b corpus {arm_path.name} sha256 drifted from {manifest_path.name} "
                f"({actual[:12]}… != {expected[:12]}…); rebuild it with build_p3b_arm_corpus.py "
                "before spending any budget, or treatment minus control is not the registered contrast"
            )
        binding = str(_relative(manifest_path))
    return {
        "path": str(_relative(corpus)),
        "bytes": corpus.stat().st_size,
        "sha256": actual,
        "manifest_binding": binding,
    }


def _guard_budget_tier(tier: str, symbols: int) -> dict[str, object]:
    calibration = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    measured = float(calibration.get("steps_per_second", calibration.get("throughput", 0.0)))
    if abs(measured - CALIBRATED_STEPS_PER_SECOND) > 1.0:
        raise SystemExit(
            f"throughput calibration file says {measured} steps/s, this runner is frozen to "
            f"{CALIBRATED_STEPS_PER_SECOND}; re-calibrate and update BUDGET_TIERS together"
        )
    return {
        "budget_tier": tier,
        "max_symbols": int(symbols),
        "calibrated_steps_per_second": measured,
        "estimated_hours": round(symbols / measured / 3600.0, 2),
        "calibration_report": str(CALIBRATION_REPORT.relative_to(PROJECT_ROOT)),
    }


def _refuse_protected(path: Path) -> Path:
    resolved = path.resolve()
    if resolved in {p.resolve() for p in PROTECTED_OUTPUTS}:
        raise SystemExit(f"refusing to overwrite a protected checkpoint: {resolved}")
    return resolved


def _install_legacy_guard() -> dict[str, object]:
    """Relax the identity-organ guard in this process only (same patch P3a evaluated with)."""

    from scripts.training.probe_taiji_cap0_legacy_load import _install_legacy_guard as install

    install()
    return {
        "relax_legacy_guard": True,
        "constrained_decode": False,
        "scope": "in-process only; taiji/model.py and the product loader are untouched",
    }


def build_config_from_checkpoint(checkpoint: Path) -> tuple[SeedConfig, dict[str, object]]:
    envelope = torch.load(checkpoint, weights_only=False)
    if not isinstance(envelope, dict) or "config" not in envelope:
        raise SystemExit(f"{checkpoint} is not a seed envelope (no 'config' key)")
    config = SeedConfig.from_dict(dict(envelope["config"]))
    meta = envelope.get("metadata") or {}
    substrate = envelope.get("substrate") or {}
    return config, {
        "start_checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "start_tick": int(meta.get("tick", 0)),
        "start_substrate_format": str(substrate.get("format", "")),
        "config_rebuilt_from_envelope": True,
        "identity_organ_enabled": bool(getattr(config.taiji, "identity_organ_enabled", False)),
    }


def train(
    *,
    arm: str,
    corpus: Path,
    checkpoint_path: Path,
    progress_path: Path,
    run_report: Path,
    max_symbols: int,
    checkpoint_every: int,
    progress_every: int,
    resume_from: Path | None,
    device: str,
    budget_meta: dict[str, object],
    data_meta: dict[str, object],
) -> dict[str, object]:
    config, start = build_config_from_checkpoint(resume_from or START_CHECKPOINT)
    guard = _install_legacy_guard()
    started = time.perf_counter()
    print(
        json.dumps(
            {
                "event": "p3b_train_start",
                "arm": arm,
                "start_tick": start["start_tick"],
                "max_symbols": max_symbols,
                "checkpoint_every": checkpoint_every,
                "device": device,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    summary = run_training(
        corpus_paths=[corpus],
        config=config,
        epochs=1,
        checkpoint_path=checkpoint_path,
        progress_path=progress_path,
        checkpoint_every=checkpoint_every,
        progress_every=progress_every,
        max_symbols=max_symbols,
        resume_checkpoint=resume_from or START_CHECKPOINT,
        device=device,
    )
    record: dict[str, object] = {
        "format": "taiji-p3b-training-run-v1",
        "status": "completed",
        "arm": arm,
        "preregistration": PREREG_PATH,
        "objective_function_unchanged": "observe(symbol, learn=True)  # byte-level next symbol",
        "architecture_changed": False,
        "loader_changed": False,
        "chain": guard,
        "start": start,
        "data": data_meta,
        "budget": budget_meta,
        "checkpoint_written": str(_relative(checkpoint_path)),
        "progress_log": str(_relative(progress_path)),
        "summary": {key: float(value) for key, value in summary.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "does_not_do": [
            "no score is produced here; capability scoring is eval_taiji_cap0_baseline.py",
            "no claim about collaboration or promotion; growth_admitted/can_promote untouched",
        ],
    }
    run_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = run_report.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(run_report)
    print(
        json.dumps(
            {"event": "p3b_train_done", "arm": arm, "ticks": summary.get("ticks")},
            ensure_ascii=True,
        ),
        flush=True,
    )
    return record


def _relative(path: Path) -> Path | str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:  # pragma: no cover - scratch paths during smoke tests
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3b 对齐语言训练（不改架构/损失/加载器）")
    parser.add_argument("--budget-tier", choices=(*BUDGET_TIERS, "custom"), default="48h")
    parser.add_argument(
        "--max-symbols", type=int, default=None, help="only with --budget-tier custom"
    )
    parser.add_argument(
        "--arm",
        choices=("treatment", "control"),
        required=True,
        help="required: a defaulted arm would silently write the other arm's checkpoint and logs",
    )
    parser.add_argument("--corpus", type=Path, default=None, help="defaults by --arm")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None, help="continue a p3b checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.budget_tier == "custom":
        if not args.max_symbols or args.max_symbols <= 0:
            parser.error("--max-symbols is required with --budget-tier custom")
        symbols = int(args.max_symbols)
        if symbols > 200_000:
            print(
                "note: a custom budget above 200k symbols is beyond the plumbing-smoke range "
                "and is recorded as an exploratory run",
                flush=True,
            )
    else:
        if args.max_symbols:
            parser.error("--max-symbols only applies with --budget-tier custom")
        symbols = BUDGET_TIERS[args.budget_tier]

    default_checkpoint, progress_path, run_report = arm_paths(args.arm)
    corpus = Path(args.corpus or (SUBSET_PATH if args.arm == "treatment" else CONTROL_CORPUS))
    output = _refuse_protected(Path(args.checkpoint or default_checkpoint))
    data_meta = _guard_data_provenance(corpus)
    budget_meta = _guard_budget_tier(args.budget_tier, int(symbols))
    resume_from = None
    if args.resume_from is not None:
        resume_from = _refuse_protected(Path(args.resume_from))
        if not resume_from.exists():
            parser.error(f"--resume-from does not exist: {resume_from}")
    train(
        arm=args.arm,
        corpus=corpus,
        checkpoint_path=output,
        progress_path=progress_path,
        run_report=run_report,
        max_symbols=int(symbols),
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        resume_from=resume_from,
        device=args.device,
        budget_meta=budget_meta,
        data_meta=data_meta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
