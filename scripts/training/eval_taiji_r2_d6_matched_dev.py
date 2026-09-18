"""R2-D6 matched-dev adjudication for H-B induction copy (contract section 5).

Arm B2 (fixture v2 train + graph v8 induction + lambda=1.0) trains over three
seeds and is evaluated on the sealed v1 dev split with the D3/D4 G1-G6 gates
plus the contracted key readouts K1 (multi-byte full rate), K2 (M4 flip) and
K3 (coverage-harm reversal).  Two historical baselines are referenced only
after bitwise drift checks proving the v8 code still reproduces them: A00 (the
D4 T2 run, v1 train) and A01 (the D5 coverage-only run, v2 train, no
induction).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from scripts.training.eval_taiji_r2_d1_surface_policies import metrics  # noqa: E402
from scripts.training.eval_taiji_r2_d5_matched_dev import (  # noqa: E402
    _evaluate,
    _load,
)
from scripts.training.probe_taiji_r2_d4_copy_supervision import value_mask_for  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    EVIDENCE_PER_POSITION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURES = {
    "v1": Path("tests/fixtures/r2_d1_measurement_v1.jsonl"),
    "v2": Path("tests/fixtures/r2_d1_measurement_v2.jsonl"),
}
V1_CORPUS_DIGEST = "53ac9f88695f135d0bb04b4d25d98ab686c82175c45ca8d4d53e64639efbeecb"
OUT_REPORT = Path("reports/r2_d6_matched_dev_20260918.json")
CHECKPOINT_ROOT = Path("reports/r2_d6_checkpoints/matched")
D4_CHECKPOINTS = Path("reports/r2_d4_checkpoints/matched")
D5_CHECKPOINTS = Path("reports/r2_d5_checkpoints/matched")
D4_REPORT = Path("reports/r2_d4_matched_dev_20260918.json")
SEEDS = (20260917, 20260918, 20260919)
EPOCHS = 30
LR = 0.01
MICROBATCH = 8
LAMBDA_COPY = 1.0
SURFACE_M1 = 0.1735
SURFACE_M3 = 0.0581
SURFACE_M4 = 0.0


def _build(
    fixture: str, induction: bool, seed: int, revision: str
) -> tuple[SequenceWorkspacePrototype, SequenceWorkspaceTrainer]:
    torch.manual_seed(seed)
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=seed,
            evidence_source=EVIDENCE_PER_POSITION,
            copy_mixture=True,
            question_conditioned_start=True,
            readout_heads=4,
            positional_keys=True,
            copy_induction=induction,
        )
    )
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=LR, code_revision=revision)
    trainer.enable_copy_value_supervision(LAMBDA_COPY)
    return prototype, trainer


def _run_training(
    fixture: str,
    induction: bool,
    seed: int,
    revision: str,
    train_rows: list[dict[str, Any]],
) -> SequenceWorkspacePrototype:
    episodes = tuple(
        (row["prefix"].encode("utf-8"), row["response"].encode("utf-8")) for row in train_rows
    )
    masks = tuple(value_mask_for(row["shape"], row["response"]) for row in train_rows)
    prototype, trainer = _build(fixture, induction, seed, revision)
    trainer.set_episodes(episodes)
    for _ in range(EPOCHS):
        for start in range(0, len(episodes), MICROBATCH):
            batch = list(episodes[start : start + MICROBATCH])
            batch_masks = [masks[index] for index in range(start, start + len(batch))]
            trainer.train_step(batch, value_masks=batch_masks)
    return prototype


def _drift_check(
    fixture: str,
    train_rows: list[dict[str, Any]],
    frozen_path: Path,
    revision: str,
) -> dict[str, Any]:
    """v8 code with the flag off must reproduce the frozen baseline bitwise."""

    prototype = _run_training(fixture, False, SEEDS[0], revision, train_rows)
    reproduced = prototype.parameter_payload()
    frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)["parameters"]
    max_diff = 0.0
    for name, tensor in reproduced.items():
        max_diff = max(max_diff, (tensor.float() - frozen[name].float()).abs().max().item())
    return {
        "seed": SEEDS[0],
        "frozen_checkpoint": str(frozen_path),
        "bitwise_identical": max_diff == 0.0,
        "max_abs_diff": max_diff,
    }


def main() -> int:
    raw1 = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURES["v1"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if content_digest(raw1) != V1_CORPUS_DIGEST:
        raise RuntimeError("R2-D1 v1 corpus digest mismatch")
    dev_rows = _load("v1", "dev")
    train_by_fixture = {
        "v1": _load("v1", "train"),
        "v2": _load("v2", "train"),
    }

    a00_drift = _drift_check(
        "v1",
        train_by_fixture["v1"],
        PROJECT_ROOT / D4_CHECKPOINTS / str(SEEDS[0]) / "epoch30.pt",
        "r2-d6-drift-a00",
    )
    a01_drift = _drift_check(
        "v2",
        train_by_fixture["v2"],
        PROJECT_ROOT / D5_CHECKPOINTS / "A01_coverage" / str(SEEDS[0]) / "epoch30.pt",
        "r2-d6-drift-a01",
    )
    if not (a00_drift["bitwise_identical"] and a01_drift["bitwise_identical"]):
        raise RuntimeError(f"historical drift checks failed: {a00_drift} / {a01_drift}")

    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        started = time.monotonic()
        prototype = _run_training(
            "v2", True, seed, f"r2-d6-b2-{seed}", train_by_fixture["v2"]
        )
        target_dir = PROJECT_ROOT / CHECKPOINT_ROOT / str(seed)
        target_dir.mkdir(parents=True, exist_ok=True)
        trainer = SequenceWorkspaceTrainer(
            prototype, learning_rate=LR, code_revision=f"r2-d6-b2-{seed}"
        )
        trainer.save(target_dir / "epoch30.pt")
        evaluation = _evaluate(prototype, dev_rows)
        runs.append(
            {
                "seed": seed,
                "parameter_count": prototype.parameter_count(),
                "copy_induce_bias_end": float(
                    prototype._parameters["copy_induce_bias"].detach().item()
                ),
                "elapsed_seconds": time.monotonic() - started,
                "eval": evaluation,
            }
        )

    d4_report = json.loads((PROJECT_ROOT / D4_REPORT).read_text(encoding="utf-8"))
    a00 = {run["seed"]: run for run in d4_report["runs"]["T2_copy_value_supervision"]}

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    m3 = [float(r["eval"]["full"]["M3_content_exact"]["value"]) for r in runs]
    m4 = [float(r["eval"]["full"]["M4_flip_pair"]["value"]) for r in runs]
    m1 = [float(r["eval"]["full"]["M1_exact"]["value"]) for r in runs]
    m3_nc = [float(r["eval"]["no_context"]["M3_content_exact"]["value"]) for r in runs]
    m1_nc = [float(r["eval"]["no_context"]["M1_exact"]["value"]) for r in runs]
    m4_mis = [float(r["eval"]["copy_misbind"]["M4_flip_pair"]["value"]) for r in runs]
    base_m3 = [float(a00[seed]["eval"]["full"]["M3_content_exact"]["value"]) for seed in SEEDS]
    base_m4 = [float(a00[seed]["eval"]["full"]["M4_flip_pair"]["value"]) for seed in SEEDS]
    delta_m3 = [a - b for a, b in zip(m3, base_m3, strict=True)]
    delta_m4 = [a - b for a, b in zip(m4, base_m4, strict=True)]
    context = [a - b for a, b in zip(m3, m3_nc, strict=True)]
    mis_drop = [a - b for a, b in zip(m4, m4_mis, strict=True)]
    fact_flip = [float(r["eval"]["pair_rates"]["fact_flip"]["value"]) for r in runs]
    combo_flip = [float(r["eval"]["pair_rates"]["combo_flip"]["value"]) for r in runs]
    boundary = [r["eval"]["boundary_stop_rate"] for r in runs]
    completion = [r["eval"]["multibyte_completion"] for r in runs]
    completion_mean = {
        "first_byte_rate": mean([c["first_byte_rate"] for c in completion]),
        "full_rate": mean([c["full_rate"] for c in completion]),
        "mean_byte_completion": mean([c["mean_byte_completion"] for c in completion]),
    }
    gates = {
        "G1_m3_ceiling_and_delta": mean(m3) >= 0.26 and mean(delta_m3) >= 0.20,
        "G2_m4_flip_fact_subset": (
            mean(m4) >= 0.50 and mean(delta_m4) >= 0.25 and mean(fact_flip) >= 2 / 6
        ),
        "G3_copy_misbind_drop": mean(mis_drop) >= 0.50,
        "G4_context_margin": mean(context) >= 0.30 and max(m1_nc) <= 0.50,
        "G5_seed_consistency": sum(1 for d in delta_m4 if d > 0) >= 2 and min(delta_m4) >= 0,
        "G6_boundary_and_restore": min(boundary) >= 0.95,
    }
    keys = {
        "K1_multibyte_full_rate_ge_0_30": completion_mean["full_rate"] >= 0.30,
        "K2_dev_m4_flip_ge_0_50": mean(m4) >= 0.50,
        "K3_beats_coverage_only_by_10x": mean(m3) > 0.008 * 10,
    }
    passed = all(gates.values()) and keys["K1_multibyte_full_rate_ge_0_30"]
    report = {
        "format": "taiji-r2-d6-matched-dev-v1",
        "version": 1,
        "contract": "plans/reference/M5_R2_D6_INDUCTION_COPY_CONTRACT_FROZEN_20260918.md",
        "seeds": list(SEEDS),
        "arm": "B2 (v2 train + graph v8 induction + lambda 1.0)",
        "lambda_copy_value": LAMBDA_COPY,
        "drift_checks": {"A00_d4_t2": a00_drift, "A01_d5_coverage": a01_drift},
        "a00_baseline": {
            "source": str(D4_REPORT),
            "m3_mean": mean(base_m3),
            "m4_mean": mean(base_m4),
        },
        "a01_reference": {
            "source": "reports/r2_d5_matched_dev_20260918.json",
            "m3_mean": 0.007751937984496124,
        },
        "runs": {"B2_induction_multibyte": runs},
        "gates": gates,
        "keys": keys,
        "metrics_mean": {
            "m1": mean(m1),
            "m3": mean(m3),
            "m4": mean(m4),
            "delta_m3_vs_a00": mean(delta_m3),
            "fact_flip": mean(fact_flip),
            "combo_flip": mean(combo_flip),
            "multibyte_completion": completion_mean,
            "bias_end": mean([r["copy_induce_bias_end"] for r in runs]),
        },
        "outcome": "passed" if passed else "failed",
        "growth_admitted": False,
        "can_promote": False,
    }
    (PROJECT_ROOT / OUT_REPORT).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "drift_a00": a00_drift["bitwise_identical"],
                "drift_a01": a01_drift["bitwise_identical"],
                "outcome": report["outcome"],
                "gates": gates,
                "keys": keys,
                "metrics_mean": report["metrics_mean"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
