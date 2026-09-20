"""R2-D5 matched-dev 2x2 adjudication (contract section 5).

Three new arms (A01 coverage-only, A10 persistence-only, A11 both) train on
their contracted fixtures over three seeds; the A00 baseline is the frozen D4
T2 history, referenced only after a one-seed bitwise drift check.  Gates G1-G6
use the D4 contract wording with delta baselines against A00, per arm.  The
decisive descriptive readout is the per-byte completion rate of multi-byte dev
answers (first-byte policy vs full copy) for every arm.
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

from scripts.training.build_taiji_r2_d1_measurement_fixture import (  # noqa: E402
    remove_material_clause,
)
from scripts.training.eval_taiji_r2_d1_surface_policies import metrics  # noqa: E402
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
OUT_REPORT = Path("reports/r2_d5_matched_dev_20260918.json")
CHECKPOINT_ROOT = Path("reports/r2_d5_checkpoints/matched")
D4_REPORT = Path("reports/r2_d4_matched_dev_20260918.json")
D4_CHECKPOINT_DIR = Path("reports/r2_d4_checkpoints/matched")
SEEDS = (20260917, 20260918, 20260919)
EPOCHS = 30
LR = 0.01
MICROBATCH = 8
LAMBDA_COPY = 1.0
SURFACE_M1 = 0.1735
SURFACE_M3 = 0.0581
SURFACE_M4 = 0.0
ARMS = {
    # arm -> (fixture, copy_persistence)
    "A01_coverage": ("v2", False),
    "A10_persistence": ("v1", True),
    "A11_both": ("v2", True),
}
COPY_SUPPORTED_SHAPES = ("fact", "negation", "same_opening_fact")


def _load(fixture: str, split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURES[fixture]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row["split"] == split]


def _rows_for_scoring(
    prototype: SequenceWorkspacePrototype,
    rows: list[dict[str, Any]],
    *,
    condition: str,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        if condition == "full":
            prefix = row["prefix"]
            kwargs: dict[str, Any] = {}
        elif condition == "no_context":
            prefix = remove_material_clause(row["prefix"])
            kwargs = {}
        elif condition == "zero_read":
            prefix = row["prefix"]
            kwargs = {"zero_read": True}
        elif condition == "copy_misbind":
            prefix = row["prefix"]
            kwargs = {"entry_rotation": len(row["prefix"]) // 2}
        elif condition == "value_misbind":
            prefix = row["prefix"]
            kwargs = {"value_rotation": len(row["prefix"]) // 2}
        else:
            raise ValueError(f"unknown condition {condition}")
        generated = prototype.generate(prefix.encode("utf-8"), max_bytes=12, **kwargs)
        text = generated.bytes_out.decode("utf-8", errors="replace")
        scored.append(
            {
                "id": row["id"],
                "shape": row["shape"],
                "source_group": row["source_group"],
                "content_dependent": row["content_dependent"],
                "pair_id": row["pair_id"],
                "pair_type": row["pair_type"],
                "pair_role": row["pair_role"],
                "gold": row["response"],
                "prediction": text,
                "correct": text == row["response"],
            }
        )
    return scored


def _pair_type_rates(scored: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        by_type.setdefault(str(row["pair_type"]), []).append(row)
    for pair_type, members in sorted(by_type.items()):
        members_by_pair: dict[str, set[str]] = {}
        hits: dict[str, set[str]] = {}
        for row in members:
            members_by_pair.setdefault(str(row["pair_id"]), set()).add(row["id"])
            if row["correct"]:
                hits.setdefault(str(row["pair_id"]), set()).add(row["id"])
        complete_pairs = {pid for pid, ids in members_by_pair.items() if len(ids) == 2}
        pair_hits = sum(1 for pid in complete_pairs if hits.get(pid, set()) == members_by_pair[pid])
        out[pair_type] = {
            "pair_hits": pair_hits,
            "denominator": len(complete_pairs),
            "value": pair_hits / len(complete_pairs) if complete_pairs else 0.0,
        }
    return out


def _multibyte_completion(
    prototype: SequenceWorkspacePrototype, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-byte common-prefix completion on multi-byte-value dev episodes.

    ``first_byte_rate`` measures the "copy starts at the right row" policy;
    ``full_rate`` measures complete multi-byte copying.  D4's signature failure
    is first-byte ≈ 1 with full ≈ 0.
    """

    first_hits = 0
    full_hits = 0
    prefix_bytes = 0.0
    members = 0
    with torch.no_grad():
        for row in rows:
            if row["shape"] not in COPY_SUPPORTED_SHAPES:
                continue
            gold = row["response"]
            value = gold[2:] if row["shape"] == "negation" else gold
            if len(value.encode("utf-8")) <= 3:
                continue
            members += 1
            generated = prototype.generate(row["prefix"].encode("utf-8"), max_bytes=12)
            text = generated.bytes_out.decode("utf-8", errors="replace")
            gold_bytes = gold.encode("utf-8")
            got_bytes = text.encode("utf-8")
            common = 0
            while common < len(gold_bytes) and got_bytes[: common + 1] == gold_bytes[: common + 1]:
                common += 1
            prefix_bytes += common / len(gold_bytes)
            if got_bytes[:3] == gold_bytes[:3]:
                first_hits += 1
            if text == gold:
                full_hits += 1
    return {
        "episodes": members,
        "first_byte_rate": first_hits / members if members else 0.0,
        "full_rate": full_hits / members if members else 0.0,
        "mean_byte_completion": prefix_bytes / members if members else 0.0,
    }


def _evaluate(prototype: SequenceWorkspacePrototype, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in ("full", "no_context", "zero_read", "copy_misbind", "value_misbind"):
        out[condition] = metrics(_rows_for_scoring(prototype, rows, condition=condition))
    out["pair_rates"] = _pair_type_rates(_rows_for_scoring(prototype, rows, condition="full"))
    out["multibyte_completion"] = _multibyte_completion(prototype, rows)
    stops = 0
    for row in rows:
        result = prototype.generate(row["prefix"].encode("utf-8"), max_bytes=12)
        if result.stopped_on_boundary:
            stops += 1
    out["boundary_stop_rate"] = stops / len(rows)
    return out


def _build(
    fixture: str, persistence: bool, seed: int
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
            copy_persistence=persistence,
        )
    )
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=LR, code_revision=f"r2-d5-{seed}")
    trainer.enable_copy_value_supervision(LAMBDA_COPY)
    return prototype, trainer


def _train(
    arm: str,
    fixture: str,
    persistence: bool,
    seed: int,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.monotonic()
    episodes = tuple(
        (row["prefix"].encode("utf-8"), row["response"].encode("utf-8")) for row in train_rows
    )
    masks = tuple(value_mask_for(row["shape"], row["response"]) for row in train_rows)
    prototype, trainer = _build(fixture, persistence, seed)
    trainer.set_episodes(episodes)
    target_dir = PROJECT_ROOT / CHECKPOINT_ROOT / arm / str(seed)
    target_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(target_dir / "zero_step.pt")
    for _ in range(EPOCHS):
        for start in range(0, len(episodes), MICROBATCH):
            batch = list(episodes[start : start + MICROBATCH])
            batch_masks = [masks[index] for index in range(start, start + len(batch))]
            trainer.train_step(batch, value_masks=batch_masks)
    trainer.save(target_dir / "epoch30.pt")
    return {
        "seed": seed,
        "parameter_count": prototype.parameter_count(),
        "copy_persist_bias_end": (
            float(prototype._parameters["copy_persist_bias"].detach().item())
            if persistence
            else None
        ),
        "elapsed_seconds": time.monotonic() - started,
        "eval": _evaluate(prototype, dev_rows),
    }


def _verify_a00_no_drift(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A00 = D4 T2 history; retrain seed 20260917 (v1 + v6 + lambda=1) and
    demand bit-identical parameters against the frozen D4 checkpoint."""

    prototype, trainer = _build("v1", False, SEEDS[0])
    episodes = tuple(
        (row["prefix"].encode("utf-8"), row["response"].encode("utf-8")) for row in train_rows
    )
    masks = tuple(value_mask_for(row["shape"], row["response"]) for row in train_rows)
    trainer.set_episodes(episodes)
    for _ in range(EPOCHS):
        for start in range(0, len(episodes), MICROBATCH):
            batch = list(episodes[start : start + MICROBATCH])
            batch_masks = [masks[index] for index in range(start, start + len(batch))]
            trainer.train_step(batch, value_masks=batch_masks)
    reproduced = trainer.prototype.parameter_payload()
    frozen = torch.load(
        PROJECT_ROOT / D4_CHECKPOINT_DIR / str(SEEDS[0]) / "epoch30.pt",
        map_location="cpu",
        weights_only=False,
    )["parameters"]
    max_diff = 0.0
    for name, tensor in reproduced.items():
        max_diff = max(max_diff, (tensor.float() - frozen[name].float()).abs().max().item())
    return {
        "seed": SEEDS[0],
        "frozen_source": str(D4_CHECKPOINT_DIR / str(SEEDS[0]) / "epoch30.pt"),
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

    drift = _verify_a00_no_drift(train_by_fixture["v1"])
    if not drift["bitwise_identical"]:
        raise RuntimeError(f"A00 drift check failed: {drift}")

    d4_report = json.loads((PROJECT_ROOT / D4_REPORT).read_text(encoding="utf-8"))
    a00 = {run["seed"]: run for run in d4_report["runs"]["T2_copy_value_supervision"]}

    runs: dict[str, list[dict[str, Any]]] = {}
    for arm, (fixture, persistence) in ARMS.items():
        runs[arm] = [
            _train(arm, fixture, persistence, seed, train_by_fixture[fixture], dev_rows)
            for seed in SEEDS
        ]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def arm_metric(arm: str, condition: str, metric_key: str) -> list[float]:
        return [float(run["eval"][condition][metric_key]["value"]) for run in runs[arm]]

    def base_metric(condition: str, metric_key: str) -> list[float]:
        return [float(a00[seed]["eval"][condition][metric_key]["value"]) for seed in SEEDS]

    adjudication: dict[str, Any] = {}
    for arm in ARMS:
        m3 = arm_metric(arm, "full", "M3_content_exact")
        m4 = arm_metric(arm, "full", "M4_flip_pair")
        m1 = arm_metric(arm, "full", "M1_exact")
        m3_nc = arm_metric(arm, "no_context", "M3_content_exact")
        m1_nc = arm_metric(arm, "no_context", "M1_exact")
        m4_mis = arm_metric(arm, "copy_misbind", "M4_flip_pair")
        delta_m3 = [a - b for a, b in zip(m3, base_metric("full", "M3_content_exact"), strict=True)]
        delta_m4 = [a - b for a, b in zip(m4, base_metric("full", "M4_flip_pair"), strict=True)]
        context = [a - b for a, b in zip(m3, m3_nc, strict=True)]
        mis_drop = [a - b for a, b in zip(m4, m4_mis, strict=True)]
        fact_flip = [float(r["eval"]["pair_rates"]["fact_flip"]["value"]) for r in runs[arm]]
        combo_flip = [float(r["eval"]["pair_rates"]["combo_flip"]["value"]) for r in runs[arm]]
        boundary = [r["eval"]["boundary_stop_rate"] for r in runs[arm]]
        completion = [r["eval"]["multibyte_completion"] for r in runs[arm]]
        gates = {
            "G1_m3_ceiling_and_delta": mean(m3) >= 0.26 and mean(delta_m3) >= 0.20,
            "G2_m4_flip_fact_subset": (
                mean(m4) >= 0.50 and mean(delta_m4) >= 0.25 and mean(fact_flip) >= 2 / 6
            ),
            "G3_copy_misbind_drop": mean(mis_drop) >= 0.50,
            "G4_context_margin": mean(context) >= 0.30 and max(m1_nc) <= 0.50,
            "G5_seed_consistency": (sum(1 for d in delta_m4 if d > 0) >= 2 and min(delta_m4) >= 0),
            "G6_boundary_and_restore": min(boundary) >= 0.95,
        }
        passed = all(gates.values())
        adjudication[arm] = {
            "passed": passed,
            "gates": gates,
            "m1_mean": mean(m1),
            "m3_mean": mean(m3),
            "m4_mean": mean(m4),
            "delta_m3_mean": mean(delta_m3),
            "delta_m4_mean": mean(delta_m4),
            "fact_flip_mean": mean(fact_flip),
            "combo_flip_mean": mean(combo_flip),
            "multibyte_completion_mean": {
                "first_byte_rate": mean([c["first_byte_rate"] for c in completion]),
                "full_rate": mean([c["full_rate"] for c in completion]),
                "mean_byte_completion": mean([c["mean_byte_completion"] for c in completion]),
            },
        }
    a00_per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        payload = torch.load(
            PROJECT_ROOT / D4_CHECKPOINT_DIR / str(seed) / "epoch30.pt",
            map_location="cpu",
            weights_only=False,
        )
        frozen = SequenceWorkspaceTrainer.from_checkpoint(payload).prototype
        a00_per_seed.append(_multibyte_completion(frozen, dev_rows))
    a00_completion: dict[str, Any] = {
        "first_byte_rate": mean([c["first_byte_rate"] for c in a00_per_seed]),
        "full_rate": mean([c["full_rate"] for c in a00_per_seed]),
        "mean_byte_completion": mean([c["mean_byte_completion"] for c in a00_per_seed]),
        "per_seed": a00_per_seed,
    }
    summary: dict[str, Any] = {
        "format": "taiji-r2-d5-matched-dev-v1",
        "version": 1,
        "contract": "plans/reference/M5_R2_D5_MULTIBYTE_FACTORIAL_CONTRACT_FROZEN_20260918.md",
        "seeds": list(SEEDS),
        "arms": {arm: {"fixture": fx, "copy_persistence": p} for arm, (fx, p) in ARMS.items()},
        "a00_baseline": {
            "source": str(D4_REPORT),
            "drift_check": drift,
            "m3_mean": mean(base_metric("full", "M3_content_exact")),
            "m4_mean": mean(base_metric("full", "M4_flip_pair")),
            "multibyte_completion_mean": {
                "first_byte_rate": a00_completion["first_byte_rate"],
                "full_rate": a00_completion["full_rate"],
                "mean_byte_completion": a00_completion["mean_byte_completion"],
            },
        },
        "runs": runs,
        "adjudication": adjudication,
        "outcome": {arm: data["passed"] for arm, data in adjudication.items()},
        "growth_admitted": False,
        "can_promote": False,
    }
    (PROJECT_ROOT / OUT_REPORT).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "drift_ok": drift["bitwise_identical"],
                "a00_m3_mean": summary["a00_baseline"]["m3_mean"],
                "a00_completion": {
                    "first_byte_rate": a00_completion["first_byte_rate"],
                    "full_rate": a00_completion["full_rate"],
                },
                "adjudication": adjudication,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if any(data["passed"] for data in adjudication.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
