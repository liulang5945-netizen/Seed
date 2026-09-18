"""R2-D3 matched-dev evaluation for graph v6 H-G (contract section 4, G1-G6).

Trains three arms over three seeds (micro-batch 8, fixed order, 30 epochs,
lr 0.01), then evaluates the sealed R2-D1 dev split (never read during
training or probe selection) under four conditions: full, material-removed
(same checkpoint), zero-read ablation, and copy-key misalignment.  Produces
M1-M5 per run, the G1-G6 gate adjudication and per-seed/worst-seed stats.
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
from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    EVIDENCE_BROADCAST_FINAL,
    EVIDENCE_PER_POSITION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURE = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
CORPUS_DIGEST = "53ac9f88695f135d0bb04b4d25d98ab686c82175c45ca8d4d53e64639efbeecb"
OUT_REPORT = Path("reports/r2_d3_matched_dev_20260918.json")
CHECKPOINT_ROOT = Path("reports/r2_d3_checkpoints/matched")
SEEDS = (20260917, 20260918, 20260919)
EPOCHS = 30
LR = 0.01
MICROBATCH = 8
SURFACE_M1 = 0.1735
SURFACE_M3 = 0.0581
SURFACE_M4 = 0.0
ARMS = {
    "H4_per_position_multih_pe": {
        "evidence_source": EVIDENCE_PER_POSITION,
        "readout_heads": 4,
    },
    "H1_singlehead_pe_control": {
        "evidence_source": EVIDENCE_PER_POSITION,
        "readout_heads": 1,
    },
    "C4_broadcast_multih_pe": {
        "evidence_source": EVIDENCE_BROADCAST_FINAL,
        "readout_heads": 4,
    },
}


def _load(split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines()
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
    """Per-pair-type rate of both members correct with the required answer
    relation (flip pairs must predict differently, invariance identically)."""

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for item in scored:
        if item["pair_id"]:
            by_pair.setdefault(str(item["pair_id"]), []).append(item)
    out: dict[str, dict[str, Any]] = {}
    for pair_type in ("fact_flip", "combo_flip", "invariance"):
        members_by_pair = [
            members
            for members in by_pair.values()
            if members and members[0]["pair_type"] == pair_type
        ]
        need_same = pair_type == "invariance"
        hits = 0
        for members in members_by_pair:
            if len(members) != 2 or not all(bool(m["correct"]) for m in members):
                continue
            if (members[0]["prediction"] == members[1]["prediction"]) == need_same:
                hits += 1
        out[pair_type] = {
            "numerator": hits,
            "denominator": len(members_by_pair),
            "value": hits / len(members_by_pair) if members_by_pair else 0.0,
        }
    return out


def _evaluate(prototype: SequenceWorkspacePrototype, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in (
        "full",
        "no_context",
        "zero_read",
        "copy_misbind",
        "value_misbind",
    ):
        scored = _rows_for_scoring(prototype, rows, condition=condition)
        out[condition] = metrics(scored)
    full_scored = _rows_for_scoring(prototype, rows, condition="full")
    out["pair_rates"] = _pair_type_rates(full_scored)
    stops = 0
    for row in rows:
        result = prototype.generate(row["prefix"].encode("utf-8"), max_bytes=12)
        if result.stopped_on_boundary:
            stops += 1
    out["boundary_stop_rate"] = stops / len(rows)
    return out


def _train(
    arm: str,
    arm_config: dict[str, Any],
    seed: int,
    episodes: tuple[tuple[bytes, bytes], ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.monotonic()
    torch.manual_seed(seed)
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=seed,
            evidence_source=arm_config["evidence_source"],
            copy_mixture=True,
            question_conditioned_start=True,
            readout_heads=arm_config["readout_heads"],
            positional_keys=True,
        )
    )
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=LR, code_revision=f"r2-d3-{arm}-{seed}"
    )
    trainer.set_episodes(episodes)
    target_dir = PROJECT_ROOT / CHECKPOINT_ROOT / arm / str(seed)
    target_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(target_dir / "zero_step.pt")
    for _ in range(EPOCHS):
        for start in range(0, len(episodes), MICROBATCH):
            trainer.train_step(list(episodes[start : start + MICROBATCH]))
    trainer.save(target_dir / "epoch30.pt")
    evaluation = _evaluate(prototype, rows)
    return {
        "seed": seed,
        "parameter_count": prototype.parameter_count(),
        "elapsed_seconds": time.monotonic() - started,
        "eval": evaluation,
    }


def main() -> int:
    raw = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    digest = content_digest(raw)
    if digest != CORPUS_DIGEST:
        raise RuntimeError(f"R2-D1 corpus digest mismatch: {digest}")
    train_rows = _load("train")
    dev_rows = _load("dev")
    episodes = tuple(
        (row["prefix"].encode("utf-8"), row["response"].encode("utf-8")) for row in train_rows
    )
    runs: dict[str, list[dict[str, Any]]] = {}
    for arm, arm_config in ARMS.items():
        runs[arm] = [_train(arm, arm_config, seed, episodes, dev_rows) for seed in SEEDS]

    def v(arm: str, condition: str, metric: str) -> list[float]:
        return [float(run["eval"][condition][metric]["value"]) for run in runs[arm]]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    h4_m3, h1_m3 = v("H4_per_position_multih_pe", "full", "M3_content_exact"), v(
        "H1_singlehead_pe_control", "full", "M3_content_exact"
    )
    h4_m4, h1_m4 = v("H4_per_position_multih_pe", "full", "M4_flip_pair"), v(
        "H1_singlehead_pe_control", "full", "M4_flip_pair"
    )
    h4_m1_full = v("H4_per_position_multih_pe", "full", "M1_exact")
    h4_m3_nc = v("H4_per_position_multih_pe", "no_context", "M3_content_exact")
    h4_m1_nc = v("H4_per_position_multih_pe", "no_context", "M1_exact")
    h4_m4_misbind = v("H4_per_position_multih_pe", "copy_misbind", "M4_flip_pair")
    delta_m3 = [a - b for a, b in zip(h4_m3, h1_m3, strict=True)]
    delta_m4 = [a - b for a, b in zip(h4_m4, h1_m4, strict=True)]
    context_margin = [a - b for a, b in zip(h4_m3, h4_m3_nc, strict=True)]
    misbind_drop = [a - b for a, b in zip(h4_m4, h4_m4_misbind, strict=True)]
    h4_fact_flip = [
        float(run["eval"]["pair_rates"]["fact_flip"]["value"])
        for run in runs["H4_per_position_multih_pe"]
    ]
    h4_combo_flip = [
        float(run["eval"]["pair_rates"]["combo_flip"]["value"])
        for run in runs["H4_per_position_multih_pe"]
    ]
    boundary_all = [
        run["eval"]["boundary_stop_rate"] for arm_runs in runs.values() for run in arm_runs
    ]
    gates = {
        "G1_m3_ceiling_and_delta": {
            "passed": mean(h4_m3) >= 0.26 and mean(delta_m3) >= 0.20,
            "h4_m3_mean": mean(h4_m3),
            "delta_m3_mean": mean(delta_m3),
        },
        "G2_m4_flip_fact_subset": {
            "passed": (
                mean(h4_m4) >= 0.50 and mean(delta_m4) >= 0.25 and mean(h4_fact_flip) >= 2 / 6
            ),
            "h4_m4_mean": mean(h4_m4),
            "delta_m4_mean": mean(delta_m4),
            "h4_fact_flip_mean": mean(h4_fact_flip),
            "h4_combo_flip_mean_routing": mean(h4_combo_flip),
        },
        "G3_copy_misbind_drop": {
            "passed": mean(misbind_drop) >= 0.50,
            "m4_drop_mean": mean(misbind_drop),
        },
        "G4_context_margin": {
            "passed": mean(context_margin) >= 0.30 and max(h4_m1_nc) <= 0.50,
            "delta_m3_context_mean": mean(context_margin),
            "no_context_m1_max": max(h4_m1_nc),
        },
        "G5_seed_consistency": {
            "passed": sum(1 for d in delta_m4 if d > 0) >= 2 and min(delta_m4) >= 0,
            "wins": sum(1 for d in delta_m4 if d > 0),
            "worst_delta_m4": min(delta_m4),
        },
        "G6_boundary_and_restore": {
            "passed": min(boundary_all) >= 0.95,
            "boundary_min": min(boundary_all),
        },
        "surface_ceiling": {
            "h4_m1_beats_surface": mean(h4_m1_full) > SURFACE_M1,
            "h4_m3_beats_surface": mean(h4_m3) > SURFACE_M3,
            "h4_m4_beats_surface": mean(h4_m4) > SURFACE_M4,
        },
    }
    all_g = [name for name in gates if name.startswith("G")]
    passed = all(gates[name]["passed"] for name in all_g)
    report = {
        "format": "taiji-r2-d3-matched-dev-v1",
        "version": 1,
        "contract": "plans/reference/M5_R2_D3_FIRST_STEP_GEOMETRY_CONTRACT_FROZEN_20260918.md",
        "corpus_digest": digest,
        "dev_split": "dev",
        "epochs": EPOCHS,
        "lr": LR,
        "microbatch": MICROBATCH,
        "seeds": list(SEEDS),
        "runs": runs,
        "gates": gates,
        "outcome": "passed" if passed else "failed",
        "growth_admitted": False,
        "can_promote": False,
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outcome": report["outcome"],
                "h4_M1": [round(x, 3) for x in h4_m1_full],
                "h4_M3": [round(x, 3) for x in h4_m3],
                "h4_M4": [round(x, 3) for x in h4_m4],
                "h1_M4": [round(x, 3) for x in h1_m4],
                "misbind_M4": [round(x, 3) for x in h4_m4_misbind],
                "no_context_M3": [round(x, 3) for x in h4_m3_nc],
                "boundary_min": round(min(boundary_all), 3),
                "gates": {name: gates[name]["passed"] for name in all_g},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
