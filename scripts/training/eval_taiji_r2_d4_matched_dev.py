"""R2-D4 matched-dev evaluation for H-T copy-value supervision (contract §5).

Trains the T2 arm (v6 graph unchanged, lambda=1.0 auxiliary copy-component NLL
on answer value bytes) over three seeds, then evaluates the sealed R2-D1 dev
split under the same five conditions and M1-M5 metrics as D3.  The lambda=0
control is the frozen D3 matched run's H4 arm (same seeds, same graph, same
order); the contract requires a one-seed bitwise checkpoint-equivalence check
before referencing it, performed here first.  Gates G1-G6 with the D3 H4 arm
as the delta baseline; combo_flip is recorded separately (C/B routing).
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

FIXTURE = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
CORPUS_DIGEST = "53ac9f88695f135d0bb04b4d25d98ab686c82175c45ca8d4d53e64639efbeecb"
OUT_REPORT = Path("reports/r2_d4_matched_dev_20260918.json")
CHECKPOINT_ROOT = Path("reports/r2_d4_checkpoints/matched")
D3_REPORT = Path("reports/r2_d3_matched_dev_20260918.json")
D3_H4_ARM = "H4_per_position_multih_pe"
D3_PROBE_CHECKPOINT = Path("reports/r2_d3_checkpoints/multihead_probe/hg_h4_seed20260917_epoch30.pt")
SEEDS = (20260917, 20260918, 20260919)
EPOCHS = 30
LR = 0.01
MICROBATCH = 8
LAMBDA_COPY = 1.0
SURFACE_M1 = 0.1735
SURFACE_M3 = 0.0581
SURFACE_M4 = 0.0


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
        pair_hits = sum(
            1
            for pid in complete_pairs
            if hits.get(pid, set()) == members_by_pair[pid]
        )
        out[pair_type] = {
            "pair_hits": pair_hits,
            "denominator": len(complete_pairs),
            "value": pair_hits / len(complete_pairs) if complete_pairs else 0.0,
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
    seed: int,
    episodes: tuple[tuple[bytes, bytes], ...],
    masks: tuple[tuple[bool, ...], ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.monotonic()
    torch.manual_seed(seed)
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=seed,
            evidence_source=EVIDENCE_PER_POSITION,
            copy_mixture=True,
            question_conditioned_start=True,
            readout_heads=4,
            positional_keys=True,
        )
    )
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=LR, code_revision=f"r2-d4-t2-{seed}"
    )
    trainer.enable_copy_value_supervision(LAMBDA_COPY)
    trainer.set_episodes(episodes)
    target_dir = PROJECT_ROOT / CHECKPOINT_ROOT / str(seed)
    target_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(target_dir / "zero_step.pt")
    for _ in range(EPOCHS):
        for start in range(0, len(episodes), MICROBATCH):
            batch = list(episodes[start : start + MICROBATCH])
            batch_masks = [masks[index] for index in range(start, start + len(batch))]
            trainer.train_step(batch, value_masks=batch_masks)
    trainer.save(target_dir / "epoch30.pt")
    evaluation = _evaluate(prototype, rows)
    return {
        "seed": seed,
        "parameter_count": prototype.parameter_count(),
        "elapsed_seconds": time.monotonic() - started,
        "eval": evaluation,
    }


def _verify_lambda0_equivalence(episodes: tuple[tuple[bytes, bytes], ...]) -> dict[str, Any]:
    """Contract §5: the lambda=0 training path must reproduce the D3 H4 probe
    checkpoint bit-for-bit at the shared seed before the historical control is
    referenced."""

    torch.manual_seed(SEEDS[0])
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(
            seed=SEEDS[0],
            evidence_source=EVIDENCE_PER_POSITION,
            copy_mixture=True,
            question_conditioned_start=True,
            readout_heads=4,
            positional_keys=True,
        )
    )
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=LR, code_revision="r2-d4-lambda0-equivalence"
    )
    trainer.set_episodes(episodes)
    for _ in range(EPOCHS):
        for start in range(0, len(episodes), MICROBATCH):
            trainer.train_step(list(episodes[start : start + MICROBATCH]))
    reproduced = trainer.prototype.parameter_payload()
    frozen = torch.load(D3_PROBE_CHECKPOINT, map_location="cpu", weights_only=False)["parameters"]
    identical = True
    max_diff = 0.0
    for name, tensor in reproduced.items():
        diff = (tensor.float() - frozen[name].float()).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff != 0.0:
            identical = False
    return {
        "seed": SEEDS[0],
        "reproduced_checkpoint": "lambda0 fresh training, 30 epochs, microbatch 8",
        "frozen_checkpoint": str(D3_PROBE_CHECKPOINT),
        "bitwise_identical": identical,
        "max_abs_diff": max_diff,
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
    masks = tuple(value_mask_for(row["shape"], row["response"]) for row in train_rows)

    equivalence = _verify_lambda0_equivalence(episodes)
    if not equivalence["bitwise_identical"]:
        raise RuntimeError(
            f"lambda=0 path no longer reproduces the frozen D3 checkpoint: {equivalence}"
        )

    d3_report = json.loads((PROJECT_ROOT / D3_REPORT).read_text(encoding="utf-8"))
    d3_baseline = {run["seed"]: run for run in d3_report["runs"][D3_H4_ARM]}

    runs = [_train(seed, episodes, masks, dev_rows) for seed in SEEDS]

    def v(metric: str, condition: str = "full") -> list[float]:
        return [float(run["eval"][condition][metric]["value"]) for run in runs]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    t2_m3 = v("M3_content_exact")
    t2_m4 = v("M4_flip_pair")
    t2_m1 = v("M1_exact")
    t2_m3_nc = v("M3_content_exact", "no_context")
    t2_m1_nc = v("M1_exact", "no_context")
    t2_m4_misbind = v("M4_flip_pair", "copy_misbind")
    base_m3 = [float(d3_baseline[seed]["eval"]["full"]["M3_content_exact"]["value"]) for seed in SEEDS]
    base_m4 = [float(d3_baseline[seed]["eval"]["full"]["M4_flip_pair"]["value"]) for seed in SEEDS]
    delta_m3 = [a - b for a, b in zip(t2_m3, base_m3, strict=True)]
    delta_m4 = [a - b for a, b in zip(t2_m4, base_m4, strict=True)]
    context_margin = [a - b for a, b in zip(t2_m3, t2_m3_nc, strict=True)]
    misbind_drop = [a - b for a, b in zip(t2_m4, t2_m4_misbind, strict=True)]
    fact_flip = [float(run["eval"]["pair_rates"]["fact_flip"]["value"]) for run in runs]
    combo_flip = [float(run["eval"]["pair_rates"]["combo_flip"]["value"]) for run in runs]
    invariance = [float(run["eval"]["pair_rates"]["invariance"]["value"]) for run in runs]
    boundary_all = [run["eval"]["boundary_stop_rate"] for run in runs]
    gates = {
        "G1_m3_ceiling_and_delta": {
            "passed": mean(t2_m3) >= 0.26 and mean(delta_m3) >= 0.20,
            "t2_m3_mean": mean(t2_m3),
            "delta_m3_mean_vs_d3h4": mean(delta_m3),
        },
        "G2_m4_flip_fact_subset": {
            "passed": (
                mean(t2_m4) >= 0.50 and mean(delta_m4) >= 0.25 and mean(fact_flip) >= 2 / 6
            ),
            "t2_m4_mean": mean(t2_m4),
            "delta_m4_mean_vs_d3h4": mean(delta_m4),
            "fact_flip_mean": mean(fact_flip),
            "combo_flip_mean_routing": mean(combo_flip),
            "invariance_mean": mean(invariance),
        },
        "G3_copy_misbind_drop": {
            "passed": mean(misbind_drop) >= 0.50,
            "m4_drop_mean": mean(misbind_drop),
        },
        "G4_context_margin": {
            "passed": mean(context_margin) >= 0.30 and max(t2_m1_nc) <= 0.50,
            "delta_m3_context_mean": mean(context_margin),
            "no_context_m1_max": max(t2_m1_nc),
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
            "t2_m1_beats_surface": mean(t2_m1) > SURFACE_M1,
            "t2_m3_beats_surface": mean(t2_m3) > SURFACE_M3,
            "t2_m4_beats_surface": mean(t2_m4) > SURFACE_M4,
        },
    }
    all_g = [name for name in gates if name.startswith("G")]
    passed = all(gates[name]["passed"] for name in all_g)
    report = {
        "format": "taiji-r2-d4-matched-dev-v1",
        "version": 1,
        "contract": "plans/reference/M5_R2_D4_COPY_SUPERVISION_CONTRACT_FROZEN_20260918.md",
        "corpus_digest": digest,
        "dev_split": "dev",
        "epochs": EPOCHS,
        "lr": LR,
        "microbatch": MICROBATCH,
        "lambda_copy_value": LAMBDA_COPY,
        "seeds": list(SEEDS),
        "lambda0_equivalence_check": equivalence,
        "d3_baseline_arm": D3_H4_ARM,
        "d3_baseline_source": str(D3_REPORT),
        "runs": {"T2_copy_value_supervision": runs},
        "gates": gates,
        "outcome": "passed" if passed else "failed",
        "growth_admitted": False,
        "can_promote": False,
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / OUT_REPORT).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "outcome": report["outcome"],
                "t2_M1": [round(x, 3) for x in t2_m1],
                "t2_M3": [round(x, 3) for x in t2_m3],
                "t2_M4": [round(x, 3) for x in t2_m4],
                "delta_m3_vs_d3h4": [round(x, 3) for x in delta_m3],
                "delta_m4_vs_d3h4": [round(x, 3) for x in delta_m4],
                "misbind_M4": [round(x, 3) for x in t2_m4_misbind],
                "no_context_M3": [round(x, 3) for x in t2_m3_nc],
                "fact_flip": [round(x, 3) for x in fact_flip],
                "combo_flip": [round(x, 3) for x in combo_flip],
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
