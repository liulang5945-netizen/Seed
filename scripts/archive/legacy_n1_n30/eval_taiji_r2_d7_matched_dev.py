"""R2-D7 matched-dev adjudication for the char-token graph (contract section 5).

Arm T1 (v2 train + char graph + induction + lambda=1.0) trains over three
seeds and is evaluated on the sealed v1 dev split.  Gates G1-G6 run on
absolute readings (the char family has no same-family historical control; the
byte-graph D4 T2 arm is referenced descriptively only, and the cross-granular
delta is explicitly not a claim).  The decisive contracted gate is K1: dev
multi-character value full_rate >= 0.30, with K2 = M4 flip >= 0.50 as the
binding-quality follow-through and first-byte (首字) rate held against the
byte-graph level 0.529.
"""

from __future__ import annotations

import argparse
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
from scripts.training.probe_taiji_r2_d7_char_token import (  # noqa: E402
    _load_train,
    value_mask_for_char,
)
from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_char_workspace import (  # noqa: E402
    CharVocab,
    SequenceCharConfig,
    SequenceCharTrainer,
    SequenceCharWorkspace,
)

FIXTURES = {
    "v1": Path("tests/fixtures/r2_d1_measurement_v1.jsonl"),
    "v2": Path("tests/fixtures/r2_d1_measurement_v2.jsonl"),
    # R2-D8 H-S scale pack: same shapes/templates/order rules, train pool scaled up
    # (32 objects x 12 colours).  dev/final are byte-identical to v1, so the dev
    # evaluation below keeps reading v1 unchanged.
    "v3": Path("tests/fixtures/r2_d1_measurement_v3.jsonl"),
}
V1_CORPUS_DIGEST = "53ac9f88695f135d0bb04b4d25d98ab686c82175c45ca8d4d53e64639efbeecb"
TRAIN_CORPUS_DIGESTS = {
    "v1": V1_CORPUS_DIGEST,
    "v2": None,  # v2 train differs from v1; the digest is recorded, not pinned
    "v3": "a73703de2759491c923066e24beac56f910d820b365ed6148e062f12565c3554",
}
OUT_REPORT = Path("reports/r2_d7_matched_dev_20260918.json")
CHECKPOINT_ROOT = Path("reports/r2_d7_checkpoints/matched")
SEEDS = (20260917, 20260918, 20260919)
EPOCHS = 30
LR = 0.01
MICROBATCH = 8
LAMBDA_COPY = 1.0
SURFACE_M1 = 0.1735
SURFACE_M3 = 0.0581
SURFACE_M4 = 0.0
COPY_SUPPORTED_SHAPES = ("fact", "negation", "same_opening_fact")


def _raw(fixture: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURES[fixture]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _score_row(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "shape": row["shape"],
        "source_group": row["source_group"],
        "content_dependent": row["content_dependent"],
        "pair_id": row["pair_id"],
        "pair_type": row["pair_type"],
        "pair_role": row["pair_role"],
        "gold": row["response"],
        "prediction": prediction,
        "correct": prediction == row["response"],
    }


def _rate(members: list[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(1 for row in members if row["correct"])
    return {
        "numerator": hits,
        "denominator": len(members),
        "value": hits / len(members) if members else 0.0,
    }


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
        complete = {pid for pid, ids in members_by_pair.items() if len(ids) == 2}
        pair_hits = sum(1 for pid in complete if hits.get(pid, set()) == members_by_pair[pid])
        out[pair_type] = {
            "pair_hits": pair_hits,
            "denominator": len(complete),
            "value": pair_hits / len(complete) if complete else 0.0,
        }
    return out


def _metrics(scored: list[dict[str, Any]]) -> dict[str, Any]:
    content = [row for row in scored if row["content_dependent"]]
    flip_members = [row for row in scored if row["pair_type"] in ("fact_flip", "combo_flip")]
    flip_pairs: dict[str, list[dict[str, Any]]] = {}
    for row in flip_members:
        flip_pairs.setdefault(str(row["pair_id"]), []).append(row)
    flip_hits = sum(
        1
        for members in flip_pairs.values()
        if len(members) == 2 and all(m["correct"] for m in members)
    )
    inv_members = [row for row in scored if row["pair_type"] == "invariance"]
    inv_pairs: dict[str, list[dict[str, Any]]] = {}
    for row in inv_members:
        inv_pairs.setdefault(str(row["pair_id"]), []).append(row)
    inv_hits = sum(
        1
        for members in inv_pairs.values()
        if len(members) == 2 and all(m["correct"] for m in members)
    )
    shapes = sorted({row["shape"] for row in scored})
    macro = (
        sum(_rate([r for r in scored if r["shape"] == shape])["value"] for shape in shapes)
        / len(shapes)
        if shapes
        else 0.0
    )
    return {
        "M1_exact": _rate(scored),
        "M2_shape_macro": {"value": macro},
        "M3_content_exact": _rate(content),
        "M4_flip_pair": {
            "numerator": flip_hits,
            "denominator": len(flip_pairs),
            "value": flip_hits / len(flip_pairs) if flip_pairs else 0.0,
        },
        "M5_inv_pair": {
            "numerator": inv_hits,
            "denominator": len(inv_pairs),
            "value": inv_hits / len(inv_pairs) if inv_pairs else 0.0,
        },
    }


def _multibyte_completion(
    workspace: SequenceCharWorkspace, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Character-unit completion on dev multi-character-value episodes."""

    first_hits = 0
    full_hits = 0
    members = 0
    with torch.no_grad():
        for row in rows:
            if row["shape"] not in COPY_SUPPORTED_SHAPES:
                continue
            value = row["response"][2:] if row["shape"] == "negation" else row["response"]
            if len(value) <= 1:
                continue
            members += 1
            text = workspace.generate(row["prefix"]).text
            gold = row["response"]
            if text[:1] == gold[:1]:
                first_hits += 1
            if text == gold:
                full_hits += 1
    return {
        "episodes": members,
        "first_byte_rate": first_hits / members if members else 0.0,
        "full_rate": full_hits / members if members else 0.0,
    }


def _evaluate(workspace: SequenceCharWorkspace, dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in ("full", "no_context", "copy_misbind", "value_misbind"):
        scored = []
        for row in dev_rows:
            if condition == "full":
                text = workspace.generate(row["prefix"]).text
            elif condition == "no_context":
                text = workspace.generate(remove_material_clause(row["prefix"])).text
            elif condition == "copy_misbind":
                prefix = row["prefix"]
                rotation = len(prefix) // 2
                text = _generate_lesioned(
                    workspace, prefix, row["response"], entry_rotation=rotation
                )
            else:
                prefix = row["prefix"]
                rotation = len(prefix) // 2
                text = _generate_lesioned(
                    workspace, prefix, row["response"], value_rotation=rotation
                )
            scored.append(_score_row(row, text))
        out[condition] = _metrics(scored)
    full_scored = [_score_row(r, workspace.generate(r["prefix"]).text) for r in dev_rows]
    out["pair_rates"] = _pair_type_rates(full_scored)
    out["multibyte_completion"] = _multibyte_completion(workspace, dev_rows)
    stops = sum(1 for r in dev_rows if workspace.generate(r["prefix"]).stopped_on_boundary)
    out["boundary_stop_rate"] = stops / len(dev_rows)
    return out


def _generate_lesioned(
    workspace: SequenceCharWorkspace,
    prefix: str,
    response: str,
    *,
    entry_rotation: int = 0,
    value_rotation: int = 0,
) -> str:
    """Greedy generation under an evaluation lesion (module-level twin)."""

    return workspace.generate_lesioned(
        prefix,
        value_rotation=value_rotation,
        entry_rotation=entry_rotation,
        max_chars=12,
    ).text


def _train(
    seed: int, train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    started = time.monotonic()
    vocab = CharVocab("".join(r["prefix"] + r["response"] for r in train_rows))
    torch.manual_seed(seed)
    workspace = SequenceCharWorkspace(vocab, SequenceCharConfig(seed=seed, copy_induction=True))
    trainer = SequenceCharTrainer(
        workspace, learning_rate=LR, code_revision=f"r2-d7-matched-{seed}"
    )
    trainer.enable_copy_value_supervision(LAMBDA_COPY)
    episodes = tuple((r["prefix"], r["response"]) for r in train_rows)
    masks = tuple(value_mask_for_char(r["shape"], r["response"]) for r in train_rows)
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
    evaluation = _evaluate(workspace, dev_rows)
    return {
        "seed": seed,
        "vocab_size": int(vocab.size),
        "parameter_count": workspace.parameter_count(),
        "copy_induce_bias_end": float(workspace._parameters["copy_induce_bias"].detach().item()),
        "elapsed_seconds": time.monotonic() - started,
        "eval": evaluation,
    }


def main() -> int:
    global OUT_REPORT, CHECKPOINT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        choices=sorted(FIXTURES),
        default="v1",
        help="train split source; dev keeps reading v1 (v3 dev is byte-identical to v1)",
    )
    parser.add_argument("--out-report", type=Path, default=OUT_REPORT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    args = parser.parse_args()
    OUT_REPORT = args.out_report
    CHECKPOINT_ROOT = args.checkpoint_root

    if content_digest(_raw("v1")) != V1_CORPUS_DIGEST:
        raise RuntimeError("R2-D1 v1 corpus digest mismatch")
    dev_rows = [r for r in _raw("v1") if r["split"] == "dev"]
    train_rows = _load_train(FIXTURES[args.fixture])
    train_digest = TRAIN_CORPUS_DIGESTS.get(args.fixture)
    if train_digest is not None and content_digest(_raw(args.fixture)) != train_digest:
        raise RuntimeError(f"R2-D1 {args.fixture} corpus digest mismatch")

    runs = [_train(seed, train_rows, dev_rows) for seed in SEEDS]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    m1 = [float(r["eval"]["full"]["M1_exact"]["value"]) for r in runs]
    m3 = [float(r["eval"]["full"]["M3_content_exact"]["value"]) for r in runs]
    m4 = [float(r["eval"]["full"]["M4_flip_pair"]["value"]) for r in runs]
    m1_nc = [float(r["eval"]["no_context"]["M1_exact"]["value"]) for r in runs]
    m3_nc = [float(r["eval"]["no_context"]["M3_content_exact"]["value"]) for r in runs]
    m4_mis = [float(r["eval"]["copy_misbind"]["M4_flip_pair"]["value"]) for r in runs]
    fact_flip = [float(r["eval"]["pair_rates"]["fact_flip"]["value"]) for r in runs]
    combo_flip = [float(r["eval"]["pair_rates"]["combo_flip"]["value"]) for r in runs]
    boundary = [r["eval"]["boundary_stop_rate"] for r in runs]
    completion = [r["eval"]["multibyte_completion"] for r in runs]
    completion_mean = {
        "first_byte_rate": mean([c["first_byte_rate"] for c in completion]),
        "full_rate": mean([c["full_rate"] for c in completion]),
    }
    context = [a - b for a, b in zip(m3, m3_nc, strict=True)]
    mis_drop = [a - b for a, b in zip(m4, m4_mis, strict=True)]
    gates = {
        "G1_m3_ceiling": mean(m3) >= 0.26,
        "G2_m4_flip": mean(m4) >= 0.50 and mean(fact_flip) >= 2 / 6,
        "G3_copy_misbind_drop": mean(mis_drop) >= 0.50,
        "G4_context_margin": mean(context) >= 0.30 and max(m1_nc) <= 0.50,
        # D8 contract section 5 **redefines** G5 for the scale pack: the core
        # hypothesis is variance convergence, so this gate is about the per-seed
        # spread of the multibyte `full_rate` (>=2 seeds above 0.10, worst >= 0).
        # The v1/v2 path keeps the D7 definition unchanged (that pack is closed).
        "G5_seed_consistency": (
            (
                sum(1 for c in completion if c["full_rate"] > 0.10) >= 2
                and min(c["full_rate"] for c in completion) >= 0.0
            )
            if args.fixture == "v3"
            else (min(m3) >= 0.20 and min(m4) > 0)
        ),
        "G6_boundary": min(boundary) >= 0.95,
    }
    keys = {
        "K1_multibyte_full_rate_ge_0_30": completion_mean["full_rate"] >= 0.30,
        "K2_m4_flip_ge_0_50": mean(m4) >= 0.50,
        "first_byte_holds_byte_graph_level_0_529": completion_mean["first_byte_rate"] >= 0.529,
    }
    passed = all(gates.values()) and keys["K1_multibyte_full_rate_ge_0_30"]
    report = {
        "format": "taiji-r2-d7-matched-dev-v1",
        "version": 1,
        "contract": "plans/reference/M5_R2_D7_CHAR_TOKEN_CONTRACT_FROZEN_20260918.md",
        "scale_pack_contract": (
            "plans/reference/M5_R2_D8_SCALE_CONTRACT_FROZEN_20260918.md"
            if args.fixture == "v3"
            else None
        ),
        "train_fixture": args.fixture,
        "train_corpus_digest": content_digest(_raw(args.fixture)),
        "dev_split": "v1 (byte-identical to v3 dev)",
        "g5_definition": (
            "D8 contract section 5: >=2 seeds with full_rate > 0.10 and worst >= 0"
            if args.fixture == "v3"
            else "D7: min(m3) >= 0.20 and min(m4) > 0"
        ),
        "multibyte_full_rate_per_seed": [c["full_rate"] for c in completion],
        "loss_increases_gate_note": (
            "probe-stage gate: treated as single-seed jitter (1/3) per user adjudication "
            "(c) -> (b); see plans/reference/M5_R2_D8_PROBE_MULTISEED_20260918.md"
        ),
        "graph": "char-v1",
        "seeds": list(SEEDS),
        "arm": f"T1 ({args.fixture} train + char graph + induction + lambda 1.0)",
        "byte_history_reference_descriptive": {
            "A00_d4_t2_m3_mean": 0.178,
            "note": "cross-granularity deltas are descriptive, not gates",
        },
        "runs": {"T1_char_multibyte": runs},
        "gates": gates,
        "keys": keys,
        "metrics_mean": {
            "m1": mean(m1),
            "m3": mean(m3),
            "m4": mean(m4),
            "no_context_m1_max": max(m1_nc),
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
                "outcome": report["outcome"],
                "gates": gates,
                "keys": keys,
                "metrics_mean": report["metrics_mean"],
                "per_seed_m3": [round(x, 3) for x in m3],
                "per_seed_m4": [round(x, 3) for x in m4],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
