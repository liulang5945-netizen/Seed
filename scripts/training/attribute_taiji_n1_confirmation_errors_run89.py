"""PLAN-N1-91 甲路：只读已封存的 288 行确认报告，出版逐题错因归因。

零查询、零训练、零远端动作。本件不是能力读数也不是臂差读数（PLAN-N1-91 §0），
它只回答"三枚终端在同 32 题上以什么形态错、错在选择侧还是续跑侧"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/taiji_n1_transition_factorial_confirmation_run89.json"
FACE = ROOT / "plans/manifests/n1_fresh_confirmation_v5_20261016.json"
PLAN = ROOT / "plans/reference/PLAN-N1-91_per_item_error_attribution_run89_20261017.md"
DEFAULT_OUT = ROOT / "reports/taiji_n1_confirmation_error_attribution_run89.json"

ARMS = ("position", "terminal", "cut")
SEEDS = (11, 12, 13)
BOUNDARY = 256
SHAPES = (
    "empty_output",
    "first_symbol_wrong",
    "prefix_then_diverge",
    "truncated_prefix",
    "overshoot",
    "equal_output_not_exact",
)
AGREEMENTS = (
    "all_three_right",
    "all_three_wrong",
    "exactly_one_right",
    "exactly_two_right",
    "other",
)
CLAIMS_FORBIDDEN = (
    "任何臂更强",
    "机制已查明",
    "训练不足或训练充分",
    "C3 或 C4 或三器官某一格已达成",
)


class Refuse(Exception):
    """一条守卫不成立：本件不出结论，rc=2。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_normalized(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load() -> tuple[dict, dict]:
    require(REPORT.is_file(), f"sealed confirmation report is missing: {REPORT}")
    require(FACE.is_file(), f"v5 face manifest is missing: {FACE}")
    require(PLAN.is_file(), f"PLAN-N1-91 pre-registration is missing: {PLAN}")
    return (
        json.loads(REPORT.read_text(encoding="utf-8")),
        json.loads(FACE.read_text(encoding="utf-8")),
    )


def core_symbols(row: dict) -> list[int]:
    target = row["target_symbols"]
    require(bool(target), f"{row['id']} has an empty target_symbols list")
    require(
        target[-1] == BOUNDARY,
        f"{row['id']} target does not end with boundary {BOUNDARY}",
    )
    return list(target[:-1])


def output_symbols(row: dict) -> list[int]:
    return list(bytes.fromhex(row["output_hex"]))


def coherence_checks(report: dict) -> dict:
    rows = report["rows"]
    boundary_bad = 0
    for row in rows:
        if (
            row["stop_reason"] == "boundary"
            and row["generated_steps"] != len(output_symbols(row)) + 1
        ):
            boundary_bad += 1
    cap_steps = sorted({row["generated_steps"] for row in rows if row["stop_reason"] == "cap"})
    observed_max = max(row["generated_steps"] for row in rows)
    return {
        "boundary_rows_incoherent": boundary_bad,
        "cap_rows": len([row for row in rows if row["stop_reason"] == "cap"]),
        "cap_generated_steps_values": cap_steps,
        "observed_max_generated_steps": observed_max,
        "protocol_max_generated_bytes": report["protocol"].get("max_generated_bytes"),
    }


def guards(report: dict, face: dict) -> dict:
    rows = report["rows"]
    # B1 shape
    require(len(rows) == 288, f"B1 rows must be 288, found {len(rows)}")
    require({row["arm"] for row in rows} == set(ARMS), "B1 arms are not the three families")
    require({row["seed"] for row in rows} == set(SEEDS), "B1 seeds are not 11/12/13")
    require({row["split"] for row in rows} == {"holdout"}, "B1 split is not uniformly holdout")
    per_cell = Counter((row["seed"], row["arm"]) for row in rows)
    require(
        all(per_cell[(seed, arm)] == 32 for seed in SEEDS for arm in ARMS),
        "B1 each (seed, arm) must carry exactly 32 rows",
    )
    face_ids = [item["id"] for item in face["episodes"]]
    require(len(face_ids) == 32 and len(set(face_ids)) == 32, "B1 face must hold 32 unique ids")
    require(
        {row["id"] for row in rows} == set(face_ids),
        "B1 row ids differ from the face ids",
    )
    # B2 face seal, disclosing both digest conventions
    seal = report["confirmation_seals"]["v5"]
    raw, normalized = sha_raw(FACE), sha_normalized(FACE)
    matched = [
        name
        for name, digest in (("raw", raw), ("crlf_normalized_to_lf", normalized))
        if digest == seal["sha256"]
    ]
    require(bool(matched), "B2 face digest matches neither convention recorded in the seal")
    require(seal.get("bytes") == FACE.stat().st_size, "B2 face byte length drifted")
    # B3 aggregate recompute
    recomputed = {arm: 0 for arm in ARMS}
    per_cell_free: dict = defaultdict(int)
    for row in rows:
        if row["free_exact"]:
            recomputed[row["arm"]] += 1
            per_cell_free[(row["seed"], row["arm"])] += 1
    require(
        {arm: recomputed[arm] for arm in ARMS} == report["aggregate_exact"],
        f"B3 recomputed per-arm totals differ from the sealed aggregate: {recomputed}",
    )
    for seed_text, arms in report["summary"].items():
        for arm, block in arms.items():
            require(
                block["free_exact"] == per_cell_free[(int(seed_text), arm)],
                f"B3 summary[{seed_text}][{arm}].free_exact disagrees with the rows",
            )
    # B4 field semantics
    coherence = coherence_checks(report)
    require(
        coherence["boundary_rows_incoherent"] == 0,
        f"B4 {coherence['boundary_rows_incoherent']} boundary rows break the step/symbol identity",
    )
    require(
        len(coherence["cap_generated_steps_values"]) <= 1,
        f"B4 cap rows disagree on generated_steps: {coherence['cap_generated_steps_values']}",
    )
    if coherence["cap_generated_steps_values"]:
        require(
            coherence["cap_generated_steps_values"][0] == coherence["observed_max_generated_steps"],
            "B4 cap rows are not at the observed maximum step count",
        )
    return {
        "B1_shape": {
            "rows": len(rows),
            "arms": sorted(ARMS),
            "seeds": list(SEEDS),
            "ids_equal_face": True,
        },
        "B2_face_seal": {
            "recorded": seal["sha256"],
            "matched_convention": matched,
            "raw": raw,
            "crlf_normalized_to_lf": normalized,
            "bytes": seal.get("bytes"),
        },
        "B3_aggregates": {
            "per_arm_free_exact": {arm: recomputed[arm] for arm in ARMS},
            "sealed_per_arm_free_exact": report["aggregate_exact"],
            "per_cell_checked": len(per_cell_free),
        },
        "B4_field_semantics": coherence,
    }


def agreement_class(flags: list[bool]) -> str:
    right = sum(flags)
    if right == 3:
        return "all_three_right"
    if right == 0:
        return "all_three_wrong"
    if right == 1:
        return "exactly_one_right"
    if right == 2:
        return "exactly_two_right"
    return "other"


def shape_class(row: dict) -> tuple[str, bool]:
    core = core_symbols(row)
    out = output_symbols(row)
    capped = row["stop_reason"] == "cap"
    if not out:
        return "empty_output", capped
    match = 0
    while match < len(out) and match < len(core) and out[match] == core[match]:
        match += 1
    if match == 0:
        return "first_symbol_wrong", capped
    if out == core:
        return "equal_output_not_exact", capped
    if match == len(out) < len(core):
        return "truncated_prefix", capped
    if match == len(core) < len(out):
        return "overshoot", capped
    return "prefix_then_diverge", capped


def attribute(report: dict) -> dict:
    rows = report["rows"]
    by_cell: dict = defaultdict(dict)
    for row in rows:
        by_cell[(row["seed"], row["id"])][row["arm"]] = row

    axis1 = Counter()
    axis1_by_seed: dict = defaultdict(Counter)
    one_right_by_arm = Counter()
    two_right_missing_arm = Counter()
    axis5 = Counter()
    all_wrong_items: list = []
    triple_total = 0
    for (seed, item_id), arms in sorted(by_cell.items()):
        flags = [arms[arm]["free_exact"] for arm in ARMS]
        label = agreement_class(flags)
        axis1[label] += 1
        axis1_by_seed[seed][label] += 1
        if label == "all_three_wrong":
            all_wrong_items.append(f"seed{seed}:{item_id}")
        if label == "exactly_one_right":
            one_right_by_arm[ARMS[flags.index(True)]] += 1
        if label == "exactly_two_right":
            two_right_missing_arm[ARMS[flags.index(False)]] += 1
        if label == "all_three_wrong":
            triple_total += 1
            hexes = {arm: arms[arm]["output_hex"] for arm in ARMS}
            for first, second in (
                ("position", "terminal"),
                ("position", "cut"),
                ("terminal", "cut"),
            ):
                if hexes[first] == hexes[second]:
                    axis5[f"{first}_equals_{second}"] += 1
            if hexes["position"] == hexes["terminal"] == hexes["cut"]:
                axis5["all_three_identical"] += 1
    require(triple_total == axis1["all_three_wrong"], "axis5 denominator drifted")
    require(len(all_wrong_items) == triple_total, "axis5 item list drifted from its counter")

    wrong_rows = [row for row in rows if not row["free_exact"]]
    shapes = Counter()
    capped = 0
    unclassified = 0
    per_kind_shape: dict = defaultdict(Counter)
    for row in wrong_rows:
        label, is_capped = shape_class(row)
        shapes[label] += 1
        capped += int(is_capped)
        per_kind_shape[row["kind"]][label] += 1
        if label not in SHAPES:
            unclassified += 1
    require(
        sum(shapes[key] for key in SHAPES) == len(wrong_rows),
        "axis2 shape classes are not complete over the free-wrong rows",
    )

    compounding = len([row for row in wrong_rows if row["teacher_exact"]])
    selection = len([row for row in wrong_rows if not row["teacher_exact"]])
    require(compounding + selection == len(wrong_rows), "axis3 split is not exhaustive")
    missing = Counter(
        len(row["target_symbols"]) - row["teacher_tokens_correct"] for row in wrong_rows
    )

    axis4: dict = {}
    for seed in SEEDS:
        for arm in ARMS:
            cell = [row for row in rows if row["seed"] == seed and row["arm"] == arm]
            axis4[f"{seed}_{arm}"] = {
                kind: {
                    "items": len([row for row in cell if row["kind"] == kind]),
                    "free_exact": len(
                        [row for row in cell if row["kind"] == kind and row["free_exact"]]
                    ),
                    "teacher_exact": len(
                        [row for row in cell if row["kind"] == kind and row["teacher_exact"]]
                    ),
                }
                for kind in sorted({row["kind"] for row in cell})
            }

    return {
        "axis1_agreement": {key: axis1[key] for key in AGREEMENTS},
        "axis1_agreement_by_seed": {
            str(seed): {key: axis1_by_seed[seed][key] for key in AGREEMENTS} for seed in SEEDS
        },
        "axis1_groups_total": sum(axis1[key] for key in AGREEMENTS),
        "axis1_exactly_one_right_by_arm": dict(sorted(one_right_by_arm.items())),
        "axis1_exactly_two_right_missing_arm": dict(sorted(two_right_missing_arm.items())),
        "axis2_shape_classes": {key: shapes[key] for key in SHAPES},
        "axis2_no_boundary_cap": capped,
        "axis2_unclassified": unclassified,
        "axis2_free_wrong_rows": len(wrong_rows),
        "axis2_shapes_by_kind": {
            kind: dict(sorted(counter.items())) for kind, counter in sorted(per_kind_shape.items())
        },
        "axis3_compounding_only": compounding,
        "axis3_selection_side": selection,
        "axis3_teacher_tokens_missing_histogram": {str(k): v for k, v in sorted(missing.items())},
        "axis4_kind_seed_arm": axis4,
        "axis5_identical_outputs_within_all_wrong": dict(sorted(axis5.items())),
        "axis5_all_three_wrong_groups": triple_total,
        "axis5_all_wrong_items": all_wrong_items,
    }


def build(out_path: Path) -> dict:
    report, face = load()
    started = time.process_time()
    guarded = guards(report, face)
    reading = attribute(report)
    require(reading["axis1_groups_total"] == 96, "axis1 must group the rows into 96 triples")
    require(reading["axis2_unclassified"] == 0, "axis2 left rows unclassified")
    acceptance = {
        "guards_B1_B4_true": True,
        "axis1_total_96": reading["axis1_groups_total"] == 96,
        "axis2_complete_and_exclusive": reading["axis2_unclassified"] == 0,
        "axis3_exhaustive": reading["axis3_compounding_only"] + reading["axis3_selection_side"]
        == reading["axis2_free_wrong_rows"],
        "axis5_denominator_matches": reading["axis5_all_three_wrong_groups"]
        == reading["axis1_agreement"]["all_three_wrong"],
    }
    payload = {
        "format": "taiji-n1-91-per-item-error-attribution-v1",
        "not_a_capability_reading": True,
        "not_an_arm_advantage_reading": True,
        "claims_forbidden_by_preregistration": list(CLAIMS_FORBIDDEN),
        "plan": str(PLAN),
        "plan_sha256": sha_raw(PLAN),
        "inputs": {
            "report": str(REPORT),
            "report_sha256_raw": sha_raw(REPORT),
            "report_sha256_crlf_normalized_to_lf": sha_normalized(REPORT),
            "face": str(FACE),
        },
        "guards": guarded,
        **reading,
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
        "confirmation_queries": 0,
        "training_updates": 0,
        "cpu_seconds": round(time.process_time() - started, 4),
        "status": "published" if all(acceptance.values()) else "stopped",
    }
    require(not out_path.exists(), f"attribution output already exists: {out_path}")
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readiness", action="store_true", help="只走 §1 四条守卫，不写任何文件")
    parser.add_argument("--approved", action="store_true", help="出版单枚归因报告")
    parser.add_argument("--out-report", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args(argv)
    try:
        require(
            arguments.readiness != arguments.approved,
            "choose exactly one of --readiness/--approved",
        )
        if arguments.readiness:
            report, face = load()
            summary = guards(report, face)
            print(
                json.dumps(
                    {"mode": "readiness", "guards": summary}, ensure_ascii=False, sort_keys=True
                )
            )
            return 0
        payload = build(arguments.out_report)
        print(
            json.dumps(
                {
                    "mode": "approved",
                    "accepted": payload["accepted"],
                    "status": payload["status"],
                    "axis1": payload["axis1_agreement"],
                    "axis2": payload["axis2_shape_classes"],
                    "axis3": {
                        "compounding_only": payload["axis3_compounding_only"],
                        "selection_side": payload["axis3_selection_side"],
                    },
                    "axis5": payload["axis5_identical_outputs_within_all_wrong"],
                    "out": str(arguments.out_report),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if payload["accepted"] else 1
    except Refuse as refuse:
        print(f"REFUSE: {refuse}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
