"""Score the R2 content-binding static data: policies, reference, pairwise.

Contract: plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md §3.2

Static implementation-gate battery.  All policies are deterministic and
content-blind in the binding sense: they may consult the TRAIN lexicon (value
words, object words) and scan the rendered material, but they never resolve
which object the question queries and never read the structured labels.

Policies
- ``fixed_unknown``        always answers 未知.
- ``first_value``          first train-value word in the material (未知 if none).
- ``last_value``           last train-value word in the material (未知 if none).
- ``second_object_value``  value of the second parsed statement, falling back to
                           the first statement, then 未知 (positional shortcut).

Gate (contract §3.2): on the train split, the pairwise-correct macro average of
every policy over {fact_flip, object_swap, relation_flip} must be <= 0.25;
otherwise the instrument is broken and no training may start.  The reference
key must answer every item of every split correctly.  The same pairwise scorer
(``score_records``) is the seven-class scoring rule later used for calibration
and confirmation predictions; policies here are its first consumers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_r2_content_binding_data import (  # noqa: E402
    FLIP_GATE_CLASSES,
    GROUP_CLASSES,
    POOLS,
    reference_answer,
    split_material_body,
)
from taiji.internalization import content_digest  # noqa: E402

FIXTURES = {
    "train": Path("tests/fixtures/r2_content_binding_v1_train.jsonl"),
    "calibration": Path("tests/fixtures/r2_content_binding_v1_calibration.jsonl"),
    "sealed": Path("tests/fixtures/r2_content_binding_v1_sealed.jsonl"),
}
DATA_REPORT = Path("reports/r2_content_binding_v1/data_contract_v1_20260919.json")
OUT_REPORT = Path("reports/r2_content_binding_v1/static_policy_baselines_v1_20260919.json")
REPORT_FORMAT = "r2-content-binding-static-baselines-v1"
POLICY_GATE_MAX = 0.25


def load_fixtures() -> dict[str, list[dict[str, Any]]]:
    raw: dict[str, list[dict[str, Any]]] = {}
    for split, relative in FIXTURES.items():
        path = PROJECT_ROOT / relative
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(row["split"] != split for row in rows):
            raise RuntimeError(f"fixture {relative} carries a foreign split row")
        raw[split] = rows
    return raw


def _statement_values(split: str, material: str) -> list[tuple[str, str, str]]:
    """Parsed (object, polarity, value) statements in material order."""

    facts: list[tuple[str, str, str]] = []
    for part in re.split(r"[，。；]", split_material_body(split, material)):
        part = part.strip()
        if not part:
            continue
        for obj in POOLS[split]["objects"]:
            if part.startswith(obj):
                rest = part[len(obj) :]
                if rest.startswith("不是"):
                    facts.append((obj, "不是", rest[2:]))
                elif rest.startswith("是"):
                    facts.append((obj, "是", rest[1:]))
                break
    return facts


def build_policies() -> dict[str, Callable[[dict[str, Any]], str]]:
    """Content-blind fixed strategies; lexicon = train split pools only."""

    train_values = sorted(
        list(POOLS["train"]["values_single"]) + list(POOLS["train"]["values_double"]),
        key=len,
        reverse=True,
    )
    train_objects = set(POOLS["train"]["objects"])

    def fixed_unknown(record: dict[str, Any]) -> str:
        return "未知"

    def _scan_values(record: dict[str, Any]) -> list[tuple[int, str]]:
        body = split_material_body(record["split"], record["material"])
        hits: list[tuple[int, str]] = []
        for value in train_values:
            start = body.find(value)
            while start != -1:
                hits.append((start, value))
                start = body.find(value, start + 1)
        return sorted(hits)

    def first_value(record: dict[str, Any]) -> str:
        hits = _scan_values(record)
        return hits[0][1] if hits else "未知"

    def last_value(record: dict[str, Any]) -> str:
        hits = _scan_values(record)
        return hits[-1][1] if hits else "未知"

    def second_object_value(record: dict[str, Any]) -> str:
        statements = [
            (obj, polarity, value)
            for obj, polarity, value in _statement_values(record["split"], record["material"])
            if obj in train_objects
        ]
        if len(statements) >= 2:
            return statements[1][2]
        if len(statements) == 1:
            return statements[0][2]
        return "未知"

    return {
        "fixed_unknown": fixed_unknown,
        "first_value": first_value,
        "last_value": last_value,
        "second_object_value": second_object_value,
    }


def score_records(
    records: list[dict[str, Any]], predict: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    """Seven-class pairwise scorer (the confirmation scoring rule).

    A group is pair-correct when both members match gold exactly.  Reports
    per-class item exact and pair rates plus the macro average over the three
    flip categories that the contract's policy gate restricts.
    """

    rows: list[dict[str, Any]] = []
    for record in records:
        prediction = predict(record)
        rows.append(
            {
                "id": record["id"],
                "group_id": record["group_id"],
                "group_class": record["group_class"],
                "gold": record["response"],
                "prediction": prediction,
                "correct": prediction == record["response"],
            }
        )
    by_class: dict[str, dict[str, Any]] = {}
    for cls in GROUP_CLASSES:
        members = [row for row in rows if row["group_class"] == cls]
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in members:
            groups.setdefault(row["group_id"], []).append(row)
        pair_hits = sum(
            1
            for members_of_group in groups.values()
            if len(members_of_group) == 2 and all(m["correct"] for m in members_of_group)
        )
        item_hits = sum(1 for row in members if row["correct"])
        by_class[cls] = {
            "items": len(members),
            "item_exact": item_hits / len(members) if members else None,
            "groups": len(groups),
            "pair_correct": pair_hits,
            "pair_rate": pair_hits / len(groups) if groups else None,
        }
    gate_values = [by_class[cls]["pair_rate"] for cls in FLIP_GATE_CLASSES]
    return {
        "items": rows,
        "per_class": by_class,
        "flip_pairwise_macro": sum(gate_values) / len(gate_values) if gate_values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="optional JSON {item_id: prediction} to score instead of the policies",
    )
    parser.add_argument("--report", type=Path, default=OUT_REPORT)
    args = parser.parse_args()

    fixtures = load_fixtures()
    observed_digest = content_digest(
        [row for split in ("train", "calibration", "sealed") for row in fixtures[split]]
    )
    data_report = json.loads((PROJECT_ROOT / DATA_REPORT).read_text(encoding="utf-8"))
    digest_match = data_report["corpus_digest"] == observed_digest

    results: OrderedDict[str, Any] = OrderedDict()
    if args.predictions is not None:
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        for split in ("train", "calibration", "sealed"):
            results[split] = score_records(
                fixtures[split], lambda record: predictions[record["id"]]
            )
        label = "predictions"
    else:
        policies = build_policies()
        for name, predict in policies.items():
            results[name] = {
                split: score_records(fixtures[split], predict)
                for split in ("train", "calibration", "sealed")
            }
        results["reference_key"] = {
            split: score_records(fixtures[split], reference_answer)
            for split in ("train", "calibration", "sealed")
        }
        label = "policies"

    gates: dict[str, Any] = {}
    if label == "policies":
        policy_macros = {
            name: payload["train"]["flip_pairwise_macro"]
            for name, payload in results.items()
            if name != "reference_key"
        }
        gates["static_policy_pairwise_macro_le_0_25"] = {
            "passed": all(m <= POLICY_GATE_MAX for m in policy_macros.values()),
            "per_policy_train": policy_macros,
            "threshold": POLICY_GATE_MAX,
        }
        reference_all = results["reference_key"]
        gates["reference_key_all_groups_correct"] = {
            "passed": all(
                reference_all[split]["per_class"][cls]["pair_rate"] == 1.0
                for split in ("train", "calibration", "sealed")
                for cls in GROUP_CLASSES
            ),
            "train_flip_macro": reference_all["train"]["flip_pairwise_macro"],
        }
    gates["corpus_digest_match"] = {"passed": bool(digest_match)}

    all_passed = all(gate["passed"] for gate in gates.values())
    report = {
        "format": REPORT_FORMAT,
        "version": 1,
        "contract": "plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_DRAFT_20260919.md",
        "mode": label,
        "corpus_digest": observed_digest,
        "gates": gates,
        "results": results,
        "outcome": "passed" if all_passed else "failed",
    }
    out_path = PROJECT_ROOT / args.report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": label,
                "outcome": report["outcome"],
                "train_flip_macros": {
                    name: round(payload["train"]["flip_pairwise_macro"], 4)
                    for name, payload in results.items()
                    if name != "reference_key"
                }
                if label == "policies"
                else None,
                "reference_train_macro": (
                    results["reference_key"]["train"]["flip_pairwise_macro"]
                    if label == "policies"
                    else None
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
