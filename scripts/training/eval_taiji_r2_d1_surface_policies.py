"""Score the frozen surface-policy battery on the R2-D1 dev fixture.

Contract: plans/reference/M5_R2_D1_DEV_MEASUREMENT_CONTRACT_FROZEN_20260917.md

Policies read question-visible surface information only (never the deciding
background content) plus answer frequencies on the train split.  Every policy
is deterministic.  b3_oracle_shape consumes gold shape labels and is a
diagnostic arm only: it is reported but never enters a gate and must never be
juxtaposed with model scores as an acceptable baseline.

Metrics M1-M5 and qualification gates Q1-Q5 follow contract sections 3 and 5.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_r2_d1_measurement_fixture import (  # noqa: E402
    MATERIAL_CLAUSE,
    POOLS,
    SHAPES,
    reference_answer,
)
from taiji.internalization import content_digest  # noqa: E402

FIXTURE = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
DATA_REPORT = Path("reports/r2_d1_data_contract_20260917.json")
OUT_REPORT = Path("reports/r2_d1_surface_baselines_20260917.json")
REPORT_FORMAT = "r2-d1-surface-baselines-v1"

CONTENT_GROUPS = tuple(s for s in SHAPES if s not in ("unknown", "same_opening_unknown"))
ACCEPTABLE = ("b0_global", "b1_stem", "b2_template")
DIAGNOSTIC = ("b3_oracle_shape",)


def load_fixture() -> dict[str, list[dict[str, Any]]]:
    path = PROJECT_ROOT / FIXTURE
    raw: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        raw.setdefault(record["split"], []).append(record)
    if set(raw) != {"train", "dev", "final"}:
        raise RuntimeError(f"fixture missing splits: {sorted(raw)}")
    return raw


def slot_normalize(text: str, split: str) -> str:
    """Replace every entity/color token with a generic slot.

    Same-color vs different-color combo materials collapse to one string under
    this normalization ({o1}是{c}，{o2}是{c} and {o1}是{c1}，{o2}是{c2} both
    become {O}是{C}，{O}是{C}), which is what denies a surface policy the
    same/different content distinction (contract section 4, b2).
    """

    normalized = text
    for token in sorted(POOLS[split]["colors"], key=len, reverse=True):
        normalized = normalized.replace(token, "{C}")
    for token in sorted(POOLS[split]["objects"], key=len, reverse=True):
        normalized = normalized.replace(token, "{O}")
    return normalized


UNKNOWN_MATERIAL_MARKERS = ("没有关于", "的信息缺失", "的资料为空")


def stem_class(record: dict[str, Any]) -> str:
    """Question-stem class visible without reading any background content."""

    stem = MATERIAL_CLAUSE.split(record["prefix"], maxsplit=1)[0]
    judged = "请判断" in stem
    # combo stems themselves contain 的颜色, so the sameness question must be
    # recognized before the color-query family.
    if ("相同" in stem) or ("一致" in stem):
        return "combo"
    if ("的颜色" in stem) or ("是什么颜色" in stem):
        return "query_judge" if judged else "query_plain"
    if ("吗" in stem) or ("是否为" in stem) or ("属于" in stem):
        return "negation"
    raise RuntimeError(f"unclassified stem: {record['prefix']}")


def template_class(record: dict[str, Any]) -> str:
    """Full sentence-template class after generic slot normalization."""

    stem, material = MATERIAL_CLAUSE.split(record["prefix"], maxsplit=1)
    normalized_material = slot_normalize(material, record["split"])
    judged = "请判断" in stem
    if ("相同" in stem) or ("一致" in stem):
        # same/different share one normalized material template
        return "combo"
    if ("的颜色" in stem) or ("是什么颜色" in stem):
        if any(marker in normalized_material for marker in UNKNOWN_MATERIAL_MARKERS):
            return "unknown_judge" if judged else "unknown_plain"
        return "fact_judge" if judged else "fact_plain"
    return "negation"


Classifier = Callable[[dict[str, Any]], str]


def class_constant(rows: list[dict[str, Any]], classifier: Classifier, cls: str) -> str:
    """Most frequent answer within one surface class; ties are broken by the
    first occurrence in the canonical (fixture-file) train order."""

    members = [row for row in rows if classifier(row) == cls]
    counter: Counter[str] = Counter(row["response"] for row in members)
    best = max(counter.values())
    winners = {answer for answer, count in counter.items() if count == best}
    for row in members:
        if row["response"] in winners:
            return str(row["response"])
    raise RuntimeError(f"empty frequency table for class {cls}")


def build_policies(train: list[dict[str, Any]]) -> dict[str, Callable[[dict[str, Any]], str]]:
    # b0: single global most frequent train answer
    global_counts: Counter[str] = Counter(r["response"] for r in train)
    best = max(global_counts.values())
    global_winners = {answer for answer, count in global_counts.items() if count == best}
    b0_answer: str = next(r["response"] for r in train if r["response"] in global_winners)

    # b1: per question-stem class; color-query stems are preset to 未知
    stem_classes = sorted({stem_class(record) for record in train})
    stem_table = {cls: class_constant(train, stem_class, cls) for cls in stem_classes}

    # b2: per full sentence-template class (combo same/diff merged)
    tmpl_classes = sorted({template_class(record) for record in train})
    tmpl_table = {cls: class_constant(train, template_class, cls) for cls in tmpl_classes}

    def b0(_: dict[str, Any]) -> str:
        return b0_answer

    def b1(record: dict[str, Any]) -> str:
        cls = stem_class(record)
        if cls in ("query_plain", "query_judge"):
            return "未知"
        return stem_table.get(cls, b0_answer)

    def b2(record: dict[str, Any]) -> str:
        return tmpl_table.get(template_class(record), b0_answer)

    def b3(record: dict[str, Any]) -> str:
        shape = record["shape"]
        if shape in ("unknown", "same_opening_unknown"):
            return "未知"
        if shape == "combination_same":
            return "相同"
        if shape == "combination_different":
            return "不同"
        cls = {
            "fact": "fact_plain",
            "same_opening_fact": "fact_judge",
            "negation": "negation",
        }[shape]
        return tmpl_table.get(cls, b0_answer)

    return {"b0_global": b0, "b1_stem": b1, "b2_template": b2, "b3_oracle_shape": b3}


def score_items(
    dev: list[dict[str, Any]], predict: Callable[[dict[str, Any]], str]
) -> list[dict[str, Any]]:
    rows = []
    for record in dev:
        prediction = predict(record).rstrip()
        rows.append(
            {
                "id": record["id"],
                "shape": record["shape"],
                "source_group": record["source_group"],
                "content_dependent": record["content_dependent"],
                "pair_id": record["pair_id"],
                "pair_type": record["pair_type"],
                "gold": record["response"],
                "prediction": prediction,
                "correct": prediction == record["response"],
            }
        )
    return rows


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    per_group: dict[str, dict[str, Any]] = {}
    for group in SHAPES:
        members = [r for r in rows if r["source_group"] == group]
        hits = sum(1 for r in members if r["correct"])
        per_group[group] = {
            "correct": hits,
            "denominator": len(members),
            "exact": hits / len(members) if members else None,
        }
    macro = sum(g["exact"] for g in per_group.values() if g["exact"] is not None) / len(
        [g for g in per_group.values() if g["exact"] is not None]
    )
    content_rows = [r for r in rows if r["content_dependent"]]
    content_hits = sum(1 for r in content_rows if r["correct"])
    content_per_group: dict[str, dict[str, Any]] = {}
    for group in CONTENT_GROUPS:
        members = [r for r in content_rows if r["source_group"] == group]
        hits = sum(1 for r in members if r["correct"])
        content_per_group[group] = {
            "correct": hits,
            "denominator": len(members),
            "exact": hits / len(members) if members else None,
        }

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["pair_id"]:
            by_pair.setdefault(row["pair_id"], []).append(row)

    flip_pairs = [
        (pid, members)
        for pid, members in by_pair.items()
        if members and members[0]["pair_type"] in ("fact_flip", "combo_flip")
    ]
    inv_pairs = [
        (pid, members)
        for pid, members in by_pair.items()
        if members and members[0]["pair_type"] == "invariance"
    ]

    def pair_rate(members: list[dict[str, Any]], need_same: bool) -> bool:
        if len(members) != 2 or not all(bool(m["correct"]) for m in members):
            return False
        same = members[0]["prediction"] == members[1]["prediction"]
        return bool(same) == need_same

    flip_hits = sum(1 for _, members in flip_pairs if pair_rate(members, need_same=False))
    inv_hits = sum(1 for _, members in inv_pairs if pair_rate(members, need_same=True))
    return {
        "M1_exact": {"numerator": correct, "denominator": total, "value": correct / total},
        "M2_shape_macro": {"value": macro},
        "M3_content_exact": {
            "numerator": content_hits,
            "denominator": len(content_rows),
            "value": content_hits / len(content_rows),
        },
        "M4_flip_pair": {
            "numerator": flip_hits,
            "denominator": len(flip_pairs),
            "value": flip_hits / len(flip_pairs),
        },
        "M5_inv_pair": {
            "numerator": inv_hits,
            "denominator": len(inv_pairs),
            "value": inv_hits / len(inv_pairs),
        },
        "per_group_exact": per_group,
        "content_per_group_exact": content_per_group,
    }


def git_identity() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": run(["rev-parse", "HEAD"]), "dirty": bool(run(["status", "--porcelain"]))}


def main() -> int:
    fixture = load_fixture()
    train, dev = fixture["train"], fixture["dev"]
    observed_digest = content_digest(fixture["train"] + fixture["dev"] + fixture["final"])

    data_report_path = PROJECT_ROOT / DATA_REPORT
    data_report = json.loads(data_report_path.read_text(encoding="utf-8"))
    digest_match = data_report["corpus_digest"] == observed_digest
    q1 = data_report["outcome"] == "passed" and digest_match

    policies = build_policies(train)
    results: OrderedDict[str, Any] = OrderedDict()
    for name, predict in policies.items():
        rows = score_items(dev, predict)
        results[name] = {"items": rows, "metrics": metrics(rows)}

    reference_rows = score_items(dev, reference_answer)
    results["reference_key"] = {"items": reference_rows, "metrics": metrics(reference_rows)}

    def value(name: str, metric: str) -> float:
        return float(results[name]["metrics"][metric]["value"])

    acceptable = {
        name: {m: value(name, m) for m in ("M1_exact", "M3_content_exact", "M4_flip_pair")}
        for name in ACCEPTABLE
    }
    ceiling_m1 = max(v["M1_exact"] for v in acceptable.values())
    ceiling_m3 = max(v["M3_content_exact"] for v in acceptable.values())
    ceiling_m4 = max(v["M4_flip_pair"] for v in acceptable.values())

    ref = results["reference_key"]["metrics"]
    gates = {
        "Q1_structure": {
            "passed": bool(q1),
            "data_report_outcome": data_report["outcome"],
            "corpus_digest_match": digest_match,
        },
        "Q2_overall_ceiling": {"passed": ceiling_m1 <= 0.50, "max_M1": ceiling_m1},
        "Q3_content_ceiling": {"passed": ceiling_m3 <= 0.50, "max_M3": ceiling_m3},
        "Q4_flip_structure": {"passed": ceiling_m4 == 0.0, "max_M4": ceiling_m4},
        "Q5_reference_key": {
            "passed": (
                ref["M1_exact"]["value"] == 1.0
                and ref["M3_content_exact"]["value"] == 1.0
                and ref["M4_flip_pair"]["value"] == 1.0
                and ref["M5_inv_pair"]["value"] == 1.0
            ),
            "M1": ref["M1_exact"]["value"],
            "M3": ref["M3_content_exact"]["value"],
            "M4": ref["M4_flip_pair"]["value"],
            "M5": ref["M5_inv_pair"]["value"],
        },
    }
    all_passed = all(gate["passed"] for gate in gates.values())

    report = {
        "format": REPORT_FORMAT,
        "version": 1,
        "corpus_digest": observed_digest,
        "data_report": str(DATA_REPORT),
        "fixture": str(FIXTURE),
        "generator_git": git_identity(),
        "acceptable_policies": list(ACCEPTABLE),
        "diagnostic_arms": list(DIAGNOSTIC) + ["reference_key"],
        "note": "b3_oracle_shape consumes gold shape labels: diagnostic only, "
        "never enters gates and is never comparable to model scores.",
        "acceptable_ceiling": {
            "M1_exact": ceiling_m1,
            "M3_content_exact": ceiling_m3,
            "M4_flip_pair": ceiling_m4,
        },
        "gates": gates,
        "policies": results,
        "growth_admitted": False,
        "can_promote": False,
        "outcome": "passed" if all_passed else "failed",
    }
    out_path = PROJECT_ROOT / OUT_REPORT
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outcome": report["outcome"],
                "ceiling": report["acceptable_ceiling"],
                "reference": {k: gates["Q5_reference_key"][k] for k in ("M1", "M3", "M4", "M5")},
            },
            ensure_ascii=False,
        )
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
