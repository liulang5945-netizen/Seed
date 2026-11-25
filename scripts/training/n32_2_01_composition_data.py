"""N32.2.01 (zero training): 23.1 tick encoding, data pool, rule baselines and gates.

Pre-registered: plans/reference/DESIGN-N32-2-01_20261125.md (D-20261125-506).  Builds the
fixed training pool (fresh worlds disjoint from every frozen partition), the validation
set (frozen ``train`` partition) and the report-only fit set; computes per-kind answer
distributions and rule baselines; derives the per-kind validation gates with the rule
written in DESIGN-N32-2-00 §4 *before* any model is trained; checks presentation.

Usage: python scripts/training/n32_2_01_composition_data.py
Outputs: reports/n32_2_01_data.json, reports/n32_2_pool_v1.json (pool, sha256 in report)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.n31_guard import preflight  # noqa: E402
from taiji import composition_encoding as ce  # noqa: E402
from taiji.native_composition import NAMES, _relation, _relative_position  # noqa: E402

CARD = "N32.2.01"
REPORTS = ROOT / "reports"
POOL_WORLDS = 12_000
GATE_FLOOR = 0.85
GATE_OVER_RULE = 0.15
PRESENTATION_MAX_RULE = 0.70


def _entity(row: dict[str, Any]) -> str:
    return next(n for n in NAMES if n in row["text"])


def rule_predictions(row: dict[str, Any]) -> dict[str, str]:
    bodies = row["scene"]["bodies"]
    ent = _entity(row)
    preds = {
        "always_unknown": "未知",
        "copy_entity": ent,
        "position_of_entity": _relative_position(ent, bodies),
    }
    axes = row["axes"]
    if row["kind"] == "q2":
        d = axes["direction"]
        opp = {"左边": "右边", "右边": "左边", "前边": "后边", "后边": "前边"}[d]
        preds["opposite_direction"] = _relation(ent, opp, bodies) or "未知"
    if row["kind"] == "q3":
        d1, d2 = axes["pair_order"]
        preds["first_hop_only"] = _relation(ent, d1, bodies) or "未知"
        preds["second_hop_from_entity"] = _relation(ent, d2, bodies) or "未知"
    return preds


def baselines(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        dist = Counter(r["answer"] for r in sub)
        rules: dict[str, float] = {"majority": max(dist.values()) / len(sub)}
        names = set().union(*(rule_predictions(r).keys() for r in sub))
        for name in sorted(names):
            hits = [rule_predictions(r).get(name) == r["answer"] for r in sub]
            rules[name] = sum(hits) / len(sub)
        best = max(rules, key=lambda k: rules[k])
        out[kind] = {
            "n": len(sub),
            "answer_distribution": {k: v / len(sub) for k, v in dist.most_common()},
            "rules": rules,
            "best_rule": best,
            "best_rule_acc": rules[best],
        }
    return out


def main() -> int:
    design = preflight(CARD, [], record=False)
    started = time.process_time()
    pool = ce.training_rows(POOL_WORLDS)
    val = ce.validation_rows()
    fit = ce.frozen_rows("fit")
    pool_json = json.dumps(
        [
            {k: r[k] for k in ("id", "seed", "kind", "text", "answer", "axes", "scene")}
            for r in pool
        ],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    pool_path = REPORTS / "n32_2_pool_v1.json"
    pool_path.write_text(pool_json + "\n", encoding="utf-8")
    pool_sha = hashlib.sha256((pool_json + "\n").encode("utf-8")).hexdigest()
    lengths = [len(r["scene"]["bodies"]) + len(r["text"]) for r in [*pool, *val, *fit]]
    coords = [float(v) for r in pool for b in r["scene"]["bodies"] for v in list(b["position"])[:3]]
    base_val = baselines(val)
    gates = {k: max(GATE_FLOOR, v["best_rule_acc"] + GATE_OVER_RULE) for k, v in base_val.items()}
    impossible = {k: g for k, g in gates.items() if g > 1.0}
    weak = {
        k: v["best_rule_acc"]
        for k, v in base_val.items()
        if v["best_rule_acc"] > PRESENTATION_MAX_RULE
    }
    leak = sorted({r["world_digest"] for r in pool} & {r["world_digest"] for r in [*val, *fit]})
    frozen_seed_hits = [r["seed"] for r in pool if ce.frozen_seed(r["seed"])]
    if leak or frozen_seed_hits:
        exit_ = "data_invalid"
    elif weak or impossible:
        exit_ = "presentation_failure"
    else:
        exit_ = "data_ready"
    report = {
        "schema": "taiji-n32-2-01-v1",
        "card": CARD,
        "design": design.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "encoding": ce.describe(),
        "pool": {
            "worlds": POOL_WORLDS,
            "rows": len(pool),
            "sha256": pool_sha,
            "path": str(pool_path.relative_to(ROOT)),
        },
        "validation": {"source": "frozen train partition", "rows": len(val)},
        "fit_report_only": {"rows": len(fit)},
        "max_ticks": max(lengths),
        "coord_range": [min(coords), max(coords)],
        "baselines": {"pool": baselines(pool), "validation": base_val, "fit": baselines(fit)},
        "gates_validation": gates,
        "presentation": {
            "max_rule_allowed": PRESENTATION_MAX_RULE,
            "too_easy": weak,
            "impossible_gate": impossible,
        },
        "leakage": {
            "world_digest_overlap": len(leak),
            "frozen_seeds_in_pool": len(frozen_seed_hits),
        },
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "training": False},
    }
    out = REPORTS / "n32_2_01_data.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{CARD} exit={exit_} gates={gates} cpu={report['cost']['cpu']} -> {out.name}")
    return 0 if exit_ == "data_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
