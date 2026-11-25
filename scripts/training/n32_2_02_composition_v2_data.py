"""N32.2.02 (zero training): data contract v2 for R1 -- scenes, shortcut baselines, gates.

Pre-registered: plans/reference/DESIGN-N32-2-02_20261125.md (D-20261125-509).
  1. generate 8,000 worlds once (``taiji.native_composition_v2.write_scenes``) unless the
     stored files already exist; record digests in the contract manifest;
  2. compute the corrected shortcut set per question kind (reference solutions excluded;
     lookup-table rules are fitted on train and scored on val);
  3. gate_k = max(0.85, best shortcut on val + 0.15); presentation fails if any best
     shortcut > 0.70 or any gate > 1.0.  On presentation failure the line closes
     (DESIGN-N32-2-00 §5 as amended by D-509) -- no v3.

Usage: python scripts/training/n32_2_02_composition_v2_data.py
Outputs: reports/n32_2_02_data.json,
         plans/manifests/taiji_23_composition_v2_contract_20261125.json
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.n31_guard import preflight  # noqa: E402
from taiji import composition_encoding as ce  # noqa: E402
from taiji import native_composition_v2 as nc2  # noqa: E402
from taiji.native_composition import DIRECTIONS, _relation, _relative_position  # noqa: E402

CARD = "N32.2.02"
REPORTS = ROOT / "reports"
MANIFEST = ROOT / "plans/manifests/taiji_23_composition_v2_contract_20261125.json"
GATE_FLOOR = 0.85
GATE_OVER_RULE = 0.15
PRESENTATION_MAX_RULE = 0.70
V1_POOL_SEEDS = (ce.TRAIN_SEED_BASE, ce.TRAIN_SEED_BASE + 12_000)
Rule = Callable[[dict[str, Any]], str]


def _xy(body: dict[str, Any]) -> tuple[float, float]:
    return float(body["position"][0]), float(body["position"][1])


def _others(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in row["scene"]["bodies"] if b["name"] != row["entity"]]


def _dist(row: dict[str, Any], body: dict[str, Any]) -> float:
    me = next(b for b in row["scene"]["bodies"] if b["name"] == row["entity"])
    (ax, ay), (bx, by) = _xy(me), _xy(body)
    return math.hypot(ax - bx, ay - by)


def _first_tick_body(row: dict[str, Any]) -> str:
    bodies = row["scene"]["bodies"]
    return str(bodies[ce.scene_order(int(row["scene"]["seed"]), len(bodies))[0]]["name"])


def _third(row: dict[str, Any]) -> str:
    first = row.get("first_hop")
    rest = [b["name"] for b in _others(row) if b["name"] != first]
    return str(rest[0]) if first is not None and rest else "未知"


def fixed_rules(kind: str) -> dict[str, Rule]:
    rules: dict[str, Rule] = {
        "always_unknown": lambda r: "未知",
        "copy_entity": lambda r: str(r["entity"]),
    }
    if kind == "q1":
        rules["position_of_first_tick_body"] = lambda r: _relative_position(
            _first_tick_body(r), r["scene"]["bodies"]
        )
        rules["always_middle"] = lambda r: "中间"
    if kind == "q2":
        rules["opposite_direction"] = lambda r: (
            _relation(r["entity"], nc2.OPPOSITE[r["axes"]["d"]], r["scene"]["bodies"]) or "未知"
        )
        rules["nearest_other"] = lambda r: str(min(_others(r), key=lambda b: _dist(r, b))["name"])
        rules["farthest_other"] = lambda r: str(max(_others(r), key=lambda b: _dist(r, b))["name"])
    if kind == "q3":
        rules["exist_d1_then_copy"] = lambda r: (
            str(r["entity"]) if r["first_hop"] is not None else "未知"
        )
        rules["first_hop_only"] = lambda r: str(r["first_hop"] or "未知")
        rules["second_hop_from_entity"] = lambda r: (
            _relation(r["entity"], r["axes"]["pair"][1], r["scene"]["bodies"]) or "未知"
        )
        rules["third_body_if_d1"] = _third
    return rules


def _role(row: dict[str, Any], name: str) -> str:
    if name == "未知":
        return "U"
    if name == row["entity"]:
        return "X"
    if name == row.get("first_hop"):
        return "F1"
    return "T"


def _from_role(row: dict[str, Any], role: str) -> str:
    return {
        "U": "未知",
        "X": str(row["entity"]),
        "F1": str(row.get("first_hop") or "未知"),
        "T": _third(row),
    }[role]


def _exists(row: dict[str, Any], d: str) -> bool:
    return _relation(row["entity"], d, row["scene"]["bodies"]) is not None


def table_keys(kind: str) -> dict[str, Callable[[dict[str, Any]], Any]]:
    if kind != "q3":
        return {}
    return {
        "role_by_pair": lambda r: tuple(r["axes"]["pair"]),
        "role_by_pair_exist_d1": lambda r: (tuple(r["axes"]["pair"]), r["first_hop"] is not None),
        "role_by_pair_exist_d1_exist_d2_from_x": lambda r: (
            tuple(r["axes"]["pair"]),
            r["first_hop"] is not None,
            _exists(r, r["axes"]["pair"][1]),
        ),
    }


def diagnostic_keys() -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Not a shortcut: X's full one-hop signature (four Q2 facts) -- report only."""
    return {
        "role_by_pair_and_x_one_hop_signature": lambda r: (
            tuple(r["axes"]["pair"]),
            tuple(
                _role(r, _relation(r["entity"], d, r["scene"]["bodies"]) or "未知")
                for d in DIRECTIONS
            ),
        )
    }


def fit_table(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[Any, str]:
    tab: dict[Any, Counter[str]] = defaultdict(Counter)
    for r in rows:
        tab[key(r)][_role(r, r["answer"])] += 1
    return {k: c.most_common(1)[0][0] for k, c in tab.items()}


def score(rows: list[dict[str, Any]], pred: Rule) -> float:
    return sum(pred(r) == r["answer"] for r in rows) / len(rows)


def baselines(train: list[dict[str, Any]], val: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in nc2.KINDS:
        tr = [r for r in train if r["kind"] == kind]
        va = [r for r in val if r["kind"] == kind]
        majority = Counter(r["answer"] for r in tr).most_common(1)[0][0]
        rules: dict[str, float] = {"majority": score(va, lambda r, m=majority: m)}
        for name, rule in fixed_rules(kind).items():
            rules[name] = score(va, rule)
        for name, key in table_keys(kind).items():
            tab = fit_table(tr, key)
            rules[name] = score(va, lambda r, t=tab, k=key: _from_role(r, t.get(k(r), "U")))
        diag: dict[str, float] = {}
        if kind == "q3":
            for name, key in diagnostic_keys().items():
                tab = fit_table(tr, key)
                diag[name] = score(va, lambda r, t=tab, k=key: _from_role(r, t.get(k(r), "U")))
        best = max(rules, key=lambda k: rules[k])
        out[kind] = {
            "n_train": len(tr),
            "n_val": len(va),
            "answer_distribution_val": {
                k: v / len(va) for k, v in Counter(r["answer"] for r in va).most_common()
            },
            "shortcuts": rules,
            "best_shortcut": best,
            "best_shortcut_acc": rules[best],
            "diagnostic_not_shortcut": diag,
        }
    return out


def main() -> int:
    design = preflight(CARD, [], record=False)
    started = time.process_time()
    if nc2.SCENES_PATH.exists() and nc2.SEALED_PATH.exists():
        generated = False
    else:
        nc2.write_scenes()
        generated = True
    open_text = nc2.SCENES_PATH.read_text(encoding="utf-8")
    sealed_bytes = nc2.SEALED_PATH.read_bytes()
    digests = {
        "open_file_sha256": hashlib.sha256(open_text.encode("utf-8")).hexdigest(),
        "confirm_file_sha256": hashlib.sha256(sealed_bytes).hexdigest(),
    }
    gen_cpu = round(time.process_time() - started, 2)
    train = nc2.rows("train")
    val = nc2.rows("val")
    seeds = [int(s["seed"]) for s in json.loads(open_text)]
    lengths = [len(r["scene"]["bodies"]) + len(r["text"]) for r in [*train[:5000], *val]]
    frozen_hits = [s for s in seeds if ce.frozen_seed(s)]
    v1_hits = [s for s in seeds if V1_POOL_SEEDS[0] <= s < V1_POOL_SEEDS[1]]
    unknown_answers = sorted({r["answer"] for r in [*train, *val]} - set(ce.ANSWER_INDEX))
    base = baselines(train, val)
    gates = {k: max(GATE_FLOOR, v["best_shortcut_acc"] + GATE_OVER_RULE) for k, v in base.items()}
    too_easy = {
        k: v["best_shortcut_acc"]
        for k, v in base.items()
        if v["best_shortcut_acc"] > PRESENTATION_MAX_RULE
    }
    impossible = {k: g for k, g in gates.items() if g > 1.0}
    if frozen_hits or v1_hits or unknown_answers or max(lengths) > ce.HORIZON:
        exit_ = "data_invalid"
    elif too_easy or impossible:
        exit_ = "presentation_failure"
    else:
        exit_ = "data_ready"
    criteria = {
        "fixture": nc2.FIXTURE_ID,
        "encoding": ce.ENCODING_VERSION,
        "world_seed_base": nc2.WORLD_SEED_BASE,
        "splits": {k: list(v) for k, v in nc2.SPLITS.items()},
        "pairs": [list(p) for p in nc2.PAIRS],
        "templates": nc2.TEMPLATES,
        "digests": digests,
        "gates_val": gates,
    }
    blob = json.dumps(criteria, sort_keys=True, ensure_ascii=False).encode("utf-8")
    manifest = {
        "schema": "taiji-23-composition-v2-contract-v1",
        "card": CARD,
        "status": "frozen" if exit_ == "data_ready" else exit_,
        "criteria": criteria,
        "criteria_sha256": hashlib.sha256(blob).hexdigest(),
        "confirm_reads": 0,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    report = {
        "schema": "taiji-n32-2-02-v1",
        "card": CARD,
        "design": design.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "generated_this_run": generated,
        "counts": {"train_rows": len(train), "val_rows": len(val), "worlds": len(seeds)},
        "max_ticks": max(lengths),
        "baselines": base,
        "gates_val": gates,
        "presentation": {
            "max_shortcut_allowed": PRESENTATION_MAX_RULE,
            "too_easy": too_easy,
            "impossible_gate": impossible,
        },
        "leakage": {
            "frozen_23_1_seeds": len(frozen_hits),
            "v1_pool_seeds": len(v1_hits),
            "unknown_answers": unknown_answers,
        },
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "criteria_sha256": manifest["criteria_sha256"],
        "exit": exit_,
        "cost": {"cpu": round(time.process_time() - started, 2), "generation_cpu": gen_cpu},
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0, "training": False},
    }
    out = REPORTS / "n32_2_02_data.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best = {k: round(v["best_shortcut_acc"], 3) for k, v in base.items()}
    print(f"{CARD} exit={exit_} best_shortcut={best} gates={gates} cpu={report['cost']['cpu']}")
    return 0 if exit_ == "data_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
