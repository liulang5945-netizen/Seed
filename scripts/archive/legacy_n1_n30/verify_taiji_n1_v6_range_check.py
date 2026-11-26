"""PLAN-N1-93 §3：牺牲子集上的动态范围检验（先测尺，再谈能力）。

只有 `sacrifice` 那 20 题会被查询；`confirm` 那 20 题在本件出版 `range_check_passed=true`
之前一次都不查——这条不是纪律口号，而是 `queried_ids()` 里的一枚硬守卫：任何落在确认子集里的
id 都让本件 rc=2 且不查询。九枚端点的装载与解码全部复用既有链路（90 的执行器与 85 的 measure），
本件只写"线的算法"和"子集的把关"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation as base85,
)
from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation_run89 as bind,
)

FACE = ROOT / "plans/manifests/n1_fresh_confirmation_v6_20261017.json"
CHECKS = ROOT / "reports/taiji_n1_v6_face_construction_checks_20261017.json"
OUT = ROOT / "reports/taiji_n1_v6_range_check.json"
ARMS = ("position", "terminal", "cut")
SEEDS = (11, 12, 13)
KINDS = ("lookup", "revision", "transition", "unknown")
CLASS_ERROR_FLOOR = 3
TRIPLE_ERROR_FLOOR = 6
MIN_USABLE_ENDPOINTS = 6
CPU_LIMIT_SECONDS = 180
#: 本件不控制生成预算：真正生效的那一位住在 `measure()` 里，点名它而不是再造一个同名常量。
GENERATION_CAP_SITE = "scripts/training/eval_taiji_n1_transition_factorial_confirmation.py:114"


class Refuse(Exception):
    """一条准入条件不成立：rc=2，不查询任何端点。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_face() -> dict:
    require(FACE.is_file(), f"signed v6 face is missing: {FACE}")
    require(CHECKS.is_file(), f"v6 construction checks are missing: {CHECKS}")
    face = json.loads(FACE.read_text(encoding="utf-8"))
    report = json.loads(CHECKS.read_text(encoding="utf-8"))
    require(report.get("face_sha256") == sha(FACE), "the v6 face differs from its check report")
    require(
        report.get("accepted") is True and report.get("queries") == 0,
        "the v6 construction checks did not pass cleanly",
    )
    subsets = face.get("subsets")
    require(isinstance(subsets, dict), "the v6 face carries no subset partition")
    require(
        sorted(subsets["sacrifice"]) and sorted(subsets["confirm"]),
        "both subsets must be non-empty",
    )
    require(
        not set(subsets["sacrifice"]) & set(subsets["confirm"]),
        "the two subsets overlap: the sacrifice half would stop being sacrificial",
    )
    ids = {episode["id"] for episode in face["episodes"]}
    require(
        ids == set(subsets["sacrifice"]) | set(subsets["confirm"]),
        "subset id lists do not cover the face exactly",
    )
    return face


def queried_ids(face: dict) -> list:
    """把关：只允许牺牲子集被查询。"""
    confirm = set(face["subsets"]["confirm"])
    chosen = [
        episode for episode in face["episodes"] if episode["id"] in face["subsets"]["sacrifice"]
    ]
    require(bool(chosen), "the sacrifice subset is empty")
    require(
        not any(episode["id"] in confirm for episode in chosen),
        "a confirmation-subset item leaked into the range check",
    )
    return [episode["id"] for episode in chosen]


def range_lines(results: list) -> dict:
    """只有"合计错误够多的端点枚数"这一条 gate `passed`；逐类那条现仅出版，不 gate（PLAN-N1-93 §3 文句与仪器口径待 owner 裁定）。"""
    per_class = {}
    for kind in KINDS:
        errors = [
            sum(1 for row in entry["rows"] if row["kind"] == kind and not row["free_exact"])
            for entry in results
        ]
        per_class[kind] = {
            "mean_errors": round(sum(errors) / len(errors), 4) if errors else None,
            "endpoints_over_floor": sum(1 for value in errors if value >= CLASS_ERROR_FLOOR),
        }
    triple = {
        (entry["seed"], entry["arm"]): sum(
            1 for row in entry["rows"] if row["kind"] != "unknown" and not row["free_exact"]
        )
        for entry in results
    }
    decodable = sum(1 for row in results[0]["rows"] if row["kind"] != "unknown") if results else 0
    over = {key: value for key, value in triple.items() if value >= TRIPLE_ERROR_FLOOR}
    usable = len(over)
    passed = (
        bool(results) and len(results) == len(SEEDS) * len(ARMS) and usable >= MIN_USABLE_ENDPOINTS
    )
    by_arm = {
        arm: sum(1 for (_seed, a) in over if a == arm) for arm in ("position", "terminal", "cut")
    }
    return {
        "per_class": per_class,
        "endpoints_with_triple_errors": usable,
        "endpoints_total": len(results),
        "decodable_items_per_endpoint": decodable,
        "range_check_passed": passed,
        "usable_by_arm": by_arm,
        "usable_competent_endpoints": by_arm["position"] + by_arm["cut"],
        "enforced_conditions": [
            f"endpoints with >= {TRIPLE_ERROR_FLOOR}/{decodable} decodable errors must number >= {MIN_USABLE_ENDPOINTS} of {len(SEEDS) * len(ARMS)}"
        ],
        "reported_not_enforced": [
            f"per-class mean native errors >= {CLASS_ERROR_FLOOR} on {decodable // 3 if decodable else 0} items per class"
        ],
        "lines": {
            "class_error_floor": CLASS_ERROR_FLOOR,
            "triple_error_floor": TRIPLE_ERROR_FLOOR,
            "min_usable_endpoints": MIN_USABLE_ENDPOINTS,
        },
    }


class Deadline:
    """85 执行器要求一个带 `guard(reserve)` 的预算对象；这里只做同形的截止检查。"""

    def __init__(self, started: float, seconds: float):
        self.expiry = started + seconds

    def guard(self, reserve: int = 0) -> None:
        if time.process_time() + reserve >= self.expiry:
            raise Refuse("the CPU reserve for the range check was reached")


def measure_endpoint(seed: int, arm: str, path: Path, items: list, budget: Deadline) -> dict:
    """整条测量链走 85 的 `evaluate`（含会话清理与"参数未变"的收尾核验），本件不重写。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = bind.load_with_family_protocol(payload)
    rows = base85.evaluate(model, optimizer, seed, arm, items, budget)
    require(len(rows) == len(items), f"seed {seed} {arm} returned {len(rows)} of {len(items)} rows")
    return {
        "seed": seed,
        "arm": arm,
        "rows": [
            {"id": row["id"], "kind": row["kind"], "free_exact": row["free_exact"]} for row in rows
        ],
    }


def run_readiness() -> int:
    face = load_face()
    ids = queried_ids(face)
    receipts = bind.family_receipts()
    endpoints, _guards = bind.endpoints(receipts)
    print(
        json.dumps(
            {
                "mode": "readiness",
                "sacrifice_items": len(ids),
                "confirm_items": len(face["subsets"]["confirm"]),
                "confirm_queried": False,
                "endpoints": sorted(f"{seed}-{arm}" for (seed, arm) in endpoints),
                "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_approved() -> int:
    started = time.process_time()
    require(not OUT.exists(), f"range check output already exists: {OUT}")
    face = load_face()
    wanted = set(queried_ids(face))
    items = [episode for episode in face["episodes"] if episode["id"] in wanted]
    receipts = bind.family_receipts()
    endpoints, _guards = bind.endpoints(receipts)
    budget = Deadline(started, CPU_LIMIT_SECONDS)
    results = []
    for seed in SEEDS:
        for arm in ARMS:
            results.append(measure_endpoint(seed, arm, endpoints[(seed, arm)], items, budget))
    endpoints_payload = []
    for entry in results:
        errors = sum(1 for row in entry["rows"] if not row["free_exact"])
        by_kind = {
            kind: sum(1 for row in entry["rows"] if row["kind"] == kind and not row["free_exact"])
            for kind in KINDS
        }
        require(
            sum(by_kind.values()) == errors,
            f"seed {entry['seed']} {entry['arm']}: the class split misses an error",
        )
        endpoints_payload.append(
            {
                "seed": entry["seed"],
                "arm": entry["arm"],
                "free_exact": len(entry["rows"]) - errors,
                "errors": errors,
                "errors_by_kind": by_kind,
                "addressable_errors": errors - by_kind["unknown"],
            }
        )
    adjudication = range_lines(results)
    recomputed = sum(
        1 for item in endpoints_payload if item["addressable_errors"] >= TRIPLE_ERROR_FLOOR
    )
    require(
        recomputed == adjudication["endpoints_with_triple_errors"],
        f"the published columns reproduce {recomputed} usable endpoints while the "
        f"adjudication says {adjudication['endpoints_with_triple_errors']}",
    )
    payload = {
        "format": "taiji-n1-93-v6-range-check-v2",
        "face": str(FACE),
        "face_sha256": sha(FACE),
        "subsets_queried": ["sacrifice"],
        "confirm_subset_queried": False,
        "range_check_is_capability_claim": False,
        "parameter_updates": 0,
        "items_queried": len(items) * len(results),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": round(time.process_time() - started, 4),
        "generation_cap_site": GENERATION_CAP_SITE,
        "endpoints": endpoints_payload,
        "adjudication": adjudication,
    }
    payload["status"] = (
        "passed" if payload["adjudication"]["range_check_passed"] else "range_not_met"
    )
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "mode": "approved",
                "status": payload["status"],
                "cpu_seconds": payload["cpu_seconds"],
                "adjudication": payload["adjudication"],
                "out": str(OUT),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if payload["adjudication"]["range_check_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readiness", action="store_true", help="只核题面与九枚端点，不查询")
    parser.add_argument("--execute-approved", action="store_true", help="牺牲子集范围检验")
    arguments = parser.parse_args(argv)
    try:
        require(
            arguments.readiness != arguments.execute_approved,
            "choose exactly one of --readiness/--execute-approved",
        )
        if arguments.readiness:
            return run_readiness()
        return run_approved()
    except Refuse as refusal:
        print(f"REFUSE: {refusal}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
