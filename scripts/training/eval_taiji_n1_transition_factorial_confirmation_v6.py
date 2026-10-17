"""PLAN-N1-94：v6 确认子集的一次性查询与四门判级。

发射顺序被本文件的守卫锁死：题面必须已签署且与机检报告同源 → 牺牲子集范围检验必须已在库且
`range_check_passed=true` 且自证没查确认半 → 九枚端点与三族共享守卫一致 → 写靶此前不存在。
测量链不重抄：装载走 90 的族回执与 `load_with_family_protocol`，逐题解码走 85 的 `evaluate`
（自带会话清理与"参数／优化器／RNG／历史未变"的收尾核验）。本文件只写 94 的三件新东西：
分母适配后的四门、双口径来源摘要（把 DEBT-G147 那格假阴性拆掉）、以及一次性消费记账。
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
from scripts.training import (  # noqa: E402
    run_taiji_n1_native_evidence_paired as paired,
)
from scripts.training import (  # noqa: E402
    verify_taiji_n1_v6_range_check as range_gate,
)

PLAN = ROOT / "plans/reference/PLAN-N1-94_v6_confirmation_query_20261017.md"
FACE = ROOT / "plans/manifests/n1_fresh_confirmation_v6_20261017.json"
CHECKS = ROOT / "reports/taiji_n1_v6_face_construction_checks_20261017.json"
RANGE = ROOT / "reports/taiji_n1_v6_range_check.json"
REPORT = ROOT / "reports/taiji_n1_transition_factorial_confirmation_v6.json"
ATTEMPT = ROOT / "reports/taiji_n1_transition_factorial_confirmation_v6_attempt.json"

ARMS = ("position", "terminal", "cut")
SEEDS = (11, 12, 13)
KINDS = ("lookup", "revision", "transition", "unknown")
ITEMS_PER_MODEL = 20
CPU_LIMIT_SECONDS = 180
GENERATION_CAP_SITE = "scripts/training/eval_taiji_n1_transition_factorial_confirmation.py:114"
GATES = {
    "each_position_seed": 10,
    "each_position_kind": 2,
    "advantage_terminal": 12,
    "advantage_cut": 12,
}


#: 统一用一个拒绝类型：范围检验仪在前置里抛的就是它，别去造父子两个同名类——
#: 捕获子类接不住父类实例，那是最隐蔽的一种"守卫不生效"。
Refuse = range_gate.Refuse


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_normalized(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preconditions() -> dict:
    """P1–P5：任一缺失即响亮失败，且在写 attempt 之前。"""
    require(PLAN.is_file(), f"PLAN-N1-94 is missing: {PLAN}")
    face = range_gate.load_face()
    confirm_ids = list(face["subsets"]["confirm"])
    require(
        len(confirm_ids) == ITEMS_PER_MODEL, f"the confirm subset holds {len(confirm_ids)} items"
    )
    episodes = {episode["id"]: episode for episode in face["episodes"]}
    items = [episodes[item_id] for item_id in confirm_ids]
    require(
        all(episode["split"] == "holdout" for episode in items),
        "the confirm subset must be entirely held out",
    )
    require(RANGE.is_file(), f"the range check report is missing: {RANGE}")
    reading = read_json(RANGE)
    adjudication = reading.get("adjudication", {})
    require(
        adjudication.get("range_check_passed") is True,
        "the ruler was never shown to have range: no confirmation query is allowed",
    )
    require(
        reading.get("confirm_subset_queried") is False,
        "the range report says the confirm subset was already queried",
    )
    require(
        reading.get("subsets_queried") == ["sacrifice"],
        f"the range report did not restrict itself to the sacrifice subset: {reading.get('subsets_queried')}",
    )
    require(
        reading.get("face_sha256") == sha_raw(FACE), "the range check ran against a different face"
    )
    receipts = bind.family_receipts()
    endpoints, guards = bind.endpoints(receipts)
    conflicts = bind.cross_family_guard_conflicts(receipts)
    require(not conflicts, f"families disagree on a shared guarded input: {sorted(conflicts)[:3]}")
    require(len(endpoints) == 9, "nine frozen endpoints required")
    require(not REPORT.exists(), f"confirmation report already exists: {REPORT}")
    return {
        "items": items,
        "endpoints": endpoints,
        "guards": guards,
        "face": face,
        "range": reading,
    }


def source_guard_digests(paths: dict) -> dict:
    """双口径摘要：按各路径所属批次的 `evidence_sha256_policy` 判定，另一枚仅披露。

    判定规则（PLAN-N1-94 §3）：`.json` 按 CRLF 归一后的值与回执登记值比，其余按原始字节比；
    两口径都不等的路径才是"真实改动"，只有 raw 不等的路径一律归为口径差异并计数出版。
    """
    detail = {}
    judged_changed = 0
    convention_only = 0
    for raw_path, recorded in sorted(paths.items()):
        path = Path(raw_path)
        if not path.is_file():
            detail[raw_path] = {"present": False}
            judged_changed += 1
            continue
        raw = sha_raw(path)
        normalized = sha_normalized(path)
        normalized_only = path.suffix.lower() == ".json"
        judged = normalized if normalized_only else raw
        matches = judged == recorded
        raw_matches = raw == recorded
        if not matches:
            judged_changed += 1
        elif not raw_matches and normalized_only:
            convention_only += 1
        detail[raw_path] = {
            "sha_raw": raw,
            "sha_json_normalized": normalized,
            "convention_used": "crlf_normalized_to_lf" if normalized_only else "raw_bytes",
            "matches_under_convention": matches,
            "matches_under_raw": raw_matches,
        }
    return {
        "sources_unchanged": judged_changed == 0,
        "judged_mismatch_count": judged_changed,
        "convention_only_count": convention_only,
        "real_change_count": sum(
            1
            for entry in detail.values()
            if not entry.get("present", True)
            or (not entry["matches_under_convention"] and not entry["matches_under_raw"])
        ),
        "guards": detail,
    }


def evaluate_gates(rows: list) -> dict:
    per_cell: dict = {}
    for row in rows:
        per_cell.setdefault((row["seed"], row["arm"]), []).append(row)
    require(len(per_cell) == 9, f"expected nine cells, got {len(per_cell)}")
    by_arm = {arm: 0 for arm in ARMS}
    per_seed: dict = {}
    per_kind: dict = {}
    for (seed, arm), cell in sorted(per_cell.items()):
        require(len(cell) == ITEMS_PER_MODEL, f"seed {seed} {arm} queried {len(cell)} items")
        exact = sum(1 for row in cell if row["free_exact"])
        by_arm[arm] += exact
        per_seed[f"{seed}_{arm}"] = exact
        for kind in KINDS:
            hits = sum(1 for row in cell if row["kind"] == kind and row["free_exact"])
            per_kind.setdefault(f"{seed}_{arm}", {})[kind] = hits
    position_per_seed = {seed: per_seed[f"{seed}_position"] for seed in SEEDS}
    kind_cells = [
        (f"{seed}_position", kind, per_kind[f"{seed}_position"][kind])
        for seed in SEEDS
        for kind in KINDS
    ]
    position_total = sum(position_per_seed.values())
    advantages = {
        "terminal": position_total - by_arm["terminal"],
        "cut": position_total - by_arm["cut"],
    }
    gates = {
        "each_position_seed": {
            "values": position_per_seed,
            "line": GATES["each_position_seed"],
            "passed": all(
                value >= GATES["each_position_seed"] for value in position_per_seed.values()
            ),
        },
        "each_position_kind": {
            "values": {f"{seed}:{kind}": hits for seed, kind, hits in kind_cells},
            "line": GATES["each_position_kind"],
            "passed": all(hits >= GATES["each_position_kind"] for _s, _k, hits in kind_cells),
        },
        "advantage_terminal": {
            "value": advantages["terminal"],
            "line": GATES["advantage_terminal"],
            "denominator": ITEMS_PER_MODEL * len(SEEDS),
            "passed": advantages["terminal"] >= GATES["advantage_terminal"],
        },
        "advantage_cut": {
            "value": advantages["cut"],
            "line": GATES["advantage_cut"],
            "denominator": ITEMS_PER_MODEL * len(SEEDS),
            "passed": advantages["cut"] >= GATES["advantage_cut"],
        },
    }
    return {
        "gates": gates,
        "per_cell_free_exact": per_seed,
        "per_cell_free_by_kind": per_kind,
        "all_four_gates": all(block["passed"] for block in gates.values()),
    }


def run_readiness() -> int:
    started = time.process_time()
    context = preconditions()
    print(
        json.dumps(
            {
                "mode": "readiness",
                "confirm_items": len(context["items"]),
                "endpoints": sorted(f"{seed}-{arm}" for (seed, arm) in context["endpoints"]),
                "range_check_passed": True,
                "guards_total": len(context["guards"]),
                "cpu_limit_seconds": CPU_LIMIT_SECONDS,
                "preparation_cpu_seconds": round(time.process_time() - started, 4),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_approved() -> int:
    started = time.process_time()
    context = preconditions()
    ATTEMPT.write_text(
        json.dumps(
            {
                "format": "taiji-n1-94-v6-confirmation-attempt-v1",
                "started_process_time": started,
                "face_sha256_raw": sha_raw(FACE),
                "confirm_items": len(context["items"]),
                "range_report_sha256_raw": sha_raw(RANGE),
                "consumed_on_write": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    budget = range_gate.Deadline(started, CPU_LIMIT_SECONDS)
    rows = []
    accounting = {"confirmation_queries": 0}
    restore_checks = []
    status = "published"
    error = None
    try:
        for (seed, arm), path in sorted(context["endpoints"].items()):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model, optimizer = bind.load_with_family_protocol(payload)
            before = paired.fingerprint(model.bundle(optimizer))
            rows.extend(
                base85.evaluate(
                    model,
                    optimizer,
                    seed,
                    arm,
                    context["items"],
                    budget,
                    accounting=accounting,
                )
            )
            restore_checks.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "unchanged": before == paired.fingerprint(model.bundle(optimizer)),
                }
            )
    except Exception as failure:  # §3：中断也要如实出版 partial 行
        status = "interrupted"
        error = f"{type(failure).__name__}: {failure}"
    payload = {
        "format": "taiji-n1-94-v6-confirmation-v1",
        "plan": str(PLAN),
        "plan_sha256": sha_raw(PLAN),
        "face": str(FACE),
        "face_sha256_raw": sha_raw(FACE),
        "range_check": str(RANGE),
        "gate_source_of_truth": "PLAN-N1-94 §2 (unrescaled absolute values)",
        "gates_lines": GATES,
        "items_per_model": ITEMS_PER_MODEL,
        "generation_cap_note": GENERATION_CAP_SITE,
        "status": status,
        "error": error,
        "confirmation_queries": accounting["confirmation_queries"],
        "rows": rows,
        "parameter_updates": 0,
        "endpoint_neutrality": restore_checks,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": round(time.process_time() - started, 4),
        "unseen_claim_allowed": True,
        "consumed_face": "v6-confirm-20",
        "retry_allowed": False,
    }
    payload.update(source_guard_digests(context["guards"]))
    if rows:
        payload.update(evaluate_gates(rows))
        complete = {
            "rows_queried": len(rows) == ITEMS_PER_MODEL * 9,
            "queries_match_rows": payload["confirmation_queries"] == len(rows),
            "endpoint_neutrality_all_true": (
                all(entry["unchanged"] for entry in restore_checks) if restore_checks else False
            ),
            "status_published": status == "published",
        }
        payload["complete_checks"] = complete
        payload["behavioral_pass"] = payload["all_four_gates"] and all(complete.values())
        payload["can_promote"] = False
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "mode": "approved",
                "status": status,
                "queries": payload["confirmation_queries"],
                "gates": {
                    name: block.get("passed", block.get("value"))
                    for name, block in payload.get("gates", {}).items()
                },
                "behavioral_pass": payload.get("behavioral_pass"),
                "out": str(REPORT),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if status == "published" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readiness", action="store_true", help="只核前置，不装载模型、不查询")
    parser.add_argument("--execute-approved", action="store_true", help="一次性确认查询")
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
