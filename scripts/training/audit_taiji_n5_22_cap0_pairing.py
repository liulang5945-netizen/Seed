"""批次 C 两臂 CAP 读数的**配对器**（`PLAN-N5-22` 的 J-N5u-1／2／3 载体；判据先冻，㊵-773）。

它只读两份已落盘的 `eval_taiji_cap0_baseline.py` 件，**不重跑评测、不重抄生成链**
（[[dont-reimplement-the-chain]]）：能力数住在两份件里，这里只做"能不能配对＋逐题翻转"。

三件要紧的事：

* **资格先于方向**（J-N5u-1）：评测集 sha、`git_head`、`chain` 两枚键、`scoring_discipline` 五条、
  `trained_during_eval` 与逐维 `error` 都要齐，才算"同一把尺量了两个臂"。任何一项不齐 ⇒ rc=1，
  本件只出版两份读数的并排，不出版方向。
* **分母不等就不比**（J-N5u-2）：一枚维度只有两臂 `machine_scored_items` **都 >0 且相等**才入场。
  按已入库件的真形状，B／G 的机检数是 0（20 题全在待人工复核），A／F／H 是 `not_executed` ⇒
  入场集预期是 {C, D, E}；这不是我猜的，是 `reports/taiji_n5_07_base_beta_cap0_20261010.json` 现读的。
* **只看逐题翻转，不看绝对分**（J-N5u-3）：一题入场要两臂同 `id` 都在、`load_ok` 都真、
  `pending_human_review` 都假；否则它记 `excluded_ids` 并出版原因，**不**当 0 分参与翻转
  （把"没测到"写成"测到且错"是本仓反复记过的失效方式，[[probe-output-must-be-verified-present]]）。

方向四支互斥：`capability_gain`／`capability_loss`／`not_resolved`／`direction_inconsistent`，
外加一支"没有入场维"＝`no_eligible_dimension`。阈值 ≥3 的来由与"读平就记读平、不许换 B／G 人工面找话说"
都冻在 `PLAN-N5-22` §2／§5，本文件的注释不复述判据、只解释形状。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FORMAT = "taiji-n5-22-cap0-pairing-v1"
PREREG = "plans/reference/PLAN-N5-22_batch_c_capability_pairing_prereg_20261013.md \u00a72"
DIRECTION_MARGIN = 3
ID_LIST_CAP = 12


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"CAP report missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "dimensions" not in payload or "identity" not in payload:
        raise ValueError(f"not a CAP baseline report: {path}")
    return payload


def _item_map(dimension: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = dimension.get("items")
    if not isinstance(items, list):
        return {}
    return {str(row["id"]): row for row in items if isinstance(row, dict) and "id" in row}


def _driven(payload: dict[str, Any]) -> list[str]:
    return sorted(
        code
        for code, dimension in payload["dimensions"].items()
        if isinstance(dimension, dict) and "tally" in dimension
    )


def _pair_dimension(code: str, treated: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    treated_items = _item_map(treated)
    control_items = _item_map(control)
    treated_scored = treated["tally"]["machine_scored_items"]
    control_scored = control["tally"]["machine_scored_items"]
    denominators_equal = treated_scored == control_scored
    common_ids = sorted(set(treated_items) & set(control_items))
    missing_ids = sorted((set(treated_items) | set(control_items)) - set(common_ids))

    eligible: list[str] = []
    excluded: dict[str, list[str]] = {
        "not_in_both_arms": missing_ids[:ID_LIST_CAP],
        "load_failed": [],
        "pending_human_review": [],
        "score_missing": [],
    }
    flips_treated_only: list[str] = []
    flips_control_only: list[str] = []
    for identifier in common_ids:
        left, right = treated_items[identifier], control_items[identifier]
        if not left.get("load_ok") or not right.get("load_ok"):
            excluded["load_failed"].append(identifier)
            continue
        if left.get("pending_human_review") or right.get("pending_human_review"):
            excluded["pending_human_review"].append(identifier)
            continue
        if not isinstance(left.get("score"), int) or not isinstance(right.get("score"), int):
            excluded["score_missing"].append(identifier)
            continue
        eligible.append(identifier)
        if left["score"] > right["score"]:
            flips_treated_only.append(identifier)
        elif right["score"] > left["score"]:
            flips_control_only.append(identifier)

    enters = bool(denominators_equal and treated_scored > 0 and eligible)
    return {
        "dimension": code,
        "machine_scored_items": {"treated": treated_scored, "control": control_scored},
        "denominators_equal": denominators_equal,
        "error_present": {"treated": treated.get("error"), "control": control.get("error")},
        "items_in_both_arms": len(common_ids),
        "eligible_items": len(eligible),
        "excluded_ids": {key: value[:ID_LIST_CAP] for key, value in excluded.items()},
        "excluded_counts": {key: len(value) for key, value in excluded.items()},
        "flips_treated_only": len(flips_treated_only),
        "flips_treated_only_ids": flips_treated_only[:ID_LIST_CAP],
        "flips_control_only": len(flips_control_only),
        "flips_control_only_ids": flips_control_only[:ID_LIST_CAP],
        "delta": len(flips_treated_only) - len(flips_control_only),
        "enters_verdict": enters,
    }


def _direction(rows: list[dict[str, Any]]) -> str:
    entering = [row for row in rows if row["enters_verdict"]]
    if not entering:
        return "no_eligible_dimension"
    best_gain = max(row["delta"] for row in entering)
    best_loss = min(row["delta"] for row in entering)
    gain = best_gain >= DIRECTION_MARGIN
    loss = best_loss <= -DIRECTION_MARGIN
    if gain and loss:
        return "direction_inconsistent"
    if gain:
        return "capability_gain"
    if loss:
        return "capability_loss"
    return "not_resolved"


def compare(treated: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    treated_dims, control_dims = treated["dimensions"], control["dimensions"]
    only_treated = sorted(set(treated_dims) - set(control_dims))
    only_control = sorted(set(control_dims) - set(treated_dims))
    codes = sorted(set(treated_dims) & set(control_dims))
    rows = [
        _pair_dimension(code, treated_dims[code], control_dims[code])
        for code in codes
        if "tally" in treated_dims[code]
        and "tally" in control_dims[code]
        and code in _driven(treated)
        and code in _driven(control)
    ]
    pairing = {
        "eval_set_sha256_equal": treated["identity"]["eval_set_sha256"]
        == control["identity"]["eval_set_sha256"],
        "git_head_equal": treated["identity"]["git_head"] == control["identity"]["git_head"],
        "chain_equal": treated.get("chain") == control.get("chain"),
        "scoring_discipline_equal": treated.get("scoring_discipline")
        == control.get("scoring_discipline"),
        "neither_arm_trained_during_eval": not treated.get("trained_during_eval")
        and not control.get("trained_during_eval"),
        "dimension_sets_equal": not only_treated and not only_control,
        "no_dimension_errors": all(
            row["error_present"]["treated"] is None and row["error_present"]["control"] is None
            for row in rows
        ),
    }
    pairing_valid = all(pairing.values())
    direction = _direction(rows) if pairing_valid else "pairing_invalid"
    verdicts = dict(pairing)
    verdicts["at_least_one_eligible_dimension"] = any(row["enters_verdict"] for row in rows)
    failing = sorted(key for key, ok in verdicts.items() if not ok)
    return {
        "identity": {
            "treated_checkpoint_sha256": treated["identity"]["checkpoint_sha256"],
            "control_checkpoint_sha256": control["identity"]["checkpoint_sha256"],
            "eval_set_sha256": treated["identity"]["eval_set_sha256"],
            "git_head": treated["identity"]["git_head"],
        },
        "pairing": pairing,
        "pairing_valid": pairing_valid,
        "direction": direction,
        "direction_margin": DIRECTION_MARGIN,
        "dimensions": rows,
        "not_compared_dimensions": sorted(
            code for code in codes if code not in {row["dimension"] for row in rows}
        ),
        "only_in_one_arm": {"treated": only_treated, "control": only_control},
        "verdicts": verdicts,
        "failing_verdicts": failing,
        "verdict": "holds" if pairing_valid else "rejected",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批次 C 两臂 CAP 读数的配对与逐题翻转")
    parser.add_argument("--treated-report", required=True, help="治疗臂的 CAP 件")
    parser.add_argument("--control-report", required=True, help="对照臂的 CAP 件")
    parser.add_argument(
        "--out-report", required=True, help="落盘路径（件已存在则拒覆盖 rc=2，除非 --overwrite）"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    started = time.perf_counter()
    paths = []
    for raw in (args.treated_report, args.control_report):
        path = Path(raw)
        paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    try:
        treated = _load(paths[0])
        control = _load(paths[1])
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError) as error:
        print(f"REJECT {error}", file=sys.stderr)
        return 2

    body = compare(treated, control)
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "prereg": PREREG,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs_as_given": [str(args.treated_report), str(args.control_report)],
        "reading_limit": (
            "本件只说"
            "「这两份读数能不能配对」与「逐题翻转的方向」；它不说幅度（幅度需同配置重复臂）、"
            "不说机制（机制需 DEBT-G131 那台集合差仪）、也不动 B／G 的人工复核面。"
            "方向句的引用单位不可拆：必须与 J-N5r-4（配置面单变量）与 J-N5t-1（两臂同流）一起引。"
        ),
        **body,
    }
    report["wall_clock_ms"] = round((time.perf_counter() - started) * 1000, 3)

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    summary = {
        "verdict": report["verdict"],
        "direction": report["direction"],
        "failing_verdicts": report["failing_verdicts"],
        "entering_dimensions": [
            row["dimension"] for row in report["dimensions"] if row["enters_verdict"]
        ],
        "deltas": {row["dimension"]: row["delta"] for row in report["dimensions"]},
        "out": out.as_posix(),
    }
    sys.stdout.buffer.write(json.dumps(summary, ensure_ascii=True).encode("ascii") + b"\n")
    return 0 if report["verdict"] == "holds" else 1


if __name__ == "__main__":
    raise SystemExit(main())
