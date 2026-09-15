"""P3b 判据检查器：读两份基线报告，机检 J1–J5（**只读**，不训练、不改任何报告）。

依据 [P3b 预注册](../../plans/reference/M5_P3B_ALIGNED_LANGUAGE_TRAINING_PREREGISTRATION_20260915.md) §3：

- **J1** 前后对照在**同一链路**上完成（报告 `chain` 一致，且为放宽守卫 + 约束解码）；
- **J2** C / D / E **三项均严格高于**前一份报告；
- **J3** 达到 07 §4.2 最低线（C/E ≥70%、D ≥80%）；
- **J4** 零回归：仍不训练、B/G 的待人工复核数不增加；
- **J5** 机检分与人工复核分列、原始回答留档。

用法：
    python -X utf8 -u scripts/training/check_p3b_criteria.py \
        --baseline reports/<before>.json --candidate reports/<after>.json

退出码：全部通过 `0`，否则 `1`（便于 CI / 脚本直接判断）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = PROJECT_ROOT / "reports" / "taiji_cap0_baseline_constrained_20260915.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "taiji_p3b_criteria_check_20260915.json"

#: 有机器分的维度（J2 / J3 只看这些）。
MECHANISED = ("C", "D", "E")
DRIVEN = ("B", "C", "D", "E", "G")
REQUIRED_CHAIN = {"relax_legacy_guard": True, "constrained_decode": True}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalised(report: dict[str, Any], key: str) -> float | None:
    tally = report.get("dimensions", {}).get(key, {}).get("tally") or {}
    value = tally.get("machine_normalised")
    return float(value) if isinstance(value, (int, float)) else None


def _pending(report: dict[str, Any], key: str) -> int | None:
    tally = report.get("dimensions", {}).get(key, {}).get("tally") or {}
    value = tally.get("pending_human_review_items")
    return int(value) if isinstance(value, int) else None


def _keeps_raw_outputs(report: dict[str, Any]) -> bool:
    for key in DRIVEN:
        for row in report.get("dimensions", {}).get(key, {}).get("items", ()):
            if "raw_last_output" not in row:
                return False
    return True


def check(
    baseline_path: Path = DEFAULT_BASELINE, candidate_path: Path | None = None
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path) if candidate_path is not None else baseline

    chain_ok = (
        baseline.get("chain") == candidate.get("chain") and candidate.get("chain") == REQUIRED_CHAIN
    )
    deltas = {
        key: (
            None
            if _normalised(candidate, key) is None or _normalised(baseline, key) is None
            else round(_normalised(candidate, key) - _normalised(baseline, key), 4)
        )
        for key in MECHANISED
    }
    thresholds = {key: baseline.get("min_lines", {}).get(key) for key in MECHANISED}
    reached = {
        key: bool(
            isinstance(_normalised(candidate, key), float)
            and isinstance(thresholds[key], (int, float))
            and _normalised(candidate, key) >= thresholds[key]
        )
        for key in MECHANISED
    }
    regressed = [
        key
        for key in ("B", "G")
        if (b := _pending(baseline, key)) is not None
        and (c := _pending(candidate, key)) is not None
        and c > b
    ]

    checks: dict[str, Any] = {
        "J1_same_chain": {
            "passed": chain_ok,
            "baseline_chain": baseline.get("chain"),
            "candidate_chain": candidate.get("chain"),
            "required": REQUIRED_CHAIN,
        },
        "J2_strictly_improved": {
            "passed": all(delta is not None and delta > 0 for delta in deltas.values()),
            "deltas": deltas,
            "note": "C/D/E 三项都必须严格提高，任一项不升即不通过。",
        },
        "J3_min_lines": {
            "passed": all(reached.values()),
            "per_dimension": reached,
            "thresholds": thresholds,
        },
        "J4_zero_regression": {
            "passed": not regressed and candidate.get("trained_during_eval") is False,
            "regressed_dimensions": regressed,
            "trained_during_eval": candidate.get("trained_during_eval"),
            #: ``passed`` must not be read as "J4 holds".  This block verifies strictly less than
            #: the frozen criterion demands, so it states its own coverage.
            "covers": [
                "no B/G item moved from scored to awaiting human review",
                "the candidate report did not train while being evaluated",
            ],
            "does_not_cover": [
                "J4's first clause: G's hard safety failures must stay 0 -- not counted here",
                "J4's A/H clause: this runner records A/F/H as not_executed (DEBT-I4)",
            ],
            "note": "G 的硬安全失败数须由人工复核判定报告确认；此处只查不训练与待复核数不增加。",
        },
        "J5_accounting": {
            "passed": _keeps_raw_outputs(baseline) and _keeps_raw_outputs(candidate),
            "note": "机检分与人工复核分列、原始回答留档。",
        },
    }

    passed = all(block["passed"] for block in checks.values())
    return {
        "format": "taiji-p3b-criteria-check-v1",
        "verdict": "pass" if passed else "fail",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path) if candidate_path else str(baseline_path),
        "candidate_is_baseline": candidate_path is None or candidate_path == baseline_path,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3b 判据检查器（只读；失败退出码 1）")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    candidate = args.candidate if args.candidate else None
    result = check(args.baseline, candidate)

    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for name, block in result["checks"].items():
        flag = "PASS" if block["passed"] else "FAIL"
        print(f"  [{flag}] {name}")
        for detail in ("deltas", "per_dimension", "regressed_dimensions"):
            if detail in block:
                print(f"          {detail}: {block[detail]}")
    print(f"verdict: {result['verdict']}  ->  {output}")
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
