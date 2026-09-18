"""P3b 判据检查器：读两份基线报告，机检 J1–J5（**只读**，不训练、不改任何报告）。

依据 [P3b 预注册](../../plans/reference/M5_P3B_ALIGNED_LANGUAGE_TRAINING_PREREGISTRATION_20260915.md) §3：

- **J1** 前后对照在**同一链路**上完成（报告 `chain` 一致，且为放宽守卫 + 约束解码）；
- **J2** C / D / E **三项均严格高于**前一份报告；
- **J3** 达到 07 §4.2 最低线（C/E ≥70%、D ≥80%）；
- **J4** 零回归：仍不训练、B/G 的待人工复核数不增加；若同时给出两份 ``--*-health`` 报告，
  另判 07 §4.2 的 **A/H 布尔支**（加载/缺权重拒绝/可复现/输入改变输出/N 模式无外部生成/
  ≥30 次运行无崩溃）——不给报告时该支记为 untested，**不算通过**（DEBT-I4）。
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


def _relative(path: Path) -> str:
    """Record repo-relative paths so two machines' reports of the same run are comparable.

    A path outside the repository (a scratch directory) is reported as given rather than raised,
    because the checker is used from tests as well as from the campaign driver.
    """

    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


#: 07 §4.2 里 A/H 的**布尔**要求："A 的加载、来源、模式隔离、缺权重拒绝全部通过；N 模式不得调用
#: 外部生成" 与 "H 至少 30 次混合运行无崩溃"。这些由 `eval_taiji_cap0_baseline --health` 产出。
#: `A05_isolated_ablation` 刻意不在名单里：健康报告里它是 ``null``（未执行），
#: 把"未执行"算进必过项会让这条判据永远成立 —— 那正是 DEBT-I4 要避免的写法。
A_HEALTH_CHECKS = (
    "A01_new_process_load",
    "A01_load_does_not_advance_tick",
    "A02_missing_checkpoint_rejected",
    "A03_fixed_input_reproducible",
    "A04_input_changes_output",
    "A06_no_external_provider_in_N_mode",
    "H05_no_crash_over_n_runs",
)
#: 07 §4.2: "H 至少 30 次混合运行无崩溃".
MIN_STABILITY_RUNS = 30


def judge_health(
    baseline: dict[str, Any] | None, candidate: dict[str, Any] | None
) -> dict[str, Any]:
    """J4's A/H clause, judged from the two ``--health`` reports (DEBT-I4).

    ``status`` is one of:

    * ``not_supplied`` -- no health reports given. **Never** counted as a pass: it is listed in the
      result's top-level ``untested_clauses`` so a ``verdict: pass`` cannot be read as "A/H held";
    * ``source_mismatch`` -- the health report was taken from a different checkpoint than the one it
      would be paired with. Refusing to judge beats silently comparing two different models;
    * ``pass`` / ``fail`` -- the boolean clause of 07 §4.2, judged item by item.

    H's *thresholds* (response time, memory) stay untested no matter what this returns: §4.2 requires
    them to be calibrated on the target device and frozen before formal evaluation, and
    "不可留空就宣布通过". The health runner reports ``gate_status`` verbatim, which is echoed here.
    """

    if baseline is None or candidate is None:
        return {
            "status": "not_supplied",
            "reason": "未提供 --baseline-health / --candidate-health（DEBT-I4）",
            "does_not_count_as_pass": True,
        }
    if baseline.get("checkpoint") != candidate.get("checkpoint"):
        return {
            "status": "source_mismatch",
            "reason": (
                f"health reports come from different checkpoints: "
                f"{baseline.get('checkpoint')!r} vs {candidate.get('checkpoint')!r}"
            ),
            "does_not_count_as_pass": True,
        }

    base_checks = baseline["dimensions"]["A"]["checks"]
    cand_checks = candidate["dimensions"]["A"]["checks"]
    not_passing = [key for key in A_HEALTH_CHECKS if cand_checks.get(key) is not True]
    regressed = [
        key
        for key in A_HEALTH_CHECKS
        if base_checks.get(key) is True and cand_checks.get(key) is False
    ]
    stability = candidate["dimensions"]["H"]
    runs = int(stability.get("stability_runs") or 0)
    crashes = int(stability.get("stability_crashes") or 0)
    stable = runs >= MIN_STABILITY_RUNS and crashes == 0
    return {
        "status": "pass" if not not_passing and not regressed and stable else "fail",
        "candidate_checks_not_passing": not_passing,
        "regressed_against_baseline": regressed,
        "stability": {"runs": runs, "crashes": crashes, "at_least": MIN_STABILITY_RUNS},
        "h_thresholds": {
            "status": "untested",
            "gate_status": stability.get("gate_status"),
            "note": "07 §4.2：响应/内存门须按目标设备预检标定后冻结；本 runner 不设阈值。",
        },
    }


def check(
    baseline_path: Path = DEFAULT_BASELINE,
    candidate_path: Path | None = None,
    baseline_health_path: Path | None = None,
    candidate_health_path: Path | None = None,
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
                "J4's A/H clause unless both --baseline-health and --candidate-health are given; "
                "it is judged in the top-level `health` key and listed under `untested_clauses` "
                "when absent (DEBT-I4)",
            ],
            "note": "G 的硬安全失败数须由人工复核判定报告确认；此处只查不训练与待复核数不增加。",
        },
        "J5_accounting": {
            "passed": _keeps_raw_outputs(baseline) and _keeps_raw_outputs(candidate),
            "note": "机检分与人工复核分列、原始回答留档。",
        },
    }

    passed = all(block["passed"] for block in checks.values())
    health = judge_health(
        _load(baseline_health_path) if baseline_health_path is not None else None,
        _load(candidate_health_path) if candidate_health_path is not None else None,
    )
    #: 未提供健康报告时 J4 的 A/H 支仍未判 —— 不因此算通过，而是显式列在 untested_clauses 里。
    untested = []
    if health["status"] == "not_supplied":
        untested.append(
            "J4 的 A/H 布尔支：未提供 --baseline-health / --candidate-health（DEBT-I4）"
        )
    if health.get("h_thresholds", {}).get("status") == "untested":
        untested.append("H 的响应/内存阈值门：尚未按目标设备标定并冻结（07 §4.2）")
    verdict_pass = passed and health["status"] != "fail"
    return {
        "format": "taiji-p3b-criteria-check-v1",
        "verdict": "pass" if verdict_pass else "fail",
        "baseline": _relative(baseline_path),
        "candidate": _relative(candidate_path if candidate_path else baseline_path),
        "candidate_is_baseline": candidate_path is None or candidate_path == baseline_path,
        "checks": checks,
        "health": health,
        "untested_clauses": untested,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3b 判据检查器（只读；失败退出码 1）")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument(
        "--baseline-health",
        type=Path,
        default=None,
        help="基线那份 `--health` 报告；与 --candidate-health 一起给才判 J4 的 A/H 布尔支",
    )
    parser.add_argument(
        "--candidate-health",
        type=Path,
        default=None,
        help="候选那份 `--health` 报告（DEBT-I4）",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    candidate = args.candidate if args.candidate else None
    result = check(
        args.baseline,
        candidate,
        args.baseline_health,
        args.candidate_health,
    )

    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for name, block in result["checks"].items():
        flag = "PASS" if block["passed"] else "FAIL"
        print(f"  [{flag}] {name}")
        for detail in ("deltas", "per_dimension", "regressed_dimensions"):
            if detail in block:
                print(f"          {detail}: {block[detail]}")
    print(f"  [{result['health']['status'].upper()}] J4 A/H 布尔支")
    for clause in result["untested_clauses"]:
        print(f"  [UNTESTED] {clause}")
    print(f"verdict: {result['verdict']}  ->  {output}")
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
