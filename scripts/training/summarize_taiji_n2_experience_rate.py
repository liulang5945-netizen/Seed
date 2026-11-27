"""每单位经验的增益（速率）摘要器（PLAN-N2-05 的执行体；零算力、只读已封存件）。

**它只做三件事**：从同一枚件里取分子与三枚分母、复算分子、做三个除法并判定。
统计推断一条都不在这里发生——分子 `j_n2b_prime_delta_quality` 是既有判读器已经出版过的键，
本器**不重算判据、不重抄生成链**（本仓纪律）。

**为什么值得单独有一台仪器**：N2 的出口交付原文要"能力增量／单位经验（速率本体读数）"，
而这一列到今天没被出版过——分子与分母都在盘上，缺的只是一个先冻住的除法。冻在
`plans/reference/PLAN-N2-05_experience_rate_prereg_20261011.md`：三枚分母必须并列（J-N2e-2）、
分母非正必须响亮拒绝（J-N2e-1）、**分子为负时这一列只能叫代价不许叫增益**（J-N2e-3）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POWERUP = PROJECT_ROOT / "reports" / "taiji_n2b_powerup_20261008.json"
FORMAT = "taiji-n2-05-experience-rate-v1"
RECOMPUTE_TOL = 1e-9
DENOMINATORS = {
    "cycles": ("organs", "stats", "cycles"),
    "max_symbols": ("organs", "max_symbols"),
    "texts": ("organs", "texts"),
}


def _dig(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def summarize(powerup_path: Path) -> dict[str, Any]:
    payload = json.loads(powerup_path.read_text(encoding="utf-8"))
    numerator = payload.get("j_n2b_prime_delta_quality")
    if not isinstance(numerator, (int, float)):
        raise ValueError("分子 j_n2b_prime_delta_quality 不在件里 ⇒ 不许由别处搬来")
    treated = _dig(payload, ("material_treated_window", "quality_mean"))
    control = _dig(payload, ("material_control_window", "quality_mean"))
    if not (isinstance(treated, (int, float)) and isinstance(control, (int, float))):
        raise ValueError("两臂 quality_mean 缺失 ⇒ 分子无法复算")
    recomputed = float(treated) - float(control)
    same_source = abs(recomputed - float(numerator)) <= RECOMPUTE_TOL

    doses: dict[str, float] = {}
    missing: list[str] = []
    for name, path in DENOMINATORS.items():
        value = _dig(payload, path)
        if isinstance(value, (int, float)):
            doses[name] = float(value)
        else:
            missing.append(name)
    if missing:
        return {
            "format": FORMAT,
            "status": "dose_self_report_absent",
            "missing_denominators": missing,
            "powerup_report": str(powerup_path),
        }
    nonpositive = sorted(name for name, value in doses.items() if value <= 0.0)
    if nonpositive:
        return {
            "format": FORMAT,
            "status": "not_computable_nonpositive_dose",
            "nonpositive_denominators": nonpositive,
            "denominators": doses,
            "powerup_report": str(powerup_path),
        }
    rates = {name: float(numerator) / value for name, value in doses.items()}
    degenerate = sorted(
        f"{a}=={b}"
        for index, a in enumerate(sorted(doses))
        for b in sorted(doses)[index + 1 :]
        if doses[a] == doses[b]
    )
    #: **只出版能为假的判定**：`J-N2e-1`／`J-N2e-2` 走到这一行时按构造必然为真（不可算与缺自述
    #: 都在上面 `return` 掉了），所以它们不进件——"没算成"这件事由 `status` 与 rc=2 承担；
    #: `J-N2e-3` 有两个取值，留下来。恒真键的教训同 ㊵-618 的 `shadow_materialized`。
    return {
        "format": FORMAT,
        "status": "ok",
        "prereg": "plans/reference/PLAN-N2-05_experience_rate_prereg_20261011.md",
        "powerup_report": str(powerup_path),
        "numerator_delta_quality": numerator,
        "numerator_recomputed": recomputed,
        "guards": {
            "G_N2e_2_numerator_recomputable": same_source,
        },
        "denominators": doses,
        "degenerate_denominators": degenerate,
        "rates": {name: round(value, 9) for name, value in rates.items()},
        "j_n2e_3": "cost_not_gain" if float(numerator) < 0.0 else "gain_direction_unqualified",
        "reading_limit": (
            "the numerator is the already-adjudicated window delta of the second power-up, whose "
            "own criterion J-N2b' was judged not_holds; a rate built on a negative numerator is a "
            "COST per unit of experience, never a capability gain rate, and it does not revisit "
            "that judgement. cycles == texts in this dose because cycles_per_text=1, so those two "
            "denominators are not independent evidence. No inference happens here."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="experience-rate summary for PLAN-N2-05 (read-only)"
    )
    parser.add_argument("--powerup-report", type=Path, default=DEFAULT_POWERUP)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        payload = summarize(args.powerup_report)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "rate_failed", "error": str(error)}, ensure_ascii=True))
        return 2
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rates": payload.get("rates"),
                "degenerate_denominators": payload.get("degenerate_denominators"),
                "j_n2e_3": payload.get("j_n2e_3"),
                "guards": payload.get("guards"),
            },
            ensure_ascii=True,
        )
    )
    if payload["status"] != "ok":
        return 2
    return 0 if payload["guards"]["G_N2e_2_numerator_recomputable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
