"""§95 那条**上界式**的仓内计算器：任何"必须在换行（LF）之后才开火"的收口规则，最多能多买多少自停。

来历（2026-10-03，PLAN-A-30 第九十五次停靠）：判"要不要为解码侧干预动产品"时不需要重放模型——
生成环自回归于**已发字节**，而这类规则只做一件事（让循环更早退出），更早退出**不改变退出点之前的任何一步**
⇒ 一次拖写要能被这条规则转成"停"，它**原本就必须发过至少一次 `0x0A`**。于是：

* `Δpp 上界 ＝ 发过 ≥1 次 LF 的拖写代 ÷ 总代 × 100`（从未换行的拖写代，规则碰不到）；
* `Δpp 下界（按阈值 R）＝ min_ratio ≤ R 的拖写代 ÷ 总代 × 100`——"存在一次 LF+1 的比值 ≤R"
  与"该代所有 LF+1 里最好的一次 ≤R"是**同一个命题**，所以用逐代 `min_ratio` 就能算全（不需逐步表）。

两台仪器的列名不同（`lf_next_probe_v30` 与 `lf_margins_v34`），这里**一处实现、自动认列并披露用的是哪一列**
（一副档只住一处；分两处写就是两把尺子）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: §91/§95 先冻的阈值网格（顺序即引用顺序，不许四选一）。
R_GRID = (1.05, 1.20, 1.50, 2.00)

#: 两代仪器里"该代 LF 之后那一步的最好一次比值"分别住在哪一列。
MARGIN_COLUMNS = (
    ('lf_margins_v34', 'min_ratio'),
    ('lf_next_probe_v30', 'min_ratio_best_over_boundary_after_lf'),
)
#: 两代仪器里"该代发过几次 LF"分别住在哪一列。
LF_COUNT_COLUMNS = (
    ('lf_trace_v29', 'lf_step_count'),
    ('lf_next_probe_v30', 'lf_count'),
    ('lf_margins_v34', 'lf_steps'),
)


def iter_generations(report: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in report.get('per_item', []):
        for index, generation in enumerate(item.get('endstep_probe_v22', [])):
            generation['item_id'] = item.get('id')
            generation['generation_index'] = index
            out.append(generation)
    return out


def _pick(generation: dict[str, Any], table) -> Any:
    for block, key in table:
        payload = generation.get(block)
        if isinstance(payload, dict) and key in payload:
            return payload[key], f'{block}.{key}'
    return None, None


def lf_count(generation: dict[str, Any]) -> int:
    value, _ = _pick(generation, LF_COUNT_COLUMNS)
    return int(value) if isinstance(value, int) else 0


def best_margin(generation: dict[str, Any]) -> float | None:
    value, _ = _pick(generation, MARGIN_COLUMNS)
    return float(value) if value else None


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    generations = iter_generations(report)
    total = len(generations)
    stops = [g for g in generations if g.get('terminal_decision') is not None]
    eaters = [g for g in generations if g.get('ate_full_budget')]
    with_lf = [g for g in eaters if lf_count(g) >= 1]
    observed = [g for g in with_lf if best_margin(g) is not None]
    #: 比值列**结构上不存在**（v29 及更早的件）与"有列但这些代都没开火"是两件不同的事：
    #: 前者只能报"不可知"，报成 0.0 会被读成"这条规则买不到任何停"（实测踩过：同一台仪器
    #: 对 v29/v30/v31 三枚同配置件跑出的下界，v29 那行是假的 0）。
    margin_column = next(
        (found for found in (_pick(g, MARGIN_COLUMNS)[1] for g in generations) if found), None
    )
    rows = []
    for threshold in R_GRID:
        if margin_column is None:
            rows.append({'R': threshold, 'fired_eaters': None, 'delta_pp_lower_bound': None})
            continue
        fired = [g for g in observed if best_margin(g) <= threshold]
        rows.append(
            {
                'R': threshold,
                'fired_eaters': len(fired),
                'delta_pp_lower_bound': round(100.0 * len(fired) / total, 2) if total else None,
            }
        )
    return {
        'report': None,
        'format': report.get('format'),
        'lower_bound_available': margin_column is not None,
        'checkpoint_sha256': report.get('checkpoint_sha256'),
        'circuit': 'seed-A' if report.get('circuit') else None,
        'circuit_sha256': (report.get('circuit_sha256') or '')[:16] or None,
        'product_window_steps': report.get('product_window_steps'),
        'max_length': report.get('max_length'),
        'generations': total,
        'self_stop_count': len(stops),
        'self_stop_rate': round(len(stops) / total, 4) if total else None,
        'eater_count': len(eaters),
        'eaters_with_lf': len(with_lf),
        #: 上界只取决于"发过 LF 的拖写代"——不看阈值，因为规则没有 LF 就没有输入。
        'delta_pp_upper_bound': round(100.0 * len(with_lf) / total, 2) if total else None,
        'eaters_with_observable_margin': len(observed),
        'eaters_with_lf_but_no_next_row': len(with_lf) - len(observed),
        'lf_total_emissions': sum(lf_count(g) for g in generations),
        'per_threshold': rows,
        'margin_column_used': margin_column,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--report', action='append', required=True, help='探针件路径（可重复）')
    args = parser.parse_args(argv)
    rows = []
    for path in args.report:
        payload = summarize(json.loads(Path(path).read_text(encoding='utf-8')))
        payload['report'] = Path(path).name
        rows.append(payload)
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    #: 结论性 rc：任何一枚件**没有 LF 列**（仪器太旧、算不出这条界）就响亮 2，
    #: 而不是默默给一个 `None` 让人当"上界为零"读。
    return 0 if all(row['generations'] and row['margin_column_used'] for row in rows) else 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
