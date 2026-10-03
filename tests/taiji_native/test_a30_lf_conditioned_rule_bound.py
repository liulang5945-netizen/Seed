"""§95 上界计算器的守卫：这条界**必须既算得出已知数，也能在缺列的旧件上响亮失败**。

已知-good 锚点不是我口供：四个数（`+2.78／+9.38／+3.82／+17.01pp`）是 2026-10-03 手工在同一批件上算过、
并已写进 PLAN-A-30 §95/§96 的读数 ⇒ 这台仪器一入库就背负"复现这四个数"的义务（改坏即红）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from bound_taiji_a30_lf_conditioned_rule import (  # noqa: E402
    R_GRID,
    best_margin,
    lf_count,
    main,
    summarize,
)

REPORTS = PROJECT_ROOT / "reports"
#: §95/§96 里手工算过的四档：(c) 底无回路、装机底无回路、装机底挂回路门 OFF、门 ON(K=128)。
KNOWN_CEILINGS = {
    'taiji_a30_stop_failure_c_v31_winner_nocircuit96_20261003.json': (2.78, 20, 8),
    'taiji_a30_stop_failure_self_v31_winner_nocircuit96_20261003.json': (9.38, 80, 27),
    'taiji_a30_stop_failure_self_v34_margins_circuitseedA_96_20261003.json': (3.82, 247, 11),
    'taiji_a30_stop_failure_self_v34_margins_productk128_96_20261003.json': (17.01, 206, 49),
}


def _gen(steps=256, lf=0, margin=None, stopped=False, column='v34'):
    generation = {'generation_steps': steps, 'ate_full_budget': not stopped, 'terminal_decision': None}
    if stopped:
        generation['terminal_decision'] = {'step': steps}
    if column == 'v34':
        generation['lf_trace_v29'] = {'lf_step_count': lf}
        generation['lf_margins_v34'] = {'lf_steps': lf, 'min_ratio': margin}
    else:
        generation['lf_trace_v29'] = {'lf_step_count': lf}
        generation['lf_next_probe_v30'] = {'lf_count': lf, 'min_ratio_best_over_boundary_after_lf': margin}
    return generation


def _report(generations):
    return {
        'format': 'taiji-a30-stop-failure-v34',
        'per_item': [{'id': 'V001', 'endstep_probe_v22': generations}],
    }


def test_the_bound_is_driven_by_lf_emission_not_by_the_threshold() -> None:
    """上界只看"发过 LF 的拖写代"；从未换行的拖写代对任何 R 都不可达（这条界能为零）。"""

    payload = summarize(_report([_gen(lf=0), _gen(lf=0), _gen(lf=1, margin=50.0), _gen(lf=3, margin=1.1)]))
    assert payload['generations'] == 4
    assert payload['eater_count'] == 4, payload
    assert payload['eaters_with_lf'] == 2, payload
    assert payload['delta_pp_upper_bound'] == 50.0, payload
    #: R=1.2 只开火一次（比值 50 那代差太远）⇒ 下界随阈值移动、上界不动。
    row = {item['R']: item for item in payload['per_threshold']}
    assert row[1.2]['fired_eaters'] == 1 and row[1.2]['delta_pp_lower_bound'] == 25.0, row
    assert row[2.0]['fired_eaters'] == 1, row
    assert payload['margin_column_used'] == 'lf_margins_v34.min_ratio'


def test_a_real_stop_is_not_counted_as_something_the_rule_must_buy() -> None:
    """上界的可及集合是**拖写代里发过 LF 的那些**：已真停的代虽然也发 LF（§73），但不许算进规则能买的增量。"""

    payload = summarize(
        _report([_gen(lf=1, margin=1.0, stopped=True), _gen(lf=1, margin=9.0), _gen(lf=0)])
    )
    assert payload['self_stop_count'] == 1, payload
    assert payload['eater_count'] == 2 and payload['eaters_with_lf'] == 1, payload
    #: 3 代里只有 1 代是"拖写且发过 LF"⇒ 上界 33.33pp；若把真停那代也算进来会虚高到 66.67。
    assert payload['delta_pp_upper_bound'] == 33.33, payload
    assert all(item['fired_eaters'] == 0 for item in payload['per_threshold']), payload
    assert payload['lf_total_emissions'] == 2, payload


def test_the_two_instrument_shapes_are_read_by_one_implementation() -> None:
    """v31 与 v34 的列名不同 ⇒ 认列要自动、且件里必须披露用的是哪一列（两把尺子之戒）。"""

    v34 = summarize(_report([_gen(lf=2, margin=1.4)]))
    v31 = summarize(_report([_gen(lf=2, margin=1.4, column='v31')]))
    assert v34['margin_column_used'] == 'lf_margins_v34.min_ratio', v34
    assert v31['margin_column_used'] == 'lf_next_probe_v30.min_ratio_best_over_boundary_after_lf', v31
    assert v34['delta_pp_upper_bound'] == v31['delta_pp_upper_bound'] == 100.0, (v34, v31)
    assert v34['per_threshold'] == v31['per_threshold'], (v34, v31)
    #: `min_ratio` 为 `None`＝发过 LF 但没有可观测的后继行 ⇒ 计入 `with_lf`、计入"无后继"那一列。
    blind = summarize(_report([_gen(lf=1, margin=None)]))
    assert blind['eaters_with_lf'] == 1 and blind['eaters_with_observable_margin'] == 0, blind
    assert blind['eaters_with_lf_but_no_next_row'] == 1, blind
    assert all(item['fired_eaters'] == 0 for item in blind['per_threshold']), blind


def test_a_report_without_lf_columns_fails_loudly_instead_of_reading_zero(tmp_path: Path) -> None:
    """旧件（v27 那批没有 LF 列）**不许**被读成"上界 0pp"——那是仪器缺列，不是事实。"""

    legacy = tmp_path / 'legacy.json'
    legacy.write_text(
        json.dumps({'format': 'v27', 'per_item': [{'id': 'V1', 'endstep_probe_v22': [{'generation_steps': 10}]}]}),
        encoding='utf-8',
    )
    payload = summarize(json.loads(legacy.read_text(encoding='utf-8')))
    assert payload['margin_column_used'] is None and payload['delta_pp_upper_bound'] == 0.0, payload
    assert main(['--report', str(legacy)]) == 2, '缺列必须 rc=2，不能 rc=0'
    assert lf_count({'generation_steps': 10}) == 0 and best_margin({'generation_steps': 10}) is None


def test_a_v29_style_report_reports_unknown_instead_of_a_fake_zero(tmp_path: Path) -> None:
    """只有 LF 计数、没有比值列的件（v29 那批）：上界照算，四阈值下界必须报 `None` 而不是 `0.0`。

    实测踩过：同一装配三次独立取数（v29／v30／v31）里 v29 那行的下界全是 0.0，
    而它的意思只是"这台仪器在这一档没有比值列"——被读成"这条规则买不到任何停"就是假事实。
    """

    v29 = tmp_path / 'v29.json'
    v29.write_text(
        json.dumps(
            {
                'format': 'v29',
                'per_item': [
                    {
                        'id': 'V1',
                        'endstep_probe_v22': [
                            {
                                'generation_steps': 10,
                                'ate_full_budget': True,
                                'lf_trace_v29': {'lf_step_count': 2},
                            },
                            {
                                'generation_steps': 10,
                                'ate_full_budget': True,
                                'lf_trace_v29': {'lf_step_count': 0},
                            },
                        ],
                    }
                ],
            }
        ),
        encoding='utf-8',
    )
    payload = summarize(json.loads(v29.read_text(encoding='utf-8')))
    assert payload['lower_bound_available'] is False, payload
    assert payload['margin_column_used'] is None, payload
    assert payload['eaters_with_lf'] == 1 and payload['delta_pp_upper_bound'] == 50.0, payload
    assert all(row['fired_eaters'] is None for row in payload['per_threshold']), payload
    assert all(row['delta_pp_lower_bound'] is None for row in payload['per_threshold']), payload
    assert main(['--report', str(v29)]) == 2, '缺比值列也要 rc=2'
    #: 反向：有列时这条旗标必须为真——否则它自己就是一条恒假的守卫。
    assert summarize(_report([_gen(lf=1, margin=1.4)]))['lower_bound_available'] is True


def test_the_grid_is_the_frozen_one() -> None:
    assert R_GRID == (1.05, 1.20, 1.50, 2.00)


def test_known_readings_from_the_state_doc_are_reproduced_exactly() -> None:
    """已知-good 干跑：§95/§96 手工算过并发表过的四个上界必须逐值复现（改坏即红）。"""

    for name, (ceiling, eaters, with_lf) in KNOWN_CEILINGS.items():
        path = REPORTS / name
        if not path.is_file():
            raise AssertionError(f'锚点件不在库里：{name}')
        payload = summarize(json.loads(path.read_text(encoding='utf-8')))
        assert payload['generations'] == 288, (name, payload['generations'])
        assert payload['eater_count'] == eaters, (name, payload['eater_count'])
        assert payload['eaters_with_lf'] == with_lf, (name, payload['eaters_with_lf'])
        assert payload['delta_pp_upper_bound'] == ceiling, (name, payload['delta_pp_upper_bound'], ceiling)
    assert main(['--report', str(REPORTS / next(iter(KNOWN_CEILINGS)))]) == 0
