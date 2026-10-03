"""逐格配对器的守卫：四格表**必须能为假**，且已知锚点（§103／§104 发表过的数）必须逐值复现。

写它的理由（都是本线真付过的代价）：
① 聚合相减会把"有没有反向的格"这个问题抹掉——所以配对器要能把 `停→拖` 与 `拖→停` **分开点名**；
② 按交集悄悄缩分母是最容易自我美化的一张表（少一格就少一个反向证据），所以键集不一致必须**响亮失败**；
③ `other`（既不自停也不吃满）这一类若存在必须披露——它会让四格合计对不上总格数而不被发现。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / 'scripts' / 'training'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from pair_taiji_a30_stop_cells import intersect_loss, load_cells, main, pair  # noqa: E402

REPORTS = PROJECT_ROOT / 'reports'
NOC96 = REPORTS / 'taiji_a30_stop_failure_self_v31_winner_nocircuit96_20261003.json'
OFF96 = REPORTS / 'taiji_a30_stop_failure_self_v34_margins_circuitseedA_96_20261003.json'
ON96 = REPORTS / 'taiji_a30_stop_failure_self_v34_margins_productk128_96_20261003.json'


def _write(tmp_path: Path, name: str, shapes, *, circuit=None, window=None, base='abc', items_sha=None):
    items = {}
    for index, (item_id, turn, shape) in enumerate(shapes):
        item = items.setdefault(item_id, {'id': item_id, 'endstep_probe_v22': []})
        while len(item['endstep_probe_v22']) <= turn:
            item['endstep_probe_v22'].append({})
        generation = item['endstep_probe_v22'][turn]
        generation['terminal_decision'] = {'step': 1} if shape == 'stop' else None
        generation['ate_full_budget'] = shape == 'eat'
    payload = {
        'checkpoint_sha256': base,
        'circuit': circuit,
        'product_window_steps': window,
        'per_item': list(items.values()),
    }
    if items_sha is not None:
        payload['items_sha256'] = items_sha
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_the_table_names_both_directions_separately(tmp_path: Path) -> None:
    """反向的两格必须分得开：合成一个"2 格由停转拖、1 格由拖转停"的面。"""

    left = _write(
        tmp_path,
        'left.json',
        [('V001', 0, 'stop'), ('V002', 0, 'stop'), ('V003', 0, 'eat'), ('V004', 0, 'stop')],
    )
    right = _write(
        tmp_path,
        'right.json',
        [('V001', 0, 'eat'), ('V002', 0, 'eat'), ('V003', 0, 'stop'), ('V004', 0, 'stop')],
    )
    result = pair(left, right)
    assert result['table'] == {'stop->eat': 2, 'eat->stop': 1, 'stop->stop': 1}, result
    assert result['harmed_stop_to_eat'] == 2 and result['rescued_eat_to_stop'] == 1, result
    assert result['rescued_examples'] == ['V003/0'], result
    assert result['harmed_examples'] == ['V001/0', 'V002/0'], result
    assert result['cells'] == 4 and result['other_shape_count'] == 0, result


def test_an_other_shape_is_disclosed_not_absorbed(tmp_path: Path) -> None:
    """既不自停也不吃满的格要单列出来（否则四格合计会悄悄不等于总格数）。"""

    left = _write(tmp_path, 'l2.json', [('V001', 0, 'stop'), ('V002', 0, 'stop')])
    right = _write(tmp_path, 'r2.json', [('V001', 0, 'other'), ('V002', 0, 'stop')])
    result = pair(left, right)
    assert result['table'] == {'stop->other': 1, 'stop->stop': 1}, result
    assert result['other_shape_count'] == 1, result
    assert sum(result['table'].values()) == result['cells'] == 2, result


def test_mismatched_cell_sets_refuse_to_pair(tmp_path: Path) -> None:
    """键集不齐 ⇒ 响亮失败。交集配表＝少一格就可能少一个反向证据。"""

    left = _write(tmp_path, 'l3.json', [('V001', 0, 'stop'), ('V002', 0, 'eat')])
    right = _write(tmp_path, 'r3.json', [('V001', 0, 'eat')])
    with pytest.raises(RuntimeError, match='格集合不一致'):
        pair(left, right)
    empty = tmp_path / 'empty.json'
    empty.write_text(json.dumps({'per_item': [{'id': 'V001', 'endstep_probe_v22': []}]}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='endstep_probe_v22'):
        load_cells(empty)


def test_the_three_way_salvage_count_is_its_own_cell(tmp_path: Path) -> None:
    """三枚件连配：只数"被回路弄坏的那批里门救回几格"，不与全件的救回数混用。"""

    base = _write(
        tmp_path, 'base.json', [('V001', 0, 'stop'), ('V002', 0, 'stop'), ('V003', 0, 'eat'), ('V004', 0, 'stop')]
    )
    off = _write(
        tmp_path, 'off.json', [('V001', 0, 'eat'), ('V002', 0, 'eat'), ('V003', 0, 'eat'), ('V004', 0, 'eat')]
    )
    on = _write(
        tmp_path, 'on.json', [('V001', 0, 'stop'), ('V002', 0, 'eat'), ('V003', 0, 'stop'), ('V004', 0, 'eat')]
    )
    result = intersect_loss(base, off, on)
    #: 弄坏的是 V001/V002/V004 三格；V003 本来就拖写，所以它被救回**不算** salvage。
    assert result == {
        'circuit_broken_cells': 3,
        'gate_saved_of_those': 1,
        'salvage_rate': 0.3333,
        'still_broken': 2,
        #: 旧件（v35 之前）没有题面指纹 ⇒ 必须显式承认"不可知"，不许静当作已证同源。
        'items_fingerprint_status': 'unknown_pre_v35',
        'items_fingerprints': [None, None, None],
    }, result


def test_published_readings_are_reproduced_on_committed_reports() -> None:
    """已知-good 锚点：§103／§104 发表过的三组数必须在同一批件上逐值复现，件不在库就 raise。"""

    for path in (NOC96, OFF96, ON96):
        if not path.is_file():
            raise AssertionError(f'锚点件不在库里：{path.name}')
    circuit = pair(NOC96, OFF96)
    assert circuit['cells'] == 288, circuit
    assert circuit['table'] == {'stop->eat': 167, 'stop->stop': 41, 'eat->eat': 80}, circuit['table']
    assert circuit['harmed_stop_to_eat'] == 167 and circuit['rescued_eat_to_stop'] == 0, circuit
    assert circuit['other_shape_count'] == 0, circuit
    assert circuit['left_base'] == circuit['right_base'] == 'ca2628077b21bc4c', circuit
    assert circuit['left_circuit'] is False and circuit['right_circuit'] is True, circuit

    gate = pair(OFF96, ON96)
    assert gate['table'] == {'eat->eat': 206, 'stop->stop': 41, 'eat->stop': 41}, gate['table']
    assert gate['rescued_eat_to_stop'] == 41 and gate['harmed_stop_to_eat'] == 0, gate
    assert gate['left_window_steps'] is None and gate['right_window_steps'] == 128, gate

    three = intersect_loss(NOC96, OFF96, ON96)
    assert three['circuit_broken_cells'] == 167 and three['gate_saved_of_those'] == 37, three
    assert three['salvage_rate'] == 0.2216 and three['still_broken'] == 130, three
    assert main(['--left', str(NOC96), '--right', str(OFF96), '--third', str(ON96)]) == 0


def test_the_fingerprint_check_proves_same_items_only_when_both_sides_carry_it(tmp_path: Path) -> None:
    """DEBT-G31 的正反两支：两边都有且相等 ⇒ `equal`；不等 ⇒ 响亮拒绝配对（不许硬配两批题）。"""

    shapes = [('V001', 0, 'stop'), ('V002', 0, 'eat')]
    left = _write(tmp_path, 'fa.json', shapes, items_sha='same16value')
    right = _write(tmp_path, 'fb.json', [('V001', 0, 'eat'), ('V002', 0, 'eat')], items_sha='same16value')
    result = pair(left, right)
    assert result['items_fingerprint'] == {
        'left': 'same16value', 'right': 'same16value', 'status': 'equal'
    }, result

    other = _write(tmp_path, 'fc.json', [('V001', 0, 'eat'), ('V002', 0, 'eat')], items_sha='different16')
    with pytest.raises(RuntimeError, match='题面不同'):
        pair(left, other)

    #: 三枚连配也走同一条判定；不一致就拒，缺键就披露不可知（上面已测）。
    third = _write(tmp_path, 'fd.json', shapes, items_sha='same16value')
    assert intersect_loss(left, right, third)['items_fingerprint_status'] == 'equal'
    with pytest.raises(RuntimeError, match='题面指纹不一致'):
        intersect_loss(left, third, other)


def test_the_probe_itself_records_the_fingerprint() -> None:
    """仪器必须**自己写**这一列——否则配对器永远只能报"不可知"，那条债就没还。"""

    source = (PROJECT_ROOT / 'scripts' / 'training' / 'probe_taiji_a30_stop_failure.py').read_text(encoding='utf-8')
    assert '"items_sha256": _items_fingerprint(items)' in source

