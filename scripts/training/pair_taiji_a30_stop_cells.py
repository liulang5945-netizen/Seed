"""逐格配对器：把"挂不挂回路／门开不开"对**停止行为**的影响按 `(题面 id, 轮序)` 配出来，而不是聚合相减。

来历（2026-10-03，PLAN-A-30 §103／§104）：聚合差值只能说明"自停少了 39/72"，**说不出有没有反向的格**；
而决策要的正是那个形状——"167 格由停转拖、0 格由拖转停"与"门只救回其中 37 格"是聚合值给不出的两句话
（§2v 那次因为只能做到聚合级，代价是把两笔账混成了一笔）。

**形状只有三种**（沿用探针的列，不自造判据）：
* `stop` ＝ 该格有 `terminal_decision`（边界符胜出的终止步，v27 起才补得出来）；
* `eat` ＝ `ate_full_budget`（吃满生成预算）；
* `other` ＝ 两者皆非（例如预算外退出）；非零时**必须**披露，因为它会悄悄改变四格表的行/列合计。

两侧都必须来自**同一批题面、同一枚底、同一生成预算**的件；键集不完全相同 ⇒ 响亮失败（宁可不出表，
也不要一张按交集悄悄缩过分母的表）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _shape(generation: dict[str, Any]) -> str:
    if generation.get('terminal_decision') is not None:
        return 'stop'
    if generation.get('ate_full_budget'):
        return 'eat'
    return 'other'


def load_cells(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """返回 `(逐格字典, 件本身)`——装配自述（底 sha／回路／K）要跟着一起出来，表才知道是哪两面在比。"""

    report = json.loads(path.read_text(encoding='utf-8'))
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    for item in report.get('per_item', []):
        for index, generation in enumerate(item.get('endstep_probe_v22', [])):
            cells[(str(item.get('id')), index)] = generation
    if not cells:
        raise RuntimeError(f'{path.name}: 件里没有逐代数据（endstep_probe_v22 为空）——这档太旧，配对不了')
    return cells, report


def fingerprint_check(left_report: dict[str, Any], right_report: dict[str, Any]) -> dict[str, Any]:
    """题面同源的判定要**从件内**做，不靠"两边都读了同一个路径"这条口供。

    三种结果：`equal`（两边都有 `items_sha256` 且相等）／`different`（都有但不等 ⇒ 配对无效，响亮失败）／
    `unknown_pre_v35`（任一边没有该键——v35 之前的件都如此，此时**不判**、只披露，
    并说明本轮 §103／§104 是靠 manifest 的 git 史外部核过的，不是靠件内自证）。
    """

    left = left_report.get('items_sha256')
    right = right_report.get('items_sha256')
    if left is None or right is None:
        return {'items_fingerprint': {'left': left, 'right': right, 'status': 'unknown_pre_v35'}}
    if left != right:
        raise RuntimeError(f'两枚件读的题面不同（{left} 对 {right}），配对无效——不许按 (id,轮序) 硬配两批题')
    return {'items_fingerprint': {'left': left, 'right': right, 'status': 'equal'}}


def pair(left_path: Path, right_path: Path) -> dict[str, Any]:
    left_cells, left_report = load_cells(left_path)
    right_cells, right_report = load_cells(right_path)
    if set(left_cells) != set(right_cells):
        only_left = sorted(set(left_cells) - set(right_cells))[:3]
        only_right = sorted(set(right_cells) - set(left_cells))[:3]
        raise RuntimeError(
            f'两枚件的格集合不一致，不配对（分母会被交集悄悄改掉）：'
            f'仅左 {len(only_left)}+ 例 {only_left}／仅右 {len(only_right)}+ 例 {only_right}'
        )
    table: dict[str, int] = {}
    rescued: list[str] = []
    lost: list[str] = []
    for key in sorted(left_cells):
        here = f'{_shape(left_cells[key])}->{_shape(right_cells[key])}'
        table[here] = table.get(here, 0) + 1
        if here == 'eat->stop':
            rescued.append(f'{key[0]}/{key[1]}')
        elif here == 'stop->eat':
            lost.append(f'{key[0]}/{key[1]}')
    other = sum(count for name, count in table.items() if 'other' in name)
    return {
        **fingerprint_check(left_report, right_report),
        'left': left_path.name,
        'right': right_path.name,
        'left_base': left_report.get('checkpoint_sha256'),
        'right_base': right_report.get('checkpoint_sha256'),
        'left_circuit': bool(left_report.get('circuit')),
        'right_circuit': bool(right_report.get('circuit')),
        'left_window_steps': left_report.get('product_window_steps'),
        'right_window_steps': right_report.get('product_window_steps'),
        'cells': len(left_cells),
        'table': table,
        'other_shape_count': other,
        'rescued_eat_to_stop': len(rescued),
        'harmed_stop_to_eat': len(lost),
        'rescued_examples': rescued[:5],
        'harmed_examples': lost[:5],
    }


def intersect_loss(left_path: Path, middle_path: Path, right_path: Path) -> dict[str, Any]:
    """三枚件连配（不挂回路／挂回路门 OFF／挂回路门 ON）：回路弄坏的格里，门救回多少。"""

    base_cells, base_report = load_cells(left_path)
    off_cells, off_report = load_cells(middle_path)
    on_cells, on_report = load_cells(right_path)
    if not (set(base_cells) == set(off_cells) == set(on_cells)):
        raise RuntimeError('三枚件的格集合不一致，不连配')
    prints = [base_report.get('items_sha256'), off_report.get('items_sha256'), on_report.get('items_sha256')]
    status = 'unknown_pre_v35' if any(p is None for p in prints) else ('equal' if len(set(prints)) == 1 else 'different')
    if status == 'different':
        raise RuntimeError(f'三枚件的题面指纹不一致：{prints}')
    broken = [
        key for key in base_cells
        if _shape(base_cells[key]) == 'stop' and _shape(off_cells[key]) == 'eat'
    ]
    saved = [key for key in broken if _shape(on_cells[key]) == 'stop']
    return {
        'circuit_broken_cells': len(broken),
        'gate_saved_of_those': len(saved),
        'salvage_rate': round(len(saved) / len(broken), 4) if broken else None,
        'still_broken': len(broken) - len(saved),
        'items_fingerprint_status': status,
        'items_fingerprints': prints,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--left', required=True, help='参照面件（通常是"不挂回路"或"门 OFF"）')
    parser.add_argument('--right', required=True, help='比较面件')
    parser.add_argument(
        '--third',
        default=None,
        help='可选：第三枚件（给了就连配三张，输出"回路弄坏的格里门救回多少"）',
    )
    args = parser.parse_args(argv)
    payload: dict[str, Any] = {'pair': pair(Path(args.left), Path(args.right))}
    if args.third:
        payload['three_way'] = intersect_loss(Path(args.left), Path(args.right), Path(args.third))
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
