"""A30 报告件的逐字段比较器：给「默认关闭 ⇒ 逐位不变」这条验收式一个仓内工具。

来历（2026-10-02，DECISION-A30 §3 甲）：判定"两份读数是否逐位相同"这件事当时只存在于一次性
脚本里；owner 若批准发射时序门控立项，验收要跑的就是这个式子，所以它必须在仓内、必须有守卫。

两类差异必须**分开报**，混在一起就会误报（今天实测到的第一处差异就是后加的披露字段）：
* `behavior` —— 同一字段在两件里**都有值**但不同（或列表长度/元素不同）；
* `schema` —— 一边有该字段一边没有，**或一边是 `None` 一边有值**（典型是仪器后加自述字段、旧件按零补）。

默认只把 `behavior` 算作"不逐位相同"；`--strict` 时 `schema` 也算（用于同版本比对的场合）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def diff_reports(
    left: Dict[str, Any],
    right: Dict[str, Any],
    trail: str = '',
) -> Tuple[List[Tuple[str, str, Any, Any]], List[Tuple[str, str]]]:
    """Return (behavior_diffs, schema_diffs) between two nested report structures."""
    behavior: List[Tuple[str, str, Any, Any]] = []
    schema: List[Tuple[str, str]] = []

    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            here = f'{trail}.{key}' if trail else key
            if key not in left or key not in right:
                schema.append((here, 'present_in_only_one_side'))
                continue
            b, s = diff_reports(left[key], right[key], here)
            behavior.extend(b)
            schema.extend(s)
        return behavior, schema

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            behavior.append((trail or '<root>', 'list_length', len(left), len(right)))
            return behavior, schema
        for index, (item_l, item_r) in enumerate(zip(left, right)):
            b, s = diff_reports(item_l, item_r, f'{trail}[{index}]')
            behavior.extend(b)
            schema.extend(s)
        return behavior, schema

    if (left is None) != (right is None):
        # 只有一边为空、另一边有值才算：这是"旧件按零补"的指纹，属信封形状而不是模型行为。
        # （两边同为 None 是相等——实测踩过：`control_no_circuit.circuit` 两枚件都是 None，
        #   旧写法把它报成 schema 差异，导致 --strict 会把两份完全相同的件判成不相同。）
        schema.append((trail or '<root>', 'null_filled_other_side_has_value'))
        return behavior, schema

    if type(left) is not type(right):
        behavior.append((trail or '<root>', 'type', type(left).__name__, type(right).__name__))
        return behavior, schema

    if left != right:
        behavior.append((trail or '<root>', 'value', left, right))
    return behavior, schema


def compare_files(
    left_path: Path,
    right_path: Path,
    subtree: Optional[str] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    left = json.loads(left_path.read_text(encoding='utf-8'))
    right = json.loads(right_path.read_text(encoding='utf-8'))
    if subtree is not None:
        if subtree not in left or subtree not in right:
            return {
                'identical': False,
                'reason': 'subtree_missing',
                'subtree': subtree,
                'present_left': subtree in left,
                'present_right': subtree in right,
            }
        left, right = left[subtree], right[subtree]
    behavior, schema = diff_reports(left, right)
    decisive = behavior + (schema if strict else [])
    return {
        'identical': not decisive,
        'behavior_diff_count': len(behavior),
        'schema_diff_count': len(schema),
        'first_behavior_diff': behavior[0] if behavior else None,
        'first_schema_diff': schema[0] if schema else None,
        'strict': strict,
        'left': left_path.name,
        'right': right_path.name,
        'subtree': subtree,
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--left', required=True, help='第一份报告件（仓内相对或绝对路径）')
    parser.add_argument('--right', required=True, help='第二份报告件')
    parser.add_argument('--subtree', default=None, help='只比某个顶层键（如 control_no_circuit）')
    parser.add_argument(
        '--strict',
        action='store_true',
        help='把 schema 差异也算作不逐位相同（只在同一仪器版本之间用）',
    )
    args = parser.parse_args(argv)
    result = compare_files(
        Path(args.left), Path(args.right), subtree=args.subtree, strict=args.strict
    )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    # 存在性即结论：不一致 ⇒ rc=1，让这条验收式能直接被门使用，而不是靠人读输出。
    return 0 if result['identical'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
