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
from typing import Any

DEFAULT_IGNORE = ('format', 'started_utc')
_IGNORE_PREFIXES = ('format_note_',)


def _is_ignored(trail: str) -> bool:
    """披露类字段（版本号、墙钟、逐版格式说明）**不算行为**。

    为什么必须有这一步（2026-10-03 实测）：同一配置、只差一个仪器版本的两个档，
    `behavior_diff_count` 是 **2** 且第一处就是 `format` 本身（v31 对 v32）——
    也就是说"只加字段不改行为"这件事**在跨版本比对里永远证不出来**，
    而 `DECISION-A30` §3 甲要的恰恰是这个证明。忽略项必须**在输出里报数**，
    否则这条豁免就成了无声放宽门柱。
    """

    head = trail.split('.', 1)[0].split('[', 1)[0]
    return head in DEFAULT_IGNORE or head.startswith(_IGNORE_PREFIXES)


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def diff_reports(
    left: dict[str, Any],
    right: dict[str, Any],
    trail: str = '',
) -> tuple[list[tuple[str, str, Any, Any]], list[tuple[str, str]]]:
    """Return (behavior_diffs, schema_diffs) between two nested report structures."""
    behavior: list[tuple[str, str, Any, Any]] = []
    schema: list[tuple[str, str]] = []

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
        for index, (item_l, item_r) in enumerate(zip(left, right, strict=True)):
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
    subtree: str | None = None,
    strict: bool = False,
    ignore_disclosure: bool = True,
) -> dict[str, Any]:
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
    ignored: list[str] = []
    if ignore_disclosure:
        ignored = [str(d[0]) for d in behavior + schema if _is_ignored(str(d[0]))]
        behavior = [d for d in behavior if not _is_ignored(str(d[0]))]
        schema = [d for d in schema if not _is_ignored(str(d[0]))]
    decisive = behavior + (schema if strict else [])
    return {
        'identical': not decisive,
        'behavior_diff_count': len(behavior),
        'schema_diff_count': len(schema),
        'first_behavior_diff': behavior[0] if behavior else None,
        'first_schema_diff': schema[0] if schema else None,
        #: 被豁免掉的披露类差异必须**可见**，不许变成无声的门柱放宽。
        'ignored_disclosure_count': len(ignored),
        'first_ignored_disclosure': ignored[0] if ignored else None,
        'ignore_disclosure': ignore_disclosure,
        'strict': strict,
        'left': left_path.name,
        'right': right_path.name,
        'subtree': subtree,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--left', required=True, help='第一份报告件（仓内相对或绝对路径）')
    parser.add_argument('--right', required=True, help='第二份报告件')
    parser.add_argument('--subtree', default=None, help='只比某个顶层键（如 control_no_circuit）')
    parser.add_argument(
        '--strict',
        action='store_true',
        help='把 schema 差异也算作不逐位相同（只在同一仪器版本之间用）',
    )
    parser.add_argument(
        '--no-ignore-disclosure',
        action='store_true',
        help='连版本号／墙钟／逐版格式说明一起算（回到旧口径；用它才能证明默认那条豁免确实在起作用）',
    )
    args = parser.parse_args(argv)
    result = compare_files(
        Path(args.left),
        Path(args.right),
        subtree=args.subtree,
        strict=args.strict,
        ignore_disclosure=not args.no_ignore_disclosure,
    )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    # 存在性即结论：不一致 ⇒ rc=1，让这条验收式能直接被门使用，而不是靠人读输出。
    return 0 if result['identical'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
