"""DEBT-G150 的守卫：协议里那枚 `max_generated_bytes` 必须与同文件的生成上限实参同值。

读码现取的事实（本册第一条测复现它）：`scripts/training/` 下**没有任何一处**读
`max_generated_bytes`，真正限制步数的是 `model.generate_trace(tape, session, question, N)`
里按位置的第四个实参字面量。今天两侧同值 ⇒ 零行为影响；本册要拦的是"改了一侧而另一侧
不动"这种静默不同源，因为两个被钉住的件（`eval_taiji_n1_transition_factorial_confirmation.py`
被 90 的 `base_evaluator` 钉、`audit_taiji_n1_native_evidence_train_fit.py` 被 75 枚 2026-10-15
在库报告钉）都不许为刷绿改字节，所以这条债只能以"把巧合钉成守卫"的形式落地。

扫描面按目录现取（`scripts/training/*.py`，一层，不递归），命中数本身入册：
两侧都在的文件、只有字面量的文件、只有协议定义的文件各自计数，三条计数被钉成常数——
新增一处就要显式改这三个数并写明理由，这比"顺手加一列"贵，也正是这条债要的代价。
"""

from __future__ import annotations

import ast
import pathlib
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
FACE = REPO / "scripts" / "training"

# 现读于本册第一次实跑；改动任何一条都要同轮更新 05 债表的 DEBT-G150 行。
PAIRED_FILES = 3
#: 〔㊵-1104 复测抬成 10〕本枚常数是"扫到的形状计数"，不是白名单：第 10 枚是 2026-10-17
#: `5aea3fbef` 新建的 `eval_taiji_n1_input_factorial_diagnostic.py`（`:344` 处 `generate_trace(..., 64)`），
#: 落在 DEBT-G150"字面量仍是真源头、协议键仍无消费处"那半未覆盖段里 ⇒ 计数照实抬，件名写在这里，
#: 谁把它改成读协议键就把它从这一档里减掉（减了这条测会红，那是正确的红，见 test 里的负对照）。
CAP_ONLY_FILES = 10
PROTOCOL_ONLY_FILES = 1
LITERAL_CAP_VALUES = {8, 32, 64}


def protocol_cap(source: str) -> int | None:
    """该文件协议字面量里 `max_generated_bytes=` 的整数值，没有则 None。"""
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.keyword)
            and node.arg == "max_generated_bytes"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            return node.value.value
    return None


def trace_caps(source: str) -> tuple[list[tuple[int, int]], list[int]]:
    """`generate_trace(...)` 的第四个位置实参：按常整数收成 (行号, 值)，其余收成行号列表。"""
    literal: list[tuple[int, int]] = []
    other: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "generate_trace"):
            continue
        if len(node.args) < 4:
            continue
        cap = node.args[3]
        if isinstance(cap, ast.Constant) and isinstance(cap.value, int):
            literal.append((node.lineno, cap.value))
        else:
            other.append(node.lineno)
    return literal, other


def scan(face: pathlib.Path) -> dict:
    paired, cap_only, protocol_only, non_literal, mismatches = [], [], [], [], []
    for path in sorted(face.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        protocol = protocol_cap(source)
        literal, other = trace_caps(source)
        if other:
            non_literal.append((path.name, other))
        if protocol is not None and literal:
            paired.append((path.name, protocol, literal))
            mismatches.extend(
                (path.name, line, protocol, value) for line, value in literal if value != protocol
            )
        elif literal:
            cap_only.append((path.name, literal))
        elif protocol is not None:
            protocol_only.append((path.name, protocol))
    return {
        "scanned": len(sorted(face.glob("*.py"))),
        "paired": paired,
        "cap_only": cap_only,
        "protocol_only": protocol_only,
        "non_literal": non_literal,
        "mismatches": mismatches,
    }


def test_scan_face_and_shape_are_the_measured_ones() -> None:
    result = scan(FACE)
    assert result["scanned"] >= 100, "the scan face must not be empty"
    assert len(result["paired"]) == PAIRED_FILES
    assert len(result["cap_only"]) == CAP_ONLY_FILES
    assert len(result["protocol_only"]) == PROTOCOL_ONLY_FILES
    assert result["non_literal"] == [], "a cap passed as an expression is outside this guard"
    assert {value for _name, _lines in result["cap_only"] for _line, value in _lines} <= (
        LITERAL_CAP_VALUES
    )


def test_protocol_and_call_site_never_disagree() -> None:
    result = scan(FACE)
    assert (
        result["mismatches"] == []
    ), f"protocol key and cap literal diverged: {result['mismatches']}"
    # 守卫必须真看见过同值的两侧，否则上一条的 ==[] 是空过。
    assert all(
        value == protocol
        for _name, protocol, literal in result["paired"]
        for _line, value in literal
    )
    assert result["paired"], "the paired face collapsed to zero"


def test_the_guard_goes_red_when_one_side_moves() -> None:
    moved_protocol = "PROTOCOL = dict(max_generated_bytes=32)\nmodel.generate_trace(a, b, c, 64)\n"
    literal, _ = trace_caps(moved_protocol)
    assert protocol_cap(moved_protocol) == 32
    assert literal == [(2, 64)]

    diverged = scan(_tmp_face(moved_protocol))
    assert [row[:2] for row in diverged["mismatches"]] == [("probe.py", 2)], diverged["mismatches"]

    agreed = scan(
        _tmp_face("PROTOCOL = dict(max_generated_bytes=64)\nmodel.generate_trace(a, b, c, 64)\n")
    )
    assert agreed["mismatches"] == []
    assert len(agreed["paired"]) == 1


def _tmp_face(source: str) -> pathlib.Path:
    directory = pathlib.Path(tempfile.mkdtemp())
    (directory / "probe.py").write_text(source, encoding="utf-8")
    return directory
