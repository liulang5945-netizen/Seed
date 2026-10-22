"""ast-canonical-v1 封印哈希的红绿双向守卫（Phase 11 S2）。

铁律: 守卫必须红/绿各跑一次证明能响。绿向 = black 重排 / docstring / 注解不变；
红向 = 常量 / 改名 / 控制流必变。落地时曾临时禁用 docstring 剥离分支，
G2 真实跑红一次留痕（提交信息存档），随后恢复——非装饰性探针。

ADR: outputs/project-audit-20261021-2315/artifacts/11-seal-migration-design.md
"""

from __future__ import annotations

import black

from scripts.training import seal_ast

FIXTURE = '''\
"""Fixture module docstring."""

from __future__ import annotations

import math

THRESHOLD: float = 1.5


def score( values:list[int],ceiling:int = 9 )->dict[str,int]:
    """Score docstring: metadata only."""
    result: dict[str, int] = {}
    for name, value in values and enumerate( values ):
        label = f'score:{name}'
        if value<ceiling and value>0 :
            result[label] = int( math.floor( value*THRESHOLD ) )
    return result
'''


def _hash(source: str) -> str:
    return seal_ast.ast_hash(source)


def test_g1_black_reformat_does_not_change_hash():
    reformatted = black.format_str(FIXTURE, mode=black.Mode())
    assert reformatted != FIXTURE  # 重排确实发生了
    assert _hash(reformatted) == _hash(FIXTURE)


def test_g2_docstring_only_edit_does_not_change_hash():
    edited = FIXTURE.replace("Fixture module docstring.", "Rewritten module doc.")
    edited = edited.replace("Score docstring: metadata only.", "Rewritten function doc.")
    assert edited != FIXTURE
    assert _hash(edited) == _hash(FIXTURE)


def test_g3_annotation_only_edit_does_not_change_hash():
    edited = FIXTURE.replace("THRESHOLD: float = 1.5", "THRESHOLD = 1.5")
    edited = edited.replace(
        "def score( values:list[int],ceiling:int = 9 )->dict[str,int]:",
        "def score(values, ceiling=9):",
    )
    edited = edited.replace("result: dict[str, int] = {}", "result = {}")
    assert edited != FIXTURE
    assert _hash(edited) == _hash(FIXTURE)


def test_r1_numeric_constant_change_changes_hash():
    edited = FIXTURE.replace("THRESHOLD: float = 1.5", "THRESHOLD: float = 2.5")
    assert edited != FIXTURE
    assert _hash(edited) != _hash(FIXTURE)


def test_r2_identifier_rename_changes_hash():
    edited = FIXTURE.replace("result[label]", "bucket[label]").replace(
        "result: dict", "bucket: dict"
    )
    assert edited != FIXTURE
    assert _hash(edited) != _hash(FIXTURE)


def test_r3_control_flow_rewrite_changes_hash():
    edited = FIXTURE.replace(
        "if value<ceiling and value>0 :",
        "if 0 < value < ceiling:",
    )
    assert edited != FIXTURE
    assert _hash(edited) != _hash(FIXTURE)


def test_self_check_golden_is_pinned():
    assert seal_ast.self_check() is True


def test_r4_fabricated_hash_is_rejected_by_all_channels():
    """git 历史取证通道的红向: 从不存在过的哈希在任何通道都不能通过。"""
    ok, failures = seal_ast.verify_hash_dict(
        {"E:/Seed/scripts/training/seal_ast.py": "0" * 64}, allow_git_history=True
    )
    assert ok is False
    assert failures == ["E:/Seed/scripts/training/seal_ast.py"]


def test_green_channel_real_file_and_git_attestation():
    """绿向（真实文件）: 在位字节通过；同一记录值经 git 历史取证也通过。"""
    path = "E:/Seed/scripts/training/seal_ast.py"
    digest = seal_ast.sha256_file(path)
    ok, failures = seal_ast.verify_hash_dict({path: digest})
    assert ok is True and failures == []
    assert seal_ast.verify_path_pin(path, digest) is True
