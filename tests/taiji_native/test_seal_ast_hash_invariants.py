"""ast-canonical-v1 封印哈希的红绿双向守卫（Phase 11 S2）。

铁律: 守卫必须红/绿各跑一次证明能响。绿向 = black 重排 / docstring / 注解不变；
红向 = 常量 / 改名 / 控制流必变。落地时曾临时禁用 docstring 剥离分支，
G2 真实跑红一次留痕（提交信息存档），随后恢复——非装饰性探针。

ADR: outputs/project-audit-20261021-2315/artifacts/11-seal-migration-design.md
"""

from __future__ import annotations

import hashlib

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


def test_out_of_root_missing_pin_is_a_clean_failure_not_a_crash(tmp_path):
    """仓外且缺失的钉点必须返回失败清单，而不是把 relative_to 的 ValueError 抛给调用方。"""
    outside = tmp_path / "missing.json"
    assert outside.is_relative_to(seal_ast.ROOT) is False
    ok, failures = seal_ast.verify_hash_dict({str(outside): "0" * 64})
    assert ok is False
    assert failures == [str(outside)]


def test_recorded_value_attested_accepts_current_and_history_but_rejects_fabricated():
    rel = "scripts/training/seal_ast.py"
    path = seal_ast.ROOT / rel
    commits = seal_ast.file_commits(rel)
    assert commits
    historical = seal_ast.git_blob_bytes(commits[-1], rel)
    assert historical is not None
    historical_sha = hashlib.sha256(historical).hexdigest()
    assert seal_ast.recorded_value_attested(path, seal_ast.sha256_file(path)) is True
    assert seal_ast.recorded_value_attested(path, historical_sha) is True
    assert seal_ast.recorded_value_attested(path, "0" * 64) is False


RUNNER_LAYOUT = "D:/a/Seed/Seed/scripts/training/seal_ast.py"


def test_cross_machine_absolute_prefix_maps_onto_the_repo_file():
    """绿向（跨机映射）: 维护机布局的绝对路径在非维护机上归位到仓库内同一文件。

    D:/a/Seed/Seed 是 GitHub runner 的检出布局——在任何开发机上都不存在，
    因此在位通道必然落空，归位是唯一通路。维护机上 E:/Seed 在位命中，
    两条路殊途同归（同文件同字节）。
    """
    digest = seal_ast.sha256_file(seal_ast.ROOT / "scripts/training/seal_ast.py")
    ok, failures = seal_ast.verify_hash_dict({RUNNER_LAYOUT: digest})
    assert ok is True and failures == []
    assert seal_ast._repo_relative(RUNNER_LAYOUT) == "scripts/training/seal_ast.py"
    assert seal_ast._repo_relative("scripts\\training\\seal_ast.py") == (
        "scripts/training/seal_ast.py"
    )
    assert seal_ast._repo_relative("no\\such\\file.json") is None


def test_cross_machine_prefix_with_fabricated_hash_stays_red():
    """红向（映射后仍判哈希）: 路径归位只决定核验哪个文件，伪造哈希照样全红。"""
    ok, failures = seal_ast.verify_hash_dict({RUNNER_LAYOUT: "0" * 64})
    assert ok is False
    assert failures == [RUNNER_LAYOUT]


def test_unmappable_out_of_repo_layout_stays_fail_closed(tmp_path):
    """红向（映射失败）: 后缀不落在任何仓库内文件的绝对路径保持 fail-closed。"""
    outside = tmp_path / "nested" / "missing.json"
    ok, failures = seal_ast.verify_hash_dict({str(outside): "0" * 64})
    assert ok is False
    assert failures == [str(outside)]
