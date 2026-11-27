"""R-S2a 契约测试（ADR-S2 §3.3 C-3）：verify_ci_deselect_list 的三态判定。

**本文件存在的核心理由（C-3 硬要求的回归用例）**：旧版守卫的判定是
**子串包含**（`if method in collected.stdout`）——只要某条用例名是另一条
node id 的子串，落空就会被静默判成命中。本文件构造两条互为子串的用例名、
删掉其中一条 ⇒ 新守卫**必须判红**（旧版会误判绿）。这与守卫自己要防的
「deselect 静默失效」反向同源，是本次改写的语义校正本体。

三态语义（取代旧版把「收集失败」与「条目落空」报同一种红）：
- MISS（红）：条目 node ∉ 收集集合 且其文件被成功收集；
- NOFILE（红）：条目文件在磁盘上不存在；
- UNCOLLECTABLE（黄，不判红）：条目所在文件本次收集产出 collection error
  （该事实由 pytest step 自己的非零退出承担）。

判定必须是**精确成员判定**（含 `::Class::` 与参数化 ``[...]`` 后缀的整体匹配），
禁止任何形式的 str-in-str。
"""

from __future__ import annotations

from pathlib import Path

from scripts.training import verify_ci_deselect_list as guard
from scripts.training.verify_ci_deselect_list import (
    _error_file_of,
    collect_listed_files,
    dedupe_preserving_order,
    judge,
    listed_files,
    parse_collection_output,
)

SAMPLE_NODE = "tests/x/test_a.py::test_alpha_beta"
SAMPLE_OUTPUT = "tests/x/test_a.py::test_alpha_beta\n" "=== 1 collected in 0.01s ===\n"


def _skeleton_repo(tmp_path: Path, relpaths: list[str]) -> Path:
    """建骨架文件使 judge 的磁盘存在性检查通过（单测只验证三态归属本身）。"""
    for rel in relpaths:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    return tmp_path


def test_substring_regression_deleted_case_must_be_red(tmp_path: Path):
    """回归核心：`test_alpha`（已删）是 `test_alpha_beta`（在册）的子串。
    旧版 `method in stdout` 会把两条都判命中（绿）；新版必须把 test_alpha 判 MISS（红）。"""
    entries = ["tests/x/test_a.py::test_alpha", "tests/x/test_a.py::test_alpha_beta"]
    nodes, error_files = parse_collection_output(SAMPLE_OUTPUT)
    assert nodes == {SAMPLE_NODE}
    assert error_files == set()
    verdict = judge(
        entries, nodes, error_files, repo=_skeleton_repo(tmp_path, ["tests/x/test_a.py"])
    )
    assert verdict.misses == ["tests/x/test_a.py::test_alpha(该文件里没有同名用例)"]
    assert verdict.hits == ["tests/x/test_a.py::test_alpha_beta"]
    assert verdict.red is True


def test_exact_membership_parametrized_suffix(tmp_path: Path):
    """参数化后缀整体匹配：`test_k` 不存在时不得被子串 `test_k[case1]` 掩护成绿。"""
    nodes, _ = parse_collection_output(
        "tests/x/test_p.py::test_k[case1]\n" "tests/x/test_p.py::test_k[case2]\n"
    )
    verdict = judge(
        ["tests/x/test_p.py::test_k", "tests/x/test_p.py::test_k[case1]"],
        nodes,
        set(),
        repo=_skeleton_repo(tmp_path, ["tests/x/test_p.py"]),
    )
    assert verdict.hits == ["tests/x/test_p.py::test_k[case1]"]
    assert len(verdict.misses) == 1 and verdict.misses[0].startswith(
        "tests/x/test_p.py::test_k(该文件里没有同名用例)"
    )


def test_exact_membership_class_chain(tmp_path: Path):
    """`::Class::` 链整体匹配：`TestA::test_m` 与 `TestB::test_m` 互不为对方掩护。"""
    nodes, _ = parse_collection_output(
        "tests/x/test_c.py::TestA::test_m\ntests/x/test_c.py::TestB::test_m\n"
    )
    verdict = judge(
        ["tests/x/test_c.py::TestA::test_m", "tests/x/test_c.py::TestC::test_m"],
        nodes,
        set(),
        repo=_skeleton_repo(tmp_path, ["tests/x/test_c.py"]),
    )
    assert verdict.hits == ["tests/x/test_c.py::TestA::test_m"]
    assert len(verdict.misses) == 1


def test_uncollectable_is_warning_not_red(tmp_path: Path):
    """收集失败 ⇒ UNCOLLECTABLE（黄），不进 misses，不判红。"""
    verdict = judge(
        ["tests/x/test_bad.py::test_boom"],
        set(),
        {"tests/x/test_bad.py"},
        repo=_skeleton_repo(tmp_path, ["tests/x/test_bad.py"]),
    )
    assert verdict.uncollectable == [
        "tests/x/test_bad.py::test_boom(所在文件本次收集失败: tests/x/test_bad.py)"
    ]
    assert verdict.misses == []
    assert verdict.red is False


def test_nofile_is_red():
    """条目文件在磁盘上不存在 ⇒ NOFILE（红，㊵-467 原病）。"""
    verdict = judge(["tests/nowhere/test_gone.py::test_x"], set(), set())
    assert verdict.nofile and verdict.nofile[0].startswith(
        "tests/nowhere/test_gone.py::test_x(路径不存在"
    )
    assert verdict.red is True


def test_malformed_entry_without_method_is_red():
    verdict = judge(["tests/x/test_a.py"], set(), set())
    assert verdict.misses == ["tests/x/test_a.py(条目缺 ::方法名)"]
    assert verdict.red is True


def test_dedupe_counts_unique_not_legs():
    """两腿 env 各一份 ⇒ 扫到 172 行；清单条数是去重后的条数（86），不得把 172 当条数。"""
    leg = ["tests/a.py::t1", "tests/a.py::t2"]
    assert dedupe_preserving_order(leg + leg) == leg


def test_listed_files_are_sorted_unique_file_paths():
    files = listed_files(["tests/b.py::t2", "tests/a.py::t1", "tests/b.py::t1"])
    assert files == ["tests/a.py", "tests/b.py"]


def test_error_lines_are_parsed_not_swallowed_as_nodes():
    stdout = (
        "tests/x/test_ok.py::test_fine\n"
        "______________________ ERROR collecting tests/x/test_bad.py ______________________\n"
        "tests/x/test_bad.py:1: in <module>\n"
        "E   ModuleNotFoundError: No module named 'missing_dep'\n"
        "=== short test summary info ===\n"
        "ERROR tests/x/test_bad.py - ModuleNotFoundError: No module named 'missing_dep'\n"
        "=== 1 collected, 1 error in 0.02s ===\n"
    )
    nodes, error_files = parse_collection_output(stdout)
    assert nodes == {"tests/x/test_ok.py::test_fine"}
    assert error_files == {"tests/x/test_bad.py"}


def test_error_file_of_both_line_shapes():
    assert _error_file_of("______ ERROR collecting tests/a.py ______") == "tests/a.py"
    assert _error_file_of("ERROR tests/a.py - ImportError: boom") == "tests/a.py"
    assert _error_file_of("=== 1 collected, 1 error in 0.02s ===") is None
    assert _error_file_of("some traceback noise :: x[1:2:3]") is None


def test_end_to_end_substring_injection_with_real_pytest_collection(tmp_path: Path):
    """端到端红绿双向：用真实 pytest --co 收集一个只含 test_alpha_beta 的文件，
    清单里写已删的 test_alpha ⇒ 新守卫判红；test_alpha_beta 判绿。
    （旧版两行 stdout 里互相是子串，会双双误判绿。）"""
    (tmp_path / "test_sample.py").write_text(
        "def test_alpha_beta():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    collected = collect_listed_files(["test_sample.py"], repo=tmp_path)
    assert collected != 3
    nodes, error_files = collected
    assert nodes == {"test_sample.py::test_alpha_beta"}
    verdict = judge(
        ["test_sample.py::test_alpha", "test_sample.py::test_alpha_beta"],
        nodes,
        error_files,
        repo=tmp_path,
    )
    assert verdict.hits == ["test_sample.py::test_alpha_beta"]
    assert verdict.misses == ["test_sample.py::test_alpha(该文件里没有同名用例)"]
    assert verdict.red is True


def test_true_prefix_entry_is_miss_while_old_substring_logic_would_pass(tmp_path: Path):
    """批次 0 ②（AR 规格）：条目是某真实 node-id 的**真前缀**
    （条目 `...::test_a`，实际收集 `...::test_ab`）。
    红绿双向——旧版子串判定 `method in collected.stdout` 在本场景为真（误判 OK），
    新守卫必须 MISS 判红；断言消息含判据原文，便于与其它失败通道区分。"""
    repo = _skeleton_repo(tmp_path, ["tests/x/test_a.py"])
    stdout = "tests/x/test_a.py::test_ab\n=== 1 collected in 0.01s ===\n"
    nodes, error_files = parse_collection_output(stdout)
    assert nodes == {"tests/x/test_a.py::test_ab"}

    # 红腿：旧版子串逻辑的判定式在本场景成立（旧版会判绿——这正是缺陷本体）
    assert "test_a" in stdout

    # 绿腿：新守卫精确成员判定 ⇒ 真前缀条目 MISS 判红
    verdict = judge(["tests/x/test_a.py::test_a"], nodes, error_files, repo=repo)
    assert verdict.misses == ["tests/x/test_a.py::test_a(该文件里没有同名用例)"]
    assert verdict.hits == []
    assert verdict.red is True


def test_empty_deselect_list_is_exit2(tmp_path: Path, monkeypatch, capsys):
    """A6（AR 附录 A 顺带项）：fail-closed——清单被删/为空 ⇒ exit 2，
    不允许退化为「没清单＝全通过」（守卫头注释第 5 条的判据原文）。
    两个 return-2 通道各自触发一次：
    ① WORKFLOW 指向不存在的路径（数据源不可读）；
    ② WORKFLOW 存在但零条目（清单被清空/键名被改）。"""
    # ① 数据源不可读 ⇒ return 2
    monkeypatch.setattr(guard, "WORKFLOW", tmp_path / "no_such_ci.yml")
    assert guard.main() == 2

    # ② 清单被清空 ⇒ return 2（文件合法存在，但没有任何行首 --deselect 条目）
    empty = tmp_path / "ci.yml"
    empty.write_text("name: no-deselect-here\n", encoding="utf-8")
    monkeypatch.setattr(guard, "WORKFLOW", empty)
    assert guard.main() == 2
    out = capsys.readouterr().out
    assert "::error::" in out and "没读到任何" in out and guard.KEY in out
