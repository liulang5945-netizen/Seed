"""R-S3 补充契约测试（ADR-S3 §3.4 C-4 + §6.2 ⑤）：split_ci_junit_faces 的合成回归。

**为什么必须有本文件**：派生正确性依赖 junit 的 classname 形态学与
`Path.is_relative_to` 归属算法，其中**尾点陷阱**（`tests.seed` 字符串前缀会误吞
兄弟目录 `tests.seed_platform`，实测 8 文件）是本脚本的存在理由之一——
这条判据的合成回归用例必须入库（architecture-reviewer §6.2 ⑤：
"必须写成用例"），否则归属算法退回字符串前缀时 CI 不会红。

判据口径 = 主理人 C-4 契约修正（08-test.md §12.0 两行表，2026-10-26 定版）：
- 空 classname 判**子元素**：`<error message="collection failure">` / `<failure>` /
  未知形态 ⇒ exit 5 + 打印模块清单；`<skipped>` ⇒ 合法模块级 skip
  （importorskip / allow_module_level）⇒ 排除出面 + 单独计数 module_skipped，不判红；
- 第 2 级分母 = 唯一 node-id 数（classname+name 去重），不用 pytest 的 `tests=`
  属性（它把模块级 skip 行也计入，基线 native 面 3903 vs 3901 恒差 4）；
- 基线锚（run 40171918472）：uid 3901+185+490=4576、module_skipped=4、
  派生面 vs 同 run 独立面双向 diff=0（本文件用合成样本锁同一套行为）。

退出码 0/2/3/4/5 见 ADR §3.8。脚本 stdlib only + fail-closed + 幂等。

**exit-4 通道的区分（批次 0 ①④ 规格）**：exit 4 有三个子通道，断言消息必须含
判据原文以区分失败原因：
- 疫苗通道（尾点陷阱的机械后果）："面混入非对应目录文件"；
- 空面通道："某个面被悄悄吞掉"；
- 三面之和不等通道：**从 CLI 输入面上不可构造**——正常归属是
  `<testcase>` 集合的划分（每条恰落一桶），uid 守恒由构造保证；它只在
  **实现退化**（如字符串前缀误归属）时可能发生，而那种退化会先被疫苗通道拦截
  （误归属的文件不在对应目录下）。故本文件用「退化实现注入」的 shim 让
  守恒层的前置疫苗真实触发 exit 4，并断言疫苗通道判据原文；空面通道由
  test_seed_face_empty_is_exit4 以判据原文锁定。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "training" / "split_ci_junit_faces.py"

XML_DECL = '<?xml version="1.0" encoding="utf-8"?>'


def _suite(
    *cases: str, failures: int = 0, skipped: int = 0, errors: int = 0, time: str = "1.000"
) -> str:
    return (
        f'<testsuite name="pytest" errors="{errors}" failures="{failures}" '
        f'skipped="{skipped}" tests="{len(cases)}" time="{time}">' + "".join(cases) + "</testsuite>"
    )


def _suites(*suites: str) -> str:
    return XML_DECL + '\n<testsuites name="pytest">' + "".join(suites) + "</testsuites>\n"


def _tc(classname: str, name: str, time: str = "0.500", child: str = "") -> str:
    return f'<testcase classname="{classname}" name="{name}" time="{time}">{child}</testcase>'


def _repo_skeleton(tmp_path: Path) -> Path:
    for rel in (
        "tests/taiji_native/fake_a.py",
        "tests/seed/fake_b.py",
        "tests/seed_platform/fake_c.py",
        "tests/test_root_mod.py",
    ):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    return tmp_path


def _run(tmp_path: Path, source_xml: str, repo: Path, tag: str) -> subprocess.CompletedProcess:
    source = tmp_path / f"report_{tag}.xml"
    source.write_text(source_xml, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--native",
            str(tmp_path / f"native_{tag}.xml"),
            "--seed",
            str(tmp_path / f"seed_{tag}.xml"),
            "--repo-root",
            str(repo),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_face_of_rejects_seed_platform_sibling_trap(tmp_path: Path):
    """尾点陷阱的直接断言：tests/seed_platform 不是 tests/seed 的子孙。
    归属必须 is_relative_to（逐目录分量），字符串前缀 `tests.seed` 在这里会误吞。"""
    from scripts.training.split_ci_junit_faces import face_of

    repo = _repo_skeleton(tmp_path)
    assert face_of(repo / "tests/seed/fake_b.py", repo) == "seed"
    assert face_of(repo / "tests/taiji_native/fake_a.py", repo) == "native"
    # 兄弟目录与 tests/ 根：两面都不进
    assert face_of(repo / "tests/seed_platform/fake_c.py", repo) is None
    assert face_of(repo / "tests/test_root_mod.py", repo) is None


def test_collection_error_form_is_exit5_with_module_list(tmp_path: Path):
    repo = _repo_skeleton(tmp_path)
    xml = _suites(
        _suite(
            _tc("tests.taiji_native.fake_a", "test_ok"),
            _tc(
                "",
                "tests.taiji_native.test_bad",
                child='<error message="collection failure">ImportError: boom</error>',
            ),
        )
    )
    proc = _run(tmp_path, xml, repo, "err")
    assert proc.returncode == 5, proc.stdout
    assert "tests.taiji_native.test_bad" in proc.stdout


def test_unknown_empty_classname_form_fails_closed(tmp_path: Path):
    """classname=='' 且无 <skipped> 子节点（无子节点/未知形态）⇒ fail-closed exit 5。"""
    repo = _repo_skeleton(tmp_path)
    xml = _suites(_suite(_tc("", "tests.mystery.module", time="0.000")))
    proc = _run(tmp_path, xml, repo, "unknown")
    assert proc.returncode == 5, proc.stdout


def test_module_level_skip_is_excluded_and_counted_not_red(tmp_path: Path):
    """合法模块级 skip（<skipped>）⇒ exit 0；排除出面 + module_skipped 单独记账；
    taiji_native 模块的 skip 行**不得**混进派生 native 面（C-4 裁决）。"""
    repo = _repo_skeleton(tmp_path)
    skip_native = _tc(
        "",
        "tests.taiji_native.fake_a",
        time="0.000",
        child='<skipped message="collection skipped" type="pytest.skip"/>',
    )
    skip_root = _tc(
        "",
        "tests.test_root_mod",
        time="0.000",
        child='<skipped message="collection skipped" type="pytest.skip"/>',
    )
    xml = _suites(
        _suite(
            _tc("tests.taiji_native.fake_a", "test_native_a"),
            _tc("tests.seed.fake_b", "test_seed_a"),
            skip_native,
            skip_root,
        )
    )
    proc = _run(tmp_path, xml, repo, "skip")
    assert proc.returncode == 0, proc.stdout
    assert "module_skipped=2" in proc.stdout
    native_out = (tmp_path / "native_skip.xml").read_text(encoding="utf-8")
    seed_out = (tmp_path / "seed_skip.xml").read_text(encoding="utf-8")
    assert "test_native_a" in native_out and "fake_a" in native_out
    assert "test_seed_a" in seed_out
    # skip 记录（无 classname 属性值）不出现在任何派生面
    assert 'classname=""' not in native_out
    assert 'classname=""' not in seed_out


def test_dot_trap_seed_platform_never_enters_faces_and_conservation_holds(tmp_path: Path):
    """§6.2 ⑤ 尾点陷阱回归：tests.seed_platform 的 classname 与 tests.seed 并存时，
    不得被前缀误吞进 seed 面（也不得进 native 面），唯一 node-id 三面守恒。"""
    repo = _repo_skeleton(tmp_path)
    xml = _suites(
        _suite(
            _tc("tests.taiji_native.fake_a", "test_native_a"),
            _tc("tests.taiji_native.fake_a", "TestG::test_native_b"),
            _tc("tests.seed.fake_b", "test_seed_a"),
            _tc("tests.seed.fake_b", "test_seed_b"),
            _tc("tests.seed_platform.fake_c", "test_platform_a"),
            _tc("tests.test_root_mod", "test_root_a"),
        )
    )
    proc = _run(tmp_path, xml, repo, "trap")
    assert proc.returncode == 0, proc.stdout
    assert "uid_native=2" in proc.stdout
    assert "uid_seed=2" in proc.stdout
    assert "uid_other=2" in proc.stdout
    assert "uid_total=6" in proc.stdout
    native_out = (tmp_path / "native_trap.xml").read_text(encoding="utf-8")
    seed_out = (tmp_path / "seed_trap.xml").read_text(encoding="utf-8")
    assert "seed_platform" not in native_out
    assert "seed_platform" not in seed_out
    assert "test_platform_a" not in native_out + seed_out


def test_dot_trap_red_green_degenerate_prefix_impl_must_exit4(tmp_path: Path):
    """批次 0 ①（AR 规格）尾点陷阱红绿双向：
    - 绿腿：现行实现（is_relative_to）⇒ tests.seed_platform 不进两面，exit 0；
    - 红腿：注入**退化实现**（不带尾点的字符串前缀匹配）⇒ 守恒层的前置疫苗
      必须 exit 4（红），且断言疫苗通道判据原文与空面通道区分。
    退化实现通过子进程 shim 注入：加载真实脚本模块后替换 face_of，
    不改动仓库内的脚本本身（脚本行为零变化）。"""
    repo = _repo_skeleton(tmp_path)
    xml = _suites(
        _suite(
            _tc("tests.taiji_native.fake_a", "test_native_a"),
            _tc("tests.seed.fake_b", "test_seed_a"),
            _tc("tests.seed_platform.fake_c", "test_platform_a"),
        )
    )
    # 绿腿：现行实现
    ok = _run(tmp_path, xml, repo, "green")
    assert ok.returncode == 0, ok.stdout
    assert "uid_seed=1" in ok.stdout  # 只有真 tests/seed 用例，seed_platform 未被吞
    seed_out = (tmp_path / "seed_green.xml").read_text(encoding="utf-8")
    assert "test_seed_a" in seed_out
    assert "test_platform_a" not in seed_out

    # 红腿：退化实现（不带尾点前缀）注入 ⇒ 必须 exit 4
    shim = tmp_path / "degenerate_prefix_shim.py"
    shim.write_text(
        "\n".join(
            [
                "import importlib.util",
                "import sys",
                "from pathlib import Path",
                "",
                "spec = importlib.util.spec_from_file_location(",
                "    'orig_script', {script!r}",
                ")",
                "mod = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(mod)",
                "",
                "def degenerate_face_of(path, repo_root):",
                "    # 退化实现：不带尾点的字符串前缀匹配（tests.seed 吞掉 tests.seed_platform）",
                "    posix = str(path).replace(chr(92), '/')",
                "    if posix.startswith(str(Path(repo_root) / 'tests' / 'seed').replace(chr(92), '/')):",
                "        return 'seed'",
                "    if posix.startswith(",
                "        str(Path(repo_root) / 'tests' / 'taiji_native').replace(chr(92), '/')",
                "    ):",
                "        return 'native'",
                "    return None",
                "",
                "mod.face_of = degenerate_face_of",
                "sys.argv = [",
                "    'split_ci_junit_faces.py',",
                "    '--source', {source!r},",
                "    '--native', {native!r},",
                "    '--seed', {seed!r},",
                "    '--repo-root', {repo!r},",
                "]",
                "raise SystemExit(mod.main())",
            ]
        ).format(
            script=str(SCRIPT),
            source=str(tmp_path / "report_degenerate.xml"),
            native=str(tmp_path / "native_degenerate.xml"),
            seed=str(tmp_path / "seed_degenerate.xml"),
            repo=str(repo),
        ),
        encoding="utf-8",
    )
    (tmp_path / "report_degenerate.xml").write_text(xml, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(shim)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 4, proc.stdout + proc.stderr
    # 判据原文：疫苗通道（与空面通道"某个面被悄悄吞掉"区分）
    assert "面混入非对应目录文件" in proc.stdout
    assert "seed_platform" in proc.stdout


def test_collection_error_fixture_is_exit5_with_module_list(tmp_path: Path):
    """批次 0 ③：合成 <error message="collection failure"> fixture
    ⇒ exit 5 并打印模块清单（fail-closed，S1/S6 共同安全网）。"""
    repo = _repo_skeleton(tmp_path)
    xml = _suites(
        _suite(
            _tc("tests.taiji_native.fake_a", "test_ok"),
            _tc(
                "",
                "tests.taiji_native.test_broken_module",
                child='<error message="collection failure">ImportError: boom</error>',
            ),
            _tc(
                "",
                "tests.seed.test_other_broken",
                child='<error message="collection failure">ImportError: bang</error>',
            ),
        )
    )
    proc = _run(tmp_path, xml, repo, "collfail")
    assert proc.returncode == 5, proc.stdout
    assert "tests.taiji_native.test_broken_module" in proc.stdout
    assert "tests.seed.test_other_broken" in proc.stdout


def test_seed_face_empty_is_exit4(tmp_path: Path):
    repo = _repo_skeleton(tmp_path)
    xml = _suites(
        _suite(
            _tc("tests.taiji_native.fake_a", "test_native_a"),
            _tc("tests.taiji_native.fake_a", "test_native_c"),
        )
    )
    proc = _run(tmp_path, xml, repo, "empty")
    assert proc.returncode == 4, proc.stdout
    # 判据原文：与疫苗通道（"面混入非对应目录文件"）区分失败原因
    assert "某个面被悄悄吞掉" in proc.stdout
    assert "uid(seed)=0" in proc.stdout


def test_unresolvable_classname_is_exit3(tmp_path: Path):
    repo = _repo_skeleton(tmp_path)
    xml = _suites(_suite(_tc("tests.taiji_native.no_such_mod_xyz", "test_ghost")))
    proc = _run(tmp_path, xml, repo, "ghost")
    assert proc.returncode == 3, proc.stdout


def test_missing_source_is_exit2(tmp_path: Path):
    """source 指向不存在文件 ⇒ exit 2（fail-closed）。"""
    repo = _repo_skeleton(tmp_path)
    missing = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(tmp_path / "no_such_report.xml"),
            "--native",
            str(tmp_path / "n.xml"),
            "--seed",
            str(tmp_path / "s.xml"),
            "--repo-root",
            str(repo),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert missing.returncode == 2


def test_idempotent_bytes_for_same_input(tmp_path: Path):
    repo = _repo_skeleton(tmp_path)
    xml = _suites(
        _suite(
            _tc("tests.taiji_native.fake_a", "test_native_a"),
            _tc("tests.seed.fake_b", "test_seed_a"),
        )
    )
    first = _run(tmp_path, xml, repo, "idem1")
    second = _run(tmp_path, xml, repo, "idem2")
    assert first.returncode == 0 and second.returncode == 0
    assert (tmp_path / "native_idem1.xml").read_bytes() == (
        tmp_path / "native_idem2.xml"
    ).read_bytes()
    assert (tmp_path / "seed_idem1.xml").read_bytes() == (tmp_path / "seed_idem2.xml").read_bytes()


def test_derived_face_keeps_failure_and_recomputes_counters(tmp_path: Path):
    """派生面深拷贝 failure 子节点并重算 testsuite 计数（缺字段补 0、不留旧值）。"""
    repo = _repo_skeleton(tmp_path)
    xml = _suites(
        _suite(
            _tc("tests.taiji_native.fake_a", "test_ok", time="1.000"),
            _tc(
                "tests.taiji_native.fake_a",
                "test_boom",
                time="0.250",
                child='<failure message="AssertionError">boom</failure>',
            ),
            _tc("tests.seed.fake_b", "test_seed_a", time="0.750"),
            failures=1,
        ),
    )
    proc = _run(tmp_path, xml, repo, "recount")
    assert proc.returncode == 0, proc.stdout
    native_out = (tmp_path / "native_recount.xml").read_text(encoding="utf-8")
    assert "<failure" in native_out
    assert 'tests="2"' in native_out
    assert 'failures="1"' in native_out
    assert 'errors="0"' in native_out
    assert 'skipped="0"' in native_out
    assert 'time="1.250000"' in native_out


def test_parametrized_ids_never_collide_across_faces(tmp_path: Path):
    """参数化后缀整体属于 node-id 的一部分：同基名不同参数是不同 id，去重不得吞并。"""
    repo = _repo_skeleton(tmp_path)
    xml = _suites(
        _suite(
            _tc("tests.taiji_native.fake_a", "test_k[case1]"),
            _tc("tests.taiji_native.fake_a", "test_k[case2]"),
            _tc("tests.seed.fake_b", "test_m[case1]"),
        )
    )
    proc = _run(tmp_path, xml, repo, "param")
    assert proc.returncode == 0, proc.stdout
    assert "uid_native=2" in proc.stdout
    assert "uid_seed=1" in proc.stdout
