"""控制台字形普查仪（`audit_taiji_console_glyph_encodability.py`）的契约测。

这台仪器的**全部价值**在于分档正确，所以本册钉的都是"哪一档必须为空／必须非空"：

* stdout 档＝会崩的那一面，`⇒` 落在这里必须进 `crash_face`；
* stderr 档＝实测**不崩**（`sys.stderr.errors=backslashreplace`），绝不能被算进崩溃面；
* pool 档＝没做数据流的保守上界，必须单独存在、不许与前两档相加；
* docstring 永远不算消息；`reconfigure(encoding="utf-8")` 必须真的能豁免；
* 解析不了的件要**披露**而不是让整门死掉，也不许悄悄少扫。

另有一支真树安全钉：主线这次要跑的那五枚件**不许出现在崩溃面里**——这一句在本轮之前只有"我以为"，
现在是可复算的断言（`train_seed_corpus.py` 里那些 `⇒` 只进异常消息，落在 pool 档）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_console_glyph_encodability.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_17_console_glyph_census", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()

#: 主线本轮真正要执行的命令面（双臂续训＋跑道器四步）。它们出现在崩溃里就意味着
#: owner 在终端会看到 traceback 而不是拒绝清单。
MAINLINE_RUN_SET = (
    "scripts/training/train_seed_corpus.py",
    "scripts/training/run_taiji_n5_retention_lane.py",
    "scripts/training/check_taiji_n5_pairing_preflight.py",
    "scripts/training/eval_taiji_cap0_baseline.py",
    "scripts/training/measure_taiji_a30_repetition_penalty.py",
    "scripts/training/adjudicate_taiji_n2_04_retention_pair.py",
)

STDOUT_GUARD = """
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main() -> int:
    print("REFUSE: 缺前置 \u21d2 不跑")
    return 2
"""

STDOUT_BARE = """
def main() -> int:
    print("REFUSE: 缺前置 \u21d2 不跑")
    return 2
"""

STDERR_BARE = """
import sys

def main() -> int:
    print("REFUSE: 缺前置 \u21d2 不跑", file=sys.stderr)
    return 2
"""

DOCSTRING_ONLY = '''"""这段说明里有 \u21d2 字形，但它永远不流向控制台。"""


def main() -> int:
    return 0
'''

POOL_ONLY = """
def build() -> list[str]:
    rows = []
    rows.append(f"缺前置 \u21d2 不跑")
    return rows
"""

MIXED = """
import sys

print("stdout 侧 \u21d2 会崩")
print("stderr 侧 \u21d2 只是难看", file=sys.stderr)
"""


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "root"
    for name, body in files.items():
        path = root / "scripts" / "training" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8", newline="\n")
    return root


def _crash_files(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {one["file"]: one for one in payload["crash_face"]}


def test_a_stdout_glyph_print_lands_in_the_crash_face(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"bare.py": STDOUT_BARE})
    payload = MODULE.audit(root, "gbk")
    row = _crash_files(payload)["scripts/training/bare.py"]
    assert row["stdout_lines"], row
    assert row["glyphs"] == ["U21D2"], row


def test_the_reconfigure_guard_actually_exempts(tmp_path: Path) -> None:
    """这一支防的是"豁免档是我口头说的"：同一枚消息，加了守卫就必须从崩溃面消失。"""

    root = _tree(tmp_path, {"guarded.py": STDOUT_GUARD})
    payload = MODULE.audit(root, "gbk")
    readings = {one["file"]: one for one in payload["readings"]}
    assert readings["scripts/training/guarded.py"]["console_guard"] is True
    assert payload["crash_face"] == []
    assert payload["verdicts"]["no_unguarded_stdout_glyph_print"] is True


def test_a_stderr_glyph_print_is_never_counted_as_a_crash(tmp_path: Path) -> None:
    """本轮真正的更正：stderr 走 backslashreplace，同一个字形**不崩**，两档不许相加。"""

    root = _tree(tmp_path, {"noisy.py": STDERR_BARE})
    payload = MODULE.audit(root, "gbk")
    assert payload["crash_face"] == []
    assert payload["constant_count_stderr_print"] == 1
    assert payload["escape_face_file_count"] == 1


def test_a_docstring_is_not_a_message(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"prosey.py": DOCSTRING_ONLY})
    payload = MODULE.audit(root, "gbk")
    assert payload["constant_count_stdout_print"] == 0
    assert payload["constant_count_message_pool"] == 0
    assert payload["crash_face"] == []


def test_an_unprinted_message_goes_to_the_conservative_pool_only(tmp_path: Path) -> None:
    """pool 档必须**单独**存在：它今天抓住了判读器与训练器，但那些字形只进异常消息，不构成崩溃面。"""

    root = _tree(tmp_path, {"pooled.py": POOL_ONLY})
    payload = MODULE.audit(root, "gbk")
    assert payload["crash_face"] == []
    assert payload["constant_count_message_pool"] == 1
    assert payload["unclassified_pool_face_file_count"] == 1


def test_two_sinks_in_one_file_stay_in_their_own_faces(tmp_path: Path) -> None:
    """同一枚文件两种落点 ⇒ 定罪只看 stdout 那一半，而两档**同时在场**（它们不是互斥分档）。"""

    root = _tree(tmp_path, {"mixed.py": MIXED})
    payload = MODULE.audit(root, "gbk")
    row = _crash_files(payload)["scripts/training/mixed.py"]
    assert len(row["stdout_lines"]) == 1, row
    assert payload["constant_count_stdout_print"] == 1, payload
    assert payload["constant_count_stderr_print"] == 1, payload
    #: 同一枚文件既进崩溃面也进转义面——相加会把它数两次，所以两档各有各的计数。
    assert payload["crash_face_file_count"] == 1
    assert payload["escape_face_file_count"] == 1


def test_an_unparsable_file_is_disclosed_not_fatal(tmp_path: Path) -> None:
    """㊵-649 的实测形状：`scripts/archive/legacy_*.py` 带 BOM，`ast.parse` 会 SyntaxError。

    整门不该为一枚遗留件死掉，但**也不许悄悄少扫**——`unparsed_files` 与判定必须一起出版。
    """

    root = _tree(tmp_path, {"broken.py": "def f(:\n    pass\n"})
    payload = MODULE.audit(root, "gbk")
    assert payload["unparsed_file_count"] == 1
    assert payload["unparsed_files"][0]["file"] == "scripts/training/broken.py"
    assert payload["verdicts"]["census_covered_every_scanned_file"] is False


def test_an_empty_scan_face_refuses_instead_of_reporting_zero(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    rc = MODULE.main(["--out-report", str(out), "--root", str(tmp_path / "nothing")])
    assert rc == 2, rc
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "census_failed", payload
    assert "扫面为空" in payload["error"], payload


def test_the_sealed_mainline_run_set_is_absent_from_the_crash_face() -> None:
    """真树钉：owner 要跑的那五枚**不在崩溃面**里。谁往它们里加一支带字形的 stdout print，这里当场红。"""

    payload = MODULE.audit(REPO, "gbk")
    present = {one["file"] for one in payload["crash_face"]}
    hit = sorted(name for name in MAINLINE_RUN_SET if name in present)
    assert hit == [], hit
    #: 反向核对：这条断言不是靠"扫到 0 枚文件"蒙对的。
    assert payload["files_scanned"] > 900, payload["files_scanned"]
    assert payload["crash_face_file_count"] > 0, payload["crash_face_file_count"]


def test_the_live_training_face_crash_list_is_down_to_the_library_only() -> None:
    """㊵-652：三枚入口脚本加守卫后，`scripts/training` 的崩溃面只剩 `utils.py` 一枚——**而且它是库**。

    库里不许改全局 stdout 编码（import 副作用会落到 7 枚调用方上），所以这一枚的修法只能在入口做。
    谁往那三枚已守卫的入口里删掉守卫，或往 `scripts/training` 的别的入口加字形 print，这一支当场红。
    """

    payload = MODULE.audit(REPO, "gbk")
    live = sorted(
        one["file"] for one in payload["crash_face"] if one["file"].startswith("scripts/training")
    )
    assert live == ["scripts/training/utils.py"], live
    guarded = {
        one["file"]: one["console_guard"]
        for one in payload["readings"]
        if one["file"]
        in {
            "scripts/training/train_cross_domain_collab.py",
            "scripts/training/train_tinystories.py",
            "scripts/training/train_tinystories_field.py",
        }
    }
    assert len(guarded) == 3 and all(guarded.values()), guarded


def test_help_names_every_flag() -> None:
    import subprocess

    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, check=False)
    text = proc.stdout.decode("utf-8", errors="replace")
    assert proc.returncode == 0, proc.returncode
    for flag in ("--out-report", "--root", "--codec"):
        assert flag in text, (flag, text)


HELP_DOC_ONLY = '''"""这支仪器的用法屏带着 \\u21d2 字形。"""

import argparse

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--x")
'''

HELP_KWARG = """
import argparse

p = argparse.ArgumentParser()
p.add_argument("--x", help="缺前置 \\u21d2 不跑")
"""

DOCSTRING_WITHOUT_PARSER = '''"""这里的 \\u21d2 永远不上任何屏，因为这枚文件没有 ArgumentParser。"""

import sys

sys.stdout.write("干净的输出")
'''


def test_the_help_screen_is_its_own_face_not_a_crash(tmp_path: Path) -> None:
    """㊵-651 的实测形状：`description=__doc__` 把 docstring 送上 `--help` ⇒ 必须算，但不能算成"正常跑会崩"。

    并档的代价是真数：第一版把两档合并时 `crash_face` 从 **46 枚跳到 108 枚**，
    而用法屏只在 `--help` 那条路写 stdout（`parser.error` 的 usage 落 stderr ⇒ 转义不崩）。
    """

    root = _tree(tmp_path, {"helpdoc.py": HELP_DOC_ONLY, "helpkw.py": HELP_KWARG})
    payload = MODULE.audit(root, "gbk")
    assert payload["help_face_file_count"] == 2, payload["help_face"]
    assert payload["constant_count_argparse_surface"] == 2, payload["help_face"]
    assert payload["crash_face"] == [], payload["crash_face"]
    assert payload["verdicts"]["no_unguarded_argparse_help_glyph"] is False


def test_a_docstring_without_argparse_is_still_not_a_message(tmp_path: Path) -> None:
    """反向那一支：排除 docstring 的规矩没被整条废掉——只有被喂进用法屏时它才是消息。"""

    root = _tree(tmp_path, {"nodoc.py": DOCSTRING_WITHOUT_PARSER})
    payload = MODULE.audit(root, "gbk")
    assert payload["constant_count_argparse_surface"] == 0, payload
    assert payload["constant_count_message_pool"] == 0, payload
    assert payload["crash_face"] == [] and payload["help_face"] == []


def _tree_rel(tmp_path: Path, files: dict[str, str]) -> Path:
    """按**相对路径**铺树（`_tree` 只会往 `scripts/training` 铺，分不了档）。"""

    root = tmp_path / "root"
    for relative, body in files.items():
        path = root / Path(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8", newline="\n")
    return root


def test_the_bucket_split_separates_unreachable_code_from_live_code(tmp_path: Path) -> None:
    """㊵-668：标题数把 42 枚归档件与 1 枚活件混着数，分档必须把它们分开而**不删任何一行**。"""

    root = _tree_rel(
        tmp_path,
        {"scripts/archive/dead.py": STDOUT_BARE, "scripts/training/live.py": STDOUT_BARE},
    )
    payload = MODULE.audit(root, "gbk")
    assert payload["crash_face_file_count"] == 2, payload["crash_face"]
    assert payload["crash_face_live_file_count"] == 1, payload["crash_face_live"]
    assert payload["crash_face_archived_file_count"] == 1, payload["crash_face"]
    assert payload["crash_face_live"] == ["scripts/training/live.py"], payload["crash_face_live"]
    #: 等式在**这里**钉，不在仪器里钉：仪器出版的两枚计数由同一批行对象数出来，
    #: 把等式写成键就是一件恒真读数（㊵-618 对 `shadow_materialized` 的同一课）。
    #: 这一行按 `crash_face` 行的 `bucket` 字段重数一遍，取法与那两枚计数不同。
    recounted = sorted(
        one["file"] for one in payload["crash_face"] if one["bucket"] == "live"
    )
    assert recounted == payload["crash_face_live"], (recounted, payload["crash_face_live"])
    #: 整面判定不因为分档而翻绿——两枚都还红着。
    assert payload["verdicts"]["no_unguarded_stdout_glyph_print"] is False
    assert payload["verdicts"]["no_unguarded_live_stdout_glyph_print"] is False


def test_guarding_the_live_file_alone_flips_only_the_live_verdict(tmp_path: Path) -> None:
    """能为 false 也能为 true：给活码加守卫 ⇒ 活码面清零，而整面照旧红（归档件还在）。

    这一支是分档唯一有判别力的方向——上一支两枚判定都是 False，看不出分档有没有在做的事；
    这一支里两枚判定**分岔**，所以"拿活码面代答已清"会被当场抓住。
    """

    root = _tree_rel(
        tmp_path,
        {"scripts/archive/dead.py": STDOUT_BARE, "scripts/training/live.py": STDOUT_GUARD},
    )
    payload = MODULE.audit(root, "gbk")
    assert payload["crash_face_live_file_count"] == 0, payload["crash_face_live"]
    assert payload["crash_face_archived_file_count"] == 1, payload["crash_face"]
    assert payload["crash_face_file_count"] == 1, payload["crash_face"]
    assert payload["verdicts"]["no_unguarded_live_stdout_glyph_print"] is True
    assert payload["verdicts"]["no_unguarded_stdout_glyph_print"] is False


def test_scripts_legacy_is_counted_live_because_nothing_proves_it_unimported(
    tmp_path: Path,
) -> None:
    """豁免必须有出处：`scripts/legacy/` 不在归档前缀里，它的字形必须算进活码面。"""

    root = _tree_rel(tmp_path, {"scripts/legacy/old.py": STDOUT_BARE})
    payload = MODULE.audit(root, "gbk")
    assert payload["archived_prefixes"] == ["scripts/archive/"], payload["archived_prefixes"]
    assert payload["crash_face_live_file_count"] == 1, payload["crash_face_live"]
    assert payload["crash_face_archived_file_count"] == 0, payload["crash_face"]


def test_the_real_tree_live_crash_face_is_the_library_and_the_partition_is_exhaustive() -> None:
    """真树双读数：仪器的 bucket 与按前缀手工重数必须给同一张活码面（现在＝`utils.py` 一枚）。

    现读（㊵-668）：`crash_face_file_count` 43＝live 1＋archived 42；`help_face` 62 枚全在 live。
    谁往活码目录加一支带字形的 stdout print，`crash_face_live` 立刻变长，这一支当场红。
    """

    payload = MODULE.audit(REPO, "gbk")
    assert payload["crash_face_live"] == ["scripts/training/utils.py"], payload["crash_face_live"]
    manual_live = sorted(
        one["file"]
        for one in payload["crash_face"]
        if not one["file"].startswith(tuple(payload["archived_prefixes"]))
    )
    assert manual_live == payload["crash_face_live"], (manual_live, payload["crash_face_live"])
    assert (
        payload["crash_face_live_file_count"] + payload["crash_face_archived_file_count"]
        == payload["crash_face_file_count"]
    ), payload
    #: 用法屏面一台机器上都跑得到 ⇒ 归档档必须为空；不为空就是有人把新仪器写进了 archive。
    assert payload["help_face_archived_file_count"] == 0, payload["help_face_archived_file_count"]
    assert payload["help_face_live_file_count"] == payload["help_face_file_count"], payload



def test_the_fixed_a30_probe_is_absent_from_every_console_face() -> None:
    """真树钉：㊵-651 给 `probe_taiji_a30_stop_failure.py` 加了守卫（那支长红因此结清）。

    谁删掉守卫，或往它的 `help=`／docstring 里加回不可编码字形而不加守卫，这一支当场红。
    """

    payload = MODULE.audit(REPO, "gbk")
    rows = [
        one
        for one in payload["readings"]
        if one["file"].endswith("probe_taiji_a30_stop_failure.py")
    ]
    assert len(rows) == 1, rows
    assert rows[0]["console_guard"] is True, rows[0]
    for face in ("crash_face", "help_face"):
        hits = [
            one for one in payload[face] if one["file"].endswith("probe_taiji_a30_stop_failure.py")
        ]
        assert hits == [], (face, hits)
