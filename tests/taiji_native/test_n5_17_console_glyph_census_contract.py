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


def test_help_names_every_flag() -> None:
    import subprocess

    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, check=False)
    text = proc.stdout.decode("utf-8", errors="replace")
    assert proc.returncode == 0, proc.returncode
    for flag in ("--out-report", "--root", "--codec"):
        assert flag in text, (flag, text)
