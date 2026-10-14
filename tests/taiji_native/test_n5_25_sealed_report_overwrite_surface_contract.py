"""`audit_taiji_sealed_report_overwrite_surface.py` 的契约册：三分类每档都要有自己的夹具。

要紧的六支（都是我这轮实跑或沙箱里真踩过的形状）：
* **只拒绝、不给覆写口**（`calibrate_taiji_r2_readout_retrain.py:215` 的 `parser.error`）**不许算进暴露面**；
* 拒绝出口不必是异常——`print(...)；return 2` 同样是拒绝（我自己那三枚就是这个形状）；
* **读侧的 `exists()`**（`if not path.exists(): raise`）不算护栏，否则真暴露面会少算；
* 默认值指向未入库路径的不许算；零命中必须响亮失败；
* 三档计数之和必须等于命中总数（分类漏一档就红）。
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_sealed_report_overwrite_surface.py"

HEAD = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out-report", default="reports/sealed.json")
args = parser.parse_args()
from pathlib import Path
target = Path(args.out_report)
"""

BARE = HEAD

REFUSE_ONLY = HEAD + """if target.exists():
    parser.error("already exists; this instrument never overwrites a report")
"""

REFUSE_WITH_FLAG = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out-report", default="reports/sealed.json")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
from pathlib import Path
target = Path(args.out_report)
if target.exists() and not args.overwrite:
    raise SystemExit(2)
"""

REFUSE_BY_RETURN = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out-report", default="reports/sealed.json")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
from pathlib import Path
target = Path(args.out_report)
if target.exists() and not args.overwrite:
    print("refusing")
    return 2
"""

READ_SIDE_RAISES = HEAD + """if not target.exists():
    raise SystemExit("input report is missing")
"""


def _load():
    spec = importlib.util.spec_from_file_location("n5_25_overwrite_surface", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "sealed.json").write_text('{"old": 1}\n', encoding="utf-8")
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "reports/sealed.json"], capture_output=True, check=True
    )
    shapes = {
        "bare.py": BARE,
        "refuse_only.py": REFUSE_ONLY,
        "refuse_with_flag.py": REFUSE_WITH_FLAG,
        "read_side_exists.py": READ_SIDE_RAISES,
        "fresh_target.py": """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out-report", default="reports/not_tracked.json")
""",
    }
    for name, source in shapes.items():
        (tmp_path / "scripts" / name).write_text(source, encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", f"scripts/{name}"], capture_output=True, check=True
        )
    return tmp_path


def test_shapes_are_classified_into_the_three_categories() -> None:
    module = _load()
    cases = {
        BARE: module.CATEGORY_NONE,
        REFUSE_ONLY: module.CATEGORY_REFUSE_ONLY,
        REFUSE_WITH_FLAG: module.CATEGORY_REFUSE_WITH_FLAG,
        REFUSE_BY_RETURN: module.CATEGORY_REFUSE_WITH_FLAG,
        READ_SIDE_RAISES: module.CATEGORY_NONE,
    }
    for source, expected in cases.items():
        category, evidence = module.classify(ast.parse(source), source)
        assert category == expected, (expected, category, source[:60])
        if expected == module.CATEGORY_NONE:
            assert evidence == []
        else:
            assert evidence, source[:60]


def test_fixture_repo_separates_the_four_instruments(tmp_path: Path) -> None:
    module = _load()
    report = module.build_report(_repo(tmp_path))
    by_file = {row["file"]: row["category"] for row in report["rows"]}
    assert by_file == {
        "scripts/bare.py": module.CATEGORY_NONE,
        "scripts/read_side_exists.py": module.CATEGORY_NONE,
        "scripts/refuse_only.py": module.CATEGORY_REFUSE_ONLY,
        "scripts/refuse_with_flag.py": module.CATEGORY_REFUSE_WITH_FLAG,
    }, by_file
    counts = report["category_counts"]
    assert counts[module.CATEGORY_NONE] == 2
    assert counts[module.CATEGORY_REFUSE_ONLY] == 1
    assert counts[module.CATEGORY_REFUSE_WITH_FLAG] == 1
    assert report["exposure_targets"] == 2
    assert report["exposure_files"] == 2
    assert "scripts/fresh_target.py" not in by_file


def test_three_categories_partition_the_hits() -> None:
    module = _load()
    report = module.build_report(PROJECT_ROOT)
    assert report["script_files_scanned"] > 500
    assert sum(report["category_counts"].values()) == report["default_targets_on_tracked_files"]
    assert report["exposure_targets"] == report["category_counts"][module.CATEGORY_NONE]
    for row in report["rows"]:
        assert (PROJECT_ROOT / row["default_target"]).is_file(), row
        assert row["flags"]


def test_zero_hits_fail_loudly_instead_of_reading_as_clean(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "empty"
    (root / "scripts").mkdir(parents=True)
    (root / "notes.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "init", str(root)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(root), "add", "notes.md"], capture_output=True, check=True)
    try:
        module.build_report(root)
    except ValueError as exc:
        assert "零命中" in str(exc)
    else:
        raise AssertionError("零命中必须响亮失败，不许读成『这面已经干净』")


def test_this_instrument_follows_its_own_rule(tmp_path: Path, monkeypatch) -> None:
    module = _load()
    target = tmp_path / "surface.json"
    target.write_text('{"sentinel": 1}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["prog", "--root", str(PROJECT_ROOT), "--out-report", str(target)]
    )
    assert module.main() == 2
    assert target.read_text(encoding="utf-8") == '{"sentinel": 1}\n'
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--root", str(PROJECT_ROOT), "--out-report", str(target), "--overwrite"],
    )
    assert module.main() == 0


def test_a_write_through_the_shared_writer_still_counts_as_a_write_target() -> None:
    """㊵-790 的成对两支：统一落盘助手（`report_io.write_text_report(target, …)`）**必须**被认成
    "把 target 当写靶用过"——否则 DEBT-G115 修法③ 每迁一台仪器就从本件的识别面上抹掉一个写靶，
    等于我自己把这道暴露门弄瞎。反面对照同册：只绑字面路径、走 `read_text` 的不算（那是读数件常量）。
    """

    module = _load()
    via_helper = (
        "import _taiji_report_io as report_io\n"
        'target = "reports/sealed.json"\n'
        "def run(text):\n"
        "    report_io.write_text_report(target, text)\n"
    )
    assert "target" in module._written_names(ast.parse(via_helper))
    category, _ = module.classify(ast.parse(via_helper), via_helper)
    assert category == "no_refusal_found", category
    #: 本件"看得见落点"的形式是**字面字符串**（或 argparse 的 default）。
    #: ㊵-790 曾把"拼串那一形（`PROJECT_ROOT / "reports" / "x.json"`）看不见"钉在下面那条断言里，
    #: 本意就是"谁扩了识别面，这一支会红并提醒他改口"——㊵-870（DEBT-G137 修法①）正是来扩它的：
    #: 扩完这条钉子按设计红了，随之改口（`37` 那枚下界也换态，v4→v5 新旧不可比，已在台账 ㊵-870 标注）。
    assert module._literal_target_names(ast.parse(via_helper)) == {
        "target": ["reports/sealed.json"]
    }
    joined = (
        "from pathlib import Path\nimport _taiji_report_io as report_io\n"
        'PROJECT_ROOT = Path(".")\n'
        'target = PROJECT_ROOT / "reports" / "sealed.json"\n'
        "def run(text):\n    report_io.write_text_report(target, text)\n"
    )
    assert "target" in module._written_names(ast.parse(joined))
    assert module._literal_target_names(ast.parse(joined)) == {"target": ["reports/sealed.json"]}
    #: 反面对照（G137 反复强调的"只绑不写不算"之外的第二类假阳性）：**片段只在拼串里出现**
    #: 但没有任何一片恰好是 `reports` 时，不许被读成落点。
    not_a_target = (
        "from pathlib import Path\n"
        'ROOT = Path(".")\n'
        'target = ROOT / "my_reports" / "sealed.json"\n'
        "def run(text):\n    target.write_text(text)\n"
    )
    assert module._literal_target_names(ast.parse(not_a_target)) == {}

    read_only = via_helper.replace(
        "    report_io.write_text_report(target, text)\n",
        '    return (target.read_text(encoding="utf-8"), "reports/sealed.json")\n',
    )
    assert "target" not in module._written_names(ast.parse(read_only))
    #: ㊵-821（DEBT-G139）：把助手那条分支关掉，同一个靶就必须**看不见**——
    #: 这正是"敏感性"的定义，也是本册里这枚新键唯一能被称为证据的形式。
    assert "target" not in module._written_names(ast.parse(via_helper), include_shared=False)


def test_the_sensitivity_key_names_only_rows_that_only_the_helper_branch_can_see(
    tmp_path: Path,
) -> None:
    """DEBT-G139 修法②：`shared_writer_sensitivity` 要能为真也要能为假。

    三枚夹具都往同一枚已入库字面靶 `reports/sealed.json` 上写，区别只在**靠哪条分支被认出来**：
    只走助手的 ⇒ 入列；助手＋普通写法都有的 ⇒ 不入列（关掉助手仍看得见，所以不依赖它）；
    只有普通写法的 ⇒ 也不入列。三枚合计 ⇒ `count == 1` 且 `rows_total` 等于命中总数。
    """

    module = _load()
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    (tmp_path / "reports" / "sealed.json").write_text('{"old": 1}\n', encoding="utf-8")
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    helper_only = (
        "import _taiji_report_io as report_io\n"
        'target = "reports/sealed.json"\n'
        "def emit(text):\n"
        "    report_io.write_json_report(target, text)\n"
    )
    helper_and_plain = (
        "import _taiji_report_io as report_io\n"
        'target = "reports/sealed.json"\n'
        "def emit(text):\n"
        "    report_io.write_json_report(target, text)\n"
        "def emit_again(text):\n"
        '    open(target, "w", encoding="utf-8").write(text)\n'
    )
    plain_only = (
        'target = "reports/sealed.json"\n'
        "def emit(text):\n"
        '    target.write_text(text, encoding="utf-8")\n'
    )
    for name, source in {
        "helper_only.py": helper_only,
        "helper_and_plain.py": helper_and_plain,
        "plain_only.py": plain_only,
    }.items():
        (tmp_path / "scripts" / name).write_text(source, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)

    report = module.build_report(tmp_path)
    block = report["shared_writer_sensitivity"]
    assert report["default_targets_on_tracked_files"] == 3, report["rows"]
    assert block["rows_total"] == 3
    assert block["count"] == 1, block
    assert block["files_recognized_only_via_shared_writer"] == ["scripts/helper_only.py"], block


def test_the_errors_keyword_is_not_a_refusal() -> None:
    #: 2026-10-13 现查的假护栏：v1 用"渲染后找子串"，`errors='replace'` 里的 `errors=` 命中了 `error`，
    #: 于是 `measure_taiji_n2_06_write_contract_values.py` 被读成"写前拒绝"，而它其实会直接覆写在册件。
    module = _load()
    source = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out-report", default="reports/sealed.json")
from pathlib import Path
target = Path("reports/other.json")
if target.exists():
    text = target.read_text(encoding="utf-8", errors="replace")
"""
    category, evidence = module.classify(ast.parse(source), source)
    assert category == module.CATEGORY_NONE
    assert evidence == []


def test_a_plain_string_return_is_not_a_refusal() -> None:
    #: 第二种假护栏：`return 'bypass_confirmed（…）'` 命中子串 `return` ⇒ 判成拒绝。
    #: 拒绝只认 `return <整数>`（我那三枚仪器的形状）／`raise`／`x.error(...)`／`exit(...)`。
    module = _load()
    source = """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out-report", default="reports/sealed.json")
from pathlib import Path
target = Path(args.out_report)
def verdict():
    if target.exists():
        return "bypass_confirmed（有答复带 U+FFFD）"
"""
    category, evidence = module.classify(ast.parse(source), source)
    assert category == module.CATEGORY_NONE
    assert evidence == []


def _literal_repo(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    (tmp_path / "reports" / "sealed.json").write_text('{"old": 1}\n', encoding="utf-8")
    (tmp_path / "reports" / "other.json").write_text('{"old": 2}\n', encoding="utf-8")
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "reports/sealed.json", "reports/other.json"],
        capture_output=True,
        check=True,
    )
    return tmp_path


def test_a_hardcoded_write_target_is_found_on_the_second_face(tmp_path: Path) -> None:
    #: v2 新增的那张面：写靶不走 argparse，而是 `out = R / "reports/sealed.json"` 这种硬编形。
    #: v1 整面看不见它——而"看不见"在这里不是中性，是把"会覆写在册件"读成"没有暴露"。
    module = _load()
    repo = _literal_repo(tmp_path)
    (repo / "scripts" / "literal.py").write_text(
        """from pathlib import Path

R = Path(".")
out = R / "reports/sealed.json"
out.write_text("{}")
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "scripts/literal.py"], capture_output=True, check=True
    )
    report = module.build_report(repo)
    rows = [row for row in report["rows"] if row["face"] == "literal_assignment"]
    assert [(row["file"], row["default_target"], row["category"]) for row in rows] == [
        ("scripts/literal.py", "reports/sealed.json", module.CATEGORY_NONE)
    ]
    assert report["face_counts"]["literal_assignment"] == 1
    assert report["exposure_targets"] == 1


def test_a_reports_constant_that_is_only_read_is_not_a_hit(tmp_path: Path) -> None:
    #: 反向控制：绑过 `reports/` 字面路径但**只读**它的不算写靶；
    #: 同一文件里真正被 write_text 的那枚（另一枚在册件）才该出现。
    module = _load()
    repo = _literal_repo(tmp_path)
    (repo / "scripts" / "mixed.py").write_text(
        """from pathlib import Path

R = Path(".")
SOURCE_REPORT = R / "reports/sealed.json"
data = SOURCE_REPORT.read_text(encoding="utf-8")
out = R / "reports/other.json"
out.write_text(data)
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "scripts/mixed.py"], capture_output=True, check=True
    )
    report = module.build_report(repo)
    targets = sorted(row["default_target"] for row in report["rows"])
    assert targets == ["reports/other.json"]
