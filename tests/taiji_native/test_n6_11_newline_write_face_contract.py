"""`DEBT-G115` 修法① 前置普查的契约册（仪器＝`audit_taiji_newline_write_face.py`）。

这台仪器最容易写错的地方不是判据，是**取哪一层当写靶**：`x.write_text(payload)` 的 args[0] 是内容，
路径是接收者 `x`。我第一版探针就是按 args[0] 取的，于是把 payload 里那句 `"evidence": "reports/…json"`
当成了写靶，点名了一枚根本没写 `reports/` 的文件——那是一枚**假阳性**，而且它朝"通过"的方向偏
（多算的处数会被读成"待办很多"，而真正的门却该盯"能不能证明"）。所以本册第一支测就是这个。

第二件事：A 桶为 0 **不等于**待办为 0。B 桶（写靶是变量、静态读不出）必须一直在件里出版，
否则下一批会拿 A 桶的绿去结清一条本来要逐枚现读的债。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_newline_write_face.py"


def _module():
    spec = importlib.util.spec_from_file_location("newline_write_face", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instrument():
    return _module()


def test_the_target_is_the_receiver_not_the_payload(instrument) -> None:
    """假阳性的那一支：payload 里含 `reports/` 而写靶是变量 ⇒ 必须进 B 桶，不许进 A 桶。"""

    payload_mentions_reports = (
        "import json\n"
        "def run(policy, ROOT):\n"
        '    policy.write_text(json.dumps({"evidence": "reports/x.json"}), encoding="utf-8")\n'
    )
    buckets = instrument.classify_source(payload_mentions_reports)
    assert buckets["a"] == [], buckets
    assert len(buckets["b"]) == 1, buckets

    literal_reports_target = (
        "def run(ROOT, data):\n"
        '    (ROOT / "reports/x.json").write_text(data, encoding="utf-8")\n'
    )
    buckets = instrument.classify_source(literal_reports_target)
    assert len(buckets["a"]) == 1, buckets
    assert buckets["b"] == [], buckets


def test_a_site_that_passes_newline_is_not_counted(instrument) -> None:
    """同一把尺必须能数出"没带"，也必须放过"带了"——否则这是一支恒红或恒绿的尺。"""

    clean = 'def run(ROOT, data):\n    (ROOT / "reports/x.json").write_text(data, newline="\\n")\n'
    assert instrument.classify_source(clean) == {"a": [], "b": [], "c": [], "helper": []}


def test_a_write_through_the_shared_helper_lands_only_in_the_helper_bucket(instrument) -> None:
    """㊵-789 的 E 桶：走统一落盘助手（DEBT-G115 修法③）的写入**天生带行尾参数**，
    所以它不许进 A／B／C 任何一桶——否则"助手已被采用"会被读成"这里还缺 `newline=`"。
    反面对照同样必须为假：把方法名抄错（`write_text_repot`）就不算采用，
    这一支的存在理由就是它能拦住"检测器匹配一切"那种假象。
    """

    adopted = (
        "import _taiji_report_io as report_io\n"
        "def run(p, text):\n"
        "    report_io.write_text_report(p, text)\n"
        '    report_io.write_json_report(p, {"a": 1})\n'
    )
    buckets = instrument.classify_source(adopted)
    assert buckets["helper"] == [3, 4], buckets
    assert buckets["a"] == [] and buckets["b"] == [] and buckets["c"] == []

    typo = "def run(p, text):\n    report_io.write_text_repot(p, text)\n"
    wrong = instrument.classify_source(typo)
    assert wrong["helper"] == [], wrong
    assert wrong["a"] == [] and wrong["b"] == [] and wrong["c"] == []


def test_binary_and_read_modes_are_not_text_write_targets(instrument) -> None:
    """`open(p,"wb")` 不该被算成文本写靶；`open(p,"r")` 更不是写入。"""

    #: 裸 `open(p)` 的默认档是只读。第一版把"没写 mode"当成写入，于是读件的地方被虚算成待办——
    #: 那里的注释写对了、代码写反了，这一支就是钉住那一条（虚高的数会让下一批去修不存在的问题）。
    assert instrument.classify_source(
        "def run(p):\n    with open(p) as fh:\n        return fh.read()\n"
    ) == {
        "a": [],
        "b": [],
        "c": [],
        "helper": [],
    }

    assert instrument.classify_source(
        'def run(p):\n    with open(p, "wb") as fh:\n        fh.write(b"x")\n'
    ) == {
        "a": [],
        "b": [],
        "c": [],
        "helper": [],
    }
    assert instrument.classify_source(
        'def run(p):\n    with open(p, "r") as fh:\n        return fh.read()\n'
    ) == {
        "a": [],
        "b": [],
        "c": [],
        "helper": [],
    }


def test_open_with_a_literal_reports_path_and_write_mode_is_a_face(instrument) -> None:
    source = 'def run():\n    with open("reports/x.json", "w", encoding="utf-8") as fh:\n        fh.write("{}")\n'
    assert len(instrument.classify_source(source)["a"]) == 1


def test_a_variable_write_target_lands_in_b_not_a(instrument) -> None:
    """静态读不出落点就是读不出——不许靠名字里带 "report" 就把它算成 A（那是拟合）。"""

    source = 'def run(path, text):\n    with open(path, "w") as fh:\n        fh.write(text)\n'
    buckets = instrument.classify_source(source)
    assert buckets["a"] == []
    assert len(buckets["b"]) == 1, buckets


def test_the_d_bucket_narrows_the_fix_face_without_inventing_a_verdict(
    instrument, tmp_path
) -> None:
    """D 桶＝"这台仪器的 argparse 默认落点自己就写着 `reports/`"：它是**待办收窄**，不是新判决。

    两支都能为假：默认值不含 `reports/` 的文件不许进 D；含的必须进 D，且 D 的行号必须也在 B 里
    （D ⊆ B——写靶仍是静态读不出的，D 只回答"该先现读哪几枚文件"）。D 不入 `verdicts`：
    把它算成红就等于宣称"这些文件已经确认在写 reports/"，而那正是还没做的逐枚现读。
    """

    import ast

    with_default = (
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        'p.add_argument("--out-report", default="reports/x.json")\n'
    )
    without_default = (
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        'p.add_argument("--face", default="output/")\n'
    )
    assert instrument.has_reports_default(ast.parse(with_default)) is True
    assert instrument.has_reports_default(ast.parse(without_default)) is False

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    (root / "writer_with.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        'p.add_argument("--out-report", default="reports/x.json")\n'
        "def run(args, text):\n"
        "    path = args.out_report\n"
        '    with open(path, "w", encoding="utf-8") as fh:\n'
        "        fh.write(text)\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "writer_without.py").write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        'p.add_argument("--out-dir", default="output/")\n'
        "def run(args, text):\n"
        "    path = args.out_dir\n"
        '    open(path, "w").write(text)\n',
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, capture_output=True)
    report = instrument.scan(root, ".")
    assert sorted(report["d_face"]["files"]) == ["writer_with.py"], report["d_face"]["files"]
    b_names = [
        name
        for name in ("writer_with.py", "writer_without.py")
        if instrument.classify_source((root / name).read_text(encoding="utf-8"))["b"]
    ]
    assert set(report["d_face"]["files"]) <= set(b_names)
    assert report["verdict"] == "holds", report["failing_verdicts"]
    assert "d_face_at_or_below_baseline" not in report["verdicts"]


def test_the_real_face_reads_zero_on_a_and_keeps_b_published(instrument, tmp_path) -> None:
    """真面：A 桶必须为 0（门的前提），B 桶必须**不为空**（否则"待办已清"就是这句话的假证）。"""

    out = tmp_path / "face.json"
    assert instrument.main(["--out-report", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["a_face"]["sites"] == 0
    assert report["b_face"]["sites"] > 0, report["b_face"]
    assert report["unparsed_files"] == sorted(instrument.PARSE_BASELINE)
    assert report["verdict"] == "holds", report["failing_verdicts"]
    assert report["tracked_python_total"] == len(instrument.tracked_python(PROJECT_ROOT, "."))


def test_an_unreadable_file_is_a_failing_verdict_not_a_silent_skip(instrument, tmp_path) -> None:
    """面里有新的文件 parse 不动 ⇒ 必须响亮，而不是"少扫几枚看不出来"。

    已知那三枚 `scripts/archive/legacy_*` 在 `PARSE_BASELINE` 里点名（它们是这台仪器**看不到**的面），
    判据写成子集：第四枚出现就红，谁把它们修到能 parse 只是让基线缩小。
    """

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    (root / "broken.py").write_text("def broken(:\n", encoding="utf-8", newline="\n")
    (root / "ok.py").write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, capture_output=True)
    report = instrument.scan(root, ".")
    assert report["unparsed_files"] == ["broken.py"], report["unparsed_files"]
    assert "no_unparseable_file_beyond_the_known_three" in report["failing_verdicts"]
    # 同一个面若把坏件拿掉，判定必须转绿——证明红的是那枚件，不是这台尺。
    (root / "broken.py").unlink()
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, capture_output=True)
    assert instrument.scan(root, ".")["verdict"] == "holds"


def test_an_empty_face_or_a_non_repo_fails_closed(instrument, tmp_path) -> None:
    """`--face` 指到没有已跟踪 .py 的地方 ⇒ rc=2；不是"干净"。"""

    out = tmp_path / "none.json"
    assert instrument.main(["--root", str(tmp_path), "--out-report", str(out)]) == 2
    assert not out.exists()


def test_the_instrument_refuses_to_overwrite_and_writes_lf(instrument, tmp_path) -> None:
    out = tmp_path / "already.json"
    out.write_text("{}\n", encoding="utf-8", newline="\n")
    assert instrument.main(["--out-report", str(out)]) == 2
    assert out.read_text(encoding="utf-8") == "{}\n"
    fresh = tmp_path / "fresh.json"
    assert instrument.main(["--out-report", str(fresh)]) == 0
    assert fresh.read_bytes().count(b"\r") == 0


def test_the_help_face_survives_a_gbk_console_and_lists_every_flag() -> None:
    env = dict(os.environ, PYTHONIOENCODING="gbk")
    proc = subprocess.run(
        [sys.executable, str(INSTRUMENT), "--help"],
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    for flag in ("--root", "--face", "--out-report", "--overwrite"):
        assert flag in proc.stdout, flag
