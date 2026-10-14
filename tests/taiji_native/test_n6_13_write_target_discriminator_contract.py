"""DEBT-G137／PLAN-G138 的契约册：判别式必须按**工具自己的数据流**分类，四种形状都要能解析到。

在册仪器＝`scripts/training/audit_taiji_test_write_target_discriminator.py`（G-G138-2：仓外探针只算前置测量，
读数只许引这一枚）。本册里每一支都能红：合成源码进、角色出，且正反两支都走（同种子必须被接受那一族的写法）。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "scripts/training/audit_taiji_test_write_target_discriminator.py"
SPEC = importlib.util.spec_from_file_location("discriminator", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _tree(source: str) -> ast.Module:
    return ast.parse(source)


def _role(source: str, dest: str) -> str:
    return module.dest_role(_tree(source), dest)[0]


def test_plain_path_binding_into_write_text_is_write() -> None:
    source = (
        "def main(args):\n"
        "    out = Path(args.out)\n"
        "    out.write_text('x', encoding='utf-8')\n"
    )
    assert _role(source, "out") == "write"


def test_ternary_and_path_wrapped_shapes_are_both_parsed() -> None:
    """The shape that defeats name-only heuristics: the binding sits inside an IfExp."""

    source = (
        "def main(args, alt):\n"
        "    out = Path(args.out) if alt is None else Path(alt)\n"
        "    out.write_bytes(b'x')\n"
    )
    assert _role(source, "out") == "write"
    # And the same tool with the flag only ever read must not be classified as write.
    reading = (
        "def main(args, alt):\n"
        "    src = Path(args.out) if alt is None else Path(alt)\n"
        "    return json.loads(src.read_text(encoding='utf-8'))\n"
    )
    assert _role(reading, "out") == "read"


def test_for_tuple_transport_is_followed() -> None:
    source = (
        "def main(args):\n"
        "    for name, value in [(('out'), args.out)]:\n"
        "        Path(value).write_text('x')\n"
    )
    assert _role(source, "out") == "write"


def test_cross_function_rename_is_followed_one_hop() -> None:
    source = (
        "def save(target):\n"
        "    Path(target).write_text('x', encoding='utf-8')\n"
        "\n"
        "def main(args):\n"
        "    save(args.out)\n"
    )
    assert _role(source, "out") == "write"


def test_a_flag_that_reaches_nothing_is_unresolved_not_write() -> None:
    source = "def main(args):\n    return args.out\n"
    assert _role(source, "out") == "unresolved"


def test_assertion_messages_are_not_flags_but_a_loose_matcher_would_take_them() -> None:
    """G-G138-1: the strict regex keeps them out; the guard must be able to fail."""

    source = (
        "raise AssertionError('--only 打空必须拒绝')\n"
        "raise AssertionError('--help 没有按惯例退出 0')\n"
        "run(['--out-report'])\n"
    )
    tree = _tree(source)
    loose = module.loose_only_literals(tree)
    assert {"--only 打空必须拒绝", "--help 没有按惯例退出 0"} <= loose
    assert all(not module.FLAG_STRICT.match(text) for text in loose)
    # The one real flag survives the strict filter, so the filter is not simply rejecting everything.
    assert module.FLAG_STRICT.match("--out-report")


def test_call_site_pairs_pair_tool_with_flag_and_honour_one_hop_constants() -> None:
    source = (
        'TOOL = "scripts/training/foo.py"\n'
        "def test():\n"
        "    run([sys.executable, TOOL, '--out-report', p])\n"
        "    run(['scripts/training/bar.py', '--report', q])\n"
    )
    pairs = set(module.call_site_pairs(_tree(source)))
    assert ("scripts/training/foo.py", "--out-report") in pairs
    assert ("scripts/training/bar.py", "--report") in pairs


@pytest.fixture(scope="module")
def report() -> dict:
    return module.build_report(REPO)


def test_the_instrument_publishes_the_three_numbers_and_the_qualifier(report: dict) -> None:
    """J-G138-4: never a single total, and the reading face has to be named in the text."""

    assert {"pairs_seen_in_call_sites", "domain_pairs", "test_side_role_tally"} <= set(report)
    tally = report["test_side_role_tally"]
    assert report["domain_pairs"] == sum(tally.values())
    assert tally["write"] == len(report["test_side_write_pairs"])
    assert "在 tracked 测的调用面里" in report["reading_limit"]


def test_write_bucket_excludes_the_named_heuristic_false_positive(report: dict) -> None:
    """J-G138-1: the ㊵-807 counterexample must not sit in the write bucket, and the buckets must be live."""

    joined = report["test_side_write_pairs"]
    assert not any(
        p.startswith("tests/taiji_native/test_p3b_review_worksheet_contract.py :: ")
        and p.endswith(" --output")
        for p in joined
    )
    assert report["test_side_not_write_buckets_total"] > 0
    assert all(
        item["role"] in {"unresolved", "both"}
        for item in report["test_side_not_write_buckets_named"]
    )


def test_the_new_buckets_never_piggyback_on_the_sealed_report_face(report: dict) -> None:
    """J-G138-3: those four numbers belong to another instrument; this one must not publish them."""

    legacy = {
        "default_targets_on_tracked_files",
        "distinct_files",
        "exposure_targets",
        "exposure_files",
    }
    assert legacy.isdisjoint(set(report))
    assert "test_side_write_pairs" in report
    assert not any("exposure" in key or "default_targets" in key for key in report)


def test_observation_can_overrule_the_ast_classification(tmp_path: Path) -> None:
    """J-G138-2 的仲裁必须真的接在仪器上：观测说它写，分类为 read 的那对就要降级并点名。"""

    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    tool = tmp_path / "scripts/tool.py"
    tool.write_text(
        "import argparse\n"
        "def main():\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument('--report')\n"
        "    args = p.parse_args()\n"
        "    src = Path(args.report)\n"
        "    return json.loads(src.read_text())\n",
        encoding="utf-8",
    )
    test = tmp_path / "tests/test_call.py"
    test.write_text(
        'def t():\n    run(["scripts/tool.py", "--report", "x.json"])\n', encoding="utf-8"
    )
    # The scan face comes from version control, so the fixture has to be a repo too.
    for step in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "f"],
    ):
        subprocess.run(["git", *step], cwd=tmp_path, capture_output=True, text=True, check=True)
    label = "tests/test_call.py :: scripts/tool.py --report"
    ast_only = module.build_report(tmp_path)
    assert ast_only["test_side_role_tally"]["read"] == 1
    assert ast_only["observation_downgraded_total"] == 0
    with_observation = module.build_report(tmp_path, {label})
    assert with_observation["observation_downgraded_total"] == 1
    assert with_observation["observation_downgraded_pairs"][0]["pair"] == label
    assert with_observation["test_side_role_tally"]["read"] == 0
    assert with_observation["test_side_role_tally"]["unresolved"] == 1


def test_legacy_heuristic_candidate_key_is_publishable(report: dict) -> None:
    """㊵-807 那批启发式候选要能被复算出名单（哪怕口径是旧的），否则经验复核无处下手。"""

    assert "legacy_heuristic_candidates" in report
    assert report["legacy_heuristic_candidate_total"] == len(report["legacy_heuristic_candidates"])
    for item in report["legacy_heuristic_candidates"]:
        assert {"test", "report_path", "flag"} <= set(item)


def test_the_instrument_writes_through_the_shared_helper(tmp_path: Path) -> None:
    """G115 修法③ dogfood plus the refusal branch: an existing target keeps its bytes."""

    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "report_io.write_json_report(" in source
    target = tmp_path / "existing.json"
    target.write_text("keep me\n", encoding="utf-8", newline="\n")
    rc = module.main(["--out-report", str(target)])
    assert rc == 2
    assert target.read_text(encoding="utf-8") == "keep me\n"
    rc = module.main(["--out-report", str(target), "--overwrite"])
    assert rc == 0
    published = json.loads(target.read_text(encoding="utf-8"))
    assert published["format"] == module.REPORT_FORMAT
