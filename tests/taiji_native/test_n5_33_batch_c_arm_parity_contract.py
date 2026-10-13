"""`DEBT-G127` 修法① 的契约册：批次 C 两臂必须被**并排**证明只差旗标，否则结论当场收回。

判据冻在 `PLAN-N5-20` §2（J-N5r-1／J-N5r-2／J-N5r-4），本册不重冻也不放松，只钉仪器的形状：
①配置面不等 ⇒ 红，并把不等的那枚键点名（冻件原文：「差异不可归因 ⇒ 两臂同 0 或同非 0 都不算结论」）；
②关旗标那一臂若留下任何写入痕迹 ⇒ 红（J-N5r-1 的"为假的情形"）；
③**结果面绝不判等**——把两臂的墙钟与摘要故意改成不同，仪器仍须为绿；这一支是防"造一支永远为假的尺"；
④缺件、缺键都走响亮出口，不许被读成"差为空"（DEBT-G58 同族）。

夹具取的是**已入库那份**治疗臂自述（`reports/taiji_n5_20_batchC_treated_A_20261013.json`，㊵-755 落盘），
所以本册在任何一台机器上都跑得动，不依赖 `output/` 里的在飞产物；对照臂真件的正式读数排在它跑完之后。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_batch_c_arm_parity.py"
SEALED_TREATED = PROJECT_ROOT / "reports" / "taiji_n5_20_batchC_treated_A_20261013.json"

#: 对照臂夹具的会话号：它落在"跑器最后一次改动之前"，与真对照臂同版（J-N5r-4 的代码面那一支）。
CONTROL_SESSION = "A-2026-10-13T02:44:40Z"


def _module():
    spec = importlib.util.spec_from_file_location("batch_c_parity", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _treated_report() -> dict[str, Any]:
    return json.loads(SEALED_TREATED.read_text(encoding="utf-8"))


def _campaign_of(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "trainer": report["trainer"],
        "symbols_budget": report["symbols_budget"],
        "skip_symbols": report["skip_symbols"],
        "base_checkpoint_sha256": report["base_checkpoint_sha256"],
        "status": report["status"],
        "wall_seconds": report["session"]["wall_seconds"],
        "arms_status": {"A": report["status"]},
    }


def _control_report() -> dict[str, Any]:
    """从真治疗臂自述造对照臂：只把旗标关掉，并把它派生的三枚计数一起归零。"""

    payload = copy.deepcopy(_treated_report())
    payload["session"]["session_id"] = CONTROL_SESSION
    payload["generated_feed"] = {
        **payload["generated_feed"],
        "every_symbols": 0,
        "feeds": 0,
        "generated_write_count": 0,
        "generated_row_weight_max_abs": 0.0,
    }
    return payload


def _write_pair(
    tmp_path: Path, treated: dict[str, Any], control: dict[str, Any]
) -> tuple[str, str]:
    for name, payload in (("treated", treated), ("control", control)):
        arm = tmp_path / name / "A"
        arm.mkdir(parents=True)
        (arm / "run_report.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
        (tmp_path / name / "campaign_report.json").write_text(
            json.dumps(_campaign_of(payload), ensure_ascii=False), encoding="utf-8", newline="\n"
        )
    return str(tmp_path / "treated" / "A"), str(tmp_path / "control" / "A")


@pytest.fixture(scope="module")
def instrument():
    return _module()


def test_the_sealed_fixture_carries_every_field_the_face_reads(instrument) -> None:
    """夹具不许是空的，也不许悄悄漏面：三张面点名的每一条路径都必须**两臂同源地**在场。

    唯一被允许的例外是 `DISCLOSURE_PATHS` 那两枚——它们 ㊵-756 才进跑器，而批次 C 两臂都载自更早那版，
    所以真件里两臂都没有。这条断言把这件事钉成"要么两边都在、要么两边都不在"，
    **单边缺席**（一臂有一臂没有）一律算归因问题。
    """

    treated = _treated_report()
    treated["campaign"] = _campaign_of(treated)
    face = (*instrument.CONFIG_FACE, *instrument.FLAG_FACE, *instrument.OUTCOME_FACE)
    absent = [dotted for dotted in face if instrument._dig(treated, dotted) == instrument._MISSING]
    assert absent == [], absent
    disclosure_absent = [
        dotted
        for dotted in instrument.DISCLOSURE_PATHS
        if instrument._dig(treated, dotted) == instrument._MISSING
    ]
    assert disclosure_absent == list(instrument.DISCLOSURE_PATHS), disclosure_absent


def test_a_pair_differing_only_in_the_flag_is_green(instrument, tmp_path) -> None:
    """J-N5r-4 的正形状：配置面逐字段相等、旗标面确实不同、两支判据各自为真。"""

    treated, control = _treated_report(), _control_report()
    out = tmp_path / "pair.json"
    treated_dir, control_dir = _write_pair(tmp_path, treated, control)
    rc = instrument.main(
        ["--treated-dir", treated_dir, "--control-dir", control_dir, "--out-report", str(out)]
    )
    assert rc == 0, json.loads(out.read_text(encoding="utf-8"))["failing_verdicts"]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdict"] == "holds", report["failing_verdicts"]
    assert report["differing_config_keys"] == []
    assert "generated_feed.every_symbols" in report["differing_flag_keys"]
    assert report["runner_commit_per_arm"]["treated"] == report["runner_commit_per_arm"]["control"]
    assert report["verdicts"]["j_n5r_2_treated_row_weight_positive"] is True
    assert report["verdicts"]["j_n5r_1_control_no_side_effects"] is True


def test_a_second_config_difference_is_red_and_names_the_key(instrument, tmp_path) -> None:
    """冻件的反半句：两臂还差 corpus／base sha／fresh 中任一项 ⇒ 差异不可归因，当场为红。"""

    control = _control_report()
    control["corpus"]["fingerprint"] = '[{"name":"other.jsonl","bytes":1}]'
    out = tmp_path / "drift.json"
    treated_dir, control_dir = _write_pair(tmp_path, _treated_report(), control)
    rc = instrument.main(
        ["--treated-dir", treated_dir, "--control-dir", control_dir, "--out-report", str(out)]
    )
    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["differing_config_keys"] == ["corpus.fingerprint"], report[
        "differing_config_keys"
    ]
    assert "config_face_identical" in report["failing_verdicts"]


def test_the_control_arm_leaving_a_write_trace_is_j_n5r_1_false(instrument, tmp_path) -> None:
    """关旗标那一臂若仍有写入痕迹 ⇒ J-N5r-1 为假，而不是"把非零解释成无害"。"""

    control = _control_report()
    control["generated_feed"]["generated_write_count"] = 5
    out = tmp_path / "leak.json"
    treated_dir, control_dir = _write_pair(tmp_path, _treated_report(), control)
    rc = instrument.main(
        ["--treated-dir", treated_dir, "--control-dir", control_dir, "--out-report", str(out)]
    )
    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "j_n5r_1_control_no_side_effects" in report["failing_verdicts"]


def test_the_outcome_face_is_published_but_never_compared(instrument, tmp_path) -> None:
    """③防"永远为假的尺"：结果面故意不同，仪器仍须为绿，但那几枚不等必须**看得见**。

    两支各自能红：`rc == 0` 证结果面没被拿来判等；`equal is False` 那几枚证这台尺确实扫到了差异——
    只验前一支的话，把结果面整面删掉也能绿（同一族教训见 DEBT-G81③ 的"零命中要连范围一起报"）。
    """

    control = _control_report()
    control["session"]["wall_seconds"] = 1.234
    control["session"]["seconds_per_symbol"] = 0.987654
    control["digests_after"]["predictive_readout"] = "deadbeef"
    control["generated_feed"]["measured_symbols_per_second"] = 999.0
    out = tmp_path / "outcome.json"
    treated_dir, control_dir = _write_pair(tmp_path, _treated_report(), control)
    rc = instrument.main(
        ["--treated-dir", treated_dir, "--control-dir", control_dir, "--out-report", str(out)]
    )
    assert rc == 0, json.loads(out.read_text(encoding="utf-8"))["failing_verdicts"]
    report = json.loads(out.read_text(encoding="utf-8"))
    rows = {row["path"]: row for row in report["faces"]["outcome"]}
    for path in (
        "session.wall_seconds",
        "digests_after.predictive_readout",
        "generated_feed.measured_symbols_per_second",
    ):
        assert rows[path]["equal"] is False, path
    assert report["verdicts"]["config_face_identical"] is True


def test_a_missing_control_report_fails_loudly_and_names_the_path(instrument, tmp_path, capsys):
    """缺件＝rc=2 且把缺的那条路径点名；不许把"文件不在"读成"两臂无差异"。"""

    treated_dir, _ = _write_pair(tmp_path, _treated_report(), _treated_report())
    empty = tmp_path / "never-ran" / "A"
    empty.mkdir(parents=True)
    rc = instrument.main(
        [
            "--treated-dir",
            treated_dir,
            "--control-dir",
            str(empty),
            "--out-report",
            str(tmp_path / "none.json"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "run_report.json" in err and "never-ran" in err
    assert not (tmp_path / "none.json").exists()


def test_a_missing_key_is_reported_as_missing_not_as_a_difference(instrument, tmp_path) -> None:
    """`observe_kwargs` 整枚缺席 ⇒ 出版 missing_paths 并为红，而不是"值不等"或"相等"。"""

    control = _control_report()
    control.pop("observe_kwargs")
    out = tmp_path / "absent.json"
    treated_dir, control_dir = _write_pair(tmp_path, _treated_report(), control)
    rc = instrument.main(
        ["--treated-dir", treated_dir, "--control-dir", control_dir, "--out-report", str(out)]
    )
    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "observe_kwargs" in report["paths_absent_from_one_arm"]
    assert "no_paths_absent_from_a_single_arm" in report["failing_verdicts"]


def test_a_disclosure_key_present_on_only_one_arm_is_red(instrument, tmp_path) -> None:
    """披露通道不是免检通道：两臂同缺＝跑器版本面差，**一臂有一臂没有**就照样归因不了。"""

    control = _control_report()
    control["generated_feed"]["generated_reward_sum_basis"] = "literal_by_construction"
    out = tmp_path / "onesided.json"
    treated_dir, control_dir = _write_pair(tmp_path, _treated_report(), control)
    rc = instrument.main(
        ["--treated-dir", treated_dir, "--control-dir", control_dir, "--out-report", str(out)]
    )
    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["disclosure"]["one_sided_absence"] == [
        "generated_feed.generated_reward_sum_basis"
    ]
    assert "no_disclosure_path_is_one_sided" in report["failing_verdicts"]


def test_the_batch_c_artifacts_do_not_self_describe_the_zero_reward_basis(
    instrument, tmp_path
) -> None:
    """真面读数：批次 C 两臂件里那两枚依据键同缺 ⇒ `holds` 仍可为真，但缺口必须点名在册。

    这一支钉的是**引用纪律**：`generated_reward_sum=0.0` 印在两臂件里，可"这是构造性的字面量、
    不是实测"那句话不在件里（跑器 ㊵-756 才补上这句自述，而两臂都载自更早那版）。
    引用批次 C 的零奖励时必须带上这句，见台账 ㊵-759。
    """

    out = tmp_path / "disclosure.json"
    treated_dir, control_dir = _write_pair(tmp_path, _treated_report(), _control_report())
    assert (
        instrument.main(
            ["--treated-dir", treated_dir, "--control-dir", control_dir, "--out-report", str(out)]
        )
        == 0
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["disclosure"]["absent_on_both_arms"] == list(instrument.DISCLOSURE_PATHS)
    assert report["disclosure"]["present_on_both_arms"] == []
    assert report["verdict"] == "holds"


def test_a_bad_shape_in_the_flag_block_fails_closed(instrument, tmp_path) -> None:
    """`feeds` 写成非数 ⇒ 那一支判假，仪器不炸栈（读数面坏掉要能在件里看见）。"""

    treated = _treated_report()
    treated["generated_feed"]["feeds"] = "not-a-number"
    out = tmp_path / "shape.json"
    treated_dir, control_dir = _write_pair(tmp_path, treated, _control_report())
    rc = instrument.main(
        ["--treated-dir", treated_dir, "--control-dir", control_dir, "--out-report", str(out)]
    )
    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert "treated_feeds_match_budget_arithmetic" in report["failing_verdicts"]


def test_the_instrument_refuses_to_overwrite_an_existing_report(instrument, tmp_path) -> None:
    treated_dir, control_dir = _write_pair(tmp_path, _treated_report(), _control_report())
    out = tmp_path / "already.json"
    out.write_text("{}\n", encoding="utf-8", newline="\n")
    rc = instrument.main(
        ["--treated-dir", treated_dir, "--control-dir", control_dir, "--out-report", str(out)]
    )
    assert rc == 2
    assert out.read_text(encoding="utf-8") == "{}\n"


def test_the_report_is_written_with_lf_endings(instrument, tmp_path) -> None:
    treated_dir, control_dir = _write_pair(tmp_path, _treated_report(), _control_report())
    out = tmp_path / "lf.json"
    assert (
        instrument.main(
            ["--treated-dir", treated_dir, "--control-dir", control_dir, "--out-report", str(out)]
        )
        == 0
    )
    assert out.read_bytes().count(b"\r") == 0


def test_the_help_face_survives_a_gbk_console_and_lists_every_flag() -> None:
    """DEBT-G81／G85 口径：真子进程＋`PYTHONIOENCODING=gbk`，capsys 不算；每个旗标名都要在帮助里。"""

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
    for flag in ("--treated-dir", "--control-dir", "--out-report", "--overwrite"):
        assert flag in proc.stdout, flag
