"""`PLAN-N5-21` §6.1 第 3 条的载体（`audit_taiji_batch_c_moved_face_sets.py`）的契约测。

这台仪器做的事只有一件：拿两趟已在册的比较器读数做**动名集合差**。所以测的重点不是集合运算
（那是 `set` 的活），而是它**不能说的话**：
参照系不同源必须红、任一趟是响亮拒绝必须红、差集为空必须走 `no_localization_signal` 而不是
"喂入没动任何东西"，以及镜像两根不许把一枚面数成两枚（[[aggregate-column-two-questions-before-use]] 的
"分子分母同一集合"＋[[guard-must-be-able-to-fail]]）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_batch_c_moved_face_sets.py"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_moved_faces", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CMP = _load()


def _report(
    path: Path,
    *,
    names: list[str],
    reference_sha: str = SHA_A,
    status: str = "measured",
    report_format: str = "taiji-n5-arm-tensor-comparison-v3",
) -> Path:
    payload = {
        "format": report_format,
        "status": status,
        "files": [{"path": "base.pt", "file_sha256": reference_sha, "bytes": 1}],
        "pairs": [{"label": "base.pt -> arm.pt", "differing_slots_names": sorted(names)}],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path


def _run(tmp_path: Path, treated: Path, control: Path, name: str = "r.json") -> tuple[int, dict]:
    out = tmp_path / name
    rc = CMP.main(
        [
            "--treated-report",
            str(treated),
            "--control-report",
            str(control),
            "--out-report",
            str(out),
        ]
    )
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_a_face_moved_only_in_the_treated_arm_is_named(tmp_path: Path) -> None:
    treated = _report(
        tmp_path / "t.json",
        names=[
            "substrate.memory.association.edge_weight",
            "taiji.kernel.memory.association.edge_weight",
            "substrate.predictive_readout.weights",
        ],
    )
    control = _report(tmp_path / "c.json", names=["substrate.predictive_readout.weights"])
    rc, payload = _run(tmp_path, treated, control)
    assert rc == 0
    assert payload["format"] == "taiji-batch-c-moved-face-sets-v2"
    assert payload["verdict"] == "holds"
    #: 镜像两根并成一枚：这里只有 `memory.association.edge_weight` 算治疗臂独有。
    assert payload["sets"]["only_in_treated"] == 1
    assert payload["only_in_treated_names"] == ["memory.association.edge_weight"]
    assert payload["only_in_treated_names_raw"] == [
        "substrate.memory.association.edge_weight",
        "taiji.kernel.memory.association.edge_weight",
    ]
    assert payload["faces"]["only_in_treated"] == {"memory": 2}
    assert payload["differing_slots"] == {"treated": 3, "control": 1}


def test_a_different_reference_face_must_go_red(tmp_path: Path) -> None:
    #: 两趟的基座侧不是同一份字节时，两个"差异名集"相减得出的东西没有意义——它必须红，
    #: 而不是安静地出一份看着像定位的件。治疗臂这里**有意留一枚独有名字**，
    #: 让 failing 只剩参照系那一条（不然两支同时红，就证不了是 sha 那条在起作用）。
    treated = _report(
        tmp_path / "t.json",
        names=["substrate.memory.association.edge_weight", "substrate.readout.w"],
    )
    control = _report(
        tmp_path / "c.json",
        names=["substrate.readout.w"],
        reference_sha=SHA_B,
    )
    rc, payload = _run(tmp_path, treated, control, "sha.json")
    assert rc == 1
    assert payload["failing_verdicts"] == ["reference_is_the_same_bytes"]
    assert payload["sets"]["only_in_treated"] == 1


def test_two_instrument_versions_are_not_one_face(tmp_path: Path) -> None:
    """两趟输入必须出自**同一版比较器**，否则集合差是在比两台面。

    v3（DEBT-G133 修法①）起"名集不齐的未声明缺席"按交集出版差异名集，而 v2 那种情形整趟拒绝、
    名集根本不出——把 v3 的交集面与 v2 的拒绝面相减，得到的"只在治疗臂"是一句关于仪器版本的话，
    不是关于权重的话。这一支**能为假**（同版本时它为真），所以它不是仪式。
    """

    treated = _report(
        tmp_path / "t.json",
        names=["substrate.memory.last_event", "substrate.memory.association.edge_weight"],
    )
    control = _report(
        tmp_path / "c.json",
        names=["substrate.memory.last_event"],
        report_format="taiji-n5-arm-tensor-comparison-v2",
    )
    rc, payload = _run(tmp_path, treated, control, "ver.json")
    assert rc == 1
    assert payload["failing_verdicts"] == ["same_instrument_version"]
    assert payload["inputs"]["treated_report_format"].endswith("-v3")
    assert payload["inputs"]["control_report_format"].endswith("-v2")


def test_a_refused_input_report_is_not_a_reading(tmp_path: Path) -> None:
    treated = _report(tmp_path / "t.json", names=["substrate.readout.w"])
    control = _report(tmp_path / "c.json", names=[], status="refuse_name_sets_differ")
    rc, payload = _run(tmp_path, treated, control, "refused.json")
    assert rc == 1
    assert payload["failing_verdicts"] == ["both_reports_measured"]


def test_an_empty_difference_is_no_signal_not_an_absence_claim(tmp_path: Path) -> None:
    #: 差集为空＝这台仪器**没读到**定位信号，不等于"喂入没动任何东西"。
    #: 这一支保证 verdict 能为假：同样的输入只要有一枚独有名字就转绿（见第一支）。
    names = ["substrate.memory.association.edge_weight", "substrate.readout.w"]
    treated = _report(tmp_path / "t.json", names=names)
    control = _report(tmp_path / "c.json", names=names)
    rc, payload = _run(tmp_path, treated, control, "same.json")
    assert rc == 1
    assert payload["verdict"] == "rejected"
    assert payload["failing_verdicts"] == ["localization_signal_present"]
    assert payload["sets"]["only_in_treated"] == 0
    assert payload["sets"]["in_both"] == 2
    assert "不是「喂入无效」" in payload["reading_limit"]


def test_missing_file_and_non_comparator_json_both_refuse(tmp_path: Path) -> None:
    treated = _report(tmp_path / "t.json", names=["substrate.readout.w"])
    rc = CMP.main(
        [
            "--treated-report",
            str(treated),
            "--control-report",
            str(tmp_path / "never-ran.json"),
            "--out-report",
            str(tmp_path / "x.json"),
        ]
    )
    assert rc == 2
    junk = tmp_path / "junk.json"
    junk.write_text("{}", encoding="utf-8", newline="\n")
    rc_junk = CMP.main(
        [
            "--treated-report",
            str(treated),
            "--control-report",
            str(junk),
            "--out-report",
            str(tmp_path / "y.json"),
        ]
    )
    assert rc_junk == 2


def test_help_walks_and_refuses_to_overwrite(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        CMP.main(["--help"])
    assert exit_info.value.code == 0
    treated = _report(tmp_path / "t.json", names=["substrate.readout.w"])
    control = _report(tmp_path / "c.json", names=[])
    out = tmp_path / "sealed.json"
    out.write_text("{}\n", encoding="utf-8", newline="\n")
    rc = CMP.main(
        [
            "--treated-report",
            str(treated),
            "--control-report",
            str(control),
            "--out-report",
            str(out),
        ]
    )
    assert rc == 2
    assert out.read_text(encoding="utf-8") == "{}\n"
