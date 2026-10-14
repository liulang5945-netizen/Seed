"""`audit_taiji_n5_23_repeat_arm_noise_floor.py` 的契约册（J-N5v-2／J-N5v-3 的机检形）。

判据函数必须在合成阈值表上**两支都跑过**：地板离开与不离开各一次、地板恰为零一次、
指标缺一枚一次、配置不齐一次、没跑到预算一次。这样「仪器正确」与「仪器跑过一次」
被分开证明——真读数还要等重复臂退出，本册只证仪器。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_n5_23_repeat_arm_noise_floor.py"


def _load():
    spec = importlib.util.spec_from_file_location("n5_23_floor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FLOOR = _load()


def _run_report(
    surprise: float,
    accuracy: float,
    *,
    consumed: int = 8_900_000,
    feed_every: int = 50,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "base_checkpoint_sha256": "sha-base",
        "symbols_budget": 8_900_000,
        "skip_symbols": 11_199_800,
        "fresh": True,
        "generated_feed": {"every_symbols": feed_every},
        "last_progress": {
            "symbols_budget": 8_900_000,
            "symbols_consumed": consumed,
            "cumulative_mean_surprise": surprise,
            "cumulative_online_accuracy": accuracy,
        },
    }


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    return path


def _parity(tmp_path: Path, name: str, treated: str, control: str) -> Path:
    """Synthesise the two-armed parity report the floor instrument now requires as input."""

    return _write(
        tmp_path / name,
        {"verdict": "holds", "runner_commit_per_arm": {"treated": treated, "control": control}},
    )


def _args(tmp_path: Path, four: list[dict[str, Any]], out_name: str) -> list[str]:
    paths = [_write(tmp_path / f"arm{index}.json", payload) for index, payload in enumerate(four)]
    parity = [
        _parity(tmp_path, "parity_batchc.json", "commit-A", "commit-A"),
        _parity(tmp_path, "parity_repeat.json", "commit-A", "commit-A"),
    ]
    return [
        "--batchc-treated",
        str(paths[0]),
        "--batchc-control",
        str(paths[1]),
        "--repeat-treated",
        str(paths[2]),
        "--repeat-control",
        str(paths[3]),
        "--parity-batchc",
        str(parity[0]),
        "--parity-repeat",
        str(parity[1]),
        "--out-report",
        str(tmp_path / out_name),
    ]


def test_a_floor_the_observed_difference_leaves_is_supported(tmp_path: Path) -> None:
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(6.9, 0.255),
        _run_report(6.05, 0.261),
    ]
    rc = FLOOR.main(_args(tmp_path, four, "r.json"))
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["verdict"] == "holds"
    surprise = payload["floors"]["cumulative_mean_surprise"]
    #: 浮点末位：7.0−6.9 在二进制里不是恰 0.1，断言按**近似**取，不为了凑整数改判据。
    assert surprise["floor"] == pytest.approx(0.1, abs=1e-9)
    assert surprise["observed_difference_batchc"] == pytest.approx(1.0, abs=1e-9)
    assert surprise["verdict"] == "magnitude_supported"


def test_a_floor_the_difference_does_not_leave_is_blocked(tmp_path: Path) -> None:
    #: 这一支同时钉住「恰好等于地板」那一格：观察差 1.0、地板 1.0 ⇒ **不算离开**（严格大于）。
    #: 少了这一支，`observed > floor` 会被悄悄写成 `>=`，而两者在真读数上常常就差最后一位。
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(8.0, 0.25),
        _run_report(5.0, 0.26),
    ]
    rc = FLOOR.main(_args(tmp_path, four, "b.json"))
    payload = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert rc == 0
    surprise = payload["floors"]["cumulative_mean_surprise"]
    assert surprise["floor"] == pytest.approx(1.0, abs=1e-9)
    assert surprise["observed_difference_batchc"] == pytest.approx(1.0, abs=1e-9)
    assert surprise["verdict"] == "magnitude_blocked"
    assert payload["verdicts"]["any_metric_leaves_its_floor"] is False


def test_a_zero_floor_is_unjudged_not_small(tmp_path: Path) -> None:
    same = _run_report(7.0, 0.25)
    other = _run_report(6.0, 0.26)
    four = [same, other, json.loads(json.dumps(same)), json.loads(json.dumps(other))]
    rc = FLOOR.main(_args(tmp_path, four, "z.json"))
    payload = json.loads((tmp_path / "z.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["floors"]["cumulative_mean_surprise"]["floor"] == 0.0
    assert payload["floors"]["cumulative_mean_surprise"]["verdict"] == "no_spread_unjudged"
    assert payload["verdicts"]["any_metric_leaves_its_floor"] is False


def test_an_arm_that_stopped_short_cannot_price_the_spread(tmp_path: Path) -> None:
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(6.9, 0.255, consumed=8_000_000),
        _run_report(6.05, 0.261),
    ]
    rc = FLOOR.main(_args(tmp_path, four, "s.json"))
    payload = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["verdict"] == "rejected"
    assert payload["failing_verdicts"] == ["all_four_arms_at_budget"]


def test_a_fourth_arm_is_not_a_repeat(tmp_path: Path) -> None:
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(6.9, 0.255, feed_every=25),
        _run_report(6.05, 0.261),
    ]
    rc = FLOOR.main(_args(tmp_path, four, "c.json"))
    payload = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["failing_verdicts"] == ["same_configuration_within_role"]


def test_a_missing_metric_column_means_no_floor_not_zero(tmp_path: Path) -> None:
    broken = _run_report(7.0, 0.25)
    del broken["last_progress"]["cumulative_online_accuracy"]
    four = [broken, _run_report(6.0, 0.26), _run_report(6.9, 0.255), _run_report(6.05, 0.261)]
    rc = FLOOR.main(_args(tmp_path, four, "m.json"))
    payload = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["floors"]["cumulative_online_accuracy"]["verdict"] == "metric_absent"
    assert payload["floors"]["cumulative_online_accuracy"].get("floor") is None
    assert payload["failing_verdicts"] == ["floors_formed"]


def test_cap_dimensions_get_their_own_floor(tmp_path: Path) -> None:
    batch = {
        "verdict": "holds",
        "pairing_valid": True,
        "dimensions": [
            {"dimension": "D", "delta": -1, "enters_verdict": True},
            {"dimension": "C", "delta": 0, "enters_verdict": True},
            {"dimension": "B", "delta": 5, "enters_verdict": False},
        ],
    }
    repeat = {
        "verdict": "holds",
        "pairing_valid": True,
        "dimensions": [
            {"dimension": "D", "delta": 2, "enters_verdict": True},
            {"dimension": "C", "delta": 0, "enters_verdict": True},
        ],
    }
    paths = [_write(tmp_path / "capb.json", batch), _write(tmp_path / "capr.json", repeat)]
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(6.9, 0.255),
        _run_report(6.05, 0.261),
    ]
    rc = FLOOR.main(
        _args(tmp_path, four, "cap.json")
        + ["--cap-pairing-batchc", str(paths[0]), "--cap-pairing-repeat", str(paths[1])]
    )
    payload = json.loads((tmp_path / "cap.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["cap"]["dimensions"] == ["C", "D"]
    assert payload["cap"]["per_dimension"]["D"]["floor"] == 3
    assert payload["cap"]["per_dimension"]["C"]["floor"] == 0


def test_an_unqualified_cap_pairing_is_refused_not_priced(tmp_path: Path) -> None:
    """配对不合格的 CAP 件不许进来定价——那条要求原先**只是件里的一句文字**。

    ㊵-797 实测：一枚 `verdict=rejected`／`pairing_valid=false` 的配对件会被原样读进来，
    地板仪照样出版 `cap`、总判据照样 `holds`。两支都要走：合格件出数（上一支测），
    不合格件响亮拒绝且不落半件（这一支）。缺键按不合格处理——不假定"键不在就是旧格式没问题"。
    """
    row = {"dimension": "D", "delta": 1, "enters_verdict": True}
    good = {"verdict": "holds", "pairing_valid": True, "dimensions": [row]}
    rejected = {"verdict": "rejected", "pairing_valid": False, "dimensions": [row]}
    missing_keys = {"dimensions": [row]}
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(6.9, 0.255),
        _run_report(6.05, 0.261),
    ]
    p_good = _write(tmp_path / "capg.json", good)
    offenders = (
        (_write(tmp_path / "caprej.json", rejected), "floor_rej.json"),
        (_write(tmp_path / "capmiss.json", missing_keys), "floor_miss.json"),
    )
    for offender, out_name in offenders:
        rc = FLOOR.main(
            _args(tmp_path, four, out_name)
            + ["--cap-pairing-batchc", str(p_good), "--cap-pairing-repeat", str(offender)]
        )
        assert rc == 2, (out_name, rc)
        assert not (tmp_path / out_name).exists(), f"refusal must leave no artifact: {out_name}"


def test_missing_input_wrong_kind_and_overwrite_guard(tmp_path: Path) -> None:
    good = _write(tmp_path / "good.json", _run_report(7.0, 0.25))
    not_run = _write(tmp_path / "not_run.json", {"format": "something-else"})
    parity = [
        _parity(tmp_path, "p_batchc.json", "commit-A", "commit-A"),
        _parity(tmp_path, "p_repeat.json", "commit-A", "commit-A"),
    ]
    base = [
        "--batchc-treated",
        str(good),
        "--batchc-control",
        str(good),
        "--repeat-treated",
        str(good),
        "--repeat-control",
        str(good),
        "--parity-batchc",
        str(parity[0]),
        "--parity-repeat",
        str(parity[1]),
        "--out-report",
        str(tmp_path / "never.json"),
    ]
    assert FLOOR.main(base) == 0
    written = (tmp_path / "never.json").read_bytes()
    assert FLOOR.main(base) == 2
    assert (
        tmp_path / "never.json"
    ).read_bytes() == written, "refusal must leave the existing bytes untouched"
    missing = list(base)
    missing[1] = str(tmp_path / "gone.json")
    assert FLOOR.main(missing) == 2
    wrong_kind = list(base)
    wrong_kind[1] = str(not_run)
    assert FLOOR.main(wrong_kind) == 2


def test_published_reading_limit_carries_the_pairing_refusal(tmp_path: Path) -> None:
    """件里的自述必须与模块常量同源，并且**真的写着**"拒收不合格配对件"这件事。

    ㊵-797 给仪器加了拒绝分支，自述句子却没跟上——只读件的人会以为 CAP 维地板在配对不合格时
    也能出数。这条测把"那半句在场"钉住（删掉即红），并用截回迁移前长度的一支证明守卫能为假。
    """
    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(6.9, 0.255),
        _run_report(6.05, 0.261),
    ]
    assert FLOOR.main(_args(tmp_path, four, "limit.json")) == 0
    payload = json.loads((tmp_path / "limit.json").read_text(encoding="utf-8"))
    clause = payload["reading_limit"]
    assert clause == FLOOR.READING_LIMIT
    for token in ("pairing_valid", "holds", "rc=2"):
        assert token in clause, token
    assert not all(
        token in clause[:108] for token in ("pairing_valid", "holds", "rc=2")
    ), "the refusal clause must be the part that arrived, not already-present wording"


def test_a_runner_version_difference_is_not_a_repeat(tmp_path: Path) -> None:
    """跑器版差必须把地板仪拦下来——㊵-833 实测这一支原先**没人守**。

    两臂 `runner_commit` 同为 `fe39dfadef6e…`、批次 C 那枚是 `9299352d439b…` 时，仪器当时照样给
    `verdict=holds` 与 `same_configuration_within_role=true`，而 `J-N5v-1` 按 §2 原文已判红。
    两支都走：同版出数（上面那批测）、异版必须 rc=1，且**点名的失败只有新加的那一条**——
    连带把原有三条资格一起打翻的话，这条红就说不清是谁的错。
    """

    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
    ]
    other = _parity(tmp_path, "p_other.json", "commit-B", "commit-B")
    rc = FLOOR.main(_args(tmp_path, four, "verdiff.json") + ["--parity-repeat", str(other)])
    payload = json.loads((tmp_path / "verdiff.json").read_text(encoding="utf-8"))
    assert rc == 1, rc
    assert payload["verdict"] == "rejected"
    assert payload["failing_verdicts"] == ["runner_commit_equal_across_campaigns"]
    assert payload["runner_commits"]["batchc"]["treated"] == "commit-A"
    assert payload["runner_commits"]["repeat"]["treated"] == "commit-B"
    assert payload["verdicts"]["same_configuration_within_role"] is True


def test_a_parity_report_that_is_not_holds_is_refused(tmp_path: Path) -> None:
    """并排件自己不合格 ⇒ rc=2 且不落半件：资格面不许建立在一次失败的并排取证上。"""

    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
    ]
    bad = _write(
        tmp_path / "p_bad.json",
        {"verdict": "rejected", "runner_commit_per_arm": {"treated": "x", "control": "x"}},
    )
    rc = FLOOR.main(_args(tmp_path, four, "parity_refused.json") + ["--parity-batchc", str(bad)])
    assert rc == 2, rc
    assert not (tmp_path / "parity_refused.json").exists(), "refusal must leave no artifact"
    # 缺键同样按不合格处理：不许把"键不在"当成"旧格式没问题"。
    shapeless = _write(tmp_path / "p_shape.json", {"verdict": "holds"})
    assert (
        FLOOR.main(
            _args(tmp_path, four, "shape_refused.json") + ["--parity-batchc", str(shapeless)]
        )
        == 2
    )
    assert not (tmp_path / "shape_refused.json").exists()


def test_parity_inputs_are_required_on_the_command_face(tmp_path: Path) -> None:
    """两枚并排入参必须是**必需**的：可选就等于允许没人传，那条资格会静默失效。"""

    four = [
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
        _run_report(7.0, 0.25),
        _run_report(6.0, 0.26),
    ]
    without = _args(tmp_path, four, "no_parity.json")
    cut = without.index("--parity-batchc")
    del without[cut : cut + 4]
    with pytest.raises(SystemExit) as stop:
        FLOOR.main(without)
    assert stop.value.code == 2
    assert not (tmp_path / "no_parity.json").exists()
