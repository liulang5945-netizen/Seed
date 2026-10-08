"""N2 第二次通电判读器（`count_taiji_n2b_faces.py`）的契约测。

这台仪器的全部价值在于**每条判定都能为假**，所以正反两支都必须实走：

* **正支**＝真实在库的面件读出 `does_not_hold`，并把七列前后差逐格钉死（数字取自 2026-10-08 的跑）；
* **拒绝支**＝缺件／面无效（有项未被模型作答）／两档不同源（题集或行数）⇒ rc=2 响亮拒绝；
* **能为真支**＝把"巩固后"换成回退面（＝与巩固前同一份权重）⇒ `j_n2a_prime` 必须翻成 True，
  这条防的是"保持侧写死成假"那种恒真/恒假式判据（DEBT-G46 的一类）；
* **自相矛盾支**＝通电件口头声明 `holds` 而两个 Δ 的符号说 `not_holds` ⇒ 仪器必须拒收（rc=2），
  结论不许由生产者自述、必须由分子分母推。

真实面件只读不改；被改的副本一律落在 pytest 的 `tmp_path` 里。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/count_taiji_n2b_faces.py"

FACES = {
    "--cap0-before": "reports/taiji_n2_cap0_before_20261008.json",
    "--cap0-after": "reports/taiji_n2b_cap0_after_20261008.json",
    "--cap0-rollback": "reports/taiji_n2b_cap0_rollback_20261008.json",
    "--replay-before": "reports/taiji_n2_replay24_before_20261008.json",
    "--replay-after": "reports/taiji_n2b_replay24_after_20261008.json",
    "--main-column": "reports/taiji_n2b_stop24_main_20261008.json",
    "--powerup": "reports/taiji_n2b_powerup_20261008.json",
    "--guard-default": "reports/taiji_n2b_guard_default_20261008.json",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("n2b_faces_reader", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patched(tmp_path: Path, flag: str, name: str, mutate) -> str:
    payload = copy.deepcopy(json.loads((REPO / FACES[flag]).read_text(encoding="utf-8")))
    mutate(payload)
    out = tmp_path / name
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    return str(out)


def _run(
    module, tmp_path: Path, overrides: dict[str, str] | None = None
) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "verdict.json"
    args = dict(FACES)
    args.update(overrides or {})
    argv = ["--out", str(out)]
    for flag, value in args.items():
        argv += [flag, value]
    rc = module.main(argv)
    return int(rc), json.loads(out.read_text(encoding="utf-8"))


def test_baseline_verdict_pins_every_column(tmp_path: Path) -> None:
    """真实面件的读数逐格钉死：J-持久化绿、J-N2a'/J-N2b' 两支都红，七列前后差一字不改。"""

    module = _load_module()
    rc, payload = _run(module, tmp_path)
    assert rc == 0, payload
    assert payload["verdict"] == "does_not_hold"
    assert payload["failed_criteria"] == ["J-N2a'", "J-N2b'"]
    criteria = payload["criteria"]
    assert criteria["j_persistence"] is True
    assert criteria["j_n2a_prime"] is False
    assert criteria["j_n2b_prime"] == "not_holds"
    assert criteria["j_n2b_prime_delta_quality"] == -1.1086209986387523
    assert criteria["j_n2b_prime_delta_accuracy"] == 0.0078125
    assert criteria["j_n2b_reader_agrees_with_driver"] is True
    columns = payload["j_n2a_prime_columns"]
    assert columns["dropped_by_group"] == {
        "cap0_per_dimension_machine_scored_correct": ["E"],
        "replay24_strict_hits": ["1.0", "2.0"],
        "replay24_well_formed_texts": ["0.0", "0.5", "1.0", "2.0"],
    }
    assert columns["unverified_by_group"] == {}
    main = columns["stop24_main_column"]
    assert main["direction"] == "smaller"
    assert main["all_held"] is True
    assert main["columns"]["never_lf_eaters_counted"]["delta"] == -9
    assert payload["ruler"]["cap0_total_correct_before_after_rollback"] == [9, 10, 9]
    assert payload["ruler"]["cap0_ruler_usable"] is True
    assert payload["fail_closed_pass"] is True


def test_rollback_face_reads_identical_to_before(tmp_path: Path) -> None:
    """G-N2-1 的面级那半：回退档的 CAP-0 读数与巩固前**逐位同**（剥易变＋剥出处后）。"""

    module = _load_module()
    _, payload = _run(module, tmp_path)
    assert payload["guards"]["g_n2_1_rollback_face_identical_to_before"] is True
    assert payload["guards"]["g_n2_1_rollback_digest_matches_mother"] is True
    before = payload["provenance_before"]["identity"]
    rollback = payload["provenance_rollback"]["identity"]
    #: 出处差必须**确实存在**才证明比的是两张不同的面（同 eval_set、不同档文件）。
    assert before["eval_set_sha256"] == rollback["eval_set_sha256"]
    assert before["checkpoint_sha256"] != rollback["checkpoint_sha256"]


def test_missing_face_is_loud_rejection(tmp_path: Path) -> None:
    module = _load_module()
    rc, payload = _run(module, tmp_path, {"--cap0-rollback": str(tmp_path / "nope.json")})
    assert rc == 2
    assert payload["verdict"] == "not_judged_missing_faces"
    assert any("nope.json" in item for item in payload["missing"])


def test_unanswered_item_voids_the_face(tmp_path: Path) -> None:
    """一项 `load_ok=false`＝这项没被模型答过，整张面作废；不许读成"能力归零"。"""

    module = _load_module()
    bad = _patched(
        tmp_path,
        "--cap0-after",
        "cap0_after_loadfail.json",
        lambda o: o["dimensions"]["C"]["items"][0].__setitem__("load_ok", False),
    )
    rc, payload = _run(module, tmp_path, {"--cap0-after": bad})
    assert rc == 2
    assert payload["verdict"] == "not_judged_face_invalid"
    assert payload["cap0_load_failures_by_face"]["cap0_after"] == 1


def test_different_item_sets_are_not_comparable(tmp_path: Path) -> None:
    """主列两档的 `items_sha256` 不同 ⇒ 前后差不可解释，整件拒判（跨题集版"换尺"）。"""

    module = _load_module()

    def flip(node: Any) -> bool:
        done = {"v": False}

        def walk(value: Any) -> None:
            if done["v"]:
                return
            if isinstance(value, dict):
                if "items_sha256" in value:
                    value["items_sha256"] = "deadbeefdeadbeef"
                    done["v"] = True
                    return
                for inner in value.values():
                    walk(inner)
            elif isinstance(value, list):
                for inner in value:
                    walk(inner)

        walk(node)
        return done["v"]

    payload = copy.deepcopy(json.loads((REPO / FACES["--main-column"]).read_text(encoding="utf-8")))
    assert flip(payload) is True
    bad = tmp_path / "main_sha_drift.json"
    bad.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    rc, verdict = _run(module, tmp_path, {"--main-column": str(bad)})
    assert rc == 2
    assert verdict["verdict"] == "not_judged_faces_not_comparable"
    assert any("items_sha256" in reason for reason in verdict["reasons"])


def test_j_n2a_prime_can_hold(tmp_path: Path) -> None:
    """保持侧必须**能为真**：把"巩固后"换成回退面、复述后换成复述前 ⇒ 一列都不跌。"""

    module = _load_module()
    same_after = _patched(tmp_path, "--cap0-rollback", "cap0_after_same.json", lambda o: None)
    same_replay = _patched(tmp_path, "--replay-before", "replay_after_same.json", lambda o: None)
    rc, payload = _run(
        module, tmp_path, {"--cap0-after": same_after, "--replay-after": same_replay}
    )
    assert rc == 0, payload
    assert payload["criteria"]["j_n2a_prime"] is True
    assert payload["failed_criteria"] == ["J-N2b'"]
    #: 同一份权重当"巩固后"时这把尺量不动 ⇒ 记 `ruler_usable=false`，"保持住了"不许当增益证据。
    assert payload["ruler"]["cap0_ruler_usable"] is False


def test_dose_window_must_equal_experience_budget(tmp_path: Path) -> None:
    """J-N2b' 的分子只在窗长＝本 pass 经验预算时才是这一轮的剂量；不等 ⇒ fail-closed 红。"""

    module = _load_module()
    bad = _patched(
        tmp_path,
        "--powerup",
        "powerup_window_drift.json",
        lambda o: o.__setitem__("window_bytes", 999),
    )
    rc, payload = _run(module, tmp_path, {"--powerup": bad})
    assert rc == 2
    assert payload["criteria"]["dose_window_equals_experience_budget"] is False
    assert payload["fail_closed_pass"] is False


def test_default_guard_red_when_organs_claim_to_have_run(tmp_path: Path) -> None:
    module = _load_module()
    bad = _patched(
        tmp_path,
        "--guard-default",
        "guard_ran_true.json",
        lambda o: o["organs"].__setitem__("ran", True),
    )
    rc, payload = _run(module, tmp_path, {"--guard-default": bad})
    assert rc == 2
    assert payload["guards"]["g_n2_2_default_position_untouched"] is False


def test_reader_refuses_driver_claim_that_deltas_deny(tmp_path: Path) -> None:
    """通电件若自述 `holds` 而 Δquality≤0，仪器的两个符号必须把它顶回去（rc=2）。"""

    module = _load_module()
    drift = _patched(
        tmp_path,
        "--powerup",
        "powerup_claim_drift.json",
        lambda o: o.__setitem__("j_n2b_prime", "holds"),
    )
    rc, payload = _run(module, tmp_path, {"--powerup": drift})
    assert rc == 2
    assert payload["criteria"]["j_n2b_recomputed_from_deltas"] == "not_holds"
    assert payload["criteria"]["j_n2b_reader_agrees_with_driver"] is False
