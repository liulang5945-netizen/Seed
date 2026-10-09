"""N3 甲判读器（`adjudicate_taiji_n3a_scaling_probe.py`）的契约测。

三支都必须能为假，且都是本轮实测逼出来的：

* **DEBT-G63 的筛除**：进度面最后一行 `window_ticks=0` 却出版 `online_accuracy=0.0` ⇒
  判读器必须**丢掉那一行**，并且末段均值不能因为它就变成 0；
* **§8.7 五项缺任一项 ⇒ `ran_not_measured` 且 rc=2**（缺的与齐的两面都造一次，不许只验一支）；
* **两臂 `mean_revisits` 相等 ⇒ 仪器坏**的响亮标志必须为真（PLAN-N3-10 §2 的判对条件）；
* 缺面文件 ⇒ rc=2；斜率不超过自取带 ⇒ `slope_gt_own_band` 为假。

夹具全部落 `tmp_path`，不碰 `output/` 与 `reports/`。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "adjudicate_taiji_n3a_scaling_probe.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n3a_judge", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _progress(path: Path, accs: list[float], *, drop_last_window: bool = True) -> Path:
    rows = []
    for index, acc in enumerate(accs, start=1):
        rows.append(
            {
                "ticks": index * 10000,
                "window_ticks": 9999,
                "online_accuracy": acc,
                "mean_surprise": 3.0 - 0.01 * index,
                "holdout_surprise": 2.9,
                "holdout_surprise_v2": 3.0 - 0.02 * index,
                "elapsed_seconds": index * 10,
                "epoch": 0,
            }
        )
    if drop_last_window:
        #: 真实形态：最后一行是收尾行，窗口已归零、acc 出版成 0.0（DEBT-G63）。
        rows[-1]["window_ticks"] = 0
        rows[-1]["online_accuracy"] = 0.0
        rows[-1]["mean_surprise"] = 0.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return path


def _exit(path: Path, **over: Any) -> Path:
    record = {
        "exit_reason": "max_symbols_reached",
        "reached_budget": True,
        "budget_max_symbols": 250000,
        "ticks_at_exit": 250000,
        "unique_documents": 315,
        "document_visits": 315,
        "mean_revisits": 1.0,
        "elapsed_seconds": 4014.0,
        "checkpoint_sha256": "a" * 64,
        "corpus_fingerprint": '[{"name":"dialogue_extended_clean.jsonl","bytes":108327171}]',
    }
    record.update(over)
    #: 形状必须与产品一致：`exit_record_path().write_text(json.dumps(..., indent=2))` 是**多行对象**，
    #: 单行夹具会让"按行读"这种缺陷在测里隐身。
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def _args(tmp_path: Path, **over: Any) -> list[str]:
    arm_a = over.get("arm_a") or _progress(
        tmp_path / "a.jsonl", [0.20 + 0.01 * k for k in range(25)]
    )
    arm_b = over.get("arm_b") or _progress(
        tmp_path / "b.jsonl", [0.20 + 0.002 * k for k in range(25)]
    )
    exit_a = over.get("exit_a") or _exit(tmp_path / "a_exit.json")
    exit_b = over.get("exit_b") or _exit(
        tmp_path / "b_exit.json", unique_documents=32, document_visits=315, mean_revisits=9.84
    )
    return [
        "--arm-a",
        str(arm_a),
        "--arm-b",
        str(arm_b),
        "--exit-a",
        str(exit_a),
        "--exit-b",
        str(exit_b),
        "--out",
        str(tmp_path / "out.json"),
    ]


def _payload(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))


def test_window_zero_closing_line_is_dropped_and_does_not_poison_the_band(tmp_path: Path) -> None:
    rc = MODULE.main(_args(tmp_path))
    arm = _payload(tmp_path)["arms"]["arm_A"]
    assert rc == 2  # §8.7 的 sequence_length 缺失，见下一条
    assert arm["progress_lines"] == 25
    assert arm["lines_dropped_window_zero"] == 1
    #: 若不筛除，末段会被那个 0.0 拉低甚至归零。
    assert arm["acc_last_segment_mean"] is not None
    assert float(arm["acc_last_segment_mean"]) > 0.4


def test_missing_eighth_seven_field_makes_it_ran_not_measured(tmp_path: Path) -> None:
    rc = MODULE.main(_args(tmp_path))
    payload = _payload(tmp_path)
    assert rc == 2
    judgement = payload["judgement"]["arm_A"]
    assert judgement["measurement_complete"] is False
    assert judgement["J_N3a"] == "ran_not_measured"
    assert "sequence_length" in payload["arms"]["arm_A"]["section_8_7_completeness"]


def test_complete_five_fields_allow_a_real_verdict(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        exit_a=_exit(tmp_path / "a_exit_full.json", sequence_length=1),
        exit_b=_exit(
            tmp_path / "b_exit_full.json",
            unique_documents=32,
            document_visits=315,
            mean_revisits=9.84,
            sequence_length=1,
        ),
    )
    rc = MODULE.main(args)
    payload = _payload(tmp_path)
    assert rc == 0
    judgement = payload["judgement"]["arm_A"]
    assert judgement["measurement_complete"] is True
    assert judgement["slope_gt_own_band"] is True
    assert judgement["J_N3a"] in {"holds", "not_holds", "degradation_blocks"}


def test_flat_arm_fails_the_own_noise_band(tmp_path: Path) -> None:
    flat = _progress(tmp_path / "flat.jsonl", [0.20] * 25)
    rc = MODULE.main(_args(tmp_path, arm_a=flat))
    payload = _payload(tmp_path)
    assert rc == 2
    judgement = payload["judgement"]["arm_A"]
    assert judgement["slope_gt_own_band"] is False


def test_equal_mean_revisits_is_flagged_as_a_broken_instrument(tmp_path: Path) -> None:
    exit_b = _exit(tmp_path / "b_exit_same.json", unique_documents=315, mean_revisits=1.0)
    rc = MODULE.main(_args(tmp_path, exit_b=exit_b))
    payload = _payload(tmp_path)
    assert rc == 2
    check = payload["single_variable_check"]
    assert check["instrument_broken_if_revisits_equal"] is True
    assert check["unique_documents_differ"] is False
    assert check["ticks_equal"] is True


def test_missing_face_file_is_a_loud_reject(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args[args.index("--arm-b") + 1] = str(tmp_path / "nope.jsonl")
    assert MODULE.main(args) == 2


def test_different_corpus_fingerprint_between_arms_voids_the_contrast(tmp_path: Path) -> None:
    """G-N3f-3／G-N3g-1：两臂吃的不是同一份材料 ⇒ 对照作废、两侧都不判，不许由人读命令比对。"""

    other = _exit(
        tmp_path / "b_exit_othercorpus.json",
        unique_documents=32,
        document_visits=315,
        mean_revisits=9.84,
        corpus_fingerprint='[{"name":"other_corpus.jsonl","bytes":123}]',
    )
    rc = MODULE.main(_args(tmp_path, exit_b=other))
    payload = _payload(tmp_path)
    assert rc == 2
    check = payload["single_variable_check"]
    assert check["corpus_fingerprints_equal"] is False
    assert payload["verdict"] == "arms_not_same_source"
    assert payload["judgement"]["arm_A"]["J_N3a"] == "not_judged"
    assert payload["judgement"]["arm_B"]["J_N3a"] == "not_judged"


def test_interpretation_limit_is_published_verbatim(tmp_path: Path) -> None:
    MODULE.main(_args(tmp_path))
    payload = _payload(tmp_path)
    assert payload["interpretation_limit"] == MODULE.INTERPRETATION_LIMIT
    assert "等符号暴露" in payload["interpretation_limit"]
    assert "不许" in payload["interpretation_limit"] or "不允许" in payload["interpretation_limit"]


#: §8.7 第四项的取值面（PLAN-N3-13）：收尾件自述优先，其次只认守卫齐了的同源复算件。
def _sidecar(path: Path, **over: Any) -> Path:
    payload: dict[str, Any] = {
        "format": "taiji-n3a-sequence-length-recompute-v1",
        "status": "ok",
        "checks": {
            "corpus_fingerprint_matches_exit": True,
            "max_symbols_matches_exit_budget": True,
            "replayed_visits_match_exit": True,
        },
        "sequence_length": {
            "documents_counted": 315,
            "min": 84,
            "median": 626.0,
            "max": 2858,
            "mean": 795.888889,
        },
    }
    payload.update(over)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def test_ok_sidecar_unlocks_the_fourth_item_and_is_disclosed(tmp_path: Path) -> None:
    args = _args(tmp_path) + [
        "--seq-a",
        str(_sidecar(tmp_path / "seq_a.json")),
        "--seq-b",
        str(_sidecar(tmp_path / "seq_b.json", sequence_length={"documents_counted": 324})),
    ]
    rc = MODULE.main(args)
    payload = _payload(tmp_path)
    assert rc == 0, payload["arms"]["arm_A"]["section_8_7_completeness"]
    for name in ("arm_A", "arm_B"):
        arm = payload["arms"][name]
        assert arm["sequence_length_source"] == "recomputed_same_stream", arm
        assert arm["measurement_complete"] is True, arm["section_8_7_completeness"]
        assert payload["judgement"][name]["J_N3a"] != "ran_not_measured"


def test_self_reported_column_takes_precedence_over_a_sidecar(tmp_path: Path) -> None:
    seq = {"documents_counted": 315, "min": 84, "median": 626.0, "max": 2858, "mean": 795.9}
    exit_a = _exit(tmp_path / "sa_exit.json", sequence_length=seq)
    args = _args(tmp_path, exit_a=exit_a) + [
        "--seq-a",
        str(_sidecar(tmp_path / "sa_seq.json", status="guard_failed")),
        "--seq-b",
        str(_sidecar(tmp_path / "sa_seq_b.json")),
    ]
    rc = MODULE.main(args)
    arm = _payload(tmp_path)["arms"]["arm_A"]
    #: 收尾件自述在场时**不去读**复算件——守卫不齐的件不能把好件顶掉。
    assert arm["sequence_length_source"] == "exit_record", arm
    assert arm["sequence_length"] == seq
    assert arm["section_8_7_completeness"]["sequence_length"] is True
    assert rc == 0, _payload(tmp_path)["arms"]["arm_B"]["section_8_7_completeness"]


def test_guard_failed_sidecar_does_not_count_as_present(tmp_path: Path) -> None:
    args = _args(tmp_path) + [
        "--seq-a",
        str(_sidecar(tmp_path / "bad_seq.json", status="guard_failed")),
        "--seq-b",
        str(_sidecar(tmp_path / "bad_seq_b.json", status="stream_shorter_than_budget")),
    ]
    rc = MODULE.main(args)
    payload = _payload(tmp_path)
    assert rc == 2
    for name in ("arm_A", "arm_B"):
        arm = payload["arms"][name]
        assert arm["sequence_length"] is None, arm
        assert arm["sequence_length_source"].startswith("sidecar_guard_failed"), arm
        assert arm["section_8_7_completeness"]["sequence_length"] is False
        assert payload["judgement"][name]["J_N3a"] == "ran_not_measured"


def test_missing_sidecar_path_is_a_loud_reject(tmp_path: Path) -> None:
    args = _args(tmp_path) + ["--seq-a", str(tmp_path / "nope.json"), "--seq-b", ""]
    assert MODULE.main(args) == 2
    assert not (tmp_path / "out.json").is_file(), "给了路径却读不到时不许照常出件"


def _prime(arm_last: float | None, control_last: float | None, *, complete: bool = True) -> dict:
    judged = {"slope_gt_own_band": True, "holdout_v2_within_limit": True}
    arm_a = {"acc_last_segment_mean": arm_last, "measurement_complete": complete}
    control = (
        None
        if control_last is None
        else {"acc_last_segment_mean": control_last, "measurement_complete": complete}
    )
    return MODULE.adjudicate_prime(judged, arm_a, control)


def test_prime_without_control_arm_is_not_judged_and_names_the_fallback_ban() -> None:
    out = _prime(0.352, None)
    assert out["J_N3a_prime"] == "not_judged"
    assert out["forbid_fallback_to_old_anchor"] is True
    #: 不许把旧锚 0.594120 当默认值塞回来代答（DEBT-G64 那条跨链值）。
    assert "0.594120" not in json.dumps(out, ensure_ascii=False)


def test_prime_holds_only_when_the_margin_is_cleared() -> None:
    out = _prime(0.40, 0.30)
    assert out["gt_control_plus_margin"] is True
    assert out["required_strictly_greater_than"] == 0.32
    assert out["J_N3a_prime"] == "holds"
    #: 判据写的是**严格大于** ⇒ 恰好等于门槛不算越线（先把这条钉住，防止以后有人改成 ≥）。
    tie = _prime(0.32, 0.30)
    assert tie["gt_control_plus_margin"] is False
    assert tie["J_N3a_prime"] == "not_holds"


def test_prime_not_holds_on_the_real_pairing_0352_vs_033966() -> None:
    #: ㊵-548 的真读数：甲臂末段 0.352 对 对照臂末段 0.33966 + 0.02 = 0.35966 ⇒ 差 +0.01234 未过线。
    out = _prime(0.352, 0.33966)
    assert out["required_strictly_greater_than"] == 0.35966
    assert out["gt_control_plus_margin"] is False
    assert out["J_N3a_prime"] == "not_holds"


def test_prime_incomplete_side_locks_ran_not_measured(tmp_path: Path) -> None:
    assert _prime(0.352, 0.34, complete=False)["J_N3a_prime"] == "ran_not_measured"
    #: 端到端：对照臂的 §8.7 缺列（不给复算件）⇒ 顶层 rc=2，甲臂再齐也不能判。
    ctrl_progress = _progress(tmp_path / "c.jsonl", [0.20 + 0.005 * k for k in range(25)])
    ctrl_exit = _exit(tmp_path / "c_exit.json", unique_documents=315, document_visits=315)
    args = _args(tmp_path) + [
        "--seq-a",
        str(_sidecar(tmp_path / "p_seq.json")),
        "--seq-b",
        str(_sidecar(tmp_path / "p_seq_b.json")),
        "--control-progress",
        str(ctrl_progress),
        "--control-exit",
        str(ctrl_exit),
    ]
    rc = MODULE.main(args)
    prime = _payload(tmp_path)["J_N3a_prime"]
    assert rc == 2, prime
    assert prime["J_N3a_prime"] == "ran_not_measured"
    assert prime["both_sides_measurement_complete"] is False


def test_prime_voided_when_control_arm_is_not_same_source(tmp_path: Path) -> None:
    ctrl_progress = _progress(tmp_path / "c2.jsonl", [0.20 + 0.005 * k for k in range(25)])
    ctrl_exit = _exit(
        tmp_path / "c2_exit.json",
        sequence_length={"documents_counted": 10},
        corpus_fingerprint='[{"name":"other_corpus.jsonl","bytes":123}]',
    )
    args = _args(tmp_path) + [
        "--seq-a",
        str(_sidecar(tmp_path / "q_seq.json")),
        "--seq-b",
        str(_sidecar(tmp_path / "q_seq_b.json")),
        "--control-progress",
        str(ctrl_progress),
        "--control-exit",
        str(ctrl_exit),
    ]
    rc = MODULE.main(args)
    prime = _payload(tmp_path)["J_N3a_prime"]
    assert prime["control_corpus_fingerprint_matches_arm_a"] is False
    assert prime["J_N3a_prime"] == "not_judged", prime
    assert "G-N3g-1" in prime["reason"]
    assert rc == 0, "不同源只作废新锚那条合取，不改顶层对两臂的判级"


def test_control_flag_without_exit_is_a_loud_reject(tmp_path: Path) -> None:
    args = _args(tmp_path) + [
        "--control-progress",
        str(_progress(tmp_path / "c3.jsonl", [0.3] * 6)),
    ]
    assert MODULE.main(args) == 2
