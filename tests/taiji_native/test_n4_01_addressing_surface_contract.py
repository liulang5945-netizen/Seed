"""PLAN-N4-01 的寻址面仪器（`measure_taiji_n4_addressing_surface.py`）契约测。

这台仪器的全部主张是"母量只有 `wrong_top1_rate` 一条，且四态不许塌成一个空元组"，所以
**每一条拒绝支与两条判级支都必须实走**（DEBT-G58 就是被"四态塌成 `()`"这条骗出来的）：

* 零干扰夹具 ⇒ 率 0.0 且 `ruler_usable=false`（尺没有动态范围，J-N4-1 不达）；
* 故意造干扰的夹具 ⇒ 率 > 0.0 且 `ruler_usable=true`；
* 零配对／缺件／非正容量／非正 limit ⇒ rc=2；
* 非空库却拿到零命中 ⇒ 响亮拒绝（产品 `retrieve` 语义被改动时本件不许静默）；
* 挂载态必须自述成 `harness`、第四态必须自述成"不可观测"，不许被读侧当产品读数或当"没过滤"。

真实产品件只读；夹具与读数件一律落 pytest 的 `tmp_path`。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "measure_taiji_n4_addressing_surface.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n4_addr", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _read(report: Path) -> dict[str, Any]:
    return json.loads(report.read_text(encoding="utf-8"))


def _clean_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """三条两两正交的线索，每条查询就等于自己那条 ⇒ top-1 必为真值，率恒 0。"""

    materials = tmp_path / "materials.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_rows(
        materials,
        [
            {"memory_id": f"m{index}", "cue": [1.0 if j == index else 0.0 for j in range(3)]}
            for index in range(3)
        ],
    )
    _write_rows(
        queries,
        [
            {
                "expected_memory_id": f"m{index}",
                "cue": [1.0 if j == index else 0.0 for j in range(3)],
            }
            for index in range(3)
        ],
    )
    return materials, queries


def test_zero_interference_fixture_has_no_dynamic_range(tmp_path: Path) -> None:
    module = _load()
    materials, queries = _clean_fixture(tmp_path)
    report = tmp_path / "report.json"
    rc = module.main(
        [
            "--materials",
            str(materials),
            "--queries",
            str(queries),
            "--capacity",
            "8",
            "--limit",
            "3",
            "--out-report",
            str(report),
        ]
    )
    payload = _read(report)
    assert rc == 0
    assert payload["wrong_top1_rate"] == 0.0
    #: 这正是 J-N4-1 要否证的形状：全场取对 ⇒ 这把尺量不出"干扰"，不许据此说 S4 有用。
    assert payload["ruler_usable"] is False
    assert payload["true_rank_p1_rate"] == 1.0
    assert payload["paired_queries"] == 3
    assert payload["total_queries"] == 3


def test_interference_fixture_is_detected(tmp_path: Path) -> None:
    module = _load()
    materials = tmp_path / "materials.jsonl"
    queries = tmp_path / "queries.jsonl"
    #: 真值 `target` 与干扰 `distractor` 只差在第二维的符号；查询落在干扰那一侧 ⇒ top-1 取错。
    _write_rows(
        materials,
        [
            {"memory_id": "target", "cue": [1.0, 0.0]},
            {"memory_id": "distractor", "cue": [1.0, 0.1]},
        ],
    )
    _write_rows(queries, [{"expected_memory_id": "target", "cue": [1.0, 0.05]}] * 6)
    report = tmp_path / "report.json"
    rc = module.main(
        [
            "--materials",
            str(materials),
            "--queries",
            str(queries),
            "--capacity",
            "8",
            "--limit",
            "1",
            "--out-report",
            str(report),
        ]
    )
    payload = _read(report)
    assert rc == 0
    assert payload["wrong_top1_rate"] == 1.0
    #: 恒 1 也不算有动态范围（J-N4-1 要求严格落在开区间里）。
    assert payload["ruler_usable"] is False
    assert payload["true_rank_p1_rate"] == 0.0
    #: `limit=1` 且真值不在第 1 位＝真值**越界**，这条必须单列，不许折进中位数或当成零命中。
    assert payload["true_rank_outside_top_limit"] == 6
    assert payload["true_rank_median"] is None
    assert payload["noise_band_adjacent_block_max"] == 0.0


def test_mixed_fixture_gives_a_usable_ruler_and_self_consistent_counts(tmp_path: Path) -> None:
    module = _load()
    materials = tmp_path / "materials.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_rows(
        materials,
        [
            {"memory_id": "target", "cue": [1.0, 0.0]},
            {"memory_id": "distractor", "cue": [1.0, 0.1]},
        ],
    )
    #: 一半查询偏向真值、一半偏向干扰 ⇒ 率落在开区间 (0,1)，四态等式也要自洽。
    rows = []
    for _ in range(4):
        rows.append({"expected_memory_id": "target", "cue": [1.0, 0.0]})
    for _ in range(4):
        rows.append({"expected_memory_id": "target", "cue": [1.0, 0.05]})
    _write_rows(queries, rows)
    report = tmp_path / "report.json"
    rc = module.main(
        [
            "--materials",
            str(materials),
            "--queries",
            str(queries),
            "--capacity",
            "8",
            "--limit",
            "2",
            "--out-report",
            str(report),
        ]
    )
    payload = _read(report)
    assert rc == 0
    assert payload["wrong_top1_rate"] == 0.5
    assert payload["ruler_usable"] is True
    counts = payload["state_counts"]
    assert (
        counts["store_absent"] + counts["store_empty"] + counts["emitted"]
        == payload["total_queries"]
    )
    assert payload["capacity"] == 8
    assert payload["cue_dim"] == 2
    #: 8 条配对查询 ⇒ 段宽 1、带下界 1.0 ⇒ J-N4-2 在这个 n 上算术上不可能成立（DEBT-G59）。
    assert payload["block_size"] == 1
    assert payload["noise_band_floor"] == 1.0


def test_noise_band_floor_shrinks_with_query_count(tmp_path: Path) -> None:
    """带下界必须是面内自述的事实，而不是事后推算：20 条配对查询 ⇒ 段宽 4、下界 0.25。"""

    module = _load()
    materials = tmp_path / "materials.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_rows(
        materials,
        [
            {"memory_id": "target", "cue": [1.0, 0.0]},
            {"memory_id": "distractor", "cue": [1.0, 0.1]},
        ],
    )
    rows = [{"expected_memory_id": "target", "cue": [1.0, 0.0]}] * 10
    rows += [{"expected_memory_id": "target", "cue": [1.0, 0.05]}] * 10
    _write_rows(queries, rows)
    report = tmp_path / "report.json"
    rc = module.main(
        [
            "--materials",
            str(materials),
            "--queries",
            str(queries),
            "--capacity",
            "8",
            "--limit",
            "2",
            "--out-report",
            str(report),
        ]
    )
    payload = _read(report)
    assert rc == 0
    assert payload["paired_queries"] == 20
    assert payload["wrong_top1_rate"] == 0.5
    assert payload["block_size"] == 4
    assert payload["noise_band_floor"] == 0.25


@pytest.mark.parametrize(
    "extra_args,expected_reject",
    [
        (["--capacity", "0"], "capacity_must_be_positive"),
        (["--limit", "0"], "limit_must_be_positive"),
    ],
)
def test_nonpositive_knobs_are_loud_rejects(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list,
    expected_reject: str,
) -> None:
    module = _load()
    materials, queries = _clean_fixture(tmp_path)
    report = tmp_path / "report.json"
    args = ["--materials", str(materials), "--queries", str(queries)]
    args += extra_args
    args += ["--out-report", str(report)]
    assert module.main(args) == 2
    #: 拒绝必须点名是哪一枚旋钮，否则"rc=2"可以是任何别的原因。
    assert expected_reject in capsys.readouterr().out
    assert not report.exists()


def test_missing_materials_file_is_rc_2(tmp_path: Path) -> None:
    module = _load()
    _, queries = _clean_fixture(tmp_path)
    assert (
        module.main(
            [
                "--materials",
                str(tmp_path / "nope.jsonl"),
                "--queries",
                str(queries),
                "--out-report",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )


def test_zero_paired_queries_is_rc_2_and_publishes_no_rate(tmp_path: Path) -> None:
    module = _load()
    materials = tmp_path / "materials.jsonl"
    queries = tmp_path / "queries.jsonl"
    _write_rows(materials, [{"memory_id": "target", "cue": [1.0, 0.0]}])
    #: 真值不在名册里 ⇒ 配对数为 0，母量分母为 0 ⇒ 不许出版率。
    _write_rows(queries, [{"expected_memory_id": "absent", "cue": [1.0, 0.0]}] * 3)
    report = tmp_path / "report.json"
    rc = module.main(
        [
            "--materials",
            str(materials),
            "--queries",
            str(queries),
            "--out-report",
            str(report),
        ]
    )
    assert rc == 2
    payload = _read(report)
    assert payload["paired_queries"] == 0
    assert payload["verdict"] == "no_paired_queries"
    assert payload["wrong_top1_rate"] is None


def test_nonempty_store_returning_zero_hits_is_a_loud_reject(tmp_path: Path, monkeypatch) -> None:
    """产品 `retrieve` 若被改成"非空库也能给 ()"，本件必须炸而不是把它数成一次寻址失败。"""

    module = _load()
    materials, queries = _clean_fixture(tmp_path)
    monkeypatch.setattr(module.EpisodicMemoryStore, "retrieve", lambda self, cue, *, limit=1: ())
    assert (
        module.main(
            [
                "--materials",
                str(materials),
                "--queries",
                str(queries),
                "--out-report",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )


def test_mount_layer_and_gated_state_are_self_reported(tmp_path: Path) -> None:
    module = _load()
    materials, queries = _clean_fixture(tmp_path)
    report = tmp_path / "report.json"
    assert (
        module.main(
            [
                "--materials",
                str(materials),
                "--queries",
                str(queries),
                "--out-report",
                str(report),
            ]
        )
        == 0
    )
    payload = _read(report)
    assert payload["mount_layer"] == "harness"
    assert payload["product_default_mounted"] is False
    #: 第四态必须自述成"不可观测"，读侧拿不到它就只能显式承认自己不知道。
    assert payload["adapter_gated_state"] == module.NOT_OBSERVABLE
    assert "gated_to_empty" not in payload["state_counts"]


def test_wall_clock_column_is_published_and_face_keys_do_not_shrink(tmp_path: Path) -> None:
    """㊵-563：把 `wall_clock_ms` 钉住（㊵-561④ 自报的那条"跑过一次不等于被钉住"）。

    两面都要断言：①这一列存在、是非负浮点；②**面读数的键集只许多不许少**——
    计时是加性自述，谁把它当成"可以顺手删掉的装饰列"，这册当场红。
    夹具自带（不依赖别处的 helper），材料侧给带唯一尾值的数值线索 ⇒ 非空库必有取回。
    """
    import importlib.util

    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "training"
        / "measure_taiji_n4_addressing_surface.py"
    )
    spec = importlib.util.spec_from_file_location("n4_face_wallclock_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    materials = tmp_path / "materials.jsonl"
    queries = tmp_path / "queries.jsonl"
    rows = []
    for index in range(40):
        rows.append(
            {
                "memory_id": f"m{index:03d}",
                "cue": [float((index * 7 + k) % 97 + 1) for k in range(12)] + [float(index + 1)],
            }
        )
    questions = []
    for step in range(12):
        pick = rows[(step * 3 + 1) % len(rows)]
        questions.append({"cue": pick["cue"][:-1], "expected_memory_id": pick["memory_id"]})
    materials.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    queries.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions),
        encoding="utf-8",
        newline="\n",
    )
    report = tmp_path / "report.json"
    rc = module.main(
        [
            "--materials",
            str(materials),
            "--queries",
            str(queries),
            "--capacity",
            "64",
            "--limit",
            "3",
            "--out-report",
            str(report),
        ]
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert rc == 0, payload.get("rejection")
    assert "rejection" not in payload, payload
    #: ①计时列存在且是非负浮点。
    assert "wall_clock_ms" in payload, sorted(payload)
    assert isinstance(payload["wall_clock_ms"], float), payload["wall_clock_ms"]
    assert payload["wall_clock_ms"] >= 0.0
    #: ②键集只许多不许少（面读数的既有自述一条不许漂走）。
    required = {
        "format",
        "capacity",
        "limit",
        "paired_queries",
        "ruler_usable",
        "wrong_top1_rate",
        "block_size",
        "noise_band_floor",
        "adapter_gated_state",
    }
    missing = sorted(required - set(payload))
    assert not missing, missing
    #: 落字节必须 LF：这台仪器的件要入库（`write_text` 不给 newline 在本机写出过 CRLF 件）。
    assert b"\r\n" not in report.read_bytes(), report.read_bytes()[:40]
