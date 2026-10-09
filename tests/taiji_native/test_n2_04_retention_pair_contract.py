"""N2-04 A 档仪器（`adjudicate_taiji_n2_04_retention_pair.py`）的契约测。

这台仪器的全部主张是"保持侧只数现成七列，且两条机检都能为假"，所以**判级两支与各条拒绝支都必须实走**：

* 七列一列不跌 ⇒ `retention_holds`（rc=0），**收益侧仍标 `not_judged_here`**（不许被读成整条合取成立）；
* 任一列跌 ≥1 ⇒ `cost_persists` 并点名是哪一列（负结果照常出版）；
* 保持材料与巩固材料**有字节交集** ⇒ rc=2 `retention_not_disjoint`（这条"保持"是泄露的题做出来的）；
* 分离取不到数／缺件／题集披露冲突／列取不到 ⇒ 各自 rc=2，**不许**把"没算"读成"没重叠"或"零跌幅"。

夹具全部落在 pytest 的 `tmp_path`，不碰 `reports/` 与 `output/`。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "adjudicate_taiji_n2_04_retention_pair.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n2_04_pair", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()

RETENTION_TEXT = "保持集的一条长文本，用来做字节级交集机检，长度必须超过 24 字节窗口。"


def _cap0_report(correct_e: int = 7, sha: str = "cap0sha") -> dict[str, Any]:
    return {
        "eval_set_sha256": sha,
        "dimensions": {
            name: {
                "item_count": 36,
                "tally": {"machine_scored_correct": correct_e, "machine_scored_items": 36},
                "items": [{"load_ok": True}],
            }
            for name in ("B", "C", "D", "E", "G")
        },
    }


def _replay_report(
    strict_1: int = 5,
    strict_2: int = 3,
    wf: tuple[int, int, int, int] = (9, 8, 7, 6),
    sha: str = "replaysha",
) -> dict[str, Any]:
    penalties = ("0.0", "0.5", "1.0", "2.0")
    return {
        "items_sha256": sha,
        "item_offset": 0,
        "limit": 24,
        "arms": [
            {
                "repetition_penalty": float(penalty),
                "items": 24,
                "texts": 24,
                "strict_hits": strict_1 if penalty == "1.0" else strict_2,
                "well_formed_texts": wf[index],
            }
            for index, penalty in enumerate(penalties)
        ],
    }


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path


def _manifest(tmp_path: Path, texts: list[str]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"items": [{"text": text} for text in texts]}, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _corpus(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def _args(tmp_path: Path, **over: Any) -> list[str]:
    cap0_before = over.get("cap0_before") or _write(tmp_path, "cap0_before.json", _cap0_report())
    cap0_after = over.get("cap0_after") or _write(tmp_path, "cap0_after.json", _cap0_report())
    replay_before = over.get("replay_before") or _write(
        tmp_path, "replay_before.json", _replay_report()
    )
    replay_after = over.get("replay_after") or _write(
        tmp_path, "replay_after.json", _replay_report()
    )
    manifest = over.get("manifest") or _manifest(tmp_path, [RETENTION_TEXT])
    corpus = over.get("corpus") or _corpus(
        tmp_path, "consolidation.jsonl", "完全不相关的巩固材料文本。\n"
    )
    return [
        "--cap0-before",
        str(cap0_before),
        "--cap0-after",
        str(cap0_after),
        "--replay-before",
        str(replay_before),
        "--replay-after",
        str(replay_after),
        "--consolidation-corpus",
        str(corpus),
        "--retention-manifest",
        str(manifest),
        "--out",
        str(tmp_path / "out.json"),
    ]


def _read_out(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))


def test_seven_columns_are_the_frozen_set_and_no_new_column_appears() -> None:
    from scripts.training import count_taiji_n2_03_attribution as attrib

    assert MODULE.SEVEN_COLUMNS == attrib.DROP_COLUMNS
    assert len(MODULE.SEVEN_COLUMNS) == 7


def test_no_drop_publishes_retention_holds_and_never_the_gain_side(tmp_path: Path) -> None:
    rc = MODULE.main(_args(tmp_path))
    payload = _read_out(tmp_path)
    assert rc == 0
    assert payload["verdict"] == "retention_holds"
    crit = payload["criteria"]
    assert crit["columns_compared"] == 7
    assert crit["columns_dropped"] == []
    #: 收益侧不许被这台仪器代答——它沿用 N2-02 的同窗判读器。
    assert crit["J_N2c_gain"] == "not_judged_here"
    assert crit["J_N2c_retention"] == "retention_holds"
    assert payload["guards"]["G_N2c_4_disjointness"]["status"] == "measured"


def test_single_column_drop_is_cost_persists_and_names_the_column(tmp_path: Path) -> None:
    after = _cap0_report(correct_e=6)
    rc = MODULE.main(_args(tmp_path, cap0_after=_write(tmp_path, "cap0_after2.json", after)))
    payload = _read_out(tmp_path)
    assert rc == 0
    assert payload["verdict"] == "cost_persists"
    assert payload["criteria"]["columns_dropped"] == ["cap0:E"]
    assert payload["per_column"]["cap0:E"]["delta"] == -1


def test_byte_intersection_with_consolidation_material_is_rejected(tmp_path: Path) -> None:
    leaked = _corpus(tmp_path, "leaked.jsonl", RETENTION_TEXT)
    rc = MODULE.main(_args(tmp_path, corpus=leaked))
    payload = _read_out(tmp_path)
    assert rc == 2
    assert payload["status"] == "retention_not_disjoint"
    assert payload["verdict"] == "not_judged"
    #: 交集数出来才算这条机检有判别力。
    assert payload["guards"]["G_N2c_4_disjointness"]["result"]["windows_found_in_corpus"] > 0


def test_missing_manifest_text_is_unverified_not_assumed_disjoint(tmp_path: Path) -> None:
    empty = _manifest(tmp_path, [])
    rc = MODULE.main(_args(tmp_path, manifest=empty))
    payload = _read_out(tmp_path)
    assert rc == 2
    assert payload["status"] == "separation_unverified"
    assert payload["guards"]["G_N2c_4_disjointness"]["status"] == "unverified"


def test_missing_input_file_is_a_loud_reject(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args[args.index("--replay-after") + 1] = str(tmp_path / "nope.json")
    assert MODULE.main(args) == 2
    payload = _read_out(tmp_path)
    assert payload["status"] == "missing_inputs"
    assert "replay_after" in payload["missing"]


def test_conflicting_eval_set_disclosure_is_not_same_source(tmp_path: Path) -> None:
    conflicting = _write(tmp_path, "cap0_after_conflict.json", _cap0_report(sha="different"))
    rc = MODULE.main(_args(tmp_path, cap0_after=conflicting))
    payload = _read_out(tmp_path)
    assert rc == 2
    assert payload["status"] == "faces_not_same_source"
    assert "eval_set_sha256" in payload["guards"]["G_N2c_3_cap0_same_source"]["conflicts"]


def test_unreadable_column_layout_is_rejected_not_counted_as_zero(tmp_path: Path) -> None:
    broken = copy.deepcopy(_cap0_report())
    del broken["dimensions"]["E"]
    rc = MODULE.main(
        _args(
            tmp_path,
            cap0_before=_write(tmp_path, "cap0_before_broken.json", broken),
            cap0_after=_write(tmp_path, "cap0_after_broken.json", broken),
        )
    )
    payload = _read_out(tmp_path)
    assert rc == 2
    assert payload["status"] == "column_layout_unreadable"


def test_nested_identity_disclosure_is_found_and_taken_is_published() -> None:
    """㊵-553 的那条取法错：出件方把 `eval_set_sha256` 出版在 `identity` 子字典里，
    而本件第一版只在顶层找 ⇒ 真件带着摘要却报 `missing`。两侧都要能取到，且**取到哪一层必须出版**。
    """

    before = {"identity": {"eval_set_sha256": "a" * 64}}
    after = {"identity": {"eval_set_sha256": "a" * 64}}
    out = MODULE._same_source_disclosure(before, after)
    assert out["status"] == "ok", out
    assert out["checked"] == ["eval_set_sha256"]
    assert out["took"] == {"eval_set_sha256": "identity->identity"}
    assert out["cross_layer"] == []


def test_top_level_and_nested_are_both_accepted_but_cross_layer_is_a_conflict() -> None:
    #: 两侧同名键取到**不同层**不算"同一处自述"——不许把一份顶层值与一份 identity 值当成同源证据。
    out = MODULE._same_source_disclosure(
        {"eval_set_sha256": "b" * 64}, {"identity": {"eval_set_sha256": "b" * 64}}
    )
    assert out["status"] == "conflict", out
    assert out["cross_layer"] == ["eval_set_sha256"]
    assert out["took"] == {"eval_set_sha256": "top_level->identity"}


def test_value_conflict_still_detected_after_the_lookup_widened() -> None:
    out = MODULE._same_source_disclosure(
        {"identity": {"eval_set_sha256": "c" * 64}}, {"identity": {"eval_set_sha256": "d" * 64}}
    )
    assert out["status"] == "conflict"
    assert out["conflicts"] == ["eval_set_sha256"]
    assert out["cross_layer"] == []


def test_missing_means_no_disclosure_anywhere_not_a_zero_match() -> None:
    #: 反面：不许把"两侧都没有"折叠成 `None == None` 的假绿。
    out = MODULE._same_source_disclosure({"format": "x"}, {"identity": {"other": 1}})
    assert out["status"] == "missing"
    assert out["checked"] == []
    assert out["took"] == {}
