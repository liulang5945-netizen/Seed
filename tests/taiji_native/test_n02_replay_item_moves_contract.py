"""`adjudicate_taiji_n02_replay_item_moves.py` 的合同测：代价只由**变差**那一侧决定，且逐条列必须与聚合列同分母。

四支判读值各有真走过的一支（`no_moves`／`hit_only`／`formed_only`／`hit_with_formed`），
另外六条拒绝支也都配了能为假的断言——尤其"少了 `--emit-per-item`"那一条，
它必须响亮拒而不是把"没有逐条数据"读成"没有代价"。
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "adjudicate_taiji_n02_replay_item_moves.py"


def _module():
    spec = importlib.util.spec_from_file_location("n02_item_moves_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(item_id: str, hit: bool, formed: int, texts: int = 3, cyclic: int = 0) -> dict:
    return {
        "id": item_id,
        "hit": hit,
        "cyclic": bool(cyclic),
        "expected_contains": [f"tok-{item_id}"],
        "text_range": [0, texts],
        "texts": texts,
        "formed_texts": formed,
        "cyclic_tail_texts": cyclic,
    }


def _report(
    checkpoint: str, rows: list[dict], *, items_sha: str = "same-sha", base_ok: bool = True
) -> dict:
    shifted = []
    cursor = 0
    for row in rows:
        item = dict(row)
        item["text_range"] = [cursor, cursor + row["texts"]]
        cursor += row["texts"]
        shifted.append(item)
    arm = {
        "repetition_penalty": 1.0,
        "items": len(shifted),
        "texts": cursor,
        "strict_hits": sum(1 for row in shifted if row["hit"]),
        "well_formed_texts": sum(int(row["formed_texts"]) for row in shifted),
        "cyclic_texts": sum(int(row["cyclic_tail_texts"]) for row in shifted),
        "per_item": shifted,
    }
    return {
        "format": "taiji-a30-repetition-penalty-v1",
        "chain": "base_raw_bytes_with_product_mask",
        "checkpoint": checkpoint,
        "items_sha256": items_sha,
        "limit": 24,
        "offset": 0,
        "base_sha256_unchanged": base_ok,
        "arms": [arm],
    }


def _run(module, tmp_path, before, after, out_name="moves.json"):
    b = tmp_path / "before.json"
    a = tmp_path / "after.json"
    b.write_text(json.dumps(before, ensure_ascii=False), encoding="utf-8", newline="\n")
    a.write_text(json.dumps(after, ensure_ascii=False), encoding="utf-8", newline="\n")
    out = tmp_path / out_name
    rc = module.main(["--before", str(b), "--after", str(a), "--out-report", str(out)])
    return rc, out


BASE_ROWS = [_row("V001", True, 3), _row("V002", False, 2), _row("V003", True, 3)]


def test_a_hit_loss_without_a_formed_change_is_named_hit_only(tmp_path) -> None:
    module = _module()
    after_rows = copy.deepcopy(BASE_ROWS)
    after_rows[0]["hit"] = False
    rc, out = _run(
        module, tmp_path, _report("mother.pt", BASE_ROWS), _report("candidate.pt", after_rows)
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "hit_only"
    lost = payload["per_penalty"][0]["hit_lost"][0]
    assert lost["id"] == "V001" and lost["hit"] == [True, False]
    assert lost["formed_texts"] == [3, 3], "hit_only 的支定义就是成句数没动"
    assert lost["expected_contains"] == ["tok-V001"]


def test_a_hit_loss_that_also_loses_a_formed_text_is_named_jointly(tmp_path) -> None:
    module = _module()
    after_rows = copy.deepcopy(BASE_ROWS)
    after_rows[0]["hit"] = False
    after_rows[0]["formed_texts"] = 1
    rc, out = _run(
        module, tmp_path, _report("mother.pt", BASE_ROWS), _report("candidate.pt", after_rows)
    )
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "hit_with_formed"


def test_a_formed_only_drop_is_not_called_a_hit_loss(tmp_path) -> None:
    module = _module()
    after_rows = copy.deepcopy(BASE_ROWS)
    after_rows[1]["formed_texts"] = 0
    rc, out = _run(
        module, tmp_path, _report("mother.pt", BASE_ROWS), _report("candidate.pt", after_rows)
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "formed_only"
    assert payload["per_penalty"][0]["hit_lost"] == []


def test_a_gain_alone_is_not_a_cost(tmp_path) -> None:
    """只有题变好 ⇒ `no_moves`，且变好那题仍要出版在 hit_gained 里，不能被词表吞掉。"""
    module = _module()
    after_rows = copy.deepcopy(BASE_ROWS)
    after_rows[1]["hit"] = True
    rc, out = _run(
        module, tmp_path, _report("mother.pt", BASE_ROWS), _report("candidate.pt", after_rows)
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "no_moves"
    assert payload["per_penalty"][0]["hit_gained"][0]["id"] == "V002"


def test_the_four_verdicts_are_all_reachable_and_exhaustive(tmp_path) -> None:
    module = _module()
    produced = set()
    for index, mutation in enumerate(
        (
            lambda rows: rows,
            lambda rows: rows[0].__setitem__("hit", False) or rows,
            lambda rows: rows[1].__setitem__("formed_texts", 0) or rows,
            lambda rows: (
                rows[0].__setitem__("hit", False),
                rows[0].__setitem__("formed_texts", 1),
                rows,
            )[-1],
        )
    ):
        after_rows = mutation(copy.deepcopy(BASE_ROWS))
        rc, out = _run(
            module,
            tmp_path,
            _report("mother.pt", BASE_ROWS),
            _report(f"candidate{index}.pt", after_rows),
            out_name=f"moves{index}.json",
        )
        assert rc == 0
        produced.add(json.loads(out.read_text(encoding="utf-8"))["verdict"])
    assert produced == set(module.VERDICTS)


def test_a_per_item_list_that_disagrees_with_the_aggregate_is_refused(tmp_path) -> None:
    """同分母守卫：逐条列加起来不等于聚合列 ⇒ 切片错位，题级结论一律不出。"""
    module = _module()
    after = _report(
        "candidate.pt", [_row("V001", True, 3), _row("V002", False, 2), _row("V003", True, 3)]
    )
    after["arms"][0]["strict_hits"] = 99
    rc, out = _run(module, tmp_path, _report("mother.pt", BASE_ROWS), after)
    assert rc == 2
    assert out.exists() is False


def test_a_readout_made_without_the_flag_is_refused_not_read_as_zero_cost(tmp_path) -> None:
    module = _module()
    after = _report("candidate.pt", copy.deepcopy(BASE_ROWS))
    del after["arms"][0]["per_item"]
    assert _run(module, tmp_path, _report("mother.pt", BASE_ROWS), after)[0] == 2


def test_a_different_item_subset_is_refused(tmp_path) -> None:
    module = _module()
    after = _report("candidate.pt", copy.deepcopy(BASE_ROWS), items_sha="other-sha")
    assert _run(module, tmp_path, _report("mother.pt", BASE_ROWS), after)[0] == 2


def test_the_same_checkpoint_on_both_sides_is_refused(tmp_path) -> None:
    module = _module()
    assert (
        _run(
            module,
            tmp_path,
            _report("same.pt", BASE_ROWS),
            _report("same.pt", copy.deepcopy(BASE_ROWS)),
        )[0]
        == 2
    )


def test_a_readout_that_moved_weights_is_refused(tmp_path) -> None:
    module = _module()
    after = _report("candidate.pt", copy.deepcopy(BASE_ROWS), base_ok=False)
    assert _run(module, tmp_path, _report("mother.pt", BASE_ROWS), after)[0] == 2


def test_a_published_moves_artifact_is_not_clobbered(tmp_path) -> None:
    module = _module()
    after_rows = copy.deepcopy(BASE_ROWS)
    after_rows[0]["hit"] = False
    before, after = _report("mother.pt", BASE_ROWS), _report("candidate.pt", after_rows)
    rc, out = _run(module, tmp_path, before, after)
    assert rc == 0
    first = out.read_bytes()
    b = tmp_path / "before.json"
    a = tmp_path / "after.json"
    assert module.main(["--before", str(b), "--after", str(a), "--out-report", str(out)]) == 2
    assert out.read_bytes() == first
