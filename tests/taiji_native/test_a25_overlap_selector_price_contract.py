"""规则选择器定价仪器的契约测试（不加载基座）。

这支仪器在**进程内**替换 `ToldContentStore.best_match`（类级补丁）。
两件事必须被证住：① 装/卸是可逆的且不残留——同一 pytest 进程里后面还要跑别的电路测试；
② 三种退回路径各自计数正确，因为"覆盖完整"是这条读数成立与否的先决条件。
"""

from __future__ import annotations

import pytest
import torch

from scripts.training import price_taiji_r2_a25_overlap_selector as price
from taiji.copy_circuit import ToldContentStore


def _store_with_two_events() -> ToldContentStore:
    store = ToldContentStore(cue_dim=4, max_events=4)
    #: cue 全同＝余弦并列，让**原键**无法区分两件事件；只有规则键该分出胜负。
    store.record("我住在苏州。".encode(), torch.ones(4))
    store.record("地铁上被人踩了一脚。".encode(), torch.ones(4))
    return store


@pytest.fixture
def patched():
    original_store = ToldContentStore.best_match
    price._install()
    try:
        yield price
    finally:
        price._uninstall()
        price.PROBE.update({"query": None, "used": 0, "no_query": 0, "empty_store": 0})
        assert ToldContentStore.best_match is original_store


def test_install_and_uninstall_are_reversible(patched) -> None:
    import score_taiji_r2_copy_circuit_chat_cap as cap_module

    #: 装补丁期间：消费点上站的是包装件，`original` 里存着真身。
    original_store = price._overlap_best_match.original
    original_answer_raw = price._answer_raw_with_query.original
    assert cap_module._answer_raw is price._answer_raw_with_query
    assert original_answer_raw is not price._answer_raw_with_query
    price._uninstall()
    assert ToldContentStore.best_match is original_store
    assert cap_module._answer_raw is original_answer_raw
    price._install()  #: fixture 收尾还会再卸一次并核对复位


def test_rule_selects_the_event_sharing_characters_with_the_question(patched) -> None:
    store = _store_with_two_events()
    patched.PROBE["query"] = "我住哪？"
    picked = store.best_match(torch.ones(4))
    assert bytes(picked.content) == "我住在苏州。".encode()
    assert patched.PROBE["used"] == 1
    assert patched.PROBE["no_query"] == 0 and patched.PROBE["empty_store"] == 0


def test_empty_store_is_a_benign_fallback_not_a_coverage_gap(patched) -> None:
    store = ToldContentStore(cue_dim=4, max_events=4)
    patched.PROBE["query"] = "我住哪？"
    assert store.best_match(torch.ones(4)) is None
    assert patched.PROBE["empty_store"] == 1
    assert patched.PROBE["no_query"] == 0


def test_missing_query_with_events_to_choose_is_a_loud_coverage_gap(patched) -> None:
    store = _store_with_two_events()
    patched.PROBE["query"] = None
    #: 走原键：cue 并列时返回**先**入库的那条（`best_match` 只在严格大于时换），
    #: 与规则键的"同分取新"相反——所以这一格真发生时必须被计数器抓到。
    picked = store.best_match(torch.ones(4))
    assert bytes(picked.content) == "我住在苏州。".encode()
    assert patched.PROBE["no_query"] == 1
    assert patched.PROBE["used"] == 0


def test_ties_go_to_the_newer_event_matching_the_separability_probe(patched) -> None:
    store = ToldContentStore(cue_dim=4, max_events=4)
    store.record("甲乙丙。".encode(), torch.ones(4))
    store.record("丁乙戊。".encode(), torch.ones(4))
    patched.PROBE["query"] = "乙"
    picked = store.best_match(torch.ones(4))
    assert (
        bytes(picked.content) == "丁乙戊。".encode()
    ), "平手必须取 event_id 较大的一条（与 §9 oracle 档同序）"


def test_first_event_selector_is_the_ceiling_not_a_lucky_pick(tmp_path) -> None:
    """天花板档：两条告知都在库里，但永远选**最早**那条＝含答案的告知。

    它与 §13 的 `--store target` 消融同义（那条删干扰，这条不删但永不选它），
    所以两档读数应当对得上；对不上就说明有一档的构造写错了。
    """
    from taiji.copy_circuit import ToldContentStore

    original = ToldContentStore.best_match
    price._install("first")
    try:
        store = ToldContentStore(cue_dim=4, max_events=4)
        store.record("我住在苏州。".encode(), torch.ones(4))
        store.record("我表哥住在西安。".encode(), torch.ones(4))
        picked = store.best_match(torch.zeros(4))
        assert bytes(picked.content) == "我住在苏州。".encode()
    finally:
        price._uninstall()
        ToldContentStore.best_match = original


def test_answer_raw_wrapper_only_stamps_the_query(patched) -> None:
    seen: list[tuple[str, object]] = []

    def fake(runtime, prompt, history):
        seen.append((prompt, history))
        return "答复"

    original = price._answer_raw_with_query.original
    price._answer_raw_with_query.original = fake
    try:
        out = price._answer_raw_with_query(None, "我住哪？", [("我住在苏州。", "苏州")])
    finally:
        price._answer_raw_with_query.original = original
    assert out == "答复"
    assert price.PROBE["query"] == "我住哪？"
    assert seen == [("我住哪？", [("我住在苏州。", "苏州")])]
