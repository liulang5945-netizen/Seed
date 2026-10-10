"""P2-11 Taiji-owned IDE language chain Gate."""

from __future__ import annotations

import pytest

from scripts.training import eval_taiji_ide_language_chain as chain


@pytest.fixture(scope="module")
def report() -> dict:
    """一次 `evaluate()` 给两条判读用——这一支的门要建三次运行时＋三次种子，跑两遍不值。"""
    return chain.evaluate()


def test_ide_language_chain_gate_passes(report: dict) -> None:
    assert report["format"] == "taiji-w7-p2-11-ide-language-chain-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())


def test_the_ambiguity_branch_is_actually_walked(report: dict) -> None:
    #: DEBT-G87 的复算入口：这一支必须是因为「证据歧义」停下，不是因为目标不在场停下。
    #: 只看 `gate.passed` 分不开两者——两种情形都"在执行前停下"，但只有前者是被测行为。
    policy = report["policy"]
    assert policy["ambiguous_reason"] == "language_evidence_ambiguous"
    assert policy["ambiguous_status"] == "needs_clarification"


def test_a_missing_fixture_refuses_instead_of_faking_a_verdict(monkeypatch) -> None:
    #: 守卫能为假支：把夹具换成不在场的名字，必须响亮拒绝而不是跑出一份红判读。
    monkeypatch.setattr(chain, "REQUIRED_FIXTURES", ("nope/not-here.h",))
    with pytest.raises(RuntimeError, match="not-here.h"):
        chain._require_fixtures()


def test_the_consumed_fixture_is_the_historical_bytes() -> None:
    #: ㊵-662 取法更正的锚：夹具字节＝被 `d8bb37d67` 删掉那一份的逐字节副本（19 B）。
    #: 数值钉在测里而不是"能跑就行"，因为换夹具＝换这条支的被测对象。
    provenance = chain._fixture_provenance(chain.AMBIGUOUS_PATH)
    assert provenance["bytes"] == 19
    assert provenance["sha256"] == "d3139b187138b0a2b1fd9b889746cb293f2dfc1ec526153bb345b1617eca9ef4"
    assert provenance["path"] == chain.AMBIGUOUS_PATH
