"""Contract guard for the P3b arm-corpus builder and the two shipped arm manifests.

Why this exists: the frozen P3b preregistration (§2.2) removed two confounds (the fresh identity
organ the relaxed guard attaches, and "any budget is more training") but not a third one that was
only measured after the arms were wired -- **data novelty**.  ``seed_beta.pt`` sits at tick
16,000,000 while its parent ``resumed_seed_corpus.pt`` sits at 4,800,200 on a *different* corpus,
so the state had already consumed the first 11,199,800 symbols of ``simple_zh_texts.jsonl``.  A
stream that starts at row 0 therefore spends a fifth to a quarter of its budget on replay, and
``treatment - control`` would mix "denser dialogue" with "less familiar data".

Both arms are now sliced from the same unseen window by one code path, differing only in the row
rule.  The pairing is pinned here from the **committed manifests** (the data files themselves are
gitignored), so a future edit that silently points an arm back at an unsliced stream fails:

* the skip is derived from checkpoint lineage, not typed by hand;
* both arms open at the same source row and claim zero replay;
* each arm covers the frozen 48 h budget with margin;
* dialogue density really does differ between the arms (the manipulation happened);
* the two streams are not the same stream (shared leading prefix under the cap);
* the builder refuses to emit a corpus it cannot fill, or one that duplicates its partner.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "scripts" / "training" / "build_p3b_arm_corpus.py"
TRAINER = REPO / "scripts" / "training" / "train_p3b_aligned.py"
DIALOGUE_MANIFEST = REPO / "plans" / "manifests" / "p3b_dialogue_fresh_manifest.json"
ALL_MANIFEST = REPO / "plans" / "manifests" / "p3b_all_fresh_manifest.json"

#: Frozen by train_p3b_aligned.BUDGET_TIERS["48h"].
BUDGET_SYMBOLS = 47_280_000
#: seed_beta.tick - resumed_seed_corpus.tick, measured from the two envelopes.
SEEN_PREFIX_SYMBOLS = 11_199_800
UNSEEN_FIRST_ROW = 2893

DIALOGUE_TEXT = "老师：甲\n乙：乙\n"
META_ONLY_TEXT = "作者：佚名\n正文一段\n"
PLAIN_TEXT = "这是一段没有角色名的叙述文字\n"
#: 1 boundary symbol + the UTF-8 length, so the arithmetic below stays readable.
DIALOGUE_SYMBOLS = 1 + len(DIALOGUE_TEXT.encode("utf-8"))


def _row(text: str) -> str:
    return json.dumps({"text": text}, ensure_ascii=False) + "\n"


def _load(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    for entry in (str(path.parent), str(REPO)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> Any:
    return _load("_p3b_arm_corpus_builder_under_test", BUILDER)


@pytest.fixture(scope="module")
def trainer() -> Any:
    return _load("_p3b_trainer_under_test", TRAINER)


def _source(path: Path, texts: list[str]) -> Path:
    path.write_text("".join(_row(text) for text in texts), encoding="utf-8")
    return path


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The row rule and the lineage-derived slice point
# --------------------------------------------------------------------------- #


def test_the_two_arms_differ_only_by_the_row_rule(builder: Any) -> None:
    assert builder.qualifies(PLAIN_TEXT, "all") is True
    assert builder.qualifies(DIALOGUE_TEXT, "dialogue") is True
    # a metadata speaker is not a speaker, so this row joins control but not treatment
    assert builder.qualifies(META_ONLY_TEXT, "dialogue") is False
    assert builder.qualifies(META_ONLY_TEXT, "all") is True


def test_unknown_rule_fails_closed(builder: Any) -> None:
    with pytest.raises(SystemExit, match="unknown rule"):
        builder.qualifies("anything", "shuffle")


def test_seen_prefix_is_derived_from_lineage_and_refuses_to_guess(builder: Any) -> None:
    kwargs: dict[str, Any] = {
        "resume_tick": 16_000_000,
        "prior_tick": 4_800_200,
        "resume_corpus_name": "simple_zh_texts.jsonl",
        "prior_corpus_name": "dialogue_extended_clean.jsonl",
        "corpus_name": "simple_zh_texts.jsonl",
    }
    assert builder.seen_prefix_symbols(**kwargs) == SEEN_PREFIX_SYMBOLS

    # the parent already trained on the target corpus: the seen prefix is longer than the tick
    # difference and the envelope records no offset, so slicing here would be a guess
    with pytest.raises(SystemExit, match="cannot bound the seen prefix"):
        builder.seen_prefix_symbols(**{**kwargs, "prior_corpus_name": "simple_zh_texts.jsonl"})
    with pytest.raises(SystemExit, match="not simple_zh_texts.jsonl"):
        builder.seen_prefix_symbols(**{**kwargs, "resume_corpus_name": "other.jsonl"})
    with pytest.raises(SystemExit, match="ahead of resume tick"):
        builder.seen_prefix_symbols(**{**kwargs, "prior_tick": 17_000_000})


# --------------------------------------------------------------------------- #
# Emission, and the two ways the builder refuses to produce a corpus
# --------------------------------------------------------------------------- #


def test_build_skips_the_seen_prefix_and_keeps_only_matching_rows(
    builder: Any, tmp_path: Path
) -> None:
    skip = 2 * (1 + 1)  # two one-character rows, boundaries included
    source = _source(tmp_path / "src.jsonl", ["a", "b", DIALOGUE_TEXT, PLAIN_TEXT, DIALOGUE_TEXT])
    out = tmp_path / "dialogue.jsonl"
    meta = builder.build(
        source, out, rule="dialogue", skip_symbols=skip, max_symbols=2 * DIALOGUE_SYMBOLS
    )
    assert meta["first_emitted_row"] == 2
    assert meta["emitted_rows"] == 2
    assert meta["dropped_rows_in_window"] == 1
    assert meta["replay_symbols_from_seen_region"] == 0
    assert meta["dialogue_density_of_emitted_rows"] == 1.0


def test_build_refuses_a_corpus_it_cannot_fill(builder: Any, tmp_path: Path) -> None:
    source = _source(tmp_path / "src.jsonl", [DIALOGUE_TEXT, DIALOGUE_TEXT])
    out = tmp_path / "all.jsonl"
    with pytest.raises(SystemExit, match="cannot cover the budget"):
        builder.build(source, out, rule="all", skip_symbols=0, max_symbols=10**9)
    assert not out.exists(), "a half-filled corpus must not survive as an arm's input"


def test_build_refuses_an_arm_that_duplicates_its_partner(
    builder: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts = [DIALOGUE_TEXT, PLAIN_TEXT, DIALOGUE_TEXT, PLAIN_TEXT]
    source = _source(tmp_path / "src.jsonl", texts)
    partner = _source(tmp_path / "partner_src.jsonl", texts)
    built = tmp_path / "partner.jsonl"
    builder.build(partner, built, rule="all", skip_symbols=0, max_symbols=10)
    monkeypatch.setattr(builder, "MAX_SHARED_PREFIX_SYMBOLS", 8)
    out = tmp_path / "also_all.jsonl"
    with pytest.raises(SystemExit, match="did not separate the streams"):
        builder.build(source, out, rule="all", skip_symbols=0, max_symbols=10, other_arm=built)
    assert not out.exists()


# --------------------------------------------------------------------------- #
# The shipped pairing: both arms read the same unseen window
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [DIALOGUE_MANIFEST, ALL_MANIFEST],
    ids=["treatment", "control"],
)
def test_arm_corpus_is_sliced_past_the_seen_prefix(builder: Any, path: Path) -> None:
    meta = _manifest(path)
    derivation = meta["skip_derivation"]
    assert meta["rule"] in builder.RULES
    assert meta["skip_symbols"] == SEEN_PREFIX_SYMBOLS
    assert int(derivation["beta"]["tick"]) - int(derivation["prior"]["tick"]) == SEEN_PREFIX_SYMBOLS
    assert derivation["beta"]["corpus_names"] == ["simple_zh_texts.jsonl"]
    assert derivation["prior"]["corpus_names"] == ["dialogue_extended_clean.jsonl"]
    assert meta["first_emitted_row"] == UNSEEN_FIRST_ROW
    assert meta["replay_symbols_from_seen_region"] == 0
    assert meta["emitted_symbols"] >= BUDGET_SYMBOLS, "the arm must be able to spend its budget"
    assert meta["boundary_symbol"] == builder.BOUNDARY


def test_the_two_arm_streams_are_the_same_window_but_not_the_same_stream() -> None:
    dialogue = _manifest(DIALOGUE_MANIFEST)
    everything = _manifest(ALL_MANIFEST)
    assert dialogue["source"] == everything["source"]
    assert dialogue["first_emitted_row"] == everything["first_emitted_row"]
    check = everything["arm_overlap_check"]
    assert check["passed"] is True
    assert check["against"].endswith("p3b_dialogue_fresh.jsonl")
    assert check["shared_prefix_symbols"] <= check["cap"]
    assert check["cap"] <= BUDGET_SYMBOLS // 100, "two arms may not share a whole budget window"


def test_the_manipulation_actually_happened() -> None:
    dialogue = _manifest(DIALOGUE_MANIFEST)
    everything = _manifest(ALL_MANIFEST)
    assert dialogue["dialogue_density_of_emitted_rows"] == 1.0
    # control is the untouched distribution: far fewer multi-speaker documents
    assert everything["dialogue_density_of_emitted_rows"] < 0.5
    assert (
        dialogue["dialogue_density_of_emitted_rows"]
        > 2 * everything["dialogue_density_of_emitted_rows"]
    )
    assert (
        dialogue["source_rows_scanned"] > everything["source_rows_scanned"]
    ), "a denser filter must read further into the same source to fill the same budget"


def test_the_trainer_defaults_to_the_sliced_arm_corpora(trainer: Any) -> None:
    assert {trainer.SUBSET_PATH.name, trainer.CONTROL_CORPUS.name} == {
        "p3b_dialogue_fresh.jsonl",
        "p3b_all_fresh.jsonl",
    }
    assert set(trainer.ARM_MANIFESTS) == {trainer.SUBSET_PATH, trainer.CONTROL_CORPUS}
    for corpus, manifest in trainer.ARM_MANIFESTS.items():
        registered = _manifest(REPO / "plans" / "manifests" / manifest.name)
        assert registered["output"].endswith(corpus.name)
