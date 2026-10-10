"""Rule-revision seal guard (WP-3 exit 5, N2 preregistration I8/T-h).

Landing HANDOFF-M4 must not touch rule_revision=0 evidence: historical reports keep their
numbers and only gain a written version pointer.  "We did not change them" is worthless
unless something checks it, so the hashes below are a lockfile: if any of these artifacts
is rewritten, or any interpreting document loses its version label, this test fails.

The expected digests were captured immediately before the landing commit and are recorded
in the plan documents (推进计划修订, WP-3 证据封条表).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path

from reference_doc_history import reference_text
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports"
DOCS = REPO / "plans" / "reference"

#: sha256 of every rule_revision=0 artifact, captured before HANDOFF-M4 landed.
SEALED_REPORTS: dict[str, str] = {
    "taiji_p5_2b_group_causal_corpora_20260913.json": (
        "bcc72d366889e2d9451dca7225ff1ae134324dc1c91a91211905c9e1cd81c6eb"
    ),
    "taiji_b0_handoff_feasibility_probe_20260913.json": (
        "361a3d215bfc961ab681cdfaa6ccf75158f9f89f0b65ee45bc9c57109f88b55e"
    ),
    "taiji_b0_m1_counterfactual_20260913.json": (
        "f304025e48415301705295dab392c90cf138fd7199a58b07553574f6b040a6b2"
    ),
    "taiji_b0_m4_artifact_audit_20260913.json": (
        "3a1562c19405b6f94f05fd8a52af239c4ff82a15bc85ec0a3f6299fb1a7d4e0e"
    ),
    "taiji_b0_m4_hardening_20260913.json": (
        "d934ed3cb8c735eec055e155b3ec29a86795b0b8ad8ab952370bf458d41d3712"
    ),
    "taiji_b0_structure_space_probe_20260913.json": (
        "b3d678bbb53d93e07f4a129d5346474e9fd68d6b8d7dec269a27a797fb1a9733"
    ),
    "taiji_b0_structure_space_probe_wide_20260915.json": (
        "f527b78a739cdab65f8f8a038ef39e7e5f172f1981b144dae05ceb6a78375ae3"
    ),
    "taiji_b0_n2_stop_reason_disposition_20260914.json": (
        "b92cee4dabaeb67fb6ce2adbc213d5879f36c05323ca5f77fa4102c55e326f9e"
    ),
}

#: Documents whose numbers are only valid under rule_revision=0 and must say so.
VERSION_LABELLED_DOCS: tuple[str, ...] = (
    "M5_B0_HANDOFF_PROBE_RESULT_20260913.md",
    "M5_B0_M1_COUNTERFACTUAL_RESULT_20260913.md",
    "M5_B0_M4_ARTIFACT_AUDIT_20260913.md",
    "M5_B0_M4_HARDENING_RESULT_20260913.md",
    "M5_B0_STRUCTURE_SPACE_RESULT_20260913.md",
    "M5_B0_MECHANISM_AND_TASK_PRECHECK_20260913.md",
)

GATE = REPO / "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py"
COUNTERFACTUAL = REPO / "scripts/training/probe_taiji_b0_m1_counterfactual.py"


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


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def gate() -> Any:
    return _load("_seal_gate", GATE)


@pytest.fixture(scope="module")
def counterfactual() -> Any:
    return _load("_seal_counterfactual", COUNTERFACTUAL)


# --------------------------------------------------------------------------- #
# I8 / T-h: the revision-0 evidence is byte-for-byte untouched
# --------------------------------------------------------------------------- #
#
# H19g (2026-09-30): these eight digests were re-taken from the **git blob**, not from a
# working tree.  All eight had previously been recorded from a Windows checkout where
# `core.autocrlf = true` rewrote every reports/*.json to CRLF, so the bytes differed from
# the blob -- one file alone by 254 line endings.  On CI (Linux, LF) the identical commit
# therefore failed every one of these assertions: the seal was green only on the platform
# that produced it, which is the opposite of what a byte-for-byte seal is for.
#
# `.gitattributes` now pins `reports/*.json text eol=lf`, so the checkout matches the blob
# on every platform and these digests mean the same thing everywhere.
#
# If you ever need to re-seal: hash `git show HEAD:reports/<name>` (or any checkout under
# the new attribute), never a working-tree copy.  `git status` cannot catch the difference
# because normalization hides it -- compare raw bytes, as _digest does.


def test_every_sealed_report_still_matches_its_seal() -> None:
    for name, expected in SEALED_REPORTS.items():
        path = REPORTS / name
        assert path.exists(), f"revision-0 evidence disappeared: {name}"
        assert _digest(path) == expected, f"{name} was rewritten after the seal was taken"


def test_the_landing_wrote_a_new_report_not_the_sealed_one(gate: Any) -> None:
    """The gate's default output must not be any sealed artifact."""

    assert gate.REVISION_0_REPORT.name in SEALED_REPORTS
    assert gate.DEFAULT_REPORT != gate.REVISION_0_REPORT
    assert gate.DEFAULT_REPORT.name not in SEALED_REPORTS
    assert _digest(gate.REVISION_0_REPORT) == SEALED_REPORTS[gate.REVISION_0_REPORT.name]


def test_the_landed_report_is_not_written_into_a_revision_0_name(gate: Any) -> None:
    assert "20260913" not in gate.DEFAULT_REPORT.name


def test_no_instrument_defaults_its_output_onto_sealed_evidence() -> None:
    """One forgotten ``--output`` must not be able to erase this round's baseline.

    The gate was fixed first, but six other B0 instruments still defaulted onto a sealed
    filename -- and ``_write_json`` replaces in place, so a post-landing re-run would have
    written new numbers into a file whose name promises revision 0.
    """

    assignment = re.compile(r"DEFAULT_(?:OUTPUT|REPORT)\s*=\s*(?:\([\s\S]*?\)|[^\n]*)")
    offenders: list[str] = []
    for script in sorted((REPO / "scripts" / "training").glob("*.py")):
        text = script.read_text(encoding="utf-8")
        for match in assignment.finditer(text):
            hit = next((name for name in SEALED_REPORTS if name in match.group(0)), None)
            if hit is not None:
                offenders.append(f"{script.name} -> {hit}")
    assert not offenders, "default output points at sealed evidence: " + "; ".join(offenders)


# --------------------------------------------------------------------------- #
# Exit 5: interpreting documents must declare which revision they describe
# --------------------------------------------------------------------------- #


def test_each_revision_0_document_labels_its_version() -> None:
    for name in VERSION_LABELLED_DOCS:
        text = reference_text(DOCS / name)
        assert "rule_revision = 0" in text or "rule_revision=0" in text, name


# --------------------------------------------------------------------------- #
# Exit 1: exactly one composition rule ships, and the counterfactual says so
# --------------------------------------------------------------------------- #


def test_the_gate_ships_exactly_one_composition_rule(gate: Any) -> None:
    text = GATE.read_text(encoding="utf-8")
    body = text.split("def _member_episode", 1)[1].split("\ndef ", 1)[0]
    revision_0 = "chosen = bindable[0]" in body
    revision_1 = 'return finish("all_members_blocked")' in body
    assert revision_0 != revision_1, "exactly one composition rule may be in the body"
    assert (0 if revision_0 else 1) == gate.RULE_REVISION
    assert getattr(gate, "COMPOSITION_RULE", None) == (
        "priority_fallback" if revision_0 else "m4_failure_handoff"
    )


def test_the_shipped_variant_is_identity_and_the_others_still_refuse(
    counterfactual: Any, gate: Any
) -> None:
    """Identity is allowed only for a replacement that actually shipped."""

    _, delta = counterfactual.build_counterfactual(gate, "m4_failure_handoff")
    assert delta["variant_is_identity"] is True
    # C2：``added_lines == 0`` 与 ``len(already_applied) == replacement_count`` 都由同一组 status
    # 算出（identity 一成立就必然为真），断言它们等于没测。在一支纯文本变换函数里唯一"测出来"
    # 的事实是关于**已发布源码**的命中次数——那正是 status 记的东西，所以钉原始量而不是摘要。
    assert [item["status"] for item in delta["replacements"]] == [
        "already_applied",
        "already_applied",
    ]
    # 试过 `episode_fn.__code__.co_code == gate._member_episode.__code__.co_code`，**不成立**：
    # 实测两处 co_code 等长（2920 B）、co_names/co_varnames/co_freevars/co_cellvars 全等，只有
    # 偏移 175 的一个 oparg 差 1（0x0a vs 0x0b），原因是 exec 出来的臂把合成文件名
    # ``<counterfactual:...>`` 带进了它内嵌的 `finish`/`<genexpr>` 常量，常量表去重结果随之位移。
    # ⇒ 指令级比较不能当 identity 判据（同理 getsource 也取不到，文件名不在磁盘上）。
    # 这里能钉的就是原始量本身：两条置换都真的在已发布源码里命中 0 次、目标文本已在。
    refusals = 0
    for variant in ("m1a_no_progress", "m1b_tick_rotation", "m2a_progress_plus_handoff"):
        try:
            counterfactual.build_counterfactual(gate, variant)
        except SystemExit:
            refusals += 1
    assert refusals == 3, "revision-0 counterfactuals must not silently become identity"


def test_the_frozen_attribute_is_never_rebound(counterfactual: Any, gate: Any) -> None:
    """Building an arm must not rebind the mechanism under test, or the probe is circular.

    C2: the removed line compared one object's source with **itself**.  What is checkable is a
    cross-object comparison (the in-memory body against the bytes on disk) and the delta field,
    which the probe now genuinely measures instead of comparing to a freshly ``exec``'d object.
    """

    before = gate._member_episode
    episode_fn, delta = counterfactual.build_counterfactual(gate, "m4_failure_handoff")
    assert gate._member_episode is before
    assert delta["frozen_attribute_unchanged"] is True
    assert inspect.getsource(before) in GATE.read_text(encoding="utf-8")
    assert episode_fn is not before
