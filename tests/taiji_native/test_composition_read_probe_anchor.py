"""14.3.0 anchor: the lifted read probe must agree with 10.21.1's fifteen published rows.

The probe moved out of ``verify_taiji_10_21_1.py`` into ``taiji/composition_read_probe.py``
so 14.3 can reuse it instead of writing a second read path. This test is the price of that
move: same model, same records, same descriptors, and every published field of the fifteen
rows has to come out identical -- floats included, not approximated.

The assertion branches on the bundle's identity (ledger ``D-20261111-03``).  A CI host that
rebuilds the bundle does not reproduce the sealed sha256, and calling that a read failure
would make this test red for a host property.  So:

``published_rows``       the file is the sealed asset: every published row must match.
``structural_invariants``  any other rebuild: only quantities that do not depend on bit
                           identity are asserted, and the branch is written to
                           ``output/ci_verify/composition_probe_branch.json``.

The test never skips.  A skip would hide the assertion on every CI run, which is worse than
a red.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import tempfile
from types import ModuleType

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.candidate_manifest import build_manifest, load_candidate  # noqa: E402
from taiji.composition_read_probe import descriptor_read  # noqa: E402
from taiji.core_09_bundle import EXPECTED_SHA256, sha256_of_file  # noqa: E402
from taiji.state_migration import resume_subject  # noqa: E402

CARD = ROOT / "reports/taiji_core_10_21_1_preflight_20261101.json"
# 与四个消费者同一口径（TAIJI_BUNDLE_PATH 可覆盖），这样「非封存重建件」那支
# 分支在本机就能被跑到，而不是只在 CI 上第一次执行。
_override = os.environ.get("TAIJI_BUNDLE_PATH")
BUNDLE = pathlib.Path(_override) if _override else ROOT / "output/taiji_core_09/subject_bundle.pt"
if not BUNDLE.is_absolute():
    BUNDLE = ROOT / BUNDLE
BRANCH_REPORT = ROOT / "output/ci_verify/composition_probe_branch.json"

COMPARE = (
    "target_index",
    "target_source_id",
    "target_mass",
    "target_density",
    "null_mass",
    "top1",
    "top1_source_id",
    "top1_is_target",
    "record_only_top1_is_target",
    "selected_entity",
    "recalled_norm",
)
READING_FIELDS = set(COMPARE) | {"record_masses", "groups", "top1_ambiguous"}


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    import importlib

    return importlib.import_module("scripts.training.verify_taiji_10_21_1")


@pytest.fixture(scope="module")
def published() -> list[dict]:
    return json.loads(CARD.read_text(encoding="utf-8"))["read"]["rows"]


@pytest.fixture(scope="module")
def subject():
    torch.set_num_threads(1)
    manifest = build_manifest(ROOT, BUNDLE)
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = pathlib.Path(directory) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")
        _loaded, payload = load_candidate(ROOT, manifest_path, BUNDLE, "frozen_eval")
    model, state, _optimizer = resume_subject(payload)
    return model, state


def build_records(runner: ModuleType) -> dict[str, list]:
    real = runner.records_from_pairs(runner.real_pairs())
    return {
        "real": real,
        "permuted": runner.records_from_pairs(runner.permuted_pairs()),
        "reversed": runner.reverse_records(real),
    }


def record_branch(branch: str, observed: str, extra: dict) -> None:
    BRANCH_REPORT.parent.mkdir(parents=True, exist_ok=True)
    try:
        label = str(BUNDLE.resolve().relative_to(ROOT))
    except ValueError:
        label = str(BUNDLE)
    payload = {
        "branch": branch,
        "bundle": label,
        "observed_sha256": observed,
        "expected_sha256": EXPECTED_SHA256,
        **extra,
    }
    BRANCH_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_lifted_probe_reproduces_every_published_row(runner, subject, published) -> None:
    model, state = subject
    observed = sha256_of_file(BUNDLE)
    sealed = observed == EXPECTED_SHA256
    records = build_records(runner)

    if sealed:
        mismatches = []
        compared = 0
        for row in published:
            reading = descriptor_read(model, state, row["descriptor"], records[row["condition"]])
            for field in COMPARE:
                compared += 1
                if reading[field] != row[field]:
                    mismatches.append(
                        {
                            "condition": row["condition"],
                            "descriptor": row["descriptor"],
                            "field": field,
                            "published": row[field],
                            "now": reading[field],
                        }
                    )
            compared += 1
            if reading["record_masses"] != row["record_masses"]:
                mismatches.append(
                    {
                        "condition": row["condition"],
                        "descriptor": row["descriptor"],
                        "field": "record_masses",
                        "published": row["record_masses"],
                        "now": reading["record_masses"],
                    }
                )
            for group_now, group_published in zip(reading["groups"], row["groups"], strict=True):
                for key in ("target_mass", "null_mass", "top1", "top1_is_target"):
                    compared += 1
                    if group_now[key] != group_published[key]:
                        mismatches.append(
                            {
                                "descriptor": row["descriptor"],
                                "field": f"group.{key}",
                                "now": group_now[key],
                            }
                        )
        record_branch("published_rows", observed, {"rows": len(published), "compared": compared})
        assert not mismatches, json.dumps(mismatches[:5], ensure_ascii=True)
        assert compared == len(published) * (len(COMPARE) + 1 + 4 * 4), compared
        return

    readings = [
        (row, descriptor_read(model, state, row["descriptor"], records[row["condition"]]))
        for row in published
    ]
    problems: list[dict] = []

    for row, reading in readings:
        if set(reading) != READING_FIELDS:
            problems.append(
                {"descriptor": row["descriptor"], "issue": "field set", "now": sorted(reading)}
            )
            continue
        if not 0.0 <= reading["target_mass"] <= 1.0:
            problems.append({"descriptor": row["descriptor"], "issue": "target_mass range"})
        if reading["null_mass"] < 0.0 or reading["null_mass"] > 1.0:
            problems.append({"descriptor": row["descriptor"], "issue": "null_mass range"})
        if not (math.isfinite(reading["recalled_norm"]) and reading["recalled_norm"] >= 0.0):
            problems.append({"descriptor": row["descriptor"], "issue": "recalled_norm"})
        if not isinstance(reading["target_index"], int) or not (
            0 <= reading["target_index"] < len(records[row["condition"]])
        ):
            problems.append({"descriptor": row["descriptor"], "issue": "target_index"})
        if len(reading["record_masses"]) != len(row["record_masses"]) or not all(
            math.isfinite(value) and value >= 0.0 for value in reading["record_masses"]
        ):
            problems.append(
                {
                    "descriptor": row["descriptor"],
                    "issue": "record_masses",
                    "now": len(reading["record_masses"]),
                }
            )
        if len(reading["groups"]) != 4:
            problems.append({"descriptor": row["descriptor"], "issue": "group count"})
        for name in ("top1_is_target", "record_only_top1_is_target"):
            if not isinstance(reading[name], bool):
                problems.append({"descriptor": row["descriptor"], "issue": name})

    # Order invariance: reversing the record order cannot move the target mass.
    # 1e-12 relative, not bit equality: the permutation changes the summation order,
    # so the last bits are allowed to move on a host that rounds differently.
    by_descriptor = {row["descriptor"]: reading for row, reading in readings}
    for row, reading in readings:
        if row["condition"] == "reversed":
            twin = by_descriptor[row["descriptor"]]
            if not math.isclose(
                reading["target_mass"], twin["target_mass"], rel_tol=1e-12, abs_tol=0.0
            ):
                problems.append({"descriptor": row["descriptor"], "issue": "order invariance"})

    # Source ids may only take values the published read already produced.
    allowed = {row["top1_source_id"] for row in published}
    for row, reading in readings:
        if reading["top1_source_id"] not in allowed:
            problems.append(
                {
                    "descriptor": row["descriptor"],
                    "issue": "top1_source_id",
                    "now": reading["top1_source_id"],
                }
            )

    record_branch(
        "structural_invariants",
        observed,
        {"rows": len(published), "problems": len(problems)},
    )
    assert not problems, json.dumps(problems[:5], ensure_ascii=True)
