"""14.3.0 anchor: the lifted read probe must reproduce 10.21.1's fifteen published rows.

The probe moved out of ``verify_taiji_10_21_1.py`` into ``taiji/composition_read_probe.py``
so 14.3 can reuse it instead of writing a second read path. This test is the price of that
move: same model, same records, same descriptors, and every published field of the fifteen
rows has to come out identical -- floats included, not approximated.
"""

from __future__ import annotations

import json
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
from taiji.state_migration import resume_subject  # noqa: E402

CARD = ROOT / "reports/taiji_core_10_21_1_preflight_20261101.json"
BUNDLE = ROOT / "output/taiji_core_09/subject_bundle.pt"

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


def test_lifted_probe_reproduces_every_published_row(runner, subject, published) -> None:
    model, state = subject
    records = {
        "real": runner.records_from_pairs(runner.real_pairs()),
        "permuted": runner.records_from_pairs(runner.permuted_pairs()),
        "reversed": runner.reverse_records(runner.records_from_pairs(runner.real_pairs())),
    }
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
    assert not mismatches, json.dumps(mismatches[:5], ensure_ascii=True)
    assert compared == len(published) * (len(COMPARE) + 1 + 4 * 4), compared
