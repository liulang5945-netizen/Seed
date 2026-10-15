"""Post-fidelity diagnostic keeps the exact original intervention identities."""

import json

import pytest

from scripts.training import audit_taiji_n1_fidelity_content_read as diagnostic


def test_interventions_pair_by_identity_not_report_order():
    manifest = json.loads(diagnostic.calibration.MANIFEST.read_text(encoding="utf-8"))
    old = json.loads(diagnostic.OLD_REPORT.read_text(encoding="utf-8"))
    pairs = diagnostic.paired_inputs(manifest, list(reversed(old["rows"])))
    lookup = {row["id"]: row["replacements"] for row in old["rows"]}
    assert len(pairs) == 96
    for item, replacements in pairs:
        assert item["split"] == "train"
        assert replacements == (lookup[item["id"]]["target"], lookup[item["id"]]["distractor"])
    with pytest.raises(ValueError, match="96 unique"):
        diagnostic.paired_inputs(manifest, old["rows"][:-1] + [old["rows"][0]])


def test_no_heldout_fidelity_reconstruction():
    with pytest.raises(ValueError, match="train episodes"):
        diagnostic.reconstruct(None, {"split": "holdout"}, "forbidden", None, 0, "abcdef")
