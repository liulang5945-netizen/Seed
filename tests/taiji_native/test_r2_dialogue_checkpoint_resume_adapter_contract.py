import hashlib
import json

import pytest

from scripts.training.run_taiji_r2_checkpoint_resume_adapter import verify_receipt_ancestry
from scripts.training.run_taiji_r2_daily_dialogue_continuation_extension import Refuse


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_chain(tmp_path):
    parent = tmp_path / "commit-parent.json"
    parent.write_text(json.dumps({"bundle": "parent.pt"}))
    bundle = tmp_path / "child.pt"
    bundle.write_bytes(b"saved child weights")
    child = tmp_path / "commit-child.json"
    child.write_text(
        json.dumps(
            {"bundle": bundle.name, "sha256": sha(bundle), "previous_receipt_sha256": sha(parent)}
        )
    )
    (tmp_path / "latest.json").write_text(
        json.dumps({"receipt": child.name, "receipt_sha256": sha(child)})
    )
    return parent, child, bundle


def test_trained_child_can_resume_from_its_verified_parent_chain(tmp_path):
    parent, _, _ = make_chain(tmp_path)
    verify_receipt_ancestry(tmp_path, sha(parent))


@pytest.mark.parametrize("drift", ["bundle", "latest", "parent"])
def test_drift_is_refused_before_training(tmp_path, drift):
    parent, child, bundle = make_chain(tmp_path)
    anchor = sha(parent)
    if drift == "bundle":
        bundle.write_bytes(b"changed")
    elif drift == "latest":
        child.write_text("{}")
    else:
        parent.unlink()
    with pytest.raises(Refuse):
        verify_receipt_ancestry(tmp_path, anchor)
