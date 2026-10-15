import json
from copy import deepcopy

import pytest

from scripts.training import audit_taiji_n1_transfer_binding as audit


def test_frozen_inputs_match_original_and_trained_pairs():
    rows = json.loads(audit.MANIFEST.read_text(encoding="utf-8"))["rows"]
    train = audit.source.checked_train()
    pairs = audit.source.checked_pairs(train)
    assert audit.validate_inputs(rows, train, pairs) is rows
    wrong = deepcopy(rows)
    wrong[0]["variants"]["query_switch"]["answer"] = "wrong"
    with pytest.raises(ValueError, match="trained pair"):
        audit.validate_inputs(wrong, train, pairs)
    wrong = deepcopy(rows)
    wrong[0]["variants"]["rename"]["split"] = "holdout"
    with pytest.raises(ValueError, match="train"):
        audit.validate_inputs(wrong, train, pairs)
    with pytest.raises(ValueError, match="96 unique"):
        audit.validate_inputs(rows[:-1] + [rows[0]], train, pairs)


def test_single_model_protocol_fields_are_nested():
    payload = {"protocol": dict(audit.source.PROTOCOL), "completed_updates": 6144}
    assert "seed" not in payload and "arm" not in payload
    assert audit.checked_payload(payload) is payload
    with pytest.raises(ValueError, match="protocol"):
        audit.checked_payload(dict(payload, completed_updates=9216))
