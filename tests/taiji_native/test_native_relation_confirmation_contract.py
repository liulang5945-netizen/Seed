"""Fixed face generation and paired initialization without model queries."""

from copy import deepcopy

import pytest

from scripts.training import _taiji_n1_relation_paired_transfer as transfer
from scripts.training import audit_taiji_n1_local_relation as engineering
from scripts.training import build_taiji_n1_confirmation_v4 as face


def test_face_disjoint_complete_deterministic_and_exclusive(tmp_path, monkeypatch):
    manifest = face.build()
    assert manifest == face.build()
    summary = face.validate(manifest)
    assert summary["items"] == 32 and summary["materials"] == 48
    assert summary["by_kind"] == {kind: 8 for kind in face.KINDS}
    bad = deepcopy(manifest)
    bad["episodes"][0]["split"] = "train"
    with pytest.raises(ValueError, match="exact"):
        face.validate(bad)
    monkeypatch.setattr(face, "MANIFEST", tmp_path / "face.json")
    monkeypatch.setattr(face, "REPORT", tmp_path / "receipt.json")
    receipt = face.execute()
    assert receipt["summary"]["model_queries"] == 0
    with pytest.raises(FileExistsError):
        face.execute()


def test_seed_initialization_preserves_old_parameters_and_moments():
    transfer.torch.set_num_threads(1)
    _, payload, _ = transfer.source_payload()
    signatures = []
    shared = []
    new = []
    for seed in (11, 12, 13, 11):
        model, optimizer = transfer.migrate(payload, seed)
        bundle = model.bundle(optimizer)
        signatures.append(transfer.source.paired.fingerprint(bundle))
        shared.append(
            transfer.source.paired.fingerprint(
                {k: v for k, v in bundle["state"].items() if k not in transfer.NEW_PARAMETERS}
            )
        )
        new.append(
            transfer.source.paired.fingerprint(
                {k: v for k, v in bundle["state"].items() if k in transfer.NEW_PARAMETERS}
            )
        )
        assert transfer.source.paired.fingerprint(
            bundle["optimizer"]
        ) == transfer.source.paired.fingerprint(
            payload["candidate"]["optimizer"]
            | {"param_groups": bundle["optimizer"]["param_groups"]}
        )
        if seed == 11:
            ready = transfer.json.loads(engineering.REPORT.read_text(encoding="utf-8"))
            expected = deepcopy(
                transfer.torch.load(
                    ready["restore_check"]["path"], map_location="cpu", weights_only=True
                )["candidate"]
            )
            for session in engineering.SESSIONS:
                del expected["sessions"][session]
            assert transfer.source.paired.fingerprint(bundle) == transfer.source.paired.fingerprint(
                expected
            )
    assert signatures[0] == signatures[3]
    assert len(set(shared)) == 1 and len(set(new[:3])) == 3
    with pytest.raises(ValueError, match="seeds"):
        transfer.migrate(payload, 14)
