"""Artifact-free face generation and lossless full evidence admission contracts."""

import gzip
import json
from copy import deepcopy

import pytest

from scripts.training import _taiji_n1_evidence_shard as shard
from scripts.training import build_taiji_n1_fresh_paired_admission as admission


def evidence():
    identity = dict(seed=11, arm="position", protocol=dict(schema="synthetic76"), phase="complete")
    record = dict(
        identity,
        completed_updates=2,
        losses=[dict(qa=0.123456789123), dict(qa=0.0)],
        measurements=[dict(groups=dict(writer=dict(actual_l2=0.001))), dict(groups={})],
        rows=[
            dict(raw_hex="e79fb3", answer="石", free_exact=True),
            dict(free_exact=False),
            dict(raw_hex="00"),
        ],
        additional=dict(all_fields_retained=[True, None, 1, "utf8"]),
    )
    return record, identity, dict(losses=2, measurements=2, rows=3)


def test_full_round_trip_identity_exact_floats_counts_and_exclusivity(tmp_path):
    record, identity, counts = evidence()
    path = tmp_path / "full.json.gz"
    seal = shard.write_new(path, record, identity, counts)
    assert shard.read_checked(path, seal, identity=identity) == record
    assert path.read_bytes()[4:8] == bytes(4)
    assert gzip.decompress(path.read_bytes()).endswith(b"\n")
    with pytest.raises(FileExistsError):
        shard.write_new(path, record, identity, counts)
    with pytest.raises(ValueError, match="identity"):
        shard.read_checked(path, seal, identity=dict(identity, seed=12))


@pytest.mark.parametrize(
    "field",
    (
        "compressed_sha256",
        "uncompressed_sha256",
        "compressed_bytes",
        "uncompressed_bytes",
        "counts",
    ),
)
def test_tampered_seal_is_rejected(tmp_path, field):
    record, identity, counts = evidence()
    path = tmp_path / "full.json.gz"
    seal = shard.write_new(path, record, identity, counts)
    altered = deepcopy(seal)
    if field.endswith("sha256"):
        altered[field] = "0" * 64
    elif field.endswith("bytes"):
        altered[field] += 1
    else:
        altered["counts"]["rows"] -= 1
    with pytest.raises(ValueError):
        shard.read_checked(path, altered)


def test_resealed_truncated_content_still_fails_exact_count(tmp_path):
    record, identity, counts = evidence()
    path = tmp_path / "full.json.gz"
    seal = shard.write_new(path, record, identity, counts)
    payload = json.loads(gzip.decompress(path.read_bytes()))
    payload["record"]["rows"].pop()
    raw = json.dumps(payload).encode()
    compressed = gzip.compress(raw, mtime=0)
    path.write_bytes(compressed)
    seal.update(
        compressed_sha256=shard.digest(compressed),
        compressed_bytes=len(compressed),
        uncompressed_sha256=shard.digest(raw),
        uncompressed_bytes=len(raw),
    )
    with pytest.raises(ValueError, match="counts"):
        shard.read_checked(path, seal)
    record["measurements"].pop()
    with pytest.raises(ValueError, match="counts"):
        shard.write_new(tmp_path / "incomplete.json.gz", record, identity, counts)


def test_fresh_face_exact_disjoint_and_complete_without_models():
    manifest = admission.build()
    assert manifest == admission.build()
    assert admission.validate(manifest) == dict(
        items=32,
        by_kind={k: 8 for k in admission.KINDS},
        entities=96,
        materials=48,
        model_queries=0,
    )
    bad = deepcopy(manifest)
    bad["episodes"][0]["split"] = "train"
    with pytest.raises(ValueError, match="exact"):
        admission.validate(bad)
    names, words = admission.exclusions()
    assert not {n for e in manifest["episodes"] for n in e["entities"]} & names
    assert all(len(e["observations"]) == 2 for e in manifest["episodes"])
    assert admission.FORMAL["total_updates"] == 221184
    assert admission.FORMAL["total_qa_sequences"] == 331776
    assert admission.FORMAL["cpu_limit_seconds"] == 9000
    assert admission.FORMAL["confirmation_queries"] == 0


def test_exact_lf_exclusive_face_prevents_reselection_before_readiness(tmp_path, monkeypatch):
    path = tmp_path / "face.json"
    record = dict(text="石", model_queries=0)
    admission.write_lf_new(path, record)
    assert b"\r" not in path.read_bytes()
    assert json.loads(path.read_bytes()) == record
    monkeypatch.setattr(admission, "MANIFEST", path)
    monkeypatch.setattr(admission, "REPORT", tmp_path / "receipt.json")

    def forbidden():
        raise AssertionError("reselection reached readiness")

    monkeypatch.setattr(admission, "readiness", forbidden)
    with pytest.raises(FileExistsError, match="exclusive"):
        admission.execute()
