"""Sealed data survives checkout newline transport; any other edit is rejected."""

import subprocess

import pytest

from scripts.training import _taiji_n1_v4_seal as seal


def test_worktree_and_git_blob_are_only_frozen_variants():
    current = seal.check_all()
    assert len(current) == 2
    for name, pair in seal.SEALS.items():
        blob = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=seal.ROOT)
        checked = seal.check_bytes(blob, *pair)
        assert checked["original_sha256"] == current[name]["original_sha256"]
        with pytest.raises(ValueError, match="unrecognized"):
            seal.check_bytes(blob + b" ", *pair)
        with pytest.raises(ValueError, match="unrecognized"):
            seal.check_bytes(blob.replace(b"\n", b"\r\n", 1), *pair)


def test_exact_crlf_and_lf_roundtrip():
    for name, pair in seal.SEALS.items():
        content = (seal.ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        assert seal.check_bytes(content, *pair)["variant"] == "git-lf"
        assert (
            seal.check_bytes(content.replace(b"\n", b"\r\n"), *pair)["variant"] == "original-crlf"
        )
