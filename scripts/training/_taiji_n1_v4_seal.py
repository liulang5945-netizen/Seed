"""Strict PLAN63 transport seal: only the two frozen LF/CRLF byte artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEALS = {
    "plans/manifests/n1_relation_confirmation_v4_20261015.json": (
        "a148dbf6c65812b08b3f25103b014da82b97796c18134724dc2f03dcb6f12529",
        "ea1cd7df5ff18c90a9799220d771c14ac740fc7642d0761a0050e2d8942684b4",
    ),
    "reports/taiji_n1_v4_face_freeze_20261015.json": (
        "221982613df310877b2fc37cb767f8fbafce864710e8b8086138bfad7856cd47",
        "8fe8ae348aa93644b9fb62b2a80ab3f91da9bdd99a08df5f9f110046b737235b",
    ),
}


def check_bytes(content, raw_sha, lf_sha):
    digest = hashlib.sha256(content).hexdigest()
    if digest == raw_sha:
        canonical = content.replace(b"\r\n", b"\n")
        if hashlib.sha256(canonical).hexdigest() != lf_sha:
            raise ValueError("frozen CRLF bytes do not match the LF transport seal")
        variant = "original-crlf"
    elif digest == lf_sha:
        if (
            b"\r" in content
            or hashlib.sha256(content.replace(b"\n", b"\r\n")).hexdigest() != raw_sha
        ):
            raise ValueError("frozen LF bytes do not reconstruct the original seal")
        variant = "git-lf"
    else:
        raise ValueError("unrecognized confirmation bytes; no normalization of changed data")
    return dict(
        original_sha256=raw_sha, actual_sha256=digest, git_lf_sha256=lf_sha, variant=variant
    )


def check_all(root=ROOT):
    return {name: check_bytes((root / name).read_bytes(), *seals) for name, seals in SEALS.items()}
