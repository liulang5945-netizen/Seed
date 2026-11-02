"""Manifest-bound loading for one candidate: identity, provenance, code hashes, modes.

The manifest is the only thing a frozen evaluation is allowed to trust. It binds

* ``candidate_id`` -- a digest over every parameter tensor (name, dtype, shape, bytes);
* ``modules``      -- sha256 of every module on the scoring path;
* ``provenance``   -- the bundle file, its bytes and its sha256;
* ``config`` and ``state_schema`` -- what the loader must rebuild before touching state;
* ``modes``        -- ``frozen_eval`` (writes forbidden), ``controlled_learning``,
  ``continuous``.

``load_candidate`` refuses on any drift. It never writes anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

MODULES = (
    "taiji/living_core.py",
    "taiji/growth_credit.py",
    "taiji/constrained_learning.py",
    "taiji/state_migration.py",
    "taiji/native_development.py",
    "taiji/subject_action_scorer.py",
)
MODES = ("frozen_eval", "controlled_learning", "continuous")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_digest(parameters: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(parameters):
        tensor = parameters[name]
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_manifest(root: Path, bundle_path: Path) -> dict:
    payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
    parameters = payload["parameters"]
    return {
        "schema": "taiji-candidate-manifest-v1",
        "candidate_id": parameter_digest(parameters),
        "subject_id": payload["subject_id"],
        "revision": payload["revision"],
        "step": payload["step"],
        "config": payload["config"],
        "state_schema": {
            "keys": "tuple[str, int]",
            "h": list(payload["h"].shape),
            "fast": list(payload["fast"].shape),
            "nu_ema": list(payload["nu_ema"].shape),
            "parameters": len(parameters),
        },
        "encoding": {"bytes": "utf-8", "tokens_per_byte": 1},
        "modules": {relative: sha256_file(root / relative) for relative in MODULES},
        "provenance": {
            "bundle": str(bundle_path),
            "bytes": bundle_path.stat().st_size,
            "sha256": sha256_file(bundle_path),
        },
        "modes": {
            "frozen_eval": {"writes_forbidden": True, "learning_forbidden": True},
            "controlled_learning": {"writes_forbidden": False, "learning_forbidden": False},
            "continuous": {"writes_forbidden": False, "learning_forbidden": False},
        },
    }


def check_manifest(root: Path, manifest: dict, bundle_path: Path) -> list[str]:
    problems = []
    if manifest.get("schema") != "taiji-candidate-manifest-v1":
        problems.append(f"schema {manifest.get('schema')!r}")
    for relative, expected in (manifest.get("modules") or {}).items():
        path = root / relative
        if not path.exists():
            problems.append(f"missing module {relative}")
        elif sha256_file(path) != expected:
            problems.append(f"module drift {relative}")
    provenance = manifest.get("provenance") or {}
    if not bundle_path.exists():
        problems.append("missing bundle")
    else:
        if bundle_path.stat().st_size != provenance.get("bytes"):
            problems.append("bundle size drift")
        if sha256_file(bundle_path) != provenance.get("sha256"):
            problems.append("bundle hash drift")
    return problems


def load_candidate(root: Path, manifest_path: Path, bundle_path: Path, mode: str):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    problems = check_manifest(root, manifest, bundle_path)
    if problems:
        raise ValueError("manifest refuses: " + "; ".join(problems))
    payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
    if parameter_digest(payload["parameters"]) != manifest["candidate_id"]:
        raise ValueError("parameter digest does not match candidate_id")
    schema = manifest["state_schema"]
    if list(payload["h"].shape) != schema["h"] or list(payload["fast"].shape) != schema["fast"]:
        raise ValueError("state schema does not match the bundle")
    return manifest, payload
