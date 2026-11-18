"""Subject registry: every trained subject is named, fingerprinted and verified.

Three readings on 2026-11-18 were void because a probe silently used an untrained
model.  Naming alone would not have prevented that: the loader must refuse anything
it cannot verify.  So subjects are addressed by key, never by path, and every load
checks the sha256 against the registry, failing closed.

Auto-discovery is deliberately disabled: a glob over ``data/experiments`` also
finds bundles from other lines (c2, p4_1), and registering them here would recreate
the very mixing this file exists to prevent.

Usage:
    python -m scripts.training.subject_registry --build
    python -m scripts.training.subject_registry --list
    python -m scripts.training.subject_registry --verify
    python -m scripts.training.subject_registry --drill
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REGISTRY = ROOT / "plans/manifests/subject_registry_20261118.json"
A5_RUN = "data/experiments/taiji_u2/a5/20261116T0520Z"
SEEDS = ("20261110", "20261111", "20261112")

DECLARED = (
    [
        (f"a5-pre-seed{seed}", f"{A5_RUN}/seed-{seed}/bundle-pre-feedback.pt", "a5-pre")
        for seed in SEEDS
    ]
    + [
        (
            "e5-organ-seed20261110",
            "data/experiments/taiji_u2/e5_9/bundle-parent-plus-organ.pt",
            "e5-organ",
        ),
    ]
    + [
        (f"a5-post-seed{seed}", f"{A5_RUN}/seed-{seed}/bundle-post-feedback.pt", "a5-post")
        for seed in SEEDS
    ]
)

FAMILIES = {
    "e5-organ": {
        "trained_on": "frozen a5 parent + trained record organ (E5.9)",
        "verified": {
            "a_fit_accuracy": 1.0,
            "b_with_record": 1.0,
            "source": "probe_e5_9_organ_20261118",
        },
        "may_be_used_for": ["E5 correction measurements", "E7 organ comparisons"],
        "must_not_be_used_for": ["claims that the parent itself learned correction"],
    },
    "a5-pre": {
        "trained_on": "task A only (A.5 protocol, 600 updates)",
        "verified": {"task_a_fit_accuracy": 1.0, "source": "probe_e5_eos_contrast_v2_20261118"},
        "may_be_used_for": ["E5 mechanism probes", "task-A reproduction"],
        "must_not_be_used_for": ["correction-rate claims (no feedback applied)"],
    },
    "a5-post": {
        "trained_on": "task A, then one feedback pass (A.5 protocol)",
        "verified": {"note": "not independently re-verified; cite the A.5 card"},
        "may_be_used_for": ["post-feedback comparisons"],
        "must_not_be_used_for": ["correctness claims without calibration"],
    },
}
NOTE = (
    "Checkpoints under output/ are transient arm states with mixed protocols; they "
    "are not registered and must not be cited as subjects. Every entry is declared "
    "by hand, never discovered by glob."
)


class SubjectError(RuntimeError):
    """Raised when a subject cannot be resolved or verified."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    subjects = {}
    for key, relative, family in DECLARED:
        path = ROOT / relative
        if not path.is_file():
            raise SubjectError(f"declared_subject_missing: {relative}")
        payload = torch.load(path, weights_only=False)
        subjects[key] = {
            "path": relative,
            "sha256": _sha(path),
            "update": payload.get("update"),
            "revision": payload.get("revision"),
            "tensors": len(payload.get("model", {})),
            "family": family,
            **FAMILIES[family],
        }
    return {
        "schema": "taiji-subject-registry-v1",
        "date": "2026-11-18",
        "note": NOTE,
        "subjects": subjects,
    }


def load_registry() -> dict:
    if not REGISTRY.exists():
        raise SubjectError(f"registry_missing: {REGISTRY}")
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def resolve(key: str) -> dict:
    registry = load_registry()
    entry = registry["subjects"].get(key)
    if entry is None:
        raise SubjectError(f"unknown_subject: {key!r}; registered: {sorted(registry['subjects'])}")
    path = ROOT / entry["path"]
    if not path.is_file():
        raise SubjectError(f"subject_file_missing: {entry['path']}")
    actual = _sha(path)
    if actual != entry["sha256"]:
        raise SubjectError(
            f"subject_digest_mismatch: {key} expected {entry['sha256'][:16]} got {actual[:16]}"
        )
    return entry


def load_subject(key: str, contract: dict | None = None):
    """Return (model, entry) for a registered subject.  Fails closed."""
    from scripts.training import run_a5_22_5 as a5

    entry = resolve(key)
    if contract is None:
        contract = json.loads(
            (ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json").read_text(
                encoding="utf-8"
            )
        )
    payload = torch.load(ROOT / entry["path"], weights_only=False)
    if payload.get("update") != entry["update"]:
        raise SubjectError(
            f"subject_update_mismatch: {key} registry {entry['update']!r} file {payload.get('update')!r}"
        )
    model = a5.build_subject(contract)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, entry


def drill() -> int:
    """Both directions must fire: a good key loads, a bad key and a tampered digest fail."""
    registry = load_registry()
    key = sorted(registry["subjects"])[0]
    print("positive:", key, "->", resolve(key)["path"])
    try:
        resolve("no-such-subject")
        print("RED MISSED: unknown key resolved")
        return 1
    except SubjectError as error:
        print("red 1 ok:", str(error)[:90])
    tampered = json.loads(json.dumps(registry))
    tampered["subjects"][key]["sha256"] = "0" * 64
    REGISTRY.write_text(json.dumps(tampered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        resolve(key)
        print("RED MISSED: tampered digest resolved")
        return 1
    except SubjectError as error:
        print("red 2 ok:", str(error)[:90])
    finally:
        REGISTRY.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--drill", action="store_true")
    args = parser.parse_args()
    if args.build:
        data = build()
        REGISTRY.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("registered", len(data["subjects"]), "subjects ->", REGISTRY.relative_to(ROOT))
    if args.list:
        for key, entry in sorted(load_registry()["subjects"].items()):
            print(
                f"{key:20s} {entry['sha256'][:12]}  update={str(entry['update']):9s} {entry['trained_on']}"
            )
    if args.verify:
        registry = load_registry()
        for key in sorted(registry["subjects"]):
            resolve(key)
        print("verified", len(registry["subjects"]), "subjects")
    if args.drill:
        return drill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
