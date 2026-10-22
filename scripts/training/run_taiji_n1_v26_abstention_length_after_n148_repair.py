"""PLAN-N1-149A：隔离修复readiness中的N1-148 verifier模块别名。"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v25_abstention_after_n147 as n148_builder
from scripts.training import build_taiji_n1_v26_abstention_length_after_n148 as face_builder
from scripts.training import eval_taiji_n1_v26_abstention_length_after_n148 as frozen_eval
from scripts.training import seal_ast  # noqa: E402

REPAIR_PLAN = ROOT / "plans/reference/PLAN-N1-149A_readiness_module_binding_correction_20261020.md"
REPAIR_PLAN_SHA256 = "9b57cdd68b36df8b6d5035ca1ee2803a387dd522ac054472c0ddc5695f295876"
N149_PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-149_matched_abstention_context_length_curve_diagnostic_draft_20261019.md"
)
N149_PLAN_SHA256 = "d6dfb5a9e6e6f5d563a7d137ef838a197c7bbff98cbca52d85fe4a11de2e6497"
N149_BUILDER_SHA256 = "0122f9baa7e6fe3464efc62de9c7696928b1038793afc4c37680a15e88b5db9a"
N149_EVALUATOR_SHA256 = "119ee5c8ad6eca014fc09d64e09658e008399ac6d093e860108aa0bf4b0820da"
N148_BUILDER_SHA256 = "c348fe023f397dfcd3ddbe09207b94fddc9570285610480a356ffa0b8d3bde49"
FACE = face_builder.FACE
FACE_SHA256 = "3c2bc75b61bbdfd609f79d3d3c2dbea9857e57b140d389cec3b3d3de3e588788"
FAILURE_RECEIPT = ROOT / "reports/taiji_n1_v26_abstention_length_preflight_failure_20261020.json"
FAILURE_RECEIPT_SHA256 = "682a60d24b7137bc24df59965ba6bcee1b1386396d7b91311d4a06bf7b2ca4a9"
TEST = ROOT / "tests/taiji_native/test_n1_plan149a_v26_readiness_binding_repair_contract.py"

_CONFIGURED = False
_ORIGINAL_SOURCE_HASHES = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_pinned(path: Path, expected: str) -> None:
    # Phase 11: 冻结钉点为历史事实——在位字节或 git 历史 blob 任一核验通过即可
    if not seal_ast.verify_path_pin(path, expected):
        raise RuntimeError("frozen source hash mismatch: " + str(path))


def _source_hashes_with_addendum(prepared: dict, specs: list[dict]) -> dict[str, str]:
    if _ORIGINAL_SOURCE_HASHES is None:
        raise RuntimeError("N1-149A runtime hooks have not been configured")
    hashes = _ORIGINAL_SOURCE_HASHES(prepared, specs)
    for path in (REPAIR_PLAN, Path(__file__).resolve(), TEST, FAILURE_RECEIPT):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"N1-149A readiness source missing: {path}")
        hashes[str(path)] = sha256(path)
    return hashes


def configure_runtime() -> dict:
    """Attach the shared N1-144 hash checker in memory and bind repair provenance."""
    global _CONFIGURED, _ORIGINAL_SOURCE_HASHES
    for path, expected in (
        (REPAIR_PLAN, REPAIR_PLAN_SHA256),
        (N149_PLAN, N149_PLAN_SHA256),
        (Path(face_builder.__file__).resolve(), N149_BUILDER_SHA256),
        (Path(frozen_eval.__file__).resolve(), N149_EVALUATOR_SHA256),
        (Path(n148_builder.__file__).resolve(), N148_BUILDER_SHA256),
        (FACE, FACE_SHA256),
        (FAILURE_RECEIPT, FAILURE_RECEIPT_SHA256),
    ):
        _require_pinned(Path(path), expected)
    if not TEST.is_file():
        raise FileNotFoundError(f"N1-149A contract test missing: {TEST}")

    shared = n148_builder.n147_builder.n144_builder
    verifier = shared._verify_hashes
    existing = getattr(n148_builder, "n144_builder", None)
    if existing is not None and existing is not shared:
        raise RuntimeError("N1-148 already exposes a different N1-144 verifier module")
    n148_builder.n144_builder = shared

    if not _CONFIGURED:
        _ORIGINAL_SOURCE_HASHES = frozen_eval._source_hashes
        frozen_eval._source_hashes = _source_hashes_with_addendum
        _CONFIGURED = True
    return {
        "verifier": verifier,
        "n149_builder_sha256": N149_BUILDER_SHA256,
        "n149_evaluator_sha256": N149_EVALUATOR_SHA256,
        "repair_plan_sha256": REPAIR_PLAN_SHA256,
        "repair_entrypoint_sha256": sha256(Path(__file__).resolve()),
        "repair_test_sha256": sha256(TEST),
        "failure_receipt_sha256": sha256(FAILURE_RECEIPT),
    }


def main() -> int:
    configure_runtime()
    return frozen_eval.main()


if __name__ == "__main__":
    raise SystemExit(main())
