"""PLAN-N1-147A：以独立入口修复readiness校验器绑定，不改冻结源码。"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v23_confirmation_after_n143 as n144_builder
from scripts.training import build_taiji_n1_v24_abstention_after_n146 as face_builder
from scripts.training import eval_taiji_n1_v24_abstention_after_n146 as frozen_eval
from scripts.training import seal_ast  # noqa: E402

REPAIR_PLAN = (
    ROOT / "plans/reference/PLAN-N1-147A_readiness_verifier_binding_correction_20261019.md"
)
REPAIR_PLAN_SHA256 = "7f92146eb583cea46951b439e49294738ff5800656b2b63d4185848abd383d39"
N147_PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-147_matched_known_unknown_abstention_diagnostic_draft_20261019.md"
)
N147_PLAN_SHA256 = "4d6c9799741d3290f5f1e6cda4325268ce086027e4d14a72f82a7f3662e09be8"
N147_BUILDER_SHA256 = "d3f1d03b483e731faed9ec323c959cc16f691c08b7ccc50338da2dd5248942a8"
N147_EVALUATOR_SHA256 = "611ab328f4b896cfde86891a815acbf6a1ad7fe77897b9231dd59ddb2db3fe80"
N144_BUILDER_SHA256 = "d7b4954f1b97b39dd708818c8454b88fd087d4ee66b9e03e7c3510923171e848"
FACE_SHA256 = "0622de968749c4f573007ad5bc5fcc8ee30a09ae660b3fecfd7db10e52ce7216"
FAILURE_RECEIPT = ROOT / "reports/taiji_n1_v24_abstention_preflight_failure_20261019.json"
FAILURE_RECEIPT_SHA256 = "77b3a0d8c3256b7cab72f20c30d6ac0aacac0d7e4c1a5a6244a1b73bcd050a25"
TEST = ROOT / "tests/taiji_native/test_n1_plan147a_v24_readiness_runner_repair_contract.py"

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


def _source_hashes_with_repair_bindings(prepared: dict, specs: list[dict]) -> dict[str, str]:
    if _ORIGINAL_SOURCE_HASHES is None:
        raise RuntimeError("repair runtime hooks are not configured")
    hashes = _ORIGINAL_SOURCE_HASHES(prepared, specs)
    for path in (REPAIR_PLAN, Path(__file__).resolve(), TEST, FAILURE_RECEIPT):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"N1-147A readiness source missing: {path}")
        hashes[str(path)] = sha256(path)
    return hashes


def configure_runtime() -> dict:
    """Bind the existing N1-144 verifier and add repair provenance in memory only."""
    global _CONFIGURED, _ORIGINAL_SOURCE_HASHES
    for path, expected in (
        (REPAIR_PLAN, REPAIR_PLAN_SHA256),
        (N147_PLAN, N147_PLAN_SHA256),
        (Path(face_builder.__file__).resolve(), N147_BUILDER_SHA256),
        (Path(frozen_eval.__file__).resolve(), N147_EVALUATOR_SHA256),
        (Path(n144_builder.__file__).resolve(), N144_BUILDER_SHA256),
        (face_builder.FACE, FACE_SHA256),
        (FAILURE_RECEIPT, FAILURE_RECEIPT_SHA256),
    ):
        _require_pinned(Path(path), expected)
    if not TEST.is_file():
        raise FileNotFoundError(f"N1-147A contract test missing: {TEST}")

    verifier = n144_builder._verify_hashes
    existing = getattr(face_builder, "_verify_hashes", None)
    if existing is not None and existing is not verifier:
        raise RuntimeError("N1-147 builder already exposes a different verifier")
    face_builder._verify_hashes = verifier

    if not _CONFIGURED:
        _ORIGINAL_SOURCE_HASHES = frozen_eval._source_hashes
        frozen_eval._source_hashes = _source_hashes_with_repair_bindings
        _CONFIGURED = True
    return {
        "verifier": verifier,
        "original_builder_sha256": N147_BUILDER_SHA256,
        "original_evaluator_sha256": N147_EVALUATOR_SHA256,
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
