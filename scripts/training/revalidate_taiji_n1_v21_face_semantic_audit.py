"""为v21 face的一次性语义审计提供格式修订后的零查询哈希复核。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import audit_taiji_n1_v21_face_semantics as audit
from scripts.training import build_taiji_n1_v21_confirmation_after_n141 as builder

PLAN = builder.PLAN
FACE = builder.FACE
SOURCE_EXCLUSIONS = builder.SOURCE_EXCLUSIONS
FACE_CHECKS = builder.FACE_CHECKS
PREPARED = builder.PREPARED
PRIOR_AUDIT = audit.REPORT
REPORT = ROOT / "reports/taiji_n1_v21_face_semantic_audit_revalidation_20261019.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _require_replay_matches(prior: dict, semantic: dict, face_sha: str, exclusion_sha: str) -> bool:
    if (
        prior.get("complete") is not True
        or prior.get("face_sha256") != face_sha
        or prior.get("source_exclusion_sha256") != exclusion_sha
        or prior.get("semantic_checks") is None
    ):
        raise ValueError("original one-shot semantic audit is not bound to current v21 bytes")
    if semantic != prior["semantic_checks"]:
        raise ValueError("semantic replay differs from the original v21 face audit")
    return True


def revalidate() -> dict:
    if REPORT.exists():
        raise FileExistsError("v21 face semantic revalidation is write-once")
    prior = read_json(PRIOR_AUDIT)
    face = read_json(FACE)
    exclusions = read_json(SOURCE_EXCLUSIONS)
    semantic = audit.audit_face(face, exclusions)
    _require_replay_matches(prior, semantic, sha256(FACE), sha256(SOURCE_EXCLUSIONS))
    source_paths = (
        PLAN,
        FACE,
        SOURCE_EXCLUSIONS,
        FACE_CHECKS,
        PREPARED,
        PRIOR_AUDIT,
        Path(audit.__file__),
        Path(__file__),
    )
    source_hashes = {str(Path(path).resolve()): sha256(Path(path)) for path in source_paths}
    unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in source_hashes.items()
    )
    if not unchanged:
        raise ValueError("v21 semantic audit revalidation source changed")
    old_audit_script = next(
        (
            digest
            for path, digest in prior.get("source_hashes", {}).items()
            if Path(path).resolve() == Path(audit.__file__).resolve()
        ),
        None,
    )
    report = {
        "schema": "n1-plan142-v21-face-semantic-audit-revalidation-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": sha256(PLAN),
        "face_sha256": sha256(FACE),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "prior_audit_sha256": sha256(PRIOR_AUDIT),
        "prior_audit_script_sha256": old_audit_script,
        "current_audit_script_sha256": sha256(Path(audit.__file__)),
        "semantic_replay_equal": True,
        "semantic_checks": semantic,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "v3_holdout_queries": 0,
        "can_promote": False,
        "summary_zh": "原始v21逐题语义审计与格式修订后的代码重放完全一致；face、manifest和既有检查报告字节未变。",
    }
    write_new(REPORT, report)
    return report


def main() -> int:
    result = revalidate()
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "face_sha256": result["face_sha256"],
                "semantic_replay_equal": result["semantic_replay_equal"],
                "model_queries": result["model_queries"],
                "parameter_updates": result["parameter_updates"],
                "can_promote": result["can_promote"],
                "summary_zh": result["summary_zh"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
