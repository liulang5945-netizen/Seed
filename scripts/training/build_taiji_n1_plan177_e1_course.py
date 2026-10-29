"""Build the N177 E1 train/fixture course without opening sealed splits.

The builder materializes only the 128 train episodes and 8 public fixture
episodes. Query/holdout remain family identities and digests; their text is not
written and no model or checkpoint is loaded. The report is write-once and
records the structural leak audit, file SHAs and zero model/optimizer activity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

CPU_START = time.process_time()
WALL_START = time.monotonic()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.internalization import content_digest  # noqa: E402
from taiji.joint_state_consumption_course import (  # noqa: E402
    E1_COURSE_FORMAT,
    audit_course,
    build_fixture_episodes,
    build_train_episodes,
    e1_course_contract,
    e1_split_digest,
    episode_to_json,
)

DEFAULT_OUTPUT_DIR = ROOT / "data" / "course" / "n1_plan177_e1_20261029"
DEFAULT_REPORT = ROOT / "reports" / "taiji_n1_plan177_e1_course_20261029.json"
MODULE = "scripts.training.build_taiji_n1_plan177_e1_course"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, episodes) -> None:
    path.write_text(
        "".join(
            json.dumps(episode_to_json(episode), ensure_ascii=False) + "\n" for episode in episodes
        ),
        encoding="utf-8",
    )


def _source_hashes() -> dict[str, str]:
    paths = [
        ROOT / "taiji/joint_state_consumption_course.py",
        ROOT / "taiji/joint_t1_course.py",
        ROOT / "taiji/joint_input_producer.py",
        ROOT / "plans/reference/SPEC-N1-177_E1_state_consumption_contract_20261029.md",
        Path(__file__),
    ]
    return {str(path.relative_to(ROOT)): _sha256_file(path) for path in paths}


def build(output_dir: Path, report_path: Path) -> dict:
    if output_dir.exists() or report_path.exists():
        raise ValueError("E1 course output is write-once; target already exists")
    audit = audit_course()
    if not audit["ok"]:
        raise ValueError("E1 course structural audit failed")
    contract = e1_course_contract()
    train = build_train_episodes()
    fixture = build_fixture_episodes()
    if len(train) != 128 or len(fixture) != 8:
        raise ValueError("E1 train/fixture episode counts differ from the frozen contract")
    if any(episode.partition != "train" for episode in train + fixture):
        raise ValueError("E1 builder materialized a non-train episode")

    output_dir.mkdir(parents=True)
    train_path = output_dir / "train.jsonl"
    fixture_path = output_dir / "fixture.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(fixture_path, fixture)
    train_sha = _sha256_file(train_path)
    fixture_sha = _sha256_file(fixture_path)

    manifest_payload = {
        "format": "taiji-n177-e1-course-manifest-v1",
        "version": 1,
        "course": contract,
        "audit": audit,
        "train": {"path": train_path.name, "sha256": train_sha, "episodes": len(train)},
        "fixture": {"path": fixture_path.name, "sha256": fixture_sha, "episodes": len(fixture)},
        "sealed": {
            "materialized": False,
            "text_read": 0,
            "query_family_ids": contract["query_family_ids"],
            "holdout_family_ids": contract["holdout_family_ids"],
        },
        "source_sha256": _source_hashes(),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    manifest = {**manifest_payload, "manifest_sha256": content_digest(manifest_payload)}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = {
        "schema": "taiji-n177-e1-course-build-v1",
        "status": "built",
        "module": MODULE,
        "course_format": E1_COURSE_FORMAT,
        "course_sha256": contract["course_sha256"],
        "train_split_sha256": e1_split_digest(train),
        "fixture_split_sha256": e1_split_digest(fixture),
        "families_per_split": contract["families_per_split"],
        "episodes_per_family": contract["episodes_per_family"],
        "turns_per_episode": contract["turns_per_episode"],
        "train": {"path": str(train_path.relative_to(ROOT)), "sha256": train_sha},
        "fixture": {"path": str(fixture_path.relative_to(ROOT)), "sha256": fixture_sha},
        "manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": _sha256_file(manifest_path),
        },
        "audit": audit,
        "sealed_materialized": False,
        "sealed_text_read": 0,
        "model_calls": 0,
        "checkpoint_loads": 0,
        "optimizer_updates": 0,
        "can_promote": False,
        "cpu_seconds": time.process_time() - CPU_START,
        "wall_seconds": time.monotonic() - WALL_START,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the N177 E1 course (no sealed text)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run the structural audit without writing any course or report file",
    )
    args = parser.parse_args(argv)
    if args.check_only:
        audit = audit_course()
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0 if audit["ok"] else 2
    report = build(args.output_dir, args.report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "train_episodes": report["train"]["path"],
                "fixture_episodes": report["fixture"]["path"],
                "course_sha256": report["course_sha256"],
                "sealed_materialized": report["sealed_materialized"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
