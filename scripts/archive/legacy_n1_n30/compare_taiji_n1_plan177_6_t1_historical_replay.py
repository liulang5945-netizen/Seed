"""N177.6 E0：把当前树回执与各历史检出的复算并成一张逐臂对照表（只读）。

判据每条都能为假：
①每个历史面上的 15 臂，主进程与自己起的子进程指纹一致（该面自己的回执里已记）；
②每臂在**它自己绑定的 revision** 那个检出里复算出的指纹，等于当前树指纹；
③各面用的是同一份仪器字节（`instrument_sha256` 相同），否则"跨面相等"可能只是同一支坏仪器。

历史检出只读：本仪器不改它们，只记录 HEAD 与脏面（应只剩那份未跟踪的回执仪器）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "reports/taiji_n1_plan177_6_t1_endpoint_receipts_20261029.json"
REPORT = ROOT / "reports/taiji_n1_plan177_6_t1_historical_replay_20261029.json"
INSTRUMENT_RELATIVE = "scripts/training/verify_taiji_n1_plan177_6_t1_endpoint_receipts.py"
DEFAULT_TREES = {
    "5c9e7f97c": "C:/Users/23747/seed-hist-5c9e7f97c101",
    "f8c11d08e": "C:/Users/23747/seed-hist-f8c11d08e319",
    "ebbd71e87": "C:/Users/23747/seed-hist-ebbd71e879b5",
    "0f16776f5": "C:/Users/23747/seed-hist-0f16776f5f",
}


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(tree), *args],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    ).stdout


def load_face(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path).replace("\\", "/"),
        "instrument_sha256": payload.get("instrument_sha256"),
        "all_fresh_match": payload.get("all_fresh_match"),
        "all_isolated": payload.get("all_isolated"),
        "revisions": dict(payload.get("revisions", {})),
        "digests": {row["arm"]: row["fingerprint"]["digest"] for row in payload["receipts"]},
    }


def describe_tree(tree: Path) -> dict[str, Any]:
    instrument = tree / INSTRUMENT_RELATIVE
    dirty = [
        line
        for line in _git(tree, "status", "--porcelain", "--untracked-files=all").splitlines()
        if line
    ]
    return {
        "path": str(tree).replace("\\", "/"),
        "head": _git(tree, "rev-parse", "HEAD").strip(),
        "dirty_paths": dirty,
        "instrument_sha256": (
            hashlib.sha256(instrument.read_bytes()).hexdigest() if instrument.is_file() else None
        ),
    }


def build(current_path: Path, trees: dict[str, Path], faces: dict[str, Path]) -> dict[str, Any]:
    missing = sorted(set(trees) ^ set(faces))
    if missing:
        raise ValueError(f"每个面都要同时给检出与回执，缺: {missing}")
    current = load_face(current_path)
    current_head = _git(ROOT, "rev-parse", "HEAD").strip()
    described = {name: describe_tree(path) for name, path in trees.items()}
    loaded = {name: load_face(path) for name, path in faces.items()}

    rows = []
    for arm in sorted(current["digests"]):
        bound = current["revisions"].get(arm, "")
        prefix = bound[:9]
        digests = {name: loaded[name]["digests"].get(arm) for name in loaded}
        rows.append(
            {
                "arm": arm,
                "bound_revision": bound,
                "bound_face": prefix if prefix in loaded else None,
                "current_digest": current["digests"][arm],
                "digests": digests,
                "bound_digest_equals_current": (
                    digests[prefix] == current["digests"][arm] if prefix in loaded else None
                ),
                "every_face_equals_current": all(
                    digest == current["digests"][arm] for digest in digests.values()
                ),
            }
        )

    instrument_shas = {
        "current": current["instrument_sha256"],
        **{name: loaded[name]["instrument_sha256"] for name in loaded},
    }
    tree_shas = {name: described[name]["instrument_sha256"] for name in described}
    verdicts = {
        "current_face_fresh_process_match": current["all_fresh_match"] is True,
        "current_face_scoring_isolated": current["all_isolated"] is True,
        "every_historical_face_fresh_process_match": all(
            loaded[name]["all_fresh_match"] is True for name in loaded
        ),
        "every_historical_face_scoring_isolated": all(
            loaded[name]["all_isolated"] is True for name in loaded
        ),
        "one_instrument_across_all_faces": len(set(instrument_shas.values())) == 1
        and all(sha == instrument_shas["current"] for sha in tree_shas.values()),
        "every_arm_replayed_under_its_bound_revision": all(
            row["bound_digest_equals_current"] is True for row in rows
        ),
        "every_historical_digest_equals_current": all(
            row["every_face_equals_current"] is True for row in rows
        ),
        "historical_worktrees_clean_except_instrument": all(
            [line for line in described[name]["dirty_paths"] if INSTRUMENT_RELATIVE not in line]
            == []
            for name in described
        ),
        "arms_covered_15": len(rows) == 15,
    }
    return {
        "format": "taiji-n177-6-t1-historical-replay-v2",
        "date": "2026-10-29",
        "current_head": current_head,
        "current_face": current["path"],
        "instrument_sha256": instrument_shas,
        "trees": described,
        "faces": {
            name: {
                "report": loaded[name]["path"],
                "revision": described[name]["head"],
                "all_fresh_match": loaded[name]["all_fresh_match"],
                "all_isolated": loaded[name]["all_isolated"],
            }
            for name in loaded
        },
        "coverage": {
            "arms": len(rows),
            "historical_faces": sorted(loaded),
            "bound_revisions": sorted({row["bound_face"] for row in rows if row["bound_face"]}),
            "comparisons": sum(len(row["digests"]) for row in rows),
        },
        "rows": rows,
        "verdicts": verdicts,
        "scope": (
            "15 endpoints x 4 historical code identities + current tree; fixed first train "
            "episode only (holdout/query never read, no retraining); the digest covers the "
            "scorer's per-turn score payload only"
        ),
        "all_verdicts_ok": all(verdicts.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177.6 E0 历史复算对照（只读）")
    parser.add_argument("--current", type=Path, default=CURRENT)
    parser.add_argument(
        "--tree", action="append", default=None, metavar="NAME=PATH", help="历史检出；可重复"
    )
    parser.add_argument(
        "--face", action="append", default=None, metavar="NAME=REPORT", help="该检出的回执；可重复"
    )
    parser.add_argument("--out-report", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    pairs = args.tree or [f"{name}={path}" for name, path in DEFAULT_TREES.items()]
    trees = {item.split("=", 1)[0]: Path(item.split("=", 1)[1]) for item in pairs}
    if args.face:
        faces = {item.split("=", 1)[0]: Path(item.split("=", 1)[1]) for item in args.face}
    else:
        stamp = Path(tempfile.gettempdir())
        faces = {
            name: stamp / f"n177_6_e0_hist_{path.name.replace('seed-hist-', '')}.json"
            for name, path in trees.items()
        }

    report = build(args.current, trees, faces)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_report.with_name(args.out_report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.out_report)
    for key, value in report["verdicts"].items():
        print(f"{key}={value}")
    print(f"ALL_VERDICTS_OK={report['all_verdicts_ok']} out={args.out_report.name}")
    return 0 if report["all_verdicts_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
