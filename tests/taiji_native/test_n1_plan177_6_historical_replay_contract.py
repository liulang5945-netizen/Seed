"""E0 历史复算对照器的判据册：三条判据都要**能为假**。

用合成回执造两副面：
①全等 ⇒ 各判据为真；
②某一臂在一个面上换了指纹、或某面换了仪器字节、或某臂绑定的 revision 没有对应检出
   ⇒ 对应当判据必须掉下来（否则"跨面相等"是恒真式）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = (
    PROJECT_ROOT / "scripts" / "training" / "compare_taiji_n1_plan177_6_t1_historical_replay.py"
)
ARMS = tuple(f"arm_{index}__seed_0" for index in range(15))


def _module():
    spec = importlib.util.spec_from_file_location("n177_6_historical_replay", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instrument():
    return _module()


def _receipts(path: Path, *, digest_shift: dict[str, str] | None = None, sha: str = "s" * 64):
    shift = digest_shift or {}
    payload = {
        "format": "taiji-n177-6-t1-endpoint-receipts-v2",
        "instrument_sha256": sha,
        "revisions": {arm: "aaaaaaaaa" for arm in ARMS},
        "receipts": [
            {
                "arm": arm,
                "fingerprint": {
                    "digest": shift.get(arm, f"digest-{arm}"),
                    "scoring_isolation": {"unchanged": True},
                },
                "fresh_process": {"matches_in_process": True},
            }
            for arm in ARMS
        ],
        "all_fresh_match": True,
        "all_isolated": True,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def _tree(path: Path, instrument: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.email=a@b",
            "-c",
            "user.name=a",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    target = path / "scripts" / "training" / "verify_taiji_n1_plan177_6_t1_endpoint_receipts.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(instrument.read_bytes())
    return path


@pytest.fixture()
def faces(instrument, tmp_path):
    instrument_file = PROJECT_ROOT / instrument.INSTRUMENT_RELATIVE
    sha = hashlib.sha256(instrument_file.read_bytes()).hexdigest()
    tree = _tree(tmp_path / "seed-hist-aaaaaaaaa", instrument_file)
    current = _receipts(tmp_path / "current.json", sha=sha)
    face = _receipts(tmp_path / "face.json", sha=sha)
    return instrument, current, {"aaaaaaaaa": tree}, {"aaaaaaaaa": face}


def test_identical_faces_hold_every_verdict(faces) -> None:
    instrument, current, trees, face_map = faces
    report = instrument.build(current, trees, face_map)
    assert report["all_verdicts_ok"] is True, report["verdicts"]
    assert report["verdicts"]["every_arm_replayed_under_its_bound_revision"] is True
    assert report["coverage"]["comparisons"] == len(ARMS)


def test_one_different_digest_breaks_the_comparison(faces) -> None:
    instrument, current, trees, face_map = faces
    _receipts(face_map["aaaaaaaaa"], digest_shift={ARMS[0]: "drifted"})
    report = instrument.build(current, trees, face_map)
    assert report["verdicts"]["every_historical_digest_equals_current"] is False
    assert report["verdicts"]["every_arm_replayed_under_its_bound_revision"] is False
    assert report["all_verdicts_ok"] is False


def test_a_different_instrument_byte_breaks_identity(faces) -> None:
    instrument, current, trees, face_map = faces
    _receipts(face_map["aaaaaaaaa"], sha="t" * 64)
    report = instrument.build(current, trees, face_map)
    assert report["verdicts"]["one_instrument_across_all_faces"] is False


def test_a_bound_revision_without_a_face_is_not_excused(faces) -> None:
    instrument, current, trees, face_map = faces
    payload = json.loads(current.read_text(encoding="utf-8"))
    payload["revisions"][ARMS[0]] = "bbbbbbbbb"
    current.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    report = instrument.build(current, trees, face_map)
    row = next(item for item in report["rows"] if item["arm"] == ARMS[0])
    assert row["bound_face"] is None
    assert row["bound_digest_equals_current"] is None
    assert report["verdicts"]["every_arm_replayed_under_its_bound_revision"] is False


def test_a_dirty_worktree_is_reported(faces) -> None:
    instrument, current, trees, face_map = faces
    (trees["aaaaaaaaa"] / "stray.txt").write_text("x\n", encoding="utf-8", newline="\n")
    report = instrument.build(current, trees, face_map)
    assert report["verdicts"]["historical_worktrees_clean_except_instrument"] is False


def test_mismatched_face_names_fail_closed(faces) -> None:
    instrument, current, trees, _ = faces
    with pytest.raises(ValueError, match="缺"):
        instrument.build(current, trees, {})
