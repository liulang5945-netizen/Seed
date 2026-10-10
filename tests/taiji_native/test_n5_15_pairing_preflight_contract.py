"""保持侧配对预检（`check_taiji_n5_pairing_preflight.py`）的契约测。

这台仪器的全部主张是"在烧两臂 60k 之前，先说清同源核对**实际比的是哪几枚键**"，所以
四条分类支与两条拒绝支都必须实走，不许只验"能跑"：

* cap0 半：声明的 `eval_set_sha256` 与**重算**的盘上题集摘要一致 ⇒ `verified`；
  题集被换 ⇒ `mismatch`（仪器必须现算，不许只回显件里的字符串）；题集不在 ⇒ `manifest_absent`；
  两层都没有那枚键 ⇒ `declaration_absent`（㊵-553 那族：键住 `identity` 而只查顶层＝假缺席）。
* replay 半：四枚键齐全 ⇒ `content_hash_available`；只剩偏移与首件 ⇒
  `degraded_to_offset_and_first_item`（判读器不会拦，但"题集内容变了"拦不住——这句话必须能被出版）；
  一枚都没有 ⇒ `no_disclosure`。
* 件不在场 ⇒ rc=2 且出件 `preflight_failed`＝"没算"，不许读成"核对通过"。

真树那一支钉的是 2026-10-10 现读：cap0 `verified`、replay `degraded_to_offset_and_first_item`
——⇒ **双臂跑完之后，保持侧的 replay 半只能声称"偏移与首件相同"，不能声称"题集内容逐字相同"**。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "check_taiji_n5_pairing_preflight.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_15_pairing_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()

MANIFEST_BYTES = b'{"items": [{"id": "V001"}]}\n'


def _face_dir(tmp_path: Path, *, eval_sha: str | None, replay_keys: tuple[str, ...]) -> Path:
    root = tmp_path / "tree"
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "plans" / "manifests").mkdir(parents=True, exist_ok=True)
    manifest = root / "plans/manifests/cap0_eval_set_v2.json"
    manifest.write_bytes(MANIFEST_BYTES)
    actual = hashlib.sha256(MANIFEST_BYTES).hexdigest()
    cap0: dict[str, Any] = {"eval_set": "plans/manifests/cap0_eval_set_v2.json", "identity": {}}
    declared = actual if eval_sha is None else eval_sha
    cap0["identity"]["eval_set_sha256"] = declared
    (root / MODULE.CAP0_BEFORE).write_text(
        json.dumps(cap0, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    replay: dict[str, Any] = {"checkpoint": "checkpoints/base.pt"}
    values = {
        "items_sha256": "abc",
        "item_offset": 0,
        "first_item": "V001",
        "limit": 24,
    }
    for key in replay_keys:
        replay[key] = values[key]
    (root / MODULE.REPLAY_BEFORE).write_text(
        json.dumps(replay, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    return root


def test_cap0_half_verifies_against_a_recomputed_hash(tmp_path: Path) -> None:
    root = _face_dir(tmp_path, eval_sha=None, replay_keys=("item_offset", "first_item"))
    payload = MODULE.preflight(root)
    assert payload["cap0"]["cap0_identity"] == "verified", payload["cap0"]
    assert payload["cap0"]["declared_layer"] == "identity"
    #: 重算值必须出版，不然"verified"只是回显。
    assert (
        payload["cap0"]["recomputed_manifest_sha256"] == payload["cap0"]["declared_eval_set_sha256"]
    )


def test_a_swapped_manifest_is_a_mismatch_not_a_silent_pass(tmp_path: Path) -> None:
    root = _face_dir(tmp_path, eval_sha=None, replay_keys=("item_offset", "first_item"))
    (root / "plans/manifests/cap0_eval_set_v2.json").write_bytes(b'{"items": [{"id": "V002"}]}\n')
    payload = MODULE.preflight(root)
    assert payload["cap0"]["cap0_identity"] == "mismatch", payload["cap0"]
    assert (
        payload["cap0"]["recomputed_manifest_sha256"] != payload["cap0"]["declared_eval_set_sha256"]
    )


def test_missing_keys_are_typed_apart_from_zero_matches(tmp_path: Path) -> None:
    #: 三种"没东西"必须是三种读数：题集不在、声明不在、replay 一枚键都没有。
    root = _face_dir(tmp_path, eval_sha=None, replay_keys=())
    (root / "plans/manifests/cap0_eval_set_v2.json").unlink()
    payload = MODULE.preflight(root)
    assert payload["cap0"]["cap0_identity"] == "manifest_absent"
    assert payload["replay"]["replay_identity"] == "no_disclosure"
    assert payload["replay"]["keys_present"] == []

    other = _face_dir(tmp_path / "b", eval_sha=None, replay_keys=("first_item",))
    del_payload = json.loads((other / MODULE.CAP0_BEFORE).read_text(encoding="utf-8"))
    del_payload["identity"].pop("eval_set_sha256")
    (other / MODULE.CAP0_BEFORE).write_text(
        json.dumps(del_payload, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    assert MODULE.preflight(other)["cap0"]["cap0_identity"] == "declaration_absent"
    #: 只有首件（没有偏移）⇒ 判读器手里只剩一枚键，这比"退化"更弱，必须另分类。
    assert MODULE.preflight(other)["replay"]["replay_identity"] == "partial_disclosure"


def test_replay_half_distinguishes_a_content_hash_from_two_weak_keys(tmp_path: Path) -> None:
    strong = MODULE.preflight(
        _face_dir(
            tmp_path / "strong",
            eval_sha=None,
            replay_keys=("items_sha256", "item_offset", "first_item", "limit"),
        )
    )
    assert strong["replay"]["replay_identity"] == "content_hash_available"
    weak = MODULE.preflight(
        _face_dir(tmp_path / "weak", eval_sha=None, replay_keys=("item_offset", "first_item"))
    )
    assert weak["replay"]["replay_identity"] == "degraded_to_offset_and_first_item"


def test_the_real_faces_read_verified_and_degraded_today() -> None:
    payload = MODULE.preflight(REPO)
    assert payload["cap0"]["cap0_identity"] == "verified", payload["cap0"]
    assert payload["cap0"]["manifest_path"] == "plans/manifests/cap0_eval_set_v2.json"
    #: 现读钉值：committed 的 replay before 面**没有** `items_sha256` 也没有 `limit`。
    #: 出件方若开始出版那两枚键，本支会红 ⇒ 那时要连 05/08 一起改口，不是悄悄换判据。
    assert payload["replay"]["keys_present"] == ["item_offset", "first_item"], payload["replay"]
    assert payload["replay"]["replay_identity"] == "degraded_to_offset_and_first_item"
    assert payload["checkpoint_of_before_faces"]["cap0_checkpoint"].endswith(
        "seed_a31self_with_circuit.pt"
    )
    assert "pre-flight on the committed before-faces" in payload["reading_limit"]


def test_a_missing_before_face_is_preflight_failed_not_verified(tmp_path: Path) -> None:
    report = tmp_path / "out.json"
    rc = MODULE.main(["--root", str(tmp_path / "empty"), "--out-report", str(report)])
    assert rc == 2, rc
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "preflight_failed"
    assert "cap0" not in payload


def test_pointing_at_a_hashed_face_is_honoured(tmp_path: Path) -> None:
    """`--replay-before` 必须真的换得动面：DEBT-G79 出路①重产的那张件要靠它被指向。"""

    root = _face_dir(tmp_path, eval_sha=None, replay_keys=("item_offset", "first_item"))
    hashed = root / "reports" / "replay_hashed.json"
    hashed.write_text(
        json.dumps({"items_sha256": "abc", "item_offset": 0, "first_item": "V001", "limit": 24}),
        encoding="utf-8",
        newline="\n",
    )
    payload = MODULE.preflight(root, replay_face=hashed)
    assert payload["replay"]["replay_identity"] == "content_hash_available", payload["replay"]
    assert payload["faces_used"]["replay"] == str(hashed)
    #: 不指名时仍是旧件那张 ⇒ 退化读数不会因为换了参数就自己消失。
    assert (
        MODULE.preflight(root)["replay"]["replay_identity"] == "degraded_to_offset_and_first_item"
    )


def test_a_named_face_that_is_absent_fails_loudly(tmp_path: Path) -> None:
    report = tmp_path / "out.json"
    rc = MODULE.main(
        [
            "--root",
            str(REPO),
            "--replay-before",
            "reports/definitely-not-a-face.json",
            "--out-report",
            str(report),
        ]
    )
    assert rc == 2, rc
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "preflight_failed"


def test_help_lists_all_four_flags(monkeypatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(sys.stdout, "write", lambda text: captured.append(text) or len(text))
    try:
        MODULE.main(["--help"])
    except SystemExit as exit_info:
        assert exit_info.code == 0, exit_info.code
    else:
        raise AssertionError("--help 没有按惯例退出 0")
    joined = "".join(captured)
    for flag in ("--out-report", "--root", "--cap0-before", "--replay-before"):
        assert flag in joined, (flag, joined[:200])
