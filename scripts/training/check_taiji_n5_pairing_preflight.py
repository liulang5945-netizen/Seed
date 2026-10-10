"""保持侧配对（PLAN-N5-04 §3 G-N5d-1）的**起跑前预检**（零算力，只读件与题集）。

**为什么要在训练之前跑它**：`adjudicate_taiji_n2_04_retention_pair.py` 的同源核对是在**四张面都齐了
之后**才执行的。如果 before 面本身带不出可比的摘要，最坏的结局不是红，而是"两臂各 60k 符号跑完
（单臂实测 334.63762799999676～473.23540269979276 秒）之后，判读器在 `G_N2c_3_*` 上 rc=2"。
本器把那次拒绝**提前**成一份读数。

两半各自独立出版，且都不许被合并成一句"同源"：

* **cap0 半**：读 `reports/taiji_n2_cap0_before_20261008.json` 声明的题集路径与 `eval_set_sha256`
  （该键在 `identity` 子字典里，不在顶层——㊵-553 那族的取法），**重算**盘上题集的 sha256 并比较。
  相等 ⇒ `cap0_identity=verified`；不等或题集不在 ⇒ `mismatch`／`manifest_absent`。
* **replay 半**：把判读器 `_replay_items_disclosure()` 实际会用的那四枚键
  （`items_sha256`／`item_offset`／`first_item`／`limit`）逐枚查 before 面里**有没有**，
  然后据实分类：有 `items_sha256` ⇒ `content_hash_available`；只有偏移与首件 ⇒
  `degraded_to_offset_and_first_item`（这时"题集内容变了"是**拦不住**的，必须说出来）。

**rc 语义**：`0`＝预检跑完（结论可以是"退化"）；`2`＝预检本身没法跑（件缺、JSON 坏）＝"没算"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAP0_BEFORE = Path("reports/taiji_n2_cap0_before_20261008.json")
REPLAY_BEFORE = Path("reports/taiji_n2_replay24_before_20261008.json")
#: 面可指名 ⇒ 重产出的那一张（DEBT-G79 出路①）能被预检指向，而不必先覆盖旧件路径。
#: 判读器 `_replay_items_disclosure()` 的那四枚键（同源抄法＝按名字列出来，不 import 私有函数）。
REPLAY_KEYS = ("items_sha256", "item_offset", "first_item", "limit")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _declared_eval_sha(payload: dict[str, Any]) -> tuple[str | None, str]:
    """`eval_set_sha256` 可能住顶层也可能住 `identity` ⇒ 取到哪一层要一起出版（㊵-553 的教训）。"""

    if "eval_set_sha256" in payload:
        return str(payload["eval_set_sha256"]), "top_level"
    identity = payload.get("identity")
    if isinstance(identity, dict) and "eval_set_sha256" in identity:
        return str(identity["eval_set_sha256"]), "identity"
    return None, "absent"


def preflight(
    root: Path,
    *,
    cap0_face: Path = CAP0_BEFORE,
    replay_face: Path = REPLAY_BEFORE,
) -> dict[str, Any]:
    cap0_path = cap0_face if cap0_face.is_absolute() else root / cap0_face
    replay_path = replay_face if replay_face.is_absolute() else root / replay_face
    if not cap0_path.is_file() or not replay_path.is_file():
        raise FileNotFoundError(
            f"missing before face: {cap0_path if not cap0_path.is_file() else replay_path}"
        )

    cap0 = _read(cap0_path)
    declared, layer = _declared_eval_sha(cap0)
    raw_manifest = str(cap0.get("eval_set", "")).replace("\\", "/")
    manifest = root / raw_manifest
    cap0_out: dict[str, Any] = {
        "declared_eval_set_sha256": declared,
        "declared_layer": layer,
        "manifest_path": raw_manifest,
        "manifest_present": manifest.is_file(),
    }
    if declared is None:
        cap0_out["cap0_identity"] = "declaration_absent"
    elif not manifest.is_file():
        cap0_out["cap0_identity"] = "manifest_absent"
    else:
        actual = _sha256(manifest)
        cap0_out["recomputed_manifest_sha256"] = actual
        cap0_out["cap0_identity"] = "verified" if actual == declared else "mismatch"

    replay = _read(replay_path)
    present = [key for key in REPLAY_KEYS if key in replay]
    if "items_sha256" in present:
        replay_outcome = "content_hash_available"
    elif {"item_offset", "first_item"} <= set(present):
        #: 判读器不会 rc=2（它有两枚可比键），但"题集内容变了"这件事**拦不住**——必须说出来。
        replay_outcome = "degraded_to_offset_and_first_item"
    elif present:
        replay_outcome = "partial_disclosure"
    else:
        replay_outcome = "no_disclosure"

    return {
        "format": "taiji-n5-pairing-preflight-v1",
        "prereg": "PLAN-N5-04 §3 G-N5d-1",
        #: 预检指向哪两张件必须随件出版——"退化"与"哈希齐"的差别就在这两张件上。
        "faces_used": {"cap0": str(cap0_path), "replay": str(replay_path)},
        "cap0": cap0_out,
        "replay": {
            "keys_present": present,
            "keys_the_adjudicator_compares": list(REPLAY_KEYS),
            "replay_identity": replay_outcome,
        },
        "checkpoint_of_before_faces": {
            "cap0_checkpoint": cap0.get("checkpoint"),
            "replay_checkpoint": replay.get("checkpoint"),
        },
        "reading_limit": (
            "this is a pre-flight on the committed before-faces only; it does not measure whether "
            "the after-faces will be produced successfully"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="retention-pair pre-flight (static)")
    parser.add_argument("--out-report", type=Path, required=True, help="where to write the JSON")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="tree holding the faces")
    parser.add_argument(
        "--cap0-before",
        type=Path,
        default=CAP0_BEFORE,
        help="cap0 before face (relative to --root, or absolute)",
    )
    parser.add_argument(
        "--replay-before",
        type=Path,
        default=REPLAY_BEFORE,
        help="replay before face (relative to --root, or absolute)",
    )
    args = parser.parse_args(argv)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = preflight(args.root, cap0_face=args.cap0_before, replay_face=args.replay_before)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        args.out_report.write_text(
            json.dumps(
                {
                    "format": "taiji-n5-pairing-preflight-v1",
                    "status": "preflight_failed",
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return 2
    args.out_report.write_text(
        json.dumps({"status": "ok", **payload}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
