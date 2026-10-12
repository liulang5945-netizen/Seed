"""DEBT-G104 批次 B 的装机面双臂普查：盘上信封在现行来源枚数与退回一枚的对照下各自能否载入。

这块要答的是 owner 二裁⑴ 那句「迁移载入是乙路的全部内容」——㊵-727 用一次性 `.tmp_probe_727*.py`
现读过一次（两臂同为 12 载入／1 拒载，唯一拒载那枚是 DEBT-G47），但探针跑完即删，盘上没有能重取
这组读数的东西。本仪器把那条链钉下来：

面（face）＝盘上 `checkpoints/*.pt`（按大小升序），入库 `.pt` 另存一枚计数（本仓现在为 0 枚）——两个数各自点名，
             不混成一个"件数"。
入口（entry）＝训练栈实际走的那条 `Seed.from_checkpoint(envelope).architecture.memory`；
             另跑一条**反例臂**：把同一枚信封喂给平面入口 `Taiji.from_checkpoint`，它必然报错，
             那一臂只用于证明"点错入口的整片拒绝不是读数"（㊵-727 ② 的教训）。
臂（arm）＝治疗臂用现行来源枚数，对照臂把 `EPISODIC_PROVENANCE_KINDS` 与 `EpisodicField.PROVENANCE_KINDS`
          同时退回掉最后一枚——即这些件被写出时的那档。

零产品码、只读盘上件；不更新任何权重、不写回任何 `.pt`。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import torch  # noqa: E402
from read_taiji_envelope_recipe_face import (  # noqa: E402
    DEFAULT_ROOTS,
    ROW_LIST_CAP,
    enumerate_face,
    repo_head,
    select_batch,
)

REPORT_FORMAT = "taiji-disk-load-face-two-arms-v1"
LEDGER_ORIGIN = "㊵-727"
DEBT = "DEBT-G104"
ENTRY_CONTROL_LIMIT = 4


def tracked_pt_count(root: Path) -> int:
    proc = subprocess.run(
        ["git", "ls-files", "*.pt"],
        capture_output=True,
        cwd=root,
    )
    if proc.returncode != 0:
        return -1
    return len([line for line in proc.stdout.decode("utf-8", "replace").splitlines() if line])


def _apply_vocabulary(kinds: tuple[str, ...]) -> None:
    import taiji.config as config_module
    from taiji.memory import EpisodicField

    config_module.EPISODIC_PROVENANCE_KINDS = kinds
    EpisodicField.PROVENANCE_KINDS = kinds


def _default_vocabulary() -> tuple[str, ...]:
    from taiji.config import EPISODIC_PROVENANCE_KINDS

    return tuple(EPISODIC_PROVENANCE_KINDS)


def _read_name(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_one(path: Path) -> Any:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _run_arm(envelopes: dict[str, Any], kinds: tuple[str, ...]) -> dict[str, object]:
    from seed.model import Seed

    rows = []
    reasons: dict[str, int] = {}
    loaded = 0
    inherited = 0
    for name, envelope in envelopes.items():
        try:
            model = Seed.from_checkpoint(envelope)
        except Exception as exc:  # noqa: BLE001 - 拒载原因本身就是要出版的东西
            key = type(exc).__name__ + "|" + str(exc).replace("\n", " ")[:70]
            reasons[key] = reasons.get(key, 0) + 1
            rows.append({"file": name, "loaded": False, "reason": key})
            continue
        field = model.architecture.memory
        width = int(field.provenance_encoder.in_features)
        rows.append(
            {
                "file": name,
                "loaded": True,
                "encoder_width": width,
                "readout_rows": int(field.provenance_readout.out_features),
                "vocabulary": len(field.PROVENANCE_KINDS),
                "receptors_inherited": bool(field.receptors_inherited),
            }
        )
        loaded += 1
        inherited += int(bool(field.receptors_inherited))
    return {
        "kinds": list(kinds),
        "attempted": len(rows),
        "loaded": loaded,
        "refused": len(rows) - loaded,
        "receptors_inherited_count": inherited,
        "refusal_reason_counts": reasons,
        "refusing_files": [row["file"] for row in rows if not row["loaded"]],
        "rows": rows,
        "rows_total": len(rows),
        "rows_listed": len(rows[:ROW_LIST_CAP]),
    }


def _entry_control(envelopes: dict[str, Any]) -> dict[str, object]:
    """把同一批信封喂给**平面**检查点入口：整片报错，证明那条取路不产出读数。"""

    from taiji.model import Taiji

    errors: dict[str, int] = {}
    attempted = 0
    for name, envelope in list(envelopes.items())[:ENTRY_CONTROL_LIMIT]:
        attempted += 1
        try:
            Taiji.from_checkpoint(envelope)
        except Exception as exc:  # noqa: BLE001
            key = type(exc).__name__
            errors[key] = errors.get(key, 0) + 1
            continue
        errors["UNEXPECTED_LOAD"] = errors.get("UNEXPECTED_LOAD", 0) + 1
    return {
        "attempted": attempted,
        "error_counts": errors,
        "meaning": "反例臂：入口点错时的整片拒绝不是装机面的读数（㊵-727 ②）",
    }


def read_face(
    root: Path,
    roots: list[str],
    max_files: int,
    max_mb: int,
    only: list[str],
) -> dict[str, object]:
    from taiji.config import EPISODIC_PROVENANCE_KINDS

    default_kinds = tuple(EPISODIC_PROVENANCE_KINDS)
    points = enumerate_face(root, roots)
    chosen, skipped_by_size, skipped_by_count = select_batch(points, max_files, max_mb, only)
    if not chosen:
        raise ValueError(
            f"面上没有任何 `.pt` 可开（roots={roots}，盘上计数={len(points)}）——"
            "空面不许出一份看起来像结论的件"
        )
    envelopes = {_read_name(root, path): _load_one(path) for path in chosen}
    control_kinds = default_kinds[:-1]
    original = default_kinds
    try:
        _apply_vocabulary(default_kinds)
        treated = _run_arm(envelopes, default_kinds)
        _apply_vocabulary(control_kinds)
        control = _run_arm(envelopes, control_kinds)
    finally:
        _apply_vocabulary(original)

    both_refusing = sorted(set(treated["refusing_files"]) & set(control["refusing_files"]))
    treated_only = sorted(set(treated["refusing_files"]) - set(control["refusing_files"]))
    face_total = len(points)
    selected = len(chosen)
    verdict = (
        "full_face_two_arm_pair"
        if selected == face_total and treated["attempted"] == selected
        else "partial_selected_lt_face"
    )
    return {
        "format": REPORT_FORMAT,
        "debt": DEBT,
        "ledger_origin": LEDGER_ORIGIN,
        "git_head": repo_head(root),
        "face": {
            "roots_scanned": list(roots),
            "disk_pt_total": face_total,
            "tracked_pt_in_git": tracked_pt_count(root),
            "selected": selected,
            "skipped_by_size": skipped_by_size,
            "skipped_by_count": skipped_by_count,
            "selection_order": "size_ascending",
            "batch_mode": "only_named" if only else "budget",
        },
        "verdict": verdict,
        "arms": {"treated": treated, "control": control},
        "pairing": {
            "refused_on_both_arms": both_refusing,
            "refused_only_on_treated_arm": treated_only,
            "load_counts_equal": treated["loaded"] == control["loaded"],
        },
        "entry_point_control": _entry_control(envelopes),
        "detail_row_cap": ROW_LIST_CAP,
        "what_this_does_not_prove": [
            "本件只证『能装／不能装』与两臂是否同形，不证载入后的权重逐位中性（那由 J-N5o-4 的进程内件证过）",
            "对照臂是进程内把枚举退回一枚，不是『用改版前的代码跑』——基板抽样与运行链都相同",
            "面上件多为未入库的盘上产物：tracked_pt_in_git 这一枚计数必须与 disk_pt_total 一起读，"
            "少了它就把『盘上有 13 枚』误当成『装机面有 13 枚在册』",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="盘上信封载入面的双臂普查（治疗臂＝现行来源枚数，对照臂＝退回一枚）"
    )
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="扫描根（契约测指向临时面）")
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS), help="面内子目录")
    parser.add_argument("--max-files", type=int, default=64, help="一次最多开几枚（按大小升序）")
    parser.add_argument("--max-mb", type=int, default=1024, help="本批体积上限（兆字节）")
    parser.add_argument("--only", action="append", default=[], help="点名枚（不存在即报错）")
    parser.add_argument(
        "--out-report", required=True, help="落盘路径（件已存在则拒覆盖，除非 --overwrite）"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report = read_face(root, args.roots, args.max_files, args.max_mb, args.only)

    out = Path(args.out_report)
    if not out.is_absolute():
        out = root / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    treated = report["arms"]["treated"]
    control = report["arms"]["control"]
    summary = {
        "verdict": report["verdict"],
        "disk_pt_total": report["face"]["disk_pt_total"],
        "selected": report["face"]["selected"],
        "treated_loaded_refused": [treated["loaded"], treated["refused"]],
        "control_loaded_refused": [control["loaded"], control["refused"]],
        "refused_on_both_arms": len(report["pairing"]["refused_on_both_arms"]),
        "refused_only_on_treated": len(report["pairing"]["refused_only_on_treated_arm"]),
        "entry_control_errors": report["entry_point_control"]["error_counts"],
        "out": out.as_posix(),
    }
    sys.stdout.buffer.write(
        json.dumps(summary, ensure_ascii=True).encode("ascii") + b"\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
