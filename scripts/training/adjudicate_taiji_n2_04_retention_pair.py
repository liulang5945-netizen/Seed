"""PLAN-N2-04 §5 的 **A 档**：保持集前后对表与分离机检的只读仪器（零训练、零产品码改动）。

为什么先做这一档（缺陷本体）：N2-04 把"保持集"冻成了成员＋判据＋六条守卫，但 owner 2026-10-08 第七次弹窗裁的是
**乙层**（"只在巩固前后对表，不进产品码"，㊵-529）⇒ 本件不试图减少代价，只回答"这一剂的代价到底在不在"。
它能为 false 的两面都必须走：七列全不跌 ⇒ `retention_holds`；任一列跌 ≥1 ⇒ `cost_persists`（负结果照常出版）。

三条硬规矩：
* **取数不重抄**：七列的取法逐字复用 `count_taiji_n2_faces.py` 的 `_cap0_tally`／`_replay_arms`
  与 `count_taiji_n2_03_attribution.py` 的 `DROP_COLUMNS`；`overlap` 复用 `audit_taiji_n3a_data_face.py` 的 `_disjointness`；
* **面不同源就拒判**（G-N2c-3）：两侧题集披露缺失或不一致 ⇒ rc=2，**不许**用"题集看起来一样"代答；
* **分离取不到数就 `unverified`**（G-N2c-4）：拿不到 `overlap` 时整件不判，而不是当成"大概不相交"。

收益侧（J-N2c-收益）不住在这台仪器里——它沿用 N2-02 那台的同窗读数，本件把那一侧标成 `not_judged_here`，
免得读者把"保持侧绿"读成"整条合取成立"。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_sibling(name: str) -> Any:
    path = PROJECT_ROOT / "scripts" / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FACES = _load_sibling("count_taiji_n2_faces")
ATTRIB = _load_sibling("count_taiji_n2_03_attribution")
N3A = _load_sibling("audit_taiji_n3a_data_face")

#: 保持侧的七列＝N2-03 已冻的那七列（本件**不新列**，也不允许事后加列）。
SEVEN_COLUMNS: tuple[tuple[str, str], ...] = ATTRIB.DROP_COLUMNS


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _column_table(cap0: dict[str, Any], replay: dict[str, Any]) -> dict[str, int]:
    """按七列的键名取现成计数——**取法逐字复用 N2-02 那台仪器**，本件不自己数第二遍。

    * `cap0:E` ⇒ `_cap0_tally(...)["per_dimension"]["E"]["correct"]`（严格命中分子）；
    * `replay_strict_hits:<惩罚档>` ⇒ `_replay_arms(...)["<档>"]["strict_hits"]`；
    * `replay_well_formed:<惩罚档>` ⇒ `_replay_arms(...)["<档>"]["well_formed_texts"]`。
    缺任一列/档 ⇒ `KeyError`，由调用方转成响亮拒绝（不许当 0 读）。
    """

    tally = FACES._cap0_tally(cap0)
    arms = FACES._replay_arms(replay)
    per_dimension = tally.get("per_dimension") or {}
    #: `_cap0_tally` 对**缺块**是宽容的（`dimensions.get(key) or {}` ⇒ 算出 correct=0），
    #: 而"缺一维"绝不能被读成"这一维的能力是 0"（㊵-487 同族：装载失败/缺块不是零分）。
    #: ⇒ 先验原始件里那一维的 `tally` 在不在，再取 `_cap0_tally` 的数（同一个来源，不重抄算法）。
    raw_dimensions = cap0.get("dimensions") or {}
    table: dict[str, int] = {}
    for source, key in SEVEN_COLUMNS:
        if source == "cap0":
            raw_block = raw_dimensions.get(key)
            if not isinstance(raw_block, dict) or not isinstance(raw_block.get("tally"), dict):
                raise KeyError(
                    f"cap0 件里没有维度 {key!r} 的 tally ⇒ 缺块不许读成零分（G-N2c-3 同源纪律）"
                )
            block = per_dimension.get(key)
            if block is None:
                raise KeyError(f"cap0 缺维度 {key!r}（或被点名维度不在 DRIVEN_DIMENSIONS 里）")
            table[f"{source}:{key}"] = int(block["correct"])
        elif source == "replay_strict_hits":
            arm = arms.get(key)
            if arm is None or arm.get("strict_hits") is None:
                raise KeyError(f"replay 缺惩罚档 {key!r} 的 strict_hits")
            table[f"{source}:{key}"] = int(arm["strict_hits"])
        elif source == "replay_well_formed":
            arm = arms.get(key)
            if arm is None or arm.get("well_formed_texts") is None:
                raise KeyError(f"replay 缺惩罚档 {key!r} 的 well_formed_texts")
            table[f"{source}:{key}"] = int(arm["well_formed_texts"])
        else:  # pragma: no cover - 七列全集已覆盖，出现即说明有人往冻列里加了新源
            raise KeyError(f"未知列源 {source!r}")
    return table


def _same_source_disclosure(
    cap0_before: dict[str, Any], cap0_after: dict[str, Any]
) -> dict[str, Any]:
    keys = ("eval_set_sha256", "source_sha256", "manifest_sha256")
    found = {
        k: (cap0_before.get(k), cap0_after.get(k))
        for k in keys
        if k in cap0_before or k in cap0_after
    }
    if not found:
        return {"status": "missing", "checked": [], "conflicts": []}
    conflicts = [k for k, (a, b) in found.items() if a != b]
    return {
        "status": "ok" if not conflicts else "conflict",
        "checked": sorted(found),
        "conflicts": conflicts,
    }


def _replay_items_disclosure(
    replay_before: dict[str, Any], replay_after: dict[str, Any]
) -> dict[str, Any]:
    keys = ("items_sha256", "item_offset", "first_item", "limit")
    found = {
        k: (replay_before.get(k), replay_after.get(k))
        for k in keys
        if k in replay_before or k in replay_after
    }
    if not found:
        return {"status": "missing", "checked": [], "conflicts": []}
    conflicts = [k for k, (a, b) in found.items() if a != b]
    return {
        "status": "ok" if not conflicts else "conflict",
        "checked": sorted(found),
        "conflicts": conflicts,
    }


def judge(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    payload: dict[str, Any] = {
        "format": "taiji-n2-04-retention-pair-v1",
        "prereg": "PLAN-N2-04 §2/§3/§4",
    }
    paths = {
        "cap0_before": _resolve(args.cap0_before),
        "cap0_after": _resolve(args.cap0_after),
        "replay_before": _resolve(args.replay_before),
        "replay_after": _resolve(args.replay_after),
    }
    missing = [name for name, p in paths.items() if not p.is_file()]
    for name, p in paths.items():
        payload[f"{name}_path"] = str(p)
        payload[f"{name}_sha256"] = (
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        )
    if missing:
        #: 缺件不是"零跌幅"——响亮拒绝。
        payload["status"] = "missing_inputs"
        payload["missing"] = sorted(missing)
        payload["verdict"] = "not_judged"
        return payload, 2

    cap0_before = _read(paths["cap0_before"])
    cap0_after = _read(paths["cap0_after"])
    replay_before = _read(paths["replay_before"])
    replay_after = _read(paths["replay_after"])

    guards: dict[str, Any] = {}
    guards["G_N2c_3_cap0_same_source"] = _same_source_disclosure(cap0_before, cap0_after)
    guards["G_N2c_3_replay_same_source"] = _replay_items_disclosure(replay_before, replay_after)

    try:
        before = _column_table(cap0_before, replay_before)
        after = _column_table(cap0_after, replay_after)
    except KeyError as error:
        payload["status"] = "column_layout_unreadable"
        payload["rejection"] = str(error)
        payload["verdict"] = "not_judged"
        payload["guards"] = guards
        return payload, 2

    per_column = {
        name: {
            "before": before[name],
            "after": after[name],
            "delta": after[name] - before[name],
            "dropped": after[name] < before[name],
        }
        for name in before
    }
    dropped = sorted(name for name, row in per_column.items() if row["dropped"])

    #: G-N2c-4 分离机检：巩固材料 vs 保持材料的字节级交集必须为空。
    corpus = [_resolve(p) for p in (args.consolidation_corpus or [])]
    windows: list[bytes] = []
    manifest_path = _resolve(args.retention_manifest)
    rejection: str | None = None
    if not manifest_path.is_file():
        rejection = f"retention manifest 缺件：{manifest_path}"
    elif not corpus:
        rejection = "没点名巩固材料 ⇒ overlap 无从算起"
    else:
        for bad in (p for p in corpus if not p.is_file()):
            rejection = f"巩固材料缺件：{bad}"
    if rejection is None:
        manifest = _read(manifest_path)
        items = manifest.get("items") if isinstance(manifest, dict) else None
        if not isinstance(items, list) or not items:
            rejection = "manifest 里没有 items 数组 ⇒ 保持侧窗口取不到"
        else:
            for item in items:
                text = item.get("text") if isinstance(item, dict) else item
                if isinstance(text, str) and text:
                    windows.append(text.encode("utf-8"))
            if not windows:
                rejection = "manifest 的 items 里没有可用文本 ⇒ 分离机检取不到数"

    if rejection is not None:
        guards["G_N2c_4_disjointness"] = {"status": "unverified", "reason": rejection}
        payload["status"] = "separation_unverified"
        payload["verdict"] = "not_judged"
        payload["guards"] = guards
        #: 取不到 overlap ⇒ 整件不判（不许把"没算"读成"没重叠"）。
        return payload, 2

    overlap = N3A._disjointness(corpus, windows)
    guards["G_N2c_4_disjointness"] = {
        "status": "measured",
        "retention_windows": len(windows),
        "result": overlap,
    }
    #: `_disjointness` 的返回里 `windows_found_in_corpus` 才是"交集"指示（它按窗口逐条数命中）。
    found = overlap.get("windows_found_in_corpus")
    intersecting = bool(found) if isinstance(found, int) else None

    for name in ("G_N2c_3_cap0_same_source", "G_N2c_3_replay_same_source"):
        if guards[name]["status"] == "conflict":
            payload["status"] = "faces_not_same_source"
            payload["verdict"] = "not_judged"
            payload["guards"] = guards
            return payload, 2
    if intersecting:
        #: 保持集与巩固材料有字节交集 ⇒ 这条"保持"是被泄露的题做出来的，整件作废。
        payload["status"] = "retention_not_disjoint"
        payload["verdict"] = "not_judged"
        payload["guards"] = guards
        return payload, 2

    payload["status"] = "ok"
    payload["per_column"] = per_column
    payload["criteria"] = {
        "columns_compared": len(per_column),
        "columns_dropped": dropped,
        "J_N2c_retention": "retention_holds" if not dropped else "cost_persists",
        "J_N2c_gain": "not_judged_here",
        "conjunction_note": "收益侧沿用 N2-02 同窗判读器，本件不代答；两侧都成立才谈『保持集有效』",
    }
    payload["verdict"] = payload["criteria"]["J_N2c_retention"]
    payload["guards"] = guards
    return payload, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="N2-04 A 档：保持集前后对表＋分离机检（只读、零训练）"
    )
    parser.add_argument("--cap0-before", required=True)
    parser.add_argument("--cap0-after", required=True)
    parser.add_argument("--replay-before", required=True)
    parser.add_argument("--replay-after", required=True)
    parser.add_argument("--consolidation-corpus", action="append", default=[])
    parser.add_argument("--retention-manifest", default="plans/manifests/cap0_eval_set_v2.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    payload, rc = judge(args)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    #: Windows 控制台是 GBK ⇒ 只打 ASCII 摘要行，中文正文全在 --out 里。
    crit = payload.get("criteria") or {}
    print(
        "STATUS",
        payload["status"],
        "VERDICT",
        payload["verdict"],
        "DROPPED",
        json.dumps(crit.get("columns_dropped", [])),
        "COMPARED",
        crit.get("columns_compared"),
        "RC",
        rc,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
