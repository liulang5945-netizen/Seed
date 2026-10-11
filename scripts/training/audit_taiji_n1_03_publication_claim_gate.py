"""PLAN-N1-03 §7「发表资格前置」的机检件：三条前置各自的态＋"有没有人已经把它说出去"。

为什么是这一台
--------------
㊵-707／㊵-708 把 §3 四条判据都接上了执行体，但 §7 那三条前置今天仍靠**人读三件件再下一句结论**。
本件把那句结论变成读数：三条前置各自出**三态**（`satisfied`／`not_established`／`violated`），
再扫"状态面文档"里有没有人已写出 §6 第一条出口那句宣称。

设计上的三条硬规定
------------------
* **第三个态不许并进红或绿**：`J-N1c-3` 那件今天判 `no_carrier_outside_weight_artifacts`——那是
  "没有可读物"，不是"验证过"，也不是"坏了"。把它读成红会让人以为前置被否证，读成绿就是 ㊵-708
  那条假绿的反面。
* **只读已入库读数，且要核 provenance**：每件都必须带**生产它的仪器**当前的 `REPORT_FORMAT`，
  否则就是"仪器升了版而件还是旧的"——那属于结构性问题，响亮失败（rc=2）而不是判绿。
* **扫描面必须由 `--doc` 显式给**：判据原文本身就写着那几句宣称（预注册件里 2／1／2 处），
  把全文扫一遍会让门永远红在自己身上——这正是 [[gate-green-must-prove-it-scanned-the-diff]] 里
  "把违禁词抄进禁令会自指命中"那一族。所以本件只扫**状态面**（03／09／完成度件那类），
  并披露扫了几件几行、以及被否定标记挡掉几行。

零算力：只读件与 markdown。落件 LF，stdout 只打 ASCII。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FORMAT = "taiji-n1-03-publication-claim-gate-v1"
ROLE_PREFIX = "reports/taiji_n1_03_role_table_face_"
WRITE_ONLY_PREFIX = "reports/taiji_write_only_key_face_"
PROVENANCE_PREFIX = "reports/taiji_answer_source_provenance_"
#: 兼容旧的按件名引用（契约测用件名造夹具）；判定面走 `newest_artifact()`，不钉死日期。
ROLE_FACE = ROLE_PREFIX + "20261011.json"
WRITE_ONLY_FACE = WRITE_ONLY_PREFIX + "20261011.json"
PROVENANCE_FACE = PROVENANCE_PREFIX + "20261011.json"
PRODUCER_MODULES = {
    ROLE_PREFIX: "scripts/training/audit_taiji_n1_03_role_table_face.py",
    WRITE_ONLY_PREFIX: "scripts/training/audit_taiji_write_only_key_face.py",
    PROVENANCE_PREFIX: "scripts/training/audit_taiji_answer_source_provenance.py",
}
CLAIMS = ("已展开且有承载表", "N1 出口交付第三项已交付", "N1 已把回答链与角色通路做完")
NEGATIONS = ("不宣布", "不许", "不得", "仍不", "未兑现", "不能宣布", "过期")
STATES = ("satisfied", "not_established", "violated")
VERDICTS = (
    "publishable_and_claimed",
    "publishable_not_claimed",
    "pending_not_claimed",
    "claims_without_precondition",
)


def _load_producer_format(root: Path, rel: Path) -> str:
    module_path = root / PRODUCER_MODULES[rel]
    spec = importlib.util.spec_from_file_location(f"producer_{module_path.stem}", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.REPORT_FORMAT)


def newest_artifact(root: Path, prefix: str) -> Path:
    """取**当期最新**那张读数，而不是钉死某个日期。

    门读封存件就会永远拦不住现行（㊵-712 实测：仪器现面已报 1 枚载体，门还在按昨天的件报"无载体"）。
    文件名里的日期是单调的（`_YYYYMMDD`），同一天多次复算再按 mtime 取新（共享工作树下的既有规矩）。
    """

    found = sorted(root.glob(prefix + "*.json"), key=lambda q: (q.name, q.stat().st_mtime))
    if not found:
        raise ValueError(
            f"前置件缺失：没有任何 `{prefix}*.json`（缺件不许读成「没宣称就没问题」，那是要修的仪器）"
        )
    return found[-1]


def _read_artifact(root: Path, prefix: str) -> tuple[Path, dict[str, object]]:
    path = newest_artifact(root, prefix)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    declared = artifact.get("format")
    expected = _load_producer_format(root, prefix)
    if declared != expected:
        raise ValueError(
            f"{path.as_posix()} 的 format={declared!r} 与生产它的仪器现行 REPORT_FORMAT={expected!r} 不同源："
            "仪器已升版而读数未重算，本门拒绝拿旧件判今天的前置"
        )
    return path, artifact


def judge_role(artifact: dict[str, object]) -> dict[str, object]:
    verdict = artifact["j_n1c_2"]["verdict"]
    if verdict == "table_registered":
        state = "satisfied"
    elif verdict == "table_incomplete":
        state = "violated"
    else:
        raise ValueError(f"角色面判级词跑出词典：{verdict!r}")
    return {"member": "§7① 两张表逐格非空", "state": state, "evidence": verdict}


def judge_write_only(artifact: dict[str, object]) -> dict[str, object]:
    rows = artifact.get("rows", [])
    if not rows:
        return {
            "member": "§7② 每个被点名键的读写命中数",
            "state": "not_established",
            "evidence": "rows 为空＝这张面今天没有可点名的键",
        }
    offenders = []
    for row in rows:
        readers = sum(
            int(row.get(field, 0) or 0)
            for field in ("layer2_alias_reads", "layer3_action_reads", "unclassified_reads")
        )
        if int(row.get("writer_sites", 0) or 0) == 0 or readers == 0:
            offenders.append(f"{row.get('key')}＝w{row.get('writer_sites')}/r{readers}")
    state = "satisfied" if not offenders else "violated"
    return {
        "member": "§7② 每个被点名键的读写命中数",
        "state": state,
        "evidence": f"{len(rows)} 枚键，读写两处都有命中的 {len(rows) - len(offenders)} 枚"
        + (f"；缺侧 {offenders[:4]}" if offenders else ""),
    }


def judge_provenance(artifact: dict[str, object]) -> dict[str, object]:
    verdict = artifact["carrier_face"]["verdict"]
    if verdict == "consistent_within_carriers":
        state = "satisfied"
    elif verdict == "answer_source_mismatch":
        state = "violated"
    elif verdict == "no_carrier_outside_weight_artifacts":
        state = "not_established"
    else:
        raise ValueError(f"来源面判级词跑出词典：{verdict!r}")
    return {
        "member": "§7③ answer_source 的一致性复算",
        "state": state,
        "evidence": f"{verdict}（载体 {artifact['carrier_face']['carrier_count']} 枚）",
        #: 「绿」不许裸奔：这一枚复算证的只是**信封内部一致**，而三枚配方键由同一份 CLI 参数写出，
        #: 所以一致近乎由构造保证——载体到没到"被采纳的回答"那一层，必须由这句话一起出版。
        "scope_note": (
            "内部一致由同一份参数构造保证，且载体只到信封层（不证标签所指那条路真被走过）"
            if state == "satisfied"
            else None
        ),
    }


def scan_claims(root: Path, docs: list[str]) -> dict[str, object]:
    if not docs:
        raise ValueError("扫描面为空：没有 --doc 就没有「宣称面」，本门不许默认全仓")
    hits: list[str] = []
    skipped_negated = 0
    lines_scanned = 0
    for rel in docs:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            lines_scanned += 1
            if not any(claim in line for claim in CLAIMS):
                continue
            if any(mark in line for mark in NEGATIONS):
                skipped_negated += 1
                continue
            hits.append(f"{rel}:{number}")
    return {
        "docs_scanned": len(docs),
        "lines_scanned": lines_scanned,
        "claim_hits": hits,
        "lines_skipped_as_negated": skipped_negated,
    }


def adjudicate(root: Path, docs: list[str]) -> dict[str, object]:
    role_path, role = _read_artifact(root, ROLE_PREFIX)
    write_only_path, write_only = _read_artifact(root, WRITE_ONLY_PREFIX)
    provenance_path, provenance = _read_artifact(root, PROVENANCE_PREFIX)
    members = [judge_role(role), judge_write_only(write_only), judge_provenance(provenance)]
    for member in members:
        if member["state"] not in STATES:
            raise ValueError(f"前置态跑出词典：{member}")
    claims = scan_claims(root, docs)
    publishable = all(member["state"] == "satisfied" for member in members)
    claimed = bool(claims["claim_hits"])
    if claimed and publishable:
        verdict = "publishable_and_claimed"
    elif claimed and not publishable:
        verdict = "claims_without_precondition"
    elif not claimed and publishable:
        verdict = "publishable_not_claimed"
    else:
        verdict = "pending_not_claimed"
    if verdict not in VERDICTS:
        raise ValueError(f"总判级词跑出词典：{verdict}")
    return {
        "format": REPORT_FORMAT,
        "prereg": "plans/reference/PLAN-N1-03_chain_and_roles_prereg_20261011.md",
        "verdict": verdict,
        "publishable": publishable,
        "members": members,
        "state_counts": {
            state: sum(1 for member in members if member["state"] == state) for state in STATES
        },
        "claims": claims,
        #: 必须出版"这一判用的是哪一件"：封存件的名字一旦被硬编码，门就会永远按旧读数判今天。
        "artifacts_used": [
            role_path.relative_to(root).as_posix(),
            write_only_path.relative_to(root).as_posix(),
            provenance_path.relative_to(root).as_posix(),
        ],
        "what_this_does_not_prove": [
            "本门只守「三件同现且词表合法」与「有没有人已宣称」，不重判任何一条判据本身——"
            "判据的判定权在各仪器自己的读数件里",
            "`not_established` 是「没有可读物」，既不是红也不是绿；把它并进任一侧都会造出假结论",
            "否定标记（不宣布／不许／…）只用于**同行内**的引用豁免，它不校验跨行的语气，"
            "所以 `lines_skipped_as_negated` 必须一起出版",
        ],
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--doc",
        action="append",
        default=[],
        help="要扫的**状态面**文档（可重复）；预注册件不要给，它会自指命中",
    )
    parser.add_argument(
        "--out-report",
        default="reports/taiji_n1_03_publication_claim_gate_20261011.json",
        help="落盘路径（默认落 reports/；件已存在则拒覆盖 rc=2，要改须显式 --overwrite）",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    try:
        report = adjudicate(root, args.doc)
    except ValueError as exc:
        print(str(exc).encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr)
        return 2
    out = Path(args.out_report)
    if not out.is_absolute():
        out = root / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"verdict": report["verdict"], "members": report["state_counts"]}))
    #: 宣称而未满足前置 ⇒ 门红（rc=1）；其余三态是"如实报告"，不是失败。
    return 1 if report["verdict"] == "claims_without_precondition" else 0


if __name__ == "__main__":
    raise SystemExit(main())
