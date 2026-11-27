"""只读分批打开 `.pt` 信封，把 `metadata` 里的来源记录复算一遍（owner 四裁④＝「授权只读打开，分批读」）。

为什么要这一台
--------------
`J-N1c-3`（来源记录不许反写）的 JSON 面今天只有**一枚**载体（`output/seed_corpus_smoke_progress_exit.json`，
㊵-712），因为配方键是 ㊵-709 才加进退出件的；而**真正的载体一直住在 `.pt` 信封的 `metadata` 里**。
本件按只读方式把它取出来，回答一个此前只能猜的问题：**有多少枚信封确实带着来源记录**。

现取的两条事实（决定了判级词的形状，2026-10-11 实测）
----------------------------------------------------
* `output/n5_10_arms/treated/checkpoint.pt`（21.9 MB）的 `metadata` **已经**带
  `answer_source=corpus`／`answer_chunking=stream`／`self_answers_path=None`——加宽之前的跑就在写；
* `checkpoints/seed_beta.pt`（4.1 MB，产品默认基座）的 `metadata` 只有
  `corpus_fingerprint`／`profile`／`saved_at_utc`／`tick`／`trainer` ⇒ **无来源记录**。
  「无记录」不是「反写」，所以它单独一支判级词，不许并进 violated，也不许因为「没矛盾」被判成一致。

批次与覆盖（这把尺的判别力全在这里）
----------------------------------
两面现读：`checkpoints`＝**装载面 13 枚／154.7 MB**（装机真正会读的），`output`＝217 枚／2969.7 MB。
默认**只开装载面且全开**——不按体积升序抽样，那样会系统性挑到没有 `metadata` 的小电路件，
等于把分组条件套在结果上（[[aggregate-column-two-questions-before-use]] 的第一问）。
`--roots output` 才扩到历史跑，并强制出版 `coverage`：扫了哪些根、读了几枚、跳了几枚、面上一共几枚。
`face_is_partial` 为真时总判级词一律带 `partial_` 前缀——**抽样读数不许被读成全面结论**
（[[aggregate-column-two-questions-before-use]]：分子分母要同一集合）。
明细列表按 `ROW_LIST_CAP` 截断（装载面 13 枚只印前 12 行），**总数一律另记在 `coverage` 里**：
被切掉的那行如果不同场出版，「这一面没记录」就可能只是「我没列出来」——这台仪器自己的坑自己得堵。

零写回：只 `torch.load(..., weights_only=False)` 读，绝不 `save`、不改写任何一枚件。
落件 LF；stdout 只打 ASCII。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

#: DEBT-G115 修法③：落盘走统一落盘助手（`newline=` 由助手必备参数兜住；拒覆写由本件前置守卫承担）。
import _taiji_report_io as report_io  # noqa: E402

REPORT_FORMAT = "taiji-envelope-recipe-face-v1"
RECIPE_KEY = "answer_source"
WEIGHT_FACE_ROOTS = ("checkpoints", "output")
#: 默认只开**装载面**（装机真正会读的那批）；`--roots output` 才扩到历史跑。
DEFAULT_ROOTS = ("checkpoints",)
EXCLUDED_WEIGHT_FACES = ("reports", "outputs")
ENVELOPE_METADATA_KEY = "metadata"
#: 明细列表只印前这么多个，**总数一律另记一格**——否则被切掉的样本会伪装成「这一面没有」。
ROW_LIST_CAP = 12
VERDICTS = (
    "consistent",
    "answer_source_mismatch",
    "no_recorded_recipe_in_sample",
    "partial_consistent",
    "partial_answer_source_mismatch",
    "partial_no_recorded_recipe_in_sample",
)


def _provenance_predicate():
    """复用 `audit_taiji_answer_source_provenance` 的矛盾式，不在这里手写第二份判据。"""

    path = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_answer_source_provenance.py"
    spec = importlib.util.spec_from_file_location("envelope_recipe_predicate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def repo_head(root: Path) -> str:
    """取 `root` 所在仓库的短 SHA；`root` 不在仓库里时如实记 `not_a_git_repo`。

    这一格是**披露**不是判据：夹具根（`tmp_path`）本来就不是仓库，为它崩掉整台仪器
    会把"读信封"降级成"只有真面才读得了"，而编一个 SHA 出来又是假披露。
    """

    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "not_a_git_repo"


def enumerate_face(root: Path, roots: list[str]) -> list[Path]:
    found: list[Path] = []
    for face in roots:
        base = root / face
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in {"node_modules", "__pycache__"}]
            found += [Path(dirpath) / name for name in filenames if name.endswith(".pt")]
    return sorted(found, key=lambda q: (q.stat().st_size, q.as_posix()))


def select_batch(
    points: list[Path], max_files: int, max_mb: int, only: list[str]
) -> tuple[list[Path], int, int]:
    if only:
        chosen = []
        for rel in only:
            path = Path(rel)
            if not path.is_absolute():
                path = PROJECT_ROOT / rel
            if not path.is_file():
                raise ValueError(f"--only 指向不存在的件：{rel}（点名错的不许被静默跳过）")
            chosen.append(path)
        return chosen, 0, 0
    budget = max_mb * 1_000_000
    chosen: list[Path] = []
    skipped_by_size = 0
    skipped_by_count = 0
    for path in points:
        size = path.stat().st_size
        if len(chosen) >= max_files:
            skipped_by_count += 1
            continue
        if size > budget:
            skipped_by_size += 1
            continue
        chosen.append(path)
        budget -= size
    return chosen, skipped_by_size, skipped_by_count


def classify_envelope(env: object, where: str, predicate) -> dict[str, object]:
    metadata = env.get(ENVELOPE_METADATA_KEY) if isinstance(env, dict) else None
    if not isinstance(metadata, dict):
        return {"file": where, "has_recipe": False, "reason": "no metadata dict", "mismatches": []}
    if RECIPE_KEY not in metadata:
        return {
            "file": where,
            "has_recipe": False,
            "reason": "metadata has no answer_source",
            "metadata_keys": sorted(str(k) for k in metadata)[:12],
            "mismatches": [],
        }
    return {
        "file": where,
        "has_recipe": True,
        RECIPE_KEY: metadata.get(RECIPE_KEY),
        "self_answers_path": metadata.get("self_answers_path"),
        "answer_chunking": metadata.get("answer_chunking"),
        "mismatches": predicate.mismatches(metadata, where),
    }


def read_face(
    root: Path, max_files: int, max_mb: int, only: list[str], roots: list[str]
) -> dict[str, object]:
    predicate = _provenance_predicate()
    points = enumerate_face(root, roots)
    total_mb = round(sum(p.stat().st_size for p in points) / 1e6, 1)
    chosen, skipped_by_size, skipped_by_count = select_batch(points, max_files, max_mb, only)
    if not chosen:
        raise ValueError(
            f"批次为空（面上一共 {len(points)} 枚）：--max-files／--max-mb 把每一枚都挡住了。"
            "空样本不许出版判级"
        )
    rows: list[dict[str, object]] = []
    mismatches: list[str] = []
    loaded = 0
    unreadable: list[str] = []
    for path in chosen:
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
        try:
            import torch

            env = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 — 任何加载失败都必须登记，不许默默少算
            unreadable.append(f"{rel}: {type(exc).__name__}")
            continue
        loaded += 1
        row = classify_envelope(env, rel, predicate)
        rows.append(row)
        mismatches += list(row["mismatches"])
    with_recipe = [row for row in rows if row["has_recipe"]]
    reason_counts: dict[str, int] = {}
    for row in rows:
        if not row["has_recipe"]:
            reason = str(row["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    bare = (
        "no_recorded_recipe_in_sample"
        if not with_recipe
        else ("answer_source_mismatch" if mismatches else "consistent")
    )
    partial = loaded < len(points)
    verdict = f"partial_{bare}" if partial else bare
    if verdict not in VERDICTS:
        raise ValueError(f"判级词跑出预注册词典：{verdict}")
    return {
        "format": REPORT_FORMAT,
        "prereg": "plans/reference/PLAN-N1-03_chain_and_roles_prereg_20261011.md",
        "criterion": "J-N1c-3",
        "opened_read_only": True,
        "anything_written": False,
        "verdict": verdict,
        "coverage": {
            "roots_scanned": sorted(roots),
            #: `--only` 点名的件可以在扫描根之外，此时 `files_in_face` 与 `files_selected` 不同源；
            #: 把挑法写成一枚词，读的人就不必猜"13 枚的面上怎么只读了 1 枚"。
            "batch_mode": "only_named" if only else "budget",
            "other_known_roots": sorted(set(WEIGHT_FACE_ROOTS) - set(roots)),
            "files_in_face": len(points),
            "face_total_mb": total_mb,
            "files_selected": len(chosen),
            "files_loaded": loaded,
            "selected_mb": round(sum(p.stat().st_size for p in chosen) / 1e6, 1),
            "skipped_over_byte_budget": skipped_by_size,
            "skipped_over_file_budget": skipped_by_count,
            "face_is_partial": partial,
            #: 部分批次必须连**挑法**一起出版：`enumerate_face` 按体积升序，所以小预算＝系统性挑到
            #: 最小（因而最可能整档没有 `metadata`）的那批——分组条件套在结果上，不出版就是假读数。
            "selection_order": "size_ascending",
            "unreadable": unreadable[:10],
            "with_recipe_total": len(with_recipe),
            "without_recipe_total": len(rows) - len(with_recipe),
            "no_recipe_reason_counts": dict(sorted(reason_counts.items())),
            "row_list_cap": ROW_LIST_CAP,
        },
        "carriers_with_recipe": [
            {k: row[k] for k in row if k != "mismatches"} for row in with_recipe
        ][:ROW_LIST_CAP],
        "carriers_without_recipe": [
            {"file": row["file"], "reason": row["reason"]} for row in rows if not row["has_recipe"]
        ][:ROW_LIST_CAP],
        "mismatches": mismatches[:12],
        "what_this_does_not_prove": [
            "「metadata 里没有 answer_source」是**无记录**，不是反写也不是合规——旧基座（如 "
            "`checkpoints/seed_beta.pt`）整档无该键，只能登记为无记录",
            "本件证的是信封内部两两不矛盾（三枚键由同一份 CLI 参数写出，一致近乎由构造保证），"
            "**不证**标签所指那条生成路径真的被走过；那需要回答级的来源记录，今天不存在",
            "抽样面（`coverage.files_loaded` < `files_in_face`）时判级词一律带 `partial_` 前缀；"
            "不带全量复算不许把这句话读成「全仓信封都合规」",
        ],
        "git_head": repo_head(root),
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--roots",
        action="append",
        default=[],
        help="要开的根（可重复）；默认只开装载面 checkpoints/",
    )
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--max-mb", type=int, default=1024)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--out-report",
        default="reports/taiji_envelope_recipe_face_20261012.json",
        help="落盘路径（默认落 reports/；件已存在则拒覆盖 rc=2）",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)

    #: DEBT-G115 修法③＋#81 出口标准：落点求解与守卫**前置到 `read_face()` 之前**（那一步要
    #: `torch.load` 最多 `--max-mb` 兆的信封）⇒ 落点已存在且无旗标时立刻 rc=2，不白读整面。
    #: 拒绝出口统一为 `stdout JSON ＋ 退出码 2`（旧态是 stderr 散文——CLIXML 假崩溃，㊵-864…867 在案）。
    #: 落点形保留「`root / out`」并把 `reports/…` 字面量留在 `--out-report` 默认值里（㊵-854②）。
    out = Path(args.out_report)
    if not out.is_absolute():
        out = root / out
    if out.exists() and not args.overwrite:
        print(
            json.dumps(
                {
                    "format": REPORT_FORMAT,
                    "status": "refuse_overwrite",
                    "path": str(out),
                },
                ensure_ascii=True,
            )
        )
        return 2

    try:
        report = read_face(
            root, args.max_files, args.max_mb, args.only, args.roots or list(DEFAULT_ROOTS)
        )
    except ValueError as exc:
        #: 零样本／点名打空等 ⇒ 响亮拒绝（**同一出口形状**：stdout JSON＋rc=2）。
        print(
            json.dumps(
                {
                    "status": "refuse_publish",
                    "error": str(exc).encode("ascii", "backslashreplace").decode("ascii"),
                },
                ensure_ascii=True,
            )
        )
        return 2
    report_io.write_json_report(out, report, newline="\n", overwrite=True)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "loaded": report["coverage"]["files_loaded"],
                "in_face": report["coverage"]["files_in_face"],
                "with_recipe": report["coverage"]["with_recipe_total"],
                "without_recipe": report["coverage"]["without_recipe_total"],
                "listed_rows": len(report["carriers_with_recipe"])
                + len(report["carriers_without_recipe"]),
                "no_recipe_reasons": report["coverage"]["no_recipe_reason_counts"],
                "row_list_cap": report["coverage"]["row_list_cap"],
                "mismatches": len(report["mismatches"]),
                "unreadable": len(report["coverage"]["unreadable"]),
                "out": out.as_posix(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
