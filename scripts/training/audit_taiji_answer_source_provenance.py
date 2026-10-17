"""PLAN-N1-03 §3 `J-N1c-3`（来源记录不许反写）的执行体：三条面各自出数，零权重读取。

判据原文（`plans/reference/PLAN-N1-03_chain_and_roles_prereg_20261011.md` §3）
--------------------------------------------------------------------------
"任何被采纳的回答，其 `answer_source` 必须与生成路径的自述一致；若一条回答的
`answer_source` 指向参考侧而产出路径是模型侧 ⇒ 判 `answer_source_mismatch`。零算力：只读已存在的件。"

为什么这一条要分三张面
----------------------
* **载体面（JSON）**：今天的 `answer_source` 只由跑道器写进**检查点信封**（`train_seed_corpus.py:547`
  的 `attach_metadata(extra={...})`），**退出件不带它**（`exit_record_path` 那支从 714 行起只加
  `checkpoint_*`／`machine_load`／`history_*`）。所以"只读已存在的件"这句话在 JSON 面上**没有载体**。
  更麻烦的是：仓里有两枚**审计件**把 `answer_source` 当**数据**登着（`"answer_source": []`、
  `"answer_source": "scripts/training/train_seed_corpus.py"`），裸按字段名扫会把它们当成载体——
  那是仪器读自己的披露当证据。载体谓词因此要求跑道器信封的**同族键**（`trainer`／`answer_chunking`／
  `self_answers_path`…）同时在场。
* **强制面（跑道器源码）**：反写要么"发生并被看见"，要么"根本写不出来"。后者由 AST 现取：
  `main()` 里那几支以 `answer_source` 为条件、以 `parser.error` 为出口的守卫（1344／1349／1351 行）。
  再加**两支真跑**的负对照证明静态面不是纸面上的——它们都是**非法组合**，`parser.error` 在任何
  加载／训练之前就把进程关掉（一致性组合一律拒探，见 `_probe_argv`）。
* **产品面**：`answer_source` 今天不由产品码写（㊵-701 判 `read_only_outside_product`）。这条不改判据，
  但它决定"反写"这件事今天可能发生在哪张面上——所以单独出版一个计数，不并进载体面的结论。

零算力：不加载检查点、不开 `.pt`（现读 230 枚／3124.4 MB，开面需 owner 授权），不训练、不动权重。
落件行尾 LF；stdout 只打 ASCII（GBK 控制台会因中文崩，见 DEBT-G46⑤）。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = "scripts/training/train_seed_corpus.py"
REPORT_FORMAT = "taiji-answer-source-provenance-v1"
#: 契约册（`test_n1_07`／`test_g14`／`test_n1_09`）按文件路径加载本件、也被起成真子进程，
#: 脚本自己那层目录先上 `sys.path`，统一落盘助手才解析得到（同㊵-790／802 那条例外约定）。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

RECIPE_KEY = "answer_source"
VALID_SOURCES = ("corpus", "self")
#: 跑道器 `attach_metadata(extra=...)` 与 `answer_source` **同源同批**写出的兄弟键（:539-551）。
#: 只有和它们同在一个对象里出现的 `answer_source` 才是"一趟跑的自述"，而不是别人登的账。
ENVELOPE_SIBLINGS = (
    "trainer",
    "answer_chunking",
    "answer_max_chars",
    "self_answers_path",
    "end_boundary_after_newline",
)
SCAN_SUFFIXES = (".json", ".jsonl")
MAX_SCAN_BYTES = 8_000_000
JSON_FACES = ("reports", "output", "checkpoints")
#: 权重面分两张：**能承载信封**的两面（报价面）与其余历史面。计数不带面名就会被读成全仓（㊵-712）。
WEIGHT_FACE_ROOTS = ("checkpoints", "output")
EXCLUDED_WEIGHT_FACES = ("reports", "outputs")
CARRIER_VERDICTS = (
    "answer_source_mismatch",
    "consistent_within_carriers",
    "no_carrier_outside_weight_artifacts",
)
#: 只允许探测**非法**组合（一致性组合会真的开始跑）。键＝旗标值，值＝给 CLI 的参数表。
NEGATIVE_CONTROLS: tuple[dict[str, object], ...] = (
    {
        "combo": "answer_source=self 而不给 --self-answers",
        "argv": ["--answer-source", "self", "--answer-chunking", "per-answer"],
        "expect_fragment": "--answer-source self requires --self-answers",
    },
    {
        "combo": "answer_source=corpus 而给了 --self-answers",
        "argv": ["--answer-source", "corpus", "--self-answers", "nope.jsonl"],
        "expect_fragment": "without --answer-source self",
    },
)


def _json_objects(root: Path, rel: str, text: str) -> list[dict[str, object]]:
    """把一件 JSON/JSONL 拆成对象列表；解析失败返回空（不炸整轮扫描）。"""

    if rel.endswith(".jsonl"):
        rows: list[dict[str, object]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                node = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(node, dict):
                rows.append(node)
        return rows
    try:
        node = json.loads(text)
    except json.JSONDecodeError:
        return []
    found: list[dict[str, object]] = []

    def walk(item: object) -> None:
        if isinstance(item, dict):
            found.append(item)
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)

    walk(node)
    return found


def is_carrier(obj: dict[str, object]) -> bool:
    """载体＝带 `answer_source` **且**带至少一枚跑道器信封兄弟键的对象。

    这条谓词是给"仪器读自己的披露"准备的：审计件里 `"answer_source": []` 是按键名归的桶，
    兄弟键一个都没有，因此不算一趟跑的自述。
    """

    return RECIPE_KEY in obj and any(sibling in obj for sibling in ENVELOPE_SIBLINGS)


def mismatches(obj: dict[str, object], where: str) -> list[str]:
    """按跑道器自己的规则判这一条记录内部是否矛盾（规则出处＝强制面那几支守卫）。"""

    source = obj.get(RECIPE_KEY)
    path = obj.get("self_answers_path")
    has_path = bool(path) and path != "null"
    chunking = obj.get("answer_chunking")
    out: list[str] = []
    if source not in VALID_SOURCES:
        out.append(
            f"{where}: answer_source={source!r} 不在跑道器自己的取值 {list(VALID_SOURCES)} 内"
        )
    if source == "self" and not has_path:
        out.append(f"{where}: answer_source=self 而 self_answers_path 为空（参考侧记录）")
    if source == "self" and chunking != "per-answer":
        out.append(f"{where}: answer_source=self 而 answer_chunking={chunking!r}（自述是连续流）")
    if source == "corpus" and has_path:
        out.append(
            f"{where}: answer_source=corpus 却带 self_answers_path={path!r}（模型侧路径标成参考侧）"
        )
    return out


def scan_carrier_face(root: Path) -> dict[str, object]:
    files: list[Path] = []
    tracked = subprocess.run(
        ["git", "ls-files", "--", "reports"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    files += [root / rel for rel in tracked if rel.endswith(SCAN_SUFFIXES)]
    for face in ("output", "checkpoints"):
        base = root / face
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in {"node_modules", "__pycache__"}]
            files += [Path(dirpath) / name for name in filenames if name.endswith(SCAN_SUFFIXES)]
    scanned = 0
    skipped_large = 0
    carriers: list[dict[str, object]] = []
    mismatch_rows: list[str] = []
    mention_list: list[str] = []
    for path in files:
        try:
            if path.stat().st_size > MAX_SCAN_BYTES:
                skipped_large += 1
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1
        if RECIPE_KEY not in text:
            continue
        found = [obj for obj in _json_objects(path, path.name, text) if is_carrier(obj)]
        if not found:
            mention_list.append(path.relative_to(root).as_posix())
            continue
        for obj in found:
            where = path.relative_to(root).as_posix()
            carriers.append(
                {
                    "file": where,
                    RECIPE_KEY: obj.get(RECIPE_KEY),
                    "self_answers_path": obj.get("self_answers_path"),
                    "answer_chunking": obj.get("answer_chunking"),
                }
            )
            mismatch_rows += mismatches(obj, where)
    return {
        "files_scanned": scanned,
        "files_skipped_over_size": skipped_large,
        "scan_faces": list(JSON_FACES),
        "carrier_count": len(carriers),
        "carriers": carriers[:12],
        # 点名**是哪几枚**：本件自己的落盘文件也算一枚"提及"（它含 `answer_source` 字样却没有那个字段），
        # 所以这个计数会随每次复算自增——它不是载体数，列出名字才可检查。
        "name_mental_only_files": len(mention_list),
        #: 不截断：上一版 `[:12]` 把 39 枚压成 12 枚样本，而上面那句注释要的正事是"列出名字才可检查"——
        #: 样本形状让一句"某枚在不在提及名单里"的断言按**字母序**失败（不是按行为），
        #: 这就是 `test_real_face_reproduces_today_reading` 整面红、单跑也红的真因（2026-10-18 ㊵-1111／DEBT-G163）。
        "name_mental_only_files_list": sorted(mention_list),
        "mismatches": mismatch_rows,
        "verdict": (
            "answer_source_mismatch"
            if mismatch_rows
            else (
                "consistent_within_carriers" if carriers else "no_carrier_outside_weight_artifacts"
            )
        ),
    }


def scan_enforcement_face(root: Path) -> dict[str, object]:
    """AST 现取跑道器里"以 `answer_source` 为条件、以 `parser.error` 为出口"的守卫。"""

    source = (root / RUNNER).read_text(encoding="utf-8")
    tree = ast.parse(source)
    guards: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if RECIPE_KEY not in test:
            continue
        rejected = [
            ast.unparse(call.args[0]) if call.args else ""
            for stmt in node.body
            for call in ast.walk(stmt)
            if isinstance(call, ast.Call) and ast.unparse(call.func).endswith("parser.error")
        ]
        if rejected:
            guards.append({"line": node.lineno, "condition": test, "message": rejected[0][:120]})
    if not guards:
        raise ValueError(
            f"强制面零命中：{RUNNER} 里找不到任何以 answer_source 为条件的 parser.error 守卫。"
            "零命中不许读成「没有可反写的形状」——那可能只是守卫被删了。"
        )
    return {"runner": RUNNER, "guard_count": len(guards), "guards": guards}


def _probe_argv(combo_argv: list[str]) -> list[str]:
    """把负对照拼成完整命令；一致性组合在这里就响亮拒绝（绝不因探测而起跑）。"""

    joined = " ".join(combo_argv)
    if "--answer-source self" in joined and "--self-answers" in joined:
        raise ValueError(f"这是**一致**组合，不许用探测的名义启动一次真跑：{combo_argv}")
    if "--answer-source corpus" in joined and "--self-answers" not in joined:
        raise ValueError(f"这是**一致**组合，不许用探测的名义启动一次真跑：{combo_argv}")
    return [
        sys.executable,
        RUNNER,
        "--checkpoint",
        "output/_answer_source_probe_should_not_exist/checkpoint.pt",
        *combo_argv,
    ]


def run_negative_controls(root: Path, limit: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for control in NEGATIVE_CONTROLS[: limit if limit is not None else len(NEGATIVE_CONTROLS)]:
        argv = _probe_argv(list(control["argv"]))  # type: ignore[arg-type]
        proc = subprocess.run(argv, cwd=root, capture_output=True, check=False)
        err = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
        rows.append(
            {
                "combo": control["combo"],  # type: ignore[index]
                "expect_fragment": control["expect_fragment"],  # type: ignore[index]
                "rc": proc.returncode,
                "rejected": proc.returncode != 0,
                "fragment_seen": str(control["expect_fragment"]) in err,  # type: ignore[arg-type]
            }
        )
    return rows


def scan_product_face(root: Path) -> dict[str, object]:
    """产品码今天写不写 `answer_source`（㊵-701 判 `read_only_outside_product`，此处现取不复用其结论）。"""

    tracked = subprocess.run(
        ["git", "ls-files", "--", "taiji/*.py", "seed/*.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    writers = [
        rel
        for rel in tracked
        if f'"{RECIPE_KEY}"' in (root / rel).read_text(encoding="utf-8", errors="replace")
        or f"{RECIPE_KEY}=" in (root / rel).read_text(encoding="utf-8", errors="replace")
    ]
    return {"tracked_product_files": len(tracked), "product_writer_files": writers}


def weight_face_price(root: Path) -> dict[str, object]:
    """只报价、不开面：**计数必须带面名**（㊵-712——本块先前出版的数被读成"全仓"，其实只是这两个根）。

    全仓现取是 867 枚（`reports` 396／`outputs` 182／`output` 217／`checkpoints` 13，
    再加运行自生物 59）；本块故意只报"训练产物"那两面，因为那才是**能承载信封**的面。
    """

    points = [
        Path(dirpath) / name
        for face in WEIGHT_FACE_ROOTS
        for dirpath, _dirnames, filenames in os.walk(root / face)
        if (root / face).is_dir()
        for name in filenames
        if name.endswith(".pt")
    ]
    return {
        "scan_faces": sorted(WEIGHT_FACE_ROOTS),
        "excluded_faces": sorted(EXCLUDED_WEIGHT_FACES),
        "pt_count": len(points),
        "pt_total_mb": round(sum(p.stat().st_size for p in points) / 1e6, 1),
        "opened": False,
        "reason": (
            "this instrument prices the face only; opening envelopes is a separate batched, "
            "read-only step (owner granted read-only access 2026-10-11, 四裁④)"
        ),
    }


def build_report(root: Path, run_controls: bool = True) -> dict[str, object]:
    carrier = scan_carrier_face(root)
    enforcement = scan_enforcement_face(root)
    report = {
        "format": REPORT_FORMAT,
        "prereg": "plans/reference/PLAN-N1-03_chain_and_roles_prereg_20261011.md",
        "criterion": "J-N1c-3",
        "carrier_face": carrier,
        "enforcement_face": enforcement,
        "negative_controls": run_negative_controls(root) if run_controls else [],
        "product_face": scan_product_face(root),
        "weight_artifact_face": weight_face_price(root),
        "what_this_does_not_prove": [
            "三枚配方键由同一份 CLI 参数同时写出，所以「内部一致」只证这条记录没被手改／被 patch 过，"
            "不证标签所指的那条路真的被走过",
            "载体面 0 枚时判 no_carrier_outside_weight_artifacts：这条判据今天**既不成立也不被否证**，"
            "不许读成「来源记录已被验证」",
            "本件不开 `.pt`（真正的载体在那上面），因此 J-N1c-3 的「实跑侧一致性」仍未复算；"
            "强制面只证非法组合写不出来，不证合法组合里的标签与事实相符",
            "产品面写入计数为 0 不等于产品侧永远不会反写，只说今天的产品码不产生这个键",
        ],
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if carrier["verdict"] not in CARRIER_VERDICTS:
        raise ValueError(f"载体判级词跑出预注册词典：{carrier['verdict']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--out-report", default="reports/taiji_answer_source_provenance_20261011.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-controls", action="store_true", help="不跑两支负对照（调试用；出版读数必须跑）"
    )
    args = parser.parse_args()
    try:
        report = build_report(Path(args.root), run_controls=not args.skip_controls)
    except ValueError as exc:
        print(str(exc).encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr)
        return 2
    out = Path(args.out_report)
    if not out.is_absolute():
        out = Path(args.root) / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    report_io.write_json_report(out, report, newline="\n", overwrite=args.overwrite)
    carrier = report["carrier_face"]
    print(
        json.dumps(
            {
                "carrier_verdict": carrier["verdict"],
                "carrier_count": carrier["carrier_count"],
                "name_mental_only_files": carrier["name_mental_only_files"],
                "mismatches": len(carrier["mismatches"]),
                "guard_count": report["enforcement_face"]["guard_count"],
                "controls": [
                    f"rc={row['rc']} rejected={row['rejected']} fragment={row['fragment_seen']}"
                    for row in report["negative_controls"]
                ],
                "product_writer_files": report["product_face"]["product_writer_files"],
                "pt_count": report["weight_artifact_face"]["pt_count"],
                "out": out.as_posix(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
