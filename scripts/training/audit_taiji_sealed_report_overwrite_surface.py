"""DEBT-G35／G50 的暴露面读数：哪些入库仪器的默认写靶直接落在**已入库**的报告件上。

G35 的处方写着「训练仪器的 `--out` 写靶必须按 sha 后缀命名或写前把旧件改名，让『覆盖』变『并存』」，
G50 又写着「与 DEBT-G35 的拒绝面需一起设计以免互相覆盖」——两行到今天只有散文，**没有暴露面计数**。
本件把那一面做成读数：扫版本控制里的 `scripts/**/*.py`，用 AST 取 argparse 的写靶旗标默认值，
只对**已 tracked 的路径**计数，并按"写前会不会拒绝"分**三类**出版。

为什么是三分类而不是"有／无护栏"两分类
--------------------------------------
起草时我原本按「有 `--overwrite` 旗标 **且** 有 `exists()` 检查」算"有护栏"，
但实读 `scripts/training/calibrate_taiji_r2_readout_retrain.py:215` 是另一种合法形状：

    if report_path.exists():
        parser.error(f"{report_path} already exists; this instrument never overwrites a report")

它**没有** `--overwrite` 旗标，却是最严格的那种"永不覆写"。用两分类会把它误算进暴露面——
那正是"仪器把自己的错报成事实"的形状。所以分三档，且每档都要给出**命中的证据原文**：

* `refusal_without_override_flag`：写前拒绝，但不给覆写口（最严，`calibrate_taiji_r2` 那一形）；
* `refusal_with_override_flag`：写前拒绝＋`--overwrite`／`--force` 显式口子（本轮新落三枚仪器那一形）；
* `no_refusal_found`：既没拒绝也没旗标 ⇒ 才算**暴露面**。

`exists()` 单独出现不算护栏：`verify_core_reasoning_coverage.py:153`／`:167` 的 `exists()` 是**读**检查。

2026-10-13（DEBT-G132／㊵-774）升版 v2：加**第二张面** `literal_assignment`＝源码里硬编的字面落盘路径
（`out = R / "reports/x.json"` 且 `out` 被当写靶用过）。v1 只认 argparse 的 `default`，于是
`measure_taiji_n2_06_write_contract_values.py` 那种"把落盘靶硬编成一枚已入库读数件"的仪器**整面看不见**——
它既不报命中也不报暴露，而我差点就照着"暴露面里没有它"去跑它、覆盖掉那份 20261011 的在册件。
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: 契约册（`test_n5_25`）按文件路径加载本件、也被起成真子进程，脚本自己那层目录先上 `sys.path`，
#: 统一落盘助手才解析得到（同㊵-790／802 那条例外约定）。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

#: ㊵-790：写靶识别面**变宽**（统一落盘助手的两个入口也算写入）⇒ 按"识别边界变了就升版"的规矩走 v3。
#: ㊵-821：读出面**变宽**（新增 `shared_writer_sensitivity` 一枚披露块）⇒ 按"读出面变了就升版"走 v4。
#: ㊵-870（DEBT-G137 修法①②）：识别面**再变宽**（`PROJECT_ROOT / "reports" / "x.json"` 这种**被 `/` 拆开**的
#: 拼接形现在读得出）＋新增 `unresolved_target_forms` 一枚披露块 ⇒ 走 v5。
#: **`37／36／22／21` 那组数是 v4 面（旧识别面）的下界，与 v5 面**不可比**——旧引用句一律不许当现行**。
REPORT_FORMAT = "taiji-sealed-report-overwrite-surface-v5"
WRITE_TARGET_FLAGS = {"out", "report", "arm-report", "out-report"}
#: 字面路径里出现这个前缀才算"落盘到读数件面"（第二张面的判据之一）。
REPORT_MARKER = "reports/"
#: ㊵-870（G137 修法②）：「看起来像落盘路径」的尾巴——只为让**下界本身可读**，不是判据。
DATA_SUFFIXES = (".json", ".jsonl", ".pt", ".md", ".txt")
#: ㊵-870：`unresolved_target_forms.rows` 的明细上限；**总数另记在 `count` 里**
#: （被切掉的行不许伪装成「这一面没有」——本件自己的旧坑）。
UNRESOLVED_LIST_CAP = 40
OVERRIDE_FLAGS = ("--overwrite", "--force")
CATEGORY_REFUSE_ONLY = "refusal_without_override_flag"
CATEGORY_REFUSE_WITH_FLAG = "refusal_with_override_flag"
CATEGORY_NONE = "no_refusal_found"


#: 统一落盘助手的两个写入口（`scripts/training/_taiji_report_io.py`）。它们与 `write_text` 等价地
#: "把一个变量当写靶用过"——不认它们，修法③ 每迁一台仪器就从本件面上少一个写靶。
SHARED_WRITER_CALLS = ("write_text_report", "write_json_report")


def _tracked(root: Path) -> list[str]:
    return subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()


def _write_flag_defaults(tree: ast.Module) -> list[tuple[list[str], str]]:
    found: list[tuple[list[str], str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "attr", "") != "add_argument":
            continue
        names = [
            a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        if not names or names[0].lstrip("-") not in WRITE_TARGET_FLAGS:
            continue
        default = None
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                default = kw.value.value
        if isinstance(default, str):
            found.append((names, default))
    return found


def _declares_override_flag(tree: ast.Module) -> bool:
    """同一文件里是否声明了 `--overwrite`／`--force`（拒绝覆写要能被显式解除）。"""

    return any(
        isinstance(node.args[0], ast.Constant) and node.args[0].value in OVERRIDE_FLAGS
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", "") == "add_argument"
        and node.args
    )


def _negated_positions(test_node: ast.expr) -> list[ast.expr]:
    """返回被 `not` 直接作用的 `.exists()` 调用（读侧检查的形状）。"""

    negated: list[ast.expr] = []
    for node in ast.walk(test_node):
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Call)
            and isinstance(node.operand.func, ast.Attribute)
            and node.operand.func.attr == "exists"
        ):
            negated.append(node.operand)
    return negated


def _body_has_refusal(body: list[ast.stmt]) -> bool:
    """拒绝出口只认**节点形状**：`x.error(...)`／`raise`／`sys.exit`／`return <整数>`。

    v1 用的是"渲染成文本后找子串"，于是两处假护栏把暴露面算少了（2026-10-13 现查，DEBT-G132）：
    `fam.read_text(encoding='utf-8', errors='replace')` 里的 `errors=` 命中了 `error`，
    而函数中途的 `return 'bypass_confirmed（…）'` 命中了 `return`。安全面宁可多报暴露，
    不能把"其实会覆写"读成"有护栏"（[[guard-must-be-able-to-fail]]）。
    """

    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ("error", "exit", "_exit"):
                return True
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
            #: `isinstance(True, int)` 为真，但 `return True` 不是拒绝出口。
            and not isinstance(node.value.value, bool)
        ):
            return True
    return False


def _exists_guard_bodies(tree: ast.Module) -> list[str]:
    """找 `if <...>.exists():` 且体内有拒绝出口（`parser.error`／`raise`／`exit`／`return <int>`）的那些体原文。"""

    bodies: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        #: 测试条件常写成 `if path.exists() and not args.overwrite:`（`BoolOp`），
        #: 只认裸 `Call` 会把**自己家的护栏**漏掉、误报成暴露面（沙箱实测抓到这一形）。
        exists_calls = [
            sub
            for sub in ast.walk(node.test)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "exists"
        ]
        if not exists_calls:
            continue
        #: `if not path.exists(): raise` 是**读侧存在性检查**（输入没准备好），
        #: 不是"写前拒绝覆写"。把它当护栏会把真暴露面漏掉（实跑 `verify_core_reasoning_coverage` 抓到）。
        negated = _negated_positions(node.test)
        if exists_calls and len(negated) == len(exists_calls):
            #: 全部 exists() 都是被 `not` 作用的读侧检查 ⇒ 不构成"写前拒绝覆写"的护栏。
            continue
        if _body_has_refusal(node.body):
            bodies.append("\n".join(ast.unparse(child) for child in node.body))
    return bodies


def _base_name(expr: ast.expr) -> str | None:
    """`target.write_text(...)`／`Path(target).write_text(...)`／`open(target, "w")` 里的宿主变量名。"""

    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.args:
        inner = expr.args[0]
        return inner.id if isinstance(inner, ast.Name) else None
    return None


def _path_parts(expr: ast.expr) -> list[str]:
    """把 `A / "b" / "c"`（或 f-string）里的**字符串常量片段按顺序**取出来。

    只走 `BinOp(Div)` 与 `JoinedStr` 两种节点，遇到别的（`Name`／`Call`／`Attribute`）就当"底座"丢掉——
    底座（如 `PROJECT_ROOT`）本来就不进落点字符串。
    """

    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        return _path_parts(expr.left) + _path_parts(expr.right)
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [expr.value]
    if isinstance(expr, ast.JoinedStr):
        return [
            piece.value
            for piece in expr.values
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str)
        ]
    return []


def _path_shaped_hint(expr: ast.expr) -> str | None:
    """㊵-870（G137 修法②）：这一坨**看起来像**落盘路径吗？——只为让下界本身可读，不是判据。"""

    texts = [
        piece.value
        for piece in ast.walk(expr)
        if isinstance(piece, ast.Constant) and isinstance(piece.value, str)
    ]
    if not texts:
        return None
    looks_like_path = any(text.endswith(DATA_SUFFIXES) for text in texts) or any(
        "/" in text for text in texts
    )
    if not looks_like_path:
        return None
    return " / ".join(texts) if len(texts) > 1 else texts[0]


def _unresolved_write_targets(
    tree: ast.Module, *, written: set[str], resolved: set[str]
) -> list[dict[str, str]]:
    """㊵-870（G137 修法②）：**被当写靶用过**但本件读不出落点的变量——单独出一桶，让下界可读。

    它是披露不是红支：把这些行藏起来，`37` 那种下界就会被读成全集——G137 记的具体后果是
    下一轮照着一份说它安全的名单去跑别人名下的仪器，把在册历史读数件覆盖掉。
    """

    unresolved: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in written or target.id in resolved:
            continue
        hint = _path_shaped_hint(node.value)
        if hint is not None:
            unresolved.append({"name": target.id, "shape": hint})
    return unresolved


def _literal_target_names(tree: ast.Module) -> dict[str, list[str]]:
    """变量名 → 它绑定的 `reports/` **字面**路径（含 `R / "reports/x.json"` 这种拼接形）。

    这是第二张面：写靶不走 argparse 默认值，而是硬编在源码里。2026-10-13 发现本器整面漏着它
    （`measure_taiji_n2_06_write_contract_values.py` 就是把落盘路径硬编成**一枚已入库读数件**，
    所以它既不在命中里也不在暴露面里 ⇒ DEBT-G132）。
    """

    bound: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        literals = [
            sub.value
            for sub in ast.walk(node.value)
            if isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and REPORT_MARKER in sub.value
        ]
        #: ㊵-870（DEBT-G137 修法①）：`PROJECT_ROOT / "reports" / "x.json"` 这种**被 `/` 拆开**的拼接形，
        #: 上面那支读不出（没有任何**单枚**常量自带 `reports/`）⇒ 把 `/` 链上的字符串片段按顺序拼回。
        #: **只在某一片段恰好等于 `"reports"`**（marker 去掉尾斜杠）时才拼——否则 `"my_reports" / "x.json"`
        #: 会被误读成一个落点（那是"只绑不写不算"之外的第二类假阳性）。
        parts = _path_parts(node.value)
        if REPORT_MARKER.rstrip("/") in parts:
            index = parts.index(REPORT_MARKER.rstrip("/"))
            tail = REPORT_MARKER + "/".join(parts[index + 1 :])
            if tail != REPORT_MARKER and tail not in literals:
                literals.append(tail)
        if literals:
            bound.setdefault(target.id, []).extend(literals)
    return bound


def _written_names(tree: ast.Module, *, include_shared: bool = True) -> set[str]:
    """被**当写靶用过**的变量名。只绑过字面路径而没往里写的不算（那是读数件常量或读检查）。

    ㊵-790：统一落盘助手（DEBT-G115 修法③，`_taiji_report_io.py`）把写入从 `x.write_text(...)`
    收进 `report_io.write_text_report(x, ...)`——**如果这里只认 `write_text`，迁移一台仪器就从本件的
    识别面上抹掉一个写靶**，暴露面会被自己主导的收口动作悄悄缩小。故助手那两个名字同样算写入。

    ㊵-821（DEBT-G139 修法①）：`include_shared=False` 给本件一个**自测口子**——同一棵树上跑两遍，
    差出来的那批字面靶就是"只有靠助手识别才看得见"的行。这条差是**出版读数**，不是调试输出：
    现行树上它今天为 0（㊵-820 的摘除对照实测七个出版值一枚不动），所以"门认得助手写入"这句
    在没有这枚键之前**不可观测**；今后哪台仪器的字面靶只剩助手这一条识别路，它会当场非零。
    """

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = getattr(func, "attr", None) or (func.id if isinstance(func, ast.Name) else "")
        if include_shared and called in SHARED_WRITER_CALLS and node.args:
            #: 助手的第一个参数就是写靶；`report_io` 这个接收者本身不是靶。
            name = _base_name(node.args[0])
            if name:
                names.add(name)
            continue
        if isinstance(func, ast.Attribute) and func.attr in ("write_text", "write_bytes"):
            name = _base_name(func.value)
            if name:
                names.add(name)
        elif isinstance(func, ast.Name) and func.id == "open" and node.args:
            mode = node.args[1] if len(node.args) > 1 else None
            writing = (
                isinstance(mode, ast.Constant)
                and isinstance(mode.value, str)
                and any(char in mode.value for char in "wax+")
            )
            if writing:
                name = _base_name(node.args[0])
                if name:
                    names.add(name)
    return names


def classify(tree: ast.Module, source: str) -> tuple[str, list[str]]:
    bodies = _exists_guard_bodies(tree)
    if not bodies:
        return CATEGORY_NONE, []
    evidence = bodies[:2]
    if _declares_override_flag(tree):
        return CATEGORY_REFUSE_WITH_FLAG, evidence
    return CATEGORY_REFUSE_ONLY, evidence


def scan(
    root: Path,
    tracked_set: set[str],
    *,
    dependent: list[str] | None = None,
    unresolved: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """`dependent`（㊵-821）就地收下"这一行的识别**只**来自统一落盘助手那条分支"的文件名。

    `unresolved`（㊵-870／G137 修法②）就地收下"**被当写靶用过、但本件读不出落点**"的变量行——
    这是**披露面**：把下界本身出版，免得 `37` 被读成全集。
    """

    rows: list[dict[str, object]] = []
    for rel in sorted(
        p for p in sorted(tracked_set) if p.startswith("scripts/") and p.endswith(".py")
    ):
        source = (root / rel).read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        category, evidence = classify(tree, source)
        for names, default in _write_flag_defaults(tree):
            if default in tracked_set:
                rows.append(
                    {
                        "file": rel,
                        "face": "argparse_default",
                        "flags": names,
                        "default_target": default,
                        "category": category,
                        "refusal_evidence": evidence,
                    }
                )
        written = _written_names(tree)
        written_no_shared = _written_names(tree, include_shared=False)
        literal_targets = _literal_target_names(tree)
        if unresolved is not None:
            for item in _unresolved_write_targets(
                tree, written=written, resolved=set(literal_targets)
            ):
                unresolved.append({"file": rel, **item, "category": category})
        for name, literals in sorted(literal_targets.items()):
            if name not in written:
                #: 只绑过字面路径、没往里写过 ⇒ 那是读数件常量或读侧检查，不是落盘靶。
                continue
            if dependent is not None and name not in written_no_shared and rel not in dependent:
                #: 摘掉助手那条分支就看不见这个靶 ⇒ 本行的识别**依赖**助手识别。
                dependent.append(rel)
            for literal in sorted(set(literals)):
                if literal in tracked_set:
                    rows.append(
                        {
                            "file": rel,
                            "face": "literal_assignment",
                            "flags": [name],
                            "default_target": literal,
                            "category": category,
                            "refusal_evidence": evidence,
                        }
                    )
    return rows


def build_report(root: Path) -> dict[str, object]:
    tracked = set(_tracked(root))
    helper_dependent: list[str] = []
    unresolved_rows: list[dict[str, object]] = []
    rows = scan(root, tracked, dependent=helper_dependent, unresolved=unresolved_rows)
    if not rows:
        raise ValueError(
            "零命中：没有一枚仪器的默认写靶落在已入库件上。先确认扫描面（tracked 的 scripts/**/*.py），"
            "不许把零命中读成「这面已经干净」"
        )
    counts = {
        category: sum(1 for row in rows if row["category"] == category)
        for category in (CATEGORY_REFUSE_ONLY, CATEGORY_REFUSE_WITH_FLAG, CATEGORY_NONE)
    }
    face_counts = {
        face: sum(1 for row in rows if row["face"] == face)
        for face in ("argparse_default", "literal_assignment")
    }
    return {
        "format": REPORT_FORMAT,
        "debts": ["DEBT-G35", "DEBT-G50", "DEBT-G132", "DEBT-G137"],
        "face": (
            "tracked 的 scripts/**/*.py；两张面并立——`argparse_default`＝写靶旗标的默认值，"
            "`literal_assignment`＝源码里硬编的字面路径且该变量被当写靶用过"
            "（v2 新增这一面：v1 只认 argparse 默认值，把硬编靶整面漏掉）"
        ),
        "script_files_scanned": len(
            [p for p in tracked if p.startswith("scripts/") and p.endswith(".py")]
        ),
        "default_targets_on_tracked_files": len(rows),
        "face_counts": face_counts,
        "distinct_files": len({row["file"] for row in rows}),
        "category_counts": counts,
        "exposure_targets": counts[CATEGORY_NONE],
        "exposure_files": len({row["file"] for row in rows if row["category"] == CATEGORY_NONE}),
        #: ㊵-821＝DEBT-G139 修法①：这条识别分支**有没有真实载体**今天是可出版的读数了。
        #: `count 0` 不是"分支没用"，而是"现行树上还没有哪一行**只**靠它才看得见"——
        #: ㊵-820 的摘除对照当时只能证"聚合数不动"，证不了"哪天瞎"；这一枚把那个盲区变成非零即报警。
        "shared_writer_sensitivity": {
            "rows_total": len(rows),
            "count": len(helper_dependent),
            "files_recognized_only_via_shared_writer": sorted(helper_dependent),
            "definition": (
                "同一棵树上跑两遍 `_written_names`（第二遍 `include_shared=False`）；"
                "`literal_assignment` 那一面里，某枚字面靶的宿主变量**只在第一遍**被认成写靶 ⇒ 该文件入列。"
                "**v5 起 `37／36／22／21` 那组口径已随识别面变宽而换态**（G137 修法①），"
                "旧值与 v5 值**新旧不可比**——跨面比较必须带 `format`。"
            ),
        },
        #: ㊵-870（DEBT-G137 修法②）：把**下界本身**出版。`default_targets_on_tracked_files` 只是下界
        #: （识别面读不出落点的那批靶既不在命中里、也不在暴露面里），这一桶登记它们的数量与文件。
        "unresolved_target_forms": {
            "count": len(unresolved_rows),
            "rows_listed": min(len(unresolved_rows), UNRESOLVED_LIST_CAP),
            "files": sorted({str(row["file"]) for row in unresolved_rows}),
            "definition": (
                "**被当写靶用过**（`write_text`／`open(...,'w')`／统一落盘助手的第一个实参）"
                "但本件读不出 `reports/` 落点的变量 ⇒ `default_targets_on_tracked_files` 只算**下界**，"
                "不把它们读成全集。这是**披露面，不是红支**（G137 记的具体后果：照一份说它安全的名单"
                "去跑别人名下的仪器，把在册历史读数件覆盖掉）。"
            ),
            "rows": unresolved_rows[:UNRESOLVED_LIST_CAP],
        },
        "rows": rows,
        "categories": {
            CATEGORY_REFUSE_ONLY: "写前拒绝覆写、但不给覆写口（最严；`calibrate_taiji_r2` 那一形）",
            CATEGORY_REFUSE_WITH_FLAG: "写前拒绝＋`--overwrite`／`--force` 显式口子",
            CATEGORY_NONE: "既没写前拒绝也没旗标 ⇒ 计入暴露面",
        },
        "scope_note": "只读普查：不改别人名下仪器的默认落点（改了会让既有引用失效，属别线半径）",
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="扫描根（契约测指向临时仓）")
    parser.add_argument(
        "--out-report",
        default="reports/taiji_sealed_report_overwrite_surface_20261011.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = build_report(Path(args.root))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out = Path(args.out_report)
    if not out.is_absolute():
        out = Path(args.root) / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    report_io.write_json_report(out, report, newline="\n", overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "script_files_scanned": report["script_files_scanned"],
                "default_targets_on_tracked_files": report["default_targets_on_tracked_files"],
                "distinct_files": report["distinct_files"],
                "face_counts": report["face_counts"],
                "category_counts": report["category_counts"],
                "exposure_targets": report["exposure_targets"],
                "exposure_files": report["exposure_files"],
                "unresolved_target_forms": report["unresolved_target_forms"]["count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
