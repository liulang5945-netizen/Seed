"""PLAN-N1-03 守卫 G-N1c-2 的执行体：把「某个键是不是只写不读」从人工 grep 变成可复算读数。

为什么要三层，而不是「有没有读者」两档
--------------------------------------
㊵-698 那次我把「按 `\\.key` 找到了读者」当成「这个键被消费了」，写下了一条越界死讯，
㊵-699 把它撤回。真相的形状是三层，缺任何一层都会读错：

* **第 1 层 生产者**：写侧（赋值、字典字面量的键、构造关键字参数）。
* **第 2 层 别名读者**：读了以后**只是原样返回**的访问器（property 或薄函数）——它把值传到下一跳，
  但本身不对值做任何动作。`taiji/model.py:903-908` 就是这一层。把它算成「消费者」会**误杀**
  真有读者的键，也会把「写-only」误判成「已消费」。
* **第 3 层 动作读者**：读到的值进入**分支、写入或异常**（``if``/``while``/``assert``/``raise``/
  比较式／传入另一个调用）。只有这一层配叫消费点。`taiji/model.py:968` 是这一层。

第四件事：可达性不猜
--------------------
「代码里存在第 3 层读者」与「默认装配下有人会走到它」是**两句**（`last_decision` 两句同时为真：
第三层在 `--n5-shadow`＝`store_true`、默认关时才可达）。静态可达性这台仪器**不算**，
它把每个第三层点位（文件／行／所在函数／原文）出版出来，并要求调用方在
`--reachability-ledger` 里**逐枚登记**该点位的默认位可达性；没登记的键一律
`reachability_not_registered`，**不许**被读成「不可达」或「可达」，也不许据它下 `write_only`。

零算力：不加载检查点、不训练、不更新权重；只读版本控制面内的源码文本。
键表（13 枚）的名字**逐枚点名到声明它的产品文件**（见 `KEY_ORIGINS`），契约测按这张表核对；
本件**不**声称这些键由某份预注册件点名——那是一句我写下来并被现读否证的错引（㊵-701）。
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
REPORT_FORMAT = "taiji-write-only-key-face-v1"
#: 契约册（`test_n1_04`／`test_n1_05`）按文件路径加载本件，也被起成真子进程；脚本自己那层目录先上
#: `sys.path`，统一落盘助手才解析得到（同㊵-790／802 那一条例外约定）。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

# 每一枚键的**名字出处**（㊵-699 立下的规矩：名字一律从定义处取，不许从缺口反推）。
# 我早先在任务描述里把这 13 枚写成「PLAN-N1-03 §1／§2 点名」——现读否证：那件里只出现 5 枚，
# 其余 8 枚在整棵 plans 树里**零命中**，真出处是产品码的字段声明。契约测逐枚核对下表，
# 任何一枚改到别处（或声明被删）即红。
KEY_ORIGINS: dict[str, str] = {
    # `taiji/state.py` 的 `MemoryRecall` 字段（7 枚）与同文件的两枚运动侧字段
    "action_probabilities": "taiji/state.py",
    "outcome_probabilities": "taiji/state.py",
    "cortical_feedback": "taiji/state.py",
    "time_code": "taiji/state.py",
    "episode_code": "taiji/state.py",
    "provenance_probabilities": "taiji/state.py",
    "expected_reward": "taiji/state.py",
    "motor_context": "taiji/state.py",
    "motor_probabilities": "taiji/state.py",
    # R4 触发器的决策留痕
    "last_decision": "taiji/adaptive_residual_growth.py",
    # 恢复台账的信封字段
    "episodic_ids": "taiji/contracts.py",
    # 跑道器自述的两枚信封键（产品侧今天不写它们，故预期读成 read_only_outside_product）
    "answer_source": "scripts/training/train_seed_corpus.py",
    "unique_documents": "scripts/training/train_seed_corpus.py",
}

NAMED_KEYS = tuple(KEY_ORIGINS)

VERDICTS = (
    "write_only",
    "alias_only",
    "consumed_with_action",
    "read_without_writer",
    "key_absent_entirely",
)

# 已点名的**跨名转发**：同一个值用另一个名字暴露出去。不做隐式推断——每一枚都要有原文依据，
# 依据由契约测钉住（改名或删掉那条访问器时测红，逼我回来重登记）。
# `last_decision` 的这枚就是 ㊵-699 那条结论的现场：`taiji/model.py:903-908` 定义属性
# `adaptive_residual_growth_decision`，返回 `self._adaptive_residual_growth_trigger.last_decision`，
# 而 `taiji/model.py:968` 在 `propose_adaptive_residual_growth_candidate()` 里读它并据此 raise。
EXTRA_NAMES = {"last_decision": ("adaptive_residual_growth_decision",)}

# 产品面（`taiji/`＋`seed/`）专用的判级词。判定权在这一层，其余面只披露不判级。
PRODUCT_VERDICTS = (
    "consumed_in_product",
    "write_only_in_product",
    "alias_only_in_product",
    "produced_and_used_locally_only",
    "unclassified_read_in_product",
    "read_only_outside_product",
    "key_absent_entirely",
)

# 读到值之后「带动作」的直接语境：分支、写入侧的使用、异常、断言、以及把值交给另一个调用。
ACTION_PARENT_TYPES = (
    ast.If,
    ast.IfExp,
    ast.While,
    ast.Assert,
    ast.Raise,
    ast.Compare,
)


def tracked_python(root: Path) -> list[Path]:
    """扫描面按版本控制取（rglob 事后过滤会挂死，且会把未跟踪 scratch 当契约）。"""

    out = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [root / rel for rel in out]


def _key_aliases(name: str) -> set[str]:
    """一个逻辑键在源码里的写法：键名本身、私有形式、以及**已登记的同值别名**。"""

    aliases = {name, f"_{name}"}
    aliases.update(EXTRA_NAMES.get(name, ()))
    return aliases


def _assignment_target_ids(tree: ast.AST) -> set[int]:
    """赋值／for 目标本身**不是读**。

    不排掉它们的话，`self._last_decision = decision` 会被算成"第 3 层读者"，
    于是"产品在写、没人读"的键被仪器自己刷成"已消费"——那是比漏报更贵的错，
    因为它让 ``write_only_in_product`` 永远读不出来（本函数落地前实测就发生过一次）。
    只排"整个目标节点"：``self.store["x"].v = 1`` 里 ``store`` 与 ``"x"`` 仍是读。
    """

    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            ids.update(id(target) for target in targets)
        elif isinstance(node, ast.For):
            ids.add(id(node.target))
    return ids


def _read_nodes(tree: ast.AST, aliases: set[str]) -> list[tuple[ast.AST, str]]:
    """所有**读**该键的位置：`obj.name`／`obj._name`／`d["name"]`／`d.get("name")`。"""

    targets = _assignment_target_ids(tree)
    reads: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if id(node) in targets:
            continue
        if isinstance(node, ast.Attribute) and node.attr in aliases:
            reads.append((node, "attribute"))
        elif isinstance(node, ast.Name) and node.id in aliases:
            # 同一作用域内的**局部名读**（`float(expected_reward)` 那一形）。它证明值在产出处被用过，
            # 但**不**证明存下来的字段有人消费，所以必须与"字段读"分开计数。
            reads.append((node, "bare_name"))
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str) and node.slice.value in aliases:
                reads.append((node, "subscript"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "pop"}
            and node.args
        ):
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value in aliases:
                reads.append((node, f"method:{node.func.attr}"))
    return reads


def _write_nodes(tree: ast.AST, aliases: set[str]) -> list[tuple[ast.AST, str]]:
    """所有**写**该键的位置：下标赋值、属性赋值、字典字面量键、构造关键字参数。"""

    writes: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in aliases:
                    writes.append((target, "attribute_assign"))
                elif (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value in aliases
                ):
                    writes.append((target, "subscript_assign"))
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value in aliases:
                    writes.append((key, "dict_literal_key"))
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in aliases:
                    writes.append((keyword, "call_keyword"))
    return writes


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    chain: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            chain[id(child)] = node
    return chain


def _classify_read(read: ast.AST, tree: ast.AST, parents: dict[int, ast.AST]) -> tuple[str, str]:
    """返回 (层, 证据)。层＝``alias`` 或 ``action`` 或 ``bare``（读了但看不出用途）。

    判据（要能为假，所以每条都点名取法）：
    * **裸 return 的访问器** ⇒ 第 2 层：读取点的父链上第一站是 ``Return``，且它所在函数体
      只有 ``Return`` 与「None 兜底」两种语句（这正是薄访问器的形状）。
    * **一次名字转发**：读值赋给一个只被赋值一次的局部名，再看那个名字用在哪——若用在
      分支／异常／比较／调用实参 ⇒ 第 3 层（这是 ``decision = self.x`` 之后 ``if not decision...`` 的形状）。
    * 直接落在分支／异常／断言／比较／调用实参位置上 ⇒ 第 3 层。
    """

    parent = parents.get(id(read))
    hops = 0
    while parent is not None and hops < 4:
        if isinstance(parent, ACTION_PARENT_TYPES):
            return "action", f"direct_in_{type(parent).__name__}"
        if isinstance(parent, ast.Call):
            return "action", "passed_to_call"
        if isinstance(parent, ast.Assign):
            names = [t for t in parent.targets if isinstance(t, ast.Name)]
            if not names:
                return "action", "assigned_into_container"
            forwarded = names[0].id
            uses = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id == forwarded and n is not parent.value
            ]
            for use in uses:
                user = parents.get(id(use))
                if isinstance(user, (*ACTION_PARENT_TYPES, ast.Call)):
                    return "action", f"forwarded_via_{forwarded}_into_{type(user).__name__}"
            enclosing = _enclosing_function(tree, getattr(read, "lineno", 0))
            if enclosing is not None and _is_thin_accessor(enclosing):
                return "alias", "thin_accessor_return"
            return "bare", f"forwarded_via_{forwarded}_no_action_use"
        if isinstance(parent, ast.Return):
            enclosing = _enclosing_function(tree, getattr(read, "lineno", 0))
            if enclosing is not None and _is_thin_accessor(enclosing):
                return "alias", "thin_accessor_return"
            return "bare", "returned_from_non_accessor"
        parent = parents.get(id(parent))
        hops += 1
    return "bare", "unclassified_parent_chain"


def _is_thin_accessor(fn: ast.FunctionDef) -> bool:
    """薄访问器＝函数体只有 ``Return``，或「``if X is None: return None`` ＋ ``Return``」。"""

    kinds = {type(stmt).__name__ for stmt in fn.body}
    if kinds == {"Return"}:
        return True
    if kinds <= {"If", "Return"} and len(fn.body) <= 2:
        for stmt in fn.body:
            if isinstance(stmt, ast.If) and len(stmt.body) == 1:
                if not isinstance(stmt.body[0], ast.Return):
                    return False
                continue
            if not isinstance(stmt, (ast.Return, ast.If)):
                return False
        return True
    return False


def _enclosing_function(tree: ast.AST, lineno: int):
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= lineno <= (node.end_lineno or node.lineno)
    ]
    return max(candidates, key=lambda fn: fn.lineno) if candidates else None


def _enclosing_class(tree: ast.AST, lineno: int) -> str:
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.lineno <= lineno <= (node.end_lineno or node.lineno)
    ]
    return max(candidates, key=lambda c: c.lineno).name if candidates else "<module>"


def _face_of(rel: str) -> str:
    """点位住在哪张面。判定权只在**产品面**，其余面是披露——把测试里的断言当成消费者，
    会让「只有测试读它」伪装成「被产品消费」，那是本件最贵的一种假读数。"""

    if rel.startswith("taiji/") or rel.startswith("seed/"):
        return "product"
    if rel.startswith("scripts/training/"):
        return "training"
    if rel.startswith("scripts/"):
        return "scripts"
    if rel.startswith("tests/"):
        return "tests"
    return "other"


def scan_keys(root: Path, keys: tuple[str, ...]) -> dict[str, object]:
    per_key: dict[str, dict[str, list[dict[str, object]]]] = {
        key: {"writes": [], "alias_reads": [], "action_reads": [], "bare_reads": []} for key in keys
    }
    parsed = 0
    for path in tracked_python(root):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        parsed += 1
        lines = source.splitlines()
        parents = _parents(tree)
        rel = path.relative_to(root).as_posix()
        for key in keys:
            aliases = _key_aliases(key)
            bucket = per_key[key]
            for node, kind in _write_nodes(tree, aliases):
                fn = _enclosing_function(tree, node.lineno)
                bucket["writes"].append(
                    {
                        "file": rel,
                        "face": _face_of(rel),
                        "line": node.lineno,
                        "kind": kind,
                        "function": fn.name if fn is not None else "<module>",
                        "class": _enclosing_class(tree, node.lineno),
                        "source": (lines[node.lineno - 1].strip()[:160] if lines else ""),
                    }
                )
            for node, kind in _read_nodes(tree, aliases):
                layer, evidence = _classify_read(node, tree, parents)
                fn = _enclosing_function(tree, node.lineno)
                row = {
                    "file": rel,
                    "face": _face_of(rel),
                    "line": node.lineno,
                    "kind": kind,
                    "layer": layer,
                    "evidence": evidence,
                    "function": fn.name if fn is not None else "<module>",
                    "class": _enclosing_class(tree, node.lineno),
                    "source": (lines[node.lineno - 1].strip()[:160] if lines else ""),
                }
                bucket[
                    {"alias": "alias_reads", "action": "action_reads"}.get(layer, "bare_reads")
                ].append(row)
    return {"files_parsed": parsed, "per_key": per_key}


def judge(
    face: dict[str, object], reachability: dict[str, dict[str, str]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, buckets in face["per_key"].items():  # type: ignore[union-attr]
        writers = len(buckets["writes"])
        alias = len(buckets["alias_reads"])
        action = len(buckets["action_reads"])
        bare = len(buckets["bare_reads"])
        if writers and not (alias or action or bare):
            verdict = "write_only"
        elif action:
            verdict = "consumed_with_action"
        elif alias or bare:
            verdict = "alias_only" if alias else "read_without_writer"
        else:
            verdict = "key_absent_entirely"

        # 判级只按**产品面**，且只按**字段读**（局部名读另计，见 `_in_face`）。
        prod_writers = _in_face(buckets["writes"], "product")
        prod_field_alias = _in_face(buckets["alias_reads"], "product", field_only=True)
        prod_field_action = _in_face(buckets["action_reads"], "product", field_only=True)
        prod_field_bare = _in_face(buckets["bare_reads"], "product", field_only=True)
        prod_local_reads = (
            _in_face(buckets["action_reads"], "product", field_only=False)
            + _in_face(buckets["alias_reads"], "product", field_only=False)
            + _in_face(buckets["bare_reads"], "product", field_only=False)
        )
        nonprod_action = (
            _in_face(buckets["action_reads"], None, field_only=True) - prod_field_action
        )
        nonprod_alias = _in_face(buckets["alias_reads"], None, field_only=True) - prod_field_alias
        all_sites = writers + alias + action + bare
        if all_sites == 0:
            product_verdict = "key_absent_entirely"
        elif prod_field_action:
            product_verdict = "consumed_in_product"
        elif prod_field_alias:
            product_verdict = "alias_only_in_product"
        elif prod_field_bare:
            product_verdict = "unclassified_read_in_product"
        elif prod_local_reads:
            product_verdict = "produced_and_used_locally_only"
        elif prod_writers:
            product_verdict = "write_only_in_product"
        else:
            product_verdict = "read_only_outside_product"

        missing = [
            f"{row['file']}:{row['line']}"
            for row in buckets["action_reads"]
            if row["face"] == "product"
            and row["kind"] != "bare_name"
            and f"{key}@{row['file']}:{row['line']}" not in reachability.get(key, {})
        ]
        rows.append(
            {
                "key": key,
                "writer_sites": writers,
                "layer2_alias_reads": alias,
                "layer3_action_reads": action,
                "unclassified_reads": bare,
                "verdict": verdict,
                "product_writer_sites": prod_writers,
                "product_field_layer2_alias_reads": prod_field_alias,
                "product_field_layer3_action_reads": prod_field_action,
                "product_unclassified_field_reads": prod_field_bare,
                "product_local_name_reads": prod_local_reads,
                "nonproduct_field_layer3_action_reads": nonprod_action,
                "nonproduct_field_layer2_alias_reads": nonprod_alias,
                "product_verdict": product_verdict,
                "reachability_registered": prod_field_action - len(missing),
                "reachability_missing_for": missing,
                "reachability": reachability.get(key, {}),
                "action_read_sites": [
                    f"{row['file']}:{row['line']}@{row['function']}"
                    for row in buckets["action_reads"]
                    if row["face"] == "product" and row["kind"] != "bare_name"
                ],
                "nonproduct_action_read_sites": [
                    f"{row['file']}:{row['line']}@{row['function']}"
                    for row in buckets["action_reads"]
                    if row["face"] != "product" and row["kind"] != "bare_name"
                ][:12],
            }
        )
    rows.sort(key=lambda row: row["key"])
    return rows


def _in_face(
    items: list[dict[str, object]], face_name: str | None, field_only: bool | None = None
) -> int:
    """按面（可选）与"字段读 vs 局部名读"（可选）计数。

    ``field_only=True`` 只数**字段读**（``obj.k``／``d["k"]``／``d.get("k")``）——
    判"存下来的值有没有人消费"要看这一类；``field_only=False`` 只数**局部名读**。
    """

    total = 0
    for row in items:
        if face_name is not None and row["face"] != face_name:
            continue
        is_local = row["kind"] == "bare_name"
        if field_only is True and is_local:
            continue
        if field_only is False and not is_local:
            continue
        total += 1
    return total


def build_report(
    root: Path, keys: tuple[str, ...], reachability: dict[str, dict[str, str]]
) -> dict:
    face = scan_keys(root, keys)
    rows = judge(face, reachability)
    if all(row["verdict"] == "key_absent_entirely" for row in rows):
        raise ValueError(
            "13 枚键**一枚都没找到**：这台仪器今天什么都没测到。先确认扫描面（版本控制里的 .py）"
            "与键名写法，零命中不许读成「这些键都不存在」或「没有写-only 键」"
        )
    absent = [row["key"] for row in rows if row["verdict"] == "key_absent_entirely"]
    unregistered = [row["key"] for row in rows if row["reachability_missing_for"]]
    stray = [
        row["key"]
        for row in rows
        if row["verdict"] not in VERDICTS or row["product_verdict"] not in PRODUCT_VERDICTS
    ]
    if stray:
        raise ValueError(f"判级词不在声明的词典里（仪器自己走歪了）：{stray}")
    return {
        "format": REPORT_FORMAT,
        "guard": "G-N1c-2（PLAN-N1-03 §4）",
        "face": "tracked .py（`git ls-files -- *.py`），AST 逐键分类；写＝赋值／字典键／构造关键字参数",
        "files_parsed": face["files_parsed"],
        "keys_named": list(keys),
        "key_origins": {key: KEY_ORIGINS[key] for key in keys},
        "extra_names": {key: list(names) for key, names in EXTRA_NAMES.items() if key in keys},
        "rows": rows,
        "tally": {
            "write_only": sum(1 for r in rows if r["verdict"] == "write_only"),
            "alias_only": sum(1 for r in rows if r["verdict"] == "alias_only"),
            "consumed_with_action": sum(1 for r in rows if r["verdict"] == "consumed_with_action"),
            "read_without_writer": sum(1 for r in rows if r["verdict"] == "read_without_writer"),
            "key_absent_entirely": len(absent),
        },
        "product_tally": {
            word: sum(1 for r in rows if r["product_verdict"] == word) for word in PRODUCT_VERDICTS
        },
        "keys_absent_entirely": absent,
        "keys_with_unregistered_reachability": unregistered,
        "limits": [
            "**跨名转发只按已登记的 `EXTRA_NAMES` 折叠**（本件出版 `extra_names` 说明折了哪些）；"
            "没登记的别名看不见——`last_decision` 那枚若被人改名，本件会把它读成「没人读」，"
            "契约测里钉了那条属性的原文，改名即红",
            "**经生成式序列化（如 ``dataclasses.asdict``）间接落到信封里的字段，静态读看不见**："
            "所以 ``produced_and_used_locally_only`` 只说「产品码里没有对该字段的读取点」，"
            "**不许**被读成「这个值没被存下来」或「这个值没被任何人用到」（`expected_reward` 就是这一档）",
            "判定权只在**产品面**（`taiji/`＋`seed/`）：训练脚本／评测脚本／测试里读到一个键，"
            "**不**算产品消费它——所以本件同时出版 `product_verdict` 与全仓 `verdict` 两栏，"
            "并把非产品面的第三层点位单列在 `nonproduct_action_read_sites`（截前 12 处）。"
            "`read_only_outside_product` 这个词是专门留给「只有测试在读」那一档的，"
            "它既不许被读成 `write_only_in_product`，也不许被读成「已消费」",
            "第 3 层的判定只看**直接语境**与**一次名字转发**（``decision = self.x`` 之后被用于分支／调用）；"
            "两次以上转发、或经容器传递后再用的，会被记成 ``unclassified_reads``，"
            "**不**自动算「没有消费者」也不自动算「有」",
            "默认位可达性**不由本件计算**：它要求 ``--reachability-ledger`` 逐枚登记"
            "（登记内容＝该点位在默认装配下会不会被走到，出处要点名到文件与旗标）；"
            "未登记的键会出现在 ``keys_with_unregistered_reachability``，"
            "此时既不许宣布 ``write_only`` 也不许宣布「默认链上没人消费」",
            "接收者是裸参数名（没有绑定到某个类）时静态读认不出类型——这是已知盲区，"
            "与 ``audit_taiji_episodic_read_sharing_face`` 里 ``store`` 那一支同形",
        ],
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _load_ledger(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    ledger: dict[str, dict[str, str]] = {}
    for key, entries in payload.items():
        if not isinstance(entries, dict):
            raise ValueError(f"可达性账本第 {key} 项必须是 点位->结论 的字典")
        ledger[str(key)] = {str(point): str(conclusion) for point, conclusion in entries.items()}
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="扫描根（契约测指向临时目录用）")
    parser.add_argument(
        "--out-report",
        default="reports/taiji_write_only_key_face_20261011.json",
        help="读数件落点；已存在则拒绝覆盖（要覆盖给 --overwrite）",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reachability-ledger",
        type=Path,
        default=None,
        help="JSON：键 -> {文件:行 -> 默认位可达性结论与出处}；缺省＝全部未登记（如实出版）",
    )
    args = parser.parse_args()
    try:
        report = build_report(Path(args.root), NAMED_KEYS, _load_ledger(args.reachability_ledger))
    except (ValueError, json.JSONDecodeError) as exc:
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
                "files_parsed": report["files_parsed"],
                "tally": report["tally"],
                "product_tally": report["product_tally"],
                "keys_with_unregistered_reachability": report[
                    "keys_with_unregistered_reachability"
                ],
                "out": out.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
