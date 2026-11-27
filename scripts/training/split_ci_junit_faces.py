"""ADR-S3（C-4）：从全量 junit（report.xml）派生 native / seed 两个面。

背景：S3 去重后 Ubuntu 腿只保留一趟全量 pytest（pytest tests/ --cov，
junit 写 report.xml），`tests/taiji_native` 与 `tests/seed` 两个面**不再各跑
一趟 pytest**，改由同一趟执行的 report.xml 派生。派生文件头写明
`derived from report.xml … not a separate pytest run`——任何人打开都能看出
它不是独立执行面。上传 artifact 名 native-report.xml / seed-report.xml 不变。

归属算法（终裁，逐条可落地）：
1. `key = classname or name`（classname 为 ``''`` 时是收集失败记录，其 name 是
   点分模块路径——但那由第 1 级前置门禁先行拦截，见下）；
2. `parts = key.split('.')`；循环拼 `"/".join(parts) + ".py"`，命中
   `(repo_root/cand).is_file()` 即定位成功；否则 `parts.pop()` 继续；耗尽 ⇒
   **UNRESOLVED ⇒ exit 3（fail-closed）**；
3. 用 **`Path.is_relative_to()` 判归属（禁止字符串前缀）**：在 `tests/taiji_native/`
   下 ⇒ native 面；在 `tests/seed/` 下 ⇒ seed 面；`tests/seed_platform/`、
   `tests/resonance/`、`tests/` 根 ⇒ 两面都不进（与去重前 step25/26 语义一致）。
   尾点陷阱（`tests.seed` 字符串前缀会误吞兄弟目录 `tests.seed_platform`）由此
   结构性关闭。

**两级自校验（顺序不可颠倒）**：
- 第 1 级（前置门禁，先跑）：全量面 report.xml 中出现 **collection failure 形态**的
  ``<testcase>`` ⇒ 打印这些模块清单 + **exit 5**。这条把「collection error ⇒ 默默
  少跑一批模块」变成显式红，是 S1 与 S6 的共同安全网。必须先做——否则只要本仓
  存在 collection error，第 2 级的三面之和必然不等，产生假红。
  **语义校正（2026-10-26，基线 artifact 实证 + 主理人裁决 C-4 修正）**：ADR §3.4 初稿把
  门禁写成「classname=='' 即 exit 5」，但真实基线 run 40171918472（conclusion=success）的
  report.xml 里本就有 4 条 classname=='' 记录——它们是模块级 ``pytest.importorskip`` /
  ``pytest.skip(..., allow_module_level=True)``（tests/test_taiji_browser_environment.py:14、
  tests/test_taiji_browser_tool_loop.py:18、tests/taiji_native/test_r2_fast_ledger_resume_v4_contract.py:10、
  tests/taiji_native/test_r2_worker_namespace_integration.py:16），junit 形态为
  classname=='' + name=点分模块路径 + **<skipped>**，**合法绿跳**。字面执行会让每次
  CI 永久假红。故按 08-test.md §12.0 的两行表裁决修正：判据是**子元素**，不是 classname——
  classname=='' 且 `<error message="collection failure">`（08-test.md §10 实证的 collection
  error 形态）/<failure>/无子节点等未知形态 ⇒ exit 5（未知形态 fail-closed）；
  classname=='' 且 `<skipped>` ⇒ **合法模块级 skip：排除出面，单独计数
  （module_skipped），不判红**（排除出面而非按模块路径归属——主理人裁决，2026-10-26）。
- 第 2 级（第 1 级通过后再做）：**分母不用 pytest 的 tests= 属性**（它把模块级 skip 行也
  计入：基线 native 面原生 junit tests=3903 但唯一 node-id 只有 3901，全量 4580 vs 4576）。
  用「唯一 node-id 数（classname+name 去重）」做守恒：uid(native) + uid(seed) + uid(other)
  == uid(全量) 严格相等（基线锚 3901+185+490=4576），且 uid(native) > 0 ∧ uid(seed) > 0
  （悄悄吞面 ⇒ 立刻红）⇒ 违反 **exit 4**。模块级 skip 数单独记账打印，不进守恒和。

派生文件：深拷贝 `<testsuites>`/`<testsuite>` 结构与 testcase 全部子节点
（`<system-out>`/`<system-err>`/`<failure>`/`<error>`/`<skipped>`/`<properties>`），
只筛 `<testcase>`，**重算** testsuite 与 testsuites 的 tests/failures/errors/
skipped/time（缺字段补 0，不得留旧值）。

依赖：stdlib only。幂等：同一 report.xml ⇒ 字节级相同的两个输出。
退出码总表见 ADR §3.8：0 成功 / 2 源文件缺失或不可解析 / 3 testcase 无法归属 /
4 第 2 级自校验失败 / 5 第 1 级前置门禁失败（打印模块清单）。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

DERIVED_COMMENT = (
    "<!-- derived from report.xml by split_ci_junit_faces.py; " "not a separate pytest run -->"
)
XML_DECL = '<?xml version="1.0" encoding="utf-8"?>'

NATIVE_DIR = ("tests", "taiji_native")
SEED_DIR = ("tests", "seed")

# testsuite/testsuites 上需要重算的计数字段
COUNTERS = ("tests", "failures", "errors", "skipped")


def resolve_file(key: str, repo_root: Path) -> Path | None:
    """classname（点分模块路径）→ 磁盘真实 .py 文件。耗尽 ⇒ None（UNRESOLVED）。"""
    parts = key.split(".")
    while parts:
        cand = "/".join(parts) + ".py"
        path = repo_root / cand
        if path.is_file():
            return path
        parts.pop()
    return None


def face_of(path: Path, repo_root: Path) -> str | None:
    """磁盘文件路径 → 'native' / 'seed' / None（两面都不进）。
    必须 is_relative_to（逐目录分量比较），禁止字符串前缀——
    `tests/seed_platform` 不是 `tests/seed` 的子孙，尾点陷阱由此关闭。"""
    native_root = repo_root.joinpath(*NATIVE_DIR)
    seed_root = repo_root.joinpath(*SEED_DIR)
    if path.is_relative_to(native_root):
        return "native"
    if path.is_relative_to(seed_root):
        return "seed"
    return None


def _suite_attr(suite: ET.Element, name: str) -> int:
    raw = suite.get(name)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


def _recount(suite: ET.Element) -> None:
    """重算单个 <testsuite> 的计数字段（缺字段补 0，不得留旧值）。"""
    cases = suite.findall("testcase")
    suite.set("tests", str(len(cases)))
    suite.set("failures", str(sum(1 for tc in cases if tc.find("failure") is not None)))
    suite.set("errors", str(sum(1 for tc in cases if tc.find("error") is not None)))
    suite.set("skipped", str(sum(1 for tc in cases if tc.find("skipped") is not None)))
    total = 0.0
    for tc in cases:
        with contextlib.suppress(ValueError):
            total += float(tc.get("time", 0))
    suite.set("time", f"{total:.6f}")


def _derive_face(source_root: ET.Element, keep_ids: set[int]) -> ET.Element:
    """深拷贝结构、只筛命中的 <testcase>，重算两级计数字段。"""
    new_root = ET.Element(source_root.tag, dict(source_root.attrib))
    for suite in source_root.findall("testsuite"):
        new_suite = ET.SubElement(new_root, suite.tag, dict(suite.attrib))
        # 保留 testsuite 下的非 testcase 子节点（properties 等）
        for child in suite:
            if child.tag != "testcase":
                new_suite.append(copy.deepcopy(child))
        for tc in suite.findall("testcase"):
            if id(tc) in keep_ids:
                new_suite.append(copy.deepcopy(tc))
        _recount(new_suite)
    # 根级 testsuites 计数 = 各 suite 之和
    agg = {name: 0 for name in COUNTERS}
    total_time = 0.0
    for suite in new_root.findall("testsuite"):
        for name in COUNTERS:
            agg[name] += _suite_attr(suite, name)
        with contextlib.suppress(ValueError):
            total_time += float(suite.get("time", 0))
    for name, value in agg.items():
        new_root.set(name, str(value))
    new_root.set("time", f"{total_time:.6f}")
    return new_root


def serialize(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return f"{XML_DECL}\n{DERIVED_COMMENT}\n{body}\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive native/seed junit faces from the full junit report."
    )
    parser.add_argument("--source", required=True, help="全量 junit（report.xml）")
    parser.add_argument("--native", required=True, help="派生输出：native-report.xml")
    parser.add_argument("--seed", required=True, help="派生输出：seed-report.xml")
    parser.add_argument(
        "--repo-root",
        required=True,
        help="仓库根（必填且显式：classname 是相对 rootdir 的点分路径，不得依赖 cwd）",
    )
    args = parser.parse_args()

    source = Path(args.source)
    repo_root = Path(args.repo_root).resolve()

    if not source.is_file():
        print(f"::error::源文件缺失：{source}")
        return 2
    try:
        tree = ET.parse(source)
    except ET.ParseError as exc:
        print(f"::error::源文件不可解析：{source}（{exc}）")
        return 2
    source_root = tree.getroot()

    testcases = source_root.findall(".//testcase")
    if not testcases:
        print("::error::全量面没有任何 <testcase>——report.xml 为空或结构异常")
        return 4

    # ── 第 1 级（前置门禁，先跑）：全量面不允许出现 collection failure 形态的记录 ──
    # 形态学（实证 + 主理人裁决 C-4 修正，见模块 docstring）：
    #   classname=='' + <error message="collection failure"> ⇒ collection error，exit 5；
    #   classname=='' + <skipped> ⇒ 合法模块级 skip（importorskip / allow_module_level）：
    #     排除出面 + 单独计数 module_skipped，不判红；
    #   classname=='' + 其他形态（<failure>/无子节点）⇒ 未知，fail-closed exit 5。
    module_skips: list[ET.Element] = []
    normal_cases: list[ET.Element] = []
    broken: list[ET.Element] = []
    for tc in testcases:
        if (tc.get("classname") or "").strip():
            normal_cases.append(tc)
            continue
        if len(tc) == 1 and tc[0].tag == "skipped":
            module_skips.append(tc)
        else:
            broken.append(tc)
    if broken:
        print(
            '::error::全量面存在 collection error 记录（classname=""）——'
            "这批模块在本趟执行里没有被收集，少跑的面已伪装成绿。模块清单："
        )
        for tc in broken:
            print(f"::error::collection failure: {tc.get('name')}")
        print(
            "模块清单（"
            + str(len(broken))
            + " 个）："
            + ", ".join(tc.get("name") or "<unnamed>" for tc in broken)
        )
        return 5

    # ── 归属（classname → 磁盘文件 → is_relative_to）；模块级 skip 已排除出面 ──
    faces: dict[str, list[ET.Element]] = {"native": [], "seed": [], "other": []}
    for tc in normal_cases:
        key = tc.get("classname") or tc.get("name") or ""
        path = resolve_file(key, repo_root)
        if path is None:
            print(
                f"::error::testcase 无法归属到磁盘文件（UNRESOLVED）：key={key!r} "
                f"name={tc.get('name')!r} —— fail-closed，不猜测"
            )
            return 3
        face = face_of(path, repo_root)
        bucket = face if face is not None else "other"
        faces[bucket].append(tc)

    # 分母口径（主理人裁决，api-tester 基线实证）：唯一 node-id（classname+name 去重），
    # 不用 pytest 的 tests= 属性——它把模块级 skip 行也计入（基线 native 面原生 junit
    # tests=3903 但唯一 node-id 3901；全量 4580 vs 4576），用它守恒会恒差 4。
    def _uid(items: list[ET.Element]) -> set[tuple[str, str]]:
        return {(tc.get("classname") or "", tc.get("name") or "") for tc in items}

    uid_native = len(_uid(faces["native"]))
    uid_seed = len(_uid(faces["seed"]))
    uid_other = len(_uid(faces["other"]))
    uid_total = len(_uid(normal_cases))
    n_total = len(testcases)

    # 尾点陷阱疫苗：面上的每个文件必须真的落在对应目录下。
    native_root = repo_root.joinpath(*NATIVE_DIR)
    seed_root = repo_root.joinpath(*SEED_DIR)
    for face_name, face_dir in (("native", native_root), ("seed", seed_root)):
        for tc in faces[face_name]:
            key = tc.get("classname") or tc.get("name") or ""
            resolved = resolve_file(key, repo_root)
            if resolved is None or not resolved.is_relative_to(face_dir):
                print(
                    f"::error::{face_name} 面混入非对应目录文件：key={key!r} resolved={resolved!r}"
                )
                return 4

    # ── 第 2 级（第 1 级通过后再做）：唯一 node-id 三面之和守恒 + 两面都非空 ──
    if uid_native + uid_seed + uid_other != uid_total:
        print(
            "::error::自校验失败：uid(native)+uid(seed)+uid(other) != uid(全量) "
            f"（{uid_native}+{uid_seed}+{uid_other} != {uid_total}）"
        )
        return 4
    if uid_native == 0 or uid_seed == 0:
        print(
            "::error::自校验失败：某个面被悄悄吞掉 "
            f"（uid(native)={uid_native}, uid(seed)={uid_seed}）——归属算法或 rootdir 变了"
        )
        return 4

    native_root_el = _derive_face(source_root, {id(tc) for tc in faces["native"]})
    seed_root_el = _derive_face(source_root, {id(tc) for tc in faces["seed"]})

    Path(args.native).write_text(serialize(native_root_el), encoding="utf-8")
    Path(args.seed).write_text(serialize(seed_root_el), encoding="utf-8")

    print(
        f"faces derived: rows_total={n_total} uid_total={uid_total} "
        f"uid_native={uid_native} uid_seed={uid_seed} uid_other={uid_other} "
        f"（other=tests/seed_platform、tests/resonance、tests/ 根，"
        "与去重前 step25/26 语义一致，两面都不进）；"
        f"module_skipped={len(module_skips)}"
        + (
            "（" + ", ".join(tc.get("name") or "<unnamed>" for tc in module_skips) + "）"
            if module_skips
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
