"""校验 `plans/` 下 Markdown 文档的相对链接：文件存在 ＋ 锚点能解析。

**为什么需要这道门**：2026-10-08（㊵-518）修`08_UPSTREAM_SYNC_PLAYBOOK.md` 的超长行格式
（64 行超长，最长 25998 字符＝一整段挤成一行→ 13 行，最长 3788 字符）后，
**切分改变了行号，也可能改变标题锚点**——而本仓大量链接是带锚点的中文 slug
（形如 `../../reference/VISION_FUTURE_TECHNOLOGY.md#22-长期运行的补全设计...`）。
**锚点失配不会让任何门变红**，只会让人点击后落在文档顶部、找不到那段话——
这与已记的「守卫静默失效」是同一族。

用法：
    python scripts/training/check_plans_markdown_links.py            # 全量体检并打印报告
    python scripts/training/check_plans_markdown_links.py --json      # 机器可读
退出码：0＝无坏链；1＝有坏链（逐条点名）。
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLANS = REPO / "plans"

#: `[文字](目标)` —— 目标可为空（纯锚点）或以 `#` 分成路径与锚点两段。
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: 显式跳过的目标前缀（非仓库内相对链接）。
SKIP_PREFIXES = ("http://", "https://", "mailto:", "ftp://")

#: **现行层**＝承担导航职责的文档；只有这里的坏链是真问题。
#: `plans/archive/` 是冻结历史（owner 2026-10-07 蒸馏政策的存放处，
#: 墓碑见 `plans/reference/DISTILLATION_TOMBSTONE_20261007.md`），
#: 它里面指向已蒸馏件的链接**是政策预期而非缺陷**——按层分档是为了
#: **不让455 条常红淹没现行层的真坏链**（常红的门等于没有门）。
CURRENT_LAYER_PREFIXES = ("plans/active/", "plans/README.md")


def is_current_layer(source: Path) -> bool:
    """该源文件是否属于现行层（承担导航职责）。"""
    relative = source.relative_to(REPO).as_posix()
    return any(relative.startswith(prefix) for prefix in CURRENT_LAYER_PREFIXES)


def slugify(heading: str) -> str:
    """把标题转成 GitHub 风格的锚点 slug。

        GitHub 的规则：去标点、小写化、空格转 `-`、非 ASCII 保留。
        `plans/` 下的标题绝大多数是中文 ⇒ 这条规则决定了锚点能不能对上，
        所以**必须按 GitHub 的实际行为写，不能自创**。

        **实测校准（2026-10-08）**：规则必须用仓库里**现存的锚点**反推，
    不能自创。本轮在同一个方向上错了两次，两次都被现存锚点当场否证：
        - 错法①以为全角括号 `（）` 属「文字」应当保留 ⇒ 14 条锚点全部变**失效**。
        - 错法②以为连字符 `2026-09-17` 的 `-`（`Pd`）应当保留 ⇒ 又失效。
        - 终版＝**Unicode `P*`／`S*` 全剥（含 `Pd` 连字符）**，
          决定性验证＝三条现存锚点逐字符吻合（3/3）：
          `### 2.1 统一认知闭环验收原则（2026-09-17用户确认）`
          → `21-统一认知闭环验收原则20260917用户确认`（括号**与**连字符都去掉）；
          `### 19.12 D8 (a2)后的推荐候选：…`
          → `1912-d8-a2后的推荐候选从发射延续转向可学习内容绑定`。
        ⇒ 判据：**校准锚点规则要拿≥3 条现存锚点逐字符对**，
        只对1 条容易把「巧合」当成「规则」。
        **一个产出假红的门和一个不产红的门同样有害**——
        前者会让人把真问题当噪音忽略（本轮就差点让我去改 14 条本来正确的链接）。
    """
    text = heading.strip().lower()
    # 去掉行内标记（`##` 标题里的 `**`、反引号等）
    text = re.sub(r"[`*_~]", "", text)
    # 去掉**变体选择符与零宽字符**（U+FE0E/F、U+200B 等）：
    # `⚠️` 被剥成`⚠`（它属 `S*`），但**变体选择符 U+FE0F 属 `Mn`（非间距标记）**，
    # 不在 P/S 之列⇒ 会在 slug 开头留下一个不可见字符。
    # 实测踩到过：`ARCHITECTURE_COMPROMISE_ORIGINS.md` 的
    # `## ⚠️ 架构本源定性（…）` 被算成 `️-架构本源定性…`（开头有幽灵字符）。
    # GitHub 不会产生这种 slug ⇒ 显式剔除。
    text = re.sub("[\ufe00-\ufe0f\u200b-\u200f\u2060]", "", text)
    # 去掉标点与符号（Unicode 类别 P*与 S*），保留文字（含中日韩）
    out: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            continue
        out.append(char)
    text = "".join(out)
    # 空格与全角空格转 `-`
    text = re.sub(r"[\s\u3000]+", "-", text)
    return text.strip("-")


def strip_code_blocks(text: str) -> str:
    """把围栏代码块与行内代码替换成等长空白，**保持所有行号不变**。

    必须这么做：代码块里出现 `[x](field_state)` 这类形态会被链接正则误抓
    （实测 `COMPREHENSIVE_NEURON_ARCHITECTURE_PLAN.md:1356` 的
    `h = h + self.field_read_layers[i](field_state)` 就是一行Python，
    被报成「链接指向不存在的 field_state」）。
    **一个把代码当链接的门，它的红是假的** ⇒ 先测准再谈门。
    行号保持不变，报错才能指到原文那一行。
    """
    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(" " * len(line))
            continue
        if in_fence:
            out.append(" " * len(line))
            continue
        out.append(re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), line))
    return "\n".join(out)


def collect_anchors(path: Path) -> set[str]:
    """收集一个 Markdown 文件里所有可锚定的 slug（标题与显式 `{#id}`）。"""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return anchors
    for line in text.split("\n"):
        match = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if match:
            base = slugify(match.group(2))
            # GitHub 对重复标题追加 `-1`、`-2`…
            count = seen.get(base, 0)
            seen[base] = count + 1
            anchors.add(base if count == 0 else f"{base}-{count}")
            continue
        explicit = re.search(r"\{#([^}]+)\}\s*$", line)
        if explicit:
            anchors.add(explicit.group(1).strip())
    return anchors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    files = sorted(PLANS.rglob("*.md"))
    anchor_cache: dict[Path, set[str]] = {}
    problems: list[dict[str, object]] = []
    checked = 0

    for source in files:
        try:
            raw = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        text = strip_code_blocks(raw)
        for match in LINK_RE.finditer(text):
            target = match.group(1).strip()
            if not target or target.startswith(SKIP_PREFIXES):
                continue
            if target.startswith("#"):
                path_part, anchor = "", target[1:]
            elif "#" in target:
                path_part, anchor = target.split("#", 1)
            else:
                path_part, anchor = target, ""
            if path_part.startswith("/") or ":" in path_part.split("/")[0]:
                continue  # 绝对路径或带 scheme 的写法，本门不管
            checked += 1
            resolved = (source.parent / path_part).resolve() if path_part else source
            line_no = text[: match.start()].count("\n") + 1
            scope = "current" if is_current_layer(source) else "archive"
            if not resolved.is_file():
                problems.append(
                    {
                        "kind": "missing_file",
                        "scope": scope,
                        "policy_expected": scope == "archive",
                        "source": source.relative_to(REPO).as_posix(),
                        "line": line_no,
                        "target": target,
                    }
                )
                continue
            if anchor:
                anchors = anchor_cache.get(resolved)
                if anchors is None:
                    anchors = collect_anchors(resolved)
                    anchor_cache[resolved] = anchors
                if anchor not in anchors:
                    problems.append(
                        {
                            "kind": "missing_anchor",
                            "scope": scope,
                            "policy_expected": scope == "archive",
                            "source": source.relative_to(REPO).as_posix(),
                            "line": line_no,
                            "target": target,
                            "resolved": resolved.relative_to(REPO).as_posix(),
                        }
                    )

    if args.json:
        print(
            json.dumps(
                {
                    "files": len(files),
                    "links_checked": checked,
                    "problems": problems,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        current = [p for p in problems if p["scope"] == "current"]
        archive = [p for p in problems if p["scope"] == "archive"]
        print(f"扫描 {len(files)} 个 Markdown 文件、校验 {checked} 条仓库内链接")
        print(
            f"**现行层**（{'/'.join(CURRENT_LAYER_PREFIXES)}）：{len(current)} 条坏链"
            " ← 这一档是门禁口径"
        )
        print(f"**归档层**（plans/archive/）：{len(archive)} 条 —— **政策预期，不计门禁**")
        if current:
            print("\n现行层坏链明细：")
            for item in current[:40]:
                print(f"  [{item['kind']}] {item['source']}:{item['line']} → {item['target']}")
            if len(current) > 40:
                print(f"  …另有 {len(current) - 40} 条")
        else:
            print("现行层**无坏链**。")
        if archive:
            by_kind: dict[str, int] = {}
            for item in archive:
                key = str(item["kind"])
                by_kind[key] = by_kind.get(key, 0) + 1
            summary = "、".join(f"{k} {v} 条" for k, v in sorted(by_kind.items()))
            print(f"\n归档层明细仅供追溯（{summary}）：")
            for item in archive[:10]:
                print(f"  {item['source']}:{item['line']} → {item['target']}")
            if len(archive) > 10:
                print(f"  …另有 {len(archive) - 10} 条（`--json` 看全量）")
            print(
                "  口径依据：owner 2026-10-07 蒸馏政策（墓碑见 "
                "plans/reference/DISTILLATION_TOMBSTONE_20261007.md）——"
                "实验过程文档蒸馏后删除，指向它们的链接是政策预期而非缺陷。"
            )
    # **门禁只对现行层负责**：归档层常红会让门失去分辨力（常红的门等于没有门）。
    return 1 if any(p["scope"] == "current" for p in problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
