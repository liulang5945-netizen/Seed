"""报告／清单的"运行条件"必填门。

判据来源：[能力卡 §13](../../plans/reference/CAPABILITY-CARD_20261103.md) 与
[决策台账 D-20261110-05](../../plans/reference/DECISION-LEDGER_20261109.md)：
同一模型、同一输入、同一权重，torch intra-op 线程数不同会改变浮点归约顺序，
末位差异会被摘要放大。因此**任何相等类或内容摘要类字段，都必须随结论记录运行
条件**（至少 torch 线程数；`OMP_NUM_THREADS`／`MKL_NUM_THREADS` 若被设置也要记）。

本模块提供两种用法：

1. 命令行（CI 与本机自检）::

       python scripts/training/check_report_run_conditions.py --check

   扫描 `reports/` 与 `plans/manifests/` 里**已被 git 跟踪**的 JSON：
   带相等／摘要字段、又没有运行条件记录、且不在基线清单里的文件，逐个判失败并
   指出缺哪个字段。旧报告不回写，它们的身份是基线清单里的
   `report_level_claim_not_machine_verifiable`。

2. 写盘点（新脚本必须走这里）::

       from check_report_run_conditions import write_report
       write_report(destination, payload)

   缺运行条件即抛 `MissingRunConditions`，**写盘失败**，不产生半成品文件。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 相等类字段：结论本身是"两块结果一样"。
EQUALS_KEY = re.compile(
    r"^(?:max_abs_(?:delta|diff)|max_delta|byte(?:s)?_identical|bitwise_identical"
    r"|identical|equal|equal_fields|same_bytes|unchanged)$"
    r"|_identical$",
    re.IGNORECASE,
)

#: 内容摘要字段：结论依赖浮点结果，线程预算会改末位。
#: 注意**不含** `*_sha256` 与 `fixture_digest`／`module_digest` 一类的文件身份哈希，
#: 它们是字节哈希，与归约顺序无关。
DIGEST_KEY = re.compile(
    r"^(?:content_digest|probe_digest|checkpoint_digest|param_digest|state_digest"
    r"|trace_digest|logits_digest|weight_digest|candidate_digest|digest)$",
    re.IGNORECASE,
)

#: 运行条件：`threads`／`*_threads` 键，或显式环境变量／调用名。
CONDITION_KEY = re.compile(r"(?:^|_)threads$", re.IGNORECASE)
CONDITION_TOKENS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "set_num_threads",
)

SCAN_ROOTS = ("reports", "plans/manifests")
BASELINE_PATH = REPO_ROOT / "plans/manifests/report_run_conditions_baseline_20261110.json"
BASELINE_SCHEMA = "report_run_conditions_baseline_v1"
LEGACY_STATUS = "report_level_claim_not_machine_verifiable"


class MissingRunConditions(RuntimeError):
    """一份带相等／摘要字段的报告，缺少运行条件记录。"""

    def __init__(self, destination: str | Path, fields: Iterable[str]) -> None:
        self.destination = str(destination)
        self.fields = sorted(set(fields))
        super().__init__(
            f"{self.destination}: 字段 {self.fields} 依赖浮点归约顺序，但报告没有记录运行条件；"
            "至少补 threads（torch.get_num_threads()），设置过 OMP/MKL 时一并补上"
        )


@dataclass(frozen=True)
class Survey:
    """一份 JSON 里的三类字段。"""

    equality: tuple[str, ...]
    digest: tuple[str, ...]
    conditions: tuple[str, ...]

    @property
    def claims(self) -> tuple[str, ...]:
        return self.equality + self.digest

    @property
    def needs_conditions(self) -> bool:
        return bool(self.claims) and not self.conditions


def _walk(node: Any) -> Iterator[Any]:
    yield node
    if isinstance(node, Mapping):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def survey(payload: Any) -> Survey:
    equality: set[str] = set()
    digest: set[str] = set()
    conditions: set[str] = set()
    for node in _walk(payload):
        if isinstance(node, Mapping):
            for key in node:
                name = str(key)
                if EQUALS_KEY.match(name):
                    equality.add(name)
                elif DIGEST_KEY.match(name):
                    digest.add(name)
                if CONDITION_KEY.search(name):
                    conditions.add(name)
        elif isinstance(node, str):
            for token in CONDITION_TOKENS:
                if token in node:
                    conditions.add(token)
    return Survey(tuple(sorted(equality)), tuple(sorted(digest)), tuple(sorted(conditions)))


def missing_conditions(payload: Any) -> tuple[str, ...]:
    """返回缺失运行条件的字段名；空元组表示合规。"""

    found = survey(payload)
    return found.claims if found.needs_conditions else ()


def assert_run_conditions(destination: str | Path, payload: Any) -> None:
    missing = missing_conditions(payload)
    if missing:
        raise MissingRunConditions(destination, missing)


def write_report(destination: str | Path, payload: Any) -> Path:
    """新报告的唯一写盘入口：先校验运行条件，再落盘（UTF-8、LF 行尾）。"""

    path = Path(destination)
    assert_run_conditions(path, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def tracked_json(repo_root: Path) -> list[Path]:
    """被 git 跟踪的 JSON。未跟踪草稿不参与判红，但仍会出现在基线的重建里。"""

    try:
        listing = subprocess.run(
            ["git", "ls-files", *SCAN_ROOTS],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        listing = [
            str(path.relative_to(repo_root)).replace("\\", "/")
            for root in SCAN_ROOTS
            for path in sorted((repo_root / root).rglob("*.json"))
        ]
    return [repo_root / name for name in listing if name.endswith(".json")]


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    files: dict[str, list[str]] = payload.get("files", {})
    return files


def load_json_lenient(path: Path) -> Any | None:
    """按 BOM 自适应解码（仓库里有少量 UTF-16 遗留 JSON）；解不出就返回 None。"""

    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def scan(repo_root: Path) -> dict[str, list[str]]:
    """返回 {仓库相对路径: 缺失运行条件的字段}。"""

    findings: dict[str, list[str]] = {}
    for path in tracked_json(repo_root):
        payload = load_json_lenient(path)
        if payload is None:
            continue
        missing = missing_conditions(payload)
        if missing:
            key = str(path.relative_to(repo_root)).replace("\\", "/")
            findings[key] = list(missing)
    return findings


def write_baseline(findings: Mapping[str, list[str]], path: Path = BASELINE_PATH) -> Path:
    payload = {
        "schema": BASELINE_SCHEMA,
        "rule": "CAPABILITY-CARD §13 / DECISION-LEDGER D-20261110-05",
        "generated": "2026-11-10",
        "status": LEGACY_STATUS,
        "note": (
            "旧报告一律不回写、不补字段：列在这里只表示它们的结论依赖未记录的运行条件，"
            "按『报告级声明、不可机器复核』引用。新报告若带相等／摘要字段而缺运行条件，"
            "由 check_report_run_conditions.py 判失败。"
        ),
        "count": len(findings),
        "files": dict(sorted(findings.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="只判红（默认动作）")
    parser.add_argument("--write-baseline", action="store_true", help="按当前仓库重建基线清单")
    parser.add_argument("--list-legacy", action="store_true", help="打印基线清单里的文件")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    findings = scan(args.repo_root)
    baseline = load_baseline(args.baseline)

    if args.write_baseline:
        written = write_baseline(findings, args.baseline)
        print(f"baseline written: {written.relative_to(args.repo_root)} ({len(findings)} files)")
        return 0

    if args.list_legacy:
        for name in sorted(baseline):
            print(name)
        print(f"# {len(baseline)} legacy files", file=sys.stderr)
        return 0

    new_violations = {name: fields for name, fields in findings.items() if name not in baseline}
    stale = sorted(set(baseline) - set(findings))

    print(f"scanned={len(tracked_json(args.repo_root))} claim_files={len(findings)}")
    print(f"baseline={len(baseline)} new_violations={len(new_violations)} stale={len(stale)}")
    for name, fields in sorted(new_violations.items()):
        print(f"::error file={name}::缺运行条件：{', '.join(fields)}", file=sys.stderr)
    for name in stale:
        print(
            f"::warning file={name}::基线里有、扫描里没有（改名或删除）；请重建基线",
            file=sys.stderr,
        )
    return 1 if new_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
