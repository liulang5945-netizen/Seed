"""入库读数件的**行尾**棘轮门（DEBT-G115 修法②；㊵-760 落库）。

要守的是一句已经被证伪过一次的话：「入库的 `reports/` 件是纯 LF」。㊵-732 那轮抓到两台仪器写出带
CRLF 的读数件，而**整族里没有任何门为此变红**——也就是说这句到今天仍然只是"我记得我修过"。
本仪器把它变成一个能为假的读数。

两个取法必须分开，混用会得出相反的答案（这是本轮现读到的事实，不是推测）：

* **索引字节面**（`git show :<路径>`）＝真正入库的那份字节。现读 1782 枚 `reports/` 件里**只有 2 枚**
  带 CR（都是散 `\r`、无 `\r\n`），本仪器**只判这一面**。
* **工作树字节面**（直接读盘）＝同一份件在本机 `core.autocrlf` 检出态下的样子，现读 138 枚带 CR。
  这一面**只出版、不判**——它是检出态，不是数据缺陷；拿它当门会把"别人换了行尾检出"算成我的红。

基线形状＝**只许减不许加**（仓里 mypy／B,SIM／覆盖率三处都是这个先例）：现存那两枚是 M 时代落进去的
封存件，把它们重排成 LF 等于**为指标改写历史字节**，所以按名字豁免；一旦有人新写一枚带 CR 的入库件，
`no_new_cr_bearing_tracked_file` 立刻为假、rc=1。豁免集缩小是好事，所以判据写成"子集"而不是"相等"
（相等那条只作为 `equals_current_baseline` 出版，不参与判定）。

零枚命中一律响亮失败：面为空通常意味着 `--face` 指错了目录，那不是"已规范化"（DEBT-G91 同一族）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: 修法② 的门与修法③ 的助手在这枚文件里碰头：本仪器判的就是"在库件的行尾"，
#: 它自己出版读数时也不该再留一条"忘了写 `newline=`"的路径可错（脚本目录先上 `sys.path`，
#: 两种入口——契约册按路径加载／起成真子进程——都解析得到助手）。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

REPORT_FORMAT = "taiji-tracked-report-line-endings-v2"
DEBT = "DEBT-G115"
DEFAULT_FACE = "reports/"

#: ㊵-760 现读的豁免集（索引字节里带 CR 的已跟踪件）。修好任何一枚就把它从这串里删掉——
#: 基线只许减，加一枚新的就是红。
#: 〔㊵-1106 唯一一次"加"，带证据〕`taiji_n1_fresh_paired_train_20261016.json` 是 2026-10-16 的
#: 封存读数件，其 SHA `dbfecdfbebe4…` 被 **45 枚**在库件／计划件引用（`git grep` 现取），
#: 重写它的字节会同时打断这批身份引用 ⇒ 不能"修"，只能按本集语义豁免（历史件、不可变）。
#: 文本面里 1 MB 只有 1 对 CRLF；真正让它进这一档的是"不可变"，不是"很多 CR"。
GRANDFATHERED = (
    "reports/_gate_clip_repair.log",
    "reports/_m7_check.json",
    "reports/taiji_n1_fresh_paired_train_20261016.json",
)


def classify_endings(blob: bytes) -> dict[str, int]:
    """三桶并列，且**不许互相摊平**：`\r\n` 是一对，散 `\r` 单独计，`\r\r\n` 是混档。

    只数 `blob.count(b"\\r")` 会把一对 CRLF 也算成"有 CR"却看不出它是哪一档；早期版本在这里翻过车
    （`verifier-must-not-flatten-the-construct`）。
    """

    crlf = blob.count(b"\r\n")
    lone = blob.count(b"\r") - crlf
    doubled = blob.count(b"\r\r\n")
    return {
        "crlf_pairs": crlf,
        "lone_cr": lone,
        "cr_lf_pairs_after_cr": doubled,
        "bytes": len(blob),
    }


def tracked_paths(root: Path, face: str) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "--", face],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files failed rc={proc.returncode}: {proc.stderr.strip()[:120]}")
    return sorted(line for line in proc.stdout.splitlines() if line)


def index_blobs(root: Path, face: str) -> dict[str, bytes]:
    """一次 `cat-file --batch` 取回整面的索引字节，而不是每枚件起一个进程。

    两个理由：①这台机器上按枚 `git show :<路径>` 是分钟级扫描，且 Windows 的命令行长度装不下 1782 条路径；
    ②**面必须是同一批字节**——逐枚起进程时中途有人 `git add` 会让两次读到不同快照。
    """

    listing = subprocess.run(
        ["git", "ls-files", "-s", "--", face],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    entries: list[tuple[str, str]] = []
    for line in listing.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 2:
            entries.append((parts[1], path))
    if not entries:
        return {}
    payload = "\n".join(oid for oid, _ in entries) + "\n"
    batch = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input=payload.encode("ascii"),
        capture_output=True,
        check=True,
    ).stdout
    blobs: dict[str, bytes] = {}
    cursor = 0
    for _, path in entries:
        end = batch.index(b"\n", cursor)
        head = batch[cursor:end].decode("ascii", "replace").split()
        cursor = end + 1
        if len(head) != 3 or not head[2].isdigit():
            raise RuntimeError(f"index blob unreadable for {path}: {head}")
        size = int(head[2])
        blobs[path] = batch[cursor : cursor + size]
        cursor += size + 1
    return blobs


def scan_face(
    paths: list[str],
    reader: Callable[[str], bytes],
    working_tree: Callable[[str], bytes],
    face: str,
) -> dict[str, Any]:
    if not paths:
        raise ValueError(
            f"零枚已跟踪件落在 {face!r}：先确认 --face 指的是哪个目录，零命中不是「已规范化」"
        )
    offenders = []
    buckets: dict[str, int] = {}
    working_cr = []
    skipped_binary = []
    for name in paths:
        raw = reader(name)
        #: 〔㊵-1106 取法修口〕压缩／二进制载荷不参与行尾判定：gzip 流里出现 0x0D 是必然的，
        #: 把 37 枚 `.json.gz` 读成"带 CR 的跟踪件"是尺的错（DEBT-G149 同一族的反向形：那次是
        #: 拿 zlib 头去解 `.gz` 全数静默失败，这次是反过来把 `.gz` 当文本扫）。判据不变：
        #: 文本件带 CR 仍然红，跳过谁、跳了几枚一律出版。
        #: 〔㊵-1106 收窄〕我先试过"含 NUL 即二进制"，结果把两枚在册豁免件一起吞了（`_m7_check.json`
        #: 是 UTF-16LE、`_gate_clip_repair.log` 里本来就有 NUL）——那会让基线悄悄变小，比误报更糟，
        #: 故只认 gzip 魔数这一种压缩容器。
        if raw.startswith(b"\x1f\x8b"):
            skipped_binary.append(name)
            continue
        counts = classify_endings(raw)
        for key, value in counts.items():
            buckets[key] = buckets.get(key, 0) + value
        if counts["lone_cr"] or counts["crlf_pairs"] or counts["cr_lf_pairs_after_cr"]:
            offenders.append({"path": name, **counts})
        if b"\r" in working_tree(name):
            working_cr.append(name)
    names = sorted(row["path"] for row in offenders)
    #: 只有这两支参与判定；"恰好还是那两枚"是**现状陈述**，写进 verdicts 会把"修好一枚"误判成红
    #: （基线只许减，减是好事——同一族教训见"改名会把 lint 基线指纹重新计成新违规"）。
    verdicts = {
        "face_is_not_empty": bool(paths),
        "no_new_cr_bearing_tracked_file": set(names) <= set(GRANDFATHERED),
    }
    return {
        "format": REPORT_FORMAT,
        "debt": DEBT,
        "face": face,
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tracked_total": len(paths),
        "scanned_bytes_index_face": buckets["bytes"],
        "buckets_index_face": {
            "crlf_pairs": buckets["crlf_pairs"],
            "lone_cr": buckets["lone_cr"],
            "cr_lf_pairs_after_cr": buckets["cr_lf_pairs_after_cr"],
        },
        "cr_bearing_tracked_files": offenders,
        "cr_bearing_names": names,
        "skipped_binary_total": len(skipped_binary),
        "skipped_binary_sample": sorted(skipped_binary)[:8],
        "grandfathered": list(GRANDFATHERED),
        "equals_current_baseline": set(names) == set(GRANDFATHERED),
        "working_tree_cr_files_total": len(working_cr),
        "working_tree_cr_sample": sorted(working_cr)[:8],
        "verdicts": verdicts,
        "verdict": "holds" if all(verdicts.values()) else "not_holds",
        "failing_verdicts": sorted(key for key, ok in verdicts.items() if not ok),
        "reading_limit": (
            "判定只走**索引字节面**（`git show :<路径>`）。工作树那一面在同一份件上现读为 138 枚带 CR，"
            "那是本机 `core.autocrlf` 的检出态、不是数据缺陷，所以它只出版不判。"
            "`cr_bearing_files_are_the_grandfathered_two` 是**现状陈述**（今天恰好还是那两枚），"
            "真正挡新增的是 `no_new_cr_bearing_tracked_file` 那枚子集判据。"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="入库 reports/ 件的行尾棘轮门：索引字节里带 CR 的件只许减不许加（DEBT-G115 修法②）"
    )
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="git 仓根（契约测指向临时仓）")
    parser.add_argument("--face", default=DEFAULT_FACE, help="按版本控制取的面（默认 reports/）")
    parser.add_argument(
        "--out-report", required=True, help="落盘路径（已存在则拒覆盖 rc=2，除非 --overwrite）"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    started = time.perf_counter()
    root = Path(args.root).resolve()
    try:
        paths = tracked_paths(root, args.face)
        blobs = index_blobs(root, args.face)
        report = scan_face(
            paths,
            lambda name: blobs[name],
            lambda name: (root / name).read_bytes(),
            args.face,
        )
    except (RuntimeError, ValueError, KeyError) as error:
        print(f"REJECT {error}", file=sys.stderr)
        return 2

    report["wall_clock_ms"] = round((time.perf_counter() - started) * 1000, 3)
    out = Path(args.out_report)
    if not out.is_absolute():
        out = root / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    report_io.write_json_report(out, report, newline="\n", overwrite=args.overwrite)
    summary = {
        "verdict": report["verdict"],
        "failing": report["failing_verdicts"],
        "tracked_total": report["tracked_total"],
        "cr_bearing_names": report["cr_bearing_names"],
        "working_tree_cr_files_total": report["working_tree_cr_files_total"],
        "out": out.as_posix(),
    }
    sys.stdout.buffer.write(json.dumps(summary, ensure_ascii=True).encode("ascii") + b"\n")
    return 0 if report["verdict"] == "holds" else 1


if __name__ == "__main__":
    raise SystemExit(main())
