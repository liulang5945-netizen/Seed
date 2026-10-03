"""把"已入库读数的源件还在不在、字节变没变"从**口头规矩**变成一份可复算的清单（`DEBT-G35` 的 ③）。

为什么需要它（不是清洁洁癖）：`DEBT-G35` 的成因是**路径正确而字节被换**——
`output/a31_ding3_boundary/checkpoint.pt` 被第二次跑档原地覆写，于是 §2ai–§2an 那 10 份读数记的
`checkpoint_sha256_before = 79b1a99cedf3…` 在盘上再也找不到对应字节。这类损失：
①`git ls-files` 查不出（件不在 git 里）；②markdown 链接审计查不出（路径是对的）；
③只有**把件里的 sha 与盘上的字节重算一遍**才看得见。本件做的就是 ③。

它**不是门**：不 skip、不 raise，只把每一条形如"路径＋sha"的声明归类成
`ok` / `sha_drift`（路径在场但字节变了）/ `missing_file`（路径不在盘上）/ `unresolvable_path`
（只有裸文件名且解析不到），并把四类的计数与逐条明细写进报告。判读由人做，可见性由本件保证。

落盘规矩照本仓既有修法：`--out-report` **必给**，目标已存在 ⇒ 打 `[拒绝落盘]` 并返回 2（重取换新文件名，两件并存可比）。

已知-good／已知-bad 锚点（守卫钉的是**分类本身**，不是某次的计数）：
`79b1a99cedf3e65b…`（§2ai–§2an 重训臂）必须以 `sha_drift` 或 `missing_file` 显形——它是我们已经独立
核过一次的真实损失；而 `ca2628077b21bc4c…`（装机基底）必须以 `ok` 显形。两条都为 false 时守卫会红。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 声明"某个字节身份"的键名特征
SHA_HINTS = ("sha256", "sha_256", "digest_of_file", "sha16")
PATH_HINTS = ("path", "checkpoint", "base", "circuit", "out", "answers", "artifact", "file")
ARTIFACT_SUFFIXES = (".pt", ".lzma", ".jsonl", ".bin", ".pth", ".safetensors")
#: 只有文件名没有目录时的解析候选（`model_reality.default_checkpoint` 这类裸名就靠它）
BARE_NAME_DIRS = ("checkpoints", "output")
#: 明显的"非文件身份"键：题面/摘要/概率指纹，不是盘上的件
NOT_A_FILE_MARKERS = ("items_sha256", "manifest_sha256", "docs_sha256", "prompts_sha256",
                      "corpus_sha256", "question", "seed", "state_dict", "payload", "digest")

_cache: dict[str, str | None] = {}


def _file_sha256(path: Path) -> str | None:
    key = str(path)
    if key not in _cache:
        _cache[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return _cache[key]


def _looks_like_sha(value: str) -> bool:
    return len(value) >= 16 and all(c in "0123456789abcdefABCDEF" for c in value)


def _resolve(raw: str) -> Path | None:
    """把件里写的路径解析成盘上的绝对路径；裸文件名按候选目录解析。"""
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.suffix in ARTIFACT_SUFFIXES else None
    if candidate.suffix in ARTIFACT_SUFFIXES:
        return PROJECT_ROOT / candidate
    if candidate.parent != Path("."):
        probe = PROJECT_ROOT / candidate
        if probe.suffix in ARTIFACT_SUFFIXES:
            return probe
    for base in BARE_NAME_DIRS:
        probe = PROJECT_ROOT / base / raw
        if probe.suffix in ARTIFACT_SUFFIXES:
            return probe
    return None


def _classify(path: Path, want: str) -> str:
    actual = _file_sha256(path)
    if actual is None:
        return "missing_file"
    if actual.startswith(want.lower()):
        return "ok"
    return "sha_drift"


def _walk(node, where, claims, ancestors, skipped) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _walk(item, f"{where}[{index}]", claims, ancestors, skipped)
        return
    if not isinstance(node, dict):
        return
    #: 一个 dict 里可能有**多枚**件与**多个** sha。配对必须保守，否则会把"这枚件的 sha"
    #: 配到"另一枚件的路径"上——第一版就是这么把 `ca262807`（实测正确）误报成 31 条 sha_drift 的。
    #: 规则：①路径键的值必须以产物后缀结尾（挡掉 `enable_copy_circuit`／`none`／把 sha 当路径）；
    #: ②sha 键名含 `sha` 且路径键名不含 `sha`/`digest`；③只在**词根相同**（最长公共前缀 ≥4）
    #: 或"本 dict 恰有一枚路径与一枚 sha"时才配对。
    path_keys = [
        k
        for k in node
        if isinstance(node[k], str)
        and "sha" not in k.lower()
        and "digest" not in k.lower()
        and any(h in k.lower() for h in PATH_HINTS)
        and node[k].strip().lower().endswith(ARTIFACT_SUFFIXES)
    ]
    sha_keys = [
        k
        for k in node
        if isinstance(node[k], str)
        and "sha" in k.lower()
        and _looks_like_sha(node[k])
        and not any(m in k.lower() for m in NOT_A_FILE_MARKERS)
    ]
    unambiguous = len(path_keys) == 1 and len(sha_keys) == 1
    paired_sha: set[str] = set()

    def _record(pk: str, sk: str, raw: str) -> None:
        resolved = _resolve(raw)
        if resolved is None:
            verdict, abs_path = "unresolvable_path", raw
        else:
            verdict, abs_path = _classify(resolved, node[sk]), resolved.as_posix()
        claims.append(
            {
                "report": where,
                "report_file": where.split(".json", 1)[0] + ".json",
                "path_key": pk,
                "sha_key": sk,
                "recorded_path": raw,
                "resolved_path": abs_path,
                "recorded_sha16": str(node[sk])[:16],
                "verdict": verdict,
            }
        )

    #: 配对靠**词根**而不是字符串前缀：`saved_checkpoint` ↔ `saved_sha256_16` 的公共前缀只有 `s`，
    #: 前缀规则会把这条真声明漏掉（第一版就漏了 on-policy 四臂那批）。
    #: 反过来，一个 sha 词根若同时配得上**多枚**路径（`checkpoint_sha256_before` 对
    #: `retrain_checkpoint` 与 `base_checkpoint`），一律判歧义不硬配——那正是第一版造出
    #: 31 条假 `sha_drift` 的形状。
    def _pairable(sk: str) -> list[str]:
        stems_sk = _stems(sk)
        return [pk for pk in path_keys if _stems(pk) & stems_sk]

    for sk in sha_keys:
        matches = _pairable(sk)
        if len(matches) == 1:
            _record(matches[0], sk, node[matches[0]].strip())
            paired_sha.add(sk)
        elif unambiguous and len(path_keys) == 1:
            _record(path_keys[0], sk, node[path_keys[0]].strip())
            paired_sha.add(sk)

    #: 跨层的本仓约定：路径写在**外层**（`retrain_checkpoint`），sha 写在**臂块里**
    #: （`runs.retrain.checkpoint_sha256_before`）。只看同一个 dict 就会把 §2ai–§2an 那批
    #: ——也就是本件存在的理由——静默丢掉（第一版正是这么漏掉已知-bad 锚点的）。
    arm = where.rsplit(".", 1)[-1].split("[")[0] if "." in where else ""
    for sk in sha_keys:
        if sk in paired_sha or "checkpoint" not in sk.lower():
            continue
        for anc in ancestors:
            cand = [k for k in anc if k.lower().endswith("_checkpoint") and _is_pathish(anc[k])]
            if arm and f"{arm}_checkpoint" in cand:
                _record(f"{arm}_checkpoint", sk, anc[f"{arm}_checkpoint"].strip())
                paired_sha.add(sk)
                break

    #: **不许静默丢弃**：仍然配不出路径的 sha 声明记进 skipped，让"看不见"变成"看得见但没分类"。
    for sk in sha_keys:
        if sk not in paired_sha:
            skipped.append(
                {
                    "report": where,
                    "report_file": where.split(".json", 1)[0] + ".json",
                    "sha_key": sk,
                    "recorded_sha16": str(node[sk])[:16],
                }
            )

    for key, value in node.items():
        if isinstance(value, (dict, list)):
            _walk(value, f"{where}.{key}" if where else key, claims, ancestors + [node], skipped)


def _is_pathish(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower().endswith(ARTIFACT_SUFFIXES)


_SHA_TOKENS = ("sha", "256", "16", "before", "after", "hex", "digest", "of", "file")


def _stems(key: str) -> set:
    """键名词根集：`saved_sha256_16` → {saved}，`saved_checkpoint` → {saved, checkpoint}。"""

    return {t for t in key.lower().replace("-", "_").split("_") if t and t not in _SHA_TOKENS}


def audit(reports_dir: Path) -> dict:
    claims: list[dict] = []
    skipped: list[dict] = []
    scanned = 0
    self_excluded = 0
    for file in sorted(reports_dir.rglob("*.json")):
        try:
            payload = json.loads(file.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        #: 自喂防护：本件的输出里 `resolved_path`／`recorded_sha16` 成对出现，下一趟会把**自己的判定**
        #: 当成一条新的"路径＋sha 声明"再扫一遍 ⇒ 计数随运行次数膨胀，而判定内容并没有新证据。
        if isinstance(payload, dict) and payload.get("format") == "taiji-artifact-sha-drift-v1":
            self_excluded += 1
            continue
        scanned += 1
        _walk(payload, file.name, claims, [], skipped)
    counts: dict[str, int] = {}
    for claim in claims:
        counts[claim["verdict"]] = counts.get(claim["verdict"], 0) + 1
    bad = [c for c in claims if c["verdict"] != "ok"]
    bad.sort(key=lambda c: (c["verdict"], c["report"], c["recorded_path"]))
    return {
        "format": "taiji-artifact-sha-drift-v1",
        "reports_scanned": scanned,
        "own_outputs_excluded": self_excluded,
        "claims_total": len(claims),
        "verdict_counts": counts,
        "sha_claims_unpaired_with_path": len(skipped),
        "unpaired_examples": sorted(skipped, key=lambda s: s["report"])[:40],
        "distinct_non_ok_paths": sorted({c["resolved_path"] for c in bad}),
        "non_ok_claims": bad,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "本件不判红不判绿：sha_drift 可能是合法的续训覆写，也可能是 G35 那种不可逆损失——"
        "分类由人做，可见性由本件保证。`sha_claims_unpaired_with_path` 数是「配不出来」的量，"
        "它必须被看见而不是被当成零。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument(
        "--out-report",
        required=True,
        help="清单落点（必给；已存在即拒绝落盘，重取请换新文件名）",
    )
    args = parser.parse_args(argv)

    out = Path(args.out_report)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    if out.exists():
        print(f"[拒绝落盘] 目标已存在：{out}（换新文件名重取）", file=sys.stderr)
        return 2
    payload = audit(Path(args.reports_dir))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "format", "reports_scanned", "claims_total", "verdict_counts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
