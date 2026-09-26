"""B-4 门：**核心推理路径**的行覆盖率（审计 `reports/project_audit_2026-08-23.md` §7 第 16 条）。

口径（为什么是这几个文件）：审计把"冻结基线名不副实"定位在**核心推理路径**上
（S3 生成主循环在 `cortex.py`、KV cache/RoPE 在 `layers.py`；S9 滑窗在 `layers.py`），
一次 `generate()` 实际穿过的模块即本门度量面：

    brain/cortex.py + brain/_cortex_*.py                        推理主对象与其纯函数分部
    resonance/ensemble.py / continuous.py / field.py             前向引擎与共振场
    layers.py                                                    注意力 / RoPE / KV cache

不在面内（有意排除，避免把死代码算进分母）：`brain/working_memory.py` —— cortex.py:219-223
自己写明"仅注册未接入"（真正的上下文记忆由 `neuroplex/agent/working_memory` 经 ContextManager
承担），且实测 `Cortex.generate` 不调它。该模块另有一个 FIFO 丢位后 `round_marks` 整体错位的
缺陷（见本轮登记），属独立债，不混进覆盖率口径。

⚠️ 分母只算上面这些文件（coverage 的 `--cov` 面按包取，再按本清单过滤）。
按 `pyproject.toml` 的既有教训：只跑测试子集或不收窄度量面都会产生**假低/假高**读数。
全局门（`fail_under=21.8`，分母含全仓 0% 遗留代码）与本门**并存且互不换算**。

用法：
    python scripts/training/verify_core_reasoning_coverage.py                 # 跑全量 tests/ 并度量
    python scripts/training/verify_core_reasoning_coverage.py --from-report output/cov.json
    python scripts/training/verify_core_reasoning_coverage.py --tests "tests/test_cortex*"
    阈值可用 --threshold 覆盖（缺省 60.0，即审计目标）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from _verify_emit import emit_and_exit  # noqa: E402

DEFAULT_THRESHOLD = 60.0

# 度量面：相对路径，与 audit 定位的核心推理路径一一对应（见模块 docstring）
CORE_FACE = (
    "neuroplex/brain/cortex.py",
    "neuroplex/brain/_cortex_helpers.py",
    "neuroplex/brain/_cortex_quality.py",
    "neuroplex/brain/_cortex_alignment.py",
    "neuroplex/brain/_cortex_routing.py",
    "neuroplex/brain/_cortex_generation.py",
    "neuroplex/resonance/ensemble.py",
    "neuroplex/resonance/continuous.py",
    "neuroplex/resonance/field.py",
    "neuroplex/layers.py",
)


def _norm(path: str) -> str:
    """coverage JSON 的键是绝对路径（Windows 还是反斜杠）⇒ 归一成 `neuroplex/...` 相对形式。

    ⚠️ 不能用 `split("neuroplex/")[-1]`：那会把前缀本身切掉，过滤后一个文件都不剩
    （第一版就是这样，门直接报 0 语句却仍然"跑通了"）。
    """

    unified = path.replace("\\", "/")
    marker = "neuroplex/"
    idx = unified.rfind(marker)
    if idx < 0:
        return unified
    return unified[idx:]


def summarize(report: dict, threshold: float) -> dict:
    """按 CORE_FACE 过滤 coverage JSON，算面内行覆盖率。"""

    files = report.get("files", {})
    per_file = {}
    covered = total = 0
    for raw, data in files.items():
        rel = _norm(raw)
        if not rel.startswith("neuroplex/"):
            continue
        if rel not in CORE_FACE:
            continue
        summary = data["summary"]
        covered += summary["covered_lines"]
        total += summary["num_statements"]
        per_file[rel] = {
            "statements": summary["num_statements"],
            "covered": summary["covered_lines"],
            "pct": round(summary["percent_covered"], 2),
            "missing_ranges": summary.get("missing_lines_count", 0),
        }
    missing_members = sorted(set(CORE_FACE) - set(per_file))
    pct = 100.0 * covered / total if total else 0.0
    face_ok = not missing_members
    target_ok = pct >= threshold
    return {
        # ⚠️ 必须显式给 status：_verify_emit.normalize 只扫**顶层** *_pass 键，
        # 判据嵌在 checks 里时它会退回空集合 ⇒ 恒 fail（一条永远红的门）。
        "status": "pass" if (face_ok and target_ok) else "fail",
        "name": "core_reasoning_coverage",
        "metrics": {
            "core_line_pct": round(pct, 2),
            "threshold": threshold,
            "audit_target": DEFAULT_THRESHOLD,
            "covered_lines": covered,
            "statements": total,
        },
        "checks": {
            # 面内文件必须在报告里出现，否则"高分"可能只是漏算了难覆盖的文件
            "core_face_complete_pass": face_ok,
            "core_coverage_pass": target_ok,
        },
        "per_file": per_file,
        "missing_from_report": missing_members,
        "threshold_basis": "audit 2026-08-23 §7-16 目标 ≥60%（度量面＝一次 generate() 穿过的模块）",
    }


def run_pytest(tests: list[str], json_out: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *tests,
        "-q",
        "-p",
        "no:cacheprovider",
        "--cov=neuroplex.brain",
        "--cov=neuroplex.resonance",
        "--cov=neuroplex.layers",
        f"--cov-report=json:{json_out}",
        "--cov-fail-under=0",
    ]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-report", default="", help="复用既有 coverage JSON（不重跑测试）")
    ap.add_argument("--tests", nargs="*", default=["tests"], help="测试目标，缺省全量 tests/")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out", default="reports/core_reasoning_coverage_20260926.json")
    args = ap.parse_args()

    if args.from_report:
        report_path = Path(args.from_report)
        if not report_path.is_absolute():
            report_path = PROJECT_ROOT / report_path
        if not report_path.exists():
            # 缺席的报告不能读成"覆盖率 0%"——那是插桩没跑成的另一种失败
            return emit_and_exit(
                "core_reasoning_coverage",
                {
                    "status": "fail",
                    "metrics": {},
                    "checks": {"coverage_report_exists_pass": False},
                    "note": f"--from-report 指向的文件不存在：{report_path}",
                },
            )
    else:
        tmp = Path(tempfile.mkdtemp(prefix="corecov_")) / "coverage.json"
        rc = run_pytest(args.tests, tmp)
        if not tmp.exists():
            return emit_and_exit(
                "core_reasoning_coverage",
                {
                    "status": "fail",
                    "metrics": {"pytest_rc": rc},
                    "checks": {"coverage_report_produced_pass": False},
                    "note": f"pytest rc={rc} 且未产出 coverage JSON（崩/挂 ⇒ 读数不可用）",
                },
            )
        report_path = tmp

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    result = summarize(raw, args.threshold)
    result["report_source"] = str(report_path)
    result["totals_pct_from_runner"] = raw.get("totals", {}).get("percent_covered")
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    result["report"] = str(out)
    return emit_and_exit("core_reasoning_coverage", result)


if __name__ == "__main__":
    sys.exit(main())
