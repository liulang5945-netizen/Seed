"""verify 脚本统一输出 schema（F11）。

所有 verify_*.py 在退出前调用 ``emit_and_exit(name, report)``：
归一化为 ``{"name", "status", "metrics", "checks"}``，并以
``VERIFY_RESULT: <compact json>`` 单行打印，便于 CI 日志与收集器稳定提取；
同时保留各脚本既有的报告落盘（详细数据仍写入 reports/）。

约定：
- status ∈ {"pass", "fail"}；
- metrics：数值指标（报告无 metrics 键时取顶层数值字段）；
- checks：布尔判据（优先取 report["checks"]，否则收集 ``*_pass`` 键）。
"""

from __future__ import annotations

import json
import numbers
from typing import Any, Dict, Mapping

VERIFY_RESULT_PREFIX = "VERIFY_RESULT: "


def _coerce_pass(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "pass"
    return bool(value)


def normalize(name: str, report: Mapping[str, Any]) -> Dict[str, Any]:
    """归一化任意 verify 报告为统一 schema。"""

    if "status" in report:
        status = "pass" if _coerce_pass(report["status"]) else "fail"
    else:
        # 旧式 *_pass 布尔键：全部通过才算通过
        pass_flags = {k: v for k, v in report.items() if k.endswith("_pass")}
        status = (
            "pass" if pass_flags and all(_coerce_pass(v) for v in pass_flags.values()) else "fail"
        )

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {
            k: v
            for k, v in report.items()
            if isinstance(v, numbers.Number) and not isinstance(v, bool)
        }

    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        checks = {k: _coerce_pass(v) for k, v in report.items() if k.endswith("_pass")}

    return {"name": name, "status": status, "metrics": dict(metrics), "checks": dict(checks)}


def emit(name: str, report: Mapping[str, Any]) -> Dict[str, Any]:
    """打印 VERIFY_RESULT 行并返回归一化结果。"""

    result = normalize(name, report)
    print(VERIFY_RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def emit_and_exit(name: str, report: Mapping[str, Any]) -> int:
    """打印统一结果行并返回进程退出码（0=pass / 1=fail）。

    供 ``raise SystemExit(main())`` 与 ``sys.exit(...)`` 两种风格共用。
    """

    result = emit(name, report)
    return 0 if result["status"] == "pass" else 1
