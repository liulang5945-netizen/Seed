"""C6 Z1 probe: can the workbench capability snapshot become native-trainable corpus?

Zero product change: it reads ``GET /api/workbench/capabilities`` (or a captured
snapshot file), renders each declared capability into the dialogue shape the
native trainer accepts, writes the result OUTSIDE the trainer's scan roots, and
judges it with ``seed.datasets.inspect_native_dataset``.  It answers only whether
material exists and is renderable -- it does not train, admit, or wire anything,
and it adds no product type or source kind.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed.datasets import inspect_native_dataset  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8000"
CAPABILITIES_PATH = "/api/workbench/capabilities"

# The corpus this runtime was trained on is Chinese dialogue in 问：/答： form, so
# a capability reaches the model only as an answer to something.  These two
# strings are the probe's own wording and are recorded in the report.
QUESTION = "这个运行环境的工作台现在可以执行哪些操作？"
ANSWER_LEAD = "当前工作台声明了以下能力。"


def _load_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    if args.snapshot is not None:
        raw = Path(args.snapshot).read_text(encoding="utf-8")
    else:
        url = f"{args.url.rstrip('/')}{CAPABILITIES_PATH}"
        with urllib.request.urlopen(url, timeout=args.timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise SystemExit("workbench capabilities snapshot is not a JSON object")
    return parsed


def _render(capability: dict[str, Any]) -> str:
    parts = [f"{capability.get('capability_id', '')}：{capability.get('description', '')}"]
    risk = capability.get("risk")
    if isinstance(risk, str) and risk:
        parts.append(f"风险等级 {risk}")
    reversible = capability.get("reversible")
    if isinstance(reversible, bool):
        parts.append("可逆" if reversible else "不可逆")
    category = capability.get("category")
    if isinstance(category, str) and category:
        parts.append(f"类别 {category}")
    parameters = capability.get("parameters")
    if isinstance(parameters, dict) and parameters:
        rendered = "、".join(f"{name}={value}" for name, value in sorted(parameters.items()))
        parts.append(f"参数 {rendered}")
    return "；".join(part for part in parts if part) + "。"


def _cjk_ratio(text: str) -> float:
    letters = [char for char in text if not char.isspace()]
    if not letters:
        return 0.0
    cjk = [char for char in letters if "\u4e00" <= char <= "\u9fff"]
    return len(cjk) / len(letters)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="runtime root to read")
    parser.add_argument("--snapshot", help="path to a captured snapshot instead of reading it")
    parser.add_argument("--out", default="output/c6_probe", help="where to write the probe corpus")
    parser.add_argument("--report", required=True, help="where to write the JSON reading")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="seconds to wait for the endpoint"
    )
    args = parser.parse_args()

    snapshot = _load_snapshot(args)
    capabilities = [item for item in snapshot.get("capabilities", []) if isinstance(item, dict)]

    lines = []
    for capability in capabilities:
        lines.append({"text": f"问：{QUESTION}\n答：{ANSWER_LEAD}{_render(capability)}"})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "c6-workbench-capability-probe.jsonl"
    body = "".join(f"{json.dumps(line, ensure_ascii=False)}\n" for line in lines)
    corpus_path.write_text(body, encoding="utf-8")

    inspected = inspect_native_dataset(corpus_path)
    lengths = [len(str(line["text"])) for line in lines]
    ratios = [_cjk_ratio(str(line["text"])) for line in lines]

    reading = {
        "probe": "taiji_c6_workbench_capability_projection",
        "establishes": (
            "the runtime's declared capabilities can be rendered into the 问/答 shape the native "
            "trainer accepts, and that file is judged native_trainable; it establishes no capability, "
            "no admission, and no wiring"
        ),
        "snapshot": {
            "format": snapshot.get("format"),
            "version": snapshot.get("version"),
            "revision": snapshot.get("revision"),
            "snapshot_id": snapshot.get("snapshot_id"),
            "declared_capabilities": len(capabilities),
        },
        "rendered": {
            "path": str(corpus_path),
            "outside_trainer_scan_roots": True,
            "question": QUESTION,
            "answer_lead": ANSWER_LEAD,
            "documents": inspected.documents,
            "invalid_records": inspected.invalid_records,
            "native_trainable": inspected.native_trainable,
            "chars_min": min(lengths) if lengths else 0,
            "chars_median": statistics.median(lengths) if lengths else 0,
            "chars_max": max(lengths) if lengths else 0,
            "cjk_ratio_min": round(min(ratios), 3) if ratios else 0.0,
            "cjk_ratio_median": round(statistics.median(ratios), 3) if ratios else 0.0,
            "cjk_ratio_max": round(max(ratios), 3) if ratios else 0.0,
        },
        "sample": str(lines[0]["text"]) if lines else "",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(reading, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(reading["rendered"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
