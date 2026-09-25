"""B-2 前置：从**迁移前的当前实现**采集 `_infer_domain` 的黄金向量（零训练、秒级）。

黄金向量 = 文本 × neuron_domains 网格上的全部输出。迁移后由等价测试加载比对：
迁移后的 `helpers.infer_domain(neuron_domains, text)` 必须逐值复现本件采到的每一个输出。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuroplex.brain.cortex import Cortex  # noqa: E402

TEXTS = [
    "def foo():\n    return 1",
    "const x = 1;",
    "import os\nfrom collections import defaultdict",
    "from 0 to 1",
    "from the beginning",
    "SELECT * FROM users WHERE id = 1",
    "CREATE TABLE t (id INT)",
    "docker run --rm -it ubuntu",
    "git commit -m 'x'",
    "npm install",
    "```python\nprint('hi')\n```",
    "if __name__ == '__main__':\n    print('x')",
    "print('hello')",
    "lambda x: x + 1",
    "try:\n    pass\nexcept Exception:\n    raise",
    "std::vector<int> v;",
    "def __init__(self):",
    "求 f(x) = x^2 的 derivative",
    "E = mc^2",
    "sin(x) + cos(x) = 1",
    "∫ f(x) dx",
    "the theorem proves the equation by bayes rule",
    "x² + y² = r²",
    "α β γ δ ε",
    "log(ln(x)) + tan(theta)",
    "这是一段中文文本。",
    "今天天气很好，我们去公园散步。",
    "The quick brown fox jumps over the lazy dog.",
    "hello world 123",
    "",
    "   ",
    "1234567890 + (2*3)/4 = 11",
    "plain text without domain markers",
]

DOMAIN_SETS = [
    [],
    ["zh"],
    ["en"],
    ["code"],
    ["math"],
    ["zh", "en"],
    ["zh", "code"],
    ["en", "math"],
    ["general"],
    ["zh_aug0_dialogue"],
]


def main() -> int:
    grid: dict[str, dict[str, str]] = {}
    for domains in DOMAIN_SETS:
        key = "|".join(domains) if domains else "(empty)"
        # 该方法只读 self.neurons.keys() ⇒ 用最小替身当 self（不实例化 Cortex）
        fake = SimpleNamespace(neurons={f"{d}_unit": object() for d in domains})
        grid[key] = {
            text: Cortex._infer_domain(fake, text) for text in TEXTS
        }

    out = PROJECT_ROOT / "reports" / "cortex_infer_domain_golden_20260925.json"
    payload = {
        "format": "cortex-infer-domain-golden-v1",
        "captured_from": "迁移前的当前实现（Cortex._infer_domain，neurons 用最小替身）",
        "note": "迁移后 helpers.infer_domain(neuron_domains, text) 必须逐值复现本件每一个输出",
        "domain_sets": ["|".join(d) if d else "(empty)" for d in DOMAIN_SETS],
        "texts": TEXTS,
        "grid": grid,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "cells": sum(len(v) for v in grid.values())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
