"""B-3 C-1 第二刀前置：从**迁移前的当前实现**采 `_rolling_nll_quality` 的黄金向量（零训练）。

夹具：`neuroplex.loader.create_cortex()`（生产构造路径）+ 自装 TokenizerHub/general SP
（与 C-2 采集脚本同款，`create_cortex` 本机 fallback 模式下 hub=None 故必须自装）。
输入：固定种子的合成 `round1_logits`（含一个词表不符的 neuron，用来钉"跳过"分支）。

迁移后 `_cortex_quality.rolling_nll_quality(hub, device, result, gen_text, domain, window)`
与 `Cortex._rolling_nll_quality` 委托都必须逐值复现本件。

⚠️ **本件是一次性锚点**：已于 2026-09-26 迁移前采集入库（提交见 B-3 C-1 第二刀）。
迁移后 cortex.py 里只有委托 ⇒ 重跑本脚本 = 用被测件自采自证，禁作再采集用途；
要重采必须先 revert 迁移提交（对照 reports/cortex_rolling_nll_golden_20260926.json 现值）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sentencepiece as spm  # noqa: E402
import torch  # noqa: E402

from neuroplex.loader import create_cortex  # noqa: E402
from neuroplex.resonance.translator import TokenizerHub  # noqa: E402


def build_cortex():
    """生产构造路径 + 自装 tokenizer（采集与守卫共用，避免两处漂移）。"""

    cortex, _ = create_cortex()

    def _load(rel: str):
        sp = spm.SentencePieceProcessor()
        sp.Load(str(PROJECT_ROOT / "neuroplex" / "domains" / rel))
        return sp

    general_sp = _load("general/sp_general.model")
    zh_sp = _load("zh/sp_zh.model")
    hub = TokenizerHub(general_tokenizer=general_sp)
    hub.register_domain("general", general_sp)
    hub.register_domain("zh", zh_sp)
    cortex.set_tokenizer_hub(hub)
    cortex.set_general_tokenizer(general_sp)
    return cortex, hub, general_sp, zh_sp


def synth_logits(zh_sp, seed: int = 7, include_short: bool = True) -> dict:
    """固定种子的 round1_logits：两支词表相符、一支词表不符、一支长度不足。

    ⚠️ `zh_unit_short`（长度 2）会把 `min(lens)-1` 钉到 1 ⇒ 任何 window 都退化成 n=1。
    要测 window 真正参与取位的那条路，必须把它从网格中拿掉（单独一条 case 钉它）。
    """

    vocab = int(zh_sp.GetPieceSize())
    gen = torch.Generator().manual_seed(seed)
    logits = {
        "zh_unit_a": torch.randn(1, 24, vocab, generator=gen),
        "zh_unit_b": torch.randn(1, 24, vocab, generator=gen) * 0.3,
        "code_unit_c": torch.randn(1, 24, vocab + 7, generator=gen),
    }
    if include_short:
        logits["zh_unit_short"] = torch.randn(1, 2, vocab, generator=gen)
    return {"round1_logits": logits}


CASES = [
    {"name": "zh_window8", "gen_text": "阿岩。我的名字是阿岩。", "domain": "zh", "window": 8},
    {"name": "zh_window1", "gen_text": "阿岩。我的名字是阿岩。", "domain": "zh", "window": 1},
    {
        "name": "zh_window4",
        "gen_text": "阿岩。我的名字是阿岩。今天天气很好。",
        "domain": "zh",
        "window": 4,
    },
    {
        "name": "general_domain_fallback",
        "gen_text": "hello world answer",
        "domain": "nope",
        "window": 6,
    },
    {"name": "window0", "gen_text": "阿岩。", "domain": "zh", "window": 0},
    {"name": "empty_text", "gen_text": "", "domain": "zh", "window": 4},
    {
        "name": "window_huge",
        "gen_text": "阿岩。我的名字是阿岩。今天天气很好。",
        "domain": "zh",
        "window": 9999,
    },
    {
        "name": "length_bound_short_neuron",
        "gen_text": "阿岩。我的名字是阿岩。",
        "domain": "zh",
        "window": 8,
        "include_short": True,
    },
]


def main() -> int:
    cortex, _hub, _general_sp, zh_sp = build_cortex()
    grid = {}
    for case in CASES:
        result = synth_logits(zh_sp, include_short=case.get("include_short", False))
        out = cortex._rolling_nll_quality(result, case["gen_text"], case["domain"], case["window"])
        grid[case["name"]] = {
            "gen_text": case["gen_text"],
            "domain": case["domain"],
            "window": case["window"],
            "include_short": case.get("include_short", False),
            "out": {k: float(v) for k, v in out.items()},
        }
    # 空 logits 与 hub=None 两条早退分支也要钉住（它们同样是迁移面的一部分）
    grid["empty_round1"] = {
        "out": cortex._rolling_nll_quality({"round1_logits": {}}, "阿岩", "zh", 4)
    }
    bare = object.__new__(type(cortex))
    bare._tokenizer_hub = None
    bare.device = torch.device("cpu")
    grid["hub_missing"] = {
        "out": type(cortex)._rolling_nll_quality(bare, synth_logits(zh_sp), "阿岩", "zh", 4)
    }

    payload = {
        "format": "cortex-rolling-nll-golden-v1",
        "captured_from": "迁移前的 Cortex._rolling_nll_quality（夹具 create_cortex + 自装 hub，合成 logits 种子 7）",
        "note": "迁移后 helper 与委托都必须逐值复现本件；float 按 repr 全精度入册",
        "grid": grid,
    }
    out_path = PROJECT_ROOT / "reports" / "cortex_rolling_nll_golden_20260926.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"report": str(out_path), "cases": len(grid)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
