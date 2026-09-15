"""CAP-0 P1 只读诊断：查看 chat 链路里**字节预测的真实产物**，判定模板回落的根因。

用途：回答"没训过语言 / 训了但没接出来 / 第三种"这一诊断问题（见
`plans/reference/M5_CAP0_P1_LANGUAGE_SUPERVISION_DIAGNOSIS_20260915.md`）。

只读：不训练、不写检查点、不改任何源码。作为诊断脚本保留，便于复现证据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from taiji import InputFrame  # noqa: E402
from taiji.language_organ import _readable_surface  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_corpus.pt"


def main() -> int:
    runtime = SeedRuntime.load(CHECKPOINT)
    payload: dict[str, object] = {"tick": int(runtime.model.tick), "probes": []}

    for prompt in ("水的沸点是多少？", "你好。", "1+1 等于几？", "用一句话说明你能做什么。"):
        text = runtime._serialize(prompt, None)
        frame = InputFrame(
            input_id=f"p1:{runtime.model.tick}",
            modality="text",
            payload=text.encode("utf-8"),
            source="p1.probe",
            timestamp=runtime.model.tick,
            provenance="external",
            confidence=1.0,
        )
        raw = runtime.model.generate_input(frame, 64, stop_at_boundary=True, sample=False)
        decoded = raw.decode("utf-8", errors="replace")
        payload["probes"].append(
            {
                "prompt": prompt,
                "raw_bytes": len(raw),
                "raw_hex_head": raw[:48].hex(),
                "decoded_head": decoded[:120],
                "has_replacement_char": "\ufffd" in decoded,
                "readable_surface": _readable_surface(decoded),
                "chat_output_head": runtime.chat(prompt, learn=False)[:90],
            }
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
