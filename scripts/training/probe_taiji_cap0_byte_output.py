"""CAP-0 P1 只读诊断：查看 chat 链路里**字节预测的真实产物**，判定模板回落的根因。

用途：回答"没训过语言 / 训了但没接出来 / 第三种"这一诊断问题（见
`plans/reference/M5_CAP0_P1_LANGUAGE_SUPERVISION_DIAGNOSIS_20260915.md`）。

只读：不训练、不写检查点、不改任何源码。作为诊断脚本保留，便于复现证据。
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CAP-0 只读诊断：字节预测产物的可读性（可选 legacy 守卫放宽）"
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--relax-legacy-guard",
        action="store_true",
        help="进程内放宽 legacy 守卫（复用反事实探针的同一实现；不改源码、不训练）",
    )
    parser.add_argument("--max-length", type=int, default=64, help="每次生成的字节上限")
    args = parser.parse_args(argv)
    max_length = int(args.max_length)

    guard_info: dict[str, object] = {}
    if args.relax_legacy_guard:
        # 单一实现：复用反事实探针的进程内包装，避免两份放宽逻辑漂移。
        from scripts.training.probe_taiji_cap0_legacy_load import _install_legacy_guard

        guard_info = _install_legacy_guard()

    runtime = SeedRuntime.load(args.checkpoint)
    payload: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "guard_relaxed": bool(args.relax_legacy_guard),
        "legacy_guard": guard_info,
        "tick": int(runtime.model.tick),
        "probes": [],
    }

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
        raw = runtime.model.generate_input(frame, max_length, stop_at_boundary=True, sample=False)
        decoded = raw.decode("utf-8", errors="replace")
        # 解码分析：判定"是编码边界问题"还是"根本没学到语言"。
        #   ignore  ⇒ 丢弃非法字节后剩下什么（能看出是否接近自然语言）
        #   prefix  ⇒ 最长合法 UTF-8 前缀长度（越短说明首字节就越不合法）
        ignored = raw.decode("utf-8", errors="ignore")
        prefix = 0
        for size in range(len(raw) + 1):
            try:
                raw[:size].decode("utf-8")
            except UnicodeDecodeError:
                break
            prefix = size
        payload["probes"].append(
            {
                "prompt": prompt,
                "raw_bytes": len(raw),
                "raw_hex_head": raw[:48].hex(),
                "decoded_head": decoded[:120],
                "decoded_ignore": ignored[:120],
                "decoded_ignore_chars": len(ignored),
                "longest_valid_utf8_prefix_bytes": prefix,
                "has_replacement_char": "\ufffd" in decoded,
                "readable_surface": _readable_surface(decoded),
                "chat_output_head": runtime.chat(prompt, learn=False)[:90],
            }
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
