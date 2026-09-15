"""CAP-0 P1 只读诊断：查看 chat 链路里**字节预测的真实产物**，判定模板回落的根因。

用途：回答"没训过语言 / 训了但没接出来 / 第三种"这一诊断问题（见
`plans/reference/M5_CAP0_P1_LANGUAGE_SUPERVISION_DIAGNOSIS_20260915.md`）。

只读：不训练、不写检查点、不改任何源码。作为诊断脚本保留，便于复现证据。
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from taiji import InputFrame  # noqa: E402
from taiji.language_organ import _readable_surface  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_corpus.pt"


def _utf8_allowed(remaining: int, lead: int) -> list[int]:
    """UTF-8 DFA：给定"还期望几个续字节"与当前字符的首字节，返回合法后继字节。

    这是**精确**的 UTF-8 结构约束（含 overlong / surrogate / 超范围排除）：
    - ``remaining == 0`` ⇒ 合法首字节：ASCII 或 2/3/4 字节序列的引导字节；
    - ``remaining > 0`` ⇒ 续字节 ``0x80..0xBF``，并按首字节收紧边界。
    """

    if remaining == 0:
        return list(range(0x00, 0x80)) + list(range(0xC2, 0xF5))
    low, high = 0x80, 0xBF
    if remaining == 3 and lead == 0xE0:
        low = 0xA0
    elif remaining == 3 and lead == 0xED:
        high = 0x9F
    elif remaining == 3 and lead == 0xF0:
        low = 0x90
    elif remaining == 3 and lead == 0xF4:
        high = 0x8F
    return list(range(low, high + 1))


def _constrained_generate(model: object, prompt: bytes, length: int) -> bytes:
    """复现 ``Taiji.generate`` 的循环，但每步**只在 UTF-8 合法后继里取 argmax**。

    ``model`` 必须是 ``Taiji`` 实例（``Seed`` 是适配层，逐字节循环在它的 substrate 上）。
    """

    model.reset_dynamics(episode_id="p3a-constrained")
    step = model.observe(
        model.config.boundary_symbol,
        learn=False,
        readout="predictive",
        use_memory=False,
        use_identity=False,
    )
    for symbol in prompt:
        step = model.observe(
            int(symbol), learn=False, readout="predictive", use_memory=False, use_identity=False
        )
    out = bytearray()
    remaining = 0
    lead = 0
    for _ in range(length):
        allowed = _utf8_allowed(remaining, lead)
        probabilities = step.probabilities.detach().cpu()
        masked = probabilities.clone()
        keep = torch.zeros_like(masked, dtype=torch.bool)
        keep[torch.tensor(allowed, dtype=torch.long)] = True
        masked[~keep] = -1.0
        symbol = int(masked.argmax().item())
        if symbol == model.config.boundary_symbol:
            break
        out.append(symbol)
        if remaining == 0:
            if symbol < 0x80:
                remaining, lead = 0, 0
            elif symbol < 0xE0:
                remaining, lead = 1, symbol
            elif symbol < 0xF0:
                remaining, lead = 2, symbol
            else:
                remaining, lead = 3, symbol
        else:
            remaining -= 1
        step = model.observe(
            symbol, learn=False, readout="predictive", use_memory=False, use_identity=False
        )
    return bytes(out)


#: 包装 ``Taiji.generate`` 时必须仍能在源码里看到这行 —— 否则说明实现已变，
#: 拒绝用过期副本继续（fail-closed，避免静默降级）。
_GENERATE_ANCHOR = "next_symbol = step.predicted_symbol"


def install_constrained_decode() -> dict[str, object]:
    """进程内把 ``Taiji.generate`` 换成 UTF-8 约束解码版（**源码不动**）。

    ``Taiji.generate_input`` 直接委托 ``generate``，所以包装它即可让
    ``SeedRuntime.chat()`` 的整条链路走约束解码。带 boundary/authorization 的
    调用**不支持**（会显式报错），以免静默丢掉受控读出语义。
    """

    from taiji.adapter import Taiji

    original = Taiji.generate
    if _GENERATE_ANCHOR not in inspect.getsource(original):
        raise RuntimeError(
            f"Taiji.generate 的实现已变（缺少锚点 {_GENERATE_ANCHOR!r}）；拒绝用过期副本继续"
        )

    @functools.wraps(original)
    def patched(
        self: object,
        prompt: bytes,
        length: int,
        *,
        stop_at_boundary: bool = False,
        sample: bool = False,
        reset: bool = True,
        use_memory: bool = False,
        boundary: object = None,
        authorization: object = None,
    ) -> bytes:
        if boundary is not None or authorization is not None:
            raise RuntimeError("约束解码包装不支持带 boundary/authorization 的调用")
        # 注意：``boundary_symbol`` 是符号空间的特殊值（不保证落在 0..255 内），
        # 不能直接 ``bytes([...])``；"遇 boundary 即停"已由 _constrained_generate 处理。
        raw = _constrained_generate(self, bytes(prompt), int(length))
        # 对齐到最后一个**完整**字符：否则下游 decode 会产生替换字符，
        # 被语言器官判为"不是文本"而回落模板（长度截断不能被当成内容问题）。
        for cut in range(len(raw), max(0, len(raw) - 4), -1):
            try:
                raw[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            return raw[:cut]
        return raw

    Taiji.generate = patched  # type: ignore[method-assign]
    return {
        "patched": True,
        "anchor_present": True,
        "original": f"{original.__module__}.{original.__qualname__}",
    }


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
    parser.add_argument(
        "--constrained",
        action="store_true",
        help="额外跑一次 UTF-8 约束解码对照（只读；模型不动，只限制每步的可行字节集）",
    )
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
        entry: dict[str, object] = {
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
        if args.constrained:
            taiji = getattr(runtime.model, "substrate", runtime.model)
            constrained = _constrained_generate(taiji, text.encode("utf-8"), max_length)
            # 截断到最后一个**完整**字符：长度截断不能被误读成"约束失败"。
            trimmed = constrained
            for cut in range(len(constrained), max(0, len(constrained) - 4), -1):
                try:
                    constrained[:cut].decode("utf-8")
                except UnicodeDecodeError:
                    continue
                trimmed = constrained[:cut]
                break
            constrained_text = trimmed.decode("utf-8", errors="replace")
            entry["constrained"] = {
                "raw_bytes": len(constrained),
                "kept_bytes": len(trimmed),
                "decoded": constrained_text[:160],
                "decoded_chars": len(constrained_text),
                "has_replacement_char": "\ufffd" in constrained_text,
                "readable_surface_present": _readable_surface(constrained_text) is not None,
            }
        payload["probes"].append(entry)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
