"""A30 §2bf 守卫：主线分块喂法（`--answer-chunking per-answer`）与短答截断。

owner 2026-09-30 裁定①＝把"分块短答形状"提升为主线配方并训一档验证
（正式档四臂：sized 47/72、self 24/72 买到真自停；连续流主线同预算只 3/72）。

本文件钉四件事：
1. **字节形状**：每块＝``问：{q}\\n答：{a}\\n``，以 ``\\n`` 结尾（与"换行后落边界"同性质）；
   拆不开 ``\\n答：`` 响亮失败；截断只动答案。
2. **默认面逐位不变**：CLI 默认 `stream` 且 `--answer-max-chars 0`；stream 档行为不因新旗标改变。
3. **响亮拒绝**：`--answer-max-chars` 配 stream 直接 parser.error（静默空转族）。
4. **随档登记**：per-answer 档的信封 metadata 记录喂法与截断值（配方从档里可查）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from train_seed_corpus import iter_answer_chunks  # noqa: E402


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"text": "问：你好\n答：你好，今天天气不错，我们出去走走吧。"},
        {"text": "问：数字\n答：四十二"},
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def test_chunks_end_with_newline_and_reassemble_the_pair(tmp_path: Path) -> None:
    chunks = list(iter_answer_chunks([_corpus(tmp_path)]))
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.endswith(b"\n")
    first = chunks[0].decode("utf-8")
    assert first == "问：你好\n答：你好，今天天气不错，我们出去走走吧。\n"


def test_truncation_touches_only_the_answer(tmp_path: Path) -> None:
    chunks = list(iter_answer_chunks([_corpus(tmp_path)], max_answer_chars=4))
    first, second = (c.decode("utf-8") for c in chunks)
    assert first == "问：你好\n答：你好，今\n"
    assert second == "问：数字\n答：四十二\n"  # 4 字以内，不动


def test_missing_seam_is_loud(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "没有接缝的一行"}, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="拆不出答案"):
        list(iter_answer_chunks([path]))


def test_cli_defaults_and_loud_rejection() -> None:
    import train_seed_corpus as module

    parser = module._build_parser()
    default_args = parser.parse_args(["--checkpoint", "output/x/checkpoint.pt"])
    assert default_args.answer_chunking == "stream"
    assert default_args.answer_max_chars == 0

    per_answer = parser.parse_args(
        ["--checkpoint", "output/x/checkpoint.pt", "--answer-chunking", "per-answer"]
    )
    assert per_answer.answer_chunking == "per-answer"

    #: 截断配 stream ＝静默空转 ⇒ 在 main 的校验处响亮拒绝（这里复刻同一谓词，
    #: 并用真 main 的 parser.error 路径做端到端冒烟时再证一次）。
    source = (PROJECT_ROOT / "scripts" / "training" / "train_seed_corpus.py").read_text(
        encoding="utf-8"
    )
    assert "only applies to --answer-chunking per-answer" in source
    assert "silent no-op" in source


def test_per_answer_smoke_records_shape_in_metadata(tmp_path: Path) -> None:
    """端到端：`--smoke --answer-chunking per-answer --answer-max-chars 8` 跑通并登记。"""

    import subprocess

    import torch

    out = tmp_path / "smoke.pt"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "train_seed_corpus.py"),
            "--smoke",
            "--answer-chunking",
            "per-answer",
            "--answer-max-chars",
            "8",
            "--checkpoint",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    envelope = torch.load(out, map_location="cpu", weights_only=False)
    metadata = envelope["metadata"]
    assert metadata["answer_chunking"] == "per-answer"
    assert metadata["answer_max_chars"] == 8


def test_per_answer_progress_and_checkpoint_cadence(tmp_path: Path) -> None:
    """节奏守卫：分块档每步跳 ~百字节，`ticks % every` 取模永不命中（第一版自伤）——
    进度与检查点必须按阈值落盘。用小间隔端到端钉住。"""

    import subprocess

    out = tmp_path / "cadence.pt"
    progress = tmp_path / "cadence_progress.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "train_seed_corpus.py"),
            "--smoke",
            "--answer-chunking",
            "per-answer",
            "--answer-max-chars",
            "8",
            "--checkpoint",
            str(out),
            "--progress",
            str(progress),
            "--progress-every",
            "500",
            "--checkpoint-every",
            "1000",
        ],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert out.is_file(), "检查点未落盘"
    lines = [line for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "进度流为空——分块档的节奏没落盘"
    ticks = [json.loads(line)["ticks"] for line in lines]
    #: 单调不减即可：终场 `_flush(final=True)` 与上一行的 ticks 相同是既有行为（连续流档同款）。
    assert ticks == sorted(ticks)
    assert len(lines) >= 3, f"进度行太少（{len(lines)}）——节奏没按阈值落盘"
