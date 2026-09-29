"""A30 丁-3 配方守卫：结束边界落换行之后（目标编码对齐）成为主线训练默认。

owner 条件授权（2026-09-29，PLAN-A-30 §7-3）：决策级两档判读保持胜出 ⇒ 改主线配方重训。
证据（§2aa＋决策级）：`after_newline` 在结束位上的边界符胜出 n=30 时 26/30、出厂基座
n=300 时 **263/300**，对照 `current`/`granularity`/未训全部 0——落点（而非 episode 分组、
而非目标密度）是唯一被证明有效的变量。

本文件钉三件事：
1. **字节形状两面**：默认（配方开）＝每篇正文后补 ``0x0A`` 再落下一篇的起始边界；
   逃生口（函数级 ``end_boundary_after_newline=False``）＝旧形状逐位不变
   （archive 仪器与旧配方复现靠这条）。
2. **CLI 默认**：主线入口默认开，``--no-end-boundary-after-newline`` 显式关。
3. **随档登记**：喂入形状写进每个落盘信封的 metadata（``end_boundary_after_newline``），
   "这条读数用的哪套配方"从档里可查，不靠外部记录。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from train_seed_corpus import iter_corpus_symbols  # noqa: E402

BOUNDARY = 256  # TaijiConfig().boundary_symbol


def _corpus(tmp_path: Path) -> Path:
    import json

    path = tmp_path / "corpus.jsonl"
    rows = [{"text": "问：你好\n答：你好"}, {"text": "问：天气\n答：晴"}]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _text_symbols(path: Path) -> list[list[int]]:
    """每篇的正文符号（不含边界），供形状断言拼接。"""

    import json

    docs: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        docs.append(list(json.loads(line)["text"].encode("utf-8")))
    return docs


def test_recipe_on_yields_newline_before_each_end_boundary(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    docs = _text_symbols(corpus)
    stream = list(iter_corpus_symbols([corpus], end_boundary_after_newline=True))
    expected: list[int] = []
    for doc in docs:
        expected += [BOUNDARY, *doc, 0x0A]
    assert stream == expected


def test_recipe_off_is_bit_identical_to_the_legacy_shape(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    docs = _text_symbols(corpus)
    stream = list(iter_corpus_symbols([corpus], end_boundary_after_newline=False))
    expected: list[int] = []
    for doc in docs:
        expected += [BOUNDARY, *doc]
    assert stream == expected
    #: 函数级默认＝关（archive 仪器与旧配方复现不静默变形）。
    assert list(iter_corpus_symbols([corpus])) == stream


def test_cli_default_is_on_with_explicit_escape() -> None:
    """主线配方默认＝开；逃生口显式关。parser 从 main 抽出的 _build_parser 直接解析。"""

    import train_seed_corpus as module

    parser = module._build_parser()
    default_args = parser.parse_args(["--checkpoint", "output/x/checkpoint.pt"])
    assert default_args.end_boundary_after_newline is True
    escape_args = parser.parse_args(["--no-end-boundary-after-newline"])
    assert escape_args.end_boundary_after_newline is False


def test_envelope_metadata_records_the_feed_shape(tmp_path: Path) -> None:
    """随档登记：--smoke 跑完后信封 metadata 必须带 end_boundary_after_newline=True。"""

    import torch

    from seed import Seed

    out = tmp_path / "smoke.pt"
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "train_seed_corpus.py"),
            "--smoke",
            "--checkpoint",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    envelope = torch.load(out, map_location="cpu", weights_only=False)
    metadata = envelope["metadata"]
    assert metadata["end_boundary_after_newline"] is True
    assert metadata["trainer"] == "train_seed_corpus"
    #: 信封可被产品 loader 载入（喂入形状变化不改信封合同）。
    Seed.from_checkpoint(
        torch.load(out, map_location="cpu", weights_only=True),
    )
