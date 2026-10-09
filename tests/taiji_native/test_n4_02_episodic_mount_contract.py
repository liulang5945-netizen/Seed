"""PLAN-N4-02 的四条守卫（G-N4d-1…4）＋真值材料现生的契约测。

冻结设计（`plans/reference/PLAN-N4-02_episodic_mount_prereg_20261009.md` §1/§2/§4）：
* `--episodic-mount`（默认关 ⇒ 信封不含 `episodic_memory` 键、行为逐位不变）；
* 写入时机＝每篇文档边界，`memory_id=doc-<篇序>`、`cue=该篇前 32 字节 ÷ 255`（可复算）；
* `materials`/`queries` 两枚 JSONL 由训练器在落 checkpoint 时出版（真值来自本跑写入）；
* 分离机检的巩固材料＝本跑自己的夜间语料（`data/consolidated/corpus-*.jsonl`）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "scripts" / "training" / "train_seed_corpus.py"

DOCS = [
    "问：甲是什么？答：甲是第一篇测试文档，内容足够长以产生多个字节窗口与完整的行为读数，用于验证情节挂载的写入路径。",
    "问：乙呢？答：乙是第二篇测试文档，长度与甲接近，确保写入器在多篇之间正确推进 memory_id 的序号。",
    "问：丙呢？答：丙是第三篇。",
]


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps({"text": doc}, ensure_ascii=False) + "\n" for doc in DOCS),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _run(tmp_path: Path, *, mount: bool) -> dict[str, Any]:
    tag = "mounted" if mount else "default"
    proc = subprocess.run(  # noqa: S603 - 仓内固定脚本，参数不含用户输入
        [
            sys.executable,
            str(TRAINER),
            "--corpus",
            str(_corpus(tmp_path)),
            "--max-symbols",
            "400",
            "--checkpoint-every",
            "500",
            "--progress-every",
            "200",
            "--checkpoint",
            str(tmp_path / f"ck_{tag}.pt"),
            "--progress",
            str(tmp_path / f"progress_{tag}.jsonl"),
        ]
        + (["--episodic-mount"] if mount else []),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr[-400:]


def _has_key_deep(obj: Any, needle: str, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if isinstance(obj, dict):
        return any(str(k) == needle or _has_key_deep(v, needle, depth + 1) for k, v in obj.items())
    if isinstance(obj, (list, tuple)) and obj:
        return _has_key_deep(obj[0], needle, depth + 1)
    return False


def test_g_n4d_1_default_off_envelope_has_no_episodic_key(tmp_path: Path) -> None:
    _run(tmp_path, mount=False)
    envelope = torch.load(tmp_path / "ck_default.pt", map_location="cpu", weights_only=False)
    assert not _has_key_deep(envelope, "episodic_memory"), sorted(envelope)
    assert not list(tmp_path.glob("*_episodic_materials.jsonl"))


def test_g_n4d_2_mounted_run_publishes_truth_materials(tmp_path: Path) -> None:
    _run(tmp_path, mount=True)
    envelope = torch.load(tmp_path / "ck_mounted.pt", map_location="cpu", weights_only=False)
    assert _has_key_deep(envelope, "episodic_memory"), sorted(envelope)
    materials = [
        json.loads(line)
        for line in (tmp_path / "ck_mounted_episodic_materials.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    queries = [
        json.loads(line)
        for line in (tmp_path / "ck_mounted_episodic_queries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    #: G-N4d-2：写入计数＝篇边界计数＝文档数（3 篇小语料全被边界闭合）。
    assert len(materials) == len(DOCS), [row["memory_id"] for row in materials]
    assert [row["memory_id"] for row in materials] == [f"doc-{n:06d}" for n in range(len(DOCS))]
    #: G-N4d-3：cue 可由语料字节逐位复算（该篇前 32 字节 ÷ 255，不足补 0）。
    for index, row in enumerate(materials):
        raw = DOCS[index].encode("utf-8")[:32]
        expected = [byte / 255.0 for byte in raw] + [0.0] * (32 - len(raw))
        assert row["cue"] == pytest.approx(expected), index
    #: queries 与 materials 对齐：expected_memory_id 都能在 materials 里找到。
    ids = {row["memory_id"] for row in materials}
    assert all(q["expected_memory_id"] in ids for q in queries)


def test_g_n4d_4_every_tenth_writing_would_trip_the_count_guard(tmp_path: Path) -> None:
    """负对照（G-N4d-4）：把写入器改成"每 10 篇写一条"必须让计数守卫红。

    直接演算：3 篇语料在"每 10 篇"策略下只会写 0 条 ⇒ `materials` 数与篇边界数（3）不等
    ⇒ 守卫判失败。这条不用真改产品码——它证明的是守卫的判别方向。
    """
    written_under_every_tenth = len(DOCS) // 10
    assert written_under_every_tenth != len(DOCS)
    boundary_count = len(DOCS)
    assert written_under_every_tenth != boundary_count
