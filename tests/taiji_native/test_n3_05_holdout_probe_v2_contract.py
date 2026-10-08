"""PLAN-N3-05 的契约测：第二把尺在场、只读、且件与常量不漂移。

三支各有明确的作用，也都能为假：

1. **在场**＝进度行的周期键集里同时有 `holdout_surprise_v2` 与旧列，且两列的值**不相等**
   （相等就意味着两次 `score_bytes` 打的是同一份文本，即新常量其实是旧常量的副本）。
2. **只读**＝G-N5-2 的形状：`score_bytes` 前后权重摘要必须逐位同。这条防的是"打分顺手改了学习状态"，
   那会把一个泛化读数变成一次训练。
3. **不漂移**＝训练器里 `HOLDOUT_PROBE_V2` 的字节数必须等于验证件记录的 `chosen` 候选字节数，
   而验证件里旧探针必须仍报**非零**命中（尺有判别力）。**这条就是防"件选了 A、码里放了 B"**——
   没有它，将来任何人换了新探针文字而没重跑验收，是静默的。

放宽进度行键集那一次的真红证据写在 `test_g14_trainer_exit_accounting.py` 的 `LEGACY_KEYS` 注释里
（实测 `1 failed, 6 passed`，失败项点名 `holdout_surprise_v2`）；这里**不再造一遍恒真式**去"演示"它会拒野键。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "training"))

from train_seed_corpus import (  # noqa: E402
    HOLDOUT_PROBE,
    HOLDOUT_PROBE_V2,
    run_training,
)

VERIFICATION_REPORT = REPO / "reports/taiji_n3_05_probe_verification_20261008.json"
SHAPE_MARKERS = ("问：", "\n答：", "请")


def _config():
    from seed import SeedConfig

    return SeedConfig()


def _write_corpus(path: Path, rows: int) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(rows):
            handle.write(json.dumps({"text": f"第{index}行：这是一段用来训练的中文文本。"}) + "\n")
    return path


def _digest(model) -> str:
    from taiji import content_digest

    return content_digest(model.checkpoint())


def test_both_holdout_columns_present_and_distinct(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "corpus.jsonl", 40)
    progress = tmp_path / "progress.jsonl"
    run_training(
        corpus_paths=[corpus],
        config=_config(),
        epochs=1,
        checkpoint_path=tmp_path / "ckpt.pt",
        progress_path=progress,
        checkpoint_every=100_000,
        progress_every=15,
        max_symbols=60,
    )
    rows = [
        json.loads(line)
        for line in progress.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 2, rows
    first = rows[0]
    assert "holdout_surprise" in first and "holdout_surprise_v2" in first, sorted(first)
    #: 两列不相等＝两次打分打的确实是两份不同文本；相等就是新常量抄了旧常量。
    assert first["holdout_surprise"] != first["holdout_surprise_v2"], first
    assert "exit_reason" not in first


def test_score_bytes_does_not_touch_weights() -> None:
    """G-N5-2：打分器必须只读——权重摘要与 **torch 的 RNG 状态**都要逐位不动。

    为什么连 RNG 一起钉：加一次 `score_bytes` 调用如果消耗了随机流，后面的训练就会走到另一条
    抽样序列上，那"旧列逐位不变"就不是由代码形状保证而是碰巧。两样都钉住，G-N5-1 才有机理依据
    （机理版本的实证＝改前/改后两支同参跑的六键逐位同，见 PLAN-N3-05 判读侧记录）。
    """

    import torch

    from seed import Seed

    model = Seed(_config())
    before = _digest(model)
    rng_before = torch.get_rng_state().clone()
    cpu_before = torch.cpu_initial_seed() if hasattr(torch, "cpu_initial_seed") else None
    first = model.score_bytes(HOLDOUT_PROBE)["mean_surprise"]
    middle = _digest(model)
    rng_middle = torch.get_rng_state().clone()
    second = model.score_bytes(HOLDOUT_PROBE_V2)["mean_surprise"]
    after = _digest(model)
    assert middle == before == after, "score_bytes 改动了权重摘要"
    assert torch.equal(rng_before, rng_middle) and torch.equal(
        rng_before, torch.get_rng_state()
    ), "score_bytes 消耗了随机流 ⇒ 加一次调用会改变后续训练的抽样序列"
    assert isinstance(first, float) and isinstance(second, float)
    if cpu_before is not None:
        assert cpu_before == torch.cpu_initial_seed()


def test_probe_constants_match_the_frozen_verification() -> None:
    """件选了哪一枚、码里就得放哪一枚：字节数对齐验证件，且尺自己还能量出错。"""

    assert (
        VERIFICATION_REPORT.is_file()
    ), f"缺验证件：{VERIFICATION_REPORT}（先跑 verify_taiji_n3_05_probe.py）"
    payload = json.loads(VERIFICATION_REPORT.read_text(encoding="utf-8"))
    assert payload["verdict"] == "probe_selected", payload["verdict"]
    chosen = payload["chosen"]
    assert payload["candidates"][chosen]["pass_all"] is True
    assert len(HOLDOUT_PROBE_V2) == payload["candidates"][chosen]["bytes"], (
        len(HOLDOUT_PROBE_V2),
        payload["candidates"][chosen]["bytes"],
        "训练器里的新常量字节数 ≠ 验证件选中那枚 ⇒ 码与件已漂移",
    )
    #: 判别力（G-N5-3）：旧探针必须仍报非零命中；它报 0 就说明这把尺量不出东西，
    #: 那时新探针的"0 命中"也一文不值。
    assert payload["legacy_probe"]["windows_found_in_corpus"] > 0, payload["legacy_probe"]
    assert len(HOLDOUT_PROBE) == payload["legacy_probe"]["bytes"]


def test_new_probe_keeps_the_same_shape_and_band() -> None:
    """A-2/A-3 的常量侧影子：形状三段齐、字节数落在冻过的档里（不靠验证件也当场能核）。"""

    text = HOLDOUT_PROBE_V2.decode("utf-8")
    for marker in SHAPE_MARKERS:
        assert marker in text, marker
    assert 64 <= len(HOLDOUT_PROBE_V2) <= 512, len(HOLDOUT_PROBE_V2)
    assert HOLDOUT_PROBE_V2 != HOLDOUT_PROBE
