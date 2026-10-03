"""`DEBT-G35` 的机器侧防口守卫：训练器不许覆写一枚**不是本次 `--resume` 源**的已存在检查点。

为什么值得钉（不是清洁洁癖）：`output/a31_ding3_boundary/checkpoint.pt` 被第二次跑档原地覆写之后，
10 份已入库读数记的 `checkpoint_sha256_before = 79b1a99cedf3…` 在盘上再也找不到对应字节 ⇒ §2ai–§2an 那批
结论永远无法复算，而**按路径 grep 或 markdown 链接检查都查不出来**（路径是对的，字节换了）。
约定挡不住这件事，只能在写靶被解析的那一刻响亮拒绝。

本守卫**不需真训练**即可跑正向两支里的一支（拒绝路），另一支用 `--smoke`（5000 ticks）确认"闸门不是恒红"。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINER_REL = "scripts/training/train_seed_corpus.py"
GUARD_MESSAGE = "refusing to overwrite existing checkpoint"


@pytest.fixture(scope="module")
def trainer():
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))
    import importlib

    return importlib.import_module("train_seed_corpus")


def test_scan_face_is_the_tracked_file() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", TRAINER_REL], cwd=PROJECT_ROOT, capture_output=True, check=True
    ).stdout.decode().strip()
    assert tracked == TRAINER_REL, f"训练器不在版本控制面上（扫到 {tracked!r}），本守卫没在守任何东西"


def test_guard_is_registered_in_the_flag_surface() -> None:
    source = (PROJECT_ROOT / TRAINER_REL).read_text(encoding="utf-8")
    assert '"--allow-overwrite-checkpoint"' in source, "逃生口旗标不见了 ⇒ 原地续训会被误拒"
    # 拒绝必须发生在**读 resume 档之前**，否则覆写前的检查点已被 torch.load 打开过。
    assert source.index(GUARD_MESSAGE) < source.index("torch.load(args.resume")


def test_refuses_overwriting_a_foreign_checkpoint(
    trainer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """正身：`--checkpoint` 指向一枚已存在、且不是 `--resume` 源的件 ⇒ rc=2 且点名原因。"""

    victim = tmp_path / "checkpoint.pt"
    victim.write_bytes(b"pre-existing-bytes")
    monkeypatch.setattr(
        sys, "argv",
        ["train_seed_corpus.py", "--smoke", "--checkpoint", str(victim),
         "--progress", str(tmp_path / "progress.jsonl")],
    )
    with pytest.raises(SystemExit) as caught:
        trainer.main()
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert GUARD_MESSAGE in err, err[-300:]
    assert victim.read_bytes() == b"pre-existing-bytes", "拒绝路不许已经动过目标文件"


def test_in_place_resume_is_not_refused(
    trainer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """反向：原地续训（`--resume X` 且 `--checkpoint X`）是**合法**用法，不能被这条闸门挡死。

    这里不要求跑完：目标是让它越过闸门。越过后它会去 `torch.load` 那枚假件而撞在别的错上——
    那正好证明"拒绝"不是恒红的门。
    """

    same = tmp_path / "checkpoint.pt"
    same.write_bytes(b"not-a-torch-envelope")
    monkeypatch.setattr(
        sys, "argv",
        ["train_seed_corpus.py", "--resume", str(same), "--checkpoint", str(same),
         "--max-symbols", "10", "--progress", str(tmp_path / "progress.jsonl")],
    )
    outcome: list[str] = []
    try:
        trainer.main()
        outcome.append("returned")
    except SystemExit as exit_error:  # parser.error 会变成 SystemExit
        outcome.append(f"SystemExit:{exit_error.code}")
        err = capsys.readouterr().err
        assert GUARD_MESSAGE not in err, f"原地续训被误拒：{err[-300:]}"
    except Exception as error:  # torch.load 撞墙＝越过了闸门
        outcome.append(f"raised:{type(error).__name__}")
        assert GUARD_MESSAGE not in str(error)
    assert outcome, "main() 既没返回也没抛，测试面不成立"


def test_escape_flag_reaches_the_write_path(
    trainer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """逃生口必须**被走到**：给了 `--allow-overwrite-checkpoint` 就不该再撞这条拒绝。

    用 `--smoke` 真跑一次（5000 ticks，几十秒），终点是那枚旧件确实被覆写掉——
    这同时钉住"旗标不是装饰"。
    """

    victim = tmp_path / "checkpoint.pt"
    victim.write_bytes(b"old-bytes")
    monkeypatch.setattr(
        sys, "argv",
        ["train_seed_corpus.py", "--smoke", "--allow-overwrite-checkpoint",
         "--checkpoint", str(victim), "--progress", str(tmp_path / "progress.jsonl")],
    )
    result = trainer.main()
    assert result in (0, None), f"main 的返回约定变了（本支只证覆写被允许）：{result!r}"
    assert victim.read_bytes() != b"old-bytes", "旗标没被走到：旧件仍在"
