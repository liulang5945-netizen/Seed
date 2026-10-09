"""㊵-626：训练器命令面自述的合同（DEBT-G70① 的落码侧）。

这条债的形状很特别：**过期的不是结论，是处方**。㊵-593 那份「N5 冻结命令」把
`--pressure-record`（真实签名 `type=Path`，要一个 JSONL 路径）写成裸旗标 ⇒ 下一次照抄就在
argparse 阶段 `parser.error`（㊵-616 实测两臂各 rc=2、零训练发生）。散文里的命令没有任何东西
钉它与代码签名一致，所以修法不是"再抄一遍正确的命令"，而是**让每次跑自己把 argv 登记进件里**。

四支测各自都能为假：
1. 面头里确有 `argv` 登记（且不是重建的参数表）；
2. 信封里的 `command_surface` 填充**早于** `atomic_save`（顺序缺陷的先例：㊵-593）；
3. 真跑一次（200 符号、全新写靶）⇒ 落盘件与面头的 argv **逐字回显**本次调用；
4. 反向：面头不许塞时钟类键（复算要能逐字对比）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "scripts" / "training" / "train_seed_corpus.py"

FACE_ARGV_ANCHOR = '"argv": list(sys.argv),'
ENVELOPE_FILL_ANCHOR = 'envelope["command_surface"] = {'
SAVE_CALL = "atomic_save(envelope, checkpoint_path)"


def _text() -> str:
    return TRAINER.read_text(encoding="utf-8")


def test_face_header_registers_argv_verbatim() -> None:
    text = _text()
    assert text.count(FACE_ARGV_ANCHOR) >= 2, "面头与信封两处登记，少一处就还是散文处方"
    #: 索引必须**从面头那处往后找**——全文 `index` 会先命中更靠前的信封填充（我第一次就是这么错的）。
    face_at = text.index('"kind": "face",')
    argv_at = text.index(FACE_ARGV_ANCHOR, face_at)
    fmt_at = text.index('"taiji-n3-pressure-face-v2"', face_at)
    assert face_at < argv_at < fmt_at, (face_at, argv_at, fmt_at)


def test_command_surface_is_filled_before_the_atomic_save() -> None:
    text = _text()
    fill = text.index(ENVELOPE_FILL_ANCHOR)
    save = text.index(SAVE_CALL)
    assert fill < save, "填充必须早于落盘（㊵-593 的顺序缺陷形状不许回退）"
    assert text.count(ENVELOPE_FILL_ANCHOR) == 1


def test_no_clock_is_published_in_the_command_surface() -> None:
    #: 面头与信封的命令面要能**逐字对比**，塞时钟会让每次复算都不同＝假差异。
    text = _text()
    surface = text[text.index(ENVELOPE_FILL_ANCHOR) : text.index(SAVE_CALL)]
    for banned in ("time.", "datetime", "utc"):
        assert banned not in surface, banned


def test_a_real_run_round_trips_its_own_argv(tmp_path: Path) -> None:
    out_dir = tmp_path / "cs"
    out_dir.mkdir(parents=True)
    argv = [
        sys.executable,
        str(TRAINER),
        "--readout",
        "predictive",
        "--no-readout-position",
        "--pressure-record",
        str(out_dir / "pressure.jsonl"),
        "--developmental-fast-slow",
        "--developmental-bridge-gate",
        "1.0",
        "--growth-min-pressure",
        "0.05",
        "--n5-shadow",
        "--n5-shadow-gate",
        "1.0",
        "--max-symbols",
        "200",
        "--checkpoint",
        str(out_dir / "checkpoint.pt"),
        "--progress",
        str(out_dir / "progress.jsonl"),
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))  # noqa: S603
    assert proc.returncode == 0, proc.stderr[-600:]
    checkpoint = out_dir / "checkpoint.pt"
    assert checkpoint.is_file(), "跑绿了却没落盘＝取法或写靶有问题，别当成测过了"

    envelope = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    surface = envelope.get("command_surface")
    assert isinstance(surface, dict), sorted(envelope)
    assert surface["trainer"] == "train_seed_corpus"
    #: 逐字回显本次调用。子进程的 `sys.argv[0]` 是**脚本路径**（不是我传的 python 解释器），
    #: 所以要比的是 `argv[1:]`——把这条写明白是因为我在这里先算错了一次（写成 `argv[-19:]`）。
    assert list(surface["argv"]) == argv[1:], (surface["argv"], argv[1:])

    header = json.loads((out_dir / "pressure.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "argv" in header, sorted(header)
    assert list(header["argv"]) == argv[1:]
    #: 面头其余键不受影响（命令面是**加**进去的，不是替换）。
    assert header["kind"] == "face"
    assert header["format"] == "taiji-n3-pressure-face-v2"
