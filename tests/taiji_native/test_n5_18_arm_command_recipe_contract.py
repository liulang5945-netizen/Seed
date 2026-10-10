"""双臂 60k 续训**处方**的守卫：逐字命令必须能被现行 argparse 接受，且缩写处方确实会被拦。

为什么这一格值得花零算力的十分钟：本仓为"交接里的过期处方"付过两次学费——
㊵-593 把 `--pressure-record` 写成裸旗标 ⇒ 照抄在 argparse 阶段就 `parser.error`（㊵-616 实测两臂各 rc=2、
零训练发生）；而 ㊵-638④ 那段本身也是**形状缩写**：它列了 `--n5-shadow --growth-min-pressure 0.65
--developmental-fast-slow --developmental-bridge-gate 1`，却没写这三枚旗标的**前置**
`--pressure-record <path>`（`train_seed_corpus.py:1293-1303` 的响亮拒绝），也没写 `--checkpoint`
（:1338-1342，缺省会写产品件 ⇒ 同样拒绝）。所以"命令在 08 ㊵-638④"这句话对下一位是不够的。

本册钉四件事，全部**不训练**：
1. 逐字双臂命令过 `_build_parser().parse_args()`，并把关键 namespace 值钉成预期；
2. 两臂只差 `--n5-shadow-gate` 一枚（与跑道器 `lineage_refusals` 同一口径，不另造判据）；
3. 每个旗标名都出现在 `--help` 里（㊵-640 那族的反面：帮助文本被 `%` 插值崩过一次，无人发现）；
4. **负对照**：把 `--pressure-record` 摘掉 ⇒ 真进程 rc=2 且点名该旗标 ⇒ 证明第 1 条不是恒真。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from train_seed_corpus import _build_parser  # noqa: E402

TRAINER = PROJECT_ROOT / "scripts" / "training" / "train_seed_corpus.py"
LANE_RUNNER = PROJECT_ROOT / "scripts" / "training" / "run_taiji_n5_retention_lane.py"

BASE = "checkpoints/seed_a31self_with_circuit.pt"
CORPUS = "data/simple_zh/dialogue_extended_clean.jsonl"
#: 两枚臂目录＝03/08 里已承诺的落点；`--checkpoint` 必须显式给，否则训练器会拒绝写产品件。
ARM_OUT = {"treated": "output/n5d_rebase_treated", "control": "output/n5d_rebase_control"}


def arm_argv(arm: str) -> list[str]:
    out = ARM_OUT[arm]
    argv = [
        "--corpus",
        CORPUS,
        "--resume",
        BASE,
        "--checkpoint",
        f"{out}/checkpoint.pt",
        "--progress",
        f"{out}/progress.json",
        "--pressure-record",
        f"{out}/pressure.jsonl",
        "--readout",
        "predictive",
        "--n5-shadow",
        "--developmental-fast-slow",
        "--developmental-bridge-gate",
        "1",
        "--growth-min-pressure",
        "0.65",
        "--max-symbols",
        "60000",
    ]
    if arm == "treated":
        argv += ["--n5-shadow-gate", "1.0"]
    return argv


def _load_lane_runner() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("n5_18_lane_runner", LANE_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arm", ("treated", "control"))
def test_the_literal_arm_recipe_parses_against_the_current_trainer(arm: str) -> None:
    args = _build_parser().parse_args(arm_argv(arm))
    assert args.readout == "predictive"
    assert args.max_symbols == 60000
    assert args.resume == BASE
    assert args.corpus == [CORPUS]
    assert args.pressure_record is not None
    assert args.developmental_fast_slow is True
    assert float(args.developmental_bridge_gate) > 0.0
    assert args.growth_minimum_pressure == 0.65
    assert args.n5_shadow is True
    assert (args.n5_shadow_gate == 1.0) is (arm == "treated")


def test_the_published_pair_passes_the_lane_runners_own_lineage_gate() -> None:
    """直接走**跑道器的判定入口**，不在测里另抄一份比较逻辑。

    这一支是 ㊵-650 的来由：㊵-647 那版把 argv 摘掉旗标后逐位比较，于是两臂**各自不同的落点**
    （`--checkpoint`／`--progress`／`--pressure-record`）也被当成"第二个变量" ⇒ 双臂跑完（每臂
    334～473 秒）之后才会被这道门拦死。处方先过门，才谈得上让人去跑。
    """

    runner = _load_lane_runner()
    left, right = arm_argv("treated"), arm_argv("control")
    arms = {
        arm: {
            "argv": row,
            "corpus_fingerprint": "fp-1",
            "config": {"taiji": {"seed": 20260822}},
        }
        for arm, row in (("treated", left), ("control", right))
    }
    assert runner.lineage_refusals(arms, base_checkpoint=BASE, base_sha256="a" * 64) == []
    for flag in runner.ARM_LOCAL_FLAGS:
        assert runner._value_of(left, flag) != runner._value_of(right, flag), flag
    assert runner._gate_value(left) != runner._gate_value(right)


def test_every_flag_in_the_recipe_is_documented() -> None:
    proc = subprocess.run(
        [sys.executable, str(TRAINER), "--help"], capture_output=True, check=False
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")[:400]
    text = proc.stdout.decode("utf-8", errors="replace")
    flags = [tok for tok in arm_argv("treated") if tok.startswith("--")]
    assert len(set(flags)) >= 10, flags
    missing = [flag for flag in sorted(set(flags)) if flag not in text]
    assert missing == [], missing


def test_dropping_pressure_record_is_refused_by_the_real_process(tmp_path: Path) -> None:
    """负对照：第 1 条测不是恒真——摘掉前置旗标 ⇒ 训练器在**任何训练之前**响亮拒绝。

    落点全部指向 `tmp_path`、`--max-symbols 1`，所以即使拒绝逻辑哪天失效，这一支也只会训 1 个符号
    并把档写在仓外，不会碰到产品件。
    """

    argv = [tok for tok in arm_argv("treated")]
    index = argv.index("--pressure-record")
    del argv[index : index + 2]
    argv[argv.index("--checkpoint") + 1] = str(tmp_path / "checkpoint.pt")
    argv[argv.index("--progress") + 1] = str(tmp_path / "progress.json")
    argv[argv.index("--max-symbols") + 1] = "1"
    proc = subprocess.run([sys.executable, str(TRAINER)] + argv, capture_output=True, check=False)
    assert proc.returncode == 2, proc.returncode
    err = proc.stderr.decode("utf-8", errors="replace")
    assert "--pressure-record" in err, err[:400]
    assert not (tmp_path / "checkpoint.pt").exists(), "拒绝必须发生在任何落盘之前"


def test_the_recipe_is_published_verbatim_in_the_ledger() -> None:
    """处方与守卫不许分居两处还互相不认识：08 ㊵-650 那格必须逐字带着这两枚落点与语料名。"""

    text = (
        PROJECT_ROOT / "plans" / "active" / "roadmap" / "08_UPSTREAM_SYNC_PLAYBOOK.md"
    ).read_text(encoding="utf-8")
    entry = [l for l in text.split("\n") if l.startswith("【2026-10-10 ㊵-650 ")]
    assert len(entry) == 1, len(entry)
    for needle in (
        CORPUS,
        BASE,
        f"{ARM_OUT['treated']}/pressure.jsonl",
        f"{ARM_OUT['control']}/pressure.jsonl",
        "--developmental-bridge-gate 1",
    ):
        assert needle in entry[0], needle
