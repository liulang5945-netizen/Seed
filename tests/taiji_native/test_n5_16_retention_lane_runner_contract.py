"""N5 保持侧跑道器（`run_taiji_n5_retention_lane.py`）的契约测。

跑道器买的是"双臂跑完之后不再手拼六条命令"——㊵-593 那份把带参旗标写成裸旗标的过期处方
就是手拼的代价。所以本册钉的是**计划本身**与**花钱前的拒绝**：

* 计划必须六条（两臂 × cap0＋replay＋判读），且每一条形参里都**显式**给 `--report`／`--out-report`
  ／`--out` ⇒ 不许任何一步落到出件方的默认路径上（默认路径历史上会覆写已封存件）；
* 预检读数为"退化"时必须拒绝，且点名是哪一枚键；
* 缺检查点时 rc=2，并且**不启动任何子进程**（那是"没算"，不是"跑砸了"）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "run_taiji_n5_retention_lane.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_16_lane_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()

DEGRADED = {
    "cap0": {"cap0_identity": "verified"},
    "base_checkpoint": {"cap0_checkpoint_identity": "verified"},
    "replay": {"replay_identity": "degraded_to_offset_and_first_item"},
}
HASHED = json.loads(json.dumps(DEGRADED))
HASHED["replay"]["replay_identity"] = "content_hash_available"


def test_the_plan_is_six_steps_with_explicit_output_paths(tmp_path: Path) -> None:
    plan = MODULE.build_plan(
        treated=tmp_path / "treated.pt",
        control=tmp_path / "control.pt",
        out_dir=tmp_path / "lane",
        replay_limit=24,
        only=None,
    )
    joined = [" ".join(row) for row in plan]
    assert len(plan) == 6, joined
    #: 每一步都必须显式给输出路径——默认路径覆写封存件是这仓的真实历史（DEBT 族）。
    assert all(("--report " in row or "--out-report " in row or "--out " in row) for row in joined)
    assert sum("eval_taiji_cap0_baseline.py" in row for row in joined) == 2
    assert sum("measure_taiji_a30_repetition_penalty.py" in row for row in joined) == 2
    assert sum("adjudicate_taiji_n2_04_retention_pair.py" in row for row in joined) == 2
    #: 判读步的 before 侧默认指向**带哈希**的新面（㊵-645 出路①），不是旧的那张退化面。
    assert all(MODULE.REPLAY_BEFORE.name in row for row in joined if "adjudicate" in row)
    assert all("treated" in row or "control" in row for row in joined)


def test_a_degraded_preflight_is_refused_by_name(tmp_path: Path) -> None:
    refusals = MODULE.preflight_refusals(DEGRADED)
    assert refusals == [
        "replay.replay_identity='degraded_to_offset_and_first_item'（需要 'content_hash_available'）"
    ], refusals
    assert MODULE.preflight_refusals(HASHED) == []


BASE = "checkpoints/seed_a31self_with_circuit.pt"
BASE_SHA = "d6169a358eaee6d194d4795e3167a7bcbb42dfde57b92aa1199bcebbed89699b"
COMMON = ["train_seed_corpus.py", "--readout", "predictive", "--max-symbols", "60000"]


def _arms(treated_argv: list[str], control_argv: list[str]) -> dict[str, dict[str, Any]]:
    return {
        "treated": {
            "argv": treated_argv,
            "corpus_fingerprint": "fp-1",
            "config": {"taiji": {"seed": 7}},
        },
        "control": {
            "argv": control_argv,
            "corpus_fingerprint": "fp-1",
            "config": {"taiji": {"seed": 7}},
        },
    }


def test_a_clean_single_variable_pair_is_accepted() -> None:
    arms = _arms(
        COMMON + ["--resume", BASE, "--n5-shadow-gate", "1.0"],
        COMMON + ["--resume", BASE],
    )
    assert MODULE.lineage_refusals(arms, base_checkpoint=BASE, base_sha256=BASE_SHA) == []


def test_a_backslash_written_base_is_the_same_base(tmp_path: Path) -> None:
    """win32 上同一枚基座会被写成 `checkpoints\\x` 或 `checkpoints/x` ⇒ 比较前必须归一。"""

    arms = _arms(
        COMMON + ["--resume", BASE.replace("/", "\\"), "--n5-shadow-gate", "1.0"],
        COMMON + ["--resume", BASE],
    )
    assert MODULE.lineage_refusals(arms, base_checkpoint=BASE, base_sha256=BASE_SHA) == []


def test_a_second_moving_variable_is_refused_not_tolerated() -> None:
    """这一支钉的是我第一版的启发式错误：只按"旗标之外还有没有别的差异"放行，
    会把 `--max-symbols 60000` 对 `--max-symbols 40000` 也放过（都是数字）。"""

    arms = _arms(
        COMMON + ["--resume", BASE, "--n5-shadow-gate", "1.0", "--limit", "24"],
        COMMON + ["--resume", BASE, "--limit", "25"],
    )
    refusals = MODULE.lineage_refusals(arms, base_checkpoint=BASE, base_sha256=BASE_SHA)
    assert any("单变量前提破了" in row for row in refusals), refusals


def test_a_forgotten_resume_or_missing_command_surface_is_refused() -> None:
    arms = _arms(
        COMMON + ["--n5-shadow-gate", "1.0"],
        COMMON,
    )
    refusals = MODULE.lineage_refusals(arms, base_checkpoint=BASE, base_sha256=BASE_SHA)
    assert any("--resume" in row for row in refusals), refusals

    arms["treated"]["argv"] = None
    refusals = MODULE.lineage_refusals(arms, base_checkpoint=BASE, base_sha256=BASE_SHA)
    assert any("command_surface.argv" in row for row in refusals), refusals


def test_a_pair_that_differs_in_nothing_has_no_two_arms() -> None:
    """两臂旗标请求值相同 ⇒ 这一对不可比（控制臂必须**不给**那枚旗标，或给不同的值）。"""

    arms = _arms(
        COMMON + ["--resume", BASE, "--n5-shadow-gate", "1.0"],
        COMMON + ["--resume", BASE, "--n5-shadow-gate", "1.0"],
    )
    refusals = MODULE.lineage_refusals(arms, base_checkpoint=BASE, base_sha256=BASE_SHA)
    assert any("没有可比的两臂" in row for row in refusals), refusals


def test_a_different_corpus_fingerprint_breaks_the_pair() -> None:
    arms = _arms(
        COMMON + ["--resume", BASE, "--n5-shadow-gate", "1.0"],
        COMMON + ["--resume", BASE],
    )
    arms["control"]["corpus_fingerprint"] = "fp-OTHER"
    refusals = MODULE.lineage_refusals(arms, base_checkpoint=BASE, base_sha256=BASE_SHA)
    assert any("corpus_fingerprint" in row for row in refusals), refusals


def test_a_missing_base_digest_is_refused() -> None:
    arms = _arms(
        COMMON + ["--resume", BASE, "--n5-shadow-gate", "1.0"],
        COMMON + ["--resume", BASE],
    )
    refusals = MODULE.lineage_refusals(arms, base_checkpoint=BASE, base_sha256=None)
    assert any("拿不到基座摘要" in row for row in refusals), refusals


def test_a_missing_checkpoint_refuses_before_launching_anything(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def _boom(*args: object, **kwargs: object) -> int:
        raise AssertionError("缺检查点时不许启动任何子进程")

    monkeypatch.setattr(MODULE, "_run", _boom)
    rc = MODULE.main(
        [
            "--treated-checkpoint",
            str(tmp_path / "nope.pt"),
            "--control-checkpoint",
            str(tmp_path / "nope2.pt"),
            "--out-dir",
            str(tmp_path / "lane"),
        ]
    )
    assert rc == 2, rc
    assert "REFUSE" in capsys.readouterr().out


def test_dry_run_prints_the_plan_without_running_a_single_step(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def _boom(*args: object, **kwargs: object) -> int:
        raise AssertionError("--dry-run 不许启动任何子进程")

    monkeypatch.setattr(MODULE, "_run", _boom)
    treated = tmp_path / "treated.pt"
    control = tmp_path / "control.pt"
    treated.write_bytes(b"x")
    control.write_bytes(b"y")
    rc = MODULE.main(
        [
            "--treated-checkpoint",
            str(treated),
            "--control-checkpoint",
            str(control),
            "--out-dir",
            str(tmp_path / "lane"),
            "--dry-run",
        ]
    )
    assert rc == 0, rc
    out = capsys.readouterr().out
    assert out.count("plan: $ ") == 6, out
    assert "DRY_RUN commands=6" in out
    assert not (tmp_path / "lane").exists()
