"""N5 保持侧跑道器（`run_taiji_n5_retention_lane.py`）的契约测。

跑道器买的是"双臂跑完之后不再手拼六条命令"——㊵-593 那份把带参旗标写成裸旗标的过期处方
就是手拼的代价。所以本册钉的是**计划本身**与**花钱前的拒绝**：

* 计划必须六条（两臂 × cap0＋replay＋判读），且每一条形参里都**显式**给 `--report`／`--out-report`
  ／`--out` ⇒ 不许任何一步落到出件方的默认路径上（默认路径历史上会覆写已封存件）；
* 判读步要**逐臂**带上各自的 `--consolidation-corpus`（㊵-565③：分离机检的材料＝本次通电自己的夜间件），
  缺语料时在花钱前拒绝；单跑面时它不是前置（别造假前置）；
* 预检读数为"退化"时必须拒绝，且点名是哪一枚键；
* 缺检查点时 rc=2，并且**不启动任何子进程**（那是"没算"，不是"跑砸了"）；
* 拒绝清单在 **GBK 控制台**上也要以 rc=2 出得来——上一格它是 rc=1＋traceback，只有起真进程才看得见。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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
        consolidation_corpus=_corpus(tmp_path),
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


def _corpus(tmp_path: Path) -> dict[str, list[Path]]:
    #: 两臂各指各的语料：同名会让下面那条"逐臂线进去了吗"的断言变成恒真。
    return {
        "treated": [tmp_path / "treated-night.jsonl"],
        "control": [tmp_path / "control-night.jsonl"],
    }


def test_each_adjudication_step_carries_its_own_arms_corpus(tmp_path: Path) -> None:
    """G_N2c_4_disjointness 的输入必须**逐臂**线进去——㊵-565③ 冻的口径是"本次通电自己的夜间语料"，
    把一枚默认值同时喂两臂会让控制臂的分离检查建立到治疗臂的材料上。"""

    plan = MODULE.build_plan(
        treated=tmp_path / "treated.pt",
        control=tmp_path / "control.pt",
        out_dir=tmp_path / "lane",
        consolidation_corpus=_corpus(tmp_path),
        replay_limit=24,
        only=None,
    )
    rows = {" ".join(row): row for row in plan if "adjudicate" in " ".join(row)}
    assert len(rows) == 2, list(rows)
    for arm in MODULE.ARMS:
        row = next(r for text, r in rows.items() if f"{arm}_cap0_after" in text)
        given = [row[i + 1] for i, tok in enumerate(row) if tok == "--consolidation-corpus"]
        assert given == [str(_corpus(tmp_path)[arm][0])], (arm, given)
        #: 语料必须排在 --out 之前，且 `--out` 仍是最后一枚（覆写目标显式在场）。
        assert row[-2] == "--out", row[-4:]


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
            *_corpus_args(tmp_path),
            "--dry-run",
        ]
    )
    assert rc == 0, rc
    out = capsys.readouterr().out
    assert out.count("plan: $ ") == 6, out
    assert "DRY_RUN commands=6" in out
    assert not (tmp_path / "lane").exists()


def _corpus_args(tmp_path: Path) -> list[str]:
    argv: list[str] = []
    for arm, paths in _corpus(tmp_path).items():
        for path in paths:
            path.write_text('{"text":"ok"}\n', encoding="utf-8")
            argv += [f"--{arm}-consolidation-corpus", str(path)]
    return argv


def test_a_missing_arm_corpus_refuses_before_launching_anything(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """缺语料不是"最后一步才红"的事：四张面（约 13 分钟）先付掉，判读仍拿不到分离读数。"""

    def _boom(*args: object, **kwargs: object) -> int:
        raise AssertionError("巩固语料不齐时不许启动任何子进程")

    monkeypatch.setattr(MODULE, "_run", _boom)
    treated = tmp_path / "treated.pt"
    control = tmp_path / "control.pt"
    treated.write_bytes(b"x")
    control.write_bytes(b"y")
    for only in (None, "adjudicate"):
        argv = [
            "--treated-checkpoint",
            str(treated),
            "--control-checkpoint",
            str(control),
            "--out-dir",
            str(tmp_path / "lane"),
        ]
        if only is not None:
            argv += ["--only", only]
        rc = MODULE.main(argv)
        assert rc == 2, (only, rc)
        out = capsys.readouterr().out
        assert "没点名巩固语料" in out, (only, out)
        assert out.count("  - ") == 2, (only, out)


def test_face_only_runs_do_not_invent_a_corpus_precondition(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """单跑面时语料不是前置——把它变成假前置会让人在 `--only cap0` 上凑一枚错语料。"""

    monkeypatch.setattr(MODULE, "_run", lambda *a, **k: 0)
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
            "--only",
            "cap0",
            "--dry-run",
        ]
    )
    assert rc == 0, rc
    out = capsys.readouterr().out
    assert "DRY_RUN commands=2" in out, out


def test_the_corpus_check_can_actually_pass_and_actually_fail(tmp_path: Path) -> None:
    """双向都测：只验拒绝分支等于没验（[[probe-output-must-be-verified-present]]）。"""

    present = tmp_path / "night.jsonl"
    present.write_text('{"text":"ok"}\n', encoding="utf-8")
    assert MODULE.corpus_refusals({"treated": [present], "control": [present]}) == []
    assert len(MODULE.corpus_refusals({"treated": [], "control": []})) == 2
    refusals = MODULE.corpus_refusals({"treated": [present], "control": [tmp_path / "gone.jsonl"]})
    assert refusals == ["control: 巩固语料不在场：%s" % (tmp_path / "gone.jsonl")], refusals


def test_the_refusal_list_survives_a_gbk_console(tmp_path: Path) -> None:
    """拒绝清单是 fail-closed 唯一的输出面，而 win32 控制台默认按 GBK 编码 stdout。

    上一格实测：capsys 全绿的同时，真进程在打印第一条拒绝时 `UnicodeEncodeError` 退出 **rc=1**——
    traceback 取代了设计好的 rc=2 与逐条点名。所以这一支**必须起子进程**才测得到（[[handoff-claims-must-be-executed]]）。
    """

    treated = tmp_path / "treated.pt"
    control = tmp_path / "control.pt"
    treated.write_bytes(b"x")
    control.write_bytes(b"y")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--treated-checkpoint",
            str(treated),
            "--control-checkpoint",
            str(control),
            "--out-dir",
            str(tmp_path / "lane"),
            "--dry-run",
        ],
        capture_output=True,
        cwd=str(REPO),
        env=dict(os.environ, PYTHONIOENCODING="gbk"),
        check=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    assert proc.returncode == 2, (proc.returncode, stdout, stderr)
    assert "Traceback" not in stderr, stderr
    assert "⇒" in stdout, stdout
    assert stdout.count("  - ") == 2, stdout
    #: 拒绝发生在任何写盘之前——出件目录不该存在。
    assert not (tmp_path / "lane").exists(), stdout
