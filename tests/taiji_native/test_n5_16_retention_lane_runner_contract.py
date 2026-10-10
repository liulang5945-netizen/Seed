"""N5 保持侧跑道器（`run_taiji_n5_retention_lane.py`）的契约测。

跑道器买的是"双臂跑完之后不再手拼六条命令"——㊵-593 那份把带参旗标写成裸旗标的过期处方
就是手拼的代价。所以本册钉的是**计划本身**与**花钱前的拒绝**：

* 计划必须六条（两臂 × cap0＋replay＋判读），且每一条形参里都**显式**给 `--report`／`--out-report`
  ／`--out` ⇒ 不许任何一步落到出件方的默认路径上（默认路径历史上会覆写已封存件）；
* 判读步要**逐臂**带上各自的 `--consolidation-corpus`（㊵-565③：分离机检的材料＝本次通电自己的夜间件），
  缺语料时在花钱前拒绝；单跑面时它不是前置（别造假前置）；
* 预检读数为"退化"时必须拒绝，且点名是哪一枚键；
* 缺检查点时 rc=2，并且**不启动任何子进程**（那是"没算"，不是"跑砸了"）；
* 拒绝清单在 **GBK 控制台**上也要以 rc=2 出得来——上一格它是 rc=1＋traceback，只有起真进程才看得见；
* ㊵-654 加的两枚前置：点名材料若**整体早于该臂运行窗口** ⇒ 记 `substituted` 并在花钱前拒绝（要带
  `--allow-substituted-consolidation-material` 才放行），以及 G-N5d-4 那道"旗标有没有作用对象"的门
  必须排在四张面**之前**——它零算力，而 2×13 分钟已经不是零算力了。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
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


def test_per_arm_output_sinks_do_not_break_the_single_variable_pair() -> None:
    """㊵-650 的实测形状：两臂**必须**各写各的档，所以落点不是"第二个变量"。

    ㊵-647 那版把 argv 摘旗标后逐位比较 ⇒ 三枚落点差异也被当成破绽，双臂跑完之后才拦。
    """

    arms = _arms(
        COMMON + ["--resume", BASE, "--n5-shadow-gate", "1.0"] + _sinks("treated"),
        COMMON + ["--resume", BASE] + _sinks("control"),
    )
    assert MODULE.lineage_refusals(arms, base_checkpoint=BASE, base_sha256=BASE_SHA) == []


def test_two_arms_pointing_at_one_checkpoint_are_refused() -> None:
    """反向那一支：落点可以不同，但**不许相同**——同档会让第二臂原地覆写第一臂。"""

    shared = ["--checkpoint", "output/shared/checkpoint.pt"]
    arms = _arms(
        COMMON + ["--resume", BASE, "--n5-shadow-gate", "1.0"] + shared,
        COMMON + ["--resume", BASE] + shared,
    )
    refusals = MODULE.lineage_refusals(arms, base_checkpoint=BASE, base_sha256=BASE_SHA)
    assert any("--checkpoint" in row and "落点相同" in row for row in refusals), refusals


def _sinks(arm: str) -> list[str]:
    out = f"output/n5d_rebase_{arm}"
    return [
        "--checkpoint",
        f"{out}/checkpoint.pt",
        "--progress",
        f"{out}/progress.json",
        "--pressure-record",
        f"{out}/pressure.jsonl",
    ]


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


#: ---------------------------------------------------------------------------
#: ㊵-654 两枚新前置：材料溯源（④）与 G-N5d-4 自述在场性的门


def _night(tmp_path: Path, tag: str, stamp: str) -> Path:
    """一枚**夜间形状**的件名——溯源只认件名里的 UTC 时间戳，别给假形状。"""

    path = tmp_path / f"corpus-{stamp}-{tag}.jsonl"
    path.write_text('{"text":"ok"}\n', encoding="utf-8")
    return path


def _arm_with_accounting(tmp_path: Path, arm: str, elapsed_seconds: float) -> Path:
    """造一臂的落点：`checkpoint.pt` ＋训练器自己的完成记账 `progress_exit.json`。"""

    checkpoint = tmp_path / arm / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"x")
    (checkpoint.parent / "progress_exit.json").write_text(
        json.dumps({"elapsed_seconds": elapsed_seconds, "ticks": 1}), encoding="utf-8"
    )
    return checkpoint


def test_material_written_inside_the_arms_own_window_is_arm_local(tmp_path: Path) -> None:
    checkpoint = _arm_with_accounting(tmp_path, "treated", 600.0)
    stamp = datetime.fromtimestamp(checkpoint.stat().st_mtime - 60, UTC).strftime("%Y%m%dT%H%M%SZ")
    record = MODULE.consolidation_provenance("treated", checkpoint, [_night(tmp_path, "a1", stamp)])
    assert record["provenance"] == "arm_local", record
    assert "newest_material_before_run_utc" not in record, record


def test_material_older_than_the_run_is_named_substituted(tmp_path: Path) -> None:
    checkpoint = _arm_with_accounting(tmp_path, "control", 600.0)
    record = MODULE.consolidation_provenance(
        "control", checkpoint, [_night(tmp_path, "b2", "20200101T000000Z")]
    )
    assert record["provenance"] == "substituted", record
    assert record["newest_material_before_run_utc"].startswith("2020-01-01"), record


def test_uncomputable_provenance_is_unknown_not_arm_local(tmp_path: Path) -> None:
    """双向：拿不到窗口／拿不到时间戳，都只能记 `unknown`——把它读成"材料是对的"就是假通过。"""

    orphan = tmp_path / "no_accounting" / "checkpoint.pt"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"x")
    record = MODULE.consolidation_provenance(
        "treated", orphan, [_night(tmp_path, "c3", "20200101T000000Z")]
    )
    assert record["provenance"] == "unknown", record
    assert record["run_window_utc"] is None, record

    checkpoint = _arm_with_accounting(tmp_path, "control", 600.0)
    plain = tmp_path / "hand_collected.jsonl"
    plain.write_text('{"text":"ok"}\n', encoding="utf-8")
    bare = MODULE.consolidation_provenance("control", checkpoint, [plain])
    assert bare["provenance"] == "unknown", bare
    assert bare["night_timestamps_utc"] == {}, bare


def test_substituted_material_refuses_unless_the_caller_signs_it(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """④ 的拒绝必须在花钱前，且不写盘；显式旗标那条支也要真走一遍（否则它可能永远走不到）。"""

    def _boom(*args: object, **kwargs: object) -> int:
        raise AssertionError("材料溯源不齐时不许启动任何子进程")

    monkeypatch.setattr(MODULE, "_run", _boom)
    treated = _arm_with_accounting(tmp_path, "treated", 600.0)
    control = _arm_with_accounting(tmp_path, "control", 600.0)
    old = _night(tmp_path, "d4", "20200101T000000Z")
    argv = [
        "--treated-checkpoint",
        str(treated),
        "--control-checkpoint",
        str(control),
        "--out-dir",
        str(tmp_path / "lane"),
        "--treated-consolidation-corpus",
        str(old),
        "--control-consolidation-corpus",
        str(old),
    ]
    assert MODULE.main(argv + ["--dry-run"]) == 2
    out = capsys.readouterr().out
    assert "早于运行窗口" in out, out
    assert not (tmp_path / "lane").exists(), out

    assert MODULE.main(argv + ["--allow-substituted-consolidation-material", "--dry-run"]) == 0
    allowed = capsys.readouterr().out
    assert allowed.count("provenance=substituted") == 2, allowed
    #: 门排在四张面旁边一起出版——它零算力，所以该在花钱前就说话。
    assert "gate: $ " in allowed, allowed


def test_the_g_n5d_4_gate_names_an_existing_instrument() -> None:
    """旗标"有请求"不等于"有作用对象"：㊵-654 那次两臂 `n5_shadow` 整块缺席、
    `should_propose` 真值 0/9,727 ⇒ 两档 238/238 张量逐位相同，配对没有对象，而 2×13 分钟已付。
    本条只钉"门指向的仪器真的在场"，读数由下一条与真跑负责。"""

    assert (REPO / MODULE.SHADOW_PRESENCE).is_file(), MODULE.SHADOW_PRESENCE


def test_both_before_faces_reach_the_preflight_not_just_the_replay_one(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """㊵-656 修的是我自己上一格引进的缺口：`--cap0-before` 覆盖**没**传进预检。

    后果不是报错而是**静默不同源**——判读步读我点名的新基线面，预检却按默认常量去比
    `reports/taiji_n2_cap0_before_20261008.json`（另一条链的底面），于是
    `cap0_identity`／`cap0_checkpoint_identity` 两枚"verified"证的是**另一对面件**。
    换基座（甲路从 `seed_beta.pt` 起）时这条必然为真，所以它必须在跑双臂之前就被钉住。
    """

    monkeypatch.setattr(MODULE, "_run", lambda *a, **k: 0)
    treated = tmp_path / "treated.pt"
    control = tmp_path / "control.pt"
    treated.write_bytes(b"x")
    control.write_bytes(b"y")
    mine_cap0 = tmp_path / "my_base_cap0.json"
    mine_cap0.write_text("{}\n", encoding="utf-8")
    mine_replay = tmp_path / "my_base_replay.json"
    mine_replay.write_text("{}\n", encoding="utf-8")
    rc = MODULE.main(
        [
            "--treated-checkpoint",
            str(treated),
            "--control-checkpoint",
            str(control),
            "--out-dir",
            str(tmp_path / "lane"),
            "--cap0-before",
            str(mine_cap0),
            "--replay-before",
            str(mine_replay),
            *_corpus_args(tmp_path),
            "--allow-substituted-consolidation-material",
            "--dry-run",
        ]
    )
    assert rc == 0, rc
    out = capsys.readouterr().out
    preflight_line = [row for row in out.splitlines() if row.startswith("preflight: $ ")]
    assert len(preflight_line) == 1, out
    row = preflight_line[0]
    assert f"--cap0-before {mine_cap0}" in row, row
    assert f"--replay-before {mine_replay}" in row, row
    #: 反证：默认常量不许出现在被点名的面上——否则"覆盖了"这句话没有判别力。
    #: 取**裸文件名**比对而不是整路径：win32 上 `str(Path)` 会把分隔符写成反斜杠，
    #: 用 `as_posix()` 比会因分隔符不同而恒真（[[log-grouping-must-parse-per-block]] 的同族坑）。
    assert MODULE.CAP0_BEFORE.name not in row, row
    assert MODULE.REPLAY_BEFORE.name not in row, row


def _lane_args(tmp_path: Path, treated: Path, control: Path) -> list[str]:
    return [
        "--treated-checkpoint",
        str(treated),
        "--control-checkpoint",
        str(control),
        "--out-dir",
        str(tmp_path / "lane"),
        *_corpus_args(tmp_path),
        "--allow-substituted-consolidation-material",
        "--dry-run",
    ]


def test_the_presence_gate_gets_both_pressure_faces_when_they_exist(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """G-N5d-2 把四元组的核对**外包**给这台仪器（"本件不另写一份"），所以两面必须线进去。

    只给档不给面时它会出 `gate_differs_quadruple_unverified`⇒"核对过了"那句是我转述的。
    反向那一支也要走到：面不在场就出 `disclosure:`，不许静默省略。"""

    monkeypatch.setattr(MODULE, "_run", lambda *a, **k: 0)
    treated_dir = tmp_path / "treated"
    control_dir = tmp_path / "control"
    treated_dir.mkdir()
    control_dir.mkdir()
    treated = treated_dir / "checkpoint.pt"
    control = control_dir / "checkpoint.pt"
    treated.write_bytes(b"x")
    control.write_bytes(b"y")
    (treated_dir / "pressure.jsonl").write_text('{"kind":"face"}\n', encoding="utf-8")

    rc = MODULE.main(_lane_args(tmp_path, treated, control))
    assert rc == 0, rc
    out = capsys.readouterr().out
    gate_line = [row for row in out.splitlines() if row.startswith("gate: $ ")]
    assert len(gate_line) == 1, out
    assert f"--face-treated {treated_dir / 'pressure.jsonl'}" in gate_line[0], gate_line
    #: 控制臂没有面 ⇒ 不许静默，要出披露。
    assert "--face-control" not in gate_line[0], gate_line
    assert any(row.startswith("disclosure: --face-control") for row in out.splitlines()), out



