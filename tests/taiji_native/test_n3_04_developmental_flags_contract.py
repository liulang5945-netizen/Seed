"""PLAN-N3-04 §5bis 两个默认关旗标的契约测（训练器侧＋判读器侧）。

两支都必须走到，理由是本件要防的正是“以为接上了”：

* **默认关那半**＝不开旗标时**根本不调**发育装配的 API（类层面记账），面首自述里
  `fast_slow_requested=false`、`bridge_gate_actual=0.0`、模式读数是 `read_only`。
  “逐位不变”的另一半证明住在守卫臂的进度行八键对照里（真实面），不在这里冒充。
* **开旗标那半**＝开了之后必须真的挂上 bundle、真的把模式设成 `fast_slow`、
  gate 的**请求值与读回值**都进面件（在场性不许由命令行反推，§3）。
* **判读器三条拒绝支各自 rc=2**：缺 developmental 自述键／模式读数里出现 `read_only`／
  `bridge_gate_actual ≤ 0`（那一臂没放行，`activity_saturation` 结构上恒零）。
  每条都造一份真件跑过去，不接受“看代码应该会拒”。
* **默认档形状不变**＝`--prereg n3-01` 的输出里没有 `assembly` 那半，步骤一的四枚在库读数件
  因此还能逐位复算。
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.training import train_seed_corpus as trainer  # noqa: E402

READER = REPO / "scripts/training/count_taiji_n3_pressure_thresholds.py"

#: 只有这三个 API 属于“新旗标才会碰”的集合；`enable_adaptive_residual_bridge` /
#: `enable_adaptive_residual_growth` 是步骤一压强面本来就做的，不记在这里以免把“挂载”误说成“装配改动”。
ASSEMBLY_APIS = (
    "migrate_f1_to_developmental_synapses",
    "set_developmental_f1_learning_mode",
    "set_adaptive_residual_bridge_gate",
)

PRESSURE_WEIGHTS = {
    "residual_error": 0.30,
    "fast_slow_conflict": 0.25,
    "activity_saturation": 0.20,
    "utility_gap": 0.25,
}


class _Drop:
    """哨兵：让合成件**真的缺掉**某个自述键（测拒绝支用）。"""


DROP = _Drop()


def _reader():
    spec = importlib.util.spec_from_file_location("n3_pressure_reader", READER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    docs = ["问：甲是什么？答：甲是一个符号串。", "问：乙呢？答：乙是另一个符号串。"]
    path.write_text(
        "".join(json.dumps({"text": doc}) + "\n" for doc in docs), encoding="utf-8", newline="\n"
    )
    return path


class _Spy:
    def __init__(self) -> None:
        from taiji.model import Taiji

        self.counts = {name: 0 for name in ASSEMBLY_APIS}
        self._originals: dict[str, Any] = {}
        for name in ASSEMBLY_APIS:
            original = getattr(Taiji, name)
            self._originals[name] = original
            self._install(name, original)

    def _install(self, name: str, original: Any) -> None:
        from taiji.model import Taiji

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.counts[name] += 1
            return original(*args, **kwargs)

        setattr(Taiji, name, wrapper)

    def restore(self) -> None:
        from taiji.model import Taiji

        for name, original in self._originals.items():
            setattr(Taiji, name, original)


def _train(tmp_path: Path, **kwargs: Any) -> Path:
    from seed import SeedConfig

    pressure = tmp_path / "pressure.jsonl"
    run = trainer.run_training
    run(
        corpus_paths=[_corpus(tmp_path)],
        config=SeedConfig(),
        epochs=1,
        checkpoint_path=tmp_path / "checkpoint.pt",
        progress_path=tmp_path / "progress.jsonl",
        checkpoint_every=1_000_000,
        progress_every=1_000_000,
        max_symbols=120,
        readout="predictive",
        pressure_record=pressure,
        **kwargs,
    )
    return pressure


def _line(pressure: Path, kind: str) -> dict[str, Any]:
    for raw in pressure.read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        if record.get("kind") == kind:
            return record
    raise AssertionError(f"面里没有 {kind} 行")


def test_default_off_never_calls_the_assembly_apis(tmp_path: Path) -> None:
    spy = _Spy()
    try:
        pressure = _train(tmp_path)
    finally:
        spy.restore()
    dev = _line(pressure, "face")["developmental"]
    assert spy.counts == dict.fromkeys(ASSEMBLY_APIS, 0), spy.counts
    assert dev["fast_slow_requested"] is False
    assert dev["bundle"] is None
    assert dev["bridge_gate_requested"] is None
    assert float(dev["bridge_gate_actual"]) == 0.0
    assert [item["learning_mode"] for item in dev["mode_readings"]] == ["read_only"]
    tail = _line(pressure, "tail")
    assert tail["learning_mode_at_close"] == "read_only"


def test_flags_on_mounts_bundle_sets_mode_and_reads_back_gate(tmp_path: Path) -> None:
    pressure = _train(tmp_path, developmental_fast_slow=True, developmental_bridge_gate=0.25)
    dev = _line(pressure, "face")["developmental"]
    assert dev["fast_slow_requested"] is True
    assert dev["bundle"] is not None
    assert dev["bundle"]["format"] == "taiji-developmental-f1-migration-v1"
    #: 挂载即 `fast_is_zero=true`（PLAN-N3-03 §3：光挂载不够，必须靠写入模式才真往 fast_delta 写）。
    assert dev["bundle"]["fast_is_zero"] is True
    assert dev["bridge_gate_requested"] == 0.25
    assert float(dev["bridge_gate_actual"]) == 0.25
    assert [item["learning_mode"] for item in dev["mode_readings"]] == ["fast_slow"]
    tail = _line(pressure, "tail")
    assert tail["learning_mode_at_close"] == "fast_slow"
    assert float(tail["bridge_gate_at_close"]) == 0.25


def test_position_input_and_developmental_bundle_are_mutually_exclusive() -> None:
    """产品自己的响亮守卫（`taiji/model.py` 的 `_reject_position_input_without_learning_path`）：
    位置输入开着就想挂发育 bundle ⇒ ValueError。这条决定了真实面必须带 `--no-readout-position`。"""

    from taiji import TaijiConfig
    from taiji.model import Taiji

    substrate = Taiji(TaijiConfig(readout_utf8_position_input=True))
    with pytest.raises(ValueError, match="developmental F1"):
        substrate.migrate_f1_to_developmental_synapses()


def test_help_lists_both_new_flags() -> None:
    #: argparse 会对 help 做 % 插值：裸百分号让 `--help` 直接崩（本仓踩过），所以这里断言
    #: 两个旗标名**确实出现在帮助输出里**，而不是只断言它们进了 parser。
    help_text = trainer._build_parser().format_help()
    assert "--developmental-fast-slow" in help_text
    assert "--developmental-bridge-gate" in help_text


@pytest.mark.parametrize(
    "argv,expect",
    [
        (["--developmental-fast-slow"], "--pressure-record"),
        (["--developmental-bridge-gate", "0.25"], "--pressure-record"),
        (["--developmental-bridge-gate=1.5", "--pressure-record", "x.jsonl"], "0.0..1.0"),
        (["--developmental-bridge-gate=-0.1", "--pressure-record", "x.jsonl"], "0.0..1.0"),
    ],
)
def test_flag_misuse_is_rejected_loudly(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expect: str
) -> None:
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "argv", ["train_seed_corpus.py", *argv])
    with redirect_stderr(buffer), pytest.raises(SystemExit) as exit_info:
        trainer.main()
    assert exit_info.value.code == 2
    assert expect in buffer.getvalue(), buffer.getvalue()


def _synthetic_face(rows: int = 520, zero_every: int = 40, **dev_overrides: Any) -> list[str]:
    dev: dict[str, Any] = {
        "fast_slow_requested": True,
        "bundle": {
            "format": "taiji-developmental-f1-migration-v1",
            "fast_is_zero": True,
            "effective_parameter_count": 12912,
        },
        "bridge_gate_requested": 0.25,
        "bridge_gate_actual": 0.25,
        "mode_readings": [{"after": "mount", "tick": 0, "learning_mode": "fast_slow"}],
        "_mode_at_close": "fast_slow",
    }
    for key, value in dev_overrides.items():
        if isinstance(value, _Drop):
            dev.pop(key, None)
        else:
            dev[key] = value
    reported = {key: value for key, value in dev.items() if not key.startswith("_")}
    lines = [
        json.dumps(
            {
                "kind": "face",
                "format": "taiji-n3-pressure-face-v1",
                "bridge": {"gate": reported.get("bridge_gate_actual", 0.0)},
                "growth": {"parent_checkpoint_digest": "y"},
                "policy": {
                    "minimum_pressure": 0.70,
                    "required_pressure_steps": 3,
                    "growth_resource_cost": 0.05,
                },
                "developmental": reported,
                "readout": "predictive",
                "seed": 20260822,
            },
            ensure_ascii=False,
        )
    ]
    for index in range(rows):
        signals = {
            "residual_error": 0.5 + 0.001 * (index % 3),
            "fast_slow_conflict": 0.4 if index % zero_every else 0.0,
            "activity_saturation": 0.3,
            "utility_gap": 0.2,
        }
        #: `pressure` 必须等于五信号加权和，否则判读器先撞“记录件不自洽”那条守卫，
        #: 我要测的就不是装配那一支了。
        pressure = sum(PRESSURE_WEIGHTS[f] * v for f, v in signals.items())
        lines.append(
            json.dumps(
                {
                    "kind": "pressure",
                    "tick": index + 1,
                    **signals,
                    "resource_state": 1.0,
                    "pressure": pressure,
                    "decision_should_propose": False,
                }
            )
        )
    lines.append(
        json.dumps(
            {
                "kind": "tail",
                "ticks_at_close": rows,
                "records_written": rows,
                "learning_mode_at_close": dev.get("_mode_at_close", "fast_slow"),
                "bridge_gate_at_close": reported.get("bridge_gate_actual"),
            }
        )
    )
    return lines


def _judge(tmp_path: Path, lines: list[str], prereg: str = "n3-04") -> tuple[int, dict[str, Any]]:
    """判读器**两条分支都把 stdout 当唯一取法**（拒绝支不写出件），避免同一测里两种读数口径。"""

    reader = _reader()
    face = tmp_path / "face.jsonl"
    face.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    captured = io.StringIO()
    with redirect_stdout(captured):
        rc = reader.main(["--pressure", str(face), "--prereg", prereg])
    return int(rc), json.loads(captured.getvalue())


def test_reader_refuses_missing_developmental_self_report(tmp_path: Path) -> None:
    rc, payload = _judge(tmp_path, _synthetic_face(mode_readings=DROP))
    assert rc == 2
    assert payload["status"] == "refused"
    assert "developmental" in payload["error"]


def test_reader_refuses_read_only_mode_reading(tmp_path: Path) -> None:
    rc, payload = _judge(
        tmp_path,
        _synthetic_face(
            mode_readings=[{"after": "mount", "tick": 0, "learning_mode": "read_only"}]
        ),
    )
    assert rc == 2
    assert "read_only" in payload["error"]


def test_reader_refuses_arm_that_did_not_open_the_bridge(tmp_path: Path) -> None:
    rc, payload = _judge(tmp_path, _synthetic_face(bridge_gate_actual=0.0))
    assert rc == 2
    assert "bridge_gate_actual" in payload["error"]


def test_reader_refuses_missing_close_mode_reading(tmp_path: Path) -> None:
    lines = _synthetic_face()
    tail = json.loads(lines[-1])
    tail.pop("learning_mode_at_close")
    lines[-1] = json.dumps(tail)
    rc, payload = _judge(tmp_path, lines)
    assert rc == 2
    assert "learning_mode_at_close" in payload["error"]


def test_reader_accepts_present_dims_and_reports_dynamic_range(tmp_path: Path) -> None:
    rc, payload = _judge(tmp_path, _synthetic_face())
    assert rc == 0, payload
    assert payload["format"] == "taiji-n3-pressure-thresholds-v2"
    assembly = payload["assembly"]
    dims = assembly["dims_present"]
    assert dims["fast_slow_conflict"]["zero_share"] < 0.05
    assert dims["activity_saturation"]["nonzero_count"] == dims["activity_saturation"]["records"]
    assert assembly["j_n3b_present"] == {"fast_slow_conflict": True, "activity_saturation": True}
    assert assembly["dynamic_range"]["proven"] is False
    assert assembly["assembly_verdict"] == "present_without_dynamic_range"
    assert assembly["assembly_self_report"]["developmental_bundle_mounted"] is True


def test_reader_reports_not_present_when_the_opened_dim_stays_zero(
    tmp_path: Path,
) -> None:
    """判据为假不等于仪器出错：那一维仍然 1/7 的零占比过线 ⇒ rc=0 且 verdict=not_present。"""

    rc, payload = _judge(tmp_path, _synthetic_face(zero_every=7))
    assert rc == 0, payload
    assembly = payload["assembly"]
    assert assembly["dims_present"]["fast_slow_conflict"]["zero_share"] > 0.05
    assert assembly["j_n3b_present"]["fast_slow_conflict"] is False
    assert assembly["assembly_verdict"] == "not_present"


def test_reader_default_prereg_keeps_the_step_one_shape(tmp_path: Path) -> None:
    rc, payload = _judge(tmp_path, _synthetic_face(), prereg="n3-01")
    assert rc == 0, payload
    assert payload["format"] == "taiji-n3-pressure-thresholds-v1"
    assert "assembly" not in payload
    assert payload["threshold_verdict"] == "not_frozen_caliber_unchecked"
