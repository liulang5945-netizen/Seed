"""N5 通电判读器（`adjudicate_taiji_n5_shadow_gate.py`）的契约测。

PLAN-N5-02 J-N5b-1／J-N5b-2 的每一支都必须能为假，且 fail-closed 那一支要能在真件上走出来：
本仓已被「缺列被读成满足」绊过（㊵-590 的 `get()` 假 null），所以这里的正负例都直接构造检查点。

夹具全部落 `tmp_path`（`torch.save` 一个含 `envelope` 的 dict），不碰 `output/` 与 `reports/`。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "adjudicate_taiji_n5_shadow_gate.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_gate_judge", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = _load()


def _face(tmp_path: Path, name: str, block: dict[str, Any] | None) -> Path:
    path = tmp_path / name
    envelope: dict[str, Any] = {"format": "seed-native-v1"}
    if block is not None:
        envelope["n5_shadow"] = block
    torch.save({"envelope": envelope}, str(path))
    return path


def _full(**over: Any) -> dict[str, Any]:
    block = {
        "candidate_id": "r4-candidate:abc",
        "bridge_id": "predictive_residual.bridge",
        "gate": 1.0,
        "unit_count": 97,
        "shadow_gate_requested": 1.0,
        "candidate_gate": 0.5,
        "candidate_utility": 0.31,
        "candidate_counterfactual_utility": 0.11,
    }
    block.update(over)
    return block


def _run(tmp_path: Path, treated: Path, control: Path | None = None) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "verdict.json"
    argv = ["--arm-treated", str(treated), "--out", str(out)]
    if control is not None:
        argv += ["--arm-control", str(control)]
    rc = JUDGE.main(argv)
    import json

    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_learning_branch_is_learned(tmp_path: Path) -> None:
    rc, payload = _run(tmp_path, _face(tmp_path, "t.pt", _full()))
    assert rc == 0
    arm = payload["arms"]["treated"]
    assert arm["j_n5b_1"] == "present"
    assert arm["j_n5b_2"] == "shadow_learned"


def test_powered_but_zero_utilities_is_inert(tmp_path: Path) -> None:
    face = _face(
        tmp_path,
        "t.pt",
        _full(candidate_utility=0.0, candidate_counterfactual_utility=0.0),
    )
    rc, payload = _run(tmp_path, face)
    assert rc == 1
    assert payload["arms"]["treated"]["j_n5b_2"] == "shadow_inert"


def test_unpowered_is_not_powered_not_inert(tmp_path: Path) -> None:
    #: F 跑的形状：物化并挂上，但 `_gate` 仍 0 ⇒ 归 `not_powered`，不混进 `shadow_inert`。
    rc, payload = _run(tmp_path, _face(tmp_path, "t.pt", _full(gate=0.0)))
    assert rc == 1
    assert payload["arms"]["treated"]["j_n5b_2"] == "not_powered"


def test_missing_self_report_key_is_fail_closed(tmp_path: Path) -> None:
    block = _full()
    del block["candidate_counterfactual_utility"]
    rc, payload = _run(tmp_path, _face(tmp_path, "t.pt", block))
    arm = payload["arms"]["treated"]
    assert rc == 2
    assert arm["j_n5b_1"] == "ran_not_measured"
    assert arm["missing_self_report_keys"] == ["candidate_counterfactual_utility"]
    assert arm["j_n5b_2"] == "unverified_missing_face"


def test_block_absent_entirely_reports_all_five_missing(tmp_path: Path) -> None:
    rc, payload = _run(tmp_path, _face(tmp_path, "t.pt", None))
    arm = payload["arms"]["treated"]
    assert rc == 2
    assert len(arm["missing_self_report_keys"]) == 5


def test_two_arms_with_identical_requested_gate_are_unpairable(tmp_path: Path) -> None:
    treated = _face(tmp_path, "t.pt", _full())
    control = _face(tmp_path, "c.pt", _full())
    rc, payload = _run(tmp_path, treated, control)
    assert rc == 2
    assert payload["pairing"]["status"] == "pairing_invalid"


def test_missing_face_file_is_rejected(tmp_path: Path) -> None:
    #: `main()` 返回码就是判据（`__main__` 才包 `raise SystemExit`），所以这里读 rc 不读异常。
    rc = JUDGE.main(["--arm-treated", str(tmp_path / "nope.pt"), "--out", str(tmp_path / "v.json")])
    assert rc == 2
