"""DEBT-G84 修法（#23）的契约测：影子旗标的"请求值"必须无条件自述，且**在场 ≠ 被走到**。

来历：㊵-654 那次双臂塌成一枚权重，事后看是这条区分不开——`shadow_gate_requested` 原先只写在
`if adaptive_shadow is not None:` 分支里 ⇒ "请求了但没物化"与"根本没请求"在读侧同形。
owner 弹窗 #23② 批的形状＝块外新键 `n5_shadow_request`，块内五枚 J-N5b-1 必需键一字不动。

五态各自能为假：`not_reported`（旧档没这个键）专用于挡住"键缺席＝没请求"这族假缺席；
`walked` 那一支保证 `j_n5b_1=present` 不再被当成"影子走过"。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "adjudicate_taiji_n5_shadow_gate.py"
TRAINER = REPO / "scripts" / "training" / "train_seed_corpus.py"
CORPUS = REPO / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"
BASE = REPO / "checkpoints" / "seed_beta.pt"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_gate_request", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = _load()

#: 现读自 `adjudicate_taiji_n5_shadow_gate.py`（我第一版按 `n5_shadow` 块里的键"背"了一遍，背错了：
#: `candidate_id`／`bridge_id`／`unit_count` 不在必需五枚之内）——这正是本测要钉的东西，不能凭记忆写。
FROZEN_KEYS = (
    "gate",
    "shadow_gate_requested",
    "candidate_gate",
    "candidate_utility",
    "candidate_counterfactual_utility",
)


def _face(tmp_path: Path, name: str, envelope_block: dict[str, Any] | None, request: dict[str, Any] | None = None) -> Path:
    path = tmp_path / name
    envelope: dict[str, Any] = {"format": "seed-native-v1"}
    if envelope_block is not None:
        envelope["n5_shadow"] = envelope_block
    if request is not None:
        envelope["n5_shadow_request"] = request
    torch.save({"envelope": envelope}, str(path))
    return path


def _block(**over: Any) -> dict[str, Any]:
    block = {
        "candidate_id": "r4-candidate:abc",
        "bridge_id": "predictive_residual.bridge",
        "gate": 1.0,
        "unit_count": 97,
        "shadow_gate_requested": 1.0,
        "candidate_gate": 0.5,
        "candidate_utility": 0.31,
        "candidate_counterfactual_utility": 0.11,
        "shadow_forward_hits": 5,
        "shadow_learn_hits": 5,
        "shadow_branch_hits": 10,
    }
    block.update(over)
    return block


def _judge(path: Path) -> dict[str, Any]:
    return JUDGE.judge_arm("treated", path)


def test_the_five_frozen_required_keys_are_unchanged() -> None:
    #: 批准的前提就是"不动那五枚"——扩必需键会把已入库的 G/H 读数追认成 `ran_not_measured`。
    assert tuple(JUDGE.REQUIRED_KEYS) == FROZEN_KEYS


def test_present_block_with_counters_reports_walked(tmp_path: Path) -> None:
    arm = _judge(_face(tmp_path, "a.pt", _block()))
    assert arm["j_n5b_1"] == "present"
    assert arm["walked"] == {"state": "walked", "forward_hits": 5, "walked": True}
    #: 块在场但没写请求键（旧档形状）⇒ 只能是 `not_reported`：**不许**拿块内那枚
    #: `shadow_gate_requested` 顶替块外新键，否则 #23 要修的"同形"问题会被测洗掉。
    assert arm["shadow_request"]["state"] == "not_reported"


def test_present_block_without_counters_is_not_read_as_walked(tmp_path: Path) -> None:
    #: 「在场 ≠ 被走到」那一支的反例：在场性是键的事，走过与否必须有计数器才算。
    block = _block()
    for key in ("shadow_forward_hits", "shadow_learn_hits", "shadow_branch_hits"):
        del block[key]
    arm = _judge(_face(tmp_path, "a.pt", block))
    assert arm["j_n5b_1"] == "present"
    assert arm["presence_counters"]["status"] == "absent_from_block"
    assert arm["walked"]["state"] == "counters_absent"
    assert arm["walked"]["walked"] is None


def test_zero_counters_say_not_walked_rather_than_absent(tmp_path: Path) -> None:
    arm = _judge(_face(tmp_path, "a.pt", _block(shadow_forward_hits=0, shadow_learn_hits=0, shadow_branch_hits=0)))
    assert arm["walked"] == {"state": "not_walked", "forward_hits": 0, "walked": False}


def test_requested_but_not_materialized_is_distinguishable_from_not_requested(tmp_path: Path) -> None:
    #: DEBT-G84 的本体：块缺席时两条状态从前同形，现在由块外新键分开。
    requested = _face(tmp_path, "r.pt", None, {"flag": True, "gate_requested": 1.0})
    untouched = _face(tmp_path, "u.pt", None, {"flag": False, "gate_requested": None})
    legacy = _face(tmp_path, "l.pt", None)
    states = {
        "requested": _judge(requested)["shadow_request"]["state"],
        "not_requested": _judge(untouched)["shadow_request"]["state"],
        "legacy": _judge(legacy)["shadow_request"]["state"],
    }
    assert states == {
        "requested": "requested_with_gate",
        "not_requested": "not_requested",
        "legacy": "not_reported",
    }
    #: 三者都仍是 `ran_not_measured`——新键**不许**把"没物化"洗成"已测"。
    assert _judge(requested)["j_n5b_1"] == "ran_not_measured"
    assert _judge(legacy)["shadow_request"]["gate_requested"] is None


def test_a_real_short_training_run_puts_the_request_key_on_disk(tmp_path: Path) -> None:
    """真跑证明：键必须进**磁盘**，不是只改内存对象（㊵-593 那次就是栽在填充在 `atomic_save` 之后）。

    形状按训练器自己的响亮拒绝规则给全（`--n5-shadow` 需 `--developmental-bridge-gate >0` 与
    `--pressure-record`，见 `train_seed_corpus.py` 的入参校验段）。200 符号不足以满足六道 EMA 合取
    ⇒ 影子**不会物化**，于是这一支同时验证"请求了但没物化"在真实产物里可读。
    """
    if not (CORPUS.is_file() and BASE.is_file()):
        #: 现场缺失按 skip 处理，但**本机现读两者都在**⇒ 这支是真跑不是空跳（跳了要在汇总里看得见）。
        pytest.skip(f"缺现场件：corpus={CORPUS.is_file()} base={BASE.is_file()}")
    ckpt = tmp_path / "req.pt"
    argv = [
        sys.executable,
        str(TRAINER),
        "--corpus",
        str(CORPUS),
        "--resume",
        str(BASE),
        "--checkpoint",
        str(ckpt),
        "--progress",
        str(tmp_path / "progress.json"),
        "--pressure-record",
        str(tmp_path / "pressure.jsonl"),
        "--readout",
        "predictive",
        "--n5-shadow",
        "--developmental-fast-slow",
        "--developmental-bridge-gate",
        "1",
        "--growth-min-pressure",
        "0.64",
        "--max-symbols",
        "200",
        "--keep-checkpoints",
        "off",
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, cwd=str(REPO))
    assert proc.returncode == 0, (proc.stdout or "")[-400:] + (proc.stderr or "")[-400:]
    loaded = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    #: 训练器存出的顶层**就是**信封（`_shadow_block` 也是按 `get("envelope", 自身)` 读的），别猜包装层。
    envelope = loaded.get("envelope", loaded)
    assert envelope["n5_shadow_request"] == {"flag": True, "gate_requested": None}
    assert "n5_shadow" not in envelope or envelope["n5_shadow"] is None
    arm = _judge(ckpt)
    assert arm["shadow_request"]["state"] == "requested_without_gate"
    assert arm["j_n5b_1"] == "ran_not_measured"
    payload = json.dumps({k: arm[k] for k in ("shadow_request", "walked")}, ensure_ascii=False)
    assert "counters_absent" in payload
