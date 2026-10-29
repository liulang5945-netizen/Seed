"""E0 回执仪器的判据册：指纹与"评分只读"检查都必须**能为假**。

两支负对照：
①同一个张量改 1 ulp ⇒ `state_digest` 必须变（否则"参数字节未变"是恒真式）；
②真的走一步优化器 ⇒ `isolation_snapshot` 前后的 `unchanged` 必须为 False；
   只做 `no_grad` 前向 ⇒ 必须为 True（否则它根本分不开"读"与"写"）。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = (
    PROJECT_ROOT / "scripts" / "training" / "verify_taiji_n1_plan177_6_t1_endpoint_receipts.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("n177_6_endpoint_receipts", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instrument():
    return _module()


class _Tower(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
        self.parameter_revision = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def _stub_trainer(width: int = 4) -> SimpleNamespace:
    model = _Tower(width)
    producer = _Tower(width)
    return SimpleNamespace(
        model=model,
        producer=producer,
        update_cursor=2560,
        producer_optimizer=torch.optim.AdamW(producer.parameters(), lr=0.1),
        language_trainer=SimpleNamespace(
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        ),
    )


def test_state_digest_separates_one_ulp(instrument) -> None:
    base = torch.tensor([1.0, 2.0], dtype=torch.float32)
    bumped = base.clone()
    bumped[1] = torch.nextafter(base[1], torch.tensor(3.0))
    assert bumped[1].item() != base[1].item()
    assert instrument.state_digest({"w": base}) != instrument.state_digest({"w": bumped})


def test_state_digest_is_stable_across_rebuilds(instrument) -> None:
    left = {"a": torch.zeros(3), "b": {"c": torch.ones(2)}}
    right = {"b": {"c": torch.ones(2)}, "a": torch.zeros(3)}
    assert instrument.state_digest(left) == instrument.state_digest(right)


def test_a_forward_pass_leaves_the_snapshot_unchanged(instrument) -> None:
    trainer = _stub_trainer()
    before = instrument.isolation_snapshot(trainer)
    with torch.no_grad():
        trainer.model(torch.ones(1, 4))
    after = instrument.isolation_snapshot(trainer)
    assert before == after
    assert before["update_cursor"] == 2560


def test_a_real_optimizer_step_is_caught(instrument) -> None:
    trainer = _stub_trainer()
    before = instrument.isolation_snapshot(trainer)
    loss = trainer.model(torch.ones(1, 4)).pow(2).sum()
    loss.backward()
    trainer.language_trainer.optimizer.step()
    after = instrument.isolation_snapshot(trainer)
    assert before != after
    assert before["model_state"] != after["model_state"]
    assert before["optimizer_state"] != after["optimizer_state"]


def test_a_cursor_write_alone_is_caught(instrument) -> None:
    trainer = _stub_trainer()
    before = instrument.isolation_snapshot(trainer)
    trainer.update_cursor += 1
    after = instrument.isolation_snapshot(trainer)
    assert before != after


def test_the_help_face_lists_every_flag() -> None:
    proc = subprocess.run(
        [sys.executable, str(INSTRUMENT), "--help"],
        capture_output=True,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    for flag in ("--run-root", "--arm", "--all-fingerprints", "--train-cache", "--out-report"):
        assert flag in proc.stdout, flag
