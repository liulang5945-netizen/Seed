"""P2-11 Taiji-owned IDE language chain Gate."""

from scripts.training.eval_taiji_ide_language_chain import evaluate


def test_ide_language_chain_gate_passes() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p2-11-ide-language-chain-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
