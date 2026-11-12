"""Select a frozen engineering run through the normal command entry point."""

import json
from pathlib import Path

import pytest

from scripts.training import run_taiji_u2_22_4_1 as runner


def contract_file(directory: Path, identity: str) -> Path:
    path = directory / "contract.json"
    path.write_text(json.dumps({"run_id": identity}), encoding="utf-8")
    return path


def test_selected_contract_binds_new_run(tmp_path: Path) -> None:
    contract = contract_file(tmp_path, "selected-u2")
    run_path = tmp_path / "run.json"
    bound = runner.bind_run(run_path, contract_path=contract)
    assert bound == {"run_id": "selected-u2"}
    assert json.loads(run_path.read_text(encoding="utf-8")) == bound
    assert runner.bind_run(run_path, contract_path=contract) == bound


def test_selected_contract_rejects_reused_root(tmp_path: Path) -> None:
    contract = contract_file(tmp_path, "selected-u2")
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps({"run_id": "older-run"}), encoding="utf-8")
    with pytest.raises(runner.BudgetError, match="older-run"):
        runner.bind_run(run_path, contract_path=contract)
    assert json.loads(run_path.read_text(encoding="utf-8"))["run_id"] == "older-run"


def test_default_binding_preserves_u1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = contract_file(tmp_path, "legacy-u1")
    monkeypatch.setattr(runner, "DRAFT", contract)
    assert runner.bind_run(tmp_path / "run.json") == {"run_id": "legacy-u1"}


def test_cli_passes_selected_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = contract_file(tmp_path, "selected-u2")
    seen = []

    def execute(root: Path, *, contract_path: Path | None = None) -> dict:
        seen.append((root, contract_path))
        return {"status": "checked_without_step"}

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.main(["--run", "--run-root", str(tmp_path), "--run-contract", str(contract)]) == 0
    assert seen == [(tmp_path, contract)]
