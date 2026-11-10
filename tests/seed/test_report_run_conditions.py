"""运行条件门的契约测试（能力卡 §13 / 台账 D-20261110-05）。"""

from __future__ import annotations

import pytest

from scripts.training import check_report_run_conditions as gate


def test_digest_without_conditions_is_rejected(tmp_path):
    payload = {"checks": {"content_digest": "abc", "max_abs_delta": 0.0}}
    assert gate.missing_conditions(payload) == ("max_abs_delta", "content_digest")
    with pytest.raises(gate.MissingRunConditions) as excinfo:
        gate.write_report(tmp_path / "report.json", payload)
    assert "content_digest" in str(excinfo.value)
    assert not (tmp_path / "report.json").exists()


def test_recorded_threads_satisfy_the_gate(tmp_path):
    payload = {"threads": 1, "checks": {"content_digest": "abc", "max_abs_delta": 0.0}}
    assert gate.missing_conditions(payload) == ()
    path = gate.write_report(tmp_path / "report.json", payload)
    assert b"\r\n" not in path.read_bytes()


def test_env_thread_tokens_also_count():
    payload = {"logits_digest": "abc", "env": {"OMP_NUM_THREADS": "1"}}
    assert gate.missing_conditions(payload) == ()


def test_file_identity_hashes_are_not_claims():
    # frozen_files／fixture_sha256 是字节哈希，与浮点归约顺序无关，不该要求线程记录。
    payload = {"frozen_files": {"a.py": "sha256"}, "fixture_sha256": "deadbeef"}
    assert gate.missing_conditions(payload) == ()


def test_repository_has_no_violation_outside_the_baseline():
    baseline = gate.load_baseline()
    new = {
        name: fields for name, fields in gate.scan(gate.REPO_ROOT).items() if name not in baseline
    }
    assert new == {}
