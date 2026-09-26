from __future__ import annotations

from pathlib import Path

import pytest

from scripts.training.eval_taiji_runtime_retention_store_separation import evaluate


def test_runtime_retention_does_not_delete_or_resurrect_external_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 这件评测默认把 43 MB 检查点写进产品工件目录；测试必须改道临时目录，
    # 否则同一次套件里"产品目录只放版本化工件"那道守卫会变成顺序相关的红。
    monkeypatch.setenv("R2_RETENTION_SCRATCH_ROOT", str(tmp_path))
    report = evaluate()
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
