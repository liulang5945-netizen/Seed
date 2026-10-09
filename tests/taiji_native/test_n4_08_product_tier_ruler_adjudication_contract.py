"""产品档寻址尺判读器的契约测：PLAN-N4-06 的三支分支都必须能为假。

夹具全是**合成报告**（写在 `tmp_path`，不碰 `reports/`）：数值用手推的小整数，
不抄真跑输出。五支分别覆盖在场性、两档差与噪声带的比较、方向性禁令、溯源档位。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "adjudicate_taiji_n4_product_tier_ruler.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n4_ruler_judge", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JUDGE = _load()


def _report(
    tmp_path: Path,
    name: str,
    wrong: float,
    band: float,
    *,
    layer: str = "product",
    writes: int = 120,
    drop: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "wrong_top1_rate": wrong,
        "noise_band_adjacent_block_max": band,
        "block_means": [wrong, wrong, wrong, wrong, wrong],
        "faces_provenance": {"mount_layer": layer, "episodic_writes": writes},
    }
    if drop is not None:
        payload.pop(drop)
    path = tmp_path / name
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    return path


def _run(tmp_path: Path, small: Path, large: Path) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "verdict.json"
    rc = JUDGE.main(["--report-small", str(small), "--report-large", str(large), "--out", str(out)])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_usable_and_direction_correct_yields_angle_sensitive(tmp_path: Path) -> None:
    #: 手推：0.30-0.10=0.20 ≥ max(带 0.12,0.15)=0.15 ⇒ 尺可用；大角度更差 ⇒ angle_sensitive。
    small = _report(tmp_path, "s.json", 0.10, 0.12)
    large = _report(tmp_path, "l.json", 0.30, 0.15)
    rc, payload = _run(tmp_path, small, large)
    assert payload["verdict"] == "angle_sensitive"
    assert payload["ruler_usable"] is True
    assert rc == 0


def test_delta_smaller_than_band_is_ruler_unusable(tmp_path: Path) -> None:
    #: 反例：0.20-0.15=0.05 < 带 0.12 ⇒ 不许比优劣，只许说"这把尺答不了"。
    small = _report(tmp_path, "s.json", 0.15, 0.12)
    large = _report(tmp_path, "l.json", 0.20, 0.12)
    rc, payload = _run(tmp_path, small, large)
    assert payload["verdict"] == "ruler_unusable"
    assert payload["j_n4f_2"] == "ruler_unusable"
    assert rc == 1


def test_reverse_direction_is_not_angle_sensitive_even_if_numbers_look_good(tmp_path: Path) -> None:
    #: 禁令落地：小角度反而更差 ⇒ 即使差值巨大也不给"寻址有效"，并且非零退出。
    small = _report(tmp_path, "s.json", 0.60, 0.10)
    large = _report(tmp_path, "l.json", 0.10, 0.10)
    rc, payload = _run(tmp_path, small, large)
    assert payload["verdict"] == "not_angle_sensitive"
    assert payload["ruler_usable"] is True
    assert rc == 1


def test_missing_self_report_field_fails_closed(tmp_path: Path) -> None:
    small = _report(tmp_path, "s.json", 0.10, 0.12, drop="block_means")
    large = _report(tmp_path, "l.json", 0.30, 0.15)
    rc, payload = _run(tmp_path, small, large)
    assert payload["status"] == "ran_not_measured"
    assert "block_means" in payload["missing"]["small"]
    assert rc == 2


def test_harness_tier_materials_cannot_be_called_product_tier(tmp_path: Path) -> None:
    small = _report(tmp_path, "s.json", 0.10, 0.12, layer="harness")
    large = _report(tmp_path, "l.json", 0.30, 0.15, layer="harness")
    rc, payload = _run(tmp_path, small, large)
    assert payload["j_n4f_5"] == "provenance_not_product_tier"
    assert payload["verdict"] == "not_adjudicable_provenance"
    assert rc == 2
