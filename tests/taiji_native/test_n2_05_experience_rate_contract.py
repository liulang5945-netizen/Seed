"""每单位经验速率摘要器（`summarize_taiji_n2_experience_rate.py`，PLAN-N2-05）的契约测。

这台的判据只有一条是**两侧都能为假**的（`J-N2e-3` 的符号限定），加上两条拒绝分支，所以本册就是
"正常档三枚并列＋两枚拒绝＋一枚反向"。恒真键（`J-N2e-1`／`J-N2e-2`／`G-N2e_1`）在第一版被我
写进件里，现已删——不可算与缺自述都由 `status` 与 rc=2 承担（同 ㊵-618 撤 `shadow_materialized` 一课）。

* 真件那一支钉的是**三枚速率并列**与 `cycles == texts` 的退化点名（漏任一枚就是把分母挑掉了）；
* 复算不符那一支：件**照样落**且 rc=2（"复算不符"本身是读数，不是"没跑"）；
* 分母非正那一支：不落速率值，只出版 `not_computable_nonpositive_dose`；
* 分子为正那一支：符号限定必须翻成 `gain_direction_unqualified`，否则这条判定就是一侧的装饰。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "summarize_taiji_n2_experience_rate.py"
REAL_POWERUP = REPO / "reports" / "taiji_n2b_powerup_20261008.json"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n2_05_experience_rate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _powerup_payload(treated: float, control: float, numerator: float, cycles: float) -> str:
    return json.dumps(
        {
            "j_n2b_prime_delta_quality": numerator,
            "material_treated_window": {"quality_mean": treated},
            "material_control_window": {"quality_mean": control},
            "organs": {
                "stats": {"cycles": cycles},
                "max_symbols": 64,
                "texts": 8,
                "cycles_per_text": 1,
            },
        }
    )


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "powerup.json"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_the_sealed_powerup_yields_three_side_by_side_rates() -> None:
    payload = MODULE.summarize(REAL_POWERUP)
    assert payload["status"] == "ok", payload
    assert set(payload["rates"]) == {"cycles", "max_symbols", "texts"}, payload["rates"]
    assert payload["guards"]["G_N2e_2_numerator_recomputable"] is True, payload
    assert payload["j_n2e_3"] == "cost_not_gain", payload
    #: 本档 cycles == texts（批文里 cycles_per_text=1），必须被点名，否则会被读成两条独立分母。
    assert payload["degenerate_denominators"] == ["cycles==texts"], payload
    #: 分母值也钉住：它们是**剂量自述**，不是可以从命令行反推的东西（DEBT-G70 那一族）。
    assert payload["denominators"] == {"cycles": 8.0, "max_symbols": 64.0, "texts": 8.0}, payload
    #: 手算一次对照（不是替换读数，是证明这台仪器没把除法做错）：-1.1086209986387523 / 8。
    assert payload["rates"]["cycles"] == round(
        payload["numerator_delta_quality"] / 8.0, 9
    ), payload["rates"]


def test_a_non_recomputable_numerator_is_published_and_refused(tmp_path: Path) -> None:
    """复算不符 ⇒ rc=2，但**件照样落**：这一支的产物就是"两臂均值与分子不同源"这条读数本身。"""

    src = _write(tmp_path, _powerup_payload(-14.0, -13.0, -9.99, 8.0))
    out = tmp_path / "rate.json"
    rc = MODULE.main(["--powerup-report", str(src), "--out-report", str(out)])
    assert rc == 2, rc
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["guards"]["G_N2e_2_numerator_recomputable"] is False, payload
    assert payload["status"] == "ok", payload


def test_a_zero_dose_refuses_instead_of_dividing(tmp_path: Path) -> None:
    src = _write(tmp_path, _powerup_payload(-14.0, -13.0, -1.0, 0.0))
    out = tmp_path / "rate.json"
    rc = MODULE.main(["--powerup-report", str(src), "--out-report", str(out)])
    assert rc == 2, rc
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "not_computable_nonpositive_dose", payload
    assert payload["nonpositive_denominators"] == ["cycles"], payload
    assert "rates" not in payload, "分母非正时不许出版任何速率值"


def test_a_missing_dose_self_report_is_named_not_guessed(tmp_path: Path) -> None:
    src = tmp_path / "sparse.json"
    src.write_text(
        json.dumps(
            {
                "j_n2b_prime_delta_quality": -1.0,
                "material_treated_window": {"quality_mean": -2.0},
                "material_control_window": {"quality_mean": -1.0},
                "organs": {"max_symbols": 64},
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    out = tmp_path / "rate.json"
    rc = MODULE.main(["--powerup-report", str(src), "--out-report", str(out)])
    assert rc == 2, rc
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "dose_self_report_absent", payload
    assert payload["missing_denominators"] == ["cycles", "texts"], payload


def test_a_positive_numerator_flips_the_sign_qualification(tmp_path: Path) -> None:
    """反向那一支：分子为正 ⇒ 限定语必须换掉，否则 `J-N2e-3` 就是一侧的装饰。"""

    src = _write(tmp_path, _powerup_payload(-12.0, -13.0, 1.0, 8.0))
    payload = MODULE.summarize(src)
    assert payload["status"] == "ok", payload
    assert payload["j_n2e_3"] == "gain_direction_unqualified", payload
    assert payload["rates"]["cycles"] == 0.125, payload
