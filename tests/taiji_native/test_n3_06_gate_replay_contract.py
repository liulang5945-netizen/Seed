"""PLAN-N3-06 重放仪（`replay_taiji_n3_06_gate_attribution.py`）的契约测。

这台仪器的全部风险是"我把六道闸自己算了一遍，然后当成产品算的"。所以它的**每一条锚点都必须被实走一次为假**：

* **正支**＝真实在库面读出 `single_gate_blocking`，且 A-1/A-2 两条逐行锚点的失配计数为 **0**（这条为真的证据只能是读数）；
* **A-1 支**＝只翻掉面里一行的 `decision_should_propose` ⇒ rc=2，并点名首个失配行；
* **A-2 支**＝只扰动一行原始信号（摘要会变、`should_propose` 未必变）⇒ rc=2 指认摘要不同源；
* **A-3 支**＝把面截到样本下限以下／抽掉一个必需字段 ⇒ 各自 rc=2；
* **A-5 支与表外原因支**＝拿一支假 trigger 替掉产品那支，让 `reasons` 与六条 EMA 不自洽 ⇒ 仪器必须拒判
  （这两支证明的不是产品，是"仪器不会把不自洽的读数放出去"）。

真实面件只读；被改的副本一律落 pytest 的 `tmp_path`。
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/replay_taiji_n3_06_gate_attribution.py"
FACE = REPO / "output/n3_04/beta_gate025/pressure.jsonl"

#: 12 支里 **10 支**要读这枚在盘面（CI run 37764787792 实测：本册 10 枚红，全是 `FileNotFoundError`
#: 形状），而 `output/n3_04/` 按 `.gitignore` 不入库（09 §3.2 第 4 项把它列为"owner 排期"的挂账资产）。
#: ⇒ 按 05 §9（H19j／H19n）既定策略在模块级声明一次，不需要的两支用 `no_local_artifacts` 豁免；
#: 这样"缺产物"在 CI 上是**点名的 skip**，不是 23 枚红里的 10 枚噪声。
LOCAL_ONLY_ARTIFACTS = ("output/n3_04/beta_gate025/pressure.jsonl",)


def _load_module():
    spec = importlib.util.spec_from_file_location("n3_06_gate_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


READER = _load_module()


def _rows(path: Path = FACE) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write(tmp: Path, records: list[dict[str, Any]]) -> str:
    out = tmp / "pressure.jsonl"
    out.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return str(out)


def _run(face_path: str, tmp: Path) -> tuple[int, dict[str, Any]]:
    out = tmp / "verdict.json"
    rc = READER.main(["--face", face_path, "--out-report", str(out)])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_real_face_is_faithful_and_indicts_the_composite_gate(tmp_path: Path) -> None:
    assert FACE.is_file(), "在库面缺失——本测读的是真件，不是夹具"
    rc, payload = _run(str(FACE), tmp_path)
    assert rc == 0, payload
    face = payload["faces"][0]
    assert face["g_n3c_a1_rows_compared"] == 3999
    assert face["g_n3c_a1_should_propose_mismatches"] == 0
    assert face["g_n3c_a2_digests_compared"] == 3999
    assert face["g_n3c_a2_digest_mismatches"] == 0
    assert face["verdict"] == "single_gate_blocking"
    assert face["dead_gates"] == ["pressure_ema"]
    assert face["g_n3c_meets_pressure_crosscheck_equal"] is True
    #: 六道阈里只有合成量那道被面头自述过——这条把 DEBT-G53 的"缺自述"钉成可机检事实。
    self_reported = {
        label: gate["threshold_self_reported_in_face"] for label, gate in face["per_gate"].items()
    }
    assert self_reported["pressure_ema"] is True
    assert sum(1 for flag in self_reported.values() if not flag) == 5
    assert face["assumed_from_product_defaults"] == [
        "minimum_residual_error",
        "minimum_fast_slow_conflict",
        "minimum_activity_saturation",
        "minimum_utility_gap",
        "minimum_resource_state",
        "ema_rate",
    ]


def test_a1_flipped_decision_refuses(tmp_path: Path) -> None:
    records = _rows()
    pressures = [record for record in records if record.get("kind") == "pressure"]
    assert pressures, "面上一个观测行都没有——夹具坏了"
    pressures[0]["decision_should_propose"] = not pressures[0]["decision_should_propose"]
    rc, payload = _run(_write(tmp_path, records), tmp_path)
    assert rc == 2
    assert payload["faces_judged"] == 0
    error = payload["refused"][0]["error"]
    assert "A-1" in error
    #: 失配行号必须是 0——仪器只点名**首个**失配，行号算错就等于把证据指到别处。
    assert "第 0 行" in error


def test_a2_perturbed_signal_breaks_the_digest_anchor(tmp_path: Path) -> None:
    records = _rows()
    for record in records:
        if record.get("kind") == "pressure":
            #: 只动一个原始信号：`should_propose` 仍是假（远够不着阈），但产品自己算的 `pressure_digest` 会变
            record["utility_gap"] = float(record["utility_gap"]) + 1e-6
            break
    rc, payload = _run(_write(tmp_path, records), tmp_path)
    assert rc == 2
    error = payload["refused"][0]["error"]
    assert "A-2" in error or "摘要" in error


def test_a3_short_face_and_missing_field_each_refuse(tmp_path: Path) -> None:
    records = _rows()
    header = [record for record in records if record.get("kind") == "face"]
    pressures = [record for record in records if record.get("kind") == "pressure"]
    rc, payload = _run(_write(tmp_path, header + pressures[:200]), tmp_path)
    assert rc == 2
    assert "样本下限" in payload["refused"][0]["error"]

    stripped = json.loads(json.dumps(pressures[0]))
    del stripped["evidence_id"]
    rc, payload = _run(_write(tmp_path, header + [stripped] + pressures[1:]), tmp_path)
    assert rc == 2
    assert "evidence_id" in payload["refused"][0]["error"]


def _fake_trigger(reasons: list[str], meets_all: bool) -> type:
    """造一支"看起来像产品"的假 trigger：六道 EMA 全部拉到 0.9（推导为真），但 `reasons` 说了算。"""

    class FakeTrigger:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def observe(self, observation: Any, *, structural_budget: int) -> Any:
            ema = 0.9 if meets_all else 0.1
            return types.SimpleNamespace(
                should_propose=False,
                reasons=list(reasons),
                consecutive_pressure_steps=0,
                pressure=ema,
                residual_error_ema=ema,
                fast_slow_conflict_ema=ema,
                activity_saturation_ema=ema,
                utility_gap_ema=ema,
                resource_state_ema=ema,
            )

    return FakeTrigger


def _patched(monkeypatch, trigger_type: type) -> None:
    monkeypatch.setattr(READER, "AdaptiveResidualGrowthTrigger", trigger_type)


def test_a5_inconsistent_reasons_refuse(tmp_path: Path, monkeypatch) -> None:
    #: 假 trigger 把六道 EMA 都判为过阈（推导 3,999 步真），却仍在 `reasons` 里发 `pressure_below_threshold`
    #: ⇒ 我的推导与产品的自述不同源，仪器必须闭嘴。
    _patched(monkeypatch, _fake_trigger(["pressure_below_threshold"], meets_all=True))
    rc, payload = _run(str(FACE), tmp_path)
    assert rc == 2
    assert "六道闸合取与产品自述不一致" in payload["refused"][0]["error"]


def test_unknown_reason_string_refuses(tmp_path: Path, monkeypatch) -> None:
    #: 决策侧词表若被改出第五种原因，本件的指认前提（:460-467 四条）就不成立 ⇒ 拒判而不是照算。
    _patched(monkeypatch, _fake_trigger(["brand_new_reason"], meets_all=False))
    rc, payload = _run(str(FACE), tmp_path)
    assert rc == 2
    assert "表外原因" in payload["refused"][0]["error"]


def test_pressure_ema_identity_holds_on_real_face() -> None:
    """钉住 PLAN-N3-07 §2 那条恒等式：`Σ wᵢ·EMAᵢ ≡ EMA(Σ wᵢ·xᵢ)`。

    那条恒等式是"旧面可以重放出 EMA 口径"的唯一依据，所以它**不许只是文档里的论证**：
    这里用产品自己的 trigger 与一支我自己维护的线性 EMA 逐步对表，任一步偏离即红。
    """

    from taiji.adaptive_residual_growth import (
        AdaptiveResidualGrowthPressure,
        AdaptiveResidualGrowthTrigger,
    )

    weights = {
        "residual_error": 0.30,
        "fast_slow_conflict": 0.25,
        "activity_saturation": 0.20,
        "utility_gap": 0.25,
    }
    #: trigger 只接受 `bridge_id` 相同的观测（:405-406 是产品自己的守卫，本测不绕开）⇒ 用面里的桥名。
    face_bridge_id = str(next(row["bridge_id"] for row in _rows() if row.get("kind") == "pressure"))
    trigger = AdaptiveResidualGrowthTrigger(bridge_id=face_bridge_id)
    rate = float(trigger.policy.ema_rate)
    mine = 0.0
    worst = 0.0
    for row in _rows():
        if row.get("kind") != "pressure":
            continue
        observation = AdaptiveResidualGrowthPressure.create(
            bridge_id=str(row["bridge_id"]),
            tick=int(row["tick"]),
            residual_error=float(row["residual_error"]),
            fast_slow_conflict=float(row["fast_slow_conflict"]),
            activity_saturation=float(row["activity_saturation"]),
            utility_gap=float(row["utility_gap"]),
            resource_state=float(row["resource_state"]),
            evidence_id=str(row["evidence_id"]),
            parent_checkpoint_digest=str(row.get("parent_checkpoint_digest") or ""),
        )
        raw_composite = sum(weight * float(row[field]) for field, weight in weights.items())
        mine = (1.0 - rate) * mine + rate * raw_composite
        decision = trigger.observe(observation, structural_budget=1)
        worst = max(worst, abs(float(decision.pressure) - mine))
    assert worst < 1e-12, worst
    #: 反证：权重抄错一位就必须偏离（否则这条测只会恒真）。
    wrong = 0.0
    other = AdaptiveResidualGrowthTrigger(bridge_id=face_bridge_id)
    for row in _rows():
        if row.get("kind") != "pressure":
            continue
        raw_composite = 0.35 * float(row["residual_error"]) + 0.25 * float(
            row["fast_slow_conflict"]
        )
        wrong = (1.0 - float(other.policy.ema_rate)) * wrong + 0.25 * raw_composite
        other_observation = AdaptiveResidualGrowthPressure.create(
            bridge_id=str(row["bridge_id"]),
            tick=int(row["tick"]),
            residual_error=float(row["residual_error"]),
            fast_slow_conflict=float(row["fast_slow_conflict"]),
            activity_saturation=float(row["activity_saturation"]),
            utility_gap=float(row["utility_gap"]),
            resource_state=float(row["resource_state"]),
            evidence_id=str(row["evidence_id"]),
            parent_checkpoint_digest=str(row.get("parent_checkpoint_digest") or ""),
        )
        other.observe(other_observation, structural_budget=1)
    assert abs(float(other.pressure_ema) - wrong) > 1e-6


#: 写侧源文件：PLAN-N3-08 的"零抄写"与"版本集一致"两条守卫都靠扫它，不靠运行时导入它（它会拉 torch）。
TRAINER = REPO / "scripts/training/train_seed_corpus.py"
#: 任何 `"minimum_*": <小数字面量>` 或 `"ema_rate": <小数字面量>` 都算抄写（值必须读自 `trigger.policy`）。
HARDCODED_POLICY_LITERAL = re.compile(r'"(?:minimum_\w+|ema_rate)":\s*0?\.\d')


@pytest.mark.no_local_artifacts
def test_trainer_reports_face_format_the_reader_accepts() -> None:
    """G-N3c-4：写侧输出的版本名必须落在读侧接受集里，且**只有**在读侧有对应夹具。"""

    source = TRAINER.read_text(encoding="utf-8")
    written = re.findall(r'"format":\s*"(taiji-n3-pressure-face-v\d+)"', source)
    assert written, "写侧找不到压强面格式名——它被改名了，本测的扫描面失效"
    assert set(written) <= set(READER.FACE_FORMATS), (written, READER.FACE_FORMATS)
    #: 反证：表外版本名必须让读侧响亮拒绝（不是"当成 v1 猜着读"）。
    assert "taiji-n3-pressure-face-v9" not in READER.FACE_FORMATS


@pytest.mark.no_local_artifacts
def test_trainer_pressure_section_has_no_hardcoded_thresholds(tmp_path: Path) -> None:
    """G-N3c-3"零抄写"：面头的九个 policy 值全部来自 `trigger.policy`，源里不许出现阈值字面量。"""

    source = TRAINER.read_text(encoding="utf-8")
    offenders = [line for line in source.splitlines() if HARDCODED_POLICY_LITERAL.search(line)]
    assert offenders == [], offenders
    #: 反例支（守卫必须能为假）：插一个写死阈值进临时副本，同一把尺必须抓到它。
    tampered = source.replace('"policy": {', '"policy": {"minimum_pressure": 0.70,', 1)
    assert tampered != source, "替换没生效——写侧形状变了，本测的反例支不再覆盖"
    assert HARDCODED_POLICY_LITERAL.search(tampered), "插入的写死值没被抓到＝守卫恒真"


def _v2_header() -> dict[str, Any]:
    policy = {
        "minimum_pressure": 0.7,
        "minimum_residual_error": 0.55,
        "minimum_fast_slow_conflict": 0.4,
        "minimum_activity_saturation": 0.4,
        "minimum_utility_gap": 0.35,
        "minimum_resource_state": 0.4,
        "required_pressure_steps": 3,
        "growth_resource_cost": 1,
        "ema_rate": 0.25,
    }
    return {
        "kind": "face",
        "format": "taiji-n3-pressure-face-v2",
        "bridge": {"gate": 0.0},
        "growth": {"parent_checkpoint_digest": ""},
        "policy": policy,
        "ema_initial": {
            "residual_error_ema": 0.0,
            "fast_slow_conflict_ema": 0.0,
            "activity_saturation_ema": 0.0,
            "utility_gap_ema": 0.0,
            "resource_state_ema": 1.0,
            "consecutive_pressure_steps": 0,
        },
    }


def test_v2_face_leaves_no_assumed_thresholds(tmp_path: Path) -> None:
    """J-N3c-自述：六道阈全部面内自述 ⇒ `assumed_from_product_defaults` 必须空、自述为真必须 6 道。

    观测行沿用真实 v1 面（同一批数），只把 face 头换成补齐版 ⇒ 这里测的是**读侧能不能停止回落默认**，
    不是重跑一次训练。
    """

    records = _rows()
    header = _v2_header()
    pressures = [record for record in records if record.get("kind") == "pressure"]
    rc, payload = _run(_write(tmp_path, [header] + pressures), tmp_path)
    assert rc == 0, payload
    face = payload["faces"][0]
    assert face["assumed_from_product_defaults"] == []
    assert len(face["per_gate"]) == 6
    assert all(gate["threshold_self_reported_in_face"] for gate in face["per_gate"].values())
    #: 补齐自述不许改变任何一道闸的判定——否则"补披露"就变成了"改读数"。
    real = _run(str(FACE), tmp_path)[1]["faces"][0]
    assert {k: v["true_steps"] for k, v in face["per_gate"].items()} == {
        k: v["true_steps"] for k, v in real["per_gate"].items()
    }


def test_unknown_face_format_refuses(tmp_path: Path) -> None:
    header = _v2_header()
    header["format"] = "taiji-n3-pressure-face-v9"
    records = _rows()
    pressures = [record for record in records if record.get("kind") == "pressure"]
    rc, payload = _run(_write(tmp_path, [header] + pressures), tmp_path)
    assert rc == 2
    assert "不在读侧接受集" in payload["refused"][0]["error"]


def test_foreign_policy_field_refuses(tmp_path: Path) -> None:
    #: 面头里出现表外 policy 字段 ⇒ 说明写/读两侧的字段集分家了，必须响亮拒绝而不是忽略。
    header = _v2_header()
    header["policy"]["minimum_surprise"] = 0.1
    records = _rows()
    pressures = [record for record in records if record.get("kind") == "pressure"]
    rc, payload = _run(_write(tmp_path, [header] + pressures), tmp_path)
    assert rc == 2
    assert "表外 policy 字段" in payload["refused"][0]["error"]


@pytest.mark.no_local_artifacts
def test_the_local_only_declaration_covers_the_face_it_claims() -> None:
    """**在 CI 上也要跑的一支**：它验的不是产品，是"本册的 skip 声明仍然成立"。

    CI run 37764787792 上本册 10 枚红全是 `FileNotFoundError`——那枚在盘面按 `.gitignore` 不入库，
    而本册此前没声明（05 §9 的 H19j／H19n 策略要求声明）。补声明之后，这条元测钉三件事：
    ① 声明的路径就是 `FACE`（改名会让声明失效而 skip 不生效）；
    ② 该路径**确实不入库**（谁把面件提交进来，这条红——那时 10 枚 skip 应当变回真跑）；
    ③ 豁免标记的数量与位置（只许贴在两支不读面件的测上；多贴＝把该跑的悄悄关掉）。
    """

    import sys as _sys

    module = _sys.modules[__name__]
    declared = tuple(getattr(module, "LOCAL_ONLY_ARTIFACTS", ()))
    assert declared == (FACE.relative_to(REPO).as_posix(),), declared
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", declared[0]],
        capture_output=True,
        check=False,
    )
    assert tracked.returncode != 0, tracked.stdout.decode("utf-8", errors="replace")[:200]
    source = Path(__file__).read_text(encoding="utf-8")
    #: 按**行首**取被应用的决定器，不用子串计数——本支自己要把那串字面量写进断言里，
    #: 子串计数会把自己抄的那两处也算进去（㊵-629/630 记过的自指形状：违禁词抄进禁令就命中自己）。
    exempt = [
        line for line in source.splitlines() if line.startswith("@pytest.mark.no_local_artifacts")
    ]
    assert len(exempt) == 3, exempt
    for name in (
        "def test_trainer_reports_face_format_the_reader_accepts",
        "def test_trainer_pressure_section_has_no_hardcoded_thresholds",
    ):
        assert ("@pytest.mark.no_local_artifacts\n" + name) in source, name
