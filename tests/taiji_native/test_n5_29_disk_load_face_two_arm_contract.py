"""`measure_taiji_disk_load_face_two_arms.py` 的契约册：装机面的双臂配对必须能为假。

这台仪器要钉的是 ㊵-727 那组一次性读数（盘上 13 枚信封：4 枚档 12 载入／1 拒载，5 枚档同样 12 载入／1 拒载，
唯一拒载那枚是 DEBT-G47 在两臂上同形的再观测）。它只有读没有写，所以风险不是改坏东西，而是**把取路错当成结论**：

① ㊵-727 ② 的真实教训是第一版探针拿 `Taiji.from_checkpoint` 去喂 Seed 信封，13/13 全报
   `KeyError: 'region_sizes'`——那是入口点错，不是装机面拒载。所以本册把"反例臂"钉成断言：
   同批信封在平面入口下**必须**报错，且不许出现 `UNEXPECTED_LOAD`。
② 两臂的配对形状是双向的：盘上件写在窄词表下 ⇒ 治疗臂靠迁移支才装得进；合成件写在现行词表下 ⇒
   **对照臂（更窄）必须拒**。两支方向合起来才证明"载入结果随词表宽度配对变化"，而不是仪器恒绿。
③ 覆盖率与原因计数必须自述（`selected == disk_pt_total`、两臂 `attempted == selected`、
   原因计数之和 == 拒载数），否则"这一面没拒载"可能只是"我没读全"。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from seed import Seed, SeedConfig
from taiji import Taiji, TaijiConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "measure_taiji_disk_load_face_two_arms.py"
REPORT = PROJECT_ROOT / "reports" / "taiji_disk_load_face_two_arms_20261012.json"


def _small_seed_config() -> SeedConfig:
    return SeedConfig(taiji=TaijiConfig(region_sizes=(16,), synapse_fan_in=4, motor_fan_in=8))


def _load():
    spec = importlib.util.spec_from_file_location("n5_29_disk_load_face", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_envelope_face(root: Path) -> Path:
    """在临时根下摆一枚由产品自己产出的信封面（不碰仓里任何一枚真件）。"""

    face = root / "checkpoints"
    face.mkdir(parents=True, exist_ok=True)
    path = face / "synthetic.pt"
    torch.save(Seed(_small_seed_config()).checkpoint(), path)
    return path


def _write_calls(source: str) -> list[str]:
    hits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            name = ""
        if name.endswith("save") or name == "dump":
            hits.append(name)
        if name == "open" and len(node.args) > 1:
            modes = [arg.value for arg in node.args[1:] if isinstance(arg, ast.Constant)]
            if any(isinstance(mode, str) and ("w" in mode or "a" in mode) for mode in modes):
                hits.append("open_for_write")
    return hits


def test_the_instrument_refuses_a_face_with_no_checkpoints(tmp_path: Path) -> None:
    module = _load()
    with pytest.raises(ValueError, match="空面"):
        module.read_face(tmp_path, ["checkpoints"], 64, 1024, [])


def test_the_synthetic_face_loads_wide_and_refuses_narrow(tmp_path: Path) -> None:
    """合成件写在现行词表下 ⇒ 治疗臂载入、对照臂必须拒——这是配对的另一半方向。"""

    module = _load()
    _synthetic_envelope_face(tmp_path)
    report = module.read_face(tmp_path, ["checkpoints"], 64, 1024, [])

    treated = report["arms"]["treated"]
    control = report["arms"]["control"]
    assert report["verdict"] == "full_face_two_arm_pair"
    assert treated["loaded"] == 1 and treated["refused"] == 0
    assert treated["rows"][0]["encoder_width"] == len(treated["kinds"])
    assert control["loaded"] == 0 and control["refused"] == 1
    assert control["refusing_files"] == ["checkpoints/synthetic.pt"]
    assert report["pairing"]["refused_on_both_arms"] == []
    assert report["pairing"]["refused_only_on_treated_arm"] == []


def test_the_wrong_entry_control_refuses_every_envelope_it_attempts(tmp_path: Path) -> None:
    """㊵-727 ② 的教训钉成断言：平面入口喂信封时整片报错，不许有一枚"意外载入"。"""

    module = _load()
    _synthetic_envelope_face(tmp_path)
    report = module.read_face(tmp_path, ["checkpoints"], 64, 1024, [])
    control = report["entry_point_control"]
    assert control["attempted"] >= 1
    assert "UNEXPECTED_LOAD" not in control["error_counts"]
    assert sum(control["error_counts"].values()) == control["attempted"]

    # 同一条反例在真实产品链上也要成立（不经仪器，直接两个入口各走一次）。
    envelope = Seed(_small_seed_config()).checkpoint()
    Seed.from_checkpoint(envelope)
    with pytest.raises(KeyError):
        Taiji.from_checkpoint(envelope)


def test_the_instrument_never_writes_into_the_face() -> None:
    """零产品码、零盘上写：出现任何 `*.save(`／`open(..., "w")` 就说明这台仪器越界了。"""

    source = SCRIPT.read_text(encoding="utf-8")
    assert _write_calls(source) == []
    assert "torch.load" in source

    # 负对照：同一条谓词必须能拒绝一段自己造的越界样本。
    fabricated = "import torch\n\ndef f(p, obj):\n    torch.save(obj, p)\n"
    assert _write_calls(fabricated) == ["save"]


def test_the_sealed_disk_face_report_is_internally_consistent() -> None:
    """已入库的盘上面读数：只核自述的算术，不用它代替现行复跑。"""

    if not REPORT.is_file():
        pytest.skip(f"装机面读数件不在盘上：{REPORT.name}")
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    face = payload["face"]
    treated = payload["arms"]["treated"]
    control = payload["arms"]["control"]

    assert face["tracked_pt_in_git"] == 0, "入库 .pt 计数变了，本件的半径要重报"
    assert face["selected"] == face["disk_pt_total"], payload["verdict"]
    assert payload["verdict"] == "full_face_two_arm_pair"
    for arm in (treated, control):
        assert arm["attempted"] == face["selected"]
        assert arm["loaded"] + arm["refused"] == arm["attempted"]
        assert sum(arm["refusal_reason_counts"].values()) == arm["refused"]
    assert treated["loaded"] == control["loaded"]
    assert treated["refusing_files"] == control["refusing_files"]
    assert payload["pairing"]["refused_only_on_treated_arm"] == []
    assert payload["pairing"]["load_counts_equal"] is True
    assert "UNEXPECTED_LOAD" not in payload["entry_point_control"]["error_counts"]
