"""零面普查的守卫：**分类、镜像去重、分母**三件必须在读数件里如实反映。

来历：`SPEC-A-22` §23 自报"不是全模型普查"，本仓据此补了
`scripts/training/audit_taiji_zero_face_census.py`。普查这类仪器最危险的不是算错范数，
而是**口径悄悄错**——把结构性索引当可学面、把活动状态当权重、或把 v10 信封里互为镜像的两份
载荷各算一遍（那样"零面占比"会被系统性放大近一倍，而头条数字就是假的）。
这四支测试钉的正是这三件口径，加上"一个面都没有 ⇒ 不出结论"的分母纪律。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "training" / "audit_taiji_zero_face_census.py"
)


@pytest.fixture(scope="module")
def census_module():
    spec = importlib.util.spec_from_file_location("zero_face_census_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "payload.pt"
    torch.save(payload, path)
    return path


def test_mirror_copies_are_deduplicated(tmp_path, census_module) -> None:
    """v10 信封的两份镜像逐位相同时，只许算一次（否则占比虚高一倍）。"""

    same = torch.ones(4)
    payload = {
        "format": "seed-native-v1",
        "taiji": {"kernel": {"fabric": {"decoders": [{"edge_weight": same.clone()}]}}},
        "substrate": {"fabric": {"decoders": [{"edge_weight": same.clone()}]}},
    }
    report = census_module.census(_write(tmp_path, payload))
    assert report["instrument_guard"]["mirror_duplicates_dropped"] == 1
    assert report["instrument_guard"]["mirror_conflicts"] == []
    faces = [face for face in report["faces"] if face["face"].endswith("edge_weight")]
    assert len(faces) == 1, faces


def test_mirror_difference_is_reported_loudly(tmp_path, census_module) -> None:
    """镜像**不相等**时不许静默取一份：两份都算，并记冲突。"""

    payload = {
        "format": "seed-native-v1",
        "taiji": {"kernel": {"fabric": {"decoders": [{"edge_weight": torch.ones(4)}]}}},
        "substrate": {"fabric": {"decoders": [{"edge_weight": torch.zeros(4)}]}},
    }
    report = census_module.census(_write(tmp_path, payload))
    conflicts = report["instrument_guard"]["mirror_conflicts"]
    assert len(conflicts) == 1, conflicts
    assert report["instrument_guard"]["mirror_duplicates_dropped"] == 0


def test_three_way_classification_keeps_indexes_and_state_out_of_learnable(
    tmp_path, census_module
) -> None:
    """结构性索引与活动状态都不许混进"可学面"，否则零面占比没有意义。"""

    payload = {
        "format": "seed-native-v1",
        "substrate": {
            "fabric": {
                "decoders": [
                    {"pre_index": torch.zeros(6, dtype=torch.long), "edge_weight": torch.ones(6)}
                ]
            },
            "state": {"regions": [{"trace": torch.zeros(3)}]},
        },
    }
    report = census_module.census(_write(tmp_path, payload))
    learnable = {face["face"] for face in report["faces"]}
    assert learnable == {"substrate.fabric.decoders[0].edge_weight"}, learnable
    assert report["summary"]["structural_faces"] == 1
    assert report["summary"]["state_faces"] == 1
    assert report["summary"]["zero_faces"] == 0
    assert report["summary"]["zero_share_of_learnable"] == 0.0


def test_zero_face_is_detected_and_reported(tmp_path, census_module) -> None:
    payload = {
        "format": "seed-native-v1",
        "substrate": {"fabric": {"consolidation_decoders": [{"edge_weight": torch.zeros(8)}]}},
    }
    report = census_module.census(_write(tmp_path, payload))
    zero = report["zero_face_list"]
    assert [face["face"] for face in zero] == [
        "substrate.fabric.consolidation_decoders[0].edge_weight"
    ]
    assert report["summary"]["zero_share_of_learnable"] == 1.0


def test_empty_payload_refuses_to_conclude(tmp_path, census_module) -> None:
    """分母为 0 ⇒ 不出结论（退出码 2），不许把"什么都没量到"读成"没有零面"。"""

    path = _write(tmp_path, {"format": "seed-native-v1"})
    report = census_module.census(path)
    assert report["instrument_guard"]["denominator_nonzero"] is False
    assert (
        census_module.main(["--checkpoint", str(path), "--report", str(tmp_path / "r.json")]) == 2
    )
