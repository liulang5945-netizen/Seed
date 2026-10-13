"""DEBT-G92 的修法：逐张量比较器（`compare_taiji_n5_arm_tensors.py`）的契约测。

这条仪器存在的理由就是"㊵-656 那个数是仓外跑的"，所以测里必须**双向**钉住它能为真也能为假：
自比为 0、扰动一枚必须被点名、名集不齐与镜像分叉都要响亮拒绝。
另两支是当日现场锚：三臂封存件里的 0／20／10，以及"list 里的张量不许进名表"。
2026-10-13 再加两支（DEBT-G129）：形状不齐的两枚必须点名出版而不是崩，且 delta 表与形状名表
必须是差异槽的一个**划分**（不重不漏）——否则 `top_deltas` 会被读成覆盖了全部差异。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "compare_taiji_n5_arm_tensors.py"
REAL_REPORT = REPO / "reports" / "taiji_n5_08_arm_tensor_diff_20261010.json"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_tensor_compare", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CMP = _load()


def _save(tmp_path: Path, name: str, envelope: dict[str, Any]) -> Path:
    path = tmp_path / name
    torch.save(envelope, str(path))
    return path


def _mirror(value: float) -> dict[str, Any]:
    #: 两个根各存一份，正是真信封的形状（`substrate.*` 与 `taiji.kernel.*`）。
    return {
        "substrate": {"readout": {"w": torch.tensor([value, 0.0])}},
        "taiji": {"kernel": {"readout": {"w": torch.tensor([value, 0.0])}}},
    }


def test_a_checkpoint_never_differs_from_itself(tmp_path: Path) -> None:
    left = _save(tmp_path, "a.pt", _mirror(1.0))
    out = tmp_path / "r.json"
    rc = CMP.main(["--checkpoint", str(left), "--checkpoint", str(left), "--out-report", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["status"] == "measured"
    row = payload["pairs"][0]
    assert row["verdict"] == "identical"
    assert row["differing_slots"] == 0
    assert row["distinct_content_pairs"] == 0
    #: 镜像核对数的是**源侧槽**（`substrate.*` 那一枚），不是两根的总槽数 ⇒ 一份镜像对＝1。
    mirror = payload["mirror_consistency"][0]
    assert mirror["mirrored_slots"] == 1
    assert mirror["agree_within_arm"] == 1
    assert mirror["one_sided_slots"] == 0


def test_one_perturbed_tensor_is_named_and_counted_in_both_units(tmp_path: Path) -> None:
    #: 「槽」与「枚」两个单位必须同时出版：这里 2 个槽＝1 枚内容对。
    base = _save(tmp_path, "a.pt", _mirror(1.0))
    moved = _save(tmp_path, "b.pt", _mirror(2.0))
    out = tmp_path / "r.json"
    CMP.main(["--checkpoint", str(base), "--checkpoint", str(moved), "--out-report", str(out)])
    row = json.loads(out.read_text(encoding="utf-8"))["pairs"][0]
    assert row["differing_slots"] == 2
    assert row["distinct_content_pairs"] == 1
    assert row["distinct_group_sizes"] == [2]
    assert row["differing_slots_names"] == [
        "substrate.readout.w",
        "taiji.kernel.readout.w",
    ]
    assert row["top_deltas"][0]["max_abs_delta"] == pytest.approx(1.0)


def test_tensors_held_in_lists_are_counted_but_never_given_index_names(tmp_path: Path) -> None:
    #: 存盘 list 里的张量用索引当名字会在跨档改序时错位 ⇒ 只计数、不进名表。
    envelope = {"substrate": {"w": torch.tensor([1.0])}, "history": [torch.zeros(3), torch.ones(2)]}
    path = _save(tmp_path, "a.pt", envelope)
    out = tmp_path / "r.json"
    CMP.main(["--checkpoint", str(path), "--checkpoint", str(path), "--out-report", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tensor_slots"] == 1
    assert payload["list_tensor_count"] == 2


def test_a_mirror_that_diverges_within_one_arm_is_a_loud_refusal(tmp_path: Path) -> None:
    #: 同臂内两根不一致＝存盘路径分叉，比"两臂有差"更该响亮 ⇒ rc=2 且状态点名。
    broken = {
        "substrate": {"readout": {"w": torch.tensor([1.0, 0.0])}},
        "taiji": {"kernel": {"readout": {"w": torch.tensor([9.0, 0.0])}}},
    }
    left = _save(tmp_path, "a.pt", broken)
    right = _save(tmp_path, "b.pt", _mirror(1.0))
    out = tmp_path / "r.json"
    rc = CMP.main(["--checkpoint", str(left), "--checkpoint", str(right), "--out-report", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 2
    assert payload["status"] == "refuse_mirror_divergence"
    assert payload["mirror_consistency"][0]["disagreeing"]


def test_differing_name_sets_are_refused_not_padded(tmp_path: Path) -> None:
    left = _save(tmp_path, "a.pt", _mirror(1.0))
    right_env = _mirror(1.0)
    right_env["substrate"]["extra"] = torch.ones(2)
    right = _save(tmp_path, "b.pt", right_env)
    out = tmp_path / "r.json"
    rc = CMP.main(["--checkpoint", str(left), "--checkpoint", str(right), "--out-report", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 2
    assert payload["status"] == "refuse_name_sets_differ"
    assert payload["pairs"][0]["only_in_right"] == ["substrate.extra"]


def test_one_checkpoint_or_a_missing_file_refuses(tmp_path: Path) -> None:
    single = _save(tmp_path, "a.pt", _mirror(1.0))
    rc = CMP.main(["--checkpoint", str(single)])
    assert rc == 2
    rc_missing = CMP.main(["--checkpoint", str(single), "--checkpoint", str(tmp_path / "gone.pt")])
    assert rc_missing == 2


def test_real_three_arm_seal_publishes_zero_then_twenty_slots_ten_weights() -> None:
    #: 当日现场锚（㊵-663）：控制臂 vs 第三档＝0 of 220，vs 治疗臂＝20 槽＝10 枚。
    #: 「第三档与控制臂逐位相同」这句是 20/220 归因给单变量的唯一排除条件，必须有件。
    payload = json.loads(REAL_REPORT.read_text(encoding="utf-8"))
    assert payload["status"] == "measured"
    assert payload["tensor_slots"] == 220
    rows = {row["label"].split("-> ")[1].split("/")[-2]: row for row in payload["pairs"]}
    assert rows["determinism"]["differing_slots"] == 0
    assert rows["treated"]["differing_slots"] == 20
    assert rows["treated"]["distinct_content_pairs"] == 10
    assert set(rows["treated"]["distinct_group_sizes"]) == {2}
    for side in payload["mirror_consistency"]:
        assert side["agree_within_arm"] == side["mirrored_slots"] == 94


def test_differently_shaped_tensors_are_named_not_subtracted(tmp_path: Path) -> None:
    #: DEBT-G129 的现场：基座（批次 B 之前，那一面宽 4）对治疗臂（宽 5）时，减法在 broadcast 上抛
    #: `RuntimeError`——整台仪器 rc=1、件根本没落。形状不齐必须**点名出版**，既不崩也不静默成"没差"。
    base = {
        "substrate": {"edge": torch.zeros(4, 2)},
        "taiji": {"kernel": {"edge": torch.zeros(4, 2)}},
    }
    arm = {
        "substrate": {"edge": torch.zeros(5, 2)},
        "taiji": {"kernel": {"edge": torch.zeros(5, 2)}},
    }
    left = _save(tmp_path, "base.pt", base)
    right = _save(tmp_path, "arm.pt", arm)
    out = tmp_path / "r.json"
    rc = CMP.main(["--checkpoint", str(left), "--checkpoint", str(right), "--out-report", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    row = payload["pairs"][0]
    assert rc == 0
    assert payload["status"] == "measured"
    assert row["differing_slots"] == 2
    assert row["shape_mismatch_slots"] == 2
    assert row["delta_comparable_slots"] == 0
    assert row["top_deltas"] == []
    assert row["shape_pairs"] == ["[4, 2] vs [5, 2]"]
    assert row["shape_mismatch_names"] == ["substrate.edge", "taiji.kernel.edge"]
    assert row["verdict"] == "differing"


def test_the_delta_table_and_the_shape_set_partition_the_differences(tmp_path: Path) -> None:
    #: 两半必须**不重不漏**：可比的那批进 delta 表，形状不齐的那批进名表，相加＝总差异槽。
    #: 少了这条，`top_deltas` 会被读成"覆盖了所有差异"，而它其实只覆盖可比的一半。
    base = {"substrate": {"a": torch.zeros(3), "b": torch.zeros(3)}}
    arm = {"substrate": {"a": torch.zeros(4), "b": torch.ones(3)}}
    left = _save(tmp_path, "base.pt", base)
    right = _save(tmp_path, "arm.pt", arm)
    out = tmp_path / "r.json"
    rc = CMP.main(["--checkpoint", str(left), "--checkpoint", str(right), "--out-report", str(out)])
    row = json.loads(out.read_text(encoding="utf-8"))["pairs"][0]
    assert rc == 0
    assert row["differing_slots"] == 2
    assert row["delta_comparable_slots"] + row["shape_mismatch_slots"] == row["differing_slots"]
    assert [entry["slot"] for entry in row["top_deltas"]] == ["substrate.b"]
    assert row["shape_mismatch_names"] == ["substrate.a"]


def test_the_three_buckets_partition_the_differing_slots(tmp_path: Path) -> None:
    #: PLAN-N5-21 §6（㊵-770）：计量面按构造必然不等（喂入那一步多走一次 observe），
    #: 所以"第四类"的判定单位只能是权重面＋拓扑面。这一支钉三件事：
    #: 分档正确、三档相加＝总差异槽、计量面被单独点名而不是混进判决面。
    base = {
        "substrate": {
            "readout": {"w": torch.zeros(2)},
            "perception": {"dynamic": {"symbol_exposure": torch.zeros(3)}},
            "memory": {"provenance_readout": {"pre_index": torch.zeros(2, dtype=torch.long)}},
        }
    }
    moved = {
        "substrate": {
            "readout": {"w": torch.ones(2)},
            "perception": {"dynamic": {"symbol_exposure": torch.full((3,), 4.0)}},
            "memory": {"provenance_readout": {"pre_index": torch.ones(2, dtype=torch.long)}},
        }
    }
    left = _save(tmp_path, "base.pt", base)
    right = _save(tmp_path, "arm.pt", moved)
    out = tmp_path / "r.json"
    rc = CMP.main(["--checkpoint", str(left), "--checkpoint", str(right), "--out-report", str(out)])
    row = json.loads(out.read_text(encoding="utf-8"))["pairs"][0]
    assert rc == 0
    assert row["differing_slots"] == 3
    assert row["differing_by_bucket"] == {"weight": 1, "topology": 1, "meter": 1}
    assert sum(row["differing_by_bucket"].values()) == row["differing_slots"]
    assert row["differing_bucket_names"]["meter"] == [
        "substrate.perception.dynamic.symbol_exposure"
    ]
    assert row["differing_bucket_names"]["topology"] == [
        "substrate.memory.provenance_readout.pre_index"
    ]
    assert row["differing_bucket_names"]["weight"] == ["substrate.readout.w"]
    assert "计量面" in row["meter_face_note"]
