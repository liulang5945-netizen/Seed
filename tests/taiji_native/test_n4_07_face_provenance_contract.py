"""PLAN-N4-05 J-N4e-5 的前置修复：面仪器必须**转述材料来源档位**，但不许与自己那把库互相代答。

`measure_taiji_n4_addressing_surface.py` 的 `mount_layer="harness"`／`product_default_mounted=False`
说的是**它自己自建并挂载的那把 store**（㊵-606③ 就是把这两枚当成"材料档位"读了，才暴露缺口）。
现在新增 `--faces-meta`：随件的来源自述被原样搬进 `faces_provenance`，
两枚旧键**保持不变** ⇒ 报告里同时存在"仪器档"与"材料档"，谁也不替谁答。

四支都能为假：给了 meta ⇒ `status="present"` 且带 `mount_layer="product"`；
不给 ⇒ `absent`；给个不存在的路径 ⇒ `missing_file`（不猜、不静默）；
最后一支钉住"互相代答"没有发生：meta 说 product 为真时，仪器自述仍是 harness/False。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "measure_taiji_n4_addressing_surface.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n4_face_provenance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()
CUE_DIM = 8


def _cue(offset: int) -> list[float]:
    base = [0.0] * CUE_DIM
    base[offset % CUE_DIM] = 1.0
    base[-1] = 0.25 + 0.01 * offset
    return base


def _faces(tmp_path: Path) -> tuple[Path, Path, Path]:
    materials = [{"memory_id": f"m-{index:03d}", "cue": _cue(index)} for index in range(24)]
    queries = [
        {"expected_memory_id": row["memory_id"], "cue": list(row["cue"])} for row in materials
    ]
    mpath = tmp_path / "materials.jsonl"
    qpath = tmp_path / "queries.jsonl"
    newline = "\n"
    mpath.write_text(
        newline.join(json.dumps(r, ensure_ascii=False) for r in materials) + newline,
        encoding="utf-8",
        newline=newline,
    )
    qpath.write_text(
        newline.join(json.dumps(r, ensure_ascii=False) for r in queries) + newline,
        encoding="utf-8",
        newline=newline,
    )
    meta = tmp_path / "faces_meta.json"
    meta.write_text(
        json.dumps(
            {
                "format": "taiji-n4-product-tier-faces-v1",
                "mount_layer": "product",
                "product_default_mounted": True,
                "write_trigger": "settle_action",
                "cue_source": "percept.features",
                "cue_dim": CUE_DIM,
                "episodic_writes": len(materials),
                "store_write_call_sites_this_file": 0,
            },
            ensure_ascii=False,
        )
        + newline,
        encoding="utf-8",
        newline=newline,
    )
    return mpath, qpath, meta


def _run(tmp_path: Path, *extra: str) -> dict[str, Any]:
    mpath, qpath, meta = _faces(tmp_path)
    report = tmp_path / "report.json"
    argv = ["--materials", str(mpath), "--queries", str(qpath), "--out-report", str(report), *extra]
    MODULE.main(argv)
    return json.loads(report.read_text(encoding="utf-8"))


def test_faces_meta_is_carried_verbatim(tmp_path: Path) -> None:
    _, _, meta = _faces(tmp_path)
    report = _run(tmp_path, "--faces-meta", str(meta))
    prov = report["faces_provenance"]
    assert prov["status"] == "present"
    assert prov["mount_layer"] == "product"
    assert prov["product_default_mounted"] is True
    assert prov["write_trigger"] == "settle_action"
    assert prov["episodic_writes"] == 24
    assert Path(prov["faces_meta"]) == meta


def test_no_meta_is_reported_absent_not_assumed(tmp_path: Path) -> None:
    report = _run(tmp_path)
    assert report["faces_provenance"] == {"status": "absent", "faces_meta": None}


def test_missing_meta_file_is_loud_not_silent(tmp_path: Path) -> None:
    report = _run(tmp_path, "--faces-meta", str(tmp_path / "nowhere.json"))
    assert report["faces_provenance"]["status"] == "missing_file"


def test_instrument_and_material_layers_do_not_answer_for_each_other(tmp_path: Path) -> None:
    _, _, meta = _faces(tmp_path)
    report = _run(tmp_path, "--faces-meta", str(meta))
    #: 材料档说 product，仪器那把库仍自述 harness／False——两行并存才是诚实的口径。
    assert report["faces_provenance"]["mount_layer"] == "product"
    assert report["mount_layer"] == "harness"
    assert report["product_default_mounted"] is False
