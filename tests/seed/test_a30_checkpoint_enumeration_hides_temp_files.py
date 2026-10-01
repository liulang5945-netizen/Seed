"""DEBT-G15 守卫：产品枚举检查点时**不得**暴露点前缀临时件。

为什么需要这条守卫（不是清洁洁癖）：`pathlib.Path.glob("*.pt")` **会**匹配以 `.` 开头的文件
（与 shell 和 `glob.glob` 不同），而崩溃/中断留下的 `.xxx.pt` 就散在 `checkpoints/` 里
（登记当天实测到三枚，其中两枚 43 MB）。两个产品入口都用 `pathlib.glob` ⇒ 隐藏临时件会被列成"可用基座"。

本文件把三件事钉住，任何一件不成立都应当红：
1. 训练面板 `GET /api/train/checkpoints` 只列非隐藏件；
2. 工件清单 `GET /api/artifacts` 只列非隐藏件；
3. **未过滤会匹上**这一前提本身（否则本守卫就是在守一个不存在的行为——那才是真红）。
"""

from __future__ import annotations

from pathlib import Path

import api.routes_artifacts as routes_artifacts
import api.training.checkpoints as training_checkpoints

_HIDDEN = ".p2-12-conflict.pt"
_VISIBLE = "seed_real_with_circuit.pt"


def _seed_dir(tmp_path: Path) -> Path:
    """造一个只含两枚 .pt 的目录：一枚正常、一枚隐藏临时件（字节够小，只为被枚举到）。"""

    root = tmp_path / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    (root / _VISIBLE).write_bytes(b"x")
    (root / _HIDDEN).write_bytes(b"y")
    return root


def test_unfiltered_pathlib_glob_would_expose_hidden(tmp_path: Path) -> None:
    """前提守卫：证明 `pathlib.glob` 确实匹配点前缀 ⇒ 本文件不是在守一个不存在的洞。"""

    root = _seed_dir(tmp_path)
    names = {path.name for path in root.glob("*.pt")}
    assert _HIDDEN in names, "pathlib 不再匹配点前缀文件了 ⇒ DEBT-G15 的成因消失，应改判此测试"


def test_training_panel_hides_temp_checkpoints(tmp_path: Path, monkeypatch) -> None:
    root = _seed_dir(tmp_path)
    monkeypatch.setattr(training_checkpoints, "_CHECKPOINT_DIR", root)

    listed = training_checkpoints.list_checkpoints()

    assert listed["status"] == "ok"
    names = [item["filename"] for item in listed["checkpoints"]]
    assert _HIDDEN not in names, f"隐藏临时件又被列成可用基座：{names}"
    assert names == [_VISIBLE], f"期望只剩一枚可见件，实得 {names}"


def test_artifact_inventory_hides_temp_checkpoints(tmp_path: Path, monkeypatch) -> None:
    root = _seed_dir(tmp_path)
    monkeypatch.setattr(routes_artifacts, "get_external_path", lambda rel: str(root))

    payload = routes_artifacts.list_artifacts()

    names = [item["artifact_id"] for item in payload.get("artifacts", [])]
    # 先堵"空过"：可见件不在列就说明读的根本不是这个结构，那时本测试必须红而不是静默通过
    assert _VISIBLE in names, f"工件清单没列出可见件（结构或键名已变）：{names} / {sorted(payload)}"
    assert _HIDDEN not in names, f"工件清单暴露了隐藏临时件：{names}"
