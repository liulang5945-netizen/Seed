"""HEAD 对表面（PLAN-N5-06 §2 J-N6a-4 的尺）：同一 tiny 配置＋同一 64 字节输入，比较 **HEAD 树**与**工作树**的观测摘要。

**为什么需要它**：㊵-636 那三枚锚点摘要出自一次性 scratch 面脚本（件已不在仓里），而 `taiji/` 此后又进了
多笔提交 ⇒ 历史锚点**不能**当今天的对照。本件唯一有效的对照形状是"同一算法、同一输入，HEAD vs 工作树"，
而这件事要能复算，就必须把尺入库而不是每次重写。

**三档形状**（`--modes` 可指名子集）：

* `position_off`：默认位（`readout_utf8_position_input=False`）的普通链；
* `position_on`：同一配置开位置输入 ⇒ 摘要**必须**与上一档不同（否则这把尺没有动态范围）；
* `developmental_off_position`：发育叠加层＋位置关（旧形状）。挂载序列照训练器那一支
  （`train_seed_corpus.py:725-765`）：先 `enable_adaptive_residual_bridge(gate=…)` 再
  `enable_adaptive_residual_growth()`，然后 `migrate_f1_to_developmental_synapses()`，最后
  `set_developmental_f1_learning_mode("fast_slow")`。少前两步会分别撞
  `taiji/model.py:1346`（must be mounted）与 `:2083`（read-only until R2）——两处都是响亮拒绝。

**一条 fail-closed**：任何一档在两棵树上取不到摘要 ⇒ `status=parity_indeterminate`＋rc=2。
理由不是洁癖：发育档第一次跑时两棵树**都**返回 `None`，而 `None == None` 会把"逐位相同"读成真绿。

尺的自证：`position_off` 复现 `tests/taiji_native/test_n4_05_product_default_mount_contract.py:31`
的 `BASELINE_DIGEST`（逐字相同），所以这把重做的尺复算了一条已入库的钉子，不是只说自己好。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACE_MODULES = ("taiji", "seed")
ALL_MODES = ("position_off", "position_on", "developmental_off_position")
PROBE_NAME = "n6_head_face_probe.py"

#: 面脚本本体被写进**仓外**临时目录再执行（cwd＋PYTHONPATH 指向被测树），
#: 所以整件对表期间不会往仓里落任何文件——门在跑的时候这一点是硬要求。
FACE_SCRIPT = """
import json
from seed import Seed, SeedConfig
from taiji.config import TaijiConfig
from taiji.internalization import content_digest

TINY = dict(
    region_sizes=(8,), synapse_fan_in=2, motor_fan_in=4, predictive_context_fan_in=2,
    memory_units=16, memory_fan_in=2, memory_readout_fan_in=2, memory_meta_dim=4,
    memory_time_dim=2, memory_episode_dim=2, lateral_fan_in=2,
    identity_organ_capacity=8, concept_capacity=8, seed=101,
)
SYMBOLS = bytes(range(32, 96))


def _rows(model, symbols):
    out = []
    for symbol in symbols:
        step = model.observe(symbol, learn=True, readout="predictive")
        prior = getattr(step, "prior_prediction", None)
        surprise = getattr(step, "surprise", None)
        out.append([
            None if prior is None else int(prior),
            None if surprise is None else round(float(surprise), 12),
        ])
    return out


def face(**over):
    config = TaijiConfig(**dict(TINY, **over))
    model = Seed(SeedConfig(taiji=config)).substrate
    return content_digest({"rows": _rows(model, SYMBOLS)})


def developmental_face():
    config = TaijiConfig(**dict(TINY))
    model = Seed(SeedConfig(taiji=config)).substrate
    model.enable_adaptive_residual_bridge(gate=1.0)
    model.enable_adaptive_residual_growth()
    model.migrate_f1_to_developmental_synapses()
    model.set_developmental_f1_learning_mode("fast_slow")
    return content_digest({"rows": _rows(model, SYMBOLS)})


out = {}
errors = {}
try:
    out["position_off"] = face()
except Exception as error:  # noqa: BLE001 - 两棵树的错误必须同形状出版
    out["position_off"] = None
    errors["position_off"] = type(error).__name__ + ": " + str(error)[:200]
try:
    out["position_on"] = face(readout_utf8_position_input=True)
except Exception as error:  # noqa: BLE001
    out["position_on"] = None
    errors["position_on"] = type(error).__name__ + ": " + str(error)[:200]
try:
    out["developmental_off_position"] = developmental_face()
except Exception as error:  # noqa: BLE001
    out["developmental_off_position"] = None
    errors["developmental_off_position"] = type(error).__name__ + ": " + str(error)[:200]
out["errors"] = errors
print(json.dumps(out))
"""


def _run_face(tree: Path, scratch: Path, modes: tuple[str, ...]) -> dict[str, Any]:
    script = scratch / PROBE_NAME
    script.write_text(FACE_SCRIPT, encoding="utf-8", newline="\n")
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        check=False,
        cwd=str(tree),
        env=dict(os.environ, PYTHONPATH=str(tree)),
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"face 进程 rc={proc.returncode} stderr={tail}")
    payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    errors = dict(payload.pop("errors", {}) or {})
    selected = {key: payload[key] for key in modes}
    return {"digests": selected, "errors": {key: errors[key] for key in modes if key in errors}}


def _extract_head(repo: Path, target: Path) -> Path:
    archive = target / "head.tar"
    with open(archive, "wb") as handle:
        subprocess.run(
            ["git", "-C", str(repo), "archive", "HEAD", *FACE_MODULES],
            stdout=handle,
            check=True,
        )
    root = target / "head"
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    archive.unlink(missing_ok=True)
    return root


def _purge(directory: Path) -> None:
    for path in sorted(directory.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    directory.rmdir()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    #: `newline="\n"` 不是装饰：这台机器上 `open(...,'w')` 默认按 CRLF 落盘，
    #: 而本仓的行尾门与被测件都按 LF 计（㊵-636 那族踩过两次）。
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parity(repo: Path, modes: tuple[str, ...]) -> dict[str, Any]:
    head = (
        subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    scratch = Path(tempfile.mkdtemp(prefix="n6_parity_"))
    try:
        head_run = _run_face(_extract_head(repo, scratch), scratch, modes)
        work_run = _run_face(repo, scratch, modes)
    finally:
        _purge(scratch)
    verdicts = {}
    for mode in modes:
        head_value = head_run["digests"].get(mode)
        work_value = work_run["digests"].get(mode)
        if head_value is None or work_value is None:
            verdicts[mode] = "unmeasured"
        else:
            verdicts[mode] = "identical" if head_value == work_value else "diverged"
    #: 位置关与位置开必须不同——否则这把尺对"位置输入"没有分辨力，"逐位相同"就只是没动。
    off = work_run["digests"].get("position_off")
    on = work_run["digests"].get("position_on")
    return {
        "format": "taiji-n6-head-face-parity-v1",
        "prereg": "PLAN-N5-06 §2 J-N6a-4",
        "head_commit": head,
        "input_bytes": 64,
        "modes": list(modes),
        "head_digests": head_run["digests"],
        "worktree_digests": work_run["digests"],
        "face_errors": {
            "head": head_run["errors"],
            "worktree": work_run["errors"],
        },
        "verdicts": verdicts,
        "worktree_position_switch_is_visible": None if off is None or on is None else off != on,
    }


def parity_rc(payload: dict[str, Any]) -> int:
    """判定与出版同源：任何一档 `unmeasured` ⇒ 本件不算判过。"""

    if any(value == "unmeasured" for value in payload["verdicts"].values()):
        return 2
    return 0 if all(value == "identical" for value in payload["verdicts"].values()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HEAD-vs-worktree observation face (zero training, read-only)"
    )
    parser.add_argument("--out-report", type=Path, required=True, help="where to write the JSON")
    parser.add_argument(
        "--repo", type=Path, default=PROJECT_ROOT, help="tree to compare against HEAD"
    )
    parser.add_argument(
        "--modes",
        action="append",
        choices=ALL_MODES,
        default=None,
        help="subset of faces to run (repeatable); default is all three",
    )
    args = parser.parse_args(argv)
    modes = tuple(args.modes) if args.modes else ALL_MODES

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = parity(args.repo, modes)
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        _write_json(
            args.out_report,
            {
                "format": "taiji-n6-head-face-parity-v1",
                "status": "face_failed",
                "error": str(error),
            },
        )
        print(json.dumps({"status": "face_failed"}, ensure_ascii=True))
        return 2
    rc = parity_rc(payload)
    payload["status"] = "parity_indeterminate" if rc == 2 else "ok"
    payload["verdict"] = payload["verdicts"]
    _write_json(args.out_report, payload)
    print(
        json.dumps(
            {"status": payload["status"], "verdicts": payload["verdicts"]}, ensure_ascii=True
        )
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
